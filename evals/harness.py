"""Runs one scenario through the REAL call pipeline, in real time, without a phone.

Two simulated sides talk to a real CallSession:

  CallerLine  - the caller's phone: sends one 20 ms audio frame every 20 ms,
                forever, just like Twilio does. Speech when the caller is
                talking, room tone (noise or line hiss) otherwise.
  FakeTwilio  - receives the agent's audio and "plays" it at real speed. It
                echoes each mark only when the audio before it would have
                finished playing, and on `clear` drops queued audio, as
                Twilio does.

Real-time pacing matters: VAD, Deepgram endpointing and barge-in all depend
on timing, so feeding audio faster than real time would make the latency
and interruption numbers meaningless.

Client-side latency: from the last frame of the caller's speech to the first
agent audio received. That's what the caller experiences (minus phone
network time, which no server-side test can see).
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import uuid
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

import numpy as np

from evals.audio import background_frame, degrade, mulaw_frames
from evals.caller import HANGUP, INTERJECTIONS, CallerBrain
from evals.scenarios import Scenario
from evals.score import FieldResult, completed, score_restaurant, score_shelter, score_volunteer
from hungercue.audio.mulaw import mulaw_to_pcm16, pcm16_to_mulaw
from hungercue.db.repo import Repo
from hungercue.pipeline.call_session import CallSession, Services
from hungercue.pipeline.costs import DEEPGRAM_TTS_PER_1K_CHARS
from hungercue.tts.aura import TTSClient

log = logging.getLogger(__name__)

FRAME_S = 0.02
HUMAN_PAUSE_S = 0.35  # how long a person waits after the agent stops before answering
AGENT_SILENT_TIMEOUT_S = 8.0  # if the agent never answers, the caller says something again
INTERRUPT_AFTER_S = 1.2  # how far into the agent's reply the interrupting caller cuts in
BARGE_IN_OK_S = 1.0  # agent must stop talking within this time to count as a successful barge-in


@dataclass
class Utterance:
    frames: list[bytes]
    started: asyncio.Event = field(default_factory=asyncio.Event)
    ended: asyncio.Event = field(default_factory=asyncio.Event)
    started_at: float = 0.0
    ended_at: float = 0.0


class FakeTwilio:
    def __init__(self) -> None:
        self.loop = asyncio.get_running_loop()
        self.inbox: asyncio.Queue[str] = asyncio.Queue()
        self.play_end = 0.0  # when queued agent audio will finish playing
        self.media_times: list[float] = []  # arrival time of each agent audio chunk
        self.clears: list[float] = []
        self._pending: list[tuple[asyncio.TimerHandle, str]] = []
        self._echoed: set[str] = set()

    def playing(self) -> bool:
        return self.play_end > self.loop.time()

    async def send(self, raw: str) -> None:
        """CallSession's outbound messages land here (instead of a WebSocket)."""
        msg = json.loads(raw)
        now = self.loop.time()
        if msg["event"] == "media":
            n = len(base64.b64decode(msg["media"]["payload"]))
            self.media_times.append(now)
            self.play_end = max(now, self.play_end) + n / 8000
        elif msg["event"] == "mark":
            name = msg["mark"]["name"]
            handle = self.loop.call_later(max(0.0, self.play_end - now), self._echo, name)
            self._pending.append((handle, name))
        elif msg["event"] == "clear":
            self.clears.append(now)
            self.play_end = now
            for handle, name in self._pending:
                handle.cancel()
                self._echo(name)  # Twilio returns marks for cleared audio right away
            self._pending = []

    def _echo(self, name: str) -> None:
        if name not in self._echoed:
            self._echoed.add(name)
            self.inbox.put_nowait(json.dumps({"event": "mark", "mark": {"name": name}}))

    def first_media_after(self, t: float) -> float | None:
        return next((m for m in self.media_times if m >= t), None)


class CallerLine:
    """The caller's microphone: a steady 20 ms frame clock."""

    def __init__(self, twilio: FakeTwilio, noise: str, snr_db: float, seed: int) -> None:
        self.twilio = twilio
        self.noise, self.snr_db = noise, snr_db
        self.rng = np.random.default_rng(seed)
        self.queue: deque[tuple[bytes, Utterance | None, str]] = deque()  # (frame, utterance, position)
        self.running = True
        self.timestamp_ms = 0

    def say(self, frames: list[bytes]) -> Utterance:
        utt = Utterance(frames)
        for i, f in enumerate(frames):
            pos = "first" if i == 0 else "last" if i == len(frames) - 1 else ""
            self.queue.append((f, utt, pos))
        if len(frames) == 1:
            self.queue[-1] = (frames[0], utt, "only")
        return utt

    @property
    def speaking(self) -> bool:
        return bool(self.queue)

    async def run(self) -> None:
        loop = asyncio.get_running_loop()
        next_t = loop.time()
        while self.running:
            now = loop.time()
            if self.queue:
                frame, utt, pos = self.queue.popleft()
                if utt is not None and pos in ("first", "only"):
                    utt.started_at = now
                    utt.started.set()
                if utt is not None and pos in ("last", "only"):
                    utt.ended_at = now + FRAME_S  # the frame finishes 20 ms after it starts
                    utt.ended.set()
            else:
                frame = background_frame(self.noise, self.snr_db, self.rng)
            payload = base64.b64encode(frame).decode()
            self.twilio.inbox.put_nowait(json.dumps(
                {"event": "media", "media": {"payload": payload, "timestamp": str(self.timestamp_ms)}}))
            self.timestamp_ms += 20
            # Schedule against an absolute clock so small delays don't accumulate into drift.
            next_t += FRAME_S
            await asyncio.sleep(max(0.0, next_t - loop.time()))


@dataclass
class EvalActions:
    """Stands in for Twilio call control: records how the call ended."""

    ended_by: str = ""

    async def transfer(self, call_sid: str, to_number: str, say: str) -> None:
        self.ended_by = "transfer"

    async def voicemail(self, call_sid: str, say: str, callback_url: str) -> None:
        self.ended_by = "voicemail"

    async def hangup(self, call_sid: str) -> None:
        self.ended_by = "hangup"


@dataclass
class CallResult:
    scenario_id: str
    caller_type: str
    language: str
    conditions: dict[str, Any]
    classified_as: str = ""
    fields: list[dict[str, Any]] = field(default_factory=list)
    completed: bool = False
    server_turns: list[dict[str, Any]] = field(default_factory=list)
    client_v2v_ms: list[float] = field(default_factory=list)
    barge_ins: list[dict[str, Any]] = field(default_factory=list)
    agent_cost_usd: float = 0.0
    eval_cost_usd: float = 0.0  # simulated caller (LLM + TTS); not part of the product's cost
    duration_s: float = 0.0
    ended_by: str = ""
    error: str = ""
    transcript: list[dict[str, Any]] = field(default_factory=list)


ServicesFactory = Callable[[EvalActions], Services]
UtteranceHook = Callable[[str, Utterance], Awaitable[None]]


def latest_agent_text(session: CallSession) -> str:
    """Everything the agent said since the caller last spoke."""
    out: list[str] = []
    for m in reversed(session.state.messages):
        if m["role"] == "user":
            break
        if m["role"] == "assistant" and m.get("content"):
            out.append(m["content"].replace(" [interrupted by caller]", ""))
    return " ".join(reversed(out)).strip()


async def wait_agent_done(session: CallSession, twilio: FakeTwilio, actions: EvalActions,
                          after_turn: int, deadline: float) -> None:
    """Wait until the agent has answered turn `after_turn` and finished playing it."""
    loop = asyncio.get_running_loop()
    asked_at = loop.time()
    quiet_since: float | None = None
    while True:
        now = loop.time()
        if now > deadline:
            raise TimeoutError("call exceeded its time limit")
        if actions.ended_by or session.finished:
            return
        answered = session.turn_index > after_turn or (now - asked_at) > AGENT_SILENT_TIMEOUT_S
        idle = (session.response_task is None or session.response_task.done()) and not twilio.playing()
        if answered and idle and session.awaiting_mark is None:
            quiet_since = quiet_since or now
            if now - quiet_since >= HUMAN_PAUSE_S:
                return
        else:
            quiet_since = None
        await asyncio.sleep(0.05)


async def synthesize_frames(tts: TTSClient, text: str, scenario: Scenario, rng: np.random.Generator) -> tuple[list[bytes], int]:
    """Caller speech: TTS in the caller's voice, then noise/mumbling applied."""
    audio = b"".join([chunk async for chunk in tts.synthesize(text, scenario.language)])
    c = scenario.conditions
    pcm = degrade(mulaw_to_pcm16(audio), c.noise, c.snr_db, c.mumble, rng)
    return mulaw_frames(pcm16_to_mulaw(pcm)), len(text)


async def run_scenario(
    scenario: Scenario,
    services_factory: ServicesFactory,
    brain: CallerBrain,
    caller_tts: TTSClient,
    repo: Repo,
    call_date: datetime,
    max_turns: int = 16,
    max_seconds: float = 300.0,
    on_utterance: UtteranceHook | None = None,
) -> CallResult:
    loop = asyncio.get_running_loop()
    c = scenario.conditions
    result = CallResult(scenario.id, scenario.caller_type, scenario.language, asdict(c))
    twilio = FakeTwilio()
    seed = int(scenario.id[1:]) if scenario.id[1:].isdigit() else 0
    line = CallerLine(twilio, c.noise, c.snr_db, seed=seed)
    rng = np.random.default_rng(seed + 1)
    actions = EvalActions()
    sid = f"CA{scenario.id}{uuid.uuid4().hex[:24]}"
    if scenario.returning:
        t = scenario.truth
        await repo.upsert_restaurant(scenario.phone, name=t["business_name"],
                                     address=f"{t['address_number']} {t['address_street']}")

    session = CallSession(services_factory(actions), twilio.send)
    run_task = asyncio.create_task(session.run(twilio.inbox.get))
    pump_task = asyncio.create_task(line.run())
    started = loop.time()
    twilio.inbox.put_nowait(json.dumps({"event": "start", "streamSid": f"MZ{scenario.id}",
                                        "start": {"callSid": sid, "customParameters": {"from": scenario.phone}}}))
    caller_chars = 0
    deadline = started + max_seconds

    async def speak(text: str) -> Utterance:
        nonlocal caller_chars
        frames, chars = await synthesize_frames(caller_tts, text, scenario, rng)
        caller_chars += chars
        utt = line.say(frames)
        if on_utterance is not None:
            asyncio.create_task(on_utterance(text, utt))
        return utt

    try:
        interrupted = False
        replies = 0
        # The agent has "answered" once its turn counter passes this value. -1 means
        # the greeting is enough (no caller turn to answer yet).
        answer_after = -1
        for _ in range(max_turns):
            await wait_agent_done(session, twilio, actions, answer_after, deadline)
            if actions.ended_by or session.finished:
                break
            text = await brain.reply(latest_agent_text(session))
            if text == HANGUP:
                break
            answer_after = session.turn_index
            utt = await speak(text)
            await utt.ended.wait()
            # Client-side voice-to-voice: wait for the first agent audio after we stopped talking.
            while twilio.first_media_after(utt.ended_at) is None and loop.time() - utt.ended_at < AGENT_SILENT_TIMEOUT_S:
                await asyncio.sleep(0.01)
            first = twilio.first_media_after(utt.ended_at)
            if first is not None:
                result.client_v2v_ms.append(round((first - utt.ended_at) * 1000, 1))
            replies += 1

            if c.interrupt and not interrupted and replies >= 2 and first is not None:
                await asyncio.sleep(max(0.0, first + INTERRUPT_AFTER_S - loop.time()))
                if twilio.playing():
                    interrupted = True
                    words = INTERJECTIONS[scenario.language]
                    brain.note_said(words)
                    answer_after = session.turn_index  # the agent should now answer the interjection
                    utt2 = await speak(words)
                    await utt2.started.wait()
                    t0 = utt2.started_at
                    while not any(x >= t0 for x in twilio.clears) and loop.time() - t0 < 2 * BARGE_IN_OK_S:
                        await asyncio.sleep(0.01)
                    cleared = next((x for x in twilio.clears if x >= t0), None)
                    reaction = round((cleared - t0) * 1000, 1) if cleared is not None else None
                    result.barge_ins.append({"reaction_ms": reaction,
                                             "success": reaction is not None and reaction <= BARGE_IN_OK_S * 1000})
                    await utt2.ended.wait()
    except Exception as e:  # one broken call shouldn't sink a 100-call run; it's recorded
        log.exception("scenario %s failed", scenario.id)
        result.error = f"{type(e).__name__}: {e}"
    finally:
        line.running = False
        twilio.inbox.put_nowait(json.dumps({"event": "stop"}))
        try:
            await asyncio.wait_for(run_task, 20)
        except (asyncio.TimeoutError, Exception) as e:  # noqa: BLE001 - recorded, not raised
            result.error = result.error or f"shutdown: {type(e).__name__}"
        pump_task.cancel()
        await asyncio.gather(pump_task, return_exceptions=True)

    result.duration_s = round(loop.time() - started, 1)
    result.ended_by = actions.ended_by or "caller_hangup"
    result.server_turns = list(session.metrics.turns)
    result.agent_cost_usd = session.metrics.cost.total if session.metrics.cost else 0.0
    result.eval_cost_usd = round(brain.cost_usd + caller_chars / 1000 * DEEPGRAM_TTS_PER_1K_CHARS, 5)
    await _score(result, scenario, repo, sid, call_date)
    return result


async def _score(result: CallResult, s: Scenario, repo: Repo, sid: str, call_date: datetime) -> None:
    call = await repo.get_call(sid)
    result.classified_as = call.caller_type if call else ""
    result.transcript = call.transcript if call else []
    pickup = await repo.pickup_for_call(sid)
    restaurant = await repo.restaurant_by_phone(s.phone)
    shelter = await repo.shelter_by_phone(s.phone)
    volunteer = await repo.volunteer_by_phone(s.phone)
    fields: list[FieldResult]
    if s.caller_type == "restaurant":
        fields = score_restaurant(s.truth, pickup, restaurant, call_date, s.returning)
    elif s.caller_type == "shelter":
        fields = score_shelter(s.truth, shelter)
    else:
        fields = score_volunteer(s.truth, volunteer)
    result.fields = [asdict(f) for f in fields]
    result.completed = completed(s.caller_type, pickup, shelter, volunteer)
