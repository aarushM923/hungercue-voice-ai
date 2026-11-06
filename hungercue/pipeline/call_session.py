"""CallSession: runs one phone call from first ring to hang-up.

Three things happen *at the same time* during a call, each as an asyncio task:

  1. Twilio reader (the `run` loop): receives 20 ms audio frames, forwards them
     to Deepgram, feeds the VAD, and records them.
  2. STT reader (`_read_stt`): receives transcripts from Deepgram and decides
     when the caller has finished their turn.
  3. Responder (`_respond`): streams the LLM reply, turns each sentence into
     speech, and sends audio to Twilio.

Barge-in, i.e. the caller talking over the agent:
  VAD hears speech while the responder is active or audio is still playing
  -> cancel the responder task (stops the LLM and TTS streams mid-flight)
  -> send Twilio `clear` (drops audio it buffered but hasn't played)
  -> the caller's new words become the next turn.

Pipelining for latency: the reply is split into sentences. TTS for sentence 1
starts the moment it's written, while the LLM is still writing sentence 2,
so audio reaches the caller long before the full reply exists.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hungercue.audio.recorder import CallRecorder
from hungercue.audio.vad import SAMPLE_RATE, VadSegmenter
from hungercue.config import Settings
from hungercue.db.repo import CallerProfile, Repo
from hungercue.llm.agent import Agent, ChatClient
from hungercue.llm.prompts import VOICEMAIL_LINE, build_system_prompt, greeting
from hungercue.llm.tools import Geocoder, ToolContext
from hungercue.phone import normalize_phone
from hungercue.pipeline.costs import CallCost
from hungercue.pipeline.latency import TurnTimer
from hungercue.state.session_store import CallState, SessionStore
from hungercue.stt.deepgram import STTStream
from hungercue.telephony import twilio_stream as ts
from hungercue.telephony.actions import CallActions
from hungercue.timeutil import now_local
from hungercue.tts.aura import TTSClient

log = logging.getLogger(__name__)

STT_FAILURE_LINE = "Sorry, we're having technical trouble. Please leave a message after the tone."


@dataclass
class CallSummary:
    """Handed to the end-of-call hook (Google Sheets logging)."""

    call_sid: str
    from_number: str
    caller_type: str
    outcome: str
    language: str
    transcript_text: str
    pickup_id: int | None


@dataclass
class Services:
    """Everything a call needs. main.py builds real ones; tests and evals build fakes."""

    settings: Settings
    repo: Repo
    store: SessionStore
    llm_client: ChatClient
    tts: TTSClient
    stt_factory: Callable[[], STTStream]
    vad_factory: Callable[[], VadSegmenter]
    actions: CallActions | None = None
    geocode: Geocoder | None = None
    on_donation: Callable[[int], Awaitable[None]] | None = None
    on_directory_update: Callable[[str, dict[str, Any]], Awaitable[None]] | None = None
    on_call_end: Callable[[CallSummary], Awaitable[None]] | None = None
    # Background caller-type guess (see llm/classify.py); None = rely on the saving tools only.
    classifier: Callable[[list[str]], Awaitable[tuple[str, Any]]] | None = None


@dataclass
class BargeIn:
    caller_started: float  # loop time the caller started speaking
    cleared: float  # loop time we sent Twilio `clear`

    @property
    def reaction_ms(self) -> float:
        return round((self.cleared - self.caller_started) * 1000, 1)


@dataclass
class CallMetrics:
    """Collected during the call; the eval harness reads these directly."""

    turns: list[dict[str, Any]] = field(default_factory=list)
    barge_ins: list[BargeIn] = field(default_factory=list)
    tts_chars: int = 0
    cost: CallCost | None = None


class CallSession:
    def __init__(self, services: Services, send: Callable[[str], Awaitable[None]]) -> None:
        self.s = services
        self.cfg = services.settings
        self.send = send
        self.loop = asyncio.get_running_loop()
        self.recorder = CallRecorder()
        self.vad = services.vad_factory()
        self.metrics = CallMetrics()

        self.stream_sid = ""
        self.state = CallState(call_sid="")
        self.profile = CallerProfile(kind="")
        self.ctx: ToolContext | None = None
        self.agent: Agent | None = None
        self.stt: STTStream | None = None
        self.stt_task: asyncio.Task[None] | None = None
        self.response_task: asyncio.Task[None] | None = None
        self.awaiting_mark: str | None = None  # final mark of the last reply, not yet echoed
        self.mark_counter = 0
        # One mark per sentence: Twilio echoes a mark once the audio before it has
        # PLAYED, so these tell us exactly which sentences the caller really heard.
        self.sentence_marks: list[tuple[str, str]] = []  # (mark name, sentence) for the current reply
        self.played_marks: set[str] = set()
        self.turn_index = 0
        self.current_timer: TurnTimer | None = None
        self.last_turn: dict[str, Any] | None = None  # latency record of the latest reply
        self.last_turn_id: int | None = None
        self.side_tasks: set[asyncio.Task[None]] = set()  # background work like classification
        self.finished = False

        # Turn-taking state
        self.pending_parts: list[str] = []  # final transcript pieces of the current caller turn
        self.has_interim = False  # Deepgram is still refining words not yet final
        self.utterance_language = ""
        self.speech_end_at: float | None = None
        self.finalized_this_pause = False
        self.last_final_at: float | None = None

    # ------------------------------------------------------------------ main loop

    async def run(self, receive: Callable[[], Awaitable[str]]) -> None:
        """Process Twilio messages until the stream stops or the socket closes."""
        try:
            while True:
                event = ts.parse_message(await receive())
                if isinstance(event, ts.StartEvent):
                    await self.on_start(event)
                elif isinstance(event, ts.MediaEvent):
                    await self.on_media(event)
                elif isinstance(event, ts.MarkEvent):
                    await self.on_mark(event.name)
                elif isinstance(event, ts.DtmfEvent):
                    await self.on_dtmf(event.digit)
                elif isinstance(event, ts.StopEvent):
                    break
        finally:
            await self.finish()

    def now(self) -> float:
        return self.loop.time()

    # ------------------------------------------------------------------ call start

    async def on_start(self, ev: ts.StartEvent) -> None:
        self.stream_sid = ev.stream_sid
        phone = normalize_phone(ev.from_number)
        repo = self.s.repo
        await repo.start_call(ev.call_sid, phone)
        profile: CallerProfile = await repo.lookup_caller(phone)

        self.state = CallState(call_sid=ev.call_sid, from_number=phone, caller_type=profile.kind)
        greet = greeting(profile, "en")
        system = build_system_prompt(now_local(self.cfg.timezone), profile, "en", phone)
        self.profile = profile
        self.state.messages = [{"role": "system", "content": system}, {"role": "assistant", "content": greet}]
        self.ctx = ToolContext(
            repo=repo,
            state=self.state,
            geocode=self.s.geocode,
            on_donation=self.s.on_donation,
            on_directory_update=self.s.on_directory_update,
        )
        self.agent = Agent(self.s.llm_client, self.cfg.llm_model, self.ctx)
        await self.s.store.save(self.state)

        try:
            self.stt = self.s.stt_factory()
            await self.stt.start()
        except Exception:
            # Without speech recognition we can't converse; don't leave the caller in silence.
            log.exception("STT failed to start for %s", ev.call_sid)
            self.stt = None
            self.ctx.pending_action = "voicemail"
            self.state.outcome = "voicemail"
            self._start_response(self._fixed([STT_FAILURE_LINE]), timer=None, log_text=True)
            return
        self.stt_task = asyncio.create_task(self._read_stt())
        await repo.add_transcript(ev.call_sid, "agent", greet)
        self._start_response(self._fixed([greet]), timer=None, log_text=False)

    # ------------------------------------------------------------------ inbound audio

    async def on_media(self, ev: ts.MediaEvent) -> None:
        self.recorder.add_inbound(ev.payload)
        if self.stt is not None:
            try:
                await self.stt.send(ev.payload)
            except Exception:
                log.warning("STT send failed", exc_info=True)
        processed_ms = self.vad.samples_seen * 1000 // SAMPLE_RATE
        for vad_event in self.vad.push_mulaw(ev.payload):
            processed_ms = self.vad.samples_seen * 1000 // SAMPLE_RATE
            # Convert "position in the audio stream" into a wall-clock time:
            # the event happened (processed - event position) ms before now.
            happened_at = self.now() - (processed_ms - vad_event.stream_ms) / 1000
            if vad_event.kind == "speech_start":
                self.speech_end_at = None  # a new utterance: any earlier end time is stale
                if self.agent_busy():
                    await self.barge_in(caller_started=happened_at)
            elif vad_event.kind == "speech_pause":
                # Don't wait for Deepgram's own (slower) endpointing. At the first short
                # pause, ask it to finalise what it has heard. That round trip then
                # overlaps with the rest of our end-of-turn silence wait, so the final
                # transcript is usually already here when we decide the caller is done.
                # If they keep talking, nothing is lost: Deepgram keeps transcribing.
                await self._finalize_stt()
                self.finalized_this_pause = True
            else:
                self.speech_end_at = happened_at
                if not self.finalized_this_pause:  # e.g. pause and end thresholds configured equal
                    await self._finalize_stt()
                self.finalized_this_pause = False
                await self.maybe_trigger(source="vad")

    async def _finalize_stt(self) -> None:
        # Always sent, even if nothing is pending: for a short "yes" Deepgram may not
        # have sent an interim result yet, and Finalize costs nothing when there's nothing to flush.
        if self.stt is not None:
            try:
                await self.stt.finalize()
            except Exception:
                log.warning("STT finalize failed", exc_info=True)

    def agent_busy(self) -> bool:
        """Is the agent thinking or talking (including audio still queued at Twilio)?"""
        responding = self.response_task is not None and not self.response_task.done()
        return responding or self.awaiting_mark is not None

    async def barge_in(self, caller_started: float) -> None:
        if self.current_timer is not None:
            self.current_timer.interrupted = True
        if self.ctx is not None:
            self.ctx.pending_action = ""  # caller has more to say; don't hang up or transfer yet
        await self._cancel_response()
        await self.send(ts.clear_message(self.stream_sid))
        self.metrics.barge_ins.append(BargeIn(caller_started, self.now()))
        self.recorder.clear()
        self.awaiting_mark = None
        await self._note_what_was_heard()
        log.info("barge-in on %s", self.state.call_sid)

    async def _note_what_was_heard(self) -> None:
        """Tell the LLM how much of its last reply the caller actually heard.

        The LLM usually finishes writing long before the audio finishes playing,
        so its history claims it said everything. If the caller cut in after one
        sentence, the model must know the rest was never heard.
        """
        heard = [text for name, text in self.sentence_marks if name in self.played_marks]
        unheard = [text for name, text in self.sentence_marks if name not in self.played_marks]
        self.sentence_marks = []
        if not unheard:
            return
        note = "The caller interrupted you. "
        note += f'They only heard: "{" ".join(heard)}". ' if heard else "They heard none of your last reply. "
        note += f'They did NOT hear: "{" ".join(unheard)}".'
        self.state.messages.append({"role": "system", "content": note})
        # If the reply had already finished, its latency row was saved as
        # uninterrupted; correct it.
        if self.last_turn is not None and not self.last_turn["interrupted"]:
            self.last_turn["interrupted"] = True
            if self.last_turn_id is not None:
                await self.s.repo.mark_turn_interrupted(self.last_turn_id)

    async def _cancel_response(self) -> None:
        task = self.response_task
        if task is not None and not task.done():
            task.cancel()
            # Wait for its cleanup (partial transcript, latency row) to finish.
            with contextlib.suppress(asyncio.CancelledError):
                await task

    # ------------------------------------------------------------------ transcripts / turn-taking

    async def _read_stt(self) -> None:
        assert self.stt is not None
        try:
            async for t in self.stt.events():
                if t.text and t.is_final:
                    self.pending_parts.append(t.text)
                    self.last_final_at = self.now()
                    self.has_interim = False
                    if t.language:
                        self.utterance_language = t.language
                elif t.text:
                    self.has_interim = True
                if t.utterance_end:
                    # Deepgram's word-gap backup: works even when constant background noise
                    # keeps our VAD from ever hearing silence.
                    await self.maybe_trigger(source="utterance_end")
                elif (t.speech_final or (t.from_finalize and t.is_final)) and not self.vad.in_speech:
                    # The words are final AND our VAD already heard the end of speech.
                    await self.maybe_trigger(source="stt")
        except asyncio.CancelledError:
            raise
        except Exception:
            log.warning("STT stream ended with error", exc_info=True)

    async def maybe_trigger(self, source: str) -> None:
        """Start the agent's turn if the caller has said something and has stopped.

        Our VAD is the authority on "the caller has stopped" (224 ms of silence);
        Deepgram supplies the words. The turn starts when BOTH hold:
          - VAD heard the end of speech (source="vad"), and every word is final;
          - or a final transcript arrives after the VAD already heard the end (source="stt").
        Backup: Deepgram's UtteranceEnd (source="utterance_end") for noisy rooms
        where the VAD may never hear real silence.

        An earlier version also started the turn on the early "forced final" we
        request at a 96 ms pause. That cut people off mid-sentence, and the
        eval exposed it through an impossible latency number.
        """
        text = " ".join(self.pending_parts).strip()
        if not text:
            return
        if source == "vad" and self.has_interim:
            return
        self.pending_parts = []
        await self.start_turn(text)

    async def start_turn(self, text: str) -> None:
        assert self.agent is not None
        if self.agent_busy():
            # Caller kept talking and VAD didn't catch it as a barge-in; treat it as one.
            await self.barge_in(caller_started=self.now())
        self.turn_index += 1
        timer = TurnTimer(
            turn_index=self.turn_index,
            t_speech_end=self.speech_end_at,
            t_transcript=self.last_final_at,
            t_trigger=self.now(),
        )
        self.speech_end_at = None
        self._update_language(text)
        self.state.messages.append({"role": "user", "content": text})
        await self.s.repo.add_transcript(self.state.call_sid, "caller", text)
        self._start_response(self.agent.respond(self.state.messages), timer=timer, log_text=True)
        if self.s.classifier is not None and not self.state.caller_type_confirmed:
            # Runs alongside the reply, not before it: the caller never waits on it.
            task = asyncio.create_task(self._classify())
            self.side_tasks.add(task)
            task.add_done_callback(self.side_tasks.discard)

    async def _classify(self) -> None:
        assert self.s.classifier is not None
        lines = [m["content"] for m in self.state.messages if m["role"] == "user"]
        try:
            guess, usage = await self.s.classifier(lines)
        except Exception:
            log.warning("caller classification failed", exc_info=True)
            return
        if usage is not None and self.agent is not None:
            self.agent._add_usage(usage)  # counted in this call's cost
        # A saving tool may have settled the type while we were waiting; it wins.
        if guess and not self.state.caller_type_confirmed and guess != self.state.caller_type:
            self.state.caller_type = guess
            await self.s.repo.update_call(self.state.call_sid, caller_type=guess)

    def _update_language(self, text: str) -> None:
        """Switch to Spanish (or back) when a full utterance is detected in it.

        Detected on the first turn as the spec asks, but also allowed later
        in case someone hands the phone to a coworker.
        """
        lang = "es" if self.utterance_language.startswith("es") else "en" if self.utterance_language else ""
        self.utterance_language = ""
        if not lang or lang == self.state.language:
            return
        if self.turn_index > 1 and len(text.split()) < 3:
            return  # a lone "sí" or "okay" isn't enough to switch mid-call
        self.state.language = lang
        self.state.messages[0]["content"] = build_system_prompt(
            now_local(self.cfg.timezone), self.profile, lang, self.state.from_number
        )

    # ------------------------------------------------------------------ outbound speech

    @staticmethod
    async def _fixed(texts: list[str]) -> AsyncIterator[str]:
        for t in texts:
            yield t

    def _start_response(self, pieces: AsyncIterator[str], timer: TurnTimer | None, log_text: bool) -> None:
        self.current_timer = timer
        self.sentence_marks = []
        self.last_turn = None
        self.last_turn_id = None
        self.response_task = asyncio.create_task(self._respond(pieces, timer, log_text))

    async def _respond(self, pieces: AsyncIterator[str], timer: TurnTimer | None, log_text: bool) -> None:
        """Speak a stream of text pieces, pipelining LLM -> TTS -> Twilio."""
        spoken: list[str] = []
        language = self.state.language
        # A queue of (sentence, audio queue) pairs keeps sentences in order while
        # letting TTS for later sentences run ahead (defined below, before use).

        async def tts_into(text: str, out: asyncio.Queue[bytes | None]) -> None:
            try:
                async for chunk in self.s.tts.synthesize(text, language):
                    await out.put(chunk)
            except asyncio.CancelledError:
                raise
            except Exception:
                log.warning("TTS failed for %r", text, exc_info=True)
            finally:
                await out.put(None)  # end-of-sentence marker, even on failure

        async def produce() -> None:
            try:
                # aclosing() guarantees the LLM generator is closed even if we're
                # cancelled between sentences, so it records what was said.
                async with contextlib.aclosing(pieces):  # type: ignore[type-var]
                    async for piece in pieces:
                        if timer is not None and timer.t_first_token is None and self.agent is not None:
                            timer.t_first_token = self.agent.first_token_at
                        if timer is not None and timer.t_tts_request is None:
                            timer.t_tts_request = self.now()
                        spoken.append(piece)
                        self.metrics.tts_chars += len(piece)
                        q: asyncio.Queue[bytes | None] = asyncio.Queue()
                        tg.create_task(tts_into(piece, q))
                        await sentence_queues.put((piece, q))
            finally:
                await sentence_queues.put(None)

        async def consume() -> None:
            while (item := await sentence_queues.get()) is not None:
                piece, q = item
                # Register the sentence before sending audio, so a barge-in
                # mid-sentence knows this sentence was cut off.
                self.mark_counter += 1
                name = f"m{self.mark_counter}"
                self.sentence_marks.append((name, piece))
                sent_audio = False
                while (chunk := await q.get()) is not None:
                    if timer is not None and timer.t_first_audio is None:
                        timer.t_first_audio = self.now()
                    await self.send(ts.media_message(self.stream_sid, chunk))
                    self.recorder.add_outbound(chunk)
                    sent_audio = True
                if not sent_audio:
                    # TTS produced nothing (it failed): nothing was heard and there is
                    # nothing to bookmark, so don't wait on a mark for it.
                    self.sentence_marks.remove((name, piece))
                    continue
                # Bookmark the end of this sentence; Twilio echoes it once played.
                await self.send(ts.mark_message(self.stream_sid, name))

        sentence_queues: asyncio.Queue[tuple[str, asyncio.Queue[bytes | None]] | None] = asyncio.Queue()
        try:
            # TaskGroup: if this task is cancelled (barge-in), every child task
            # (the LLM stream, each TTS request, the sender) is cancelled with it.
            async with asyncio.TaskGroup() as tg:
                tg.create_task(produce())
                tg.create_task(consume())
            # The reply is fully sent. It's still "playing" until Twilio echoes
            # the last sentence's mark (which may already have happened).
            last = self.sentence_marks[-1][0] if self.sentence_marks else None
            self.awaiting_mark = last if last and last not in self.played_marks else None
        finally:
            text = " ".join(spoken)
            if timer is not None and timer.interrupted:
                text += " [interrupted]"
            # Shield the bookkeeping so a cancellation can't leave it half-written.
            await asyncio.shield(self._after_reply(text if log_text else "", timer))

    async def _after_reply(self, text: str, timer: TurnTimer | None) -> None:
        try:
            if text.strip():
                await self.s.repo.add_transcript(self.state.call_sid, "agent", text)
            if timer is not None:
                breakdown = timer.breakdown()
                self.last_turn = {"turn_index": timer.turn_index, **breakdown}
                self.metrics.turns.append(self.last_turn)
                row = await self.s.repo.add_turn(self.state.call_sid, timer.turn_index, **breakdown)
                self.last_turn_id = row.id
                log.info("turn %s latency %s", timer.turn_index, breakdown)
            await self.s.store.save(self.state)
            # Nothing left to play (no audio, or it all played already): act now
            # instead of waiting for a mark that will never come.
            if self.awaiting_mark is None and not self.finished and self.ctx and self.ctx.pending_action:
                await self.perform_action()
        except Exception:
            log.warning("post-reply bookkeeping failed", exc_info=True)

    async def on_mark(self, name: str) -> None:
        self.played_marks.add(name)
        if name != self.awaiting_mark:
            return  # a sentence mark mid-reply, or one from a reply cleared by barge-in
        self.awaiting_mark = None
        if self.ctx is not None and self.ctx.pending_action:
            await self.perform_action()

    async def on_dtmf(self, digit: str) -> None:
        """Pressing 0 at any time reaches a person (a classic phone-menu habit)."""
        if digit == "0" and self.ctx is not None:
            if self.agent_busy():
                await self.barge_in(caller_started=self.now())
            self.ctx.pending_action = "transferred"
            self.state.outcome = "transferred"
            await self.perform_action()

    async def perform_action(self) -> None:
        """Carry out transfer / voicemail / hang-up after the agent finished speaking."""
        assert self.ctx is not None
        action, self.ctx.pending_action = self.ctx.pending_action, ""
        actions, sid = self.s.actions, self.state.call_sid
        if actions is None:
            log.warning("no CallActions configured; cannot %s", action)
            return
        lang = self.state.language
        if action == "transferred" and not self.cfg.coordinator_phone:
            action = "voicemail"  # nobody to transfer to: take a message instead
            self.state.outcome = "voicemail"
        try:
            if action == "transferred":
                await actions.transfer(sid, self.cfg.coordinator_phone, "")
            elif action == "voicemail":
                await actions.voicemail(sid, VOICEMAIL_LINE.get(lang, VOICEMAIL_LINE["en"]), self.cfg.public_base_url + "/voicemail")
            elif action == "end_call":
                await actions.hangup(sid)
        except Exception:
            log.exception("call action %s failed", action)

    # ------------------------------------------------------------------ call end

    async def finish(self) -> None:
        if self.finished:
            return
        self.finished = True
        await self._cancel_response()
        if self.stt_task is not None:
            self.stt_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self.stt_task
        if self.stt is not None:
            await self.stt.close()
        if not self.state.call_sid:
            return
        if self.side_tasks:
            # Let an in-flight classification land (briefly), so the call record has the final type.
            _done, still_running = await asyncio.wait(list(self.side_tasks), timeout=5)
            for task in still_running:
                task.cancel()
            await asyncio.gather(*still_running, return_exceptions=True)

        sid = self.state.call_sid
        audio_path = ""
        if self.recorder.duration_s > 0:
            try:
                audio_path = str(self.recorder.save(Path(self.cfg.recordings_dir) / f"{sid}.wav"))
            except OSError:
                log.exception("could not save recording for %s", sid)
        llm_usd = self.agent.usage.cost_usd(self.cfg.llm_model) if self.agent else 0.0
        self.metrics.cost = CallCost(self.recorder.duration_s / 60, self.metrics.tts_chars, llm_usd)
        call = await self.s.repo.get_call(sid)
        await self.s.repo.update_call(
            sid,
            ended_at=now_local(self.cfg.timezone),
            language=self.state.language,
            caller_type=self.state.caller_type,
            outcome=self.state.outcome or (call.outcome if call else "") or "no_action",
            audio_path=audio_path,
            cost_usd=self.metrics.cost.total,
        )
        if self.s.on_call_end is not None:
            transcript = " | ".join(f"{t['role']}: {t['text']}" for t in (call.transcript if call else []))
            summary = CallSummary(
                sid, self.state.from_number, self.state.caller_type, self.state.outcome or "no_action",
                self.state.language, transcript, self.state.pickup_id,
            )
            try:
                await self.s.on_call_end(summary)
            except Exception:
                log.exception("on_call_end hook failed")
        await self.s.store.delete(sid)
