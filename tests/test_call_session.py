"""End-to-end tests of one call through CallSession, with every external service faked."""

import asyncio
import json
import wave
from pathlib import Path

import pytest

from hungercue.config import Settings
from hungercue.db.repo import Repo
from hungercue.pipeline.call_session import CallSession, CallSummary, Services
from hungercue.state.session_store import SessionStore, make_redis
from tests.fakes import (
    LOUD, QUIET, FakeActions, FakeOpenAI, FakeSTT, FakeTTS, loudness_vad, text_chunks, tool_chunks, until,
)


class Harness:
    """Plays Twilio's side of the WebSocket."""

    def __init__(self, repo: Repo, tmp_path: Path, scripts: list, tts: FakeTTS | None = None,
                 stt_fails: bool = False, **settings) -> None:
        self.inbox: asyncio.Queue[str] = asyncio.Queue()
        self.out: list[dict] = []
        self.stt = FakeSTT()
        self.tts = tts or FakeTTS()
        self.actions = FakeActions()
        self.llm = FakeOpenAI(scripts)
        self.summaries: list[CallSummary] = []
        self.store = SessionStore(make_redis(""))
        cfg = Settings(_env_file=None, recordings_dir=str(tmp_path / "rec"), **settings)

        def stt_factory():
            if stt_fails:
                raise ConnectionError("deepgram down")
            return self.stt

        async def on_end(s: CallSummary) -> None:
            self.summaries.append(s)

        self.services = Services(cfg, repo, self.store, self.llm, self.tts, stt_factory, loudness_vad,
                                 actions=self.actions, on_call_end=on_end)

    async def start(self, phone: str = "+12125550100") -> None:
        async def send(msg: str) -> None:
            self.out.append(json.loads(msg))

        self.session = CallSession(self.services, send)
        self.task = asyncio.create_task(self.session.run(self.inbox.get))
        self.push({"event": "start", "streamSid": "MZ1",
                   "start": {"callSid": "CA1", "customParameters": {"from": phone}}})

    def push(self, msg: dict) -> None:
        self.inbox.put_nowait(json.dumps(msg))

    def audio(self, frame: bytes, n: int) -> None:
        import base64
        for _ in range(n):
            self.push({"event": "media", "media": {"payload": base64.b64encode(frame).decode(), "timestamp": "0"}})

    def kinds(self) -> list[str]:
        return [m["event"] for m in self.out]

    def marks(self) -> list[str]:
        return [m["mark"]["name"] for m in self.out if m["event"] == "mark"]

    async def reply_done(self) -> None:
        """Wait until the agent has finished sending its reply to the CURRENT turn.

        turn_index goes up a moment before the new reply task is created, so
        "response_task is done" alone can match the previous (greeting) task and
        return too early. On a slow CI machine that race really happened. When a
        caller turn exists, require the running reply to belong to that turn.
        """
        def done() -> bool:
            s = self.session
            if s.response_task is None or not s.response_task.done():
                return False
            if s.turn_index == 0:
                return True  # greeting: no caller turn yet
            return s.current_timer is not None and s.current_timer.turn_index == s.turn_index

        await until(done)

    def echo_marks(self, limit: int | None = None) -> None:
        """Twilio echoes a mark once the audio before it has played."""
        pending = [m for m in self.marks() if m not in self.session.played_marks]
        for name in pending[:limit]:
            self.push({"event": "mark", "mark": {"name": name}})

    async def play_out(self) -> None:
        """Pretend the caller's phone finished playing everything."""
        await self.reply_done()
        self.echo_marks()
        await until(lambda: self.session.awaiting_mark is None)

    async def caller_says(self, text: str, language: str = "") -> None:
        """Speech audio, then silence (VAD end), then Deepgram's final transcript."""
        self.audio(LOUD, 15)
        self.audio(QUIET, 15)
        await until(lambda: self.session.speech_end_at is not None or self.session.turn_index > 0)
        self.stt.say(text, language)

    async def hang_up(self) -> None:
        self.push({"event": "stop"})
        await asyncio.wait_for(self.task, 3)


async def test_greeting_turn_and_hangup(repo: Repo, tmp_path: Path) -> None:
    h = Harness(repo, tmp_path, [text_chunks("Great, what food do you have today? ")])
    await h.start()
    await h.reply_done()
    assert h.tts.requests[0][0].startswith("Hi, this is HungerCue")
    await h.play_out()

    await h.caller_says("I want to donate some bagels")
    await until(lambda: h.session.turn_index == 1)
    await h.reply_done()
    assert h.llm.requests[0]["messages"][-1] == {"role": "user", "content": "I want to donate some bagels"}
    assert ("Great, what food do you have today?", "en") in h.tts.requests

    turn = h.session.metrics.turns[0]
    assert turn["v2v_ms"] is not None and turn["v2v_ms"] >= turn["llm_first_token_ms"] >= 0
    await h.hang_up()

    call = await repo.get_call("CA1")
    assert [t["role"] for t in call.transcript] == ["agent", "caller", "agent"]
    assert call.ended_at is not None and call.outcome == "no_action" and call.cost_usd > 0
    with wave.open(call.audio_path) as w:
        assert w.getnchannels() == 2 and w.getframerate() == 8000
    assert h.summaries[0].call_sid == "CA1" and h.stt.closed
    assert await h.store.load("CA1") is None
    rows = await repo.turns_since(call.started_at)
    assert len(rows) == 1 and rows[0].v2v_ms == turn["v2v_ms"]


async def test_barge_in_clears_audio_and_cancels_reply(repo: Repo, tmp_path: Path) -> None:
    long_reply = "".join(f"Sentence number {i} is here. " for i in range(20))
    h = Harness(repo, tmp_path, [text_chunks(long_reply), text_chunks("Sure, go ahead. ")],
                tts=FakeTTS(delay=0.02))
    await h.start()
    await h.play_out()
    await h.caller_says("tell me everything")
    await until(lambda: h.kinds().count("media") >= 4)  # agent has started talking
    h.audio(LOUD, 10)  # caller talks over the agent
    await until(lambda: "clear" in h.kinds())
    assert len(h.session.metrics.barge_ins) == 1
    assert h.session.metrics.barge_ins[0].reaction_ms >= 0
    await until(lambda: h.session.metrics.turns and h.session.metrics.turns[0]["interrupted"])
    # The history records what was cut off, so the LLM knows the caller didn't hear it all.
    notes = [m["content"] for m in h.session.state.messages if m["role"] == "system"][1:]
    assert notes and "did NOT hear" in notes[0]
    sent_before = h.kinds().count("media")
    await asyncio.sleep(0.15)
    assert h.kinds().count("media") == sent_before  # no more audio after the clear
    await h.hang_up()


async def test_end_call_hangs_up_after_playback(repo: Repo, tmp_path: Path) -> None:
    scripts = [tool_chunks("end_call", {"reason": "done"}), text_chunks("Thanks, goodbye! ")]
    h = Harness(repo, tmp_path, scripts)
    await h.start()
    await h.play_out()
    await h.caller_says("that's all, bye")
    await until(lambda: h.session.turn_index == 1)
    await h.reply_done()
    assert h.actions.log == []  # not until the goodbye has actually played
    await h.play_out()
    assert h.actions.log == [("hangup", "CA1")]
    await h.hang_up()


async def test_spanish_detected_on_first_turn(repo: Repo, tmp_path: Path) -> None:
    h = Harness(repo, tmp_path, [text_chunks("Claro, ¿qué comida tiene? ")])
    await h.start()
    await h.play_out()
    await h.caller_says("hola quiero donar comida", language="es")
    await until(lambda: h.session.turn_index == 1)
    await h.reply_done()
    assert h.session.state.language == "es"
    assert "Current caller language: Spanish" in h.llm.requests[0]["messages"][0]["content"]
    assert h.tts.requests[-1][1] == "es"
    await h.hang_up()
    assert (await repo.get_call("CA1")).language == "es"


async def test_stt_failure_goes_to_voicemail(repo: Repo, tmp_path: Path) -> None:
    h = Harness(repo, tmp_path, [], stt_fails=True)
    await h.start()
    await h.reply_done()
    assert "technical trouble" in h.tts.requests[0][0]
    await h.play_out()
    assert h.actions.log == [("voicemail", "CA1")]
    await h.hang_up()
    assert (await repo.get_call("CA1")).outcome == "voicemail"


@pytest.mark.parametrize("coordinator,expected", [("+12125550199", "transfer"), ("", "voicemail")])
async def test_press_zero_for_a_person(repo: Repo, tmp_path: Path, coordinator: str, expected: str) -> None:
    h = Harness(repo, tmp_path, [], coordinator_phone=coordinator)
    await h.start()
    await h.reply_done()
    h.push({"event": "dtmf", "dtmf": {"digit": "0"}})
    await until(lambda: h.actions.log)
    assert h.actions.log[0][0] == expected
    await h.hang_up()


async def test_returning_restaurant_is_greeted_by_name(repo: Repo, tmp_path: Path) -> None:
    await repo.upsert_restaurant("+12125550100", name="the bagel shop", address="200 W 112th St")
    h = Harness(repo, tmp_path, [])
    await h.start()
    await until(lambda: h.tts.requests)
    assert h.tts.requests[0][0] == "Hi, this is HungerCue. Is this another pickup from the bagel shop on 112th?"
    assert "RETURNING RESTAURANT" in h.session.state.messages[0]["content"]
    await h.hang_up()


async def test_interrupt_after_reply_written_but_still_playing(repo: Repo, tmp_path: Path) -> None:
    """The common real case: the LLM finished writing, the audio is still playing."""
    h = Harness(repo, tmp_path, [text_chunks("First part here. Second part here. Third part here. ")])
    await h.start()
    await h.play_out()
    await h.caller_says("what are the steps")
    await until(lambda: h.session.turn_index == 1)
    await h.reply_done()  # everything sent to Twilio...
    assert h.session.metrics.turns[0]["interrupted"] is False
    h.echo_marks(limit=1)  # ...but only the first sentence has played
    await until(lambda: len(h.session.played_marks) == 2)  # greeting + first sentence
    h.audio(LOUD, 10)
    await until(lambda: "clear" in h.kinds())
    note = h.session.state.messages[-1]["content"]
    assert 'They only heard: "First part here."' in note and "Third part here." in note
    assert h.session.metrics.turns[0]["interrupted"] is True
    rows = await repo.turns_since((await repo.get_call("CA1")).started_at)
    assert rows[0].interrupted  # database row corrected too
    await h.hang_up()


async def test_action_runs_immediately_when_nothing_to_play(repo: Repo, tmp_path: Path) -> None:
    # The model hangs up without saying anything: no audio, so no mark will ever come.
    h = Harness(repo, tmp_path, [tool_chunks("end_call", {"reason": "silence"}), []])
    await h.start()
    await h.play_out()
    await h.caller_says("...")
    await until(lambda: h.actions.log == [("hangup", "CA1")])
    await h.hang_up()
