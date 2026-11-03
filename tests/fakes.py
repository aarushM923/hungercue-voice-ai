"""Reusable fakes for tests (and the offline parts of the eval harness)."""

from __future__ import annotations

import json
from types import SimpleNamespace as NS
from typing import Any


def text_chunks(text: str, size: int = 7) -> list[Any]:
    """Split text into streaming chunks like the OpenAI API sends."""
    parts = [text[i : i + size] for i in range(0, len(text), size)]
    return [NS(usage=None, choices=[NS(delta=NS(content=p, tool_calls=None))]) for p in parts]


def tool_chunks(name: str, args: dict[str, Any], call_id: str = "call_1") -> list[Any]:
    """A tool call whose JSON arguments arrive in 3 fragments, as in real streams."""
    raw = json.dumps(args)
    third = len(raw) // 3 + 1
    frags = [raw[i : i + third] for i in range(0, len(raw), third)]
    out = [NS(usage=None, choices=[NS(delta=NS(content=None, tool_calls=[
        NS(index=0, id=call_id, function=NS(name=name, arguments=frags[0]))]))])]
    for f in frags[1:]:
        out.append(NS(usage=None, choices=[NS(delta=NS(content=None, tool_calls=[
            NS(index=0, id=None, function=NS(name=None, arguments=f))]))]))
    return out


def usage_chunk(prompt: int = 100, completion: int = 20, cached: int = 0) -> Any:
    return NS(usage=NS(prompt_tokens=prompt, completion_tokens=completion,
                       prompt_tokens_details=NS(cached_tokens=cached)), choices=[])


class FakeStream:
    def __init__(self, chunks: list[Any]) -> None:
        self.chunks = chunks

    def __aiter__(self):
        async def gen():
            for c in self.chunks:
                yield c
        return gen()


class FakeOpenAI:
    """Replays one scripted list of chunks per create() call and records requests."""

    def __init__(self, scripts: list[list[Any]]) -> None:
        self.scripts = list(scripts)
        self.requests: list[dict[str, Any]] = []
        self.chat = NS(completions=NS(create=self._create))

    async def _create(self, **kwargs: Any) -> FakeStream:
        # Snapshot messages: the agent mutates the list after the call.
        self.requests.append({**kwargs, "messages": [dict(m) for m in kwargs["messages"]]})
        return FakeStream(self.scripts.pop(0))


# ---------------------------------------------------------------- call pipeline fakes

import asyncio  # noqa: E402

import numpy as np  # noqa: E402

from hungercue.audio.vad import VadSegmenter  # noqa: E402
from hungercue.stt.deepgram import Transcript  # noqa: E402

LOUD = bytes([0x00, 0x80] * 80)  # 20 ms of max-amplitude mu-law: "speech" for the fake VAD
QUIET = b"\xff" * 160  # 20 ms of mu-law silence


def loudness_vad(**kw: Any) -> VadSegmenter:
    """A VAD whose 'model' is just loudness, so tests control speech exactly."""
    return VadSegmenter(lambda chunk: 0.95 if float(np.abs(chunk).mean()) > 0.1 else 0.01, **kw)


class FakeSTT:
    def __init__(self) -> None:
        self.queue: asyncio.Queue[Transcript | None] = asyncio.Queue()
        self.frames = 0
        self.closed = False

    async def start(self) -> None:
        pass

    async def send(self, audio: bytes) -> None:
        self.frames += 1

    async def events(self):
        while (t := await self.queue.get()) is not None:
            yield t

    async def close(self) -> None:
        self.closed = True
        await self.queue.put(None)

    def say(self, text: str, language: str = "") -> None:
        """Simulate Deepgram finalising an utterance and detecting end of speech."""
        self.queue.put_nowait(Transcript(text, is_final=True, speech_final=True, language=language))


class FakeTTS:
    def __init__(self, delay: float = 0.0, chunks_per_text: int = 2) -> None:
        self.requests: list[tuple[str, str]] = []
        self.delay = delay
        self.chunks_per_text = chunks_per_text

    async def synthesize(self, text: str, language: str = "en"):
        self.requests.append((text, language))
        for _ in range(self.chunks_per_text):
            if self.delay:
                await asyncio.sleep(self.delay)
            yield b"\x7f" * 160


class FakeActions:
    def __init__(self) -> None:
        self.log: list[tuple[str, ...]] = []

    async def transfer(self, call_sid: str, to_number: str, say: str) -> None:
        self.log.append(("transfer", call_sid, to_number))

    async def voicemail(self, call_sid: str, say: str, callback_url: str) -> None:
        self.log.append(("voicemail", call_sid))

    async def hangup(self, call_sid: str) -> None:
        self.log.append(("hangup", call_sid))


async def until(cond, timeout: float = 3.0) -> None:
    """Wait until cond() is true (polling), failing the test after `timeout` seconds."""
    loop = asyncio.get_running_loop()
    end = loop.time() + timeout
    while not cond():
        if loop.time() > end:
            raise AssertionError("condition not met in time")
        await asyncio.sleep(0.005)
