"""Streaming speech-to-text with Deepgram.

We open one WebSocket to Deepgram per call and forward Twilio's mu-law audio
as-is (Deepgram accepts 8 kHz mu-law directly, so no conversion). Deepgram
sends back transcripts *while the caller is still talking*:

  is_final=False  -> interim guess, may still change ("I have three tr...")
  is_final=True   -> this stretch of words is locked in
  speech_final    -> Deepgram's own endpointing thinks the caller paused
  from_finalize   -> the final result we forced by sending "Finalize"

Latency: Deepgram's own endpointing proved slow to fire (0.4-3 s after the
caller stopped, in the first eval run). Our Silero VAD knows within ~250 ms,
so when it hears the end of speech we send {"type": "Finalize"}, which makes
Deepgram return the final transcript for everything it has heard right away.

Starting the LLM on the final text the instant the caller stops (instead of
uploading a whole recording afterwards) is what makes sub-second replies possible.

`language=multi` turns on Nova-3's English/Spanish code-switching, and each
result reports which language was heard; we use that on the first turn.
Docs: https://developers.deepgram.com/docs/live-streaming-audio
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import urlencode

LISTEN_URL = "wss://api.deepgram.com/v1/listen"


@dataclass
class Transcript:
    text: str
    is_final: bool
    speech_final: bool
    language: str = ""
    utterance_end: bool = False  # Deepgram's backup "they've stopped" signal
    from_finalize: bool = False  # this final result answers our Finalize request


class STTStream(Protocol):
    """Anything that accepts audio and yields transcripts (real Deepgram or a test fake)."""

    async def start(self) -> None: ...
    async def send(self, mulaw_audio: bytes) -> None: ...
    async def finalize(self) -> None: ...
    def events(self) -> AsyncIterator[Transcript]: ...
    async def close(self) -> None: ...


def build_listen_url(model: str, endpointing_ms: int = 300, language: str = "multi") -> str:
    params = {
        "model": model,
        "encoding": "mulaw",
        "sample_rate": 8000,
        "channels": 1,
        "interim_results": "true",
        "endpointing": endpointing_ms,  # ms of silence before speech_final
        "utterance_end_ms": 1000,  # word-gap based fallback, robust to background noise
        "smart_format": "true",  # "three pm" -> "3 PM", numbers and addresses formatted
        "language": language,
    }
    return f"{LISTEN_URL}?{urlencode(params)}"


def parse_message(raw: str | bytes) -> Transcript | None:
    """Convert one Deepgram JSON message into a Transcript (None for metadata)."""
    msg: dict[str, Any] = json.loads(raw)
    kind = msg.get("type")
    if kind == "UtteranceEnd":
        return Transcript(text="", is_final=True, speech_final=True, utterance_end=True)
    if kind != "Results":
        return None
    alternatives = msg.get("channel", {}).get("alternatives", [])
    if not alternatives:
        return None
    alt = alternatives[0]
    languages = alt.get("languages") or []
    return Transcript(
        text=alt.get("transcript", "").strip(),
        is_final=bool(msg.get("is_final")),
        speech_final=bool(msg.get("speech_final")),
        language=languages[0] if languages else "",
        from_finalize=bool(msg.get("from_finalize")),
    )


class _WebSocket(Protocol):
    async def send(self, message: str | bytes) -> None: ...
    async def close(self) -> None: ...
    def __aiter__(self) -> AsyncIterator[str | bytes]: ...


Connector = Callable[[str, dict[str, str]], Awaitable[_WebSocket]]


async def _default_connect(url: str, headers: dict[str, str]) -> _WebSocket:
    from websockets.asyncio.client import connect

    return await connect(url, additional_headers=headers, max_size=None)  # type: ignore[return-value]


class DeepgramSTT:
    def __init__(
        self,
        api_key: str,
        model: str = "nova-3",
        endpointing_ms: int = 300,
        language: str = "multi",
        connect: Connector = _default_connect,
    ) -> None:
        self.url = build_listen_url(model, endpointing_ms, language)
        self.headers = {"Authorization": f"Token {api_key}"}
        self._connect = connect
        self.ws: _WebSocket | None = None

    async def start(self) -> None:
        self.ws = await self._connect(self.url, self.headers)

    async def send(self, mulaw_audio: bytes) -> None:
        if self.ws is not None:
            await self.ws.send(mulaw_audio)  # binary frame = raw audio

    async def finalize(self) -> None:
        """Ask Deepgram to finalise what it has heard so far, immediately."""
        if self.ws is not None:
            await self.ws.send(json.dumps({"type": "Finalize"}))

    async def events(self) -> AsyncIterator[Transcript]:
        if self.ws is None:
            return
        async for raw in self.ws:
            t = parse_message(raw)
            if t is not None:
                yield t

    async def close(self) -> None:
        if self.ws is not None:
            try:
                # Ask Deepgram to flush any last words before closing.
                await self.ws.send(json.dumps({"type": "CloseStream"}))
                await self.ws.close()
            except Exception:
                pass  # already closed by the other side; nothing to clean up
            self.ws = None
