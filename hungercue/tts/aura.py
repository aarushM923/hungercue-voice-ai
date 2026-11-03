"""Streaming text-to-speech with Deepgram Aura.

We ask Aura for raw 8 kHz mu-law, exactly what Twilio plays, so audio bytes
flow from Deepgram to the caller with no resampling or re-encoding. The HTTP
response is *streamed*: we forward the first chunk to Twilio as soon as it
arrives (typically ~150-250 ms) instead of waiting for the whole sentence.

The call pipeline sends one request per sentence while the LLM is still
writing the next one, so the caller hears sentence 1 while sentence 2 is
being generated.
Docs: https://developers.deepgram.com/docs/streaming-text-to-speech
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol

import httpx

SPEAK_URL = "https://api.deepgram.com/v1/speak"


class TTSClient(Protocol):
    def synthesize(self, text: str, language: str) -> AsyncIterator[bytes]: ...


class AuraTTS:
    def __init__(self, api_key: str, voice_en: str, voice_es: str, client: httpx.AsyncClient | None = None) -> None:
        self.api_key = api_key
        self.voices = {"en": voice_en, "es": voice_es}
        # One shared client keeps the TLS connection open between sentences,
        # saving a ~100 ms handshake on every request after the first.
        self.client = client or httpx.AsyncClient(timeout=httpx.Timeout(10.0, connect=5.0))

    def voice_for(self, language: str) -> str:
        return self.voices.get(language, self.voices["en"])

    async def synthesize(self, text: str, language: str = "en") -> AsyncIterator[bytes]:
        """Yield mu-law audio chunks for `text` as they arrive."""
        params = {
            "model": self.voice_for(language),
            "encoding": "mulaw",
            "sample_rate": 8000,
            "container": "none",  # raw bytes, no WAV header that would click on the line
        }
        headers = {"Authorization": f"Token {self.api_key}", "Content-Type": "application/json"}
        async with self.client.stream("POST", SPEAK_URL, params=params, headers=headers, json={"text": text}) as r:
            r.raise_for_status()
            async for chunk in r.aiter_bytes():
                if chunk:
                    yield chunk

    async def aclose(self) -> None:
        await self.client.aclose()
