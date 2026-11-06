"""Aura TTS tests with respx faking the HTTP endpoint (no network, no key)."""

import httpx
import pytest
import respx

from hungercue.tts.aura import SPEAK_URL, AuraTTS


@respx.mock
async def test_streams_mulaw_and_picks_voice() -> None:
    route = respx.post(SPEAK_URL).mock(return_value=httpx.Response(200, content=b"\x01\x02\x03"))
    tts = AuraTTS("KEY", "aura-en", "aura-es")
    chunks = [c async for c in tts.synthesize("Hola", "es")]
    assert b"".join(chunks) == b"\x01\x02\x03"
    req = route.calls.last.request
    assert req.url.params["model"] == "aura-es" and req.url.params["encoding"] == "mulaw"
    assert req.url.params["sample_rate"] == "8000" and req.headers["Authorization"] == "Token KEY"
    assert tts.voice_for("fr") == "aura-en"  # unknown language falls back to English
    await tts.aclose()


@respx.mock
async def test_http_error_raises() -> None:
    respx.post(SPEAK_URL).mock(return_value=httpx.Response(401))
    tts = AuraTTS("BAD", "a", "b")
    with pytest.raises(httpx.HTTPStatusError):
        _ = [c async for c in tts.synthesize("hi")]


@respx.mock
async def test_retries_once_on_dropped_connection_before_any_audio() -> None:
    route = respx.post(SPEAK_URL).mock(side_effect=[
        httpx.RemoteProtocolError("Server disconnected without sending a response."),
        httpx.Response(200, content=b"\x01\x02"),
    ])
    tts = AuraTTS("KEY", "a", "b")
    assert b"".join([c async for c in tts.synthesize("hi")]) == b"\x01\x02"
    assert route.call_count == 2


@respx.mock
async def test_gives_up_after_second_failure() -> None:
    respx.post(SPEAK_URL).mock(side_effect=httpx.ConnectError("down"))
    with pytest.raises(httpx.ConnectError):
        _ = [c async for c in AuraTTS("KEY", "a", "b").synthesize("hi")]
