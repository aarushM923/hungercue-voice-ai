"""Deepgram STT tests with a fake WebSocket (no network, no key)."""

import json
from urllib.parse import parse_qs, urlparse

from hungercue.stt.deepgram import DeepgramSTT, Transcript, build_listen_url, parse_message


def results(text: str, is_final: bool = True, speech_final: bool = False, langs: list | None = None) -> str:
    alt = {"transcript": text}
    if langs:
        alt["languages"] = langs
    return json.dumps({"type": "Results", "is_final": is_final, "speech_final": speech_final,
                       "channel": {"alternatives": [alt]}})


def test_url_has_phone_audio_settings() -> None:
    q = parse_qs(urlparse(build_listen_url("nova-3", 250)).query)
    assert q["encoding"] == ["mulaw"] and q["sample_rate"] == ["8000"]
    assert q["endpointing"] == ["250"] and q["interim_results"] == ["true"] and q["language"] == ["multi"]


def test_parse_messages() -> None:
    assert parse_message(results(" hola ", True, True, ["es"])) == Transcript("hola", True, True, "es")
    assert parse_message(results("I have", False)) == Transcript("I have", False, False, "")
    assert parse_message('{"type":"UtteranceEnd"}').utterance_end
    assert parse_message('{"type":"Metadata"}') is None
    assert parse_message('{"type":"Results","channel":{"alternatives":[]}}') is None


class FakeWS:
    def __init__(self, incoming: list[str]) -> None:
        self.incoming, self.sent, self.closed = incoming, [], False

    async def send(self, m: object) -> None:
        self.sent.append(m)

    async def close(self) -> None:
        self.closed = True

    def __aiter__(self):
        async def gen():
            for m in self.incoming:
                yield m
        return gen()


async def test_stream_round_trip() -> None:
    fake = FakeWS([results("hi", False), '{"type":"Metadata"}', results("hi there", True, True)])
    seen = {}

    async def connect(url: str, headers: dict) -> FakeWS:
        seen.update(url=url, headers=headers)
        return fake

    stt = DeepgramSTT("KEY", connect=connect)
    await stt.send(b"ignored before start")
    await stt.start()
    assert seen["headers"] == {"Authorization": "Token KEY"}
    await stt.send(b"\xff" * 160)
    got = [t async for t in stt.events()]
    assert [t.text for t in got] == ["hi", "hi there"] and got[-1].speech_final
    await stt.close()
    assert fake.sent == [b"\xff" * 160, '{"type": "CloseStream"}'] and fake.closed
