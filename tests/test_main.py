"""Tests for the FastAPI entry points and app wiring."""

import base64
import json
from pathlib import Path

from fastapi.testclient import TestClient
from twilio.request_validator import RequestValidator

from hungercue.config import Settings
from hungercue.main import create_app
from hungercue.pipeline.call_session import Services
from hungercue.services import build_container, missing_call_keys
from tests.fakes import FakeOpenAI, FakeSTT, FakeTTS, loudness_vad, text_chunks


def settings(tmp_path: Path, **overrides: object) -> Settings:
    return Settings(_env_file=None, public_base_url="https://hc.example", twilio_auth_token="tok",
                    database_url=f"sqlite+aiosqlite:///{tmp_path / 'app.db'}",
                    recordings_dir=str(tmp_path / "rec"), **overrides)


def signed(path: str, params: dict) -> dict:
    return {"X-Twilio-Signature": RequestValidator("tok").compute_signature(f"https://hc.example{path}", params)}


def test_healthz(tmp_path: Path) -> None:
    with TestClient(create_app(settings(tmp_path))) as c:
        assert c.get("/healthz").json() == {"ok": True}


def test_voice_rejects_unsigned_request(tmp_path: Path) -> None:
    with TestClient(create_app(settings(tmp_path))) as c:
        assert c.post("/voice", data={"From": "+1"}).status_code == 403


def test_voice_returns_stream_twiml_when_signed(tmp_path: Path) -> None:
    params = {"From": "+12125550100", "CallSid": "CA1"}
    with TestClient(create_app(settings(tmp_path))) as c:
        r = c.post("/voice", data=params, headers=signed("/voice", params))
    assert r.status_code == 200
    assert '<Stream url="wss://hc.example/media">' in r.text
    assert 'value="+12125550100"' in r.text
    assert "recorded" in r.text  # privacy notice is played first


def test_media_echoes_without_keys(tmp_path: Path) -> None:
    with TestClient(create_app(settings(tmp_path))) as client:
        assert client.app.state.container.call_services is None
        with client.websocket_connect("/media") as ws:
            ws.send_text(json.dumps({"event": "start", "streamSid": "MZ1", "start": {"callSid": "CA1"}}))
            payload = base64.b64encode(b"\x10\x20").decode()
            ws.send_text(json.dumps({"event": "media", "media": {"payload": payload, "timestamp": "0"}}))
            echoed = json.loads(ws.receive_text())
            assert echoed["media"]["payload"] == payload and echoed["streamSid"] == "MZ1"
            ws.send_text(json.dumps({"event": "stop"}))


def test_voicemail_callback_saves_recording(tmp_path: Path) -> None:
    params = {"CallSid": "CA9", "RecordingUrl": "https://api.twilio.com/rec/RE1"}
    with TestClient(create_app(settings(tmp_path))) as c:
        repo = c.app.state.container.repo
        c.portal.call(repo.start_call, "CA9", "+12125550100")  # run async code from a sync test
        assert c.post("/voicemail", data=params, headers=signed("/voicemail", params)).status_code == 200
        call = c.portal.call(repo.get_call, "CA9")
    assert call.voicemail_url.endswith("RE1") and call.outcome == "voicemail"


def test_media_runs_real_call_session_with_services(tmp_path: Path) -> None:
    """The full /media route with fake AI services: greeting audio comes back."""
    cfg = settings(tmp_path)

    async def builder(s: Settings):
        c = await build_container(s)
        c.call_services = Services(s, c.repo, c.store, FakeOpenAI([text_chunks("Hi. ")]), FakeTTS(),
                                   FakeSTT, loudness_vad)
        return c

    with TestClient(create_app(cfg, builder)) as client:
        with client.websocket_connect("/media") as ws:
            ws.send_text(json.dumps({"event": "start", "streamSid": "MZ1",
                                     "start": {"callSid": "CA1", "customParameters": {"from": "+12125550100"}}}))
            first = json.loads(ws.receive_text())
            assert first["event"] == "media" and first["streamSid"] == "MZ1"
            ws.send_text(json.dumps({"event": "stop"}))
        call = client.portal.call(client.app.state.container.repo.get_call, "CA1")
        assert call.transcript[0]["text"].startswith("Hi, this is HungerCue")


async def test_build_container_with_keys_creates_services(tmp_path: Path) -> None:
    s = settings(tmp_path, openai_api_key="sk-test", deepgram_api_key="dg-test")
    assert missing_call_keys(s) == []
    c = await build_container(s)
    assert c.call_services is not None and c.call_services.actions is None  # no Twilio creds given
    vad = c.call_services.vad_factory()
    assert vad.prob_fn.session is not None
    await c.close()
    assert missing_call_keys(settings(tmp_path)) == ["OPENAI_API_KEY", "DEEPGRAM_API_KEY"]
