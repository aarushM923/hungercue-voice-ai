"""Tests for the FastAPI entry points (milestone 1: TwiML + echo)."""

import base64
import json

from fastapi.testclient import TestClient
from twilio.request_validator import RequestValidator

from hungercue.config import Settings
from hungercue.main import create_app


def make_client(**overrides: object) -> TestClient:
    s = Settings(_env_file=None, public_base_url="https://hc.example", twilio_auth_token="tok", **overrides)
    return TestClient(create_app(s))


def test_healthz() -> None:
    assert make_client().get("/healthz").json() == {"ok": True}


def test_voice_rejects_unsigned_request() -> None:
    assert make_client().post("/voice", data={"From": "+1"}).status_code == 403


def test_voice_returns_stream_twiml_when_signed() -> None:
    params = {"From": "+12125550100", "CallSid": "CA1"}
    sig = RequestValidator("tok").compute_signature("https://hc.example/voice", params)
    r = make_client().post("/voice", data=params, headers={"X-Twilio-Signature": sig})
    assert r.status_code == 200
    assert '<Stream url="wss://hc.example/media">' in r.text
    assert 'value="+12125550100"' in r.text
    assert "recorded" in r.text  # privacy notice is played first


def test_media_echo() -> None:
    client = make_client()
    with client.websocket_connect("/media") as ws:
        ws.send_text(json.dumps({"event": "start", "streamSid": "MZ1", "start": {"callSid": "CA1"}}))
        payload = base64.b64encode(b"\x10\x20").decode()
        ws.send_text(json.dumps({"event": "media", "media": {"payload": payload, "timestamp": "0"}}))
        echoed = json.loads(ws.receive_text())
        assert echoed["media"]["payload"] == payload and echoed["streamSid"] == "MZ1"
        ws.send_text(json.dumps({"event": "stop"}))
