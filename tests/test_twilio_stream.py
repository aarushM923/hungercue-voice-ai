"""Tests for the Twilio Media Streams message helpers."""

import base64
import json

from twilio.request_validator import RequestValidator

from hungercue.telephony import twilio_stream as ts


def test_parse_start() -> None:
    raw = json.dumps({
        "event": "start", "streamSid": "MZ1",
        "start": {"callSid": "CA1", "streamSid": "MZ1", "customParameters": {"from": "+12125550100"}},
    })
    ev = ts.parse_message(raw)
    assert isinstance(ev, ts.StartEvent)
    assert (ev.stream_sid, ev.call_sid, ev.from_number) == ("MZ1", "CA1", "+12125550100")


def test_parse_media_decodes_base64() -> None:
    raw = json.dumps({"event": "media", "media": {"payload": base64.b64encode(b"\x01\x02").decode(), "timestamp": "40"}})
    ev = ts.parse_message(raw)
    assert isinstance(ev, ts.MediaEvent)
    assert ev.payload == b"\x01\x02" and ev.timestamp_ms == 40


def test_parse_mark_stop_dtmf_other() -> None:
    assert ts.parse_message('{"event":"mark","mark":{"name":"m1"}}') == ts.MarkEvent("m1")
    assert ts.parse_message('{"event":"stop","stop":{"callSid":"CA1"}}') == ts.StopEvent("CA1")
    assert ts.parse_message('{"event":"dtmf","dtmf":{"digit":"5"}}') == ts.DtmfEvent("5")
    assert ts.parse_message('{"event":"connected"}') == ts.OtherEvent("connected")


def test_outgoing_messages() -> None:
    m = json.loads(ts.media_message("MZ1", b"\xff\xff"))
    assert m == {"event": "media", "streamSid": "MZ1", "media": {"payload": "//8="}}
    assert json.loads(ts.clear_message("MZ1")) == {"event": "clear", "streamSid": "MZ1"}
    assert json.loads(ts.mark_message("MZ1", "t1"))["mark"]["name"] == "t1"


def test_stream_twiml_escapes_values() -> None:
    xml = ts.stream_twiml("wss://x/media", "+1<2>", "Hi & welcome")
    assert '<Stream url="wss://x/media">' in xml
    assert "Hi &amp; welcome" in xml and "+1&lt;2&gt;" in xml


def test_signature_validation() -> None:
    token, url, params = "secret", "https://x/voice", {"From": "+1"}
    good = RequestValidator(token).compute_signature(url, params)
    assert ts.is_valid_twilio_request(token, url, params, good)
    assert not ts.is_valid_twilio_request(token, url, params, "bad")
    assert not ts.is_valid_twilio_request("", url, params, good)
