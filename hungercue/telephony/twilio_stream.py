"""Twilio Media Streams protocol: parse incoming messages, build outgoing ones.

When a call connects, our /voice webhook returns TwiML containing
<Connect><Stream url="wss://.../media"/></Connect>. Twilio then opens a
WebSocket to /media and sends JSON messages:

  connected -> start (call info) -> media, media, media ... (20 ms audio chunks) -> stop

We can send back:
  media  - audio for the caller to hear (base64 mu-law, 8 kHz)
  mark   - a named bookmark; Twilio echoes it back once all audio before it has PLAYED,
           which is how we know when the agent has actually finished speaking
  clear  - throw away all queued audio immediately (used for barge-in)

Docs: https://www.twilio.com/docs/voice/media-streams/websocket-messages
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from typing import Any
from xml.sax.saxutils import escape

from twilio.request_validator import RequestValidator


@dataclass
class StartEvent:
    stream_sid: str
    call_sid: str
    from_number: str = ""
    custom_parameters: dict[str, str] = field(default_factory=dict)


@dataclass
class MediaEvent:
    payload: bytes  # raw mu-law bytes (already base64-decoded)
    timestamp_ms: int  # milliseconds since the stream started, per Twilio
    track: str = "inbound"


@dataclass
class MarkEvent:
    name: str


@dataclass
class DtmfEvent:
    digit: str


@dataclass
class StopEvent:
    call_sid: str = ""


@dataclass
class OtherEvent:
    event: str


TwilioEvent = StartEvent | MediaEvent | MarkEvent | DtmfEvent | StopEvent | OtherEvent


def parse_message(raw: str) -> TwilioEvent:
    """Turn one JSON text frame from Twilio into a typed event object."""
    msg: dict[str, Any] = json.loads(raw)
    event = msg.get("event", "")
    if event == "start":
        start = msg.get("start", {})
        params = start.get("customParameters", {}) or {}
        return StartEvent(
            stream_sid=msg.get("streamSid") or start.get("streamSid", ""),
            call_sid=start.get("callSid", ""),
            from_number=params.get("from", ""),
            custom_parameters=params,
        )
    if event == "media":
        media = msg.get("media", {})
        return MediaEvent(
            payload=base64.b64decode(media.get("payload", "")),
            timestamp_ms=int(media.get("timestamp", 0) or 0),
            track=media.get("track", "inbound"),
        )
    if event == "mark":
        return MarkEvent(name=msg.get("mark", {}).get("name", ""))
    if event == "dtmf":
        return DtmfEvent(digit=msg.get("dtmf", {}).get("digit", ""))
    if event == "stop":
        return StopEvent(call_sid=msg.get("stop", {}).get("callSid", ""))
    return OtherEvent(event=event)


def media_message(stream_sid: str, mulaw_audio: bytes) -> str:
    """Build a message that plays audio to the caller."""
    return json.dumps(
        {
            "event": "media",
            "streamSid": stream_sid,
            "media": {"payload": base64.b64encode(mulaw_audio).decode("ascii")},
        }
    )


def mark_message(stream_sid: str, name: str) -> str:
    """Build a mark message; Twilio echoes it back when playback reaches this point."""
    return json.dumps({"event": "mark", "streamSid": stream_sid, "mark": {"name": name}})


def clear_message(stream_sid: str) -> str:
    """Build a clear message: Twilio drops all audio it has buffered but not yet played."""
    return json.dumps({"event": "clear", "streamSid": stream_sid})


def stream_twiml(ws_url: str, from_number: str, notice: str) -> str:
    """TwiML for an incoming call: play the recording notice, then open the media stream.

    The caller's number is passed as a custom <Parameter> because the WebSocket
    'start' message does not otherwise include it.
    """
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Response>"
        f"<Say>{escape(notice)}</Say>"
        "<Connect>"
        f'<Stream url="{escape(ws_url)}">'
        f'<Parameter name="from" value="{escape(from_number)}"/>'
        "</Stream>"
        "</Connect>"
        "</Response>"
    )


def is_valid_twilio_request(auth_token: str, url: str, params: dict[str, str], signature: str) -> bool:
    """Check the X-Twilio-Signature header so strangers can't post fake calls or SMS.

    Twilio signs every webhook with HMAC-SHA1 of the URL plus the sorted form
    parameters, keyed with our auth token. If the signature doesn't match, the
    request did not come from Twilio.
    """
    if not auth_token or not signature:
        return False
    return bool(RequestValidator(auth_token).validate(url, params, signature))
