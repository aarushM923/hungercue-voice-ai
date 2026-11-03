"""Mid-call actions through Twilio's REST API: transfer, voicemail, hang up.

While a <Connect><Stream> is running, Twilio is "inside" our WebSocket. To do
something else with the call we *update* the live call with new TwiML; Twilio
then closes the stream and follows the new instructions.

The Twilio Python library is synchronous (it blocks while waiting for the
HTTP reply), so we run it with asyncio.to_thread. Otherwise one slow request
would freeze audio for every call on this server.
"""

from __future__ import annotations

import asyncio
from typing import Protocol
from xml.sax.saxutils import escape

from twilio.rest import Client


class CallActions(Protocol):
    async def transfer(self, call_sid: str, to_number: str, say: str) -> None: ...
    async def voicemail(self, call_sid: str, say: str, callback_url: str) -> None: ...
    async def hangup(self, call_sid: str) -> None: ...


def transfer_twiml(to_number: str, say: str) -> str:
    # The agent already said "connecting you now", so `say` is usually empty;
    # an empty <Say> is invalid TwiML, so only include it when there's text.
    say_xml = f"<Say>{escape(say)}</Say>" if say else ""
    return f"<Response>{say_xml}<Dial>{escape(to_number)}</Dial></Response>"


def voicemail_twiml(say: str, callback_url: str) -> str:
    # <Record> plays a beep, records up to 2 minutes, and posts the recording URL to callback_url.
    return (
        f"<Response><Say>{escape(say)}</Say>"
        f'<Record maxLength="120" playBeep="true" recordingStatusCallback="{escape(callback_url)}"/>'
        "</Response>"
    )


class TwilioCallActions:
    def __init__(self, account_sid: str, auth_token: str) -> None:
        self.client = Client(account_sid, auth_token)

    async def _update(self, call_sid: str, **kwargs: str) -> None:
        await asyncio.to_thread(lambda: self.client.calls(call_sid).update(**kwargs))

    async def transfer(self, call_sid: str, to_number: str, say: str) -> None:
        await self._update(call_sid, twiml=transfer_twiml(to_number, say))

    async def voicemail(self, call_sid: str, say: str, callback_url: str) -> None:
        await self._update(call_sid, twiml=voicemail_twiml(say, callback_url))

    async def hangup(self, call_sid: str) -> None:
        await self._update(call_sid, status="completed")
