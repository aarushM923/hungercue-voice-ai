"""Sending text messages.

Every text goes through `Texter.send`, which ALWAYS logs it to the database
(so coordinators see it on the dashboard) and only actually sends it through
Twilio when SMS_ENABLED is on. SMS from our toll-free number is blocked by US
carriers until Twilio's toll-free verification is approved, so until then
messages are saved with status "pending_verification".
"""

from __future__ import annotations

import asyncio
import logging
from typing import Protocol

from twilio.rest import Client

from hungercue.db.repo import Repo

log = logging.getLogger(__name__)


class SmsGateway(Protocol):
    async def send(self, to: str, body: str) -> None: ...


class TwilioSms:
    def __init__(self, account_sid: str, auth_token: str, from_number: str) -> None:
        self.client = Client(account_sid, auth_token)
        self.from_number = from_number

    async def send(self, to: str, body: str) -> None:
        # Twilio's library blocks while waiting for the HTTP reply; run it in a thread.
        await asyncio.to_thread(lambda: self.client.messages.create(to=to, from_=self.from_number, body=body))


class Texter:
    def __init__(self, repo: Repo, gateway: SmsGateway | None, enabled: bool) -> None:
        self.repo = repo
        self.gateway = gateway
        self.enabled = enabled and gateway is not None

    async def send(self, to: str, body: str, kind: str, pickup_id: int | None = None) -> str:
        """Send (or queue as pending) one text. Returns the status recorded."""
        if not to or to.startswith("unknown:"):
            status = "failed"
            log.warning("no phone number for %s text (pickup %s)", kind, pickup_id)
        elif not self.enabled:
            status = "pending_verification"
            log.info("SMS disabled, logged only -> %s: %s", to, body)
        else:
            try:
                assert self.gateway is not None
                await self.gateway.send(to, body)
                status = "sent"
            except Exception:
                log.exception("SMS to %s failed", to)
                status = "failed"
        await self.repo.log_sms("outbound", to, body, kind=kind, pickup_id=pickup_id, status=status)
        return status
