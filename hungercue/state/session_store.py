"""Per-call state in Redis.

While a call is live we keep its working memory (language, who's calling,
the LLM message history) in Redis instead of in Python variables. If the
server restarts or a second instance picks up an SMS reply, the state is
still there. Each key expires after a few hours, so abandoned calls clean
themselves up.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

import fakeredis
import redis.asyncio as aioredis

SESSION_TTL_SECONDS = 4 * 60 * 60


@dataclass
class CallState:
    call_sid: str
    from_number: str = ""
    language: str = "en"  # detected on the first caller turn
    caller_type: str = ""  # restaurant | shelter | volunteer once known
    caller_type_confirmed: bool = False  # set by a saving tool; the background guess can't override it
    messages: list[dict[str, Any]] = field(default_factory=list)  # OpenAI chat history
    pickup_id: int | None = None
    outcome: str = ""


def make_redis(url: str) -> aioredis.Redis:
    """Real Redis if a URL is configured, otherwise an in-memory fake."""
    if url:
        return aioredis.from_url(url, decode_responses=True)
    return fakeredis.FakeAsyncRedis(decode_responses=True)


class SessionStore:
    def __init__(self, client: aioredis.Redis) -> None:
        self.r = client

    @staticmethod
    def _key(call_sid: str) -> str:
        return f"call:{call_sid}"

    async def load(self, call_sid: str) -> CallState | None:
        raw = await self.r.get(self._key(call_sid))
        return CallState(**json.loads(raw)) if raw else None

    async def save(self, state: CallState) -> None:
        # `ex` sets the expiry; every save pushes it back, so live calls never expire.
        await self.r.set(self._key(state.call_sid), json.dumps(asdict(state)), ex=SESSION_TTL_SECONDS)

    async def delete(self, call_sid: str) -> None:
        await self.r.delete(self._key(call_sid))
