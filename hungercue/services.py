"""Builds the real services the app uses (the "wiring" in one place).

Each external dependency (OpenAI, Deepgram, Twilio, Google Sheets) is only
created if its credentials are configured, so the server can start (and the
tests and dashboard can run) without every key in place.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from hungercue.audio.vad import SileroVAD, VadSegmenter, load_session
from hungercue.config import Settings
from hungercue.db.repo import Repo
from hungercue.db.session import Database
from hungercue.pipeline.call_session import Services
from hungercue.state.session_store import SessionStore, make_redis
from hungercue.stt.deepgram import DeepgramSTT
from hungercue.telephony.actions import TwilioCallActions
from hungercue.tts.aura import AuraTTS

log = logging.getLogger(__name__)


@dataclass
class AppContainer:
    """Long-lived objects shared by every request, created once at startup."""

    settings: Settings
    db: Database
    repo: Repo
    store: SessionStore
    call_services: Services | None = None  # None until AI keys are configured
    extras: dict[str, Any] = field(default_factory=dict)  # matcher, notifier, sheets, scheduler

    async def close(self) -> None:
        tts = getattr(self.call_services, "tts", None) if self.call_services else None
        if isinstance(tts, AuraTTS):
            await tts.aclose()
        await self.db.close()


def missing_call_keys(settings: Settings) -> list[str]:
    """Which env vars a real (non-echo) call still needs."""
    needed = {"OPENAI_API_KEY": settings.openai_api_key, "DEEPGRAM_API_KEY": settings.deepgram_api_key}
    return [name for name, value in needed.items() if not value]


async def build_container(settings: Settings) -> AppContainer:
    db = Database(settings.database_url)
    await db.create_tables()
    repo = Repo(db)
    store = SessionStore(make_redis(settings.redis_url))
    container = AppContainer(settings=settings, db=db, repo=repo, store=store)

    missing = missing_call_keys(settings)
    if missing:
        log.warning("AI keys missing (%s): calls will run in echo mode", ", ".join(missing))
        return container

    from openai import AsyncOpenAI  # imported lazily: only needed once keys exist

    vad_session = load_session()  # load the ONNX model once for all calls
    actions = (
        TwilioCallActions(settings.twilio_account_sid, settings.twilio_auth_token)
        if settings.twilio_account_sid and settings.twilio_auth_token
        else None
    )
    container.call_services = Services(
        settings=settings,
        repo=repo,
        store=store,
        llm_client=AsyncOpenAI(api_key=settings.openai_api_key),
        tts=AuraTTS(settings.deepgram_api_key, settings.tts_voice_en, settings.tts_voice_es),
        stt_factory=lambda: DeepgramSTT(
            settings.deepgram_api_key, settings.stt_model, endpointing_ms=settings.stt_endpointing_ms
        ),
        vad_factory=lambda: VadSegmenter(
            SileroVAD(session=vad_session),
            min_speech_ms=settings.vad_min_speech_ms,
            min_silence_ms=settings.vad_min_silence_ms,
        ),
        actions=actions,
    )
    return container
