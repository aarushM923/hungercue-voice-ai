"""Builds the real services the app uses (the "wiring" in one place).

Each external dependency (OpenAI, Deepgram, Twilio, Google Sheets) is only
created if its credentials are configured, so the server can start (and the
tests and dashboard can run) without every key in place.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

from hungercue.audio.vad import SileroVAD, VadSegmenter, load_session
from hungercue.config import Settings
from hungercue.db.repo import Repo
from hungercue.db.session import Database
from hungercue.matching.geo import Geocoder
from hungercue.matching.service import Matcher
from hungercue.notify.followup import FollowUp
from hungercue.notify.sms import Texter, TwilioSms
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
    matcher: Matcher
    followup: FollowUp
    geocoder: Geocoder
    call_services: Services | None = None  # None until AI keys are configured
    background: list[asyncio.Task[None]] = field(default_factory=list)  # sweeper etc.

    def start_background(self) -> None:
        self.background.append(asyncio.create_task(self.followup.run_sweeper()))

    async def close(self) -> None:
        # Cancel background tasks AND wait for them to finish before closing the
        # database. A task cancelled mid-query makes SQLAlchemy discard its
        # connection in a follow-up task; if we closed the database (and the event
        # loop) without waiting, that cleanup could be left waiting forever on a
        # driver thread that had already exited. That was an intermittent shutdown hang.
        for task in self.background:
            task.cancel()
        await asyncio.gather(*self.background, return_exceptions=True)
        self.background.clear()
        tts =getattr(self.call_services, "tts", None) if self.call_services else None
        if isinstance(tts, AuraTTS):
            await tts.aclose()
        await self.geocoder.aclose()
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
    matcher = Matcher(repo, settings)
    has_twilio = bool(settings.twilio_account_sid and settings.twilio_auth_token)
    gateway = (
        TwilioSms(settings.twilio_account_sid, settings.twilio_auth_token, settings.twilio_phone_number)
        if has_twilio
        else None
    )
    texter = Texter(repo, gateway, enabled=settings.sms_enabled)
    followup = FollowUp(repo, matcher, texter, settings)
    container = AppContainer(
        settings=settings, db=db, repo=repo, store=store, matcher=matcher, followup=followup, geocoder=Geocoder()
    )

    missing = missing_call_keys(settings)
    if missing:
        log.warning("AI keys missing (%s): calls will run in echo mode", ", ".join(missing))
        return container

    from openai import AsyncOpenAI  # imported lazily: only needed once keys exist

    vad_session = load_session()  # load the ONNX model once for all calls
    actions = TwilioCallActions(settings.twilio_account_sid, settings.twilio_auth_token) if has_twilio else None

    async def on_donation(pickup_id: int) -> None:
        # Matching + texting runs in the background so the caller isn't kept waiting.
        followup.donation_logged_background(pickup_id)

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
        geocode=container.geocoder,
        on_donation=on_donation,
    )
    return container
