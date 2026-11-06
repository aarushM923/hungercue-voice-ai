"""Builds the real services the app uses (the "wiring" in one place).

Each external dependency (OpenAI, Deepgram, Twilio, Google Sheets) is only
created if its credentials are configured, so the server can start (and the
tests and dashboard can run) without every key in place.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path

from hungercue.audio.vad import SileroVAD, VadSegmenter, load_session
from hungercue.config import Settings
from hungercue.db.repo import Repo
from hungercue.db.session import Database
from hungercue.llm.classify import make_classifier
from hungercue.matching.geo import Geocoder
from hungercue.matching.service import Matcher
from hungercue.notify.followup import FollowUp
from hungercue.notify.sms import Texter, TwilioSms
from hungercue.pipeline.call_session import CallSummary, Services
from hungercue.privacy.retention import RecordingDeleter, run_retention, twilio_recording_deleter
from hungercue.sheets.logger import SheetsLogger, open_from_settings
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
    sheets: SheetsLogger | None = None  # None when Sheets isn't configured
    recording_deleter: RecordingDeleter | None = None  # deletes Twilio voicemails after 30 days
    call_services: Services | None = None  # None until AI keys are configured
    background: list[asyncio.Task[None]] = field(default_factory=list)  # sweeper etc.

    def start_background(self) -> None:
        self.background.append(asyncio.create_task(self.followup.run_sweeper()))
        self.background.append(
            asyncio.create_task(run_retention(self.repo, self.settings.audio_retention_days, self.recording_deleter))
        )

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


def build_sheets(settings: Settings) -> SheetsLogger | None:
    """The coordinator spreadsheet logger, or None if it isn't set up on this machine."""
    if not settings.sheets_enabled:
        return None
    if not Path(settings.google_credentials_json).exists():
        log.info("Google Sheets off: %s not found", settings.google_credentials_json)
        return None
    opener = open_from_settings(settings.google_credentials_json, settings.google_sheet_id, settings.google_sheet_name)
    return SheetsLogger(opener, role_tabs=settings.role_tabs, tz=settings.timezone)


def call_end_logger(repo: Repo, sheets: SheetsLogger) -> Callable[[CallSummary], Awaitable[None]]:
    """After each call, add a row to the caller's tab, like v1 did.

    Notes = the call outcome plus what the caller said, so a coordinator can
    see at a glance what happened without opening the dashboard.
    """

    async def hook(summary: CallSummary) -> None:
        if not summary.caller_type:
            return  # caller hung up before saying who they were; nothing useful to log
        profile = await repo.lookup_caller(summary.from_number)
        said = [part[len("caller: "):] for part in summary.transcript_text.split(" | ") if part.startswith("caller: ")]
        notes = f"[{summary.outcome}] " + " / ".join(said)
        await sheets.log_contact(summary.caller_type.title(), profile.name or "Unknown", summary.from_number, notes[:1000])

    return hook


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
    sheets = build_sheets(settings)
    followup = FollowUp(repo, matcher, texter, settings, on_pickup_change=sheets.sync_pickup if sheets else None)
    container = AppContainer(
        settings=settings, db=db, repo=repo, store=store, matcher=matcher, followup=followup,
        geocoder=Geocoder(), sheets=sheets,
        recording_deleter=(
            twilio_recording_deleter(settings.twilio_account_sid, settings.twilio_auth_token) if has_twilio else None
        ),
    )

    missing = missing_call_keys(settings)
    if missing:
        log.warning("AI keys missing (%s): calls will run in echo mode", ", ".join(missing))
        return container

    from openai import AsyncOpenAI  # imported lazily: only needed once keys exist

    vad_session = load_session()  # load the ONNX model once for all calls
    actions = TwilioCallActions(settings.twilio_account_sid, settings.twilio_auth_token) if has_twilio else None

    llm_client = AsyncOpenAI(api_key=settings.openai_api_key)

    async def on_donation(pickup_id: int) -> None:
        # Matching + texting runs in the background so the caller isn't kept waiting.
        followup.donation_logged_background(pickup_id)

    container.call_services = Services(
        settings=settings,
        repo=repo,
        store=store,
        llm_client=llm_client,
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
        on_call_end=call_end_logger(repo, sheets) if sheets else None,
        classifier=make_classifier(llm_client, settings.llm_model),
    )
    return container
