"""Raw-audio retention: delete call audio after 30 days unless the caller agreed otherwise.

What counts as raw audio:
- our stereo call recordings (recordings/<CallSid>.wav), and
- voicemails, which Twilio stores on its side (we only keep the URL).

Transcripts are text and are kept; they're only visible through the
password-protected dashboard. A coordinator can tick "caller agreed to keep
audio" on a call (keep_audio=True), and this job then leaves that call alone.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta
from pathlib import Path

from hungercue.db.repo import Repo
from hungercue.timeutil import now_local

log = logging.getLogger(__name__)

RecordingDeleter = Callable[[str], Awaitable[None]]  # takes a Twilio recording SID (RE...)

RECORDING_SID = re.compile(r"(RE[0-9a-f]{32})")


def twilio_recording_deleter(account_sid: str, auth_token: str) -> RecordingDeleter:
    from twilio.rest import Client

    client = Client(account_sid, auth_token)

    async def delete(recording_sid: str) -> None:
        # The Twilio library blocks while it waits on the network, so run it in a thread.
        await asyncio.to_thread(lambda: client.recordings(recording_sid).delete())

    return delete


async def purge_old_audio(
    repo: Repo,
    days: int,
    now: datetime | None = None,
    delete_recording: RecordingDeleter | None = None,
) -> int:
    """Delete expired audio and clear the references. Returns how many calls were purged."""
    cutoff = (now or now_local()) - timedelta(days=days)
    purged = 0
    for call in await repo.calls_with_audio_before(cutoff):
        changes: dict[str, str] = {}
        if call.audio_path:
            Path(call.audio_path).unlink(missing_ok=True)  # already gone is fine
            changes["audio_path"] = ""
        if call.voicemail_url:
            m = RECORDING_SID.search(call.voicemail_url)
            if m and delete_recording is not None:
                try:
                    await delete_recording(m.group(1))
                    changes["voicemail_url"] = ""
                except Exception:
                    # Keep the URL so the next run retries; never lose track of audio we hold.
                    log.exception("could not delete Twilio recording for %s", call.call_sid)
            elif m is None:
                changes["voicemail_url"] = ""  # not a Twilio recording URL; nothing to delete there
        if changes:
            await repo.update_call(call.call_sid, **changes)
            purged += 1
    if purged:
        log.info("retention: purged audio from %d calls older than %s", purged, cutoff)
    return purged


async def run_retention(
    repo: Repo, days: int, delete_recording: RecordingDeleter | None, interval_s: float = 3600.0
) -> None:
    """Hourly loop started with the app (cheap: one indexed query when nothing is due)."""
    while True:
        try:
            await purge_old_audio(repo, days, delete_recording=delete_recording)
        except Exception:
            log.exception("retention run failed")
        await asyncio.sleep(interval_s)
