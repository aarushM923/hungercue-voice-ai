"""30-day raw-audio retention tests."""

from datetime import datetime, timedelta
from pathlib import Path

from hungercue.db.repo import Repo
from hungercue.privacy.retention import purge_old_audio

NOW = datetime(2026, 12, 10, 12)
REC = "https://api.twilio.com/2010-04-01/Accounts/AC1/Recordings/RE" + "a" * 32


async def make_call(repo: Repo, sid: str, age_days: int, wav: Path | None = None, **fields) -> None:
    await repo.start_call(sid, "+15550000100")
    if wav is not None:
        wav.write_bytes(b"RIFF")
        fields["audio_path"] = str(wav)
    await repo.update_call(sid, started_at=NOW - timedelta(days=age_days), **fields)


async def test_deletes_only_expired_unconsented_audio(repo: Repo, tmp_path: Path) -> None:
    old, new, kept = tmp_path / "old.wav", tmp_path / "new.wav", tmp_path / "kept.wav"
    await make_call(repo, "CA_old", 31, old)
    await make_call(repo, "CA_new", 29, new)
    await make_call(repo, "CA_kept", 90, kept, keep_audio=True)
    assert await purge_old_audio(repo, 30, now=NOW) == 1
    assert not old.exists() and new.exists() and kept.exists()
    assert (await repo.get_call("CA_old")).audio_path == ""
    assert (await repo.get_call("CA_old")).transcript == []  # transcripts are kept (none here, untouched)
    assert await purge_old_audio(repo, 30, now=NOW) == 0  # idempotent


async def test_twilio_voicemail_deleted_and_retried_on_failure(repo: Repo, tmp_path: Path) -> None:
    await make_call(repo, "CA_vm", 40, voicemail_url=REC)
    deleted: list[str] = []

    async def failing(sid: str) -> None:
        raise ConnectionError("twilio down")

    assert await purge_old_audio(repo, 30, now=NOW, delete_recording=failing) == 0
    assert (await repo.get_call("CA_vm")).voicemail_url == REC  # kept so the next run retries

    async def ok(sid: str) -> None:
        deleted.append(sid)

    assert await purge_old_audio(repo, 30, now=NOW, delete_recording=ok) == 1
    assert deleted == ["RE" + "a" * 32] and (await repo.get_call("CA_vm")).voicemail_url == ""


async def test_missing_file_is_fine(repo: Repo, tmp_path: Path) -> None:
    await make_call(repo, "CA_gone", 45, audio_path=str(tmp_path / "already-deleted.wav"))
    assert await purge_old_audio(repo, 30, now=NOW) == 1
