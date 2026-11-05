"""Dashboard tests: access control, pages, audio, consent, rebalance, chart data."""

from datetime import datetime, timedelta
from pathlib import Path

from fastapi.testclient import TestClient

from hungercue.config import Settings
from hungercue.dashboard.routes import latency_rows, latency_stats
from hungercue.db.models import Turn
from hungercue.main import create_app
from hungercue.timeutil import now_local

AUTH = ("coordinator", "s3cret")
SAME_ORIGIN = {"origin": "http://testserver"}


def client(tmp_path: Path, **kw) -> TestClient:
    s = Settings(_env_file=None, database_url=f"sqlite+aiosqlite:///{tmp_path / 'd.db'}",
                 recordings_dir=str(tmp_path / "rec"), dashboard_password=kw.pop("password", "s3cret"), **kw)
    return TestClient(create_app(s))


def test_disabled_without_password(tmp_path: Path) -> None:
    with client(tmp_path, password="") as c:
        assert c.get("/dashboard", auth=AUTH).status_code == 404


def test_requires_correct_login(tmp_path: Path) -> None:
    with client(tmp_path) as c:
        r = c.get("/dashboard")
        assert r.status_code == 401 and "Basic" in r.headers["www-authenticate"]
        assert c.get("/dashboard", auth=("coordinator", "wrong")).status_code == 401
        assert c.get("/dashboard", auth=("someone", "s3cret")).status_code == 401
        assert c.get("/dashboard", auth=AUTH).status_code == 200


def test_index_shows_pickups_sms_banner_and_chart(tmp_path: Path) -> None:
    with client(tmp_path) as c:
        repo = c.app.state.container.repo
        today = now_local().replace(hour=15, minute=0, second=0)

        async def seed():
            r = await repo.upsert_restaurant("+15550000200", name="Bagel <Shop>", address="200 W 112th St")
            p = await repo.create_pickup(r.id, items="bagels", quantity="3 bags",
                                         pickup_start=today, pickup_end=today + timedelta(hours=2))
            await repo.update_pickup(p.id, status="unmatched")
            await repo.start_call("CA1", "+15550000200")
            await repo.add_turn("CA1", 1, stt_ms=100, endpoint_ms=20, llm_first_token_ms=300, tts_first_byte_ms=150, v2v_ms=650)
            await repo.log_sms("outbound", "+15550000200", "hello", kind="offer", status="pending_verification")

        c.portal.call(seed)
        html = c.get("/dashboard", auth=AUTH).text
    assert "Bagel &lt;Shop&gt;" in html  # escaped: names come from callers
    assert "Needs a coordinator" in html and "pending verification" in html
    assert "Texting is pending toll-free verification" in html
    assert "650 ms" in html and "✓ under" in html and '"v2v_ms": 650' in html


def test_bad_day_param(tmp_path: Path) -> None:
    with client(tmp_path) as c:
        assert c.get("/dashboard?day=tomorrow", auth=AUTH).status_code == 400
        assert c.get("/dashboard?day=2026-11-03", auth=AUTH).status_code == 200


def test_call_page_audio_and_consent(tmp_path: Path) -> None:
    with client(tmp_path) as c:
        repo = c.app.state.container.repo
        wav = tmp_path / "rec" / "CA1.wav"
        wav.parent.mkdir()
        wav.write_bytes(b"RIFF....WAVE")

        async def seed():
            await repo.start_call("CA1", "+15550000100")
            await repo.add_transcript("CA1", "caller", "I have <b>soup</b>")
            await repo.update_call("CA1", audio_path=str(wav))
            await repo.start_call("CA2", "+1")
            await repo.update_call("CA2", audio_path="/etc/passwd")  # never served: outside recordings

        c.portal.call(seed)
        page = c.get("/dashboard/calls/CA1", auth=AUTH).text
        assert "I have &lt;b&gt;soup&lt;/b&gt;" in page and "<audio" in page
        audio = c.get("/dashboard/calls/CA1/audio", auth=AUTH)
        assert audio.status_code == 200 and audio.headers["content-type"] == "audio/wav"
        assert "no-store" in audio.headers["cache-control"]
        assert c.get("/dashboard/calls/CA1/audio").status_code == 401
        assert c.get("/dashboard/calls/CA2/audio", auth=AUTH).status_code == 404
        assert c.get("/dashboard/calls/NOPE", auth=AUTH).status_code == 404

        # Consent toggle: rejected cross-site, accepted from our own page.
        assert c.post("/dashboard/calls/CA1/keep-audio", auth=AUTH, headers={"origin": "https://evil.example"}).status_code == 403
        assert c.post("/dashboard/calls/CA1/keep-audio", auth=AUTH).status_code == 403  # no origin at all
        r = c.post("/dashboard/calls/CA1/keep-audio", auth=AUTH, headers=SAME_ORIGIN, follow_redirects=False)
        assert r.status_code == 303
        assert c.portal.call(repo.get_call, "CA1").keep_audio is True


def test_rebalance_button(tmp_path: Path) -> None:
    with client(tmp_path) as c:
        repo = c.app.state.container.repo
        today = now_local().replace(hour=15, minute=0, second=0)

        async def seed():
            await repo.upsert_shelter("+1000", name="Hope", lat=40.78, lon=-73.97, receiving_start="00:00", receiving_end="23:59")
            await repo.upsert_volunteer("+2000", name="Ana", lat=40.80, lon=-73.95, has_car=True)
            r = await repo.upsert_restaurant("+3000", name="Deli")
            p = await repo.create_pickup(r.id, items="soup", pickup_start=today, pickup_end=today + timedelta(hours=1),
                                         lat=40.80, lon=-73.96)
            return p.id

        pid = c.portal.call(seed)
        r = c.post("/dashboard/rebalance", auth=AUTH, headers=SAME_ORIGIN, follow_redirects=False)
        assert r.status_code == 303
        p = c.portal.call(repo.get_pickup, pid)
        sms = c.portal.call(repo.recent_sms)
    assert p.status == "offered" and p.volunteer.name == "Ana"
    assert sms[0].kind == "offer" and sms[0].status == "pending_verification"


def test_latency_rows_fill_other_and_skip_unmeasured() -> None:
    t = Turn(call_sid="CA12345", turn_index=2, stt_ms=100, endpoint_ms=None, llm_first_token_ms=300,
             tts_first_byte_ms=150, v2v_ms=700, interrupted=False, created_at=datetime(2026, 11, 3, 9))
    skip = Turn(call_sid="CA1", turn_index=3, v2v_ms=None, created_at=datetime(2026, 11, 3, 9))
    rows = latency_rows([t, skip])
    assert len(rows) == 1
    r = rows[0]
    assert r["endpoint_ms"] == 0 and r["other_ms"] == 150 and r["label"] == "2345#2"
    assert sum(r[k] for k in ("stt_ms", "endpoint_ms", "llm_first_token_ms", "tts_first_byte_ms", "other_ms")) == r["v2v_ms"]
    stats = latency_stats([t, skip])
    assert stats["count"] == 1 and stats["p50"] == 700
