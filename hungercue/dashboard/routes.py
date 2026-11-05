"""Coordinator dashboard: today's pickups, match status, calls with transcripts
and audio, the SMS log, and a latency chart.

Access control (the spec asks for transcripts in access-controlled storage):
- HTTP Basic auth with DASHBOARD_USER / DASHBOARD_PASSWORD. If no password is
  set the dashboard doesn't exist (404): safe by default.
- Passwords are compared with secrets.compare_digest, which takes the same
  time whether the first or last character is wrong, so attackers can't
  guess the password one character at a time by timing responses.
- Browsers resend Basic credentials automatically, so another website could
  try to submit our forms on a coordinator's behalf (CSRF). POSTs therefore
  must come from a page whose Origin/Referer is this same host.
- Call audio is served only through this authenticated route, never as a
  public static file.
"""

from __future__ import annotations

import json
import secrets
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, Response
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.templating import Jinja2Templates

from hungercue.config import Settings
from hungercue.db.models import Turn
from hungercue.pipeline.latency import percentile
from hungercue.timeutil import now_local

TEMPLATES = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
basic = HTTPBasic(auto_error=False)

# Pickup status -> (label, icon). Status is never shown by color alone.
STATUS = {
    "pending_match": ("Finding match", "○"),
    "offered": ("Offered, awaiting YES", "◔"),
    "accepted": ("Volunteer confirmed", "✓"),
    "unmatched": ("Needs a coordinator", "!"),
    "completed": ("Completed", "✓"),
    "cancelled": ("Cancelled", "×"),
}
SEGMENTS = [  # (key, legend label) in stack order, bottom to top: the order time is spent
    ("stt_ms", "Speech-to-text"),
    ("endpoint_ms", "Endpointing"),
    ("llm_first_token_ms", "LLM first token"),
    ("tts_first_byte_ms", "TTS first byte"),
    ("other_ms", "Other"),
]
V2V_TARGET_P50_MS = 800
V2V_TARGET_P95_MS = 1500


def latency_rows(turns: list[Turn]) -> list[dict[str, Any]]:
    """One row per turn with every stack segment filled in.

    The measured stages don't add up exactly to voice-to-voice (e.g. waiting for
    the first full sentence), so the remainder is shown as "Other" rather than
    silently dropped; the column height then always equals the real v2v.
    """
    rows = []
    for t in turns:
        if t.v2v_ms is None:
            continue
        parts = {k: float(getattr(t, k) or 0.0) for k, _ in SEGMENTS[:-1]}
        parts["other_ms"] = max(0.0, t.v2v_ms - sum(parts.values()))
        rows.append({
            "label": f"{t.call_sid[-4:]}#{t.turn_index}",
            "call_sid": t.call_sid,
            "at": t.created_at.strftime("%b %d %H:%M"),
            "v2v_ms": round(t.v2v_ms),
            "interrupted": t.interrupted,
            **{k: round(v) for k, v in parts.items()},
        })
    return rows


def latency_stats(turns: list[Turn]) -> dict[str, Any]:
    values = [t.v2v_ms for t in turns if t.v2v_ms is not None]
    return {
        "count": len(values),
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "target_p50": V2V_TARGET_P50_MS,
        "target_p95": V2V_TARGET_P95_MS,
    }


def build_router(settings: Settings) -> APIRouter:
    router = APIRouter(prefix="/dashboard")

    def require_login(creds: HTTPBasicCredentials | None = Depends(basic)) -> str:
        if not settings.dashboard_password:
            raise HTTPException(status_code=404)
        ok = creds is not None and secrets.compare_digest(
            creds.username.encode(), settings.dashboard_user.encode()
        ) & secrets.compare_digest(creds.password.encode(), settings.dashboard_password.encode())
        if not ok:
            raise HTTPException(status_code=401, headers={"WWW-Authenticate": 'Basic realm="HungerCue"'})
        return creds.username  # type: ignore[union-attr]

    def require_same_origin(request: Request) -> None:
        source = request.headers.get("origin") or request.headers.get("referer") or ""
        if urlparse(source).netloc != request.url.netloc:
            raise HTTPException(status_code=403, detail="cross-site form post rejected")

    def container(request: Request) -> Any:
        return request.app.state.container

    @router.get("", response_class=HTMLResponse)
    async def index(request: Request, day: str = "", _: str = Depends(require_login)) -> Response:
        c = container(request)
        try:
            date = datetime.strptime(day, "%Y-%m-%d") if day else now_local(settings.timezone)
        except ValueError:
            raise HTTPException(status_code=400, detail="day must be YYYY-MM-DD")
        start = date.replace(hour=0, minute=0, second=0, microsecond=0)
        pickups = await c.repo.pickups_between(start, start + timedelta(days=1))
        chart_turns = await c.repo.recent_turns(40)
        stats_turns = await c.repo.recent_turns(500)
        rows = latency_rows(chart_turns)
        counts: dict[str, int] = {}
        for p in pickups:
            counts[p.status] = counts.get(p.status, 0) + 1
        return TEMPLATES.TemplateResponse(request, "index.html", {
            "day": start, "prev_day": start - timedelta(days=1), "next_day": start + timedelta(days=1),
            "pickups": pickups, "status": STATUS, "counts": counts,
            "calls": await c.repo.recent_calls(25),
            "sms": await c.repo.recent_sms(25),
            "sms_enabled": settings.sms_enabled,
            "stats": latency_stats(stats_turns),
            "rows": rows,
            # Embedded inside <script>; escaping "</" means no value can end the tag early.
            "chart_json": json.dumps({"rows": rows, "segments": SEGMENTS, "target": V2V_TARGET_P50_MS}).replace(
                "</", "<\\/"
            ),
        })

    @router.get("/calls/{call_sid}", response_class=HTMLResponse)
    async def call_detail(request: Request, call_sid: str, _: str = Depends(require_login)) -> Response:
        c = container(request)
        call = await c.repo.get_call(call_sid)
        if call is None:
            raise HTTPException(status_code=404)
        turns = [t for t in await c.repo.turns_since(call.started_at) if t.call_sid == call_sid]
        has_audio = bool(call.audio_path) and Path(call.audio_path).exists()
        return TEMPLATES.TemplateResponse(request, "call.html", {
            "call": call, "turns": latency_rows(turns), "has_audio": has_audio,
            "retention_days": settings.audio_retention_days,
        })

    @router.get("/calls/{call_sid}/audio")
    async def call_audio(request: Request, call_sid: str, _: str = Depends(require_login)) -> Response:
        call = await container(request).repo.get_call(call_sid)
        if call is None or not call.audio_path or not Path(call.audio_path).exists():
            raise HTTPException(status_code=404)
        # Only paths the app itself recorded are ever stored, but double-check the
        # file lives inside the recordings folder before serving it.
        path = Path(call.audio_path).resolve()
        if Path(settings.recordings_dir).resolve() not in path.parents:
            raise HTTPException(status_code=404)
        return FileResponse(path, media_type="audio/wav", headers={"Cache-Control": "private, no-store"})

    @router.post("/calls/{call_sid}/keep-audio")
    async def keep_audio(
        request: Request, call_sid: str, _: str = Depends(require_login), __: None = Depends(require_same_origin)
    ) -> Response:
        """Record that the caller agreed to keep their audio past 30 days (or undo it)."""
        c = container(request)
        call = await c.repo.get_call(call_sid)
        if call is None:
            raise HTTPException(status_code=404)
        await c.repo.update_call(call_sid, keep_audio=not call.keep_audio)
        return RedirectResponse(f"/dashboard/calls/{call_sid}", status_code=303)

    @router.post("/rebalance")
    async def rebalance(
        request: Request, _: str = Depends(require_login), __: None = Depends(require_same_origin)
    ) -> Response:
        """Re-optimise today's unaccepted pickups with the Hungarian algorithm, then send offers."""
        c = container(request)
        today = now_local(settings.timezone)
        _result, needs_offer = await c.matcher.rebalance_day(today)
        for pickup_id in needs_offer:
            await c.followup.send_offer(pickup_id)
        return RedirectResponse("/dashboard", status_code=303)

    return router
