"""Time helpers. All stored times are naive local times (see db/models.py)."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

DEFAULT_TZ = "America/New_York"


def now_local(tz: str = DEFAULT_TZ) -> datetime:
    """Current local wall-clock time with the timezone label stripped off."""
    return datetime.now(ZoneInfo(tz)).replace(tzinfo=None, microsecond=0)


def parse_hhmm(value: str) -> tuple[int, int]:
    """'09:30' -> (9, 30). Raises ValueError on bad input."""
    hours, minutes = value.strip().split(":")
    h, m = int(hours), int(minutes)
    if not (0 <= h <= 23 and 0 <= m <= 59):
        raise ValueError(f"bad time {value!r}")
    return h, m
