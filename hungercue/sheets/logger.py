"""Google Sheets log for coordinators (carried over from v1's utils.log_to_sheets).

Coordinators already work from the "HungerCue Calls" spreadsheet, so v2 keeps
its tabs and columns exactly (Volunteers / Restaurants / Shelters with
Type, Name, Phone, Notes, Timestamp) and adds a Pickups tab with one row per
pickup that is updated in place as its status changes.

gspread is a *blocking* library (each call waits on Google's API), so every
call runs in a worker thread via asyncio.to_thread; otherwise one slow Sheets
request would freeze live call audio. And every public method swallows its
errors: a Sheets outage must never break a phone call or a text.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import Any

from hungercue.db.models import Pickup
from hungercue.timeutil import now_local

log = logging.getLogger(__name__)

CONTACT_HEADERS = ["Type", "Name", "Phone", "Notes", "Timestamp"]  # unchanged from v1
TAB_BY_ROLE = {"Volunteer": "Volunteers", "Restaurant": "Restaurants", "Shelter": "Shelters"}
PICKUP_TAB = "Pickups"
PICKUP_HEADERS = [
    "Pickup ID", "Restaurant", "Restaurant Phone", "Items", "Quantity", "Pickup Window",
    "Address", "Shelter", "Volunteer", "Status", "Updated",
]


def pickup_row(p: Pickup, updated: str) -> list[str]:
    window = f"{p.pickup_start:%Y-%m-%d %H:%M}-{p.pickup_end:%H:%M}"
    return [
        str(p.id),
        p.restaurant.name if p.restaurant else "",
        p.restaurant.phone if p.restaurant else "",
        p.items, p.quantity, window, p.address,
        p.shelter.name if p.shelter else "",
        p.volunteer.name if p.volunteer else "",
        p.status, updated,
    ]


class SheetsLogger:
    def __init__(
        self,
        open_spreadsheet: Callable[[], Any],
        role_tabs: bool = True,
        tz: str = "America/New_York",
    ) -> None:
        """`open_spreadsheet` returns a gspread Spreadsheet; injected so tests can fake it."""
        self._open = open_spreadsheet
        self.role_tabs = role_tabs
        self.tz = tz
        self._sheet: Any = None
        self._tabs: dict[str, Any] = {}
        self._lock = asyncio.Lock()  # one write at a time keeps row order sane

    # ---------- blocking helpers (run in a thread) ----------

    def _worksheet(self, title: str, headers: list[str]) -> Any:
        """Get a tab, creating it with a header row if it doesn't exist (as v1 did)."""
        if title in self._tabs:
            return self._tabs[title]
        if self._sheet is None:
            self._sheet = self._open()
        import gspread  # local import: only needed when Sheets is actually used

        try:
            ws = self._sheet.worksheet(title)
        except gspread.exceptions.WorksheetNotFound:
            ws = self._sheet.add_worksheet(title=title, rows=1000, cols=len(headers))
            ws.append_row(headers)
        self._tabs[title] = ws
        return ws

    def _append_contact(self, role: str, name: str, phone: str, notes: str) -> None:
        ts = now_local(self.tz).strftime("%Y-%m-%d %H:%M:%S")
        row = [role, name, phone, notes, ts]
        if self.role_tabs:
            self._worksheet(TAB_BY_ROLE.get(role, "Volunteers"), CONTACT_HEADERS).append_row(row)
        else:
            if self._sheet is None:
                self._sheet = self._open()
            self._sheet.sheet1.append_row(row)

    def _upsert_pickup(self, row: list[str]) -> None:
        ws = self._worksheet(PICKUP_TAB, PICKUP_HEADERS)
        ids = ws.col_values(1)  # column A holds pickup IDs; index 0 is the header
        if row[0] in ids:
            line = ids.index(row[0]) + 1  # Sheets rows are 1-based
            ws.update(range_name=f"A{line}:K{line}", values=[row])
        else:
            ws.append_row(row)

    # ---------- async API used by the app ----------

    async def log_contact(self, role: str, name: str, phone: str, notes: str) -> bool:
        return await self._run(self._append_contact, role, name, phone, notes)

    async def sync_pickup(self, pickup: Pickup) -> bool:
        updated = now_local(self.tz).strftime("%Y-%m-%d %H:%M:%S")
        return await self._run(self._upsert_pickup, pickup_row(pickup, updated))

    async def _run(self, fn: Callable[..., None], *args: Any) -> bool:
        async with self._lock:
            try:
                await asyncio.to_thread(fn, *args)
                return True
            except Exception:
                log.exception("Google Sheets write failed (%s)", fn.__name__)
                self._tabs.clear()  # stale handles after an error; reconnect next time
                self._sheet = None
                return False


def open_from_settings(credentials_json: str, sheet_id: str, sheet_name: str) -> Callable[[], Any]:
    """Factory matching v1: service-account file, open by ID if given else by name."""

    def _open() -> Any:
        import gspread

        gc = gspread.service_account(filename=credentials_json)
        return gc.open_by_key(sheet_id) if sheet_id else gc.open(sheet_name)

    return _open
