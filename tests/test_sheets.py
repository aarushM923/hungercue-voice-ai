"""Google Sheets logger tests with an in-memory fake spreadsheet."""

from datetime import datetime
from pathlib import Path

import gspread

from hungercue.config import Settings
from hungercue.db.repo import Repo
from hungercue.pipeline.call_session import CallSummary
from hungercue.services import build_sheets, call_end_logger
from hungercue.sheets.logger import CONTACT_HEADERS, PICKUP_HEADERS, SheetsLogger


class FakeWorksheet:
    def __init__(self) -> None:
        self.rows: list[list[str]] = []

    def append_row(self, row: list[str]) -> None:
        self.rows.append(list(row))

    def col_values(self, col: int) -> list[str]:
        return [r[col - 1] for r in self.rows]

    def update(self, range_name: str, values: list[list[str]]) -> None:
        line = int(range_name.split(":")[0][1:])  # "A3:K3" -> 3
        self.rows[line - 1] = list(values[0])


class FakeSpreadsheet:
    def __init__(self, existing: tuple[str, ...] = ()) -> None:
        self.tabs = {t: FakeWorksheet() for t in existing}
        self.sheet1 = FakeWorksheet()
        self.opened = 0

    def worksheet(self, title: str) -> FakeWorksheet:
        if title not in self.tabs:
            raise gspread.exceptions.WorksheetNotFound(title)
        return self.tabs[title]

    def add_worksheet(self, title: str, rows: int, cols: int) -> FakeWorksheet:
        self.tabs[title] = FakeWorksheet()
        return self.tabs[title]


def logger_for(sheet: FakeSpreadsheet, **kw) -> SheetsLogger:
    def opener():
        sheet.opened += 1
        return sheet
    return SheetsLogger(opener, **kw)


async def test_contact_rows_keep_v1_tabs_and_columns() -> None:
    sheet = FakeSpreadsheet(existing=("Volunteers",))
    sheet.tabs["Volunteers"].append_row(CONTACT_HEADERS)
    lg = logger_for(sheet)
    assert await lg.log_contact("Volunteer", "Ana", "+15550000100", "[volunteer_updated] Harlem, weekends")
    assert await lg.log_contact("Restaurant", "Deli", "+15550000200", "notes")
    assert sheet.tabs["Volunteers"].rows[1][:4] == ["Volunteer", "Ana", "+15550000100", "[volunteer_updated] Harlem, weekends"]
    assert sheet.tabs["Restaurants"].rows[0] == CONTACT_HEADERS  # missing tab created with headers
    assert sheet.opened == 1  # spreadsheet opened once and reused


async def test_single_sheet_mode() -> None:
    sheet = FakeSpreadsheet()
    assert await logger_for(sheet, role_tabs=False).log_contact("Shelter", "Hope", "+1", "n")
    assert sheet.sheet1.rows[0][0] == "Shelter" and sheet.tabs == {}


async def test_pickup_row_is_updated_in_place(repo: Repo) -> None:
    r = await repo.upsert_restaurant("+15550000200", name="Bagel Shop")
    p = await repo.create_pickup(r.id, items="bagels", quantity="3 bags", address="200 W 112th St",
                                 pickup_start=datetime(2026, 11, 3, 15), pickup_end=datetime(2026, 11, 3, 17))
    sheet = FakeSpreadsheet()
    lg = logger_for(sheet)
    await lg.sync_pickup(await repo.get_pickup(p.id))
    await repo.update_pickup(p.id, status="accepted")
    await lg.sync_pickup(await repo.get_pickup(p.id))
    rows = sheet.tabs["Pickups"].rows
    assert rows[0] == PICKUP_HEADERS and len(rows) == 2  # header + ONE row, not two
    assert rows[1][:2] == [str(p.id), "Bagel Shop"] and rows[1][5] == "2026-11-03 15:00-17:00" and rows[1][9] == "accepted"


async def test_failures_are_swallowed_and_reconnect() -> None:
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionError("google down")
        return FakeSpreadsheet()

    lg = SheetsLogger(flaky)
    assert await lg.log_contact("Volunteer", "Ana", "+1", "x") is False  # no exception
    assert await lg.log_contact("Volunteer", "Ana", "+1", "x") is True  # recovers next time


async def test_call_end_logger_builds_notes(repo: Repo) -> None:
    await repo.upsert_volunteer("+15550000100", name="Ana Ruiz")
    sheet = FakeSpreadsheet()
    hook = call_end_logger(repo, logger_for(sheet))
    await hook(CallSummary("CA1", "+15550000100", "volunteer", "volunteer_updated", "en",
                           "agent: Hi | caller: I can help weekends | agent: Great | caller: in Harlem", None))
    row = sheet.tabs["Volunteers"].rows[1]
    assert row[:4] == ["Volunteer", "Ana Ruiz", "+15550000100", "[volunteer_updated] I can help weekends / in Harlem"]
    await hook(CallSummary("CA2", "+1", "", "no_action", "en", "", None))  # unknown caller type: skipped
    assert len(sheet.tabs["Volunteers"].rows) == 2


def test_build_sheets_off_without_credentials(tmp_path: Path) -> None:
    missing = Settings(_env_file=None, google_credentials_json=str(tmp_path / "nope.json"))
    assert build_sheets(missing) is None
    assert build_sheets(Settings(_env_file=None, sheets_enabled=False)) is None
    creds = tmp_path / "creds.json"
    creds.write_text("{}")
    assert build_sheets(Settings(_env_file=None, google_credentials_json=str(creds))) is not None
