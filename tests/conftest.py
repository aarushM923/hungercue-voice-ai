"""Shared pytest fixtures."""

from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from hungercue.db.repo import Repo
from hungercue.db.session import Database


@pytest.fixture
async def db(tmp_path: Path) -> AsyncIterator[Database]:
    """A brand-new SQLite database file per test, deleted afterwards by pytest."""
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'test.db'}")
    await database.create_tables()
    yield database
    await database.close()


@pytest.fixture
def repo(db: Database) -> Repo:
    return Repo(db)
