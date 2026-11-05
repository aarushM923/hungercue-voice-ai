"""App container lifecycle tests."""

import asyncio
from pathlib import Path

from hungercue.config import Settings
from hungercue.services import build_container


async def test_close_leaves_no_stray_tasks_even_mid_sweep(tmp_path: Path) -> None:
    """Regression test for an intermittent shutdown hang.

    The sweeper runs its first sweep immediately, so closing right after
    startup cancels it mid-query. close() must wait for that cancellation
    (including SQLAlchemy discarding the connection) to finish before
    closing the database; otherwise a cleanup task is left behind and event
    loop shutdown can wait on it forever.
    """
    for i in range(25):
        s = Settings(_env_file=None, database_url=f"sqlite+aiosqlite:///{tmp_path / f'{i}.db'}",
                     recordings_dir=str(tmp_path))
        c = await build_container(s)
        await c.repo.upsert_volunteer("+15550000100", name="Ana")  # give the sweep real work
        c.start_background()
        for _ in range(i % 5):  # vary how far into the sweep we get before closing
            await asyncio.sleep(0)
        await c.close()
        stray = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        assert stray == [], stray
