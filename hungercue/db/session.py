"""Database connection setup."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from hungercue.db.models import Base


class Database:
    """Owns the connection pool and hands out short-lived sessions.

    A *session* is one unit of work: open it, read/write, commit, close.
    `expire_on_commit=False` keeps loaded objects readable after commit,
    which we need because we often return a row and then use it elsewhere.
    """

    def __init__(self, url: str) -> None:
        self.engine = create_async_engine(url)
        self.sessionmaker = async_sessionmaker(self.engine, expire_on_commit=False)

    async def create_tables(self) -> None:
        """Create any missing tables (safe to call on every startup)."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        """`async with db.session() as s:` commits on success, rolls back on error."""
        async with self.sessionmaker() as s:
            try:
                yield s
                await s.commit()
            except Exception:
                await s.rollback()
                raise

    async def close(self) -> None:
        await self.engine.dispose()
