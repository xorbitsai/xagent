"""Keep authentication fixture Sessions owned entirely by the auth worker."""

from collections.abc import AsyncIterator, Callable, Generator
from contextlib import contextmanager

from sqlalchemy.orm import Session

from xagent.web.models.auth_database import SyncAuthSessionFactory


def auth_db_override(
    provider: Callable[[], Generator[Session, None, None]],
) -> Callable[[], AsyncIterator[SyncAuthSessionFactory]]:
    async def override() -> AsyncIterator[SyncAuthSessionFactory]:
        yield contextmanager(provider)

    return override
