"""Run one PostgreSQL DDL statement under a lock timeout, from an Alembic migration.

A DDL statement queues for its table lock, and every later statement on the
table queues behind it. A lock timeout bounds how long it waits for the lock,
and a bounded number of retries rides out a short blocking transaction. The
timeout does not bound the statement once it holds the lock.
"""

from __future__ import annotations

import re
import time
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

LOCK_NOT_AVAILABLE = "55P03"
# PostgreSQL reads a lock timeout of 0 as "wait forever", so only a positive
# whole number of milliseconds or seconds is accepted. The value is spliced into
# the SQL text, so nothing else may pass.
_LOCK_TIMEOUT_PATTERN = re.compile(r"[1-9][0-9]*(ms|s)")


def _check_lock_timeout(lock_timeout: str) -> None:
    if not _LOCK_TIMEOUT_PATTERN.fullmatch(lock_timeout):
        raise ValueError(
            "lock_timeout must be a positive whole number of ms or s, "
            f"got {lock_timeout!r}"
        )


def _is_lock_timeout(exc: sa.exc.DBAPIError) -> bool:
    # psycopg2 reports ``pgcode``; psycopg 3 reports ``sqlstate``.
    original = exc.orig
    code = getattr(original, "pgcode", None) or getattr(original, "sqlstate", None)
    return code == LOCK_NOT_AVAILABLE


def run_ddl_with_lock_retry(
    statement: str,
    *,
    table: str,
    lock_timeout: str = "5s",
    retry_delays: Sequence[float] = (2.0, 4.0),
) -> None:
    """Run ``statement`` with ``lock_timeout`` set, retrying only lock timeouts.

    Only valid inside an Alembic migration on a PostgreSQL connection; any other
    dialect, and offline SQL (``--sql``), raise ``ValueError`` before a statement
    runs or is emitted. The statement runs in an autocommit block, so it commits
    on its own. After each lock timeout
    the call waits for the next value of ``retry_delays`` and tries again; when
    they are used up it raises ``RuntimeError`` naming ``table``. Any other
    error is raised as it is. The session's lock timeout from before the call is
    restored afterwards, whatever the outcome.
    """
    _check_lock_timeout(lock_timeout)
    context = op.get_context()
    if context.dialect.name != "postgresql":
        raise ValueError("run_ddl_with_lock_retry only supports PostgreSQL")
    if context.as_sql:
        # Checked before the autocommit block, which would already emit COMMIT.
        raise ValueError("use offline_ddl_with_lock_timeout in --sql mode")
    with context.autocommit_block():
        bind = op.get_bind()
        previous = bind.execute(sa.text("SHOW lock_timeout")).scalar_one()
        try:
            op.execute(sa.text(f"SET lock_timeout = '{lock_timeout}'"))
            delays = iter(retry_delays)
            while True:
                try:
                    op.execute(sa.text(statement))
                    return
                except sa.exc.DBAPIError as exc:
                    if not _is_lock_timeout(exc):
                        raise
                    delay = next(delays, None)
                    if delay is None:
                        raise RuntimeError(
                            f"could not lock {table}; stop task execution "
                            "processes and retry the migration"
                        ) from exc
                    time.sleep(delay)
        finally:
            bind.execute(
                sa.text("SELECT set_config('lock_timeout', :previous, false)"),
                {"previous": previous},
            )


def offline_ddl_with_lock_timeout(statement: str, *, lock_timeout: str = "5s") -> None:
    """Emit ``statement`` between ``SET`` and ``RESET lock_timeout`` for offline SQL.

    The script is run by an operator in a new session, whose default lock
    timeout is the value from before, so ``RESET`` restores it.
    """
    _check_lock_timeout(lock_timeout)
    with op.get_context().autocommit_block():
        op.execute(sa.text(f"SET lock_timeout = '{lock_timeout}'"))
        op.execute(sa.text(statement))
        op.execute(sa.text("RESET lock_timeout"))
