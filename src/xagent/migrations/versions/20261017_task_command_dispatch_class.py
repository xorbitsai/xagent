"""add dispatch_class to task_execution_commands

``dispatch_class`` records whether a command was classified ``interactive`` or
``batch`` when it was accepted. Existing rows read ``batch``, which means
"not classified": the column is written once, in the acceptance transaction,
and never updated afterwards.

On PostgreSQL the column is added with a constant default, so no rewrite is
needed, but ``ALTER TABLE`` still takes an ``ACCESS EXCLUSIVE`` lock on the
table. The statement runs under a session ``lock_timeout`` of five seconds and
is attempted up to three times, waiting two and then four seconds between
attempts, so a long transaction on the table fails the migration with a clear
message instead of queueing every other statement behind it. Run the migration
with the task execution processes stopped.

Revision ID: 20261017_task_cmd_dispatch_class
Revises: 20261016_email_session_merge
Create Date: 2026-10-17

"""

import time
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "20261017_task_cmd_dispatch_class"
down_revision: Union[str, None] = "20261016_email_session_merge"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

TABLE = "task_execution_commands"
COLUMN = "dispatch_class"
_LOCK_TIMEOUT = "5s"
# Seconds to wait after the first and the second failed attempt.
_LOCK_RETRY_DELAYS = (2.0, 4.0)
_LOCK_NOT_AVAILABLE = "55P03"
_ADD_COLUMN_SQL = (
    f"ALTER TABLE {TABLE} ADD COLUMN {COLUMN} VARCHAR(16) NOT NULL DEFAULT 'batch'"
)


def _inspector() -> sa.Inspector:
    return sa.inspect(op.get_bind())


def _has_table(inspector: sa.Inspector) -> bool:
    return TABLE in inspector.get_table_names()


def _has_column(inspector: sa.Inspector) -> bool:
    return any(column["name"] == COLUMN for column in inspector.get_columns(TABLE))


def _is_lock_timeout(exc: sa.exc.DBAPIError) -> bool:
    # psycopg2 reports ``pgcode``; psycopg 3 reports ``sqlstate``.
    original = exc.orig
    code = getattr(original, "pgcode", None) or getattr(original, "sqlstate", None)
    return code == _LOCK_NOT_AVAILABLE


def _add_column_postgresql() -> None:
    with op.get_context().autocommit_block():
        op.execute(sa.text(f"SET lock_timeout = '{_LOCK_TIMEOUT}'"))
        try:
            delays = iter(_LOCK_RETRY_DELAYS)
            while True:
                try:
                    op.execute(sa.text(_ADD_COLUMN_SQL))
                    return
                except sa.exc.DBAPIError as exc:
                    if not _is_lock_timeout(exc):
                        raise
                    delay = next(delays, None)
                    if delay is None:
                        raise RuntimeError(
                            f"could not lock {TABLE}; stop task execution "
                            "processes and retry the migration"
                        ) from exc
                    time.sleep(delay)
        finally:
            op.execute(sa.text("RESET lock_timeout"))


def upgrade() -> None:
    context = op.get_context()
    if context.as_sql:
        if context.dialect.name == "postgresql":
            with context.autocommit_block():
                op.execute(sa.text(f"SET lock_timeout = '{_LOCK_TIMEOUT}'"))
                op.execute(sa.text(_ADD_COLUMN_SQL))
                op.execute(sa.text("RESET lock_timeout"))
        else:
            op.add_column(TABLE, _column())
        return

    inspector = _inspector()
    if not _has_table(inspector) or _has_column(inspector):
        return
    if context.dialect.name == "postgresql":
        _add_column_postgresql()
    else:
        op.add_column(TABLE, _column())


def _column() -> sa.Column:
    return sa.Column(
        COLUMN,
        sa.String(16),
        server_default=sa.text("'batch'"),
        nullable=False,
    )


def downgrade() -> None:
    context = op.get_context()
    if context.as_sql:
        op.drop_column(TABLE, COLUMN)
        return
    inspector = _inspector()
    if _has_table(inspector) and _has_column(inspector):
        op.drop_column(TABLE, COLUMN)
