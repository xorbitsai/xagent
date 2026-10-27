"""add requested_dispatch_class to tasks

``requested_dispatch_class`` records the dispatch class the caller declared when
it created the task, exactly as sent. It is nullable with no default, so
existing rows read NULL (no declaration) and PostgreSQL adds the column by
changing the catalog only, without rewriting the table.

``ALTER TABLE`` still takes an ``ACCESS EXCLUSIVE`` lock on ``tasks``, and lease
renewals write that table continuously. Each statement runs under a session
``lock_timeout`` of five seconds and is attempted up to three times, waiting two
and then four seconds between attempts, so a long transaction on the table fails
the migration with a clear message instead of queueing every other statement
behind it. Dropping the column on downgrade takes the same lock and follows the
same rules. Stop the task execution processes before running the migration.

Revision ID: 20261026_task_requested_dispatch_class
Revises: 20261026_task_cmd_unfinished_idx
Create Date: 2026-10-26

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

from xagent.migrations.ddl_lock_retry import (
    offline_ddl_with_lock_timeout,
    run_ddl_with_lock_retry,
)

revision: str = "20261026_task_requested_dispatch_class"
down_revision: Union[str, None] = "20261026_task_cmd_unfinished_idx"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

TABLE = "tasks"
COLUMN = "requested_dispatch_class"
_LOCK_TIMEOUT = "5s"
# Seconds to wait after the first and the second lock timeout.
_LOCK_RETRY_DELAYS = (2.0, 4.0)
_ADD_COLUMN_SQL = f"ALTER TABLE {TABLE} ADD COLUMN {COLUMN} VARCHAR(16)"
_DROP_COLUMN_SQL = f"ALTER TABLE {TABLE} DROP COLUMN {COLUMN}"


def _has_table() -> bool:
    return TABLE in sa.inspect(op.get_bind()).get_table_names()


def _has_column() -> bool:
    inspector = sa.inspect(op.get_bind())
    return any(column["name"] == COLUMN for column in inspector.get_columns(TABLE))


def _run_ddl(statement: str) -> None:
    run_ddl_with_lock_retry(
        statement,
        table=TABLE,
        lock_timeout=_LOCK_TIMEOUT,
        retry_delays=_LOCK_RETRY_DELAYS,
    )


def upgrade() -> None:
    context = op.get_context()
    if context.as_sql:
        if context.dialect.name == "postgresql":
            offline_ddl_with_lock_timeout(_ADD_COLUMN_SQL, lock_timeout=_LOCK_TIMEOUT)
        else:
            op.add_column(TABLE, sa.Column(COLUMN, sa.String(16), nullable=True))
        return

    if not _has_table() or _has_column():
        return
    if context.dialect.name == "postgresql":
        _run_ddl(_ADD_COLUMN_SQL)
    else:
        op.add_column(TABLE, sa.Column(COLUMN, sa.String(16), nullable=True))


def downgrade() -> None:
    context = op.get_context()
    if context.as_sql:
        if context.dialect.name == "postgresql":
            offline_ddl_with_lock_timeout(_DROP_COLUMN_SQL, lock_timeout=_LOCK_TIMEOUT)
        else:
            op.drop_column(TABLE, COLUMN)
        return

    if not _has_table() or not _has_column():
        return
    if context.dialect.name == "postgresql":
        _run_ddl(_DROP_COLUMN_SQL)
    else:
        op.drop_column(TABLE, COLUMN)
