"""store the structured result of a task's latest completed run

``tasks.structured_output_record`` is a nullable ``json`` column with no
default. NULL means the latest completed run produced no structured result,
and every existing row reads NULL.

It is ``json``, not ``jsonb``, on PostgreSQL: the column keeps the text it was
given, so the key order of the stored output survives a read.

On PostgreSQL both statements, on upgrade and downgrade, run through
``run_ddl_with_lock_retry``: a five-second ``lock_timeout``, up to three
attempts, two and then four seconds apart. Adding a nullable column without a
default does not rewrite the table; the lock retry bounds the wait for the
``ACCESS EXCLUSIVE`` lock. SQLite adds the column with one ``ADD COLUMN``; a
batch rebuild is not used because it drops the ``AUTOINCREMENT`` of ``tasks``.

Downgrade drops the column and the structured results in it.

Revision ID: 20261105_task_structured_output_record
Revises: 20261107_split_tool_categories
Create Date: 2026-11-05

"""

from typing import Callable, Sequence, Union

import sqlalchemy as sa
from alembic import op

from xagent.migrations.ddl_lock_retry import (
    offline_ddl_with_lock_timeout,
    run_ddl_with_lock_retry,
)

revision: str = "20261105_task_structured_output_record"
down_revision: Union[str, None] = "20261107_split_tool_categories"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

TABLE = "tasks"
COLUMN = "structured_output_record"
_LOCK_TIMEOUT = "5s"
# Seconds to wait after the first and the second lock timeout.
_LOCK_RETRY_DELAYS = (2.0, 4.0)

ADD_COLUMN_SQL = f"ALTER TABLE {TABLE} ADD COLUMN {COLUMN} JSON"
DROP_COLUMN_SQL = f"ALTER TABLE {TABLE} DROP COLUMN {COLUMN}"


def _columns() -> set[str] | None:
    """The column names of ``tasks`` now, or None without it; never cached."""
    inspector = sa.inspect(op.get_bind())
    if TABLE not in inspector.get_table_names():
        return None
    return {column["name"] for column in inspector.get_columns(TABLE)}


def _apply(statement: str, fallback: Callable[[], None]) -> None:
    """Run ``statement`` under the lock retry on PostgreSQL, else ``fallback``."""
    if op.get_context().dialect.name != "postgresql":
        fallback()
        return
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
            offline_ddl_with_lock_timeout(ADD_COLUMN_SQL, lock_timeout=_LOCK_TIMEOUT)
        else:
            op.add_column(TABLE, sa.Column(COLUMN, sa.JSON(), nullable=True))
        return

    columns = _columns()
    if columns is None or COLUMN in columns:
        return
    _apply(
        ADD_COLUMN_SQL,
        lambda: op.add_column(TABLE, sa.Column(COLUMN, sa.JSON(), nullable=True)),
    )


def downgrade() -> None:
    context = op.get_context()
    if context.as_sql:
        if context.dialect.name == "postgresql":
            offline_ddl_with_lock_timeout(DROP_COLUMN_SQL, lock_timeout=_LOCK_TIMEOUT)
        else:
            op.drop_column(TABLE, COLUMN)
        return

    if COLUMN not in (_columns() or set()):
        return
    _apply(DROP_COLUMN_SQL, lambda: op.drop_column(TABLE, COLUMN))
