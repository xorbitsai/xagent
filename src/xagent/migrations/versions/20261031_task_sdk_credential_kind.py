"""record which kind of SDK credential created a task

``tasks.sdk_credential_kind`` and ``expired_task_tombstones.sdk_credential_kind``
are ``VARCHAR(16)``, nullable, with no default. NULL means the task was not
created by a personal key (a runtime key, or a path that is not the SDK);
``personal`` means a personal key created it. Existing rows read NULL, so every
runtime key keeps reaching exactly the tasks it reached before. A named CHECK on
each table admits only those two values.

On PostgreSQL every statement, on upgrade and downgrade, runs through
``run_ddl_with_lock_retry``: a five-second ``lock_timeout``, up to three
attempts, two and then four seconds apart. Each column is added first without a
constraint, and the CHECK is added afterwards as ``NOT VALID``, so adding it
does not scan the table under the ``ACCESS EXCLUSIVE`` lock; every row that
exists then is NULL and satisfies it, and every later write is checked. SQLite
adds each column with one ``ADD COLUMN`` whose definition carries the named
CHECK; a batch rebuild is not used because it drops the ``AUTOINCREMENT`` of
``tasks``.

Downgrade first runs :data:`ROLLBACK_REWRITE_SQL` on both tables and only then
drops the constraints and the columns. Code that predates the column scopes an
agent key by ``source = 'sdk'`` and ``agent_id`` alone, so without the rewrite
that key could read, and append to, a task a personal key created, and the
append would run as the key holder. ``internal`` is a source that code already
knows: it treats these tasks as the holder's own web tasks. Once the column is
gone the rewrite cannot be told apart from a real web task, so it is permanent.
Run the downgrade only with every process that writes tasks stopped.

Code that predates this revision refuses to start against a database at this
revision, so going back to that code always means running this downgrade, with
those processes stopped, before the older code starts.

Revision ID: 20261031_task_sdk_credential_kind
Revises: 20261102_zendesk_side_conv
Create Date: 2026-10-31

"""

from functools import partial
from typing import Callable, Sequence, Union

import sqlalchemy as sa
from alembic import op

from xagent.migrations.ddl_lock_retry import (
    offline_ddl_with_lock_timeout,
    run_ddl_with_lock_retry,
)

revision: str = "20261031_task_sdk_credential_kind"
down_revision: Union[str, None] = "20261102_zendesk_side_conv"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

TASKS = "tasks"
TOMBSTONES = "expired_task_tombstones"
TABLES = (TASKS, TOMBSTONES)
COLUMN = "sdk_credential_kind"
CHECK_CONDITION = f"{COLUMN} IS NULL OR {COLUMN} = 'personal'"
# Named explicitly; both stay within PostgreSQL's 63-character identifier limit.
CONSTRAINTS = {
    TASKS: "ck_tasks_sdk_credential_kind",
    TOMBSTONES: "ck_expired_task_tombstones_sdk_credential_kind",
}
ROLLBACK_REWRITE_SQL = {
    TASKS: "UPDATE tasks SET source = 'internal' WHERE source = 'sdk' AND sdk_credential_kind = 'personal'",
    TOMBSTONES: "UPDATE expired_task_tombstones SET source = 'internal' WHERE source = 'sdk' AND sdk_credential_kind = 'personal'",
}
_LOCK_TIMEOUT = "5s"
# Seconds to wait after the first and the second lock timeout.
_LOCK_RETRY_DELAYS = (2.0, 4.0)


def _add_column_sql(table: str) -> str:
    return f"ALTER TABLE {table} ADD COLUMN {COLUMN} VARCHAR(16)"


def _add_check_sql(table: str) -> str:
    return (
        f"ALTER TABLE {table} ADD CONSTRAINT {CONSTRAINTS[table]} "
        f"CHECK ({CHECK_CONDITION}) NOT VALID"
    )


def _drop_check_sql(table: str) -> str:
    return f"ALTER TABLE {table} DROP CONSTRAINT IF EXISTS {CONSTRAINTS[table]}"


def _drop_column_sql(table: str) -> str:
    return f"ALTER TABLE {table} DROP COLUMN {COLUMN}"


def _column(table: str) -> sa.Column:
    return sa.Column(
        COLUMN,
        sa.String(16),
        sa.CheckConstraint(CHECK_CONDITION, name=CONSTRAINTS[table]),
        nullable=True,
    )


def _columns(table: str) -> set[str] | None:
    """The column names of ``table`` now, or None without it; never cached."""
    inspector = sa.inspect(op.get_bind())
    if table not in inspector.get_table_names():
        return None
    return {column["name"] for column in inspector.get_columns(table)}


def _has_check(table: str) -> bool:
    inspector = sa.inspect(op.get_bind())
    return any(
        item.get("name") == CONSTRAINTS[table]
        for item in inspector.get_check_constraints(table)
    )


def _apply(statement: str, table: str, fallback: Callable[[], None]) -> None:
    """Run ``statement`` under the lock retry on PostgreSQL, else ``fallback``."""
    if op.get_context().dialect.name != "postgresql":
        fallback()
        return
    run_ddl_with_lock_retry(
        statement,
        table=table,
        lock_timeout=_LOCK_TIMEOUT,
        retry_delays=_LOCK_RETRY_DELAYS,
    )


def _execute(statement: str) -> Callable[[], None]:
    return lambda: op.execute(sa.text(statement))


def upgrade() -> None:
    context = op.get_context()
    is_postgresql = context.dialect.name == "postgresql"
    if context.as_sql:
        for table in TABLES:
            if is_postgresql:
                for statement in (_add_column_sql(table), _add_check_sql(table)):
                    offline_ddl_with_lock_timeout(statement, lock_timeout=_LOCK_TIMEOUT)
            else:
                op.add_column(table, _column(table))
        return

    for table in TABLES:
        columns = _columns(table)
        if columns is None:
            continue
        if not is_postgresql:
            # The CHECK travels with the column definition.
            if COLUMN not in columns:
                op.add_column(table, _column(table))
            continue
        if COLUMN not in columns:
            _apply(_add_column_sql(table), table, lambda: None)
        if not _has_check(table):
            _apply(_add_check_sql(table), table, lambda: None)


def downgrade() -> None:
    context = op.get_context()
    is_postgresql = context.dialect.name == "postgresql"
    if context.as_sql:
        for table in TABLES:
            if is_postgresql:
                offline_ddl_with_lock_timeout(
                    ROLLBACK_REWRITE_SQL[table], lock_timeout=_LOCK_TIMEOUT
                )
            else:
                op.execute(sa.text(ROLLBACK_REWRITE_SQL[table]))
        for table in TABLES:
            if is_postgresql:
                for statement in (_drop_check_sql(table), _drop_column_sql(table)):
                    offline_ddl_with_lock_timeout(statement, lock_timeout=_LOCK_TIMEOUT)
            else:
                op.drop_column(table, COLUMN)
        return

    present = [table for table in TABLES if COLUMN in (_columns(table) or set())]
    # Every rewrite runs before anything is dropped.
    for table in present:
        statement = ROLLBACK_REWRITE_SQL[table]
        _apply(statement, table, _execute(statement))
    for table in present:
        if is_postgresql and _has_check(table):
            _apply(_drop_check_sql(table), table, lambda: None)
        _apply(_drop_column_sql(table), table, partial(op.drop_column, table, COLUMN))
