"""add dispatch priority to tasks and task_execution_commands

``tasks.dispatch_priority`` (nullable, no default) is the priority the caller
declared when it created the task; existing rows read NULL.
``task_execution_commands.priority`` (``NOT NULL DEFAULT 0``) is the priority a
command was accepted with; existing rows read 0. Neither column rewrites the
table on PostgreSQL 11 or later.

The partial index ``ix_task_commands_claimable_fair_priority`` orders the
claimable commands by ``(fairness_key, dispatch_class, priority DESC, id)``.
A plain ``CREATE INDEX`` that fails rolls back whole, so an index of this name
is taken to be complete. SQLite gets it through ``sqlite_where``; any other
dialect gets it without the predicate.

On PostgreSQL every statement, on upgrade and downgrade, runs through
``run_ddl_with_lock_retry``: a five-second ``lock_timeout``, up to three
attempts, two and then four seconds apart. ``ALTER TABLE`` takes an
``ACCESS EXCLUSIVE`` lock, and building the index blocks every write to the
table for as long as the build takes, so stop the task execution processes
before running the migration.

Revision ID: 20261027_task_dispatch_priority
Revises: 20261026_auto_sequential_failover
Create Date: 2026-10-27

"""

from typing import Callable, Sequence, Union

import sqlalchemy as sa
from alembic import op

from xagent.migrations.ddl_lock_retry import (
    offline_ddl_with_lock_timeout,
    run_ddl_with_lock_retry,
)

revision: str = "20261027_task_dispatch_priority"
down_revision: Union[str, None] = "20261026_auto_sequential_failover"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

TASKS = "tasks"
TASK_COLUMN = "dispatch_priority"
COMMANDS = "task_execution_commands"
COMMAND_COLUMN = "priority"
INDEX = "ix_task_commands_claimable_fair_priority"
# Includes the predicate column ``status``.
INDEX_REQUIRED_COLUMNS = frozenset(
    {"fairness_key", "dispatch_class", COMMAND_COLUMN, "id", "status"}
)
PREDICATE = sa.text("status IN ('pending', 'processing')")
_LOCK_TIMEOUT = "5s"
# Seconds to wait after the first and the second lock timeout.
_LOCK_RETRY_DELAYS = (2.0, 4.0)
_ADD_TASK_COLUMN_SQL = f"ALTER TABLE {TASKS} ADD COLUMN {TASK_COLUMN} SMALLINT"
_DROP_TASK_COLUMN_SQL = f"ALTER TABLE {TASKS} DROP COLUMN {TASK_COLUMN}"
_ADD_COMMAND_COLUMN_SQL = (
    f"ALTER TABLE {COMMANDS} ADD COLUMN {COMMAND_COLUMN} SMALLINT NOT NULL DEFAULT 0"
)
_DROP_COMMAND_COLUMN_SQL = f"ALTER TABLE {COMMANDS} DROP COLUMN {COMMAND_COLUMN}"
_CREATE_INDEX_SQL = (
    f"CREATE INDEX {INDEX} ON {COMMANDS} "
    f"(fairness_key, dispatch_class, {COMMAND_COLUMN} DESC, id) WHERE {PREDICATE}"
)
_DROP_INDEX_SQL = f"DROP INDEX IF EXISTS {INDEX}"


def _task_column() -> sa.Column:
    return sa.Column(TASK_COLUMN, sa.SmallInteger(), nullable=True)


def _command_column() -> sa.Column:
    return sa.Column(
        COMMAND_COLUMN, sa.SmallInteger(), nullable=False, server_default="0"
    )


def _create_index_op() -> None:
    op.create_index(
        INDEX,
        COMMANDS,
        ["fairness_key", "dispatch_class", sa.text(f"{COMMAND_COLUMN} DESC"), "id"],
        sqlite_where=PREDICATE,
    )


def _columns(table: str) -> set[str] | None:
    """The column names of ``table`` now, or None without it; never cached."""
    inspector = sa.inspect(op.get_bind())
    if table not in inspector.get_table_names():
        return None
    return {column["name"] for column in inspector.get_columns(table)}


def _has_index() -> bool:
    inspector = sa.inspect(op.get_bind())
    return any(item.get("name") == INDEX for item in inspector.get_indexes(COMMANDS))


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


def upgrade() -> None:
    context = op.get_context()
    if context.as_sql:
        if context.dialect.name == "postgresql":
            for statement in (
                _ADD_TASK_COLUMN_SQL,
                _ADD_COMMAND_COLUMN_SQL,
                _CREATE_INDEX_SQL,
            ):
                offline_ddl_with_lock_timeout(statement, lock_timeout=_LOCK_TIMEOUT)
        else:
            op.add_column(TASKS, _task_column())
            op.add_column(COMMANDS, _command_column())
            _create_index_op()
        return

    task_columns = _columns(TASKS)
    if task_columns is not None and TASK_COLUMN not in task_columns:
        _apply(
            _ADD_TASK_COLUMN_SQL, TASKS, lambda: op.add_column(TASKS, _task_column())
        )
    command_columns = _columns(COMMANDS)
    if command_columns is None:
        return
    if COMMAND_COLUMN not in command_columns:
        _apply(
            _ADD_COMMAND_COLUMN_SQL,
            COMMANDS,
            lambda: op.add_column(COMMANDS, _command_column()),
        )
    # Read again: the earlier listing lacks the columns just added.
    if INDEX_REQUIRED_COLUMNS <= (_columns(COMMANDS) or set()) and not _has_index():
        _apply(_CREATE_INDEX_SQL, COMMANDS, _create_index_op)


def downgrade() -> None:
    context = op.get_context()
    if context.as_sql:
        if context.dialect.name == "postgresql":
            for statement in (
                _DROP_INDEX_SQL,
                _DROP_COMMAND_COLUMN_SQL,
                _DROP_TASK_COLUMN_SQL,
            ):
                offline_ddl_with_lock_timeout(statement, lock_timeout=_LOCK_TIMEOUT)
        else:
            op.drop_index(INDEX, table_name=COMMANDS)
            op.drop_column(COMMANDS, COMMAND_COLUMN)
            op.drop_column(TASKS, TASK_COLUMN)
        return

    command_columns = _columns(COMMANDS)
    if command_columns is not None and _has_index():
        _apply(
            _DROP_INDEX_SQL,
            COMMANDS,
            lambda: op.drop_index(INDEX, table_name=COMMANDS),
        )
    if command_columns is not None and COMMAND_COLUMN in command_columns:
        _apply(
            _DROP_COMMAND_COLUMN_SQL,
            COMMANDS,
            lambda: op.drop_column(COMMANDS, COMMAND_COLUMN),
        )
    task_columns = _columns(TASKS)
    if task_columns is not None and TASK_COLUMN in task_columns:
        _apply(_DROP_TASK_COLUMN_SQL, TASKS, lambda: op.drop_column(TASKS, TASK_COLUMN))
