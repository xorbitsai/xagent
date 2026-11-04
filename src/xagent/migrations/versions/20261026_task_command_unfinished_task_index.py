"""add partial index on task_execution_commands by task over unfinished commands

The index orders the commands that are not ``completed`` or ``failed`` by
``(task_id, id)``, so the oldest unfinished command of one task is the first
entry of its range. Settled commands are not in it, so it stays small however
large the command history grows. It serves the dispatcher's lookup of the
oldest claimable command of each running task, and it also lets the check in
every claimable query, that a task has no unfinished command before the
candidate, read only the unfinished commands of that task.

On PostgreSQL the index is built with a plain ``CREATE INDEX`` through
``run_ddl_with_lock_retry``. While it builds, every write to the table is
blocked, task workers and the web processes that accept commands alike, for as
long as the build takes, which grows with the whole table and not only with the
unfinished rows; reads continue. The lock timeout only limits how long the
statement waits for its lock, not the build itself, and a write that is still
uncommitted keeps the statement waiting, so run the migration with the task
execution processes stopped.

A same-named index whose columns, predicate or validity differ is dropped
first. ``DROP INDEX`` takes an ``ACCESS EXCLUSIVE`` lock, which even reads
block. The drop and the build commit separately: if the build fails after the
drop, the table has no such index until the migration runs again.

SQLite gets the same partial index through ``sqlite_where``. Any other dialect
gets a plain ``CREATE INDEX`` without a predicate.

Accepted cost: every insert, and every status change that leaves a row
unfinished, maintains one more index entry.

Revision ID: 20261026_task_cmd_unfinished_idx
Revises: 20261023_add_mcp_signed_uploads
Create Date: 2026-10-26

"""

import re
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

from xagent.migrations.ddl_lock_retry import (
    offline_ddl_with_lock_timeout,
    run_ddl_with_lock_retry,
)

revision: str = "20261026_task_cmd_unfinished_idx"
down_revision: Union[str, None] = "20261023_add_mcp_signed_uploads"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

TABLE = "task_execution_commands"
INDEX = "ix_task_commands_unfinished_task"
INDEX_COLUMNS = ("task_id", "id")
# The predicate column itself, so a table that lacks ``status`` is recognized
# as not ready for this index, same as the index columns.
REQUIRED_COLUMNS = (*INDEX_COLUMNS, "status")
PREDICATE = sa.text("status NOT IN ('completed', 'failed')")
_LOCK_TIMEOUT = "5s"
# Seconds to wait after the first and the second lock timeout.
_LOCK_RETRY_DELAYS = (2.0, 4.0)
_CREATE_SQL = (
    f"CREATE INDEX {INDEX} ON {TABLE} ({', '.join(INDEX_COLUMNS)}) WHERE {PREDICATE}"
)
_DROP_SQL = f"DROP INDEX IF EXISTS {INDEX}"

# A same-named index with a stale or missing partial predicate has the same
# column list as a correct one, so the deparsed predicate is read too.
POSTGRES_INDEX_STATE_SQL = sa.text(
    """
    SELECT i.indisvalid, pg_catalog.pg_get_expr(i.indpred, i.indrelid)
    FROM pg_catalog.pg_index AS i
    WHERE i.indexrelid = pg_catalog.to_regclass(:index_name)
    """
)


def _inspector() -> sa.Inspector:
    return sa.inspect(op.get_bind())


def _columns(inspector: sa.Inspector) -> set[str]:
    if TABLE not in inspector.get_table_names():
        return set()
    return {column["name"] for column in inspector.get_columns(TABLE)}


def _index_columns(inspector: sa.Inspector, index_name: str) -> tuple[str, ...] | None:
    if TABLE not in inspector.get_table_names():
        return None
    for item in inspector.get_indexes(TABLE):
        if item.get("name") == index_name:
            return tuple(str(name) for name in item.get("column_names") or ())
    return None


def _postgres_index_state() -> tuple[bool, str | None] | None:
    """Return (is valid, deparsed predicate) of the index, or None if absent."""

    row = (
        op.get_bind()
        .execute(POSTGRES_INDEX_STATE_SQL, {"index_name": INDEX})
        .one_or_none()
    )
    return None if row is None else (bool(row[0]), row[1])


def _normalize_pg_predicate(expression: str) -> str:
    """Normalize a Postgres-deparsed predicate for structural comparison.

    ``pg_get_expr()`` reformats the predicate it stored: it wraps the column
    reference in parentheses, casts the column and the literals to the
    operator's resolved type, rewrites an ``IN`` list into
    ``= ANY (ARRAY[...])``, a ``NOT IN`` list into ``<> ALL (ARRAY[...])`` and
    does not guarantee whitespace. None of that changes what the predicate
    selects, so the comparison removes casts, parentheses and whitespace,
    restores an ``= ANY (ARRAY[...])`` list to ``IN`` and a
    ``<> ALL (ARRAY[...])`` list to ``NOT IN``, and compares the whole string.
    """

    normalized = expression.lower()
    normalized = re.sub(r"::[a-z_ ]+(\([0-9]+\))?(\[\])?", "", normalized)
    normalized = re.sub(r"[()\s]", "", normalized)
    normalized = re.sub(r"=anyarray\[(.*)\]$", r"in\1", normalized)
    return re.sub(r"<>allarray\[(.*)\]$", r"notin\1", normalized)


def _postgres_predicate_matches_expected(expression: str | None) -> bool:
    """Whether ``expression`` is our own PREDICATE (a non-partial index,
    ``expression is None``, never matches)."""

    if expression is None:
        return False
    return _normalize_pg_predicate(expression) == _normalize_pg_predicate(
        str(PREDICATE)
    )


def _run_ddl(statement: str) -> None:
    run_ddl_with_lock_retry(
        statement,
        table=TABLE,
        lock_timeout=_LOCK_TIMEOUT,
        retry_delays=_LOCK_RETRY_DELAYS,
    )


def _upgrade_postgresql() -> None:
    inspector = _inspector()
    if not set(REQUIRED_COLUMNS) <= _columns(inspector):
        return

    state = _postgres_index_state()
    existing_columns = _index_columns(inspector, INDEX)
    if (
        state is not None
        and state[0]
        and existing_columns == INDEX_COLUMNS
        and _postgres_predicate_matches_expected(state[1])
    ):
        return

    if state is not None or existing_columns is not None:
        _run_ddl(_DROP_SQL)
    _run_ddl(_CREATE_SQL)


def upgrade() -> None:
    context = op.get_context()
    if context.as_sql:
        if context.dialect.name == "postgresql":
            offline_ddl_with_lock_timeout(_CREATE_SQL, lock_timeout=_LOCK_TIMEOUT)
        else:
            # Dialects other than SQLite ignore ``sqlite_where``.
            op.create_index(INDEX, TABLE, list(INDEX_COLUMNS), sqlite_where=PREDICATE)
        return

    if context.dialect.name == "postgresql":
        _upgrade_postgresql()
        return

    inspector = _inspector()
    # A database whose table lacks one of these columns (an Alembic-only
    # install stopped short of the migration that added it, or a legacy
    # fixture in a test) cannot support this predicate or these index columns
    # yet; skip rather than fail the whole chain.
    if not set(REQUIRED_COLUMNS) <= _columns(inspector):
        return
    if _index_columns(inspector, INDEX) is not None:
        return
    # Dialects other than SQLite ignore ``sqlite_where``.
    op.create_index(INDEX, TABLE, list(INDEX_COLUMNS), sqlite_where=PREDICATE)


def downgrade() -> None:
    context = op.get_context()
    if context.dialect.name == "postgresql":
        if context.as_sql:
            offline_ddl_with_lock_timeout(_DROP_SQL, lock_timeout=_LOCK_TIMEOUT)
        elif _index_columns(_inspector(), INDEX) is not None:
            _run_ddl(_DROP_SQL)
    elif context.as_sql or _index_columns(_inspector(), INDEX) is not None:
        op.drop_index(INDEX, table_name=TABLE)
