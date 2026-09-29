"""add partial index on task_chat_messages for pending deliveries

The R6 lease-recovery loop (task_lease_recovery.py) queries
task_chat_messages rows with delivery_status = 'pending' every tick, both to
page orphaned task_ids and to close one task's rows inside its own recovery
transaction. There was no index on delivery_status, so each tick scanned the
whole table. Almost every row settles out of "pending" quickly, so a partial
index scoped to that one value stays small regardless of table growth.

(task_id, created_at) are carried alongside the predicate: task_id backs the
per-task equality lookup both call sites use, and created_at backs the
sweep's additional "created_at < created_before" filter over the same rows.

On PostgreSQL, ``CREATE INDEX`` alone takes a SHARE lock that blocks writes
to the table for as long as the build runs -- unacceptable on the hot
``task_chat_messages`` table, which every user turn and every recovery sweep
tick writes to. This mirrors 20260725_add_task_lease_recovery_index.py:
build the index with ``CONCURRENTLY`` outside the migration's own
transaction (``autocommit_block()``), detect and rebuild a leftover
``INVALID`` index from an interrupted prior attempt (Postgres marks a
concurrent build that failed partway as invalid rather than rolling it back),
and keep the operation idempotent so a re-run of the migration is safe.
Non-PostgreSQL dialects (SQLite in tests, the only other dialect this app
runs on) keep the plain, transactional ``CREATE INDEX`` path.

Accepted cost: a partial index whose predicate reads ``delivery_status``
disables Postgres's HOT (heap-only tuple) update optimization for any row
whose ``delivery_status`` changes, because the index must be maintained
whenever the predicate's outcome could change. In practice this means one
extra non-HOT update per user turn -- the write that flips a row from
``pending`` to ``dispatched`` or its terminal state -- which is exactly the
row the index exists to make cheap to find in the first place.

Revision ID: 20260927_pending_delivery_index
Revises: 20260926_expired_task_tombstones
Create Date: 2026-09-27

"""

import re
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "20260927_pending_delivery_index"
down_revision: Union[str, None] = "20260926_expired_task_tombstones"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

TABLE = "task_chat_messages"
INDEX = "ix_task_chat_messages_pending_delivery"
INDEX_COLUMNS = ("task_id", "created_at")
# The predicate column itself, so a database whose task_chat_messages
# predates 20260710_add_chat_message_delivery_state (which introduced
# delivery_status) is recognized as not ready for this index yet, same as
# the two index columns below.
REQUIRED_COLUMNS = (*INDEX_COLUMNS, "delivery_status")
PREDICATE = sa.text("delivery_status = 'pending'")

POSTGRES_INDEX_VALIDITY_SQL = sa.text(
    """
    SELECT i.indisvalid
    FROM pg_catalog.pg_index AS i
    WHERE i.indexrelid = pg_catalog.to_regclass(:index_name)
    """
)

# A same-named (task_id, created_at) index that lacks -- or has a stale --
# partial predicate is column-list-identical to a correct one, so the column
# check above cannot tell them apart. ``pg_get_expr(indpred, indrelid)``
# deparses the stored predicate back into SQL text (NULL when the index is
# not partial at all); comparing that against our own PREDICATE catches both
# a plain, non-partial (task_id, created_at) index and one scoped to the
# wrong value.
POSTGRES_INDEX_PREDICATE_SQL = sa.text(
    """
    SELECT pg_catalog.pg_get_expr(i.indpred, i.indrelid)
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


def _indexes(inspector: sa.Inspector) -> set[str]:
    if TABLE not in inspector.get_table_names():
        return set()
    return {
        name
        for item in inspector.get_indexes(TABLE)
        if (name := item.get("name")) is not None
    }


def _index_columns(inspector: sa.Inspector, index_name: str) -> tuple[str, ...] | None:
    if TABLE not in inspector.get_table_names():
        return None
    for item in inspector.get_indexes(TABLE):
        if item.get("name") == index_name:
            return tuple(str(name) for name in item.get("column_names") or ())
    return None


def _postgres_index_validity() -> bool | None:
    """Return whether the current-schema index exists and is usable."""

    return (
        op.get_bind()
        .execute(
            POSTGRES_INDEX_VALIDITY_SQL,
            {"index_name": INDEX},
        )
        .scalar_one_or_none()
    )


def _postgres_index_predicate() -> str | None:
    """Return the current-schema index's deparsed predicate, if any."""

    return (
        op.get_bind()
        .execute(
            POSTGRES_INDEX_PREDICATE_SQL,
            {"index_name": INDEX},
        )
        .scalar_one_or_none()
    )


def _normalize_pg_predicate(expression: str) -> str:
    """Normalize a Postgres-deparsed predicate for structural comparison.

    ``pg_get_expr()`` reformats the predicate it stored: it wraps the
    column reference in parentheses, casts the column and/or literal to
    match the operator's resolved type (this migration's own
    ``delivery_status = 'pending'`` comes back as
    ``((delivery_status)::text = 'pending'::text)``), and does not
    guarantee whitespace. None of that changes what the predicate selects,
    so comparison lowercases and strips casts, parentheses, and whitespace
    rather than requiring an exact string match.
    """

    normalized = expression.lower()
    normalized = re.sub(r"::[a-z_ ]+(\([0-9]+\))?", "", normalized)
    normalized = re.sub(r"[()]", "", normalized)
    normalized = re.sub(r"\s+", "", normalized)
    return normalized


def _postgres_predicate_matches_expected(expression: str | None) -> bool:
    """Whether ``expression`` is our own PREDICATE (a non-partial index,
    ``expression is None``, never matches)."""

    if expression is None:
        return False
    return _normalize_pg_predicate(expression) == _normalize_pg_predicate(
        str(PREDICATE)
    )


def _upgrade_postgresql() -> None:
    """Build the pending-delivery index without blocking table writes."""

    inspector = _inspector()
    if not set(REQUIRED_COLUMNS) <= _columns(inspector):
        return

    validity = _postgres_index_validity()
    existing_columns = _index_columns(inspector, INDEX)
    is_up_to_date = (
        validity is True
        and existing_columns == INDEX_COLUMNS
        and _postgres_predicate_matches_expected(_postgres_index_predicate())
    )
    if is_up_to_date:
        return

    with op.get_context().autocommit_block():
        if validity is not None or existing_columns is not None:
            op.drop_index(
                INDEX,
                table_name=TABLE,
                if_exists=True,
                postgresql_concurrently=True,
            )
        op.create_index(
            INDEX,
            TABLE,
            list(INDEX_COLUMNS),
            if_not_exists=True,
            postgresql_concurrently=True,
            postgresql_where=PREDICATE,
        )


def upgrade() -> None:
    context = op.get_context()
    if context.as_sql:
        if context.dialect.name == "postgresql":
            with context.autocommit_block():
                op.create_index(
                    INDEX,
                    TABLE,
                    list(INDEX_COLUMNS),
                    postgresql_concurrently=True,
                    postgresql_where=PREDICATE,
                )
        elif context.dialect.name == "sqlite":
            op.create_index(INDEX, TABLE, list(INDEX_COLUMNS), sqlite_where=PREDICATE)
        else:
            op.create_index(INDEX, TABLE, list(INDEX_COLUMNS))
        return

    if context.dialect.name == "postgresql":
        _upgrade_postgresql()
        return

    inspector = _inspector()
    # A database whose task_chat_messages predates one of these columns (an
    # Alembic-only install stopped short of the migration that added it, or a
    # legacy fixture in a test) cannot support this predicate or these index
    # columns yet; skip rather than fail the whole chain, matching the
    # guard-clause convention 20260922_task_last_activity_at.py and
    # 20260710_add_chat_message_delivery_state.py use for the same table.
    if not set(REQUIRED_COLUMNS) <= _columns(inspector):
        return
    if INDEX in _indexes(inspector):
        return
    kwargs: dict[str, object] = {}
    if context.dialect.name == "sqlite":
        kwargs["sqlite_where"] = PREDICATE
    op.create_index(INDEX, TABLE, list(INDEX_COLUMNS), **kwargs)


def downgrade() -> None:
    context = op.get_context()
    is_postgresql = context.dialect.name == "postgresql"
    if context.as_sql:
        if is_postgresql:
            with context.autocommit_block():
                op.drop_index(
                    INDEX,
                    table_name=TABLE,
                    postgresql_concurrently=True,
                )
        else:
            op.drop_index(INDEX, table_name=TABLE)
        return

    if is_postgresql:
        with context.autocommit_block():
            op.drop_index(
                INDEX,
                table_name=TABLE,
                if_exists=True,
                postgresql_concurrently=True,
            )
        return

    inspector = _inspector()
    if INDEX in _indexes(inspector):
        op.drop_index(INDEX, table_name=TABLE)
