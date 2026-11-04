"""index tasks.last_checkpoint_trace_event_id

``fk_tasks_last_checkpoint_trace_event_id`` points tasks at trace_events, so
PostgreSQL checks it on every trace_events delete by looking tasks up on
``last_checkpoint_trace_event_id``. With no index on that column each check is
a sequential scan of tasks, paid once per deleted trace row. Checkpoint
pruning deletes trace rows after every persisted checkpoint, and task deletion
and the retention purge delete them in bulk, so the cost scales with both the
delete rate and the size of tasks.

On PostgreSQL the index is built with ``CONCURRENTLY`` outside the migration's
own transaction (``autocommit_block()``), so a large tasks table keeps taking
writes during the build. This mirrors 20260725_add_task_lease_recovery_index.py:
a leftover ``INVALID`` index from an interrupted concurrent build, or a
same-named index on other columns, is dropped and rebuilt, and a re-run is a
no-op. Other dialects use a plain ``CREATE INDEX``.

Revision ID: 20261031_task_checkpoint_pointer_index
Revises: 20261030_merge_history_freshdesk
Create Date: 2026-10-31

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "20261031_task_checkpoint_pointer_index"
down_revision: Union[str, None] = "20261030_merge_history_freshdesk"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

TABLE = "tasks"
INDEX = "ix_tasks_last_checkpoint_trace_event_id"
INDEX_COLUMNS = ("last_checkpoint_trace_event_id",)

POSTGRES_INDEX_VALIDITY_SQL = sa.text(
    """
    SELECT i.indisvalid
    FROM pg_catalog.pg_index AS i
    WHERE i.indexrelid = pg_catalog.to_regclass(:index_name)
    """
)


def _online_indexes() -> set[str]:
    inspector = sa.inspect(op.get_bind())
    if TABLE not in inspector.get_table_names():
        return set()
    return {
        name
        for item in inspector.get_indexes(TABLE)
        if (name := item.get("name")) is not None
    }


def _online_index_columns(index_name: str) -> tuple[str, ...] | None:
    inspector = sa.inspect(op.get_bind())
    if TABLE not in inspector.get_table_names():
        return None
    for item in inspector.get_indexes(TABLE):
        if item.get("name") == index_name:
            return tuple(str(name) for name in item.get("column_names") or ())
    return None


def _online_columns() -> set[str]:
    inspector = sa.inspect(op.get_bind())
    if TABLE not in inspector.get_table_names():
        return set()
    return {item["name"] for item in inspector.get_columns(TABLE)}


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


def _upgrade_postgresql() -> None:
    """Build the pointer index without blocking task-table writes."""

    if not set(INDEX_COLUMNS).issubset(_online_columns()):
        return

    validity = _postgres_index_validity()
    existing_columns = _online_index_columns(INDEX)
    if validity is True and existing_columns == INDEX_COLUMNS:
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
                )
        else:
            op.create_index(INDEX, TABLE, list(INDEX_COLUMNS))
        return

    if context.dialect.name == "postgresql":
        _upgrade_postgresql()
        return

    if set(INDEX_COLUMNS).issubset(_online_columns()) and INDEX not in (
        _online_indexes()
    ):
        op.create_index(INDEX, TABLE, list(INDEX_COLUMNS))


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

    if INDEX in _online_indexes():
        op.drop_index(INDEX, table_name=TABLE)
