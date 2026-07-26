"""add uploaded file recovery lookup index

Revision ID: 20260725_add_uploaded_file_recovery_index
Revises: 20260725_add_task_lease_recovery_index
Create Date: 2026-07-25 00:10:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "20260725_add_uploaded_file_recovery_index"
down_revision: Union[str, None] = "20260725_add_task_lease_recovery_index"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

TABLE = "uploaded_files"
INDEX = "ix_uploaded_files_status_updated_at_id"
INDEX_COLUMNS = ("storage_status", "updated_at", "id")

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
    return (
        op.get_bind()
        .execute(
            POSTGRES_INDEX_VALIDITY_SQL,
            {"index_name": INDEX},
        )
        .scalar_one_or_none()
    )


def _upgrade_postgresql() -> None:
    """Build the recovery index without blocking uploaded-file writes."""

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
