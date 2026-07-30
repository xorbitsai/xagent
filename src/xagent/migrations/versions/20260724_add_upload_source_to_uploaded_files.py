"""add nullable upload_source marker to uploaded_files

Public share-channel hardening (#973): task-less public-share uploads are
created before any task/owner binding, so they can never be reaped by a
plain ``task_id IS NULL`` sweep without also catching logged-in users'
un-sent draft attachments. This adds a provenance marker so orphan GC can
scope its predicate to exactly those task-less public uploads. NULL for all
existing rows and for every other upload path.

Also adds a composite index serving the hourly GC predicate
(``upload_source + task_id + created_at``): marked rows keep their marker
after binding (provenance), so without an index the sweep's scan grows with
total upload history. On PostgreSQL the index is built concurrently so the
hot uploaded_files table is not write-locked; a previously failed concurrent
build (``pg_index.indisvalid = false``) is dropped and rebuilt rather than
trusted (mirrors 20260725_add_uploaded_file_recovery_index).

Revision ID: 20260724_add_upload_source_to_uploaded_files
Revises: 20260725_add_uploaded_file_recovery_index
Create Date: 2026-07-24

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.engine.reflection import Inspector

revision: str = "20260724_add_upload_source_to_uploaded_files"
down_revision: Union[str, None] = "20260725_add_uploaded_file_recovery_index"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

TABLE = "uploaded_files"
COLUMN = "upload_source"
INDEX = "ix_uploaded_files_orphan_gc"
INDEX_COLUMNS = (COLUMN, "task_id", "created_at")

# uploaded_files is migration-created (20260225) with FKs to users/tasks,
# which are create_all()-owned and may not exist in a migrations-only
# database (SQLite tolerates dangling FK targets). SQLite batch recreate
# reflects the table; without this, resolving those FKs raises
# NoSuchTableError. The FK DDL itself is still carried over by name.
BATCH_REFLECT_KWARGS = {"resolve_fks": False}

POSTGRES_INDEX_VALIDITY_SQL = sa.text(
    """
    SELECT i.indisvalid
    FROM pg_catalog.pg_index AS i
    WHERE i.indexrelid = pg_catalog.to_regclass(:index_name)
    """
)


def _existing_columns(inspector: Inspector, table: str) -> list[str]:
    return [col["name"] for col in inspector.get_columns(table)]


def _existing_indexes(inspector: Inspector, table: str) -> set[str]:
    return {ix["name"] for ix in inspector.get_indexes(table) if ix.get("name")}


def _online_index_columns() -> tuple[str, ...] | None:
    inspector = sa.inspect(op.get_bind())
    for item in inspector.get_indexes(TABLE):
        if item.get("name") == INDEX:
            return tuple(str(name) for name in item.get("column_names") or ())
    return None


def _postgres_index_validity() -> bool | None:
    return (
        op.get_bind()
        .execute(POSTGRES_INDEX_VALIDITY_SQL, {"index_name": INDEX})
        .scalar_one_or_none()
    )


def _create_index_postgresql() -> None:
    """Build (or repair) the GC index without blocking uploaded-file writes.

    A failed ``CREATE INDEX CONCURRENTLY`` leaves an invalid index entry
    (``indisvalid = false``) that plain existence checks — and
    ``IF NOT EXISTS`` — would treat as complete forever, so validity and
    column shape are checked explicitly and any broken/mismatched index is
    dropped concurrently and rebuilt.
    """
    validity = _postgres_index_validity()
    existing_columns = _online_index_columns()
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
    from alembic import context

    if context.is_offline_mode():
        # Offline (--sql) supplies a MockConnection that cannot be
        # reflected/inspected: emit deterministic DDL unconditionally.
        op.add_column(TABLE, sa.Column(COLUMN, sa.String(length=64), nullable=True))
        if op.get_context().dialect.name == "postgresql":
            with op.get_context().autocommit_block():
                op.create_index(
                    INDEX,
                    TABLE,
                    list(INDEX_COLUMNS),
                    if_not_exists=True,
                    postgresql_concurrently=True,
                )
        else:
            op.create_index(INDEX, TABLE, list(INDEX_COLUMNS))
        return

    bind = context.get_bind()
    inspector = Inspector.from_engine(bind)
    if TABLE not in inspector.get_table_names():
        return
    # Guarded so the migration is re-runnable on a partially-applied DB.
    if COLUMN not in _existing_columns(inspector, TABLE):
        with op.batch_alter_table(
            TABLE, reflect_kwargs=BATCH_REFLECT_KWARGS
        ) as batch_op:
            batch_op.add_column(sa.Column(COLUMN, sa.String(length=64), nullable=True))
    if op.get_context().dialect.name == "postgresql":
        _create_index_postgresql()
    elif INDEX not in _existing_indexes(Inspector.from_engine(bind), TABLE):
        op.create_index(INDEX, TABLE, list(INDEX_COLUMNS))


def downgrade() -> None:
    from alembic import context

    if context.is_offline_mode():
        # Deterministic offline DDL, no inspection (see upgrade()).
        if op.get_context().dialect.name == "postgresql":
            with op.get_context().autocommit_block():
                op.drop_index(
                    INDEX,
                    table_name=TABLE,
                    if_exists=True,
                    postgresql_concurrently=True,
                )
        else:
            op.drop_index(INDEX, table_name=TABLE)
        op.drop_column(TABLE, COLUMN)
        return

    bind = context.get_bind()
    inspector = Inspector.from_engine(bind)
    if TABLE not in inspector.get_table_names():
        return
    if op.get_context().dialect.name == "postgresql":
        # Covers a valid, an invalid (failed concurrent build), or an absent
        # index alike; concurrent drop keeps writes unblocked.
        with op.get_context().autocommit_block():
            op.drop_index(
                INDEX,
                table_name=TABLE,
                if_exists=True,
                postgresql_concurrently=True,
            )
    elif INDEX in _existing_indexes(inspector, TABLE):
        op.drop_index(INDEX, table_name=TABLE)
    if COLUMN in _existing_columns(inspector, TABLE):
        with op.batch_alter_table(
            TABLE, reflect_kwargs=BATCH_REFLECT_KWARGS
        ) as batch_op:
            batch_op.drop_column(COLUMN)
