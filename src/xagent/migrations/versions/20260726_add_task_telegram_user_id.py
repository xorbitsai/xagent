"""add Telegram sender ownership to tasks

Revision ID: 20260726_add_task_telegram_user_id
Revises: 20260802_add_workforce_run_last_activity_at
Create Date: 2026-07-26

"""

from typing import Any, Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "20260726_add_task_telegram_user_id"
down_revision: Union[str, None] = "20260802_add_workforce_run_last_activity_at"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

TABLE = "tasks"
COLUMN = "telegram_user_id"
INDEX = "ix_tasks_telegram_user_id"

# Resolve the index through the catalogs constrained by the target table and
# schema. to_regclass() follows search_path, so a same-name index in an earlier
# schema could otherwise be inspected -- and later dropped -- while the real
# target-schema index stayed invalid.
POSTGRES_INDEX_VALIDITY_SQL = sa.text(
    """
    SELECT i.indisvalid
    FROM pg_catalog.pg_index AS i
    JOIN pg_catalog.pg_class AS idx ON idx.oid = i.indexrelid
    JOIN pg_catalog.pg_class AS tbl ON tbl.oid = i.indrelid
    JOIN pg_catalog.pg_namespace AS ns ON ns.oid = tbl.relnamespace
    WHERE idx.relname = :index_name
      AND tbl.relname = :table_name
      AND ns.nspname = :schema_name
    """
)


# The schema of the *visible* tasks relation. version_table_schema names only
# the Alembic version table, and current_schema() is merely the first entry on
# search_path, so neither identifies the relation an unqualified reference
# actually resolves to. Ask PostgreSQL which one it resolves.
POSTGRES_VISIBLE_TABLE_SCHEMA_SQL = sa.text(
    """
    SELECT ns.nspname
    FROM pg_catalog.pg_class AS cls
    JOIN pg_catalog.pg_namespace AS ns ON ns.oid = cls.relnamespace
    WHERE cls.oid = pg_catalog.to_regclass(:table_name)
    """
)


def _target_schema() -> str | None:
    """The schema holding the tasks relation this migration operates on.

    Returns None when it cannot be resolved, so callers fall back to plain
    unqualified behaviour instead of addressing a guessed schema.
    """

    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        resolved = bind.execute(
            POSTGRES_VISIBLE_TABLE_SCHEMA_SQL, {"table_name": TABLE}
        ).scalar()
        if resolved:
            return str(resolved)
    schema = op.get_context().version_table_schema
    return str(schema) if schema else None


def _postgres_index_validity(schema: str | None) -> bool | None:
    """Return whether the index exists and is usable, or None if absent."""

    return (
        op.get_bind()
        .execute(
            POSTGRES_INDEX_VALIDITY_SQL,
            {
                "index_name": INDEX,
                "table_name": TABLE,
                "schema_name": schema,
            },
        )
        .scalar_one_or_none()
    )


def _online_columns(schema: str | None) -> set[str]:
    inspector = sa.inspect(op.get_bind())
    if TABLE not in inspector.get_table_names(schema=schema):
        return set()
    return {str(item["name"]) for item in inspector.get_columns(TABLE, schema=schema)}


def _online_indexes(schema: str | None) -> set[str]:
    inspector = sa.inspect(op.get_bind())
    if TABLE not in inspector.get_table_names(schema=schema):
        return set()
    return {
        name
        for item in inspector.get_indexes(TABLE, schema=schema)
        if (name := item.get("name")) is not None
    }


def _online_index_definition(
    index_name: str,
    schema: str | None,
) -> dict[str, Any] | None:
    """Return the reflected definition of a same-name index, if any."""

    inspector = sa.inspect(op.get_bind())
    if TABLE not in inspector.get_table_names(schema=schema):
        return None
    for item in inspector.get_indexes(TABLE, schema=schema):
        if item.get("name") == index_name:
            return item
    return None


def _index_definition_matches(definition: dict[str, Any] | None) -> bool:
    """Whether an existing index provides the exact lookup this migration needs.

    Key columns alone are not enough. A UNIQUE index would stop a Telegram
    sender from owning a second task, and a partial index would not cover every
    row, so either must be rebuilt rather than accepted.
    """

    if definition is None:
        return False
    if tuple(str(name) for name in definition.get("column_names") or ()) != (COLUMN,):
        return False
    if bool(definition.get("unique")):
        return False
    dialect_options = definition.get("dialect_options") or {}
    # A predicate under any dialect key means the index is partial.
    if any(key.endswith("_where") for key in dialect_options):
        return False
    if definition.get("expressions") or definition.get("include_columns"):
        return False
    return True


def _online_table_exists(schema: str | None) -> bool:
    return TABLE in sa.inspect(op.get_bind()).get_table_names(schema=schema)


def upgrade() -> None:
    context = op.get_context()
    is_postgresql = context.dialect.name == "postgresql"

    # Offline (--sql) generation has a MockConnection, so reflection is
    # unavailable. Emit the unconditional DDL instead of inspecting.
    if context.as_sql:
        op.add_column(TABLE, sa.Column(COLUMN, sa.String(length=32), nullable=True))
        if is_postgresql:
            with context.autocommit_block():
                op.create_index(
                    INDEX,
                    TABLE,
                    [COLUMN],
                    postgresql_concurrently=True,
                )
        else:
            op.create_index(INDEX, TABLE, [COLUMN])
        return

    # Address the same relation the catalog lookup inspects, so reflection and
    # DDL can never diverge onto different schemas.
    schema = _target_schema()

    if not _online_table_exists(schema):
        return

    if COLUMN not in _online_columns(schema):
        op.add_column(
            TABLE,
            sa.Column(COLUMN, sa.String(length=32), nullable=True),
            schema=schema,
        )

    if COLUMN not in _online_columns(schema):
        return

    # A plain CREATE INDEX holds a SHARE lock and blocks writes to the live
    # tasks table for the whole build, so PostgreSQL builds it concurrently.
    if is_postgresql:
        validity = _postgres_index_validity(schema)
        definition = _online_index_definition(INDEX, schema)
        # A same-name index that is valid but has different semantics (wrong
        # columns, UNIQUE, or partial) would otherwise be accepted, letting
        # Alembic stamp the revision without the lookup this migration needs.
        if validity is True and _index_definition_matches(definition):
            return
        with context.autocommit_block():
            # A failed CREATE INDEX CONCURRENTLY leaves the index present but
            # invalid. IF NOT EXISTS would skip the rebuild and let Alembic
            # stamp the revision with an unusable index, so drop it first.
            if validity is not None or definition is not None:
                op.drop_index(
                    INDEX,
                    table_name=TABLE,
                    schema=schema,
                    if_exists=True,
                    postgresql_concurrently=True,
                )
            op.create_index(
                INDEX,
                TABLE,
                [COLUMN],
                schema=schema,
                if_not_exists=True,
                postgresql_concurrently=True,
            )
        return

    definition = _online_index_definition(INDEX, schema)
    if _index_definition_matches(definition):
        return
    if definition is not None:
        # Drifted same-name index: rebuild it rather than stamping the revision
        # with an index that does not serve the ownership lookup.
        op.drop_index(INDEX, table_name=TABLE, schema=schema)
    op.create_index(INDEX, TABLE, [COLUMN], schema=schema)


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
        op.drop_column(TABLE, COLUMN)
        return

    schema = _target_schema()

    if not _online_table_exists(schema):
        return

    if is_postgresql:
        with context.autocommit_block():
            op.drop_index(
                INDEX,
                table_name=TABLE,
                schema=schema,
                if_exists=True,
                postgresql_concurrently=True,
            )
    elif INDEX in _online_indexes(schema):
        op.drop_index(INDEX, table_name=TABLE, schema=schema)

    if COLUMN in _online_columns(schema):
        op.drop_column(TABLE, COLUMN, schema=schema)
