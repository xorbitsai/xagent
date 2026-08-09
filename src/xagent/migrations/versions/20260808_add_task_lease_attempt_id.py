"""add lease attempt identity to tasks

Revision ID: 20260808_add_task_lease_attempt_id
Revises: 20260807_seed_notion_mcp_app
Create Date: 2026-08-08

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "20260808_add_task_lease_attempt_id"
down_revision: Union[str, None] = "20260807_seed_notion_mcp_app"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

TABLE = "tasks"
COLUMN = "lease_attempt_id"

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
    """The schema holding the tasks relation this migration operates on."""
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        resolved = bind.execute(
            POSTGRES_VISIBLE_TABLE_SCHEMA_SQL, {"table_name": TABLE}
        ).scalar()
        if resolved:
            return str(resolved)
    schema = op.get_context().version_table_schema
    return str(schema) if schema else None


def _online_columns(schema: str | None) -> set[str]:
    inspector = sa.inspect(op.get_bind())
    if TABLE not in inspector.get_table_names(schema=schema):
        return set()
    return {str(item["name"]) for item in inspector.get_columns(TABLE, schema=schema)}


def _online_table_exists(schema: str | None) -> bool:
    return TABLE in sa.inspect(op.get_bind()).get_table_names(schema=schema)


def upgrade() -> None:
    context = op.get_context()

    # Offline (--sql) generation has a MockConnection, so reflection is
    # unavailable. Emit the unconditional DDL instead of inspecting. This is
    # the 20260726 shape, not the inspector-only shape #1137 used -- the
    # latter raises under --sql on both dialects.
    if context.as_sql:
        op.add_column(TABLE, sa.Column(COLUMN, sa.String(length=64), nullable=True))
        return

    # Address the same relation the catalog lookup inspects, so reflection and
    # DDL can never diverge onto different schemas.
    schema = _target_schema()

    if not _online_table_exists(schema):
        return

    if COLUMN not in _online_columns(schema):
        op.add_column(
            TABLE,
            sa.Column(COLUMN, sa.String(length=64), nullable=True),
            schema=schema,
        )


def downgrade() -> None:
    context = op.get_context()

    if context.as_sql:
        op.drop_column(TABLE, COLUMN)
        return

    schema = _target_schema()

    if not _online_table_exists(schema):
        return

    if COLUMN in _online_columns(schema):
        op.drop_column(TABLE, COLUMN, schema=schema)
