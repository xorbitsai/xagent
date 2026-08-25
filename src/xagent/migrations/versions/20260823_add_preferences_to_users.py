"""add preferences to users table

Revision ID: 20260823_add_preferences_to_users
Revises: a3b70c638cc3
Create Date: 2026-08-23 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "20260823_add_preferences_to_users"
# Repointed twice, both times because a sibling branch landed on main
# first while this migration was still in review: originally
# 20260818_user_oauth_resource_owner, then a3b70c638cc3 (the merge that
# folded that head together with the Salesforce connector's branch), now
# 20260824_seed_google_search_console_mcp_app - the Google Search
# Console connector branched from the same a3b70c638cc3 parent and
# landed first. This migration still hasn't landed on main, so moving
# its parent again is safe; no reconciling merge revision is needed
# since 20260824_seed_google_search_console_mcp_app already descends
# from a3b70c638cc3.
down_revision: Union[str, None] = "20260824_seed_google_search_console_mcp_app"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

TABLE = "users"
COLUMN = "preferences"

# The schema of the *visible* users relation. version_table_schema names only
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
    """The schema holding the users relation this migration operates on."""
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
    # unavailable. Emit the unconditional DDL instead of inspecting (the
    # 20260808/20260726 shape, not the inspector-only shape this migration
    # used before - the latter raises under --sql on both dialects).
    if context.as_sql:
        op.add_column(TABLE, sa.Column(COLUMN, sa.JSON(), nullable=True))
        return

    # Address the same relation the catalog lookup inspects, so reflection and
    # DDL can never diverge onto different schemas.
    schema = _target_schema()

    if not _online_table_exists(schema):
        return

    if COLUMN not in _online_columns(schema):
        op.add_column(
            TABLE,
            sa.Column(COLUMN, sa.JSON(), nullable=True),
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
