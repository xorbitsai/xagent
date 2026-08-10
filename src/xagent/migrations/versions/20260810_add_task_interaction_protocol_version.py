"""add tasks.interaction_protocol_version

Revision ID: 20260810_add_task_interaction_protocol_version
Revises: 20260809_add_task_interaction_requests
Create Date: 2026-08-10

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "20260810_add_task_interaction_protocol_version"
down_revision: Union[str, None] = "20260809_add_task_interaction_requests"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

TABLE = "tasks"
COLUMN = "interaction_protocol_version"
CONSTRAINT_NAME = "ck_tasks_interaction_protocol_version"
CONSTRAINT_CONDITION = (
    "interaction_protocol_version IS NULL OR interaction_protocol_version = 1"
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


def _online_check_constraints(schema: str | None) -> set[str]:
    inspector = sa.inspect(op.get_bind())
    if TABLE not in inspector.get_table_names(schema=schema):
        return set()
    return {
        str(item["name"])
        for item in inspector.get_check_constraints(TABLE, schema=schema)
        if item.get("name")
    }


def upgrade() -> None:
    context = op.get_context()

    # Offline (--sql) generation has a MockConnection, so reflection is
    # unavailable. Emit the unconditional DDL instead of inspecting. Adding a
    # CHECK to an existing table has no --sql-mode-safe path on SQLite (see
    # the migration docstring in the model's __table_args__ comment), so only
    # PostgreSQL gets the constraint here.
    if context.as_sql:
        op.add_column(TABLE, sa.Column(COLUMN, sa.Integer(), nullable=True))
        if context.dialect.name == "postgresql":
            op.create_check_constraint(CONSTRAINT_NAME, TABLE, CONSTRAINT_CONDITION)
        return

    # Address the same relation the catalog lookup inspects, so reflection and
    # DDL can never diverge onto different schemas.
    schema = _target_schema()

    if not _online_table_exists(schema):
        return

    if COLUMN not in _online_columns(schema):
        op.add_column(
            TABLE,
            sa.Column(COLUMN, sa.Integer(), nullable=True),
            schema=schema,
        )

    # The column guard above can skip add_column (e.g. a create_all-built
    # database already has the column), but the CHECK must still be checked
    # independently -- otherwise upgrading such a database would silently
    # leave it without the constraint.
    if context.dialect.name != "postgresql":
        return

    if CONSTRAINT_NAME not in _online_check_constraints(schema):
        op.create_check_constraint(
            CONSTRAINT_NAME, TABLE, CONSTRAINT_CONDITION, schema=schema
        )


def downgrade() -> None:
    context = op.get_context()

    # Constraint dropped before column, both here and in the online branch
    # below. PostgreSQL does not require this order -- DROP COLUMN auto-drops
    # a single-column CHECK that depends on the dropped column, no CASCADE
    # needed. The explicit drop_constraint keeps the downgrade symmetric with
    # upgrade's add-column-then-add-constraint sequence and independent of
    # that auto-drop behaviour.
    if context.as_sql:
        if context.dialect.name == "postgresql":
            op.drop_constraint(CONSTRAINT_NAME, TABLE, type_="check")
        op.drop_column(TABLE, COLUMN)
        return

    schema = _target_schema()

    if not _online_table_exists(schema):
        return

    if context.dialect.name == "postgresql":
        if CONSTRAINT_NAME in _online_check_constraints(schema):
            op.drop_constraint(CONSTRAINT_NAME, TABLE, type_="check", schema=schema)

    if COLUMN in _online_columns(schema):
        op.drop_column(TABLE, COLUMN, schema=schema)
