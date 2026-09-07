"""add the frozen tool call table

One row holds the arguments a gated MCP call will run with if approved, so
an approval executes what was shown rather than whatever the model writes
next. Model-only plumbing at this revision: the write gate that produces
rows is off unless a host installs it, so nothing writes here yet.

upgrade()/downgrade() fork on context.as_sql because offline (--sql)
generation runs against a MockConnection, where reflection is unavailable.
The offline branch emits the DDL unconditionally; the online branch keeps
existence guards -- the same shape as 20260809_add_task_interaction_requests.py.

The online branch carries two guards: the table already existing makes
upgrade() a no-op, which keeps a create_all-first startup idempotent
against a subsequent `alembic upgrade head`; a missing parent table also
makes it a no-op, because the two backends fail asymmetrically on a
dangling foreign key -- PostgreSQL raises UndefinedTable while creating the
table, SQLite lets the CREATE TABLE succeed and only fails on a later
INSERT.

The one CHECK and the named foreign key are rendered inline inside
op.create_table on both backends, so no ALTER TABLE ADD CONSTRAINT is ever
emitted and downgrade() is a single op.drop_table. The name-for-name
contract with the model's __table_args__ is enforced by the
create_all/migration parity tests, not by the downgrade path.

downgrade() is state-based, not provenance-based: it drops whatever table is
there, including one a create_all built rather than this revision.

Revision ID: 20260903_add_frozen_tool_calls
Revises: 20260902_oauth_flow_generation
Create Date: 2026-09-03

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "20260903_add_frozen_tool_calls"
down_revision: Union[str, None] = "20260902_oauth_flow_generation"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

TABLE = "frozen_tool_calls"
PARENT_TABLES = ("tasks",)
INDEXES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("ix_frozen_tool_calls_task_status", ("task_id", "status")),
    ("ix_frozen_tool_calls_expires_at", ("expires_at",)),
)

# Mirrors FrozenToolCall.__table_args__ (src/xagent/web/models/frozen_tool_call.py)
# -- the model is this list's source of truth, and the create_all/migration
# parity tests compare both name and expression against it.
CHECKS: tuple[tuple[str, str], ...] = (
    ("ck_frozen_tool_calls_status", "status IN ('pending','executed','voided')"),
)

POSTGRES_VISIBLE_TABLE_SCHEMA_SQL = sa.text(
    """
    SELECT ns.nspname
    FROM pg_catalog.pg_class AS cls
    JOIN pg_catalog.pg_namespace AS ns ON ns.oid = cls.relnamespace
    WHERE cls.oid = pg_catalog.to_regclass(:table_name)
    """
)


def _target_schema() -> str | None:
    """The schema holding the parent relations this table hangs off.

    Resolved from a parent, not from TABLE: TABLE does not exist yet on the
    run that creates it, so to_regclass() on it would always be NULL.
    """
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        resolved = bind.execute(
            POSTGRES_VISIBLE_TABLE_SCHEMA_SQL, {"table_name": PARENT_TABLES[0]}
        ).scalar()
        if resolved:
            return str(resolved)
    schema = op.get_context().version_table_schema
    return str(schema) if schema else None


def _create_table(schema: str | None) -> None:
    op.create_table(
        TABLE,
        sa.Column("interaction_id", sa.String(length=255), nullable=False),
        sa.Column("task_id", sa.Integer(), nullable=False),
        sa.Column("tool_name", sa.String(length=255), nullable=False),
        sa.Column("server_name", sa.String(length=255), nullable=False),
        sa.Column("write_hint", sa.String(length=32), nullable=False),
        sa.Column("arguments", sa.JSON(none_as_null=True), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("settled_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("interaction_id", name="pk_frozen_tool_calls"),
        sa.ForeignKeyConstraint(
            ["task_id"],
            ["tasks.id"],
            name="fk_frozen_tool_calls_task_id_tasks",
            ondelete="CASCADE",
        ),
        *(sa.CheckConstraint(expr, name=name) for name, expr in CHECKS),
        schema=schema,
    )
    for index_name, columns in INDEXES:
        op.create_index(index_name, TABLE, list(columns), schema=schema)


def upgrade() -> None:
    context = op.get_context()

    if context.as_sql:
        _create_table(None)
        return

    schema = _target_schema()
    inspector = sa.inspect(op.get_bind())
    tables = set(inspector.get_table_names(schema=schema))

    # Guard 1: a create_all-first startup already has this table.
    if TABLE in tables:
        return
    # Guard 2: a missing parent would fail asymmetrically across backends.
    if not set(PARENT_TABLES).issubset(tables):
        return

    _create_table(schema)


def downgrade() -> None:
    context = op.get_context()

    if context.as_sql:
        op.drop_table(TABLE)
        return

    schema = _target_schema()
    inspector = sa.inspect(op.get_bind())
    if TABLE not in set(inspector.get_table_names(schema=schema)):
        return
    op.drop_table(TABLE, schema=schema)
