"""add durable terminal task-command events

Revision ID: 20260828_terminal_cmd_events
Revises: 20260821_actor_oauth_flow_states
Create Date: 2026-08-28

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "20260828_terminal_cmd_events"
down_revision: Union[str, None] = "20260821_actor_oauth_flow_states"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

TABLE = "task_command_terminal_events"
COMMAND_TABLE = "task_execution_commands"
TARGET_STATE_VERSION = "target_state_version"
POSTGRES_VISIBLE_TABLE_SCHEMA_SQL = sa.text(
    """
    SELECT ns.nspname
    FROM pg_catalog.pg_class AS cls
    JOIN pg_catalog.pg_namespace AS ns ON ns.oid = cls.relnamespace
    WHERE cls.oid = pg_catalog.to_regclass(:table_name)
    """
)


def _target_schema() -> str | None:
    """Resolve the schema containing the visible command table."""

    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        resolved = bind.execute(
            POSTGRES_VISIBLE_TABLE_SCHEMA_SQL,
            {"table_name": COMMAND_TABLE},
        ).scalar()
        if resolved:
            return str(resolved)
    schema = op.get_context().version_table_schema
    return str(schema) if schema else None


def _create_terminal_event_table(schema: str | None) -> None:
    op.create_table(
        TABLE,
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("event_id", sa.String(36), nullable=False),
        sa.Column(
            "task_command_id",
            sa.Integer(),
            sa.ForeignKey("task_execution_commands.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "task_id",
            sa.Integer(),
            sa.ForeignKey("tasks.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("task_run_id", sa.String(64), nullable=True),
        sa.Column("task_state_version", sa.Integer(), nullable=True),
        sa.Column("command_id", sa.String(64), nullable=False),
        sa.Column("command_kind", sa.String(32), nullable=False),
        sa.Column(
            "actor_user_id",
            sa.Integer(),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "task_owner_user_id",
            sa.Integer(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("outcome_version", sa.Integer(), nullable=False),
        sa.Column("outcome", sa.String(32), nullable=False),
        sa.Column("message_code", sa.String(64), nullable=True),
        sa.Column(
            "resend_safe",
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
        sa.Column(
            "include_command_identity",
            sa.Boolean(),
            nullable=False,
            server_default=sa.true(),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.UniqueConstraint("event_id", name="uq_task_command_terminal_event_id"),
        sa.UniqueConstraint(
            "task_command_id",
            "outcome_version",
            name="uq_task_command_terminal_outcome_version",
        ),
        schema=schema,
    )
    op.create_index(
        "ix_task_command_terminal_events_task_cursor",
        TABLE,
        ["task_id", "id"],
        schema=schema,
    )


def upgrade() -> None:
    context = op.get_context()
    if context.as_sql:
        op.add_column(
            COMMAND_TABLE,
            sa.Column(TARGET_STATE_VERSION, sa.Integer(), nullable=True),
        )
        _create_terminal_event_table(None)
        return

    schema = _target_schema()
    inspector = sa.inspect(op.get_bind())
    tables = set(inspector.get_table_names(schema=schema))
    if not {"tasks", "users", COMMAND_TABLE}.issubset(tables):
        return
    command_columns = {
        column["name"] for column in inspector.get_columns(COMMAND_TABLE, schema=schema)
    }
    if TARGET_STATE_VERSION not in command_columns:
        op.add_column(
            COMMAND_TABLE,
            sa.Column(TARGET_STATE_VERSION, sa.Integer(), nullable=True),
            schema=schema,
        )
    if TABLE not in tables:
        _create_terminal_event_table(schema)


def downgrade() -> None:
    if op.get_context().as_sql:
        op.drop_table(TABLE)
        op.drop_column(COMMAND_TABLE, TARGET_STATE_VERSION)
        return

    schema = _target_schema()
    inspector = sa.inspect(op.get_bind())
    tables = set(inspector.get_table_names(schema=schema))
    if TABLE in tables:
        op.drop_table(TABLE, schema=schema)
    if COMMAND_TABLE in tables:
        command_columns = {
            column["name"]
            for column in inspector.get_columns(COMMAND_TABLE, schema=schema)
        }
        if TARGET_STATE_VERSION in command_columns:
            op.drop_column(COMMAND_TABLE, TARGET_STATE_VERSION, schema=schema)
