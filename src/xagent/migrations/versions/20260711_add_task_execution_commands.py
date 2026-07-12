"""add durable task execution command inbox

Revision ID: 20260711_task_commands
Revises: 20260711_add_trace_events_task_idx
Create Date: 2026-07-11

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "20260711_task_commands"
down_revision: Union[str, None] = "20260711_add_trace_events_task_idx"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

TABLE = "task_execution_commands"


def upgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    tables = set(inspector.get_table_names())
    if TABLE in tables:
        return
    # Base application tables are created by SQLAlchemy in production. The
    # migration integration CLI also exercises an Alembic-only empty schema;
    # follow the existing task-related migrations and skip until those base
    # tables exist instead of creating invalid foreign keys.
    if not {"tasks", "users"}.issubset(tables):
        return
    op.create_table(
        TABLE,
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "task_id",
            sa.Integer(),
            sa.ForeignKey("tasks.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "actor_user_id",
            sa.Integer(),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("command_id", sa.String(64), nullable=False),
        sa.Column("kind", sa.String(32), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("target_run_id", sa.String(64), nullable=True),
        sa.Column("target_runner_id", sa.String(255), nullable=True),
        sa.Column("status", sa.String(32), nullable=False, server_default="pending"),
        sa.Column("claimed_by", sa.String(255), nullable=True),
        sa.Column("claim_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("attempt_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("failure_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("defer_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("result", sa.JSON(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint("task_id", "command_id", name="uq_task_command_identity"),
    )
    op.create_index(
        "ix_task_execution_commands_actor_user_id", TABLE, ["actor_user_id"]
    )
    op.create_index("ix_task_commands_status_created", TABLE, ["status", "created_at"])
    op.create_index("ix_task_commands_task_order", TABLE, ["task_id", "id"])


def downgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    if TABLE in inspector.get_table_names():
        op.drop_table(TABLE)
