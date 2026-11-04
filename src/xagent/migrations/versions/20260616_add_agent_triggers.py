"""add agent triggers

Revision ID: 20260616_add_agent_triggers
Revises: 20260611_backfill_basic_agents_web_search
Create Date: 2026-06-16 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "20260616_add_agent_triggers"
down_revision: Union[str, tuple[str, str], None] = (
    "20260611_backfill_basic_agents_web_search"
)
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _index_names(table_name: str) -> set[str]:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if table_name not in inspector.get_table_names():
        return set()
    return {index["name"] for index in inspector.get_indexes(table_name)}


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing_tables = inspector.get_table_names()

    if "agent_triggers" not in existing_tables:
        constraints = [
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("webhook_token"),
        ]
        if "users" in existing_tables:
            constraints.append(
                sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE")
            )
        if "agents" in existing_tables:
            constraints.append(
                sa.ForeignKeyConstraint(["agent_id"], ["agents.id"], ondelete="CASCADE")
            )

        op.create_table(
            "agent_triggers",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("user_id", sa.Integer(), nullable=False),
            sa.Column("agent_id", sa.Integer(), nullable=False),
            sa.Column("type", sa.String(length=32), nullable=False),
            sa.Column("name", sa.String(length=200), nullable=False),
            sa.Column(
                "enabled",
                sa.Boolean(),
                nullable=False,
                server_default=sa.true(),
            ),
            sa.Column("config", sa.JSON(), nullable=False),
            sa.Column("prompt_template", sa.Text(), nullable=True),
            sa.Column("webhook_token", sa.String(length=128), nullable=True),
            sa.Column("secret_hash", sa.String(length=64), nullable=True),
            sa.Column("next_run_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("last_run_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("last_error", sa.Text(), nullable=True),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                server_default=sa.text("CURRENT_TIMESTAMP"),
                nullable=True,
            ),
            sa.Column(
                "updated_at",
                sa.DateTime(timezone=True),
                server_default=sa.text("CURRENT_TIMESTAMP"),
                nullable=True,
            ),
            *constraints,
        )

    if "trigger_runs" not in existing_tables:
        constraints = [
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("idempotency_key"),
        ]
        if "agent_triggers" in inspector.get_table_names():
            constraints.append(
                sa.ForeignKeyConstraint(
                    ["trigger_id"], ["agent_triggers.id"], ondelete="CASCADE"
                )
            )
        if "tasks" in existing_tables:
            constraints.append(
                sa.ForeignKeyConstraint(["task_id"], ["tasks.id"], ondelete="SET NULL")
            )
        if "background_jobs" in existing_tables:
            constraints.append(
                sa.ForeignKeyConstraint(
                    ["background_job_id"], ["background_jobs.id"], ondelete="SET NULL"
                )
            )

        op.create_table(
            "trigger_runs",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("trigger_id", sa.Integer(), nullable=False),
            sa.Column("task_id", sa.Integer(), nullable=True),
            sa.Column("background_job_id", sa.String(length=36), nullable=True),
            sa.Column("status", sa.String(length=32), nullable=False),
            sa.Column("source_event_id", sa.String(length=255), nullable=True),
            sa.Column("payload_snapshot", sa.JSON(), nullable=True),
            sa.Column("idempotency_key", sa.String(length=255), nullable=False),
            sa.Column("error_message", sa.Text(), nullable=True),
            sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                server_default=sa.text("CURRENT_TIMESTAMP"),
                nullable=True,
            ),
            sa.Column(
                "updated_at",
                sa.DateTime(timezone=True),
                server_default=sa.text("CURRENT_TIMESTAMP"),
                nullable=True,
            ),
            *constraints,
        )

    for table_name, index_name, columns in (
        ("agent_triggers", "ix_agent_triggers_id", ["id"]),
        ("agent_triggers", "ix_agent_triggers_user_id", ["user_id"]),
        ("agent_triggers", "ix_agent_triggers_agent_id", ["agent_id"]),
        ("agent_triggers", "ix_agent_triggers_type", ["type"]),
        ("agent_triggers", "ix_agent_triggers_webhook_token", ["webhook_token"]),
        ("agent_triggers", "ix_agent_triggers_next_run_at", ["next_run_at"]),
        ("trigger_runs", "ix_trigger_runs_id", ["id"]),
        ("trigger_runs", "ix_trigger_runs_trigger_id", ["trigger_id"]),
        ("trigger_runs", "ix_trigger_runs_task_id", ["task_id"]),
        ("trigger_runs", "ix_trigger_runs_background_job_id", ["background_job_id"]),
        ("trigger_runs", "ix_trigger_runs_status", ["status"]),
        ("trigger_runs", "ix_trigger_runs_source_event_id", ["source_event_id"]),
        ("trigger_runs", "ix_trigger_runs_idempotency_key", ["idempotency_key"]),
    ):
        if table_name in inspector.get_table_names() and index_name not in _index_names(
            table_name
        ):
            op.create_index(index_name, table_name, columns, unique=False)


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if "trigger_runs" in inspector.get_table_names():
        for index_name in (
            "ix_trigger_runs_idempotency_key",
            "ix_trigger_runs_source_event_id",
            "ix_trigger_runs_status",
            "ix_trigger_runs_background_job_id",
            "ix_trigger_runs_task_id",
            "ix_trigger_runs_trigger_id",
            "ix_trigger_runs_id",
        ):
            if index_name in _index_names("trigger_runs"):
                op.drop_index(index_name, table_name="trigger_runs")
        op.drop_table("trigger_runs")

    if "agent_triggers" in inspector.get_table_names():
        for index_name in (
            "ix_agent_triggers_next_run_at",
            "ix_agent_triggers_webhook_token",
            "ix_agent_triggers_type",
            "ix_agent_triggers_agent_id",
            "ix_agent_triggers_user_id",
            "ix_agent_triggers_id",
        ):
            if index_name in _index_names("agent_triggers"):
                op.drop_index(index_name, table_name="agent_triggers")
        op.drop_table("agent_triggers")
