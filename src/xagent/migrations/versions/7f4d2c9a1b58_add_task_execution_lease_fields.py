"""Add task execution lease fields

Revision ID: 7f4d2c9a1b58
Revises: 20260509_add_delegate_agent_ids_to_tasks
Create Date: 2026-05-09
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.engine.reflection import Inspector

# revision identifiers, used by Alembic.
revision = "7f4d2c9a1b58"
down_revision = "20260509_add_delegate_agent_ids_to_tasks"
branch_labels = None
depends_on = None


def _column_names(table_name: str) -> set[str]:
    bind = op.get_bind()
    inspector = Inspector.from_engine(bind)
    return {column["name"] for column in inspector.get_columns(table_name)}


def upgrade() -> None:
    bind = op.get_bind()
    inspector = Inspector.from_engine(bind)
    if "tasks" not in inspector.get_table_names():
        return

    columns = _column_names("tasks")
    if "runner_id" not in columns:
        op.add_column("tasks", sa.Column("runner_id", sa.String(length=255)))
    if "lease_expires_at" not in columns:
        op.add_column(
            "tasks", sa.Column("lease_expires_at", sa.DateTime(timezone=True))
        )
    if "last_heartbeat_at" not in columns:
        op.add_column(
            "tasks", sa.Column("last_heartbeat_at", sa.DateTime(timezone=True))
        )
    if "last_checkpoint_event_id" not in columns:
        op.add_column(
            "tasks",
            sa.Column("last_checkpoint_event_id", sa.String(length=255)),
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = Inspector.from_engine(bind)
    if "tasks" not in inspector.get_table_names():
        return

    columns = _column_names("tasks")
    if "last_checkpoint_event_id" in columns:
        op.drop_column("tasks", "last_checkpoint_event_id")
    if "last_heartbeat_at" in columns:
        op.drop_column("tasks", "last_heartbeat_at")
    if "lease_expires_at" in columns:
        op.drop_column("tasks", "lease_expires_at")
    if "runner_id" in columns:
        op.drop_column("tasks", "runner_id")
