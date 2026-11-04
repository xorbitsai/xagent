"""add task visibility flag

Revision ID: 20260525_add_task_visibility
Revises: 20260525_add_trace_checkpoint_blobs
Create Date: 2026-05-25 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect

revision: str = "20260525_add_task_visibility"
down_revision: Union[str, None] = "20260525_add_trace_checkpoint_blobs"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    if "tasks" not in inspector.get_table_names():
        return

    existing_columns = {col["name"] for col in inspector.get_columns("tasks")}
    if "is_visible" not in existing_columns:
        op.add_column(
            "tasks",
            sa.Column(
                "is_visible",
                sa.Boolean(),
                server_default=sa.true(),
                nullable=False,
            ),
        )

    existing_indexes = {idx["name"] for idx in inspector.get_indexes("tasks")}
    if "ix_tasks_is_visible" not in existing_indexes:
        op.create_index("ix_tasks_is_visible", "tasks", ["is_visible"])


def downgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    if "tasks" not in inspector.get_table_names():
        return

    existing_indexes = {idx["name"] for idx in inspector.get_indexes("tasks")}
    if "ix_tasks_is_visible" in existing_indexes:
        op.drop_index("ix_tasks_is_visible", table_name="tasks")

    existing_columns = {col["name"] for col in inspector.get_columns("tasks")}
    if "is_visible" in existing_columns:
        op.drop_column("tasks", "is_visible")
