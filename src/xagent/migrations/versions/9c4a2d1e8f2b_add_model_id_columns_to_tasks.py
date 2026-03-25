"""add_model_id_columns_to_tasks

Revision ID: 9c4a2d1e8f2b
Revises: 805d5a835b7b
Create Date: 2026-03-06

Adds internal model_id columns to tasks for backward-compatible migration away
from overloading the existing *_model_name columns.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "9c4a2d1e8f2b"
down_revision: Union[str, None] = "805d5a835b7b"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing_tables = set(inspector.get_table_names())

    if "tasks" not in existing_tables:
        return

    task_columns = {column["name"] for column in inspector.get_columns("tasks")}

    if "model_id" not in task_columns:
        op.add_column(
            "tasks", sa.Column("model_id", sa.String(length=255), nullable=True)
        )
        task_columns.add("model_id")
    if "small_fast_model_id" not in task_columns:
        op.add_column(
            "tasks",
            sa.Column("small_fast_model_id", sa.String(length=255), nullable=True),
        )
        task_columns.add("small_fast_model_id")
    if "visual_model_id" not in task_columns:
        op.add_column(
            "tasks",
            sa.Column("visual_model_id", sa.String(length=255), nullable=True),
        )
        task_columns.add("visual_model_id")
    if "compact_model_id" not in task_columns:
        op.add_column(
            "tasks",
            sa.Column("compact_model_id", sa.String(length=255), nullable=True),
        )
        task_columns.add("compact_model_id")

    model_columns = (
        {column["name"] for column in inspector.get_columns("models")}
        if "models" in existing_tables
        else set()
    )

    # Safe best-effort backfill: only copy legacy values when they match an existing
    # registered internal model_id (avoid guessing from provider model_name).
    if "model_id" in model_columns:
        if {"model_id", "model_name"} <= task_columns:
            op.execute(
                """
                UPDATE tasks
                SET model_id = model_name
                WHERE model_id IS NULL
                  AND model_name IS NOT NULL
                  AND EXISTS (
                    SELECT 1 FROM models m WHERE m.model_id = tasks.model_name
                  )
                """
            )
        if {"small_fast_model_id", "small_fast_model_name"} <= task_columns:
            op.execute(
                """
                UPDATE tasks
                SET small_fast_model_id = small_fast_model_name
                WHERE small_fast_model_id IS NULL
                  AND small_fast_model_name IS NOT NULL
                  AND EXISTS (
                    SELECT 1 FROM models m WHERE m.model_id = tasks.small_fast_model_name
                  )
                """
            )
        if {"visual_model_id", "visual_model_name"} <= task_columns:
            op.execute(
                """
                UPDATE tasks
                SET visual_model_id = visual_model_name
                WHERE visual_model_id IS NULL
                  AND visual_model_name IS NOT NULL
                  AND EXISTS (
                    SELECT 1 FROM models m WHERE m.model_id = tasks.visual_model_name
                  )
                """
            )
        if {"compact_model_id", "compact_model_name"} <= task_columns:
            op.execute(
                """
                UPDATE tasks
                SET compact_model_id = compact_model_name
                WHERE compact_model_id IS NULL
                  AND compact_model_name IS NOT NULL
                  AND EXISTS (
                    SELECT 1 FROM models m WHERE m.model_id = tasks.compact_model_name
                  )
                """
            )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if "tasks" not in set(inspector.get_table_names()):
        return

    task_columns = {column["name"] for column in inspector.get_columns("tasks")}

    if "compact_model_id" in task_columns:
        op.drop_column("tasks", "compact_model_id")
    if "visual_model_id" in task_columns:
        op.drop_column("tasks", "visual_model_id")
    if "small_fast_model_id" in task_columns:
        op.drop_column("tasks", "small_fast_model_id")
    if "model_id" in task_columns:
        op.drop_column("tasks", "model_id")
