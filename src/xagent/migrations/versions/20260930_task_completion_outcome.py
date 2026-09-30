"""Preserve the semantic outcome separately from execution control status."""

import sqlalchemy as sa
from alembic import op

revision = "20260930_task_completion_outcome"
down_revision = "20260929_task_attachment_detachment"
branch_labels = None
depends_on = None


def upgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    if inspector.has_table("tasks") and "completion_outcome" not in {
        column["name"] for column in inspector.get_columns("tasks")
    }:
        op.add_column(
            "tasks", sa.Column("completion_outcome", sa.String(20), nullable=True)
        )


def downgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    if inspector.has_table("tasks") and "completion_outcome" in {
        column["name"] for column in inspector.get_columns("tasks")
    }:
        # No FK, index, or CHECK references this column. Drop it in place so
        # SQLite preserves inbound rows and the other columns' inline CHECKs.
        op.drop_column("tasks", "completion_outcome")
