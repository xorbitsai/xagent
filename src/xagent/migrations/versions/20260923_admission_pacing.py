"""Add opt-in startup pacing without altering existing admission budgets."""

import sqlalchemy as sa
from alembic import op

revision = "20260923_admission_pacing"
down_revision = "20260923_task_admission"
branch_labels = None
depends_on = None


def upgrade() -> None:
    inspector = None if op.get_context().as_sql else sa.inspect(op.get_bind())
    if inspector is not None:
        if not inspector.has_table("task_admission_buckets") or inspector.has_table(
            "task_admission_pacing"
        ):
            return
    op.create_table(
        "task_admission_pacing",
        sa.Column(
            "bucket_key",
            sa.String(255),
            sa.ForeignKey("task_admission_buckets.key", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("interval_seconds", sa.Float(), nullable=False),
        sa.Column("burst", sa.Integer(), nullable=False),
        sa.Column("lane", sa.String(16), nullable=False),
        sa.Column("next_start_at", sa.Float(), nullable=False, server_default="0"),
    )


def downgrade() -> None:
    if op.get_context().as_sql or sa.inspect(op.get_bind()).has_table(
        "task_admission_pacing"
    ):
        op.drop_table("task_admission_pacing")
