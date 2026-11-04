"""Add durable execution admission without changing existing command state."""

import sqlalchemy as sa
from alembic import op

revision = "20260923_task_admission"
down_revision = "20260922_task_last_activity_at"
branch_labels = None
depends_on = None


def upgrade() -> None:
    offline = op.get_context().as_sql
    inspector = None if offline else sa.inspect(op.get_bind())
    if inspector is not None and not inspector.has_table("task_execution_commands"):
        return
    if inspector is None or not inspector.has_table("task_admission_buckets"):
        op.create_table(
            "task_admission_buckets",
            sa.Column("key", sa.String(255), primary_key=True),
            sa.Column("capacity", sa.Integer(), nullable=False),
            sa.Column("max_pending", sa.Integer(), nullable=False),
        )
    if inspector is None or not inspector.has_table("task_admission_tickets"):
        op.create_table(
            "task_admission_tickets",
            sa.Column(
                "command_id",
                sa.Integer(),
                sa.ForeignKey("task_execution_commands.id", ondelete="CASCADE"),
                primary_key=True,
            ),
            sa.Column(
                "bucket_key",
                sa.String(255),
                sa.ForeignKey("task_admission_buckets.key"),
                nullable=False,
            ),
            sa.Column(
                "task_id",
                sa.Integer(),
                sa.ForeignKey("tasks.id", ondelete="CASCADE"),
                nullable=False,
            ),
            sa.Column("runner_id", sa.String(255), nullable=True),
            sa.Column("owner_attempt_id", sa.String(64), nullable=True),
        )
        op.create_index(
            "ix_task_admission_bucket_command",
            "task_admission_tickets",
            ["bucket_key", "command_id"],
        )
        op.create_index(
            "ix_task_admission_tickets_task_id", "task_admission_tickets", ["task_id"]
        )


def downgrade() -> None:
    offline = op.get_context().as_sql
    inspector = None if offline else sa.inspect(op.get_bind())
    for name in ("task_admission_tickets", "task_admission_buckets"):
        if inspector is None or inspector.has_table(name):
            op.drop_table(name)
