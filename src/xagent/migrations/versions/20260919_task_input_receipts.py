"""Persist first-input identity without adding another execution lease."""

import sqlalchemy as sa
from alembic import op

revision = "20260919_task_input_receipts"
down_revision = "20260918_command_retry_at"
branch_labels = None
depends_on = None


def upgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    # These parent tables are metadata-owned in Alembic-only installations.
    if inspector.has_table("task_input_receipts") or not all(
        inspector.has_table(name) for name in ("tasks", "task_execution_commands")
    ):
        return
    op.create_table(
        "task_input_receipts",
        sa.Column("identity_hash", sa.String(64), primary_key=True),
        sa.Column("payload_hash", sa.String(64), nullable=False),
        sa.Column(
            "task_id", sa.Integer, sa.ForeignKey("tasks.id", ondelete="SET NULL")
        ),
        sa.Column(
            "command_db_id",
            sa.Integer,
            sa.ForeignKey("task_execution_commands.id", ondelete="SET NULL"),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    for column in ("task_id", "command_db_id"):
        op.create_index(
            f"ix_task_input_receipts_{column}", "task_input_receipts", [column]
        )


def downgrade() -> None:
    if sa.inspect(op.get_bind()).has_table("task_input_receipts"):
        op.drop_table("task_input_receipts")
