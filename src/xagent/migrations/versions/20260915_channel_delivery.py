"""Persist recoverable final channel delivery destinations and claims."""

import sqlalchemy as sa
from alembic import op

revision = "20260915_channel_delivery"
down_revision = "20260914_update_github_description"
branch_labels = None
depends_on = None


def upgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    if not inspector.has_table("task_execution_commands") or inspector.has_table(
        "task_channel_deliveries"
    ):
        return
    op.create_table(
        "task_channel_deliveries",
        sa.Column(
            "command_id",
            sa.Integer(),
            sa.ForeignKey("task_execution_commands.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column(
            "channel_id",
            sa.Integer(),
            sa.ForeignKey("user_channels.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("destination", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(16), nullable=False, server_default="pending"),
        sa.Column("claim_token", sa.String(64), nullable=True),
        sa.Column("available_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("delivered_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index(
        "ix_task_channel_delivery_pending",
        "task_channel_deliveries",
        ["channel_id", "status", "available_at"],
    )


def downgrade() -> None:
    if sa.inspect(op.get_bind()).has_table("task_channel_deliveries"):
        op.drop_index(
            "ix_task_channel_delivery_pending", table_name="task_channel_deliveries"
        )
        op.drop_table("task_channel_deliveries")
