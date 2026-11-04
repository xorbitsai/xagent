"""Bound channel delivery retries without losing existing pending destinations."""

import sqlalchemy as sa
from alembic import op

revision = "20260915_delivery_retry_budget"
down_revision = "20260915_channel_delivery"
branch_labels = None
depends_on = None


def upgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    if not inspector.has_table("task_channel_deliveries") or "failure_count" in {
        column["name"] for column in inspector.get_columns("task_channel_deliveries")
    }:
        return
    with op.batch_alter_table("task_channel_deliveries") as batch:
        batch.add_column(
            sa.Column("failure_count", sa.Integer(), nullable=False, server_default="0")
        )


def downgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    if not inspector.has_table("task_channel_deliveries") or "failure_count" not in {
        column["name"] for column in inspector.get_columns("task_channel_deliveries")
    }:
        return
    with op.batch_alter_table("task_channel_deliveries") as batch:
        batch.drop_column("failure_count")
