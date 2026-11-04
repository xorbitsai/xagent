"""Separate command retry scheduling from processing ownership.

Run with all old execution processes stopped. Processing rows retain their
application evidence; the task owner reconciles them instead of resetting them.
"""

import sqlalchemy as sa
from alembic import op

revision = "20260918_command_retry_at"
down_revision = "20260916_update_hubspot_description"
branch_labels = None
depends_on = None


def upgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    # Core task tables are metadata-owned and can be absent in Alembic-only runs.
    if not inspector.has_table("task_execution_commands"):
        return
    if any(
        column["name"] == "retry_available_at"
        for column in inspector.get_columns("task_execution_commands")
    ):
        return
    op.add_column(
        "task_execution_commands",
        sa.Column("retry_available_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.execute(
        sa.text(
            "UPDATE task_execution_commands SET retry_available_at = claim_expires_at "
            "WHERE status = 'pending'"
        )
    )


def downgrade() -> None:
    if not sa.inspect(op.get_bind()).has_table("task_execution_commands"):
        return
    op.drop_column("task_execution_commands", "retry_available_at")
