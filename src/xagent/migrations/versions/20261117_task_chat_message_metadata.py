"""Add message_metadata to task_chat_messages.

Server-set metadata for a user message (``hidden``, ``kind``,
``interaction_id``); ``hidden`` is a display flag, not "hidden from the model".
Nullable; existing rows stay NULL.

Revision ID: 20261117_chat_message_metadata
Revises: 20261117_seed_chargebee_mcp_app
"""

import sqlalchemy as sa
from alembic import op

revision = "20261117_chat_message_metadata"
down_revision = "20261117_seed_chargebee_mcp_app"
branch_labels = None
depends_on = None


def upgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    if not inspector.has_table("task_chat_messages"):
        return
    columns = {c["name"] for c in inspector.get_columns("task_chat_messages")}
    if "message_metadata" not in columns:
        op.add_column(
            "task_chat_messages",
            sa.Column("message_metadata", sa.JSON(), nullable=True),
        )


def downgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    if not inspector.has_table("task_chat_messages"):
        return
    columns = {c["name"] for c in inspector.get_columns("task_chat_messages")}
    if "message_metadata" in columns:
        with op.batch_alter_table("task_chat_messages") as batch:
            batch.drop_column("message_metadata")
