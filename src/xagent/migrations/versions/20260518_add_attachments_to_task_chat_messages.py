"""add attachments column to task_chat_messages

Revision ID: 20260518_add_chat_message_attachments
Revises: 20260509_add_durable_file_storage_fields
Create Date: 2026-05-18 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.engine.reflection import Inspector

# revision identifiers, used by Alembic.
revision: str = "20260518_add_chat_message_attachments"
down_revision: Union[str, None] = "20260509_add_durable_file_storage_fields"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    from alembic import context

    bind = context.get_bind()
    inspector = Inspector.from_engine(bind)

    if "task_chat_messages" not in inspector.get_table_names():
        # Base table doesn't exist yet; nothing to do for this migration.
        return

    existing_columns = {
        col["name"] for col in inspector.get_columns("task_chat_messages")
    }
    if "attachments" not in existing_columns:
        op.add_column(
            "task_chat_messages",
            sa.Column("attachments", sa.JSON(), nullable=True),
        )


def downgrade() -> None:
    from alembic import context

    bind = context.get_bind()
    inspector = Inspector.from_engine(bind)

    if "task_chat_messages" not in inspector.get_table_names():
        return

    existing_columns = {
        col["name"] for col in inspector.get_columns("task_chat_messages")
    }
    if "attachments" in existing_columns:
        op.drop_column("task_chat_messages", "attachments")
