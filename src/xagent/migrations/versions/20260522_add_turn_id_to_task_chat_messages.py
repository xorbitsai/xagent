"""add turn_id to task_chat_messages

Revision ID: 20260522_add_task_chat_message_turn_id
Revises: 20260521_merge_alembic_heads
Create Date: 2026-05-22 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.engine.reflection import Inspector

# revision identifiers, used by Alembic.
revision: str = "20260522_add_task_chat_message_turn_id"
down_revision: Union[str, None] = "20260521_merge_alembic_heads"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    from alembic import context

    bind = context.get_bind()
    inspector = Inspector.from_engine(bind)

    if "task_chat_messages" not in inspector.get_table_names():
        return

    existing_columns = {
        col["name"] for col in inspector.get_columns("task_chat_messages")
    }
    if "turn_id" not in existing_columns:
        op.add_column(
            "task_chat_messages",
            sa.Column("turn_id", sa.String(length=64), nullable=True),
        )

    inspector = Inspector.from_engine(bind)
    existing_indexes = {
        idx["name"] for idx in inspector.get_indexes("task_chat_messages")
    }
    if "ix_task_chat_messages_turn_id" not in existing_indexes:
        op.create_index(
            op.f("ix_task_chat_messages_turn_id"),
            "task_chat_messages",
            ["turn_id"],
            unique=False,
        )


def downgrade() -> None:
    from alembic import context

    bind = context.get_bind()
    inspector = Inspector.from_engine(bind)

    if "task_chat_messages" not in inspector.get_table_names():
        return

    existing_indexes = {
        idx["name"] for idx in inspector.get_indexes("task_chat_messages")
    }
    if "ix_task_chat_messages_turn_id" in existing_indexes:
        op.drop_index(op.f("ix_task_chat_messages_turn_id"), "task_chat_messages")

    existing_columns = {
        col["name"] for col in inspector.get_columns("task_chat_messages")
    }
    if "turn_id" in existing_columns:
        op.drop_column("task_chat_messages", "turn_id")
