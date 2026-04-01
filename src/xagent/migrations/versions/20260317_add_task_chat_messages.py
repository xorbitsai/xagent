from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.engine.reflection import Inspector

revision: str = "20260317_add_task_chat_messages"
down_revision: Union[str, None] = "44a6d3a54c35"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    from alembic import context

    bind = context.get_bind()
    inspector = Inspector.from_engine(bind)

    # Check if table already exists
    existing_tables = inspector.get_table_names()
    if "task_chat_messages" not in existing_tables:
        # Build foreign key constraints dynamically based on what tables exist
        # This handles running migrations from empty database
        foreign_keys = []

        # Only add task_id FK if tasks table exists
        if "tasks" in existing_tables:
            foreign_keys.append(
                sa.ForeignKeyConstraint(["task_id"], ["tasks.id"], ondelete="CASCADE")
            )

        # Only add user_id FK if users table exists
        if "users" in existing_tables:
            foreign_keys.append(
                sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE")
            )

        op.create_table(
            "task_chat_messages",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("task_id", sa.Integer(), nullable=False),
            sa.Column("user_id", sa.Integer(), nullable=False),
            sa.Column("role", sa.String(length=32), nullable=False),
            sa.Column("content", sa.Text(), nullable=False),
            sa.Column(
                "message_type",
                sa.String(length=64),
                nullable=False,
                server_default="message",
            ),
            sa.Column("interactions", sa.JSON(), nullable=True),
            sa.Column(
                "created_at", sa.DateTime(timezone=True), server_default=sa.func.now()
            ),
            *foreign_keys,
            sa.PrimaryKeyConstraint("id"),
        )

    # Check and create indexes
    existing_indexes = (
        [idx["name"] for idx in inspector.get_indexes("task_chat_messages")]
        if "task_chat_messages" in existing_tables
        else []
    )
    if "ix_task_chat_messages_id" not in existing_indexes:
        op.create_index(
            op.f("ix_task_chat_messages_id"), "task_chat_messages", ["id"], unique=False
        )
    if "ix_task_chat_messages_task_id" not in existing_indexes:
        op.create_index(
            op.f("ix_task_chat_messages_task_id"),
            "task_chat_messages",
            ["task_id"],
            unique=False,
        )
    if "ix_task_chat_messages_user_id" not in existing_indexes:
        op.create_index(
            op.f("ix_task_chat_messages_user_id"),
            "task_chat_messages",
            ["user_id"],
            unique=False,
        )


def downgrade() -> None:
    from alembic import context

    bind = context.get_bind()
    inspector = Inspector.from_engine(bind)

    # Check if table exists before dropping
    existing_tables = inspector.get_table_names()
    if "task_chat_messages" in existing_tables:
        op.drop_index(
            op.f("ix_task_chat_messages_user_id"), table_name="task_chat_messages"
        )
        op.drop_index(
            op.f("ix_task_chat_messages_task_id"), table_name="task_chat_messages"
        )
        op.drop_index(op.f("ix_task_chat_messages_id"), table_name="task_chat_messages")
        op.drop_table("task_chat_messages")
