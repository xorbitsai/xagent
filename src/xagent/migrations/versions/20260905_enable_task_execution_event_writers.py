"""Allow explicitly created event-backed test tasks; keep legacy defaults."""

import sqlalchemy as sa
from alembic import op

revision = "20260905_execution_event_writers"
down_revision = "20260905_task_execution_events"
branch_labels = None
depends_on = None


def _replace_check(expression: str) -> None:
    if op.get_bind().dialect.name == "sqlite":
        # This column is not referenced by any FK/index. Replacing the column
        # avoids rebuilding tasks and triggering inbound ON DELETE CASCADE.
        # Upgrade starts with only version 1; downgrade verifies that below.
        op.drop_column("tasks", "conversation_storage_version")
        op.execute(
            "ALTER TABLE tasks ADD COLUMN conversation_storage_version INTEGER "
            "DEFAULT 1 NOT NULL CONSTRAINT ck_tasks_conversation_storage_version "
            f"CHECK ({expression})"
        )
    else:
        op.drop_constraint(
            "ck_tasks_conversation_storage_version", "tasks", type_="check"
        )
        op.create_check_constraint(
            "ck_tasks_conversation_storage_version", "tasks", expression
        )


def upgrade() -> None:
    offline = op.get_context().as_sql
    inspector = None if offline else sa.inspect(op.get_bind())
    if offline or inspector.has_table("task_chat_messages"):
        if offline or "execution_event_id" not in {
            c["name"] for c in inspector.get_columns("task_chat_messages")
        }:
            op.add_column(
                "task_chat_messages",
                sa.Column("execution_event_id", sa.String(36), nullable=True),
            )
            op.create_index(
                "ix_task_chat_messages_execution_event_id",
                "task_chat_messages",
                ["execution_event_id"],
                unique=True,
            )
    if not op.get_context().as_sql:
        inspector = sa.inspect(op.get_bind())
        if not inspector.has_table("tasks"):
            return
        check = next(
            c
            for c in inspector.get_check_constraints("tasks")
            if c["name"] == "ck_tasks_conversation_storage_version"
        )
        if "IN" in check["sqltext"].upper() or "ANY" in check["sqltext"].upper():
            return  # create_all already installed this schema
    _replace_check("conversation_storage_version IN (1, 2)")


def downgrade() -> None:
    if op.get_context().as_sql:
        raise RuntimeError("Event-writer downgrade requires an online version check")
    inspector = sa.inspect(op.get_bind())
    has_tasks = inspector.has_table("tasks")
    if has_tasks and op.get_bind().scalar(
        sa.text("SELECT count(*) FROM tasks WHERE conversation_storage_version <> 1")
    ):
        raise RuntimeError("Cannot downgrade while event-backed tasks exist")
    if inspector.has_table("task_chat_messages"):
        op.drop_index(
            "ix_task_chat_messages_execution_event_id", table_name="task_chat_messages"
        )
        op.drop_column("task_chat_messages", "execution_event_id")
    if has_tasks:
        _replace_check("conversation_storage_version = 1")
