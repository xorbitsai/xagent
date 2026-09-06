"""Add unwired execution-event storage and a legacy-only task version.

Revision ID: 20260905_task_execution_events
Revises: 20260902_seed_magento_mcp_app
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "20260905_task_execution_events"
down_revision: Union[str, None] = "20260902_seed_magento_mcp_app"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

TABLE = "task_execution_events"


def _create_events() -> None:
    op.create_table(
        TABLE,
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("event_id", sa.String(36), nullable=False),
        sa.Column(
            "task_id",
            sa.Integer(),
            sa.ForeignKey("tasks.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("sequence", sa.BigInteger(), nullable=False),
        sa.Column("scope_id", sa.String(255), nullable=False),
        sa.Column("run_id", sa.String(64), nullable=True),
        sa.Column("turn_id", sa.String(64), nullable=True),
        sa.Column("assistant_message_id", sa.String(255), nullable=True),
        sa.Column("tool_attempt_id", sa.String(255), nullable=True),
        sa.Column("idempotency_key", sa.String(255), nullable=False),
        sa.Column("kind", sa.String(64), nullable=False),
        sa.Column("payload_version", sa.Integer(), nullable=False),
        sa.Column(
            "payload", sa.JSON().with_variant(JSONB(), "postgresql"), nullable=False
        ),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.UniqueConstraint("event_id", name="uq_task_execution_events_event_id"),
        sa.UniqueConstraint(
            "task_id", "sequence", name="uq_task_execution_events_sequence"
        ),
        sa.UniqueConstraint(
            "task_id",
            "scope_id",
            "idempotency_key",
            name="uq_task_execution_events_idempotency",
        ),
        sa.CheckConstraint("sequence > 0", name="ck_task_execution_events_sequence"),
        sa.CheckConstraint(
            "payload_version > 0", name="ck_task_execution_events_payload_version"
        ),
        sa.CheckConstraint(
            "scope_id <> '' AND idempotency_key <> '' AND kind <> ''",
            name="ck_task_execution_events_identity",
        ),
    )
    op.create_index(
        "ix_task_execution_events_scope_cursor",
        TABLE,
        ["task_id", "scope_id", "sequence"],
    )


def upgrade() -> None:
    offline = op.get_context().as_sql
    if not offline:
        inspector = sa.inspect(op.get_bind())
        if not inspector.has_table("tasks"):
            return
        columns = {column["name"] for column in inspector.get_columns("tasks")}
    else:
        columns = set()
    # Inline CHECKs work with ADD COLUMN on both supported databases. Avoid
    # rebuilding SQLite's tasks table (and disturbing its inbound FKs).
    if "conversation_storage_version" not in columns:
        op.execute(
            "ALTER TABLE tasks ADD COLUMN conversation_storage_version INTEGER "
            "DEFAULT 1 NOT NULL CONSTRAINT ck_tasks_conversation_storage_version "
            "CHECK (conversation_storage_version = 1)"
        )
    if "conversation_event_sequence" not in columns:
        op.execute(
            "ALTER TABLE tasks ADD COLUMN conversation_event_sequence BIGINT "
            "DEFAULT 0 NOT NULL CONSTRAINT ck_tasks_conversation_event_sequence "
            "CHECK (conversation_event_sequence >= 0)"
        )
    if offline or not inspector.has_table(TABLE):
        _create_events()


def downgrade() -> None:
    offline = op.get_context().as_sql
    if not offline:
        inspector = sa.inspect(op.get_bind())
        if inspector.has_table(TABLE):
            op.drop_table(TABLE)
        if not inspector.has_table("tasks"):
            return
        columns = {column["name"] for column in inspector.get_columns("tasks")}
    else:
        op.drop_table(TABLE)
        columns = {"conversation_event_sequence", "conversation_storage_version"}
    # DROP COLUMN also removes its inline CHECK, without copying tasks.
    if "conversation_event_sequence" in columns:
        op.drop_column("tasks", "conversation_event_sequence")
    if "conversation_storage_version" in columns:
        op.drop_column("tasks", "conversation_storage_version")
