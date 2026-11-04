"""add trigger provider foundation

Revision ID: 20260702_add_trigger_provider_foundation
Revises: 20260629_add_gmail_watch_states
Create Date: 2026-07-02 00:00:00.000000

Adds the unified TriggerProvider identity fields on agent_triggers, the
trigger_audits table with a nullable SET NULL trigger reference, and the
per-mailbox Gmail provisioning fields on gmail_watch_states. All new columns
are nullable. Existing triggers are backfilled so the unified pipeline can
resolve them: provider defaults to the trigger type, and Gmail triggers get
resource_id resolved from their bound OAuth account's email.
"""

import json
from typing import Any, Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "20260702_add_trigger_provider_foundation"
down_revision: Union[str, None] = "20260629_add_gmail_watch_states"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _inspector() -> sa.engine.reflection.Inspector:
    return sa.inspect(op.get_bind())


def _column_names(table_name: str) -> set[str]:
    inspector = _inspector()
    if table_name not in inspector.get_table_names():
        return set()
    return {column["name"] for column in inspector.get_columns(table_name)}


def _index_names(table_name: str) -> set[str]:
    inspector = _inspector()
    if table_name not in inspector.get_table_names():
        return set()
    return {
        name
        for index in inspector.get_indexes(table_name)
        if (name := index["name"]) is not None
    }


def _agent_trigger_columns() -> tuple[sa.Column, ...]:
    return (
        sa.Column("provider", sa.String(length=64), nullable=True),
        sa.Column("callback_id", sa.String(length=128), nullable=True),
        sa.Column("resource_id", sa.String(length=255), nullable=True),
        sa.Column("secret_encrypted", sa.Text(), nullable=True),
        sa.Column("provisioning_status", sa.String(length=32), nullable=True),
        sa.Column("provisioning_error", sa.Text(), nullable=True),
    )


def _gmail_watch_columns() -> tuple[sa.Column, ...]:
    return (
        sa.Column("callback_id", sa.String(length=128), nullable=True),
        sa.Column("push_audience", sa.Text(), nullable=True),
        sa.Column("subscription_name", sa.String(length=512), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=True),
    )


def _trigger_config_oauth_account_id(config: Any) -> int | None:
    if isinstance(config, str):
        try:
            config = json.loads(config)
        except ValueError:
            return None
    if not isinstance(config, dict):
        return None
    value = config.get("oauth_account_id")
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _backfill_trigger_identity() -> None:
    """Backfill provider and Gmail resource_id on pre-existing triggers.

    Legacy rows predate the unified pipeline: provider mirrors the trigger
    type, and Gmail triggers resolve resource_id (the watched mailbox email)
    from the OAuth account referenced by config.oauth_account_id. Without
    this, provider lookup, resource authorization, and mailbox reference
    counting would all skip legacy triggers.
    """
    bind = op.get_bind()
    bind.execute(
        sa.text("UPDATE agent_triggers SET provider = type WHERE provider IS NULL")
    )

    if "user_oauth" not in _inspector().get_table_names():
        return
    gmail_rows = bind.execute(
        sa.text(
            "SELECT id, config FROM agent_triggers "
            "WHERE type = 'gmail' AND resource_id IS NULL"
        )
    ).fetchall()
    for trigger_id, config in gmail_rows:
        oauth_account_id = _trigger_config_oauth_account_id(config)
        if oauth_account_id is None:
            continue
        email_row = bind.execute(
            sa.text("SELECT email FROM user_oauth WHERE id = :oauth_id"),
            {"oauth_id": oauth_account_id},
        ).fetchone()
        email = str(email_row[0] or "").strip().lower() if email_row else ""
        if not email:
            continue
        bind.execute(
            sa.text("UPDATE agent_triggers SET resource_id = :email WHERE id = :id"),
            {"email": email, "id": trigger_id},
        )


def upgrade() -> None:
    inspector = _inspector()
    existing_tables = inspector.get_table_names()

    if "agent_triggers" in existing_tables:
        existing_columns = _column_names("agent_triggers")
        for column in _agent_trigger_columns():
            if column.name not in existing_columns:
                op.add_column("agent_triggers", column)
        _backfill_trigger_identity()

    if "gmail_watch_states" in existing_tables:
        existing_columns = _column_names("gmail_watch_states")
        for column in _gmail_watch_columns():
            if column.name not in existing_columns:
                op.add_column("gmail_watch_states", column)

    if "trigger_audits" not in existing_tables:
        constraints: list[sa.schema.SchemaItem] = [sa.PrimaryKeyConstraint("id")]
        if "agent_triggers" in existing_tables:
            constraints.append(
                sa.ForeignKeyConstraint(
                    ["trigger_id"], ["agent_triggers.id"], ondelete="SET NULL"
                )
            )
        op.create_table(
            "trigger_audits",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("trigger_id", sa.Integer(), nullable=True),
            sa.Column("provider", sa.String(length=64), nullable=True),
            sa.Column("callback_id", sa.String(length=128), nullable=True),
            sa.Column("outcome", sa.String(length=64), nullable=False),
            sa.Column("detail", sa.JSON(), nullable=True),
            sa.Column("remote_ip", sa.String(length=64), nullable=True),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                server_default=sa.text("CURRENT_TIMESTAMP"),
                nullable=True,
            ),
            *constraints,
        )

    for table_name, index_name, columns, unique in (
        ("agent_triggers", "ix_agent_triggers_provider", ["provider"], False),
        ("agent_triggers", "ix_agent_triggers_callback_id", ["callback_id"], True),
        ("agent_triggers", "ix_agent_triggers_resource_id", ["resource_id"], False),
        (
            "gmail_watch_states",
            "ix_gmail_watch_states_callback_id",
            ["callback_id"],
            True,
        ),
        ("trigger_audits", "ix_trigger_audits_id", ["id"], False),
        ("trigger_audits", "ix_trigger_audits_trigger_id", ["trigger_id"], False),
        ("trigger_audits", "ix_trigger_audits_provider", ["provider"], False),
        ("trigger_audits", "ix_trigger_audits_callback_id", ["callback_id"], False),
        ("trigger_audits", "ix_trigger_audits_outcome", ["outcome"], False),
        ("trigger_audits", "ix_trigger_audits_created_at", ["created_at"], False),
    ):
        if table_name in _inspector().get_table_names() and (
            index_name not in _index_names(table_name)
        ):
            op.create_index(index_name, table_name, columns, unique=unique)


def downgrade() -> None:
    inspector = _inspector()

    if "trigger_audits" in inspector.get_table_names():
        for index_name in (
            "ix_trigger_audits_created_at",
            "ix_trigger_audits_outcome",
            "ix_trigger_audits_callback_id",
            "ix_trigger_audits_provider",
            "ix_trigger_audits_trigger_id",
            "ix_trigger_audits_id",
        ):
            if index_name in _index_names("trigger_audits"):
                op.drop_index(index_name, table_name="trigger_audits")
        op.drop_table("trigger_audits")

    if "gmail_watch_states" in inspector.get_table_names():
        if "ix_gmail_watch_states_callback_id" in _index_names("gmail_watch_states"):
            op.drop_index(
                "ix_gmail_watch_states_callback_id", table_name="gmail_watch_states"
            )
        existing_columns = _column_names("gmail_watch_states")
        for column in reversed(_gmail_watch_columns()):
            if column.name in existing_columns:
                op.drop_column("gmail_watch_states", column.name)

    if "agent_triggers" in inspector.get_table_names():
        for index_name in (
            "ix_agent_triggers_resource_id",
            "ix_agent_triggers_callback_id",
            "ix_agent_triggers_provider",
        ):
            if index_name in _index_names("agent_triggers"):
                op.drop_index(index_name, table_name="agent_triggers")
        existing_columns = _column_names("agent_triggers")
        for column in reversed(_agent_trigger_columns()):
            if column.name in existing_columns:
                op.drop_column("agent_triggers", column.name)
