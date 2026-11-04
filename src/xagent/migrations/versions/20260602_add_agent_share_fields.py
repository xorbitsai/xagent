"""add agent share fields

Revision ID: 20260602_add_agent_share_fields
Revises: 20260529_add_oidc_consumed_tokens
Create Date: 2026-06-02 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "20260602_add_agent_share_fields"
down_revision: Union[str, tuple[str, str], None] = "20260529_add_oidc_consumed_tokens"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _index_names(table_name: str) -> set[str]:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if table_name not in inspector.get_table_names():
        return set()
    return {index["name"] for index in inspector.get_indexes(table_name)}


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "agents" not in inspector.get_table_names():
        return

    existing_columns = {col["name"] for col in inspector.get_columns("agents")}
    if "share_enabled" not in existing_columns:
        op.add_column(
            "agents",
            sa.Column(
                "share_enabled",
                sa.Boolean(),
                nullable=False,
                server_default=sa.false(),
            ),
        )
    if "share_token" not in existing_columns:
        op.add_column(
            "agents",
            sa.Column("share_token", sa.String(length=255), nullable=True),
        )
    if "share_updated_at" not in existing_columns:
        op.add_column(
            "agents",
            sa.Column("share_updated_at", sa.DateTime(timezone=True), nullable=True),
        )

    existing_indexes = _index_names("agents")
    if "ix_agents_share_token" not in existing_indexes:
        op.create_index(
            "ix_agents_share_token", "agents", ["share_token"], unique=False
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "agents" not in inspector.get_table_names():
        return

    existing_indexes = _index_names("agents")
    if "ix_agents_share_token" in existing_indexes:
        op.drop_index("ix_agents_share_token", table_name="agents")

    existing_columns = {col["name"] for col in inspector.get_columns("agents")}
    if "share_updated_at" in existing_columns:
        op.drop_column("agents", "share_updated_at")
    if "share_token" in existing_columns:
        op.drop_column("agents", "share_token")
    if "share_enabled" in existing_columns:
        op.drop_column("agents", "share_enabled")
