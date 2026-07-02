"""add mcp oauth connector tables and columns

Revision ID: 5bb3df522a7d
Revises: 20260629_add_gmail_watch_states
Create Date: 2026-07-02

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.engine.reflection import Inspector

revision: str = "5bb3df522a7d"
down_revision: Union[str, None] = "20260629_add_gmail_watch_states"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = Inspector.from_engine(bind)

    # Check if prerequisite tables exist
    tables = inspector.get_table_names()
    if "users" not in tables or "mcp_servers" not in tables:
        # Tables don't exist yet, will be created by SQLAlchemy
        return

    op.add_column("mcp_servers", sa.Column("oauth_client", sa.JSON(), nullable=True))
    op.add_column(
        "mcp_servers", sa.Column("auth_server_metadata", sa.JSON(), nullable=True)
    )

    op.create_table(
        "mcp_user_oauth_tokens",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("mcpserver_id", sa.Integer(), nullable=False),
        sa.Column("access_token", sa.Text(), nullable=True),
        sa.Column("refresh_token", sa.Text(), nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("token_type", sa.String(length=50), nullable=True),
        sa.Column("scope", sa.Text(), nullable=True),
        sa.Column(
            "status", sa.String(length=20), nullable=False, server_default="pending"
        ),
        sa.Column("pkce_verifier", sa.Text(), nullable=True),
        sa.Column("state", sa.String(length=255), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now()
        ),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["mcpserver_id"], ["mcp_servers.id"], ondelete="CASCADE"
        ),
        sa.UniqueConstraint("user_id", "mcpserver_id", name="uq_mcp_user_server_token"),
    )
    op.create_index(
        "ix_mcp_user_oauth_tokens_user_id", "mcp_user_oauth_tokens", ["user_id"]
    )
    op.create_index(
        "ix_mcp_user_oauth_tokens_mcpserver_id",
        "mcp_user_oauth_tokens",
        ["mcpserver_id"],
    )
    op.create_index(
        "ix_mcp_user_oauth_tokens_state", "mcp_user_oauth_tokens", ["state"]
    )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = Inspector.from_engine(bind)

    tables = inspector.get_table_names()
    if "mcp_user_oauth_tokens" in tables:
        op.drop_index(
            "ix_mcp_user_oauth_tokens_state", table_name="mcp_user_oauth_tokens"
        )
        op.drop_index(
            "ix_mcp_user_oauth_tokens_mcpserver_id",
            table_name="mcp_user_oauth_tokens",
        )
        op.drop_index(
            "ix_mcp_user_oauth_tokens_user_id", table_name="mcp_user_oauth_tokens"
        )
        op.drop_table("mcp_user_oauth_tokens")

    if "mcp_servers" in tables:
        columns = [col["name"] for col in inspector.get_columns("mcp_servers")]
        if "auth_server_metadata" in columns:
            op.drop_column("mcp_servers", "auth_server_metadata")
        if "oauth_client" in columns:
            op.drop_column("mcp_servers", "oauth_client")
