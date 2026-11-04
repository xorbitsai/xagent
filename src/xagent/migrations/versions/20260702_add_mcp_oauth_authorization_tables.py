"""add MCP OAuth authorization tables

Revision ID: 20260702_add_mcp_oauth_tables
Revises: 20260629_add_gmail_watch_states
Create Date: 2026-07-02 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "20260702_add_mcp_oauth_tables"
down_revision: Union[str, None] = "20260629_add_gmail_watch_states"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _index_names(table_name: str) -> set[str]:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if table_name not in inspector.get_table_names():
        return set()
    return {index["name"] for index in inspector.get_indexes(table_name)}


def _create_index_if_missing(
    table_name: str, index_name: str, columns: list[str], *, unique: bool = False
) -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if table_name not in inspector.get_table_names():
        return
    if index_name not in _index_names(table_name):
        op.create_index(index_name, table_name, columns, unique=unique)


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing_tables = set(inspector.get_table_names())

    if "mcp_oauth_clients" not in existing_tables:
        constraints = [
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint(
                "lookup_hash",
                name="uq_mcp_oauth_clients_server_issuer_client",
            ),
        ]
        if "mcp_servers" in existing_tables:
            constraints.append(
                sa.ForeignKeyConstraint(
                    ["mcp_server_id"], ["mcp_servers.id"], ondelete="CASCADE"
                )
            )

        op.create_table(
            "mcp_oauth_clients",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("mcp_server_id", sa.Integer(), nullable=False),
            sa.Column("lookup_hash", sa.String(length=64), nullable=False),
            sa.Column("issuer", sa.String(length=1000), nullable=False),
            sa.Column("authorization_endpoint", sa.String(length=1000), nullable=False),
            sa.Column("token_endpoint", sa.String(length=1000), nullable=False),
            sa.Column("client_id", sa.String(length=1000), nullable=False),
            sa.Column("client_secret", sa.Text(), nullable=True),
            sa.Column(
                "token_endpoint_auth_method",
                sa.String(length=100),
                nullable=False,
                server_default="none",
            ),
            sa.Column("redirect_uri", sa.String(length=1000), nullable=False),
            sa.Column("metadata", sa.JSON(), nullable=True),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                server_default=sa.text("CURRENT_TIMESTAMP"),
                nullable=True,
            ),
            sa.Column(
                "updated_at",
                sa.DateTime(timezone=True),
                server_default=sa.text("CURRENT_TIMESTAMP"),
                nullable=True,
            ),
            *constraints,
        )

    if "mcp_oauth_grants" not in existing_tables:
        constraints = [
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint(
                "lookup_hash",
                name="uq_mcp_oauth_grants_lookup",
            ),
        ]
        if "mcp_servers" in existing_tables:
            constraints.append(
                sa.ForeignKeyConstraint(
                    ["mcp_server_id"], ["mcp_servers.id"], ondelete="CASCADE"
                )
            )
        if "users" in existing_tables:
            constraints.append(
                sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE")
            )
        constraints.append(
            sa.ForeignKeyConstraint(
                ["mcp_oauth_client_id"],
                ["mcp_oauth_clients.id"],
                ondelete="CASCADE",
            )
        )

        op.create_table(
            "mcp_oauth_grants",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("mcp_server_id", sa.Integer(), nullable=False),
            sa.Column("user_id", sa.Integer(), nullable=False),
            sa.Column("mcp_oauth_client_id", sa.Integer(), nullable=False),
            sa.Column("lookup_hash", sa.String(length=64), nullable=False),
            sa.Column("resource_owner_key", sa.String(length=512), nullable=False),
            sa.Column("issuer", sa.String(length=1000), nullable=False),
            sa.Column("resource", sa.String(length=1000), nullable=False),
            sa.Column(
                "scope", sa.String(length=1000), nullable=False, server_default=""
            ),
            sa.Column("access_token", sa.Text(), nullable=False),
            sa.Column("refresh_token", sa.Text(), nullable=True),
            sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column(
                "token_type",
                sa.String(length=50),
                nullable=False,
                server_default="Bearer",
            ),
            sa.Column(
                "status",
                sa.String(length=50),
                nullable=False,
                server_default="active",
            ),
            sa.Column("metadata", sa.JSON(), nullable=True),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                server_default=sa.text("CURRENT_TIMESTAMP"),
                nullable=True,
            ),
            sa.Column(
                "updated_at",
                sa.DateTime(timezone=True),
                server_default=sa.text("CURRENT_TIMESTAMP"),
                nullable=True,
            ),
            sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
            *constraints,
        )

    if "mcp_oauth_flow_states" not in existing_tables:
        constraints = [
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("state", name="uq_mcp_oauth_flow_states_state"),
        ]
        if "mcp_servers" in existing_tables:
            constraints.append(
                sa.ForeignKeyConstraint(
                    ["mcp_server_id"], ["mcp_servers.id"], ondelete="CASCADE"
                )
            )
        if "users" in existing_tables:
            constraints.append(
                sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE")
            )
        constraints.append(
            sa.ForeignKeyConstraint(
                ["mcp_oauth_client_id"],
                ["mcp_oauth_clients.id"],
                ondelete="CASCADE",
            )
        )

        op.create_table(
            "mcp_oauth_flow_states",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("state", sa.String(length=255), nullable=False),
            sa.Column("mcp_server_id", sa.Integer(), nullable=False),
            sa.Column("user_id", sa.Integer(), nullable=False),
            sa.Column("mcp_oauth_client_id", sa.Integer(), nullable=False),
            sa.Column("resource_owner_key", sa.String(length=512), nullable=False),
            sa.Column("issuer", sa.String(length=1000), nullable=False),
            sa.Column("resource", sa.String(length=1000), nullable=False),
            sa.Column("scope", sa.Text(), nullable=False, server_default=""),
            sa.Column("code_verifier", sa.Text(), nullable=False),
            sa.Column("redirect_after", sa.String(length=1000), nullable=True),
            sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("consumed_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                server_default=sa.text("CURRENT_TIMESTAMP"),
                nullable=True,
            ),
            *constraints,
        )

    for table_name, index_name, columns in (
        ("mcp_oauth_grants", "ix_mcp_oauth_grants_user_id", ["user_id"]),
        (
            "mcp_oauth_grants",
            "ix_mcp_oauth_grants_mcp_oauth_client_id",
            ["mcp_oauth_client_id"],
        ),
        ("mcp_oauth_grants", "ix_mcp_oauth_grants_expires_at", ["expires_at"]),
        (
            "mcp_oauth_flow_states",
            "ix_mcp_oauth_flow_states_mcp_server_id",
            ["mcp_server_id"],
        ),
        ("mcp_oauth_flow_states", "ix_mcp_oauth_flow_states_user_id", ["user_id"]),
        (
            "mcp_oauth_flow_states",
            "ix_mcp_oauth_flow_states_mcp_oauth_client_id",
            ["mcp_oauth_client_id"],
        ),
        (
            "mcp_oauth_flow_states",
            "ix_mcp_oauth_flow_states_expires_at",
            ["expires_at"],
        ),
    ):
        _create_index_if_missing(table_name, index_name, columns)


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    for table_name, index_name in (
        ("mcp_oauth_flow_states", "ix_mcp_oauth_flow_states_expires_at"),
        ("mcp_oauth_flow_states", "ix_mcp_oauth_flow_states_mcp_oauth_client_id"),
        ("mcp_oauth_flow_states", "ix_mcp_oauth_flow_states_user_id"),
        ("mcp_oauth_flow_states", "ix_mcp_oauth_flow_states_mcp_server_id"),
        ("mcp_oauth_grants", "ix_mcp_oauth_grants_expires_at"),
        ("mcp_oauth_grants", "ix_mcp_oauth_grants_mcp_oauth_client_id"),
        ("mcp_oauth_grants", "ix_mcp_oauth_grants_user_id"),
    ):
        if table_name in inspector.get_table_names() and index_name in _index_names(
            table_name
        ):
            op.drop_index(index_name, table_name=table_name)

    for table_name in (
        "mcp_oauth_flow_states",
        "mcp_oauth_grants",
        "mcp_oauth_clients",
    ):
        if table_name in inspector.get_table_names():
            op.drop_table(table_name)
