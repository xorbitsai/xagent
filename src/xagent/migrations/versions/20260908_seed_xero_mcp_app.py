"""Seed the Xero provider-OAuth connector without replacing operator settings.

Revision ID: 20260908_seed_xero_mcp_app
Revises: 20260901_seed_zendesk_mcp_app
"""

import os

import sqlalchemy as sa
from alembic import op

revision = "20260908_seed_xero_mcp_app"
down_revision = "20260901_seed_zendesk_mcp_app"
branch_labels = None
depends_on = None

OAUTH_PROVIDERS_TABLE = sa.table(
    "oauth_providers",
    sa.column("provider_name", sa.String),
    sa.column("name", sa.String),
    sa.column("client_id", sa.String),
    sa.column("client_secret", sa.String),
    sa.column("auth_url", sa.String),
    sa.column("token_url", sa.String),
    sa.column("redirect_uri", sa.String),
    sa.column("userinfo_url", sa.String),
    sa.column("user_id_path", sa.String),
    sa.column("email_path", sa.String),
    sa.column("default_scopes", sa.JSON),
)

PUBLIC_MCP_APPS_TABLE = sa.table(
    "public_mcp_apps",
    sa.column("app_id", sa.String),
    sa.column("name", sa.String),
    sa.column("description", sa.Text),
    sa.column("icon", sa.String),
    sa.column("transport", sa.String),
    sa.column("provider_name", sa.String),
    sa.column("category", sa.String),
    sa.column("oauth_scopes", sa.JSON),
    sa.column("is_visible_in_connector", sa.Boolean),
    sa.column("launch_config", sa.JSON),
)


def upgrade() -> None:
    bind = op.get_bind()
    provider = {
        "provider_name": "xero",
        "name": "Xero",
        "client_id": os.environ.get("XERO_CLIENT_ID", ""),
        "client_secret": os.environ.get("XERO_CLIENT_SECRET", ""),
        "auth_url": "https://login.xero.com/identity/connect/authorize",
        "token_url": "https://identity.xero.com/connect/token",
        "redirect_uri": os.environ.get("XERO_REDIRECT_URI", ""),
        "userinfo_url": "https://identity.xero.com/connect/userinfo",
        "user_id_path": "sub",
        "email_path": "email",
        "default_scopes": ["openid", "profile", "email"],
    }
    app = {
        "app_id": "xero",
        "name": "Xero",
        "description": "Connect to Xero to manage contacts, invoices, payments, and accounting records.",
        "icon": "https://www.google.com/s2/favicons?domain=xero.com&sz=128",
        "transport": "oauth",
        "provider_name": "xero",
        "category": "Finance",
        "oauth_scopes": [
            "openid",
            "profile",
            "email",
            "accounting.contacts",
            "accounting.settings",
            "accounting.invoices",
            "accounting.payments",
            "accounting.banktransactions",
            "accounting.manualjournals",
            "offline_access",
        ],
        "is_visible_in_connector": True,
        # Match the existing catalog launcher; inject only the actor's token.
        "launch_config": {
            "command": "npx",
            "args": ["-y", "@xeroapi/xero-mcp-server@latest"],
            "env_mapping": {"XERO_CLIENT_BEARER_TOKEN": "access_token"},
        },
    }
    for table, identity, row in (
        (OAUTH_PROVIDERS_TABLE, "provider_name", provider),
        (PUBLIC_MCP_APPS_TABLE, "app_id", app),
    ):
        existing = bind.execute(
            sa.select(table.c[identity]).where(table.c[identity] == "xero")
        ).first()
        if existing is None:
            bind.execute(sa.insert(table), row)


def downgrade() -> None:
    # Existing and seeded Xero rows are indistinguishable. Preserve connections.
    pass
