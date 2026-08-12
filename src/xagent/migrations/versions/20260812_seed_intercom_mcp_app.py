"""seed built-in Intercom (OAuth) MCP connector

Revision ID: 20260812_seed_intercom_mcp_app
Revises: 20260810_add_task_interaction_protocol_version
Create Date: 2026-08-12 00:00:00.000000

"""

import os
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "20260812_seed_intercom_mcp_app"
down_revision: Union[str, None] = "20260810_add_task_interaction_protocol_version"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

FULL_OAUTH_PROVIDERS_TABLE = sa.table(
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

APP_ID = "intercom"


def _filter_row(row: dict[str, object], allowed_columns: set[str]) -> dict[str, object]:
    return {key: value for key, value in row.items() if key in allowed_columns}


def _intercom_provider_row() -> dict[str, object]:
    return {
        "provider_name": "intercom",
        "name": "Intercom",
        "client_id": os.environ.get("INTERCOM_CLIENT_ID", ""),
        "client_secret": os.environ.get("INTERCOM_CLIENT_SECRET", ""),
        # Single global authorize host: once a public app clears Intercom App
        # Review, Intercom replicates it across US/EU/AU automatically.
        "auth_url": "https://app.intercom.com/oauth",
        # No region prefix: api.intercom.io auto-routes to the workspace's
        # actual hosting region (US/EU/AU), unlike Intercom's hosted MCP
        # server (mcp.intercom.com / mcp.eu.intercom.com) which has separate
        # per-region endpoints and does not support AU workspaces at all.
        "token_url": "https://api.intercom.io/auth/eagle/token",
        "redirect_uri": os.environ.get("INTERCOM_REDIRECT_URI", ""),
        "userinfo_url": "https://api.intercom.io/me",
        "user_id_path": "id",
        "email_path": "email",
        # Intercom has no `scope` authorize-URL param; granted permissions
        # come entirely from the app's Authentication settings in the
        # Developer Hub, so there is nothing to list here.
        "default_scopes": [],
    }


def _intercom_app_row() -> dict[str, object]:
    return {
        "app_id": APP_ID,
        "name": "Intercom",
        "description": "Connect to Intercom to search contacts, review conversations, and reply to customers.",
        "icon": "https://www.google.com/s2/favicons?domain=intercom.com&sz=128",
        "transport": "oauth",
        "provider_name": "intercom",
        "category": "Support",
        "oauth_scopes": [],
        "is_visible_in_connector": True,
        "launch_config": {
            "command": "python",
            "args": ["-m", "xagent.web.tools.mcp.intercom"],
            "env_mapping": {"INTERCOM_ACCESS_TOKEN": "access_token"},
        },
    }


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing_tables = set(inspector.get_table_names())

    if "oauth_providers" in existing_tables:
        oauth_columns = {
            column["name"] for column in inspector.get_columns("oauth_providers")
        }
        existing_provider_names = set(
            bind.execute(
                sa.select(FULL_OAUTH_PROVIDERS_TABLE.c.provider_name)
            ).scalars()
        )
        if "intercom" not in existing_provider_names:
            bind.execute(
                sa.insert(FULL_OAUTH_PROVIDERS_TABLE),
                [_filter_row(_intercom_provider_row(), oauth_columns)],
            )

    if "public_mcp_apps" in existing_tables:
        app_columns = {
            column["name"] for column in inspector.get_columns("public_mcp_apps")
        }
        existing_app_ids = set(
            bind.execute(sa.select(PUBLIC_MCP_APPS_TABLE.c.app_id)).scalars()
        )
        if APP_ID not in existing_app_ids:
            bind.execute(
                sa.insert(PUBLIC_MCP_APPS_TABLE),
                [_filter_row(_intercom_app_row(), app_columns)],
            )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing_tables = set(inspector.get_table_names())

    if "public_mcp_apps" in existing_tables:
        # Only the catalog entry is removed. Any user OAuth connections created
        # against this app are not owned by this migration and are cleaned up
        # through the normal disconnect path.
        bind.execute(
            sa.delete(PUBLIC_MCP_APPS_TABLE).where(
                PUBLIC_MCP_APPS_TABLE.c.app_id == APP_ID
            )
        )

    if "oauth_providers" not in existing_tables:
        return

    if "public_mcp_apps" in existing_tables:
        remaining_intercom_apps = bind.execute(
            sa.select(sa.func.count())
            .select_from(PUBLIC_MCP_APPS_TABLE)
            .where(PUBLIC_MCP_APPS_TABLE.c.provider_name == "intercom")
        ).scalar()
        if remaining_intercom_apps:
            return

    # Delete unconditionally by provider_name, matching the sibling seed
    # migrations (slack, meta, google-maps). A shape-matching guard would
    # protect the wrong thing: an admin recreating the *real* Intercom
    # provider enters Intercom's canonical URLs, which would match the guard
    # and be deleted anyway -- only a differently-shaped row would survive.
    bind.execute(
        sa.delete(FULL_OAUTH_PROVIDERS_TABLE).where(
            FULL_OAUTH_PROVIDERS_TABLE.c.provider_name == "intercom"
        )
    )
