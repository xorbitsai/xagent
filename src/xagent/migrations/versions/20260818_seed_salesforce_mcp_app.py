"""seed built-in Salesforce (OAuth) MCP connector

Revision ID: 20260818_seed_salesforce_mcp_app
Revises: 20260818_add_instance_url_to_user_oauth
Create Date: 2026-08-18 00:00:00.000000

"""

import os
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "20260818_seed_salesforce_mcp_app"
down_revision: Union[str, None] = "20260818_add_instance_url_to_user_oauth"
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

APP_ID = "salesforce"

SALESFORCE_SCOPES = ["api", "refresh_token", "openid"]


def _filter_row(row: dict[str, object], allowed_columns: set[str]) -> dict[str, object]:
    return {key: value for key, value in row.items() if key in allowed_columns}


def _salesforce_provider_row() -> dict[str, object]:
    return {
        "provider_name": "salesforce",
        "name": "Salesforce",
        "client_id": os.environ.get("SALESFORCE_CLIENT_ID", ""),
        "client_secret": os.environ.get("SALESFORCE_CLIENT_SECRET", ""),
        "auth_url": "https://login.salesforce.com/services/oauth2/authorize",
        "token_url": "https://login.salesforce.com/services/oauth2/token",
        "redirect_uri": os.environ.get("SALESFORCE_REDIRECT_URI", ""),
        # Left empty on purpose, not because the URL is unknown -- see the
        # matching comment on the registry row for why.
        "userinfo_url": "",
        "user_id_path": "user_id",
        "email_path": "email",
        "default_scopes": SALESFORCE_SCOPES,
    }


def _salesforce_app_row() -> dict[str, object]:
    return {
        "app_id": APP_ID,
        "name": "Salesforce",
        "description": "Connect to Salesforce to query and manage records (accounts, contacts, leads, opportunities, and custom objects) with SOQL/SOSL, and browse object schemas.",
        "icon": "https://www.google.com/s2/favicons?domain=salesforce.com&sz=128",
        "transport": "oauth",
        "provider_name": "salesforce",
        "category": "CRM",
        "oauth_scopes": SALESFORCE_SCOPES,
        "is_visible_in_connector": True,
        "launch_config": {
            "command": "python",
            "args": ["-m", "xagent.web.tools.mcp.salesforce"],
            "env_mapping": {
                "SALESFORCE_ACCESS_TOKEN": "access_token",
                "SALESFORCE_INSTANCE_URL": "instance_url",
            },
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
        if "salesforce" not in existing_provider_names:
            bind.execute(
                sa.insert(FULL_OAUTH_PROVIDERS_TABLE),
                [_filter_row(_salesforce_provider_row(), oauth_columns)],
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
                [_filter_row(_salesforce_app_row(), app_columns)],
            )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing_tables = set(inspector.get_table_names())

    if "public_mcp_apps" in existing_tables:
        # Only delete the catalog entry when it still matches the static
        # shape this migration seeded, mirroring the oauth_providers guard
        # below -- an unconditional delete-by-app_id would remove a
        # pre-existing operator row that happened to already occupy app_id
        # "salesforce" before this migration ever ran (upgrade()'s own
        # `if APP_ID not in existing_app_ids` check would have skipped
        # inserting over it, so upgrade and downgrade must agree on what
        # "this migration's row" means). name/transport/provider_name are
        # NOT NULL-or-always-set core columns present since the table's
        # creation and none of them are env-dependent, so they can be
        # matched unconditionally. Any user OAuth connections created
        # against this app are not owned by this migration either way and
        # are cleaned up through the normal disconnect path.
        seeded_app = _salesforce_app_row()
        bind.execute(
            sa.delete(PUBLIC_MCP_APPS_TABLE)
            .where(PUBLIC_MCP_APPS_TABLE.c.app_id == APP_ID)
            .where(PUBLIC_MCP_APPS_TABLE.c.name == seeded_app["name"])
            .where(PUBLIC_MCP_APPS_TABLE.c.transport == seeded_app["transport"])
            .where(PUBLIC_MCP_APPS_TABLE.c.provider_name == seeded_app["provider_name"])
        )

    if "oauth_providers" not in existing_tables:
        return

    if "public_mcp_apps" in existing_tables:
        remaining_salesforce_apps = bind.execute(
            sa.select(sa.func.count())
            .select_from(PUBLIC_MCP_APPS_TABLE)
            .where(PUBLIC_MCP_APPS_TABLE.c.provider_name == "salesforce")
        ).scalar()
        if remaining_salesforce_apps:
            return

    # Only delete the provider row when it still matches the static shape this
    # migration seeded, so an admin-created "salesforce" provider (via
    # POST /admin/mcp/providers) is preserved. client_id/client_secret are
    # env-dependent and intentionally not part of the guard. name/auth_url/
    # token_url are NOT NULL core columns present since the table's creation,
    # so they can be matched unconditionally.
    seeded_provider = _salesforce_provider_row()
    bind.execute(
        sa.delete(FULL_OAUTH_PROVIDERS_TABLE)
        .where(FULL_OAUTH_PROVIDERS_TABLE.c.provider_name == "salesforce")
        .where(FULL_OAUTH_PROVIDERS_TABLE.c.name == seeded_provider["name"])
        .where(FULL_OAUTH_PROVIDERS_TABLE.c.auth_url == seeded_provider["auth_url"])
        .where(FULL_OAUTH_PROVIDERS_TABLE.c.token_url == seeded_provider["token_url"])
    )
