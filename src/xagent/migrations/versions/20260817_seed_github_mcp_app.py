"""seed built-in GitHub (OAuth) MCP connector

Revision ID: 20260817_seed_github_mcp_app
Revises: 20260813_trace_json_columns_to_jsonb
Create Date: 2026-08-17 00:00:00.000000

"""

import os
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "20260817_seed_github_mcp_app"
down_revision: Union[str, None] = "20260813_trace_json_columns_to_jsonb"
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

OAUTH_PROVIDERS_TABLE = sa.table(
    "oauth_providers",
    sa.column("provider_name", sa.String),
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

APP_ID = "github"

GITHUB_SCOPES = ["repo", "read:org", "user:email"]

# Identity-only, matching the other seeded providers (e.g. zoom) -- the
# functional scopes above live on the app row (_github_app_row) and are
# merged in at authorize time, so the provider row doesn't need to
# duplicate the full list.
GITHUB_PROVIDER_DEFAULT_SCOPES = ["read:user"]


def _filter_row(row: dict[str, object], allowed_columns: set[str]) -> dict[str, object]:
    return {key: value for key, value in row.items() if key in allowed_columns}


def _github_provider_row() -> dict[str, object]:
    return {
        "provider_name": "github",
        "name": "GitHub",
        "client_id": os.environ.get("GITHUB_CLIENT_ID", ""),
        "client_secret": os.environ.get("GITHUB_CLIENT_SECRET", ""),
        "auth_url": "https://github.com/login/oauth/authorize",
        "token_url": "https://github.com/login/oauth/access_token",
        "redirect_uri": os.environ.get("GITHUB_REDIRECT_URI", ""),
        "userinfo_url": "https://api.github.com/user",
        "user_id_path": "id",
        "email_path": "login",
        "default_scopes": GITHUB_PROVIDER_DEFAULT_SCOPES,
    }


def _github_app_row() -> dict[str, object]:
    return {
        "app_id": APP_ID,
        "name": "GitHub",
        "description": "Connect to GitHub to search repositories and code, read and create issues and pull requests, comment, and browse file contents and commit history.",
        "icon": "https://www.google.com/s2/favicons?domain=github.com&sz=128",
        "transport": "oauth",
        "provider_name": "github",
        "category": "Development",
        "oauth_scopes": GITHUB_SCOPES,
        "is_visible_in_connector": True,
        "launch_config": {
            "command": "python",
            "args": ["-m", "xagent.web.tools.mcp.github"],
            "env_mapping": {"GITHUB_ACCESS_TOKEN": "access_token"},
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
            bind.execute(sa.select(OAUTH_PROVIDERS_TABLE.c.provider_name)).scalars()
        )
        if "github" not in existing_provider_names:
            bind.execute(
                sa.insert(FULL_OAUTH_PROVIDERS_TABLE),
                [_filter_row(_github_provider_row(), oauth_columns)],
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
                [_filter_row(_github_app_row(), app_columns)],
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
        remaining_github_apps = bind.execute(
            sa.select(sa.func.count())
            .select_from(PUBLIC_MCP_APPS_TABLE)
            .where(PUBLIC_MCP_APPS_TABLE.c.provider_name == "github")
        ).scalar()
        if remaining_github_apps:
            return

    # Only delete the provider row when it still matches the static shape this
    # migration seeded, so an admin-created "github" provider (via
    # POST /admin/mcp/providers) is preserved. client_id/client_secret are
    # env-dependent and intentionally not part of the guard. name/auth_url/
    # token_url are NOT NULL core columns present since the table's creation,
    # so they can be matched unconditionally.
    seeded_provider = _github_provider_row()
    bind.execute(
        sa.delete(FULL_OAUTH_PROVIDERS_TABLE)
        .where(FULL_OAUTH_PROVIDERS_TABLE.c.provider_name == "github")
        .where(FULL_OAUTH_PROVIDERS_TABLE.c.name == seeded_provider["name"])
        .where(FULL_OAUTH_PROVIDERS_TABLE.c.auth_url == seeded_provider["auth_url"])
        .where(FULL_OAUTH_PROVIDERS_TABLE.c.token_url == seeded_provider["token_url"])
    )
