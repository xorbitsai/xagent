"""seed built-in Linear (OAuth) MCP connector

Revision ID: 20260818_seed_linear_mcp_app
Revises: 20260813_trace_json_columns_to_jsonb
Create Date: 2026-08-18 00:00:00.000000

"""

import os
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "20260818_seed_linear_mcp_app"
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

APP_ID = "linear"

LINEAR_APP_SCOPES = ["read", "write"]
# The provider row only carries the identity-only "read" scope -- the
# functional "write" scope lives on the app row and is merged in at
# authorize time by _merge_oauth_scopes (see the matching comment on the
# registry row in builtin_mcp_registry.py).
LINEAR_PROVIDER_SCOPES = ["read"]


def _filter_row(row: dict[str, object], allowed_columns: set[str]) -> dict[str, object]:
    return {key: value for key, value in row.items() if key in allowed_columns}


def _linear_provider_row() -> dict[str, object]:
    return {
        "provider_name": "linear",
        "name": "Linear",
        "client_id": os.environ.get("LINEAR_CLIENT_ID", ""),
        "client_secret": os.environ.get("LINEAR_CLIENT_SECRET", ""),
        "auth_url": "https://linear.app/oauth/authorize",
        "token_url": "https://api.linear.app/oauth/token",
        "redirect_uri": os.environ.get("LINEAR_REDIRECT_URI", ""),
        # Linear's API is GraphQL-only; no flat REST userinfo endpoint fits
        # this callback's GET-plus-flat-dict.get() lookup — see the matching
        # comment on the registry row.
        "userinfo_url": "",
        "user_id_path": "id",
        "email_path": "email",
        "default_scopes": LINEAR_PROVIDER_SCOPES,
    }


def _linear_app_row() -> dict[str, object]:
    return {
        "app_id": APP_ID,
        "name": "Linear",
        "description": "Connect to Linear to search and manage issues, comments, teams, and projects.",
        "icon": "https://www.google.com/s2/favicons?domain=linear.app&sz=128",
        "transport": "oauth",
        "provider_name": "linear",
        "category": "Productivity",
        "oauth_scopes": LINEAR_APP_SCOPES,
        "is_visible_in_connector": True,
        "launch_config": {
            "command": "python",
            "args": ["-m", "xagent.web.tools.mcp.linear"],
            "env_mapping": {"LINEAR_ACCESS_TOKEN": "access_token"},
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
        if "linear" not in existing_provider_names:
            bind.execute(
                sa.insert(FULL_OAUTH_PROVIDERS_TABLE),
                [_filter_row(_linear_provider_row(), oauth_columns)],
            )

    if "public_mcp_apps" in existing_tables:
        app_columns = {
            column["name"] for column in inspector.get_columns("public_mcp_apps")
        }
        # Same guard as seed_builtin_oauth_and_public_mcp_apps in
        # builtin_mcp_registry.py, and for the same reason:
        # is_visible_in_connector is load-bearing for the hidden-rollout
        # gate some builtin rows ship behind -- _filter_row silently
        # dropping a missing column and falling back to the table's
        # visible-by-default default would seed a row that's supposed to
        # ship hidden as visible instead. Belt-and-suspenders here too,
        # since this column has been present since the table's creation.
        if "is_visible_in_connector" not in app_columns:
            raise RuntimeError(
                "public_mcp_apps.is_visible_in_connector is missing; "
                "builtin apps that ship hidden must not seed visible"
            )
        existing_app_ids = set(
            bind.execute(sa.select(PUBLIC_MCP_APPS_TABLE.c.app_id)).scalars()
        )
        if APP_ID not in existing_app_ids:
            bind.execute(
                sa.insert(PUBLIC_MCP_APPS_TABLE),
                [_filter_row(_linear_app_row(), app_columns)],
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
        remaining_linear_apps = bind.execute(
            sa.select(sa.func.count())
            .select_from(PUBLIC_MCP_APPS_TABLE)
            .where(PUBLIC_MCP_APPS_TABLE.c.provider_name == "linear")
        ).scalar()
        if remaining_linear_apps:
            return

    # Only delete the provider row when it still matches the static shape this
    # migration seeded, so an admin-created "linear" provider (via
    # POST /admin/mcp/providers) is preserved. client_id/client_secret are
    # env-dependent and intentionally not part of the guard. name/auth_url/
    # token_url are NOT NULL core columns present since the table's creation,
    # so they can be matched unconditionally.
    seeded_provider = _linear_provider_row()
    bind.execute(
        sa.delete(FULL_OAUTH_PROVIDERS_TABLE)
        .where(FULL_OAUTH_PROVIDERS_TABLE.c.provider_name == "linear")
        .where(FULL_OAUTH_PROVIDERS_TABLE.c.name == seeded_provider["name"])
        .where(FULL_OAUTH_PROVIDERS_TABLE.c.auth_url == seeded_provider["auth_url"])
        .where(FULL_OAUTH_PROVIDERS_TABLE.c.token_url == seeded_provider["token_url"])
    )
