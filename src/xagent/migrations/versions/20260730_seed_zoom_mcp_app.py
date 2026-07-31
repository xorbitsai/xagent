"""seed built-in Zoom (OAuth) MCP connector

Revision ID: 20260730_seed_zoom_mcp_app
Revises: 20260730_seed_google_analytics_mcp_app
Create Date: 2026-07-30 00:00:00.000000

"""

import os
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "20260730_seed_zoom_mcp_app"
down_revision: Union[str, None] = "20260730_seed_google_analytics_mcp_app"
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

APP_ID = "zoom"

ZOOM_SCOPES = [
    "meeting:read:meeting",
    "meeting:read:list_meetings",
    "meeting:read:past_meeting",
    "cloud_recording:read:list_recording_files",
    "cloud_recording:read:meeting_transcript",
    "user:read:user",
]

# Identity-only, matching the other seeded providers — the functional scopes
# above live on the app row (_zoom_app_row) and are merged in at authorize
# time, so the provider row doesn't need to duplicate the full list.
ZOOM_PROVIDER_DEFAULT_SCOPES = ["user:read:user"]


def _filter_row(row: dict[str, object], allowed_columns: set[str]) -> dict[str, object]:
    return {key: value for key, value in row.items() if key in allowed_columns}


def _zoom_provider_row() -> dict[str, object]:
    return {
        "provider_name": "zoom",
        "name": "Zoom",
        "client_id": os.environ.get("ZOOM_CLIENT_ID", ""),
        "client_secret": os.environ.get("ZOOM_CLIENT_SECRET", ""),
        "auth_url": "https://zoom.us/oauth/authorize",
        "token_url": "https://zoom.us/oauth/token",
        "redirect_uri": os.environ.get("ZOOM_REDIRECT_URI", ""),
        "userinfo_url": "https://api.zoom.us/v2/users/me",
        "user_id_path": "id",
        "email_path": "email",
        "default_scopes": ZOOM_PROVIDER_DEFAULT_SCOPES,
    }


def _zoom_app_row() -> dict[str, object]:
    return {
        "app_id": APP_ID,
        "name": "Zoom",
        "description": "Connect to Zoom to look up meetings, and read cloud recordings and transcripts.",
        "icon": "https://www.google.com/s2/favicons?domain=zoom.us&sz=128",
        "transport": "oauth",
        "provider_name": "zoom",
        "category": "Scheduling",
        "oauth_scopes": ZOOM_SCOPES,
        "is_visible_in_connector": True,
        "launch_config": {
            "command": "python",
            "args": ["-m", "xagent.web.tools.mcp.zoom"],
            "env_mapping": {"ZOOM_ACCESS_TOKEN": "access_token"},
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
        if "zoom" not in existing_provider_names:
            bind.execute(
                sa.insert(FULL_OAUTH_PROVIDERS_TABLE),
                [_filter_row(_zoom_provider_row(), oauth_columns)],
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
                [_filter_row(_zoom_app_row(), app_columns)],
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
        remaining_zoom_apps = bind.execute(
            sa.select(sa.func.count())
            .select_from(PUBLIC_MCP_APPS_TABLE)
            .where(PUBLIC_MCP_APPS_TABLE.c.provider_name == "zoom")
        ).scalar()
        if remaining_zoom_apps:
            return

    # Only delete the provider row when it still matches the static shape this
    # migration seeded, so an admin-created "zoom" provider (via
    # POST /admin/mcp/providers) is preserved. client_id/client_secret are
    # env-dependent and intentionally not part of the guard. name/auth_url/
    # token_url are NOT NULL core columns present since the table's creation,
    # so they can be matched unconditionally.
    seeded_provider = _zoom_provider_row()
    bind.execute(
        sa.delete(FULL_OAUTH_PROVIDERS_TABLE)
        .where(FULL_OAUTH_PROVIDERS_TABLE.c.provider_name == "zoom")
        .where(FULL_OAUTH_PROVIDERS_TABLE.c.name == seeded_provider["name"])
        .where(FULL_OAUTH_PROVIDERS_TABLE.c.auth_url == seeded_provider["auth_url"])
        .where(FULL_OAUTH_PROVIDERS_TABLE.c.token_url == seeded_provider["token_url"])
    )
