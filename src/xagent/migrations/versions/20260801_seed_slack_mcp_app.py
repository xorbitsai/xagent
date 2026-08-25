"""seed built-in Slack (OAuth) MCP connector

Revision ID: 20260801_seed_slack_mcp_app
Revises: 20260730_add_slack_oauth_flow_states
Create Date: 2026-08-01 00:00:00.000000

"""

import os
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "20260801_seed_slack_mcp_app"
down_revision: Union[str, None] = "20260730_add_slack_oauth_flow_states"
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

APP_ID = "slack"

# Kept in sync with the current builtin_mcp_registry.py row (not just the
# scopes present when this migration was first written) so
# test_seed_rows_match_registry keeps catching drift — see the precedent set
# by 20260720_seed_docs_slides_hubspot.py's own HubSpot row when its scopes
# were later expanded in 20260810_add_hubspot_marketing_scopes.py. Existing
# databases are unaffected (this migration only inserts when the row is
# absent); the follow-up 20260812_add_slack_history_reactions_files_scopes and
# 20260825_add_slack_channels_join_scope migrations are what actually upgrade
# an already-seeded row.
SLACK_SCOPES = [
    "chat:write",
    "chat:write.public",
    "channels:read",
    "channels:history",
    "channels:join",
    "groups:read",
    "groups:history",
    "im:read",
    "im:history",
    "mpim:read",
    "mpim:history",
    "reactions:write",
    "files:write",
]


def _filter_row(row: dict[str, object], allowed_columns: set[str]) -> dict[str, object]:
    return {key: value for key, value in row.items() if key in allowed_columns}


def _slack_provider_row() -> dict[str, object]:
    return {
        "provider_name": "slack",
        "name": "Slack",
        "client_id": os.environ.get("SLACK_CLIENT_ID", ""),
        "client_secret": os.environ.get("SLACK_CLIENT_SECRET", ""),
        "auth_url": "https://slack.com/oauth/v2/authorize",
        "token_url": "https://slack.com/api/oauth.v2.access",
        "redirect_uri": os.environ.get("SLACK_REDIRECT_URI", ""),
        "userinfo_url": "https://slack.com/api/auth.test",
        # auth.test never returns an email for a bot token; the workspace
        # name ("team") is deliberately stored in the email slot because
        # UserOAuth.email is only consumed as the "connected account" display
        # label for non-gmail providers — without it the Slack connection
        # would show up unlabeled in the connector UI.
        "user_id_path": "team_id",
        "email_path": "team",
        "default_scopes": SLACK_SCOPES,
    }


def _slack_app_row() -> dict[str, object]:
    return {
        "app_id": APP_ID,
        "name": "Slack",
        "description": "Connect to Slack to search and read channel, thread, and DM history, post messages and replies, react to messages, upload files, and (with your approval) join public channels, e.g. incident summaries and recommended fixes.",
        "icon": "https://www.google.com/s2/favicons?domain=slack.com&sz=128",
        "transport": "oauth",
        "provider_name": "slack",
        "category": "Communication",
        "oauth_scopes": SLACK_SCOPES,
        "is_visible_in_connector": True,
        "launch_config": {
            "command": "python",
            "args": ["-m", "xagent.web.tools.mcp.slack"],
            "env_mapping": {"SLACK_ACCESS_TOKEN": "access_token"},
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
        if "slack" not in existing_provider_names:
            bind.execute(
                sa.insert(FULL_OAUTH_PROVIDERS_TABLE),
                [_filter_row(_slack_provider_row(), oauth_columns)],
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
                [_filter_row(_slack_app_row(), app_columns)],
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
        remaining_slack_apps = bind.execute(
            sa.select(sa.func.count())
            .select_from(PUBLIC_MCP_APPS_TABLE)
            .where(PUBLIC_MCP_APPS_TABLE.c.provider_name == "slack")
        ).scalar()
        if remaining_slack_apps:
            return

    # Delete unconditionally by provider_name, matching the sibling seed
    # migrations (meta, google-maps). A shape-matching guard would protect
    # the wrong thing: an admin recreating the *real* Slack provider enters
    # Slack's canonical URLs, which would match the guard and be deleted
    # anyway — only a differently-shaped row would survive.
    bind.execute(
        sa.delete(FULL_OAUTH_PROVIDERS_TABLE).where(
            FULL_OAUTH_PROVIDERS_TABLE.c.provider_name == "slack"
        )
    )
