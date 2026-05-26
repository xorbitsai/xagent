"""seed built-in Microsoft Graph MCP registry data

Revision ID: 20260526_seed_builtin_microsoft_graph_mcp_apps
Revises: 20260525_add_task_visibility
Create Date: 2026-05-26 00:00:00.000000

"""

import os
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "20260526_seed_builtin_microsoft_graph_mcp_apps"
down_revision: Union[str, None] = "20260525_add_task_visibility"
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

OAUTH_PROVIDERS_TABLE = sa.table(
    "oauth_providers",
    sa.column("provider_name", sa.String),
)

MICROSOFT_APP_IDS = ("teams", "outlook", "onedrive")


def _filter_row(row: dict[str, object], allowed_columns: set[str]) -> dict[str, object]:
    return {key: value for key, value in row.items() if key in allowed_columns}


def _microsoft_provider_row() -> dict[str, object]:
    return {
        "provider_name": "microsoft",
        "name": "Microsoft",
        "client_id": os.environ.get("MICROSOFT_CLIENT_ID", ""),
        "client_secret": os.environ.get("MICROSOFT_CLIENT_SECRET", ""),
        "auth_url": "https://login.microsoftonline.com/common/oauth2/v2.0/authorize",
        "token_url": "https://login.microsoftonline.com/common/oauth2/v2.0/token",
        "redirect_uri": os.environ.get("MICROSOFT_REDIRECT_URI", ""),
        "userinfo_url": "https://graph.microsoft.com/v1.0/me",
        "user_id_path": "id",
        "email_path": "userPrincipalName",
        "default_scopes": ["User.Read"],
    }


def _microsoft_app_rows() -> list[dict[str, object]]:
    return [
        {
            "app_id": "teams",
            "name": "Teams",
            "description": "Connect to Microsoft Teams to list teams, read channels and chats, and send messages.",
            "icon": "https://www.google.com/s2/favicons?domain=teams.microsoft.com&sz=128",
            "transport": "oauth",
            "provider_name": "microsoft",
            "category": "Communication",
            "oauth_scopes": [
                "Team.ReadBasic.All",
                "Channel.ReadBasic.All",
                "TeamMember.Read.All",
                "ChannelMessage.Read.All",
                "ChannelMessage.Send",
                "Chat.ReadWrite",
            ],
            "is_visible_in_connector": True,
            "launch_config": {
                "command": "uv",
                "args": ["run", "python", "-m", "xagent.web.tools.mcp.teams"],
                "env_mapping": {"AUTH_TOKEN": "access_token"},
            },
        },
        {
            "app_id": "outlook",
            "name": "Outlook",
            "description": "Connect to Outlook to work with mail, calendar events, contacts, and profile data.",
            "icon": "https://www.google.com/s2/favicons?domain=outlook.office.com&sz=128",
            "transport": "oauth",
            "provider_name": "microsoft",
            "category": "Communication",
            "oauth_scopes": [
                "Mail.Read",
                "Mail.Send",
                "Calendars.ReadWrite",
                "Contacts.Read",
            ],
            "is_visible_in_connector": True,
            "launch_config": {
                "command": "uv",
                "args": ["run", "python", "-m", "xagent.web.tools.mcp.outlook"],
                "env_mapping": {"AUTH_TOKEN": "access_token"},
            },
        },
        {
            "app_id": "onedrive",
            "name": "OneDrive",
            "description": "Connect to OneDrive to browse files, download content, and manage cloud storage.",
            "icon": "https://www.google.com/s2/favicons?domain=onedrive.live.com&sz=128",
            "transport": "oauth",
            "provider_name": "microsoft",
            "category": "Storage",
            "oauth_scopes": ["Files.ReadWrite"],
            "is_visible_in_connector": True,
            "launch_config": {
                "command": "uv",
                "args": ["run", "python", "-m", "xagent.web.tools.mcp.onedrive"],
                "env_mapping": {"AUTH_TOKEN": "access_token"},
            },
        },
    ]


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
        if "microsoft" not in existing_provider_names:
            bind.execute(
                sa.insert(FULL_OAUTH_PROVIDERS_TABLE),
                [_filter_row(_microsoft_provider_row(), oauth_columns)],
            )

    if "public_mcp_apps" in existing_tables:
        app_columns = {
            column["name"] for column in inspector.get_columns("public_mcp_apps")
        }
        existing_app_ids = set(
            bind.execute(sa.select(PUBLIC_MCP_APPS_TABLE.c.app_id)).scalars()
        )
        app_rows_to_insert = [
            _filter_row(row, app_columns)
            for row in _microsoft_app_rows()
            if row["app_id"] not in existing_app_ids
        ]
        if app_rows_to_insert:
            bind.execute(sa.insert(PUBLIC_MCP_APPS_TABLE), app_rows_to_insert)


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing_tables = set(inspector.get_table_names())

    if "public_mcp_apps" in existing_tables:
        bind.execute(
            sa.delete(PUBLIC_MCP_APPS_TABLE).where(
                PUBLIC_MCP_APPS_TABLE.c.app_id.in_(MICROSOFT_APP_IDS)
            )
        )

    if "oauth_providers" not in existing_tables:
        return

    if "public_mcp_apps" in existing_tables:
        remaining_microsoft_apps = bind.execute(
            sa.select(sa.func.count())
            .select_from(PUBLIC_MCP_APPS_TABLE)
            .where(PUBLIC_MCP_APPS_TABLE.c.provider_name == "microsoft")
        ).scalar_one()
        if remaining_microsoft_apps:
            return

    bind.execute(
        sa.delete(OAUTH_PROVIDERS_TABLE).where(
            OAUTH_PROVIDERS_TABLE.c.provider_name == "microsoft"
        )
    )
