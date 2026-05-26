from __future__ import annotations

import os
from typing import Any

import sqlalchemy as sa
from sqlalchemy.engine import Connection

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


def get_builtin_oauth_provider_rows() -> list[dict[str, Any]]:
    return [
        {
            "provider_name": "google",
            "name": "Google",
            "client_id": os.environ.get("GOOGLE_CLIENT_ID", ""),
            "client_secret": os.environ.get("GOOGLE_CLIENT_SECRET", ""),
            "auth_url": "https://accounts.google.com/o/oauth2/auth",
            "token_url": "https://oauth2.googleapis.com/token",
            "redirect_uri": os.environ.get("GOOGLE_REDIRECT_URI", ""),
            "userinfo_url": "https://www.googleapis.com/oauth2/v2/userinfo",
            "user_id_path": "id",
            "email_path": "email",
            "default_scopes": [
                "https://www.googleapis.com/auth/userinfo.email",
                "https://www.googleapis.com/auth/userinfo.profile",
            ],
        },
        {
            "provider_name": "linkedin",
            "name": "LinkedIn",
            "client_id": os.environ.get("LINKEDIN_CLIENT_ID", ""),
            "client_secret": os.environ.get("LINKEDIN_CLIENT_SECRET", ""),
            "auth_url": "https://www.linkedin.com/oauth/v2/authorization",
            "token_url": "https://www.linkedin.com/oauth/v2/accessToken",
            "redirect_uri": os.environ.get("LINKEDIN_REDIRECT_URI", ""),
            "userinfo_url": "https://api.linkedin.com/v2/userinfo",
            "user_id_path": "sub",
            "email_path": "email",
            "default_scopes": ["openid", "profile", "email", "w_member_social"],
        },
        {
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
        },
    ]


def get_builtin_public_mcp_app_rows() -> list[dict[str, Any]]:
    return [
        {
            "app_id": "linkedin",
            "name": "LinkedIn",
            "description": "Access LinkedIn to retrieve basic user profiles and manage your created posts.",
            "icon": "https://www.google.com/s2/favicons?domain=linkedin.com&sz=128",
            "transport": "oauth",
            "provider_name": "linkedin",
            "category": "CRM",
            "oauth_scopes": ["openid", "profile", "email", "w_member_social"],
            "is_visible_in_connector": True,
            "launch_config": {
                "command": "uv",
                "args": ["run", "python", "-m", "xagent.web.tools.mcp.linkedin"],
                "env_mapping": {"LINKEDIN_ACCESS_TOKEN": "access_token"},
            },
        },
        {
            "app_id": "gmail",
            "name": "Gmail",
            "description": "Connect to your Gmail inbox to read, search, draft, and send emails.",
            "icon": "https://www.google.com/s2/favicons?domain=mail.google.com&sz=128",
            "transport": "oauth",
            "provider_name": "google",
            "category": "Communication",
            "oauth_scopes": ["https://www.googleapis.com/auth/gmail.modify"],
            "is_visible_in_connector": True,
            "launch_config": {
                "command": "uv",
                "args": ["run", "python", "-m", "xagent.web.tools.mcp.gmail"],
                "env_mapping": {"GOOGLE_ACCESS_TOKEN": "access_token"},
            },
        },
        {
            "app_id": "google-drive",
            "name": "Google Drive",
            "description": "Access Google Drive to search for files, read documents, and manage your cloud storage.",
            "icon": "https://www.google.com/s2/favicons?domain=drive.google.com&sz=128",
            "transport": "oauth",
            "provider_name": "google",
            "category": "Support",
            "oauth_scopes": ["https://www.googleapis.com/auth/drive"],
            "is_visible_in_connector": True,
            "launch_config": {
                "command": "uv",
                "args": [
                    "run",
                    "python",
                    "-m",
                    "xagent.web.tools.mcp.google_drive",
                ],
                "env_mapping": {"GOOGLE_ACCESS_TOKEN": "access_token"},
            },
        },
        {
            "app_id": "google-calendar",
            "name": "Google Calendar",
            "description": "Connect to Google Calendar to manage events, schedule meetings, and view your daily agenda.",
            "icon": "https://www.google.com/s2/favicons?domain=calendar.google.com&sz=128",
            "transport": "oauth",
            "provider_name": "google",
            "category": "Scheduling",
            "oauth_scopes": ["https://www.googleapis.com/auth/calendar"],
            "is_visible_in_connector": True,
            "launch_config": {
                "command": "uv",
                "args": ["run", "python", "-m", "xagent.web.tools.mcp.calendar"],
                "env_mapping": {"GOOGLE_ACCESS_TOKEN": "access_token"},
            },
        },
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


def _filter_row(row: dict[str, Any], allowed_columns: set[str]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key in allowed_columns}


def seed_builtin_oauth_and_public_mcp_apps(bind: Connection) -> None:
    inspector = sa.inspect(bind)
    existing_tables = set(inspector.get_table_names())

    if "oauth_providers" in existing_tables:
        oauth_columns = {
            column["name"] for column in inspector.get_columns("oauth_providers")
        }
        existing_provider_names = set(
            bind.execute(sa.select(OAUTH_PROVIDERS_TABLE.c.provider_name)).scalars()
        )
        provider_rows_to_insert = [
            _filter_row(row, oauth_columns)
            for row in get_builtin_oauth_provider_rows()
            if row["provider_name"] not in existing_provider_names
        ]
        if provider_rows_to_insert:
            bind.execute(sa.insert(OAUTH_PROVIDERS_TABLE), provider_rows_to_insert)

    if "public_mcp_apps" in existing_tables:
        app_columns = {
            column["name"] for column in inspector.get_columns("public_mcp_apps")
        }
        existing_app_ids = set(
            bind.execute(sa.select(PUBLIC_MCP_APPS_TABLE.c.app_id)).scalars()
        )
        app_rows_to_insert = [
            _filter_row(row, app_columns)
            for row in get_builtin_public_mcp_app_rows()
            if row["app_id"] not in existing_app_ids
        ]
        if app_rows_to_insert:
            bind.execute(sa.insert(PUBLIC_MCP_APPS_TABLE), app_rows_to_insert)
