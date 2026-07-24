from __future__ import annotations

import hashlib
import json
import os
from copy import deepcopy
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
        {
            "provider_name": "hubspot",
            "name": "HubSpot",
            "client_id": os.environ.get("HUBSPOT_CLIENT_ID", ""),
            "client_secret": os.environ.get("HUBSPOT_CLIENT_SECRET", ""),
            "auth_url": "https://app.hubspot.com/oauth/authorize",
            "token_url": "https://api.hubapi.com/oauth/v1/token",
            "redirect_uri": os.environ.get("HUBSPOT_REDIRECT_URI", ""),
            # HubSpot's token-info endpoint takes the token in the URL path
            # rather than a Bearer header; the callback substitutes the
            # {{access_token}} placeholder before issuing the GET.
            "userinfo_url": "https://api.hubapi.com/oauth/v1/access-tokens/{{access_token}}",
            "user_id_path": "user_id",
            "email_path": "user",
            "default_scopes": ["oauth"],
        },
        {
            "provider_name": "meta",
            "name": "Meta",
            "client_id": os.environ.get("META_CLIENT_ID", ""),
            "client_secret": os.environ.get("META_CLIENT_SECRET", ""),
            "auth_url": "https://www.facebook.com/v25.0/dialog/oauth",
            "token_url": "https://graph.facebook.com/v25.0/oauth/access_token",
            "redirect_uri": os.environ.get("META_REDIRECT_URI", ""),
            "userinfo_url": "https://graph.facebook.com/v25.0/me?fields=id,email",
            "user_id_path": "id",
            "email_path": "email",
            "default_scopes": ["public_profile"],
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
                "command": "python",
                "args": ["-m", "xagent.web.tools.mcp.linkedin"],
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
                "command": "python",
                "args": ["-m", "xagent.web.tools.mcp.gmail"],
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
                "command": "python",
                "args": ["-m", "xagent.web.tools.mcp.google_drive"],
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
                "command": "python",
                "args": ["-m", "xagent.web.tools.mcp.calendar"],
                "env_mapping": {"GOOGLE_ACCESS_TOKEN": "access_token"},
            },
        },
        {
            "app_id": "google-docs",
            "name": "Google Docs",
            "description": "Connect to Google Docs to create documents from Markdown or text, read document content, and edit text.",
            "icon": "https://www.google.com/s2/favicons?domain=docs.google.com&sz=128",
            "transport": "oauth",
            "provider_name": "google",
            "category": "Productivity",
            "oauth_scopes": [
                "https://www.googleapis.com/auth/documents",
                "https://www.googleapis.com/auth/drive.file",
            ],
            "is_visible_in_connector": True,
            "launch_config": {
                "command": "python",
                "args": ["-m", "xagent.web.tools.mcp.google_docs"],
                "env_mapping": {"GOOGLE_ACCESS_TOKEN": "access_token"},
            },
        },
        {
            "app_id": "google-slides",
            "name": "Google Slides",
            "description": "Connect to Google Slides to create presentations, add slides, and read slide content.",
            "icon": "https://www.google.com/s2/favicons?domain=slides.google.com&sz=128",
            "transport": "oauth",
            "provider_name": "google",
            "category": "Productivity",
            "oauth_scopes": [
                "https://www.googleapis.com/auth/presentations",
                "https://www.googleapis.com/auth/drive.file",
            ],
            "is_visible_in_connector": True,
            "launch_config": {
                "command": "python",
                "args": ["-m", "xagent.web.tools.mcp.google_slides"],
                "env_mapping": {"GOOGLE_ACCESS_TOKEN": "access_token"},
            },
        },
        {
            "app_id": "google-ads",
            "name": "Google Ads",
            "description": "Connect to Google Ads to list accessible accounts, inspect campaigns, and run GAQL reports.",
            "icon": "https://www.google.com/s2/favicons?domain=ads.google.com&sz=128",
            "transport": "oauth",
            "provider_name": "google",
            "category": "Marketing",
            "oauth_scopes": ["https://www.googleapis.com/auth/adwords"],
            "is_visible_in_connector": True,
            "launch_config": {
                "command": "python",
                "args": ["-m", "xagent.web.tools.mcp.google_ads"],
                "env_mapping": {"GOOGLE_ACCESS_TOKEN": "access_token"},
                "static_env": {
                    "GOOGLE_ADS_DEVELOPER_TOKEN": "GOOGLE_ADS_DEVELOPER_TOKEN"
                },
            },
        },
        {
            "app_id": "hubspot",
            "name": "HubSpot",
            "description": "Connect to HubSpot CRM to search, create, and update contacts and companies, read deals, and log notes.",
            "icon": "https://www.google.com/s2/favicons?domain=hubspot.com&sz=128",
            "transport": "oauth",
            "provider_name": "hubspot",
            "category": "CRM",
            "oauth_scopes": [
                "crm.objects.contacts.read",
                "crm.objects.contacts.write",
                "crm.objects.companies.read",
                "crm.objects.companies.write",
                "crm.objects.deals.read",
            ],
            "is_visible_in_connector": True,
            "launch_config": {
                "command": "python",
                "args": ["-m", "xagent.web.tools.mcp.hubspot"],
                "env_mapping": {"HUBSPOT_ACCESS_TOKEN": "access_token"},
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
                "command": "python",
                "args": ["-m", "xagent.web.tools.mcp.teams"],
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
                "command": "python",
                "args": ["-m", "xagent.web.tools.mcp.outlook"],
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
                "command": "python",
                "args": ["-m", "xagent.web.tools.mcp.onedrive"],
                "env_mapping": {"AUTH_TOKEN": "access_token"},
            },
        },
        {
            "app_id": "facebook",
            "name": "Facebook Pages",
            "description": "Connect to Facebook Pages to discover managed pages and publish page posts.",
            "icon": "https://www.google.com/s2/favicons?domain=facebook.com&sz=128",
            "transport": "oauth",
            "provider_name": "meta",
            "category": "Marketing",
            "oauth_scopes": [
                "pages_show_list",
                "pages_read_engagement",
                "pages_manage_posts",
            ],
            "is_visible_in_connector": True,
            "launch_config": {
                "command": "python",
                "args": ["-m", "xagent.web.tools.mcp.facebook"],
                "env_mapping": {"META_ACCESS_TOKEN": "access_token"},
            },
        },
        {
            "app_id": "instagram",
            "name": "Instagram",
            "description": "Connect to Instagram professional accounts through Meta Graph API.",
            "icon": "https://www.google.com/s2/favicons?domain=instagram.com&sz=128",
            "transport": "oauth",
            "provider_name": "meta",
            "category": "Marketing",
            "oauth_scopes": [
                "pages_show_list",
                "pages_read_engagement",
                "instagram_basic",
                "instagram_content_publish",
            ],
            "is_visible_in_connector": True,
            "launch_config": {
                "command": "python",
                "args": ["-m", "xagent.web.tools.mcp.instagram"],
                "env_mapping": {"META_ACCESS_TOKEN": "access_token"},
            },
        },
        {
            "app_id": "google-maps",
            "name": "Google Maps",
            "description": "Geocoding, directions, place search, and more via the Google Maps APIs.",
            "icon": "https://www.google.com/s2/favicons?domain=maps.google.com&sz=128",
            "transport": "stdio",
            "provider_name": None,
            "category": "Productivity",
            "oauth_scopes": None,
            "is_visible_in_connector": True,
            # Key-based (non-oauth): connected via POST /api/mcp/apps/{id}/connect.
            # required_env tells the connector which secret(s) to prompt for.
            "launch_config": {
                "command": "npx",
                "args": ["-y", "@cablate/mcp-google-map", "--stdio"],
                "required_env": ["GOOGLE_MAPS_API_KEY"],
            },
        },
    ]


_BUILTIN_EXECUTION_FIELD_NAMES = (
    "name",
    "transport",
    "provider_name",
    "oauth_scopes",
    "launch_config",
)


def get_builtin_public_mcp_app(app_id: str) -> dict[str, Any] | None:
    for row in get_builtin_public_mcp_app_rows():
        if row["app_id"] == app_id:
            return deepcopy(row)
    return None


def is_builtin_public_mcp_app(app_id: str) -> bool:
    return any(row["app_id"] == app_id for row in get_builtin_public_mcp_app_rows())


def get_builtin_execution_fields(app_id: str) -> dict[str, Any] | None:
    row = get_builtin_public_mcp_app(app_id)
    if row is None:
        return None
    return deepcopy(
        {field_name: row[field_name] for field_name in _BUILTIN_EXECUTION_FIELD_NAMES}
    )


def _safe_configuration_hash(values: dict[str, Any]) -> str:
    serialized = json.dumps(
        values,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(serialized).hexdigest()}"


def validate_builtin_public_mcp_apps(bind: Connection) -> list[dict[str, Any]]:
    """Return safe summaries for persisted built-in catalog drift.

    Missing rows are intentionally outside this drift-only check; supported
    admin writes cannot delete built-ins, and this validator never recreates
    data. Only rows selected by exact built-in ``app_id`` are compared, so
    custom applications remain database-owned.
    """

    canonical_rows = get_builtin_public_mcp_app_rows()
    canonical_by_app_id = {row["app_id"]: row for row in canonical_rows}
    selected_columns = [
        PUBLIC_MCP_APPS_TABLE.c.app_id,
        *(PUBLIC_MCP_APPS_TABLE.c[field] for field in _BUILTIN_EXECUTION_FIELD_NAMES),
    ]
    persisted_rows = bind.execute(
        sa.select(*selected_columns).where(
            PUBLIC_MCP_APPS_TABLE.c.app_id.in_(tuple(canonical_by_app_id))
        )
    ).mappings()
    persisted_by_app_id = {row["app_id"]: row for row in persisted_rows}

    mismatches: list[dict[str, Any]] = []
    for app_id, canonical_row in canonical_by_app_id.items():
        persisted_row = persisted_by_app_id.get(app_id)
        if persisted_row is None:
            continue

        mismatched_fields = [
            field_name
            for field_name in _BUILTIN_EXECUTION_FIELD_NAMES
            if persisted_row[field_name] != canonical_row[field_name]
        ]
        if not mismatched_fields:
            continue

        canonical_values = {
            field_name: canonical_row[field_name] for field_name in mismatched_fields
        }
        persisted_values = {
            field_name: persisted_row[field_name] for field_name in mismatched_fields
        }
        mismatches.append(
            {
                "app_id": app_id,
                "mismatched_fields": mismatched_fields,
                "canonical_hash": _safe_configuration_hash(canonical_values),
                "persisted_hash": _safe_configuration_hash(persisted_values),
            }
        )

    return mismatches


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
