"""Centralized registry for MCP Applications and OAuth Providers.

This module provides a scalable structure for defining supported MCP applications,
their OAuth configurations, and server launch configurations.
"""

from typing import Any, Dict, List

# Centralized OAuth Provider Configuration
OAUTH_PROVIDERS: Dict[str, Dict[str, str]] = {
    "google": {
        "token_url": "https://oauth2.googleapis.com/token",
        "client_id_env": "GOOGLE_CLIENT_ID",
        "client_secret_env": "GOOGLE_CLIENT_SECRET",
    },
    "linkedin": {
        "token_url": "https://www.linkedin.com/oauth/v2/accessToken",
        "client_id_env": "LINKEDIN_CLIENT_ID",
        "client_secret_env": "LINKEDIN_CLIENT_SECRET",
    },
}

# Centralized MCP Apps Library
MCP_APPS_LIBRARY: List[Dict[str, Any]] = [
    {
        "id": "linkedin",
        "name": "LinkedIn",
        "description": "Access LinkedIn to retrieve basic user profiles and manage your created posts.",
        "icon": "https://www.google.com/s2/favicons?domain=linkedin.com&sz=128",
        "transport": "oauth",
        "provider": "linkedin",
        "category": "CRM",
        "oauth_scopes": ["openid", "profile", "email", "w_member_social"],
        "launch_config": {
            "command": "uv",
            "args": ["run", "python", "-m", "xagent.web.tools.mcp.linkedin"],
            "env_mapping": {"LINKEDIN_ACCESS_TOKEN": "access_token"},
        },
    },
    {
        "id": "gmail",
        "name": "Gmail",
        "description": "Connect to your Gmail inbox to read, search, draft, and send emails.",
        "icon": "https://www.google.com/s2/favicons?domain=mail.google.com&sz=128",
        "transport": "oauth",
        "provider": "google",
        "category": "Communication",
        "oauth_scopes": ["https://www.googleapis.com/auth/gmail.modify"],
        "launch_config": {
            "command": "uv",
            "args": ["run", "python", "-m", "xagent.web.tools.mcp.gmail"],
            "env_mapping": {"GOOGLE_ACCESS_TOKEN": "access_token"},
        },
    },
    {
        "id": "google-drive",
        "name": "Google Drive",
        "description": "Access Google Drive to search for files, read documents, and manage your cloud storage.",
        "icon": "https://www.google.com/s2/favicons?domain=drive.google.com&sz=128",
        "transport": "oauth",
        "provider": "google",
        "category": "Support",
        "oauth_scopes": ["https://www.googleapis.com/auth/drive"],
        "launch_config": {
            "command": "uv",
            "args": ["run", "python", "-m", "xagent.web.tools.mcp.google_drive"],
            "env_mapping": {"GOOGLE_ACCESS_TOKEN": "access_token"},
        },
    },
    {
        "id": "google-calendar",
        "name": "Google Calendar",
        "description": "Connect to Google Calendar to manage events, schedule meetings, and view your daily agenda.",
        "icon": "https://www.google.com/s2/favicons?domain=calendar.google.com&sz=128",
        "transport": "oauth",
        "provider": "google",
        "category": "Scheduling",
        "oauth_scopes": ["https://www.googleapis.com/auth/calendar"],
        "launch_config": {
            "command": "uv",
            "args": ["run", "python", "-m", "xagent.web.tools.mcp.calendar"],
            "env_mapping": {"GOOGLE_ACCESS_TOKEN": "access_token"},
        },
    },
]

# Helper functions for O(1) or encapsulated lookups
_APPS_BY_ID = {app["id"]: app for app in MCP_APPS_LIBRARY if "id" in app}
_APPS_BY_NAME = {app["name"]: app for app in MCP_APPS_LIBRARY if "name" in app}


def get_app_by_id(app_id: str) -> Dict[str, Any] | None:
    """Retrieve an MCP app configuration by its ID."""
    return _APPS_BY_ID.get(app_id)


def get_app_by_name(name: str) -> Dict[str, Any] | None:
    """Retrieve an MCP app configuration by its exact name."""
    return _APPS_BY_NAME.get(name)
