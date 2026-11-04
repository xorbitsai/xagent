"""normalize built-in MCP execution fields

Revision ID: 20260715_normalize_builtin_mcp_launch
Revises: 20260713_add_agent_visibility
Create Date: 2026-07-15

"""

import json
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "20260715_normalize_builtin_mcp_launch"
down_revision: Union[str, None] = "20260713_add_agent_visibility"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


PUBLIC_MCP_APPS_TABLE = sa.table(
    "public_mcp_apps",
    sa.column("app_id", sa.String),
    sa.column("name", sa.String),
    sa.column("transport", sa.String),
    sa.column("provider_name", sa.String),
    sa.column("oauth_scopes", sa.JSON),
    sa.column("launch_config", sa.JSON),
)

EXECUTION_FIELD_NAMES = (
    "name",
    "transport",
    "provider_name",
    "oauth_scopes",
    "launch_config",
)
JSON_EXECUTION_FIELD_NAMES = frozenset({"oauth_scopes", "launch_config"})

# This is an immutable migration snapshot. Runtime registry changes must not alter
# migrations that have already been applied to deployed databases.
CANONICAL_EXECUTION_FIELDS: dict[str, dict[str, object]] = {
    "linkedin": {
        "name": "LinkedIn",
        "transport": "oauth",
        "provider_name": "linkedin",
        "oauth_scopes": ["openid", "profile", "email", "w_member_social"],
        "launch_config": {
            "command": "python",
            "args": ["-m", "xagent.web.tools.mcp.linkedin"],
            "env_mapping": {"LINKEDIN_ACCESS_TOKEN": "access_token"},
        },
    },
    "gmail": {
        "name": "Gmail",
        "transport": "oauth",
        "provider_name": "google",
        "oauth_scopes": ["https://www.googleapis.com/auth/gmail.modify"],
        "launch_config": {
            "command": "python",
            "args": ["-m", "xagent.web.tools.mcp.gmail"],
            "env_mapping": {"GOOGLE_ACCESS_TOKEN": "access_token"},
        },
    },
    "google-drive": {
        "name": "Google Drive",
        "transport": "oauth",
        "provider_name": "google",
        "oauth_scopes": ["https://www.googleapis.com/auth/drive"],
        "launch_config": {
            "command": "python",
            "args": ["-m", "xagent.web.tools.mcp.google_drive"],
            "env_mapping": {"GOOGLE_ACCESS_TOKEN": "access_token"},
        },
    },
    "google-calendar": {
        "name": "Google Calendar",
        "transport": "oauth",
        "provider_name": "google",
        "oauth_scopes": ["https://www.googleapis.com/auth/calendar"],
        "launch_config": {
            "command": "python",
            "args": ["-m", "xagent.web.tools.mcp.calendar"],
            "env_mapping": {"GOOGLE_ACCESS_TOKEN": "access_token"},
        },
    },
    "teams": {
        "name": "Teams",
        "transport": "oauth",
        "provider_name": "microsoft",
        "oauth_scopes": [
            "Team.ReadBasic.All",
            "Channel.ReadBasic.All",
            "TeamMember.Read.All",
            "ChannelMessage.Read.All",
            "ChannelMessage.Send",
            "Chat.ReadWrite",
        ],
        "launch_config": {
            "command": "python",
            "args": ["-m", "xagent.web.tools.mcp.teams"],
            "env_mapping": {"AUTH_TOKEN": "access_token"},
        },
    },
    "outlook": {
        "name": "Outlook",
        "transport": "oauth",
        "provider_name": "microsoft",
        "oauth_scopes": [
            "Mail.Read",
            "Mail.Send",
            "Calendars.ReadWrite",
            "Contacts.Read",
        ],
        "launch_config": {
            "command": "python",
            "args": ["-m", "xagent.web.tools.mcp.outlook"],
            "env_mapping": {"AUTH_TOKEN": "access_token"},
        },
    },
    "onedrive": {
        "name": "OneDrive",
        "transport": "oauth",
        "provider_name": "microsoft",
        "oauth_scopes": ["Files.ReadWrite"],
        "launch_config": {
            "command": "python",
            "args": ["-m", "xagent.web.tools.mcp.onedrive"],
            "env_mapping": {"AUTH_TOKEN": "access_token"},
        },
    },
    "facebook": {
        "name": "Facebook Pages",
        "transport": "oauth",
        "provider_name": "meta",
        "oauth_scopes": [
            "pages_show_list",
            "pages_read_engagement",
            "pages_manage_posts",
        ],
        "launch_config": {
            "command": "python",
            "args": ["-m", "xagent.web.tools.mcp.facebook"],
            "env_mapping": {"META_ACCESS_TOKEN": "access_token"},
        },
    },
    "instagram": {
        "name": "Instagram",
        "transport": "oauth",
        "provider_name": "meta",
        "oauth_scopes": [
            "pages_show_list",
            "pages_read_engagement",
            "instagram_basic",
            "instagram_content_publish",
        ],
        "launch_config": {
            "command": "python",
            "args": ["-m", "xagent.web.tools.mcp.instagram"],
            "env_mapping": {"META_ACCESS_TOKEN": "access_token"},
        },
    },
    "google-maps": {
        "name": "Google Maps",
        "transport": "stdio",
        "provider_name": None,
        "oauth_scopes": None,
        "launch_config": {
            "command": "npx",
            "args": ["-y", "@cablate/mcp-google-map", "--stdio"],
            "required_env": ["GOOGLE_MAPS_API_KEY"],
        },
    },
}


def _update_statement(
    app_id: str, execution_fields: dict[str, object]
) -> sa.sql.dml.Update:
    return (
        sa.update(PUBLIC_MCP_APPS_TABLE)
        .where(PUBLIC_MCP_APPS_TABLE.c.app_id == app_id)
        .values(**execution_fields)
    )


def _offline_value(field_name: str, value: object, dialect_name: str) -> object:
    if field_name not in JSON_EXECUTION_FIELD_NAMES:
        return sa.null() if value is None else op.inline_literal(value)

    # Match the online sa.JSON binding contract (none_as_null=False): Python None
    # is stored as JSON ``null``, not SQL NULL, on every supported dialect.
    serialized_value = json.dumps(value, sort_keys=True)
    serialized_literal = op.inline_literal(serialized_value)
    if dialect_name == "postgresql":
        return sa.cast(serialized_literal, sa.JSON())
    return serialized_literal


def _upgrade_offline() -> None:
    dialect_name = op.get_context().dialect.name
    for app_id, execution_fields in CANONICAL_EXECUTION_FIELDS.items():
        statement = (
            sa.update(PUBLIC_MCP_APPS_TABLE)
            .where(PUBLIC_MCP_APPS_TABLE.c.app_id == op.inline_literal(app_id))
            .values(
                **{
                    field_name: _offline_value(
                        field_name,
                        execution_fields[field_name],
                        dialect_name,
                    )
                    for field_name in EXECUTION_FIELD_NAMES
                }
            )
        )
        op.execute(statement)


def upgrade() -> None:
    if op.get_context().as_sql:
        _upgrade_offline()
        return

    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "public_mcp_apps" not in inspector.get_table_names():
        return

    columns = {column["name"] for column in inspector.get_columns("public_mcp_apps")}
    required_columns = {"app_id", *EXECUTION_FIELD_NAMES}
    if not required_columns.issubset(columns):
        return

    for app_id, execution_fields in CANONICAL_EXECUTION_FIELDS.items():
        bind.execute(_update_statement(app_id, execution_fields))


def downgrade() -> None:
    # Persisted built-in execution drift is a data defect, so a downgrade keeps
    # normalized values instead of restoring environment-specific drift.
    pass
