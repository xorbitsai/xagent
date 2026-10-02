"""migrate HubSpot from local REST tools to its hosted MCP server

Revision ID: 20261002_hubspot_remote_mcp
Revises: 20260930_merge_task_outcome_identity
Create Date: 2026-10-02

Downgrading restores the legacy HubSpot OAuth provider from the historical
HUBSPOT_CLIENT_ID, HUBSPOT_CLIENT_SECRET, and HUBSPOT_REDIRECT_URI environment
variables. If those legacy variables are no longer available, an administrator
must re-enter the provider credentials after the downgrade.
"""

import os
from typing import Any, Sequence, Union

import sqlalchemy as sa
from alembic import op

from xagent.builtin_identity import builtin_provenance_identity

revision: str = "20261002_hubspot_remote_mcp"
down_revision: Union[str, None] = "20260930_merge_task_outcome_identity"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

APP_ID = "hubspot"
LEGACY_DESCRIPTION = (
    "Connect to HubSpot CRM and Marketing Hub to list, search, create, and "
    "update contacts, companies, and deals, log notes, read forms and "
    "submissions, pull traffic analytics reports, and read marketing emails "
    "and campaigns."
)
REMOTE_DESCRIPTION = (
    "Connect to HubSpot's hosted MCP server to work with CRM, activity, "
    "content, and marketing data using the authenticated user's HubSpot "
    "permissions."
)
LEGACY_SCOPES = [
    "crm.objects.contacts.read",
    "crm.objects.contacts.write",
    "crm.objects.companies.read",
    "crm.objects.companies.write",
    "crm.objects.deals.read",
    "crm.objects.deals.write",
    "forms",
]
LEGACY_LAUNCH_CONFIG = {
    "command": "python",
    "args": ["-m", "xagent.web.tools.mcp.hubspot"],
    "env_mapping": {"HUBSPOT_ACCESS_TOKEN": "access_token"},
}
BUILTIN_PROVENANCE = {
    "registry": "xagent",
    "app_id": APP_ID,
    "version": 2,
}
REMOTE_LAUNCH_CONFIG = {
    "url": "https://mcp.hubspot.com",
    "auth": {
        "type": "mcp_oauth",
        "credential_provider": "hubspot",
        "token_endpoint_auth_method": "client_secret_post",
    },
    "builtin_provenance": BUILTIN_PROVENANCE,
}
LEGACY_OAUTH_PROVIDER_STATIC_FIELDS = {
    "name": "HubSpot",
    "auth_url": "https://app.hubspot.com/oauth/authorize",
    "token_url": "https://api.hubapi.com/oauth/v1/token",
    "userinfo_url": "https://api.hubapi.com/oauth/v1/access-tokens/{{access_token}}",
    "user_id_path": "user_id",
    "email_path": "user",
    "default_scopes": ["oauth"],
}

PUBLIC_MCP_APPS = sa.table(
    "public_mcp_apps",
    sa.column("app_id", sa.String),
    sa.column("name", sa.String),
    sa.column("description", sa.Text),
    sa.column("transport", sa.String),
    sa.column("provider_name", sa.String),
    sa.column("oauth_scopes", sa.JSON),
    sa.column("launch_config", sa.JSON),
)
MCP_SERVERS = sa.table(
    "mcp_servers",
    sa.column("id", sa.Integer),
    sa.column("name", sa.String),
    sa.column("transport", sa.String),
    sa.column("url", sa.String),
    sa.column("auth", sa.JSON),
)
OAUTH_PROVIDERS = sa.table(
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
USER_OAUTH = sa.table(
    "user_oauth",
    sa.column("provider", sa.String),
    sa.column("access_token", sa.String),
    sa.column("refresh_token", sa.String),
)


def _columns(bind: sa.engine.Connection, table_name: str) -> set[str]:
    inspector = sa.inspect(bind)
    if table_name not in set(inspector.get_table_names()):
        return set()
    return {column["name"] for column in inspector.get_columns(table_name)}


def _owned_remote_launch(value: object) -> bool:
    return isinstance(value, dict) and builtin_provenance_identity(
        value.get("builtin_provenance")
    ) == builtin_provenance_identity(BUILTIN_PROVENANCE)


def _same_legacy_scope_set(value: object) -> bool:
    """Match legacy scopes without depending on JSON array ordering."""
    return (
        isinstance(value, (list, tuple))
        and all(isinstance(scope, str) for scope in value)
        and set(value) == set(LEGACY_SCOPES)
    )


def _legacy_catalog_row(row: Any) -> bool:
    return (
        row["name"] == "HubSpot"
        and row["transport"] == "oauth"
        and row["provider_name"] == "hubspot"
        and _same_legacy_scope_set(row["oauth_scopes"])
        and row["launch_config"] == LEGACY_LAUNCH_CONFIG
    )


def _delete_server_and_dependents(
    bind: sa.engine.Connection,
    *,
    name: str,
    transport: str,
    matches: Any,
) -> None:
    required = {"id", "name", "transport", "url", "auth"}
    if not required.issubset(_columns(bind, "mcp_servers")):
        return
    rows = bind.execute(
        sa.select(
            MCP_SERVERS.c.id,
            MCP_SERVERS.c.name,
            MCP_SERVERS.c.transport,
            MCP_SERVERS.c.url,
            MCP_SERVERS.c.auth,
        ).where(
            MCP_SERVERS.c.name == name,
            MCP_SERVERS.c.transport == transport,
        )
    ).mappings()
    server_ids = [int(row["id"]) for row in rows if matches(row)]
    for server_id in server_ids:
        # Be explicit for SQLite deployments where migration connections may
        # not have foreign-key cascade enforcement enabled.
        for table_name, server_column in (
            ("mcp_oauth_flow_states", "mcp_server_id"),
            ("mcp_oauth_grants", "mcp_server_id"),
            ("mcp_oauth_clients", "mcp_server_id"),
            ("user_mcpservers", "mcpserver_id"),
        ):
            if server_column not in _columns(bind, table_name):
                continue
            child = sa.table(table_name, sa.column(server_column, sa.Integer))
            bind.execute(sa.delete(child).where(child.c[server_column] == server_id))
        bind.execute(sa.delete(MCP_SERVERS).where(MCP_SERVERS.c.id == server_id))


def _delete_legacy_local_server(bind: sa.engine.Connection) -> None:
    def matches(row: Any) -> bool:
        auth = row["auth"]
        return (
            isinstance(auth, dict)
            and auth.get("app_id") == APP_ID
            and auth.get("provider") == "hubspot"
        )

    _delete_server_and_dependents(
        bind, name="HubSpot", transport="oauth", matches=matches
    )


def _delete_remote_server(bind: sa.engine.Connection) -> None:
    def matches(row: Any) -> bool:
        auth = row["auth"]
        return (
            row["url"] == "https://mcp.hubspot.com"
            and isinstance(auth, dict)
            and auth.get("type") == "mcp_oauth"
        )

    _delete_server_and_dependents(
        bind, name=APP_ID, transport="streamable_http", matches=matches
    )


def _clear_legacy_hubspot_tokens(bind: sa.engine.Connection) -> None:
    required = {"provider", "access_token"}
    if not required.issubset(_columns(bind, "user_oauth")):
        return
    values: dict[str, object] = {"access_token": ""}
    if "refresh_token" in _columns(bind, "user_oauth"):
        values["refresh_token"] = None
    bind.execute(
        sa.update(USER_OAUTH).where(USER_OAUTH.c.provider == APP_ID).values(**values)
    )


def _remove_legacy_hubspot_provider(bind: sa.engine.Connection) -> None:
    required = {"provider_name", *LEGACY_OAUTH_PROVIDER_STATIC_FIELDS}
    if not required.issubset(_columns(bind, "oauth_providers")):
        return
    row = (
        bind.execute(
            sa.select(*(OAUTH_PROVIDERS.c[name] for name in required)).where(
                OAUTH_PROVIDERS.c.provider_name == APP_ID
            )
        )
        .mappings()
        .first()
    )
    if row is None or any(
        row[name] != expected
        for name, expected in LEGACY_OAUTH_PROVIDER_STATIC_FIELDS.items()
    ):
        return
    bind.execute(
        sa.delete(OAUTH_PROVIDERS).where(OAUTH_PROVIDERS.c.provider_name == APP_ID)
    )


def _restore_legacy_hubspot_provider(bind: sa.engine.Connection) -> None:
    required = {
        "provider_name",
        "name",
        "client_id",
        "client_secret",
        "auth_url",
        "token_url",
        "redirect_uri",
        "userinfo_url",
        "user_id_path",
        "email_path",
        "default_scopes",
    }
    if not required.issubset(_columns(bind, "oauth_providers")):
        return
    exists = bind.execute(
        sa.select(OAUTH_PROVIDERS.c.provider_name).where(
            OAUTH_PROVIDERS.c.provider_name == APP_ID
        )
    ).first()
    if exists is not None:
        return
    bind.execute(
        sa.insert(OAUTH_PROVIDERS).values(
            provider_name=APP_ID,
            name=LEGACY_OAUTH_PROVIDER_STATIC_FIELDS["name"],
            client_id=os.environ.get("HUBSPOT_CLIENT_ID", ""),
            client_secret=os.environ.get("HUBSPOT_CLIENT_SECRET", ""),
            auth_url=LEGACY_OAUTH_PROVIDER_STATIC_FIELDS["auth_url"],
            token_url=LEGACY_OAUTH_PROVIDER_STATIC_FIELDS["token_url"],
            redirect_uri=os.environ.get("HUBSPOT_REDIRECT_URI", ""),
            userinfo_url=LEGACY_OAUTH_PROVIDER_STATIC_FIELDS["userinfo_url"],
            user_id_path=LEGACY_OAUTH_PROVIDER_STATIC_FIELDS["user_id_path"],
            email_path=LEGACY_OAUTH_PROVIDER_STATIC_FIELDS["email_path"],
            default_scopes=LEGACY_OAUTH_PROVIDER_STATIC_FIELDS["default_scopes"],
        )
    )


def upgrade() -> None:
    bind = op.get_bind()
    required = {
        "app_id",
        "name",
        "description",
        "transport",
        "provider_name",
        "oauth_scopes",
        "launch_config",
    }
    if not required.issubset(_columns(bind, "public_mcp_apps")):
        return
    row = (
        bind.execute(
            sa.select(*(PUBLIC_MCP_APPS.c[name] for name in required)).where(
                PUBLIC_MCP_APPS.c.app_id == APP_ID
            )
        )
        .mappings()
        .first()
    )
    if row is None:
        return
    if _owned_remote_launch(row["launch_config"]):
        # A prior registry sync may have updated the catalog before this
        # migration ran. Keep the migration idempotent by still removing
        # legacy state left behind by that partial transition.
        _delete_legacy_local_server(bind)
        _clear_legacy_hubspot_tokens(bind)
        _remove_legacy_hubspot_provider(bind)
        return
    if not _legacy_catalog_row(row):
        raise RuntimeError(
            "Cannot migrate HubSpot connector: the existing catalog row does "
            "not match the Xagent legacy HubSpot connector"
        )

    values: dict[str, object] = {
        "transport": "streamable_http",
        "provider_name": None,
        "oauth_scopes": None,
        "launch_config": REMOTE_LAUNCH_CONFIG,
    }
    if row["description"] == LEGACY_DESCRIPTION:
        values["description"] = REMOTE_DESCRIPTION
    bind.execute(
        sa.update(PUBLIC_MCP_APPS)
        .where(PUBLIC_MCP_APPS.c.app_id == APP_ID)
        .values(**values)
    )
    _delete_legacy_local_server(bind)
    _clear_legacy_hubspot_tokens(bind)
    _remove_legacy_hubspot_provider(bind)


def downgrade() -> None:
    bind = op.get_bind()
    required = {
        "app_id",
        "description",
        "transport",
        "provider_name",
        "oauth_scopes",
        "launch_config",
    }
    if not required.issubset(_columns(bind, "public_mcp_apps")):
        return
    row = (
        bind.execute(
            sa.select(*(PUBLIC_MCP_APPS.c[name] for name in required)).where(
                PUBLIC_MCP_APPS.c.app_id == APP_ID
            )
        )
        .mappings()
        .first()
    )
    if row is None or not _owned_remote_launch(row["launch_config"]):
        return

    values: dict[str, object] = {
        "transport": "oauth",
        "provider_name": "hubspot",
        "oauth_scopes": LEGACY_SCOPES,
        "launch_config": LEGACY_LAUNCH_CONFIG,
    }
    if row["description"] == REMOTE_DESCRIPTION:
        values["description"] = LEGACY_DESCRIPTION
    bind.execute(
        sa.update(PUBLIC_MCP_APPS)
        .where(PUBLIC_MCP_APPS.c.app_id == APP_ID)
        .values(**values)
    )
    _delete_remote_server(bind)
    _restore_legacy_hubspot_provider(bind)
