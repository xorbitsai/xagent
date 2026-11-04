"""switch the Zendesk connector from API tokens to per-account OAuth

Revision ID: 20261012_zendesk_oauth
Revises: 20261012_model_name_2048
Create Date: 2026-10-12

Hard switch (#2930): existing API-token connections are deleted and users
reconnect through OAuth. Zendesk OAuth needs ZENDESK_CLIENT_ID and
ZENDESK_CLIENT_SECRET (or the admin provider settings); for accounts outside
the client's own, that client must be an approved Zendesk global OAuth
client.
"""

from typing import Any, Sequence, Union

import sqlalchemy as sa
from alembic import op

from xagent.builtin_identity import builtin_provenance_identity

revision: str = "20261012_zendesk_oauth"
down_revision: Union[str, None] = "20261012_model_name_2048"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

APP_ID = "zendesk"
APP_NAME = "Zendesk"
LEGACY_DESCRIPTION = (
    "Connect to Zendesk with an API token to search and manage tickets, reply "
    "to customers or add internal notes, and look up users and organizations."
)
OAUTH_DESCRIPTION = (
    "Connect to Zendesk to search and manage tickets, reply to customers or "
    "add internal notes, and look up users and organizations."
)
LEGACY_LAUNCH_CONFIG = {
    "command": "python",
    "args": ["-m", "xagent.web.tools.mcp.zendesk"],
    "required_env": ["ZENDESK_SUBDOMAIN", "ZENDESK_EMAIL", "ZENDESK_API_TOKEN"],
}
BUILTIN_PROVENANCE = {
    "registry": "xagent",
    "app_id": APP_ID,
    "version": 1,
}
OAUTH_SCOPES = ["read", "tickets:write"]
OAUTH_LAUNCH_CONFIG = {
    "command": "python",
    "args": ["-m", "xagent.web.tools.mcp.zendesk"],
    "env_mapping": {
        "ZENDESK_ACCESS_TOKEN": "access_token",
        "ZENDESK_SUBDOMAIN": "instance_url",
    },
    "builtin_provenance": BUILTIN_PROVENANCE,
}
# The endpoints are code-owned (oauth_account_hosts.py), so the static URL
# columns stay empty. These fields identify a row this migration inserted.
OAUTH_PROVIDER_STATIC_FIELDS = {
    "name": APP_NAME,
    "auth_url": "",
    "token_url": "",
    "userinfo_url": "",
    "user_id_path": "id",
    "email_path": "email",
    "default_scopes": ["read"],
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
    sa.column("command", sa.String),
    sa.column("args", sa.JSON),
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
ACTOR_CONNECTIONS = sa.table(
    "actor_mcp_server_connections",
    sa.column("app_id", sa.String),
)
USER_OAUTH = sa.table(
    "user_oauth",
    sa.column("provider", sa.String),
    sa.column("access_token", sa.String),
    sa.column("refresh_token", sa.String),
    sa.column("expires_at", sa.DateTime),
)


def _columns(bind: sa.engine.Connection, table_name: str) -> set[str]:
    inspector = sa.inspect(bind)
    if table_name not in set(inspector.get_table_names()):
        return set()
    return {column["name"] for column in inspector.get_columns(table_name)}


def _owned_oauth_launch(value: object) -> bool:
    return isinstance(value, dict) and builtin_provenance_identity(
        value.get("builtin_provenance")
    ) == builtin_provenance_identity(BUILTIN_PROVENANCE)


def _legacy_catalog_row(row: Any) -> bool:
    # is_visible_in_connector is deliberately not part of the shape: it is
    # admin-editable, and an operator may already have shown the row.
    return (
        row["name"] == APP_NAME
        and row["transport"] == "stdio"
        and row["provider_name"] is None
        and row["oauth_scopes"] is None
        and row["launch_config"] == LEGACY_LAUNCH_CONFIG
    )


def _delete_servers_and_dependents(
    bind: sa.engine.Connection, *, name: str, transport: str, matches: Any
) -> None:
    required = {"id", "name", "transport", "command", "args", "auth"}
    if not required.issubset(_columns(bind, "mcp_servers")):
        return
    rows = bind.execute(
        sa.select(*(MCP_SERVERS.c[column] for column in sorted(required))).where(
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


def _delete_legacy_api_token_connections(bind: sa.engine.Connection) -> None:
    """Delete the API-token servers the catalog stdio connect created.

    That connect names its server after the app id and stores each user's
    email and API token on the user link, so deleting the server and its
    links removes the stored credentials. A custom server only matches when
    it also runs the builtin module, which the reserved catalog name already
    prevents a user from creating.
    """

    def runs_builtin_module(row: Any) -> bool:
        return bool(
            row["command"] == LEGACY_LAUNCH_CONFIG["command"]
            and row["args"] == LEGACY_LAUNCH_CONFIG["args"]
        )

    _delete_servers_and_dependents(
        bind, name=APP_ID, transport="stdio", matches=runs_builtin_module
    )
    # Actor (bot) connections keep their own encrypted copy of the API token.
    if "app_id" in _columns(bind, "actor_mcp_server_connections"):
        bind.execute(
            sa.delete(ACTOR_CONNECTIONS).where(ACTOR_CONNECTIONS.c.app_id == APP_ID)
        )


def _insert_oauth_provider(bind: sa.engine.Connection) -> None:
    required = {"provider_name", "client_id", "client_secret", "redirect_uri"}
    required |= set(OAUTH_PROVIDER_STATIC_FIELDS)
    if not required.issubset(_columns(bind, "oauth_providers")):
        return
    exists = bind.execute(
        sa.select(OAUTH_PROVIDERS.c.provider_name).where(
            OAUTH_PROVIDERS.c.provider_name == APP_ID
        )
    ).first()
    if exists is not None:
        # An operator-created row keeps its own credentials and settings.
        return
    # Credentials stay blank so the runtime reads ZENDESK_CLIENT_ID,
    # ZENDESK_CLIENT_SECRET, and ZENDESK_REDIRECT_URI at request time; a
    # copied value would pin whatever client was configured during this
    # migration, so a later switch to the global client would not take.
    bind.execute(
        sa.insert(OAUTH_PROVIDERS).values(
            provider_name=APP_ID,
            client_id="",
            client_secret="",
            redirect_uri="",
            **OAUTH_PROVIDER_STATIC_FIELDS,
        )
    )


def _remove_inserted_oauth_provider(bind: sa.engine.Connection) -> None:
    configured_fields = ("client_id", "client_secret", "redirect_uri")
    required = {"provider_name", *OAUTH_PROVIDER_STATIC_FIELDS, *configured_fields}
    if not required.issubset(_columns(bind, "oauth_providers")):
        return
    row = (
        bind.execute(
            sa.select(*(OAUTH_PROVIDERS.c[name] for name in sorted(required))).where(
                OAUTH_PROVIDERS.c.provider_name == APP_ID
            )
        )
        .mappings()
        .first()
    )
    if row is None or any(
        row[name] != expected for name, expected in OAUTH_PROVIDER_STATIC_FIELDS.items()
    ):
        return
    # An admin who entered credentials in the provider settings owns the row
    # now; deleting it would destroy their secret.
    if any(row[name] for name in configured_fields):
        return
    bind.execute(
        sa.delete(OAUTH_PROVIDERS).where(OAUTH_PROVIDERS.c.provider_name == APP_ID)
    )


def _delete_oauth_connections(bind: sa.engine.Connection) -> None:
    def is_zendesk_oauth_server(row: Any) -> bool:
        auth = row["auth"]
        return isinstance(auth, dict) and auth.get("app_id") == APP_ID

    _delete_servers_and_dependents(
        bind, name=APP_NAME, transport="oauth", matches=is_zendesk_oauth_server
    )
    required = {"provider", "access_token", "refresh_token", "expires_at"}
    if required.issubset(_columns(bind, "user_oauth")):
        # Keep the identity as a reconnect tombstone, as a dead refresh does;
        # the OAuth tokens are useless once the connector is API-token again.
        bind.execute(
            sa.update(USER_OAUTH)
            .where(USER_OAUTH.c.provider == APP_ID)
            .values(access_token="", refresh_token=None, expires_at=None)
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
            sa.select(*(PUBLIC_MCP_APPS.c[name] for name in sorted(required))).where(
                PUBLIC_MCP_APPS.c.app_id == APP_ID
            )
        )
        .mappings()
        .first()
    )
    if row is None:
        return
    if _owned_oauth_launch(row["launch_config"]):
        # Already switched (an earlier run, or a database seeded from the
        # current registry). Still remove legacy state a partial transition
        # may have left behind.
        _insert_oauth_provider(bind)
        _delete_legacy_api_token_connections(bind)
        return
    if not _legacy_catalog_row(row):
        raise RuntimeError(
            "Cannot migrate Zendesk connector: the existing catalog row does "
            "not match the Xagent legacy API-token Zendesk connector"
        )

    values: dict[str, object] = {
        "transport": "oauth",
        "provider_name": APP_ID,
        "oauth_scopes": OAUTH_SCOPES,
        "launch_config": OAUTH_LAUNCH_CONFIG,
    }
    if row["description"] == LEGACY_DESCRIPTION:
        values["description"] = OAUTH_DESCRIPTION
    bind.execute(
        sa.update(PUBLIC_MCP_APPS)
        .where(PUBLIC_MCP_APPS.c.app_id == APP_ID)
        .values(**values)
    )
    _insert_oauth_provider(bind)
    _delete_legacy_api_token_connections(bind)


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
            sa.select(*(PUBLIC_MCP_APPS.c[name] for name in sorted(required))).where(
                PUBLIC_MCP_APPS.c.app_id == APP_ID
            )
        )
        .mappings()
        .first()
    )
    if row is None or not _owned_oauth_launch(row["launch_config"]):
        return

    values: dict[str, object] = {
        "transport": "stdio",
        "provider_name": None,
        "oauth_scopes": None,
        "launch_config": LEGACY_LAUNCH_CONFIG,
    }
    if row["description"] == OAUTH_DESCRIPTION:
        values["description"] = LEGACY_DESCRIPTION
    bind.execute(
        sa.update(PUBLIC_MCP_APPS)
        .where(PUBLIC_MCP_APPS.c.app_id == APP_ID)
        .values(**values)
    )
    _delete_oauth_connections(bind)
    _remove_inserted_oauth_provider(bind)
