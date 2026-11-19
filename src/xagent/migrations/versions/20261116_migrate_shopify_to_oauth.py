"""switch the Shopify connector from pasted Admin API tokens to per-store OAuth

Revision ID: 20261116_shopify_oauth
Revises: 20261117_chat_message_metadata
Create Date: 2026-11-16

Hard switch (#2810), as Zendesk did (20261012_zendesk_oauth): existing
token connections are deleted and merchants reconnect through OAuth. Shopify
no longer lets stores create the legacy custom apps those tokens came from.
OAuth needs SHOPIFY_CLIENT_ID and SHOPIFY_CLIENT_SECRET (or the admin
provider settings). The catalog row is hidden after the switch; an operator
shows it once the app is configured and verified.

Unlike Zendesk's legacy row, Shopify's legacy row already carries the builtin
provenance marker, so the marker cannot tell the two shapes apart. Each state
is recognized by its full execution shape, and any other shape stops the
migration rather than being skipped. Deleted token connections cannot be
restored by the downgrade.
"""

import logging
from typing import Any, Sequence, Union

import sqlalchemy as sa
from alembic import op

from xagent.builtin_identity import builtin_provenance_identity

logger = logging.getLogger(__name__)

revision: str = "20261116_shopify_oauth"
down_revision: Union[str, None] = "20261117_chat_message_metadata"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

APP_ID = "shopify"
APP_NAME = "Shopify"
BUILTIN_PROVENANCE = {
    "registry": "xagent",
    "app_id": APP_ID,
    "version": 1,
}
MODULE_ARGS = ["-m", "xagent.web.tools.mcp.shopify"]
# Frozen copy of 20260909_seed_shopify_mcp_app's row, the only shape the
# token connector ever shipped with.
LEGACY_DESCRIPTION = (
    "Connect a Shopify custom app with a store label (for example, acme for "
    "acme.myshopify.com) and Admin API access token. Grant write_products, "
    "write_orders, and read_customers; read_all_orders is optional for orders "
    "older than 60 days."
)
LEGACY_LAUNCH_CONFIG = {
    "command": "python",
    "args": MODULE_ARGS,
    "required_env": ["SHOPIFY_STORE_DOMAIN", "SHOPIFY_ACCESS_TOKEN"],
    "required_admin_scopes": [
        "write_products",
        "write_orders",
        "read_customers",
    ],
    "optional_admin_scopes": ["read_all_orders"],
    "credential_scope": "personal",
    "builtin_provenance": BUILTIN_PROVENANCE,
}
OAUTH_DESCRIPTION = (
    "Connect to Shopify to search and manage products, look up and update "
    "orders, and look up customers and collections. Orders older than 60 days "
    "need the read_all_orders scope, which is not requested."
)
OAUTH_SCOPES = ["write_products", "write_orders", "read_customers"]
OAUTH_LAUNCH_CONFIG = {
    "command": "python",
    "args": MODULE_ARGS,
    "env_mapping": {
        "SHOPIFY_ACCESS_TOKEN": "access_token",
        "SHOPIFY_STORE_DOMAIN": "instance_url",
    },
    "builtin_provenance": BUILTIN_PROVENANCE,
}
# The endpoints are code-owned (oauth_account_hosts.py) and the identity comes
# from shopify_oauth.fetch_shopify_shop_identity, so these stay empty.
OAUTH_PROVIDER_STATIC_FIELDS = {
    "name": APP_NAME,
    "auth_url": "",
    "token_url": "",
    "userinfo_url": "",
    "user_id_path": "",
    "email_path": "",
    "default_scopes": [],
}
# Children of mcp_servers this repository defines, children first. Other
# tables pointing at mcp_servers.id (a downstream deployment's own) are found
# through their foreign keys.
KNOWN_SERVER_CHILDREN = (
    ("mcp_oauth_flow_states", "mcp_server_id"),
    ("mcp_oauth_grants", "mcp_server_id"),
    ("mcp_oauth_clients", "mcp_server_id"),
    ("user_mcpservers", "mcpserver_id"),
)

PUBLIC_MCP_APPS = sa.table(
    "public_mcp_apps",
    sa.column("app_id", sa.String),
    sa.column("name", sa.String),
    sa.column("description", sa.Text),
    sa.column("transport", sa.String),
    sa.column("provider_name", sa.String),
    sa.column("oauth_scopes", sa.JSON),
    sa.column("launch_config", sa.JSON),
    sa.column("is_visible_in_connector", sa.Boolean),
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
CATALOG_COLUMNS = (
    "app_id",
    "name",
    "description",
    "transport",
    "provider_name",
    "oauth_scopes",
    "launch_config",
    "is_visible_in_connector",
)


def _columns(bind: sa.engine.Connection, table_name: str) -> set[str]:
    inspector = sa.inspect(bind)
    if table_name not in set(inspector.get_table_names()):
        return set()
    return {column["name"] for column in inspector.get_columns(table_name)}


def _is_official(marker: object) -> bool:
    return builtin_provenance_identity(marker) == builtin_provenance_identity(
        BUILTIN_PROVENANCE
    )


def _legacy_catalog_row(row: Any) -> bool:
    # is_visible_in_connector and description are admin-editable, so they
    # are not part of the shape.
    return bool(
        row["name"] == APP_NAME
        and row["transport"] == "stdio"
        and row["provider_name"] is None
        and row["oauth_scopes"] is None
        and row["launch_config"] == LEGACY_LAUNCH_CONFIG
    )


def _oauth_catalog_row(row: Any) -> bool:
    return bool(
        row["name"] == APP_NAME
        and row["transport"] == "oauth"
        and row["provider_name"] == APP_ID
        and row["oauth_scopes"] == OAUTH_SCOPES
        and row["launch_config"] == OAUTH_LAUNCH_CONFIG
    )


def _catalog_row(bind: sa.engine.Connection) -> Any | None:
    if not set(CATALOG_COLUMNS).issubset(_columns(bind, "public_mcp_apps")):
        return None
    return (
        bind.execute(
            sa.select(*(PUBLIC_MCP_APPS.c[name] for name in CATALOG_COLUMNS)).where(
                PUBLIC_MCP_APPS.c.app_id == APP_ID
            )
        )
        .mappings()
        .first()
    )


def _server_children(bind: sa.engine.Connection) -> list[tuple[str, str]]:
    """Every (table, column) whose rows point at an mcp_servers row."""
    inspector = sa.inspect(bind)
    tables = set(inspector.get_table_names())
    children = [
        (table, column)
        for table, column in KNOWN_SERVER_CHILDREN
        if column in _columns(bind, table)
    ]
    known = {table for table, _column in KNOWN_SERVER_CHILDREN}
    for table in sorted(tables - known - {"mcp_servers"}):
        for foreign_key in inspector.get_foreign_keys(table):
            if foreign_key.get("referred_table") != "mcp_servers":
                continue
            if foreign_key.get("referred_columns") != ["id"]:
                continue
            for column in foreign_key.get("constrained_columns") or ():
                children.append((table, column))
    return children


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
    if not server_ids:
        return
    # Explicit, because a migration connection may not enforce the
    # foreign keys' ON DELETE CASCADE (SQLite without the pragma).
    for table_name, column in _server_children(bind):
        child = sa.table(table_name, sa.column(column, sa.Integer))
        bind.execute(sa.delete(child).where(child.c[column].in_(server_ids)))
    bind.execute(sa.delete(MCP_SERVERS).where(MCP_SERVERS.c.id.in_(server_ids)))


def _delete_token_connections(bind: sa.engine.Connection) -> None:
    """Delete the token servers the catalog stdio connect created.

    That connect names the server after the app id and stamps the builtin
    marker on its auth (api/mcp.py _catalog_server_provenance_auth). A
    server that merely runs the same module under the same name, without the
    marker, is not provably ours and is left alone.
    """

    def is_official_token_server(row: Any) -> bool:
        auth = row["auth"]
        return bool(
            row["command"] == LEGACY_LAUNCH_CONFIG["command"]
            and row["args"] == MODULE_ARGS
            and isinstance(auth, dict)
            and _is_official(auth.get("builtin_provenance"))
        )

    _delete_servers_and_dependents(
        bind, name=APP_ID, transport="stdio", matches=is_official_token_server
    )
    # Toby personal connections keep their own encrypted copy of the token.
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
    # Blank credentials make the runtime read SHOPIFY_CLIENT_ID,
    # SHOPIFY_CLIENT_SECRET, and SHOPIFY_REDIRECT_URI at request time.
    bind.execute(
        sa.insert(OAUTH_PROVIDERS).values(
            provider_name=APP_ID,
            client_id="",
            client_secret="",
            redirect_uri="",
            **OAUTH_PROVIDER_STATIC_FIELDS,
        )
    )


def upgrade() -> None:
    bind = op.get_bind()
    row = _catalog_row(bind)
    if row is None:
        return
    if _oauth_catalog_row(row):
        # Already switched (a rerun, or a database seeded from the current
        # registry): keep the row, visibility included, and finish the rest.
        _insert_oauth_provider(bind)
        _delete_token_connections(bind)
        return
    if not _legacy_catalog_row(row):
        raise RuntimeError(
            "Cannot migrate Shopify connector: the existing catalog row is "
            "neither the Xagent token connector nor its OAuth replacement"
        )

    values: dict[str, object] = {
        "transport": "oauth",
        "provider_name": APP_ID,
        "oauth_scopes": OAUTH_SCOPES,
        "launch_config": OAUTH_LAUNCH_CONFIG,
        "is_visible_in_connector": False,
    }
    if row["description"] == LEGACY_DESCRIPTION:
        values["description"] = OAUTH_DESCRIPTION
    bind.execute(
        sa.update(PUBLIC_MCP_APPS)
        .where(PUBLIC_MCP_APPS.c.app_id == APP_ID)
        .values(**values)
    )
    _insert_oauth_provider(bind)
    _delete_token_connections(bind)


def _delete_oauth_connections(bind: sa.engine.Connection) -> None:
    def is_shopify_oauth_server(row: Any) -> bool:
        auth = row["auth"]
        return isinstance(auth, dict) and auth.get("app_id") == APP_ID

    _delete_servers_and_dependents(
        bind, name=APP_NAME, transport="oauth", matches=is_shopify_oauth_server
    )
    required = {"provider", "access_token", "refresh_token", "expires_at"}
    if required.issubset(_columns(bind, "user_oauth")):
        # Keep the identity as a reconnect tombstone, as a dead refresh does;
        # the OAuth tokens are useless once the connector takes tokens again.
        bind.execute(
            sa.update(USER_OAUTH)
            .where(USER_OAUTH.c.provider == APP_ID)
            .values(access_token="", refresh_token=None, expires_at=None)
        )


def downgrade() -> None:
    bind = op.get_bind()
    row = _catalog_row(bind)
    if row is None:
        return
    if not _oauth_catalog_row(row):
        logger.warning(
            "Leaving the Shopify catalog row unchanged: it is not the shape "
            "20261116_shopify_oauth created"
        )
        return

    # The provider row stays: nothing proves this migration inserted it, and
    # a blank one is inert. Visibility returns to the token connector's
    # shipped default rather than a guess at an earlier operator setting.
    values: dict[str, object] = {
        "transport": "stdio",
        "provider_name": None,
        "oauth_scopes": None,
        "launch_config": LEGACY_LAUNCH_CONFIG,
        "is_visible_in_connector": True,
    }
    if row["description"] == OAUTH_DESCRIPTION:
        values["description"] = LEGACY_DESCRIPTION
    bind.execute(
        sa.update(PUBLIC_MCP_APPS)
        .where(PUBLIC_MCP_APPS.c.app_id == APP_ID)
        .values(**values)
    )
    _delete_oauth_connections(bind)
