"""Seed the built-in Shopify key-based MCP connector.

Revision ID: 20260909_seed_shopify_mcp_app
Revises: 370740d9125a
"""

import logging
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

from xagent.builtin_identity import (
    builtin_provenance_identity,
    canonicalize_builtin_identity,
)

logger = logging.getLogger(__name__)

revision: str = "20260909_seed_shopify_mcp_app"
down_revision: Union[str, None] = "370740d9125a"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

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
MCP_SERVERS_TABLE = sa.table(
    "mcp_servers",
    sa.column("name", sa.String),
    sa.column("auth", sa.JSON),
)

APP_ID = "shopify"
BUILTIN_PROVENANCE = {
    "registry": "xagent",
    "app_id": APP_ID,
    "version": 1,
}
ROW = {
    "app_id": APP_ID,
    "name": "Shopify",
    "description": "Connect a Shopify custom app with a store label (for example, acme for acme.myshopify.com) and Admin API access token. Grant write_products, write_orders, and read_customers; read_all_orders is optional for orders older than 60 days.",
    "icon": "https://www.google.com/s2/favicons?domain=shopify.com&sz=128",
    "transport": "stdio",
    "provider_name": None,
    "category": "Commerce",
    "oauth_scopes": None,
    "is_visible_in_connector": True,
    "launch_config": {
        "command": "python",
        "args": ["-m", "xagent.web.tools.mcp.shopify"],
        "required_env": ["SHOPIFY_STORE_DOMAIN", "SHOPIFY_ACCESS_TOKEN"],
        "required_admin_scopes": [
            "write_products",
            "write_orders",
            "read_customers",
        ],
        "optional_admin_scopes": ["read_all_orders"],
        "credential_scope": "personal",
        "builtin_provenance": BUILTIN_PROVENANCE,
    },
}


def _has_provenance(launch_config: object) -> bool:
    return isinstance(launch_config, dict) and builtin_provenance_identity(
        launch_config.get("builtin_provenance")
    ) == builtin_provenance_identity(BUILTIN_PROVENANCE)


def _has_server_provenance(auth: object) -> bool:
    return isinstance(auth, dict) and builtin_provenance_identity(
        auth.get("builtin_provenance")
    ) == builtin_provenance_identity(BUILTIN_PROVENANCE)


def _collides_with_shopify_identity(value: object) -> bool:
    return canonicalize_builtin_identity(value) in {
        canonicalize_builtin_identity(APP_ID),
        canonicalize_builtin_identity(ROW["name"]),
    }


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    tables = set(inspector.get_table_names())
    if "public_mcp_apps" not in tables:
        return

    columns = {column["name"] for column in inspector.get_columns("public_mcp_apps")}
    if "launch_config" not in columns:
        raise RuntimeError(
            "Cannot seed builtin Shopify identity: public_mcp_apps.launch_config "
            "is required for provenance"
        )

    catalog_rows = list(
        bind.execute(
            sa.select(
                PUBLIC_MCP_APPS_TABLE.c.app_id,
                PUBLIC_MCP_APPS_TABLE.c.name,
                PUBLIC_MCP_APPS_TABLE.c.launch_config,
            )
        ).mappings()
    )
    colliding_catalog_rows = [
        row
        for row in catalog_rows
        if _collides_with_shopify_identity(row["app_id"])
        or _collides_with_shopify_identity(row["name"])
    ]
    exact_app_rows = [row for row in catalog_rows if row["app_id"] == APP_ID]
    if exact_app_rows:
        existing = exact_app_rows[0]
        if _has_provenance(existing["launch_config"]) and colliding_catalog_rows == [
            existing
        ]:
            return
        raise RuntimeError(
            "Cannot seed builtin Shopify connector: custom or ambiguous "
            "public_mcp_apps identity collides with 'shopify'"
        )
    if colliding_catalog_rows:
        raise RuntimeError(
            "Cannot seed builtin Shopify connector: custom public_mcp_apps "
            "identity collides with 'shopify'"
        )

    if "mcp_servers" in tables:
        server_columns = {
            column["name"] for column in inspector.get_columns("mcp_servers")
        }
        if "auth" not in server_columns:
            raise RuntimeError(
                "Cannot verify builtin Shopify server provenance: "
                "mcp_servers.auth is required"
            )
        colliding_servers = [
            row
            for row in bind.execute(
                sa.select(MCP_SERVERS_TABLE.c.name, MCP_SERVERS_TABLE.c.auth)
            ).mappings()
            if _collides_with_shopify_identity(row["name"])
        ]
        trusted_round_trip = (
            len(colliding_servers) == 1
            and colliding_servers[0]["name"] == APP_ID
            and _has_server_provenance(colliding_servers[0]["auth"])
        )
        if colliding_servers and not trusted_round_trip:
            raise RuntimeError(
                "Cannot seed builtin Shopify connector: custom mcp_servers "
                "identity collides with 'shopify'"
            )

    dropped_keys = sorted(set(ROW) - columns)
    if dropped_keys:
        logger.warning(
            "public_mcp_apps is missing columns %s; seeding %r without them",
            dropped_keys,
            APP_ID,
        )
    bind.execute(
        sa.insert(PUBLIC_MCP_APPS_TABLE),
        [{key: value for key, value in ROW.items() if key in columns}],
    )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "public_mcp_apps" not in set(inspector.get_table_names()):
        return
    columns = {column["name"] for column in inspector.get_columns("public_mcp_apps")}
    if "launch_config" not in columns:
        return
    existing = bind.execute(
        sa.select(PUBLIC_MCP_APPS_TABLE.c.launch_config).where(
            PUBLIC_MCP_APPS_TABLE.c.app_id == APP_ID
        )
    ).scalar_one_or_none()
    if not _has_provenance(existing):
        return
    bind.execute(
        sa.delete(PUBLIC_MCP_APPS_TABLE).where(PUBLIC_MCP_APPS_TABLE.c.app_id == APP_ID)
    )
