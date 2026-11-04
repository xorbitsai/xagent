"""seed built-in Excel (OAuth) MCP connector

Revision ID: 20260917_seed_excel_mcp_app
Revises: 6f40e43d9e28
Create Date: 2026-09-17 00:00:00.000000

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

# revision identifiers, used by Alembic.
revision: str = "20260917_seed_excel_mcp_app"
down_revision: Union[str, None] = "6f40e43d9e28"
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

APP_ID = "excel"
BUILTIN_PROVENANCE = {
    "registry": "xagent",
    "app_id": APP_ID,
    "version": 1,
}

ROW = {
    "app_id": APP_ID,
    "name": "Excel",
    "description": "Connect to Excel to read and write worksheets, cell ranges, and tables in workbooks stored on OneDrive or SharePoint.",
    "icon": "https://www.google.com/s2/favicons?domain=office.com&sz=128",
    "transport": "oauth",
    "provider_name": "microsoft",
    "category": "Productivity",
    "oauth_scopes": ["Files.ReadWrite", "offline_access"],
    "is_visible_in_connector": True,
    "launch_config": {
        "command": "python",
        "args": ["-m", "xagent.web.tools.mcp.excel"],
        "env_mapping": {"AUTH_TOKEN": "access_token"},
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


def _collides_with_excel_identity(value: object) -> bool:
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

    columns = {c["name"] for c in inspector.get_columns("public_mcp_apps")}
    if "launch_config" not in columns:
        raise RuntimeError(
            "Cannot seed builtin Excel identity: public_mcp_apps.launch_config "
            "is required for provenance"
        )
    existing_rows = (
        bind.execute(
            sa.select(
                PUBLIC_MCP_APPS_TABLE.c.app_id,
                PUBLIC_MCP_APPS_TABLE.c.name,
                PUBLIC_MCP_APPS_TABLE.c.launch_config,
            )
        )
        .mappings()
        .all()
    )
    collisions = [
        row
        for row in existing_rows
        if _collides_with_excel_identity(row["app_id"])
        or _collides_with_excel_identity(row["name"])
    ]
    catalog_already_owned = False
    if collisions:
        if (
            len(collisions) == 1
            and collisions[0]["app_id"] == APP_ID
            and _has_provenance(collisions[0]["launch_config"])
        ):
            catalog_already_owned = True
        else:
            identities = ", ".join(
                f"app_id={row['app_id']!r}, name={row['name']!r}" for row in collisions
            )
            raise RuntimeError(
                "Cannot seed builtin Excel connector: existing public_mcp_apps "
                f"identities collide with the reserved Excel identity ({identities})"
            )

    if "mcp_servers" in tables:
        server_columns = {
            column["name"] for column in inspector.get_columns("mcp_servers")
        }
        if "auth" not in server_columns:
            raise RuntimeError(
                "Cannot verify builtin Excel server provenance: "
                "mcp_servers.auth is required"
            )
        colliding_servers = [
            row
            for row in bind.execute(
                sa.select(MCP_SERVERS_TABLE.c.name, MCP_SERVERS_TABLE.c.auth)
            ).mappings()
            if _collides_with_excel_identity(row["name"])
        ]
        trusted_round_trip = (
            len(colliding_servers) == 1
            and colliding_servers[0]["name"] == ROW["name"]
            and _has_server_provenance(colliding_servers[0]["auth"])
        )
        if colliding_servers and not trusted_round_trip:
            identities = ", ".join(f"name={row['name']!r}" for row in colliding_servers)
            raise RuntimeError(
                "Cannot seed builtin Excel connector: custom mcp_servers "
                f"identities collide with the reserved Excel identity ({identities})"
            )

    if catalog_already_owned:
        return

    dropped_keys = sorted(set(ROW) - columns)
    if dropped_keys:
        logger.warning(
            "public_mcp_apps is missing columns %s; seeding %r without "
            "them -- this row will not self-heal on a later re-run",
            dropped_keys,
            APP_ID,
        )
    row = {k: v for k, v in ROW.items() if k in columns}
    bind.execute(sa.insert(PUBLIC_MCP_APPS_TABLE), [row])


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "public_mcp_apps" not in set(inspector.get_table_names()):
        return
    columns = {c["name"] for c in inspector.get_columns("public_mcp_apps")}
    if "launch_config" not in columns:
        return
    existing = bind.execute(
        sa.select(PUBLIC_MCP_APPS_TABLE.c.launch_config).where(
            PUBLIC_MCP_APPS_TABLE.c.app_id == APP_ID
        )
    ).scalar_one_or_none()
    if not _has_provenance(existing):
        return
    # Only the catalog entry is removed. The shared "microsoft" oauth_providers
    # row is left untouched since it is reused by Outlook/Teams/OneDrive/
    # SharePoint. Any MCPServer/UserMCPServer rows created by users who already
    # connected are intentionally left in place -- connect-driven rows are not
    # owned by this migration and are cleaned up through the normal disconnect
    # path.
    bind.execute(
        sa.delete(PUBLIC_MCP_APPS_TABLE).where(PUBLIC_MCP_APPS_TABLE.c.app_id == APP_ID)
    )
