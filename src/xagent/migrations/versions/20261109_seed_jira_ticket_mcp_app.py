"""seed built-in create-only Jira ticket (key-based) MCP connector

Revision ID: 20261109_seed_jira_ticket_mcp_app
Revises: 20261105_mcp_oauth_flow_access_path
Create Date: 2026-11-09 00:00:00.000000

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
revision: str = "20261109_seed_jira_ticket_mcp_app"
down_revision: Union[str, None] = "20261105_mcp_oauth_flow_access_path"
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

APP_ID = "jira-ticket"
BUILTIN_PROVENANCE = {
    "registry": "xagent",
    "app_id": APP_ID,
    "version": 1,
}

ROW = {
    "app_id": APP_ID,
    "name": "Jira Ticket",
    "description": (
        "Connect a Jira Cloud site with a classic (unscoped) API token to "
        "create tickets in one fixed project. This connector can only create "
        "tickets: it cannot read, search or change existing ones."
    ),
    "icon": "https://www.google.com/s2/favicons?domain=atlassian.com&sz=128",
    "transport": "stdio",
    "provider_name": None,
    "category": "Productivity",
    "oauth_scopes": None,
    # Hidden until verified against a live site (xorbitsai/xagent#3661).
    "is_visible_in_connector": False,
    "launch_config": {
        "command": "python",
        "args": ["-m", "xagent.web.tools.mcp.jira_ticket"],
        "required_env": [
            "JIRA_TICKET_SITE",
            "JIRA_TICKET_EMAIL",
            "JIRA_TICKET_API_TOKEN",
            "JIRA_TICKET_PROJECT_KEY",
        ],
        "builtin_provenance": BUILTIN_PROVENANCE,
    },
}


def _has_provenance(launch_config: object) -> bool:
    return isinstance(launch_config, dict) and builtin_provenance_identity(
        launch_config.get("builtin_provenance")
    ) == builtin_provenance_identity(BUILTIN_PROVENANCE)


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "public_mcp_apps" not in set(inspector.get_table_names()):
        return

    columns = {c["name"] for c in inspector.get_columns("public_mcp_apps")}
    # The provenance marker lives in launch_config, and it is what lets the
    # registry overlay and the connect flow tell this row from an operator's
    # own. The column filter below would silently drop it and seed a row
    # nothing can trust, so its absence fails loudly instead.
    if "launch_config" not in columns:
        raise RuntimeError(
            "Cannot seed builtin Jira Ticket identity: "
            "public_mcp_apps.launch_config is required for provenance"
        )

    identity_keys = {
        canonicalize_builtin_identity(APP_ID),
        canonicalize_builtin_identity(ROW["name"]),
    }
    for existing in (
        bind.execute(
            sa.select(
                PUBLIC_MCP_APPS_TABLE.c.app_id,
                PUBLIC_MCP_APPS_TABLE.c.name,
                PUBLIC_MCP_APPS_TABLE.c.launch_config,
            )
        )
        .mappings()
        .all()
    ):
        if existing["app_id"] == APP_ID:
            if _has_provenance(existing["launch_config"]):
                return
            # Not adopted: no marker is written, so the builtin overlay keeps
            # treating this as the operator's own row.
            logger.warning(
                "Preserving existing public_mcp_apps row with app_id=%r because "
                "it has no matching builtin_provenance; the builtin Jira Ticket "
                "connector will not be seeded",
                APP_ID,
            )
            return
        if {
            canonicalize_builtin_identity(existing["app_id"]),
            canonicalize_builtin_identity(existing["name"]),
        } & identity_keys:
            # The connect flow rejects a catalog whose normalized id or name
            # collides with another row, so a seed here would be unconnectable.
            logger.warning(
                "Not seeding builtin %r: existing public_mcp_apps row "
                "app_id=%r name=%r collides with it after normalization",
                APP_ID,
                existing["app_id"],
                existing["name"],
            )
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
    # Only a row carrying this migration's own provenance marker is removed;
    # an operator's row under the same app_id is left in place. Any
    # MCPServer/UserMCPServer rows created by users who already connected are
    # intentionally left too -- connect-driven rows are cleaned up through the
    # normal disconnect path.
    if not _has_provenance(existing):
        return
    bind.execute(
        sa.delete(PUBLIC_MCP_APPS_TABLE).where(PUBLIC_MCP_APPS_TABLE.c.app_id == APP_ID)
    )
