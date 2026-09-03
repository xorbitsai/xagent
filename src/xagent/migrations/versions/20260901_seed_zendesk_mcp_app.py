"""seed built-in Zendesk (key-based) MCP connector

Revision ID: 20260901_seed_zendesk_mcp_app
Revises: 20260901_seed_mixpanel_mcp_app
Create Date: 2026-09-01 00:00:00.000000

"""

import logging
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

logger = logging.getLogger(__name__)

# revision identifiers, used by Alembic.
revision: str = "20260901_seed_zendesk_mcp_app"
down_revision: Union[str, None] = "20260901_seed_mixpanel_mcp_app"
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

APP_ID = "zendesk"

ROW = {
    "app_id": APP_ID,
    "name": "Zendesk",
    "description": "Connect to Zendesk with an API token to search and manage tickets, reply to customers or add internal notes, and look up users and organizations.",
    "icon": "https://www.google.com/s2/favicons?domain=zendesk.com&sz=128",
    "transport": "stdio",
    "provider_name": None,
    "category": "Support",
    "oauth_scopes": None,
    "is_visible_in_connector": False,
    "launch_config": {
        "command": "python",
        "args": ["-m", "xagent.web.tools.mcp.zendesk"],
        "required_env": ["ZENDESK_SUBDOMAIN", "ZENDESK_EMAIL", "ZENDESK_API_TOKEN"],
    },
}


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "public_mcp_apps" not in set(inspector.get_table_names()):
        return

    columns = {c["name"] for c in inspector.get_columns("public_mcp_apps")}
    existing = set(bind.execute(sa.select(PUBLIC_MCP_APPS_TABLE.c.app_id)).scalars())
    if APP_ID in existing:
        return

    # Silently degrades rather than failing outright (this table is never
    # expected to be missing a column that predates this migration by
    # months), but the app_id-exists guard above means a row seeded here
    # while a column was missing can never self-heal on a later re-run --
    # so at least surface which keys were dropped instead of leaving no
    # trace at all.
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
    # Only the catalog entry is removed. Zendesk has no oauth_providers row
    # (it is key-based). Any MCPServer/UserMCPServer rows created by users who
    # already connected are intentionally left in place -- connect-driven
    # rows are not owned by this migration and are cleaned up through the
    # normal disconnect path.
    bind.execute(
        sa.delete(PUBLIC_MCP_APPS_TABLE).where(PUBLIC_MCP_APPS_TABLE.c.app_id == APP_ID)
    )
