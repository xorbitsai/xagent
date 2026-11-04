"""seed built-in Notion (remote MCP, DCR OAuth) connector

Revision ID: 20260807_seed_notion_mcp_app
Revises: 20260806_seed_google_sheets_mcp_app
Create Date: 2026-08-07 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "20260807_seed_notion_mcp_app"
down_revision: Union[str, None] = "20260806_seed_google_sheets_mcp_app"
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

APP_ID = "notion"

ROW = {
    "app_id": APP_ID,
    "name": "Notion",
    "description": "Connect to Notion to search your workspace and read, create and update pages and databases through Notion's hosted MCP server.",
    "icon": "https://www.google.com/s2/favicons?domain=notion.so&sz=128",
    "transport": "streamable_http",
    "provider_name": None,
    "category": "Productivity",
    "oauth_scopes": None,
    "is_visible_in_connector": True,
    "launch_config": {
        "url": "https://mcp.notion.com/mcp",
        "auth": {"type": "mcp_oauth"},
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

    row = {k: v for k, v in ROW.items() if k in columns}
    bind.execute(sa.insert(PUBLIC_MCP_APPS_TABLE), [row])


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "public_mcp_apps" not in set(inspector.get_table_names()):
        return
    # Only the catalog entry is removed. No oauth_providers row exists for
    # Notion (auth is per-user Dynamic Client Registration, not a shared
    # static client), and any MCPServer/UserMCPServer/MCPOAuth* rows created
    # by users who already connected are intentionally left in place —
    # connect-driven rows are not owned by this migration and are cleaned up
    # through the normal disconnect path.
    bind.execute(
        sa.delete(PUBLIC_MCP_APPS_TABLE).where(PUBLIC_MCP_APPS_TABLE.c.app_id == APP_ID)
    )
