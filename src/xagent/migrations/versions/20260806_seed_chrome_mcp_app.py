"""seed built-in Chrome (keyless) MCP connector

Revision ID: 20260806_seed_chrome_mcp_app
Revises: 20260807_seed_notion_mcp_app
Create Date: 2026-08-06 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "20260806_seed_chrome_mcp_app"
down_revision: Union[str, None] = "20260807_seed_notion_mcp_app"
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

APP_ID = "chrome"

ROW = {
    "app_id": APP_ID,
    "name": "Chrome",
    "description": "Automate a Chrome browser: open pages, read content, fill forms, take screenshots, and inspect network requests.",
    "icon": "https://www.google.com/s2/favicons?domain=chrome.google.com&sz=128",
    "transport": "stdio",
    "provider_name": None,
    "category": "Productivity",
    "oauth_scopes": None,
    # Hidden until the runtime supports execution-scoped (persistent) stdio
    # MCP sessions — see builtin_mcp_registry.py's chrome row. Admins can
    # re-enable via PATCH /api/admin/mcp/apps once that lands.
    "is_visible_in_connector": False,
    # Keyless (non-oauth): no required_env — connecting only creates the
    # per-user association via POST /api/mcp/apps/{id}/connect.
    # Version pin + sandbox/telemetry flags: see builtin_mcp_registry.py's
    # chrome row for the full rationale. This is a frozen snapshot of that
    # row (test_seed_row_matches_registry enforces they never drift apart).
    "launch_config": {
        "command": "npx",
        "args": [
            "-y",
            "chrome-devtools-mcp@1.6.0",
            "--headless",
            "--isolated",
            "--chrome-arg=--no-sandbox",
            "--chrome-arg=--disable-setuid-sandbox",
            "--no-usage-statistics",
            "--no-performance-crux",
        ],
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
    # Only the catalog entry is removed. Any MCPServer/UserMCPServer rows created
    # by users who already connected are not owned by this migration and are
    # cleaned up through the normal disconnect path.
    bind.execute(
        sa.delete(PUBLIC_MCP_APPS_TABLE).where(PUBLIC_MCP_APPS_TABLE.c.app_id == APP_ID)
    )
