"""seed built-in Word (OAuth) MCP connector

Revision ID: 20260917_seed_word_mcp_app
Revises: 20260916_merge_delivery_memory
Create Date: 2026-09-17 00:00:00.000000

"""

import logging
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

logger = logging.getLogger(__name__)

# revision identifiers, used by Alembic.
revision: str = "20260917_seed_word_mcp_app"
down_revision: Union[str, None] = "20260916_merge_delivery_memory"
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

APP_ID = "word"

ROW = {
    "app_id": APP_ID,
    "name": "Word",
    "description": "Connect to Word to read, create, and edit documents stored on OneDrive or SharePoint.",
    "icon": "https://www.google.com/s2/favicons?domain=office.com&sz=128",
    "transport": "oauth",
    "provider_name": "microsoft",
    "category": "Productivity",
    "oauth_scopes": ["Files.ReadWrite"],
    "is_visible_in_connector": True,
    "launch_config": {
        "command": "python",
        "args": ["-m", "xagent.web.tools.mcp.word"],
        "env_mapping": {"AUTH_TOKEN": "access_token"},
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
    # Only the catalog entry is removed. The shared "microsoft" oauth_providers
    # row is left untouched since it is reused by Outlook/Teams/OneDrive/
    # SharePoint/Excel/Planner. Any MCPServer/UserMCPServer rows created by
    # users who already connected are intentionally left in place --
    # connect-driven rows are not owned by this migration and are cleaned up
    # through the normal disconnect path.
    bind.execute(
        sa.delete(PUBLIC_MCP_APPS_TABLE).where(PUBLIC_MCP_APPS_TABLE.c.app_id == APP_ID)
    )
