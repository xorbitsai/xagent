"""seed built-in Google Ads (OAuth) MCP connector

Revision ID: 20260724_seed_google_ads_mcp_app
Revises: 20260722_add_workforce_id_to_agent_api_keys
Create Date: 2026-07-24 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "20260724_seed_google_ads_mcp_app"
down_revision: Union[str, None] = "20260722_add_workforce_id_to_agent_api_keys"
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

APP_ID = "google-ads"

ROW = {
    "app_id": APP_ID,
    "name": "Google Ads",
    "description": "Connect to Google Ads to list accessible accounts, inspect campaigns, and run GAQL reports.",
    "icon": "https://www.google.com/s2/favicons?domain=ads.google.com&sz=128",
    "transport": "oauth",
    "provider_name": "google",
    "category": "Marketing",
    "oauth_scopes": ["https://www.googleapis.com/auth/adwords"],
    "is_visible_in_connector": True,
    "launch_config": {
        "command": "python",
        "args": ["-m", "xagent.web.tools.mcp.google_ads"],
        "env_mapping": {"GOOGLE_ACCESS_TOKEN": "access_token"},
        "static_env": {"GOOGLE_ADS_DEVELOPER_TOKEN": "GOOGLE_ADS_DEVELOPER_TOKEN"},
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
    # Only the catalog entry is removed. The shared "google" oauth_providers row
    # is left untouched since it is reused by Gmail/Drive/Calendar/Docs/Slides.
    # Any MCPServer/UserMCPServer rows created by users who already connected are
    # intentionally left in place — connect-driven rows are not owned by this
    # migration and are cleaned up through the normal disconnect path.
    bind.execute(
        sa.delete(PUBLIC_MCP_APPS_TABLE).where(PUBLIC_MCP_APPS_TABLE.c.app_id == APP_ID)
    )
