"""seed built-in Meta Ads MCP connector

Revision ID: 20260909_seed_meta_ads_mcp_app
Revises: 20260901_seed_zendesk_mcp_app
Create Date: 2026-09-09 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "20260909_seed_meta_ads_mcp_app"
down_revision: Union[str, None] = "20260901_seed_zendesk_mcp_app"
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

APP_ID = "meta-ads"

ROW = {
    "app_id": APP_ID,
    "name": "Meta Ads",
    "description": (
        "Connect to Meta Ads to list ad accounts, inspect campaigns, ad "
        "sets, and ads, and pull performance insights."
    ),
    "icon": "https://www.google.com/s2/favicons?domain=facebook.com&sz=128",
    "transport": "oauth",
    "provider_name": "meta",
    "category": "Marketing",
    "oauth_scopes": ["ads_read"],
    "is_visible_in_connector": True,
    "launch_config": {
        "command": "python",
        "args": ["-m", "xagent.web.tools.mcp.meta_ads"],
        "env_mapping": {"META_ACCESS_TOKEN": "access_token"},
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
    # Only the catalog entry is removed. The shared "meta" oauth_providers
    # row is left untouched since it is reused by Facebook/Instagram. Any
    # MCPServer/UserMCPServer rows created by users who already connected are
    # intentionally left in place -- connect-driven rows are not owned by
    # this migration and are cleaned up through the normal disconnect path.
    bind.execute(
        sa.delete(PUBLIC_MCP_APPS_TABLE).where(PUBLIC_MCP_APPS_TABLE.c.app_id == APP_ID)
    )
