"""seed built-in Google Maps (key-based) MCP connector

Revision ID: 20260703_seed_google_maps_mcp_app
Revises: 20260704_merge_alembic_heads
Create Date: 2026-07-03 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

from xagent.migrations.seed_helpers import delete_unmodified_seeded_rows

# revision identifiers, used by Alembic.
revision: str = "20260703_seed_google_maps_mcp_app"
down_revision: Union[str, None] = "20260704_merge_alembic_heads"
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

APP_ID = "google-maps"

ROW = {
    "app_id": APP_ID,
    "name": "Google Maps",
    "description": "Geocoding, directions, place search, and more via the Google Maps APIs.",
    "icon": "https://www.google.com/s2/favicons?domain=maps.google.com&sz=128",
    "transport": "stdio",
    "provider_name": None,
    "category": "Productivity",
    "oauth_scopes": None,
    "is_visible_in_connector": True,
    "launch_config": {
        "command": "npx",
        "args": ["-y", "@cablate/mcp-google-map", "--stdio"],
        "required_env": ["GOOGLE_MAPS_API_KEY"],
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
    # Only the catalog entry is removed, and only if it still matches this
    # migration's seed snapshot — an operator's pre-existing custom
    # app_id="google-maps" row is left in place. Any MCPServer/UserMCPServer
    # rows created by users who already connected (possibly holding an admin
    # platform key) are intentionally left in place — connect-driven rows are
    # not owned by this migration and are cleaned up through the normal
    # disconnect path.
    delete_unmodified_seeded_rows(bind, PUBLIC_MCP_APPS_TABLE, [ROW])
