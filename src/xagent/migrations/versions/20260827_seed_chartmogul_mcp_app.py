"""seed built-in ChartMogul (key-based) MCP connector

Revision ID: 20260827_seed_chartmogul_mcp_app
Revises: 20260821_actor_oauth_flow_states
Create Date: 2026-08-27 00:00:00.000000

"""

import logging
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

logger = logging.getLogger(__name__)

# revision identifiers, used by Alembic.
revision: str = "20260827_seed_chartmogul_mcp_app"
down_revision: Union[str, None] = "20260821_actor_oauth_flow_states"
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

APP_ID = "chartmogul"

ROW = {
    "app_id": APP_ID,
    "name": "ChartMogul",
    "description": "Connect your ChartMogul account to look up subscription metrics, customers, and revenue analytics -- this connector can also create and update customers, contacts, opportunities, tasks, plans, and invoices, not just read them.",
    "icon": "https://www.google.com/s2/favicons?domain=chartmogul.com&sz=128",
    "transport": "stdio",
    "provider_name": None,
    "category": "Analytics",
    "oauth_scopes": None,
    "is_visible_in_connector": False,
    "launch_config": {
        "command": "uv",
        "args": [
            "--directory",
            "/opt/xagent/vendor/chartmogul-mcp-server",
            "run",
            "--no-sync",
            "main.py",
        ],
        "required_env": ["CHARTMOGUL_TOKEN"],
    },
}


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "public_mcp_apps" not in set(inspector.get_table_names()):
        return

    columns = {c["name"] for c in inspector.get_columns("public_mcp_apps")}

    # is_visible_in_connector is load-bearing for the Docker-only gate this
    # row ships behind (see the registry row's comment), so its absence must
    # fail loudly rather than let the column-filter below silently drop it
    # and fall back to the table default (TRUE) -- seeding the row visible,
    # the opposite of intent. Unreachable through any normal
    # `alembic upgrade head` (the migration that adds the column is a real
    # ancestor in this chain). A plain RuntimeError rather than assert:
    # assertions are stripped under python -O/PYTHONOPTIMIZE, which would
    # silently defeat this guard. Checked before the collision branch so
    # both paths are covered. Mirrors 20260806_seed_chrome_mcp_app.py, which
    # ships hidden for the same reason.
    if "is_visible_in_connector" not in columns:
        raise RuntimeError(
            "public_mcp_apps.is_visible_in_connector is missing; the "
            "chartmogul row must not seed visible"
        )

    existing = set(bind.execute(sa.select(PUBLIC_MCP_APPS_TABLE.c.app_id)).scalars())
    if APP_ID in existing:
        # A row with this app_id already exists (e.g. hand-created by an
        # operator before this migration deployed). Such a row keeps its own
        # is_visible_in_connector, which defaults to TRUE for hand-created
        # rows -- and the builtin registry overlays the real
        # transport/launch_config onto ANY row sharing this app_id at read
        # time, so a visible pre-existing row would silently become a
        # working, one-click-connectable ChartMogul connector that only
        # functions inside the Docker image, defeating the hidden gate with
        # no further action. Enforce hidden on the collision branch too,
        # instead of returning untouched.
        bind.execute(
            sa.update(PUBLIC_MCP_APPS_TABLE)
            .where(PUBLIC_MCP_APPS_TABLE.c.app_id == APP_ID)
            .values(is_visible_in_connector=False)
        )
        return

    # Silently degrades rather than failing outright (this table is never
    # expected to be missing a column that predates this migration by
    # months, and is_visible_in_connector -- the one column that actually
    # matters for this row -- is already guaranteed present above), but the
    # app_id-exists guard above means a row seeded here while a column was
    # missing can never self-heal on a later re-run -- so at least surface
    # which keys were dropped instead of leaving no trace at all.
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
    # Only the catalog entry is removed. ChartMogul has no oauth_providers row
    # (it is key-based). Any MCPServer/UserMCPServer rows created by users who
    # already connected are intentionally left in place -- connect-driven
    # rows are not owned by this migration and are cleaned up through the
    # normal disconnect path.
    #
    # An unconditional DELETE-by-app_id is NOT safe here: upgrade()'s
    # collision branch adopts a pre-existing hand-created row by flipping
    # only is_visible_in_connector -- name/description/transport are left
    # exactly as the operator set them. Deleting unconditionally on downgrade
    # would destroy that operator's own row, not "remove the entry this
    # migration owns." Matching on name/description/transport (specific
    # enough that a hand-made row coincidentally matching all three isn't a
    # realistic concern) distinguishes a genuinely migration-created row from
    # an adopted one without a new tracking column -- a row that doesn't
    # match is left in place, restored rather than destroyed. Mirrors
    # 20260806_seed_chrome_mcp_app.py's downgrade for the same reason.
    # Also connect/disconnect routes 404 once this row is gone
    # (get_app_by_id returns None), so leftover connections from users who
    # already connected are removed via the server-level route (DELETE
    # /api/mcp/servers/{id}), not the catalog one.
    # Known edge, in the safe direction: description (unlike name/transport)
    # is admin-editable even on builtin rows, so a migration-created row
    # whose description an admin later changed is skipped too -- downgrade
    # then leaves a hidden orphan row behind rather than risking deleting
    # operator-owned data. Same tradeoff the chrome and zoom seed migrations
    # make for the identical reason.
    bind.execute(
        sa.delete(PUBLIC_MCP_APPS_TABLE)
        .where(PUBLIC_MCP_APPS_TABLE.c.app_id == APP_ID)
        .where(PUBLIC_MCP_APPS_TABLE.c.name == ROW["name"])
        .where(PUBLIC_MCP_APPS_TABLE.c.description == ROW["description"])
        .where(PUBLIC_MCP_APPS_TABLE.c.transport == ROW["transport"])
    )
