"""seed built-in Chrome (keyless) MCP connector

Revision ID: 20260806_seed_chrome_mcp_app
Revises: 20260808_add_task_lease_attempt_id
Create Date: 2026-08-06 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "20260806_seed_chrome_mcp_app"
down_revision: Union[str, None] = "20260808_add_task_lease_attempt_id"
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

# Vendor-scoped, not the generic "chrome" -- see builtin_mcp_registry.py's
# chrome row for why (collision with user-created servers literally named
# "chrome"/"Chrome").
APP_ID = "chrome-devtools"

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
            "--prefer-offline",
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

    # is_visible_in_connector is load-bearing for the release gate this row
    # ships behind, so its absence must fail loudly rather than let the
    # column-filter below silently drop it and fall back to the table
    # default (TRUE) -- seeding the row visible, the opposite of intent.
    # Unreachable through any normal `alembic upgrade head` (the migration
    # that adds the column is a real ancestor in this chain). A plain
    # RuntimeError rather than assert: assertions are stripped under
    # `python -O`/PYTHONOPTIMIZE, which would silently defeat this guard.
    # Checked before the collision branch so both paths are covered.
    if "is_visible_in_connector" not in columns:
        raise RuntimeError(
            "public_mcp_apps.is_visible_in_connector is missing; the chrome "
            "row must not seed visible"
        )

    existing = set(bind.execute(sa.select(PUBLIC_MCP_APPS_TABLE.c.app_id)).scalars())
    if APP_ID in existing:
        # A row with this app_id already exists (e.g. hand-created by an
        # operator before this migration deployed -- #1143 literally asks
        # for a Chrome connector, so that is a realistic precondition). Such
        # a row keeps its own is_visible_in_connector, which defaults to
        # TRUE for hand-created rows -- and the builtin registry overlays
        # the real transport/launch_config onto ANY row sharing this app_id
        # at read time, so a visible pre-existing row would silently become
        # a working, one-click-connectable Chrome connector, defeating the
        # hidden-rollout gate with no further action. Enforce hidden on the
        # collision branch too, instead of returning untouched.
        bind.execute(
            sa.update(PUBLIC_MCP_APPS_TABLE)
            .where(PUBLIC_MCP_APPS_TABLE.c.app_id == APP_ID)
            .values(is_visible_in_connector=False)
        )
        return

    row = {k: v for k, v in ROW.items() if k in columns}
    bind.execute(sa.insert(PUBLIC_MCP_APPS_TABLE), [row])


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "public_mcp_apps" not in set(inspector.get_table_names()):
        return
    # Only the catalog entry is removed. Any MCPServer/UserMCPServer rows
    # created by users who already connected are not owned by this migration
    # and are intentionally left in place. Note the app-scoped
    # connect/disconnect routes 404 once this row is gone (get_app_by_id
    # returns None), so leftover rows are removed via the server-level route
    # (DELETE /api/mcp/servers/{id}), not the catalog one.
    #
    # An unconditional DELETE-by-app_id is NOT safe here: upgrade()'s
    # collision branch adopts a pre-existing hand-created row (e.g. an
    # operator who created one before this migration deployed, per #1143)
    # by flipping only is_visible_in_connector -- name/description/transport
    # are left exactly as the operator set them. Deleting unconditionally on
    # downgrade would destroy that operator's own row, not "remove the entry
    # this migration owns." Matching on name/description/transport (specific
    # enough that a hand-made row coincidentally matching all three isn't a
    # realistic concern) distinguishes a genuinely migration-created row from
    # an adopted one without needing a new tracking column -- a row that
    # doesn't match is left in place, restored rather than destroyed.
    # Known edge, in the safe direction: description (unlike name/transport)
    # is admin-editable even on builtin rows, so a migration-created row
    # whose description an admin later changed is skipped too -- downgrade
    # then leaves a hidden orphan row behind rather than risking deleting
    # operator-owned data. Same tradeoff the zoom seed migration makes for
    # its provider-row guard.
    bind.execute(
        sa.delete(PUBLIC_MCP_APPS_TABLE)
        .where(PUBLIC_MCP_APPS_TABLE.c.app_id == APP_ID)
        .where(PUBLIC_MCP_APPS_TABLE.c.name == ROW["name"])
        .where(PUBLIC_MCP_APPS_TABLE.c.description == ROW["description"])
        .where(PUBLIC_MCP_APPS_TABLE.c.transport == ROW["transport"])
    )
