"""seed built-in Freshdesk (key-based) MCP connector

Revision ID: 20260923_seed_freshdesk_mcp_app
Revises: 20260922_task_last_activity_at
Create Date: 2026-09-23 00:00:00.000000

"""

import logging
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

from xagent.migrations.seed_helpers import delete_unmodified_seeded_rows

logger = logging.getLogger(__name__)

# revision identifiers, used by Alembic.
revision: str = "20260923_seed_freshdesk_mcp_app"
down_revision: Union[str, None] = "20260922_task_last_activity_at"
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

APP_ID = "freshdesk"

ROW = {
    "app_id": APP_ID,
    "name": "Freshdesk",
    "description": (
        "Connect your Freshdesk helpdesk (with your tenant subdomain and a "
        "per-user API key from Profile settings -> Your API Key) to look up, "
        "create and update tickets, read and post conversations, and search "
        "contacts and agents."
    ),
    "icon": "https://www.google.com/s2/favicons?domain=freshdesk.com&sz=128",
    "transport": "stdio",
    "provider_name": None,
    "category": "Support",
    "oauth_scopes": None,
    # Hidden until manually verified against a live Freshdesk tenant, matching
    # the zendesk/intercom precedent. Flipped by a follow-up migration once the
    # end-to-end check on xorbitsai/xagent-saas#1409 passes.
    "is_visible_in_connector": False,
    "launch_config": {
        "command": "python",
        "args": ["-m", "xagent.web.tools.mcp.freshdesk"],
        # Two values, both per-user: Freshdesk is multi-tenant by hostname, so
        # the subdomain identifies the account and the key authenticates the
        # agent within it. Both ride the existing encrypted per-user env path.
        "required_env": ["FRESHDESK_SUBDOMAIN", "FRESHDESK_API_KEY"],
    },
}


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "public_mcp_apps" not in set(inspector.get_table_names()):
        return

    columns = {c["name"] for c in inspector.get_columns("public_mcp_apps")}

    # is_visible_in_connector is load-bearing for the release gate this row
    # ships behind: without it the column filter below would drop the key and
    # the table default (TRUE) would seed the row visible. Checked before the
    # collision branch so both paths are covered. A plain RuntimeError rather
    # than assert: assertions are stripped under `python -O`/PYTHONOPTIMIZE,
    # which would silently defeat this guard. Same guard as the zendesk/chrome
    # seeds.
    if "is_visible_in_connector" not in columns:
        raise RuntimeError(
            "public_mcp_apps.is_visible_in_connector is missing; the freshdesk "
            "row must not seed visible"
        )

    existing = set(bind.execute(sa.select(PUBLIC_MCP_APPS_TABLE.c.app_id)).scalars())
    if APP_ID in existing:
        # A pre-existing row (e.g. hand-created by an operator) keeps its own
        # visibility, and the builtin registry overlays the real launch_config
        # onto any row with this app_id, so a visible one would become a
        # working connector. Force it hidden; its other fields are left alone.
        bind.execute(
            sa.update(PUBLIC_MCP_APPS_TABLE)
            .where(PUBLIC_MCP_APPS_TABLE.c.app_id == APP_ID)
            .values(is_visible_in_connector=False)
        )
        return

    # Any other missing column degrades rather than failing outright (this
    # table is never expected to be missing a column that predates this
    # migration by months), but the app_id-exists guard above means a row
    # seeded here while a column was missing can never self-heal on a later
    # re-run -- so at least surface which keys were dropped instead of
    # leaving no trace at all.
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
    # Only the catalog entry is removed. Freshdesk has no oauth_providers row
    # (it is key-based). Any MCPServer/UserMCPServer rows created by users who
    # already connected are intentionally left in place -- connect-driven rows
    # are not owned by this migration and are cleaned up through the normal
    # disconnect path. Matches chartmogul's/posthog's identical downgrade.
    #
    # Only rows still matching this migration's own seed snapshot are removed.
    # upgrade()'s collision branch adopts a pre-existing row by flipping only
    # is_visible_in_connector and leaves its other fields as the operator set
    # them, so an unconditional delete by app_id would destroy that operator's
    # own row -- the same reason atlassian/miro/fireflies/rocketlane use this
    # helper. An adopted row that differs from the snapshot is therefore left
    # in place as a hidden orphan: downgrade does not restore the visibility
    # it had before upgrade().
    delete_unmodified_seeded_rows(bind, PUBLIC_MCP_APPS_TABLE, [ROW])
