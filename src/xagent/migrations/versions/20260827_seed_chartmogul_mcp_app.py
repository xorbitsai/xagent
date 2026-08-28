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

# The full set of fields that identify "our" row. downgrade() matches on
# all of them before deleting -- a stricter bar is the safe direction
# there, since a false match destroys data while a false non-match just
# leaves a harmless orphan. upgrade() uses a narrower subset of this same
# tuple (see its own comment) for the opposite reason: a false "not ours"
# there raises and blocks a deploy, so it can't afford to trip on a field
# that's expected to legitimately drift.
IDENTITY_FIELDS = ("name", "description", "transport")

ROW = {
    "app_id": APP_ID,
    "name": "ChartMogul",
    "description": "Connect your ChartMogul account to look up subscription metrics, customers, and revenue analytics -- this connector can also create and update customers, contacts, customer notes, opportunities, plans, plan groups, subscription events, and tasks, and import invoices, not just read them.",
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

    # Excludes 'description' from IDENTITY_FIELDS (see its own comment) --
    # name/transport are core, NOT NULL columns this table has carried
    # since long before this migration, so unlike the dropped_keys handling
    # further down there's no realistic case where they're absent.
    own_row_fields = [f for f in IDENTITY_FIELDS if f != "description"]
    existing_row = (
        bind.execute(
            sa.select(*(PUBLIC_MCP_APPS_TABLE.c[f] for f in own_row_fields)).where(
                PUBLIC_MCP_APPS_TABLE.c.app_id == APP_ID
            )
        )
        .mappings()
        .first()
    )
    if existing_row is not None:
        if all(existing_row[f] == ROW[f] for f in own_row_fields):
            # Matches what this migration would have inserted -- either our
            # own row from an earlier run, or (rarer) a row that happens to
            # coincide on name/transport. Either way, force hidden rather
            # than trusting whatever is_visible_in_connector already holds:
            # it isn't part of the match (an admin flipping it later is a
            # separate, already-accepted risk -- see the registry's
            # comment), so a coincidentally-matching row that was already
            # visible would otherwise sail through untouched and start
            # getting ChartMogul's real launch_config overlaid onto it at
            # read time. A no-op if it's already hidden, which is the
            # common case for our own row.
            bind.execute(
                sa.update(PUBLIC_MCP_APPS_TABLE)
                .where(PUBLIC_MCP_APPS_TABLE.c.app_id == APP_ID)
                .values(is_visible_in_connector=False)
            )
            return
        # 'chartmogul' had no special meaning before this migration: nothing
        # stopped an operator from hand-creating a custom PublicMCPApp with
        # this exact app_id beforehand (admin_mcp.py only rejects app_ids
        # that are ALREADY builtin). Silently adopting that row here would
        # mean the builtin registry starts overlaying ChartMogul's real
        # transport/launch_config onto someone else's connector at read
        # time, and a later downgrade could delete it outright. Neither is
        # safe to do automatically without knowing whether the row is
        # actually ours, so fail loudly and let an operator resolve the
        # collision by hand instead. chrome/intercom's seed migrations
        # still use the older, weaker silent-adopt pattern for the
        # identical risk -- not backported here since it's a change to
        # already-shipped migrations, out of scope for this one connector;
        # tracked at https://github.com/xorbitsai/xagent/issues/1896.
        raise RuntimeError(
            f"public_mcp_apps already has a row with app_id={APP_ID!r} "
            "that this migration did not create (its name or transport "
            "don't match the seed row) -- rename or remove that row "
            "before upgrading, since 'chartmogul' is now a reserved "
            "built-in app_id"
        )

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
    # Degrades the same way upgrade() does if description was dropped from
    # an old schema -- see the identical comment there.
    columns = {c["name"] for c in inspector.get_columns("public_mcp_apps")}
    identity_fields = [f for f in IDENTITY_FIELDS if f in columns]

    # Only the catalog entry is removed. ChartMogul has no oauth_providers row
    # (it is key-based). Any MCPServer/UserMCPServer rows created by users who
    # already connected are intentionally left in place -- connect-driven
    # rows are not owned by this migration and are cleaned up through the
    # normal disconnect path.
    #
    # An unconditional DELETE-by-app_id is NOT safe here: upgrade() only
    # ever inserts a row matching IDENTITY_FIELDS exactly (a foreign row
    # with this app_id makes it raise instead), but there is no tracking
    # column recording that this row came from this migration. Matching on
    # IDENTITY_FIELDS (specific enough that a hand-made row coincidentally
    # matching all of them isn't a realistic concern) is the same identity
    # check upgrade() itself uses to tell "our row" from a collision -- a
    # row that doesn't match is left in place rather than destroyed.
    # Mirrors 20260806_seed_chrome_mcp_app.py's downgrade for the same
    # reason.
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
    query = sa.delete(PUBLIC_MCP_APPS_TABLE).where(
        PUBLIC_MCP_APPS_TABLE.c.app_id == APP_ID
    )
    for field in identity_fields:
        query = query.where(PUBLIC_MCP_APPS_TABLE.c[field] == ROW[field])
    bind.execute(query)
