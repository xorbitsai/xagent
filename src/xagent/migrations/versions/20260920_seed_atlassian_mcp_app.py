"""seed built-in Atlassian (remote MCP, DCR OAuth) connector

Revision ID: 20260920_seed_atlassian_mcp_app
Revises: 20260917_seed_planner_mcp_app
Create Date: 2026-09-20 00:00:00.000000

"""

import logging
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

from xagent.builtin_identity import (
    builtin_provenance_identity,
    canonicalize_builtin_identity,
)
from xagent.migrations.seed_helpers import (
    delete_unmodified_seeded_rows,
    remote_mcp_server_identity_is_claimable,
)

logger = logging.getLogger(__name__)

# revision identifiers, used by Alembic.
revision: str = "20260920_seed_atlassian_mcp_app"
down_revision: Union[str, None] = "20260917_seed_planner_mcp_app"
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

APP_ID = "atlassian"
BUILTIN_PROVENANCE = {
    "registry": "xagent",
    "app_id": APP_ID,
    "version": 1,
}

# Frozen copy of the builtin_mcp_registry.py row at the time this migration
# was written (tests/alembic pins the two against each other). Same connector
# shape as the granola/notion seeds: a vendor-hosted remote MCP server reached
# over streamable_http, authenticated per user through OAuth 2.1 + PKCE with
# Dynamic Client Registration, so there is no oauth_providers row and no
# secret in launch_config. Unlike those two seeds it carries the
# builtin_provenance ownership marker (whatsapp/shopify pattern) so that a
# pre-existing operator row under the same app_id is never adopted on upgrade
# nor deleted on downgrade.
# The vendor's current endpoint is /v2/mcp; the legacy /v1/sse endpoint is
# unsupported after 2026-06-30. This row sits alongside the separate "jira"
# row (our own local Jira tool behind Atlassian 3LO, transport "oauth"), which
# is left untouched.
ROW = {
    "app_id": APP_ID,
    "name": "Atlassian (Jira, Confluence, Bitbucket)",
    "description": "Connect to Atlassian to search and work with Jira issues, Confluence pages and Bitbucket repositories through Atlassian's hosted MCP server.",
    "icon": "https://www.google.com/s2/favicons?domain=atlassian.com&sz=128",
    "transport": "streamable_http",
    "provider_name": None,
    "category": "Productivity",
    "oauth_scopes": None,
    "is_visible_in_connector": True,
    "launch_config": {
        "url": "https://mcp.atlassian.com/v2/mcp",
        "auth": {"type": "mcp_oauth"},
        "builtin_provenance": BUILTIN_PROVENANCE,
    },
}

# mcp_servers is reconciled before the identity is claimed, through the shared
# seed_helpers.remote_mcp_server_identity_is_claimable: the connector listing
# resolves a catalog app's shared server row by normalized transport +
# app_id/display name, so a custom streamable_http server that predates this
# identity would otherwise be presented as the official card while the runtime
# kept using its own URL/auth. The helper accepts only the row our own connect
# path creates and otherwise logs an actionable error and skips seeding without
# touching any row (the granola/notion seeds get the same check via #2514).


def _filter_row(row: dict[str, object], allowed_columns: set[str]) -> dict[str, object]:
    return {key: value for key, value in row.items() if key in allowed_columns}


def _has_provenance(launch_config: object) -> bool:
    return isinstance(launch_config, dict) and builtin_provenance_identity(
        launch_config.get("builtin_provenance")
    ) == builtin_provenance_identity(BUILTIN_PROVENANCE)


def _collides_with_atlassian_identity(value: object) -> bool:
    return canonicalize_builtin_identity(value) in {
        canonicalize_builtin_identity(APP_ID),
        canonicalize_builtin_identity(ROW["name"]),
    }


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    tables = set(inspector.get_table_names())
    if "public_mcp_apps" not in tables:
        return

    columns = {column["name"] for column in inspector.get_columns("public_mcp_apps")}
    if "launch_config" not in columns:
        raise RuntimeError(
            "Cannot seed builtin Atlassian identity: public_mcp_apps.launch_config "
            "is required for provenance"
        )

    catalog_rows = list(
        bind.execute(
            sa.select(
                PUBLIC_MCP_APPS_TABLE.c.app_id,
                PUBLIC_MCP_APPS_TABLE.c.name,
                PUBLIC_MCP_APPS_TABLE.c.launch_config,
            )
        ).mappings()
    )
    exact_app_rows = [row for row in catalog_rows if row["app_id"] == APP_ID]
    if exact_app_rows:
        existing = exact_app_rows[0]
        # Idempotent re-run over a row this migration (or a prior version of
        # it) already owns: accept and stop. An unrelated row created *after*
        # this one was seeded is policed at that row's own creation time
        # (POST /admin/mcp/apps), not by failing every later
        # `alembic upgrade head` against an already-correctly-owned row.
        if _has_provenance(existing["launch_config"]):
            return
        # app_id is what the builtin execution overlay keys off of
        # (get_builtin_execution_fields / _matches_builtin_provenance look up
        # by exact app_id, never by name), so an unprovenanced row squatting
        # on it is a genuine misidentification risk: fail closed rather than
        # seed alongside it or silently skip.
        raise RuntimeError(
            "Cannot seed builtin Atlassian connector: an existing "
            "public_mcp_apps row with app_id='atlassian' has no matching "
            "builtin_provenance"
        )

    # No row claims app_id "atlassian". A *different* app_id whose display name
    # (or, via typo/casing, its app_id) normalizes to the same identity is a
    # one-time cosmetic collision in the connector picker, not a
    # misidentification risk, so it does not warrant aborting the whole
    # `alembic upgrade head` run. Skip with a warning instead.
    #
    # This skip is permanent: Alembic stamps this revision as applied whether
    # or not the insert below ran, so a re-run will NOT retry seeding even if
    # the colliding row is later renamed away. Recovering the builtin row
    # afterwards needs a manual INSERT (or a follow-up migration) using this
    # file's ROW/BUILTIN_PROVENANCE as the template.
    colliding_rows = [
        row
        for row in catalog_rows
        if _collides_with_atlassian_identity(row["app_id"])
        or _collides_with_atlassian_identity(row["name"])
    ]
    if colliding_rows:
        logger.error(
            "Permanently skipping builtin Atlassian seed: public_mcp_apps "
            "row(s) with app_id %s share its identity under a different "
            "app_id. Re-running `alembic upgrade head` will NOT retry this "
            "-- the revision is already stamped applied. Seed the row "
            "manually (see this migration's ROW/BUILTIN_PROVENANCE) once "
            "the collision is resolved.",
            sorted({row["app_id"] for row in colliding_rows}),
        )
        return

    if not remote_mcp_server_identity_is_claimable(
        bind,
        app_id=APP_ID,
        display_name=str(ROW["name"]),
        transport=str(ROW["transport"]),
        url=str(ROW["launch_config"]["url"]),
        auth=dict(ROW["launch_config"]["auth"]),
        seed_label="Atlassian",
    ):
        return

    dropped_keys = sorted(set(ROW) - columns)
    if dropped_keys:
        logger.warning(
            "public_mcp_apps is missing columns %s; seeding %r without them",
            dropped_keys,
            APP_ID,
        )
    bind.execute(sa.insert(PUBLIC_MCP_APPS_TABLE), [_filter_row(ROW, columns)])


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "public_mcp_apps" not in set(inspector.get_table_names()):
        return
    columns = {column["name"] for column in inspector.get_columns("public_mcp_apps")}
    if "launch_config" not in columns:
        # Without launch_config there is no provenance marker to compare, so
        # ownership of a same-app_id row cannot be established; leave it.
        return
    # Only the row this migration seeded is removed, and only while it still
    # matches the frozen seed snapshot: the shared helper compares every
    # seeded column (launch_config included, which is where the provenance
    # marker lives, so an unprovenanced operator row never matches) and
    # preserves a row an administrator has since edited through the admin
    # PATCH endpoint (description, icon, category, visibility). No
    # oauth_providers row exists for Atlassian (auth is per-user Dynamic Client
    # Registration, not a shared static client), and any
    # MCPServer/UserMCPServer/MCPOAuth* rows created by users who already
    # connected are intentionally left in place -- connect-driven rows are not
    # owned by this migration and are cleaned up through the normal disconnect
    # path.
    delete_unmodified_seeded_rows(bind, PUBLIC_MCP_APPS_TABLE, [ROW])
