"""seed built-in WhatsApp Business (Meta OAuth) MCP connector

Revision ID: 20260909_seed_whatsapp_mcp_app
Revises: 20260914_durable_sandbox_lifecycles
Create Date: 2026-09-09 00:00:00.000000

"""

import logging
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

from xagent.builtin_identity import (
    builtin_provenance_identity,
    canonicalize_builtin_identity,
)

logger = logging.getLogger(__name__)

# revision identifiers, used by Alembic.
revision: str = "20260909_seed_whatsapp_mcp_app"
down_revision: Union[str, None] = "20260914_durable_sandbox_lifecycles"
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

APP_ID = "whatsapp"
BUILTIN_PROVENANCE = {
    "registry": "xagent",
    "app_id": APP_ID,
    "version": 1,
}

ROW = {
    "app_id": APP_ID,
    "name": "WhatsApp Business",
    "description": "Connect to the WhatsApp Business Platform to discover business accounts and phone numbers, browse message templates, and send text, template, and media messages to customers.",
    "icon": "https://www.google.com/s2/favicons?domain=whatsapp.com&sz=128",
    "transport": "oauth",
    "provider_name": "meta",
    "category": "Communication",
    "oauth_scopes": [
        "business_management",
        "whatsapp_business_management",
        "whatsapp_business_messaging",
    ],
    "is_visible_in_connector": True,
    "launch_config": {
        "command": "python",
        "args": ["-m", "xagent.web.tools.mcp.whatsapp"],
        "env_mapping": {"META_ACCESS_TOKEN": "access_token"},
        "builtin_provenance": BUILTIN_PROVENANCE,
    },
}

# The "meta" oauth_providers row is NOT seeded here, unlike a migration that
# introduces a brand-new provider (salesforce/deputy/myob): it already exists
# (20260627_seed_meta_connectors seeded it for Facebook Pages/Instagram, and
# this connector reuses the same provider for OAuth), and every other
# public_mcp_apps row that adds an app to an already-seeded provider
# (20260703_google_maps, 20260724_google_ads, 20260730_google_analytics,
# 20260806_google_sheets, 20260824_google_search_console, and the sibling
# meta-ads connector) leaves oauth_providers alone rather than re-seeding it
# defensively. Following that precedent instead of adding a second, drifting
# copy of the frozen provider row here.
#
# No mcp_servers-table provenance check (unlike the sibling shopify seed
# migration): that check guards shopify's key-based connect_mcp_app path,
# whose MCPServer.auth carries this same builtin_provenance marker. OAuth
# apps activate through a different path (generic_oauth_callback in
# api/auth.py) whose MCPServer.auth already carries its own
# app_id/provider identity (_oauth_auth_metadata) and is already guarded
# against a mismatched custom server by _ensure_server_matches_oauth_app --
# an independent, pre-existing mechanism every oauth-transport builtin app
# (including facebook/instagram) already relies on, so this migration adds
# no new server-level guard.


def _filter_row(row: dict[str, object], allowed_columns: set[str]) -> dict[str, object]:
    return {key: value for key, value in row.items() if key in allowed_columns}


def _has_provenance(launch_config: object) -> bool:
    return isinstance(launch_config, dict) and builtin_provenance_identity(
        launch_config.get("builtin_provenance")
    ) == builtin_provenance_identity(BUILTIN_PROVENANCE)


def _collides_with_whatsapp_identity(value: object) -> bool:
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
            "Cannot seed builtin WhatsApp identity: public_mcp_apps.launch_config "
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
        # it) already owns: accept and stop. This does NOT also require the
        # row to be the only identity collision on record -- an unrelated
        # row created *after* this one was seeded (e.g. an admin later
        # names an entirely different custom connector "WhatsApp") is a
        # separate ambiguity to police at that row's own creation time
        # (POST /admin/mcp/apps), not a reason for every subsequent
        # `alembic upgrade head` to unconditionally fail closed against an
        # already-correctly-owned row it never touches.
        if _has_provenance(existing["launch_config"]):
            return
        # app_id="whatsapp" itself is the identifier the builtin execution
        # overlay actually keys off of (get_builtin_execution_fields /
        # _matches_builtin_provenance both look up by exact app_id, never
        # by name), so an unprovenanced row squatting on it is a genuine
        # misidentification risk worth failing the whole run over: seeding
        # our own row alongside it, or silently skipping, would both leave
        # an ambiguous app_id="whatsapp" row for the overlay to reason
        # about with no way to tell which one is "ours".
        raise RuntimeError(
            "Cannot seed builtin WhatsApp connector: an existing "
            "public_mcp_apps row with app_id='whatsapp' has no matching "
            "builtin_provenance"
        )

    # No row claims app_id "whatsapp" -- the identifier that actually
    # matters to the overlay is free. A *different* app_id whose display
    # name (or, via typo/casing, its app_id) happens to normalize to the
    # same identity is a one-time cosmetic collision in the connector
    # picker, not a misidentification risk, so it doesn't warrant the same
    # blast radius: failing this migration would abort every migration in
    # the same `alembic upgrade head` run (this one or any added after it),
    # not just skip seeding whatsapp. Skip with a warning instead; an
    # operator who wants the builtin row seeded can rename the conflicting
    # app and re-run.
    colliding_rows = [
        row
        for row in catalog_rows
        if _collides_with_whatsapp_identity(row["app_id"])
        or _collides_with_whatsapp_identity(row["name"])
    ]
    if colliding_rows:
        logger.warning(
            "Skipping builtin WhatsApp seed: public_mcp_apps row(s) with "
            "app_id %s share its identity under a different app_id",
            sorted({row["app_id"] for row in colliding_rows}),
        )
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
        return
    existing = bind.execute(
        sa.select(PUBLIC_MCP_APPS_TABLE.c.launch_config).where(
            PUBLIC_MCP_APPS_TABLE.c.app_id == APP_ID
        )
    ).scalar_one_or_none()
    if not _has_provenance(existing):
        return
    bind.execute(
        sa.delete(PUBLIC_MCP_APPS_TABLE).where(PUBLIC_MCP_APPS_TABLE.c.app_id == APP_ID)
    )
