"""seed built-in SharePoint (OAuth) MCP connector

Revision ID: 20260917_seed_sharepoint_mcp_app
Revises: 20260916_update_hubspot_description
Create Date: 2026-09-17 00:00:00.000000

"""

import logging
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

from xagent.builtin_identity import builtin_provenance_identity

logger = logging.getLogger(__name__)

# revision identifiers, used by Alembic.
revision: str = "20260917_seed_sharepoint_mcp_app"
down_revision: Union[str, None] = "20260916_update_hubspot_description"
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

APP_ID = "sharepoint"
BUILTIN_PROVENANCE = {
    "registry": "xagent",
    "app_id": APP_ID,
    "version": 1,
}

ROW = {
    "app_id": APP_ID,
    "name": "SharePoint",
    "description": "Connect to SharePoint to search sites, browse and manage document libraries, and read and write list items.",
    "icon": "https://www.google.com/s2/favicons?domain=sharepoint.com&sz=128",
    "transport": "oauth",
    "provider_name": "microsoft",
    "category": "Storage",
    "oauth_scopes": ["Sites.ReadWrite.All"],
    "is_visible_in_connector": True,
    "launch_config": {
        "command": "python",
        "args": ["-m", "xagent.web.tools.mcp.sharepoint"],
        "env_mapping": {"AUTH_TOKEN": "access_token"},
        "builtin_provenance": BUILTIN_PROVENANCE,
    },
}


def _has_provenance(launch_config: object) -> bool:
    return isinstance(launch_config, dict) and builtin_provenance_identity(
        launch_config.get("builtin_provenance")
    ) == builtin_provenance_identity(BUILTIN_PROVENANCE)


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "public_mcp_apps" not in set(inspector.get_table_names()):
        return

    columns = {c["name"] for c in inspector.get_columns("public_mcp_apps")}
    if "launch_config" not in columns:
        raise RuntimeError(
            "Cannot seed builtin SharePoint identity: public_mcp_apps."
            "launch_config is required for provenance"
        )

    # .first() (a Row, or None only when zero rows match), not
    # .scalar_one_or_none() -- a matching row whose launch_config happens to
    # be NULL must still be treated as "the row exists" (fail closed below),
    # not conflated with "no row at all" (scalar_one_or_none() returns None
    # for both, which would let this fall through to the INSERT and crash
    # on the app_id unique constraint instead of raising the intended,
    # clearer error).
    existing_row = bind.execute(
        sa.select(PUBLIC_MCP_APPS_TABLE.c.launch_config).where(
            PUBLIC_MCP_APPS_TABLE.c.app_id == APP_ID
        )
    ).first()
    if existing_row is not None:
        # Idempotent re-run over a row this migration (or a prior version of
        # it) already owns: accept and stop.
        if _has_provenance(existing_row[0]):
            return
        # app_id="sharepoint" is what the builtin execution overlay actually
        # keys off of, so an unprovenanced row already squatting on it (e.g.
        # a custom connector an admin created via POST /admin/mcp/apps
        # before this migration ever ran) is a genuine misidentification
        # risk worth failing the whole run over -- seeding our own row
        # alongside it isn't possible (app_id is the primary key), and
        # silently skipping would leave that ambiguity unresolved and
        # unreported.
        raise RuntimeError(
            "Cannot seed builtin SharePoint connector: an existing "
            "public_mcp_apps row with app_id='sharepoint' has no matching "
            "builtin_provenance"
        )

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
    columns = {c["name"] for c in inspector.get_columns("public_mcp_apps")}
    if "launch_config" not in columns:
        return
    # Only the catalog entry is removed, and only if this migration (or a
    # prior version of it) is the one that owns it -- a row that collided
    # with app_id="sharepoint" before this migration ever ran (see upgrade's
    # fail-closed check above) or was never seeded is left untouched rather
    # than deleted by identifier alone. The shared "microsoft" oauth_providers
    # row is left untouched too, since it is reused by Outlook/Teams/OneDrive.
    # Any MCPServer/UserMCPServer rows created by users who already connected
    # are intentionally left in place -- connect-driven rows are not owned by
    # this migration and are cleaned up through the normal disconnect path.
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
