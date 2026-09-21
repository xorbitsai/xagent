"""seed built-in Planner (OAuth) MCP connector

Revision ID: 20260917_seed_planner_mcp_app
Revises: 20260917_seed_excel_mcp_app
Create Date: 2026-09-17 00:00:00.000000

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
revision: str = "20260917_seed_planner_mcp_app"
down_revision: Union[str, None] = "20260917_seed_excel_mcp_app"
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

APP_ID = "planner"
BUILTIN_PROVENANCE = {
    "registry": "xagent",
    "app_id": APP_ID,
    "version": 1,
}

ROW = {
    "app_id": APP_ID,
    "name": "Planner",
    "description": "Connect a Microsoft 365 work or school account to manage basic Planner plans, buckets, and tasks, including checklists and assignments. Personal Microsoft accounts and Premium plans are not supported.",
    "icon": "https://www.google.com/s2/favicons?domain=tasks.office.com&sz=128",
    "transport": "oauth",
    "provider_name": "microsoft",
    "category": "Productivity",
    "oauth_scopes": ["Tasks.ReadWrite"],
    "is_visible_in_connector": True,
    "launch_config": {
        "command": "python",
        "args": ["-m", "xagent.web.tools.mcp.planner"],
        "env_mapping": {"AUTH_TOKEN": "access_token"},
        "builtin_provenance": BUILTIN_PROVENANCE,
    },
}


def _has_provenance(launch_config: object) -> bool:
    return isinstance(launch_config, dict) and builtin_provenance_identity(
        launch_config.get("builtin_provenance")
    ) == builtin_provenance_identity(BUILTIN_PROVENANCE)


def _identity_collisions(bind: sa.engine.Connection) -> list[dict[str, object]]:
    normalized_app_id = canonicalize_builtin_identity(APP_ID)
    return [
        dict(row)
        for row in bind.execute(
            sa.select(
                PUBLIC_MCP_APPS_TABLE.c.app_id,
                PUBLIC_MCP_APPS_TABLE.c.launch_config,
            )
        ).mappings()
        if canonicalize_builtin_identity(row["app_id"]) == normalized_app_id
    ]


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "public_mcp_apps" not in set(inspector.get_table_names()):
        return

    columns = {c["name"] for c in inspector.get_columns("public_mcp_apps")}
    if "launch_config" not in columns:
        raise RuntimeError(
            "Cannot seed builtin Planner identity: "
            "public_mcp_apps.launch_config is required for provenance"
        )

    collisions = _identity_collisions(bind)
    if collisions:
        owned_exact = next(
            (
                row
                for row in collisions
                if row["app_id"] == APP_ID and _has_provenance(row["launch_config"])
            ),
            None,
        )
        if owned_exact is not None:
            aliases = sorted(
                str(row["app_id"]) for row in collisions if row["app_id"] != APP_ID
            )
            if aliases:
                raise RuntimeError(
                    "Cannot use builtin Planner connector while legacy normalized "
                    f"public_mcp_apps aliases exist: {aliases!r}. Remove or recreate "
                    "those custom apps under non-reserved app_ids, then retry."
                )
            return
        if len(collisions) == 1 and collisions[0]["app_id"] == APP_ID:
            raise RuntimeError(
                "Cannot seed builtin Planner connector: an existing "
                "public_mcp_apps row with app_id='planner' has no matching "
                "builtin_provenance"
            )
        raise RuntimeError(
            "Cannot seed builtin Planner connector: public_mcp_apps contains "
            "an app_id that conflicts with the normalized builtin identity "
            f"'planner': {sorted(row['app_id'] for row in collisions)!r}"
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
    if not {"app_id", "launch_config"}.issubset(columns):
        return
    # Only the catalog entry is removed. The shared "microsoft" oauth_providers
    # row is left untouched since it is reused by Outlook/Teams/OneDrive. Any
    # MCPServer/UserMCPServer rows created by users who already connected are
    # intentionally left in place -- connect-driven rows are not owned by
    # this migration and are cleaned up through the normal disconnect path.
    existing = bind.execute(
        sa.select(PUBLIC_MCP_APPS_TABLE.c.launch_config).where(
            PUBLIC_MCP_APPS_TABLE.c.app_id == APP_ID
        )
    ).scalar_one_or_none()
    if not _has_provenance(existing):
        return
    aliases = sorted(
        str(row["app_id"])
        for row in _identity_collisions(bind)
        if row["app_id"] != APP_ID
    )
    if aliases:
        raise RuntimeError(
            "Cannot safely downgrade builtin Planner while legacy normalized "
            f"public_mcp_apps aliases exist: {aliases!r}. Remove or recreate those "
            "custom apps under non-reserved app_ids, then retry the downgrade."
        )
    bind.execute(
        sa.delete(PUBLIC_MCP_APPS_TABLE).where(PUBLIC_MCP_APPS_TABLE.c.app_id == APP_ID)
    )
