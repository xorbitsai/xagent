"""seed built-in Planner (OAuth) MCP connector

Revision ID: 20260917_seed_planner_mcp_app
Revises: 20260919_task_input_receipts
Create Date: 2026-09-17 00:00:00.000000

"""

import logging
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

logger = logging.getLogger(__name__)

# revision identifiers, used by Alembic.
revision: str = "20260917_seed_planner_mcp_app"
down_revision: Union[str, None] = "20260919_task_input_receipts"
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

ROW = {
    "app_id": APP_ID,
    "name": "Planner",
    "description": "Connect to Microsoft Planner to manage plans, buckets, and tasks, including checklists and assignments.",
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
    # Only delete a row that still looks like the one this migration seeded
    # (matched on name/description/transport, not just app_id): an operator
    # could have hand-created a custom app under this same free-form app_id
    # before this migration ever ran (upgrade() no-ops on that collision
    # rather than overwriting it), and an unconditional delete-by-app_id here
    # would then destroy that unrelated row on a later rollback.
    #
    # All three snapshot columns must exist before matching on any of them:
    # partial matching (e.g. app_id alone) is exactly the coincidence this
    # guard exists to rule out, and sa.delete(...).where() with zero
    # conditions compiles to an unconditional DELETE FROM public_mcp_apps.
    if not {"name", "description", "transport"}.issubset(columns):
        return
    # Only the catalog entry is removed. The shared "microsoft" oauth_providers
    # row is left untouched since it is reused by Outlook/Teams/OneDrive. Any
    # MCPServer/UserMCPServer rows created by users who already connected are
    # intentionally left in place -- connect-driven rows are not owned by
    # this migration and are cleaned up through the normal disconnect path.
    bind.execute(
        sa.delete(PUBLIC_MCP_APPS_TABLE)
        .where(PUBLIC_MCP_APPS_TABLE.c.app_id == APP_ID)
        .where(PUBLIC_MCP_APPS_TABLE.c.name == ROW["name"])
        .where(PUBLIC_MCP_APPS_TABLE.c.description == ROW["description"])
        .where(PUBLIC_MCP_APPS_TABLE.c.transport == ROW["transport"])
    )
