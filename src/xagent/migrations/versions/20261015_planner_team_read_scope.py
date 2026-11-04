"""add joined-team discovery scope to the builtin Planner connector

Revision ID: 20261015_planner_team_read_scope
Revises: 20261015_task_cmd_claimable_id
Create Date: 2026-10-15 00:00:00.000000

"""

import json
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "20261015_planner_team_read_scope"
down_revision: Union[str, None] = "20261015_task_cmd_claimable_id"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


PUBLIC_MCP_APPS_TABLE = sa.table(
    "public_mcp_apps",
    sa.column("app_id", sa.String),
    sa.column("provider_name", sa.String),
    sa.column("oauth_scopes", sa.JSON),
)

APP_ID = "planner"
PROVIDER_NAME = "microsoft"
PREVIOUS_SCOPES = ("Tasks.ReadWrite", "offline_access")
CURRENT_SCOPES = ("Tasks.ReadWrite", "Team.ReadBasic.All", "offline_access")


def _columns_present(bind: sa.engine.Connection) -> bool:
    inspector = sa.inspect(bind)
    if "public_mcp_apps" not in set(inspector.get_table_names()):
        return False
    columns = {column["name"] for column in inspector.get_columns("public_mcp_apps")}
    return {"app_id", "provider_name", "oauth_scopes"}.issubset(columns)


def _json_sql_literal(values: Sequence[str], dialect_name: str) -> object:
    serialized_literal = op.inline_literal(json.dumps(list(values)))
    if dialect_name == "postgresql":
        return sa.cast(serialized_literal, sa.JSON())
    return serialized_literal


def _set_scopes_as_sql(scopes: Sequence[str]) -> None:
    op.execute(
        sa.update(PUBLIC_MCP_APPS_TABLE)
        .where(
            PUBLIC_MCP_APPS_TABLE.c.app_id == op.inline_literal(APP_ID),
            PUBLIC_MCP_APPS_TABLE.c.provider_name == op.inline_literal(PROVIDER_NAME),
        )
        .values(oauth_scopes=_json_sql_literal(scopes, op.get_context().dialect.name))
    )


def _set_scopes(scopes: Sequence[str]) -> None:
    """Keep the persisted builtin row aligned with the code registry.

    The code registry drives authorize-time scopes; this persisted copy keeps
    catalog drift checks consistent. Existing grants are deliberately left
    working. Grants issued before this migration receive an actionable
    reconnect error only when they use the new joined-team discovery tool.
    """
    if op.get_context().as_sql:
        _set_scopes_as_sql(scopes)
        return

    bind = op.get_bind()
    if not _columns_present(bind):
        return
    bind.execute(
        sa.update(PUBLIC_MCP_APPS_TABLE)
        .where(
            PUBLIC_MCP_APPS_TABLE.c.app_id == APP_ID,
            PUBLIC_MCP_APPS_TABLE.c.provider_name == PROVIDER_NAME,
        )
        .values(oauth_scopes=list(scopes))
    )


def upgrade() -> None:
    _set_scopes(CURRENT_SCOPES)


def downgrade() -> None:
    _set_scopes(PREVIOUS_SCOPES)
