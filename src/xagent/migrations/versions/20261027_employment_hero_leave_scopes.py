"""add Employment Hero leave scopes and catalog description

Revision ID: 20261027_employment_hero_leave
Revises: 20261027_task_dispatch_priority
Create Date: 2026-10-27

"""

import json
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "20261027_employment_hero_leave"
down_revision: Union[str, None] = "20261027_task_dispatch_priority"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


PUBLIC_MCP_APPS_TABLE = sa.table(
    "public_mcp_apps",
    sa.column("app_id", sa.String),
    sa.column("provider_name", sa.String),
    sa.column("description", sa.Text),
    sa.column("oauth_scopes", sa.JSON),
)

APP_ID = "employment-hero"
PROVIDER_NAME = "employment-hero"
PREVIOUS_DESCRIPTION = (
    "Connect to Employment Hero to look up organisations, employees, teams, "
    "and timesheet entries."
)
CURRENT_DESCRIPTION = (
    "Connect to Employment Hero to look up organisations, employees, teams, "
    "timesheets, leave categories, balances and requests, and to create or "
    "approve leave requests with confirmation."
)
PREVIOUS_SCOPES = (
    "organisations:list",
    "employees:list",
    "employees:show",
    "teams:list",
    "timesheet_entries:list",
)
CURRENT_SCOPES = (
    *PREVIOUS_SCOPES,
    "leave_categories:list",
    "leave_balances:list",
    "leave_requests:list",
    "leave_requests:create",
    "leave_requests:approve",
)


def _columns_present(bind: sa.engine.Connection) -> bool:
    inspector = sa.inspect(bind)
    if "public_mcp_apps" not in set(inspector.get_table_names()):
        return False
    columns = {column["name"] for column in inspector.get_columns("public_mcp_apps")}
    return {
        "app_id",
        "provider_name",
        "description",
        "oauth_scopes",
    }.issubset(columns)


def _json_sql_literal(values: Sequence[str], dialect_name: str) -> object:
    serialized_literal = op.inline_literal(json.dumps(list(values)))
    if dialect_name == "postgresql":
        return sa.cast(serialized_literal, sa.JSON())
    return serialized_literal


def _set_values_as_sql(description: str, scopes: Sequence[str]) -> None:
    op.execute(
        sa.update(PUBLIC_MCP_APPS_TABLE)
        .where(
            PUBLIC_MCP_APPS_TABLE.c.app_id == op.inline_literal(APP_ID),
            PUBLIC_MCP_APPS_TABLE.c.provider_name == op.inline_literal(PROVIDER_NAME),
        )
        .values(
            description=op.inline_literal(description),
            oauth_scopes=_json_sql_literal(scopes, op.get_context().dialect.name),
        )
    )


def _set_values(description: str, scopes: Sequence[str]) -> None:
    """Keep the persisted built-in row aligned with the code registry.

    The code registry drives authorize-time scopes. Existing grants are left
    intact; users must reconnect before a new access token can contain the
    added leave permissions.
    """
    if op.get_context().as_sql:
        _set_values_as_sql(description, scopes)
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
        .values(description=description, oauth_scopes=list(scopes))
    )


def upgrade() -> None:
    _set_values(CURRENT_DESCRIPTION, CURRENT_SCOPES)


def downgrade() -> None:
    _set_values(PREVIOUS_DESCRIPTION, PREVIOUS_SCOPES)
