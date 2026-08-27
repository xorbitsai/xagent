"""narrow the Google Calendar connector's OAuth scope to calendar.events

The Calendar connector's tools (search/create/get/update/delete) only ever
operate on events, never on calendar list management, so the full
``.../auth/calendar`` scope requested more access than the feature set uses.
This updates the persisted built-in catalog row to match the narrower scope
now declared in ``builtin_mcp_registry.py``.

Revision ID: 20260817_narrow_google_calendar_scope
Revises: 20260825_add_slack_channels_join_scope
Create Date: 2026-08-17

"""

import json
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "20260817_narrow_google_calendar_scope"
down_revision: Union[str, None] = "20260825_add_slack_channels_join_scope"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


PUBLIC_MCP_APPS_TABLE = sa.table(
    "public_mcp_apps",
    sa.column("app_id", sa.String),
    sa.column("oauth_scopes", sa.JSON),
)

APP_ID = "google-calendar"
OLD_SCOPES = ["https://www.googleapis.com/auth/calendar"]
NEW_SCOPES = ["https://www.googleapis.com/auth/calendar.events"]


def _offline_scopes_literal(scopes: list[str], dialect_name: str):
    # Match the online sa.JSON binding contract (none_as_null=False) used
    # elsewhere in this migration set: values are stored as JSON, not as a
    # bare SQL string literal, on every supported dialect.
    serialized_literal = op.inline_literal(json.dumps(scopes, sort_keys=True))
    if dialect_name == "postgresql":
        return sa.cast(serialized_literal, sa.JSON())
    return serialized_literal


def _upgrade_offline() -> None:
    dialect_name = op.get_context().dialect.name
    statement = (
        sa.update(PUBLIC_MCP_APPS_TABLE)
        .where(PUBLIC_MCP_APPS_TABLE.c.app_id == op.inline_literal(APP_ID))
        .values(oauth_scopes=_offline_scopes_literal(NEW_SCOPES, dialect_name))
    )
    op.execute(statement)


def upgrade() -> None:
    if op.get_context().as_sql:
        _upgrade_offline()
        return

    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "public_mcp_apps" not in inspector.get_table_names():
        return

    columns = {column["name"] for column in inspector.get_columns("public_mcp_apps")}
    if not {"app_id", "oauth_scopes"}.issubset(columns):
        return

    bind.execute(
        sa.update(PUBLIC_MCP_APPS_TABLE)
        .where(PUBLIC_MCP_APPS_TABLE.c.app_id == APP_ID)
        .values(oauth_scopes=NEW_SCOPES)
    )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "public_mcp_apps" not in inspector.get_table_names():
        return

    columns = {column["name"] for column in inspector.get_columns("public_mcp_apps")}
    if not {"app_id", "oauth_scopes"}.issubset(columns):
        return

    bind.execute(
        sa.update(PUBLIC_MCP_APPS_TABLE)
        .where(PUBLIC_MCP_APPS_TABLE.c.app_id == APP_ID)
        .values(oauth_scopes=OLD_SCOPES)
    )
