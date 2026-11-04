"""add the Zendesk side_conversations:write scope

Revision ID: 20261102_zendesk_side_conv
Revises: 20261031_speech_settings
Create Date: 2026-11-02

"""

import json
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "20261102_zendesk_side_conv"
down_revision: Union[str, None] = "20261031_speech_settings"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


PUBLIC_MCP_APPS_TABLE = sa.table(
    "public_mcp_apps",
    sa.column("app_id", sa.String),
    sa.column("provider_name", sa.String),
    sa.column("oauth_scopes", sa.JSON),
)

APP_ID = "zendesk"
PROVIDER_NAME = "zendesk"
PREVIOUS_SCOPES = ("read", "tickets:write")
CURRENT_SCOPES = (*PREVIOUS_SCOPES, "side_conversations:write")


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
    """Keep the persisted built-in row aligned with the code registry.

    The code registry drives authorize-time scopes. Existing grants are left
    intact; users must reconnect before a new access token can contain the
    added side conversation permission.
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
