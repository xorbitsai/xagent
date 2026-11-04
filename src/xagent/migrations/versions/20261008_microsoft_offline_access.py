"""add offline_access to the builtin Microsoft connectors

Revision ID: 20261008_microsoft_offline_access
Revises: 20261008_task_auto_recovery
Create Date: 2026-10-08 00:00:00.000000

"""

import json
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "20261008_microsoft_offline_access"
down_revision: Union[str, None] = "20261008_task_auto_recovery"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


PUBLIC_MCP_APPS_TABLE = sa.table(
    "public_mcp_apps",
    sa.column("app_id", sa.String),
    sa.column("provider_name", sa.String),
    sa.column("oauth_scopes", sa.JSON),
)

PROVIDER_NAME = "microsoft"
OFFLINE_ACCESS_SCOPE = "offline_access"

# Excel already requests offline_access (20260917_seed_excel_mcp_app) and is
# deliberately absent here.
PREVIOUS_SCOPES: dict[str, tuple[str, ...]] = {
    "teams": (
        "Team.ReadBasic.All",
        "Channel.ReadBasic.All",
        "TeamMember.Read.All",
        "ChannelMessage.Read.All",
        "ChannelMessage.Send",
        "Chat.ReadWrite",
    ),
    "outlook": ("Mail.Read", "Mail.Send", "Calendars.ReadWrite", "Contacts.Read"),
    "onedrive": ("Files.ReadWrite",),
    "planner": ("Tasks.ReadWrite",),
    "powerpoint": ("Files.ReadWrite",),
    "sharepoint": ("Sites.ReadWrite.All",),
    "word": ("Files.ReadWrite.All",),
}
CURRENT_SCOPES: dict[str, tuple[str, ...]] = {
    app_id: (*scopes, OFFLINE_ACCESS_SCOPE)
    for app_id, scopes in PREVIOUS_SCOPES.items()
}


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


def _set_scopes_as_sql(scopes_by_app_id: dict[str, tuple[str, ...]]) -> None:
    dialect_name = op.get_context().dialect.name
    for app_id, scopes in scopes_by_app_id.items():
        op.execute(
            sa.update(PUBLIC_MCP_APPS_TABLE)
            .where(
                PUBLIC_MCP_APPS_TABLE.c.app_id == op.inline_literal(app_id),
                PUBLIC_MCP_APPS_TABLE.c.provider_name
                == op.inline_literal(PROVIDER_NAME),
            )
            .values(oauth_scopes=_json_sql_literal(scopes, dialect_name))
        )


def _set_scopes(scopes_by_app_id: dict[str, tuple[str, ...]]) -> None:
    """Keep persisted rows in sync with the code registry's canonical value.

    This does NOT drive what is requested at OAuth-authorize time: for a
    builtin app, _app_to_dict sources oauth_scopes from
    get_builtin_execution_fields (the code registry), never from this DB row.
    The write exists so validate_builtin_public_mcp_apps doesn't report drift.

    Existing grants are left alone. One issued without offline_access has no
    refresh_token, so it already stops counting as connected once its access
    token expires and the connector UI prompts a reconnect; clearing it here
    would only cut that short, and would also log out any grant that did get
    a refresh_token.

    Rows are matched on provider_name as well as app_id, so a custom app
    squatting one of these app_ids under another provider is not rewritten.
    """
    if op.get_context().as_sql:
        _set_scopes_as_sql(scopes_by_app_id)
        return

    bind = op.get_bind()
    if not _columns_present(bind):
        return

    for app_id, scopes in scopes_by_app_id.items():
        bind.execute(
            sa.update(PUBLIC_MCP_APPS_TABLE)
            .where(
                PUBLIC_MCP_APPS_TABLE.c.app_id == app_id,
                PUBLIC_MCP_APPS_TABLE.c.provider_name == PROVIDER_NAME,
            )
            .values(oauth_scopes=list(scopes))
        )


def upgrade() -> None:
    _set_scopes(CURRENT_SCOPES)


def downgrade() -> None:
    _set_scopes(PREVIOUS_SCOPES)
