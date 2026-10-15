"""sync the persisted Xero MCP launcher and OAuth scopes

The in-repo Xero server replaces the npm launcher and exposes report tools
that require four additional read scopes.  Built-in connector execution uses
the persisted ``public_mcp_apps`` row, so changing only the runtime registry
leaves existing databases on the old launcher.  Existing OAuth grants also do
not acquire newly requested scopes automatically; invalidate them so the UI
prompts affected users to reconnect instead of failing report calls at run
time.

Revision ID: 20261015_sync_xero_mcp_app
Revises: 20261014_task_cmd_claimable_kind
Create Date: 2026-10-15

"""

import json
import logging
from typing import Any, Sequence, Union

import sqlalchemy as sa
from alembic import op

logger = logging.getLogger(__name__)

revision: str = "20261015_sync_xero_mcp_app"
down_revision: Union[str, None] = "20261014_task_cmd_claimable_kind"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


PUBLIC_MCP_APPS_TABLE = sa.table(
    "public_mcp_apps",
    sa.column("app_id", sa.String),
    sa.column("oauth_scopes", sa.JSON),
    sa.column("launch_config", sa.JSON),
)
USER_OAUTH_TABLE = sa.table(
    "user_oauth",
    sa.column("provider", sa.String),
    sa.column("access_token", sa.String),
    sa.column("refresh_token", sa.String),
)

APP_ID = "xero"
PREVIOUS_SCOPES = [
    "openid",
    "profile",
    "email",
    "accounting.contacts",
    "accounting.settings",
    "accounting.invoices",
    "accounting.payments",
    "accounting.banktransactions",
    "accounting.manualjournals",
    "offline_access",
]
CURRENT_SCOPES = [
    "openid",
    "profile",
    "email",
    "accounting.contacts",
    "accounting.settings",
    "accounting.invoices",
    "accounting.payments",
    "accounting.banktransactions",
    "accounting.manualjournals",
    "accounting.reports.aged.read",
    "accounting.reports.profitandloss.read",
    "accounting.reports.balancesheet.read",
    "accounting.reports.trialbalance.read",
    "offline_access",
]
PREVIOUS_LAUNCH_CONFIG = {
    "command": "npx",
    "args": ["-y", "@xeroapi/xero-mcp-server@latest"],
    "env_mapping": {"XERO_CLIENT_BEARER_TOKEN": "access_token"},
}
CURRENT_LAUNCH_CONFIG = {
    "command": "python",
    "args": ["-m", "xagent.web.tools.mcp.xero"],
    "env_mapping": {"XERO_CLIENT_BEARER_TOKEN": "access_token"},
}


def _columns_present(
    bind: sa.engine.Connection, table_name: str, required_columns: set[str]
) -> bool:
    inspector = sa.inspect(bind)
    if table_name not in set(inspector.get_table_names()):
        return False
    columns = {column["name"] for column in inspector.get_columns(table_name)}
    return required_columns.issubset(columns)


def _offline_json_literal(value: Any) -> object:
    serialized = op.inline_literal(json.dumps(value))
    if op.get_context().dialect.name == "postgresql":
        return sa.cast(serialized, sa.JSON())
    return serialized


def _set_xero_app_offline(scopes: list[str], launch_config: dict[str, Any]) -> None:
    op.execute(
        sa.update(PUBLIC_MCP_APPS_TABLE)
        .where(PUBLIC_MCP_APPS_TABLE.c.app_id == op.inline_literal(APP_ID))
        .values(
            oauth_scopes=_offline_json_literal(scopes),
            launch_config=_offline_json_literal(launch_config),
        )
    )


def _set_xero_app(
    bind: sa.engine.Connection, scopes: list[str], launch_config: dict[str, Any]
) -> None:
    required = {"app_id", "oauth_scopes", "launch_config"}
    if not _columns_present(bind, "public_mcp_apps", required):
        return
    bind.execute(
        sa.update(PUBLIC_MCP_APPS_TABLE)
        .where(PUBLIC_MCP_APPS_TABLE.c.app_id == APP_ID)
        .values(oauth_scopes=scopes, launch_config=launch_config)
    )


def _invalidate_existing_xero_grants(bind: sa.engine.Connection) -> None:
    if not _columns_present(bind, "user_oauth", {"provider", "access_token"}):
        return

    values: dict[str, object] = {"access_token": ""}
    if _columns_present(bind, "user_oauth", {"refresh_token"}):
        values["refresh_token"] = None
    result = bind.execute(
        sa.update(USER_OAUTH_TABLE)
        .where(USER_OAUTH_TABLE.c.provider == APP_ID)
        .values(**values)
    )
    if result.rowcount:
        logger.warning(
            "Disconnected %d existing Xero grant(s) so users can approve the "
            "new accounting report scopes.",
            result.rowcount,
        )


def _invalidate_existing_xero_grants_offline() -> None:
    op.execute(
        sa.update(USER_OAUTH_TABLE)
        .where(USER_OAUTH_TABLE.c.provider == op.inline_literal(APP_ID))
        .values(access_token=op.inline_literal(""), refresh_token=None)
    )


def upgrade() -> None:
    if op.get_context().as_sql:
        _set_xero_app_offline(CURRENT_SCOPES, CURRENT_LAUNCH_CONFIG)
        _invalidate_existing_xero_grants_offline()
        return

    bind = op.get_bind()
    _set_xero_app(bind, CURRENT_SCOPES, CURRENT_LAUNCH_CONFIG)
    _invalidate_existing_xero_grants(bind)


def downgrade() -> None:
    # Invalidated tokens cannot be restored; users who reconnect retain a
    # valid Xero grant even if the application code is rolled back.
    if op.get_context().as_sql:
        _set_xero_app_offline(PREVIOUS_SCOPES, PREVIOUS_LAUNCH_CONFIG)
        return

    _set_xero_app(op.get_bind(), PREVIOUS_SCOPES, PREVIOUS_LAUNCH_CONFIG)
