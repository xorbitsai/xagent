"""show the Freshdesk connector in the catalog

Revision ID: 20261028_enable_freshdesk
Revises: 20261029_update_chartmogul_description
Create Date: 2026-10-28 00:00:00

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20261028_enable_freshdesk"
down_revision: str | None = "20261029_update_chartmogul_description"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


PUBLIC_MCP_APPS_TABLE = sa.table(
    "public_mcp_apps",
    sa.column("app_id", sa.String),
    sa.column("is_visible_in_connector", sa.Boolean),
)

APP_ID = "freshdesk"


def _columns_present(
    bind: sa.engine.Connection, table_name: str, required_columns: set[str]
) -> bool:
    inspector = sa.inspect(bind)
    if table_name not in set(inspector.get_table_names()):
        return False
    columns = {column["name"] for column in inspector.get_columns(table_name)}
    return required_columns.issubset(columns)


def _set_visibility(bind: sa.engine.Connection, visible: bool) -> None:
    if not _columns_present(
        bind, "public_mcp_apps", {"app_id", "is_visible_in_connector"}
    ):
        return
    bind.execute(
        sa.update(PUBLIC_MCP_APPS_TABLE)
        .where(PUBLIC_MCP_APPS_TABLE.c.app_id == APP_ID)
        .values(is_visible_in_connector=visible)
    )


def _set_visibility_offline(visible: bool) -> None:
    op.execute(
        sa.update(PUBLIC_MCP_APPS_TABLE)
        .where(PUBLIC_MCP_APPS_TABLE.c.app_id == op.inline_literal(APP_ID))
        .values(is_visible_in_connector=op.inline_literal(visible))
    )


def upgrade() -> None:
    if op.get_context().as_sql:
        _set_visibility_offline(True)
        return

    _set_visibility(op.get_bind(), True)


def downgrade() -> None:
    # Intentionally not reversed, online or offline. Hiding the row
    # unconditionally would make it match the 20260923_seed_freshdesk_mcp_app
    # snapshot again, so a full downgrade would then let that seed's
    # downgrade delete a freshdesk row an operator had made visible before
    # this migration. The code being downgraded to reads visibility from the
    # database as well, so nothing needs the flag reset. As a result, after
    # upgrading through this migration, a full downgrade leaves the row
    # visible and the seed's downgrade keeps it.
    pass
