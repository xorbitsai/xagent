"""update ChartMogul connector description for metrics tools

Revision ID: 20261029_update_chartmogul_description
Revises: 20261029_google_oauth_grants
Create Date: 2026-10-29

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "20261029_update_chartmogul_description"
down_revision: Union[str, None] = "20261029_google_oauth_grants"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


PUBLIC_MCP_APPS_TABLE = sa.table(
    "public_mcp_apps",
    sa.column("app_id", sa.String),
    sa.column("description", sa.Text),
)

APP_ID = "chartmogul"
PREVIOUS_DESCRIPTION = (
    "Connect your ChartMogul account (with a per-user API key from Profile -> "
    "API keys) to look up and manage customers, contacts, and sales opportunities."
)
CURRENT_DESCRIPTION = (
    "Connect your ChartMogul account (with a per-user API key from Profile -> "
    "API keys) to manage customers, contacts, and sales opportunities, and "
    "analyze MRR, ARR, churn, LTV, and other subscription metrics."
)


def _required_columns_present(bind: sa.engine.Connection) -> bool:
    inspector = sa.inspect(bind)
    if "public_mcp_apps" not in set(inspector.get_table_names()):
        return False
    columns = {column["name"] for column in inspector.get_columns("public_mcp_apps")}
    return {"app_id", "description"}.issubset(columns)


def _set_description_if_unchanged(
    bind: sa.engine.Connection, expected_current: str, new_value: str
) -> None:
    """Update the seeded description without overwriting admin customizations."""
    if not _required_columns_present(bind):
        return

    bind.execute(
        sa.update(PUBLIC_MCP_APPS_TABLE)
        .where(
            PUBLIC_MCP_APPS_TABLE.c.app_id == APP_ID,
            PUBLIC_MCP_APPS_TABLE.c.description == expected_current,
        )
        .values(description=new_value)
    )


def upgrade() -> None:
    _set_description_if_unchanged(
        op.get_bind(), PREVIOUS_DESCRIPTION, CURRENT_DESCRIPTION
    )


def downgrade() -> None:
    _set_description_if_unchanged(
        op.get_bind(), CURRENT_DESCRIPTION, PREVIOUS_DESCRIPTION
    )
