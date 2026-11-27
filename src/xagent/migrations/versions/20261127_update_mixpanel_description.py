"""state the Mixpanel role that user profile queries need

Revision ID: 20261127_mixpanel_description
Revises: 20261126_preview_lookup_indexes
Create Date: 2026-11-27

Mixpanel's Engage (user profile) endpoint returns 403 unless the service
account's project role can export user profiles. The connector description
now says which roles can. Like 20260908_update_google_drive_description,
only a row that still has the previous built-in description is changed, so
an admin-edited description is kept.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "20261127_mixpanel_description"
down_revision: Union[str, None] = "20261126_preview_lookup_indexes"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


PUBLIC_MCP_APPS_TABLE = sa.table(
    "public_mcp_apps",
    sa.column("app_id", sa.String),
    sa.column("description", sa.Text),
)

APP_ID = "mixpanel"

PREVIOUS_DESCRIPTION = (
    "Connect to Mixpanel with a Service Account to query events, "
    "segmentation, retention, funnels, and user profiles, and export raw "
    "event data. Supports US, EU, and India data residency."
)
CURRENT_DESCRIPTION = (
    "Connect to Mixpanel with a Service Account to query events, "
    "segmentation, retention, funnels, and user profiles, and export raw "
    "event data. User profile queries need the Analyst, Admin, or Owner "
    "project role. Supports US, EU, and India data residency."
)


def _columns_present(bind: sa.engine.Connection) -> bool:
    inspector = sa.inspect(bind)
    if "public_mcp_apps" not in set(inspector.get_table_names()):
        return False
    columns = {c["name"] for c in inspector.get_columns("public_mcp_apps")}
    return {"app_id", "description"}.issubset(columns)


def _set_description_if_unchanged(
    bind: sa.engine.Connection, expected_current: str, new_value: str
) -> None:
    if not _columns_present(bind):
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
