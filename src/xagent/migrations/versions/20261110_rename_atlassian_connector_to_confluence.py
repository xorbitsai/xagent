"""rename the Atlassian connector display name to Confluence

Revision ID: 20261110_rename_atlassian_to_confluence
Revises: 20261109_seed_jira_ticket_mcp_app
Create Date: 2026-11-10

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "20261110_rename_atlassian_to_confluence"
down_revision: Union[str, None] = "20261109_seed_jira_ticket_mcp_app"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


PUBLIC_MCP_APPS_TABLE = sa.table(
    "public_mcp_apps",
    sa.column("app_id", sa.String),
    sa.column("name", sa.String),
)

APP_ID = "atlassian"
# The value 20261105_update_atlassian_name_and_description.py left in place.
PREVIOUS_NAME = "Atlassian (Confluence, Bitbucket)"
CURRENT_NAME = "Confluence"


def _name_column_present(bind: sa.engine.Connection) -> bool:
    inspector = sa.inspect(bind)
    if "public_mcp_apps" not in set(inspector.get_table_names()):
        return False
    columns = {column["name"] for column in inspector.get_columns("public_mcp_apps")}
    return {"app_id", "name"}.issubset(columns)


def _rename_if_unchanged(
    bind: sa.engine.Connection, expected_current: str, new_value: str
) -> None:
    """Rename only while the row still holds the seeded value.

    name is code-owned (protected from admin PATCH and drift-checked by
    validate_builtin_public_mcp_apps), so only an unexpected value is left
    alone. app_id is the stable identity; the display name is not used to
    resolve existing connections.
    """
    if not _name_column_present(bind):
        return
    bind.execute(
        sa.update(PUBLIC_MCP_APPS_TABLE)
        .where(
            PUBLIC_MCP_APPS_TABLE.c.app_id == APP_ID,
            PUBLIC_MCP_APPS_TABLE.c.name == expected_current,
        )
        .values(name=new_value)
    )


def upgrade() -> None:
    _rename_if_unchanged(op.get_bind(), PREVIOUS_NAME, CURRENT_NAME)


def downgrade() -> None:
    _rename_if_unchanged(op.get_bind(), CURRENT_NAME, PREVIOUS_NAME)
