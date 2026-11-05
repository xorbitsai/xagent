"""position the Atlassian connector as Confluence + Bitbucket, not Jira

Revision ID: 20261105_update_atlassian_name_and_description
Revises: 20261031_task_sdk_credential_kind
Create Date: 2026-11-05

"""

from typing import Literal, Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "20261105_update_atlassian_name_and_description"
down_revision: Union[str, None] = "20261031_task_sdk_credential_kind"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


PUBLIC_MCP_APPS_TABLE = sa.table(
    "public_mcp_apps",
    sa.column("app_id", sa.String),
    sa.column("name", sa.String),
    sa.column("description", sa.Text),
)

APP_ID = "atlassian"
# The values 20260920_seed_atlassian_mcp_app.py seeded.
PREVIOUS_NAME = "Atlassian (Jira, Confluence, Bitbucket)"
PREVIOUS_DESCRIPTION = (
    "Connect to Atlassian to search and work with Jira issues, Confluence "
    "pages and Bitbucket repositories through Atlassian's hosted MCP server."
)
CURRENT_NAME = "Atlassian (Confluence, Bitbucket)"
CURRENT_DESCRIPTION = (
    "Connect to Atlassian to search and work with Confluence pages and "
    "Bitbucket repositories through Atlassian's hosted MCP server. For Jira, "
    "use the Jira connector."
)


CopyColumn = Literal["name", "description"]


def _column_present(bind: sa.engine.Connection, column_name: CopyColumn) -> bool:
    inspector = sa.inspect(bind)
    if "public_mcp_apps" not in set(inspector.get_table_names()):
        return False
    columns = {column["name"] for column in inspector.get_columns("public_mcp_apps")}
    return {"app_id", column_name}.issubset(columns)


def _set_if_unchanged(
    bind: sa.engine.Connection,
    column_name: CopyColumn,
    expected_current: str,
    new_value: str,
) -> None:
    """Update one seeded column only while it still holds the seeded value.

    name and description are updated independently: description is
    admin-editable, while name is code-owned (protected from admin PATCH and
    checked by validate_builtin_public_mcp_apps), so a customized description
    must not keep the name from following the registry.
    """
    if not _column_present(bind, column_name):
        return

    column = PUBLIC_MCP_APPS_TABLE.c[column_name]
    bind.execute(
        sa.update(PUBLIC_MCP_APPS_TABLE)
        .where(
            PUBLIC_MCP_APPS_TABLE.c.app_id == APP_ID,
            column == expected_current,
        )
        .values({column_name: new_value})
    )


def upgrade() -> None:
    bind = op.get_bind()
    _set_if_unchanged(bind, "name", PREVIOUS_NAME, CURRENT_NAME)
    _set_if_unchanged(bind, "description", PREVIOUS_DESCRIPTION, CURRENT_DESCRIPTION)


def downgrade() -> None:
    bind = op.get_bind()
    _set_if_unchanged(bind, "description", CURRENT_DESCRIPTION, PREVIOUS_DESCRIPTION)
    _set_if_unchanged(bind, "name", CURRENT_NAME, PREVIOUS_NAME)
