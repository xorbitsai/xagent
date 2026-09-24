"""hide Google Drive until drive.file authorization has a Picker flow

Revision ID: 20260924_hide_google_drive_until_picker
Revises: 20260924_narrow_google_oauth_scopes
Create Date: 2026-09-24 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "20260924_hide_google_drive_until_picker"
down_revision: Union[str, None] = "20260924_narrow_google_oauth_scopes"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


PUBLIC_MCP_APPS_TABLE = sa.table(
    "public_mcp_apps",
    sa.column("app_id", sa.String),
    sa.column("description", sa.Text),
    sa.column("is_visible_in_connector", sa.Boolean),
)

APP_ID = "google-drive"
PREVIOUS_DESCRIPTION = (
    "Access Google Drive to search for files, read documents, manage your "
    "cloud storage, and manage sharing on files or folders -- including "
    "granting access to someone new and revoking an existing collaborator's "
    "access."
)
CURRENT_DESCRIPTION = (
    "Access Google Drive files selected through Google's file picker or "
    "created by Xagent, including reading documents and managing those files."
)


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


def _set_description_if_unchanged(
    bind: sa.engine.Connection, expected_current: str, new_value: str
) -> None:
    # Preserve an administrator's custom description while converging the
    # canonical row seeded by the builtin registry.
    if not _columns_present(bind, "public_mcp_apps", {"app_id", "description"}):
        return
    bind.execute(
        sa.update(PUBLIC_MCP_APPS_TABLE)
        .where(
            PUBLIC_MCP_APPS_TABLE.c.app_id == APP_ID,
            PUBLIC_MCP_APPS_TABLE.c.description == expected_current,
        )
        .values(description=new_value)
    )


def _offline_update_visibility(visible: bool) -> None:
    op.execute(
        sa.update(PUBLIC_MCP_APPS_TABLE)
        .where(PUBLIC_MCP_APPS_TABLE.c.app_id == op.inline_literal(APP_ID))
        .values(is_visible_in_connector=op.inline_literal(visible))
    )


def _offline_update_description(expected_current: str, new_value: str) -> None:
    op.execute(
        sa.update(PUBLIC_MCP_APPS_TABLE)
        .where(
            PUBLIC_MCP_APPS_TABLE.c.app_id == op.inline_literal(APP_ID),
            PUBLIC_MCP_APPS_TABLE.c.description == op.inline_literal(expected_current),
        )
        .values(description=op.inline_literal(new_value))
    )


def upgrade() -> None:
    if op.get_context().as_sql:
        _offline_update_visibility(False)
        _offline_update_description(PREVIOUS_DESCRIPTION, CURRENT_DESCRIPTION)
        return

    bind = op.get_bind()
    _set_visibility(bind, False)
    _set_description_if_unchanged(bind, PREVIOUS_DESCRIPTION, CURRENT_DESCRIPTION)


def downgrade() -> None:
    if op.get_context().as_sql:
        _offline_update_visibility(True)
        _offline_update_description(CURRENT_DESCRIPTION, PREVIOUS_DESCRIPTION)
        return

    bind = op.get_bind()
    _set_visibility(bind, True)
    _set_description_if_unchanged(bind, CURRENT_DESCRIPTION, PREVIOUS_DESCRIPTION)
