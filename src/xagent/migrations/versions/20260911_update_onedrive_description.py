"""update OneDrive description for the new upload tool

Revision ID: 20260911_update_onedrive_description
Revises: 20260911_assistant_source_event
Create Date: 2026-09-11

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "20260911_update_onedrive_description"
down_revision: Union[str, None] = "20260911_assistant_source_event"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


PUBLIC_MCP_APPS_TABLE = sa.table(
    "public_mcp_apps",
    sa.column("app_id", sa.String),
    sa.column("description", sa.Text),
)

APP_ID = "onedrive"

PREVIOUS_DESCRIPTION = (
    "Connect to OneDrive to browse files, download content, and manage cloud storage."
)
CURRENT_DESCRIPTION = (
    "Connect to OneDrive to browse files, download content, upload files, "
    "and manage cloud storage."
)


def _columns_present(
    bind: sa.engine.Connection, table_name: str, required_columns: set[str]
) -> bool:
    """Whether ``table_name`` exists and has all of ``required_columns``.

    This migration must be a no-op (not an error) against a database
    mid-way through a schema this old, or an admin's reduced-schema table,
    rather than assume a table shape that matches only the current model.
    """
    inspector = sa.inspect(bind)
    if table_name not in set(inspector.get_table_names()):
        return False
    columns = {c["name"] for c in inspector.get_columns(table_name)}
    return required_columns.issubset(columns)


def _set_description_if_unchanged(
    bind: sa.engine.Connection, expected_current: str, new_value: str
) -> None:
    """Refresh the stale registry description on an already-seeded row,
    without clobbering a customization.

    ``description`` is not in admin_mcp's _BUILTIN_PROTECTED_FIELDS, so an
    operator can legitimately have edited it via the admin PATCH endpoint.
    Only overwrite when the persisted value still equals the last-known
    canonical value (i.e. it was never customized); an edited value matches
    neither the previous nor the current canonical value and is left alone
    in either direction.

    ``description`` is also excluded from builtin_mcp_registry.py's own
    _BUILTIN_EXECUTION_FIELD_NAMES drift sync (unlike oauth_scopes/
    launch_config/etc., which self-heal on every read), and the seed
    migration only ever inserts a row once -- so without this migration, an
    already-provisioned install's onedrive row would keep showing the
    pre-upload description forever, even after upgrading to a build whose
    source registry has moved on (mirrors
    20260908_update_google_drive_description.py's identical rationale).
    """
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


def upgrade() -> None:
    bind = op.get_bind()
    _set_description_if_unchanged(bind, PREVIOUS_DESCRIPTION, CURRENT_DESCRIPTION)


def downgrade() -> None:
    bind = op.get_bind()
    _set_description_if_unchanged(bind, CURRENT_DESCRIPTION, PREVIOUS_DESCRIPTION)
