"""update Google Drive description and category for the sharing tools

Revision ID: 20260908_update_google_drive_description_category
Revises: 20260904_add_auto_model_config
Create Date: 2026-09-08

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "20260908_update_google_drive_description_category"
down_revision: Union[str, None] = "20260904_add_auto_model_config"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


PUBLIC_MCP_APPS_TABLE = sa.table(
    "public_mcp_apps",
    sa.column("app_id", sa.String),
    sa.column("description", sa.Text),
    sa.column("category", sa.String),
)

APP_ID = "google-drive"

PREVIOUS_DESCRIPTION = (
    "Access Google Drive to search for files, read documents, and manage "
    "your cloud storage."
)
CURRENT_DESCRIPTION = (
    "Access Google Drive to search for files, read documents, manage your "
    "cloud storage, and share files or folders with others."
)

PREVIOUS_CATEGORY = "Support"
CURRENT_CATEGORY = "Storage"


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


def _set_field_if_unchanged(
    bind: sa.engine.Connection, column: str, expected_current: str, new_value: str
) -> None:
    """Refresh a stale registry field on an already-seeded row, without
    clobbering a customization.

    Neither ``description`` nor ``category`` is in admin_mcp's
    _BUILTIN_PROTECTED_FIELDS, so an operator can legitimately have edited
    either via the admin PATCH endpoint. Only overwrite when the persisted
    value still equals the last-known canonical value (i.e. it was never
    customized); an edited value matches neither the previous nor the
    current canonical value and is left alone in either direction.

    Both fields are also excluded from builtin_mcp_registry.py's own
    _BUILTIN_EXECUTION_FIELD_NAMES drift sync (unlike oauth_scopes/
    launch_config/etc., which self-heal on every read), and the seed
    migration only ever inserts a row once -- so without this migration, an
    already-provisioned install's google-drive row would keep showing
    "Support"/the pre-sharing description forever, even after upgrading to
    a build whose source registry has moved on.
    """
    if not _columns_present(bind, "public_mcp_apps", {"app_id", column}):
        return

    bind.execute(
        sa.update(PUBLIC_MCP_APPS_TABLE)
        .where(
            PUBLIC_MCP_APPS_TABLE.c.app_id == APP_ID,
            getattr(PUBLIC_MCP_APPS_TABLE.c, column) == expected_current,
        )
        .values(**{column: new_value})
    )


def upgrade() -> None:
    bind = op.get_bind()
    _set_field_if_unchanged(
        bind, "description", PREVIOUS_DESCRIPTION, CURRENT_DESCRIPTION
    )
    _set_field_if_unchanged(bind, "category", PREVIOUS_CATEGORY, CURRENT_CATEGORY)


def downgrade() -> None:
    bind = op.get_bind()
    _set_field_if_unchanged(
        bind, "description", CURRENT_DESCRIPTION, PREVIOUS_DESCRIPTION
    )
    _set_field_if_unchanged(bind, "category", CURRENT_CATEGORY, PREVIOUS_CATEGORY)
