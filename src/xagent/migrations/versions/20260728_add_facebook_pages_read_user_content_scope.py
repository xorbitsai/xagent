"""add pages_read_user_content scope to the Facebook connector

Revision ID: 20260728_add_facebook_pages_read_user_content_scope
Revises: 20260725_add_uploaded_file_recovery_index
Create Date: 2026-07-28

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "20260728_add_facebook_pages_read_user_content_scope"
down_revision: Union[str, None] = "20260725_add_uploaded_file_recovery_index"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


PUBLIC_MCP_APPS_TABLE = sa.table(
    "public_mcp_apps",
    sa.column("app_id", sa.String),
    sa.column("oauth_scopes", sa.JSON),
)

APP_ID = "facebook"
NEW_SCOPE = "pages_read_user_content"
PREVIOUS_SCOPES = ["pages_show_list", "pages_read_engagement", "pages_manage_posts"]
CURRENT_SCOPES = [*PREVIOUS_SCOPES, NEW_SCOPE]


def _set_scopes(bind: sa.engine.Connection, scopes: list[str]) -> None:
    inspector = sa.inspect(bind)
    if "public_mcp_apps" not in set(inspector.get_table_names()):
        return

    columns = {c["name"] for c in inspector.get_columns("public_mcp_apps")}
    if not {"app_id", "oauth_scopes"}.issubset(columns):
        return

    bind.execute(
        sa.update(PUBLIC_MCP_APPS_TABLE)
        .where(PUBLIC_MCP_APPS_TABLE.c.app_id == APP_ID)
        .values(oauth_scopes=scopes)
    )


def upgrade() -> None:
    _set_scopes(op.get_bind(), CURRENT_SCOPES)


def downgrade() -> None:
    _set_scopes(op.get_bind(), PREVIOUS_SCOPES)
