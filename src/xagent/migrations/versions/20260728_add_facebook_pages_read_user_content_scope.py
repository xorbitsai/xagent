"""add pages_read_user_content scope to the Facebook connector

Revision ID: 20260728_add_facebook_pages_read_user_content_scope
Revises: 20260724_add_upload_source_to_uploaded_files
Create Date: 2026-07-28

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "20260728_add_facebook_pages_read_user_content_scope"
down_revision: Union[str, None] = "20260724_add_upload_source_to_uploaded_files"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


PUBLIC_MCP_APPS_TABLE = sa.table(
    "public_mcp_apps",
    sa.column("app_id", sa.String),
    sa.column("oauth_scopes", sa.JSON),
)

USER_OAUTH_TABLE = sa.table(
    "user_oauth",
    sa.column("provider", sa.String),
    sa.column("access_token", sa.String),
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


def _invalidate_existing_facebook_grants(bind: sa.engine.Connection) -> None:
    """Force reconnection for Facebook grants issued under the old scope set.

    Meta never returns a ``scope`` field from its token endpoint, so a stored
    grant's actual permissions can't be inspected after the fact — there is no
    way to tell whether a given row already has ``pages_read_user_content``.
    Every row was necessarily authorized before this scope existed, so the
    access token is cleared; ``_oauth_account_can_connect`` treats a falsy
    token as disconnected, and the connector UI prompts the user to
    reconnect and grant the new permission.
    """

    inspector = sa.inspect(bind)
    if "user_oauth" not in set(inspector.get_table_names()):
        return

    columns = {c["name"] for c in inspector.get_columns("user_oauth")}
    if not {"provider", "access_token"}.issubset(columns):
        return

    bind.execute(
        sa.update(USER_OAUTH_TABLE)
        .where(USER_OAUTH_TABLE.c.provider == APP_ID)
        .values(access_token="")
    )


def upgrade() -> None:
    bind = op.get_bind()
    _set_scopes(bind, CURRENT_SCOPES)
    _invalidate_existing_facebook_grants(bind)


def downgrade() -> None:
    # The cleared access tokens are gone for good (that's the point — force a
    # reconnect); there is nothing meaningful to restore for user_oauth here.
    _set_scopes(op.get_bind(), PREVIOUS_SCOPES)
