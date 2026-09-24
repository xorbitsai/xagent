"""narrow Google Drive OAuth access and hide Gmail

Revision ID: 20260924_narrow_google_oauth_scopes
Revises: 20260923_seed_freshdesk_mcp_app
Create Date: 2026-09-24 00:00:00.000000

"""

import json
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "20260924_narrow_google_oauth_scopes"
down_revision: Union[str, None] = "20260923_seed_freshdesk_mcp_app"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


PUBLIC_MCP_APPS_TABLE = sa.table(
    "public_mcp_apps",
    sa.column("app_id", sa.String),
    sa.column("oauth_scopes", sa.JSON),
    sa.column("is_visible_in_connector", sa.Boolean),
)

GMAIL_APP_ID = "gmail"
GOOGLE_DRIVE_APP_ID = "google-drive"

PREVIOUS_GMAIL_SCOPES = ("https://www.googleapis.com/auth/gmail.modify",)
CURRENT_GMAIL_SCOPES: tuple[str, ...] = ()
PREVIOUS_GOOGLE_DRIVE_SCOPES = ("https://www.googleapis.com/auth/drive",)
CURRENT_GOOGLE_DRIVE_SCOPES = ("https://www.googleapis.com/auth/drive.file",)


def _columns_present(
    bind: sa.engine.Connection, table_name: str, required_columns: set[str]
) -> bool:
    inspector = sa.inspect(bind)
    if table_name not in set(inspector.get_table_names()):
        return False
    columns = {column["name"] for column in inspector.get_columns(table_name)}
    return required_columns.issubset(columns)


def _offline_json_literal(values: Sequence[str], dialect_name: str) -> object:
    serialized_literal = op.inline_literal(json.dumps(values))
    if dialect_name == "postgresql":
        return sa.cast(serialized_literal, sa.JSON())
    return serialized_literal


def _set_scopes_offline(app_id: str, scopes: Sequence[str]) -> None:
    statement = (
        sa.update(PUBLIC_MCP_APPS_TABLE)
        .where(PUBLIC_MCP_APPS_TABLE.c.app_id == op.inline_literal(app_id))
        .values(
            oauth_scopes=_offline_json_literal(scopes, op.get_context().dialect.name)
        )
    )
    op.execute(statement)


def _set_scopes(bind: sa.engine.Connection, app_id: str, scopes: Sequence[str]) -> None:
    if not _columns_present(bind, "public_mcp_apps", {"app_id", "oauth_scopes"}):
        return

    bind.execute(
        sa.update(PUBLIC_MCP_APPS_TABLE)
        .where(PUBLIC_MCP_APPS_TABLE.c.app_id == app_id)
        .values(oauth_scopes=list(scopes))
    )


def _set_gmail_visibility(bind: sa.engine.Connection, visible: bool) -> None:
    if not _columns_present(
        bind, "public_mcp_apps", {"app_id", "is_visible_in_connector"}
    ):
        return

    bind.execute(
        sa.update(PUBLIC_MCP_APPS_TABLE)
        .where(PUBLIC_MCP_APPS_TABLE.c.app_id == GMAIL_APP_ID)
        .values(is_visible_in_connector=visible)
    )


def upgrade() -> None:
    # Builtin OAuth execution fields come from builtin_mcp_registry.py at
    # request time. These updates converge already-seeded catalog rows so the
    # drift validator agrees with that registry and the Gmail visibility gate
    # also applies to existing installations. Already-issued user_oauth grants
    # are deliberately retained; this change only controls future requests.
    if op.get_context().as_sql:
        _set_scopes_offline(GMAIL_APP_ID, CURRENT_GMAIL_SCOPES)
        _set_scopes_offline(GOOGLE_DRIVE_APP_ID, CURRENT_GOOGLE_DRIVE_SCOPES)
        op.execute(
            sa.update(PUBLIC_MCP_APPS_TABLE)
            .where(PUBLIC_MCP_APPS_TABLE.c.app_id == op.inline_literal(GMAIL_APP_ID))
            .values(is_visible_in_connector=op.inline_literal(False))
        )
        return

    bind = op.get_bind()
    _set_scopes(bind, GMAIL_APP_ID, CURRENT_GMAIL_SCOPES)
    _set_scopes(bind, GOOGLE_DRIVE_APP_ID, CURRENT_GOOGLE_DRIVE_SCOPES)
    _set_gmail_visibility(bind, False)


def downgrade() -> None:
    if op.get_context().as_sql:
        _set_scopes_offline(GMAIL_APP_ID, PREVIOUS_GMAIL_SCOPES)
        _set_scopes_offline(GOOGLE_DRIVE_APP_ID, PREVIOUS_GOOGLE_DRIVE_SCOPES)
        op.execute(
            sa.update(PUBLIC_MCP_APPS_TABLE)
            .where(PUBLIC_MCP_APPS_TABLE.c.app_id == op.inline_literal(GMAIL_APP_ID))
            .values(is_visible_in_connector=op.inline_literal(True))
        )
        return

    bind = op.get_bind()
    _set_scopes(bind, GMAIL_APP_ID, PREVIOUS_GMAIL_SCOPES)
    _set_scopes(bind, GOOGLE_DRIVE_APP_ID, PREVIOUS_GOOGLE_DRIVE_SCOPES)
    _set_gmail_visibility(bind, True)
