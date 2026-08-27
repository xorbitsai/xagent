"""narrow the Google Calendar connector's OAuth scope to calendar.events

The Calendar connector's tools (search/create/get/update/delete) only ever
operate on events, never on calendar list management, so the full
``.../auth/calendar`` scope requested more access than the feature set uses.

For a builtin app, the scope actually requested at authorize time is sourced
live from ``builtin_mcp_registry.py``, not from this table -- this migration
only converges the persisted ``public_mcp_apps.oauth_scopes`` row so
``validate_builtin_public_mcp_apps`` stops reporting drift against that
registry value on an already-seeded database. It deliberately does not touch
already-issued ``user_oauth`` grants: narrowing a scope doesn't require
re-consent for a token that already has the (now broader) old scope to keep
working, so there's nothing to invalidate.

Revision ID: 20260817_narrow_google_calendar_scope
Revises: 20260825_add_slack_channels_join_scope
Create Date: 2026-08-17

"""

import json
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "20260817_narrow_google_calendar_scope"
# This re-cut PR was rebased onto main after 20260825_add_slack_channels_join_scope
# landed, which is why this (earlier-dated) revision's parent has a later date.
down_revision: Union[str, None] = "20260825_add_slack_channels_join_scope"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


PUBLIC_MCP_APPS_TABLE = sa.table(
    "public_mcp_apps",
    sa.column("app_id", sa.String),
    sa.column("oauth_scopes", sa.JSON),
)

APP_ID = "google-calendar"
REQUIRED_COLUMNS = {"app_id", "oauth_scopes"}
OLD_SCOPES = ("https://www.googleapis.com/auth/calendar",)
NEW_SCOPES = ("https://www.googleapis.com/auth/calendar.events",)


def _columns_present(
    bind: sa.engine.Connection, table_name: str, required_columns: set[str]
) -> bool:
    """Whether ``table_name`` exists and has all of ``required_columns``.

    Used by _set_calendar_scopes()'s online branch, called from both
    upgrade() and downgrade(): this migration must be a no-op (not an
    error) against a database mid-way through a schema this old, or an
    admin's reduced-schema table, rather than assume a table shape that
    matches only the current model.
    """
    inspector = sa.inspect(bind)
    if table_name not in set(inspector.get_table_names()):
        return False
    columns = {column["name"] for column in inspector.get_columns(table_name)}
    return required_columns.issubset(columns)


def _offline_scopes_literal(scopes: Sequence[str], dialect_name: str) -> object:
    # Match the online sa.JSON binding contract: values are stored as JSON,
    # not as a bare SQL string literal, on every supported dialect.
    serialized_literal = op.inline_literal(json.dumps(scopes))
    if dialect_name == "postgresql":
        return sa.cast(serialized_literal, sa.JSON())
    return serialized_literal


def _set_calendar_scopes(scopes: Sequence[str]) -> None:
    """Write ``scopes`` to the google-calendar row's oauth_scopes column.

    oauth_scopes is in admin_mcp's _BUILTIN_PROTECTED_FIELDS, so an operator
    can never have customized it via the admin PATCH endpoint -- safe to
    overwrite unconditionally, with no prior-value check, in both
    directions. Handles both the online and offline (``--sql``) paths itself
    -- unlike the sibling migrations' ``_set_<app>_scopes(bind, scopes)``
    helpers, this one has no live ``bind`` to take as a parameter until
    after the as_sql check below, since none of those siblings support
    offline SQL generation.
    """
    if op.get_context().as_sql:
        dialect_name = op.get_context().dialect.name
        statement = (
            sa.update(PUBLIC_MCP_APPS_TABLE)
            .where(PUBLIC_MCP_APPS_TABLE.c.app_id == op.inline_literal(APP_ID))
            .values(oauth_scopes=_offline_scopes_literal(scopes, dialect_name))
        )
        op.execute(statement)
        return

    bind = op.get_bind()
    if not _columns_present(bind, "public_mcp_apps", REQUIRED_COLUMNS):
        return

    bind.execute(
        sa.update(PUBLIC_MCP_APPS_TABLE)
        .where(PUBLIC_MCP_APPS_TABLE.c.app_id == APP_ID)
        .values(oauth_scopes=scopes)
    )


def upgrade() -> None:
    _set_calendar_scopes(NEW_SCOPES)


def downgrade() -> None:
    _set_calendar_scopes(OLD_SCOPES)
