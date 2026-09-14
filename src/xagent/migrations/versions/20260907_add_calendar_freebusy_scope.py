"""add calendar.freebusy to the Google Calendar connector's OAuth scope

The scheduling-conflict check added to the calendar connector's create/update
tools calls Google's ``freebusy.query`` endpoint to check attendees'
availability. That endpoint is not authorized by ``.../auth/calendar.events``
alone (per Google's own Calendar API scope reference) - it needs one of
``calendar``, ``calendar.readonly``, ``calendar.freebusy``, or
``calendar.events.freebusy``. ``calendar.freebusy`` is the minimal addition
that covers it without also granting broader calendar read access.

For a builtin app, the scope actually requested at authorize time is sourced
live from ``builtin_mcp_registry.py``, not from this table -- this migration
only converges the persisted ``public_mcp_apps.oauth_scopes`` row so
``validate_builtin_public_mcp_apps`` stops reporting drift against that
registry value on an already-seeded database.

Unlike the earlier narrowing migration (20260817_narrow_google_calendar_scope),
this one WIDENS the scope: an already-issued ``user_oauth`` grant only has
the narrower ``calendar.events`` scope and does not automatically pick up
the new one. Existing users must reconnect the Google Calendar connector
before attendee free/busy checks stop 403ing for them; this migration does
not (and cannot) retroactively fix already-issued tokens.

Revision ID: 20260907_add_calendar_freebusy_scope
Revises: 20260911_assistant_source_event
Create Date: 2026-09-07

"""

import json
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "20260907_add_calendar_freebusy_scope"
# This migration was originally authored before the later-dated assistant
# source migration landed. Pointing at the current main head keeps the graph
# linear; migration filenames are descriptive labels, not execution order.
down_revision: Union[str, None] = "20260911_assistant_source_event"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


PUBLIC_MCP_APPS_TABLE = sa.table(
    "public_mcp_apps",
    sa.column("app_id", sa.String),
    sa.column("oauth_scopes", sa.JSON),
)

APP_ID = "google-calendar"
OLD_SCOPES = ("https://www.googleapis.com/auth/calendar.events",)
NEW_SCOPES = (
    "https://www.googleapis.com/auth/calendar.events",
    "https://www.googleapis.com/auth/calendar.freebusy",
)


def _columns_present(
    bind: sa.engine.Connection, table_name: str, required_columns: set[str]
) -> bool:
    """Whether ``table_name`` exists and has all of ``required_columns``.

    Used by _set_calendar_scopes(), the online path upgrade() and
    downgrade() both call: this migration must be a no-op (not an error)
    against a database mid-way through a schema this old, or an admin's
    reduced-schema table, rather than assume a table shape that matches
    only the current model.
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


def _set_calendar_scopes_offline(scopes: Sequence[str]) -> None:
    dialect_name = op.get_context().dialect.name
    statement = (
        sa.update(PUBLIC_MCP_APPS_TABLE)
        .where(PUBLIC_MCP_APPS_TABLE.c.app_id == op.inline_literal(APP_ID))
        .values(oauth_scopes=_offline_scopes_literal(scopes, dialect_name))
    )
    op.execute(statement)


def _set_calendar_scopes(scopes: Sequence[str]) -> None:
    """Write ``scopes`` to the google-calendar row's oauth_scopes column.

    oauth_scopes is in admin_mcp's _BUILTIN_PROTECTED_FIELDS, so an operator
    can never have customized it via the admin PATCH endpoint -- safe to
    overwrite unconditionally, with no prior-value check, in both
    directions. Online-only: the caller has already handled the offline
    (``--sql``) path, since there's no live ``bind`` to use here otherwise.
    """
    bind = op.get_bind()
    if not _columns_present(bind, "public_mcp_apps", {"app_id", "oauth_scopes"}):
        return

    bind.execute(
        sa.update(PUBLIC_MCP_APPS_TABLE)
        .where(PUBLIC_MCP_APPS_TABLE.c.app_id == APP_ID)
        .values(oauth_scopes=scopes)
    )


def upgrade() -> None:
    # Offline (--sql) generation has a MockConnection, so reflection is
    # unavailable -- emit the unconditional UPDATE instead of inspecting.
    if op.get_context().as_sql:
        _set_calendar_scopes_offline(NEW_SCOPES)
        return

    _set_calendar_scopes(NEW_SCOPES)


def downgrade() -> None:
    if op.get_context().as_sql:
        _set_calendar_scopes_offline(OLD_SCOPES)
        return

    _set_calendar_scopes(OLD_SCOPES)
