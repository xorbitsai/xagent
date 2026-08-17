"""narrow the Google Calendar connector's OAuth scope to calendar.events

The Calendar connector's tools (search/create/get/update/delete) only ever
operate on events, never on calendar list management, so the full
``.../auth/calendar`` scope requested more access than the feature set uses.
This updates the persisted built-in catalog row to match the narrower scope
now declared in ``builtin_mcp_registry.py``.

Narrowing the catalog row only changes what *future* authorizations
request -- Google's OAuth refresh grant returns a new access token for
whatever scope was originally consented to, it never narrows it (see
``refresh_oauth_token`` in ``web/tools/config.py``). So this migration also
revokes -- deletes -- any ``user_oauth`` row whose granted scope still
contains the old, full calendar scope, forcing those users through the
OAuth flow again on their next Calendar action; only then do they get the
narrower grant. Matching is on granted-scope content, not the ``provider``
column: ``APPS_REQUIRING_APP_SCOPED_OAUTH_GRANT`` (web/mcp_apps.py) does not
include ``google-calendar``, so a bare provider-level ``google`` row is also
accepted as a Calendar credential (web/tools/config.py's
``_resolve_legacy_oauth_access_token``) and would evade a filter on
``provider == "google-calendar"`` alone.

This is a delete, matching this table's existing disconnect contract
(``web/api/mcp.py`` deletes the row outright; there is no revoked/active
status flag on ``user_oauth`` to flip instead). It does not call Google's
token-revocation endpoint: no migration in this repo makes outbound network
calls, and neither does the existing disconnect flow for this table (only
the separate ``MCPOAuthGrant`` table gets a best-effort external revoke).
Google still considers the token valid until it expires or the user revokes
app access from their own Google Account -- identical to any other
local-only disconnect today.

Revision ID: 20260817_narrow_google_calendar_scope
Revises: 20260813_trace_json_columns_to_jsonb
Create Date: 2026-08-17

"""

import json
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "20260817_narrow_google_calendar_scope"
down_revision: Union[str, None] = "20260813_trace_json_columns_to_jsonb"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


PUBLIC_MCP_APPS_TABLE = sa.table(
    "public_mcp_apps",
    sa.column("app_id", sa.String),
    sa.column("oauth_scopes", sa.JSON),
)

USER_OAUTH_TABLE = sa.table(
    "user_oauth",
    sa.column("id", sa.Integer),
    sa.column("scope", sa.String),
)

APP_ID = "google-calendar"
OLD_SCOPE = "https://www.googleapis.com/auth/calendar"
NEW_SCOPE = "https://www.googleapis.com/auth/calendar.events"
OLD_SCOPES = [OLD_SCOPE]
NEW_SCOPES = [NEW_SCOPE]


def _offline_scopes_literal(scopes: list[str], dialect_name: str):
    # Match the online sa.JSON binding contract (none_as_null=False) used
    # elsewhere in this migration set: values are stored as JSON, not as a
    # bare SQL string literal, on every supported dialect.
    serialized_literal = op.inline_literal(json.dumps(scopes, sort_keys=True))
    if dialect_name == "postgresql":
        return sa.cast(serialized_literal, sa.JSON())
    return serialized_literal


def _offline_update(scopes: list[str]) -> None:
    dialect_name = op.get_context().dialect.name
    statement = (
        sa.update(PUBLIC_MCP_APPS_TABLE)
        .where(PUBLIC_MCP_APPS_TABLE.c.app_id == op.inline_literal(APP_ID))
        .values(oauth_scopes=_offline_scopes_literal(scopes, dialect_name))
    )
    op.execute(statement)


def _revoke_grants_carrying_the_old_calendar_scope(bind: sa.engine.Connection) -> None:
    inspector = sa.inspect(bind)
    if "user_oauth" not in inspector.get_table_names():
        return

    columns = {column["name"] for column in inspector.get_columns("user_oauth")}
    if not {"id", "scope"}.issubset(columns):
        return

    rows = bind.execute(
        sa.select(USER_OAUTH_TABLE.c.id, USER_OAUTH_TABLE.c.scope)
    ).fetchall()

    # Google's token response echoes ``scope`` as a plain space-delimited
    # string (RFC 6749); split rather than substring-match so a scope that
    # merely contains "calendar" as a substring of something else can't
    # collide with the exact grant we're hunting for.
    ids_to_revoke = [
        row.id
        for row in rows
        if row.scope
        and OLD_SCOPE in row.scope.split()
        and NEW_SCOPE not in row.scope.split()
    ]
    if not ids_to_revoke:
        return

    bind.execute(
        sa.delete(USER_OAUTH_TABLE).where(USER_OAUTH_TABLE.c.id.in_(ids_to_revoke))
    )


def upgrade() -> None:
    if op.get_context().as_sql:
        _offline_update(NEW_SCOPES)
        # The revoke step below reads and judges each row's actual granted
        # scope; offline (--sql) generation runs against a MockConnection,
        # where neither reflection nor a live SELECT is possible. It is
        # deliberately skipped here, same as the row-content-dependent
        # cleanup in 20260813_trace_json_columns_to_jsonb.py's offline path.
        return

    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "public_mcp_apps" in inspector.get_table_names():
        columns = {
            column["name"] for column in inspector.get_columns("public_mcp_apps")
        }
        if {"app_id", "oauth_scopes"}.issubset(columns):
            bind.execute(
                sa.update(PUBLIC_MCP_APPS_TABLE)
                .where(PUBLIC_MCP_APPS_TABLE.c.app_id == APP_ID)
                .values(oauth_scopes=NEW_SCOPES)
            )

    _revoke_grants_carrying_the_old_calendar_scope(bind)


def downgrade() -> None:
    # Restores the catalog row's broader scope. Does not, and cannot,
    # restore any user_oauth row the upgrade revoked: deleting a grant is
    # not reversible, and a downgrade certainly shouldn't try to fabricate
    # a "the old scope was granted after all" credential.
    if op.get_context().as_sql:
        _offline_update(OLD_SCOPES)
        return

    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "public_mcp_apps" not in inspector.get_table_names():
        return

    columns = {column["name"] for column in inspector.get_columns("public_mcp_apps")}
    if not {"app_id", "oauth_scopes"}.issubset(columns):
        return

    bind.execute(
        sa.update(PUBLIC_MCP_APPS_TABLE)
        .where(PUBLIC_MCP_APPS_TABLE.c.app_id == APP_ID)
        .values(oauth_scopes=OLD_SCOPES)
    )
