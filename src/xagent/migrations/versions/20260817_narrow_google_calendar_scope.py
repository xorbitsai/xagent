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
revokes -- deletes -- every ``user_oauth`` row with ``provider ==
"google-calendar"`` whose granted scope still contains the old, full
calendar scope, forcing those users through the OAuth flow again on their
next Calendar action.

Deliberately scoped to ``provider == "google-calendar"`` only, not to any
row whose *scope content* happens to mention the old calendar scope: that
same bare "google" row is the fallback credential every other Google
connector (Gmail, Drive, Docs, ...) may also be relying on, so deleting it
on a scope-content match alone would risk breaking those too. A
``google-calendar``-provider row exists only because this connector's own
connect flow created it (``provider=(app_id or provider)`` in
web/api/auth.py), so it is unambiguously this connector's, and only this
connector's, credential -- safe to delete outright.

This leaves one thing this data migration alone can't fix: a bare
``google`` row (no app_id) could still carry the old scope via
``include_granted_scopes=true`` without ever being touched here. That gap is
closed architecturally instead, not by data cleanup: ``google-calendar`` has
been added to ``APPS_REQUIRING_APP_SCOPED_OAUTH_GRANT`` in web/mcp_apps.py,
so the resolver (web/tools/config.py's
``_resolve_legacy_oauth_access_token``) and the connected-state display
(web/api/mcp.py's ``_enrich_oauth_server_info`` /
``_connected_oauth_server_for_app``) no longer accept a bare ``google`` row
as a Calendar credential at all, regardless of what scope it carries. A
stale bare-``google`` row that happens to carry the old scope is therefore
inert for Calendar going forward even though this migration leaves it
sitting in the table.

A row that carries *both* the old and the new scope (which
``include_granted_scopes=true`` can produce on reconnect once the registry
requests only the new scope) is still revoked: presence of the old scope
is grounds enough on its own, regardless of whether the new scope also
shows up alongside it. A ``google-calendar`` row with no scope recorded at
all (``NULL`` or empty string) is *also* revoked, not skipped: per RFC 6749
5.1, a provider may omit ``scope`` entirely when the granted scope exactly
matches what was requested, and the only thing this app_id ever requested
before this migration was the old, broad scope -- so a missing scope on
this provider's row is evidence of the broad grant, not proof of a narrow
one. Note this is a point-in-time cleanup of rows that predate this
migration, not a standing guarantee against recurrence -- a user who
reconnects after this migration runs could in principle still receive a
combined grant the same way, since Google's incremental consent is per
OAuth client, not per requested scope; the ``APPS_REQUIRING_APP_SCOPED_OAUTH_GRANT``
change above is what actually prevents that combined grant from mattering,
by refusing to trust the bare row for Calendar regardless of its scope.

The online cleanup above runs entirely in the database via a single
provider-plus-scope predicate, batched by id so a large table is never
scanned or deleted in one unbounded statement (see ``REVOKE_BATCH_SIZE``
below). Offline (``--sql``) generation emits the equivalent predicate as one
literal, unbatched ``DELETE`` -- there is no live connection to page
through, and an offline-generated script is applied as a single statement
by whatever runs it, not by this migration walking pages itself.

This is a delete, matching this table's existing disconnect contract
(``web/api/mcp.py`` deletes the row outright; there is no revoked/active
status flag on ``user_oauth`` to flip instead). It does not call Google's
token-revocation endpoint: no migration in this repo makes outbound network
calls, and neither does the existing disconnect flow for this table (only
the separate ``MCPOAuthGrant`` table gets a best-effort external revoke).
Google still considers the token valid until it expires or the user revokes
app access from their own Google Account -- identical to any other
local-only disconnect today.

Rows are paged by id (keyset, not OFFSET, matching the parent revision's
convention) and deleted in bounded batches, so this doesn't load the whole
table or build one unbounded ``IN`` list on a large deployment.

Revision ID: 20260817_narrow_google_calendar_scope
Revises: 20260813_trace_json_columns_to_jsonb
Create Date: 2026-08-17

"""

import json
import logging
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

logger = logging.getLogger(__name__)

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
    sa.column("provider", sa.String),
    sa.column("scope", sa.String),
)

APP_ID = "google-calendar"
OLD_SCOPE = "https://www.googleapis.com/auth/calendar"
NEW_SCOPE = "https://www.googleapis.com/auth/calendar.events"
OLD_SCOPES = [OLD_SCOPE]
NEW_SCOPES = [NEW_SCOPE]

# Ids per revoke batch. Bounds a page of (id, scope) rows plus one IN-list
# of matched ids, not the whole table -- see the module docstring.
REVOKE_BATCH_SIZE = 500


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


def _offline_revoke_statement() -> None:
    # Mirrors the online predicate exactly (see
    # _revoke_grants_carrying_the_old_calendar_scope), expressed as SQL the
    # target database evaluates itself instead of a Python-side fetch loop:
    # offline (--sql) generation has no live connection to page through.
    # Padding scope with a leading/trailing space and searching for the old
    # scope padded the same way turns a token-exact match into a plain
    # substring search that also matches at the very start or end of the
    # string, without a false hit on a longer scope that merely contains the
    # old one as a substring (OLD_SCOPE has no spaces, and the scope
    # separator is always a single space per RFC 6749).
    space = op.inline_literal(" ")
    padded_scope = space.concat(USER_OAUTH_TABLE.c.scope).concat(space)
    like_pattern = op.inline_literal(f"%{' ' + OLD_SCOPE + ' '}%")

    statement = sa.delete(USER_OAUTH_TABLE).where(
        USER_OAUTH_TABLE.c.provider == op.inline_literal(APP_ID),
        sa.or_(
            USER_OAUTH_TABLE.c.scope.is_(None),
            USER_OAUTH_TABLE.c.scope == op.inline_literal(""),
            padded_scope.like(like_pattern),
        ),
    )
    op.execute(statement)


def _revoke_grants_carrying_the_old_calendar_scope(bind: sa.engine.Connection) -> None:
    inspector = sa.inspect(bind)
    if "user_oauth" not in inspector.get_table_names():
        return

    columns = {column["name"] for column in inspector.get_columns("user_oauth")}
    if not {"id", "provider", "scope"}.issubset(columns):
        return

    select_page = (
        sa.select(USER_OAUTH_TABLE.c.id, USER_OAUTH_TABLE.c.scope)
        .where(
            USER_OAUTH_TABLE.c.provider == APP_ID,
            USER_OAUTH_TABLE.c.id > sa.bindparam("after"),
        )
        .order_by(USER_OAUTH_TABLE.c.id)
        .limit(REVOKE_BATCH_SIZE)
    )

    revoked_count = 0
    after = 0
    while True:
        rows = bind.execute(select_page, {"after": after}).fetchall()
        if not rows:
            break
        after = rows[-1].id

        # Google's token response echoes ``scope`` as a plain space-delimited
        # string (RFC 6749); split rather than substring-match so a scope
        # that merely contains "calendar" as a substring of something else
        # can't collide with the exact grant we're hunting for. Presence of
        # the old scope alone is grounds for revocation, regardless of
        # whether the new scope is also present in the same grant. A row
        # with no scope recorded (None or "") is revoked too, not skipped --
        # see the module docstring for why a missing scope on this provider
        # is evidence of the old, broad grant rather than proof of a narrow
        # one.
        ids_to_revoke = [
            row.id for row in rows if not row.scope or OLD_SCOPE in row.scope.split()
        ]
        if ids_to_revoke:
            bind.execute(
                sa.delete(USER_OAUTH_TABLE).where(
                    USER_OAUTH_TABLE.c.id.in_(ids_to_revoke)
                )
            )
            revoked_count += len(ids_to_revoke)

    if revoked_count:
        logger.warning(
            "Revoked %d google-calendar user_oauth grant(s) still carrying "
            "the old '%s' scope; affected users must reconnect Calendar.",
            revoked_count,
            OLD_SCOPE,
        )


def upgrade() -> None:
    if op.get_context().as_sql:
        _offline_update(NEW_SCOPES)
        _offline_revoke_statement()
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
