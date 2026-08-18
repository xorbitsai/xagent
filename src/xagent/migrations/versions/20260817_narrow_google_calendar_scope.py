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
calendar scope (or has no scope recorded at all -- see below), forcing
those users through the OAuth flow again on their next Calendar action.

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
mitigated architecturally instead, not by data cleanup: ``google-calendar``
has been added to ``APPS_REQUIRING_APP_SCOPED_OAUTH_GRANT`` in
web/mcp_apps.py, so the resolver (web/tools/config.py's
``_resolve_legacy_oauth_access_token``) and the connected-state display
(web/api/mcp.py's ``_enrich_oauth_server_info`` /
``_connected_oauth_server_for_app``) no longer accept a bare ``google`` row
as a Calendar credential at all, regardless of what scope it carries.

Be precise about what that policy change does and does not establish: it
only closes the *bare-row* angle. It does **not** make the app-scoped
``google-calendar`` row's own token trustworthy going forward. Google's
authorize request still sends ``include_granted_scopes=true``
(web/api/auth.py), and the callback stores whatever ``scope`` string Google
echoes back without validating it against the app's registered
``oauth_scopes`` (web/api/auth.py's token-exchange handler). A user who
reconnects Calendar after this migration runs can, in principle, receive a
token whose granted scope again includes the old, full ``calendar`` scope
alongside ``calendar.events`` -- Google's incremental consent is scoped to
the OAuth *client*, not to what any one authorization request asked for, and
nothing here rejects or re-narrows a combined grant returned on reconnect.
Nothing in this migration, or in the ``APPS_REQUIRING_APP_SCOPED_OAUTH_GRANT``
change, prevents that recurrence -- closing it for good would mean either
requesting ``include_granted_scopes=false`` for this app, or validating (and
rejecting/stripping) the returned scope at the callback against the app's
registered ``oauth_scopes``. Neither exists today. This migration is a
point-in-time cleanup of rows that predate it, not a standing least-privilege
guarantee.

A row that carries *both* the old and the new scope (which, per the above,
can happen on reconnect even after this migration) is still revoked here:
presence of the old scope is grounds enough on its own, regardless of
whether the new scope also shows up alongside it -- if this migration is
ever re-applied against a database with such a row, revoking it again is
the correct outcome; nothing distinguishes "the same forbidden scope,
again" from "still first time."

A ``google-calendar`` row with no scope recorded at all (``NULL`` or empty
string) is *also* revoked, not skipped: per RFC 6749 5.1, a provider may
omit ``scope`` entirely when the granted scope exactly matches what was
requested, and the only thing this app_id had ever requested before this
migration first ran was the old, broad scope -- so a missing scope is
evidence of that grant, not proof of a narrow one.

That inference only holds looking backward from before this migration first
ran, which makes the empty/NULL branch the one piece of this cleanup that
is **not** safe to replay. If this revision is downgraded and re-upgraded
(or an offline script generated before some Calendar users reconnected is
applied after), any ``google-calendar`` row created *since* the first run
with an omitted ``scope`` echo -- itself entirely possible under RFC 6749
5.1 once the app requests only ``calendar.events`` -- would be revoked all
over again, even though it never carried the old scope. This migration
cannot tell the two cases apart from the row alone (there is no reliable
"created before this migration" marker to key off), so it does not try to.
Do not downgrade and re-upgrade this revision, and do not apply an
offline-generated script for it more than once, once Calendar connections
have been made under the narrow scope.

Online and offline (``--sql``) generation build and apply the exact same
predicate (see ``_revoke_predicate``) -- there are not two implementations
of "which rows qualify" to keep in sync. The one difference is how each
side supplies its literals: online binds ordinary parameters against a live
connection; offline inlines them into the emitted SQL text, since ``--sql``
generation has no connection to bind against. Neither path pages or
batches: a single ``DELETE ... WHERE <predicate>`` is what the offline path
was always going to emit (there is no way to page through rows without a
live connection to read from), and it is what the online path now issues
too -- there is no batching benefit to preserve, since every revoked row's
lock is held until this migration's single transaction commits either way.

The online path still guards on ``user_oauth`` (and its ``id``/``provider``/
``scope`` columns) existing before touching it, matching this migration's
guard on ``public_mcp_apps``. The offline path cannot: ``--sql`` generation
runs against a mock connection with no live reflection available, so its
``DELETE`` is unconditional. In the full migration chain this is never a
problem -- the migration that creates ``user_oauth``
(``c7dfa28cc67a_add_user_oauth_table``) is an ancestor of this revision, so
a whole-history offline script always creates the table before this
``DELETE`` runs against it. It only becomes a problem for a hand-rolled
partial ``--sql`` range applied to a database in a state the online guard
was written to tolerate (e.g. ``user_oauth`` manually dropped) -- an
unsupported, self-inflicted deployment shape, not a normal one.

After the revoke above, one further cleanup: a user whose *only* historical
Calendar connection was the app_id-less "Connect Google" flow (batch-connect
in web/api/auth.py) has a ``user_mcpservers`` row for the shared "Google
Calendar" ``mcp_servers`` row, backed only by a bare ``provider == "google"``
``user_oauth`` row -- never a ``google-calendar`` one. Before the
``APPS_REQUIRING_APP_SCOPED_OAUTH_GRANT`` change above, that bare row was an
accepted Calendar credential, so the connection worked. After it, the
connected-state check correctly reports "disconnected" -- but the stale
``user_mcpservers`` row itself is not gone, and per web/api/auth.py's own
comment on why the batch-connect flow now *skips creating* this row for
apps in ``APPS_REQUIRING_APP_SCOPED_OAUTH_GRANT``, the agent runtime picks
up an existing ``user_mcpservers`` row directly, bypassing the
connected-state check, and can never resolve a token for it. Left alone,
these users would see Calendar as available to attach to an agent, then
have every tool call fail at token resolution instead of the tool simply
not being offered. ``_remove_orphaned_calendar_server_rows`` deletes the
``user_mcpservers`` row for any user who, after the revoke step above, has
no ``google-calendar`` credential -- mirroring, retroactively, what the
batch-connect flow now does prospectively.

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
import logging
from typing import Callable, Sequence, Union

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
    sa.column("user_id", sa.Integer),
    sa.column("provider", sa.String),
    sa.column("scope", sa.String),
)

MCP_SERVERS_TABLE = sa.table(
    "mcp_servers",
    sa.column("id", sa.Integer),
    sa.column("name", sa.String),
)

USER_MCPSERVERS_TABLE = sa.table(
    "user_mcpservers",
    sa.column("id", sa.Integer),
    sa.column("user_id", sa.Integer),
    sa.column("mcpserver_id", sa.Integer),
)

APP_ID = "google-calendar"
CALENDAR_SERVER_NAME = "Google Calendar"
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


def _revoke_predicate(literal: Callable[[object], object]):
    """The one definition of "which user_oauth rows qualify for revocation".

    ``literal`` wraps each Python value into whatever the caller needs it to
    compile as: ``sa.literal`` for the online path (an ordinary bound
    parameter against a live connection), ``op.inline_literal`` for the
    offline (``--sql``) path (inlined into the emitted SQL text, since that
    path has no connection to bind against). The predicate shape -- and
    therefore the exact set of rows it matches -- is identical either way,
    by construction, not by two implementations being kept manually in sync.

    Padding ``scope`` with a leading/trailing space and searching for the
    old scope padded the same way turns a token-exact match into a plain
    substring search that also matches at the very start or end of the
    string, without a false hit on a longer scope that merely contains the
    old one as a substring (OLD_SCOPE has no spaces, and RFC 6749's scope
    separator is always a single space).
    """
    space = literal(" ")
    padded_scope = space.concat(USER_OAUTH_TABLE.c.scope).concat(space)
    like_pattern = literal(f"% {OLD_SCOPE} %")

    return sa.and_(
        USER_OAUTH_TABLE.c.provider == literal(APP_ID),
        sa.or_(
            USER_OAUTH_TABLE.c.scope.is_(None),
            USER_OAUTH_TABLE.c.scope == literal(""),
            padded_scope.like(like_pattern),
        ),
    )


def _offline_revoke_statement() -> None:
    op.execute(sa.delete(USER_OAUTH_TABLE).where(_revoke_predicate(op.inline_literal)))


def _revoke_grants_carrying_the_old_calendar_scope(bind: sa.engine.Connection) -> None:
    inspector = sa.inspect(bind)
    if "user_oauth" not in inspector.get_table_names():
        return

    columns = {column["name"] for column in inspector.get_columns("user_oauth")}
    if not {"id", "provider", "scope"}.issubset(columns):
        return

    result = bind.execute(
        sa.delete(USER_OAUTH_TABLE).where(_revoke_predicate(sa.literal))
    )
    if result.rowcount:
        logger.warning(
            "Revoked %d google-calendar user_oauth grant(s) still carrying "
            "the old '%s' scope; affected users must reconnect Calendar.",
            result.rowcount,
            OLD_SCOPE,
        )


def _remove_orphaned_calendar_server_rows(bind: sa.engine.Connection) -> None:
    inspector = sa.inspect(bind)
    required_tables = {"mcp_servers", "user_mcpservers", "user_oauth"}
    if not required_tables.issubset(set(inspector.get_table_names())):
        return

    mcp_servers_columns = {
        column["name"] for column in inspector.get_columns("mcp_servers")
    }
    user_mcpservers_columns = {
        column["name"] for column in inspector.get_columns("user_mcpservers")
    }
    user_oauth_columns = {
        column["name"] for column in inspector.get_columns("user_oauth")
    }
    if (
        not {"id", "name"}.issubset(mcp_servers_columns)
        or not {"id", "user_id", "mcpserver_id"}.issubset(user_mcpservers_columns)
        or not {"user_id", "provider"}.issubset(user_oauth_columns)
    ):
        return

    calendar_server_id = bind.execute(
        sa.select(MCP_SERVERS_TABLE.c.id).where(
            MCP_SERVERS_TABLE.c.name == CALENDAR_SERVER_NAME
        )
    ).scalar_one_or_none()
    if calendar_server_id is None:
        return

    connected_user_ids = sa.select(USER_OAUTH_TABLE.c.user_id).where(
        USER_OAUTH_TABLE.c.provider == APP_ID
    )
    result = bind.execute(
        sa.delete(USER_MCPSERVERS_TABLE).where(
            USER_MCPSERVERS_TABLE.c.mcpserver_id == calendar_server_id,
            USER_MCPSERVERS_TABLE.c.user_id.not_in(connected_user_ids),
        )
    )
    if result.rowcount:
        logger.warning(
            "Removed %d orphaned Calendar user_mcpservers row(s) for users "
            "with no google-calendar credential; those users must reconnect "
            "to use Calendar again.",
            result.rowcount,
        )


def upgrade() -> None:
    if op.get_context().as_sql:
        _offline_update(NEW_SCOPES)
        _offline_revoke_statement()
        # The orphaned-server cleanup below needs a live SELECT (which
        # mcp_servers row is "Google Calendar", which users still have a
        # google-calendar credential) that offline (--sql) generation has no
        # connection to run. Skipped here; see the module docstring.
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
    _remove_orphaned_calendar_server_rows(bind)


def downgrade() -> None:
    # Restores the catalog row's broader scope. Does not, and cannot,
    # restore any user_oauth or user_mcpservers row the upgrade removed:
    # deleting a grant or a server association is not reversible, and a
    # downgrade certainly shouldn't try to fabricate "the old scope was
    # granted after all" or "the user was still connected after all".
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
