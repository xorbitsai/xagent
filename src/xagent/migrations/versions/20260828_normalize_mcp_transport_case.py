r"""Normalize stored MCP transport values to their canonical form

Revision ID: 20260828_normalize_mcp_transport_case
Revises: 20260821_actor_oauth_flow_states
Create Date: 2026-08-28

`transport` is a free-form string on the MCP API models, so rows written before
the write-time normalizing validators shipped may hold a mixed-case or
whitespace-padded value (e.g. "Streamable_HTTP", "OAuth", "\tstdio"). The
readers of that column do not agree on how to compare it: `classify_app_auth`
(web/mcp_apps.py) lowercases before matching, while the connect and credential
paths match exactly -- `api/mcp.py`'s `server.transport != "oauth"` gates,
`api/auth.py:2087`, `web/tools/config.py:3547`, and the core serializer's
`transport in ["sse", "websocket", "streamable_http"]` branch. A row stored as
"OAuth" is therefore classified `builtin_oauth` by the catalog and rejected by
every exact-matching gate behind it.

A shared catalog row's transport is never rewritten by the connect path, so an
un-migrated row stays in that split state indefinitely. Backfill the two
web-layer tables once so the stored values agree with what every reader
expects.

State of the readers when this lands. This migration is the third of the three
changes tracked in #1828 and is sequenced behind both of the others, so the
descriptions above are written against the post-#1829/#1830 tree:

  1. Write-side canonicalization (#1829) introduces `normalize_transport()`
     (`web/services/mcp_runtime.py`) and applies it on every write path, so no
     new non-canonical row is created. That helper does not exist yet on this
     branch; the whitespace grammar below is specified against it.
  2. Read-side tolerance (#1830) normalizes on read, so a not-yet-backfilled
     row behaves like its canonical equivalent.

Before those land the two axes differ. A whitespace-padded row is rejected
fairly uniformly, because no current reader strips whitespace and the runtime
OAuth gate (`_is_mcp_oauth_http_server`) still matches exactly. A mixed-case row
already splits today: `classify_app_auth` lowercases, so it admits
"Streamable_HTTP" and "OAuth", while the exact-matching gates behind it reject
the same row. The backfill is worth
running either way, because the divergence above is what the row hits once the
tolerant readers are deployed, and because an admin catalog write can still
author a mixed-case value today.

Covered value set. A row is rewritten only when normalizing it yields one of
the transports the application dispatches on (see _CANONICAL_TRANSPORTS below).
That bound makes the rewrite safe to audit: the migration can only ever move a
row onto a value the readers already agree on, never invent one, and an
unrecognized string is left exactly as stored rather than reshaped into
different garbage.

Whitespace grammar, scoped to ASCII. The write-side helper canonicalizes with
Python's `str.strip().lower()`; single-argument SQL `TRIM()` strips only ASCII
spaces, so TAB/LF/CR are translated to spaces before `TRIM` runs. Translating
rather than deleting keeps `TRIM`'s edges-only semantics: an interior character
is never removed, so the migration cannot splice a canonical transport out of a
value the helper leaves alone. The non-ASCII whitespace `str.strip()` also
removes (NBSP, U+3000) is out of scope and left unfixed -- see
_PADDING_WHITESPACE for why naming it portably is not possible.

Rollout ordering. This is a one-shot UPDATE with no CHECK constraint or trigger
behind it, so it converges only if no application instance running the
pre-validator models is still serving admin writes when it lands; the startup
migration lock serializes migrators, not request sessions. Deploy #1829 and
#1830, drain the old instances, then run this backfill. Run out of order it is
still safe -- it only ever rewrites a value toward the form every reader agrees
on -- but it is not guaranteed to leave the table canonical, because a
surviving old writer can insert a fresh non-canonical row after the UPDATE
commits.

Idempotent: the normalizing expression is a no-op on already-canonical values,
and the WHERE clause skips rows that are already normalized.
"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "20260828_normalize_mcp_transport_case"
down_revision = "20260821_actor_oauth_flow_states"
branch_labels = None
depends_on = None

_TABLES = ("mcp_servers", "public_mcp_apps")

# The transports the application dispatches on: the stdio branch and
# HTTP_MCP_TRANSPORTS in the MCP model/runtime, plus "oauth", which is the
# default for public_mcp_apps.transport and drives the builtin_oauth catalog
# classification. A row is only rewritten when it normalizes onto one of these.
_CANONICAL_TRANSPORTS = ("stdio", "sse", "websocket", "streamable_http", "oauth")

# Whitespace that Python's str.strip() removes but single-argument SQL TRIM()
# does not. Spelled as code points so no literal control character is embedded
# in the emitted SQL -- an offline (--sql) artifact carrying a raw newline
# inside a string literal is a hazard for whoever runs it by hand.
#
# ASCII-only, and that boundary is a portability constraint rather than a
# preference. str.strip() also removes NBSP (U+00A0) and the ideographic space
# (U+3000), but neither can be named portably here: on a database whose encoding
# cannot represent them, PostgreSQL's chr() raises "requested character too
# large for encoding" and aborts the whole upgrade, and a U&'...' literal fails
# the same way (both verified against a SQL_ASCII server). A row padded with one
# of those is left untouched rather than fixed -- for a one-shot data migration,
# crashing on a database whose encoding it never had to care about is the worse
# trade. Such a row is also unstorable in exactly the encodings where the fix is
# unexpressible. Every code point below is single-byte ASCII and so is
# expressible on every supported encoding.
_PADDING_WHITESPACE = (
    9,  # TAB
    10,  # LF
    11,  # VT
    12,  # FF
    13,  # CR
    28,  # FS
    29,  # GS
    30,  # RS
    31,  # US
)

# PostgreSQL spells the code-point-to-character function CHR(); SQLite spells it
# CHAR(). Both take one integer argument and need no extension. Indexed rather
# than looked up with a default: these are the only two dialects the project
# supports (see db/migration_support.py), and an unsupported one should fail
# loudly here rather than silently receive a narrower backfill.
_CHAR_FUNCTION = {"postgresql": "chr", "sqlite": "char"}

# PostgreSQL's LOWER() is locale-sensitive. On a database created with a Turkish
# locale (ICU tr-TR is supported and unrestricted here), LOWER('STDIO') yields
# 'stdıo' with a dotless i (U+0131), which is not in _CANONICAL_TRANSPORTS -- so
# the guard below rejects the row and the backfill silently leaves 'STDIO'
# unfixed while every other value normalizes. Forcing the C collation on LOWER's
# *input* makes the case mapping ASCII and locale-independent. It must be the
# input, not the result: `LOWER(x) COLLATE "C"` still lowercases under the
# database locale and only relabels the output.
#
# SQLite needs no equivalent -- its LOWER() maps ASCII A-Z only, by design.
_LOWER_COLLATION = {"postgresql": ' COLLATE "C"', "sqlite": ""}


def _normalized_expression(dialect_name: str) -> str:
    """SQL canonicalizing `transport` the way normalize_transport() does."""
    char_fn = _CHAR_FUNCTION[dialect_name]
    collation = _LOWER_COLLATION[dialect_name]
    expression = "transport"
    for code_point in _PADDING_WHITESPACE:
        expression = f"REPLACE({expression}, {char_fn}({code_point}), ' ')"
    return f"LOWER(TRIM({expression}){collation})"


def _tables_with_transport() -> list[str]:
    """The subset of _TABLES this database actually has, with the column.

    Online only: reflection needs a real bind. Offline callers use _TABLES
    directly, which is sound because the names are compile-time constants.
    """
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing = set(inspector.get_table_names())
    present = []
    for table in _TABLES:
        if table not in existing:
            continue
        columns = {col["name"] for col in inspector.get_columns(table)}
        if "transport" in columns:
            present.append(table)
    return present


def _backfill_statement(table: str, normalized: str) -> str:
    """The one UPDATE this migration runs, as text.

    Split out from _backfill so a test can execute the exact statement the
    migration would rather than restating its WHERE clause -- a restated copy
    keeps passing when the real clause changes.
    """
    canonical_list = ", ".join(f"'{value}'" for value in _CANONICAL_TRANSPORTS)
    # NULL transports are left alone: both comparisons below are NULL-safe
    # (they evaluate to NULL, not true), so those rows are skipped rather than
    # rewritten to ''.
    return f"""
        UPDATE {table}
        SET transport = {normalized}
        WHERE transport IS NOT NULL
          AND transport <> {normalized}
          AND {normalized} IN ({canonical_list})
    """


def _backfill(table: str, normalized: str) -> None:
    op.execute(_backfill_statement(table, normalized))


def upgrade() -> None:
    from alembic import context

    normalized = _normalized_expression(op.get_context().dialect.name)

    if context.is_offline_mode():
        # Offline (--sql) supplies a MockConnection that cannot be inspected,
        # so reflection is skipped and both UPDATEs are emitted unconditionally.
        # The table names are compile-time constants, and an UPDATE against a
        # table the target database lacks surfaces when the operator applies the
        # script -- the right place for it, since offline mode cannot know the
        # target's shape.
        for table in _TABLES:
            _backfill(table, normalized)
        return

    for table in _tables_with_transport():
        _backfill(table, normalized)


def downgrade() -> None:
    # Deliberately not reversible: the original mixed-case/padded spellings are
    # not recorded anywhere, and restoring them would reintroduce rows the
    # application cannot connect. Normalized values remain valid for every
    # earlier revision, so leaving them in place is safe.
    pass
