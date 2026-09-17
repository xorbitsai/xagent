"""convert trace payload columns from json to jsonb on PostgreSQL

``trace_events.data`` was a ``json`` column, which validates syntax at
write time and nothing else: it happily stores the NUL escape and either
half of an unpaired UTF-16 surrogate, both of which ``->>`` then refuses
to convert back to text -- one such row failed three monitoring endpoints
whole-query (#1149). ``jsonb`` decodes escapes to native text at INSERT,
so it rejects those payloads at the source (#1248). The two checkpoint
blob tables carry the same payloads (their rows are hashed slices of the
same trace data), so all three columns convert together:

- ``trace_events.data``
- ``trace_message_blobs.message_data``
- ``trace_checkpoint_blobs.blob_data``

``ALTER ... TYPE jsonb USING <col>::jsonb`` fails on any stored row that
carries one of those escapes, so the online upgrade first rewrites such
rows with the offending code points replaced by U+FFFD -- the same
substitution the write-side sanitizer
(``web/utils/json_payload_sanitizer.py``) applies to new payloads. Rows
are found with the same normalize-then-match steps as the read guard in
``web/api/monitor.py`` (``PG_ESCAPED_BACKSLASH`` and friends -- that file
is the patterns' source of truth and documents why each step is
load-bearing); the actual rewrite happens in Python, where escape
semantics are exact, not with SQL regex surgery on the JSON text.

This migration is PostgreSQL-only. SQLite stores JSON as TEXT and has no
jsonb; both branches are no-ops there. On PostgreSQL the online branch
guards on the table existing (a bare database has none of the three until
``create_all`` runs) and on the column not already being jsonb (a
create_all-first startup builds jsonb directly from the model, and this
keeps the upgrade idempotent on rerun).

The offline (--sql) branch emits only the three ALTERs: the cleanup step
must read rows, which a MockConnection cannot. An offline script applied
to a database still holding an unconvertible row fails on that ALTER --
PostgreSQL offline scripts run inside the BEGIN/COMMIT wrapper Alembic
emits, so the failure aborts the whole script and converts nothing.
Clean such rows first (or run the migration online, which does it for
you).

``jsonb`` does not preserve key order or duplicate keys. Consumers of
these columns deserialize into Python dicts (checkpoint restore, the
monitoring endpoints), where key order already carries no meaning, and
duplicate keys cannot be produced by ``json.dumps`` from a dict in the
first place -- the write path only ever serializes dicts.

``jsonb`` does, however, re-render numbers. Its numbers are ``numeric``,
so it keeps the literal's exact value and prints it in plain notation:
``1e+16`` reads back as the int ``10000000000000000``, ``1e25`` as 10^25,
and ``-0.0`` as ``0.0`` (``numeric`` has no signed zero). A long decimal
is preserved in full -- the retype is about *notation and Python type*,
not precision.

The checkpoint blob path re-hashes payloads it reads and compares them
against the hash recorded at write time, so a *pre-existing* row carrying
a shape whose Python type changes becomes undecodable after this
migration. New writes are covered: the write-side sanitizer normalizes
both shapes before the hash is taken. Rows this migration does not touch
are deliberately left alone -- catching them would mean rewriting every
row in all three tables to defend against shapes that need a float above
2**53, or a negative zero, in trace output. That leaves such rows with a
disclosed failure and no remedy of their own; the pre-flight below counts
only the NUL rows this migration actually rewrites, and a float-shaped
row hits the same escalation with no count query to warn about it. The
supported remedy is the same one: prune the affected task's checkpoint
history.

The rows this migration *does* rewrite keep their numbers verbatim rather
than being normalized here, so ``jsonb`` applies its own conversion on
ingest identically for rewritten and untouched rows. Normalizing in the
migration would make the two diverge, because reaching the number at all
means parsing it, and a float64 round-trip is lossy where ``numeric`` is
not: it renders ``1e25`` as 10000000000000000905969664 rather than 10^25,
truncates a literal carrying more precision than a double holds, and
turns ``1e1000`` into ``inf`` -- which ``json.dumps`` writes as
``Infinity``, not JSON, failing the rewrite's own cast. Parsing to
``Decimal`` and emitting each number as its own literal avoids all three;
see ``_loads_preserving_numbers`` and ``_dumps_preserving_numbers``.

Rewriting a *blob* row has a cost the paragraph above does not cover, and
it is the one thing to weigh before running this. Replacing NUL with
U+FFFD changes the payload, so its canonical hash changes -- but the hash
is also the blob's identity: ``trace_events.data`` references a blob by
embedding the hash value itself, and strict restore re-hashes what it
read and compares it against that reference
(``_load_messages_by_refs`` / ``_load_checkpoint_blob_by_ref``,
``web/services/trace_message_storage.py``). So a rewritten blob row still
resolves but fails verification: ``CheckpointMessageDecodeError``, which
``web/api/trace_handlers.py`` logs and treats as a skip, falling back to
an older checkpoint row.

Whether that fallback finds anything depends on which blob it was. Blob
rows are deduplicated per task, and ``context.system_prompt`` and
``context.metadata`` are typically byte-identical across every checkpoint
of a task -- so for those, one poisoned row means *no* readable
checkpoint at all, and the scan escalates to ``CheckpointCorruptError``:
HTTP 400 from the a2a resume path and a non-recoverable verdict in lease
recovery. For a mid-conversation message blob, checkpoints predating that
message still decode, so the task loses progress rather than the task.

The hash columns are therefore left verbatim on purpose, and the two
alternatives were considered and rejected:

* Recomputing ``message_hash``/``blob_hash`` from the rewritten payload
  makes the row unreachable rather than unverifiable -- the references in
  ``trace_events.data`` still name the old hash -- and it can abort the
  migration outright: if the task re-checkpointed the same content after
  the sanitizer landed, the new hash already exists and the UPDATE
  violates ``uq_trace_message_blobs_task_hash``. Doing it properly would
  mean remapping the hash inside every referring payload, including the
  refs marker's own hash, which is the application's hashing contract
  reimplemented in a migration.
* Deleting the affected blob rows lands on the same
  ``CheckpointMessageDecodeError`` for strict restore, so it buys nothing
  there, and it costs the non-strict readers: every viewer path
  (``api/conversation_logs.py``, ``api/workforces.py``,
  ``services/task_setup_snapshot.py``) decodes with ``strict=False``,
  which skips hash verification and renders the cleaned payload fine. A
  deleted row breaks those too.

Only NUL can reach a blob row this way. ``canonical_json_bytes`` uses
``ensure_ascii=False``, so a payload carrying an unpaired surrogate
raises ``UnicodeEncodeError`` before it can be hashed -- such a payload
never became a blob row in the first place. It can sit in
``trace_events.data``, which has no self-hash, and rewriting that column
is harmless.

Before running this on a deployment that predates the write-side
sanitizer, count the blob rows that would be rewritten. Neither query
writes anything:

    SELECT count(*) FROM trace_message_blobs
    WHERE CAST(message_data AS text) LIKE '%\\u0000%';

    SELECT count(*) FROM trace_checkpoint_blobs
    WHERE CAST(blob_data AS text) LIKE '%\\u0000%';

A non-zero count means some tasks will lose strict checkpoint recovery.
Pruning those tasks' checkpoint history first is the supported remedy and
turns a surprise 400 into a deliberate one.

The detection patterns assume a UTF8 server encoding, which is what the
shipped compose file runs. On a server in another encoding the cast also
rejects any escape naming a character outside that charset -- including a
valid surrogate pair, which the pair-stripping step deliberately removes
before matching -- so the cleanup would miss such a row and the ALTER
would fail on it. That is a safe failure (the transaction rolls back and
nothing is converted), but it needs the row cleaned by hand before the
upgrade can proceed. The read guard in ``web/api/monitor.py`` documents
the same limitation for the same patterns.

Two further shapes are legal in ``json`` and rejected by ``jsonb``, and
this cleanup handles neither: a number outside ``numeric``'s exponent
range, and nesting past ``jsonb``'s depth limit. No application write path
can produce either -- a Python float caps at ~1.8e308, ``json.dumps``
recurses within the interpreter's own limit -- so only a manual or
external writer could have planted one. Such a row fails the ALTER the
same safe way as the encoding case above.

Operationally this is the expensive kind of migration, and on a large
deployment it should be scheduled rather than slipped into a routine
restart. ``ALTER ... TYPE`` takes an ACCESS EXCLUSIVE lock and rewrites
every row of the table -- there is no in-place path from ``json`` to
``jsonb``, because the on-disk representation genuinely differs -- and the
cleanup scan that precedes it is an unindexed full-table regex pass over
the same rows. ``env.py`` sets ``transaction_per_migration=True`` on
PostgreSQL, so all three tables' scans and ALTERs run in *one*
transaction: the ``trace_events`` lock is taken first and held until the
two blob tables have converted too. Budget the window for the whole
migration, not per table, and note that ``trace_events`` is the table the
monitoring dashboard already scans. Pruning checkpoint history first
shortens every step.

Deployment ordering matters for the same reason. An old instance still
running without the write-side sanitizer can insert a fresh unconvertible
row after the cleanup scan has passed it but before the ALTER takes its
lock; the ALTER then fails and the whole migration rolls back. Nothing is
corrupted and a retry succeeds, but roll the application out fully before
migrating, or expect to run it more than once.

Revision ID: 20260813_trace_json_columns_to_jsonb
Revises: 20260812_seed_intercom_mcp_app
Create Date: 2026-08-13

"""

import json
import re
from decimal import Decimal
from typing import Any, Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260813_trace_json_columns_to_jsonb"
down_revision: Union[str, None] = "20260812_seed_intercom_mcp_app"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# (table, column) pairs that hold trace payloads. All three receive slices
# of the same event data, so a payload rejected by one would be rejected by
# any -- they must convert together or the write path splits behaviour.
TRACE_JSON_COLUMNS: tuple[tuple[str, str], ...] = (
    ("trace_events", "data"),
    ("trace_message_blobs", "message_data"),
    ("trace_checkpoint_blobs", "blob_data"),
)

# Detection patterns, copied verbatim from web/api/monitor.py
# (PG_ESCAPED_BACKSLASH / PG_SURROGATE_PAIR_PATTERN /
# PG_UNSAFE_ESCAPE_PATTERN) -- that file is the source of truth and
# documents why each normalization step is load-bearing. Bound as
# parameters, never interpolated, so SQL literal escaping cannot distort
# them. The migration inlines the values instead of importing them: a
# migration must stay runnable after the application module moves or
# changes.
ESCAPED_BACKSLASH = "\\\\"
ESCAPED_BACKSLASH_STANDIN = "__"
SURROGATE_PAIR_PATTERN = (
    "\\\\u[dD][89abAB][0-9a-fA-F]{2}\\\\u[dD][c-fC-F][0-9a-fA-F]{2}"
)
UNSAFE_ESCAPE_PATTERN = (
    "\\\\u0000|\\\\u[dD][89abAB][0-9a-fA-F]{2}|\\\\u[dD][c-fC-F][0-9a-fA-F]{2}"
)

# The Python-side rewrite: NUL and every surrogate code point become
# U+FFFD. Mirrors the code-point half of
# web/utils/json_payload_sanitizer.py, which documents the rule; duplicated
# here because a migration must not import application code that can drift.
# The sanitizer's number normalization has deliberately no counterpart here
# -- see _sanitize.
_UNSTORABLE_CODE_POINTS = re.compile("[\x00\ud800-\udfff]")
_REPLACEMENT_CHARACTER = "�"

# Placeholder marker for a number held out of json.dumps. Long and
# namespaced so a collision with real payload text is implausible; checked
# for anyway before any substitution runs.
_NUMBER_TOKEN_PREFIX = "__xagent_jsonb_migration_number_"

# Ids per cleanup batch. This bounds a list of integers, not payloads --
# the rewrite loop reads one payload at a time (see
# _rewrite_unconvertible_rows) -- so it can be generous without putting
# blob rows in memory.
REWRITE_BATCH_SIZE = 500


def _sanitize(value: Any) -> Any:
    """Rewrite one payload. Eager rather than copy-on-write, unlike the
    app-side sanitizer this mirrors: every row that reaches here matched
    the unconvertible-escape predicate, so the "clean payload, hand it back
    untouched" fast path that motivates the copy-on-write version there
    cannot fire here. ``re.sub`` already returns the original object when
    nothing matches, so only the container spine is rebuilt, never the text
    -- a few percent of the json.loads/json.dumps pair that brackets this
    call, and far less than one payload.

    Numbers are deliberately not touched. They arrive as ``Decimal`` (see
    ``_loads_preserving_numbers``) and leave verbatim, so ``jsonb`` applies
    its own normalization on ingest -- identically for a row this rewrites
    and a row it never looks at. Normalizing here instead would make the
    two diverge: a float64 round-trip renders ``1e25`` as
    10000000000000000905969664, where the cast produces 10^25.
    """
    if isinstance(value, str):
        return _UNSTORABLE_CODE_POINTS.sub(_REPLACEMENT_CHARACTER, value)
    if isinstance(value, dict):
        return {_sanitize(key): _sanitize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize(item) for item in value]
    return value


def _loads_preserving_numbers(payload_text: str) -> Any:
    """Parse without letting a number through float64.

    ``json.loads`` maps every JSON float onto a Python float, which silently
    truncates a literal carrying more precision than a double holds and
    turns ``1e1000`` into ``inf`` -- and ``json.dumps`` then writes
    ``Infinity``, which is not JSON and fails the UPDATE's cast. ``jsonb``
    itself does neither: its numbers are ``numeric``, so it keeps the exact
    literal. Parsing to ``Decimal`` matches that, and integers are already
    arbitrary precision.
    """
    return json.loads(payload_text, parse_float=Decimal)


def _dumps_preserving_numbers(value: Any, *, payload_text: str) -> str:
    """Serialize with every ``Decimal`` written as its own literal.

    ``json.dumps`` cannot emit a ``Decimal`` (and ``default=`` would quote
    it into a string), so each one is swapped for a placeholder string and
    the placeholder's *quoted* form is substituted back afterwards. The
    prefix is checked against the original payload first: a collision would
    otherwise let payload text be replaced by a number.
    """
    if _NUMBER_TOKEN_PREFIX in payload_text:
        raise RuntimeError(
            f"payload already contains {_NUMBER_TOKEN_PREFIX!r}; refusing to "
            "rewrite it rather than risk substituting into payload text"
        )

    literals: list[str] = []

    def tokenize(item: Any) -> Any:
        if isinstance(item, Decimal):
            literals.append(str(item))
            return f"{_NUMBER_TOKEN_PREFIX}{len(literals) - 1}__"
        if isinstance(item, dict):
            return {key: tokenize(sub) for key, sub in item.items()}
        if isinstance(item, list):
            return [tokenize(sub) for sub in item]
        return item

    text = json.dumps(tokenize(value))
    for index, literal in enumerate(literals):
        quoted = json.dumps(f"{_NUMBER_TOKEN_PREFIX}{index}__")
        if quoted not in text:
            raise RuntimeError(f"number placeholder {quoted} vanished during encode")
        text = text.replace(quoted, literal, 1)
    return text


def _table_exists(name: str) -> bool:
    return sa.inspect(op.get_bind()).has_table(name)


def _column_is_jsonb(table: str, column: str) -> bool:
    """Whether the column is already jsonb. Callers must have established
    that the table exists (both do, via _table_exists three lines up); a
    missing one raises NoSuchTableError out of here rather than resolving
    to a bool.

    That is deliberate, not an oversight. This answer carries opposite
    polarity in the two directions -- upgrade() reads False as "convert
    this column", downgrade() reads False as "skip it" -- so there is no
    default a missing table could safely collapse to. Swallowing it would
    tell upgrade() to proceed against a table that is not there, and the
    cleanup SELECT would raise the same failure one step later with a
    worse message.
    """
    for col in sa.inspect(op.get_bind()).get_columns(table):
        if col["name"] == column:
            return isinstance(col["type"], postgresql.JSONB)
    return False


def _rewrite_unconvertible_rows(table: str, column: str) -> None:
    """Replace unstorable escapes in rows the jsonb cast would reject.

    Rows carrying such payloads exist only where something slipped past the
    (younger) write-side sanitizer, so the scan usually matches nothing;
    the full-table regex pass runs once, here, not on any query path.

    The scan pages over matching *ids* and then reads one payload at a
    time, so peak memory is one payload rather than a batch of them. That
    split is the point: these values are whole trace events and checkpoint
    blobs, and nothing caps their size -- the payload truncation in
    ``core/agent/runtime.py`` applies to LLM-category events only, and
    checkpoints are ``system_update_general``. Holding a batch of them, or
    the whole matching set, would make the migration's memory a function of
    how much bad output a deployment happened to emit.

    Keyset pagination, not OFFSET: the loop rewrites the rows it just read,
    so they drop out of the predicate, and an OFFSET walk would skip past
    rows as the result set shifts under it.
    """
    bind = op.get_bind()
    # noqa: S608 throughout -- table/column come from TRACE_JSON_COLUMNS
    # above, never from user input, and every payload pattern is bound as a
    # parameter rather than interpolated.
    select_ids = sa.text(
        f"SELECT id FROM {table} "  # noqa: S608
        f"WHERE id > :after "
        f"AND regexp_replace("
        f"replace(CAST({column} AS text), :bs, :standin), "
        f":pair, '', 'g') ~ :unsafe "
        f"ORDER BY id LIMIT :limit"
    )
    select_payload = sa.text(
        f"SELECT CAST({column} AS text) FROM {table} WHERE id = :id"  # noqa: S608
    )
    # The column is still json at this point -- the ALTER runs after this
    # returns -- so the rewritten payload is cast back to json, not jsonb.
    update = sa.text(
        f"UPDATE {table} SET {column} = CAST(:payload AS json) "  # noqa: S608
        f"WHERE id = :id"
    )

    after = 0
    while True:
        row_ids = [
            row[0]
            for row in bind.execute(
                select_ids,
                {
                    "after": after,
                    "bs": ESCAPED_BACKSLASH,
                    "standin": ESCAPED_BACKSLASH_STANDIN,
                    "pair": SURROGATE_PAIR_PATTERN,
                    "unsafe": UNSAFE_ESCAPE_PATTERN,
                    "limit": REWRITE_BATCH_SIZE,
                },
            )
        ]
        if not row_ids:
            return
        for row_id in row_ids:
            payload_row = bind.execute(select_payload, {"id": row_id}).fetchone()
            if payload_row is None:
                # Deleted between the id scan and this read -- checkpoint
                # pruning runs concurrently against these very tables, and
                # a row that no longer exists needs no rewrite.
                continue
            payload_text = payload_row[0]
            cleaned = _sanitize(_loads_preserving_numbers(payload_text))
            bind.execute(
                update,
                {
                    "payload": _dumps_preserving_numbers(
                        cleaned, payload_text=payload_text
                    ),
                    "id": row_id,
                },
            )
        # A rewritten row no longer matches the predicate, so the next
        # batch would find these again only by id -- advance past them.
        after = row_ids[-1]


def upgrade() -> None:
    context = op.get_context()
    if context.dialect.name != "postgresql":
        return

    # Offline (--sql) generation runs against a MockConnection: reflection
    # and the row-cleanup SELECT are both impossible, so emit the plain
    # ALTERs a migration-built database needs (see the module docstring for
    # what happens if unconvertible rows are still present).
    if context.as_sql:
        for table, column in TRACE_JSON_COLUMNS:
            op.alter_column(
                table,
                column,
                type_=postgresql.JSONB(),
                existing_nullable=False,
                postgresql_using=f"{column}::jsonb",
            )
        return

    for table, column in TRACE_JSON_COLUMNS:
        # trace_events is created by Base.metadata.create_all(), not by a
        # migration; on a bare database none of the three tables exist yet.
        if not _table_exists(table):
            continue
        # A create_all-first startup already built the column as jsonb from
        # the model; converting again is a rerun no-op.
        if _column_is_jsonb(table, column):
            continue
        _rewrite_unconvertible_rows(table, column)
        op.alter_column(
            table,
            column,
            type_=postgresql.JSONB(),
            existing_nullable=False,
            postgresql_using=f"{column}::jsonb",
        )


def downgrade() -> None:
    context = op.get_context()
    if context.dialect.name != "postgresql":
        return

    if context.as_sql:
        for table, column in TRACE_JSON_COLUMNS:
            op.alter_column(
                table,
                column,
                type_=sa.JSON(),
                existing_nullable=False,
                postgresql_using=f"{column}::json",
            )
        return

    # jsonb -> json always casts cleanly; the U+FFFD rewrites are not (and
    # cannot be) undone.
    for table, column in TRACE_JSON_COLUMNS:
        if not _table_exists(table):
            continue
        if not _column_is_jsonb(table, column):
            continue
        op.alter_column(
            table,
            column,
            type_=sa.JSON(),
            existing_nullable=False,
            postgresql_using=f"{column}::json",
        )
