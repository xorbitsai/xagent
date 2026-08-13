"""Tests for migration 20260813_trace_json_columns_to_jsonb (#1248).

Following this repo's migration-test convention (see
tests/migrations/test_20260804_add_task_checkpoint_trace_event_anchor.py):
the trace tables' pre-migration shape is built directly with SQLAlchemy
Core table objects, stamped to the parent revision, and only the migration
under test is run against it.

The migration is PostgreSQL-only, so the SQLite tests pin the no-op (both
directions leave schema and rows untouched) and everything substantive --
the row cleanup, the type change, idempotence, and the downgrade -- runs
under ``@pytest.mark.postgresql`` against a real server, where ``json``
vs ``jsonb`` semantics exist. The payload shapes mirror the seed data of
``tests/web/api/test_monitor_postgresql.py``, the read-side test that
documents why these exact escapes are hazardous.
"""

from __future__ import annotations

import importlib.util
import os
from io import StringIO
from pathlib import Path
from types import ModuleType

import pytest
import sqlalchemy as sa
from alembic import command
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, text

from xagent.db.config import create_alembic_config

PARENT_REVISION = "20260812_seed_intercom_mcp_app"
TARGET_REVISION = "20260813_trace_json_columns_to_jsonb"

TRACE_JSON_COLUMNS = (
    ("trace_events", "data"),
    ("trace_message_blobs", "message_data"),
    ("trace_checkpoint_blobs", "blob_data"),
)
ALL_TABLES = ("alembic_version",) + tuple(t for t, _ in TRACE_JSON_COLUMNS)

# Escape sequences as they sit in the stored JSON *text*: six literal
# characters each, written with an escaped backslash so no editor or JSON
# layer decodes them prematurely.
BS = chr(92)
NUL_ESCAPE = BS + "u0000"
LONE_HIGH_ESCAPE = BS + "ud800"
LONE_LOW_ESCAPE = BS + "udc00"
# A valid pair (U+1F600) that must survive conversion as a real character.
PAIR_ESCAPE = BS + "ud83d" + BS + "ude00"
# Literal text that merely looks like an escape: the JSON carries a doubled
# backslash, so the value is the six characters backslash-u0000.
LITERAL_ESCAPE_TEXT = BS + BS + "u0000"
REPLACEMENT = chr(0xFFFD)


def _migration_module() -> ModuleType:
    import xagent.migrations as migrations_pkg

    migrations_dir = Path(next(iter(migrations_pkg.__path__)))
    path = migrations_dir / "versions" / f"{TARGET_REVISION}.py"
    spec = importlib.util.spec_from_file_location(TARGET_REVISION, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pre_migration_metadata() -> sa.MetaData:
    """The three trace tables, reduced to the columns the migration touches
    plus their primary keys -- not the full production schema."""
    metadata = sa.MetaData()
    sa.Table(
        "trace_events",
        metadata,
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("task_id", sa.Integer, nullable=False),
        sa.Column("event_id", sa.String(255), nullable=False),
        sa.Column("event_type", sa.String(100), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("data", sa.JSON, nullable=False),
    )
    # The hash and bytes columns are carried here even though the migration
    # never writes them: that it leaves them verbatim is a contract (the
    # hash is the blob's identity, referenced from trace_events.data), and
    # a fixture without them cannot pin it.
    sa.Table(
        "trace_message_blobs",
        metadata,
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("task_id", sa.Integer, nullable=False),
        sa.Column("execution_id", sa.String(255), nullable=False),
        sa.Column("message_hash", sa.String(80), nullable=False),
        sa.Column("message_data", sa.JSON, nullable=False),
        sa.Column("message_bytes", sa.Integer, nullable=False),
    )
    sa.Table(
        "trace_checkpoint_blobs",
        metadata,
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("task_id", sa.Integer, nullable=False),
        sa.Column("execution_id", sa.String(255), nullable=False),
        sa.Column("blob_kind", sa.String(255), nullable=False),
        sa.Column("blob_hash", sa.String(80), nullable=False),
        sa.Column("blob_data", sa.JSON, nullable=False),
        sa.Column("blob_bytes", sa.Integer, nullable=False),
    )
    return metadata


def _stamp_parent_revision(engine: sa.engine.Engine) -> None:
    with engine.begin() as conn:
        conn.execute(
            text("CREATE TABLE alembic_version (version_num VARCHAR(255) NOT NULL)")
        )
        conn.execute(
            text(
                "INSERT INTO alembic_version (version_num) VALUES "
                f"('{PARENT_REVISION}')"
            )
        )


def _insert_trace_event(
    conn: sa.engine.Connection, *, row_id: int, payload_json_text: str
) -> None:
    """Insert with the payload cast from raw JSON text, so escape sequences
    reach the column exactly as written -- the json type accepts them, which
    is the bug this migration removes."""
    conn.execute(
        text(
            "INSERT INTO trace_events "
            "(id, task_id, event_id, event_type, timestamp, data) "
            "VALUES (:id, 1, :event_id, 'llm_call_start', "
            "'2026-08-13 00:00:00', CAST(:payload AS json))"
        ),
        {"id": row_id, "event_id": f"evt-{row_id}", "payload": payload_json_text},
    )


def _insert_message_blob(
    conn: sa.engine.Connection,
    *,
    row_id: int,
    payload_json_text: str,
    message_hash: str | None = None,
    message_bytes: int = 0,
) -> None:
    conn.execute(
        text(
            "INSERT INTO trace_message_blobs "
            "(id, task_id, execution_id, message_hash, message_data, "
            "message_bytes) VALUES (:id, 1, 'exec-1', :hash, "
            "CAST(:payload AS json), :bytes)"
        ),
        {
            "id": row_id,
            "hash": message_hash or f"sha256:msg-{row_id}",
            "payload": payload_json_text,
            "bytes": message_bytes,
        },
    )


def _insert_checkpoint_blob(
    conn: sa.engine.Connection,
    *,
    row_id: int,
    payload_json_text: str,
    blob_hash: str | None = None,
    blob_bytes: int = 0,
) -> None:
    conn.execute(
        text(
            "INSERT INTO trace_checkpoint_blobs "
            "(id, task_id, execution_id, blob_kind, blob_hash, blob_data, "
            "blob_bytes) VALUES (:id, 1, 'exec-1', 'context.metadata', "
            ":hash, CAST(:payload AS json), :bytes)"
        ),
        {
            "id": row_id,
            "hash": blob_hash or f"sha256:blob-{row_id}",
            "payload": payload_json_text,
            "bytes": blob_bytes,
        },
    )


def _alembic_config(engine: sa.engine.Engine):
    config = create_alembic_config(engine)
    config.set_main_option(
        "sqlalchemy.url", engine.url.render_as_string(hide_password=False)
    )
    return config


def _upgrade(engine: sa.engine.Engine, revision: str = TARGET_REVISION) -> None:
    command.upgrade(_alembic_config(engine), revision)


def _downgrade(engine: sa.engine.Engine, revision: str = PARENT_REVISION) -> None:
    command.downgrade(_alembic_config(engine), revision)


@pytest.fixture
def sqlite_engine(tmp_path: Path) -> sa.engine.Engine:
    engine = create_engine(f"sqlite:///{tmp_path / 'jsonb.db'}")
    _pre_migration_metadata().create_all(bind=engine)
    _stamp_parent_revision(engine)
    return engine


class TestSqliteNoOp:
    def test_upgrade_leaves_schema_and_rows_untouched(
        self, sqlite_engine: sa.engine.Engine
    ) -> None:
        payload = '{"model_name": "gpt-4o"}'
        with sqlite_engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO trace_events "
                    "(id, task_id, event_id, event_type, timestamp, data) "
                    "VALUES (1, 1, 'evt-1', 'llm_call_start', "
                    "'2026-08-13 00:00:00', :payload)"
                ),
                {"payload": payload},
            )
        before = {
            c["name"]: str(c["type"])
            for c in sa.inspect(sqlite_engine).get_columns("trace_events")
        }

        _upgrade(sqlite_engine)

        after = {
            c["name"]: str(c["type"])
            for c in sa.inspect(sqlite_engine).get_columns("trace_events")
        }
        assert after == before
        with sqlite_engine.begin() as conn:
            stored = conn.execute(
                text("SELECT data FROM trace_events WHERE id = 1")
            ).scalar_one()
        assert stored == payload

    def test_downgrade_leaves_schema_and_rows_untouched(
        self, sqlite_engine: sa.engine.Engine
    ) -> None:
        """Symmetric with the upgrade test above: the downgrade must also
        assert, not merely run without raising."""
        payload = '{"model_name": "gpt-4o"}'
        with sqlite_engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO trace_events "
                    "(id, task_id, event_id, event_type, timestamp, data) "
                    "VALUES (1, 1, 'evt-1', 'llm_call_start', "
                    "'2026-08-13 00:00:00', :payload)"
                ),
                {"payload": payload},
            )
        before = {
            c["name"]: str(c["type"])
            for c in sa.inspect(sqlite_engine).get_columns("trace_events")
        }

        _upgrade(sqlite_engine)
        _downgrade(sqlite_engine)

        after = {
            c["name"]: str(c["type"])
            for c in sa.inspect(sqlite_engine).get_columns("trace_events")
        }
        assert after == before
        with sqlite_engine.begin() as conn:
            stored = conn.execute(
                text("SELECT data FROM trace_events WHERE id = 1")
            ).scalar_one()
        assert stored == payload


def _postgres_url() -> str | None:
    return os.getenv("XAGENT_TEST_POSTGRES_URL") or os.getenv(
        "POSTGRES_TEST_DATABASE_URL"
    )


@pytest.fixture
def postgres_engine():
    url = _postgres_url()
    if not url:
        pytest.skip("XAGENT_TEST_POSTGRES_URL is not set")
    engine = create_engine(url)
    with engine.begin() as conn:
        for table in ALL_TABLES:
            conn.execute(text(f"DROP TABLE IF EXISTS {table} CASCADE"))
    _pre_migration_metadata().create_all(bind=engine)
    _stamp_parent_revision(engine)
    yield engine
    with engine.begin() as conn:
        for table in ALL_TABLES:
            conn.execute(text(f"DROP TABLE IF EXISTS {table} CASCADE"))
    engine.dispose()


def _column_type(engine, table: str, column: str) -> str:
    with engine.begin() as conn:
        return conn.execute(
            text(
                "SELECT data_type FROM information_schema.columns "
                "WHERE table_name = :t AND column_name = :c"
            ),
            {"t": table, "c": column},
        ).scalar_one()


@pytest.mark.postgresql
class TestUpgradePostgres:
    def test_converts_all_three_columns_to_jsonb(self, postgres_engine) -> None:
        _upgrade(postgres_engine)

        for table, column in TRACE_JSON_COLUMNS:
            assert _column_type(postgres_engine, table, column) == "jsonb"

    def test_rewrites_rows_the_cast_would_reject(self, postgres_engine) -> None:
        with postgres_engine.begin() as conn:
            _insert_trace_event(
                conn, row_id=1, payload_json_text='{"v": "a' + NUL_ESCAPE + 'b"}'
            )
            _insert_trace_event(
                conn, row_id=2, payload_json_text='{"v": "x' + LONE_HIGH_ESCAPE + '"}'
            )
            _insert_trace_event(
                conn, row_id=3, payload_json_text='{"v": "y' + LONE_LOW_ESCAPE + '"}'
            )

        _upgrade(postgres_engine)

        with postgres_engine.begin() as conn:
            values = dict(
                conn.execute(
                    text("SELECT id, data->>'v' FROM trace_events ORDER BY id")
                ).fetchall()
            )
        assert values == {
            1: f"a{REPLACEMENT}b",
            2: f"x{REPLACEMENT}",
            3: f"y{REPLACEMENT}",
        }

    def test_rewrites_a_payload_mixing_valid_pairs_with_orphans(
        self, postgres_engine
    ) -> None:
        """The shape the pair-strip and the unsafe predicate only interact
        on: a valid surrogate pair sitting next to an orphan. The pair must
        be stripped before matching, or the orphan behind it is missed and
        the ALTER fails on the row."""
        with postgres_engine.begin() as conn:
            _insert_trace_event(
                conn,
                row_id=1,
                payload_json_text=(
                    '{"v": "' + PAIR_ESCAPE + LONE_LOW_ESCAPE + PAIR_ESCAPE + '"}'
                ),
            )
            _insert_trace_event(
                conn,
                row_id=2,
                payload_json_text=(
                    '{"v": "' + LITERAL_ESCAPE_TEXT + LONE_HIGH_ESCAPE + '"}'
                ),
            )

        _upgrade(postgres_engine)

        with postgres_engine.begin() as conn:
            values = dict(
                conn.execute(
                    text("SELECT id, data->>'v' FROM trace_events ORDER BY id")
                ).fetchall()
            )
        emoji = chr(0x1F600)
        # Row 1: both valid pairs survive as real characters, the orphan
        # between them is replaced. Row 2: text that only looks like an
        # escape is preserved, the orphan after it is still caught.
        assert values == {
            1: f"{emoji}{REPLACEMENT}{emoji}",
            2: BS + "u0000" + REPLACEMENT,
        }

    def test_rewrites_a_large_float_alongside_the_escape(self, postgres_engine) -> None:
        """The migration's own float normalization: a row selected for its
        escape also gets its exponent-notation floats converted, so the
        rewritten payload matches what jsonb would hand back."""
        with postgres_engine.begin() as conn:
            _insert_trace_event(
                conn,
                row_id=1,
                payload_json_text='{"v": "n' + NUL_ESCAPE + '", "cost": 1e16}',
            )

        _upgrade(postgres_engine)

        with postgres_engine.begin() as conn:
            payload = conn.execute(
                text("SELECT data FROM trace_events WHERE id = 1")
            ).scalar_one()
        assert payload["v"] == f"n{REPLACEMENT}"
        assert payload["cost"] == 10000000000000000
        assert isinstance(payload["cost"], int)

    def test_rewrite_preserves_numbers_a_float_round_trip_would_damage(
        self, postgres_engine
    ) -> None:
        """The rewrite must not launder numbers through float64. jsonb's
        numbers are numeric, so it keeps the literal exactly; parsing to
        float would truncate the long decimal, turn 1e1000 into inf (which
        json.dumps writes as ``Infinity``, failing the rewrite's own cast),
        and render 1e25 as 10000000000000000905969664 rather than 10^25.
        """
        with postgres_engine.begin() as conn:
            _insert_trace_event(
                conn,
                row_id=1,
                payload_json_text=(
                    '{"v": "n' + NUL_ESCAPE + '", '
                    '"precise": 0.123456789012345678901, '
                    '"huge": 1e1000, "exp": 1e25}'
                ),
            )

        _upgrade(postgres_engine)

        with postgres_engine.begin() as conn:
            row = conn.execute(
                text(
                    "SELECT data->>'v', data->>'precise', data->>'huge', "
                    "data->>'exp' FROM trace_events WHERE id = 1"
                )
            ).one()
        assert row[0] == f"n{REPLACEMENT}"
        # Exactly what a plain CAST would have produced for these numbers.
        assert row[1] == "0.123456789012345678901"
        assert row[2] == "1" + "0" * 1000
        assert row[3] == "1" + "0" * 25

    def test_untouched_row_and_rewritten_row_agree_on_numbers(
        self, postgres_engine
    ) -> None:
        """The property that makes the choice defensible: whether or not a
        row is rewritten must not change how its numbers land."""
        numbers = '"precise": 0.123456789012345678901, "exp": 1e25'
        with postgres_engine.begin() as conn:
            _insert_trace_event(
                conn,
                row_id=1,
                payload_json_text='{"v": "n' + NUL_ESCAPE + '", ' + numbers + "}",
            )
            _insert_trace_event(
                conn, row_id=2, payload_json_text='{"v": "clean", ' + numbers + "}"
            )

        _upgrade(postgres_engine)

        with postgres_engine.begin() as conn:
            rows = conn.execute(
                text(
                    "SELECT data->>'precise', data->>'exp' FROM trace_events "
                    "ORDER BY id"
                )
            ).fetchall()
        assert rows[0] == rows[1]

    def test_benign_payloads_convert_unrewritten(self, postgres_engine) -> None:
        with postgres_engine.begin() as conn:
            _insert_trace_event(
                conn, row_id=1, payload_json_text='{"v": "' + PAIR_ESCAPE + '"}'
            )
            _insert_trace_event(
                conn,
                row_id=2,
                payload_json_text='{"v": "' + LITERAL_ESCAPE_TEXT + '"}',
            )

        _upgrade(postgres_engine)

        with postgres_engine.begin() as conn:
            values = dict(
                conn.execute(
                    text("SELECT id, data->>'v' FROM trace_events ORDER BY id")
                ).fetchall()
            )
        # The valid pair decodes to the astral character; the literal text
        # keeps its backslash. Neither gains a replacement character.
        assert values == {1: chr(0x1F600), 2: BS + "u0000"}

    def test_cleans_the_blob_tables_too(self, postgres_engine) -> None:
        with postgres_engine.begin() as conn:
            _insert_message_blob(
                conn, row_id=1, payload_json_text='{"m": "' + LONE_HIGH_ESCAPE + '"}'
            )
            _insert_checkpoint_blob(
                conn, row_id=1, payload_json_text='{"b": "' + NUL_ESCAPE + '"}'
            )

        _upgrade(postgres_engine)

        with postgres_engine.begin() as conn:
            message = conn.execute(
                text("SELECT message_data->>'m' FROM trace_message_blobs")
            ).scalar_one()
            blob = conn.execute(
                text("SELECT blob_data->>'b' FROM trace_checkpoint_blobs")
            ).scalar_one()
        assert message == REPLACEMENT
        assert blob == REPLACEMENT

    def test_blob_hash_columns_are_left_verbatim(self, postgres_engine) -> None:
        """The hash is the blob's identity, not a checksum of the column:
        ``trace_events.data`` references a blob by embedding the hash value,
        so rewriting it would make the row unreachable instead of merely
        unverifiable, and could collide with a post-sanitizer row under
        uq_trace_message_blobs_task_hash. The migration therefore rewrites
        the payload and leaves the hash alone -- a deliberate choice with a
        documented cost (see the module docstring), which this pins so it
        cannot be changed silently.
        """
        with postgres_engine.begin() as conn:
            _insert_message_blob(
                conn,
                row_id=1,
                payload_json_text='{"m": "n' + NUL_ESCAPE + '"}',
                message_hash="sha256:deadbeef",
                message_bytes=999,
            )

        _upgrade(postgres_engine)

        with postgres_engine.begin() as conn:
            row = conn.execute(
                text(
                    "SELECT message_hash, message_bytes, message_data->>'m' "
                    "FROM trace_message_blobs WHERE id = 1"
                )
            ).one()
        assert row[0] == "sha256:deadbeef"
        assert row[1] == 999
        assert row[2] == f"n{REPLACEMENT}"

    def test_jsonb_rejects_the_hazard_after_upgrade(self, postgres_engine) -> None:
        """The invariant the whole migration exists to establish: after it,
        the column can no longer hold a payload ``->>`` cannot read."""
        _upgrade(postgres_engine)

        with pytest.raises(sa.exc.DBAPIError):
            with postgres_engine.begin() as conn:
                _insert_trace_event(
                    conn,
                    row_id=99,
                    payload_json_text='{"v": "' + LONE_HIGH_ESCAPE + '"}',
                )

    def test_upgrade_is_idempotent_when_rerun(self, postgres_engine) -> None:
        migration = _migration_module()
        _upgrade(postgres_engine)

        # command.upgrade() short-circuits once stamped, so run the
        # migration body itself a second time over the live connection.
        with postgres_engine.begin() as conn:
            context = MigrationContext.configure(conn)
            with Operations.context(context):
                migration.upgrade()

        for table, column in TRACE_JSON_COLUMNS:
            assert _column_type(postgres_engine, table, column) == "jsonb"

    def test_missing_tables_are_skipped(self, postgres_engine) -> None:
        """A bare database has none of the three tables when migrations run
        (they are create_all-owned); each must be skipped independently."""
        with postgres_engine.begin() as conn:
            conn.execute(text("DROP TABLE trace_checkpoint_blobs"))

        _upgrade(postgres_engine)

        assert _column_type(postgres_engine, "trace_events", "data") == "jsonb"


@pytest.mark.postgresql
class TestDowngradePostgres:
    def test_downgrade_restores_json(self, postgres_engine) -> None:
        _upgrade(postgres_engine)
        _downgrade(postgres_engine)

        for table, column in TRACE_JSON_COLUMNS:
            assert _column_type(postgres_engine, table, column) == "json"

    def test_downgrade_preserves_rows(self, postgres_engine) -> None:
        with postgres_engine.begin() as conn:
            _insert_trace_event(
                conn, row_id=1, payload_json_text='{"model_name": "gpt-4o"}'
            )

        _upgrade(postgres_engine)
        _downgrade(postgres_engine)

        with postgres_engine.begin() as conn:
            value = conn.execute(
                text("SELECT data->>'model_name' FROM trace_events WHERE id = 1")
            ).scalar_one()
        assert value == "gpt-4o"


def _offline_sql(dialect_name: str, operation: str) -> str:
    migration = _migration_module()
    output = StringIO()
    context = MigrationContext.configure(
        dialect_name=dialect_name,
        opts={"as_sql": True, "output_buffer": output},
    )

    with Operations.context(context):
        getattr(migration, operation)()

    return output.getvalue()


class TestOfflineSql:
    def test_postgresql_upgrade_emits_the_three_alters(self) -> None:
        sql = _offline_sql("postgresql", "upgrade")

        for table, column in TRACE_JSON_COLUMNS:
            assert (
                f"ALTER TABLE {table} ALTER COLUMN {column} TYPE JSONB "
                f"USING {column}::jsonb" in sql
            )
        # No row cleanup offline: a MockConnection cannot read rows. The
        # module docstring documents the operator-facing consequence.
        assert "UPDATE" not in sql
        assert "SELECT" not in sql

    def test_postgresql_downgrade_restores_json(self) -> None:
        sql = _offline_sql("postgresql", "downgrade")

        for table, column in TRACE_JSON_COLUMNS:
            assert (
                f"ALTER TABLE {table} ALTER COLUMN {column} TYPE JSON "
                f"USING {column}::json" in sql
            )

    @pytest.mark.parametrize("operation", ["upgrade", "downgrade"])
    def test_sqlite_emits_nothing(self, operation: str) -> None:
        sql = _offline_sql("sqlite", operation)

        assert "ALTER TABLE" not in sql

    @pytest.mark.parametrize("dialect", ["sqlite", "postgresql"])
    @pytest.mark.parametrize("operation", ["upgrade", "downgrade"])
    def test_offline_sql_carries_no_bind_parameters(
        self, dialect: str, operation: str
    ) -> None:
        sql = _offline_sql(dialect, operation)

        assert "%(" not in sql
        assert ":param" not in sql
        assert "?" not in sql


def _batch_span() -> int:
    """One full id batch plus a remainder, so the cleanup loop runs twice.

    Derived from the migration's own constant rather than hardcoded: the
    point of these tests is that the loop iterates, which stops being true
    if someone raises REWRITE_BATCH_SIZE past a fixed row count here.
    """
    return _migration_module().REWRITE_BATCH_SIZE + 3


def _insert_many(conn: sa.engine.Connection, payloads: dict[int, str]) -> None:
    """Seed rows in one round trip -- a batch-span worth of single INSERTs
    dominates these tests' runtime otherwise."""
    conn.execute(
        text(
            "INSERT INTO trace_events "
            "(id, task_id, event_id, event_type, timestamp, data) "
            "VALUES (:id, 1, :event_id, 'llm_call_start', "
            "'2026-08-13 00:00:00', CAST(:payload AS json))"
        ),
        [
            {"id": row_id, "event_id": f"evt-{row_id}", "payload": payload}
            for row_id, payload in payloads.items()
        ],
    )


@pytest.mark.postgresql
class TestBatchedCleanup:
    """The cleanup pages over matching ids and reads one payload at a time,
    so neither the whole matching set nor a batch of payloads is ever
    resident. The tests above stay inside a single batch; this class forces
    the loop to iterate, which is the only way the keyset advance is
    covered.
    """

    def test_rewrites_more_rows_than_one_batch(self, postgres_engine) -> None:
        row_count = _batch_span()
        with postgres_engine.begin() as conn:
            _insert_many(
                conn,
                {
                    row_id: '{"v": "n' + NUL_ESCAPE + '"}'
                    for row_id in range(1, row_count + 1)
                },
            )

        _upgrade(postgres_engine)

        with postgres_engine.begin() as conn:
            distinct = conn.execute(
                text("SELECT DISTINCT data->>'v' FROM trace_events")
            ).fetchall()
            total = conn.execute(text("SELECT count(*) FROM trace_events")).scalar_one()
        # Every row rewritten, none skipped by the keyset advance and none
        # left behind for the cast to choke on.
        assert total == row_count
        assert distinct == [(f"n{REPLACEMENT}",)]

    def test_clean_rows_between_poisoned_ones_are_untouched(
        self, postgres_engine
    ) -> None:
        """The id scan matches only poisoned rows, so the keyset advance
        steps over clean rows rather than rewriting them."""
        span = _batch_span()
        with postgres_engine.begin() as conn:
            _insert_many(
                conn,
                {
                    row_id: (
                        '{"v": "bad' + NUL_ESCAPE + '"}'
                        if row_id % 2
                        else '{"v": "' + PAIR_ESCAPE + '"}'
                    )
                    for row_id in range(1, span + 1)
                },
            )

        _upgrade(postgres_engine)

        with postgres_engine.begin() as conn:
            good = conn.execute(
                text("SELECT count(*) FROM trace_events WHERE data->>'v' = :v"),
                {"v": chr(0x1F600)},
            ).scalar_one()
            bad = conn.execute(
                text("SELECT count(*) FROM trace_events WHERE data->>'v' = :v"),
                {"v": f"bad{REPLACEMENT}"},
            ).scalar_one()
        assert good == span // 2
        assert bad == span - span // 2
