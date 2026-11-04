"""The retention anchor's migration: backfill correctness and idempotence.

Runs on SQLite and on a disposable PostgreSQL through the shared ``engine``
fixture. Both matter: the backfill's correlated ``MAX()`` subquery and its
``UPDATE ... WHERE id IN (SELECT ... LIMIT)`` batching are the two constructs
most likely to behave differently across the pair.
"""

from __future__ import annotations

import importlib
import pathlib
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

import pytest
import sqlalchemy as sa
from alembic import op
from alembic.migration import MigrationContext
from alembic.operations import Operations

from tests.web.services.task_database_shared import engine as engine_fixture

engine = engine_fixture

MODULE = "xagent.migrations.versions.20260922_task_last_activity_at"

migration_module = importlib.import_module(MODULE)
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

NOW = datetime(2026, 9, 22, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def migration():
    return importlib.import_module(MODULE)


@contextmanager
def _migration_env(engine):
    """Drive the revision the way ``env.py`` drives it.

    Not ``engine.begin()``. A connection already inside a transaction when
    ``MigrationContext.configure`` runs is recorded as an *external*
    transaction, and ``begin_transaction()`` then returns a nullcontext
    without creating ``_transaction`` -- so ``autocommit_block()`` asserts on
    a transaction it was never given. ``env.py`` commits the reflection
    transaction before ``configure()`` for exactly this reason, and this
    revision's PostgreSQL path depends on that block, so the harness has to
    mirror it or it tests a shape production never runs.
    """
    with engine.connect() as connection:
        context = MigrationContext.configure(connection)
        with Operations.context(context), context.begin_transaction():
            yield connection


def _schema(*, with_messages: bool) -> tuple[sa.MetaData, sa.Table, sa.Table | None]:
    """The pre-migration shape of the two tables the backfill reads.

    Deliberately minimal rather than the real metadata: the migration must work
    against a database whose ``tasks`` table predates every column this PR
    adds, which is exactly what an upgrade encounters.
    """
    metadata = sa.MetaData()
    tasks = sa.Table(
        "tasks",
        metadata,
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("created_at", sa.DateTime(timezone=True)),
    )
    messages = None
    if with_messages:
        messages = sa.Table(
            "task_chat_messages",
            metadata,
            sa.Column("id", sa.Integer, primary_key=True),
            sa.Column(
                "task_id", sa.Integer, sa.ForeignKey("tasks.id", ondelete="CASCADE")
            ),
            sa.Column("created_at", sa.DateTime(timezone=True)),
        )
    return metadata, tasks, messages


def _anchors(connection: sa.Connection) -> dict[int, datetime | None]:
    # Typed columns, not a bare ``text()``: SQLite hands back the stored
    # string unless SQLAlchemy is told the column is a DateTime, and the
    # whole point of these assertions is to compare instants.
    rows = connection.execute(
        sa.select(
            sa.column("id", sa.Integer),
            sa.column("last_activity_at", sa.DateTime(timezone=True)),
        )
        .select_from(sa.table("tasks"))
        .order_by(sa.column("id"))
    ).all()
    return {
        row[0]: (
            row[1].replace(tzinfo=timezone.utc)
            if row[1] is not None and row[1].tzinfo is None
            else row[1]
        )
        for row in rows
    }


def test_harness_supplies_what_an_autocommit_block_needs(engine):
    """Pin the harness contract, not just the migration's.

    This revision's PostgreSQL path runs the backfill inside
    ``autocommit_block()``, and that block commits the transaction Alembic
    owns. A harness that hands ``MigrationContext`` a connection already in
    someone else's transaction gets a nullcontext from
    ``begin_transaction()`` instead, and the block then asserts on a
    ``_transaction`` it never received.

    SQLite cannot catch that through ``upgrade()``, because the dialect
    branch skips the block there -- which is how a harness bug reached CI as
    eight PostgreSQL failures while every local run was green. The assertion
    itself is dialect-independent, so entering a block directly pins it here.
    """
    with _migration_env(engine) as connection:
        with op.get_context().autocommit_block():
            connection.execute(sa.text("SELECT 1"))


def _script_statements(script: str) -> list[str]:
    """The migration's own statements from a rendered script.

    Alembic's bookkeeping is dropped: ``alembic_version`` is the harness's
    table, not this schema's, and BEGIN/COMMIT are supplied by the connection
    the test already holds.
    """
    # Strip comment lines before splitting: Alembic writes
    # ``-- Running upgrade X -> Y`` on its own line with no semicolon, so a
    # naive split glues it onto the statement that follows and the whole
    # thing then looks like a comment.
    body = "\n".join(
        line for line in script.splitlines() if not line.lstrip().startswith("--")
    )
    statements = []
    for raw in body.split(";"):
        statement = " ".join(raw.split())
        if not statement or statement in {"BEGIN", "COMMIT"}:
            continue
        # Drop only Alembic's own bookkeeping write, matched exactly. A
        # substring test for "alembic_version" would silently swallow any
        # other statement that merely mentions it, which is how a script
        # carrying an extra anchor-clobbering UPDATE could still pass.
        if statement.startswith("UPDATE alembic_version SET version_num"):
            continue
        statements.append(statement)
    return statements


def _has_anchor_column(connection: sa.Connection) -> bool:
    return any(
        c["name"] == "last_activity_at"
        for c in sa.inspect(connection).get_columns("tasks")
    )


def test_backfill_uses_last_message_and_falls_back_to_created_at(engine, migration):
    """Anchor = last message, or the task's own creation time when it has none.

    The fallback is not cosmetic: a NULL anchor compares as NULL against any
    cutoff, so a message-less task left NULL would never expire.
    """
    metadata, tasks, messages = _schema(with_messages=True)
    metadata.create_all(engine)
    with _migration_env(engine) as connection:
        connection.execute(
            tasks.insert(),
            [
                {"id": 1, "created_at": NOW - timedelta(days=500)},
                {"id": 2, "created_at": NOW - timedelta(days=300)},
            ],
        )
        connection.execute(
            messages.insert(),
            [
                {"id": 10, "task_id": 1, "created_at": NOW - timedelta(days=400)},
                {"id": 11, "task_id": 1, "created_at": NOW - timedelta(days=120)},
            ],
        )

        migration.upgrade()

        anchors = _anchors(connection)
        # Task 1 has messages: the most recent one wins, not the first.
        assert anchors[1] == NOW - timedelta(days=120)
        # Task 2 never carried a message: it ages from its own creation.
        assert anchors[2] == NOW - timedelta(days=300)


def test_upgrade_is_idempotent_and_finishes_a_partial_backfill(engine, migration):
    """Re-running must not re-stamp finished rows, but must finish unfinished ones.

    A run that added the column and died leaves NULLs behind, so the backfill
    is unconditional rather than gated on the column being new. That makes the
    "already done" case a no-op instead of a rewrite.
    """
    metadata, tasks, messages = _schema(with_messages=True)
    metadata.create_all(engine)
    with _migration_env(engine) as connection:
        connection.execute(
            tasks.insert(), {"id": 1, "created_at": NOW - timedelta(days=9)}
        )
        connection.execute(
            messages.insert(),
            {"id": 10, "task_id": 1, "created_at": NOW - timedelta(days=5)},
        )
        migration.upgrade()
        after_first = _anchors(connection)

        # A row that arrives (or is reset) after the first run.
        connection.execute(
            tasks.insert(), {"id": 2, "created_at": NOW - timedelta(days=3)}
        )
        migration.upgrade()
        after_second = _anchors(connection)

        assert after_second[1] == after_first[1]
        assert after_second[2] == NOW - timedelta(days=3)

        # A third run changes nothing at all.
        migration.upgrade()
        assert _anchors(connection) == after_second


def test_backfill_pages_past_one_batch(engine, migration, monkeypatch):
    """The loop must converge on a set larger than one batch, not stop at it."""
    monkeypatch.setattr(migration, "BACKFILL_BATCH_SIZE", 3)
    metadata, tasks, _ = _schema(with_messages=False)
    metadata.create_all(engine)
    with _migration_env(engine) as connection:
        connection.execute(
            tasks.insert(),
            [
                {"id": index, "created_at": NOW - timedelta(days=index)}
                for index in range(1, 11)
            ],
        )
        migration.upgrade()
        anchors = _anchors(connection)
        assert len(anchors) == 10
        assert all(value is not None for value in anchors.values())


def test_column_is_added_without_a_server_default(engine, migration):
    """A default would stamp existing rows with the migration's own clock.

    That is the value the backfill exists to avoid: it would push every
    historical task's expiry out by a full retention period.
    """
    metadata, _, _ = _schema(with_messages=False)
    metadata.create_all(engine)
    with _migration_env(engine) as connection:
        migration.upgrade()
        column = next(
            c
            for c in sa.inspect(connection).get_columns("tasks")
            if c["name"] == "last_activity_at"
        )
        assert column["default"] is None
        assert column["nullable"] is True


def test_downgrade_removes_the_column_and_is_repeatable(engine, migration):
    metadata, _, _ = _schema(with_messages=False)
    metadata.create_all(engine)
    with _migration_env(engine) as connection:
        migration.upgrade()
        assert any(
            c["name"] == "last_activity_at"
            for c in sa.inspect(connection).get_columns("tasks")
        )
        migration.downgrade()
        migration.downgrade()
        assert not any(
            c["name"] == "last_activity_at"
            for c in sa.inspect(connection).get_columns("tasks")
        )
        migration.upgrade()
        assert any(
            c["name"] == "last_activity_at"
            for c in sa.inspect(connection).get_columns("tasks")
        )


def test_alembic_only_missing_tables_are_left_to_metadata(engine, migration):
    """No ``tasks`` table means an Alembic-only install; metadata owns it."""
    with _migration_env(engine) as connection:
        migration.upgrade()
        assert not sa.inspect(connection).has_table("tasks")
        migration.downgrade()


def test_backfill_without_the_messages_table_falls_back_to_created_at(
    engine, migration
):
    """``tasks`` can exist before ``task_chat_messages`` in an Alembic-only run."""
    metadata, tasks, _ = _schema(with_messages=False)
    metadata.create_all(engine)
    with _migration_env(engine) as connection:
        connection.execute(
            tasks.insert(), {"id": 1, "created_at": NOW - timedelta(days=7)}
        )
        migration.upgrade()
        assert _anchors(connection)[1] == NOW - timedelta(days=7)


def test_backfill_converges_when_no_source_can_supply_an_instant(engine, migration):
    """A row with nothing to anchor on must terminate the loop, not spin on it.

    Selecting the remaining NULL set on every pass looks equivalent to paging
    and is not: a row written to NULL stays in that set and is handed back
    forever, so the revision dies at the batch ceiling and rolls back --
    deterministically, on every retry. The row is left NULL, which the
    predicate reads as "no anchor" and refuses to expire.
    """
    metadata, tasks, messages = _schema(with_messages=True)
    metadata.create_all(engine)
    with _migration_env(engine) as connection:
        connection.execute(
            tasks.insert(),
            [
                {"id": 1, "created_at": None},
                {"id": 2, "created_at": NOW - timedelta(days=3)},
            ],
        )
        migration.upgrade()

        anchors = _anchors(connection)
        assert anchors[1] is None
        assert anchors[2] == NOW - timedelta(days=3)

        # And a message arriving later still anchors the unanchored row.
        connection.execute(
            messages.insert(),
            {"id": 10, "task_id": 1, "created_at": NOW - timedelta(days=1)},
        )
        migration.upgrade()
        assert _anchors(connection)[1] == NOW - timedelta(days=1)


def test_upgrade_survives_a_tasks_table_with_no_created_at(engine, migration):
    """A legacy ``tasks`` shape must not take the whole upgrade chain down.

    ``tasks`` predates most of its columns, and tests/migration/test_migration.py
    builds exactly this shape. Naming ``tasks.created_at`` unconditionally
    fails the statement and every later revision with it.
    """
    metadata = sa.MetaData()
    sa.Table(
        "tasks",
        metadata,
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("source", sa.String(20)),
    )
    metadata.create_all(engine)
    with _migration_env(engine) as connection:
        connection.execute(sa.text("INSERT INTO tasks (id, source) VALUES (1, 'sdk')"))
        migration.upgrade()
        assert _has_anchor_column(connection)
        # Nothing to derive an anchor from, so the row stays unanchored
        # rather than being stamped with a guess.
        assert _anchors(connection)[1] is None


def test_backfill_uses_messages_when_tasks_has_no_created_at(engine, migration):
    """Missing ``tasks.created_at`` must not disable the message source too."""
    metadata = sa.MetaData()
    sa.Table("tasks", metadata, sa.Column("id", sa.Integer, primary_key=True))
    sa.Table(
        "task_chat_messages",
        metadata,
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("task_id", sa.Integer),
        sa.Column("created_at", sa.DateTime(timezone=True)),
    )
    metadata.create_all(engine)
    with _migration_env(engine) as connection:
        connection.execute(sa.text("INSERT INTO tasks (id) VALUES (1)"))
        connection.execute(
            sa.text(
                "INSERT INTO task_chat_messages (id, task_id, created_at) "
                "VALUES (10, 1, :ts)"
            ),
            {"ts": NOW - timedelta(days=2)},
        )
        migration.upgrade()
        assert _anchors(connection)[1] == NOW - timedelta(days=2)


def test_postgresql_backfills_outside_the_add_column_transaction(
    migration, monkeypatch
):
    """The ALTER's ACCESS EXCLUSIVE lock must not span the backfill.

    ``env.py`` gives PostgreSQL one transaction per migration, so without the
    autocommit block ``ALTER TABLE ... ADD COLUMN`` holds ACCESS EXCLUSIVE on
    ``tasks`` until the backfill finishes -- every task read and write in the
    deployment blocks behind it, for as long as the table is large. There is
    no assertion a SQLite run can make about that, so this pins the dialect
    branch instead: the one thing between the fix and a silent regression.
    """
    entered: list[str] = []

    class _Block:
        def __enter__(self):
            entered.append("in")

        def __exit__(self, *exc):
            entered.append("out")
            return False

    class _Context:
        as_sql = False

        def autocommit_block(self):
            return _Block()

    class _Dialect:
        name = "postgresql"

    class _Bind:
        dialect = _Dialect()

    class _Inspector:
        def has_table(self, name):
            return name == "tasks"

        def get_columns(self, name):
            return [
                {"name": "id"},
                {"name": "created_at"},
                {"name": "last_activity_at"},
            ]

    monkeypatch.setattr(migration.op, "get_context", _Context)
    monkeypatch.setattr(migration.op, "get_bind", _Bind)
    monkeypatch.setattr(migration.sa, "inspect", lambda _bind: _Inspector())
    monkeypatch.setattr(
        migration, "_backfill", lambda *a, **k: entered.append("backfill") or 0
    )

    migration.upgrade()
    assert entered == ["in", "backfill", "out"], entered


def test_sqlite_backfills_inside_the_migration_transaction(migration, monkeypatch):
    """SQLite shares one transaction across the whole chain; do not break it.

    Entering an autocommit block here would commit the chain mid-way, trading
    a lock problem this dialect does not have for a partial-upgrade one it
    does.
    """
    calls: list[str] = []

    class _Context:
        as_sql = False

        def autocommit_block(self):  # pragma: no cover - must not be reached
            raise AssertionError("SQLite must not break the chain transaction")

    class _Dialect:
        name = "sqlite"

    class _Bind:
        dialect = _Dialect()

    class _Inspector:
        def has_table(self, name):
            return name == "tasks"

        def get_columns(self, name):
            return [
                {"name": "id"},
                {"name": "created_at"},
                {"name": "last_activity_at"},
            ]

    monkeypatch.setattr(migration.op, "get_context", _Context)
    monkeypatch.setattr(migration.op, "get_bind", _Bind)
    monkeypatch.setattr(migration.sa, "inspect", lambda _bind: _Inspector())
    monkeypatch.setattr(
        migration, "_backfill", lambda *a, **k: calls.append("backfill") or 0
    )

    migration.upgrade()
    assert calls == ["backfill"]


def _render_offline_script(database_url: str | None = None) -> str:
    """The script ``alembic upgrade <predecessor>:<this> --sql`` produces.

    Driven as a subprocess through the real command rather than a stubbed
    ``op.get_context``. An earlier version of these tests stubbed it and
    asserted substrings, and a mutation that emitted a semantically inverted
    expression inline passed every one of them -- so what the migration
    actually renders is the only thing worth asserting on.
    """
    import os
    import subprocess
    import sys

    # The dialect comes from the environment, not from any engine the test
    # holds, so it must be pinned to whatever the script will be applied to.
    # Left ambient, the PostgreSQL parametrisation rendered SQLite DDL
    # (``ADD COLUMN ... DATETIME``) and applied it to PostgreSQL, which has no
    # such type -- green locally, red in CI.
    env = dict(os.environ)
    if database_url is not None:
        env["DATABASE_URL"] = database_url

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "alembic",
            "upgrade",
            f"{migration_module.down_revision}:{migration_module.revision}",
            "--sql",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_offline_script_carries_the_backfill_before_the_version_stamp():
    """Offline must fill, and must fill before the revision is stamped.

    Alembic appends the ``alembic_version`` update after whatever the
    migration emits. A script that adds the column without filling it stamps
    the revision anyway, so a later online ``upgrade head`` skips the backfill
    and every historical task keeps a NULL anchor -- which
    ``retention_anchor()`` resolves to ``created_at``, reporting a task
    created 400 days ago with a message from last week as expired at 90 days.
    Order is the whole guarantee, so it is asserted, not assumed.
    """
    script = _render_offline_script()

    add_column = script.index("ADD COLUMN last_activity_at")
    backfill = script.index("UPDATE tasks SET last_activity_at")
    stamp = script.index("UPDATE alembic_version")

    assert add_column < backfill < stamp
    assert migration_module.OFFLINE_ANCHOR_SQL in script


def test_offline_script_anchors_real_rows_on_their_newest_message(engine, request):
    """Apply what the command actually emitted, against seeded rows.

    Several messages per task, deliberately: with one message MIN, MAX and
    "the only row" are indistinguishable, and the review asked for
    ``last_activity_at = MAX(task_chat_messages.created_at)``. Executing a
    hand-written copy of the statement is also what let an earlier version of
    this test pass while the migration emitted something else, so the
    statements applied here are parsed out of the rendered script.
    """
    metadata, tasks, messages = _schema(with_messages=True)
    metadata.create_all(engine)
    with engine.begin() as connection:
        connection.execute(
            tasks.insert(),
            [
                {"id": 1, "created_at": NOW - timedelta(days=400)},
                {"id": 2, "created_at": NOW - timedelta(days=300)},
            ],
        )
        connection.execute(
            messages.insert(),
            [
                # Out of id order as well as out of time order, so neither
                # "first row" nor "last row" can stand in for the maximum.
                {"id": 10, "task_id": 1, "created_at": NOW - timedelta(days=200)},
                {"id": 11, "task_id": 1, "created_at": NOW - timedelta(days=10)},
                {"id": 12, "task_id": 1, "created_at": NOW - timedelta(days=90)},
            ],
        )

        script = _render_offline_script(str(engine.url))
        for statement in _script_statements(script):
            connection.execute(sa.text(statement))

        anchors = _anchors(connection)
        # The scenario the review names: created long ago, spoken to recently.
        # 400 days would be created_at, 200 the oldest message, 90 the middle.
        assert anchors[1] == NOW - timedelta(days=10)
        # No message at all: the task's own creation time is the fallback.
        assert anchors[2] == NOW - timedelta(days=300)


def test_offline_script_does_not_disturb_an_already_anchored_row(engine):
    """The emitted UPDATE is guarded, so re-application cannot clobber.

    A row anchored by an online run, or by an earlier application of this
    same script, must survive it.
    """
    metadata, tasks, messages = _schema(with_messages=True)
    metadata.create_all(engine)
    existing = NOW - timedelta(minutes=5)
    with engine.begin() as connection:
        connection.execute(
            tasks.insert(), {"id": 1, "created_at": NOW - timedelta(days=400)}
        )
        connection.execute(
            messages.insert(),
            {"id": 10, "task_id": 1, "created_at": NOW - timedelta(days=200)},
        )

        statements = _script_statements(_render_offline_script(str(engine.url)))
        for statement in statements:
            connection.execute(sa.text(statement))

        # Stand in for a later anchor, then re-run only the backfill.
        connection.execute(
            sa.text("UPDATE tasks SET last_activity_at = :ts WHERE id = 1"),
            {"ts": existing},
        )
        for statement in statements:
            if statement.startswith("UPDATE tasks SET last_activity_at"):
                connection.execute(sa.text(statement))

        assert _anchors(connection)[1] == existing


def test_offline_downgrade_still_renders(migration, monkeypatch):
    """Dropping the column needs no row work, so it stays offline-renderable."""
    dropped: list[str] = []

    class _Context:
        as_sql = True

    monkeypatch.setattr(migration.op, "get_context", _Context)
    monkeypatch.setattr(
        migration.op,
        "drop_column",
        lambda table, column: dropped.append((table, column)),
    )
    migration.downgrade()
    # Record the arguments: a stub that swallowed them passed even when the
    # migration dropped an unrelated column.
    assert dropped == [("tasks", "last_activity_at")]


def test_backfill_never_overwrites_an_anchor_set_after_the_select(engine, migration):
    """A row anchored between the SELECT and the UPDATE keeps the newer value.

    The ids come from an earlier statement and, on PostgreSQL, each batch
    commits separately -- so a transcript writer can anchor a selected task in
    between. Without ``AND last_activity_at IS NULL`` on the UPDATE the
    backfill overwrites it, and it overwrites it with a *stale* value: a
    blocked UPDATE re-checks its search condition on the target row, but the
    MAX() subquery reads another table under the original snapshot and cannot
    see the message that just arrived.

    What this test reaches, and what it does not: it anchors the row on the
    same connection before the UPDATE runs, which reproduces the *state* the
    racing writer leaves behind and proves the guard skips such a row. It
    does not reproduce the MVCC half -- no second session, no lock re-check,
    and no message inserted -- so the stale-snapshot mechanism above is
    reasoning about PostgreSQL, not something asserted here. SQLite has no
    row locks for it to exercise.
    """
    metadata, tasks, messages = _schema(with_messages=True)
    metadata.create_all(engine)
    concurrent_anchor = NOW - timedelta(minutes=1)
    with _migration_env(engine) as connection:
        connection.execute(
            tasks.insert(), {"id": 1, "created_at": NOW - timedelta(days=400)}
        )
        connection.execute(
            messages.insert(),
            {"id": 10, "task_id": 1, "created_at": NOW - timedelta(days=200)},
        )

        real_execute = connection.execute
        anchored: list[bool] = []

        def racing_execute(statement, *args, **kwargs):
            # Between the batch SELECT and its UPDATE, stand in for a
            # transcript writer that has just anchored this task.
            text = str(statement)
            if text.startswith("UPDATE tasks SET last_activity_at") and not anchored:
                anchored.append(True)
                real_execute(
                    sa.text("UPDATE tasks SET last_activity_at = :ts WHERE id = 1"),
                    {"ts": concurrent_anchor},
                )
            return real_execute(statement, *args, **kwargs)

        connection.execute = racing_execute  # type: ignore[method-assign]
        try:
            migration.upgrade()
        finally:
            connection.execute = real_execute  # type: ignore[method-assign]

        assert anchored, "the racing write never ran; the test proves nothing"
        assert _anchors(connection)[1] == concurrent_anchor


def test_revision_chains_onto_the_previous_head(migration):
    """A second head would break the repository's single-head invariant."""
    assert migration.revision == "20260922_task_last_activity_at"
    assert migration.down_revision == "20260921_word_contract"
