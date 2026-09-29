"""Populated upgrades retain attachment identity and change the task FK."""

from pathlib import Path

import pytest
import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations

from tests.shared.postgres_disposable import (
    disposable_database_factory,
    load_migration_module,
)
from xagent.db.migration import _migration_connection

MIGRATION = (
    Path(__file__).parents[2]
    / "src/xagent/migrations/versions/20260929_task_attachment_detachment.py"
)


@pytest.fixture(
    params=["sqlite", pytest.param("postgresql", marks=pytest.mark.postgresql)]
)
def engine_factory(request, tmp_path):
    if request.param == "postgresql":
        with disposable_database_factory("detach_migration") as make:
            yield make
    else:
        engines = []

        def make(tag):
            engine = sa.create_engine(f"sqlite:///{tmp_path / f'{tag}.db'}")

            @sa.event.listens_for(engine, "connect")
            def fk_on(connection, _record):
                connection.execute("PRAGMA foreign_keys=ON")

            engines.append(engine)
            return engine

        yield make
        for engine in engines:
            engine.dispose()


@pytest.fixture
def engine(engine_factory):
    return engine_factory("upgrade")


def seed(engine, *, task_foreign_key=True):
    task_reference = (
        " REFERENCES tasks(id) ON DELETE CASCADE" if task_foreign_key else ""
    )
    with engine.begin() as conn:
        for statement in (
            "CREATE TABLE users (id INTEGER PRIMARY KEY)",
            "CREATE TABLE tasks (id INTEGER PRIMARY KEY)",
            "CREATE TABLE uploaded_files (id INTEGER PRIMARY KEY, "
            "file_id VARCHAR(36) NOT NULL UNIQUE, storage_path VARCHAR(2048) NOT NULL UNIQUE, "
            "user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE, "
            f"task_id INTEGER{task_reference}, "
            "storage_status VARCHAR(32) NOT NULL)",
            "CREATE INDEX ix_existing_status ON uploaded_files(storage_status)",
            "INSERT INTO users VALUES (1)",
            "INSERT INTO tasks VALUES (1)",
            "INSERT INTO uploaded_files VALUES (1, 'attached', '/owned/file', 1, 1, 'legacy')",
            "INSERT INTO uploaded_files VALUES (2, 'draft', '/owned/draft', 1, NULL, 'legacy')",
        ):
            conn.execute(sa.text(statement))


def migrate(engine, operation):
    migration = load_migration_module(MIGRATION)
    connection_manager = (
        _migration_connection(engine)
        if engine.dialect.name == "sqlite"
        else engine.connect()
    )
    with connection_manager as conn:
        context = MigrationContext.configure(conn)
        with context.begin_transaction(), Operations.context(context):
            getattr(migration, operation)()
        if engine.dialect.name != "sqlite":
            # The direct Operations harness must own the transaction so its
            # PostgreSQL autocommit block can commit before validation.
            conn.commit()


def attachment_schema(engine):
    inspector = sa.inspect(engine)
    markers = {
        column["name"]: (str(column["type"]), column["nullable"])
        for column in inspector.get_columns("uploaded_files")
        if column["name"] in {"detached_at", "detached_reason"}
    }
    task_foreign_keys = {
        (
            tuple(foreign_key["constrained_columns"]),
            foreign_key["referred_table"],
            tuple(foreign_key["referred_columns"]),
            foreign_key["options"].get("ondelete"),
        )
        for foreign_key in inspector.get_foreign_keys("uploaded_files")
        if foreign_key["constrained_columns"] == ["task_id"]
    }
    return markers, task_foreign_keys


def test_upgrade_preserves_data_uniqueness_indexes_and_reruns(engine):
    seed(engine)
    migrate(engine, "upgrade")
    migrate(engine, "upgrade")
    inspector = sa.inspect(engine)
    assert "ix_existing_status" in {
        i["name"] for i in inspector.get_indexes("uploaded_files")
    }
    with engine.begin() as conn:
        rows = conn.execute(
            sa.text(
                "SELECT file_id, detached_reason, detached_at FROM uploaded_files ORDER BY id"
            )
        ).all()
        assert rows == [("attached", None, None), ("draft", None, None)]
        conn.execute(sa.text("DELETE FROM tasks WHERE id=1"))
        assert (
            conn.execute(
                sa.text("SELECT count(*) FROM uploaded_files WHERE task_id IS NULL")
            ).scalar()
            == 2
        )
    for duplicate in (
        "(3, 'attached', '/unique', 1, NULL, 'legacy')",
        "(3, 'unique', '/owned/file', 1, NULL, 'legacy')",
    ):
        with engine.begin() as conn, pytest.raises(sa.exc.IntegrityError):
            conn.execute(
                sa.text(
                    "INSERT INTO uploaded_files (id,file_id,storage_path,user_id,task_id,storage_status) VALUES "
                    + duplicate
                )
            )
    migrate(engine, "downgrade")
    assert "detached_at" not in {
        c["name"] for c in sa.inspect(engine).get_columns("uploaded_files")
    }
    with engine.begin() as conn:
        conn.execute(sa.text("INSERT INTO tasks VALUES (2)"))
        conn.execute(sa.text("UPDATE uploaded_files SET task_id=2 WHERE id=1"))
        conn.execute(sa.text("DELETE FROM tasks WHERE id=2"))
        assert conn.execute(sa.text("SELECT file_id FROM uploaded_files")).all() == [
            ("draft",)
        ]


@pytest.mark.parametrize("marker", ["detached_reason", "detached_at"])
def test_partial_upgrade_and_pending_obligations_block_downgrade(engine, marker):
    seed(engine)
    with engine.begin() as conn:
        sql_type = "VARCHAR(32)" if marker == "detached_reason" else "TIMESTAMP"
        conn.execute(
            sa.text(f"ALTER TABLE uploaded_files ADD COLUMN {marker} {sql_type}")
        )
    migrate(engine, "upgrade")
    with engine.begin() as conn:
        value = "'task_deleted'" if marker == "detached_reason" else "CURRENT_TIMESTAMP"
        conn.execute(sa.text(f"UPDATE uploaded_files SET {marker}={value} WHERE id=1"))
    with pytest.raises(RuntimeError, match="Reconcile detached"):
        migrate(engine, "downgrade")
    assert {"detached_at", "detached_reason"} <= {
        c["name"] for c in sa.inspect(engine).get_columns("uploaded_files")
    }
    with engine.begin() as conn:
        conn.execute(sa.text("DELETE FROM users WHERE id=1"))
        assert (
            conn.execute(sa.text("SELECT count(*) FROM uploaded_files")).scalar() == 0
        )
    migrate(engine, "downgrade")


@pytest.mark.parametrize("dialect", ["sqlite", "postgresql"])
def test_offline_migration_refuses_to_emit_incomplete_schema(dialect):
    migration = load_migration_module(MIGRATION)
    context = MigrationContext.configure(dialect_name=dialect, opts={"as_sql": True})
    with Operations.context(context), pytest.raises(RuntimeError, match="online"):
        migration.upgrade()


def test_populated_upgrade_markers_and_task_fk_match_fresh_install(
    engine, engine_factory
):
    from xagent.web.models import Base

    seed(engine)
    migrate(engine, "upgrade")
    fresh_engine = engine_factory("fresh")
    Base.metadata.create_all(fresh_engine)

    assert attachment_schema(engine) == attachment_schema(fresh_engine)


@pytest.mark.parametrize("existing_not_valid", [False, True])
def test_upgrade_rejects_dangling_task_references_before_schema_changes(
    engine, existing_not_valid
):
    if existing_not_valid and engine.dialect.name != "postgresql":
        pytest.skip("NOT VALID constraints are PostgreSQL-specific")
    seed(engine, task_foreign_key=False)
    with engine.begin() as conn:
        conn.execute(sa.text("UPDATE uploaded_files SET task_id=999 WHERE id=1"))
        if existing_not_valid:
            conn.execute(
                sa.text(
                    "ALTER TABLE uploaded_files ADD CONSTRAINT "
                    "fk_uploaded_files_task_id_tasks FOREIGN KEY (task_id) "
                    "REFERENCES tasks(id) ON DELETE SET NULL NOT VALID"
                )
            )
    inspector = sa.inspect(engine)
    columns_before = [c["name"] for c in inspector.get_columns("uploaded_files")]
    foreign_keys_before = inspector.get_foreign_keys("uploaded_files")

    with pytest.raises(
        RuntimeError, match="missing tasks.*preserving.*IDs and rows"
    ) as exc:
        migrate(engine, "upgrade")

    assert "uploaded file IDs: [1]" in str(exc.value)
    assert "missing task IDs: [999]" in str(exc.value)
    inspector = sa.inspect(engine)
    assert [
        c["name"] for c in inspector.get_columns("uploaded_files")
    ] == columns_before
    assert inspector.get_foreign_keys("uploaded_files") == foreign_keys_before
    with engine.connect() as conn:
        assert conn.execute(
            sa.text("SELECT id, task_id FROM uploaded_files ORDER BY id")
        ).all() == [(1, 999), (2, None)]
        if existing_not_valid:
            assert (
                conn.execute(
                    sa.text(
                        "SELECT convalidated FROM pg_constraint "
                        "WHERE conname='fk_uploaded_files_task_id_tasks'"
                    )
                ).scalar_one()
                is False
            )


@pytest.mark.postgresql
def test_validation_releases_ddl_lock_and_can_resume_after_interruption():
    with disposable_database_factory("detach_validate") as make:
        engine = make("retry")
        seed(engine)

        def interrupt_validation(
            conn, cursor, statement, parameters, context, executemany
        ):
            if "VALIDATE CONSTRAINT" not in statement:
                return
            # A separate writer must not wait on the ADD COLUMN/FK DDL lock.
            with engine.begin() as writer:
                writer.execute(sa.text("SET LOCAL lock_timeout = '500ms'"))
                writer.execute(
                    sa.text(
                        "UPDATE uploaded_files SET storage_status='available' WHERE id=1"
                    )
                )
            raise RuntimeError("simulated interruption before validation")

        sa.event.listen(engine, "before_cursor_execute", interrupt_validation)
        with pytest.raises(RuntimeError, match="simulated interruption"):
            migrate(engine, "upgrade")
        sa.event.remove(engine, "before_cursor_execute", interrupt_validation)
        migrate(engine, "upgrade")
        with engine.connect() as conn:
            assert (
                conn.execute(
                    sa.text(
                        "SELECT convalidated FROM pg_constraint WHERE conname='fk_uploaded_files_task_id_tasks'"
                    )
                ).scalar()
                is True
            )
            assert (
                conn.execute(
                    sa.text("SELECT storage_status FROM uploaded_files WHERE id=1")
                ).scalar()
                == "available"
            )


@pytest.mark.parametrize("parent_present", [False, True])
def test_historical_missing_task_foreign_key(engine, parent_present):
    seed(engine, task_foreign_key=False)
    if not parent_present:
        with engine.begin() as conn:
            conn.execute(sa.text("DROP TABLE tasks"))
    migrate(engine, "upgrade")
    migrate(engine, "upgrade")
    inspector = sa.inspect(engine)
    assert inspector.has_table("tasks") == parent_present
    task_fks = [
        fk
        for fk in inspector.get_foreign_keys("uploaded_files")
        if fk["constrained_columns"] == ["task_id"]
    ]
    if parent_present:
        assert task_fks[0]["options"]["ondelete"] == "SET NULL"
        with engine.begin() as conn:
            conn.execute(sa.text("DELETE FROM tasks WHERE id=1"))
            assert (
                conn.execute(
                    sa.text("SELECT count(*) FROM uploaded_files WHERE task_id IS NULL")
                ).scalar()
                == 2
            )
    else:
        assert task_fks == []
    migrate(engine, "downgrade")
    assert "detached_at" not in {
        c["name"] for c in sa.inspect(engine).get_columns("uploaded_files")
    }
