from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import sqlalchemy as sa
from alembic import context as alembic_context
from alembic.migration import MigrationContext
from alembic.operations import Operations

from tests.shared.postgres_disposable import disposable_database_factory
from xagent.web.models.uploaded_file import UploadedFile

MIGRATION_PATH = (
    Path(__file__).resolve().parents[2]
    / "src/xagent/migrations/versions/20260830_add_uploaded_file_detach_provenance.py"
)


def _load_migration():
    assert MIGRATION_PATH.exists(), "detach-provenance migration is missing"
    spec = importlib.util.spec_from_file_location(
        "detach_provenance_migration", MIGRATION_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_migration(migration, operation: str, connection) -> None:
    context = MigrationContext.configure(connection)
    with context.begin_transaction(), Operations.context(context):
        getattr(migration, operation)()


def _create_pre_migration_schema(connection) -> None:
    connection.exec_driver_sql("PRAGMA foreign_keys = ON")
    connection.execute(sa.text("CREATE TABLE users (id INTEGER PRIMARY KEY)"))
    connection.execute(sa.text("CREATE TABLE tasks (id INTEGER PRIMARY KEY)"))
    connection.execute(
        sa.text(
            "CREATE TABLE uploaded_files ("
            "id INTEGER PRIMARY KEY, file_id VARCHAR(36) NOT NULL UNIQUE, "
            "user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE, "
            "task_id INTEGER REFERENCES tasks(id) ON DELETE CASCADE, "
            "filename VARCHAR(512) NOT NULL, storage_path VARCHAR(2048) NOT NULL UNIQUE, "
            "storage_backend VARCHAR(64), storage_key VARCHAR(2048), storage_uri VARCHAR(4096), "
            "checksum VARCHAR(128), etag VARCHAR(255), workspace_relative_path VARCHAR(2048), "
            "workspace_category VARCHAR(64), upload_source VARCHAR(64), "
            "storage_status VARCHAR(32) NOT NULL, mime_type VARCHAR(255), "
            "file_size INTEGER NOT NULL DEFAULT 0, created_at DATETIME, updated_at DATETIME"
            ")"
        )
    )
    connection.execute(sa.text("INSERT INTO users (id) VALUES (1)"))
    connection.execute(sa.text("INSERT INTO tasks (id) VALUES (10)"))
    connection.execute(
        sa.text(
            "INSERT INTO uploaded_files "
            "(id, file_id, user_id, task_id, filename, storage_path, storage_status, file_size) "
            "VALUES (1, 'file-1', 1, 10, 'one.txt', '/tmp/one.txt', 'available', 1)"
        )
    )


def _create_postgresql_pre_migration_schema(connection) -> None:
    connection.execute(sa.text("CREATE TABLE users (id INTEGER PRIMARY KEY)"))
    connection.execute(sa.text("CREATE TABLE tasks (id INTEGER PRIMARY KEY)"))
    connection.execute(
        sa.text(
            "CREATE TABLE uploaded_files ("
            "id INTEGER PRIMARY KEY, file_id VARCHAR(36) NOT NULL UNIQUE, "
            "user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE, "
            "task_id INTEGER REFERENCES tasks(id) ON DELETE CASCADE, "
            "filename VARCHAR(512) NOT NULL, storage_path VARCHAR(2048) NOT NULL UNIQUE, "
            "upload_source VARCHAR(64), storage_status VARCHAR(32) NOT NULL, "
            "file_size INTEGER NOT NULL DEFAULT 0, "
            "created_at TIMESTAMPTZ, updated_at TIMESTAMPTZ"
            ")"
        )
    )
    connection.execute(sa.text("INSERT INTO users (id) VALUES (1)"))
    connection.execute(sa.text("INSERT INTO tasks (id) VALUES (10)"))
    connection.execute(
        sa.text(
            "INSERT INTO uploaded_files "
            "(id, file_id, user_id, task_id, filename, storage_path, storage_status, file_size) "
            "VALUES (1, 'file-1', 1, 10, 'one.txt', '/tmp/one.txt', 'available', 1)"
        )
    )


def _task_fk_ondelete(connection) -> str | None:
    task_fk = next(
        fk
        for fk in sa.inspect(connection).get_foreign_keys("uploaded_files")
        if fk["constrained_columns"] == ["task_id"]
    )
    return task_fk.get("options", {}).get("ondelete")


def test_model_declares_detach_provenance_and_gc_index() -> None:
    columns = UploadedFile.__table__.columns

    assert {"detached_reason", "detached_at"} <= set(columns.keys())
    task_fk = next(iter(columns["task_id"].foreign_keys))
    assert task_fk.ondelete == "SET NULL"
    index = next(
        index
        for index in UploadedFile.__table__.indexes
        if index.name == "ix_uploaded_files_detached_gc"
    )
    assert [column.name for column in index.columns] == [
        "task_id",
        "storage_status",
        "detached_at",
        "id",
    ]


def test_sqlite_upgrade_preserves_history_and_replaces_task_fk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    migration = _load_migration()
    engine = sa.create_engine("sqlite:///:memory:")

    with engine.connect() as connection:
        _create_pre_migration_schema(connection)
        monkeypatch.setattr(alembic_context, "is_offline_mode", lambda: False)
        monkeypatch.setattr(alembic_context, "get_bind", lambda: connection)
        _run_migration(migration, "upgrade", connection)

        historical = connection.execute(
            sa.text(
                "SELECT detached_reason, detached_at FROM uploaded_files WHERE id = 1"
            )
        ).one()
        assert historical == (None, None)
        assert _task_fk_ondelete(connection) == "SET NULL"

        connection.execute(sa.text("DELETE FROM tasks WHERE id = 10"))
        assert (
            connection.execute(
                sa.text("SELECT task_id FROM uploaded_files WHERE id = 1")
            ).scalar_one()
            is None
        )


@pytest.mark.parametrize(
    ("file_id", "storage_path"),
    [
        ("file-1", "/tmp/unique-path.txt"),
        ("unique-file", "/tmp/one.txt"),
    ],
)
def test_sqlite_upgrade_preserves_uploaded_file_unique_constraints(
    monkeypatch: pytest.MonkeyPatch,
    file_id: str,
    storage_path: str,
) -> None:
    migration = _load_migration()
    engine = sa.create_engine("sqlite:///:memory:")

    with engine.connect() as connection:
        _create_pre_migration_schema(connection)
        monkeypatch.setattr(alembic_context, "is_offline_mode", lambda: False)
        monkeypatch.setattr(alembic_context, "get_bind", lambda: connection)
        _run_migration(migration, "upgrade", connection)

        with pytest.raises(sa.exc.IntegrityError):
            connection.execute(
                sa.text(
                    "INSERT INTO uploaded_files "
                    "(id, file_id, user_id, filename, storage_path, storage_status, file_size) "
                    "VALUES (3, :file_id, 1, 'duplicate.txt', :storage_path, "
                    "'available', 1)"
                ),
                {"file_id": file_id, "storage_path": storage_path},
            )


@pytest.mark.parametrize("dialect_name", ["sqlite", "postgresql"])
def test_offline_migration_fails_instead_of_emitting_incomplete_fk_sql(
    monkeypatch: pytest.MonkeyPatch,
    dialect_name: str,
) -> None:
    migration = _load_migration()
    context = MigrationContext.configure(
        dialect_name=dialect_name,
        opts={"as_sql": True},
    )
    monkeypatch.setattr(alembic_context, "is_offline_mode", lambda: True)

    with (
        Operations.context(context),
        pytest.raises(RuntimeError, match="offline detach-provenance"),
    ):
        migration.upgrade()


@pytest.mark.postgresql
def test_postgresql_upgrade_preserves_history_and_installs_validated_fk_and_index() -> (
    None
):
    migration = _load_migration()

    with disposable_database_factory("xagent_upload_detach") as make_database:
        engine = make_database("upgrade")
        with engine.connect() as connection:
            _create_postgresql_pre_migration_schema(connection)
            connection.commit()

            _run_migration(migration, "upgrade", connection)

            assert connection.execute(
                sa.text(
                    "SELECT detached_reason, detached_at "
                    "FROM uploaded_files WHERE id = 1"
                )
            ).one() == (None, None)
            assert _task_fk_ondelete(connection) == "SET NULL"
            assert (
                connection.execute(
                    sa.text(
                        "SELECT convalidated FROM pg_constraint "
                        "WHERE conrelid = 'uploaded_files'::regclass AND conname = :name"
                    ),
                    {"name": migration.TASK_FK},
                ).scalar_one()
                is True
            )
            assert next(
                index
                for index in sa.inspect(connection).get_indexes("uploaded_files")
                if index["name"] == migration.INDEX
            )["column_names"] == [
                "task_id",
                "storage_status",
                "detached_at",
                "id",
            ]

            connection.execute(sa.text("DELETE FROM tasks WHERE id = 10"))
            assert (
                connection.execute(
                    sa.text("SELECT task_id FROM uploaded_files WHERE id = 1")
                ).scalar_one()
                is None
            )
            connection.commit()

            _run_migration(migration, "upgrade", connection)
            connection.execute(sa.text("INSERT INTO tasks (id) VALUES (11)"))
            connection.execute(
                sa.text("UPDATE uploaded_files SET task_id = 11 WHERE id = 1")
            )
            connection.commit()

            _run_migration(migration, "downgrade", connection)
            assert _task_fk_ondelete(connection) == "CASCADE"
            assert {
                column["name"]
                for column in sa.inspect(connection).get_columns("uploaded_files")
            }.isdisjoint({"detached_reason", "detached_at"})
            connection.execute(sa.text("DELETE FROM tasks WHERE id = 11"))
            assert (
                connection.execute(
                    sa.text("SELECT COUNT(*) FROM uploaded_files WHERE id = 1")
                ).scalar_one()
                == 0
            )


@pytest.mark.postgresql
def test_postgresql_commits_not_valid_fk_before_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    migration = _load_migration()

    with disposable_database_factory("xagent_upload_detach") as make_database:
        engine = make_database("fk_validation_lock")
        with engine.connect() as connection:
            _create_postgresql_pre_migration_schema(connection)
            connection.commit()

            real_execute = migration.op.execute
            writer_ran = False

            def execute_with_concurrent_writer(statement, *args, **kwargs):
                nonlocal writer_ran
                if "VALIDATE CONSTRAINT" in str(statement).upper():
                    with engine.connect() as writer:
                        writer.execute(sa.text("SET LOCAL lock_timeout = '500ms'"))
                        writer.execute(
                            sa.text(
                                "INSERT INTO uploaded_files "
                                "(id, file_id, user_id, task_id, filename, "
                                "storage_path, storage_status, file_size) VALUES "
                                "(3, 'file-3', 1, 10, 'three.txt', '/tmp/three.txt', "
                                "'available', 1)"
                            )
                        )
                        writer.commit()
                    writer_ran = True
                return real_execute(statement, *args, **kwargs)

            monkeypatch.setattr(migration.op, "execute", execute_with_concurrent_writer)

            _run_migration(migration, "upgrade", connection)

            assert writer_ran is True
            assert (
                connection.execute(
                    sa.text("SELECT COUNT(*) FROM uploaded_files WHERE id = 3")
                ).scalar_one()
                == 1
            )


@pytest.mark.postgresql
def test_postgresql_upgrade_repairs_invalid_same_column_index() -> None:
    migration = _load_migration()

    with disposable_database_factory("xagent_upload_detach") as make_database:
        engine = make_database("invalid_index")
        with engine.begin() as connection:
            _create_postgresql_pre_migration_schema(connection)
            connection.execute(
                sa.text(
                    "ALTER TABLE uploaded_files DROP CONSTRAINT uploaded_files_pkey"
                )
            )
            connection.execute(
                sa.text(
                    "INSERT INTO uploaded_files "
                    "(id, file_id, user_id, task_id, filename, storage_path, "
                    "storage_status, file_size) VALUES "
                    "(1, 'file-2', 1, 10, 'two.txt', '/tmp/two.txt', 'available', 1)"
                )
            )
            connection.execute(
                sa.text("ALTER TABLE uploaded_files ADD COLUMN detached_at TIMESTAMPTZ")
            )
            connection.execute(
                sa.text(
                    "ALTER TABLE uploaded_files ADD COLUMN detached_reason VARCHAR(64)"
                )
            )
            connection.execute(
                sa.text("UPDATE uploaded_files SET detached_at = CURRENT_TIMESTAMP")
            )

        with engine.connect().execution_options(
            isolation_level="AUTOCOMMIT"
        ) as connection:
            with pytest.raises(sa.exc.IntegrityError):
                connection.execute(
                    sa.text(
                        f"CREATE UNIQUE INDEX CONCURRENTLY {migration.INDEX} "
                        "ON uploaded_files "
                        "(task_id, storage_status, detached_at, id)"
                    )
                )

        with engine.connect() as connection:
            assert connection.execute(
                sa.text(
                    "SELECT indisvalid, indisunique FROM pg_index "
                    "WHERE indexrelid = to_regclass(:name)"
                ),
                {"name": migration.INDEX},
            ).one() == (False, True)
            connection.commit()

            _run_migration(migration, "upgrade", connection)

            assert connection.execute(
                sa.text(
                    "SELECT indisvalid, indisunique FROM pg_index "
                    "WHERE indexrelid = to_regclass(:name)"
                ),
                {"name": migration.INDEX},
            ).one() == (True, False)
