"""Task deletion must never give retained workspace paths to a new task."""

import sqlite3
from pathlib import Path

import pytest
import sqlalchemy as sa
from alembic import command
from alembic.migration import MigrationContext
from alembic.operations import Operations
from alembic.script import ScriptDirectory
from sqlalchemy.orm import sessionmaker

from tests.shared.postgres_disposable import (
    disposable_database_factory,
    load_migration_module,
)
from xagent.core.file_storage.factory import get_unscoped_file_storage
from xagent.core.workspace import TaskWorkspace
from xagent.db.config import create_alembic_config
from xagent.db.migration import _migration_connection
from xagent.web.config import get_upload_path
from xagent.web.models import Base, Task, UploadedFile, User
from xagent.web.services.task_deletion import purge_task_rows

MIGRATION = (
    Path(__file__).parents[2]
    / "src/xagent/migrations/versions/20260930_task_identity.py"
)


def migrate(engine):
    migration = load_migration_module(MIGRATION)
    with _migration_connection(engine) as connection:
        context = MigrationContext.configure(connection)
        with Operations.context(context):
            migration.upgrade()


@pytest.fixture(autouse=True)
def storage_config(monkeypatch, tmp_path):
    monkeypatch.setenv("XAGENT_UPLOADS_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("XAGENT_EXTERNAL_UPLOAD_DIRS", "")
    monkeypatch.setenv("XAGENT_FILE_STORAGE_URI", (tmp_path / "objects").as_uri())
    get_unscoped_file_storage.cache_clear()
    yield
    get_unscoped_file_storage.cache_clear()


@pytest.fixture(
    params=["sqlite", pytest.param("postgresql", marks=pytest.mark.postgresql)]
)
def engine(request, sqlite_engine):
    if request.param == "postgresql":
        with disposable_database_factory("task_identity") as make:
            yield make("upgrade")
    else:
        yield sqlite_engine


@pytest.fixture
def sqlite_engine(tmp_path):
    engine = sa.create_engine(f"sqlite:///{tmp_path / 'tasks.db'}")

    @sa.event.listens_for(engine, "connect")
    def enable_fks(connection, _record):
        connection.execute("PRAGMA foreign_keys=ON")

    yield engine
    engine.dispose()


@pytest.mark.parametrize("schema", ["fresh", "upgraded", "upgraded_after_delete"])
def test_delete_recreate_output_preserves_retained_file(
    engine, tmp_path, monkeypatch, schema
):
    metadata = sa.MetaData()
    for table in Base.metadata.tables.values():
        table.to_metadata(metadata)
    if schema != "fresh":
        metadata.tables["tasks"].dialect_options["sqlite"]["autoincrement"] = False
    metadata.create_all(engine)
    sessions = sessionmaker(bind=engine)
    monkeypatch.setattr(
        "xagent.web.models.database.get_session_local", lambda: sessions
    )
    monkeypatch.setattr(
        "xagent.web.services.uploaded_file_store.get_session_local", lambda: sessions
    )
    monkeypatch.setattr("xagent.core.storage.manager.create_db_session", sessions)
    with sessions() as db:
        owner = User(username="owner", password_hash="unused")
        db.add(owner)
        db.flush()
        owner_id = int(owner.id)
        task = Task(user_id=owner_id, title="old")
        db.add(task)
        db.commit()
        old_id = int(task.id)
    base = tmp_path / "uploads" / f"user_{owner_id}"
    old = TaskWorkspace(f"web_task_{old_id}", str(base))
    old_path = old.output_dir / "report.txt"
    old_path.write_bytes(b"retained bytes")
    old_file_id = old.register_file(str(old_path))
    if schema == "upgraded":
        migrate(engine)
    with sessions() as db:
        assert purge_task_rows(db, task_id=old_id, detached_reason="task_deleted")
        db.commit()
        retained = db.query(UploadedFile).filter_by(file_id=old_file_id).one()
        before = {
            c.name: getattr(retained, c.name) for c in UploadedFile.__table__.columns
        }
        durable_path = Path(str(retained.storage_uri).removeprefix("file://"))
        assert durable_path.read_bytes() == b"retained bytes"
    if schema == "upgraded_after_delete":
        migrate(engine)
    with sessions() as db:
        task = Task(user_id=owner_id, title="new")
        db.add(task)
        db.commit()
        new_id = int(task.id)
    new = TaskWorkspace(f"web_task_{new_id}", str(base))
    new_path = new.output_dir / "report.txt"
    new_path.write_bytes(b"new output")
    new_file_id = new.register_file(str(new_path))
    assert new_id > old_id
    assert new_path != old_path
    assert new_file_id != old_file_id
    assert old_path.read_bytes() == durable_path.read_bytes() == b"retained bytes"
    with sessions() as db:
        retained = db.query(UploadedFile).filter_by(file_id=old_file_id).one()
        assert {
            c.name: getattr(retained, c.name) for c in UploadedFile.__table__.columns
        } == before
        output = db.query(UploadedFile).filter_by(file_id=new_file_id).one()
        assert output.task_id == new_id
        assert output.storage_path == str(new_path)


def seed_legacy(engine):
    with engine.begin() as connection:
        connection.exec_driver_sql(
            "CREATE TABLE tasks (id INTEGER PRIMARY KEY, title TEXT)"
        )
        connection.exec_driver_sql("INSERT INTO tasks VALUES (7, 'live')")


def test_upgrade_decodes_only_uploaded_file_candidates(tmp_path):
    fetched = []

    class RecordingCursor(sqlite3.Cursor):
        recording = False

        def execute(self, statement, parameters=()):
            self.recording = (
                statement.startswith("SELECT storage_")
                and "FROM uploaded_files" in statement
            )
            return super().execute(statement, parameters)

        def fetchone(self):
            row = super().fetchone()
            if self.recording and row is not None:
                fetched.append(row)
            return row

        def fetchmany(self, size=None):
            rows = super().fetchmany(size) if size is not None else super().fetchmany()
            if self.recording:
                fetched.extend(rows)
            return rows

        def fetchall(self):
            rows = super().fetchall()
            if self.recording:
                fetched.extend(rows)
            return rows

    class RecordingConnection(sqlite3.Connection):
        def cursor(self, *args, **kwargs):
            kwargs["factory"] = RecordingCursor
            return super().cursor(*args, **kwargs)

    database = tmp_path / "recording.db"
    engine = sa.create_engine(
        "sqlite://",
        creator=lambda: sqlite3.connect(database, factory=RecordingConnection),
    )
    try:
        seed_legacy(engine)
        with engine.begin() as connection:
            connection.exec_driver_sql(
                "CREATE TABLE uploaded_files "
                "(storage_path TEXT, storage_key TEXT, storage_uri TEXT)"
            )
            connection.exec_driver_sql(
                "INSERT INTO uploaded_files VALUES (?, ?, ?)",
                [
                    (f"/ordinary/{index}/report.txt", None, "s3://bucket/report.txt")
                    for index in range(40)
                ]
                + [
                    ("/archive/web_task_701/output/a.txt", None, None),
                    (None, "tenant/task_702/output/b.txt", None),
                    (None, None, "s3://archive/web_task_703/output/c.txt"),
                ],
            )
        migrate(engine)
        with engine.begin() as connection:
            new_id = connection.exec_driver_sql(
                "INSERT INTO tasks DEFAULT VALUES"
            ).lastrowid
        assert new_id == 704
        assert len(fetched) == 3
    finally:
        engine.dispose()


@pytest.mark.parametrize(
    "source",
    [
        "local",
        "scoped",
        "windows",
        "storage_key",
        "storage_uri",
        "cleanup",
        "tombstone",
        "directory",
        "external",
        "symlink",
        "legacy_symlink",
        "legacy_user_symlink",
        "escaped_symlink",
        "workspace_named_scope",
        "uploaded_task_id",
        "legacy_custom_metadata",
        "legacy_custom_directory",
        "legacy_unscoped_metadata",
        "reattached_metadata",
    ],
)
def test_upgrade_reserves_historical_ids(sqlite_engine, tmp_path, monkeypatch, source):
    seed_legacy(sqlite_engine)
    with sqlite_engine.begin() as connection:
        if source in {"cleanup", "tombstone", "uploaded_task_id"}:
            table = (
                "uploaded_files"
                if source == "uploaded_task_id"
                else "task_cleanup_obligations"
                if source == "cleanup"
                else "expired_task_tombstones"
            )
            connection.exec_driver_sql(f"CREATE TABLE {table} (task_id INTEGER)")
            connection.exec_driver_sql(f"INSERT INTO {table} VALUES (701)")
        elif source in {"local", "scoped", "windows", "storage_key", "storage_uri"}:
            column = source if source.startswith("storage_") else "storage_path"
            paths = {
                "local": "/old/mount/web_task_701/output/report.txt",
                "scoped": "/old/mount/user_4/tenant/project/web_task_701/output/report.txt",
                "windows": r"C:\old\user_4\task_701\output\report.txt",
                "storage_key": "users/4/tenant/task_701/output/report.txt",
                "storage_uri": "s3://old-bucket/user_4/web_task_701/output/report.txt",
            }
            connection.exec_driver_sql(f"CREATE TABLE uploaded_files ({column} TEXT)")
            connection.exec_driver_sql(
                "INSERT INTO uploaded_files VALUES (?)", (paths[source],)
            )
        elif source.startswith("legacy_") and source.endswith(
            ("metadata", "directory")
        ):
            path = get_upload_path(
                "report.txt",
                task_id="701",
                folder="custom",
                user_id=None if source.startswith("legacy_unscoped") else 4,
                create_if_not_exists=source.endswith("directory"),
            )
            if source.endswith("metadata"):
                connection.exec_driver_sql(
                    "CREATE TABLE uploaded_files(storage_path TEXT)"
                )
                connection.exec_driver_sql(
                    "INSERT INTO uploaded_files VALUES (?)", (str(path),)
                )
        elif source == "reattached_metadata":
            connection.exec_driver_sql(
                "CREATE TABLE uploaded_files(task_id INTEGER, storage_path TEXT)"
            )
            connection.exec_driver_sql(
                "INSERT INTO uploaded_files VALUES (2, '/archive/web_task_701/output/report.txt')"
            )
        else:
            root = tmp_path / "uploads"
            if source == "external":
                root = tmp_path / "external"
                root.mkdir()
                monkeypatch.setenv("XAGENT_EXTERNAL_UPLOAD_DIRS", str(root))
            if source in {
                "symlink",
                "legacy_symlink",
                "legacy_user_symlink",
                "escaped_symlink",
            }:
                target = tmp_path / "scope-target"
                target.mkdir()
                alias_parent = (
                    root / "user_4" if source == "legacy_user_symlink" else root
                )
                alias_parent.mkdir(parents=True)
                legacy_alias = source in {"legacy_symlink", "legacy_user_symlink"}
                alias = "task_701" if legacy_alias else "web_task_701"
                (alias_parent / alias).symlink_to(target, target_is_directory=True)
                # A symlink loop must not stop the finite inventory.
                (target / "loop").symlink_to(root, target_is_directory=True)
                if source != "escaped_symlink":
                    monkeypatch.setenv("XAGENT_EXTERNAL_UPLOAD_DIRS", str(target))
            if source == "workspace_named_scope":
                root = root / "user_4" / "web_task_9"
            subdir = "custom" if source.startswith("legacy_") else "output"
            (
                alias_parent / alias
                if source in {"legacy_symlink", "legacy_user_symlink"}
                else root / "web_task_701"
            ).joinpath(subdir).mkdir(parents=True)
    if source == "escaped_symlink":
        with pytest.raises(RuntimeError, match="outside configured upload roots"):
            migrate(sqlite_engine)
        return
    migrate(sqlite_engine)
    migrate(sqlite_engine)
    with sqlite_engine.begin() as connection:
        new_id = connection.exec_driver_sql(
            "INSERT INTO tasks(title) VALUES ('new')"
        ).lastrowid
        assert new_id == 702
        assert (
            connection.exec_driver_sql("SELECT title FROM tasks WHERE id=7").scalar()
            == "live"
        )


def test_retry_and_downgrade_preserve_consumed_sequence(sqlite_engine):
    seed_legacy(sqlite_engine)
    migrate(sqlite_engine)
    with sqlite_engine.begin() as connection:
        connection.exec_driver_sql("INSERT INTO tasks VALUES (900, 'retired')")
        connection.exec_driver_sql("DELETE FROM tasks")
    migration = load_migration_module(MIGRATION)
    with _migration_connection(sqlite_engine) as connection:
        with Operations.context(MigrationContext.configure(connection)):
            migration.downgrade()
    migrate(sqlite_engine)
    with sqlite_engine.begin() as connection:
        assert (
            connection.exec_driver_sql("INSERT INTO tasks DEFAULT VALUES").lastrowid
            == 901
        )


def test_upgrade_preserves_constraints_indexes_triggers_and_children(sqlite_engine):
    statements = [
        "CREATE TABLE users (id INTEGER PRIMARY KEY)",
        "INSERT INTO users VALUES (1)",
        "CREATE TABLE tasks (id INTEGER PRIMARY KEY, owner INTEGER REFERENCES users(id) ON DELETE SET NULL, "
        "title TEXT NOT NULL, priority INTEGER DEFAULT 0 CHECK(priority >= 0), UNIQUE(owner,title))",
        "CREATE INDEX ix_title ON tasks(title) WHERE priority > 0",
        "CREATE INDEX ix_lower_title ON tasks(lower(title))",
        "CREATE TABLE audit (task_id INTEGER)",
        "CREATE TRIGGER task_insert AFTER INSERT ON tasks BEGIN INSERT INTO audit VALUES (new.id); END",
        "INSERT INTO tasks VALUES (7,1,'kept',2)",
    ]
    for name, action in [
        ("cascade_child", "CASCADE"),
        ("null_child", "SET NULL"),
        ("restrict_child", "RESTRICT"),
    ]:
        statements += [
            f"CREATE TABLE {name} (id INTEGER PRIMARY KEY, task_id INTEGER REFERENCES tasks(id) ON DELETE {action})",
            f"INSERT INTO {name} VALUES (1,7)",
        ]
    with sqlite_engine.begin() as connection:
        for statement in statements:
            connection.exec_driver_sql(statement)
        original_objects = connection.exec_driver_sql(
            "SELECT type,name,sql FROM sqlite_master WHERE type IN ('index','trigger') AND sql IS NOT NULL ORDER BY name"
        ).all()
    # Inspect actual SQLite constraints: SQLAlchemy reflection loses the
    # ON DELETE action in this compact legacy DDL before the migration.
    with sqlite_engine.connect() as connection:
        before_fks = {
            name: connection.exec_driver_sql(f"PRAGMA foreign_key_list('{name}')").all()
            for name in ["tasks", "cascade_child", "null_child", "restrict_child"]
        }
    migrate(sqlite_engine)
    migrate(sqlite_engine)
    with sqlite_engine.connect() as connection:
        after_fks = {
            name: connection.exec_driver_sql(f"PRAGMA foreign_key_list('{name}')").all()
            for name in before_fks
        }
    assert after_fks == before_fks
    with sqlite_engine.begin() as connection:
        assert connection.exec_driver_sql("PRAGMA foreign_keys").scalar() == 1
        assert connection.exec_driver_sql("PRAGMA foreign_key_check").all() == []
        assert connection.exec_driver_sql("SELECT * FROM tasks").all() == [
            (7, 1, "kept", 2)
        ]
        for name in ["cascade_child", "null_child", "restrict_child"]:
            assert connection.exec_driver_sql(f"SELECT * FROM {name}").all() == [(1, 7)]
        assert connection.exec_driver_sql("SELECT * FROM audit").all() == [(7,)]
        assert (
            connection.exec_driver_sql(
                "SELECT type,name,sql FROM sqlite_master WHERE type IN ('index','trigger') AND sql IS NOT NULL ORDER BY name"
            ).all()
            == original_objects
        )
    for statement in [
        "INSERT INTO tasks(owner,title) VALUES(1,'kept')",
        "INSERT INTO tasks(title,priority) VALUES('bad',-1)",
        "INSERT INTO tasks(owner,title) VALUES(99,'missing owner')",
        "DELETE FROM tasks WHERE id=7",
    ]:
        with sqlite_engine.begin() as connection, pytest.raises(sa.exc.IntegrityError):
            connection.exec_driver_sql(statement)
    with sqlite_engine.begin() as connection:
        connection.exec_driver_sql("DELETE FROM restrict_child")
        connection.exec_driver_sql("DELETE FROM tasks")
        assert connection.exec_driver_sql("SELECT * FROM cascade_child").all() == []
        assert connection.exec_driver_sql("SELECT task_id FROM null_child").all() == [
            (None,)
        ]
        assert (
            connection.exec_driver_sql(
                "INSERT INTO tasks(owner,title) VALUES(1,'new')"
            ).lastrowid
            == 8
        )
        assert connection.exec_driver_sql("SELECT * FROM audit").all() == [(7,), (8,)]
        connection.exec_driver_sql("DELETE FROM users")
        assert connection.exec_driver_sql("SELECT owner FROM tasks").scalar() is None


def test_upgrade_preserves_inline_checks_for_older_downgrades(sqlite_engine):
    seed_legacy(sqlite_engine)
    events = load_migration_module(
        MIGRATION.with_name("20260905_add_task_execution_events.py")
    )
    writers = load_migration_module(
        MIGRATION.with_name("20260905_enable_task_execution_event_writers.py")
    )
    with _migration_connection(sqlite_engine) as connection:
        with Operations.context(MigrationContext.configure(connection)):
            events.upgrade()
            writers.upgrade()
    migrate(sqlite_engine)
    migrate(sqlite_engine)
    for statement in [
        "UPDATE tasks SET conversation_storage_version=3",
        "UPDATE tasks SET conversation_event_sequence=-1",
    ]:
        with sqlite_engine.begin() as connection, pytest.raises(sa.exc.IntegrityError):
            connection.exec_driver_sql(statement)
    with _migration_connection(sqlite_engine) as connection:
        with Operations.context(MigrationContext.configure(connection)):
            writers.downgrade()
            events.downgrade()
    with sqlite_engine.begin() as connection:
        assert connection.exec_driver_sql("SELECT * FROM tasks").all() == [(7, "live")]
        connection.exec_driver_sql("DELETE FROM tasks")
        assert (
            connection.exec_driver_sql("INSERT INTO tasks DEFAULT VALUES").lastrowid
            == 8
        )


def test_interrupted_rebuild_rolls_back_and_can_retry(
    sqlite_engine, tmp_path, monkeypatch
):
    seed_legacy(sqlite_engine)
    with sqlite_engine.begin() as connection:
        connection.exec_driver_sql(
            "CREATE TABLE child (task_id INTEGER REFERENCES tasks(id) ON DELETE CASCADE)"
        )
        connection.exec_driver_sql("INSERT INTO child VALUES (7)")
        before = connection.exec_driver_sql(
            "SELECT sql FROM sqlite_master WHERE name='tasks'"
        ).scalar()
    root = tmp_path / "uploads"
    root.mkdir(exist_ok=True)
    migration = load_migration_module(MIGRATION)
    original_scandir = migration.os.scandir

    def fail_scan(path):
        if Path(path) == root:
            raise OSError(f"simulated I/O failure at {path}")
        return original_scandir(path)

    monkeypatch.setattr(migration.os, "scandir", fail_scan)
    with pytest.raises(OSError, match=str(root)):
        with _migration_connection(sqlite_engine) as connection:
            with Operations.context(MigrationContext.configure(connection)):
                migration.upgrade()
    monkeypatch.setattr(migration.os, "scandir", original_scandir)

    def interrupt(connection, cursor, statement, parameters, context, executemany):
        if statement.startswith("INSERT INTO sqlite_sequence"):
            raise RuntimeError("simulated interruption")

    sa.event.listen(sqlite_engine, "before_cursor_execute", interrupt)
    try:
        with pytest.raises(RuntimeError, match="simulated interruption"):
            migrate(sqlite_engine)
    finally:
        sa.event.remove(sqlite_engine, "before_cursor_execute", interrupt)
    with sqlite_engine.connect() as connection:
        assert connection.exec_driver_sql("PRAGMA foreign_keys").scalar() == 1
        assert (
            connection.exec_driver_sql(
                "SELECT sql FROM sqlite_master WHERE name='tasks'"
            ).scalar()
            == before
        )
        assert connection.exec_driver_sql("SELECT * FROM tasks").all() == [(7, "live")]
        assert connection.exec_driver_sql("SELECT * FROM child").all() == [(7,)]
        assert (
            connection.exec_driver_sql(
                "SELECT name FROM sqlite_master WHERE name='_alembic_tmp_tasks'"
            ).all()
            == []
        )
    migrate(sqlite_engine)
    with sqlite_engine.begin() as connection:
        connection.exec_driver_sql("DELETE FROM tasks")
        assert (
            connection.exec_driver_sql("INSERT INTO tasks DEFAULT VALUES").lastrowid
            == 8
        )


@pytest.mark.parametrize(
    "path",
    [
        "web_task_888.txt",
        "web_task_999suffix/output/report.txt",
        "output/web_task_999",
        "web_task_9/output/web_task_999/output/report.txt",
    ],
)
def test_non_workspace_names_do_not_raise_floor_beyond_real_workspace(
    sqlite_engine, tmp_path, path
):
    seed_legacy(sqlite_engine)
    with sqlite_engine.begin() as connection:
        connection.exec_driver_sql("CREATE TABLE uploaded_files(storage_path TEXT)")
        connection.exec_driver_sql(
            "INSERT INTO uploaded_files VALUES (?)",
            [
                (str(tmp_path / path),),
                (
                    str(
                        get_upload_path(
                            "document.txt",
                            user_id=1,
                            collection="task_9223372036854775806",
                        )
                    ),
                ),
            ],
        )
    root = tmp_path / "uploads"
    root.mkdir(exist_ok=True)
    (root / "web_task_999").write_text("a file, not a workspace")
    (root / "web_task_9" / "output" / "web_task_999" / "output").mkdir(parents=True)
    migrate(sqlite_engine)
    with sqlite_engine.begin() as connection:
        assert (
            connection.exec_driver_sql("INSERT INTO tasks DEFAULT VALUES").lastrowid
            == 10
        )


@pytest.mark.parametrize(
    ("value", "succeeds"),
    [
        ((1 << 53) - 2, True),
        ((1 << 53) - 1, False),
        (1 << 53, False),
        ((1 << 63) - 2, False),
        ((1 << 63) - 1, False),
    ],
)
def test_historical_ids_leave_the_next_javascript_id_safe(
    sqlite_engine, value, succeeds
):
    seed_legacy(sqlite_engine)
    with sqlite_engine.begin() as connection:
        connection.exec_driver_sql("CREATE TABLE uploaded_files(storage_path TEXT)")
        connection.exec_driver_sql(
            "INSERT INTO uploaded_files VALUES (?)",
            (f"/web_task_{value}/output/report.txt",),
        )
        before = connection.exec_driver_sql(
            "SELECT sql FROM sqlite_master WHERE name='tasks'"
        ).scalar()
    if succeeds:
        migrate(sqlite_engine)
        with sqlite_engine.begin() as connection:
            assert (
                connection.exec_driver_sql("INSERT INTO tasks DEFAULT VALUES").lastrowid
                == (1 << 53) - 1
            )
        return
    with pytest.raises(RuntimeError, match=rf"{value}.*uploaded-file path.*not safe"):
        migrate(sqlite_engine)
    with sqlite_engine.connect() as connection:
        assert (
            connection.exec_driver_sql(
                "SELECT sql FROM sqlite_master WHERE name='tasks'"
            ).scalar()
            == before
        )
        assert connection.exec_driver_sql("PRAGMA foreign_keys").scalar() == 1


def test_unsafe_direct_migration_requires_production_connection(sqlite_engine):
    seed_legacy(sqlite_engine)
    migration = load_migration_module(MIGRATION)
    with sqlite_engine.begin() as connection:
        with Operations.context(MigrationContext.configure(connection)):
            with pytest.raises(RuntimeError, match="_migration_connection"):
                migration.upgrade()


@pytest.mark.parametrize("dialect", ["sqlite", "postgresql"])
def test_offline_migration_contract(dialect):
    migration = load_migration_module(MIGRATION)
    with Operations.context(
        MigrationContext.configure(dialect_name=dialect, opts={"as_sql": True})
    ):
        if dialect == "sqlite":
            with pytest.raises(RuntimeError, match="online SQLite"):
                migration.upgrade()
        else:
            migration.upgrade()


def test_empty_database_upgrade_defers_to_create_all(sqlite_engine):
    migrate(sqlite_engine)
    assert not sa.inspect(sqlite_engine).has_table("tasks")
    Base.metadata.create_all(sqlite_engine)
    with sqlite_engine.connect() as connection:
        assert (
            "AUTOINCREMENT"
            in connection.exec_driver_sql(
                "SELECT sql FROM sqlite_master WHERE name='tasks'"
            ).scalar()
        )


def test_full_chain_preserves_absent_workspace_identity(sqlite_engine, tmp_path):
    metadata = sa.MetaData()
    for table in Base.metadata.tables.values():
        table.to_metadata(metadata)
    metadata.tables["tasks"].dialect_options["sqlite"]["autoincrement"] = False
    metadata.create_all(sqlite_engine)
    (tmp_path / "uploads" / "web_task_701" / "output").mkdir(parents=True)
    config = create_alembic_config(sqlite_engine)
    command.upgrade(config, "head")
    with sqlite_engine.begin() as connection:
        assert (
            connection.exec_driver_sql(
                "SELECT version_num FROM alembic_version"
            ).scalar()
            == ScriptDirectory.from_config(config).get_current_head()
        )
        assert (
            "AUTOINCREMENT"
            in connection.exec_driver_sql(
                "SELECT sql FROM sqlite_master WHERE name='tasks'"
            ).scalar()
        )
        assert (
            connection.exec_driver_sql(
                "SELECT seq FROM sqlite_sequence WHERE name='tasks'"
            ).scalar()
            == 701
        )


def test_startup_upgrade_preserves_populated_model_and_circular_foreign_keys(
    sqlite_engine,
):
    from datetime import UTC, datetime

    from xagent.db.migration import get_alembic_revision, try_upgrade_db
    from xagent.web.models.task import TraceEvent

    metadata = sa.MetaData()
    for table in Base.metadata.tables.values():
        table.to_metadata(metadata)
    metadata.tables["tasks"].dialect_options["sqlite"]["autoincrement"] = False
    metadata.create_all(sqlite_engine)
    with sessionmaker(bind=sqlite_engine)() as db:
        user = User(username="owner", password_hash="unused")
        db.add(user)
        db.flush()
        task = Task(id=17, title="kept", user_id=user.id)
        db.add(task)
        db.flush()
        trace = TraceEvent(
            task_id=17,
            event_id="checkpoint",
            event_type="checkpoint",
            timestamp=datetime.now(UTC),
            data={"kept": True},
        )
        db.add(trace)
        db.flush()
        task.last_checkpoint_trace_event_id = trace.id
        db.commit()
    with sqlite_engine.begin() as connection:
        connection.exec_driver_sql(
            "CREATE TABLE alembic_version (version_num VARCHAR(255) PRIMARY KEY)"
        )
        connection.exec_driver_sql(
            "INSERT INTO alembic_version VALUES ('20260929_task_attachment_detachment')"
        )
        before_task = connection.exec_driver_sql("SELECT * FROM tasks").all()
        before_trace = connection.exec_driver_sql("SELECT * FROM trace_events").all()
        inspector = sa.inspect(connection)
        before_schema = (
            [
                tuple(row[1:6])
                for row in connection.exec_driver_sql("PRAGMA table_info('tasks')")
            ],
            inspector.get_check_constraints("tasks"),
            inspector.get_indexes("tasks"),
            inspector.get_unique_constraints("tasks"),
        )
        before_fks = {
            name: sorted(
                connection.exec_driver_sql(f"PRAGMA foreign_key_list('{name}')").all(),
                key=lambda row: tuple(str(v) for v in row[2:]),
            )
            for name in sa.inspect(connection).get_table_names()
        }
    try_upgrade_db(sqlite_engine)
    try_upgrade_db(sqlite_engine)
    config = create_alembic_config(sqlite_engine)
    assert (
        get_alembic_revision(sqlite_engine)
        == ScriptDirectory.from_config(config).get_current_head()
    )
    with sqlite_engine.connect() as connection:
        assert connection.exec_driver_sql("SELECT * FROM tasks").all() == before_task
        assert (
            connection.exec_driver_sql("SELECT * FROM trace_events").all()
            == before_trace
        )
        for name, foreign_keys in before_fks.items():
            after = sorted(
                connection.exec_driver_sql(f"PRAGMA foreign_key_list('{name}')").all(),
                key=lambda row: tuple(str(v) for v in row[2:]),
            )
            # SQLite can reorder the task's outbound FK constraint IDs.
            assert [row[1:] for row in after] == [row[1:] for row in foreign_keys]
        assert connection.exec_driver_sql("PRAGMA foreign_key_check").all() == []
        inspector = sa.inspect(connection)
        assert before_schema == (
            [
                tuple(row[1:6])
                for row in connection.exec_driver_sql("PRAGMA table_info('tasks')")
            ],
            inspector.get_check_constraints("tasks"),
            inspector.get_indexes("tasks"),
            inspector.get_unique_constraints("tasks"),
        )
        assert (
            connection.exec_driver_sql(
                "SELECT seq FROM sqlite_sequence WHERE name='tasks'"
            ).scalar()
            == 17
        )
