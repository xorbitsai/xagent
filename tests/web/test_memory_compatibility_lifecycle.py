import json
import multiprocessing
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import lancedb  # type: ignore
import pyarrow as pa  # type: ignore
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from xagent.core.memory.core import MemoryNote
from xagent.core.memory.lancedb import LanceDBMemoryStore
from xagent.core.memory.lancedb_maintenance import (
    MAINTENANCE_METADATA_KEY,
    MaintenanceOutcome,
    MaintenanceStatus,
    maintain_lancedb_memory_table,
)
from xagent.core.memory.scope_columns import SCOPE_DIMS_COLUMN, USER_ID_COLUMN
from xagent.core.memory.vector_compatibility import (
    VECTOR_IDENTITY_METADATA_KEY,
    EmbeddingIdentity,
    VectorCompatibility,
    create_or_recreate_vector_capable_table,
    open_lancedb_table_if_exists,
)
from xagent.core.model import EmbeddingModelConfig
from xagent.core.model.embedding import BaseEmbedding
from xagent.core.tools.core.RAG_tools.LanceDB.schema_manager import _safe_close_table
from xagent.web import dynamic_memory_store as memory_module
from xagent.web.dynamic_memory_store import (
    DynamicMemoryStoreManager,
    MemoryAdmissionLockTimeout,
    MemoryStoreRestartRequired,
    MemoryStoreStartupAdmissionError,
    _embedding_model_config,
)
from xagent.web.models import Base, Model, User, UserDefaultModel, UserModel
from xagent.web.user_isolated_memory import UserContext, UserIsolatedMemoryStore


def _model(**overrides):
    values = {
        "id": 7,
        "model_id": "shared-embedding",
        "model_provider": "dashscope",
        "model_name": "text-embedding-v4",
        "api_key": "shared-secret",
        "base_url": "https://embedding.example/v1",
        "dimension": 768,
        "instruct": "retrieval.document",
        "max_retries": 4,
        "updated_at": "2026-09-09T00:00:00Z",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _store(connection, embedding=object()):
    base = LanceDBMemoryStore.__new__(LanceDBMemoryStore)
    base._collection_name = "memories"
    base._embedding_model = embedding
    base._vector_store = SimpleNamespace(get_raw_connection=lambda: connection)
    return UserIsolatedMemoryStore(base), base


def _fake_table(*, vector=True):
    names = ["id", "text", "metadata"] + (["vector"] if vector else [])
    return SimpleNamespace(
        close=lambda: None,
        schema=SimpleNamespace(names=names, metadata={}),
    )


def _mock_lifecycle(monkeypatch, connection):
    wrapper, base = _store(connection)
    manager = DynamicMemoryStoreManager()
    monkeypatch.setattr(manager, "_get_embedding_model_from_db", lambda **_: _model())
    monkeypatch.setattr(manager, "_create_lancedb_store", lambda _model: wrapper)
    monkeypatch.setattr(manager, "_lifecycle_lock", lambda *_: nullcontext())
    return manager, wrapper, base


class _ProcessLifecycleManager(DynamicMemoryStoreManager):
    def __init__(self, db_dir, acquired=None, release=None, attempting=None):
        super().__init__()
        self._db_dir = db_dir
        self._acquired = acquired
        self._release = release
        self._attempting = attempting
        self.lock_exit_version = None

    def _get_embedding_model_from_db(self, *, fail_fast=False):
        return _model(dimension=4)

    def _create_lancedb_store(self, embedding_model):
        return UserIsolatedMemoryStore(
            LanceDBMemoryStore(
                self._db_dir,
                embedding_model=embedding_model,
                initialize_schema=False,
                include_null_vector_fallback=True,
            )
        )

    @contextmanager
    def _lifecycle_lock(self, connection, table_name):
        if self._attempting is not None:
            self._attempting.set()
        with super()._lifecycle_lock(connection, table_name):
            if self._acquired is not None:
                self._acquired.set()
            if self._release is not None:
                self._release.wait()
            yield
            table = connection.open_table(table_name)
            try:
                self.lock_exit_version = table.version
            finally:
                _safe_close_table(table)


def _run_process_lifecycle(db_dir, queue, acquired=None, release=None, attempting=None):
    try:
        manager = _ProcessLifecycleManager(db_dir, acquired, release, attempting)
        manager.run_startup_compatibility_lifecycle()
        queue.put(("ok", manager.lock_exit_version))
    except Exception as error:
        queue.put(("error", type(error).__name__, str(error)))


def test_shared_config_and_store_are_stable_across_request_users(monkeypatch):
    config = _embedding_model_config(_model())
    assert config == EmbeddingModelConfig(
        id="shared-embedding",
        model_provider="dashscope",
        model_name="text-embedding-v4",
        api_key="shared-secret",
        base_url="https://embedding.example/v1",
        dimension=768,
        instruct=None,
        max_retries=4,
    )

    manager = DynamicMemoryStoreManager()
    shared = manager._memory_store
    monkeypatch.setattr(
        manager,
        "_get_embedding_model_from_db",
        lambda **_: pytest.fail("request acquisition resolved an embedding identity"),
    )
    with UserContext(101):
        first = manager.get_memory_store()
    with UserContext(202):
        second = manager.get_memory_store()
    assert first is second is shared


def test_published_store_rejects_authoritative_fingerprint_drift(monkeypatch):
    model = _model()
    connection = SimpleNamespace(
        list_tables=lambda: ["memories"],
        open_table=lambda _name: _fake_table(),
    )
    manager, wrapper, _base = _mock_lifecycle(monkeypatch, connection)
    monkeypatch.setattr(manager, "_get_embedding_model_from_db", lambda **_: model)
    monkeypatch.setattr(
        memory_module,
        "maintain_lancedb_memory_table",
        lambda *_args: MaintenanceOutcome(MaintenanceStatus.COMPLETE),
    )
    monkeypatch.setattr(
        memory_module,
        "create_or_recreate_vector_capable_table",
        lambda *_args: VectorCompatibility.MATCHING,
    )

    manager.run_startup_compatibility_lifecycle()
    assert manager.get_memory_store() is wrapper

    model.api_key = "rotated-secret"
    with pytest.raises(MemoryStoreRestartRequired) as exc_info:
        manager.get_memory_store()

    detail = str(exc_info.value)
    assert "restart every worker" in detail
    assert "shared-secret" not in detail
    assert "rotated-secret" not in detail


def test_manager_constructs_dormant_store_with_complete_shared_config(
    monkeypatch, tmp_path
):
    captured = {}
    monkeypatch.setattr(memory_module, "get_storage_root", lambda: tmp_path)
    monkeypatch.setattr(
        memory_module,
        "LanceDBMemoryStore",
        lambda **kwargs: captured.update(kwargs) or SimpleNamespace(),
    )

    embedding = Mock(spec=BaseEmbedding)
    DynamicMemoryStoreManager()._create_lancedb_store(embedding)

    assert captured["initialize_schema"] is False
    assert captured["include_null_vector_fallback"] is True
    assert captured["embedding_model"] is embedding


def test_startup_lifecycle_unwraps_serializes_and_orders_primitives(monkeypatch):
    events = []

    class Connection:
        def list_tables(self):
            return ["memories"]

        def open_table(self, name):
            events.append(("open", name))
            return _fake_table()

    manager, wrapper, _base = _mock_lifecycle(monkeypatch, Connection())

    def maintain(connection, name):
        assert manager._lock._is_owned()  # type: ignore[attr-defined]
        events.append(("maintain", connection, name))
        return MaintenanceOutcome(MaintenanceStatus.COMPLETE)

    def admit(connection, name, identity):
        events.append(("admit", connection, name, identity))
        return VectorCompatibility.MATCHING

    monkeypatch.setattr(memory_module, "maintain_lancedb_memory_table", maintain)
    monkeypatch.setattr(memory_module, "create_or_recreate_vector_capable_table", admit)

    manager.run_startup_compatibility_lifecycle()

    assert [event[0] for event in events] == ["open", "maintain", "admit"]
    assert events[1][1] is events[2][1]
    assert events[2][3].model_name == "text-embedding-v4"
    assert manager._memory_store is wrapper


def test_malformed_legacy_data_fails_closed_without_publication(monkeypatch, caplog):
    connection = SimpleNamespace(
        list_tables=lambda: ["memories"],
        open_table=lambda _name: _fake_table(),
    )
    manager, _wrapper, base = _mock_lifecycle(monkeypatch, connection)
    monkeypatch.setattr(
        memory_module,
        "maintain_lancedb_memory_table",
        lambda *_args: MaintenanceOutcome(
            MaintenanceStatus.INVALID_LEGACY_DATA,
            detail="duplicate legacy IDs",
        ),
    )
    recreate = Mock()
    monkeypatch.setattr(
        memory_module, "create_or_recreate_vector_capable_table", recreate
    )

    previous = manager._memory_store
    with (
        caplog.at_level("ERROR"),
        pytest.raises(
            MemoryStoreStartupAdmissionError, match="repair the table offline"
        ),
    ):
        manager.run_startup_compatibility_lifecycle()

    assert base._embedding_model is not None
    assert manager._memory_store is previous
    assert manager._is_lancedb is False
    recreate.assert_not_called()
    assert "repair the table offline" in caplog.text
    assert "duplicate legacy IDs" not in caplog.text
    assert "shared-secret" not in caplog.text


def test_missing_table_is_recreated_then_maintained(monkeypatch):
    events = []

    class Connection:
        def list_tables(self):
            return []

    manager, _wrapper, _base = _mock_lifecycle(monkeypatch, Connection())
    monkeypatch.setattr(
        memory_module,
        "create_or_recreate_vector_capable_table",
        lambda *_args: events.append("recreate") or VectorCompatibility.MATCHING,
    )
    monkeypatch.setattr(
        memory_module,
        "maintain_lancedb_memory_table",
        lambda *_args: (
            events.append("maintain") or MaintenanceOutcome(MaintenanceStatus.COMPLETE)
        ),
    )

    manager.run_startup_compatibility_lifecycle()

    assert events == ["recreate", "maintain"]


def test_mismatching_vector_space_admits_text_only(monkeypatch):
    connection = SimpleNamespace(
        list_tables=lambda: ["memories"],
        open_table=lambda _name: _fake_table(),
    )
    manager, _wrapper, base = _mock_lifecycle(monkeypatch, connection)
    monkeypatch.setattr(
        memory_module,
        "maintain_lancedb_memory_table",
        lambda *_args: MaintenanceOutcome(MaintenanceStatus.COMPLETE),
    )
    monkeypatch.setattr(
        memory_module,
        "create_or_recreate_vector_capable_table",
        lambda *_args: VectorCompatibility.MISMATCHING,
    )

    manager.run_startup_compatibility_lifecycle()

    assert base._embedding_model is None


def test_real_maintenance_failure_propagates_and_preserves_manager_state(monkeypatch):
    connection = SimpleNamespace(
        list_tables=lambda: ["memories"],
        open_table=lambda _name: _fake_table(),
    )
    manager, _wrapper, _base = _mock_lifecycle(monkeypatch, connection)
    previous = manager._memory_store
    manager._is_lancedb = True
    manager._last_embedding_model_id = 3
    monkeypatch.setattr(
        memory_module,
        "maintain_lancedb_memory_table",
        lambda *_args: (_ for _ in ()).throw(OSError("real I/O failure")),
    )

    with pytest.raises(OSError, match="real I/O failure"):
        manager.run_startup_compatibility_lifecycle()

    assert manager._memory_store is previous
    assert manager._is_lancedb is True
    assert manager._last_embedding_model_id == 3


def _use_model(manager, monkeypatch, model=None):
    monkeypatch.setattr(
        manager,
        "_get_embedding_model_from_db",
        lambda **_: model or _model(),
    )


@pytest.mark.parametrize(
    "model",
    [
        pytest.param(_model(dimension=None), id="missing-dimension"),
        pytest.param(_model(dimension=object()), id="invalid-field-type"),
        pytest.param(
            _model(model_provider="unsupported-provider"),
            id="unsupported-provider",
        ),
    ],
)
def test_invalid_model_config_preserves_manager_without_touching_filesystem(
    monkeypatch, caplog, model
):
    manager = DynamicMemoryStoreManager()
    previous = manager._memory_store
    _use_model(manager, monkeypatch, model)
    create_store = Mock(side_effect=AssertionError("filesystem was touched"))
    monkeypatch.setattr(manager, "_create_lancedb_store", create_store)

    with caplog.at_level("WARNING"):
        manager.run_startup_compatibility_lifecycle()

    assert manager._memory_store is previous
    assert manager._is_lancedb is False
    create_store.assert_not_called()
    assert "Configured default embedding model" in caplog.text
    assert "shared-secret" not in caplog.text


def test_unexpected_config_builder_type_error_is_not_degraded(monkeypatch):
    manager = DynamicMemoryStoreManager()
    _use_model(manager, monkeypatch)
    monkeypatch.setattr(
        memory_module,
        "_embedding_model_config",
        Mock(side_effect=TypeError("config builder bug")),
    )

    with pytest.raises(TypeError, match="config builder bug"):
        manager.run_startup_compatibility_lifecycle()


def test_adapter_non_config_failure_propagates_and_preserves_state(monkeypatch):
    manager = DynamicMemoryStoreManager()
    previous = manager._memory_store
    _use_model(manager, monkeypatch)
    monkeypatch.setattr(
        memory_module,
        "create_embedding_adapter",
        lambda _config: (_ for _ in ()).throw(OSError("adapter I/O failure")),
    )

    with pytest.raises(OSError, match="adapter I/O failure"):
        manager.run_startup_compatibility_lifecycle()

    assert manager._memory_store is previous
    assert manager._is_lancedb is False


@pytest.mark.parametrize(
    "error",
    [TypeError("adapter bug"), ValueError("adapter bug")],
    ids=["type-error", "value-error"],
)
def test_unexpected_adapter_error_is_not_degraded(monkeypatch, error):
    manager = DynamicMemoryStoreManager()
    _use_model(manager, monkeypatch)
    monkeypatch.setattr(
        memory_module,
        "create_embedding_adapter",
        Mock(side_effect=error),
    )

    with pytest.raises(type(error), match="adapter bug"):
        manager.run_startup_compatibility_lifecycle()


def test_admission_lock_contention_is_bounded_and_fail_fast(tmp_path, monkeypatch):
    class ContendedLock:
        def __init__(self, path, timeout):
            assert path.startswith(str(tmp_path))
            assert timeout > 0

        def acquire(self):
            raise memory_module.Timeout("busy")

    monkeypatch.setattr(memory_module, "FileLock", ContendedLock)
    connection = SimpleNamespace(uri=str(tmp_path))

    with pytest.raises(MemoryAdmissionLockTimeout, match="Timed out after"):
        with DynamicMemoryStoreManager._lifecycle_lock(connection, "memories"):
            pytest.fail("contended lifecycle entered critical section")


@pytest.fixture
def sqlite_model_db(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'models.db'}")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)

    def get_db():
        with session_factory() as session:
            yield session

    monkeypatch.setattr(memory_module, "get_db", get_db)
    yield session_factory
    engine.dispose()


def _db_model(model_id, *, active=True):
    model = Model(
        model_id=model_id,
        category="embedding",
        model_provider="dashscope",
        model_name="text-embedding-v4",
        dimension=1024,
        is_active=active,
    )
    model.api_key = "test-key"
    return model


def _add_admin_default(db, user_id, label, *, admin=True, active=True, shared=False):
    user = User(
        id=user_id,
        username=f"{label}-user",
        password_hash="hash",
        is_admin=admin,
    )
    model = _db_model(f"{label}-model", active=active)
    db.add_all([user, model])
    db.flush()
    db.add(
        UserDefaultModel(
            user_id=user_id,
            model_id=model.id,
            config_type="embedding",
        )
    )
    if shared:
        db.add(
            UserModel(
                user_id=user_id,
                model_id=model.id,
                is_owner=True,
                is_shared=True,
            )
        )
    return model


def test_real_sqlite_admin_default_predicates_and_deterministic_order(
    sqlite_model_db,
):
    with sqlite_model_db() as db:
        _add_admin_default(db, 1, "member", admin=False)
        _add_admin_default(db, 5, "inactive", active=False)
        _add_admin_default(db, 10, "first-admin")
        _add_admin_default(db, 20, "second-admin", shared=True)
        # No UserModel row exists for first_model: an unshared admin default is
        # authoritative without mutating the sharing relationship.
        db.commit()

    selected = DynamicMemoryStoreManager()._get_embedding_model_from_db(fail_fast=True)
    assert selected is not None
    assert selected.model_id == "first-admin-model"


def test_real_sqlite_visible_admin_hook_can_select_another_admin_default(
    sqlite_model_db, monkeypatch
):
    from xagent.web.services import model_service

    with sqlite_model_db() as db:
        _add_admin_default(db, 10, "first-admin")
        _add_admin_default(db, 20, "second-admin", shared=True)
        db.commit()

    monkeypatch.setattr(
        model_service, "_visible_user_ids_hook", lambda _db, _user_id: [20]
    )
    selected = DynamicMemoryStoreManager()._get_embedding_model_from_db(fail_fast=True)
    assert selected is not None
    assert selected.model_id == "second-admin-model"


def _vectorless_table(db_dir, rows):
    connection = lancedb.connect(db_dir)
    table = connection.create_table("memories", pa.Table.from_pylist(rows))
    _safe_close_table(table)
    return connection


def _real_manager(root, monkeypatch):
    manager = DynamicMemoryStoreManager()
    _use_model(manager, monkeypatch, _model(dimension=4))
    monkeypatch.setattr(memory_module, "get_storage_root", lambda: root)
    return manager


def test_real_vectorless_lifecycle_recreates_and_marks_complete_same_startup(
    tmp_path, monkeypatch
):
    db_dir = tmp_path / "memory_store"
    db_dir.mkdir()
    _vectorless_table(
        db_dir,
        [
            {
                "id": "legacy",
                "text": "remember",
                "metadata": json.dumps(
                    {"user_id": 7, "execution_scope_project": "alpha"}
                ),
            }
        ],
    )
    manager = _real_manager(tmp_path, monkeypatch)

    manager.run_startup_compatibility_lifecycle()

    base = manager._memory_store._base_store
    table = base._vector_store.get_raw_connection().open_table("memories")
    try:
        assert table.to_arrow().to_pylist() == [
            {
                "id": "legacy",
                "text": "remember",
                "metadata": json.dumps(
                    {"user_id": 7, "execution_scope_project": "alpha"}
                ),
                "vector": None,
                USER_ID_COLUMN: 7,
                SCOPE_DIMS_COLUMN: ["project=alpha"],
            }
        ]
        assert (table.schema.field(USER_ID_COLUMN).metadata or {}).get(
            MAINTENANCE_METADATA_KEY
        ) == b"1"
    finally:
        _safe_close_table(table)
    assert manager.get_store_info()["supports_vector_search"] is True
    assert (
        maintain_lancedb_memory_table(
            base._vector_store.get_raw_connection(), "memories"
        ).scanned_rows
        == 0
    )


def test_real_existing_table_without_default_is_admitted_text_only(
    tmp_path, monkeypatch
):
    db_dir = tmp_path / "memory_store"
    db_dir.mkdir()
    _vectorless_table(
        db_dir,
        [
            {
                "id": "legacy",
                "text": "remember alpha",
                "metadata": json.dumps({"user_id": 7}),
            }
        ],
    )
    manager = DynamicMemoryStoreManager()
    monkeypatch.setattr(memory_module, "get_storage_root", lambda: tmp_path)
    monkeypatch.setattr(manager, "_get_embedding_model_from_db", lambda **_: None)

    manager.run_startup_compatibility_lifecycle()

    assert manager.get_store_info()["is_lancedb"] is True
    assert manager.get_store_info()["supports_vector_search"] is False
    with UserContext(7):
        store = manager.get_memory_store()
        assert [note.id for note in store.list_all()] == ["legacy"]
        assert store.add(MemoryNote(content="new durable memory")).success
    table = lancedb.connect(db_dir).open_table("memories")
    try:
        rows = table.to_arrow().to_pylist()
        assert len(rows) == 2
        assert "legacy" in {row["id"] for row in rows}
        assert {row[USER_ID_COLUMN] for row in rows} == {7}
        assert {USER_ID_COLUMN, SCOPE_DIMS_COLUMN} <= set(table.schema.names)
    finally:
        _safe_close_table(table)


def test_no_default_and_no_table_does_not_create_persistent_artifacts(
    tmp_path, monkeypatch
):
    manager = DynamicMemoryStoreManager()
    monkeypatch.setattr(memory_module, "get_storage_root", lambda: tmp_path)
    monkeypatch.setattr(manager, "_get_embedding_model_from_db", lambda **_: None)

    manager.run_startup_compatibility_lifecycle()

    assert not (tmp_path / "memory_store").exists()
    assert manager.get_store_info()["is_lancedb"] is False


def test_real_malformed_legacy_table_blocks_scoped_reads_and_writes(
    tmp_path, monkeypatch
):
    db_dir = tmp_path / "memory_store"
    db_dir.mkdir()
    connection = _vectorless_table(
        db_dir,
        [
            {"id": "duplicate", "text": "first", "metadata": '{"user_id": 7}'},
            {"id": "duplicate", "text": "second", "metadata": '{"user_id": 7}'},
        ],
    )
    table = connection.open_table("memories")
    try:
        before_schema = table.schema
        before_rows = table.to_arrow().to_pylist()
    finally:
        _safe_close_table(table)
    manager = _real_manager(tmp_path, monkeypatch)
    previous = manager._memory_store

    with pytest.raises(MemoryStoreStartupAdmissionError):
        manager.run_startup_compatibility_lifecycle()

    assert manager._memory_store is previous
    assert manager._is_lancedb is False
    with UserContext(7):
        with pytest.raises(MemoryStoreStartupAdmissionError):
            manager.get_memory_store().list_all()
        with pytest.raises(MemoryStoreStartupAdmissionError):
            manager.get_memory_store().add(MemoryNote(content="must not migrate"))
    table = connection.open_table("memories")
    try:
        assert table.schema == before_schema
        assert table.to_arrow().to_pylist() == before_rows
    finally:
        _safe_close_table(table)


def test_real_matching_and_mismatching_tables_drive_vector_capability(
    tmp_path, monkeypatch, caplog
):
    for name, identity, expected in (
        (
            "matching",
            EmbeddingIdentity(
                "dashscope",
                "text-embedding-v4",
                "https://embedding.example/v1",
                4,
                None,
            ),
            True,
        ),
        (
            "mismatching",
            EmbeddingIdentity(
                "dashscope", "different-model", "https://embedding.example/v1", 4, None
            ),
            False,
        ),
    ):
        root = tmp_path / name
        db_dir = root / "memory_store"
        db_dir.mkdir(parents=True)
        connection = lancedb.connect(db_dir)
        create_or_recreate_vector_capable_table(connection, "memories", identity)
        manager = _real_manager(root, monkeypatch)
        with caplog.at_level("WARNING"):
            manager.run_startup_compatibility_lifecycle()
        assert manager.get_store_info()["supports_vector_search"] is expected
    assert "vector identity does not match" in caplog.text
    assert "shared-secret" not in caplog.text


def test_real_incompatible_scope_schema_propagates_and_preserves_manager(
    tmp_path, monkeypatch
):
    db_dir = tmp_path / "memory_store"
    db_dir.mkdir()
    connection = lancedb.connect(db_dir)
    table = connection.create_table(
        "memories",
        pa.table(
            {
                "id": ["legacy"],
                "text": ["remember"],
                "metadata": ["{}"],
                USER_ID_COLUMN: ["wrong-type"],
                SCOPE_DIMS_COLUMN: pa.array([[]], pa.list_(pa.string())),
            }
        ),
    )
    _safe_close_table(table)
    manager = _real_manager(tmp_path, monkeypatch)
    previous = manager._memory_store

    with pytest.raises(RuntimeError, match="incompatible_schema"):
        manager.run_startup_compatibility_lifecycle()

    assert manager._memory_store is previous
    assert manager._is_lancedb is False


def test_table_existence_helper_distinguishes_absent_and_backend_failures():
    absent = SimpleNamespace(list_tables=lambda: [])
    assert open_lancedb_table_if_exists(absent, "memories") is None

    for error in (ValueError("schema failure"), OSError("I/O failure")):
        connection = SimpleNamespace(
            list_tables=lambda: ["memories"],
            open_table=Mock(side_effect=error),
        )
        with pytest.raises(type(error), match=str(error)):
            open_lancedb_table_if_exists(connection, "memories")


def test_cross_process_lifecycle_reopens_after_lock_without_stale_overwrite(tmp_path):
    db_dir = tmp_path / "shared-memory"
    db_dir.mkdir()
    connection = _vectorless_table(
        db_dir,
        [{"id": "kept", "text": "remember", "metadata": "{}"}],
    )
    del connection

    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    first_acquired = context.Event()
    release_first = context.Event()
    second_attempting = context.Event()
    second_acquired = context.Event()
    first = context.Process(
        target=_run_process_lifecycle,
        args=(str(db_dir), queue, first_acquired, release_first),
    )
    processes = []
    try:
        first.start()
        processes.append(first)
        assert first_acquired.wait(timeout=30), (
            queue.get(timeout=5)
            if not queue.empty()
            else "first worker did not initialize"
        )

        # The second process constructs its connection while the first still owns
        # the lock and the on-disk table is vectorless. Correct admission performs
        # no table read until after it acquires the lock.
        second = context.Process(
            target=_run_process_lifecycle,
            args=(str(db_dir), queue, second_acquired, None, second_attempting),
        )
        second.start()
        processes.append(second)
        assert second_attempting.wait(timeout=30)
        second_was_blocked = not second_acquired.wait(timeout=1)
        release_first.set()
        first.join(timeout=30)
        second.join(timeout=30)
        assert second_was_blocked
        assert first.exitcode == 0
        assert second.exitcode == 0
        outcomes = [queue.get(timeout=5), queue.get(timeout=5)]
        assert all(outcome[0] == "ok" for outcome in outcomes), outcomes
        assert outcomes[0][1] == outcomes[1][1]
    finally:
        release_first.set()
        for process in processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)

    final_connection = lancedb.connect(db_dir)
    table = final_connection.open_table("memories")
    try:
        rows = table.to_arrow().to_pylist()
        assert [row["id"] for row in rows] == ["kept"]
        assert "vector" in table.schema.names
        assert VECTOR_IDENTITY_METADATA_KEY in (table.schema.metadata or {})
        assert (table.schema.field(USER_ID_COLUMN).metadata or {}).get(
            MAINTENANCE_METADATA_KEY
        ) == b"1"
    finally:
        _safe_close_table(table)
