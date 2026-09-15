import json
import math
import multiprocessing
from pathlib import Path

import lancedb  # type: ignore
import pyarrow as pa  # type: ignore
import pytest
from filelock import FileLock, Timeout

from xagent.core.memory import storage_admission, vector_compatibility
from xagent.core.memory.lancedb_maintenance import (
    MAINTENANCE_METADATA_KEY,
    MAINTENANCE_TABLE_VERSION_KEY,
    MAINTENANCE_VERSION,
    MaintenanceOutcome,
    MaintenanceStatus,
    lancedb_lock_path,
)
from xagent.core.memory.storage_admission import (
    ADMISSION_FAILED_DETAIL,
    REPAIR_REQUIRED_DETAIL,
    DormantLanceDBMemoryHandle,
    MemoryStorageMode,
    StorageAdmissionState,
    admit_lancedb_memory_storage,
)
from xagent.core.memory.vector_compatibility import EmbeddingIdentity
from xagent.core.tools.core.RAG_tools.LanceDB.schema_manager import _safe_close_table

IDENTITY = EmbeddingIdentity(
    "openai", "text-embedding-3-small", "https://api.openai.com/v1/embeddings", 4, None
)


def _connection(tmp_path, *, invalid=False):
    connection = lancedb.connect(tmp_path)
    rows = 5
    table = connection.create_table(
        "memories",
        pa.table(
            {
                "id": [
                    None if invalid and index == 4 else f"note-{index}"
                    for index in range(rows)
                ],
                "text": [f"text-{index}" for index in range(rows)],
                "metadata": [json.dumps({"user_id": index}) for index in range(rows)],
            }
        ),
    )
    _safe_close_table(table)
    return connection


def _snapshot(connection):
    table = connection.open_table("memories")
    try:
        return table.version, table.schema, table.to_arrow().to_pylist()
    finally:
        _safe_close_table(table)


def _admit(connection, **kwargs):
    return admit_lancedb_memory_storage(
        DormantLanceDBMemoryHandle(connection, "memories"),
        IDENTITY,
        writers_quiesced=True,
        batch_size=2,
        **kwargs,
    )


def _admit_after_lock_attempt(database_path, attempted, results):
    original_lock = storage_admission.FileLock

    class SignalingLock:
        def __init__(self, *args, **kwargs):
            self._lock = original_lock(*args, **kwargs)

        def __enter__(self):
            attempted.set()
            return self._lock.__enter__()

        def __exit__(self, *args):
            return self._lock.__exit__(*args)

    storage_admission.FileLock = SignalingLock
    results.put(_admit(lancedb.connect(database_path)).state.value)


def test_supported_version_atomic_null_vectors_marker_and_bounded_scan(
    tmp_path, monkeypatch
):
    connection = _connection(tmp_path)
    before_version = _snapshot(connection)[0]
    scanned = []
    null_types = []
    original = vector_compatibility._checkpoint
    original_nulls = vector_compatibility.pa.nulls

    def observe(stage, batch=None):
        if stage == "scan_batch":
            scanned.append(batch)
        original(stage, batch)

    monkeypatch.setattr(vector_compatibility, "_checkpoint", observe)

    def observe_nulls(size, type):
        null_types.append(type)
        return original_nulls(size, type)

    monkeypatch.setattr(vector_compatibility.pa, "nulls", observe_nulls)
    outcome = _admit(connection)
    table = connection.open_table("memories")
    field = table.schema.field("user_id")
    assert outcome.state is StorageAdmissionState.ADMITTED
    assert outcome.admitted.capabilities.mode is MemoryStorageMode.VECTOR
    assert outcome.admitted.capabilities.vector_search is True
    assert table.version == before_version + 1
    assert field.metadata[MAINTENANCE_METADATA_KEY] == MAINTENANCE_VERSION
    assert field.metadata[MAINTENANCE_TABLE_VERSION_KEY] == str(table.version).encode()
    assert table.schema.field("vector").type == pa.list_(pa.float32(), 4)
    assert table.to_arrow()["vector"].to_pylist() == [None] * 5
    assert null_types == [pa.list_(pa.float32(), 4)] * 3
    assert scanned == [2, 2, 1]
    scanned.clear()
    assert _admit(connection).state is StorageAdmissionState.ADMITTED
    assert scanned == []


def test_invalid_and_mid_commit_failure_leave_original_unchanged(tmp_path, monkeypatch):
    invalid = _connection(tmp_path / "invalid", invalid=True)
    before = _snapshot(invalid)
    outcome = _admit(invalid)
    assert outcome.state is StorageAdmissionState.BLOCKED_REPAIR
    assert outcome.detail == REPAIR_REQUIRED_DETAIL
    assert _snapshot(invalid) == before
    assert not list((tmp_path / "invalid").glob(".memory-stage-*"))
    assert not list((tmp_path / "invalid").glob(".memory-seen-*"))

    incompatible = lancedb.connect(tmp_path / "incompatible")
    table = incompatible.create_table(
        "memories",
        pa.table({"id": [1], "text": ["text"], "metadata": ["{}"]}),
    )
    _safe_close_table(table)
    before = _snapshot(incompatible)
    assert _admit(incompatible).state is StorageAdmissionState.BLOCKED_REPAIR
    assert _snapshot(incompatible) == before

    connection = _connection(tmp_path / "failure")
    before = _snapshot(connection)

    def fail(stage, batch=None):
        if stage == "commit_batch" and batch == 1:
            raise RuntimeError("secret backend path /do/not/expose")

    monkeypatch.setattr(vector_compatibility, "_checkpoint", fail)
    outcome = _admit(connection)
    assert outcome.state is StorageAdmissionState.RETRYABLE_UNAVAILABLE
    assert outcome.detail == ADMISSION_FAILED_DETAIL
    assert "secret" not in outcome.detail
    assert _snapshot(connection) == before


def test_absent_leaves_only_coordination_lock_and_pagination_finds_late_table(tmp_path):
    absent_path = tmp_path / "absent"
    connection = lancedb.connect(absent_path)
    before = set(absent_path.iterdir())
    admission_path = Path(lancedb_lock_path(connection, "memories", "admission"))
    outcome = _admit(connection)
    assert outcome.state is StorageAdmissionState.ABSENT
    assert set(absent_path.iterdir()) == before | {admission_path}
    assert list(connection.table_names(limit=100)) == []

    paged = lancedb.connect(tmp_path / "paged")
    for index in range(11):
        table = paged.create_table(
            f"decoy-{index:02d}", schema=pa.schema([("x", pa.string())])
        )
        _safe_close_table(table)
    table = paged.create_table(
        "memories",
        pa.table({"id": ["kept"], "text": ["text"], "metadata": ["{}"]}),
    )
    _safe_close_table(table)
    assert _admit(paged).state is StorageAdmissionState.ADMITTED


def test_absent_admission_waits_for_lifecycle_lock(tmp_path):
    database_path = tmp_path / "lifecycle-race"
    connection = lancedb.connect(database_path)
    admission_path = lancedb_lock_path(connection, "memories", "admission")
    context = multiprocessing.get_context("spawn")
    attempted = context.Event()
    results = context.Queue()
    process = context.Process(
        target=_admit_after_lock_attempt,
        args=(str(database_path), attempted, results),
    )

    with FileLock(admission_path):
        process.start()
        assert attempted.wait(10)
        table = connection.create_table(
            "memories",
            pa.table({"id": ["created"], "text": ["text"], "metadata": ["{}"]}),
        )
        _safe_close_table(table)
    process.join(20)
    if process.is_alive():
        process.terminate()
        process.join(5)
    assert process.exitcode == 0
    assert results.get(timeout=5) == StorageAdmissionState.ADMITTED.value


def test_typed_unavailable_states_and_safe_details(tmp_path, monkeypatch):
    connection = _connection(tmp_path)
    dormant = DormantLanceDBMemoryHandle(connection, "memories")
    outcome = admit_lancedb_memory_storage(dormant, IDENTITY, writers_quiesced=False)
    assert outcome.state is StorageAdmissionState.QUIESCENCE_REQUIRED

    admission_path = lancedb_lock_path(connection, "memories", "admission")
    with FileLock(admission_path):
        outcome = _admit(connection, lock_timeout=0)
    assert outcome.state is StorageAdmissionState.RETRYABLE_UNAVAILABLE
    assert outcome.detail == ADMISSION_FAILED_DETAIL

    monkeypatch.setattr(
        "xagent.core.memory.storage_admission.prepare_lancedb_memory_table",
        lambda *_args, **_kwargs: MaintenanceOutcome(MaintenanceStatus.INCOMPLETE),
    )
    outcome = _admit(connection)
    assert outcome.state is StorageAdmissionState.MAINTENANCE_INCOMPLETE
    assert outcome.detail == ADMISSION_FAILED_DETAIL

    def fail_discovery(*_args, **_kwargs):
        raise OSError("credential at /private/backend")

    monkeypatch.setattr(
        "xagent.core.memory.storage_admission._lancedb_table_exists", fail_discovery
    )
    outcome = _admit(connection)
    assert outcome.state is StorageAdmissionState.RETRYABLE_UNAVAILABLE
    assert outcome.detail == ADMISSION_FAILED_DETAIL


def test_mismatching_vectors_have_typed_text_only_capability(tmp_path):
    connection = lancedb.connect(tmp_path)
    table = connection.create_table(
        "memories",
        pa.table(
            {
                "id": ["legacy"],
                "text": ["text"],
                "metadata": ["{}"],
                "vector": pa.array([[1.0] * 4], pa.list_(pa.float32(), 4)),
            }
        ),
    )
    _safe_close_table(table)
    outcome = _admit(connection)
    assert outcome.state is StorageAdmissionState.ADMITTED
    assert outcome.admitted.capabilities.mode is MemoryStorageMode.TEXT_ONLY
    assert outcome.admitted.capabilities.vector_search is False


@pytest.mark.parametrize("case", ["nan", "wrong_length"])
def test_malformed_existing_vectors_are_blocked_without_mutation(tmp_path, case):
    connection = lancedb.connect(tmp_path)
    vectors = [[math.nan, 0.0, 0.0, 0.0]]
    vector_type = pa.list_(pa.float32(), 4)
    if case == "wrong_length":
        vectors = [[1.0, 2.0, 3.0, 4.0], [1.0, 2.0]]
        vector_type = pa.list_(pa.float32())
    rows = len(vectors)
    table = connection.create_table(
        "memories",
        pa.table(
            {
                "id": [f"note-{index}" for index in range(rows)],
                "text": ["text"] * rows,
                "metadata": ["{}"] * rows,
                "badvec": pa.array(vectors, type=vector_type),
            }
        ),
    )
    table.alter_columns({"path": "badvec", "rename": "vector"})
    _safe_close_table(table)
    before = _snapshot(connection)

    outcome = _admit(connection)
    after = _snapshot(connection)
    assert outcome.state is StorageAdmissionState.BLOCKED_REPAIR
    assert outcome.detail == REPAIR_REQUIRED_DETAIL
    assert after[:2] == before[:2]
    if case == "nan":
        assert math.isnan(after[2][0]["vector"][0])
    else:
        assert after[2] == before[2]


def test_existing_null_vectors_are_preserved(tmp_path):
    connection = lancedb.connect(tmp_path)
    vector_type = pa.list_(pa.float32(), 4)
    table = connection.create_table(
        "memories",
        pa.table(
            {
                "id": ["null", "valid"],
                "text": ["a", "b"],
                "metadata": ["{}", "{}"],
                "vector": pa.array([None, [1.0, 2.0, 3.0, 4.0]], vector_type),
            }
        ),
        on_bad_vectors="null",
    )
    _safe_close_table(table)
    outcome = _admit(connection)
    assert outcome.state is StorageAdmissionState.ADMITTED
    assert _snapshot(connection)[2][0]["vector"] is None


def test_compatibility_is_not_reinspected_after_commit(tmp_path, monkeypatch):
    connection = _connection(tmp_path)
    original = storage_admission._inspect_lancedb_vector_state
    inspected_versions = []

    def fail_after_commit(*args, **kwargs):
        table = connection.open_table("memories")
        try:
            inspected_versions.append(int(table.version))
            if int(table.version) > 1:
                raise OSError("post-commit inspection failed")
        finally:
            _safe_close_table(table)
        return original(*args, **kwargs)

    monkeypatch.setattr(
        storage_admission, "_inspect_lancedb_vector_state", fail_after_commit
    )
    outcome = _admit(connection)
    assert outcome.state is StorageAdmissionState.ADMITTED
    assert inspected_versions == [1]
    assert _snapshot(connection)[0] == 2


@pytest.mark.parametrize("metadata", ['{"user_id":1e309}', '{"user_id":Infinity}'])
def test_nonfinite_legacy_user_id_requires_repair(tmp_path, metadata):
    connection = lancedb.connect(tmp_path)
    table = connection.create_table(
        "memories",
        pa.table({"id": ["bad"], "text": ["text"], "metadata": [metadata]}),
    )
    _safe_close_table(table)
    before = _snapshot(connection)
    outcome = _admit(connection)
    assert outcome.state is StorageAdmissionState.BLOCKED_REPAIR
    assert outcome.detail == REPAIR_REQUIRED_DETAIL
    assert _snapshot(connection) == before


def test_distant_duplicate_is_rejected_without_mutation(tmp_path):
    connection = lancedb.connect(tmp_path)
    table = connection.create_table(
        "memories",
        pa.table(
            {
                "id": ["duplicate", "middle", "duplicate"],
                "text": ["a", "b", "c"],
                "metadata": ["{}", "{}", "{}"],
            }
        ),
    )
    _safe_close_table(table)
    before = _snapshot(connection)
    outcome = _admit(connection)
    assert outcome.state is StorageAdmissionState.BLOCKED_REPAIR
    assert _snapshot(connection) == before


def test_post_commit_cleanup_failure_keeps_success(tmp_path, monkeypatch):
    connection = _connection(tmp_path)
    original = vector_compatibility.os.unlink

    def fail_cleanup(path):
        if ".memory-" in Path(path).name:
            raise PermissionError("private path must not escape")
        original(path)

    monkeypatch.setattr(vector_compatibility.os, "unlink", fail_cleanup)
    outcome = _admit(connection)
    assert outcome.state is StorageAdmissionState.ADMITTED
    assert outcome.detail is None
    assert _snapshot(connection)[0] == 2


def test_admission_lock_precedes_maintenance_and_has_no_production_caller(
    tmp_path, monkeypatch
):
    connection = _connection(tmp_path)
    admission_path = lancedb_lock_path(connection, "memories", "admission")
    original = vector_compatibility.prepare_lancedb_memory_table

    def guarded(*args, **kwargs):
        with pytest.raises(Timeout):
            FileLock(admission_path, timeout=0).acquire()
        return original(*args, **kwargs)

    monkeypatch.setattr(
        "xagent.core.memory.storage_admission.prepare_lancedb_memory_table", guarded
    )
    assert _admit(connection).state is StorageAdmissionState.ADMITTED

    source_root = Path(__file__).parents[3] / "src" / "xagent"
    callers = [
        path
        for path in source_root.rglob("*.py")
        if path.name != "__init__.py"
        and "admit_lancedb_memory_storage(" in path.read_text(encoding="utf-8")
    ]
    assert callers == [source_root / "core" / "memory" / "storage_admission.py"]
