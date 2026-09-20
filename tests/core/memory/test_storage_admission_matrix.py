import json
import math
import multiprocessing
import time
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
    maintain_lancedb_memory_table,
)
from xagent.core.memory.scope_columns import SCOPE_DIMS_COLUMN, USER_ID_COLUMN
from xagent.core.memory.storage_admission import (
    ADMISSION_FAILED_DETAIL,
    REPAIR_REQUIRED_DETAIL,
    DormantLanceDBMemoryHandle,
    MemoryStorageMode,
    StorageAdmissionState,
    admit_lancedb_memory_storage,
)
from xagent.core.memory.vector_compatibility import (
    FULL_ADMISSION_METADATA_KEY,
    FULL_ADMISSION_TABLE_VERSION_KEY,
    FULL_ADMISSION_VERSION,
    VECTOR_IDENTITY_METADATA_KEY,
    EmbeddingIdentity,
    VectorCompatibility,
    create_or_recreate_vector_capable_table,
    prepare_lancedb_memory_table,
)
from xagent.core.tools.core.RAG_tools.LanceDB.schema_manager import _safe_close_table

IDENTITY = EmbeddingIdentity(
    "openai", "text-embedding-3-small", "https://api.openai.com/v1/embeddings", 4, None
)
# Same dimension as IDENTITY, so only the stored identity distinguishes the two
# vector spaces; a dimension check alone would not catch a mix-up.
FOREIGN_IDENTITY = EmbeddingIdentity(
    "openai", "text-embedding-3-large", "https://api.openai.com/v1/embeddings", 4, None
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
    assert field.metadata[FULL_ADMISSION_METADATA_KEY] == FULL_ADMISSION_VERSION
    version_key = field.metadata[FULL_ADMISSION_TABLE_VERSION_KEY]
    assert version_key == str(table.version).encode()
    assert table.schema.field("vector").type == pa.list_(pa.float32(), 4)
    assert table.to_arrow()["vector"].to_pylist() == [None] * 5
    assert null_types == [pa.list_(pa.float32(), 4)] * 3
    assert scanned == [2, 2, 1]
    scanned.clear()
    assert _admit(connection).state is StorageAdmissionState.ADMITTED
    assert scanned == []
    # Any later commit moves the version the marker is bound to, so the next
    # admission revalidates the table instead of trusting the stale marker.
    table.delete("id = 'note-0'")
    _safe_close_table(table)
    assert _admit(connection).state is StorageAdmissionState.ADMITTED
    assert scanned == [2, 2]


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
        outcome = _admit(connection, lock_timeout=0.05)
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


def test_compatibility_is_classified_from_the_committed_table(tmp_path, monkeypatch):
    """Classification describes the commit, and is derived while it is held."""
    connection = _connection(tmp_path)
    original = vector_compatibility.classify_vector_compatibility
    classified = []

    def record(schema, identity):
        classified.append(schema)
        return original(schema, identity)

    monkeypatch.setattr(vector_compatibility, "classify_vector_compatibility", record)
    outcome = _admit(connection)
    assert outcome.state is StorageAdmissionState.ADMITTED
    assert outcome.admitted.vector_compatibility is VectorCompatibility.MATCHING
    assert outcome.maintenance.vector_compatibility is VectorCompatibility.MATCHING
    version, committed, _rows = _snapshot(connection)
    assert version == 2
    # Classified exactly once, from the schema maintenance committed -- never
    # from the pre-maintenance state, which carried no vectors at all, and
    # never from a re-read a concurrent writer could have invalidated.
    assert len(classified) == 1
    assert classified[0].field("vector").type == pa.list_(pa.float32(), 4)
    assert (
        classified[0].field(USER_ID_COLUMN).metadata[FULL_ADMISSION_TABLE_VERSION_KEY]
        == str(version).encode()
    )
    identity_key = classified[0].metadata[VECTOR_IDENTITY_METADATA_KEY]
    assert identity_key == committed.metadata[VECTOR_IDENTITY_METADATA_KEY]


class _FailAfterCommitConnection:
    """Fail every table inspection attempted after the overwrite commits."""

    def __init__(self, inner):
        self._inner = inner
        self.committed = False

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def open_table(self, name):
        if self.committed:
            raise OSError("post-commit inspection at /private/backend")
        return self._inner.open_table(name)

    def create_table(self, *args, **kwargs):
        table = self._inner.create_table(*args, **kwargs)
        self.committed = True
        return table


def test_admission_never_inspects_the_table_after_the_commit(tmp_path):
    connection = _connection(tmp_path)
    guarded = _FailAfterCommitConnection(connection)

    outcome = _admit(guarded)

    # A backend failure after the durable commit used to be reported as an
    # ordinary retryable outcome, leaving the caller unable to tell "nothing
    # was mutated" from "committed but unverified". There is no such read left.
    assert outcome.state is StorageAdmissionState.ADMITTED
    assert outcome.admitted.vector_compatibility is VectorCompatibility.MATCHING
    assert outcome.admitted.capabilities.mode is MemoryStorageMode.VECTOR
    version, schema, _rows = _snapshot(connection)
    assert version == 2
    assert schema.field("vector").type == pa.list_(pa.float32(), 4)
    # The injected failure really is live: any post-commit read would have hit it.
    assert guarded.committed
    with pytest.raises(OSError, match="post-commit inspection"):
        guarded.open_table("memories")


def test_empty_vectorless_table_is_admitted_with_typed_null_vectors(tmp_path):
    connection = lancedb.connect(tmp_path)
    table = connection.create_table(
        "memories",
        schema=pa.schema(
            [("id", pa.string()), ("text", pa.string()), ("metadata", pa.string())]
        ),
    )
    assert table.count_rows() == 0
    _safe_close_table(table)

    # With nothing staged to scan, the supplied schema is the only description
    # of the table the overwrite must produce.
    outcome = _admit(connection)

    assert outcome.state is StorageAdmissionState.ADMITTED
    assert outcome.admitted.capabilities.mode is MemoryStorageMode.VECTOR
    assert outcome.admitted.vector_compatibility is VectorCompatibility.MATCHING
    version, schema, rows = _snapshot(connection)
    assert rows == []
    assert version == 2
    assert schema.field("vector").type == pa.list_(pa.float32(), 4)
    assert schema.field("vector").nullable
    assert schema.field(SCOPE_DIMS_COLUMN).type == pa.list_(pa.string())
    metadata = schema.field(USER_ID_COLUMN).metadata
    assert metadata[MAINTENANCE_METADATA_KEY] == MAINTENANCE_VERSION
    assert metadata[MAINTENANCE_TABLE_VERSION_KEY] == str(version).encode()
    assert metadata[FULL_ADMISSION_METADATA_KEY] == FULL_ADMISSION_VERSION
    assert metadata[FULL_ADMISSION_TABLE_VERSION_KEY] == str(version).encode()
    # The markers are version-bound, so the committed table is admitted again
    # through the fast path without a second rewrite.
    assert _admit(connection).state is StorageAdmissionState.ADMITTED
    assert _snapshot(connection)[0] == version


# A regression here would wait forever rather than fail, so bound the test.
@pytest.mark.timeout(30)
@pytest.mark.parametrize("lock_timeout", [0, -1.0, math.inf, math.nan, True, "1"])
def test_invalid_lock_timeout_is_rejected_before_any_lock(tmp_path, lock_timeout):
    connection = _connection(tmp_path)
    before = _snapshot(connection)
    admission_path = lancedb_lock_path(connection, "memories", "admission")
    maintenance_path = lancedb_lock_path(connection, "memories", "maintenance")
    # Both locks are held, so a timeout that reaches FileLock either waits
    # forever or gives up; neither may happen for a value the contract rejects.
    with FileLock(admission_path), FileLock(maintenance_path):
        with pytest.raises(ValueError, match="finite positive"):
            _admit(connection, lock_timeout=lock_timeout)
        with pytest.raises(ValueError, match="finite positive"):
            prepare_lancedb_memory_table(
                connection,
                "memories",
                IDENTITY,
                batch_size=2,
                lock_timeout=lock_timeout,
            )
    assert _snapshot(connection) == before


@pytest.mark.timeout(30)
@pytest.mark.parametrize("scope", ["admission", "maintenance"])
def test_contended_lock_returns_a_typed_retryable_outcome(tmp_path, scope):
    connection = _connection(tmp_path)
    before = _snapshot(connection)
    started = time.monotonic()
    with FileLock(lancedb_lock_path(connection, "memories", scope)):
        outcome = _admit(connection, lock_timeout=0.05)
    assert time.monotonic() - started < 10
    assert outcome.state is StorageAdmissionState.RETRYABLE_UNAVAILABLE
    assert outcome.detail == ADMISSION_FAILED_DETAIL
    assert _snapshot(connection) == before


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

    # The legacy recreation entry point holds the same lock across the same span,
    # so it cannot publish a vector space another admission is already reading.
    monkeypatch.setattr(vector_compatibility, "prepare_lancedb_memory_table", guarded)
    assert (
        create_or_recreate_vector_capable_table(connection, "memories", IDENTITY)
        is VectorCompatibility.MATCHING
    )

    source_root = Path(__file__).parents[3] / "src" / "xagent"
    callers = [
        path
        for path in source_root.rglob("*.py")
        if path.name != "__init__.py"
        and "admit_lancedb_memory_storage(" in path.read_text(encoding="utf-8")
    ]
    assert callers == [source_root / "core" / "memory" / "storage_admission.py"]


def _scope_maintained_connection(tmp_path, *, drop_text, vector):
    """Seed a table whose only completion marker is the scope-only one."""
    connection = lancedb.connect(tmp_path)
    row = {"id": "note-0", "text": "text-0", "metadata": json.dumps({"user_id": 3})}
    fields = [("id", pa.string()), ("text", pa.string()), ("metadata", pa.string())]
    if drop_text:
        del row["text"], fields[1]
    # LanceDB refuses to write NaNs into a column named "vector", so the legacy
    # values are staged under another name and renamed into place.
    row["badvec"] = vector
    fields.append(("badvec", pa.list_(pa.float32(), 4)))
    identity = json.dumps(IDENTITY.as_dict(), sort_keys=True, separators=(",", ":"))
    schema = pa.schema(
        fields, metadata={VECTOR_IDENTITY_METADATA_KEY: identity.encode()}
    )
    table = connection.create_table(
        "memories", pa.Table.from_pylist([row], schema=schema)
    )
    table.alter_columns({"path": "badvec", "rename": "vector"})
    _safe_close_table(table)

    assert (
        maintain_lancedb_memory_table(connection, "memories").status
        is MaintenanceStatus.COMPLETE
    )
    metadata = _snapshot(connection)[1].field(USER_ID_COLUMN).metadata
    assert metadata[MAINTENANCE_METADATA_KEY] == MAINTENANCE_VERSION
    assert FULL_ADMISSION_METADATA_KEY not in metadata
    return connection


@pytest.mark.parametrize("case", ["missing_text", "nan_vector"])
def test_scope_only_marker_never_certifies_admission(tmp_path, case):
    nan_case = case == "nan_vector"
    connection = _scope_maintained_connection(
        tmp_path,
        drop_text=not nan_case,
        vector=[math.nan, 0.0, 0.0, 0.0] if nan_case else [1.0, 2.0, 3.0, 4.0],
    )
    before = _snapshot(connection)

    outcome = _admit(connection)
    after = _snapshot(connection)
    assert outcome.state is StorageAdmissionState.BLOCKED_REPAIR
    assert outcome.detail == REPAIR_REQUIRED_DETAIL
    assert after[:2] == before[:2]
    if nan_case:
        # The rejected table keeps its own bytes; admission never "repairs" it.
        assert math.isnan(after[2][0]["vector"][0])
    else:
        assert after[2] == before[2]


def _admit_foreign_identity_after_snapshot(path, snapshotted, committed, results):
    original = storage_admission.prepare_lancedb_memory_table

    def wait_for_foreign_commit(*args, **kwargs):
        # Park where a pre-maintenance snapshot would have been taken, so the
        # other vector space lands before this admission classifies anything.
        snapshotted.set()
        assert committed.wait(30)
        return original(*args, **kwargs)

    storage_admission.prepare_lancedb_memory_table = wait_for_foreign_commit
    outcome = admit_lancedb_memory_storage(
        DormantLanceDBMemoryHandle(lancedb.connect(path), "memories"),
        FOREIGN_IDENTITY,
        writers_quiesced=True,
        batch_size=2,
    )
    admitted = outcome.admitted
    results.put(
        (
            outcome.state.value,
            None if admitted is None else admitted.capabilities.mode.value,
            None if admitted is None else admitted.vector_compatibility.value,
        )
    )


def test_interleaved_recreation_never_certifies_a_foreign_vector_space(tmp_path):
    connection = _connection(tmp_path)
    context = multiprocessing.get_context("spawn")
    snapshotted, committed = context.Event(), context.Event()
    results = context.Queue()
    process = context.Process(
        target=_admit_foreign_identity_after_snapshot,
        args=(str(tmp_path), snapshotted, committed, results),
    )

    process.start()
    try:
        assert snapshotted.wait(30)
        # The local identity commits its own vectors mid-flight.
        assert (
            prepare_lancedb_memory_table(
                connection, "memories", IDENTITY, batch_size=2, lock_timeout=10
            ).status
            is MaintenanceStatus.COMPLETE
        )
        version, schema, _rows = _snapshot(connection)
        committed.set()
        process.join(60)
    finally:
        if process.is_alive():
            process.terminate()
            process.join(5)
    assert process.exitcode == 0

    # The other caller is told the truth about the space it would read, and it
    # neither certifies itself over those vectors nor rewrites them.
    assert results.get(timeout=5) == (
        StorageAdmissionState.ADMITTED.value,
        MemoryStorageMode.TEXT_ONLY.value,
        VectorCompatibility.MISMATCHING.value,
    )
    after_version, after_schema, _after_rows = _snapshot(connection)
    assert after_version == version
    assert (
        after_schema.metadata[VECTOR_IDENTITY_METADATA_KEY]
        == (schema.metadata[VECTOR_IDENTITY_METADATA_KEY])
    )
