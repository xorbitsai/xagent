import importlib.metadata
import json
from pathlib import Path

import lancedb  # type: ignore
import pyarrow as pa  # type: ignore
import pytest
from filelock import FileLock, Timeout

from xagent.core.memory import vector_compatibility
from xagent.core.memory.lancedb_maintenance import (
    MAINTENANCE_METADATA_KEY,
    MAINTENANCE_TABLE_VERSION_KEY,
    MAINTENANCE_VERSION,
    lancedb_lock_path,
)
from xagent.core.memory.storage_admission import (
    ADMISSION_FAILED_DETAIL,
    REPAIR_REQUIRED_DETAIL,
    DormantLanceDBMemoryHandle,
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


def test_supported_version_atomic_null_vectors_marker_and_bounded_scan(
    tmp_path, monkeypatch
):
    assert importlib.metadata.version("lancedb") in {"0.24.2", "0.29.2", "0.37.1"}
    connection = _connection(tmp_path)
    before_version = _snapshot(connection)[0]
    scanned = []
    original = vector_compatibility._checkpoint

    def observe(stage, batch=None):
        if stage == "scan_batch":
            scanned.append(batch)
        original(stage, batch)

    monkeypatch.setattr(vector_compatibility, "_checkpoint", observe)
    outcome = _admit(connection)
    table = connection.open_table("memories")
    field = table.schema.field("user_id")
    assert outcome.state is StorageAdmissionState.ADMITTED
    assert outcome.admitted.capabilities.vector_search is True
    assert table.version == before_version + 1
    assert field.metadata[MAINTENANCE_METADATA_KEY] == MAINTENANCE_VERSION
    assert field.metadata[MAINTENANCE_TABLE_VERSION_KEY] == str(table.version).encode()
    assert table.schema.field("vector").type == pa.list_(pa.float32(), 4)
    assert table.to_arrow()["vector"].to_pylist() == [None] * 5
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

    connection = _connection(tmp_path / "failure")
    before = _snapshot(connection)

    def fail(stage, batch=None):
        if stage == "commit_batch" and batch == 1:
            raise RuntimeError("secret backend path /do/not/expose")

    monkeypatch.setattr(vector_compatibility, "_checkpoint", fail)
    outcome = _admit(connection)
    assert outcome.state is StorageAdmissionState.BLOCKED_REPAIR
    assert outcome.detail == ADMISSION_FAILED_DETAIL
    assert "secret" not in outcome.detail
    assert _snapshot(connection) == before


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
        and "admit_lancedb_memory_storage(" in path.read_text()
    ]
    assert callers == [source_root / "core" / "memory" / "storage_admission.py"]
