"""Real-LanceDB coverage for resumable scope-column maintenance."""

import json
import multiprocessing
import socket
import time
from datetime import timedelta

import lancedb  # type: ignore
import pyarrow as pa  # type: ignore
import pytest
from filelock import FileLock

from xagent.core.memory import lancedb_maintenance as maintenance
from xagent.core.memory.lancedb_maintenance import (
    MAINTENANCE_METADATA_KEY,
    MAINTENANCE_TABLE_VERSION_KEY,
    MAINTENANCE_VERSION,
    MaintenanceLockTimeout,
    MaintenanceStatus,
    maintain_lancedb_memory_table,
)
from xagent.core.memory.scope_columns import SCOPE_DIMS_COLUMN, USER_ID_COLUMN
from xagent.core.memory.vector_compatibility import VECTOR_IDENTITY_METADATA_KEY
from xagent.core.tools.core.RAG_tools.LanceDB.schema_manager import _safe_close_table


def _metadata(index):
    return json.dumps(
        {
            "user_id": index,
            "execution_scope_agent": f"agent-{index % 2}",
        }
    )


def _connection(
    tmp_path,
    count=2,
    *,
    ids=None,
    metadata=None,
    scoped=False,
    connection_kwargs=None,
):
    connection = lancedb.connect(tmp_path, **(connection_kwargs or {}))
    ids = ids if ids is not None else [f"note-{index}" for index in range(count)]
    metadata = (
        metadata if metadata is not None else [_metadata(i) for i in range(count)]
    )
    columns = {
        "id": ids,
        "text": [f"text-{i}" for i in range(count)],
        "metadata": pa.array(metadata, pa.string()),
        "vector": pa.array(
            [[float(i), 1.0] for i in range(count)], pa.list_(pa.float32(), 2)
        ),
    }
    if scoped:
        columns[USER_ID_COLUMN] = pa.array([None] * count, pa.int64())
        columns[SCOPE_DIMS_COLUMN] = pa.array([None] * count, pa.list_(pa.string()))
    data = pa.table(columns).replace_schema_metadata(
        {VECTOR_IDENTITY_METADATA_KEY: b"preserved"}
    )
    table = connection.create_table("memories", data)
    _safe_close_table(table)
    return connection


def _table(connection):
    return connection.open_table("memories")


def _marker(table):
    if USER_ID_COLUMN not in table.schema.names:
        return None
    return (table.schema.field(USER_ID_COLUMN).metadata or {}).get(
        MAINTENANCE_METADATA_KEY
    )


def _marker_table_version(table):
    if USER_ID_COLUMN not in table.schema.names:
        return None
    return (table.schema.field(USER_ID_COLUMN).metadata or {}).get(
        MAINTENANCE_TABLE_VERSION_KEY
    )


def _hold_file_lock(lock_path, ready, release):
    with FileLock(lock_path):
        ready.set()
        release.wait(30)


def test_complete_migration_and_second_run_is_schema_only(tmp_path, monkeypatch):
    connection = _connection(tmp_path)
    first = maintain_lancedb_memory_table(connection, "memories")
    table = _table(connection)
    assert first.status is MaintenanceStatus.COMPLETE
    assert table.to_arrow().select([USER_ID_COLUMN, SCOPE_DIMS_COLUMN]).to_pylist() == [
        {USER_ID_COLUMN: 0, SCOPE_DIMS_COLUMN: ["agent=agent-0"]},
        {USER_ID_COLUMN: 1, SCOPE_DIMS_COLUMN: ["agent=agent-1"]},
    ]
    assert _marker(table) == MAINTENANCE_VERSION
    assert _marker_table_version(table) == str(table.version).encode()
    assert table.schema.metadata[VECTOR_IDENTITY_METADATA_KEY] == b"preserved"
    _safe_close_table(table)

    monkeypatch.setattr(
        maintenance, "_read_rows", lambda _table: pytest.fail("fast path scanned")
    )
    second = maintain_lancedb_memory_table(connection, "memories")
    assert second == maintenance.MaintenanceOutcome(MaintenanceStatus.COMPLETE)


@pytest.mark.parametrize(("count", "batches"), [(512, 1), (513, 2)])
def test_512_boundary_uses_one_commit_per_batch(tmp_path, count, batches):
    connection = _connection(tmp_path, count=count)
    table = _table(connection)
    before = table.version
    _safe_close_table(table)
    outcome = maintain_lancedb_memory_table(connection, "memories")
    table = _table(connection)
    assert outcome.batches_committed == batches
    assert outcome.updated_rows == count
    assert table.version - before == batches + 2  # add columns + batches + marker
    _safe_close_table(table)


def test_sql_null_metadata_projects_to_empty_scope_without_network(
    tmp_path, monkeypatch
):
    connection = _connection(tmp_path, count=1, metadata=[None])

    def forbidden(*_args, **_kwargs):
        raise AssertionError("maintenance must not embed or use the network")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    outcome = maintain_lancedb_memory_table(connection, "memories")
    row = _table(connection).to_arrow().to_pylist()[0]
    assert outcome.status is MaintenanceStatus.COMPLETE
    assert row[USER_ID_COLUMN] is None
    assert row[SCOPE_DIMS_COLUMN] == []


@pytest.mark.parametrize(
    "ids",
    [[None, "ok"], [1, 2], ["same", "same"], ["", "ok"]],
)
def test_invalid_ids_leave_table_unchanged(tmp_path, ids):
    connection = _connection(tmp_path, ids=ids)
    table = _table(connection)
    before = (table.version, table.schema, table.to_arrow().to_pylist())
    _safe_close_table(table)
    result = maintain_lancedb_memory_table(connection, "memories")
    table = _table(connection)
    assert result.status is MaintenanceStatus.INVALID_LEGACY_DATA
    assert (table.version, table.schema, table.to_arrow().to_pylist()) == before
    _safe_close_table(table)


@pytest.mark.parametrize(
    ("metadata", "detail"),
    [
        (pa.array([7], pa.int64()), "must be a string or SQL NULL"),
        (pa.array([json.dumps({"user_id": 2**63})], pa.string()), "signed int64"),
        (
            pa.array([json.dumps({"user_id": -(2**63) - 1})], pa.string()),
            "signed int64",
        ),
    ],
)
def test_invalid_metadata_or_user_id_range_leaves_table_unchanged(
    tmp_path, metadata, detail
):
    connection = lancedb.connect(tmp_path)
    table = connection.create_table(
        "memories",
        pa.table({"id": ["note-0"], "text": ["text-0"], "metadata": metadata}),
    )
    before = (table.version, table.schema, table.to_arrow().to_pylist())
    _safe_close_table(table)

    result = maintain_lancedb_memory_table(connection, "memories")
    table = _table(connection)
    assert result.status is MaintenanceStatus.INVALID_LEGACY_DATA
    assert detail in result.detail
    assert (table.version, table.schema, table.to_arrow().to_pylist()) == before
    _safe_close_table(table)


@pytest.mark.parametrize("stage", ["columns_added", "batch_committed"])
def test_partial_failure_resumes_without_repeating_completed_rows(
    tmp_path, monkeypatch, stage
):
    connection = _connection(tmp_path, count=513)
    original = maintenance._checkpoint

    def fail(selected, batch=None):
        if selected == stage and (batch is None or batch == 1):
            raise RuntimeError("injected failure")
        original(selected, batch)

    monkeypatch.setattr(maintenance, "_checkpoint", fail)
    with pytest.raises(RuntimeError, match="injected failure"):
        maintain_lancedb_memory_table(connection, "memories")
    table = _table(connection)
    assert {USER_ID_COLUMN, SCOPE_DIMS_COLUMN} <= set(table.schema.names)
    assert _marker(table) is None
    _safe_close_table(table)

    monkeypatch.setattr(maintenance, "_checkpoint", original)
    resumed = maintain_lancedb_memory_table(connection, "memories")
    assert resumed.status is MaintenanceStatus.COMPLETE
    assert resumed.updated_rows == (513 if stage == "columns_added" else 1)


@pytest.mark.parametrize("change_metadata", [False, True])
def test_concurrent_changes_are_preserved_and_metadata_change_is_reported(
    tmp_path, monkeypatch, change_metadata
):
    connection = _connection(tmp_path, count=1, scoped=True)
    original = maintenance._backfill_batch
    changed_metadata = json.dumps({"user_id": 99})

    def concurrent_update(table, rows):
        writer = _table(connection)
        values = {"text": "writer-text", "vector": [9.0, 9.0]}
        if change_metadata:
            values["metadata"] = changed_metadata
        writer.update("id = 'note-0'", values=values)
        _safe_close_table(writer)
        return original(table, rows)

    monkeypatch.setattr(maintenance, "_backfill_batch", concurrent_update)
    result = maintain_lancedb_memory_table(connection, "memories")
    row = _table(connection).to_arrow().to_pylist()[0]
    assert (row["text"], row["vector"]) == ("writer-text", [9.0, 9.0])
    if change_metadata:
        assert result.status is MaintenanceStatus.INCOMPLETE
        assert result.cas_skipped_rows
        assert row["metadata"] == changed_metadata
        assert _marker(_table(connection)) is None
    else:
        assert result.status is MaintenanceStatus.COMPLETE
        assert row[USER_ID_COLUMN] == 0


def test_late_metadata_writer_invalidates_marker_and_resumes(tmp_path, monkeypatch):
    connection = _connection(tmp_path, count=1, scoped=True)
    changed_metadata = json.dumps({"user_id": 99})

    def late_update(stage, _batch=None):
        if stage == "before_completion":
            writer = _table(connection)
            writer.update("id = 'note-0'", values={"metadata": changed_metadata})
            _safe_close_table(writer)

    monkeypatch.setattr(maintenance, "_checkpoint", late_update)
    first = maintain_lancedb_memory_table(connection, "memories")
    table = _table(connection)
    row = table.to_arrow().to_pylist()[0]
    assert first.status is MaintenanceStatus.INCOMPLETE
    assert row["metadata"] == changed_metadata
    assert row[USER_ID_COLUMN] == 0
    assert _marker_table_version(table) != str(table.version).encode()
    _safe_close_table(table)

    monkeypatch.setattr(maintenance, "_checkpoint", lambda *_args: None)
    resumed = maintain_lancedb_memory_table(connection, "memories")
    table = _table(connection)
    assert resumed.status is MaintenanceStatus.COMPLETE
    assert table.to_arrow().to_pylist()[0][USER_ID_COLUMN] == 99
    assert _marker_table_version(table) == str(table.version).encode()
    _safe_close_table(table)


def test_table_version_change_invalidates_fast_path(tmp_path, monkeypatch):
    connection = _connection(tmp_path, count=1)
    assert (
        maintain_lancedb_memory_table(connection, "memories").status
        is MaintenanceStatus.COMPLETE
    )
    writer = _table(connection)
    writer.update("id = 'note-0'", values={"text": "new text"})
    _safe_close_table(writer)

    scanned = False
    original = maintenance._read_rows

    def record_scan(table):
        nonlocal scanned
        scanned = True
        return original(table)

    monkeypatch.setattr(maintenance, "_read_rows", record_scan)
    result = maintain_lancedb_memory_table(connection, "memories")
    assert result.status is MaintenanceStatus.COMPLETE
    assert scanned


def test_strong_consistency_detects_version_change_during_final_read(
    tmp_path, monkeypatch
):
    connection = _connection(
        tmp_path,
        count=1,
        scoped=True,
        connection_kwargs={"read_consistency_interval": timedelta(0)},
    )
    original = maintenance._read_rows
    calls = 0

    def update_after_final_read(table):
        nonlocal calls
        rows = original(table)
        calls += 1
        if calls == 2:
            writer = _table(connection)
            writer.update("id = 'note-0'", values={"text": "concurrent text"})
            _safe_close_table(writer)
        return rows

    monkeypatch.setattr(maintenance, "_read_rows", update_after_final_read)
    result = maintain_lancedb_memory_table(connection, "memories")
    table = _table(connection)
    assert result.status is MaintenanceStatus.INCOMPLETE
    assert table.to_arrow().to_pylist()[0]["text"] == "concurrent text"
    assert _marker(table) is None
    _safe_close_table(table)


def test_final_invalid_id_returns_invalid_and_preserves_concurrent_row(
    tmp_path, monkeypatch
):
    connection = _connection(
        tmp_path,
        count=1,
        connection_kwargs={"read_consistency_interval": timedelta(0)},
    )

    def append_invalid_id(stage, _batch=None):
        if stage == "batch_committed":
            writer = _table(connection)
            writer.add(
                [
                    {
                        "id": None,
                        "text": "concurrent text",
                        "metadata": "{}",
                        "vector": [3.0, 4.0],
                        USER_ID_COLUMN: None,
                        SCOPE_DIMS_COLUMN: [],
                    }
                ]
            )
            _safe_close_table(writer)

    monkeypatch.setattr(maintenance, "_checkpoint", append_invalid_id)
    result = maintain_lancedb_memory_table(connection, "memories")
    table = _table(connection)
    assert result.status is MaintenanceStatus.INVALID_LEGACY_DATA
    assert any(row["id"] is None for row in table.to_arrow().to_pylist())
    assert _marker(table) is None
    _safe_close_table(table)


def test_exact_scope_schema_preserves_existing_field_metadata(tmp_path):
    connection = _connection(tmp_path, count=1, scoped=True)
    table = _table(connection)
    metadata = {"preserved": "yes"}
    if hasattr(table, "update_field_metadata"):
        table.update_field_metadata({"path": USER_ID_COLUMN, "metadata": metadata})
    else:
        table.replace_field_metadata(USER_ID_COLUMN, metadata)
    _safe_close_table(table)

    result = maintain_lancedb_memory_table(connection, "memories")
    table = _table(connection)
    assert result.status is MaintenanceStatus.COMPLETE
    assert table.schema.field(USER_ID_COLUMN).metadata[b"preserved"] == b"yes"
    assert _marker_table_version(table) == str(table.version).encode()
    _safe_close_table(table)


@pytest.mark.parametrize(
    ("partial", "old_marker"),
    [(False, False), (True, False), (False, True)],
)
def test_incompatible_scope_schema_is_reported_without_mutation(
    tmp_path, partial, old_marker
):
    connection = lancedb.connect(tmp_path)
    columns = {
        "id": ["note-0"],
        "text": ["text-0"],
        "metadata": pa.array([json.dumps({"user_id": 0})], pa.string()),
        USER_ID_COLUMN: pa.array([0], pa.int32()),
    }
    if not partial:
        columns[SCOPE_DIMS_COLUMN] = pa.array([[]], pa.list_(pa.int32()))
    table = connection.create_table("memories", pa.table(columns))
    if old_marker:
        marker = {MAINTENANCE_METADATA_KEY.decode(): MAINTENANCE_VERSION.decode()}
        if hasattr(table, "update_field_metadata"):
            table.update_field_metadata({"path": USER_ID_COLUMN, "metadata": marker})
        else:
            table.replace_field_metadata(USER_ID_COLUMN, marker)
    before = (table.version, table.schema, table.to_arrow().to_pylist())
    _safe_close_table(table)

    result = maintain_lancedb_memory_table(connection, "memories")
    table = _table(connection)
    assert result.status is MaintenanceStatus.INCOMPATIBLE_SCHEMA
    assert USER_ID_COLUMN in result.detail
    assert (table.version, table.schema, table.to_arrow().to_pylist()) == before
    _safe_close_table(table)


def test_completion_marker_is_last_and_failure_before_it_resumes(tmp_path, monkeypatch):
    connection = _connection(tmp_path)

    def fail(stage, _batch=None):
        if stage == "before_completion":
            raise RuntimeError("before marker")

    monkeypatch.setattr(maintenance, "_checkpoint", fail)
    with pytest.raises(RuntimeError, match="before marker"):
        maintain_lancedb_memory_table(connection, "memories")
    table = _table(connection)
    assert _marker(table) is None
    version = table.version
    _safe_close_table(table)
    monkeypatch.setattr(maintenance, "_checkpoint", lambda *_args: None)
    assert (
        maintain_lancedb_memory_table(connection, "memories").status
        is MaintenanceStatus.COMPLETE
    )
    table = _table(connection)
    assert table.version == version + 1
    assert _marker(table) == MAINTENANCE_VERSION
    _safe_close_table(table)


def test_lock_timeout_is_bounded_and_success_releases_lock(tmp_path):
    connection = _connection(tmp_path)
    lock_path = maintenance._lock_path(connection, "memories")
    with FileLock(lock_path):
        started = time.monotonic()
        with pytest.raises(MaintenanceLockTimeout, match="stop the other"):
            maintain_lancedb_memory_table(connection, "memories", lock_timeout=0.05)
        assert time.monotonic() - started < 1

    maintain_lancedb_memory_table(connection, "memories", lock_timeout=0.05)
    with FileLock(lock_path, timeout=0.05):
        pass


def test_lock_timeout_against_second_process(tmp_path):
    connection = _connection(tmp_path)
    lock_path = maintenance._lock_path(connection, "memories")
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    release = context.Event()
    process = context.Process(target=_hold_file_lock, args=(lock_path, ready, release))
    process.start()
    try:
        assert ready.wait(20)
        with pytest.raises(MaintenanceLockTimeout, match="stop the other"):
            maintain_lancedb_memory_table(connection, "memories", lock_timeout=0.05)
    finally:
        release.set()
        process.join(10)
    assert process.exitcode == 0


def test_non_local_uri_is_rejected():
    class RemoteConnection:
        uri = "s3://bucket/database"

    with pytest.raises(ValueError, match="writable local database URI"):
        maintain_lancedb_memory_table(RemoteConnection(), "memories")


@pytest.mark.parametrize("timeout", [0, float("inf")])
def test_nonpositive_lock_timeout_is_rejected(tmp_path, timeout):
    connection = _connection(tmp_path)
    with pytest.raises(ValueError, match="finite positive"):
        maintain_lancedb_memory_table(connection, "memories", lock_timeout=timeout)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"batch_size": True},
        {"batch_size": 1.5},
        {"batch_size": "1"},
        {"lock_timeout": True},
        {"lock_timeout": "1"},
        {"lock_timeout": float("nan")},
    ],
)
def test_invalid_parameter_types_do_not_access_connection(kwargs):
    class ForbiddenConnection:
        @property
        def uri(self):
            raise AssertionError(
                "invalid parameters must fail before connection access"
            )

    with pytest.raises(ValueError):
        maintain_lancedb_memory_table(ForbiddenConnection(), "memories", **kwargs)
