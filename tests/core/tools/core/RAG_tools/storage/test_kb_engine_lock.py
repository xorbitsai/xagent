"""The deployment's KB engine is recorded at first start and locks the setting."""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import lancedb
import pytest
from filelock import FileLock, Timeout

from xagent.core.tools.core.RAG_tools.core.exceptions import ConfigurationError
from xagent.core.tools.core.RAG_tools.core.schemas import CollectionInfo
from xagent.core.tools.core.RAG_tools.kb import KBContextRequest, get_kb_coordinator
from xagent.core.tools.core.RAG_tools.storage import vector_backend
from xagent.core.tools.core.RAG_tools.storage.factory import get_metadata_store
from xagent.core.tools.core.RAG_tools.storage.vector_backend import (
    KB_ENGINE_RECORD,
    KBStorageBackend,
    lock_deployment_kb_engine,
)


@pytest.fixture(autouse=True)
def lancedb_setting(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (tmp_path / "lancedb").mkdir(exist_ok=True)
    monkeypatch.setenv("LANCEDB_DIR", str(tmp_path / "lancedb"))
    monkeypatch.setenv("XAGENT_VECTOR_BACKEND", "lancedb")
    monkeypatch.delenv("VECTOR_STORE_BACKEND", raising=False)


@pytest.fixture
def milvus_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XAGENT_VECTOR_BACKEND", "milvus")


def _record() -> Path:
    return Path(os.environ["LANCEDB_DIR"]) / KB_ENGINE_RECORD


def _seed(table: str) -> lancedb.table.Table:
    return lancedb.connect(os.environ["LANCEDB_DIR"]).create_table(table, [{"id": "x"}])


def test_fresh_deployment_records_the_setting() -> None:
    assert lock_deployment_kb_engine() is KBStorageBackend.LANCEDB
    assert _record().read_text() == "lancedb\n"
    assert _record().stat().st_mode & 0o777 == 0o644
    assert lock_deployment_kb_engine() is KBStorageBackend.LANCEDB


def test_upgraded_lancedb_deployment_keeps_starting() -> None:
    _seed("documents")
    _seed("embeddings_model_a")

    assert lock_deployment_kb_engine() is KBStorageBackend.LANCEDB
    assert _record().read_text() == "lancedb\n"


@pytest.mark.parametrize(
    "table",
    ["documents", "collection_config", "collection_metadata", "embeddings_model_a"],
)
def test_kb_rows_without_a_record_mean_lancedb(
    milvus_setting: None, table: str
) -> None:
    _seed(table)

    with pytest.raises(ConfigurationError) as raised:
        lock_deployment_kb_engine()

    message = str(raised.value)
    assert f"KB engine is lancedb ({table} hold data)" in message
    assert "XAGENT_VECTOR_BACKEND is milvus" in message
    assert "delete the record file and restart" in message
    assert _record().read_text() == "lancedb\n"


def test_deleted_rows_do_not_count_as_data(milvus_setting: None) -> None:
    _seed("documents").delete("id = 'x'")

    assert lock_deployment_kb_engine() is KBStorageBackend.MILVUS
    assert _record().read_text() == "milvus\n"


def test_an_empty_kb_ids_table_is_not_milvus() -> None:
    _seed("kb_ids").delete("id = 'x'")

    assert lock_deployment_kb_engine() is KBStorageBackend.LANCEDB


def test_an_unreadable_table_counts_as_data(milvus_setting: None) -> None:
    broken = Path(os.environ["LANCEDB_DIR"]) / "embeddings_broken.lance"
    broken.mkdir()
    (broken / "junk").write_text("x")

    with pytest.raises(ConfigurationError, match=r"lancedb \(embeddings_broken hold"):
        lock_deployment_kb_engine()


def test_kb_ids_rows_mean_a_milvus_deployment() -> None:
    _seed("kb_ids")
    _seed("documents")

    with pytest.raises(
        ConfigurationError,
        match=r"is milvus \(kb_ids hold data\).*XAGENT_VECTOR_BACKEND is lancedb",
    ):
        lock_deployment_kb_engine()
    assert _record().read_text() == "milvus\n"


def test_a_setting_that_differs_from_the_record_is_refused() -> None:
    _record().write_text("milvus\n")

    with pytest.raises(ConfigurationError) as raised:
        lock_deployment_kb_engine()

    message = str(raised.value)
    assert "KB engine is milvus, recorded in" in message
    assert str(_record()) in message
    assert "XAGENT_VECTOR_BACKEND is lancedb" in message
    assert "delete the record file and restart" in message
    assert _record().read_text() == "milvus\n"


def test_a_reserved_engine_is_refused_before_recording(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("XAGENT_VECTOR_BACKEND", "qdrant")

    with pytest.raises(ConfigurationError, match="not implemented"):
        lock_deployment_kb_engine()
    assert not _record().exists()


def test_a_fresh_deployment_starts_with_milvus_and_records_it(
    milvus_setting: None,
) -> None:
    assert lock_deployment_kb_engine() is KBStorageBackend.MILVUS
    assert _record().read_text() == "milvus\n"
    assert lock_deployment_kb_engine() is KBStorageBackend.MILVUS


def test_a_lancedb_record_refuses_the_milvus_setting(milvus_setting: None) -> None:
    _record().write_text("lancedb\n")

    with pytest.raises(ConfigurationError) as raised:
        lock_deployment_kb_engine()

    message = str(raised.value)
    assert "KB engine is lancedb, recorded in" in message
    assert "XAGENT_VECTOR_BACKEND is milvus" in message
    assert "delete the record file and restart" in message
    assert _record().read_text() == "lancedb\n"


@pytest.mark.parametrize("content", [b"", b"postgres\n", b"\xff\xfe"])
def test_a_corrupt_record_is_refused_and_kept(content: bytes) -> None:
    _record().write_bytes(content)

    with pytest.raises(ConfigurationError, match="delete the record file"):
        lock_deployment_kb_engine()
    assert _record().read_bytes() == content


@pytest.mark.parametrize("lock_file_exists", [False, True])
@pytest.mark.parametrize(
    ("record", "table", "fix"),
    [
        (None, "documents", None),
        ("lancedb\n", "documents", None),
        (
            None,
            "kb_ids",
            "writable (with flock support), or set XAGENT_VECTOR_BACKEND to milvus.",
        ),
        ("milvus\n", "documents", "delete the record file and restart."),
    ],
)
def test_a_read_only_directory_still_starts_and_still_refuses(
    record: str | None, table: str, fix: str | None, lock_file_exists: bool
) -> None:
    _seed(table)
    if record is not None:
        _record().write_text(record)
    if lock_file_exists:
        _record().with_name(f"{KB_ENGINE_RECORD}.lock").touch()
    _record().parent.chmod(0o555)
    try:
        if fix is not None:
            with pytest.raises(
                ConfigurationError, match="KB engine is milvus"
            ) as raised:
                lock_deployment_kb_engine()
            assert str(raised.value).endswith(fix)
        else:
            assert lock_deployment_kb_engine() is KBStorageBackend.LANCEDB
        assert _record().exists() == (record is not None)
    finally:
        _record().parent.chmod(0o755)


@pytest.mark.parametrize("lancedb_dir_set", [True, False])
def test_a_lancedb_directory_that_cannot_be_created_still_starts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, lancedb_dir_set: bool
) -> None:
    from xagent.providers.vector_store.lancedb import LanceDBConnectionManager

    root = tmp_path / "root"
    root.mkdir()
    monkeypatch.setenv("XAGENT_STORAGE_ROOT", str(root))
    monkeypatch.delenv("LANCEDB_PATH", raising=False)
    if lancedb_dir_set:
        monkeypatch.setenv("LANCEDB_DIR", str(root / "lancedb"))
    else:
        monkeypatch.delenv("LANCEDB_DIR")
    default_dir = LanceDBConnectionManager.get_default_lancedb_dir
    default_dir.cache_clear()
    root.chmod(0o555)
    try:
        assert lock_deployment_kb_engine() is KBStorageBackend.LANCEDB
    finally:
        root.chmod(0o755)
        default_dir.cache_clear()
    assert list(root.iterdir()) == []


def test_a_missing_lancedb_directory_is_created_and_recorded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("LANCEDB_DIR", str(tmp_path / "new" / "lancedb"))

    assert lock_deployment_kb_engine() is KBStorageBackend.LANCEDB
    assert _record().read_text() == "lancedb\n"


@pytest.mark.parametrize(
    ("kind", "mode"),
    [
        ("file", 0o755),
        ("directory", 0o000),
        ("directory", 0o311),
        ("directory", 0o644),
        ("parent", 0o000),
    ],
)
def test_an_unreachable_lancedb_directory_warns_and_starts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    kind: str,
    mode: int,
) -> None:
    locked = tmp_path / "unreachable"
    target = locked / "lancedb" if kind == "parent" else locked
    if kind == "file":
        target.write_text("x")
    else:
        target.mkdir(parents=True)
    monkeypatch.setenv("LANCEDB_DIR", str(target))
    locked.chmod(mode)
    try:
        with caplog.at_level(logging.WARNING, logger=vector_backend.__name__):
            assert lock_deployment_kb_engine() is KBStorageBackend.LANCEDB
    finally:
        locked.chmod(0o755)
    assert "Cannot reach the LanceDB directory" in caplog.text
    if kind == "directory":
        assert list(target.iterdir()) == []


def test_a_record_is_never_written_without_the_lock() -> None:
    lock_path = _record().with_name(f"{KB_ENGINE_RECORD}.lock")
    lock_path.touch()
    lock_path.chmod(0o444)

    assert lock_deployment_kb_engine() is KBStorageBackend.LANCEDB
    assert not _record().exists()


def test_an_empty_lancedb_dir_warns_and_starts(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("LANCEDB_DIR", "")

    with caplog.at_level(logging.WARNING, logger=vector_backend.__name__):
        assert lock_deployment_kb_engine() is KBStorageBackend.LANCEDB
    assert "LANCEDB_DIR is empty; the KB engine is not checked" in caplog.text


def test_an_unreadable_record_is_refused_without_the_delete_hint() -> None:
    _record().mkdir()

    with pytest.raises(ConfigurationError, match="Cannot read the KB engine") as raised:
        lock_deployment_kb_engine()
    assert "delete the record file" not in str(raised.value)


@pytest.mark.parametrize("failing", ["connect", "list"])
def test_a_detection_error_warns_and_starts_without_a_record(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, failing: str
) -> None:
    connects, closed, released = [], [], []

    class Connection:
        def list_tables(self) -> list[str]:
            raise RuntimeError("lance listing failed")

        def close(self) -> None:
            closed.append(1)

    def connect(_: str) -> Connection:
        connects.append(1)
        if failing == "connect":
            raise OSError("connect failed")
        return Connection()

    class SpyLock(FileLock):
        def release(self, force: bool = False) -> None:
            released.append(force)
            super().release(force)

    monkeypatch.setattr(lancedb, "connect", connect)
    monkeypatch.setattr(vector_backend, "FileLock", SpyLock)

    with caplog.at_level(logging.WARNING, logger=vector_backend.__name__):
        assert lock_deployment_kb_engine() is KBStorageBackend.LANCEDB
    assert "Cannot detect the KB engine" in caplog.text
    assert connects == [1]
    assert closed == ([1] if failing == "list" else [])
    assert released[:1] == [False]
    assert not _record().exists()


def test_detection_closes_every_table_and_the_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closed = []

    class Table:
        def __init__(self, name: str) -> None:
            self.name = name

        def count_rows(self) -> int:
            if self.name == "documents":
                raise OSError("unreadable")
            return 0

        def close(self) -> None:
            closed.append(self.name)

    class Connection:
        def list_tables(self) -> list[str]:
            return ["kb_ids", "documents", "embeddings_a"]

        def open_table(self, name: str) -> Table:
            return Table(name)

        def close(self) -> None:
            closed.append("connection")

    monkeypatch.setattr(lancedb, "connect", lambda _: Connection())

    assert lock_deployment_kb_engine() is KBStorageBackend.LANCEDB
    assert sorted(closed) == ["connection", "documents", "embeddings_a", "kb_ids"]


def test_a_held_lock_times_out_with_a_warning_naming_the_lock_file(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(vector_backend, "_LOCK_TIMEOUT_SECONDS", 0.1)
    lock_path = _record().with_name(f"{KB_ENGINE_RECORD}.lock")

    with FileLock(str(lock_path)):
        with caplog.at_level(logging.WARNING, logger=vector_backend.__name__):
            assert lock_deployment_kb_engine() is KBStorageBackend.LANCEDB
    assert str(lock_path) in caplog.text
    assert not _record().exists()


@pytest.mark.parametrize(
    ("error", "warning"),
    [
        (NotImplementedError("flock is not supported"), "flock is not supported"),
        (Timeout("lock"), "or the filesystem does not support flock"),
    ],
)
@pytest.mark.parametrize("table", ["documents", "kb_ids"])
def test_an_unusable_lock_compares_without_a_record(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    error: Exception,
    warning: str,
    table: str,
) -> None:
    class UnusableLock:
        def __init__(self, *_: object, **__: object) -> None:
            pass

        def acquire(self) -> None:
            raise error

    monkeypatch.setattr(vector_backend, "FileLock", UnusableLock)
    _seed(table)

    with caplog.at_level(logging.WARNING, logger=vector_backend.__name__):
        if table == "kb_ids":
            with pytest.raises(ConfigurationError, match="could not be written"):
                lock_deployment_kb_engine()
        else:
            assert lock_deployment_kb_engine() is KBStorageBackend.LANCEDB
    assert warning in caplog.text
    assert not _record().exists()


_FIRST_START = """
import os, sys, time
from pathlib import Path
from xagent.core.tools.core.RAG_tools.storage import vector_backend

go, writes = Path(sys.argv[1]), sys.argv[2]
replace = os.replace

def slow_replace(source, target):
    time.sleep(0.3)
    with open(writes, "a") as log:
        log.write(f"{os.getpid()}\\n")
    replace(source, target)

os.replace = slow_replace
(go.parent / f"ready-{os.getpid()}").touch()
while not go.exists():
    time.sleep(0.01)
print(vector_backend.lock_deployment_kb_engine().value)
"""


def test_concurrent_first_starts_write_the_record_once(tmp_path: Path) -> None:
    go, writes = tmp_path / "go", tmp_path / "writes"
    children = [
        subprocess.Popen(
            [sys.executable, "-c", _FIRST_START, str(go), str(writes)],
            stdout=subprocess.PIPE,
            text=True,
        )
        for _ in range(6)
    ]
    deadline = time.monotonic() + 120
    while len(list(tmp_path.glob("ready-*"))) < 6 and time.monotonic() < deadline:
        time.sleep(0.05)
    go.touch()

    outputs = [child.communicate(timeout=120)[0].split()[-1:] for child in children]

    assert [child.returncode for child in children] == [0] * 6
    assert outputs == [["lancedb"]] * 6
    assert len(writes.read_text().splitlines()) == 1
    assert _record().read_text() == "lancedb\n"


async def _context_backend(collection: CollectionInfo) -> KBStorageBackend:
    await get_metadata_store().save_collection(collection)
    context = await get_kb_coordinator().get_context(
        KBContextRequest(collection=collection.name)
    )
    return context.backend


@pytest.mark.parametrize(
    ("binding", "expected"),
    [
        (None, KBStorageBackend.MILVUS),
        ({"backend": ""}, KBStorageBackend.MILVUS),
        ({"backend": "lancedb"}, KBStorageBackend.LANCEDB),
    ],
)
def test_a_binding_wins_and_a_missing_one_means_the_deployment_engine(
    milvus_setting: None, binding: dict[str, str] | None, expected: KBStorageBackend
) -> None:
    extra = {} if binding is None else {"kb_storage": binding}
    collection = CollectionInfo(name="kb", extra_metadata=extra)

    assert asyncio.run(_context_backend(collection)) is expected


def test_a_missing_collection_resolves_to_the_deployment_engine(
    milvus_setting: None,
) -> None:
    request = KBContextRequest(collection="new", hide_missing=True)
    context = asyncio.run(get_kb_coordinator().get_context(request))

    assert context.collection_info is None
    assert context.backend is KBStorageBackend.MILVUS


def test_the_three_binding_writers_record_the_deployment_engine(
    milvus_setting: None,
) -> None:
    coordinator = get_kb_coordinator()
    writers = {
        "api": coordinator.api.ensure_collection_backend_binding,
        "pipeline": coordinator.pipeline.ensure_collection_backend_binding_async,
        "tool": coordinator.tools.ensure_agent_collection_backend_binding,
    }

    async def write_all() -> None:
        for name, writer in writers.items():
            await get_metadata_store().save_collection(CollectionInfo(name=name))
            await writer(name)

    asyncio.run(write_all())

    for name in writers:
        saved = asyncio.run(get_metadata_store().get_collection(name))
        assert saved.extra_metadata["kb_storage"] == {"backend": "milvus"}


def test_a_binding_to_another_engine_blocks_search_and_ingest_not_delete(
    milvus_setting: None,
) -> None:
    asyncio.run(
        get_metadata_store().save_collection(
            CollectionInfo(name="kb", extra_metadata={"kb_storage": "lancedb"})
        )
    )
    coordinator = get_kb_coordinator()

    with pytest.raises(ValueError, match="bound to the lancedb engine.*re-import"):
        coordinator.search_dense_sync("kb", "m", [0.1], user_id=None, is_admin=True)
    with pytest.raises(ValueError, match="re-import"):
        coordinator.vector_storage.read_chunks_for_embedding("kb", "d", "p", "m")
    with pytest.raises(ValueError, match="re-import"):
        coordinator.vector_storage.write_vectors_to_db("kb", [], user_id=1)
    assert coordinator.delete_document_record_sync("kb", "d", is_admin=True) == 0
