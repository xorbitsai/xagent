"""Embedding-migration freeze: state table, fail-closed reads, and core guards."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from xagent.core.tools.core.RAG_tools.core.exceptions import CollectionMigratingError
from xagent.core.tools.core.RAG_tools.core.schemas import (
    CrawlResult,
    IngestionResult,
    WebCrawlConfig,
)
from xagent.core.tools.core.RAG_tools.kb.coordinator import KBCoordinator
from xagent.core.tools.core.RAG_tools.management import (
    collections as collections_module,
)
from xagent.core.tools.core.RAG_tools.management import migration_state as state_module
from xagent.core.tools.core.RAG_tools.management.migration_state import (
    FROZEN,
    IDLE,
    MIGRATION_TABLE,
    CollectionMigrationState,
    admitted_async,
    admitted_context_async,
    claim_freeze,
    clear_migration_state,
    ensure_collection_writable,
    heartbeat_freeze,
    pending_collection,
    read_migration_state,
    read_raw_state,
    restore_state,
    run_in_admitted_context,
    write_migration_state,
)
from xagent.core.tools.core.RAG_tools.pipelines import document_ingestion, web_ingestion
from xagent.core.tools.core.RAG_tools.storage.factory import get_metadata_store
from xagent.core.utils.blocking import run_blocking


def _freeze(name: str = "kb") -> None:
    write_migration_state(
        name,
        CollectionMigrationState(
            state=FROZEN,
            source_model_id="model-a",
            target_model_id="model-b",
        ),
    )


def _write_raw_state(name: str, state: str) -> None:
    conn = get_metadata_store().get_raw_connection()
    state_module._ensure_table(conn)
    table = conn.open_table(MIGRATION_TABLE)
    try:
        table.merge_insert(
            ["name"]
        ).when_matched_update_all().when_not_matched_insert_all().execute(
            [
                {
                    "name": name,
                    "state": state,
                    "source_model_id": None,
                    "target_model_id": None,
                    "updated_at": None,
                }
            ]
        )
    finally:
        from xagent.core.tools.core.RAG_tools.LanceDB.schema_manager import (
            _safe_close_table,
        )

        _safe_close_table(table)


def test_missing_table_means_idle() -> None:
    assert read_migration_state("kb").state == IDLE
    ensure_collection_writable("kb")


def test_missing_row_means_idle() -> None:
    _freeze("other")

    assert read_migration_state("kb").state == IDLE
    ensure_collection_writable("kb")


def test_read_error_is_not_treated_as_idle(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(_conn: object) -> list[str]:
        raise RuntimeError("state table unreadable")

    monkeypatch.setattr(state_module, "list_table_names", _boom)

    with pytest.raises(RuntimeError):
        read_migration_state("kb")
    with pytest.raises(RuntimeError):
        ensure_collection_writable("kb")


def test_unknown_state_value_is_treated_as_frozen() -> None:
    _write_raw_state("kb", "something-new")

    assert read_migration_state("kb").state == FROZEN
    with pytest.raises(CollectionMigratingError):
        ensure_collection_writable("kb")


def test_state_round_trip_and_other_collections_unaffected() -> None:
    stamped = write_migration_state(
        "kb",
        CollectionMigrationState(
            state=FROZEN,
            source_model_id="model-a",
            target_model_id="model-b",
        ),
    )

    state = read_migration_state("kb")
    assert state.state == FROZEN
    assert state.source_model_id == "model-a"
    assert state.target_model_id == "model-b"
    assert stamped.updated_at is not None
    assert read_migration_state("other").state == IDLE


def test_write_rejects_unknown_state() -> None:
    with pytest.raises(ValueError):
        write_migration_state("kb", CollectionMigrationState(state="something-new"))


def test_frozen_collection_blocks_document_ingestion_before_any_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _freeze()

    def _boom(*args: object, **kwargs: object) -> None:
        raise AssertionError("document ingestion reached a write step")

    for name in (
        "register_document",
        "parse_document",
        "write_vectors_to_db",
        "update_collection_stats_sync",
        "ProgressManager",
    ):
        monkeypatch.setattr(document_ingestion, name, _boom)

    with pytest.raises(CollectionMigratingError):
        document_ingestion._process_document_impl("kb", "/tmp/doc.pdf")


def test_background_job_execution_refuses_frozen_collection() -> None:
    from xagent.core.tools.core.RAG_tools.pipelines.document_ingestion import (
        run_document_ingestion,
    )

    _freeze()

    with pytest.raises(CollectionMigratingError):
        run_document_ingestion(collection="kb", source_path="/tmp/doc.pdf")


@pytest.mark.asyncio
async def test_frozen_collection_blocks_web_ingestion_before_crawl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _freeze()

    def _boom(*args: object, **kwargs: object) -> None:
        raise AssertionError("web ingestion constructed a crawler")

    monkeypatch.setattr(web_ingestion, "WebCrawler", _boom)

    with pytest.raises(CollectionMigratingError):
        await web_ingestion._run_web_ingestion_impl(
            "kb", WebCrawlConfig(start_url="https://example.com")
        )


def test_unfrozen_collection_proceeds_past_guards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Reached(RuntimeError):
        pass

    monkeypatch.setattr(
        document_ingestion,
        "ProgressManager",
        lambda *a, **k: (_ for _ in ()).throw(_Reached),
    )
    with pytest.raises(_Reached):
        document_ingestion._process_document_impl("kb", "/tmp/doc.pdf")

    reached_store = {}

    def _store() -> None:
        reached_store["called"] = True
        raise _Reached

    monkeypatch.setattr(collections_module, "get_vector_index_store", _store)
    with pytest.raises(_Reached):
        collections_module._delete_document_impl(
            collection="kb", doc_id="doc-1", user_id=1, coordinator=None
        )
    assert reached_store["called"]


def _migration_names() -> list[str]:
    conn = get_metadata_store().get_raw_connection()
    if MIGRATION_TABLE not in state_module.list_table_names(conn):
        return []
    table = conn.open_table(MIGRATION_TABLE)
    try:
        return [row["name"] for row in table.search().to_arrow().to_pylist()]
    finally:
        from xagent.core.tools.core.RAG_tools.LanceDB.schema_manager import (
            _safe_close_table,
        )

        _safe_close_table(table)


def test_write_migration_state_retries_commit_conflict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_conn = get_metadata_store().get_raw_connection()
    state_module._ensure_table(real_conn)
    open_calls = {"n": 0}

    class _FlakyConn:
        def __getattr__(self, name: str) -> object:
            if name == "open_table":

                def _open(table_name: str) -> object:
                    open_calls["n"] += 1
                    if open_calls["n"] == 1:
                        raise RuntimeError("commit conflict")
                    return real_conn.open_table(table_name)

                return _open
            return getattr(real_conn, name)

    monkeypatch.setattr(
        state_module,
        "get_metadata_store",
        lambda: SimpleNamespace(get_raw_connection=lambda: _FlakyConn()),
    )

    stamped = write_migration_state("kb", CollectionMigrationState(state=FROZEN))

    assert stamped.updated_at is not None
    assert open_calls["n"] >= 2
    assert read_migration_state("kb").state == FROZEN


def test_clear_migration_state_removes_row() -> None:
    _freeze("kb")
    assert "kb" in _migration_names()

    clear_migration_state("kb")

    assert "kb" not in _migration_names()
    assert read_migration_state("kb").state == IDLE


def test_pending_scope_admits_on_first_check_then_suppresses() -> None:
    with pending_collection("kb"):
        ensure_collection_writable("kb")
        _freeze("kb")
        ensure_collection_writable("kb")

    with pytest.raises(CollectionMigratingError):
        ensure_collection_writable("kb")


@pytest.mark.asyncio
async def test_admitted_scope_suppresses_nested_check_and_does_not_leak() -> None:
    async with admitted_async("kb"):
        _freeze("kb")
        ensure_collection_writable("kb")

    with pytest.raises(CollectionMigratingError):
        ensure_collection_writable("kb")


@pytest.mark.asyncio
async def test_admitted_scope_propagates_into_run_blocking() -> None:
    async with admitted_async("kb"):
        _freeze("kb")
        await run_blocking(ensure_collection_writable, "kb")

    with pytest.raises(CollectionMigratingError):
        await run_blocking(ensure_collection_writable, "kb")


@pytest.mark.asyncio
async def test_admitted_context_carries_the_scope_into_another_thread() -> None:
    admission = await admitted_context_async("kb")
    _freeze("kb")

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, admission.run, ensure_collection_writable, "kb")

    with pytest.raises(CollectionMigratingError):
        await run_blocking(ensure_collection_writable, "kb")


class _FakePipelineFacade:
    def __init__(self) -> None:
        self.finished: list[tuple[str, str]] = []
        self.side_effects: list[dict[str, object]] = []
        self.rolled_back: list[str] = []

    @contextmanager
    def web_page_operation(self, **kwargs: object):
        yield object()

    def record_web_page_file_side_effect(
        self, operation: object, **kwargs: object
    ) -> None:
        self.side_effects.append(kwargs)

    def finish_web_page_operation(
        self, operation: object, *, status: str, message: str
    ) -> None:
        self.finished.append((status, message))

    def rollback_failed_ingestion_sync(self, request: object) -> SimpleNamespace:
        callback = getattr(request, "file_compensation", None)
        if callback is not None:
            callback()
            self.rolled_back.append(str(getattr(request, "source", "")))
        return SimpleNamespace(warnings=[], first_error=None)


@pytest.mark.asyncio
async def test_web_ingest_children_stay_admitted_after_a_freeze(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A freeze after web admission must not abort the parent or skip callbacks."""
    committed: list[str] = []
    compensated: list[str] = []

    class _Crawler:
        failed_urls: dict[str, str] = {}
        total_urls_found = 2

        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        async def crawl(self) -> list[CrawlResult]:
            _freeze("kb")
            return [
                CrawlResult(
                    url=f"https://example.com/page-{i}",
                    title=f"page {i}",
                    content_markdown="body",
                    status="success",
                    depth=0,
                    content_length=4,
                )
                for i in range(2)
            ]

        def get_statistics(self) -> dict[str, object]:
            return {}

    def _ingest(
        *, collection: str, source_path: str, **kwargs: object
    ) -> IngestionResult:
        # The real nested guard; it must be a no-op inside the parent's admission.
        ensure_collection_writable(collection)
        if Path(source_path).name.startswith("1_"):
            return IngestionResult(
                status="error", message="page failed", doc_id="doc-1"
            )
        return IngestionResult(
            status="success",
            message="page ok",
            doc_id="doc-0",
            chunk_count=1,
            embedding_count=1,
        )

    def _file_handler(temp_file: Path, title: str, collection: str, url: str):
        return {
            "file_path": str(temp_file),
            "file_id": None,
            "commit_on_success": lambda: committed.append(url),
            "file_compensation": lambda: compensated.append(url),
        }

    monkeypatch.setattr(web_ingestion, "WebCrawler", _Crawler)
    monkeypatch.setattr(web_ingestion, "run_document_ingestion", _ingest)
    facade = _FakePipelineFacade()

    result = await web_ingestion._run_web_ingestion_impl(
        "kb",
        WebCrawlConfig(start_url="https://example.com"),
        user_id=1,
        is_admin=False,
        file_handler=_file_handler,
        pipeline_facade=facade,
    )

    assert read_migration_state("kb").state == FROZEN
    assert result.status == "partial"
    assert result.documents_created == 1
    assert committed == ["https://example.com/page-0"]
    assert compensated == ["https://example.com/page-1"]
    assert sorted(status for status, _ in facade.finished) == ["error", "success"]


@pytest.mark.asyncio
async def test_run_in_admitted_context_supports_overlapping_workers() -> None:
    import threading

    admission = await admitted_context_async("kb")
    _freeze("kb")
    entered = threading.Event()
    release = threading.Event()

    def _hold() -> None:
        entered.set()
        release.wait(5)

    loop = asyncio.get_running_loop()
    holder = loop.run_in_executor(None, admission.run, _hold)
    assert await asyncio.to_thread(entered.wait, 5)
    try:
        await loop.run_in_executor(
            None,
            run_in_admitted_context,
            admission,
            ensure_collection_writable,
            "kb",
        )
    finally:
        release.set()
        await holder


def test_frozen_collection_blocks_document_delete_before_storage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _freeze()

    def _boom() -> None:
        raise AssertionError("document delete reached the vector store")

    monkeypatch.setattr(collections_module, "get_vector_index_store", _boom)

    with pytest.raises(CollectionMigratingError):
        collections_module._delete_document_impl(
            collection="kb", doc_id="doc-1", user_id=1, coordinator=None
        )


@pytest.mark.asyncio
async def test_frozen_collection_blocks_collection_delete_and_rename() -> None:
    _freeze()
    coordinator = KBCoordinator()

    with pytest.raises(CollectionMigratingError):
        await coordinator.delete_collection("kb", user_id=None, is_admin=True)
    with pytest.raises(CollectionMigratingError):
        await coordinator.rename_collection(
            "kb", "new-name", user_id=None, is_admin=True
        )


@pytest.mark.asyncio
async def test_frozen_rename_refuses_before_opening_the_collection() -> None:
    _freeze()
    coordinator = KBCoordinator()

    with patch.object(
        coordinator,
        "open_collection",
        new=AsyncMock(side_effect=AssertionError("collection was opened")),
    ):
        with pytest.raises(CollectionMigratingError):
            await coordinator.rename_collection(
                "kb", "new-name", user_id=None, is_admin=True
            )


class _FakeCollectionHandle:
    def list_collection_documents(self, **kwargs: object) -> list[str]:
        return []

    def delete_collection_data(self, **kwargs: object) -> dict[str, int]:
        return {"documents": 0}

    def count_documents(self, **kwargs: object) -> int:
        return 0

    async def delete_collection_config(self, *args: object, **kwargs: object) -> None:
        return None

    def rename_collection_data(self, *args: object, **kwargs: object) -> list[str]:
        return []

    def rename_collection_status(self, *args: object, **kwargs: object) -> list[str]:
        return []

    async def rename_collection_metadata(self, *args: object, **kwargs: object) -> None:
        return None


# --- Single admission per operation -----------------------------------------


class _RecordingHandle(_FakeCollectionHandle):
    def __init__(self) -> None:
        self.renames: list[str] = []

    def rename_collection_data(self, *args: object, **kwargs: object) -> list[str]:
        self.renames.append("data")
        return []

    def rename_collection_status(self, *args: object, **kwargs: object) -> list[str]:
        self.renames.append("status")
        return []

    async def rename_collection_metadata(self, *args: object, **kwargs: object) -> None:
        self.renames.append("metadata")


@pytest.mark.asyncio
async def test_rename_inside_admission_completes_after_a_freeze() -> None:
    write_migration_state("kb", CollectionMigrationState(state=IDLE))
    coordinator = KBCoordinator()
    handle = _RecordingHandle()

    with patch.object(
        coordinator, "open_collection", new=AsyncMock(return_value=handle)
    ):
        async with admitted_async("kb"):
            _freeze("kb")
            warnings = await coordinator.rename_collection(
                "kb", "kb-new", user_id=None, is_admin=True
            )

    assert warnings == []
    assert handle.renames == ["data", "status", "metadata"]
    # Rename never moves the row; the migration job owns its lifecycle.
    assert read_migration_state("kb").state == FROZEN
    assert read_migration_state("kb-new").state == IDLE


@pytest.mark.asyncio
async def test_direct_coordinator_rename_still_checks_outside_a_scope() -> None:
    _freeze("kb")
    coordinator = KBCoordinator()

    with pytest.raises(CollectionMigratingError):
        await coordinator.rename_collection("kb", "kb-new", user_id=None, is_admin=True)


class _TenantHandle(_RecordingHandle):
    def list_collection_documents(self, **kwargs: object) -> list[str]:
        return ["doc-a"]

    def delete_documents_data(
        self, doc_ids: list[str], **kwargs: object
    ) -> dict[str, int]:
        return {"documents": len(doc_ids)}


@pytest.mark.asyncio
async def test_delete_and_rename_leave_the_migration_row_untouched() -> None:
    write_migration_state("kb", CollectionMigrationState(state=IDLE))
    coordinator = KBCoordinator()

    with patch.object(
        coordinator,
        "open_collection",
        new=AsyncMock(return_value=_TenantHandle()),
    ):
        async with admitted_async("kb"):
            _freeze("kb")
            await coordinator.delete_collection("kb", user_id=7, is_admin=False)
            await coordinator.rename_collection(
                "kb", "kb-new", user_id=7, is_admin=False
            )

    assert "kb" in _migration_names()
    assert read_migration_state("kb").state == FROZEN
    assert read_migration_state("kb-new").state == IDLE


@pytest.mark.asyncio
async def test_direct_coordinator_rename_checks_the_target_name() -> None:
    _freeze("kb-target")
    coordinator = KBCoordinator()

    with patch.object(
        coordinator,
        "open_collection",
        new=AsyncMock(side_effect=AssertionError("collection was opened")),
    ):
        with pytest.raises(CollectionMigratingError):
            await coordinator.rename_collection(
                "kb", "kb-target", user_id=1, is_admin=False
            )


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["delete", "rename"])
async def test_coordinator_checks_the_canonical_source_name(operation: str) -> None:
    _freeze("kb")
    coordinator = KBCoordinator()

    with patch.object(
        coordinator,
        "open_collection",
        new=AsyncMock(side_effect=AssertionError("collection was opened")),
    ):
        with pytest.raises(CollectionMigratingError):
            if operation == "delete":
                await coordinator.delete_collection("  kb  ", None, True)
            else:
                await coordinator.rename_collection("  kb  ", "new-name", None, True)


def _store_on(path: Path) -> SimpleNamespace:
    import lancedb

    conn = lancedb.connect(str(path))
    return SimpleNamespace(
        get_raw_connection=lambda: conn, get_collection=AsyncMock(return_value=None)
    )


@pytest.mark.asyncio
async def test_injected_coordinator_reads_the_freeze_from_its_own_store(
    tmp_path: Path,
) -> None:
    from unittest.mock import MagicMock

    store = _store_on(tmp_path / "injected")
    state_module._merge_row(
        store.get_raw_connection(),
        {"name": "kb-b", **CollectionMigrationState(state=FROZEN).to_row()},
    )
    _freeze("kb-a")
    shim = MagicMock()
    shim.get_metadata_store.return_value = store
    coordinator = KBCoordinator(storage_shim=shim)

    with patch.object(
        coordinator, "open_collection", new=AsyncMock(return_value=_RecordingHandle())
    ):
        with pytest.raises(CollectionMigratingError):
            await coordinator.rename_collection("src", "kb-b", None, True)
        with pytest.raises(CollectionMigratingError):
            await coordinator.delete_collection("kb-b", None, True)
        # Frozen only in the ambient store: not this coordinator's concern.
        await coordinator.rename_collection("src", "kb-a", None, True)


@pytest.mark.asyncio
async def test_direct_coordinator_rename_checks_the_stripped_target_name() -> None:
    _freeze("kb-target")
    coordinator = KBCoordinator()

    with patch.object(
        coordinator,
        "open_collection",
        new=AsyncMock(side_effect=AssertionError("collection was opened")),
    ):
        with pytest.raises(CollectionMigratingError):
            await coordinator.rename_collection(
                "kb", " kb-target ", user_id=1, is_admin=False
            )


def _claim(owner: str, *, stale: float = 60.0) -> str:
    return claim_freeze(
        "kb",
        source_model_id="old",
        target_model_id="new",
        mode="switch",
        owner=owner,
        stale_after_seconds=stale,
    )


def test_claim_freeze_claims_resumes_and_refuses() -> None:
    assert _claim("a") == "claimed"
    assert _claim("a") == "resumed"
    assert _claim("b") == "busy"
    assert read_migration_state("kb").owner == "a"
    assert _claim("b", stale=0.0) == "resumed"
    assert read_migration_state("kb").owner == "b"


def test_claim_freeze_over_an_idle_record_keeps_nothing_of_it() -> None:
    write_migration_state(
        "kb",
        CollectionMigrationState(
            state=IDLE, source_model_id="x", target_model_id="old", mode="switch"
        ),
    )

    assert _claim("a") == "claimed"
    state = read_migration_state("kb")
    assert (state.state, state.owner, state.target_model_id) == (FROZEN, "a", "new")


def test_claim_freeze_loses_when_the_row_changed_after_its_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_migration_state("kb", CollectionMigrationState(state=FROZEN, owner="dead"))
    stale = state_module._read_row("kb")
    # Another run took the window between this run's read and its write.
    write_migration_state("kb", CollectionMigrationState(state=FROZEN, owner="winner"))
    real = state_module._read_row
    reads = iter([stale])
    monkeypatch.setattr(
        state_module,
        "_read_row",
        lambda name, store=None: next(reads, None) or real(name, store),
    )

    assert _claim("late", stale=0.0) == "busy"
    assert read_migration_state("kb").owner == "winner"


def test_claim_freeze_refuses_an_unreadable_state_value() -> None:
    write_migration_state("kb", CollectionMigrationState(state=FROZEN, owner="a"))
    table = get_metadata_store().get_raw_connection().open_table(MIGRATION_TABLE)
    table.update("name = 'kb'", {"state": "garbled"})

    assert _claim("b", stale=0.0) == "busy"


def test_heartbeat_renews_only_the_owners_row() -> None:
    write_migration_state("kb", CollectionMigrationState(state=FROZEN, owner="a"))
    before = read_migration_state("kb").updated_at

    assert heartbeat_freeze("kb", "b") is False
    assert heartbeat_freeze("kb", "a") is True
    assert read_migration_state("kb").updated_at >= before

    write_migration_state("kb", CollectionMigrationState(state=IDLE, owner="a"))
    assert heartbeat_freeze("kb", "a") is False


def test_restore_state_puts_the_row_back_exactly() -> None:
    write_migration_state(
        "kb",
        CollectionMigrationState(
            state=IDLE, source_model_id="x", target_model_id="y", mode="switch"
        ),
    )
    before = read_raw_state("kb")
    _claim("a")

    restore_state("kb", before)

    assert read_raw_state("kb") == before
    restore_state("kb", None)
    assert read_raw_state("kb") is None


def _old_schema_table(owner_type: Any) -> None:
    import pyarrow as pa

    fields = [
        ("name", pa.string()),
        ("state", pa.string()),
        ("source_model_id", pa.string()),
        ("target_model_id", pa.string()),
        ("updated_at", pa.timestamp("us")),
    ]
    if owner_type is not None:
        fields += [("owner", owner_type), ("mode", owner_type)]
    row = {"name": "kb", "state": IDLE, "source_model_id": "m", "target_model_id": None}
    get_metadata_store().get_raw_connection().create_table(
        MIGRATION_TABLE,
        data=pa.Table.from_pylist(
            [
                {
                    **row,
                    "updated_at": None,
                    **({"owner": None, "mode": None} if owner_type is not None else {}),
                }
            ],
            schema=pa.schema(fields),
        ),
    )


@pytest.mark.parametrize("owner_type", ["absent", "null-typed"])
def test_a_table_from_an_older_version_takes_owner_writes(owner_type: str) -> None:
    import pyarrow as pa

    _old_schema_table(None if owner_type == "absent" else pa.null())

    write_migration_state(
        "kb2", CollectionMigrationState(state=FROZEN, owner="run-1", mode="switch")
    )

    assert read_migration_state("kb2").owner == "run-1"
    assert read_migration_state("kb2").mode == "switch"
    assert read_migration_state("kb").source_model_id == "m"


def test_claim_freeze_on_a_stale_absent_read_does_not_take_a_live_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_migration_state("kb", CollectionMigrationState(state=FROZEN, owner="winner"))
    real = state_module._read_row
    calls = {"n": 0}

    def _read_row(name: str, store: Any = None) -> Any:
        calls["n"] += 1
        # This run read "no row" just before another run inserted its freeze.
        return None if calls["n"] == 1 else real(name, store)

    monkeypatch.setattr(state_module, "_read_row", _read_row)

    assert _claim("late") == "busy"


@pytest.mark.parametrize("owner_type", ["absent", "null-typed"])
def test_claim_freeze_upgrades_a_table_from_an_older_version(
    owner_type: str,
) -> None:
    import pyarrow as pa

    _old_schema_table(None if owner_type == "absent" else pa.null())

    assert _claim("run-1") == "claimed"
    state = read_migration_state("kb")
    assert (state.state, state.owner, state.target_model_id) == (FROZEN, "run-1", "new")


def test_claim_freeze_updates_only_a_row_in_the_state_it_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_migration_state(
        "kb", CollectionMigrationState(state=IDLE, target_model_id="old")
    )
    stale = state_module._read_row("kb")
    # Another run froze it between this run's read and its write.
    write_migration_state("kb", CollectionMigrationState(state=FROZEN, owner="winner"))
    real = state_module._read_row
    calls = {"n": 0}

    def _read_row(name: str, store: Any = None) -> Any:
        calls["n"] += 1
        return stale if calls["n"] == 1 else real(name, store)

    monkeypatch.setattr(state_module, "_read_row", _read_row)

    assert _claim("late") == "busy"
    state = read_migration_state("kb")
    assert (state.state, state.owner) == (FROZEN, "winner")
