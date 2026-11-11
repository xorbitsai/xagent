"""Embedding-migration freeze: state table, fail-closed reads, and core guards."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
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
    admitted,
    admitted_async,
    admitted_context_async,
    clear_migration_state,
    ensure_collection_writable,
    move_migration_state,
    read_migration_state,
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


def test_move_migration_state_rekeys_row() -> None:
    _freeze("old-kb")

    move_migration_state("old-kb", "new-kb")

    assert "old-kb" not in _migration_names()
    assert "new-kb" in _migration_names()
    assert read_migration_state("new-kb").state == FROZEN


def test_move_migration_state_noop_when_row_absent() -> None:
    move_migration_state("absent-kb", "other-kb")

    assert "other-kb" not in _migration_names()


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


@pytest.mark.asyncio
async def test_delete_collection_clears_migration_row() -> None:
    write_migration_state("kb", CollectionMigrationState(state=IDLE))
    coordinator = KBCoordinator()

    with patch.object(
        coordinator,
        "open_collection",
        new=AsyncMock(return_value=_FakeCollectionHandle()),
    ):
        await coordinator.delete_collection("kb", user_id=None, is_admin=True)

    assert "kb" not in _migration_names()


@pytest.mark.asyncio
async def test_rename_collection_moves_migration_row() -> None:
    write_migration_state("old-kb", CollectionMigrationState(state=IDLE))
    coordinator = KBCoordinator()

    with patch.object(
        coordinator,
        "open_collection",
        new=AsyncMock(return_value=_FakeCollectionHandle()),
    ):
        await coordinator.rename_collection(
            "old-kb", "new-kb", user_id=None, is_admin=True
        )

    assert "old-kb" not in _migration_names()
    assert "new-kb" in _migration_names()


# --- Single admission per operation -----------------------------------------


def test_admitted_scope_suppresses_nested_check_and_does_not_leak() -> None:
    with admitted("kb"):
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
    assert read_migration_state("kb-new").state == FROZEN
    assert read_migration_state("kb").state == IDLE


@pytest.mark.asyncio
async def test_direct_coordinator_rename_still_checks_outside_a_scope() -> None:
    _freeze("kb")
    coordinator = KBCoordinator()

    with pytest.raises(CollectionMigratingError):
        await coordinator.rename_collection("kb", "kb-new", user_id=None, is_admin=True)
