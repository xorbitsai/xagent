"""Embedding-migration freeze: state table, fail-closed reads, and core guards."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from xagent.core.tools.core.RAG_tools.core.exceptions import CollectionMigratingError
from xagent.core.tools.core.RAG_tools.core.schemas import CrawlResult, WebCrawlConfig
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
    ensure_collection_writable,
    read_migration_state,
    write_migration_state,
)
from xagent.core.tools.core.RAG_tools.pipelines import document_ingestion, web_ingestion
from xagent.core.tools.core.RAG_tools.storage.factory import get_metadata_store


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


def test_commit_gate_rechecks_freeze_mid_ingest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate_calls: list[int] = []

    def _gate() -> None:
        gate_calls.append(1)
        if len(gate_calls) == 1:
            _freeze()

    monkeypatch.setattr(
        document_ingestion, "validate_document_processing_sync", lambda **_: None
    )
    monkeypatch.setattr(
        document_ingestion,
        "resolve_effective_embedding_model_sync",
        lambda *_: "embedding-default",
    )
    config = SimpleNamespace(
        id="embedding-default", model_name="stub", model_provider="stub", dimension=2
    )
    monkeypatch.setattr(
        document_ingestion,
        "_resolve_embedding_adapter",
        lambda _cfg: (config, object()),
    )
    monkeypatch.setattr(
        document_ingestion, "initialize_collection_embedding_sync", lambda **_: None
    )
    monkeypatch.setattr(
        document_ingestion,
        "register_document",
        lambda **_: {"doc_id": "doc-1", "created": True},
    )
    monkeypatch.setattr(
        document_ingestion, "update_collection_stats_sync", lambda **_: None
    )
    monkeypatch.setattr(
        document_ingestion, "_record_ingestion_status", lambda *a, **k: None
    )

    def _boom(*args: object, **kwargs: object) -> None:
        raise AssertionError("write after the freeze was reached")

    monkeypatch.setattr(document_ingestion, "parse_document", _boom)

    with pytest.raises(CollectionMigratingError):
        document_ingestion._process_document_impl(
            "kb", "/tmp/doc.pdf", commit_gate=_gate
        )

    assert len(gate_calls) == 1


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


@pytest.mark.asyncio
async def test_web_ingestion_stops_before_page_write_after_mid_crawl_freeze(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FreezingCrawler:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.failed_urls: dict[str, str] = {}

        async def crawl(self) -> list[CrawlResult]:
            _freeze()
            return [
                CrawlResult(
                    url="https://example.com/page",
                    title="Page",
                    content_markdown="content",
                    status="success",
                    depth=0,
                    content_length=7,
                )
            ]

    monkeypatch.setattr(web_ingestion, "WebCrawler", _FreezingCrawler)
    file_handler_calls: list[int] = []

    def _file_handler(*args: object, **kwargs: object) -> dict:
        file_handler_calls.append(1)
        return {}

    with pytest.raises(CollectionMigratingError):
        await web_ingestion._run_web_ingestion_impl(
            "kb",
            WebCrawlConfig(start_url="https://example.com"),
            file_handler=_file_handler,
        )

    assert file_handler_calls == []


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
