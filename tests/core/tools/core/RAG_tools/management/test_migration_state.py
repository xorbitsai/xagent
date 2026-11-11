"""Embedding-migration freeze: state persistence, file lock, and core guards."""

from __future__ import annotations

import pytest
from filelock import Timeout

from xagent.core.tools.core.RAG_tools.core.exceptions import CollectionMigratingError
from xagent.core.tools.core.RAG_tools.core.schemas import CollectionInfo, WebCrawlConfig
from xagent.core.tools.core.RAG_tools.kb.coordinator import KBCoordinator
from xagent.core.tools.core.RAG_tools.management import (
    collections as collections_module,
)
from xagent.core.tools.core.RAG_tools.management.collection_manager import (
    _get_collection_sync_impl,
    _run_in_separate_loop,
    collection_manager,
    update_collection_stats_sync,
)
from xagent.core.tools.core.RAG_tools.management.migration_state import (
    MIGRATION_METADATA_KEY,
    CollectionMigrationState,
    EmbeddingMigrationPhase,
    collection_migration_lock,
    ensure_collection_writable,
    read_migration_state,
    write_migration_state,
)
from xagent.core.tools.core.RAG_tools.pipelines import document_ingestion, web_ingestion


def _create_collection(name: str = "kb", **overrides) -> None:
    _run_in_separate_loop(
        collection_manager.save_collection(CollectionInfo(name=name, **overrides))
    )


def _freeze(name: str = "kb") -> None:
    write_migration_state(
        name,
        CollectionMigrationState(
            state=EmbeddingMigrationPhase.FROZEN,
            source_model_id="model-a",
            target_model_id="model-b",
        ),
    )


def test_write_state_preserves_other_extra_metadata_and_collection_fields() -> None:
    _create_collection("kb", documents=5, extra_metadata={"keep": "me"})

    write_migration_state(
        "kb",
        CollectionMigrationState(
            state=EmbeddingMigrationPhase.FROZEN,
            source_model_id="model-a",
            target_model_id="model-b",
        ),
    )

    collection = _get_collection_sync_impl("kb")
    assert collection.extra_metadata["keep"] == "me"
    state = collection.extra_metadata[MIGRATION_METADATA_KEY]
    assert state["state"] == "frozen"
    assert state["source_model_id"] == "model-a"
    assert state["target_model_id"] == "model-b"
    assert state["updated_at"]
    assert collection.documents == 5


def test_stats_update_after_state_write_does_not_drop_migration_state() -> None:
    _create_collection("kb", extra_metadata={"keep": "me"})
    _freeze()

    update_collection_stats_sync("kb", documents_delta=1)

    collection = _get_collection_sync_impl("kb")
    assert collection.documents == 1
    assert collection.extra_metadata["keep"] == "me"
    assert read_migration_state("kb").state == EmbeddingMigrationPhase.FROZEN


def test_ensure_collection_writable_raises_only_when_frozen() -> None:
    _create_collection("kb")
    ensure_collection_writable("kb")
    ensure_collection_writable("missing")

    _freeze()

    with pytest.raises(CollectionMigratingError):
        ensure_collection_writable("kb")


def test_file_lock_excludes_same_collection_but_not_others() -> None:
    with collection_migration_lock("collection-a", timeout=1):
        with pytest.raises(Timeout):
            with collection_migration_lock("collection-a", timeout=0.1):
                pass
        with collection_migration_lock("collection-b", timeout=1):
            pass


def test_frozen_collection_blocks_document_ingestion_before_any_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _create_collection("kb")
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


@pytest.mark.asyncio
async def test_frozen_collection_blocks_web_ingestion_before_crawl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _create_collection("kb")
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
    _create_collection("kb")
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
    _create_collection("kb")
    _freeze()
    coordinator = KBCoordinator()

    with pytest.raises(CollectionMigratingError):
        await coordinator.delete_collection("kb", user_id=None, is_admin=True)
    with pytest.raises(CollectionMigratingError):
        await coordinator.rename_collection(
            "kb", "new-name", user_id=None, is_admin=True
        )


def test_unfrozen_collection_proceeds_past_guards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _create_collection("kb")

    class _Reached(RuntimeError):
        pass

    def _reached(*args: object, **kwargs: object) -> None:
        raise _Reached

    monkeypatch.setattr(document_ingestion, "ProgressManager", _reached)
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
