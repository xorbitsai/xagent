"""Tests for the KB API compatibility facade."""

from __future__ import annotations

import inspect
from typing import Optional

import pytest

from xagent.core.tools.core.RAG_tools.core.schemas import (
    CollectionInfo,
    IngestionResult,
    WebCrawlConfig,
    WebIngestionResult,
)
from xagent.core.tools.core.RAG_tools.kb import (
    CompensationStep,
    KBApiCompatibilityFacade,
    KBApiOperationResult,
    KBCoordinator,
    KBOperationCompatibilityFacade,
    KBOperationOutcome,
    PersistencePolicy,
    RollbackStatus,
    SideEffectPlane,
    get_kb_coordinator,
    reset_kb_coordinator_for_tests,
)


class _FakeMetadataStore:
    def __init__(self, collection: Optional[CollectionInfo]) -> None:
        self.collection = collection
        self.saved_configs: list[tuple[str, str, int]] = []
        self.saved_collections: list[CollectionInfo] = []

    async def get_collection(self, collection: str) -> CollectionInfo:
        if self.collection is None or self.collection.name != collection:
            raise ValueError(f"Collection {collection!r} not found")
        return self.collection

    async def save_collection(self, collection: CollectionInfo) -> None:
        self.saved_collections.append(collection)
        self.collection = collection

    async def save_collection_config(
        self,
        *,
        collection: str,
        config_json: str,
        user_id: int,
    ) -> None:
        self.saved_configs.append((collection, config_json, user_id))


class _ConfigOnlyMetadataStore:
    def __init__(self) -> None:
        self.saved_configs: list[tuple[str, str, int]] = []

    async def save_collection_config(
        self,
        *,
        collection: str,
        config_json: str,
        user_id: int,
    ) -> None:
        self.saved_configs.append((collection, config_json, user_id))


class _NoneReturningMetadataStore(_FakeMetadataStore):
    async def get_collection(self, collection: str) -> CollectionInfo | None:
        return None


class _FakeStorageShim:
    def __init__(
        self,
        metadata_store: object,
        vector_store: object | None = None,
        status_store: object | None = None,
    ) -> None:
        self.metadata_store = metadata_store
        self.vector_store = vector_store
        self.status_store = status_store

    def get_metadata_store(self) -> object:
        return self.metadata_store

    def get_vector_index_store(self) -> object:
        if self.vector_store is None:
            raise AssertionError("vector store was not configured")
        return self.vector_store

    def get_ingestion_status_store(self) -> object:
        if self.status_store is None:
            raise AssertionError("status store was not configured")
        return self.status_store


def test_kb_api_facade_public_surface_imports() -> None:
    import xagent.core.tools.core.RAG_tools.kb as kb

    assert hasattr(kb, "KBApiCompatibilityFacade")
    reset_kb_coordinator_for_tests()
    assert isinstance(get_kb_coordinator().api_compatibility, KBApiCompatibilityFacade)
    assert get_kb_coordinator().api is get_kb_coordinator().api_compatibility


@pytest.mark.asyncio
async def test_save_collection_config_creates_owner_neutral_backend_binding() -> None:
    metadata_store = _FakeMetadataStore(None)
    facade = KBApiCompatibilityFacade(storage_shim=_FakeStorageShim(metadata_store))

    await facade.save_collection_config(
        collection="demo",
        config_json="{}",
        user_id=7,
    )

    assert metadata_store.saved_configs == [("demo", "{}", 7)]
    assert metadata_store.saved_collections == [metadata_store.collection]
    assert metadata_store.collection is not None
    assert metadata_store.collection.owners == []
    assert metadata_store.collection.extra_metadata["kb_storage"] == {
        "backend": "lancedb"
    }


@pytest.mark.asyncio
async def test_save_collection_config_preserves_existing_backend_binding() -> None:
    existing = CollectionInfo(
        name="demo",
        extra_metadata={"kb_storage": {"backend": "postgresql"}, "other": "kept"},
    )
    metadata_store = _FakeMetadataStore(existing)
    facade = KBApiCompatibilityFacade(storage_shim=_FakeStorageShim(metadata_store))

    await facade.save_collection_config(
        collection="demo",
        config_json='{"chunk_size": 1000}',
        user_id=7,
    )

    assert metadata_store.saved_configs == [("demo", '{"chunk_size": 1000}', 7)]
    assert existing.extra_metadata["kb_storage"] == {"backend": "postgresql"}
    assert existing.extra_metadata["other"] == "kept"
    assert metadata_store.saved_collections == []


@pytest.mark.asyncio
async def test_save_collection_config_creates_backend_binding_when_store_returns_none() -> (
    None
):
    metadata_store = _NoneReturningMetadataStore(None)
    facade = KBApiCompatibilityFacade(storage_shim=_FakeStorageShim(metadata_store))

    await facade.save_collection_config(
        collection="demo",
        config_json="{}",
        user_id=7,
    )

    assert metadata_store.saved_collections == [metadata_store.collection]
    assert metadata_store.collection is not None
    assert metadata_store.collection.name == "demo"
    assert metadata_store.collection.extra_metadata["kb_storage"] == {
        "backend": "lancedb"
    }


@pytest.mark.asyncio
async def test_save_collection_config_uses_proxy_store_methods() -> None:
    metadata_store = _FakeMetadataStore(None)

    class MetadataStoreProxy:
        def __getattr__(self, name: str) -> object:
            return getattr(metadata_store, name)

    facade = KBApiCompatibilityFacade(
        storage_shim=_FakeStorageShim(MetadataStoreProxy())
    )

    await facade.save_collection_config(
        collection="demo",
        config_json="{}",
        user_id=7,
    )

    assert metadata_store.collection is not None
    assert metadata_store.collection.extra_metadata["kb_storage"] == {
        "backend": "lancedb"
    }


@pytest.mark.asyncio
async def test_save_collection_config_tolerates_config_only_test_stores() -> None:
    metadata_store = _ConfigOnlyMetadataStore()
    facade = KBApiCompatibilityFacade(storage_shim=_FakeStorageShim(metadata_store))

    await facade.save_collection_config(
        collection="demo",
        config_json="{}",
        user_id=7,
    )

    assert metadata_store.saved_configs == [("demo", "{}", 7)]


def test_coordinator_accepts_injected_api_facade() -> None:
    facade = KBApiCompatibilityFacade()
    coordinator = KBCoordinator(api_compatibility=facade)

    assert coordinator.api_compatibility is facade
    assert coordinator.api is facade


def test_api_operation_result_consumes_new_operation_outcome() -> None:
    operation_facade = KBOperationCompatibilityFacade()
    coordinator = KBCoordinator(operation_compatibility=operation_facade)
    facade = coordinator.api_compatibility

    def operation() -> IngestionResult:
        with operation_facade.start_operation(
            operation_type="document_ingestion",
            collection="demo",
        ) as active_operation:
            active_operation.finish(
                status="error",
                rollback_status=RollbackStatus.INCOMPLETE,
                side_effects_may_remain=True,
            )
        return IngestionResult(status="error", message="failed")

    api_result = facade.run_with_operation_outcome(
        operation,
        operation_type="document_ingestion",
        collection="demo",
    )

    assert api_result.result.status == "error"
    assert api_result.operation_outcome is operation_facade.last_outcome
    cleanup_decision = facade.failed_ingest_cleanup_decision(api_result)
    assert cleanup_decision.successful_documents == 0
    assert cleanup_decision.side_effects_may_remain is True

    completed_rollback = facade.with_rollback_complete(api_result, True)
    cleanup_after_rollback = facade.failed_ingest_cleanup_decision(completed_rollback)
    assert cleanup_after_rollback.side_effects_may_remain is False
    assert completed_rollback.operation_outcome is not None
    assert (
        completed_rollback.operation_outcome.rollback_status is RollbackStatus.COMPLETE
    )
    assert completed_rollback.operation_outcome.side_effects_may_remain is False


def test_api_rollback_failure_updates_operation_outcome_incomplete() -> None:
    incomplete_outcome = KBOperationOutcome(
        operation_id="op-1",
        operation_type="document_ingestion",
        collection="demo",
        status="error",
        rollback_status=RollbackStatus.INCOMPLETE,
        persistence_policy=PersistencePolicy.PRESERVE_SUCCESSFUL_CHILDREN,
        side_effects_may_remain=True,
    )
    facade = KBApiCompatibilityFacade()
    api_result = KBApiOperationResult(
        result=IngestionResult(status="error", message="failed"),
        operation_outcome=incomplete_outcome,
    )

    failed_rollback = facade.with_rollback_complete(api_result, False)

    assert failed_rollback.rollback_complete is False
    assert failed_rollback.operation_outcome is not None
    assert (
        failed_rollback.operation_outcome.rollback_status is RollbackStatus.INCOMPLETE
    )
    assert failed_rollback.operation_outcome.side_effects_may_remain is True


def test_run_failed_ingest_rollback_marks_successful_compensation_complete() -> None:
    facade = KBApiCompatibilityFacade()
    incomplete_outcome = KBOperationOutcome(
        operation_id="op-1",
        operation_type="document_ingestion",
        collection="demo",
        status="error",
        rollback_status=RollbackStatus.INCOMPLETE,
        persistence_policy=PersistencePolicy.PRESERVE_SUCCESSFUL_CHILDREN,
        side_effects_may_remain=True,
    )
    api_result = KBApiOperationResult(
        result=IngestionResult(status="error", message="failed"),
        operation_outcome=incomplete_outcome,
    )

    rollback_result = facade.run_failed_ingest_rollback(api_result, lambda: None)

    assert rollback_result.rollback_complete is True
    assert rollback_result.error is None
    assert rollback_result.operation_result.rollback_complete is True
    assert rollback_result.operation_result.operation_outcome is not None
    assert (
        rollback_result.operation_result.operation_outcome.rollback_status
        is RollbackStatus.COMPLETE
    )
    assert (
        rollback_result.operation_result.operation_outcome.side_effects_may_remain
        is False
    )


def test_run_failed_ingest_rollback_marks_failed_compensation_incomplete() -> None:
    facade = KBApiCompatibilityFacade()
    outcome = KBOperationOutcome(
        operation_id="op-1",
        operation_type="document_ingestion",
        collection="demo",
        status="error",
        rollback_status=RollbackStatus.INCOMPLETE,
        persistence_policy=PersistencePolicy.PRESERVE_SUCCESSFUL_CHILDREN,
        side_effects_may_remain=True,
    )
    api_result = KBApiOperationResult(
        result=IngestionResult(status="error", message="failed"),
        operation_outcome=outcome,
    )
    error = RuntimeError("rollback failed")

    rollback_result = facade.run_failed_ingest_rollback(
        api_result,
        lambda: (_ for _ in ()).throw(error),
    )

    assert rollback_result.rollback_complete is False
    assert rollback_result.error is error
    assert rollback_result.operation_result.rollback_complete is False
    assert rollback_result.operation_result.operation_outcome is not None
    assert (
        rollback_result.operation_result.operation_outcome.rollback_status
        is RollbackStatus.INCOMPLETE
    )
    assert (
        rollback_result.operation_result.operation_outcome.side_effects_may_remain
        is True
    )


def test_run_failed_ingest_rollback_closes_awaitable_result(recwarn) -> None:
    facade = KBApiCompatibilityFacade()
    outcome = KBOperationOutcome(
        operation_id="op-1",
        operation_type="document_ingestion",
        collection="demo",
        status="error",
        rollback_status=RollbackStatus.INCOMPLETE,
        persistence_policy=PersistencePolicy.PRESERVE_SUCCESSFUL_CHILDREN,
        side_effects_may_remain=True,
    )
    api_result = KBApiOperationResult(
        result=IngestionResult(status="error", message="failed"),
        operation_outcome=outcome,
    )

    async def rollback() -> None:
        return None

    rollback_result = facade.run_failed_ingest_rollback(api_result, rollback)

    assert rollback_result.rollback_complete is False
    assert isinstance(rollback_result.error, TypeError)
    assert "run_failed_ingest_rollback_async" in str(rollback_result.error)
    assert not any("was never awaited" in str(item.message) for item in recwarn)


def test_run_failed_ingest_rollback_can_compensate_successful_operation() -> None:
    facade = KBApiCompatibilityFacade()
    outcome = KBOperationOutcome(
        operation_id="op-1",
        operation_type="document_ingestion",
        collection="demo",
        status="success",
        rollback_status=RollbackStatus.NOT_NEEDED,
        persistence_policy=PersistencePolicy.PRESERVE_SUCCESSFUL_CHILDREN,
        compensation_steps=(
            CompensationStep(
                name="remove_registered_document",
                plane=SideEffectPlane.DOCUMENT,
            ),
        ),
    )
    api_result = KBApiOperationResult(
        result=IngestionResult(status="success", message="ok"),
        operation_outcome=outcome,
    )

    rollback_result = facade.run_failed_ingest_rollback(api_result, lambda: None)

    assert rollback_result.operation_result.rollback_complete is True
    assert rollback_result.operation_result.operation_outcome is not None
    assert (
        rollback_result.operation_result.operation_outcome.rollback_status
        is RollbackStatus.COMPLETE
    )


def test_run_with_operation_outcome_rebinds_storage_context() -> None:
    from xagent.core.tools.core.RAG_tools.storage.factory import (
        bind_storage_shim_for_current_context,
        get_bound_storage_shim_for_current_context,
    )

    outer_shim = _FakeStorageShim(_ConfigOnlyMetadataStore())
    inner_shim = _FakeStorageShim(_ConfigOnlyMetadataStore())
    facade = KBApiCompatibilityFacade(storage_shim=inner_shim)
    seen_shims: list[object | None] = []

    def operation() -> IngestionResult:
        seen_shims.append(get_bound_storage_shim_for_current_context())
        return IngestionResult(status="success", message="ok")

    with bind_storage_shim_for_current_context(outer_shim):
        api_result = facade.run_with_operation_outcome(
            operation,
            operation_type="document_ingestion",
            collection="demo",
        )
        assert get_bound_storage_shim_for_current_context() is outer_shim

    assert api_result.result.status == "success"
    assert seen_shims == [inner_shim]


@pytest.mark.asyncio
async def test_run_async_with_operation_outcome_rebinds_storage_context() -> None:
    from xagent.core.tools.core.RAG_tools.storage.factory import (
        bind_storage_shim_for_current_context,
        get_bound_storage_shim_for_current_context,
    )

    outer_shim = _FakeStorageShim(_ConfigOnlyMetadataStore())
    inner_shim = _FakeStorageShim(_ConfigOnlyMetadataStore())
    facade = KBApiCompatibilityFacade(storage_shim=inner_shim)
    seen_shims: list[object | None] = []

    async def operation() -> WebIngestionResult:
        seen_shims.append(get_bound_storage_shim_for_current_context())
        return WebIngestionResult(
            status="success",
            collection="demo",
            total_urls_found=0,
            pages_crawled=0,
            pages_failed=0,
            documents_created=0,
            chunks_created=0,
            embeddings_created=0,
            crawled_urls=[],
            failed_urls={},
            message="ok",
            warnings=[],
            elapsed_time_ms=0,
        )

    with bind_storage_shim_for_current_context(outer_shim):
        api_result = await facade.run_async_with_operation_outcome(
            operation,
            operation_type="web_ingestion",
            collection="demo",
        )
        assert get_bound_storage_shim_for_current_context() is outer_shim

    assert api_result.result.status == "success"
    assert seen_shims == [inner_shim]


@pytest.mark.asyncio
async def test_api_facade_storage_operations_rebind_storage_context() -> None:
    from xagent.core.tools.core.RAG_tools.storage.factory import (
        bind_storage_shim_for_current_context,
        get_bound_storage_shim_for_current_context,
    )

    class VectorStore:
        def __init__(self) -> None:
            self.list_calls: list[dict[str, object]] = []
            self.rename_calls: list[dict[str, object]] = []

        def list_document_records(self, **kwargs: object) -> list[str]:
            self.list_calls.append(kwargs)
            return ["record"]

        def rename_collection_data(self, **kwargs: object) -> list[str]:
            self.rename_calls.append(kwargs)
            return ["vector warning"]

    class MetadataStore(_FakeMetadataStore):
        def __init__(self) -> None:
            super().__init__(CollectionInfo(name="old"))
            self.loaded_configs: list[dict[str, object]] = []
            self.deleted_metadata: list[dict[str, object]] = []
            self.deleted_entries: list[str] = []
            self.renamed: list[dict[str, object]] = []
            self.config_owner_ids = {7, 8}

        async def get_collection_config(
            self,
            *,
            collection: str,
            user_id: int | None,
            is_admin: bool = False,
        ) -> str | None:
            self.loaded_configs.append(
                {"collection": collection, "user_id": user_id, "is_admin": is_admin}
            )
            return "{}"

        async def delete_collection_metadata(self, **kwargs: object) -> dict[str, int]:
            self.deleted_metadata.append(kwargs)
            return {"collection_config": 1}

        async def delete_collection(self, collection_name: str) -> None:
            self.deleted_entries.append(collection_name)

        def list_collection_config_owner_ids(self, collection_name: str) -> set[int]:
            assert collection_name == "old"
            return self.config_owner_ids

        async def rename_collection(self, **kwargs: object) -> None:
            self.renamed.append(kwargs)

    class StatusStore:
        def __init__(self) -> None:
            self.renamed: list[dict[str, object]] = []

        def rename_collection_status(self, **kwargs: object) -> list[str]:
            self.renamed.append(kwargs)
            return ["status warning"]

    outer_metadata = MetadataStore()
    outer_vector = VectorStore()
    outer_status = StatusStore()
    inner_metadata = MetadataStore()
    inner_vector = VectorStore()
    inner_status = StatusStore()
    outer_shim = _FakeStorageShim(outer_metadata, outer_vector, outer_status)
    inner_shim = _FakeStorageShim(inner_metadata, inner_vector, inner_status)
    facade = KBApiCompatibilityFacade(storage_shim=inner_shim)

    with bind_storage_shim_for_current_context(outer_shim):
        assert facade.list_document_records(
            collection_name="old",
            user_id=7,
            is_admin=False,
        ) == ["record"]
        await facade.save_collection_config(
            collection="old",
            config_json="{}",
            user_id=7,
        )
        assert (
            await facade.get_collection_config(
                collection="old",
                user_id=7,
                is_admin=False,
            )
            == "{}"
        )
        await facade.delete_collection_metadata(
            collection_name="old",
            user_id=7,
            is_admin=False,
            delete_orphaned_metadata=True,
        )
        assert await facade.delete_collection_metadata_entry("old") is True
        assert facade.list_collection_config_owner_ids("old") == {7, 8}
        assert await facade.rename_collection_data(
            collection_name="old",
            new_name="new",
            user_id=7,
            is_admin=False,
        ) == ["vector warning"]
        await facade.rename_collection_metadata(
            old_name="old",
            new_name="new",
            user_id=7,
            is_admin=False,
        )
        assert facade.rename_collection_status(
            old_name="old",
            new_name="new",
            user_id=7,
            is_admin=False,
        ) == ["status warning"]
        assert get_bound_storage_shim_for_current_context() is outer_shim

    assert outer_vector.list_calls == []
    assert outer_vector.rename_calls == []
    assert outer_metadata.saved_configs == []
    assert outer_metadata.loaded_configs == []
    assert outer_metadata.deleted_metadata == []
    assert outer_metadata.deleted_entries == []
    assert outer_metadata.renamed == []
    assert outer_status.renamed == []

    assert inner_vector.list_calls == [
        {"collection_name": "old", "user_id": 7, "is_admin": False}
    ]
    assert inner_vector.rename_calls == [
        {
            "collection_name": "old",
            "new_name": "new",
            "user_id": 7,
            "is_admin": False,
        }
    ]
    assert inner_metadata.saved_configs == [("old", "{}", 7)]
    assert inner_metadata.loaded_configs == [
        {"collection": "old", "user_id": 7, "is_admin": False}
    ]
    assert inner_metadata.deleted_metadata == [
        {
            "collection_name": "old",
            "user_id": 7,
            "is_admin": False,
            "delete_orphaned_metadata": True,
        }
    ]
    assert inner_metadata.deleted_entries == ["old"]
    assert inner_metadata.renamed == [
        {"old_name": "old", "new_name": "new", "user_id": 7, "is_admin": False}
    ]
    assert inner_status.renamed == [
        {"old_name": "old", "new_name": "new", "user_id": 7, "is_admin": False}
    ]


def test_api_operation_result_ignores_stale_operation_outcome() -> None:
    operation_facade = KBOperationCompatibilityFacade()
    coordinator = KBCoordinator(operation_compatibility=operation_facade)
    facade = coordinator.api_compatibility

    with operation_facade.start_operation(
        operation_type="document_ingestion",
        collection="demo",
    ) as active_operation:
        active_operation.finish(status="success")
    assert operation_facade.last_outcome is not None

    api_result = facade.run_with_operation_outcome(
        lambda: IngestionResult(status="error", message="patched failure"),
        operation_type="document_ingestion",
        collection="demo",
    )

    assert api_result.operation_outcome is None
    cleanup_decision = facade.failed_ingest_cleanup_decision(api_result)
    assert cleanup_decision.side_effects_may_remain is False


def test_api_operation_result_ignores_equal_stale_operation_outcome_copy() -> None:
    operation_facade = KBOperationCompatibilityFacade()
    coordinator = KBCoordinator(operation_compatibility=operation_facade)
    facade = coordinator.api_compatibility

    with operation_facade.start_operation(
        operation_type="document_ingestion",
        collection="demo",
    ) as active_operation:
        active_operation.finish(status="success")
    assert operation_facade.last_outcome is not None

    copied_previous_outcome = KBOperationOutcome(
        operation_id=operation_facade.last_outcome.operation_id,
        operation_type=operation_facade.last_outcome.operation_type,
        collection=operation_facade.last_outcome.collection,
        status=operation_facade.last_outcome.status,
        rollback_status=operation_facade.last_outcome.rollback_status,
        persistence_policy=operation_facade.last_outcome.persistence_policy,
        compensation_steps=operation_facade.last_outcome.compensation_steps,
        child_outcomes=operation_facade.last_outcome.child_outcomes,
        warnings=operation_facade.last_outcome.warnings,
        side_effects_may_remain=operation_facade.last_outcome.side_effects_may_remain,
        details=operation_facade.last_outcome.details,
    )

    api_result = facade.wrap_operation_result(
        IngestionResult(status="error", message="patched failure"),
        previous_outcome=copied_previous_outcome,
        operation_type="document_ingestion",
        collection="demo",
    )

    assert api_result.operation_outcome is None


def test_failed_batch_ingest_cleanup_decision_aggregates_operation_outcomes() -> None:
    facade = KBApiCompatibilityFacade()
    side_effect_outcome = KBOperationOutcome(
        operation_id="op-1",
        operation_type="document_ingestion",
        collection="demo",
        status="error",
        rollback_status=RollbackStatus.INCOMPLETE,
        persistence_policy=PersistencePolicy.PRESERVE_SUCCESSFUL_CHILDREN,
        side_effects_may_remain=True,
    )
    clean_result = KBApiOperationResult(
        result=IngestionResult(status="error", message="rolled back"),
        rollback_complete=True,
    )
    dirty_result = KBApiOperationResult(
        result=IngestionResult(status="error", message="rollback failed"),
        operation_outcome=side_effect_outcome,
    )
    success_result = KBApiOperationResult(
        result=IngestionResult(status="success", message="ok")
    )

    cleanup_decision = facade.failed_batch_ingest_cleanup_decision(
        [clean_result, dirty_result, success_result]
    )

    assert cleanup_decision.successful_documents == 1
    assert cleanup_decision.side_effects_may_remain is True


def test_failed_ingest_cleanup_decision_accepts_dict_results() -> None:
    facade = KBApiCompatibilityFacade()

    cleanup_decision = facade.failed_ingest_cleanup_decision(
        KBApiOperationResult(
            result={
                "status": "partial",
                "documents_created": "2",
                "side_effects_may_remain": True,
            }
        )
    )

    assert cleanup_decision.successful_documents == 2
    assert cleanup_decision.side_effects_may_remain is True


def test_failed_batch_ingest_cleanup_decision_counts_dict_successes() -> None:
    facade = KBApiCompatibilityFacade()

    cleanup_decision = facade.failed_batch_ingest_cleanup_decision(
        [
            KBApiOperationResult(result={"status": "success"}),
            KBApiOperationResult(result={"status": "error"}),
        ]
    )

    assert cleanup_decision.successful_documents == 1
    assert cleanup_decision.side_effects_may_remain is False


def test_list_document_records_omits_none_max_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xagent.core.tools.core.RAG_tools.storage import factory

    calls: list[dict[str, object]] = []

    class VectorStore:
        def list_document_records(self, **kwargs: object) -> list[str]:
            calls.append(kwargs)
            return ["record"]

    monkeypatch.setattr(factory, "get_vector_index_store", lambda: VectorStore())

    records = KBApiCompatibilityFacade().list_document_records(
        collection_name="demo",
        user_id=7,
        is_admin=False,
    )

    assert records == ["record"]
    assert calls == [
        {
            "collection_name": "demo",
            "user_id": 7,
            "is_admin": False,
        }
    ]


def test_web_api_list_document_records_routes_through_api_facade(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xagent.web.api import kb as kb_api

    calls: list[dict[str, object]] = []

    class Facade:
        def list_document_records(self, **kwargs: object) -> list[str]:
            calls.append(kwargs)
            return ["record"]

    monkeypatch.setattr(
        kb_api,
        "_get_api_compatibility_facade",
        lambda: Facade(),
    )

    records = kb_api.list_document_records(
        collection_name="demo",
        user_id=7,
        is_admin=False,
        max_results=25,
    )

    assert records == ["record"]
    assert calls == [
        {
            "collection_name": "demo",
            "user_id": 7,
            "is_admin": False,
            "max_results": 25,
        }
    ]


def test_web_api_document_ingestion_outcome_keeps_legacy_runner_signature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xagent.web.api import kb as kb_api

    calls: list[dict[str, object]] = []

    def fake_run_document_ingestion(
        collection: str,
        source_path: str,
        *,
        ingestion_config: object,
        file_id: str | None = None,
        user_id: int | None = None,
        progress_manager: object | None = None,
        is_admin: bool = False,
    ) -> IngestionResult:
        calls.append(
            {
                "collection": collection,
                "source_path": source_path,
                "ingestion_config": ingestion_config,
                "file_id": file_id,
                "user_id": user_id,
                "progress_manager": progress_manager,
                "is_admin": is_admin,
            }
        )
        return IngestionResult(status="success", message="ok")

    config = object()
    monkeypatch.setattr(kb_api, "run_document_ingestion", fake_run_document_ingestion)

    result = kb_api.run_document_ingestion_with_outcome(
        collection="demo",
        source_path="/tmp/demo.txt",
        ingestion_config=config,
        user_id=7,
        is_admin=False,
        file_id="file-1",
    )

    assert result.result.status == "success"
    assert calls == [
        {
            "collection": "demo",
            "source_path": "/tmp/demo.txt",
            "ingestion_config": config,
            "file_id": "file-1",
            "user_id": 7,
            "progress_manager": None,
            "is_admin": False,
        }
    ]


@pytest.mark.asyncio
async def test_web_api_web_ingestion_outcome_keeps_legacy_runner_signature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xagent.web.api import kb as kb_api

    calls: list[dict[str, object]] = []

    async def fake_run_web_ingestion(
        collection: str,
        crawl_config: WebCrawlConfig,
        *,
        ingestion_config: object,
        user_id: int,
        is_admin: bool = False,
        file_handler: object | None = None,
    ) -> WebIngestionResult:
        calls.append(
            {
                "collection": collection,
                "crawl_config": crawl_config,
                "ingestion_config": ingestion_config,
                "user_id": user_id,
                "is_admin": is_admin,
                "file_handler": file_handler,
            }
        )
        return WebIngestionResult(
            status="success",
            collection=collection,
            total_urls_found=0,
            pages_crawled=0,
            pages_failed=0,
            documents_created=0,
            chunks_created=0,
            embeddings_created=0,
            message="ok",
            elapsed_time_ms=0,
        )

    crawl_config = WebCrawlConfig(start_url="https://example.com")
    config = object()
    monkeypatch.setattr(kb_api, "run_web_ingestion", fake_run_web_ingestion)

    result = await kb_api.run_web_ingestion_with_outcome(
        collection="web",
        crawl_config=crawl_config,
        ingestion_config=config,
        user_id=7,
        is_admin=True,
    )

    assert result.result.status == "success"
    assert calls == [
        {
            "collection": "web",
            "crawl_config": crawl_config,
            "ingestion_config": config,
            "user_id": 7,
            "is_admin": True,
            "file_handler": None,
        }
    ]


@pytest.mark.asyncio
async def test_rename_collection_routes_storage_metadata_and_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xagent.core.tools.core.RAG_tools.storage import factory

    calls: list[tuple[str, str, int, bool]] = []

    class VectorStore:
        def rename_collection_data(
            self,
            *,
            collection_name: str,
            new_name: str,
            user_id: int,
            is_admin: bool,
        ) -> list[str]:
            calls.append((collection_name, new_name, user_id, is_admin))
            return ["vector warning"]

    class MetadataStore:
        async def rename_collection(
            self,
            *,
            old_name: str,
            new_name: str,
            user_id: int,
            is_admin: bool,
        ) -> None:
            calls.append((old_name, new_name, user_id, is_admin))

    class StatusStore:
        def rename_collection_status(
            self,
            *,
            old_name: str,
            new_name: str,
            user_id: int,
            is_admin: bool,
        ) -> list[str]:
            calls.append((old_name, new_name, user_id, is_admin))
            return ["status warning"]

    monkeypatch.setattr(factory, "get_vector_index_store", lambda: VectorStore())
    monkeypatch.setattr(factory, "get_metadata_store", lambda: MetadataStore())
    monkeypatch.setattr(factory, "get_ingestion_status_store", lambda: StatusStore())

    facade = KBApiCompatibilityFacade()

    assert await facade.rename_collection_data(
        collection_name="old",
        new_name="new",
        user_id=7,
        is_admin=False,
    ) == ["vector warning"]
    await facade.rename_collection_metadata(
        old_name="old",
        new_name="new",
        user_id=7,
        is_admin=False,
    )
    assert facade.rename_collection_status(
        old_name="old",
        new_name="new",
        user_id=7,
        is_admin=False,
    ) == ["status warning"]
    assert calls == [
        ("old", "new", 7, False),
        ("old", "new", 7, False),
        ("old", "new", 7, False),
    ]


def test_web_api_search_wrapper_routes_through_api_facade(monkeypatch) -> None:
    from xagent.web.api import kb as kb_api

    sentinel = object()
    calls: list[tuple[str, str, int, bool]] = []

    class Facade:
        def run_document_search(
            self,
            collection: str,
            query_text: str,
            **kwargs: object,
        ) -> object:
            calls.append(
                (
                    collection,
                    query_text,
                    int(kwargs["user_id"]),
                    bool(kwargs["is_admin"]),
                )
            )
            return sentinel

    monkeypatch.setattr(
        kb_api,
        "_get_api_compatibility_facade",
        lambda: Facade(),
    )

    result = kb_api.run_document_search(
        collection="demo",
        query_text="question",
        user_id=7,
        is_admin=True,
    )

    assert result is sentinel
    assert calls == [("demo", "question", 7, True)]


def test_web_api_delete_document_wrapper_routes_through_api_facade(
    monkeypatch,
) -> None:
    from xagent.web.api import kb as kb_api

    sentinel = object()
    calls: list[tuple[str, str, int, bool]] = []

    class Facade:
        def delete_document(
            self,
            collection: str,
            doc_id: str,
            user_id: int,
            is_admin: bool,
        ) -> object:
            calls.append((collection, doc_id, user_id, is_admin))
            return sentinel

    monkeypatch.setattr(
        kb_api,
        "_get_api_compatibility_facade",
        lambda: Facade(),
    )

    result = kb_api.delete_document(
        collection="demo",
        doc_id="doc-1",
        user_id=7,
        is_admin=True,
    )

    assert result is sentinel
    assert calls == [("demo", "doc-1", 7, True)]


def test_delete_document_api_does_not_shadow_api_facade_wrapper() -> None:
    from xagent.web.api import kb as kb_api

    source = inspect.getsource(kb_api.delete_document_api)
    assert "management.collections import delete_document" not in source


def test_failed_ingest_cleanup_decision_uses_operation_outcome() -> None:
    """Cleanup decision reads from operation outcome, not opaque coverage metadata."""
    operation_facade = KBOperationCompatibilityFacade()
    coordinator = KBCoordinator(operation_compatibility=operation_facade)
    api_facade = KBApiCompatibilityFacade(coordinator=coordinator)

    with operation_facade.start_operation(
        operation_type="web_ingestion",
        collection="demo",
        persistence_policy=PersistencePolicy.PRESERVE_SUCCESSFUL_CHILDREN,
    ) as operation:
        operation.record_side_effect(
            name="test_side_effect",
            plane=SideEffectPlane.FILE,
            payload={"file_id": "f1"},
            idempotency_key="file:demo:f1",
        )
        operation.mark_compensated_steps(planes={SideEffectPlane.FILE})

        outcome = operation.finish(
            status="error",
            rollback_status=RollbackStatus.COMPLETE,
            side_effects_may_remain=False,
        )

    operation_result = KBApiOperationResult(
        result={"status": "error"},
        operation_outcome=outcome,
        rollback_complete=None,
    )
    decision = api_facade.failed_ingest_cleanup_decision(
        operation_result=operation_result,
    )

    # When no opaque rollback_complete flag is set, fall back to
    # operation_outcome.side_effects_may_remain
    assert decision.side_effects_may_remain is False
