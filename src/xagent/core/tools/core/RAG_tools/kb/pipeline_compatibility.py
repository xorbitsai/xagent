"""Pipeline compatibility facade."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Callable, Mapping, Optional, cast

from ..core.schemas import (
    IngestionConfig,
    IngestionResult,
    IngestionStepResult,
    SearchConfig,
    SearchPipelineResult,
    WebCrawlConfig,
    WebIngestionResult,
)
from .models import KBStorageBackend
from .operation_compatibility import (
    KBOperation,
    KBOperationCompatibilityFacade,
    KBOperationOutcome,
    PersistencePolicy,
    SideEffectPlane,
    finish_ingestion_outcome,
    finish_web_ingestion_outcome,
    record_document_ingestion_side_effects,
    should_finish_document_ingestion_operation,
    step_metadata,
)

if TYPE_CHECKING:
    from ..core.schemas import CollectionInfo
    from .coordinator import KBCoordinator
    from .models import RollbackFailedIngestionRequest, RollbackFailedIngestionResult
    from .storage_shim import KBStorageShimCompatibilityFacade

KB_STORAGE_METADATA_KEY = "kb_storage"


class KBPipelineCompatibilityFacade:
    """Compatibility boundary for high-level KB pipeline entry points.

    Pipeline modules keep their historical import paths and response contracts.
    The facade centralizes coordinator-owned storage binding and collection
    backend binding while delegating parser, chunker, embedding, crawler,
    progress, and rerank behavior to the existing pipeline implementations.
    """

    def __init__(
        self,
        coordinator: KBCoordinator | None = None,
        storage_shim: KBStorageShimCompatibilityFacade | None = None,
        operation_compatibility: KBOperationCompatibilityFacade | None = None,
    ) -> None:
        self._coordinator = coordinator
        self._storage_shim = storage_shim
        self._operation_compatibility = operation_compatibility
        # Lazily-built coordinator bound to an injected shim (see
        # _active_coordinator); cached so repeated calls reuse one instance.
        self._shim_coordinator: KBCoordinator | None = None

    def _active_storage_shim(self) -> KBStorageShimCompatibilityFacade | None:
        if self._storage_shim is not None:
            return self._storage_shim
        if self._coordinator is not None:
            return self._coordinator.storage_shim
        return None

    def _active_operation_facade(self) -> KBOperationCompatibilityFacade | None:
        if self._operation_compatibility is not None:
            return self._operation_compatibility
        if self._coordinator is not None:
            return self._coordinator.operation_compatibility
        return None

    def _active_coordinator(self) -> KBCoordinator:
        if self._coordinator is not None:
            return self._coordinator

        from .coordinator import KBCoordinator, get_kb_coordinator

        # An injected shim without a coordinator must stay bound to that shim
        # (mirrors the legacy-step facade pattern, legacy_step_compatibility
        # _active_coordinator).
        if self._storage_shim is not None:
            if self._shim_coordinator is None:
                self._shim_coordinator = KBCoordinator(storage_shim=self._storage_shim)
            return self._shim_coordinator

        return get_kb_coordinator()

    def rollback_failed_ingestion_sync(
        self, request: "RollbackFailedIngestionRequest"
    ) -> "RollbackFailedIngestionResult":
        """Delegate failed-ingest rollback orchestration to the coordinator."""
        return self._active_coordinator().rollback_failed_ingestion_sync(request)

    @contextmanager
    def _storage_context(self) -> Iterator[None]:
        storage_shim = self._active_storage_shim()
        if storage_shim is None:
            yield
            return

        from ..storage.factory import bind_storage_shim_for_current_context

        with bind_storage_shim_for_current_context(storage_shim):
            yield

    @contextmanager
    def _operation_context(
        self,
        *,
        operation_type: str,
        collection: str,
        persistence_policy: PersistencePolicy = (
            PersistencePolicy.PRESERVE_SUCCESSFUL_CHILDREN
        ),
        details: Optional[Mapping[str, Any]] = None,
    ) -> Iterator[KBOperation | None]:
        operation_facade = self._active_operation_facade()
        if operation_facade is None:
            yield None
            return

        current_operation = operation_facade.current_operation()
        if current_operation is not None:
            if (
                current_operation.operation_type == "web_ingestion"
                and operation_type == "document_ingestion"
            ):
                with operation_facade.start_child_operation(
                    operation_type="web_page_ingestion",
                    collection=collection,
                    persistence_policy=persistence_policy,
                    details=details,
                ) as child_operation:
                    yield child_operation
                return

            yield current_operation
            return

        with operation_facade.start_operation(
            operation_type=operation_type,
            collection=collection,
            persistence_policy=persistence_policy,
            details=details,
        ) as operation:
            yield operation

    async def ensure_collection_backend_binding_async(
        self, collection: str
    ) -> CollectionInfo | None:
        """Ensure direct pipeline-created collections carry a backend binding."""
        storage_shim = self._active_storage_shim()
        if storage_shim is None:
            return None

        metadata_store = storage_shim.get_metadata_store()
        try:
            collection_info = await metadata_store.get_collection(collection)
        except ValueError:
            return None

        extra_metadata = dict(collection_info.extra_metadata or {})
        if extra_metadata.get(KB_STORAGE_METADATA_KEY) is not None:
            return collection_info

        extra_metadata[KB_STORAGE_METADATA_KEY] = {
            "backend": KBStorageBackend.LANCEDB.value
        }
        updated_collection = collection_info.model_copy(
            update={"extra_metadata": extra_metadata}
        )
        await metadata_store.save_collection(updated_collection)
        return updated_collection

    def ensure_collection_backend_binding(
        self, collection: str
    ) -> CollectionInfo | None:
        """Ensure direct pipeline-created collections carry a backend binding."""
        # Compatibility wrapper for sync document ingestion paths. Async
        # callers should await ensure_collection_backend_binding_async().
        from .coordinator import _run_in_separate_loop

        return _run_in_separate_loop(
            self.ensure_collection_backend_binding_async(collection)
        )

    def process_document(
        self,
        collection: str,
        source_path: str,
        *,
        config: Optional[IngestionConfig] = None,
        progress_manager: Optional[Any] = None,
        user_id: Optional[int] = None,
        is_admin: bool = False,
        file_id: Optional[str] = None,
        metadata_source_path: Optional[str] = None,
        commit_gate: Optional[Callable[[], None]] = None,
    ) -> IngestionResult:
        from ..pipelines.document_ingestion import _process_document_impl

        with self._operation_context(
            operation_type="document_ingestion",
            collection=collection,
            details={"source_path": source_path, "file_id": file_id},
        ) as operation:
            with self._storage_context():
                result = _process_document_impl(
                    collection=collection,
                    source_path=source_path,
                    config=config,
                    progress_manager=progress_manager,
                    user_id=user_id,
                    is_admin=is_admin,
                    file_id=file_id,
                    metadata_source_path=metadata_source_path,
                    commit_gate=commit_gate,
                )
                self._record_document_ingestion_side_effects(
                    operation,
                    result,
                    collection=collection,
                    source_path=source_path,
                    file_id=file_id,
                    user_id=user_id,
                    is_admin=is_admin,
                )
                self.ensure_collection_backend_binding(collection)
                if should_finish_document_ingestion_operation(operation):
                    finish_ingestion_outcome(
                        operation, status=result.status, message=result.message
                    )
                return result

    def run_document_ingestion(
        self,
        collection: str,
        source_path: str,
        *,
        ingestion_config: Optional[Any] = None,
        progress_manager: Optional[Any] = None,
        user_id: Optional[int] = None,
        is_admin: Optional[bool] = None,
        file_id: Optional[str] = None,
        metadata_source_path: Optional[str] = None,
        commit_gate: Optional[Callable[[], None]] = None,
    ) -> IngestionResult:
        from ..pipelines.document_ingestion import _run_document_ingestion_impl

        with self._operation_context(
            operation_type="document_ingestion",
            collection=collection,
            details={"source_path": source_path, "file_id": file_id},
        ) as operation:
            with self._storage_context():
                result: object = _run_document_ingestion_impl(
                    collection=collection,
                    source_path=source_path,
                    ingestion_config=ingestion_config,
                    progress_manager=progress_manager,
                    user_id=user_id,
                    is_admin=is_admin,
                    file_id=file_id,
                    metadata_source_path=metadata_source_path,
                    commit_gate=commit_gate,
                )
                # Phase-1 compatibility: _run_document_ingestion_impl() still calls
                # the public process_document symbol, which legacy tests/callers may
                # monkeypatch. Only structured results carry rollback metadata.
                if not isinstance(result, IngestionResult):
                    return cast(IngestionResult, result)
                if operation is not None and operation.outcome is None:
                    self._record_document_ingestion_side_effects(
                        operation,
                        result,
                        collection=collection,
                        source_path=source_path,
                        file_id=file_id,
                        user_id=user_id,
                        is_admin=is_admin,
                    )
                    self.ensure_collection_backend_binding(collection)
                    if should_finish_document_ingestion_operation(operation):
                        finish_ingestion_outcome(
                            operation, status=result.status, message=result.message
                        )
                elif operation is None:
                    self.ensure_collection_backend_binding(collection)
                return result

    def search_documents(
        self,
        collection: str,
        query_text: str,
        *,
        config: Optional[SearchConfig] = None,
        progress_manager: Optional[Any] = None,
        user_id: Optional[int] = None,
        is_admin: Optional[bool] = None,
    ) -> SearchPipelineResult:
        from ..pipelines.document_search import _search_documents_impl

        with self._storage_context():
            return _search_documents_impl(
                collection=collection,
                query_text=query_text,
                config=config,
                progress_manager=progress_manager,
                user_id=user_id,
                is_admin=is_admin,
            )

    def run_document_search(
        self,
        collection: str,
        query_text: str,
        *,
        config: Optional[SearchConfig | Mapping[str, Any]] = None,
        progress_manager: Optional[Any] = None,
        user_id: Optional[int] = None,
        is_admin: Optional[bool] = None,
    ) -> SearchPipelineResult:
        from ..pipelines.document_search import _run_document_search_impl

        with self._storage_context():
            return _run_document_search_impl(
                collection=collection,
                query_text=query_text,
                config=config,
                progress_manager=progress_manager,
                user_id=user_id,
                is_admin=is_admin,
            )

    async def run_web_ingestion(
        self,
        collection: str,
        crawl_config: WebCrawlConfig,
        *,
        ingestion_config: Optional[IngestionConfig] = None,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
        user_id: Optional[int] = None,
        is_admin: Optional[bool] = None,
        file_handler: Optional[Callable[..., Any]] = None,
    ) -> WebIngestionResult:
        from ..pipelines.web_ingestion import _run_web_ingestion_impl

        with self._operation_context(
            operation_type="web_ingestion",
            collection=collection,
            persistence_policy=PersistencePolicy.PRESERVE_SUCCESSFUL_CHILDREN,
            details={"start_url": crawl_config.start_url},
        ) as operation:
            with self._storage_context():
                result = await _run_web_ingestion_impl(
                    collection=collection,
                    crawl_config=crawl_config,
                    ingestion_config=ingestion_config,
                    progress_callback=progress_callback,
                    user_id=user_id,
                    is_admin=is_admin,
                    file_handler=file_handler,
                    pipeline_facade=self,
                )
                await self.ensure_collection_backend_binding_async(collection)
                outcome = finish_web_ingestion_outcome(
                    operation,
                    status=result.status,
                    documents_created=result.documents_created,
                    pages_crawled=result.pages_crawled,
                    pages_failed=result.pages_failed,
                    failed_urls=result.failed_urls,
                    message=result.message,
                )
                if (
                    outcome is not None
                    and result.side_effects_may_remain
                    != outcome.side_effects_may_remain
                ):
                    result = result.model_copy(
                        update={
                            "side_effects_may_remain": outcome.side_effects_may_remain
                        }
                    )
                return result

    @contextmanager
    def web_page_operation(
        self,
        *,
        collection: str,
        url: str,
        title: Optional[str] = None,
    ) -> Iterator[KBOperation | None]:
        operation_facade = self._active_operation_facade()
        if operation_facade is None:
            yield None
            return

        current_operation = operation_facade.current_operation()
        if current_operation is None:
            yield None
            return

        if current_operation.operation_type == "web_ingestion":
            with operation_facade.start_child_operation(
                operation_type="web_page_ingestion",
                collection=collection,
                persistence_policy=PersistencePolicy.PRESERVE_SUCCESSFUL_CHILDREN,
                details={"url": url, "title": title},
            ) as child_operation:
                yield child_operation
            return

        yield current_operation

    def record_web_page_file_side_effect(
        self,
        operation: KBOperation | None,
        *,
        collection: str,
        url: str,
        file_path: Optional[str],
        file_id: Optional[str],
        reason: str = "file_handler",
        extra_payload: Optional[Mapping[str, Any]] = None,
        compensation: Optional[Callable[[], None]] = None,
    ) -> None:
        if operation is None:
            return
        payload = {
            "collection": collection,
            "url": url,
            "file_path": file_path,
            "file_id": file_id,
            "reason": reason,
        }
        if extra_payload:
            payload.update(dict(extra_payload))
        operation.record_side_effect(
            name="cleanup_web_page_persistence",
            plane=SideEffectPlane.FILE,
            payload=payload,
            idempotency_key=f"file:{collection}:{file_id or file_path or url}",
            compensation=compensation,
        )

    @staticmethod
    def compensate_web_page_file_side_effect(
        operation: KBOperation | None,
    ) -> tuple[BaseException, ...]:
        """Execute registered web-file compensation callbacks for a page."""
        if operation is None or operation.outcome is not None:
            return ()
        return operation.execute_compensations(
            step_names={"cleanup_web_page_persistence"},
            planes={SideEffectPlane.FILE},
        )

    @staticmethod
    def finish_web_page_operation(
        operation: KBOperation | None,
        *,
        status: str,
        message: str,
        side_effects_may_remain: Optional[bool] = None,
    ) -> None:
        if operation is None or operation.outcome is not None:
            return
        if side_effects_may_remain is None:
            side_effects_may_remain = (
                status != "success" and operation.has_side_effects()
            )
        operation.finish(
            status=status,
            rollback_status=operation.infer_rollback_status(
                status,
                side_effects_may_remain=side_effects_may_remain,
            ),
            side_effects_may_remain=side_effects_may_remain,
            details={"message": message},
        )

    def _record_document_ingestion_side_effects(
        self,
        operation: KBOperation | None,
        result: IngestionResult,
        *,
        collection: str,
        source_path: str,
        file_id: Optional[str],
        user_id: Optional[int],
        is_admin: Optional[bool],
    ) -> None:
        record_document_ingestion_side_effects(
            operation,
            collection=collection,
            doc_id=result.doc_id,
            parse_hash=result.parse_hash,
            completed_steps=result.completed_steps,
            chunk_count=result.chunk_count,
            vector_count=result.vector_count,
            failed_step=result.failed_step,
            source_path=source_path,
            file_id=file_id,
            user_id=user_id,
            is_admin=is_admin,
        )

    @staticmethod
    def _step_metadata(
        completed_steps: list[IngestionStepResult],
        name: str,
    ) -> dict[str, Any] | None:
        return step_metadata(completed_steps, name)

    @staticmethod
    def _finish_document_ingestion_outcome(
        operation: KBOperation | None,
        result: IngestionResult,
    ) -> None:
        # Thin delegator kept for an existing test's monkeypatch target (#515).
        finish_ingestion_outcome(
            operation, status=result.status, message=result.message
        )

    @staticmethod
    def _record_web_ingestion_outcome(
        operation: KBOperation | None,
        result: WebIngestionResult,
    ) -> KBOperationOutcome | None:
        # Thin delegator kept for an existing test's direct call (#515).
        return finish_web_ingestion_outcome(
            operation,
            status=result.status,
            documents_created=result.documents_created,
            pages_crawled=result.pages_crawled,
            pages_failed=result.pages_failed,
            failed_urls=result.failed_urls,
            message=result.message,
        )
