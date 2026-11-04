"""API compatibility facade for KB route-facing operations."""

from __future__ import annotations

import inspect
import logging
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Callable,
    Generic,
    Optional,
    TypeVar,
    cast,
)

from ..core.schemas import (
    CollectionInfo,
    CollectionOperationResult,
    DocumentListResult,
    DocumentOperationResult,
    IngestionResult,
    ListCollectionsResult,
    SearchConfig,
    SearchPipelineResult,
    WebCrawlConfig,
    WebIngestionResult,
)
from .async_utils import maybe_await
from .collection_handle import deployment_kb_backend
from .operation_compatibility import (
    KBOperationOutcome,
    RollbackStatus,
    _close_awaitable_if_possible,
)
from .pipeline_compatibility import KB_STORAGE_METADATA_KEY

if TYPE_CHECKING:
    from ..storage.contracts import DocumentRecord
    from .coordinator import KBCoordinator
    from .operation_compatibility import KBOperationCompatibilityFacade
    from .storage_shim import KBStorageShimCompatibilityFacade

T_Result = TypeVar("T_Result")

logger = logging.getLogger(__name__)


def _has_store_method(metadata_store: object, name: str) -> bool:
    """Return True when the store exposes a callable method."""
    return callable(getattr(metadata_store, name, None))


@dataclass(frozen=True)
class KBApiOperationResult(Generic[T_Result]):
    """Route-internal result plus the rollback outcome produced by the coordinator."""

    result: T_Result
    operation_outcome: KBOperationOutcome | None = None
    rollback_complete: bool | None = None


@dataclass(frozen=True)
class KBApiFailedIngestCleanupDecision:
    """Cleanup policy for a failed ingest, from rollback outcomes or plain counts."""

    successful_documents: int = 0
    side_effects_may_remain: bool = False

    @property
    def keeps_new_collection_metadata(self) -> bool:
        """Whether a failed ingest must keep a new collection's metadata."""
        return self.successful_documents > 0 or self.side_effects_may_remain


@dataclass(frozen=True)
class KBApiFailedIngestRollbackResult(Generic[T_Result]):
    """Result of executing API-level rollback for a failed ingest operation."""

    operation_result: KBApiOperationResult[T_Result]
    error: Exception | None = None

    @property
    def rollback_complete(self) -> bool:
        """Return whether the rollback callback completed without an exception."""
        return self.error is None


class KBApiCompatibilityFacade:
    """Compatibility boundary for KB API route semantics.

    FastAPI request parsing, dependency handling, response wrappers, and HTTP
    error mapping stay in ``web.api.kb``. This facade owns the normalized KB
    operations that routes call after those API-layer concerns are resolved.
    """

    def __init__(
        self,
        coordinator: KBCoordinator | None = None,
        storage_shim: KBStorageShimCompatibilityFacade | None = None,
    ) -> None:
        self._coordinator = coordinator
        self._storage_shim = storage_shim

    def _active_storage_shim(self) -> KBStorageShimCompatibilityFacade | None:
        if self._storage_shim is not None:
            return self._storage_shim
        if self._coordinator is not None:
            return self._coordinator.storage_shim
        return None

    def _active_operation_facade(self) -> KBOperationCompatibilityFacade | None:
        if self._coordinator is not None:
            return self._coordinator.operation_compatibility
        return None

    def last_operation_outcome(self) -> KBOperationOutcome | None:
        """Return the last finalized coordinator operation in the current context."""
        operation_facade = self._active_operation_facade()
        if operation_facade is None:
            return None
        return operation_facade.last_outcome

    @contextmanager
    def _storage_context(self) -> Iterator[None]:
        storage_shim = self._active_storage_shim()
        if storage_shim is None:
            yield
            return

        from ..storage.factory import bind_storage_shim_for_current_context

        with bind_storage_shim_for_current_context(storage_shim):
            yield

    def wrap_operation_result(
        self,
        result: T_Result,
        *,
        previous_outcome: KBOperationOutcome | None = None,
        operation_type: str | tuple[str, ...] | None = None,
        collection: str | None = None,
        rollback_complete: bool | None = None,
    ) -> KBApiOperationResult[T_Result]:
        """Attach the current coordinator outcome to an API-compatible result."""
        outcome = self.last_operation_outcome()
        if outcome == previous_outcome:
            outcome = None
        if outcome is not None and operation_type is not None:
            expected = (
                (operation_type,) if isinstance(operation_type, str) else operation_type
            )
            if outcome.operation_type not in expected:
                outcome = None
        if (
            outcome is not None
            and collection is not None
            and outcome.collection != collection
        ):
            outcome = None

        return KBApiOperationResult(
            result=result,
            operation_outcome=outcome,
            rollback_complete=rollback_complete,
        )

    def run_with_operation_outcome(
        self,
        operation: Callable[[], T_Result],
        *,
        operation_type: str | tuple[str, ...],
        collection: str,
        rollback_complete: bool | None = None,
    ) -> KBApiOperationResult[T_Result]:
        """Run a sync API operation and consume its coordinator outcome."""
        previous_outcome = self.last_operation_outcome()
        with self._storage_context():
            result = operation()
        return self.wrap_operation_result(
            result,
            previous_outcome=previous_outcome,
            operation_type=operation_type,
            collection=collection,
            rollback_complete=rollback_complete,
        )

    async def run_async_with_operation_outcome(
        self,
        operation: Callable[[], Awaitable[T_Result]],
        *,
        operation_type: str | tuple[str, ...],
        collection: str,
        rollback_complete: bool | None = None,
    ) -> KBApiOperationResult[T_Result]:
        """Run an async API operation and consume its coordinator outcome."""
        previous_outcome = self.last_operation_outcome()
        with self._storage_context():
            result = await operation()
        return self.wrap_operation_result(
            result,
            previous_outcome=previous_outcome,
            operation_type=operation_type,
            collection=collection,
            rollback_complete=rollback_complete,
        )

    @staticmethod
    def with_result(
        operation_result: KBApiOperationResult[Any],
        result: T_Result,
    ) -> KBApiOperationResult[T_Result]:
        """Replace the legacy result while preserving internal operation metadata."""
        return KBApiOperationResult(
            result=result,
            operation_outcome=operation_result.operation_outcome,
            rollback_complete=operation_result.rollback_complete,
        )

    @staticmethod
    def with_rollback_complete(
        operation_result: KBApiOperationResult[T_Result],
        rollback_complete: bool,
        *,
        force: bool = False,
    ) -> KBApiOperationResult[T_Result]:
        """Record whether API-level compensation completed after a failed operation."""
        outcome = operation_result.operation_outcome
        if outcome is None or (outcome.status == "success" and not force):
            return replace(operation_result, rollback_complete=rollback_complete)

        rollback_status: RollbackStatus
        if rollback_complete:
            if outcome.rollback_status is RollbackStatus.SKIPPED_BY_POLICY:
                rollback_status = outcome.rollback_status
            else:
                rollback_status = (
                    RollbackStatus.COMPLETE
                    if (
                        outcome.compensation_steps
                        or outcome.child_outcomes
                        or outcome.rollback_status is RollbackStatus.INCOMPLETE
                    )
                    else outcome.rollback_status
                )
            side_effects_may_remain = False
        else:
            rollback_status = RollbackStatus.INCOMPLETE
            side_effects_may_remain = True

        return replace(
            operation_result,
            operation_outcome=replace(
                outcome,
                rollback_status=rollback_status,
                side_effects_may_remain=side_effects_may_remain,
            ),
            rollback_complete=rollback_complete,
        )

    def run_failed_ingest_rollback(
        self,
        operation_result: KBApiOperationResult[T_Result],
        rollback: Callable[[], Any],
    ) -> KBApiFailedIngestRollbackResult[T_Result]:
        """Execute sync failed-ingest rollback and update operation outcome state."""
        try:
            with self._storage_context():
                result = rollback()
                if inspect.isawaitable(result):
                    _close_awaitable_if_possible(result)
                    raise TypeError(
                        "run_failed_ingest_rollback_async must be used for "
                        "awaitable rollback callbacks"
                    )
        except Exception as exc:
            return KBApiFailedIngestRollbackResult(
                operation_result=self.with_rollback_complete(
                    operation_result,
                    False,
                    force=True,
                ),
                error=exc,
            )

        return KBApiFailedIngestRollbackResult(
            operation_result=self.with_rollback_complete(
                operation_result,
                True,
                force=True,
            )
        )

    async def run_failed_ingest_rollback_async(
        self,
        operation_result: KBApiOperationResult[T_Result],
        rollback: Callable[[], Any],
    ) -> KBApiFailedIngestRollbackResult[T_Result]:
        """Execute async failed-ingest rollback and update operation outcome state."""
        try:
            with self._storage_context():
                await maybe_await(rollback())
        except Exception as exc:
            return KBApiFailedIngestRollbackResult(
                operation_result=self.with_rollback_complete(
                    operation_result,
                    False,
                    force=True,
                ),
                error=exc,
            )

        return KBApiFailedIngestRollbackResult(
            operation_result=self.with_rollback_complete(
                operation_result,
                True,
                force=True,
            )
        )

    def failed_ingest_cleanup_decision(
        self,
        operation_result: KBApiOperationResult[Any],
        *,
        successful_documents: int | None = None,
        rollback_complete: bool | None = None,
    ) -> KBApiFailedIngestCleanupDecision:
        """Derive config/metadata cleanup inputs from operation rollback state."""
        effective_rollback_complete = (
            operation_result.rollback_complete
            if rollback_complete is None
            else rollback_complete
        )
        if successful_documents is None:
            successful_documents = self._legacy_successful_document_count(
                operation_result.result
            )

        return KBApiFailedIngestCleanupDecision(
            successful_documents=max(0, int(successful_documents)),
            side_effects_may_remain=self._side_effects_may_remain_after_api_rollback(
                operation_result,
                rollback_complete=effective_rollback_complete,
            ),
        )

    def failed_batch_ingest_cleanup_decision(
        self,
        operation_results: list[KBApiOperationResult[Any]],
        *,
        successful_documents: int | None = None,
    ) -> KBApiFailedIngestCleanupDecision:
        """Aggregate child operation rollback state for batch/cloud ingest cleanup."""
        if successful_documents is None:
            successful_documents = sum(
                self._legacy_successful_document_count(item.result)
                for item in operation_results
            )
        return KBApiFailedIngestCleanupDecision(
            successful_documents=max(0, int(successful_documents)),
            side_effects_may_remain=any(
                self.failed_ingest_cleanup_decision(item).side_effects_may_remain
                for item in operation_results
            ),
        )

    @staticmethod
    def _legacy_successful_document_count(result: Any) -> int:
        if isinstance(result, dict):
            documents_created = result.get("documents_created")
            status = result.get("status")
        else:
            documents_created = getattr(result, "documents_created", None)
            status = getattr(result, "status", None)
        if documents_created is not None:
            try:
                return int(documents_created)
            except (TypeError, ValueError):
                return 0
        return 1 if status == "success" else 0

    @staticmethod
    def _side_effects_may_remain_after_api_rollback(
        operation_result: KBApiOperationResult[Any],
        *,
        rollback_complete: bool | None,
    ) -> bool:
        if rollback_complete is False:
            return True
        if rollback_complete is True:
            return False

        outcome = operation_result.operation_outcome
        if outcome is not None:
            return bool(
                outcome.side_effects_may_remain
                or outcome.rollback_status is RollbackStatus.INCOMPLETE
            )

        result = operation_result.result
        if isinstance(result, dict):
            return bool(result.get("side_effects_may_remain", False))
        return bool(getattr(result, "side_effects_may_remain", False))

    def collection_holds_documents(
        self,
        *,
        collection_name: str,
        user_id: int,
        context: str,
        on_error: bool,
    ) -> bool:
        """Whether the collection currently holds any document.

        This is the single question behind both halves of the ingest lifecycle:
        a collection with documents must be published and must never be cleaned up,
        an empty one must be neither. Asking the store at decision time avoids
        trusting `collection_existed_before`, which is stamped when the request or
        job is submitted and goes stale while a sibling ingest runs.

        The two halves need opposite behaviour when the store cannot be read, so
        `on_error` is explicit: a cleanup passes `True` (unknown state must not
        authorize a delete), a publish passes `False` (unknown state must not make
        a possibly empty knowledge base visible).

        Reads are owner-scoped: every decision it gates acts on the caller's own
        rows, and an admin-wide read would match rows another tenant owns.

        Callers that roll back their own document must do so before asking.
        """
        try:
            records = self.list_document_records(
                collection_name=collection_name,
                user_id=user_id,
                is_admin=False,
                max_results=1,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to list documents of %s/user_%s during %s, assuming %s: %s",
                collection_name,
                user_id,
                context,
                "documents exist" if on_error else "no documents",
                exc,
            )
            return on_error
        return bool(records)

    async def _collection_config_exists(
        self,
        *,
        collection_name: str,
        user_id: int,
        context: str,
    ) -> bool:
        """Whether a config row is already published for this collection."""
        try:
            config = await self.get_collection_config(
                collection=collection_name,
                user_id=user_id,
                is_admin=False,
            )
        except Exception as exc:  # noqa: BLE001
            # Unreadable state must not authorize a destructive rollback.
            logger.warning(
                "Failed to read the collection config of %s/user_%s during %s: %s",
                collection_name,
                user_id,
                context,
                exc,
            )
            return True
        return config is not None

    async def failed_ingest_may_delete_collection(
        self,
        *,
        collection_name: str,
        user_id: int,
        collection_existed_before: bool,
        collection_records: list[Any],
        register_created: bool,
        doc_id: Optional[str],
        context: str,
    ) -> bool:
        """Whether a failed ingest may delete the whole collection.

        `collection_records` is the caller's read of the collection. A record is
        another ingest's unless `register_created` and its doc_id is `doc_id`; a
        caller that already removed its own document passes
        `register_created=False`, so every remaining record counts.

        `collection_existed_before` is stamped when the request or job is submitted,
        so a sibling ingest into the same new collection may have landed a document
        or published the config since. Either one makes the delete destructive, and
        both are re-read here rather than inferred from the stamp.

        The stamp itself cannot be re-read: `initialize_collection` creates the
        collection metadata row at step 0 of the ingest (collection_manager.py:598),
        so by rollback time the row exists for every run and says nothing about
        whether the collection predated this one.

        Caveat: these reads are not atomic with the delete that follows, so a
        sibling that commits inside the window is still exposed. Closing it needs a
        per-collection lock, which the codebase does not have anywhere yet — tracked
        in https://github.com/xorbitsai/xagent/issues/1242 rather than invented here.

        Caveat: when the config read itself fails, this fails safe and a brand-new
        zero-document collection keeps its metadata row with no config — invisible
        to its owner and still answering 409 for the name. That row is deliberately
        not cleaned up on this branch: the cleanup deletes this owner's config row
        first, so on the other reading of an unreadable state — a sibling published
        it — it would drop settings that sibling just saved.
        """
        # Compare doc_ids, not file_ids: same-path ingests share a file_id.
        other_document_present = any(
            not register_created
            or (
                record.get("doc_id")
                if isinstance(record, dict)
                else getattr(record, "doc_id", None)
            )
            != doc_id
            for record in collection_records
        )
        if collection_existed_before or other_document_present:
            return False
        return not await self._collection_config_exists(
            collection_name=collection_name,
            user_id=user_id,
            context=context,
        )

    async def cleanup_failed_new_collection_metadata(
        self,
        *,
        collection_name: str,
        user_id: int,
    ) -> None:
        """Remove config rows left behind when a brand-new collection ingest fails.

        Every caller reaches here from a stale `collection_existed_before`, so the
        documents check is the last guard before a row a sibling ingest owns is
        deleted by name.
        """
        # Read as the owner, and delete as the owner too: an admin-scoped delete
        # bypasses owner scoping entirely and would wipe a metadata row another
        # tenant owns under the same name, which the owner-scoped read never saw.
        if self.collection_holds_documents(
            collection_name=collection_name,
            user_id=user_id,
            context="failed-ingest metadata cleanup",
            on_error=True,
        ):
            logger.info(
                "Skipping failed-ingest collection metadata cleanup for %s/user_%s "
                "because the collection holds documents",
                collection_name,
                user_id,
            )
            return

        cleanup_result = await self.delete_collection_metadata(
            collection_name=collection_name,
            user_id=user_id,
            is_admin=False,
            delete_orphaned_metadata=True,
        )
        logger.info(
            "Cleaned failed-ingest collection metadata for %s: %s",
            collection_name,
            cleanup_result,
        )

    async def save_collection_config(
        self,
        *,
        collection: str,
        config_json: str,
        user_id: int,
    ) -> None:
        """Save tenant-scoped config and ensure owner-neutral backend binding."""
        with self._storage_context():
            from ..storage.factory import get_metadata_store

            store = get_metadata_store()

            await maybe_await(
                store.save_collection_config(
                    collection=collection,
                    config_json=config_json,
                    user_id=user_id,
                )
            )
            await self._ensure_collection_backend_binding_with_store(collection, store)

    async def ensure_collection_backend_binding(
        self,
        collection: str,
    ) -> CollectionInfo | None:
        """Create a collection-level backend binding without changing owners."""
        with self._storage_context():
            from ..storage.factory import get_metadata_store

            store = get_metadata_store()
            return await self._ensure_collection_backend_binding_with_store(
                collection,
                store,
            )

    async def _ensure_collection_backend_binding_with_store(
        self,
        collection: str,
        store: Any,
    ) -> CollectionInfo | None:
        if not _has_store_method(store, "save_collection"):
            return None

        collection_info: CollectionInfo | None = None
        if _has_store_method(store, "get_collection"):
            try:
                loaded = store.get_collection(collection)
                loaded = await maybe_await(loaded)
            except ValueError:
                collection_info = CollectionInfo(name=collection)
            else:
                if isinstance(loaded, CollectionInfo):
                    collection_info = loaded
                elif loaded is None:
                    collection_info = CollectionInfo(name=collection)
        else:
            collection_info = CollectionInfo(name=collection)

        if collection_info is None:
            return None

        extra_metadata = dict(collection_info.extra_metadata or {})
        if extra_metadata.get(KB_STORAGE_METADATA_KEY) is not None:
            return collection_info

        extra_metadata[KB_STORAGE_METADATA_KEY] = {
            "backend": deployment_kb_backend().value
        }
        updated_collection = collection_info.model_copy(
            update={"extra_metadata": extra_metadata}
        )
        await maybe_await(store.save_collection(updated_collection))
        return updated_collection

    async def get_collection_config(
        self,
        *,
        collection: str,
        user_id: Optional[int],
        is_admin: bool = False,
    ) -> str | None:
        """Read tenant-scoped config through the facade-bound metadata store."""
        with self._storage_context():
            from ..storage.factory import get_metadata_store

            return cast(
                str | None,
                await maybe_await(
                    get_metadata_store().get_collection_config(
                        collection=collection,
                        user_id=user_id,
                        is_admin=is_admin,
                    )
                ),
            )

    async def delete_collection_metadata(
        self,
        *,
        collection_name: str,
        user_id: Optional[int],
        is_admin: bool = False,
        delete_orphaned_metadata: bool = False,
    ) -> dict[str, int]:
        """Delete collection metadata through the facade-bound metadata store."""
        with self._storage_context():
            from ..storage.factory import get_metadata_store

            return cast(
                dict[str, int],
                await maybe_await(
                    get_metadata_store().delete_collection_metadata(
                        collection_name=collection_name,
                        user_id=user_id,
                        is_admin=is_admin,
                        delete_orphaned_metadata=delete_orphaned_metadata,
                    )
                ),
            )

    def list_collection_config_owner_ids(self, collection_name: str) -> set[int]:
        """List tenant config owners through the facade-bound metadata store."""
        with self._storage_context():
            from ..storage.factory import get_metadata_store

            return set(
                get_metadata_store().list_collection_config_owner_ids(collection_name)
            )

    def get_collection_sync(self, collection_name: str) -> CollectionInfo:
        if self._coordinator is not None:
            return self._coordinator.maintenance_compatibility.get_collection_sync(
                collection_name
            )

        from ..management import collection_manager as collection_manager_module

        with self._storage_context():
            return collection_manager_module.get_collection_sync(collection_name)

    def delete_collection_metadata_sync(
        self,
        *,
        collection_name: str,
        user_id: Optional[int],
        is_admin: bool = False,
        delete_orphaned_metadata: bool = False,
    ) -> dict[str, int]:
        if self._coordinator is not None:
            return self._coordinator.maintenance_compatibility.delete_collection_metadata_sync(
                collection_name=collection_name,
                user_id=user_id,
                is_admin=is_admin,
                delete_orphaned_metadata=delete_orphaned_metadata,
            )

        from ..management import collection_manager as collection_manager_module

        with self._storage_context():
            return collection_manager_module.delete_collection_metadata_sync(
                collection_name=collection_name,
                user_id=user_id,
                is_admin=is_admin,
                delete_orphaned_metadata=delete_orphaned_metadata,
            )

    async def list_collections(
        self,
        user_id: Optional[int] = None,
        is_admin: Optional[bool] = None,
        force_realtime: bool = False,
    ) -> ListCollectionsResult:
        if self._coordinator is not None:
            return await self._coordinator.management.list_collections(
                user_id=user_id,
                is_admin=is_admin,
                force_realtime=force_realtime,
            )

        from ..management import collections as collections_module

        with self._storage_context():
            return await collections_module.list_collections(
                user_id=user_id,
                is_admin=is_admin,
                force_realtime=force_realtime,
            )

    def list_documents(
        self,
        collection: str,
        user_id: Optional[int] = None,
        is_admin: bool = False,
    ) -> DocumentListResult:
        if self._coordinator is not None:
            return self._coordinator.management.list_documents(
                collection=collection,
                user_id=user_id,
                is_admin=is_admin,
            )

        from ..management import collections as collections_module

        with self._storage_context():
            return collections_module.list_documents(
                collection=collection,
                user_id=user_id,
                is_admin=is_admin,
            )

    def list_document_records(
        self,
        *,
        collection_name: Optional[str],
        user_id: Optional[int],
        is_admin: bool = False,
        max_results: Optional[int] = None,
    ) -> list[Any]:
        with self._storage_context():
            from ..storage.factory import get_vector_index_store

            store = get_vector_index_store()
            kwargs: dict[str, Any] = {
                "collection_name": collection_name,
                "user_id": user_id,
                "is_admin": is_admin,
            }
            if max_results is not None:
                kwargs["max_results"] = max_results
            return store.list_document_records(**kwargs)

    def list_document_records_by_file_ids(
        self, file_ids: Iterable[str]
    ) -> list[DocumentRecord]:
        with self._storage_context():
            from ..storage.factory import get_vector_index_store

            return get_vector_index_store().list_document_records_by_file_ids(file_ids)

    def load_ingestion_status_rows(
        self, doc_refs: Sequence[tuple[str, str]]
    ) -> list[dict[str, Any]]:
        with self._storage_context():
            from ..storage.factory import get_ingestion_status_store

            return get_ingestion_status_store().load_ingestion_status_rows(doc_refs)

    def replace_ingestion_status_rows(
        self,
        doc_refs: Sequence[tuple[str, str]],
        rows: Sequence[dict[str, Any]],
    ) -> None:
        with self._storage_context():
            from ..storage.factory import get_ingestion_status_store

            get_ingestion_status_store().replace_ingestion_status_rows(doc_refs, rows)

    def delete_document(
        self,
        collection: str,
        doc_id: str,
        user_id: int,
        is_admin: bool = False,
    ) -> DocumentOperationResult:
        if self._coordinator is not None:
            return self._coordinator.management.delete_document(
                collection=collection,
                doc_id=doc_id,
                user_id=user_id,
                is_admin=is_admin,
            )

        from ..management import collections as collections_module

        with self._storage_context():
            return collections_module.delete_document(
                collection,
                doc_id,
                user_id,
                is_admin,
            )

    def delete_collection(
        self,
        collection: str,
        user_id: Optional[int] = None,
        is_admin: bool = False,
    ) -> CollectionOperationResult:
        if self._coordinator is not None:
            return self._coordinator.management.delete_collection(
                collection=collection,
                user_id=user_id,
                is_admin=is_admin,
            )

        from ..management import collections as collections_module

        with self._storage_context():
            return collections_module.delete_collection(collection, user_id, is_admin)

    async def rename_collection_data(
        self,
        *,
        collection_name: str,
        new_name: str,
        user_id: Optional[int],
        is_admin: bool = False,
    ) -> list[str]:
        if self._coordinator is not None:
            return await self._coordinator.rename_collection(
                old_name=collection_name,
                new_name=new_name,
                user_id=user_id,
                is_admin=is_admin,
            )

        import asyncio

        with self._storage_context():
            from ..storage.factory import get_vector_index_store

            store = get_vector_index_store()
            return await asyncio.to_thread(
                store.rename_collection_data,
                collection_name=collection_name,
                new_name=new_name,
                user_id=user_id,
                is_admin=is_admin,
            )

    async def rename_collection_metadata(
        self,
        *,
        old_name: str,
        new_name: str,
        user_id: Optional[int],
        is_admin: bool = False,
    ) -> None:
        if self._coordinator is not None:
            # rename_collection_data (called earlier in the web API rename sequence)
            # already routed all three rename steps (data + status + metadata) through
            # coordinator.rename_collection().  This call is a no-op to avoid a
            # double-rename of metadata rows that no longer exist under old_name.
            return

        with self._storage_context():
            from ..storage.factory import get_metadata_store

            store = get_metadata_store()
            await store.rename_collection(
                old_name=old_name,
                new_name=new_name,
                user_id=user_id,
                is_admin=is_admin,
            )

    def rename_collection_status(
        self,
        *,
        old_name: str,
        new_name: str,
        user_id: Optional[int],
        is_admin: bool = False,
    ) -> list[str]:
        if self._coordinator is not None:
            # rename_collection_data already completed all three rename steps via
            # coordinator.rename_collection().  Return empty warnings to avoid
            # double-renaming status rows that no longer exist under old_name.
            return []

        with self._storage_context():
            from ..storage.factory import get_ingestion_status_store

            store = get_ingestion_status_store()
            return store.rename_collection_status(
                old_name=old_name,
                new_name=new_name,
                user_id=user_id,
                is_admin=is_admin,
            )

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
        commit_gate: Optional[Any] = None,
    ) -> IngestionResult:
        if self._coordinator is not None:
            return self._coordinator.pipeline_compatibility.run_document_ingestion(
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

        from ..pipelines import document_ingestion as document_ingestion_pipeline

        with self._storage_context():
            return document_ingestion_pipeline.run_document_ingestion(
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
        if self._coordinator is not None:
            return self._coordinator.pipeline_compatibility.run_document_search(
                collection=collection,
                query_text=query_text,
                config=config,
                progress_manager=progress_manager,
                user_id=user_id,
                is_admin=is_admin,
            )

        from ..pipelines import document_search as document_search_pipeline

        with self._storage_context():
            return document_search_pipeline.run_document_search(
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
        ingestion_config: Optional[Any] = None,
        progress_callback: Optional[Any] = None,
        user_id: Optional[int] = None,
        is_admin: Optional[bool] = None,
        file_handler: Optional[Any] = None,
    ) -> WebIngestionResult:
        if self._coordinator is not None:
            return await self._coordinator.pipeline_compatibility.run_web_ingestion(
                collection=collection,
                crawl_config=crawl_config,
                ingestion_config=ingestion_config,
                progress_callback=progress_callback,
                user_id=user_id,
                is_admin=is_admin,
                file_handler=file_handler,
            )

        from ..pipelines import web_ingestion as web_ingestion_pipeline

        with self._storage_context():
            return await web_ingestion_pipeline.run_web_ingestion(
                collection=collection,
                crawl_config=crawl_config,
                ingestion_config=ingestion_config,
                progress_callback=progress_callback,
                user_id=user_id,
                is_admin=is_admin,
                file_handler=file_handler,
            )

    def reconstruct_parse_result_from_db(
        self,
        collection: str,
        doc_id: str,
        parse_hash: Optional[str] = None,
        *,
        user_id: Optional[int] = None,
        is_admin: bool = False,
    ) -> tuple[list[dict[str, Any]], str | None]:
        if self._coordinator is not None:
            return self._coordinator.parse_display_compatibility.reconstruct_parse_result_from_db(
                collection=collection,
                doc_id=doc_id,
                parse_hash=parse_hash,
                user_id=user_id,
                is_admin=is_admin,
            )

        from ..parse import parse_display as parse_display_module

        with self._storage_context():
            return parse_display_module.reconstruct_parse_result_from_db(
                collection,
                doc_id,
                parse_hash,
                user_id=user_id,
                is_admin=is_admin,
            )

    def paginate_parse_results(
        self,
        elements: list[dict[str, Any]],
        page: int,
        page_size: int,
    ) -> tuple[list[Any], dict[str, Any]]:
        if self._coordinator is not None:
            return self._coordinator.parse_display_compatibility.paginate_parse_results(
                elements,
                page,
                page_size,
            )

        from ..parse import parse_display as parse_display_module

        return parse_display_module.paginate_parse_results(elements, page, page_size)
