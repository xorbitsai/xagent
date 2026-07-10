"""Semantic KB coordinator skeleton."""

from __future__ import annotations

import asyncio
import inspect
import logging
import threading
from collections.abc import Coroutine, Sequence
from contextvars import copy_context
from typing import Any, Callable, Dict, List, Optional, TypeVar, cast

from ..core.exceptions import (
    CascadeCleanupError,
    DatabaseOperationError,
    RagCoreException,
)
from ..core.schemas import (
    CollectionOperationDetail,
    CollectionOperationResult,
    DocumentProcessingStatus,
    DocumentRecordDetail,
    DocumentRecordListResult,
    RegisterDocumentRequest,
    RegisterDocumentResponse,
)
from ..storage.factory import StorageFactory
from ..utils.user_scope import resolve_user_scope
from .api_compatibility import KBApiCompatibilityFacade
from .collection_handle import (
    KBHandleProvider,
    KBMainPointerSnapshot,
    KBVersionCandidateCleanupSnapshot,
    KBVersionCandidateRollbackResult,
    LanceDBCollectionHandle,
)
from .file_compatibility import KBFileCompatibilityFacade
from .legacy_step_compatibility import KBLegacyStepCompatibilityFacade
from .maintenance_compatibility import KBMaintenanceCompatibilityFacade
from .management_facade import KBCoreManagementCompatibilityFacade
from .models import (
    KBAccessMode,
    KBBackendCapabilities,
    KBCollectionContext,
    KBContextRequest,
    KBStorageBackend,
    KBUserScope,
    KBVectorStorageCleanupResult,
    RollbackFailedIngestionRequest,
    RollbackFailedIngestionResult,
)
from .operation_compatibility import (
    KBOperationCompatibilityFacade,
    RollbackStatus,
    SideEffectPlane,
    _close_awaitable_if_possible,
)
from .parse_display_compatibility import KBParseDisplayCompatibilityFacade
from .pipeline_compatibility import KBPipelineCompatibilityFacade
from .retrieval_compatibility import KBRetrievalHelperCompatibilityFacade
from .storage_shim import KBStorageShimCompatibilityFacade
from .tool_compatibility import KBToolCompatibilityFacade
from .vector_storage_compatibility import KBVectorStorageCompatibilityFacade
from .version_compatibility import KBVersionCompatibilityFacade

T = TypeVar("T")

KB_STORAGE_METADATA_KEY = "kb_storage"

logger = logging.getLogger(__name__)


def _normalize_user_id(user_id: str | int | None) -> int | None:
    """Coerce ``user_id`` from ``str | int | None`` to ``int | None``.

    Raises:
        ValueError: If ``user_id`` is provided but cannot be converted to int.
    """
    if user_id is None:
        return None
    try:
        return int(user_id)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Invalid user_id: {user_id!r}") from exc


def _merge_positive_counts(
    target: dict[str, int], source: dict[str, int] | None
) -> None:
    """Merge ``source`` row counts into ``target``, dropping non-positive values.

    Mirrors the legacy ``_delete_collection_impl`` accounting: a ``{"documents": 0}``
    entry means "no rows of that kind were deleted" and is omitted so callers do
    not see misleading zero-count clutter in ``deleted_counts``.
    """
    for key, value in dict(source or {}).items():
        try:
            count = int(value)
        except (ValueError, TypeError):
            continue
        if count <= 0:
            continue
        target[str(key)] = target.get(str(key), 0) + count


class KBCoordinator:
    """KB-level semantic entry point for future compatibility facades."""

    def __init__(
        self,
        storage_factory: StorageFactory | None = None,
        handle_provider: KBHandleProvider | None = None,
        storage_shim: KBStorageShimCompatibilityFacade | None = None,
        file_compatibility: KBFileCompatibilityFacade | None = None,
        management_facade: KBCoreManagementCompatibilityFacade | None = None,
        parse_display_compatibility: KBParseDisplayCompatibilityFacade | None = None,
        maintenance_compatibility: KBMaintenanceCompatibilityFacade | None = None,
        version_compatibility: KBVersionCompatibilityFacade | None = None,
        retrieval_helper_compatibility: (
            KBRetrievalHelperCompatibilityFacade | None
        ) = None,
        vector_storage_compatibility: KBVectorStorageCompatibilityFacade | None = None,
        operation_compatibility: KBOperationCompatibilityFacade | None = None,
        pipeline_compatibility: KBPipelineCompatibilityFacade | None = None,
        legacy_step_compatibility: KBLegacyStepCompatibilityFacade | None = None,
        tool_compatibility: KBToolCompatibilityFacade | None = None,
        api_compatibility: KBApiCompatibilityFacade | None = None,
    ) -> None:
        self._storage_factory = storage_factory or StorageFactory.get_factory()
        self._handle_provider = handle_provider or KBHandleProvider()
        self._storage_shim = storage_shim or KBStorageShimCompatibilityFacade(
            storage_factory=self._storage_factory
        )
        self._file_compatibility = file_compatibility or KBFileCompatibilityFacade()
        self._management = management_facade or KBCoreManagementCompatibilityFacade(
            coordinator=self
        )
        self._parse_display_compatibility = (
            parse_display_compatibility
            or KBParseDisplayCompatibilityFacade(coordinator=self)
        )
        self._maintenance_compatibility = (
            maintenance_compatibility
            or KBMaintenanceCompatibilityFacade(coordinator=self)
        )
        self._version_compatibility = (
            version_compatibility or KBVersionCompatibilityFacade(coordinator=self)
        )
        self._retrieval_helper_compatibility = (
            retrieval_helper_compatibility
            or KBRetrievalHelperCompatibilityFacade(coordinator=self)
        )
        self._vector_storage_compatibility = (
            vector_storage_compatibility
            or KBVectorStorageCompatibilityFacade(coordinator=self)
        )
        self._operation_compatibility = (
            operation_compatibility or KBOperationCompatibilityFacade()
        )
        self._pipeline_compatibility = (
            pipeline_compatibility or KBPipelineCompatibilityFacade(coordinator=self)
        )
        self._legacy_step_compatibility = (
            legacy_step_compatibility
            or KBLegacyStepCompatibilityFacade(coordinator=self)
        )
        self._tool_compatibility = tool_compatibility or KBToolCompatibilityFacade(
            coordinator=self
        )
        self._api_compatibility = api_compatibility or KBApiCompatibilityFacade(
            coordinator=self
        )

    @property
    def storage_shim(self) -> KBStorageShimCompatibilityFacade:
        """Return the low-level storage compatibility facade."""
        return self._storage_shim

    @property
    def file_compatibility(self) -> KBFileCompatibilityFacade:
        """Return the uploaded-file and physical compatibility facade."""
        return self._file_compatibility

    @property
    def file_compat(self) -> KBFileCompatibilityFacade:
        """Backward-friendly short alias for the file compatibility facade."""
        return self._file_compatibility

    @property
    def management(self) -> KBCoreManagementCompatibilityFacade:
        """Return the core management compatibility facade."""
        return self._management

    @property
    def parse_display_compatibility(self) -> KBParseDisplayCompatibilityFacade:
        """Return the parse display compatibility facade."""
        return self._parse_display_compatibility

    @property
    def parse_display(self) -> KBParseDisplayCompatibilityFacade:
        """Backward-friendly short alias for the parse display facade."""
        return self._parse_display_compatibility

    @property
    def maintenance_compatibility(self) -> KBMaintenanceCompatibilityFacade:
        """Return the collection metadata maintenance compatibility facade."""
        return self._maintenance_compatibility

    @property
    def maintenance_compat(self) -> KBMaintenanceCompatibilityFacade:
        """Backward-friendly short alias for the maintenance facade."""
        return self._maintenance_compatibility

    @property
    def version_compatibility(self) -> KBVersionCompatibilityFacade:
        """Return the version-management compatibility facade."""
        return self._version_compatibility

    @property
    def version(self) -> KBVersionCompatibilityFacade:
        """Backward-friendly short alias for the version facade."""
        return self._version_compatibility

    @property
    def retrieval_helper_compatibility(self) -> KBRetrievalHelperCompatibilityFacade:
        """Return the low-level retrieval helper compatibility facade."""
        return self._retrieval_helper_compatibility

    @property
    def retrieval_helper(self) -> KBRetrievalHelperCompatibilityFacade:
        """Backward-friendly short alias for the retrieval helper facade."""
        return self._retrieval_helper_compatibility

    @property
    def vector_storage_compatibility(self) -> KBVectorStorageCompatibilityFacade:
        """Return the vector storage compatibility facade."""
        return self._vector_storage_compatibility

    @property
    def vector_storage(self) -> KBVectorStorageCompatibilityFacade:
        """Backward-friendly short alias for the vector storage facade."""
        return self._vector_storage_compatibility

    @property
    def operation_compatibility(self) -> KBOperationCompatibilityFacade:
        """Return the rollback-aware operation compatibility facade."""
        return self._operation_compatibility

    @property
    def operations(self) -> KBOperationCompatibilityFacade:
        """Backward-friendly short alias for the operation facade."""
        return self._operation_compatibility

    @property
    def pipeline_compatibility(self) -> KBPipelineCompatibilityFacade:
        """Return the high-level pipeline compatibility facade."""
        return self._pipeline_compatibility

    @property
    def pipeline(self) -> KBPipelineCompatibilityFacade:
        """Backward-friendly short alias for the pipeline facade."""
        return self._pipeline_compatibility

    @property
    def legacy_step_compatibility(self) -> KBLegacyStepCompatibilityFacade:
        """Return the legacy step helper compatibility facade."""
        return self._legacy_step_compatibility

    @property
    def legacy_steps(self) -> KBLegacyStepCompatibilityFacade:
        """Backward-friendly short alias for the legacy step facade."""
        return self._legacy_step_compatibility

    @property
    def tool_compatibility(self) -> KBToolCompatibilityFacade:
        """Return the agent/tool compatibility facade."""
        return self._tool_compatibility

    @property
    def tools(self) -> KBToolCompatibilityFacade:
        """Backward-friendly short alias for the tool facade."""
        return self._tool_compatibility

    @property
    def api_compatibility(self) -> KBApiCompatibilityFacade:
        """Return the API route compatibility facade."""
        return self._api_compatibility

    @property
    def api(self) -> KBApiCompatibilityFacade:
        """Backward-friendly short alias for the API facade."""
        return self._api_compatibility

    async def get_context(self, request: KBContextRequest) -> KBCollectionContext:
        """Resolve collection, caller scope, stores, backend, and capabilities."""
        collection = self._normalize_collection(request.collection)
        access_mode = self._normalize_access_mode(request.access_mode)
        user_scope = self._resolve_user_scope(request)
        metadata_store = self._storage_shim.get_metadata_store()
        vector_index_store = self._storage_shim.get_vector_index_store()
        ingestion_status_store = self._storage_shim.get_ingestion_status_store()
        main_pointer_store = self._storage_shim.get_main_pointer_store()

        collection_info = None
        try:
            collection_info = await metadata_store.get_collection(collection)
        except ValueError as exc:
            if not self._is_missing_collection_error(collection, exc):
                raise
            if not (request.hide_missing or request.allow_create):
                raise ValueError(f"Collection '{collection}' not found") from exc

        backend = self._resolve_backend(collection_info)
        capabilities = self._capabilities_for_backend(backend)

        return KBCollectionContext(
            collection=collection,
            user_scope=user_scope,
            access_mode=access_mode,
            allow_create=bool(request.allow_create),
            hide_missing=bool(request.hide_missing),
            metadata_store=metadata_store,
            vector_index_store=vector_index_store,
            ingestion_status_store=ingestion_status_store,
            main_pointer_store=main_pointer_store,
            backend=backend,
            capabilities=capabilities,
            collection_info=collection_info,
        )

    def get_context_sync(self, request: KBContextRequest) -> KBCollectionContext:
        """Synchronous wrapper for legacy compatibility surfaces."""
        return _run_in_separate_loop(self.get_context(request))

    async def open_collection(
        self, request: KBContextRequest
    ) -> LanceDBCollectionHandle:
        """Open a thin collection handle for the resolved context."""
        context = await self.get_context(request)
        return self._handle_provider.open(context)

    def open_collection_sync(
        self, request: KBContextRequest
    ) -> LanceDBCollectionHandle:
        """Synchronous wrapper for opening a collection handle."""
        return _run_in_separate_loop(self.open_collection(request))

    # --- Document-row lifecycle (delegated to the collection handle) ---

    async def register_document(
        self, request: RegisterDocumentRequest
    ) -> RegisterDocumentResponse:
        """Open the collection handle and register a document row."""
        handle = await self.open_collection(
            KBContextRequest(
                collection=request.collection,
                user_id=request.user_id,
                access_mode=KBAccessMode.WRITE,
                hide_missing=True,
            )
        )
        # The handle call is synchronous and blocking (file hashing + LanceDB
        # I/O); offload it so awaiting this method never stalls the event loop.
        return await asyncio.to_thread(handle.register_document, request)

    def register_document_sync(
        self, request: RegisterDocumentRequest
    ) -> RegisterDocumentResponse:
        """Synchronous wrapper for :meth:`register_document`."""
        return _run_in_separate_loop(self.register_document(request))

    async def load_document(
        self,
        collection: str,
        doc_id: str,
        *,
        user_id: Optional[int] = None,
        is_admin: bool = False,
    ) -> DocumentRecordDetail | None:
        """Open the collection handle and load a document row by id."""
        handle = await self.open_collection(
            KBContextRequest(
                collection=collection,
                user_id=user_id,
                is_admin=is_admin,
                hide_missing=True,
            )
        )
        # Blocking LanceDB read; offload so awaiting this never stalls the loop.
        return await asyncio.to_thread(
            handle.load_document, doc_id, user_id=user_id, is_admin=is_admin
        )

    def load_document_sync(
        self,
        collection: str,
        doc_id: str,
        *,
        user_id: Optional[int] = None,
        is_admin: bool = False,
    ) -> DocumentRecordDetail | None:
        """Synchronous wrapper for :meth:`load_document`."""
        return _run_in_separate_loop(
            self.load_document(collection, doc_id, user_id=user_id, is_admin=is_admin)
        )

    async def list_document_records(
        self,
        collection: str,
        *,
        user_id: Optional[int] = None,
        is_admin: bool = False,
        limit: int = 100,
    ) -> DocumentRecordListResult:
        """Open the collection handle and list document rows."""
        handle = await self.open_collection(
            KBContextRequest(
                collection=collection,
                user_id=user_id,
                is_admin=is_admin,
                hide_missing=True,
            )
        )
        # Blocking LanceDB scan; offload so awaiting this never stalls the loop.
        return await asyncio.to_thread(
            handle.list_documents, user_id=user_id, is_admin=is_admin, limit=limit
        )

    def list_document_records_sync(
        self,
        collection: str,
        *,
        user_id: Optional[int] = None,
        is_admin: bool = False,
        limit: int = 100,
    ) -> DocumentRecordListResult:
        """Synchronous wrapper for :meth:`list_document_records`."""
        return _run_in_separate_loop(
            self.list_document_records(
                collection, user_id=user_id, is_admin=is_admin, limit=limit
            )
        )

    async def delete_document_record(
        self,
        collection: str,
        doc_id: str,
        *,
        user_id: Optional[int] = None,
        is_admin: bool = False,
    ) -> int:
        """Open the collection handle and delete a document row (no cascade)."""
        handle = await self.open_collection(
            KBContextRequest(
                collection=collection,
                user_id=user_id,
                is_admin=is_admin,
                hide_missing=True,
            )
        )
        # Blocking LanceDB delete; offload so awaiting this never stalls the loop.
        return await asyncio.to_thread(
            handle.delete_document_record, doc_id, user_id=user_id, is_admin=is_admin
        )

    def delete_document_record_sync(
        self,
        collection: str,
        doc_id: str,
        *,
        user_id: Optional[int] = None,
        is_admin: bool = False,
    ) -> int:
        """Synchronous wrapper for :meth:`delete_document_record`."""
        return _run_in_separate_loop(
            self.delete_document_record(
                collection, doc_id, user_id=user_id, is_admin=is_admin
            )
        )

    # --- Ingestion-status lifecycle (delegated to the collection handle) ---

    async def write_ingestion_status(
        self,
        collection: str,
        doc_id: str,
        *,
        status: str,
        message: Optional[str] = None,
        parse_hash: Optional[str] = None,
        user_id: Optional[int] = None,
    ) -> None:
        """Open the collection handle and write ingestion status (async)."""
        handle = await self.open_collection(
            KBContextRequest(
                collection=collection,
                user_id=user_id,
                hide_missing=True,
            )
        )
        await asyncio.to_thread(
            handle.write_ingestion_status,
            doc_id,
            status=status,
            message=message,
            parse_hash=parse_hash,
            user_id=user_id,
        )

    def write_ingestion_status_sync(
        self,
        collection: str,
        doc_id: str,
        *,
        status: str,
        message: Optional[str] = None,
        parse_hash: Optional[str] = None,
        user_id: Optional[int] = None,
    ) -> None:
        """Synchronous wrapper for :meth:`write_ingestion_status`."""
        _run_in_separate_loop(
            self.write_ingestion_status(
                collection,
                doc_id,
                status=status,
                message=message,
                parse_hash=parse_hash,
                user_id=user_id,
            )
        )

    async def write_ingestion_status_async(
        self,
        collection: str,
        doc_id: str,
        *,
        status: str,
        message: Optional[str] = None,
        parse_hash: Optional[str] = None,
        user_id: Optional[int] = None,
    ) -> None:
        """Open the collection handle and write ingestion status (direct async)."""
        handle = await self.open_collection(
            KBContextRequest(
                collection=collection,
                user_id=user_id,
                hide_missing=True,
            )
        )
        await handle.write_ingestion_status_async(
            doc_id,
            status=status,
            message=message,
            parse_hash=parse_hash,
            user_id=user_id,
        )

    async def load_ingestion_status(
        self,
        collection: str,
        *,
        doc_id: Optional[str] = None,
        user_id: Optional[int] = None,
        is_admin: bool = False,
    ) -> list:
        """Open the collection handle and load ingestion status (async)."""
        handle = await self.open_collection(
            KBContextRequest(
                collection=collection,
                user_id=user_id,
                is_admin=is_admin,
                hide_missing=True,
            )
        )
        return await asyncio.to_thread(
            handle.load_ingestion_status,
            doc_id=doc_id,
            user_id=user_id,
            is_admin=is_admin,
        )

    def load_ingestion_status_sync(
        self,
        collection: str,
        *,
        doc_id: Optional[str] = None,
        user_id: Optional[int] = None,
        is_admin: bool = False,
    ) -> list:
        """Synchronous wrapper for :meth:`load_ingestion_status`."""
        return _run_in_separate_loop(
            self.load_ingestion_status(
                collection, doc_id=doc_id, user_id=user_id, is_admin=is_admin
            )
        )

    async def load_ingestion_status_async(
        self,
        collection: str,
        *,
        doc_id: Optional[str] = None,
        user_id: Optional[int] = None,
        is_admin: bool = False,
    ) -> list:
        """Open the collection handle and load ingestion status (direct async)."""
        handle = await self.open_collection(
            KBContextRequest(
                collection=collection,
                user_id=user_id,
                is_admin=is_admin,
                hide_missing=True,
            )
        )
        return await handle.load_ingestion_status_async(
            doc_id=doc_id,
            user_id=user_id,
            is_admin=is_admin,
        )

    async def clear_ingestion_status(
        self,
        collection: str,
        doc_id: str,
        *,
        user_id: Optional[int] = None,
        is_admin: bool = False,
    ) -> None:
        """Open the collection handle and clear ingestion status (async)."""
        handle = await self.open_collection(
            KBContextRequest(
                collection=collection,
                user_id=user_id,
                is_admin=is_admin,
                hide_missing=True,
            )
        )
        await asyncio.to_thread(
            handle.clear_ingestion_status,
            doc_id,
            user_id=user_id,
            is_admin=is_admin,
        )

    def clear_ingestion_status_sync(
        self,
        collection: str,
        doc_id: str,
        *,
        user_id: Optional[int] = None,
        is_admin: bool = False,
    ) -> None:
        """Synchronous wrapper for :meth:`clear_ingestion_status`."""
        _run_in_separate_loop(
            self.clear_ingestion_status(
                collection, doc_id, user_id=user_id, is_admin=is_admin
            )
        )

    async def clear_ingestion_status_async(
        self,
        collection: str,
        doc_id: str,
        *,
        user_id: Optional[int] = None,
        is_admin: bool = False,
    ) -> None:
        """Open the collection handle and clear ingestion status (direct async)."""
        handle = await self.open_collection(
            KBContextRequest(
                collection=collection,
                user_id=user_id,
                is_admin=is_admin,
                hide_missing=True,
            )
        )
        await handle.clear_ingestion_status_async(
            doc_id,
            user_id=user_id,
            is_admin=is_admin,
        )

    async def rename_collection_status(
        self,
        old_name: str,
        new_name: str,
        *,
        user_id: Optional[int] = None,
        is_admin: bool = False,
    ) -> list:
        """Open the old collection handle and rename status rows to ``new_name``."""
        handle = await self.open_collection(
            KBContextRequest(
                collection=old_name,
                user_id=user_id,
                is_admin=is_admin,
                hide_missing=True,
            )
        )
        return await asyncio.to_thread(
            handle.rename_collection_status,
            new_name,
            user_id,
            is_admin,
        )

    def rename_collection_status_sync(
        self,
        old_name: str,
        new_name: str,
        *,
        user_id: Optional[int] = None,
        is_admin: bool = False,
    ) -> list:
        """Synchronous wrapper for :meth:`rename_collection_status`."""
        return _run_in_separate_loop(
            self.rename_collection_status(
                old_name, new_name, user_id=user_id, is_admin=is_admin
            )
        )

    async def delete_collection(
        self,
        collection: str,
        user_id: str | int | None,
        is_admin: bool,
        doc_ids: list[str] | None = None,
        warnings_out: list[str] | None = None,
        delete_orphaned_metadata: bool = True,
    ) -> CollectionOperationResult:
        """Delete a collection by routing through the collection handle.

        When ``is_admin`` is ``True`` all rows are deleted via
        :meth:`LanceDBCollectionHandle.delete_collection_data`.  For a tenant
        caller, only the rows identified by ``doc_ids`` are removed via
        :meth:`LanceDBCollectionHandle.delete_documents_data`.  When
        ``doc_ids`` is ``None`` or empty and ``is_admin`` is ``False`` the
        data plane is left untouched (config-only path).

        ``delete_orphaned_metadata=True`` (default) additionally removes the
        collection config row via :meth:`LanceDBCollectionHandle.delete_collection_config`.

        Returns:
            :class:`CollectionOperationResult` with status ``success``,
            ``partial_success`` (when a :class:`DatabaseOperationError` was
            caught during the data-plane delete), or ``error``.
        """
        int_user_id = _normalize_user_id(user_id)

        handle = await self.open_collection(
            KBContextRequest(
                collection=collection,
                user_id=int_user_id,
                is_admin=is_admin,
                hide_missing=True,
            )
        )

        warnings: list[str] = warnings_out if warnings_out is not None else []
        deleted_counts: dict[str, int] = {}
        data_error: Exception | None = None

        # Collect doc_ids BEFORE deletion for affected_documents tracking.
        # Skip discovery when the caller already provided explicit doc_ids — those
        # are the affected documents.  Only query when we need auto-discovery (admin
        # deletes all, or tenant lets us discover their scope via doc_ids=None).
        affected_doc_ids: list[str] = []
        if is_admin or doc_ids is None:
            try:
                affected_doc_ids = await asyncio.to_thread(
                    handle.list_collection_documents,
                    user_id=int_user_id,
                    is_admin=is_admin,
                )
            except Exception as exc:  # noqa: BLE001
                # For a tenant caller where doc_ids=None (delete their entire collection),
                # discovery failure means we cannot determine the correct deletion scope.
                # Silently skipping data-plane delete and returning "success" would be wrong.
                if not is_admin and doc_ids is None:
                    return CollectionOperationResult(
                        status="error",
                        collection=collection,
                        message=f"Failed to list documents before delete for {collection!r}: {exc}",
                        warnings=list(warnings),
                        affected_documents=[],
                        deleted_counts={},
                    )
                warnings.append(
                    f"Failed to list documents before delete for {collection!r}: {exc}"
                )
        else:
            # Caller supplied explicit doc_ids — they are the affected documents.
            affected_doc_ids = list(doc_ids)

        # For tenant (non-admin) callers: use caller-supplied doc_ids when provided,
        # otherwise fall back to the discovered set so the data-plane delete always
        # operates on the right scope (consistent with _delete_collection_impl).
        effective_doc_ids: list[str] | None = doc_ids
        if not is_admin and effective_doc_ids is None:
            effective_doc_ids = affected_doc_ids

        try:
            if is_admin:
                result_counts = await asyncio.to_thread(
                    handle.delete_collection_data,
                    user_id=int_user_id,
                    is_admin=is_admin,
                    warnings_out=warnings,
                )
                _merge_positive_counts(deleted_counts, result_counts)
            elif effective_doc_ids:
                result_counts = await asyncio.to_thread(
                    handle.delete_documents_data,
                    effective_doc_ids,
                    user_id=int_user_id,
                    is_admin=is_admin,
                    warnings_out=warnings,
                )
                _merge_positive_counts(deleted_counts, result_counts)
            # else: config-only — no data-plane delete
        except (DatabaseOperationError, CascadeCleanupError) as exc:
            # CascadeCleanupError (admin cascade path) carries no per-doc details;
            # DatabaseOperationError (tenant batch path) may carry deleted_counts.
            data_error = exc
            details = getattr(exc, "details", {}) or {}
            if isinstance(details, dict):
                raw_counts = details.get("deleted_counts")
                if isinstance(raw_counts, dict):
                    _merge_positive_counts(deleted_counts, raw_counts)

        if delete_orphaned_metadata and data_error is None:
            # Always remove the current tenant's config row so it does not
            # become orphaned when other tenants still have documents.
            # When the collection is completely empty across all tenants, also
            # do an admin-scope cleanup to remove any remaining rows.
            # Skip config cleanup when the data-plane delete failed — removing
            # config while data rows remain would lose the user's KB state.
            try:
                remaining = await asyncio.to_thread(
                    handle.count_documents,
                    user_id=None,
                    is_admin=True,
                )
            except (RagCoreException, OSError):
                remaining = 1
            try:
                if is_admin and remaining == 0:
                    # Admin caller + collection fully empty: remove all tenant rows.
                    await handle.delete_collection_config()
                else:
                    # Non-admin caller, or other tenants still have data: only remove
                    # the current tenant's config row to preserve tenant isolation.
                    await handle.delete_collection_config(tenant_only=True)
            except Exception as cfg_exc:  # noqa: BLE001 - best-effort
                warnings.append(
                    f"Failed to delete collection config for {collection!r}: {cfg_exc}"
                )

        def _to_details(
            doc_ids: list[str], status: DocumentProcessingStatus
        ) -> list[CollectionOperationDetail]:
            return [CollectionOperationDetail(doc_id=d, status=status) for d in doc_ids]

        if data_error is not None:
            if deleted_counts:
                # Extract successfully deleted doc_ids from error details to provide
                # accurate per-document status instead of marking everything FAILED.
                err_details = getattr(data_error, "details", {}) or {}
                raw_deleted = (
                    err_details.get("deleted_doc_ids")
                    if isinstance(err_details, dict)
                    else None
                )
                deleted_doc_ids: list[str] = (
                    raw_deleted if isinstance(raw_deleted, list) else []
                )
                deleted_set = set(deleted_doc_ids)
                failed_doc_ids = [d for d in affected_doc_ids if d not in deleted_set]
                return CollectionOperationResult(
                    status="partial_success",
                    collection=collection,
                    message=f"Partially deleted collection {collection!r}: {data_error}",
                    warnings=list(warnings),
                    affected_documents=(
                        _to_details(deleted_doc_ids, DocumentProcessingStatus.SUCCESS)
                        + _to_details(failed_doc_ids, DocumentProcessingStatus.FAILED)
                    ),
                    deleted_counts=dict(deleted_counts),
                )
            return CollectionOperationResult(
                status="error",
                collection=collection,
                message=f"Failed to delete collection {collection!r}: {data_error}",
                warnings=list(warnings),
                affected_documents=_to_details(
                    affected_doc_ids, DocumentProcessingStatus.FAILED
                ),
                deleted_counts={},
            )

        # Best-effort data-plane deletes (delete_collection_data / config cleanup)
        # surface partial failures as appended warnings rather than raising.  When
        # such warnings accompany an actual deletion, report ``partial_success`` so
        # the caller is not told the operation fully succeeded — mirroring the
        # legacy ``_delete_collection_impl`` status semantics.
        something_deleted = bool(deleted_counts) or bool(affected_doc_ids)
        status = "partial_success" if warnings and something_deleted else "success"
        message = (
            f"Partially deleted collection {collection!r}."
            if status == "partial_success"
            else f"Collection {collection!r} deleted successfully."
        )
        return CollectionOperationResult(
            status=status,
            collection=collection,
            message=message,
            warnings=list(warnings),
            affected_documents=_to_details(
                affected_doc_ids, DocumentProcessingStatus.SUCCESS
            ),
            deleted_counts=dict(deleted_counts),
        )

    async def rename_collection(
        self,
        old_name: str,
        new_name: str,
        user_id: str | int | None,
        is_admin: bool,
    ) -> list[str]:
        """Rename a collection's data, status, and metadata in best-effort order.

        Calls three handle primitives sequentially:
        1. :meth:`LanceDBCollectionHandle.rename_collection_data` – vector-side data tables
        2. :meth:`LanceDBCollectionHandle.rename_collection_status` – ingestion status rows
        3. :meth:`LanceDBCollectionHandle.rename_collection_metadata` – control-plane metadata (async)

        Each step is best-effort: if one raises, the error is recorded as a
        warning and the remaining steps still execute.

        Returns:
            A list of warning strings (empty on full success).
        """
        int_user_id = _normalize_user_id(user_id)

        handle = await self.open_collection(
            KBContextRequest(
                collection=old_name,
                user_id=int_user_id,
                is_admin=is_admin,
                hide_missing=True,
            )
        )

        warnings: list[str] = []

        # The data rename is the gate for the control-plane rename: if any vector
        # row was not moved, abort before touching status/metadata to avoid a
        # split-brain collection where metadata points at new_name while vector
        # data remains under old_name.  Failures surface two ways and BOTH must
        # gate: a hard exception (e.g. no DB connection) propagates out, and
        # per-table failures are returned as a non-empty warnings list (the store
        # catches them per table rather than raising) — short-circuit on those too.
        data_warnings = await asyncio.to_thread(
            handle.rename_collection_data,
            new_name,
            int_user_id,
            is_admin,
        )
        if data_warnings:
            warnings.extend(data_warnings)
            return warnings

        try:
            status_warnings = await asyncio.to_thread(
                handle.rename_collection_status,
                new_name,
                int_user_id,
                is_admin,
            )
            if status_warnings:
                warnings.extend(status_warnings)
        except Exception as exc:  # noqa: BLE001 - best-effort
            warnings.append(
                f"rename_collection_status for {old_name!r} → {new_name!r} failed: {exc}"
            )

        try:
            await handle.rename_collection_metadata(new_name, int_user_id, is_admin)
        except Exception as exc:  # noqa: BLE001 - best-effort
            warnings.append(
                f"rename_collection_metadata for {old_name!r} → {new_name!r} failed: {exc}"
            )

        return warnings

    # --- Main-pointer lifecycle (delegated to the collection handle) ---

    async def get_main_pointer(
        self,
        collection: str,
        doc_id: str,
        step_type: str,
        *,
        model_tag: Optional[str] = None,
    ) -> Optional[dict]:
        """Open the collection handle and get a main pointer (async)."""
        handle = await self.open_collection(
            KBContextRequest(
                collection=collection,
                hide_missing=True,
            )
        )
        return await asyncio.to_thread(
            handle.get_main_pointer, doc_id, step_type, model_tag
        )

    def get_main_pointer_sync(
        self,
        collection: str,
        doc_id: str,
        step_type: str,
        *,
        model_tag: Optional[str] = None,
    ) -> Optional[dict]:
        """Synchronous wrapper for :meth:`get_main_pointer`."""
        return _run_in_separate_loop(
            self.get_main_pointer(collection, doc_id, step_type, model_tag=model_tag)
        )

    async def set_main_pointer(
        self,
        collection: str,
        doc_id: str,
        step_type: str,
        semantic_id: str,
        technical_id: str,
        *,
        model_tag: Optional[str] = None,
        operator: Optional[str] = None,
    ) -> None:
        """Open the collection handle and set a main pointer (async)."""
        handle = await self.open_collection(
            KBContextRequest(
                collection=collection,
                access_mode=KBAccessMode.WRITE,
                hide_missing=True,
            )
        )
        return await asyncio.to_thread(
            handle.set_main_pointer,
            doc_id,
            step_type,
            semantic_id,
            technical_id,
            model_tag,
            operator,
        )

    def set_main_pointer_sync(
        self,
        collection: str,
        doc_id: str,
        step_type: str,
        semantic_id: str,
        technical_id: str,
        *,
        model_tag: Optional[str] = None,
        operator: Optional[str] = None,
    ) -> None:
        """Synchronous wrapper for :meth:`set_main_pointer`."""
        _run_in_separate_loop(
            self.set_main_pointer(
                collection,
                doc_id,
                step_type,
                semantic_id,
                technical_id,
                model_tag=model_tag,
                operator=operator,
            )
        )

    async def list_main_pointers(
        self,
        collection: str,
        *,
        doc_id: Optional[str] = None,
        limit: int = 100,
    ) -> list:
        """Open the collection handle and list main pointers (async)."""
        handle = await self.open_collection(
            KBContextRequest(
                collection=collection,
                hide_missing=True,
            )
        )
        return await asyncio.to_thread(handle.list_main_pointers, doc_id, limit)

    def list_main_pointers_sync(
        self,
        collection: str,
        *,
        doc_id: Optional[str] = None,
        limit: int = 100,
    ) -> list:
        """Synchronous wrapper for :meth:`list_main_pointers`."""
        return _run_in_separate_loop(
            self.list_main_pointers(collection, doc_id=doc_id, limit=limit)
        )

    async def delete_main_pointer(
        self,
        collection: str,
        doc_id: str,
        step_type: str,
        *,
        model_tag: Optional[str] = None,
    ) -> bool:
        """Open the collection handle and delete a main pointer (async)."""
        handle = await self.open_collection(
            KBContextRequest(
                collection=collection,
                access_mode=KBAccessMode.WRITE,
                hide_missing=True,
            )
        )
        return await asyncio.to_thread(
            handle.delete_main_pointer, doc_id, step_type, model_tag
        )

    def delete_main_pointer_sync(
        self,
        collection: str,
        doc_id: str,
        step_type: str,
        *,
        model_tag: Optional[str] = None,
    ) -> bool:
        """Synchronous wrapper for :meth:`delete_main_pointer`."""
        return _run_in_separate_loop(
            self.delete_main_pointer(collection, doc_id, step_type, model_tag=model_tag)
        )

    async def list_candidates(
        self,
        collection: str,
        doc_id: str,
        step_type: str,
        *,
        model_tag: Optional[str] = None,
        state: Optional[str] = None,
        limit: int = 50,
        order_by: str = "created_at desc",
    ) -> dict:
        """Open the collection handle and list version candidates (async)."""
        handle = await self.open_collection(
            KBContextRequest(
                collection=collection,
                hide_missing=True,
            )
        )
        return await asyncio.to_thread(
            handle.list_candidates,
            doc_id,
            step_type,
            model_tag,
            state,
            limit,
            order_by,
        )

    def list_candidates_sync(
        self,
        collection: str,
        doc_id: str,
        step_type: str,
        *,
        model_tag: Optional[str] = None,
        state: Optional[str] = None,
        limit: int = 50,
        order_by: str = "created_at desc",
    ) -> dict:
        """Synchronous wrapper for :meth:`list_candidates`."""
        return _run_in_separate_loop(
            self.list_candidates(
                collection,
                doc_id,
                step_type,
                model_tag=model_tag,
                state=state,
                limit=limit,
                order_by=order_by,
            )
        )

    async def promote_version_main(
        self,
        collection: str,
        doc_id: str,
        step_type: str,
        selected_id: str,
        *,
        operator: Optional[str] = None,
        preview_only: bool = False,
        confirm: bool = False,
        model_tag: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Open the collection handle and promote a version candidate to main (async)."""
        handle = await self.open_collection(
            KBContextRequest(
                collection=collection,
                user_id=None,
                is_admin=True,
                access_mode=KBAccessMode.WRITE,
                hide_missing=True,
            )
        )
        return await asyncio.to_thread(
            handle.promote_version_main,
            doc_id,
            step_type,
            selected_id,
            operator=operator,
            preview_only=preview_only,
            confirm=confirm,
            model_tag=model_tag,
        )

    def promote_version_main_sync(
        self,
        collection: str,
        doc_id: str,
        step_type: str,
        selected_id: str,
        *,
        operator: Optional[str] = None,
        preview_only: bool = False,
        confirm: bool = False,
        model_tag: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Synchronous wrapper for :meth:`promote_version_main`."""
        return _run_in_separate_loop(
            self.promote_version_main(
                collection,
                doc_id,
                step_type,
                selected_id,
                operator=operator,
                preview_only=preview_only,
                confirm=confirm,
                model_tag=model_tag,
            )
        )

    async def cleanup_cascade(
        self,
        collection: str,
        doc_id: str,
        scope: str,
        *,
        new_parse_hash: Optional[str] = None,
        old_parse_hash: Optional[str] = None,
        model_tag: Optional[str] = None,
        user_id: Optional[int] = None,
        is_admin: Optional[bool] = None,
        preview_only: bool = True,
        confirm: bool = False,
    ) -> Dict[str, Any]:
        """Open the collection handle and run cascade cleanup (async)."""
        handle = await self.open_collection(
            KBContextRequest(
                collection=collection,
                user_id=user_id,
                is_admin=is_admin if is_admin is not None else True,
                access_mode=KBAccessMode.WRITE,
                hide_missing=True,
            )
        )
        return await asyncio.to_thread(
            handle.cleanup_cascade,
            doc_id,
            scope,
            new_parse_hash=new_parse_hash,
            old_parse_hash=old_parse_hash,
            model_tag=model_tag,
            user_id=user_id,
            is_admin=is_admin,
            preview_only=preview_only,
            confirm=confirm,
        )

    def cleanup_cascade_sync(
        self,
        collection: str,
        doc_id: str,
        scope: str,
        *,
        new_parse_hash: Optional[str] = None,
        old_parse_hash: Optional[str] = None,
        model_tag: Optional[str] = None,
        user_id: Optional[int] = None,
        is_admin: Optional[bool] = None,
        preview_only: bool = True,
        confirm: bool = False,
    ) -> Dict[str, Any]:
        """Synchronous wrapper for :meth:`cleanup_cascade`."""
        return _run_in_separate_loop(
            self.cleanup_cascade(
                collection,
                doc_id,
                scope,
                new_parse_hash=new_parse_hash,
                old_parse_hash=old_parse_hash,
                model_tag=model_tag,
                user_id=user_id,
                is_admin=is_admin,
                preview_only=preview_only,
                confirm=confirm,
            )
        )

    async def cleanup_document_cascade(
        self,
        collection: str,
        doc_id: str,
        *,
        model_tag: Optional[str] = None,
        user_id: Optional[int] = None,
        is_admin: bool = True,
        preview_only: bool = True,
        confirm: bool = False,
    ) -> Dict[str, Any]:
        """Open collection handle and run document cascade cleanup (async)."""
        handle = await self.open_collection(
            KBContextRequest(
                collection=collection,
                user_id=user_id,
                is_admin=is_admin,
                access_mode=KBAccessMode.WRITE,
                hide_missing=True,
            )
        )
        return await asyncio.to_thread(
            handle.cleanup_document_cascade,
            doc_id,
            model_tag=model_tag,
            user_id=user_id,
            is_admin=is_admin,
            preview_only=preview_only,
            confirm=confirm,
        )

    def cleanup_document_cascade_sync(
        self,
        collection: str,
        doc_id: str,
        *,
        model_tag: Optional[str] = None,
        user_id: Optional[int] = None,
        is_admin: bool = True,
        preview_only: bool = True,
        confirm: bool = False,
    ) -> Dict[str, Any]:
        """Synchronous wrapper for :meth:`cleanup_document_cascade`."""
        return _run_in_separate_loop(
            self.cleanup_document_cascade(
                collection,
                doc_id,
                model_tag=model_tag,
                user_id=user_id,
                is_admin=is_admin,
                preview_only=preview_only,
                confirm=confirm,
            )
        )

    async def cleanup_parse_cascade(
        self,
        collection: str,
        doc_id: str,
        *,
        old_parse_hash: Optional[str] = None,
        new_parse_hash: Optional[str] = None,
        user_id: Optional[int] = None,
        is_admin: bool = True,
        preview_only: bool = True,
        confirm: bool = False,
    ) -> Dict[str, Any]:
        """Open collection handle and run parse cascade cleanup (async)."""
        handle = await self.open_collection(
            KBContextRequest(
                collection=collection,
                user_id=user_id,
                is_admin=is_admin,
                access_mode=KBAccessMode.WRITE,
                hide_missing=True,
            )
        )
        return await asyncio.to_thread(
            handle.cleanup_parse_cascade,
            doc_id,
            old_parse_hash=old_parse_hash,
            new_parse_hash=new_parse_hash,
            user_id=user_id,
            is_admin=is_admin,
            preview_only=preview_only,
            confirm=confirm,
        )

    def cleanup_parse_cascade_sync(
        self,
        collection: str,
        doc_id: str,
        *,
        old_parse_hash: Optional[str] = None,
        new_parse_hash: Optional[str] = None,
        user_id: Optional[int] = None,
        is_admin: bool = True,
        preview_only: bool = True,
        confirm: bool = False,
    ) -> Dict[str, Any]:
        """Synchronous wrapper for :meth:`cleanup_parse_cascade`."""
        return _run_in_separate_loop(
            self.cleanup_parse_cascade(
                collection,
                doc_id,
                old_parse_hash=old_parse_hash,
                new_parse_hash=new_parse_hash,
                user_id=user_id,
                is_admin=is_admin,
                preview_only=preview_only,
                confirm=confirm,
            )
        )

    async def cleanup_chunk_cascade(
        self,
        collection: str,
        doc_id: str,
        *,
        old_parse_hash: Optional[str] = None,
        new_parse_hash: Optional[str] = None,
        user_id: Optional[int] = None,
        is_admin: bool = True,
        preview_only: bool = True,
        confirm: bool = False,
    ) -> Dict[str, Any]:
        """Open collection handle and run chunk cascade cleanup (async)."""
        handle = await self.open_collection(
            KBContextRequest(
                collection=collection,
                user_id=user_id,
                is_admin=is_admin,
                access_mode=KBAccessMode.WRITE,
                hide_missing=True,
            )
        )
        return await asyncio.to_thread(
            handle.cleanup_chunk_cascade,
            doc_id,
            old_parse_hash=old_parse_hash,
            new_parse_hash=new_parse_hash,
            user_id=user_id,
            is_admin=is_admin,
            preview_only=preview_only,
            confirm=confirm,
        )

    def cleanup_chunk_cascade_sync(
        self,
        collection: str,
        doc_id: str,
        *,
        old_parse_hash: Optional[str] = None,
        new_parse_hash: Optional[str] = None,
        user_id: Optional[int] = None,
        is_admin: bool = True,
        preview_only: bool = True,
        confirm: bool = False,
    ) -> Dict[str, Any]:
        """Synchronous wrapper for :meth:`cleanup_chunk_cascade`."""
        return _run_in_separate_loop(
            self.cleanup_chunk_cascade(
                collection,
                doc_id,
                old_parse_hash=old_parse_hash,
                new_parse_hash=new_parse_hash,
                user_id=user_id,
                is_admin=is_admin,
                preview_only=preview_only,
                confirm=confirm,
            )
        )

    async def cleanup_embed_cascade(
        self,
        collection: str,
        doc_id: str,
        *,
        model_tag: Optional[str] = None,
        user_id: Optional[int] = None,
        is_admin: bool = True,
        preview_only: bool = True,
        confirm: bool = False,
    ) -> Dict[str, Any]:
        """Open collection handle and run embed cascade cleanup (async)."""
        handle = await self.open_collection(
            KBContextRequest(
                collection=collection,
                user_id=user_id,
                is_admin=is_admin,
                access_mode=KBAccessMode.WRITE,
                hide_missing=True,
            )
        )
        return await asyncio.to_thread(
            handle.cleanup_embed_cascade,
            doc_id,
            model_tag=model_tag,
            user_id=user_id,
            is_admin=is_admin,
            preview_only=preview_only,
            confirm=confirm,
        )

    def cleanup_embed_cascade_sync(
        self,
        collection: str,
        doc_id: str,
        *,
        model_tag: Optional[str] = None,
        user_id: Optional[int] = None,
        is_admin: bool = True,
        preview_only: bool = True,
        confirm: bool = False,
    ) -> Dict[str, Any]:
        """Synchronous wrapper for :meth:`cleanup_embed_cascade`."""
        return _run_in_separate_loop(
            self.cleanup_embed_cascade(
                collection,
                doc_id,
                model_tag=model_tag,
                user_id=user_id,
                is_admin=is_admin,
                preview_only=preview_only,
                confirm=confirm,
            )
        )

    @staticmethod
    def _normalize_collection(collection: str) -> str:
        normalized = collection.strip() if isinstance(collection, str) else ""
        if not normalized:
            raise ValueError("collection must be a non-empty string")
        return normalized

    @staticmethod
    def _normalize_access_mode(access_mode: KBAccessMode | str) -> KBAccessMode:
        if isinstance(access_mode, KBAccessMode):
            return access_mode
        try:
            return KBAccessMode(str(access_mode).strip().lower())
        except ValueError as exc:
            allowed = ", ".join(mode.value for mode in KBAccessMode)
            raise ValueError(
                f"Invalid KB access mode {access_mode!r}; choose one of: {allowed}"
            ) from exc

    @staticmethod
    def _is_missing_collection_error(collection: str, exc: ValueError) -> bool:
        message = str(exc)
        return message in {
            f"Collection '{collection}' not found",
            "Table 'collection_metadata' was not found",
        }

    @staticmethod
    def _resolve_user_scope(request: KBContextRequest) -> KBUserScope:
        scope = resolve_user_scope(user_id=request.user_id, is_admin=request.is_admin)
        return KBUserScope(user_id=scope.user_id, is_admin=bool(scope.is_admin))

    def _resolve_backend(self, collection_info: object | None) -> KBStorageBackend:
        if collection_info is None:
            return KBStorageBackend.LANCEDB

        extra_metadata = getattr(collection_info, "extra_metadata", None) or {}
        binding = extra_metadata.get(KB_STORAGE_METADATA_KEY)
        if binding is None:
            return KBStorageBackend.LANCEDB

        if isinstance(binding, str):
            return self._parse_backend(binding)

        if isinstance(binding, dict):
            raw_backend = binding.get("backend")
            if raw_backend is None or str(raw_backend).strip() == "":
                return KBStorageBackend.LANCEDB
            return self._parse_backend(str(raw_backend))

        raise ValueError(
            f"Invalid {KB_STORAGE_METADATA_KEY} binding shape: {type(binding).__name__}"
        )

    @staticmethod
    def _parse_backend(raw_backend: str) -> KBStorageBackend:
        try:
            return KBStorageBackend(raw_backend.strip().lower())
        except ValueError as exc:
            allowed = ", ".join(backend.value for backend in KBStorageBackend)
            raise ValueError(
                f"Invalid {KB_STORAGE_METADATA_KEY} backend {raw_backend!r}; "
                f"choose one of: {allowed}"
            ) from exc

    @staticmethod
    def _capabilities_for_backend(backend: KBStorageBackend) -> KBBackendCapabilities:
        if backend is KBStorageBackend.LANCEDB:
            return KBBackendCapabilities.lancedb()
        return KBBackendCapabilities.unsupported()

    # --- Rollback snapshot/restore primitives (#513 Task 7) ---

    async def capture_main_pointer_snapshot(
        self,
        collection: str,
        doc_id: str,
        step_type: str,
        model_tag: Optional[str] = None,
    ) -> "KBMainPointerSnapshot":
        """Open collection handle and capture a main-pointer snapshot (async)."""
        handle = await self.open_collection(
            KBContextRequest(
                collection=collection,
                access_mode=KBAccessMode.READ,
                hide_missing=True,
            )
        )
        return await asyncio.to_thread(
            handle.capture_main_pointer_snapshot,
            doc_id,
            step_type,
            model_tag,
        )

    def capture_main_pointer_snapshot_sync(
        self,
        collection: str,
        doc_id: str,
        step_type: str,
        model_tag: Optional[str] = None,
    ) -> "KBMainPointerSnapshot":
        """Synchronous wrapper for :meth:`capture_main_pointer_snapshot`."""
        return _run_in_separate_loop(
            self.capture_main_pointer_snapshot(collection, doc_id, step_type, model_tag)
        )

    async def restore_main_pointer_snapshot(
        self,
        snapshot: "KBMainPointerSnapshot",
        *,
        operator: Optional[str] = None,
    ) -> bool:
        """Open collection handle and restore a main-pointer snapshot (async)."""
        handle = await self.open_collection(
            KBContextRequest(
                collection=snapshot.collection,
                access_mode=KBAccessMode.WRITE,
                hide_missing=True,
            )
        )
        return await asyncio.to_thread(
            handle.restore_main_pointer_snapshot,
            snapshot,
            operator=operator,
        )

    def restore_main_pointer_snapshot_sync(
        self,
        snapshot: "KBMainPointerSnapshot",
        *,
        operator: Optional[str] = None,
    ) -> bool:
        """Synchronous wrapper for :meth:`restore_main_pointer_snapshot`."""
        return _run_in_separate_loop(
            self.restore_main_pointer_snapshot(snapshot, operator=operator)
        )

    async def capture_candidate_cleanup_snapshot(
        self,
        collection: str,
        doc_id: str,
        scope: str,
        *,
        new_parse_hash: Optional[str] = None,
        old_parse_hash: Optional[str] = None,
        model_tag: Optional[str] = None,
        user_id: Optional[int] = None,
        is_admin: Optional[bool] = None,
    ) -> "KBVersionCandidateCleanupSnapshot":
        """Open collection handle and capture a candidate-cleanup snapshot (async)."""
        handle = await self.open_collection(
            KBContextRequest(
                collection=collection,
                user_id=user_id,
                is_admin=bool(is_admin) if is_admin is not None else True,
                access_mode=KBAccessMode.WRITE,
                hide_missing=True,
            )
        )
        return await asyncio.to_thread(
            handle.capture_candidate_cleanup_snapshot,
            doc_id,
            scope,
            new_parse_hash=new_parse_hash,
            old_parse_hash=old_parse_hash,
            model_tag=model_tag,
            user_id=user_id,
            is_admin=is_admin,
        )

    def capture_candidate_cleanup_snapshot_sync(
        self,
        collection: str,
        doc_id: str,
        scope: str,
        *,
        new_parse_hash: Optional[str] = None,
        old_parse_hash: Optional[str] = None,
        model_tag: Optional[str] = None,
        user_id: Optional[int] = None,
        is_admin: Optional[bool] = None,
    ) -> "KBVersionCandidateCleanupSnapshot":
        """Synchronous wrapper for :meth:`capture_candidate_cleanup_snapshot`."""
        return _run_in_separate_loop(
            self.capture_candidate_cleanup_snapshot(
                collection,
                doc_id,
                scope,
                new_parse_hash=new_parse_hash,
                old_parse_hash=old_parse_hash,
                model_tag=model_tag,
                user_id=user_id,
                is_admin=is_admin,
            )
        )

    async def capture_status_snapshot(
        self,
        collection: str,
        doc_id: str,
        *,
        user_id: Optional[int] = None,
        is_admin: bool = True,
    ) -> List[Dict[str, Any]]:
        """Open collection handle and capture an ingestion-status snapshot (async)."""
        handle = await self.open_collection(
            KBContextRequest(
                collection=collection,
                user_id=user_id,
                is_admin=is_admin,
                access_mode=KBAccessMode.READ,
                hide_missing=True,
            )
        )
        return await asyncio.to_thread(
            handle.capture_status_snapshot,
            doc_id,
            user_id=user_id,
            is_admin=is_admin,
        )

    def capture_status_snapshot_sync(
        self,
        collection: str,
        doc_id: str,
        *,
        user_id: Optional[int] = None,
        is_admin: bool = True,
    ) -> List[Dict[str, Any]]:
        """Synchronous wrapper for :meth:`capture_status_snapshot`."""
        return _run_in_separate_loop(
            self.capture_status_snapshot(
                collection, doc_id, user_id=user_id, is_admin=is_admin
            )
        )

    async def restore_status_snapshot(
        self,
        collection: str,
        doc_id: str,
        snapshot: List[Dict[str, Any]],
        *,
        user_id: Optional[int] = None,
    ) -> None:
        """Open collection handle and restore an ingestion-status snapshot (async)."""
        handle = await self.open_collection(
            KBContextRequest(
                collection=collection,
                user_id=user_id,
                is_admin=True,
                access_mode=KBAccessMode.WRITE,
                hide_missing=True,
            )
        )
        await asyncio.to_thread(
            handle.restore_status_snapshot,
            doc_id,
            snapshot,
            user_id=user_id,
        )

    def restore_status_snapshot_sync(
        self,
        collection: str,
        doc_id: str,
        snapshot: List[Dict[str, Any]],
        *,
        user_id: Optional[int] = None,
    ) -> None:
        """Synchronous wrapper for :meth:`restore_status_snapshot`."""
        _run_in_separate_loop(
            self.restore_status_snapshot(collection, doc_id, snapshot, user_id=user_id)
        )

    async def clear_status_snapshot(
        self,
        collection: str,
        doc_id: str,
        *,
        user_id: Optional[int] = None,
        is_admin: bool = True,
    ) -> None:
        """Open collection handle and clear the ingestion-status row (async)."""
        handle = await self.open_collection(
            KBContextRequest(
                collection=collection,
                user_id=user_id,
                is_admin=is_admin,
                access_mode=KBAccessMode.WRITE,
                hide_missing=True,
            )
        )
        await asyncio.to_thread(
            handle.clear_status_snapshot,
            doc_id,
            user_id=user_id,
            is_admin=is_admin,
        )

    def clear_status_snapshot_sync(
        self,
        collection: str,
        doc_id: str,
        *,
        user_id: Optional[int] = None,
        is_admin: bool = True,
    ) -> None:
        """Synchronous wrapper for :meth:`clear_status_snapshot`."""
        _run_in_separate_loop(
            self.clear_status_snapshot(
                collection, doc_id, user_id=user_id, is_admin=is_admin
            )
        )

    async def restore_candidate_cleanup_snapshot(
        self,
        snapshot: "KBVersionCandidateCleanupSnapshot",
        *,
        cleanup_executed: bool = False,
    ) -> "KBVersionCandidateRollbackResult":
        """Open collection handle and assess rollback feasibility (async)."""
        handle = await self.open_collection(
            KBContextRequest(
                collection=snapshot.collection,
                user_id=snapshot.user_id,
                is_admin=bool(snapshot.is_admin)
                if snapshot.is_admin is not None
                else True,
                access_mode=KBAccessMode.READ,
                hide_missing=True,
            )
        )
        return await asyncio.to_thread(
            handle.restore_candidate_cleanup_snapshot,
            snapshot,
            cleanup_executed=cleanup_executed,
        )

    def restore_candidate_cleanup_snapshot_sync(
        self,
        snapshot: "KBVersionCandidateCleanupSnapshot",
        *,
        cleanup_executed: bool = False,
    ) -> "KBVersionCandidateRollbackResult":
        """Synchronous wrapper for :meth:`restore_candidate_cleanup_snapshot`."""
        return _run_in_separate_loop(
            self.restore_candidate_cleanup_snapshot(
                snapshot, cleanup_executed=cleanup_executed
            )
        )

    # --- Failed-ingest rollback orchestration (#515) ---

    def rollback_failed_ingestion_sync(
        self, request: RollbackFailedIngestionRequest
    ) -> RollbackFailedIngestionResult:
        """Run failed-ingest compensation in DOCUMENT->FILE->STATUS->SNAPSHOT order.

        Single owner of the orchestration formerly duplicated in
        ``pipelines/web_ingestion.py`` (with-operation and callbacks-only
        copies). Orchestration only: per-plane compensation mechanics arrive
        as request callbacks or saga steps already registered on
        ``request.operation``. Compensation failures are folded into the
        result; this method never raises for a failing callback.

        Sync-first by design (spec §3.1.2): the one #515 consumer,
        web_ingestion, runs its compensation callbacks synchronously on the
        pipeline thread.
        """
        if request.operation is not None:
            return self._rollback_boundaries_with_operation(request)
        return self._rollback_boundaries_callbacks_only(request)

    async def rollback_failed_ingestion(
        self, request: RollbackFailedIngestionRequest
    ) -> RollbackFailedIngestionResult:
        """Async twin (coordinator convention; first awaited in #795)."""
        return await asyncio.to_thread(self.rollback_failed_ingestion_sync, request)

    def _rollback_boundaries_with_operation(
        self, request: RollbackFailedIngestionRequest
    ) -> RollbackFailedIngestionResult:
        operation = request.operation
        assert operation is not None
        context = dict(request.rollback_context or {})
        warnings: list[str] = []
        boundary_errors: dict[str, tuple[str, ...]] = {}
        first_error: Optional[str] = None
        attempted = False
        collection = request.collection
        source = request.source
        doc_id = request.doc_id
        key_fallback = context.get("file_id") or context.get("file_path") or source

        def _fold(boundary: str, errors: tuple[BaseException, ...]) -> None:
            nonlocal first_error
            if not errors:
                return
            boundary_errors[boundary] = tuple(str(exc) for exc in errors)
            for exc in errors:
                message = (
                    f"Web rollback {boundary} compensation failed for {source}: {exc}"
                )
                logger.warning(message)
                warnings.append(message)
            if first_error is None:
                first_error = f"{boundary} boundary compensation failed: {errors[0]}"

        # DOCUMENT boundary. A successful document compensation covers the
        # cascade planes: delete_document cascades to parse/chunk/embedding,
        # and collection initialization is shared (not rolled back per page).
        document_compensation = request.document_compensation
        if document_compensation is not None:
            attempted = True

            def _compensate_document() -> None:
                callback = document_compensation(request.ingestion_result)
                result = callback()
                if inspect.isawaitable(result):
                    _close_awaitable_if_possible(result)
                    raise TypeError(
                        "Async DOCUMENT compensation callback is not supported "
                        f"for {source}"
                    )
                operation.mark_compensated_steps(
                    planes={
                        SideEffectPlane.COLLECTION,
                        SideEffectPlane.DOCUMENT,
                        SideEffectPlane.PARSE,
                        SideEffectPlane.CHUNK,
                        SideEffectPlane.EMBEDDING,
                    }
                )

            operation.record_side_effect(
                name="remove_registered_document",
                plane=SideEffectPlane.DOCUMENT,
                payload={
                    "collection": collection,
                    "url": source,
                    "doc_id": doc_id,
                    "file_id": context.get("file_id"),
                    "rollback_kind": context.get("rollback_kind"),
                },
                idempotency_key=(
                    f"document:{collection}:{doc_id}"
                    if doc_id
                    else f"document:{collection}:{key_fallback}"
                ),
                compensation=_compensate_document,
            )
            _fold(
                "DOCUMENT",
                operation.execute_compensations(
                    step_names={"remove_registered_document"},
                    planes={SideEffectPlane.DOCUMENT},
                ),
            )

        # FILE boundary. The key must match web_ingestion's ingest-time
        # registration (file:{collection}:{file_id or file_path or url}) so
        # re-registration dedupes instead of double-running the callback.
        file_compensation = request.file_compensation
        file_succeeded = file_compensation is None
        if file_compensation is not None:
            attempted = True
            operation.record_side_effect(
                name="cleanup_web_page_persistence",
                plane=SideEffectPlane.FILE,
                payload={
                    "collection": collection,
                    "url": source,
                    "file_path": context.get("file_path"),
                    "file_id": context.get("file_id"),
                    "reason": "file_compensation",
                    **context,
                },
                idempotency_key=f"file:{collection}:{key_fallback}",
                compensation=cast(Callable[[], None], file_compensation),
            )
            file_errors = operation.execute_compensations(
                step_names={"cleanup_web_page_persistence"},
                planes={SideEffectPlane.FILE},
            )
            file_succeeded = not file_errors
            _fold("FILE", file_errors)

        # STATUS boundary.
        status_compensation = request.status_compensation
        if status_compensation is not None:
            attempted = True

            def _compensate_status() -> None:
                callback = status_compensation(request.ingestion_result)
                result = callback()
                if inspect.isawaitable(result):
                    _close_awaitable_if_possible(result)
                    raise TypeError(
                        "Async STATUS compensation callback is not supported "
                        f"for {source}"
                    )

            operation.record_side_effect(
                name="clear_ingestion_status",
                plane=SideEffectPlane.STATUS,
                payload={
                    "collection": collection,
                    "url": source,
                    "doc_id": doc_id,
                    "file_id": context.get("file_id"),
                    "rollback_kind": context.get("rollback_kind"),
                },
                idempotency_key=(
                    f"status:{collection}:{doc_id}"
                    if doc_id
                    else f"status:{collection}:{key_fallback}"
                ),
                compensation=_compensate_status,
            )
            _fold(
                "STATUS",
                operation.execute_compensations(
                    step_names={"clear_ingestion_status"},
                    planes={SideEffectPlane.STATUS},
                ),
            )

        # SNAPSHOT boundary - only clean the backup when no prior boundary
        # errored AND (no FILE compensation was registered OR it succeeded).
        snapshot_compensation = request.snapshot_compensation
        if snapshot_compensation is not None:
            attempted = True
            backup_path = context.get("backup_path")
            operation.record_side_effect(
                name="cleanup_web_page_snapshot",
                plane=SideEffectPlane.SNAPSHOT,
                payload={
                    "collection": collection,
                    "url": source,
                    "backup_path": backup_path,
                    "file_id": context.get("file_id"),
                    "rollback_kind": context.get("rollback_kind"),
                },
                idempotency_key=(
                    f"snapshot:{collection}:{backup_path or key_fallback}"
                ),
                compensation=cast(Callable[[], None], snapshot_compensation),
            )
            file_registered = file_compensation is not None
            if first_error is None and (not file_registered or file_succeeded):
                _fold(
                    "SNAPSHOT",
                    operation.execute_compensations(
                        step_names={"cleanup_web_page_snapshot"},
                        planes={SideEffectPlane.SNAPSHOT},
                    ),
                )

        side_effects_may_remain = (
            bool(boundary_errors) or operation.has_uncompensated_side_effects()
        )
        ingest_status = "error"
        if request.ingestion_result is not None and request.ingestion_result.status:
            ingest_status = request.ingestion_result.status
        rollback_status = operation.infer_rollback_status(
            ingest_status,
            side_effects_may_remain=side_effects_may_remain,
        )
        if side_effects_may_remain:
            status = "incomplete"
        elif attempted:
            status = "complete"
        else:
            status = "not_needed"
        rollback_complete = first_error is None and not side_effects_may_remain
        assert not (side_effects_may_remain and status in ("not_needed", "complete"))
        assert not (side_effects_may_remain and rollback_complete)
        return RollbackFailedIngestionResult(
            status=status,
            rollback_status=rollback_status,
            rollback_complete=rollback_complete,
            side_effects_may_remain=side_effects_may_remain,
            first_error=first_error,
            boundary_errors=boundary_errors,
            warnings=tuple(warnings),
        )

    def _rollback_boundaries_callbacks_only(
        self, request: RollbackFailedIngestionRequest
    ) -> RollbackFailedIngestionResult:
        """Callback-only path (no saga): same order and SNAPSHOT gate."""
        warnings: list[str] = []
        boundary_errors: dict[str, tuple[str, ...]] = {}
        first_error: Optional[str] = None
        attempted = False
        source = request.source

        def _fold(boundary: str, exc: BaseException) -> None:
            nonlocal first_error
            boundary_errors[boundary] = boundary_errors.get(boundary, ()) + (str(exc),)
            message = f"Web rollback {boundary} compensation failed for {source}: {exc}"
            logger.warning(message)
            warnings.append(message)
            if first_error is None:
                first_error = f"{boundary} boundary compensation failed: {exc}"

        if request.document_compensation is not None:
            attempted = True
            try:
                callback = request.document_compensation(request.ingestion_result)
                result = callback()
                if inspect.isawaitable(result):
                    _close_awaitable_if_possible(result)
                    raise TypeError(
                        "Async DOCUMENT compensation callback is not supported "
                        f"for {source}"
                    )
            except Exception as exc:  # noqa: BLE001 - fold into the result
                _fold("DOCUMENT", exc)

        file_succeeded = False
        if request.file_compensation is not None:
            attempted = True
            try:
                result = request.file_compensation()
                if inspect.isawaitable(result):
                    _close_awaitable_if_possible(result)
                    raise TypeError(
                        "Async FILE compensation callback is not supported "
                        f"for {source}"
                    )
            except Exception as exc:  # noqa: BLE001 - fold into the result
                _fold("FILE", exc)
            else:
                file_succeeded = True

        if request.status_compensation is not None:
            attempted = True
            try:
                callback = request.status_compensation(request.ingestion_result)
                result = callback()
                if inspect.isawaitable(result):
                    _close_awaitable_if_possible(result)
                    raise TypeError(
                        "Async STATUS compensation callback is not supported "
                        f"for {source}"
                    )
            except Exception as exc:  # noqa: BLE001 - fold into the result
                _fold("STATUS", exc)

        if request.snapshot_compensation is not None:
            attempted = True
            file_registered = request.file_compensation is not None
            if first_error is None and (not file_registered or file_succeeded):
                try:
                    result = request.snapshot_compensation()
                    if inspect.isawaitable(result):
                        _close_awaitable_if_possible(result)
                        raise TypeError(
                            "Async SNAPSHOT compensation callback is not supported "
                            f"for {source}"
                        )
                except Exception as exc:  # noqa: BLE001 - fold into the result
                    _fold("SNAPSHOT", exc)

        if boundary_errors:
            status, rollback_status = "incomplete", RollbackStatus.INCOMPLETE
        elif attempted:
            status, rollback_status = "complete", RollbackStatus.COMPLETE
        else:
            status, rollback_status = "not_needed", RollbackStatus.NOT_NEEDED
        return RollbackFailedIngestionResult(
            status=status,
            rollback_status=rollback_status,
            rollback_complete=first_error is None,
            side_effects_may_remain=bool(boundary_errors),
            first_error=first_error,
            boundary_errors=boundary_errors,
            warnings=tuple(warnings),
        )

    # --- Rollback vector cleanup router (#515) ---

    async def cleanup_vectors_for_operation(
        self,
        collection: str,
        *,
        doc_id: Optional[str] = None,
        parse_hash: Optional[str] = None,
        chunk_ids: Optional[Sequence[str]] = None,
        model_tag: Optional[str] = None,
        user_id: Optional[int] = None,
        is_admin: bool = False,
        preview_only: bool = True,
        confirm: bool = False,
    ) -> KBVectorStorageCleanupResult:
        """Open the collection handle and run rollback vector cleanup (async)."""
        handle = await self.open_collection(
            KBContextRequest(
                collection=collection,
                user_id=user_id,
                is_admin=is_admin,
                access_mode=KBAccessMode.WRITE,
                hide_missing=True,
            )
        )
        return await asyncio.to_thread(
            handle.cleanup_embeddings_for_operation,
            doc_id=doc_id,
            parse_hash=parse_hash,
            chunk_ids=chunk_ids,
            model_tag=model_tag,
            preview_only=preview_only,
            confirm=confirm,
        )

    def cleanup_vectors_for_operation_sync(
        self,
        collection: str,
        *,
        doc_id: Optional[str] = None,
        parse_hash: Optional[str] = None,
        chunk_ids: Optional[Sequence[str]] = None,
        model_tag: Optional[str] = None,
        user_id: Optional[int] = None,
        is_admin: bool = False,
        preview_only: bool = True,
        confirm: bool = False,
    ) -> KBVectorStorageCleanupResult:
        """Synchronous wrapper for :meth:`cleanup_vectors_for_operation`."""
        return _run_in_separate_loop(
            self.cleanup_vectors_for_operation(
                collection,
                doc_id=doc_id,
                parse_hash=parse_hash,
                chunk_ids=chunk_ids,
                model_tag=model_tag,
                user_id=user_id,
                is_admin=is_admin,
                preview_only=preview_only,
                confirm=confirm,
            )
        )

    def reset_compatibility_caches(self) -> None:
        """Clear coordinator-owned compatibility facade caches."""
        self._storage_shim.reset_coordinator_caches()
        self._handle_provider.reset_for_tests()


_coordinator_lock = threading.RLock()
_coordinator: Optional[KBCoordinator] = None


def get_kb_coordinator() -> KBCoordinator:
    """Return the process-global KB semantic coordinator."""
    global _coordinator
    if _coordinator is None:
        with _coordinator_lock:
            if _coordinator is None:
                _coordinator = KBCoordinator()
    return _coordinator


def reset_kb_coordinator_for_tests() -> None:
    """Reset process-global KB coordinator state for tests."""
    global _coordinator
    with _coordinator_lock:
        if _coordinator is not None:
            _coordinator.reset_compatibility_caches()
        _coordinator = None


def _run_in_separate_loop(awaitable: Coroutine[Any, Any, T]) -> T:
    """Run a coroutine from sync code, including inside an existing event loop."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(awaitable)

    if not loop.is_running():
        return asyncio.run(awaitable)

    result: Optional[T] = None
    error: Optional[BaseException] = None
    context = copy_context()

    def target() -> None:
        nonlocal result, error
        try:
            result = context.run(lambda: asyncio.run(awaitable))
        except BaseException as exc:  # noqa: BLE001 - propagate from worker thread
            error = exc

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join()

    if error is not None:
        raise error
    return result  # type: ignore[return-value]
