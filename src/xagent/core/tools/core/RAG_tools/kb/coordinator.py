"""Semantic KB coordinator skeleton."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Coroutine
from contextvars import copy_context
from typing import Any, Optional, TypeVar

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
from .collection_handle import KBHandleProvider, LanceDBCollectionHandle
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
)
from .operation_compatibility import KBOperationCompatibilityFacade
from .parse_display_compatibility import KBParseDisplayCompatibilityFacade
from .pipeline_compatibility import KBPipelineCompatibilityFacade
from .retrieval_compatibility import KBRetrievalHelperCompatibilityFacade
from .storage_shim import KBStorageShimCompatibilityFacade
from .tool_compatibility import KBToolCompatibilityFacade
from .vector_storage_compatibility import KBVectorStorageCompatibilityFacade
from .version_compatibility import KBVersionCompatibilityFacade

T = TypeVar("T")

KB_STORAGE_METADATA_KEY = "kb_storage"


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
