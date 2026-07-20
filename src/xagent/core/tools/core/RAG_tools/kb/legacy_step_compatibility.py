"""Legacy step compatibility facade."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from ..core.config import (
    DEFAULT_IMAGE_CONTEXT_SIZE,
    DEFAULT_TABLE_CONTEXT_SIZE,
    DEFAULT_TIKTOKEN_ENCODING,
)
from ..core.exceptions import DocumentValidationError
from ..core.schemas import (
    ChunkStrategy,
    DenseSearchResponse,
    FusionConfig,
    HybridSearchResponse,
    ParseMethod,
    RegisterDocumentRequest,
    SparseSearchResponse,
)
from .models import KBAccessMode, KBContextRequest
from .operation_compatibility import (
    KBOperation,
    KBOperationCompatibilityFacade,
    PersistencePolicy,
    finish_owned_operation,
    record_chunk_side_effect,
    record_document_registration_side_effect,
    record_parse_side_effect,
)

if TYPE_CHECKING:
    from .collection_handle import LanceDBCollectionHandle
    from .coordinator import KBCoordinator
    from .storage_shim import KBStorageShimCompatibilityFacade


class KBLegacyStepCompatibilityFacade:
    """Compatibility boundary for legacy KB step helper functions.

    Document registration, parse, chunk, and retrieval helper modules keep their
    historical import paths and sync/async behavior. The facade provides one
    coordinator-owned storage boundary while delegating to the current helper
    implementations.
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

        # An injected shim without a coordinator must keep document lifecycle
        # calls bound to that shim instead of leaking onto the process-global
        # coordinator's independent stores. Back a dedicated coordinator with
        # the injected shim so the facade's injection boundary is preserved.
        if self._storage_shim is not None:
            if self._shim_coordinator is None:
                self._shim_coordinator = KBCoordinator(storage_shim=self._storage_shim)
            return self._shim_coordinator

        return get_kb_coordinator()

    @staticmethod
    def _build_register_request(
        *,
        collection: str,
        source_path: str,
        file_type: Optional[str],
        doc_id: Optional[str],
        uploaded_at: Optional[str],
        user_id: Optional[int],
        file_id: Optional[str],
        metadata_source_path: Optional[str],
    ) -> RegisterDocumentRequest:
        """Convert legacy register args into a semantic request.

        Parses the optional ISO8601 ``uploaded_at`` string (supporting a
        trailing ``Z``); a missing or unparsable value becomes ``None`` so the
        handle applies its default timestamp.
        """
        uploaded_at_dt: Optional[datetime] = None
        if uploaded_at:
            try:
                if uploaded_at.endswith("Z"):
                    uploaded_at_dt = datetime.fromisoformat(
                        uploaded_at.replace("Z", "+00:00")
                    )
                else:
                    uploaded_at_dt = datetime.fromisoformat(uploaded_at)
                if uploaded_at_dt.tzinfo is None:
                    uploaded_at_dt = uploaded_at_dt.replace(tzinfo=timezone.utc)
            except ValueError:
                uploaded_at_dt = None

        return RegisterDocumentRequest(
            collection=collection,
            file_id=file_id,
            source_path=source_path,
            metadata_source_path=metadata_source_path,
            file_type=file_type,
            doc_id=doc_id,
            uploaded_at=uploaded_at_dt,
            user_id=user_id,
        )

    @contextmanager
    def _operation_context(
        self, *, operation_type: str, collection: str
    ) -> Iterator[tuple[KBOperation | None, bool]]:
        operation_facade = self._active_operation_facade()
        if operation_facade is None:
            yield None, False
            return

        current_operation = operation_facade.current_operation()
        if current_operation is not None:
            yield current_operation, False
            return

        with operation_facade.start_operation(
            operation_type=operation_type,
            collection=collection,
            persistence_policy=PersistencePolicy.PRESERVE_SUCCESSFUL_CHILDREN,
        ) as operation:
            yield operation, True

    @contextmanager
    def _storage_context(self) -> Iterator[None]:
        storage_shim = self._active_storage_shim()
        if storage_shim is None:
            yield
            return

        from ..storage.factory import bind_storage_shim_for_current_context

        with bind_storage_shim_for_current_context(storage_shim):
            yield

    def _open_collection_handle(
        self,
        collection: str,
        *,
        user_id: Optional[int],
        is_admin: bool,
        access_mode: "KBAccessMode" = KBAccessMode.WRITE,
    ) -> "LanceDBCollectionHandle":
        """Open the collection handle that owns parse/chunk storage (#509).

        Routed through the active coordinator so an injected shim keeps
        parse/chunk storage bound to that shim (preserves the facade's
        injection boundary).

        ``access_mode`` defaults to WRITE so existing parse/chunk callers
        keep their behaviour without any call-site changes.  Search methods
        pass ``KBAccessMode.READ``.
        """
        return self._active_coordinator().open_collection_sync(
            KBContextRequest(
                collection=collection,
                user_id=user_id,
                is_admin=is_admin,
                access_mode=access_mode,
                hide_missing=True,
            )
        )

    def register_document(
        self,
        collection: str,
        source_path: str,
        file_type: Optional[str] = None,
        doc_id: Optional[str] = None,
        uploaded_at: Optional[str] = None,
        user_id: Optional[int] = None,
        file_id: Optional[str] = None,
        metadata_source_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Register a document row via the coordinator + collection handle.

        Converts the legacy inputs into a semantic ``RegisterDocumentRequest``,
        delegates to the coordinator (which opens the handle), and converts the
        semantic response back into the legacy dict shape. The operation context
        and document side-effect recording are preserved for rollback support.
        """
        # Preserve the legacy empty-collection contract before the coordinator
        # normalizes the collection (which would raise a bare ValueError).
        if not collection:
            raise DocumentValidationError("Collection name cannot be empty")

        request = self._build_register_request(
            collection=collection,
            source_path=source_path,
            file_type=file_type,
            doc_id=doc_id,
            uploaded_at=uploaded_at,
            user_id=user_id,
            file_id=file_id,
            metadata_source_path=metadata_source_path,
        )

        with self._operation_context(
            operation_type="legacy_register_document", collection=collection
        ) as (operation, owns_operation):
            response = self._active_coordinator().register_document_sync(request)
            result: Dict[str, Any] = response.model_dump()
            self._record_document_side_effect(
                operation,
                collection=collection,
                source_path=source_path,
                file_id=file_id,
                user_id=user_id,
                result=result,
            )
            self._finish_owned_operation(operation, owns_operation, status="success")
            return result

    def get_document(self, db_dir: str, collection: str, doc_id: str) -> Optional[Any]:
        """Load a document row via the handle (legacy raw-dict shape or None).

        ``db_dir`` is accepted for backward compatibility and ignored. The
        anonymous default scope reproduces the legacy ``get_document`` behavior.
        """
        detail = self._active_coordinator().load_document_sync(collection, doc_id)
        return detail.to_legacy_dict() if detail is not None else None

    def list_documents(
        self, db_dir: str, collection: str, limit: int = 100
    ) -> list[Dict[str, Any]]:
        """List document rows via the handle (legacy raw-dict list).

        ``db_dir`` is accepted for backward compatibility and ignored. Listing
        uses admin scope to mirror the legacy file-level helper.
        """
        result = self._active_coordinator().list_document_records_sync(
            collection, user_id=None, is_admin=True, limit=limit
        )
        return result.to_legacy_dicts()

    def parse_document(
        self,
        collection: str,
        doc_id: str,
        parse_method: ParseMethod,
        params: Optional[Dict[str, Any]] = None,
        user_id: Optional[int] = None,
        is_admin: bool = False,
        progress_callback: Optional[Any] = None,
        source_path_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        from ..parse.parse_document import _parse_document_impl

        with self._operation_context(
            operation_type="legacy_parse_document", collection=collection
        ) as (operation, owns_operation):
            with self._storage_context():
                handle = self._open_collection_handle(
                    collection, user_id=user_id, is_admin=is_admin
                )
                result = _parse_document_impl(
                    collection=collection,
                    doc_id=doc_id,
                    parse_method=parse_method,
                    params=params,
                    user_id=user_id,
                    is_admin=is_admin,
                    progress_callback=progress_callback,
                    source_path_override=source_path_override,
                    handle=handle,
                )
            self._record_parse_side_effect(
                operation, collection=collection, doc_id=doc_id, result=result
            )
            self._finish_owned_operation(operation, owns_operation, status="success")
            return result

    def chunk_document(
        self,
        collection: str,
        doc_id: str,
        parse_hash: str,
        chunk_strategy: ChunkStrategy = ChunkStrategy.RECURSIVE,
        chunk_size: Optional[int] = 1000,
        chunk_overlap: int = 200,
        headers_to_split_on: Optional[List[Tuple[str, str]]] = None,
        separators: Optional[List[str]] = None,
        use_token_count: bool = False,
        tiktoken_encoding: str = DEFAULT_TIKTOKEN_ENCODING,
        enable_protected_content: bool = True,
        protected_patterns: Optional[List[str]] = None,
        table_context_size: int = DEFAULT_TABLE_CONTEXT_SIZE,
        image_context_size: int = DEFAULT_IMAGE_CONTEXT_SIZE,
        user_id: Optional[int] = None,
        is_admin: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        from ..chunk.chunk_document import _chunk_document_impl

        with self._operation_context(
            operation_type="legacy_chunk_document", collection=collection
        ) as (operation, owns_operation):
            with self._storage_context():
                handle = self._open_collection_handle(
                    collection, user_id=user_id, is_admin=is_admin
                )
                result = _chunk_document_impl(
                    collection=collection,
                    doc_id=doc_id,
                    parse_hash=parse_hash,
                    chunk_strategy=chunk_strategy,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    headers_to_split_on=headers_to_split_on,
                    separators=separators,
                    use_token_count=use_token_count,
                    tiktoken_encoding=tiktoken_encoding,
                    enable_protected_content=enable_protected_content,
                    protected_patterns=protected_patterns,
                    table_context_size=table_context_size,
                    image_context_size=image_context_size,
                    user_id=user_id,
                    is_admin=is_admin,
                    handle=handle,
                    **kwargs,
                )
            self._record_chunk_side_effect(
                operation,
                collection=collection,
                doc_id=doc_id,
                parse_hash=parse_hash,
                result=result,
            )
            self._finish_owned_operation(operation, owns_operation, status="success")
            return result

    def chunk_recursive(
        self,
        collection: str,
        doc_id: str,
        parse_hash: str,
        chunk_size: Optional[int] = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        from ..chunk.chunk_document import _chunk_recursive_impl

        with self._operation_context(
            operation_type="legacy_chunk_recursive", collection=collection
        ) as (operation, owns_operation):
            with self._storage_context():
                handle = self._open_collection_handle(
                    collection,
                    user_id=kwargs.get("user_id"),
                    is_admin=bool(kwargs.get("is_admin", False)),
                )
                result = _chunk_recursive_impl(
                    collection=collection,
                    doc_id=doc_id,
                    parse_hash=parse_hash,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    separators=separators,
                    handle=handle,
                    **kwargs,
                )
            self._record_chunk_side_effect(
                operation,
                collection=collection,
                doc_id=doc_id,
                parse_hash=parse_hash,
                result=result,
            )
            self._finish_owned_operation(operation, owns_operation, status="success")
            return result

    def chunk_markdown(
        self,
        collection: str,
        doc_id: str,
        parse_hash: str,
        chunk_size: Optional[int] = 1200,
        chunk_overlap: int = 200,
        headers_to_split_on: Optional[List[Tuple[str, str]]] = None,
        separators: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        from ..chunk.chunk_document import _chunk_markdown_impl

        with self._operation_context(
            operation_type="legacy_chunk_markdown", collection=collection
        ) as (operation, owns_operation):
            with self._storage_context():
                handle = self._open_collection_handle(
                    collection,
                    user_id=kwargs.get("user_id"),
                    is_admin=bool(kwargs.get("is_admin", False)),
                )
                result = _chunk_markdown_impl(
                    collection=collection,
                    doc_id=doc_id,
                    parse_hash=parse_hash,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    headers_to_split_on=headers_to_split_on,
                    separators=separators,
                    handle=handle,
                    **kwargs,
                )
            self._record_chunk_side_effect(
                operation,
                collection=collection,
                doc_id=doc_id,
                parse_hash=parse_hash,
                result=result,
            )
            self._finish_owned_operation(operation, owns_operation, status="success")
            return result

    def chunk_fixed_size(
        self,
        collection: str,
        doc_id: str,
        parse_hash: str,
        chunk_size: Optional[int] = 1000,
        chunk_overlap: int = 0,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        from ..chunk.chunk_document import _chunk_fixed_size_impl

        with self._operation_context(
            operation_type="legacy_chunk_fixed_size", collection=collection
        ) as (operation, owns_operation):
            with self._storage_context():
                handle = self._open_collection_handle(
                    collection,
                    user_id=kwargs.get("user_id"),
                    is_admin=bool(kwargs.get("is_admin", False)),
                )
                result = _chunk_fixed_size_impl(
                    collection=collection,
                    doc_id=doc_id,
                    parse_hash=parse_hash,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    handle=handle,
                    **kwargs,
                )
            self._record_chunk_side_effect(
                operation,
                collection=collection,
                doc_id=doc_id,
                parse_hash=parse_hash,
                result=result,
            )
            self._finish_owned_operation(operation, owns_operation, status="success")
            return result

    def _record_document_side_effect(
        self,
        operation: KBOperation | None,
        *,
        collection: str,
        source_path: str,
        file_id: Optional[str],
        user_id: Optional[int],
        result: Dict[str, Any],
    ) -> None:
        doc_id = result.get("doc_id")
        if operation is None or not doc_id:
            return
        record_document_registration_side_effect(
            operation,
            collection=collection,
            doc_id=doc_id,
            created=result.get("created"),
            source_path=source_path,
            file_id=file_id,
            user_id=user_id,
        )

    def _record_parse_side_effect(
        self,
        operation: KBOperation | None,
        *,
        collection: str,
        doc_id: str,
        result: Dict[str, Any],
    ) -> None:
        if operation is None or result.get("written") is False:
            return
        parse_hash = result.get("parse_hash")
        if not parse_hash:
            return
        record_parse_side_effect(
            operation, collection=collection, doc_id=doc_id, parse_hash=parse_hash
        )

    def _record_chunk_side_effect(
        self,
        operation: KBOperation | None,
        *,
        collection: str,
        doc_id: str,
        parse_hash: str,
        result: Dict[str, Any],
    ) -> None:
        if operation is None or result.get("created") is False:
            return
        chunk_count = int(result.get("chunk_count", 0) or 0)
        if chunk_count <= 0:
            return
        record_chunk_side_effect(
            operation,
            collection=collection,
            doc_id=doc_id,
            parse_hash=parse_hash,
            chunk_count=chunk_count,
        )

    @staticmethod
    def _finish_owned_operation(
        operation: KBOperation | None,
        owns_operation: bool,
        *,
        status: str,
    ) -> None:
        finish_owned_operation(operation, owns_operation, status=status)

    def search_dense(
        self,
        collection: str,
        model_tag: str,
        query_vector: List[float],
        *,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        readonly: bool = False,
        nprobes: Optional[int] = None,
        refine_factor: Optional[int] = None,
        user_id: Optional[int] = None,
        is_admin: bool = False,
    ) -> DenseSearchResponse:
        with self._storage_context():
            handle = self._open_collection_handle(
                collection,
                user_id=user_id,
                is_admin=is_admin,
                access_mode=KBAccessMode.READ,
            )
            return handle.search_dense(
                model_tag,
                query_vector,
                top_k=top_k,
                filters=filters,
                readonly=readonly,
                nprobes=nprobes,
                refine_factor=refine_factor,
                user_id=user_id,
                is_admin=is_admin,
            )

    async def search_dense_async(
        self,
        collection: str,
        model_tag: str,
        query_vector: List[float],
        *,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        readonly: bool = False,
        nprobes: Optional[int] = None,
        refine_factor: Optional[int] = None,
        user_id: Optional[int] = None,
        is_admin: bool = False,
    ) -> DenseSearchResponse:
        with self._storage_context():
            handle = await self._active_coordinator().open_collection(
                KBContextRequest(
                    collection=collection,
                    user_id=user_id,
                    is_admin=is_admin,
                    access_mode=KBAccessMode.READ,
                    hide_missing=True,
                )
            )
            return await handle.search_dense_async(
                model_tag,
                query_vector,
                top_k=top_k,
                filters=filters,
                readonly=readonly,
                nprobes=nprobes,
                refine_factor=refine_factor,
                user_id=user_id,
                is_admin=is_admin,
            )

    def search_sparse(
        self,
        collection: str,
        model_tag: str,
        query_text: str,
        *,
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
        readonly: bool = False,
        nprobes: Optional[int] = None,
        refine_factor: Optional[int] = None,
        user_id: Optional[int] = None,
        is_admin: bool = False,
    ) -> SparseSearchResponse:
        with self._storage_context():
            handle = self._open_collection_handle(
                collection,
                user_id=user_id,
                is_admin=is_admin,
                access_mode=KBAccessMode.READ,
            )
            return handle.search_sparse(
                model_tag,
                query_text,
                top_k=top_k,
                filters=filters,
                readonly=readonly,
                nprobes=nprobes,
                refine_factor=refine_factor,
                user_id=user_id,
                is_admin=is_admin,
            )

    async def search_sparse_async(
        self,
        collection: str,
        model_tag: str,
        query_text: str,
        *,
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
        readonly: bool = False,
        nprobes: Optional[int] = None,
        refine_factor: Optional[int] = None,
        user_id: Optional[int] = None,
        is_admin: bool = False,
    ) -> SparseSearchResponse:
        with self._storage_context():
            handle = await self._active_coordinator().open_collection(
                KBContextRequest(
                    collection=collection,
                    user_id=user_id,
                    is_admin=is_admin,
                    access_mode=KBAccessMode.READ,
                    hide_missing=True,
                )
            )
            return await handle.search_sparse_async(
                model_tag,
                query_text,
                top_k=top_k,
                filters=filters,
                readonly=readonly,
                nprobes=nprobes,
                refine_factor=refine_factor,
                user_id=user_id,
                is_admin=is_admin,
            )

    def search_hybrid(
        self,
        collection: str,
        model_tag: str,
        query_text: str,
        query_vector: List[float],
        *,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        fusion_config: Optional[FusionConfig] = None,
        readonly: bool = False,
        nprobes: Optional[int] = None,
        refine_factor: Optional[int] = None,
        user_id: Optional[int] = None,
        is_admin: bool = False,
    ) -> HybridSearchResponse:
        with self._storage_context():
            handle = self._open_collection_handle(
                collection,
                user_id=user_id,
                is_admin=is_admin,
                access_mode=KBAccessMode.READ,
            )
            return handle.search_hybrid(
                model_tag,
                query_text,
                query_vector,
                top_k=top_k,
                filters=filters,
                fusion_config=fusion_config,
                readonly=readonly,
                nprobes=nprobes,
                refine_factor=refine_factor,
                user_id=user_id,
                is_admin=is_admin,
            )
