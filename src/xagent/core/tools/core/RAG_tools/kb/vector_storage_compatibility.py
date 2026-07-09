"""Vector storage compatibility facade."""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from .cleanup_filters import resolve_cleanup_scope
from .models import KBAccessMode, KBContextRequest, KBVectorStorageCleanupResult

if TYPE_CHECKING:
    from ..core.schemas import (
        ChunkEmbeddingData,
        EmbeddingReadResponse,
        EmbeddingWriteResponse,
    )
    from .collection_handle import LanceDBCollectionHandle
    from .coordinator import KBCoordinator
    from .storage_shim import KBStorageShimCompatibilityFacade

logger = logging.getLogger(__name__)


class KBVectorStorageCompatibilityFacade:
    """Compatibility boundary for legacy vector-storage helpers.

    Public vector-storage functions keep their historical synchronous shape.
    The facade binds coordinator-owned storage access, then delegates to the
    current vector manager implementation so model-tag routing, dimension
    checks, merge error mapping, and result models remain unchanged.
    """

    def __init__(
        self,
        coordinator: "KBCoordinator | None" = None,
        storage_shim: "KBStorageShimCompatibilityFacade | None" = None,
    ) -> None:
        self._coordinator = coordinator
        self._storage_shim = storage_shim
        # Lazily-built coordinator bound to an injected shim (see
        # _active_coordinator); cached so repeated calls reuse one instance.
        self._shim_coordinator: "KBCoordinator | None" = None

    def _active_storage_shim(self) -> "KBStorageShimCompatibilityFacade | None":
        if self._storage_shim is not None:
            return self._storage_shim
        if self._coordinator is not None:
            return self._coordinator.storage_shim
        return None

    def _active_coordinator(self) -> "KBCoordinator":
        if self._coordinator is not None:
            return self._coordinator

        from .coordinator import KBCoordinator, get_kb_coordinator

        # An injected shim without a coordinator must keep embedding storage
        # bound to that shim instead of leaking onto the process-global
        # coordinator's independent stores (mirrors the parse/chunk facade).
        if self._storage_shim is not None:
            if self._shim_coordinator is None:
                self._shim_coordinator = KBCoordinator(storage_shim=self._storage_shim)
            return self._shim_coordinator

        return get_kb_coordinator()

    def _open_collection_handle(
        self, collection: str, *, user_id: Optional[int], is_admin: bool
    ) -> "LanceDBCollectionHandle":
        """Open the collection handle that owns embedding storage (#510).

        Routed through the active coordinator so an injected shim keeps
        embedding storage bound to that shim (preserves the facade's injection
        boundary).
        """
        return self._active_coordinator().open_collection_sync(
            KBContextRequest(
                collection=collection,
                user_id=user_id,
                is_admin=is_admin,
                access_mode=KBAccessMode.WRITE,
                hide_missing=True,
            )
        )

    @contextmanager
    def _storage_context(self) -> Iterator[None]:
        storage_shim = self._active_storage_shim()
        if storage_shim is None:
            yield
            return

        from ..storage.factory import bind_storage_shim_for_current_context

        with bind_storage_shim_for_current_context(storage_shim):
            yield

    def validate_query_vector(
        self,
        query_vector: List[float],
        model_tag: Optional[str] = None,
        conn: Any = None,
        user_id: Optional[int] = None,
        is_admin: bool = False,
    ) -> None:
        # Query-vector validation is a pure, collection-independent check, so it
        # delegates to the handle-owned format validator without opening a
        # (collection-scoped) handle. ``model_tag``/``conn``/``user_id``/
        # ``is_admin`` are retained for signature parity only.
        from .collection_handle import validate_query_vector_format

        validate_query_vector_format(query_vector)

    def read_chunks_for_embedding(
        self,
        collection: str,
        doc_id: str,
        parse_hash: str,
        model: str,
        filters: Optional[Dict[str, Any]] = None,
        user_id: Optional[int] = None,
        is_admin: bool = False,
    ) -> "EmbeddingReadResponse":
        with self._storage_context():
            handle = self._open_collection_handle(
                collection, user_id=user_id, is_admin=is_admin
            )
            return handle.read_chunks_needing_embedding(
                doc_id,
                parse_hash,
                model,
                filters=filters,
                user_id=user_id,
                is_admin=is_admin,
            )

    def write_vectors_to_db(
        self,
        collection: str,
        embeddings: List["ChunkEmbeddingData"],
        create_index: bool = True,
        user_id: Optional[int] = None,
    ) -> "EmbeddingWriteResponse":
        with self._storage_context():
            # Writes carry no is_admin (mirrors register_document, which opens a
            # WRITE handle with the caller's user_id and default non-admin scope).
            handle = self._open_collection_handle(
                collection, user_id=user_id, is_admin=False
            )
            return handle.write_embeddings(
                embeddings,
                create_index=create_index,
                user_id=user_id,
            )

    def cleanup_vectors_for_document(
        self,
        *,
        collection: str,
        doc_id: str,
        model_tag: Optional[str] = None,
        user_id: Optional[int] = None,
        is_admin: Optional[bool] = None,
        preview_only: bool = True,
        confirm: bool = False,
    ) -> KBVectorStorageCleanupResult:
        """Delete or preview all vectors for one document."""
        return self.cleanup_vectors_for_operation(
            collection=collection,
            doc_id=doc_id,
            model_tag=model_tag,
            user_id=user_id,
            is_admin=is_admin,
            preview_only=preview_only,
            confirm=confirm,
        )

    def cleanup_vectors_for_chunks(
        self,
        *,
        collection: str,
        doc_id: str,
        chunk_ids: Sequence[str],
        parse_hash: Optional[str] = None,
        model_tag: Optional[str] = None,
        user_id: Optional[int] = None,
        is_admin: Optional[bool] = None,
        preview_only: bool = True,
        confirm: bool = False,
    ) -> KBVectorStorageCleanupResult:
        """Delete or preview vectors for an explicit chunk set."""
        return self.cleanup_vectors_for_operation(
            collection=collection,
            doc_id=doc_id,
            parse_hash=parse_hash,
            chunk_ids=chunk_ids,
            model_tag=model_tag,
            user_id=user_id,
            is_admin=is_admin,
            preview_only=preview_only,
            confirm=confirm,
        )

    def cleanup_vectors_for_operation(
        self,
        *,
        collection: str,
        doc_id: Optional[str] = None,
        parse_hash: Optional[str] = None,
        chunk_ids: Optional[Sequence[str]] = None,
        model_tag: Optional[str] = None,
        user_id: Optional[int] = None,
        is_admin: Optional[bool] = None,
        preview_only: bool = True,
        confirm: bool = False,
    ) -> KBVectorStorageCleanupResult:
        """Delete or preview vectors created by a failed compatibility operation.

        Normalize -> delegate -> return: scope validation/user-scope fallback
        happens here, then the coordinator opens the collection handle that
        owns the cleanup (#515).
        """
        scope = resolve_cleanup_scope(
            collection=collection,
            doc_id=doc_id,
            parse_hash=parse_hash,
            chunk_ids=chunk_ids,
            model_tag=model_tag,
            user_id=user_id,
            is_admin=is_admin,
        )
        return self._active_coordinator().cleanup_vectors_for_operation_sync(
            scope.collection,
            doc_id=scope.doc_id,
            parse_hash=scope.parse_hash,
            chunk_ids=scope.chunk_ids,
            model_tag=scope.model_tag,
            user_id=scope.user_id,
            is_admin=scope.is_admin,
            preview_only=preview_only,
            confirm=confirm,
        )
