"""Core management compatibility facade for KB operations."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from ..core.schemas import (
    CollectionOperationResult,
    DocumentListResult,
    DocumentOperationResult,
    DocumentStatsResult,
    ListCollectionsResult,
)

if TYPE_CHECKING:
    from .coordinator import KBCoordinator
    from .storage_shim import KBStorageShimCompatibilityFacade


class KBCoreManagementCompatibilityFacade:
    """Compatibility boundary for legacy management module functions.

    Public management modules keep their historical import paths and signatures,
    while coordinator-owned code gets one stable surface for list, delete,
    retry/cancel, and ingestion-status operations.
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

    @contextmanager
    def _storage_context(self) -> Iterator[None]:
        storage_shim = self._active_storage_shim()
        if storage_shim is None:
            yield
            return

        from ..storage.factory import bind_storage_shim_for_current_context

        with bind_storage_shim_for_current_context(storage_shim):
            yield

    async def list_collections(
        self,
        user_id: Optional[int] = None,
        is_admin: Optional[bool] = None,
        force_realtime: bool = False,
    ) -> ListCollectionsResult:
        from ..management import collections as management_collections

        with self._storage_context():
            return await management_collections._list_collections_impl(
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
        from ..management import collections as management_collections

        with self._storage_context():
            return management_collections._list_documents_impl(
                collection=collection,
                user_id=user_id,
                is_admin=is_admin,
            )

    def get_document_stats(
        self,
        collection: str,
        doc_id: str,
        model_tag: Optional[str] = None,
        user_id: Optional[int] = None,
        is_admin: bool = False,
    ) -> DocumentStatsResult:
        from ..management import collections as management_collections

        with self._storage_context():
            return management_collections._get_document_stats_impl(
                collection=collection,
                doc_id=doc_id,
                model_tag=model_tag,
                user_id=user_id,
                is_admin=is_admin,
            )

    def delete_document(
        self,
        collection: str,
        doc_id: str,
        user_id: int,
        is_admin: bool = False,
    ) -> DocumentOperationResult:
        from ..management import collections as management_collections

        with self._storage_context():
            return management_collections._delete_document_impl(
                collection=collection,
                doc_id=doc_id,
                user_id=user_id,
                is_admin=is_admin,
            )

    def delete_collection(
        self,
        collection: str,
        user_id: Optional[int] = None,
        is_admin: bool = False,
    ) -> CollectionOperationResult:
        if self._coordinator is not None:
            from .coordinator import _run_in_separate_loop

            return _run_in_separate_loop(
                self.delete_collection_async(
                    collection=collection,
                    user_id=user_id,
                    is_admin=is_admin,
                )
            )

        from ..management import collections as management_collections

        with self._storage_context():
            return management_collections._delete_collection_impl(
                collection=collection,
                user_id=user_id,
                is_admin=is_admin,
            )

    async def delete_collection_async(
        self,
        collection: str,
        user_id: Optional[int] = None,
        is_admin: bool = False,
        doc_ids: Optional[List[str]] = None,
        delete_orphaned_metadata: bool = True,
    ) -> CollectionOperationResult:
        """Async delete routed through coordinator when available.

        When a coordinator is present, delegates to
        :meth:`KBCoordinator.delete_collection` so the operation goes through
        the handle boundary. Falls back to :func:`_delete_collection_impl` for
        legacy callers without a coordinator.
        """
        if self._coordinator is not None:
            return await self._coordinator.delete_collection(
                collection=collection,
                user_id=user_id,
                is_admin=is_admin,
                doc_ids=doc_ids,
                delete_orphaned_metadata=delete_orphaned_metadata,
            )

        import asyncio

        from ..management import collections as management_collections

        with self._storage_context():
            return await asyncio.to_thread(
                management_collections._delete_collection_impl,
                collection=collection,
                user_id=user_id,
                is_admin=is_admin,
            )

    def retry_document(
        self,
        collection: str,
        doc_id: str,
        user_id: int,
        is_admin: bool = False,
    ) -> DocumentOperationResult:
        from ..management import collections as management_collections

        with self._storage_context():
            return management_collections._retry_document_impl(
                collection=collection,
                doc_id=doc_id,
                user_id=user_id,
                is_admin=is_admin,
            )

    def cancel_document(
        self,
        collection: str,
        doc_id: str,
        user_id: int,
        is_admin: bool = False,
        reason: Optional[str] = None,
    ) -> DocumentOperationResult:
        from ..management import collections as management_collections

        with self._storage_context():
            return management_collections._cancel_document_impl(
                collection=collection,
                doc_id=doc_id,
                user_id=user_id,
                is_admin=is_admin,
                reason=reason,
            )

    def cancel_collection(
        self,
        collection: str,
        reason: Optional[str] = None,
        user_id: Optional[int] = None,
        is_admin: bool = False,
    ) -> CollectionOperationResult:
        from ..management import collections as management_collections

        with self._storage_context():
            return management_collections._cancel_collection_impl(
                collection=collection,
                reason=reason,
                user_id=user_id,
                is_admin=is_admin,
            )

    def get_document_status(self, collection: str, doc_id: str) -> Dict[str, Any]:
        from ..management import collections as management_collections

        with self._storage_context():
            return management_collections._get_document_status_impl(
                collection=collection,
                doc_id=doc_id,
            )

    def write_ingestion_status(
        self,
        collection: str,
        doc_id: str,
        *,
        status: str,
        message: Optional[str] = None,
        parse_hash: Optional[str] = None,
        user_id: Optional[int] = None,
    ) -> None:
        from ..management import status as management_status

        with self._storage_context():
            management_status._write_ingestion_status_impl(
                collection=collection,
                doc_id=doc_id,
                status=status,
                message=message,
                parse_hash=parse_hash,
                user_id=user_id,
            )

    def load_ingestion_status(
        self,
        collection: Optional[str] = None,
        doc_id: Optional[str] = None,
        user_id: Optional[int] = None,
        is_admin: bool = False,
    ) -> List[Dict[str, Any]]:
        from ..management import status as management_status

        with self._storage_context():
            return management_status._load_ingestion_status_impl(
                collection=collection,
                doc_id=doc_id,
                user_id=user_id,
                is_admin=is_admin,
            )

    def clear_ingestion_status(
        self,
        collection: str,
        doc_id: str,
        user_id: Optional[int] = None,
        is_admin: bool = False,
    ) -> None:
        from ..management import status as management_status

        with self._storage_context():
            management_status._clear_ingestion_status_impl(
                collection=collection,
                doc_id=doc_id,
                user_id=user_id,
                is_admin=is_admin,
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
        from ..management import status as management_status

        with self._storage_context():
            await management_status._write_ingestion_status_async_impl(
                collection=collection,
                doc_id=doc_id,
                status=status,
                message=message,
                parse_hash=parse_hash,
                user_id=user_id,
            )

    async def load_ingestion_status_async(
        self,
        collection: Optional[str] = None,
        doc_id: Optional[str] = None,
        user_id: Optional[int] = None,
        is_admin: bool = False,
    ) -> List[Dict[str, Any]]:
        from ..management import status as management_status

        with self._storage_context():
            return await management_status._load_ingestion_status_async_impl(
                collection=collection,
                doc_id=doc_id,
                user_id=user_id,
                is_admin=is_admin,
            )

    async def clear_ingestion_status_async(
        self,
        collection: str,
        doc_id: str,
        user_id: Optional[int] = None,
        is_admin: bool = False,
    ) -> None:
        from ..management import status as management_status

        with self._storage_context():
            await management_status._clear_ingestion_status_async_impl(
                collection=collection,
                doc_id=doc_id,
                user_id=user_id,
                is_admin=is_admin,
            )
