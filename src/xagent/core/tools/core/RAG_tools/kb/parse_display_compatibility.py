"""Parse display compatibility facade."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from .models import KBAccessMode, KBContextRequest

if TYPE_CHECKING:
    from ..core.schemas import ParsedElementDisplay
    from .collection_handle import LanceDBCollectionHandle
    from .coordinator import KBCoordinator
    from .storage_shim import KBStorageShimCompatibilityFacade


class KBParseDisplayCompatibilityFacade:
    """Compatibility boundary for legacy parse display helpers.

    Parse display helpers are synchronous read-only APIs. The facade binds
    coordinator-owned storage access while preserving the legacy helper names,
    signatures, tuple shapes, and conversion behavior.
    """

    def __init__(
        self,
        coordinator: KBCoordinator | None = None,
        storage_shim: KBStorageShimCompatibilityFacade | None = None,
    ) -> None:
        self._coordinator = coordinator
        self._storage_shim = storage_shim
        # Lazily-built coordinator bound to an injected shim (see
        # _active_coordinator); cached so repeated calls reuse one instance.
        self._shim_coordinator: KBCoordinator | None = None

    def _active_storage_shim(self) -> KBStorageShimCompatibilityFacade | None:
        if self._storage_shim is not None:
            return self._storage_shim
        if self._coordinator is not None:
            return self._coordinator.storage_shim
        return None

    def _active_coordinator(self) -> KBCoordinator:
        if self._coordinator is not None:
            return self._coordinator

        from .coordinator import KBCoordinator, get_kb_coordinator

        # An injected shim without a coordinator must keep parse-display reads
        # bound to that shim instead of the process-global coordinator's
        # independent stores (mirrors the legacy step facade boundary).
        if self._storage_shim is not None:
            if self._shim_coordinator is None:
                self._shim_coordinator = KBCoordinator(storage_shim=self._storage_shim)
            return self._shim_coordinator

        return get_kb_coordinator()

    def _open_collection_handle(
        self, collection: str, *, user_id: int | None, is_admin: bool
    ) -> "LanceDBCollectionHandle":
        """Open the read-only collection handle that owns parse storage (#509)."""
        return self._active_coordinator().open_collection_sync(
            KBContextRequest(
                collection=collection,
                user_id=user_id,
                is_admin=is_admin,
                access_mode=KBAccessMode.READ,
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

    def reconstruct_parse_result_from_db(
        self,
        collection: str,
        doc_id: str,
        parse_hash: Optional[str] = None,
        user_id: Optional[int] = None,
        is_admin: bool = False,
    ) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        from ..parse.parse_display import _reconstruct_parse_result_from_db_impl

        with self._storage_context():
            handle = self._open_collection_handle(
                collection, user_id=user_id, is_admin=is_admin
            )
            return _reconstruct_parse_result_from_db_impl(
                collection,
                doc_id,
                parse_hash=parse_hash,
                user_id=user_id,
                is_admin=is_admin,
                handle=handle,
            )

    def paginate_parse_results(
        self,
        elements: List[Dict[str, Any]],
        page: int = 1,
        page_size: int = 20,
    ) -> Tuple[List[ParsedElementDisplay], Dict[str, Any]]:
        from ..parse.parse_display import _paginate_parse_results_impl

        return _paginate_parse_results_impl(elements, page=page, page_size=page_size)
