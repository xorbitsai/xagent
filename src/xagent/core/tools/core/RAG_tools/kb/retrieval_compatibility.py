"""Retrieval helper compatibility facade."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, List, Optional

from ..core.schemas import SearchResult

if TYPE_CHECKING:
    from .coordinator import KBCoordinator
    from .storage_shim import KBStorageShimCompatibilityFacade


class KBRetrievalHelperCompatibilityFacade:
    """Compatibility boundary for low-level retrieval helper functions.

    Retrieval helpers keep their historical import paths, sync/async shapes,
    tuple return contracts, filter parsing, score conversion, index advice, and
    prompt-context formatting. The facade gives coordinator-owned code one
    retrieval boundary while delegating to the current helper implementations.
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

    def format_search_results_for_llm(
        self,
        search_results: List[SearchResult],
        top_k: Optional[int] = None,
        include_metadata: bool = False,
        separator: str = "\n---\n",
    ) -> str:
        from ..retrieval.format_context import _format_search_results_for_llm_impl

        return _format_search_results_for_llm_impl(
            search_results,
            top_k=top_k,
            include_metadata=include_metadata,
            separator=separator,
        )
