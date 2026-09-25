"""A published memory store that can be revoked after it has been handed out.

Revoking the manager's publication is not enough on its own. ``AgentService``
keeps the store it was built with in ``self.memory``, and so do its execution
adapter and its execution-scoped memory tools. The agent cache's hit path
re-checks owner and scope invariants only, never memory policy, so after an
authority vector-identity change a cache hit would keep reading and writing
through the adapter built for the *previous* vector space while the manager
reports ``restart_required``.

Invalidating the agent cache would not fix that: it is a far larger surface,
and it cannot reach an execution that is already in flight. So the manager
publishes this proxy instead of the raw store. The proxy captures the
publication generation it was published under, and every memory operation
checks that generation against the manager's current one before touching
storage. When the manager establishes or revokes a publication the generation
moves, and every reference handed out under the old one fails closed -- in a
cached agent, in a memory tool, and mid-execution -- by raising
:class:`MemoryUnavailableError` carrying the manager's current status.

The proxy is the outermost wrapper, so ``UserIsolatedMemoryStore`` keeps
working underneath it unchanged. Callers that need the storage adapter behind
the wrappers (``get_store_info`` reporting ``store_type``) go through
:func:`unwrap_memory_store`.
"""

from __future__ import annotations

from typing import Any, List, Optional, Protocol

from ..core.memory.base import MemoryStore
from ..core.memory.core import MemoryNote, MemoryResponse
from .memory_lifecycle import MemoryLifecycleStatus, MemoryUnavailableError


class PublicationGenerationSource(Protocol):
    """The slice of the manager this proxy depends on."""

    def publication_generation(self) -> int: ...

    def published_status(self) -> MemoryLifecycleStatus: ...


class RevocableMemoryStore(MemoryStore):
    """Delegates to ``base_store`` only while its publication is still current."""

    def __init__(
        self,
        base_store: MemoryStore,
        source: PublicationGenerationSource,
        generation: int,
    ) -> None:
        self._base_store = base_store
        self._source = source
        self._generation = generation

    # -- revocation ------------------------------------------------------

    def _live(self) -> MemoryStore:
        """Return the delegate, or fail closed if this reference is revoked.

        Deliberately lock-free: this runs on every memory operation, and the
        manager's lock is held across an authority read, so taking it here
        would queue ordinary memory traffic behind a database round trip. The
        manager updates its status *before* it moves the generation, so a
        reader that observes the new generation also observes the status that
        explains it.
        """
        if self._source.publication_generation() != self._generation:
            raise MemoryUnavailableError(self._source.published_status())
        return self._base_store

    @property
    def revoked(self) -> bool:
        """Whether this reference has been superseded."""
        return self._source.publication_generation() != self._generation

    # -- MemoryStore surface ---------------------------------------------

    def add(self, note: MemoryNote) -> MemoryResponse:
        return self._live().add(note)

    def get(self, note_id: str) -> MemoryResponse:
        return self._live().get(note_id)

    def update(self, note: MemoryNote) -> MemoryResponse:
        return self._live().update(note)

    def delete(self, note_id: str) -> MemoryResponse:
        return self._live().delete(note_id)

    def search(
        self,
        query: str,
        k: int = 5,
        filters: Optional[dict[str, Any]] = None,
        similarity_threshold: Optional[float] = None,
    ) -> List[MemoryNote]:
        return self._live().search(
            query=query,
            k=k,
            filters=filters,
            similarity_threshold=similarity_threshold,
        )

    def clear(self) -> None:
        self._live().clear()

    def list_all(self, filters: Optional[dict[str, Any]] = None) -> List[MemoryNote]:
        return self._live().list_all(filters)

    def get_stats(self) -> dict[str, Any]:
        return self._live().get_stats()

    def delete_by_scope_dimension(self, dim_key: str, value: Any) -> MemoryResponse:
        return self._live().delete_by_scope_dimension(dim_key, value)

    def list_scope_dimension_values(self, dim_key: str) -> set[str]:
        return self._live().list_scope_dimension_values(dim_key)


def unwrap_memory_store(store: Any) -> Any:
    """Peel the revocation and user-isolation wrappers off ``store``.

    Reports what the storage adapter actually is, without asserting that a
    revoked reference is usable -- unwrapping is description, not access.
    """
    # Imported here: user_isolated_memory has no reason to know about this
    # module, and a module-level import would make that dependency circular.
    from .user_isolated_memory import UserIsolatedMemoryStore

    while isinstance(store, (RevocableMemoryStore, UserIsolatedMemoryStore)):
        store = store._base_store
    return store


__all__ = [
    "PublicationGenerationSource",
    "RevocableMemoryStore",
    "unwrap_memory_store",
]
