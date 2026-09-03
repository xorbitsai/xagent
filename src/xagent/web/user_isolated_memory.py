"""User-isolated memory store for web application."""

import contextvars
from typing import Any, List, Optional

from xagent.core.execution_scope import (
    ExecutionScope,
    get_execution_scope,
    memory_dimension_metadata,
    metadata_carries_scope_dimensions,
)
from xagent.core.memory.base import (
    MEMORY_BACKEND_UNAVAILABLE_REASON,
    MemoryBackendUnavailableError,
    MemoryStore,
)
from xagent.core.memory.core import MemoryNote, MemoryResponse
from xagent.core.memory.scope_columns import SCOPE_EXCLUSIVE_FILTER_KEY
from xagent.core.user_context import current_user_id


class UserIsolatedMemoryStore(MemoryStore):
    """Memory store implementation that isolates memory by user ID using context.

    ExecutionScope integration (#757): when the active scope carries
    ``memory_dimensions``, adds stamp them onto note metadata (prefixed, see
    ``MEMORY_DIMENSION_METADATA_PREFIX``) and searches filter on them
    alongside ``user_id``. Visibility is one-way by default: scoped searches
    only see notes carrying their dimensions, while unscoped searches see
    everything under the user — including scoped notes. With
    ``strict_memory_isolation`` on the active scope, dimension-less searches
    additionally exclude any scope-stamped note (the flag is consumed even
    when the rest of the scope is empty). That exclusion is carried to the
    store as the ``__scope_exclusive__`` filter directive: pushed into the ``where``
    prefilter on the vector path (so it no longer crowds the top-k) and applied
    in Python on the text-search fallback.

    The by-id methods (``get``/``update``/``delete``) enforce the same scope
    dimensions as ``search``/``list_all`` (see ``_scope_permits``), so a
    caller cannot read, modify, or delete another scope's note by id even
    when both notes share a ``user_id``. A mismatch returns the combined
    "not found or access denied" response, indistinguishable from a genuine
    miss.

    Scope is a cooperative namespace, not a security boundary: ``user_id``
    remains the only access-control key.
    """

    def __init__(
        self, base_store: MemoryStore, *, require_vector_search: bool = False
    ) -> None:
        """
        Initialize with a base memory store for actual storage.

        Args:
            base_store: The underlying memory store for storage operations
        """
        self._base_store = base_store
        self._require_vector_search = require_vector_search

    def _get_current_user_id(self) -> Optional[int]:
        """Get the current user ID from context."""
        return current_user_id.get()

    def _add_user_filter(
        self, filters: Optional[dict[str, Any]] = None
    ) -> dict[str, Any]:
        """
        Add user filter to existing filters.

        Args:
            filters: Existing filters to extend

        Returns:
            Updated filters with user isolation
        """
        # Work on shallow copies so a caller-supplied ``filters`` dict (and its
        # nested ``metadata`` dict) is never mutated in place — callers may
        # reuse the same object across scopes.
        filters = dict(filters) if filters else {}

        # Add user ID to metadata filters
        metadata_filters = dict(filters.get("metadata", {}))
        user_id = self._get_current_user_id()
        if user_id is not None:
            metadata_filters["user_id"] = user_id

        # Scoped searches filter on the active scope's memory dimensions;
        # dimension-less scopes add nothing (unscoped searches see
        # everything under the user — one-way visibility by default).
        scope = get_execution_scope()
        metadata_filters.update(memory_dimension_metadata(scope))

        filters["metadata"] = metadata_filters

        # strict_memory_isolation on a dimension-less scope: exclude any
        # scope-stamped note. Pushed into the store's filter (#822 slice 003) as
        # a `where` prefilter on the vector path — so a strict search over a
        # collection dominated by scoped notes no longer has its own notes
        # crowded out of the top-k — and honored in Python on the text fallback.
        if (
            scope is not None
            and scope.strict_memory_isolation
            and not scope.memory_dimensions
        ):
            filters[SCOPE_EXCLUSIVE_FILTER_KEY] = True

        return filters

    @staticmethod
    def _scope_gates_by_id_access() -> bool:
        """Whether the active scope constrains by-id (get/update/delete)
        access. An unscoped or empty non-strict scope gates nothing, so the
        by-id path stays byte-for-byte unchanged when no scope is active."""
        scope: Optional[ExecutionScope] = get_execution_scope()
        if scope is None:
            return False
        return bool(scope.memory_dimensions) or scope.strict_memory_isolation

    @staticmethod
    def _scope_permits(note: MemoryNote) -> bool:
        """Whether the active scope may touch ``note`` by id, mirroring the
        equality-filter semantics of ``search``/``add``/``list_all``:

        - a scoped caller (non-empty ``memory_dimensions``) matches only a
          note carrying all of its dimension stamps;
        - a dimension-less caller under ``strict_memory_isolation`` cannot
          touch a scope-stamped note (the same exclusion ``search``/``list_all``
          push into the store via the ``__scope_exclusive__`` directive);
        - unscoped / dimension-less non-strict callers keep one-way
          visibility (access allowed).
        """
        scope: Optional[ExecutionScope] = get_execution_scope()
        dimension_metadata = memory_dimension_metadata(scope)
        if dimension_metadata:
            return all(
                note.metadata.get(key) == value
                for key, value in dimension_metadata.items()
            )
        if scope is not None and scope.strict_memory_isolation:
            return not metadata_carries_scope_dimensions(note.metadata)
        return True

    @staticmethod
    def _not_found_or_denied(note_id: str) -> MemoryResponse:
        """Combined miss/denial response: a scope or ownership mismatch is
        indistinguishable from a genuine miss."""
        return MemoryResponse(
            success=False,
            error="Memory note not found or access denied",
            memory_id=note_id,
        )

    def add(self, note: MemoryNote) -> MemoryResponse:
        """
        Add a memory note with user isolation.

        Args:
            note: Memory note to add

        Returns:
            Memory response
        """
        # Add user ID to metadata for isolation
        user_id = self._get_current_user_id()
        if user_id is not None:
            note.metadata["user_id"] = user_id

        # Stamp the active scope's memory dimensions so scoped searches can
        # filter on them (no-op when unscoped or dimension-less).
        note.metadata.update(memory_dimension_metadata(get_execution_scope()))

        if self._require_vector_search:
            return self._base_store.add_required_vector(note)
        return self._base_store.add(note)

    def ensure_persistence(self) -> None:
        self._base_store.ensure_persistence()

    def ensure_required_vector_search(self) -> None:
        self._base_store.ensure_required_vector_search()

    def add_required_vector(self, note: MemoryNote) -> MemoryResponse:
        user_id = self._get_current_user_id()
        if user_id is not None:
            note.metadata["user_id"] = user_id
        note.metadata.update(memory_dimension_metadata(get_execution_scope()))
        return self._base_store.add_required_vector(note)

    def get(self, note_id: str) -> MemoryResponse:
        """
        Retrieve a memory note with user isolation.

        Args:
            note_id: Memory note ID

        Returns:
            Memory response
        """
        response = self._base_store.get(note_id)
        if response.success and response.content:
            note = response.content
            # Check if the note belongs to the user
            user_id = self._get_current_user_id()
            if user_id is not None and note.metadata.get("user_id") != user_id:
                return self._not_found_or_denied(note_id)
            # Enforce the active scope's dimensions on the by-id path too
            # (mirrors search/list_all). No-op when unscoped.
            if not self._scope_permits(note):
                return self._not_found_or_denied(note_id)

        return response

    def update(self, note: MemoryNote) -> MemoryResponse:
        """
        Update a memory note with user isolation.

        Args:
            note: Memory note to update

        Returns:
            Memory response
        """
        # First verify ownership and scope access. get() enforces both;
        # call it whenever a user or a gating scope is present. When fully
        # unscoped with no user context, the original no-check path is kept.
        user_id = self._get_current_user_id()
        if note.id and (user_id is not None or self._scope_gates_by_id_access()):
            existing_response = self.get(note.id)
            if not existing_response.success:
                if self._require_vector_search and existing_response.error not in {
                    "Memory not found",
                    "Memory note not found or access denied",
                }:
                    raise MemoryBackendUnavailableError(
                        MEMORY_BACKEND_UNAVAILABLE_REASON
                    )
                return existing_response

        # Add user ID to metadata if not present
        if user_id is not None and "user_id" not in note.metadata:
            note.metadata["user_id"] = user_id

        if self._require_vector_search:
            return self._base_store.update_required_vector(note)
        return self._base_store.update(note)

    def update_required_vector(self, note: MemoryNote) -> MemoryResponse:
        user_id = self._get_current_user_id()
        if note.id and (user_id is not None or self._scope_gates_by_id_access()):
            existing_response = self.get(note.id)
            if not existing_response.success:
                if existing_response.error not in {
                    "Memory not found",
                    "Memory note not found or access denied",
                }:
                    raise MemoryBackendUnavailableError(
                        MEMORY_BACKEND_UNAVAILABLE_REASON
                    )
                return existing_response
        if user_id is not None and "user_id" not in note.metadata:
            note.metadata["user_id"] = user_id
        note.metadata.update(memory_dimension_metadata(get_execution_scope()))
        return self._base_store.update_required_vector(note)

    def delete(self, note_id: str) -> MemoryResponse:
        """
        Delete a memory note with user isolation.

        Args:
            note_id: Memory note ID

        Returns:
            Memory response
        """
        # First verify ownership and scope access (see update()). Unscoped
        # with no user context keeps the original no-check path.
        user_id = self._get_current_user_id()
        if user_id is not None or self._scope_gates_by_id_access():
            existing_response = self.get(note_id)
            if not existing_response.success:
                return existing_response

        return self._base_store.delete(note_id)

    def delete_by_scope_dimension(self, dim_key: str, value: Any) -> MemoryResponse:
        """Bulk-delete notes stamped with dimension ``dim_key=value``.

        Maintenance operation (reaping memories of a dead principal, e.g. a
        revoked client application), delegated straight to the base store: it
        is not gated by the per-user context, because the dimension predicate
        itself bounds the deletion to notes explicitly stamped with that
        dimension — dimension-less (creator-direct) notes are never touched.

        Trusted-caller-only: because it bypasses per-user isolation, this must
        only be reachable from control-plane maintenance jobs — never wire it
        to a per-user/per-request route, or one tenant could delete another
        tenant's notes.
        """
        return self._base_store.delete_by_scope_dimension(dim_key, value)

    def list_scope_dimension_values(self, dim_key: str) -> set[str]:
        """Distinct stamped values for one scope dimension, store-wide.

        Reconciliation primitive for the same maintenance flows as
        ``delete_by_scope_dimension`` — delegated ungated, and trusted-caller-
        only, for the same reasons: it enumerates values across every user.
        """
        return self._base_store.list_scope_dimension_values(dim_key)

    def search(
        self,
        query: str,
        k: int = 5,
        filters: Optional[dict[str, Any]] = None,
        similarity_threshold: Optional[float] = None,
    ) -> List[MemoryNote]:
        """
        Search memory notes with user isolation.

        Args:
            query: Search query
            k: Number of results
            filters: Additional filters
            similarity_threshold: Similarity threshold

        Returns:
            List of matching memory notes
        """
        # Add user (and scope-dimension) filters to existing filters. Strict
        # dimension-less isolation is carried inside the filters and applied by
        # the store (prefilter on the vector path), not post-filtered here.
        filtered_filters = self._add_user_filter(filters)

        search = (
            self._base_store.search_required_vector
            if self._require_vector_search
            else self._base_store.search
        )
        return search(
            query=query,
            k=k,
            filters=filtered_filters,
            similarity_threshold=similarity_threshold,
        )

    def search_required_vector(
        self,
        query: str,
        k: int = 5,
        filters: Optional[dict[str, Any]] = None,
        similarity_threshold: Optional[float] = None,
    ) -> List[MemoryNote]:
        return self._base_store.search_required_vector(
            query=query,
            k=k,
            filters=self._add_user_filter(filters),
            similarity_threshold=similarity_threshold,
        )

    def clear(self) -> None:
        """
        Clear memory notes with user isolation.
        """
        user_id = self._get_current_user_id()
        if user_id is not None:
            # Only clear notes for this user
            user_notes = self.list_all(filters={"metadata": {"user_id": user_id}})
            for note in user_notes:
                self._base_store.delete(note.id)
        else:
            # Clear all notes
            self._base_store.clear()

    def list_all(self, filters: Optional[dict[str, Any]] = None) -> List[MemoryNote]:
        """
        List all memory notes with user isolation.

        Args:
            filters: Additional filters

        Returns:
            List of memory notes
        """
        # Add user (and scope-dimension) filters to existing filters. Strict
        # dimension-less isolation is carried inside the filters and applied by
        # the store, not post-filtered here.
        filtered_filters = self._add_user_filter(filters)

        return self._base_store.list_all(filtered_filters)

    def get_stats(self) -> dict[str, Any]:
        """
        Get statistics with user isolation.

        Returns:
            Statistics dictionary
        """
        user_id = self._get_current_user_id()
        if user_id is not None:
            # Get stats for specific user
            user_notes = self.list_all()
            base_stats = self._base_store.get_stats()
            stats = self._calculate_stats(user_notes)
            # Preserve the original memory store type from base store
            stats["memory_store_type"] = base_stats.get("memory_store_type", "unknown")
            return stats
        else:
            # Get global stats
            return self._base_store.get_stats()

    def _calculate_stats(self, notes: List[MemoryNote]) -> dict[str, Any]:
        """
        Calculate statistics for a given set of notes.

        Args:
            notes: List of memory notes
            scope: Scope description for stats

        Returns:
            Statistics dictionary
        """
        total_count = len(notes)
        category_counts: dict[str, int] = {}
        tag_counts: dict[str, int] = {}

        for note in notes:
            # Count by category
            category_counts[note.category] = category_counts.get(note.category, 0) + 1

            # Count tags
            for tag in note.tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1

        return {
            "total_count": total_count,
            "category_counts": category_counts,
            "tag_counts": tag_counts,
        }


def set_user_context(user_id: Optional[int]) -> contextvars.Token:
    """
    Set the current user context for memory operations.

    Args:
        user_id: User ID to set as current context

    Returns:
        Context token that can be used to reset the context
    """
    return current_user_id.set(user_id)


def reset_user_context(token: contextvars.Token) -> None:
    """
    Reset the user context to its previous state.

    Args:
        token: Context token from set_user_context
    """
    current_user_id.reset(token)


class UserContext:
    """Context manager for setting user context."""

    def __init__(self, user_id: Optional[int]) -> None:
        self.user_id = user_id
        self.token: Optional[contextvars.Token] = None

    def __enter__(self) -> "UserContext":
        self.token = set_user_context(self.user_id)
        return self

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[Exception],
        exc_tb: Optional[object],
    ) -> None:
        if self.token is not None:
            reset_user_context(self.token)
