"""User-isolated memory store for web application."""

import contextvars
from typing import Any, List, Optional

from xagent.core.memory.base import MemoryStore
from xagent.core.memory.core import MemoryNote, MemoryResponse
from xagent.core.user_context import current_user_id


class UserIsolatedMemoryStore(MemoryStore):
    """Memory store implementation that isolates memory by user ID using context."""

    def __init__(self, base_store: MemoryStore) -> None:
        """
        Initialize with a base memory store for actual storage.

        Args:
            base_store: The underlying memory store for storage operations
        """
        self._base_store = base_store

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
        if filters is None:
            filters = {}

        # Add user ID to metadata filters
        metadata_filters = filters.get("metadata", {})
        user_id = self._get_current_user_id()
        if user_id is not None:
            metadata_filters["user_id"] = user_id

        filters["metadata"] = metadata_filters
        return filters

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

        return self._base_store.add(note)

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
                return MemoryResponse(
                    success=False,
                    error="Memory note not found or access denied",
                    memory_id=note_id,
                )

        return response

    def update(self, note: MemoryNote) -> MemoryResponse:
        """
        Update a memory note with user isolation.

        Args:
            note: Memory note to update

        Returns:
            Memory response
        """
        # First verify ownership
        user_id = self._get_current_user_id()
        if user_id is not None and note.id:
            existing_response = self.get(note.id)
            if not existing_response.success:
                return existing_response

        # Add user ID to metadata if not present
        if user_id is not None and "user_id" not in note.metadata:
            note.metadata["user_id"] = user_id

        return self._base_store.update(note)

    def delete(self, note_id: str) -> MemoryResponse:
        """
        Delete a memory note with user isolation.

        Args:
            note_id: Memory note ID

        Returns:
            Memory response
        """
        # First verify ownership
        user_id = self._get_current_user_id()
        if user_id is not None:
            existing_response = self.get(note_id)
            if not existing_response.success:
                return existing_response

        return self._base_store.delete(note_id)

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
        # Add user filter to existing filters
        filtered_filters = self._add_user_filter(filters)

        return self._base_store.search(
            query=query,
            k=k,
            filters=filtered_filters,
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
        # Add user filter to existing filters
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
