from __future__ import annotations

import threading
import uuid
from typing import Any, List, Optional

from .base import MemoryStore, comparable_timestamp
from .core import MemoryNote, MemoryResponse


class InMemoryMemoryStore(MemoryStore):
    # Filter matching (_matches_filters and friends) is inherited from the
    # MemoryStore base so stores cannot drift apart on filter semantics (#916).

    def __init__(self) -> None:
        self._store: dict[str, MemoryNote] = {}
        # Serializes the check-then-write of ``update`` against ``delete``, so
        # an update never re-creates a note deleted while it ran (the same
        # contract as ``LanceDBMemoryStore.update``).
        self._lock = threading.Lock()

    def add(self, note: MemoryNote) -> MemoryResponse:
        note_id = note.id or str(uuid.uuid4())
        note.id = note_id
        with self._lock:
            self._store[note_id] = note
        return MemoryResponse(success=True, memory_id=note_id)

    def get(self, note_id: str) -> MemoryResponse:
        note = self._store.get(note_id)
        if note:
            return MemoryResponse(success=True, memory_id=note_id, content=note)
        else:
            return MemoryResponse(
                success=False, error="Note not found", memory_id=note_id
            )

    def update(self, note: MemoryNote) -> MemoryResponse:
        """Replace a stored note, only while it still exists (never re-creates it)."""
        with self._lock:
            if note.id is None or note.id not in self._store:
                return MemoryResponse(
                    success=False,
                    error="Note not found or ID missing",
                    memory_id=note.id,
                )
            self._store[note.id] = note
        return MemoryResponse(success=True, memory_id=note.id)

    def delete(self, note_id: str) -> MemoryResponse:
        with self._lock:
            deleted = self._store.pop(note_id, None) is not None
        if deleted:
            return MemoryResponse(success=True, memory_id=note_id)
        return MemoryResponse(success=False, error="Note not found", memory_id=note_id)

    def search(
        self,
        query: str,
        k: int = 5,
        filters: Optional[dict[str, Any]] = None,
        similarity_threshold: Optional[float] = None,
    ) -> list[MemoryNote]:
        other_filters = self._flat_other_filters(filters)
        results = []
        for note in self._store.values():
            if query.lower() not in note.content.lower():
                continue
            if filters and not self._matches_filters(note, filters, other_filters):
                continue
            results.append(note)
        return results[:k]

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def list_all(self, filters: Optional[dict[str, Any]] = None) -> List[MemoryNote]:
        if filters:
            other_filters = self._flat_other_filters(filters)
            results = [
                note
                for note in self._store.values()
                if self._matches_filters(note, filters, other_filters)
            ]
        else:
            results = list(self._store.values())

        # Sort by timestamp (newest first); normalized because legacy naive
        # and current aware timestamps cannot be compared directly.
        results.sort(key=lambda x: comparable_timestamp(x.timestamp), reverse=True)

        return results

    def get_stats(self) -> dict[str, Any]:
        total_count = len(self._store)
        category_counts: dict[str, int] = {}
        tag_counts: dict[str, int] = {}

        for note in self._store.values():
            # Count by category
            category_counts[note.category] = category_counts.get(note.category, 0) + 1

            # Count tags
            for tag in note.tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1

        return {
            "total_count": total_count,
            "category_counts": category_counts,
            "tag_counts": tag_counts,
            "memory_store_type": "in_memory",
        }
