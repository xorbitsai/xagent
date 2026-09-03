from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Collection, List, Optional

from .core import MemoryNote, MemoryResponse
from .scope_columns import (
    SCOPE_EXCLUSIVE_FILTER_KEY,
    encode_scope_dims,
    scope_dim_element,
)

MEMORY_BACKEND_UNAVAILABLE_REASON = "required_memory_backend_unavailable"


class MemoryBackendUnavailableError(RuntimeError):
    """A caller-required memory backend capability is unavailable."""


def comparable_timestamp(value: Any) -> Any:
    """Normalize a datetime for cross-comparison in date-range filters.

    Stored note timestamps default to naive local time
    (``MemoryNote.timestamp``'s ``datetime.now`` factory), while caller-supplied
    ``date_from``/``date_to`` filter values may be timezone-aware (FastAPI
    parses ISO date query params with an offset into aware datetimes).
    Comparing the two directly raises ``TypeError``, which the stores' outer
    exception handlers would swallow into a silently empty result. Aware
    datetimes are therefore converted to naive local time before comparison;
    naive datetimes and non-datetime values pass through unchanged.

    Shared by every ``MemoryStore`` implementation's filter dispatch so the
    stores cannot drift apart on this policy.
    """
    if isinstance(value, datetime) and value.tzinfo is not None:
        return value.astimezone().replace(tzinfo=None)
    return value


class MemoryStore(ABC):
    """
    Abstract base class defining the interface for a memory storage backend.

    Any concrete implementation (e.g., in-memory store, ChromaDB, Redis, etc.)
    should implement all the following methods to manage MemoryNote objects.
    """

    def ensure_persistence(self) -> None:
        """Verify durable storage without changing backend state."""

        raise MemoryBackendUnavailableError(MEMORY_BACKEND_UNAVAILABLE_REASON)

    def ensure_required_vector_search(self) -> None:
        """Verify strict vector operations without network calls or mutation."""

        raise MemoryBackendUnavailableError(MEMORY_BACKEND_UNAVAILABLE_REASON)

    def add_required_vector(self, note: "MemoryNote") -> "MemoryResponse":
        """Add a note only when a valid vector can be persisted."""

        raise MemoryBackendUnavailableError(MEMORY_BACKEND_UNAVAILABLE_REASON)

    def update_required_vector(self, note: "MemoryNote") -> "MemoryResponse":
        """Atomically replace a note only after producing a valid vector."""

        raise MemoryBackendUnavailableError(MEMORY_BACKEND_UNAVAILABLE_REASON)

    def search_required_vector(
        self,
        query: str,
        k: int = 5,
        filters: Optional[dict[str, Any]] = None,
        similarity_threshold: Optional[float] = None,
    ) -> list["MemoryNote"]:
        """Search only through a compatible vector index."""

        raise MemoryBackendUnavailableError(MEMORY_BACKEND_UNAVAILABLE_REASON)

    @abstractmethod
    def add(self, note: "MemoryNote") -> "MemoryResponse":
        """
        Add a memory note to the store.

        Args:
            note (MemoryNote): The memory note to be added.

        Returns:
            MemoryResponse: Response indicating success and the note ID.
        """
        pass

    @abstractmethod
    def get(self, note_id: str) -> "MemoryResponse":
        """
        Retrieve a memory note by its ID.

        Args:
            note_id (str): The unique identifier of the memory note.

        Returns:
            MemoryResponse: Response containing the memory note or an error.
        """
        pass

    @abstractmethod
    def update(self, note: "MemoryNote") -> "MemoryResponse":
        """
        Update an existing memory note.

        Args:
            note (MemoryNote): The memory note with updated data.

        Returns:
            MemoryResponse: Response indicating success or failure.
        """
        pass

    @abstractmethod
    def delete(self, note_id: str) -> "MemoryResponse":
        """
        Delete a memory note by its ID.

        Args:
            note_id (str): The unique identifier of the memory note.

        Returns:
            MemoryResponse: Response indicating success or failure.
        """
        pass

    @abstractmethod
    def search(
        self,
        query: str,
        k: int = 5,
        filters: Optional[dict[str, Any]] = None,
        similarity_threshold: Optional[float] = None,
    ) -> list["MemoryNote"]:
        """
        Search memory notes by query text with optional filters.

        Args:
            query (str): The query string to search for.
            k (int, optional): Number of top results to return. Defaults to 5.
            filters (Dict[str, Any], optional): Additional filter criteria. Defaults to None.

        Returns:
            List[MemoryNote]: List of matching memory notes.
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """
        Clear all memory notes from the store.
        """
        pass

    @abstractmethod
    def list_all(self, filters: Optional[dict[str, Any]] = None) -> List["MemoryNote"]:
        """
        List all memory notes with optional filtering.

        Args:
            filters (Dict[str, Any], optional): Filter criteria like category, date range, etc.

        Returns:
            List[MemoryNote]: List of memory notes matching the filters.
        """
        pass

    @abstractmethod
    def get_stats(self) -> dict[str, Any]:
        """
        Get statistics about the memory store.

        Returns:
            Dict[str, Any]: Statistics including total count, counts by category, etc.
        """
        pass

    # ------------------------------------------------------------------
    # Shared filter dispatch (#916)
    #
    # Single implementation of filter matching used by every store's
    # search()/list_all(), hoisted here so backends cannot drift apart on
    # filter semantics. #906/#909/#910 unified the dispatch *within* each
    # store; this hoists it *across* stores, mirroring what #910 did for
    # comparable_timestamp(). Store-specific behavior belongs in overrides,
    # not in copies.
    # ------------------------------------------------------------------

    # Filter keys with dedicated handling in _matches_filters; anything else
    # is treated as a flat metadata-equality filter.
    _KNOWN_FILTER_KEYS = frozenset(
        {
            "category",
            "metadata",
            "date_from",
            "date_to",
            "tags",
            "keywords",
            SCOPE_EXCLUSIVE_FILTER_KEY,
        }
    )

    @staticmethod
    def _matches_metadata_filters(
        metadata: dict[str, Any], metadata_filters: dict[str, Any]
    ) -> bool:
        """Metadata equality for both the nested ``filters["metadata"]`` dict
        and flat metadata keys. String-coerced on both sides so
        ``UserIsolatedMemoryStore`` isolation and ad-hoc filters behave the
        same on every store (#842). A non-dict filter value cannot match
        anything (rather than crashing on a malformed caller-supplied
        ``filters["metadata"]``)."""
        if not isinstance(metadata_filters, dict):
            return False
        return all(
            str(metadata.get(key, "")) == str(value)
            for key, value in metadata_filters.items()
        )

    @staticmethod
    def _required_items(value: Any) -> Optional[Collection[Any]]:
        """Normalize a ``tags``/``keywords`` filter value: a plain string is a
        single required item (not iterated per character), an iterable is many,
        and anything else can't match (rather than raising ``TypeError`` on
        malformed caller-supplied filters — the same no-match policy as
        ``_matches_metadata_filters``)."""
        if isinstance(value, str):
            return (value,)
        if isinstance(value, (list, tuple, set, frozenset)):
            return value
        return None

    @classmethod
    def _flat_other_filters(cls, filters: Optional[dict[str, Any]]) -> dict[str, Any]:
        """Filter keys without dedicated handling, applied as flat metadata
        equality. Note-independent — compute once per call, not per note."""
        return {
            key: value
            for key, value in (filters or {}).items()
            if key not in cls._KNOWN_FILTER_KEYS
        }

    @classmethod
    def _matches_filters(
        cls,
        note: MemoryNote,
        filters: dict[str, Any],
        other_filters: dict[str, Any],
    ) -> bool:
        """Single filter dispatch shared by every store's ``search()`` and
        ``list_all()``, so backends cannot drift apart on filter semantics
        (#916; previously near-identical per-store copies).

        ``category`` comparison is string-coerced on both sides — a deliberate
        decision (#916): before the hoist the stores accidentally disagreed
        (InMemory exact, LanceDB coerced), and coercion is consistent with the
        string-coerced metadata-equality policy used everywhere else in this
        dispatch.

        An explicit ``None`` for ``date_from``/``date_to`` means "no bound"
        (#916) — matching how absent keys behave — rather than raising
        ``TypeError`` inside the date comparison.

        ``other_filters`` is ``_flat_other_filters(filters)``, precomputed by
        the caller so the per-note check does not rebuild it.
        """
        # #822: strict dimension-less exclusion (the ``__scope_exclusive__``
        # directive, ``SCOPE_EXCLUSIVE_FILTER_KEY``) — a note carrying any
        # scope dimension is excluded. Scope-dimension stamps live in
        # note.metadata.
        if filters.get(SCOPE_EXCLUSIVE_FILTER_KEY) and encode_scope_dims(note.metadata):
            return False

        if "category" in filters and str(note.category) != str(filters["category"]):
            return False

        # Nested metadata filters (the shape UserIsolatedMemoryStore emits
        # for user_id/scope isolation — before #842 search() never matched
        # them and list_all() ignored them, fail-open)
        if "metadata" in filters and not cls._matches_metadata_filters(
            note.metadata, filters["metadata"]
        ):
            return False

        # Date range filters (both sides tz-normalized so an aware filter
        # against the naive default timestamps cannot raise TypeError;
        # explicit None = no bound)
        date_from = filters.get("date_from")
        date_to = filters.get("date_to")
        if date_from is not None or date_to is not None:
            note_ts = comparable_timestamp(note.timestamp)
            if date_from is not None and note_ts < comparable_timestamp(date_from):
                return False
            if date_to is not None and note_ts > comparable_timestamp(date_to):
                return False

        # Tag filter (all required tags present)
        if "tags" in filters:
            required_tags = cls._required_items(filters["tags"])
            if required_tags is None or not all(
                tag in note.tags for tag in required_tags
            ):
                return False

        # Keyword filter (all required keywords present)
        if "keywords" in filters:
            required_keywords = cls._required_items(filters["keywords"])
            if required_keywords is None or not all(
                keyword in note.keywords for keyword in required_keywords
            ):
                return False

        # Other flat metadata filters (string-coerced equality)
        if other_filters and not cls._matches_metadata_filters(
            note.metadata, other_filters
        ):
            return False

        return True

    def delete_by_scope_dimension(self, dim_key: str, value: Any) -> MemoryResponse:
        """
        Delete every note stamped with the execution-scope dimension
        ``dim_key=value``.

        Maintenance operation for reaping notes whose scope dimension no longer
        maps to a live principal (e.g. all memories of a revoked client
        application). Matching is exact per-dimension string equality — a note
        carrying ``client_application_id=42`` is never touched by a delete for
        ``client_application_id=4`` — and notes carrying no scope dimensions are
        never candidates. Idempotent: re-running deletes nothing further.

        ``value`` is matched by its ``str()`` form, which must render exactly
        the string stamped at write time (write-time values are validated
        non-empty strings). A non-canonical representation — a float, a bool,
        leading zeros, alternate UUID casing — silently matches nothing
        (``success=True``, ``deleted_count=0``).

        This default walks ``list_all()`` and deletes note-by-note; backends
        with a native bulk predicate delete should override it. It assumes
        ``list_all()`` returns the complete store — a backend whose
        ``list_all()`` is bounded or paginated must override this method
        directly.

        Returns:
            MemoryResponse: ``success`` plus ``metadata["deleted_count"]``.
            On failure, ``deleted_count`` is backend-specific: this fallback
            deletes note-by-note and may report a nonzero partial count, while
            bulk-predicate overrides (e.g. LanceDB) are all-or-nothing and
            report 0.
        """
        element = scope_dim_element(dim_key, value)
        deleted = 0
        failures = 0
        for note in self.list_all():
            if note.id and element in encode_scope_dims(note.metadata):
                if self.delete(note.id).success:
                    deleted += 1
                else:
                    failures += 1
        if failures:
            return MemoryResponse(
                success=False,
                error=f"Failed to delete {failures} matching note(s)",
                metadata={"deleted_count": deleted},
            )
        return MemoryResponse(success=True, metadata={"deleted_count": deleted})

    def list_scope_dimension_values(self, dim_key: str) -> set[str]:
        """
        Distinct values stamped for one execution-scope dimension, store-wide.

        Lets a control plane reconcile a dimension against its own records —
        e.g. find the ``client_application_id`` values still present in memory
        whose rows have since been hard-deleted from the database, which no
        query on the database side can recover. Values are returned in their
        stamped string form.

        This default walks ``list_all()``, and assumes it returns the complete
        store — a backend whose ``list_all()`` is bounded or paginated must
        override this method directly, as should backends able to project the
        dimension column. Raises on backend failure rather
        than returning a partial set, so a reconciler never mistakes an error
        for "no values".
        """
        prefix = f"{dim_key}="
        values: set[str] = set()
        for note in self.list_all():
            for element in encode_scope_dims(note.metadata):
                if element.startswith(prefix):
                    values.add(element[len(prefix) :])
        return values
