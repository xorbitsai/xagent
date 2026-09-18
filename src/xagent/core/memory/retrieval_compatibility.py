"""Bounded streaming retrieval for the dormant persistent-memory path (#2346).

The live text fallback in ``LanceDBMemoryStore`` materialises the whole
candidate population (``limit(None).to_pylist()``), pushes only the null-vector
predicate into the backend, and sorts every candidate to pick ``k``. This module
is the compatible replacement for that shape, kept dormant: it is reachable only
from ``search_with_null_vector_fallback``, never from ``search``.

Three properties hold on every scan here. **Bounded residency**: rows arrive
through the backend's own batching (``to_batches(batch_size=...)``), projected to
the three columns a note is rebuilt from, so resident memory is
``O(batch_size + k)`` rather than ``O(population)``. **Backend-filtered**: the
null-vector predicate *and* the scope clause built by :func:`build_scope_where`
(owner ``user_id`` plus the execution-scope dimensions) are pushed into
``where``, so rows from other principals are never read at all; the rest
(category, tags/keywords, dates, arbitrary metadata keys) stays a Python
post-filter applied per batch. **Deterministic bounded top-k**: raw rows are
never capped before eligibility filtering and ranking, so a winner in the last
batch still wins; selection uses a heap bounded at ``k`` rather than a sort over
the population, and the rank key ends in the note id, so ties resolve
identically whatever the batch boundaries are.
"""

from __future__ import annotations

import heapq
import logging
from itertools import count
from typing import Any, Callable, Iterable, Iterator, Optional, Union

from ..tools.core.RAG_tools.LanceDB.schema_manager import _safe_close_table
from .core import MemoryNote
from .scope_columns import build_scope_where
from .storage_admission import AdmittedLanceDBMemoryStore, DormantLanceDBMemoryHandle

logger = logging.getLogger(__name__)

DEFAULT_STREAM_BATCH_SIZE = 256

# A note is rebuilt from these columns alone; nothing wider is ever read.
PROJECTED_COLUMNS = ("id", "text", "metadata")

MemorySource = Union[DormantLanceDBMemoryHandle, AdmittedLanceDBMemoryStore]
RowToNote = Callable[[dict[str, Any]], MemoryNote]
NoteFilterFactory = Callable[[dict[str, Any]], Callable[[MemoryNote], bool]]


def _checkpoint(_stage: str, _batch: int | None = None) -> None:
    """Test seam for the bounded-residency checks."""


def resolve_handle(source: MemorySource) -> DormantLanceDBMemoryHandle:
    """The layer B connection handle behind a dormant or admitted store."""
    if isinstance(source, AdmittedLanceDBMemoryStore):
        return source.handle
    return source


class _Descending:
    """Rank wrapper that inverts ordering, so ``heapq``'s min-heap keeps the
    *worst* retained candidate at its root — the one a better candidate evicts,
    which is what bounds the heap at ``k``."""

    __slots__ = ("key",)

    def __init__(self, key: tuple[int, int, str]) -> None:
        self.key = key

    def __lt__(self, other: "_Descending") -> bool:
        return other.key < self.key


def _stream_rows(
    handle: DormantLanceDBMemoryHandle,
    *,
    scope_where: Optional[str],
    null_vectors_only: bool,
    batch_size: int,
) -> Iterator[dict[str, Any]]:
    """Yield backend rows one bounded batch at a time. The ``vector IS NULL``
    term is added only when the table actually carries a vector column, so a
    legacy vectorless table streams instead of failing."""
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    table = handle.connection.open_table(handle.table_name)
    try:
        names = set(table.schema.names)
        terms = []
        if null_vectors_only and "vector" in names:
            terms.append("vector IS NULL")
        if scope_where:
            terms.append(scope_where)
        scan = table.search()
        if terms:
            scan = scan.where(" AND ".join(terms))
        projection = [column for column in PROJECTED_COLUMNS if column in names]
        if projection:
            scan = scan.select(projection)
        for batch in scan.to_batches(batch_size=batch_size):
            _checkpoint("scan_batch", batch.num_rows)
            # Only this one batch is in Python at a time.
            yield from batch.to_pylist()
    finally:
        _safe_close_table(table)


def _accepted_note(
    row: dict[str, Any],
    row_to_note: RowToNote,
    residual_filters: dict[str, Any],
    matches: Callable[[MemoryNote], bool],
) -> Optional[MemoryNote]:
    """The note for ``row``, or ``None`` when it is malformed or filtered out.
    A malformed row is skipped rather than aborting the scan (#847): one bad row
    must not truncate everything behind it in the stream."""
    try:
        note = row_to_note(row)
    except Exception as row_error:
        logger.warning("Skipping malformed memory row in streaming scan: %s", row_error)
        return None
    if residual_filters and not matches(note):
        return None
    return note


def stream_lexical_top_k(
    source: MemorySource,
    query: str,
    k: int,
    *,
    row_to_note: RowToNote,
    note_filter_factory: NoteFilterFactory,
    filters: Optional[dict[str, Any]] = None,
    exclude_ids: Iterable[str] = (),
    null_vectors_only: bool = True,
    batch_size: int = DEFAULT_STREAM_BATCH_SIZE,
) -> list[MemoryNote]:
    """The ``k`` best lexical matches for ``query``, streamed and bounded.

    Ranking is the existing fallback semantics — exact match, then prefix, then
    substring; more occurrences first; stable tie-break by id — but selected
    with a heap bounded at ``k`` across all batches instead of sorting every
    candidate.

    ``exclude_ids`` carries the ids ANN already returned. They are dropped on
    the raw row, before a note is built and before a heap slot is taken, so a
    duplicate can never displace a distinct lexical winner.
    """
    if k <= 0:
        return []
    scope_where, residual_filters = build_scope_where(filters)
    matches = note_filter_factory(residual_filters)
    excluded = set(exclude_ids)
    needle = query.casefold()
    heap: list[tuple[_Descending, int, MemoryNote]] = []
    tiebreak = count()
    for row in _stream_rows(
        resolve_handle(source),
        scope_where=scope_where,
        null_vectors_only=null_vectors_only,
        batch_size=batch_size,
    ):
        if str(row.get("id", "")) in excluded:
            continue
        folded = (row.get("text") or "").casefold()
        if needle and needle not in folded:
            continue
        note = _accepted_note(row, row_to_note, residual_filters, matches)
        if note is None:
            continue
        rank = (
            0 if folded == needle else 1 if folded.startswith(needle) else 2,
            -folded.count(needle),
            str(note.id),
        )
        entry = (_Descending(rank), next(tiebreak), note)
        if len(heap) < k:
            heapq.heappush(heap, entry)
        elif rank < heap[0][0].key:
            heapq.heapreplace(heap, entry)
        _checkpoint("retained", len(heap))
    return [note for _rank, _seq, note in sorted(heap, key=lambda item: item[0].key)]


def stream_notes(
    source: MemorySource,
    *,
    row_to_note: RowToNote,
    note_filter_factory: NoteFilterFactory,
    filters: Optional[dict[str, Any]] = None,
    batch_size: int = DEFAULT_STREAM_BATCH_SIZE,
) -> Iterator[MemoryNote]:
    """Every eligible note, streamed in bounded batches. The dedicated listing
    path: no query text, so no empty-query lexical ranking, and no ``k`` large
    enough to stand in for "all"."""
    scope_where, residual_filters = build_scope_where(filters)
    matches = note_filter_factory(residual_filters)
    for row in _stream_rows(
        resolve_handle(source),
        scope_where=scope_where,
        null_vectors_only=False,
        batch_size=batch_size,
    ):
        note = _accepted_note(row, row_to_note, residual_filters, matches)
        if note is not None:
            yield note


def stream_list_all(
    source: MemorySource,
    *,
    row_to_note: RowToNote,
    note_filter_factory: NoteFilterFactory,
    filters: Optional[dict[str, Any]] = None,
    batch_size: int = DEFAULT_STREAM_BATCH_SIZE,
) -> list[MemoryNote]:
    """Eligible notes, newest first — the streaming form of ``list_all``."""
    notes = list(
        stream_notes(
            source,
            row_to_note=row_to_note,
            note_filter_factory=note_filter_factory,
            filters=filters,
            batch_size=batch_size,
        )
    )
    notes.sort(key=lambda note: note.timestamp, reverse=True)
    return notes


def stream_stats(
    source: MemorySource,
    *,
    row_to_note: RowToNote,
    note_filter_factory: NoteFilterFactory,
    filters: Optional[dict[str, Any]] = None,
    batch_size: int = DEFAULT_STREAM_BATCH_SIZE,
) -> dict[str, Any]:
    """Store statistics accumulated incrementally over the stream. Counts are
    folded in per note, so nothing beyond the accumulators and one batch is
    resident — unlike ``get_stats``, which counts a materialised ``list_all()``."""
    total_count = 0
    category_counts: dict[str, int] = {}
    tag_counts: dict[str, int] = {}
    for note in stream_notes(
        source,
        row_to_note=row_to_note,
        note_filter_factory=note_filter_factory,
        filters=filters,
        batch_size=batch_size,
    ):
        total_count += 1
        category_counts[note.category] = category_counts.get(note.category, 0) + 1
        for tag in note.tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
    return {
        "total_count": total_count,
        "category_counts": category_counts,
        "tag_counts": tag_counts,
        "memory_store_type": "lancedb",
    }
