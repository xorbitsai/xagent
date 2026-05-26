"""Paragraph-level page index extraction for chunking and parse statistics.

Used to merge ``metadata['page_number']`` with DeepDoc PDF bounding-box rows in
``metadata['positions']`` without mutating that field.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List


def normalized_pages_from_deepdoc_positions(meta: Dict[str, Any]) -> List[int]:
    """Collect 1-based page indices from DeepDoc PDF bbox ``metadata['positions']``.

    DeepDoc enriches rows in ``xagent.providers.pdf_parser.deepdoc._build_element_metadata``:

    - **Enriched:** ``[page_num, col_id, left, right, top, bottom]`` (length >= 6)
    - **Raw:** ``[page_num, left, right, top, bottom]`` (length >= 5)

    The first element is always the page index. It may be **0-based** (common in
    raw PDF APIs) or **1-based**. Heuristic: if any valid row has ``page_num == 0``,
    treat **all** rows as 0-based and convert to 1-based for RAG metadata
    (``page + 1``). Otherwise only integers ``>= 1`` are kept as 1-based pages.

    Malformed rows (too short, non-integer page) are skipped. This function only
    reads ``positions``; it never mutates ``meta``.

    Args:
        meta: Paragraph ``metadata`` dict (may include ``positions``).

    Returns:
        Sorted unique 1-based page numbers derived solely from bbox rows.
    """
    positions = meta.get("positions")
    if not isinstance(positions, list) or not positions:
        return []

    raw_indices: list[int] = []
    for item in positions:
        if not isinstance(item, (list, tuple)) or len(item) < 5:
            continue
        try:
            page_idx = int(item[0])
        except (TypeError, ValueError):
            continue
        raw_indices.append(page_idx)

    if not raw_indices:
        return []

    if min(raw_indices) == 0:
        normalized = sorted(
            {p + 1 for p in raw_indices if isinstance(p, int) and p >= 0}
        )
    else:
        normalized = sorted({p for p in raw_indices if isinstance(p, int) and p >= 1})
    return normalized


def collect_pages_from_paragraphs(paragraphs: Iterable[Dict[str, Any]]) -> List[int]:
    """Collect sorted unique 1-based page numbers from paragraph dict metadata.

    Merges:

    - ``metadata['page_number']`` when it is an integer ``>= 1`` (primary page).
    - ``metadata['positions']`` DeepDoc / PDF bbox rows (see
      :func:`normalized_pages_from_deepdoc_positions`) without overwriting that
      field — only page indices are read for provenance.

    Handles various edge cases:

    - None or non-dict paragraph entries
    - Missing or non-dict metadata
    - Non-integer or invalid ``page_number`` / bbox rows (skipped)

    Args:
        paragraphs: Iterable of paragraph dicts (typically ``text`` + ``metadata``).

    Returns:
        Sorted list of unique 1-based page numbers (>= 1).
    """
    pages: set[int] = set()
    for para in paragraphs:
        if not para or not isinstance(para, dict):
            continue

        meta = para.get("metadata")
        if not meta or not isinstance(meta, dict):
            continue

        page_num = meta.get("page_number")
        if isinstance(page_num, int) and page_num >= 1:
            pages.add(page_num)

        for p in normalized_pages_from_deepdoc_positions(meta):
            pages.add(p)

    return sorted(pages)
