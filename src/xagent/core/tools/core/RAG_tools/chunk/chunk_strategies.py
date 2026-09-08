from __future__ import annotations

import logging
import re
from typing import Any, Callable, Dict, List, Optional

from ..core.config import DEFAULT_PROTECTED_PATTERNS, DEFAULT_TIKTOKEN_ENCODING
from ..utils.token_utils import (
    get_token_counter,
    split_text_by_tokens,
)

logger = logging.getLogger(__name__)

DEFAULT_SEPARATORS: List[str] = ["\n\n", "\n", "。", "！", "？", ". ", ", ", " "]


def _join_paragraphs(paragraphs: List[Dict[str, Any]]) -> str:
    texts = [p.get("text", "") for p in paragraphs if p.get("text")]
    return "\n\n".join(texts)


def _create_chunk_record(
    text: str, source_paragraph: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Create a chunk record with position information (no ID or timestamp).

    Args:
        text: Chunk text content
        source_paragraph: Source paragraph metadata for position info

    Returns:
        Chunk record with text, position fields, and full metadata dictionary
    """
    # Extract metadata from source paragraph if available
    metadata = (source_paragraph or {}).get("metadata", {})

    return {
        "text": text,
        "page_number": metadata.get("page_number"),
        "section": metadata.get("section"),
        "anchor": metadata.get("anchor"),
        "json_path": metadata.get("json_path"),
        "metadata": metadata,  # Preserve full metadata dictionary
    }


def _split_by_separators_core(text: str, separators: Optional[List[str]]) -> List[str]:
    """Core splitter that preserves delimiters by attaching them to the previous unit.

    This function contains the shared logic for regex construction, splitting with
    capturing groups, handling None parts, and attaching delimiters to the
    previous chunk. It returns a list of pure text chunks and is reused by
    higher-level wrappers with/without metadata.

    Args:
        text: Text to split
        separators: List of separator strings to use for splitting (defaults applied inside)

    Returns:
        List of text chunks with delimiters attached to the previous chunk.
        Joining *these* chunks reproduces the input exactly, separator-only
        runs included. The guarantee is this function's alone: callers further
        up strip and drop blank windows by design.
    """
    if not separators:
        separators = DEFAULT_SEPARATORS

    escaped_separators = [re.escape(sep) for sep in separators]
    # One capturing group around the whole alternation, not one per separator:
    # re.split emits an element for *every* group on each match, so N groups
    # make the stride N+1 while the loop below walks it in twos. With the eight
    # default separators that dropped every other delimiter, gluing words
    # together ("CSV integration" -> "CSVintegration").
    pattern = "(" + "|".join(escaped_separators) + ")"

    parts = re.split(pattern, text)

    chunks: List[str] = []
    for i in range(0, len(parts), 2):
        text_part = parts[i]
        delimiter_part = parts[i + 1] if i + 1 < len(parts) else ""
        chunk = text_part + delimiter_part
        if not chunk:
            continue
        if text_part.strip():
            chunks.append(chunk)
        elif chunks:
            # Adjacent separators ("。 " after a space) leave a piece with a
            # delimiter but no text. It is still part of the document, so it
            # joins the chunk it follows - dropping it deleted characters, and
            # keeping it standalone made a chunk of pure punctuation.
            chunks[-1] += chunk
        else:
            # Only reachable for the very first fragment, when the text opens
            # with a separator and there is nothing yet to attach it to.
            chunks.append(chunk)

    return chunks


def _split_by_separators(text: str, separators: Optional[List[str]]) -> List[str]:
    """Wrapper: returns pure text chunks using the core splitter."""
    return _split_by_separators_core(text, separators)


def _split_by_separators_with_metadata(
    text: str,
    separators: Optional[List[str]],
    source_paragraph: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Wrapper: returns structured chunks with metadata using the core splitter.

    Passes every unit through, which is what carries the core splitter's
    lossless guarantee to this level. Filtering blank units here used to drop
    the separator between two protected regions - the whole of a segment can be
    blank when a code fence ends and the next one begins.
    """
    units = _split_by_separators_core(text, separators)
    return [{"text": unit, "source_paragraph": source_paragraph} for unit in units]


def _find_protected_ranges(
    text: str,
    patterns: Optional[List[str]] = None,
) -> List[tuple[int, int]]:
    """Find protected regions (start, end) in text. Returns merged non-overlapping ranges."""
    if patterns is None:
        patterns = list(DEFAULT_PROTECTED_PATTERNS)
    if not patterns:
        return []
    ranges: List[tuple[int, int]] = []
    for pat in patterns:
        try:
            for m in re.finditer(pat, text, re.MULTILINE | re.DOTALL):
                ranges.append((m.start(), m.end()))
        except re.error:
            continue
    ranges.sort(key=lambda r: r[0])
    merged: List[tuple[int, int]] = []
    for start, end in ranges:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _split_by_separators_with_metadata_and_protection(
    text: str,
    separators: Optional[List[str]],
    source_paragraph: Optional[Dict[str, Any]],
    protected_patterns: Optional[List[str]],
) -> List[Dict[str, Any]]:
    """Split text by separators but never split inside protected regions (P1)."""
    ranges = _find_protected_ranges(text, protected_patterns)
    if not ranges:
        return _split_by_separators_with_metadata(text, separators, source_paragraph)
    result: List[Dict[str, Any]] = []
    pos = 0
    for start, end in ranges:
        if pos < start:
            normal = text[pos:start]
            units = _split_by_separators_with_metadata(
                normal, separators, source_paragraph
            )
            result.extend(units)
        protected_text = text[start:end]
        if protected_text.strip():
            result.append(
                {"text": protected_text, "source_paragraph": source_paragraph}
            )
        pos = end
    if pos < len(text):
        units = _split_by_separators_with_metadata(
            text[pos:], separators, source_paragraph
        )
        result.extend(units)
    return result


def _window_with_overlap(
    tokens: List[str], chunk_size: int, chunk_overlap: int
) -> List[str]:
    if chunk_size <= 0:
        return []
    if chunk_overlap < 0:
        chunk_overlap = 0
    chunks: List[str] = []
    start = 0
    n = len(tokens)
    while start < n:
        end = min(n, start + chunk_size)
        chunk = "".join(tokens[start:end])
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start = max(start + chunk_size - chunk_overlap, start + 1)
    return chunks


def _window_with_overlap_and_metadata(
    chunk_records: List[Dict[str, Any]], chunk_size: int, chunk_overlap: int
) -> List[Dict[str, Any]]:
    """Apply sliding window with overlap to chunk records, preserving metadata.

    Args:
        chunk_records: List of chunk records with text and source_paragraph
        chunk_size: Target chunk size in characters
        chunk_overlap: Overlap between consecutive chunks

    Returns:
        List of windowed chunk records with preserved metadata
    """
    if chunk_size <= 0:
        return []
    if chunk_overlap < 0:
        chunk_overlap = 0

    # Convert chunk records to character tokens while preserving metadata
    tokens: list[str] = []
    metadata_map: dict[
        int, dict[str, Any]
    ] = {}  # Maps character position to source paragraph

    for chunk_record in chunk_records:
        text = chunk_record["text"]
        source_paragraph = chunk_record.get("source_paragraph")

        start_pos = len(tokens)
        tokens.extend(list(text))
        end_pos = len(tokens)

        # Map character positions to source paragraph
        for i in range(start_pos, end_pos):
            if source_paragraph is not None:
                metadata_map[i] = source_paragraph

    # Apply sliding window
    windows = []
    start = 0
    n = len(tokens)

    while start < n:
        end = min(n, start + chunk_size)
        window_text = "".join(tokens[start:end])

        if window_text:
            # Find the first contributing paragraph for this window
            first_paragraph = None
            for i in range(start, end):
                if i in metadata_map and metadata_map[i] is not None:
                    first_paragraph = metadata_map[i]
                    break

            windows.append({"text": window_text, "source_paragraph": first_paragraph})

        if end == n:
            break
        start = max(start + chunk_size - chunk_overlap, start + 1)

    return windows


def _merge_units_by_token_limit(
    unit_records: List[Dict[str, Any]],
    chunk_token_size: int,
    chunk_token_overlap: int,
    num_tokens_fn: Callable[[str], int],
    tiktoken_encoding: str = DEFAULT_TIKTOKEN_ENCODING,
) -> List[Dict[str, Any]]:
    """Merge semantic units by token limit with greedy merge and overlap.

    Units are merged until adding the next would exceed chunk_token_size;
    then overlap is applied by carrying trailing units (that fit in
    chunk_token_overlap) to the next chunk. Single units that exceed
    chunk_token_size are split by token (fallback) and then merged.

    Args:
        unit_records: List of {"text": str, "source_paragraph": optional dict}.
        chunk_token_size: Max tokens per chunk.
        chunk_token_overlap: Overlap in tokens between consecutive chunks.
        num_tokens_fn: Function text -> token count.
        tiktoken_encoding: Encoding name for fallback split of long units.

    Returns:
        List of {"text": str, "source_paragraph": optional dict}.
    """
    if chunk_token_size <= 0 or not unit_records:
        return []

    if chunk_token_overlap < 0:
        chunk_token_overlap = 0
    if chunk_token_overlap >= chunk_token_size:
        chunk_token_overlap = max(0, chunk_token_size - 1)

    # Expand any unit that exceeds limit into token-sized sub-units
    expanded: List[Dict[str, Any]] = []
    for rec in unit_records:
        text = rec.get("text", "")
        para = rec.get("source_paragraph")
        n = num_tokens_fn(text)
        if n <= chunk_token_size:
            expanded.append({"text": text, "source_paragraph": para})
        else:
            for segment in split_text_by_tokens(
                text,
                max_tokens=chunk_token_size,
                overlap_tokens=chunk_token_overlap,
                encoding_name=tiktoken_encoding,
            ):
                if segment.strip():
                    expanded.append({"text": segment, "source_paragraph": para})

    if not expanded:
        return []

    # Greedy merge with overlap
    windows: List[Dict[str, Any]] = []
    current_units: List[Dict[str, Any]] = []
    current_tokens = 0
    i = 0
    while i < len(expanded):
        rec = expanded[i]
        t = num_tokens_fn(rec["text"])
        if current_tokens + t <= chunk_token_size:
            current_units.append(rec)
            current_tokens += t
            i += 1
            continue
        # Emit current chunk
        if current_units:
            chunk_text = "".join(u["text"] for u in current_units)
            first_para = current_units[0].get("source_paragraph")
            windows.append({"text": chunk_text, "source_paragraph": first_para})
        # Overlap: take trailing units that fit in chunk_token_overlap
        overlap_units: List[Dict[str, Any]] = []
        overlap_tokens = 0
        for u in reversed(current_units):
            ut = num_tokens_fn(u["text"])
            if overlap_tokens + ut <= chunk_token_overlap:
                overlap_units.insert(0, u)
                overlap_tokens += ut
            else:
                break
        current_units = overlap_units
        current_tokens = overlap_tokens
        # Do not advance i so we re-process rec in next iteration
    if current_units:
        chunk_text = "".join(u["text"] for u in current_units)
        first_para = current_units[0].get("source_paragraph")
        windows.append({"text": chunk_text, "source_paragraph": first_para})

    return windows


def apply_recursive_strategy(
    paragraphs: List[Dict[str, Any]], params: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Apply recursive chunking strategy with metadata preservation.

    This strategy splits text by separators first, then applies sliding window
    with overlap. Metadata from source paragraphs is preserved and propagated
    to the final chunks.
    """
    if not paragraphs:
        return []

    separators: Optional[List[str]] = params.get("separators")
    chunk_size_param = params.get("chunk_size")
    chunk_overlap: int = int(params.get("chunk_overlap", 200))

    # P1: optional protected content (code blocks, formulas, tables, etc.)
    enable_protected = params.get("enable_protected_content", True)
    protected_patterns: Optional[List[str]] = params.get("protected_patterns")

    all_chunk_records = []

    for paragraph in paragraphs:
        text = paragraph.get("text", "")
        if not text.strip():
            continue

        if enable_protected:
            paragraph_chunks = _split_by_separators_with_metadata_and_protection(
                text, separators, paragraph, protected_patterns
            )
        else:
            paragraph_chunks = _split_by_separators_with_metadata(
                text, separators, paragraph
            )
        all_chunk_records.extend(paragraph_chunks)

    if not all_chunk_records:
        return []

    # Apply size limit: token-based (P0) or character-based (legacy)
    use_token_count = bool(params.get("use_token_count"))
    if chunk_size_param is None:
        # User didn't set chunk_size, trust semantic splitting completely
        windows = all_chunk_records
    elif use_token_count:
        # P0: semantic unit merge by token limit (tiktoken)
        chunk_token_size = int(chunk_size_param)
        tiktoken_encoding = str(
            params.get("tiktoken_encoding", DEFAULT_TIKTOKEN_ENCODING)
        )
        num_tokens_fn = get_token_counter(tiktoken_encoding)
        windows = _merge_units_by_token_limit(
            all_chunk_records,
            chunk_token_size=chunk_token_size,
            chunk_token_overlap=chunk_overlap,
            num_tokens_fn=num_tokens_fn,
            tiktoken_encoding=tiktoken_encoding,
        )
    else:
        chunk_size = int(chunk_size_param)
        windows = _window_with_overlap_and_metadata(
            all_chunk_records, chunk_size, chunk_overlap
        )

    units_total_chars = sum(len(r.get("text", "")) for r in all_chunk_records)
    out_lens = [len(w.get("text", "")) for w in windows]
    logger.info(
        "[RAG][chunk] apply_recursive_strategy: semantic_units=%s units_total_chars=%s "
        "chunk_size=%r use_token_count=%s overlap=%s -> windows=%s "
        "window_char_min=%s max=%s sum=%s",
        len(all_chunk_records),
        units_total_chars,
        chunk_size_param,
        use_token_count,
        chunk_overlap,
        len(windows),
        min(out_lens, default=0),
        max(out_lens, default=0),
        sum(out_lens),
    )

    # Create final chunk records with preserved metadata
    return [
        _create_chunk_record(w["text"].strip(), w["source_paragraph"])
        for w in windows
        if w["text"].strip()
    ]


def _split_by_headers(
    text: str,
    headers_to_split_on: Optional[List[tuple[str, str]]],
) -> List[tuple[str, str]]:
    """Split text into (section_text, section_header) by header patterns. P1."""
    lines = text.splitlines()
    if not headers_to_split_on:
        # Default: atx-style # to ######
        header_pattern = re.compile(r"^\s{0,3}#{1,6}\s+")
        sections_with_headers = []
        current: List[str] = []
        section_header = ""
        for line in lines:
            if header_pattern.match(line):
                if current:
                    sections_with_headers.append(("\n".join(current), section_header))
                    current = []
                section_header = line.strip()
            current.append(line)
        if current:
            sections_with_headers.append(("\n".join(current), section_header))
        return sections_with_headers

    # User-provided headers: e.g. [("# ", "H1"), ("## ", "H2")]. Try longest prefix first.
    sorted_headers = sorted(headers_to_split_on, key=lambda x: len(x[0]), reverse=True)
    sections_with_headers = []
    section_lines: List[str] = []
    section_header = ""
    for line in lines:
        matched = False
        for prefix, _ in sorted_headers:
            if line.strip().startswith(prefix) or line.startswith(prefix):
                if section_lines:
                    sections_with_headers.append(
                        ("\n".join(section_lines), section_header)
                    )
                    section_lines = []
                section_header = line.strip()
                section_lines.append(line)
                matched = True
                break
        if not matched:
            section_lines.append(line)
    if section_lines:
        sections_with_headers.append(("\n".join(section_lines), section_header))
    return sections_with_headers


def apply_markdown_strategy(
    paragraphs: List[Dict[str, Any]], params: Dict[str, Any]
) -> List[Dict[str, Any]]:
    text = _join_paragraphs(paragraphs)
    if not text:
        return []
    chunk_size_param = params.get("chunk_size")
    chunk_overlap: int = int(params.get("chunk_overlap", 200))
    separators: Optional[List[str]] = params.get("separators")
    headers_to_split_on: Optional[List[tuple[str, str]]] = params.get(
        "headers_to_split_on"
    )

    # P1: Split by Markdown headers (configurable or default # to ######)
    sections_with_headers = _split_by_headers(text, headers_to_split_on)

    # If no headers found (single section with no header), fallback to recursive
    if len(sections_with_headers) == 1 and not sections_with_headers[0][1]:
        return apply_recursive_strategy(paragraphs, params)

    # For each section, further split and create chunks with section metadata (P1)
    chunks = []
    section_meta = {"section": ""}

    for sec, section_header in sections_with_headers:
        section_meta["section"] = section_header or ""
        source_para = {"metadata": dict(section_meta)}

        if separators and separators != DEFAULT_SEPARATORS:
            section_chunks = _split_by_separators(sec, separators)
            if section_chunks:
                if chunk_size_param is None:
                    windows = section_chunks
                else:
                    chunk_size = int(chunk_size_param)
                    total_chars = sum(len(chunk) for chunk in section_chunks)
                    if total_chars <= chunk_size:
                        windows = section_chunks
                    else:
                        windowed_chunks = _window_with_overlap_and_metadata(
                            [{"text": chunk} for chunk in section_chunks],
                            chunk_size,
                            chunk_overlap,
                        )
                        windows = [w["text"] for w in windowed_chunks]
            else:
                windows = [sec]
        else:
            if chunk_size_param is None:
                windows = [sec]
            else:
                chunk_size = int(chunk_size_param)
                tokens = list(sec)
                windows = _window_with_overlap(tokens, chunk_size, chunk_overlap)

        if not windows:
            continue
        for w in windows:
            if w.strip():
                chunks.append(
                    _create_chunk_record(w.strip(), source_paragraph=source_para)
                )
    return chunks


# P2: table/image context attachment
_TABLE_LINE_PATTERN = re.compile(r"\n\s*\|[^|\n]+\|", re.MULTILINE)
_IMAGE_PATTERN = re.compile(r"!\[.*?\]\(.*?\)", re.DOTALL)


def _is_table_chunk(text: str) -> bool:
    """True if text looks like a markdown table (has |...| row)."""
    # Primary: pattern match for markdown table rows
    if _TABLE_LINE_PATTERN.search(text):
        return True
    # Fallback: require at least 2 lines with pipe chars and a separator-like line
    lines = text.splitlines()
    pipe_lines = [ln for ln in lines if "|" in ln]
    if len(pipe_lines) >= 2 and text.count("|") >= 4:
        # Check for separator row pattern (e.g., |---|---|)
        for ln in lines:
            if re.search(r"\|[-:]+[-|:]*\|", ln):
                return True
    return False


def _is_image_chunk(text: str) -> bool:
    """True if text contains markdown image syntax."""
    return bool(_IMAGE_PATTERN.search(text))


def attach_media_context(
    chunks: List[Dict[str, Any]],
    table_context_size: int = 0,
    image_context_size: int = 0,
) -> None:
    """P2: Prepend/append surrounding context to table and image chunks (in-place).

    For each chunk that looks like a table or image, prepend the last N chars
    of the previous chunk and append the first N chars of the next chunk,
    so retrieval gets better context. N = table_context_size or image_context_size.

    Args:
        chunks: List of chunk dicts with "text" key; modified in place.
        table_context_size: Max chars from prev/next chunk to attach to table chunks; 0 = off.
        image_context_size: Max chars from prev/next chunk to attach to image chunks; 0 = off.
    """
    if not chunks or (table_context_size <= 0 and image_context_size <= 0):
        return
    n = len(chunks)
    for i in range(n):
        text = chunks[i].get("text", "")
        if not text:
            continue
        ctx_size = 0
        if table_context_size > 0 and _is_table_chunk(text):
            ctx_size = table_context_size
        elif image_context_size > 0 and _is_image_chunk(text):
            ctx_size = image_context_size
        if ctx_size <= 0:
            continue
        prefix = ""
        if i > 0:
            prev_text = chunks[i - 1].get("text", "")
            if prev_text:
                prefix = (
                    prev_text[-ctx_size:] if len(prev_text) > ctx_size else prev_text
                )
        suffix = ""
        if i + 1 < n:
            next_text = chunks[i + 1].get("text", "")
            if next_text:
                suffix = (
                    next_text[:ctx_size] if len(next_text) > ctx_size else next_text
                )
        if prefix:
            chunks[i]["text"] = prefix.strip() + "\n\n" + text
        if suffix:
            chunks[i]["text"] = chunks[i]["text"] + "\n\n" + suffix.strip()


def apply_fixed_size_strategy(
    paragraphs: List[Dict[str, Any]], params: Dict[str, Any]
) -> List[Dict[str, Any]]:
    text = _join_paragraphs(paragraphs)
    if not text:
        return []
    chunk_size_param = params.get("chunk_size")
    chunk_overlap: int = int(params.get("chunk_overlap", 0))

    if chunk_size_param is None:
        # User didn't set chunk_size, return whole text as one chunk
        return [_create_chunk_record(text.strip())]
    else:
        chunk_size = int(chunk_size_param)
        tokens = list(text)
        windows = _window_with_overlap(tokens, chunk_size, chunk_overlap)
        return [_create_chunk_record(w.strip()) for w in windows if w.strip()]
