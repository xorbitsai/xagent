"""Backend-agnostic KB cleanup scope helpers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Optional

from ..utils.user_scope import resolve_user_scope


@dataclass(frozen=True)
class KBCleanupScope:
    """Resolved scope for destructive KB cleanup operations."""

    collection: str
    user_id: Optional[int]
    is_admin: bool
    doc_id: Optional[str] = None
    parse_hash: Optional[str] = None
    chunk_ids: tuple[str, ...] = ()
    model_tag: Optional[str] = None


def resolve_cleanup_scope(
    *,
    collection: str,
    doc_id: Optional[str] = None,
    parse_hash: Optional[str] = None,
    chunk_ids: Optional[Sequence[object]] = None,
    model_tag: Optional[str] = None,
    user_id: Optional[int] = None,
    is_admin: Optional[bool] = None,
    require_target: bool = True,
) -> KBCleanupScope:
    """Normalize and validate a cleanup scope with request-context fallback."""
    normalized_collection = collection.strip() if isinstance(collection, str) else ""
    if not normalized_collection:
        raise ValueError("collection must be a non-empty string")

    normalized_doc_id = _normalize_optional_string(doc_id)
    normalized_parse_hash = _normalize_optional_string(parse_hash)
    normalized_chunk_ids = normalize_cleanup_chunk_ids(chunk_ids)
    if normalized_chunk_ids and normalized_doc_id is None:
        raise ValueError("doc_id is required when chunk_ids are provided")
    if require_target and not any(
        [normalized_doc_id, normalized_parse_hash, normalized_chunk_ids]
    ):
        raise ValueError("At least one of doc_id, parse_hash, or chunk_ids is required")

    user_scope = resolve_user_scope(user_id=user_id, is_admin=is_admin)
    return KBCleanupScope(
        collection=normalized_collection,
        doc_id=normalized_doc_id,
        parse_hash=normalized_parse_hash,
        chunk_ids=normalized_chunk_ids,
        model_tag=model_tag,
        user_id=user_scope.user_id,
        is_admin=user_scope.is_admin,
    )


def normalize_cleanup_chunk_ids(
    chunk_ids: Optional[Sequence[object]],
) -> tuple[str, ...]:
    """Return a stable deduplicated chunk-id tuple."""
    if not chunk_ids:
        return ()
    normalized: set[str] = set()
    for chunk_id in chunk_ids:
        if chunk_id is None:
            continue
        value = str(chunk_id)
        if value:
            normalized.add(value)
    return tuple(sorted(normalized))


def _normalize_optional_string(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    normalized = str(value)
    return normalized if normalized else None
