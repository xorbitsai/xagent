"""Canonical builders for durable-storage keys.

This module is the single source of truth for the object-storage key layout
(``users/{user_id}/uploads/...`` and ``users/{user_id}/tasks/{task_id}/outputs/...``).
Builders sanitize their inputs so the produced keys always pass strict
normalization (see ``normalize_storage_key``); keys already persisted in the
database are read back verbatim and never re-derived through these builders.

When an :class:`ExecutionScope` carries ``workspace_segments``, they are
inserted immediately after the user root
(``users/{user_id}/{segment}.../...``). The scoped prefix is an extension of
the user-bound prefix, so per-user prefix-scope enforcement
(``ScopedFileStorage``) admits scoped keys unchanged. Scope segments are
validated, never sanitized — silently rewriting one could merge two scopes'
namespaces.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence
from uuid import UUID

from ..execution_scope import validate_scope_component


def safe_storage_filename(filename: str) -> str:
    safe_name = Path(filename).name.strip()
    if safe_name in ("", ".", ".."):
        return "file"
    return _sanitize_component(safe_name) or "file"


def build_user_key_prefix(user_id: int, scope_segments: Sequence[str] = ()) -> str:
    """Compose the user-root key prefix ``users/{user_id}[/{segment}...]``.

    Single source of truth shared by the key builders below and by
    ``get_user_file_storage`` (so a scope-bound storage handle and the keys
    written through it use an identical prefix). Segments are validated,
    never sanitized.
    """
    prefix = f"users/{user_id}"
    for segment in scope_segments:
        validate_scope_component(segment, field_name="workspace_segments entry")
        prefix += f"/{segment}"
    return prefix


def build_upload_storage_key(
    user_id: int,
    file_id: str,
    filename: str,
    *,
    scope_segments: Sequence[str] = (),
) -> str:
    return (
        f"{build_user_key_prefix(user_id, scope_segments)}/uploads/"
        f"{file_id}/{safe_storage_filename(filename)}"
    )


def build_upload_generation_storage_key(
    user_id: int,
    file_id: str,
    filename: str,
    *,
    generation: str,
    scope_segments: Sequence[str] = (),
) -> str:
    """Build one immutable uploaded-file object generation.

    ``file_id`` is the logical identity retained by the database row, while
    ``generation`` gives each replacement a fresh object key. This prevents a
    replacement from overwriting bytes still referenced by an older metadata
    version and lets failed compare-and-swap attempts compensate only their
    own staged object.
    """

    validate_scope_component(file_id, field_name="file_id")
    try:
        generation_component = UUID(generation).hex
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("generation must be a UUID") from exc
    return (
        f"{build_user_key_prefix(user_id, scope_segments)}/uploads/"
        f"{file_id}/_versions/{generation_component}/"
        f"{safe_storage_filename(filename)}"
    )


def build_task_output_storage_key(
    user_id: int,
    task_id: int,
    file_id: str,
    relative_path: str,
    *,
    scope_segments: Sequence[str] = (),
) -> str:
    return (
        f"{build_user_key_prefix(user_id, scope_segments)}/tasks/{task_id}/outputs/"
        f"{file_id}/{_safe_relative_output_path(relative_path)}"
    )


def _safe_relative_output_path(relative_path: str) -> str:
    components = [
        part for part in relative_path.strip().split("/") if part not in ("", ".")
    ]
    if ".." in components:
        # A traversal component means the structure cannot be trusted;
        # keep only a safe basename.
        return safe_storage_filename(relative_path)
    if not components:
        return "file"
    return "/".join(_sanitize_component(part) for part in components)


def _sanitize_component(component: str) -> str:
    return "".join(
        "_" if ch == "\\" or ord(ch) < 32 or ord(ch) == 127 else ch for ch in component
    )
