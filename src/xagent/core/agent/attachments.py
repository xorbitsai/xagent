"""Shared projection from raw upload metadata to the UI chip shape.

Both the websocket persistence path
(``_normalize_attachments_for_persistence``) and the trace callback's
context fallback (``_files_from_context``) need to project the full
``file_info_list`` (which contains absolute filesystem paths) down to the
minimal shape the frontend FileAttachment chip needs:

    {"file_id": str, "name": str, "size": Any, "type": Any}

Keeping that projection in one place prevents the two callers from drifting
on what fields are exposed to clients (paths must never leak — the
attachments column and trace event payloads both reach the browser).
"""

from __future__ import annotations

from typing import Any, Dict, List

from ..context_ref import ContextReference
from ..file_ref import build_file_ref, guess_mime_type

# Keep the direct-context contract to the formats shared by the supported chat
# providers. SVG remains a source-inspection case for ``understand_media`` and
# uncommon raster formats can use the same fallback instead of making the
# primary chat request fail at the provider boundary.
_DIRECT_CONTEXT_IMAGE_MIME_TYPES = {
    "image/gif",
    "image/jpeg",
    "image/png",
    "image/webp",
}


def project_file_info_to_chip(file_info_list: Any) -> List[Dict[str, Any]]:
    """Project ``file_info_list`` to the chip shape; tolerant to None/garbage.

    Entries without a ``file_id`` are dropped (the chip can't be rendered or
    clicked without one). Absolute filesystem paths are *not* copied across.
    ``original_name`` is preferred over ``name`` for the chip label so a
    server-side normalized basename doesn't override the user's filename.
    """
    if not isinstance(file_info_list, list):
        return []
    projected: List[Dict[str, Any]] = []
    for info in file_info_list:
        if not isinstance(info, dict):
            continue
        file_id = info.get("file_id")
        if not file_id:
            continue
        projected.append(
            {
                "file_id": str(file_id),
                "name": str(
                    info.get("original_name") or info.get("name") or "uploaded file"
                ),
                "size": info.get("size"),
                "type": info.get("type"),
            }
        )
    return projected


def build_image_context_references(files: Any) -> tuple[ContextReference, ...]:
    """Project trusted uploaded-image metadata to durable model context refs.

    The input may be the full runtime ``file_info`` shape or the path-stripped
    attachment-chip shape persisted in chat history. Absolute paths are never
    copied: provider image bytes are resolved just in time from the registered
    ``file_id`` by :mod:`xagent.core.context_materializer`.
    """

    if not isinstance(files, list):
        return ()

    references: list[ContextReference] = []
    seen_file_ids: set[str] = set()
    for info in files:
        if not isinstance(info, dict):
            continue
        file_id = str(info.get("file_id") or "").strip()
        if not file_id or file_id in seen_file_ids:
            continue

        filename = "uploaded image"
        for key in ("original_name", "name", "filename"):
            candidate = str(info.get(key) or "").strip()
            if candidate:
                filename = candidate
                break
        declared_mime_type = str(
            info.get("mime_type") or info.get("type") or ""
        ).lower()
        declared_mime_type = declared_mime_type.split(";", 1)[0].strip()
        if declared_mime_type == "image/jpg":
            declared_mime_type = "image/jpeg"
        guessed_mime_type = guess_mime_type(filename).lower()
        if declared_mime_type in _DIRECT_CONTEXT_IMAGE_MIME_TYPES:
            mime_type = declared_mime_type
        elif declared_mime_type in {"", "application/octet-stream"}:
            mime_type = guessed_mime_type
        else:
            continue
        if mime_type not in _DIRECT_CONTEXT_IMAGE_MIME_TYPES:
            continue

        size: int | None = None
        raw_size = info.get("size")
        if raw_size is not None:
            try:
                parsed_size = int(raw_size)
            except (TypeError, ValueError):
                pass
            else:
                if parsed_size >= 0:
                    size = parsed_size
        try:
            reference = ContextReference(
                file_ref=build_file_ref(
                    file_id=file_id,
                    filename=filename,
                    mime_type=mime_type,
                    size=size,
                ),
                text_fallback=(
                    "An uploaded image was referenced by FileRef but is not "
                    "available as native visual context in this model call."
                ),
                metadata={"source": "user_upload"},
            )
        except (TypeError, ValueError):
            continue

        seen_file_ids.add(file_id)
        references.append(reference)

    return tuple(references)
