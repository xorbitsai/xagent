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
