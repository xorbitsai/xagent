from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import Any
from urllib.parse import quote

FILE_REF_MODEL_INSTRUCTIONS = """## FILE REFERENCES
Files are referenced by FileRef objects. Treat file_id as the canonical file handle.

Rules:
- Use file_id when reading files or passing files to tools.
- Do not guess storage paths such as /uploads/... or user_id/... paths.
- For HTML assets, call prepare_html_asset(file_id, alias) first.
- Use the returned html_src inside HTML, CSS, script, or image references.
- For user-visible output links, use markdown_link or file:{file_id}."""


def guess_mime_type(filename: str) -> str:
    media_type, _ = mimetypes.guess_type(filename)
    return media_type or "application/octet-stream"


def build_file_ref(
    *,
    file_id: str | None,
    filename: str,
    mime_type: str | None = None,
    size: int | None = None,
) -> dict[str, Any]:
    """Build the model/API-facing file reference for a registered file."""
    resolved_mime_type = mime_type or guess_mime_type(filename)
    result: dict[str, Any] = {
        "file_id": file_id,
        "filename": filename,
        "mime_type": resolved_mime_type,
    }
    if size is not None:
        result["size"] = int(size)

    if file_id:
        encoded_file_id = quote(file_id, safe="")
        result.update(
            {
                "preview_url": f"/api/files/preview/{encoded_file_id}",
                "download_url": f"/api/files/download/{encoded_file_id}",
                "public_preview_url": f"/api/files/public/preview/{encoded_file_id}",
                "markdown_link": f"[{filename}](file:{file_id})",
            }
        )
    else:
        result.update(
            {
                "preview_url": None,
                "download_url": None,
                "public_preview_url": None,
                "markdown_link": None,
            }
        )
    return result


def safe_asset_filename(filename: str) -> str:
    """Return a browser-safe basename for HTML bundle assets."""
    name = Path(filename).name.strip()
    if not name or name in {".", ".."}:
        return "asset"
    return name
