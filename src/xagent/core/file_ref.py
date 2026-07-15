from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote, urlsplit

FILE_REF_OUTPUT_INSTRUCTIONS = """## FILE REFERENCE OUTPUTS
When mentioning a generated or uploaded file that has a file_id, render it as a Markdown file reference:
- Files: [filename](file:file_id)
- Images intended for inline display: ![filename](file:file_id)
Do not mention only the filename when a file_id or markdown_link is available. Prefer an existing markdown_link value when one is present."""

FILE_REF_MODEL_INSTRUCTIONS = f"""## FILE REFERENCES
Files are referenced by FileRef objects. Treat file_id as the canonical file handle.

Rules:
- Use file_id when reading files or passing files to tools.
- Do not guess storage paths such as /uploads/... or user_id/... paths.
- For HTML assets, call prepare_html_asset(file_id, html_path, alias) first.
- Use the returned html_src inside HTML, CSS, script, or image references.
- For user-visible output links, follow the file reference output rules below.

{FILE_REF_OUTPUT_INSTRUCTIONS}"""


def parse_file_id_ref(value: str | None) -> str | None:
    """Extract a file id from an internal ``file:`` reference.

    ``file:<id>`` is the canonical Xagent form. ``file://<id>`` is accepted
    for compatibility with older chat and connector messages. Real file URIs
    such as ``file:///absolute/path`` are deliberately not treated as file
    ids; callers must still route paths through workspace containment checks.
    """
    if value is None:
        return None

    raw = str(value).strip()
    if not raw.startswith("file:"):
        return None

    try:
        parsed = urlsplit(raw)
    except ValueError:
        return None
    if parsed.scheme != "file" or parsed.query or parsed.fragment:
        return None

    if raw.startswith("file://"):
        # ``file://<id>`` places the legacy id in the URI authority. A path
        # means this is a real/remote file URI instead of an internal handle.
        if not parsed.netloc or parsed.path:
            return None
        candidate = parsed.netloc
    else:
        candidate = parsed.path

    file_id = unquote(candidate).strip()
    if not file_id or file_id in {".", ".."}:
        return None
    if "/" in file_id or "\\" in file_id:
        return None
    return file_id


def build_file_id_ref(file_id: str) -> str:
    """Return the canonical internal URI for a registered file id."""
    normalized = str(file_id).strip()
    if not normalized or normalized in {".", ".."}:
        raise ValueError("file_id must be a non-empty identifier")
    decoded = unquote(normalized)
    if decoded in {".", ".."} or any(
        separator in value for value in (normalized, decoded) for separator in "/\\"
    ):
        raise ValueError("file_id must not contain path separators")
    return f"file:{quote(normalized, safe='')}"


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
                "markdown_link": f"[{filename}]({build_file_id_ref(file_id)})",
            }
        )
    else:
        result.update(
            {
                "preview_url": None,
                "download_url": None,
                "markdown_link": None,
            }
        )
    return result


def build_workspace_file_ref(
    *,
    workspace: Any,
    file_path: str | Path,
    file_id: str | None = None,
    mime_type: str | None = None,
) -> dict[str, Any]:
    """Register a workspace file and build the model/API-facing FileRef."""
    resolved_path = Path(file_path).resolve()
    if not resolved_path.exists() or not resolved_path.is_file():
        raise FileNotFoundError(f"File not found for FileRef: {file_path}")
    if not hasattr(workspace, "workspace_dir"):
        raise ValueError("Workspace does not expose workspace_dir")

    final_file_id = file_id or workspace.get_file_id_from_path(str(resolved_path))
    if not final_file_id:
        final_file_id = workspace.register_file(str(resolved_path))

    workspace_root = workspace.workspace_dir.resolve()
    file_ref = build_file_ref(
        file_id=final_file_id,
        filename=resolved_path.name,
        mime_type=mime_type,
        size=resolved_path.stat().st_size,
    )
    try:
        relative_path = str(resolved_path.relative_to(workspace_root))
    except ValueError:
        relative_path = str(resolved_path)

    return {
        **file_ref,
        "relative_path": relative_path,
        "file_path": str(resolved_path),
    }


def safe_asset_filename(filename: str) -> str:
    """Return a browser-safe basename for HTML bundle assets."""
    name = Path(filename).name.strip()
    if not name or name in {".", ".."}:
        return "asset"
    return name
