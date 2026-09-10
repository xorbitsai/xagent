import base64
import json
import logging
import mimetypes
import os
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests
from mcp.server.fastmcp import FastMCP

from .utils import setup_proxy_env

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("onedrive-mcp")

setup_proxy_env()

mcp = FastMCP("onedrive-mcp")

GRAPH_BASE_URL = "https://graph.microsoft.com/v1.0"
DEFAULT_TIMEOUT_SECONDS = 30
# Sized for a single chunk PUT of up to _UPLOAD_SESSION_CHUNK_SIZE bytes over
# a slow link, not just a small JSON Graph call -- DEFAULT_TIMEOUT_SECONDS'
# 30s is realistic for the API calls elsewhere in this module but can
# legitimately be too short for a multi-megabyte binary PUT.
_CHUNK_UPLOAD_TIMEOUT_SECONDS = 120
# Microsoft's docs describe the simple content PUT's limit as "4 MB"; some
# Graph deployments enforce that as the decimal 4,000,000 bytes rather than
# 4 MiB (4,194,304 bytes). Using the smaller, decimal figure here means a
# file in that ambiguous ~194KB gap always takes the resumable upload-session
# path instead of risking a rejection right at the simple-PUT boundary.
_SIMPLE_UPLOAD_MAX_BYTES = 4_000_000
# Chunk size for the upload-session path. Must be a multiple of 320 KiB
# (327,680 bytes) per Graph's requirement for every non-final chunk; 5 MiB
# is exactly 16 * 320 KiB.
_UPLOAD_SESSION_CHUNK_SIZE = 5 * 1024 * 1024

_UPLOAD_ALLOWED_DIRS_ENV_VAR = "XAGENT_ONEDRIVE_FILE_ALLOWED_DIRS"

# Extensions of unambiguously binary formats a caller might plausibly name
# a text-upload target after. Mirrors google_drive.py's equivalent guard —
# onedrive_upload_text_file's content is always UTF-8 text (see below), so
# a binary-looking target name is always a caller mistake, not a valid use.
_KNOWN_BINARY_EXTENSIONS = {
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".xlsm", ".ppt", ".pptx",
    ".odt", ".ods", ".odp", ".epub", ".mobi",
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tif", ".tiff",
    ".heic", ".heif", ".ico", ".psd",
    ".mp3", ".wav", ".ogg", ".flac", ".m4a", ".aac",
    ".mp4", ".mov", ".avi", ".mkv", ".webm", ".wmv",
    ".zip", ".7z", ".rar", ".tar", ".gz", ".tgz", ".bz2",
    ".exe", ".dll", ".so", ".dylib", ".dmg", ".bin", ".iso", ".apk",
    ".woff", ".woff2", ".ttf", ".otf", ".wasm", ".parquet", ".sqlite", ".db",
}  # fmt: skip


def _success(**payload: Any) -> str:
    return json.dumps({"status": "success", **payload}, ensure_ascii=False)


def _error(message: str, *, details: Any = None) -> str:
    payload: dict[str, Any] = {"status": "error", "message": message}
    if details is not None:
        payload["details"] = details
    return json.dumps(payload, ensure_ascii=False)


def _graph_headers(extra_headers: dict[str, str] | None = None) -> dict[str, str]:
    token = os.environ.get("AUTH_TOKEN")
    if not token:
        raise ValueError("AUTH_TOKEN environment variable is missing")

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    if extra_headers:
        headers.update(extra_headers)
    return headers


def _graph_request(
    method: str,
    path: str,
    *,
    params: dict[str, Any] | None = None,
    body: dict[str, Any] | None = None,
    data: bytes | None = None,
    extra_headers: dict[str, str] | None = None,
    raw: bool = False,
) -> Any:
    response = requests.request(
        method=method,
        url=f"{GRAPH_BASE_URL}{path}",
        headers=_graph_headers(extra_headers),
        params=params,
        json=body,
        data=data,
        timeout=DEFAULT_TIMEOUT_SECONDS,
    )
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        response_text = response.text.strip()
        message = str(exc)
        if response_text:
            message = f"{message} - {response_text}"
        raise RuntimeError(message) from exc

    if raw:
        return response.content
    if response.status_code == 204 or not response.content:
        return {}
    return response.json()


def _normalize_path(path: str | None) -> str | None:
    if path is None:
        return None
    value = path.strip().strip("/")
    return value or None


def _item_path(base_path: str | None) -> str:
    normalized = _normalize_path(base_path)
    if not normalized:
        return "/me/drive/root"
    return f"/me/drive/root:/{quote(normalized, safe='/')}:"


def _children_path(folder_path: str | None) -> str:
    normalized = _normalize_path(folder_path)
    if not normalized:
        return "/me/drive/root/children"
    return f"/me/drive/root:/{quote(normalized, safe='/')}:/children"


def _content_path(file_path: str) -> str:
    normalized = _normalize_path(file_path)
    if not normalized:
        raise ValueError("file_path is required")
    return f"/me/drive/root:/{quote(normalized, safe='/')}:/content"


def _decode_bytes(content: bytes) -> tuple[str | None, str | None]:
    try:
        return content.decode("utf-8"), None
    except UnicodeDecodeError:
        return None, base64.b64encode(content).decode("ascii")


def _name_looks_binary(name: str) -> bool:
    return Path(name).suffix.lower() in _KNOWN_BINARY_EXTENSIONS


def _allowed_upload_dirs() -> list[Path]:
    raw_dirs = os.environ.get(_UPLOAD_ALLOWED_DIRS_ENV_VAR, "")
    if not raw_dirs.strip():
        return [Path.cwd().resolve()]
    return [
        Path(stripped).expanduser().resolve()
        for raw_dir in raw_dirs.split(",")
        if (stripped := raw_dir.strip())
    ]


def _resolve_upload_file_path(file_path: str) -> Path:
    """Restrict onedrive_upload_file to files under an allowlisted
    directory, mirroring slack_upload_file's/google_drive_upload_file's
    defense -- without it an agent could be tricked into exfiltrating
    arbitrary host files through this tool.

    Containment is checked before existence, so a path that is both
    outside the allowlist and nonexistent reports the allowlist message,
    not "not found" -- the latter would leak whether that host path
    exists at all to a caller who has no business finding out.
    """
    local_path = Path(file_path).expanduser()
    if not local_path.is_absolute():
        local_path = Path.cwd() / local_path
    local_path = local_path.resolve()

    allowed_dirs = _allowed_upload_dirs()
    if not any(local_path.is_relative_to(d) for d in allowed_dirs):
        logger.warning(
            "Rejected onedrive_upload_file path %s outside allowed directories: %s",
            local_path,
            ", ".join(str(path) for path in allowed_dirs),
        )
        raise PermissionError(
            "file_path is outside the allowed upload directories; ask the "
            "user for a file inside the task workspace or another allowed "
            "location"
        )

    if not local_path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")
    return local_path


def _upload_large_file_content(remote_path: str, fh: Any, total: int) -> dict[str, Any]:
    """Upload the next ``total`` bytes readable from ``fh`` (too large for
    the simple content PUT) via a resumable upload session, in 320 KiB-
    aligned chunks read straight off disk -- never materializing more than
    one chunk of the file in memory at a time, unlike holding the whole
    file as a single ``bytes`` object would."""
    session = _graph_request(
        "POST",
        f"{_item_path(remote_path)}/createUploadSession",
        body={"item": {"@microsoft.graph.conflictBehavior": "replace"}},
    )
    upload_url = session.get("uploadUrl")
    if not upload_url:
        raise RuntimeError("OneDrive did not return an upload session URL")

    result: dict[str, Any] = {}
    for start in range(0, total, _UPLOAD_SESSION_CHUNK_SIZE):
        end = min(start + _UPLOAD_SESSION_CHUNK_SIZE, total)
        chunk = fh.read(end - start)
        # The upload session URL is itself pre-authenticated (a token in
        # its query string) -- Graph 401s a chunk request that also carries
        # our own Authorization header, so this goes straight through
        # requests.put rather than _graph_request (which always attaches
        # one), and content type is irrelevant to the session endpoint.
        response = requests.put(
            upload_url,
            data=chunk,
            headers={
                "Content-Length": str(end - start),
                "Content-Range": f"bytes {start}-{end - 1}/{total}",
            },
            timeout=_CHUNK_UPLOAD_TIMEOUT_SECONDS,
        )
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            # Mirror _graph_request's error enrichment (this loop can't go
            # through _graph_request itself -- see the Authorization-header
            # note above) so a rejected chunk surfaces Graph's actual error
            # body instead of a bare "400 Client Error" with no detail.
            response_text = response.text.strip()
            message = str(exc)
            if response_text:
                message = f"{message} - {response_text}"
            raise RuntimeError(message) from exc
        # Only the final chunk's response carries the completed item;
        # Graph's intermediate (202) responses return upload-progress
        # info, not the item -- checked explicitly (end == total) rather
        # than "the last response that happened to have a body", which
        # would silently pick up an unrelated intermediate body if Graph's
        # progress responses ever started including one.
        if end == total:
            result = response.json() if response.content else {}
    return result


@mcp.tool()
def onedrive_get_profile() -> str:
    """Get the current Microsoft 365 user profile for OneDrive operations."""
    try:
        me = _graph_request(
            "GET",
            "/me",
            params={"$select": "id,displayName,userPrincipalName,mail"},
        )
        return _success(user=me)
    except Exception as e:
        logger.error("Error getting OneDrive profile: %s", e)
        return _error(str(e))


@mcp.tool()
def onedrive_list_items(folder_path: str | None = None, top: int = 50) -> str:
    """List files and folders in OneDrive, optionally under a folder path."""
    try:
        result = _graph_request(
            "GET",
            _children_path(folder_path),
            params={"$top": max(1, min(top, 200))},
        )
        return _success(items=result.get("value", []))
    except Exception as e:
        logger.error("Error listing OneDrive items under %s: %s", folder_path, e)
        return _error(str(e))


@mcp.tool()
def onedrive_search_files(query: str, top: int = 25) -> str:
    """Search files and folders in OneDrive by keyword."""
    try:
        if not query.strip():
            raise ValueError("query is required")
        escaped_query = query.replace("'", "''")
        result = _graph_request(
            "GET",
            f"/me/drive/root/search(q='{quote(escaped_query, safe='')}')",
            params={"$top": max(1, min(top, 100))},
        )
        return _success(items=result.get("value", []))
    except Exception as e:
        logger.error("Error searching OneDrive files: %s", e)
        return _error(str(e))


@mcp.tool()
def onedrive_get_item(path: str | None = None, item_id: str | None = None) -> str:
    """Get OneDrive metadata by path or item_id."""
    try:
        if item_id:
            result = _graph_request(
                "GET",
                f"/me/drive/items/{quote(item_id, safe='')}",
            )
        elif path:
            result = _graph_request("GET", _item_path(path))
        else:
            raise ValueError("either path or item_id is required")
        return _success(item=result)
    except Exception as e:
        logger.error("Error getting OneDrive item: %s", e)
        return _error(str(e))


@mcp.tool()
def onedrive_get_file_content(file_path: str) -> str:
    """Download file content from OneDrive by path. Returns text when possible, otherwise base64."""
    try:
        content = _graph_request("GET", _content_path(file_path), raw=True)
        text_content, base64_content = _decode_bytes(content)
        return _success(
            file_path=file_path,
            text_content=text_content,
            base64_content=base64_content,
            encoding="utf-8" if text_content is not None else "base64",
        )
    except Exception as e:
        logger.error("Error downloading OneDrive file %s: %s", file_path, e)
        return _error(str(e))


@mcp.tool()
def onedrive_upload_text_file(
    file_path: str,
    content: str,
) -> str:
    """
    Upload or overwrite a UTF-8 text file in OneDrive by path.

    content is always treated as text (it is UTF-8 encoded before upload)
    -- this tool cannot create a real PDF, image, or Office-format binary;
    naming the target "report.pdf" would produce a file with that name but
    plain-text content, not an actual PDF. To upload an already-generated
    local file's real bytes (a PDF, image, .docx, etc.), use
    onedrive_upload_file with that file's path instead.
    """
    try:
        if _name_looks_binary(file_path):
            return _error(
                f"'{file_path}' looks like a binary file, but "
                "onedrive_upload_text_file only writes text content -- "
                "uploading it here would produce a mislabeled text file, "
                "not real binary data. If you already have this file on "
                "disk (e.g. as a task output), use onedrive_upload_file "
                "with its local path to upload the real binary content "
                "instead."
            )
        result = _graph_request(
            "PUT",
            _content_path(file_path),
            extra_headers={"Content-Type": "text/plain; charset=utf-8"},
            data=content.encode("utf-8"),
        )
        return _success(item=result)
    except Exception as e:
        logger.error("Error uploading OneDrive text file %s: %s", file_path, e)
        return _error(str(e))


@mcp.tool()
def onedrive_upload_file(
    file_path: str, remote_path: str = "", mime_type: str = ""
) -> str:
    """
    Upload a local file's real bytes to OneDrive -- use this (not
    onedrive_upload_text_file) for a PDF, image, Office document, or any
    other binary content, including a file this agent already generated
    into the task workspace (e.g. an exported PDF report).

    file_path: path to a file already on disk, e.g. something written to
    the task workspace. Must be inside an allowed directory (automatically
    scoped to the current task workspace) so this tool cannot be used to
    exfiltrate arbitrary files from the host. Pass an absolute path — a
    relative path resolves against this process's own working directory,
    not the allowed directory, and will not find a file written to the
    task workspace.
    remote_path: the OneDrive path to upload to (e.g. "Documents/report.pdf"),
    overwriting any existing file there; defaults to the local file's own
    name at the OneDrive root.
    mime_type: defaults to a guess from the file's extension, falling back
    to "application/octet-stream" when it can't be guessed.
    """
    try:
        local_path = _resolve_upload_file_path(file_path)
        resolved_remote_path = remote_path.strip() or local_path.name
        resolved_mime_type = (
            mime_type.strip()
            or mimetypes.guess_type(local_path.name)[0]
            or "application/octet-stream"
        )

        # Read from one open handle throughout -- the size check, the small-
        # file read, and the large-file chunk reads all use this same fh
        # rather than each re-opening/re-stat'ing the file. For a file over
        # _SIMPLE_UPLOAD_MAX_BYTES, _upload_large_file_content reads it
        # chunk-by-chunk straight off this handle rather than this function
        # first loading the whole file into memory -- the resumable upload-
        # session path exists specifically so a large file's memory
        # footprint stays bounded to one chunk at a time.
        with local_path.open("rb") as fh:
            file_size = os.fstat(fh.fileno()).st_size
            if file_size == 0:
                raise ValueError(f"File is empty: {file_path}")

            if file_size <= _SIMPLE_UPLOAD_MAX_BYTES:
                result = _graph_request(
                    "PUT",
                    _content_path(resolved_remote_path),
                    extra_headers={"Content-Type": resolved_mime_type},
                    data=fh.read(),
                )
            else:
                result = _upload_large_file_content(resolved_remote_path, fh, file_size)

        return _success(item=result)
    except Exception as e:
        logger.error("Error uploading OneDrive file %s: %s", file_path, e)
        return _error(str(e))


@mcp.tool()
def onedrive_create_folder(
    folder_name: str,
    parent_path: str | None = None,
    conflict_behavior: str = "rename",
) -> str:
    """Create a OneDrive folder under the specified parent path."""
    try:
        if not folder_name.strip():
            raise ValueError("folder_name is required")
        normalized_behavior = conflict_behavior.strip().lower()
        if normalized_behavior not in {"rename", "fail", "replace"}:
            raise ValueError("conflict_behavior must be one of: rename, fail, replace")
        result = _graph_request(
            "POST",
            _children_path(parent_path),
            body={
                "name": folder_name,
                "folder": {},
                "@microsoft.graph.conflictBehavior": normalized_behavior,
            },
        )
        return _success(folder=result)
    except Exception as e:
        logger.error("Error creating OneDrive folder %s: %s", folder_name, e)
        return _error(str(e))


@mcp.tool()
def onedrive_rename_item(item_id: str, new_name: str) -> str:
    """Rename a OneDrive file or folder by item_id."""
    try:
        if not new_name.strip():
            raise ValueError("new_name is required")
        result = _graph_request(
            "PATCH",
            f"/me/drive/items/{quote(item_id, safe='')}",
            body={"name": new_name},
        )
        return _success(item=result)
    except Exception as e:
        logger.error("Error renaming OneDrive item %s: %s", item_id, e)
        return _error(str(e))


@mcp.tool()
def onedrive_delete_item(item_id: str) -> str:
    """Delete a OneDrive file or folder by item_id."""
    try:
        _graph_request("DELETE", f"/me/drive/items/{quote(item_id, safe='')}")
        return _success(message="Item deleted successfully")
    except Exception as e:
        logger.error("Error deleting OneDrive item %s: %s", item_id, e)
        return _error(str(e))


if __name__ == "__main__":
    mcp.run()
