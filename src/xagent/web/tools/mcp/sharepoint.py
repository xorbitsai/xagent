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

from .utils import allowed_dirs_from_env, setup_proxy_env, url_path_id

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sharepoint-mcp")

setup_proxy_env()

mcp = FastMCP("sharepoint-mcp")

GRAPH_BASE_URL = "https://graph.microsoft.com/v1.0"
DEFAULT_TIMEOUT_SECONDS = 30
_BINARY_UPLOAD_TIMEOUT_SECONDS = 120
# Graph's simple content PUT is documented inconsistently -- the OneDrive API
# concepts page says simple upload is "available for items with less than
# 4 MB of content", while the Graph v1.0 API reference for the same endpoint
# says it "supports files up to 250 MB". onedrive.py resolves this by taking
# the smaller, more conservative bound and falling back to a resumable
# upload session above it; that session-based path is not implemented here
# yet, so a file over this bound is rejected outright with a clear error
# rather than silently attempted and possibly failed by Graph.
_SIMPLE_UPLOAD_MAX_BYTES = 4_000_000

_UPLOAD_ALLOWED_DIRS_ENV_VAR = "XAGENT_SHAREPOINT_FILE_ALLOWED_DIRS"

# Extensions this connector refuses for sharepoint_upload_text_file, since
# writing arbitrary text content under one of these names would silently
# produce a mislabeled (and broken) binary file rather than the real
# document format. Deliberately a small denylist of common binary document/
# media formats, not the exhaustive text-extension allowlist onedrive.py and
# google_drive.py each carry -- a real file already on disk should go
# through sharepoint_upload_file instead.
_BINARY_ONLY_EXTENSIONS = frozenset({
    ".docx", ".dotx", ".xlsx", ".xltx", ".xlsm", ".pptx", ".potx",
    ".pdf", ".zip", ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp",
})  # fmt: skip

_MIME_TYPE_OVERRIDES = {
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
}


class _GraphRequestError(RuntimeError):
    """Graph HTTP failure that retains its status without response parsing."""

    def __init__(self, message: str, *, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


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
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> Any:
    response = requests.request(
        method=method,
        url=f"{GRAPH_BASE_URL}{path}",
        headers=_graph_headers(extra_headers),
        params=params,
        json=body,
        data=data,
        timeout=timeout,
    )
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        response_text = response.text.strip()
        message = str(exc)
        if response_text:
            message = f"{message} - {response_text}"
        raise _GraphRequestError(message, status_code=response.status_code) from exc

    if raw:
        return response.content
    if response.status_code == 204 or not response.content:
        return {}
    return response.json()


def _site_segment(site_id: str) -> str:
    """Percent-encode a caller-supplied Graph site identifier for
    interpolation into a URL path segment.

    A Graph site id is one of: the literal "root", a composite id
    ("hostname,spSiteId,spWebId"), or a "hostname:/server-relative-path"
    form -- all three are returned verbatim by sharepoint_search_sites /
    sharepoint_get_root_site / sharepoint_get_site, so callers are expected
    to pass one of those straight back rather than constructing one by
    hand. ':' and '/' are kept unescaped because they are structural to the
    third form; every other character is percent-encoded. A '.'/'..'
    segment is rejected outright (matching onedrive.py's _normalize_path)
    since requests/urllib3 normalize dot-segments out of the final URL,
    which could otherwise walk the request off "/sites/{id}/..." and onto a
    different Graph endpoint under the same token.
    """
    if not isinstance(site_id, str) or not site_id.strip():
        raise ValueError("site_id is required")
    value = site_id.strip()
    if any(segment in (".", "..") for segment in value.split("/")):
        raise ValueError(f"site_id must not contain '.' or '..' segments: {site_id!r}")
    return quote(value, safe=":/,")


def _drive_base(site_id: str, drive_id: str | None) -> str:
    site_segment = _site_segment(site_id)
    if drive_id:
        return f"/sites/{site_segment}/drives/{url_path_id(drive_id, 'drive_id')}"
    return f"/sites/{site_segment}/drive"


def _normalize_relative_path(path: str | None) -> str | None:
    """Normalize a drive-relative path for a root:/{path}: request URL,
    rejecting '.'/'..' segments -- same rationale as _site_segment above."""
    if path is None:
        return None
    value = path.strip().strip("/")
    if not value:
        return None
    if "\\" in value:
        raise ValueError("path must use '/' separators and must not contain '\\'")
    if any(segment in (".", "..") for segment in value.split("/")):
        raise ValueError(f"path must not contain '.' or '..' segments: {path!r}")
    return value


def _drive_children_path(
    site_id: str, folder_path: str | None, drive_id: str | None
) -> str:
    base = _drive_base(site_id, drive_id)
    normalized = _normalize_relative_path(folder_path)
    if not normalized:
        return f"{base}/root/children"
    return f"{base}/root:/{quote(normalized, safe='/')}:/children"


def _drive_content_path(site_id: str, file_path: str, drive_id: str | None) -> str:
    stripped_path = file_path.strip()
    if stripped_path.endswith("/"):
        raise ValueError(
            "file_path must include a filename, not end with a folder separator"
        )
    if Path(stripped_path).name.endswith("."):
        # Matches onedrive.py's _content_path guard: a trailing-dot filename
        # can be silently normalized by the backing storage (Windows/SharePoint
        # semantics), so e.g. "report.docx." would both bypass
        # sharepoint_upload_text_file's extension denylist (its suffix parses
        # as "", not ".docx") AND land as "report.docx" on the server side --
        # silently overwriting an unrelated real document with mislabeled
        # text content.
        raise ValueError(f"{file_path!r} filename must not end with a period")
    normalized = _normalize_relative_path(file_path)
    if not normalized:
        raise ValueError("file_path is required")
    base = _drive_base(site_id, drive_id)
    return f"{base}/root:/{quote(normalized, safe='/')}:/content"


def _caller_safe_drive_item(item: dict[str, Any]) -> dict[str, Any]:
    """Copy a driveItem without Graph's short-lived preauthenticated URL."""
    safe_item = dict(item)
    safe_item.pop("@microsoft.graph.downloadUrl", None)
    return safe_item


def _decode_bytes(content: bytes) -> tuple[str | None, str | None]:
    try:
        return content.decode("utf-8"), None
    except UnicodeDecodeError:
        return None, base64.b64encode(content).decode("ascii")


def _guess_mime_type(name: str) -> str | None:
    suffix = Path(name).suffix.lower()
    override = _MIME_TYPE_OVERRIDES.get(suffix)
    if override is not None:
        return override
    guessed_type, _guessed_encoding = mimetypes.guess_type(name)
    return guessed_type


def _resolve_upload_file_path(local_file_path: str) -> Path:
    """Restrict sharepoint_upload_file to files under an allowlisted
    directory, mirroring onedrive_upload_file's local-file guard.

    Containment is checked before existence, so a path that is both outside
    the allowlist and nonexistent reports the allowlist message, not "not
    found" -- the latter would leak whether that host path exists at all.
    """
    try:
        candidate_path = Path(local_file_path).expanduser()
        if not candidate_path.is_absolute():
            candidate_path = Path.cwd() / candidate_path
        local_path = candidate_path.resolve()
        if candidate_path.is_symlink():
            local_path = candidate_path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        logger.warning(
            "Could not resolve SharePoint upload path %r: %s", local_file_path, exc
        )
        raise ValueError("Could not resolve local_file_path") from exc

    try:
        allowed_dirs = allowed_dirs_from_env(_UPLOAD_ALLOWED_DIRS_ENV_VAR)
    except ValueError as exc:
        logger.warning("Invalid SharePoint upload directory configuration: %s", exc)
        raise ValueError("Upload directory configuration is invalid") from None
    if not any(local_path.is_relative_to(d) for d in allowed_dirs):
        logger.warning(
            "Rejected sharepoint_upload_file path %s outside allowed directories: %s",
            local_path,
            ", ".join(str(path) for path in allowed_dirs),
        )
        raise PermissionError(
            "local_file_path is outside the allowed upload directories; ask the "
            "user for a file inside the task workspace or another allowed "
            "location"
        )

    if local_path.exists() and not local_path.is_file():
        raise ValueError("The given path is not a regular file")
    if not local_path.is_file():
        raise FileNotFoundError("File not found at the given path")
    return local_path


def _parse_fields_json(fields_json: str) -> dict[str, Any]:
    """Parse a SharePoint list item's column values from a JSON object
    string. Taking a JSON string (rather than a raw dict parameter) matches
    google_slides_batch_update's precedent for an open-ended, caller-defined
    payload -- a SharePoint list's columns are defined per-list by whoever
    created it, so there is no fixed schema this tool could validate against.
    """
    try:
        parsed = json.loads(fields_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"fields_json is not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("fields_json must decode to a JSON object")
    return parsed


@mcp.tool()
def sharepoint_search_sites(query: str, top: int = 25) -> str:
    """Search SharePoint sites by keyword. Returns each matching site's id
    (needed by every other sharepoint_* tool) alongside its name and URL."""
    try:
        if not query.strip():
            raise ValueError("query is required")
        result = _graph_request(
            "GET",
            "/sites",
            params={"search": query, "$top": max(1, min(top, 100))},
        )
        return _success(sites=result.get("value", []))
    except Exception as e:
        logger.error("Error searching SharePoint sites: %s", e)
        return _error(str(e))


@mcp.tool()
def sharepoint_get_root_site() -> str:
    """Get the tenant's root SharePoint site, including its id."""
    try:
        result = _graph_request("GET", "/sites/root")
        return _success(site=result)
    except Exception as e:
        logger.error("Error getting SharePoint root site: %s", e)
        return _error(str(e))


@mcp.tool()
def sharepoint_get_site(site_id: str) -> str:
    """Get a SharePoint site's metadata by its Graph site id (from
    sharepoint_search_sites or sharepoint_get_root_site)."""
    try:
        result = _graph_request("GET", f"/sites/{_site_segment(site_id)}")
        return _success(site=result)
    except Exception as e:
        logger.error("Error getting SharePoint site %s: %s", site_id, e)
        return _error(str(e))


@mcp.tool()
def sharepoint_list_drives(site_id: str) -> str:
    """List the document libraries (drives) in a SharePoint site."""
    try:
        result = _graph_request("GET", f"/sites/{_site_segment(site_id)}/drives")
        return _success(drives=result.get("value", []))
    except Exception as e:
        logger.error("Error listing SharePoint drives for site %s: %s", site_id, e)
        return _error(str(e))


@mcp.tool()
def sharepoint_list_items(
    site_id: str,
    folder_path: str | None = None,
    drive_id: str | None = None,
    top: int = 50,
) -> str:
    """List files and folders in a SharePoint document library, optionally
    under a folder path. Uses the site's default document library unless
    drive_id (from sharepoint_list_drives) is given."""
    try:
        result = _graph_request(
            "GET",
            _drive_children_path(site_id, folder_path, drive_id),
            params={"$top": max(1, min(top, 200))},
        )
        return _success(
            items=[_caller_safe_drive_item(item) for item in result.get("value", [])]
        )
    except Exception as e:
        logger.error(
            "Error listing SharePoint items for site %s under %s: %s",
            site_id,
            folder_path,
            e,
        )
        return _error(str(e))


@mcp.tool()
def sharepoint_search_files(
    site_id: str, query: str, drive_id: str | None = None, top: int = 25
) -> str:
    """Search files and folders in a SharePoint document library by keyword."""
    try:
        if not query.strip():
            raise ValueError("query is required")
        escaped_query = query.replace("'", "''")
        base = _drive_base(site_id, drive_id)
        result = _graph_request(
            "GET",
            f"{base}/root/search(q='{quote(escaped_query, safe='')}')",
            params={"$top": max(1, min(top, 100))},
        )
        return _success(
            items=[_caller_safe_drive_item(item) for item in result.get("value", [])]
        )
    except Exception as e:
        logger.error("Error searching SharePoint files for site %s: %s", site_id, e)
        return _error(str(e))


@mcp.tool()
def sharepoint_get_file_content(
    site_id: str, file_path: str, drive_id: str | None = None
) -> str:
    """Download a file's content from a SharePoint document library by path.
    Returns text when possible, otherwise base64."""
    try:
        content = _graph_request(
            "GET", _drive_content_path(site_id, file_path, drive_id), raw=True
        )
        text_content, base64_content = _decode_bytes(content)
        return _success(
            file_path=file_path,
            text_content=text_content,
            base64_content=base64_content,
            encoding="utf-8" if text_content is not None else "base64",
        )
    except Exception as e:
        logger.error(
            "Error downloading SharePoint file %s for site %s: %s",
            file_path,
            site_id,
            e,
        )
        return _error(str(e))


@mcp.tool()
def sharepoint_upload_text_file(
    site_id: str,
    file_path: str,
    content: str,
    drive_id: str | None = None,
) -> str:
    """Upload or overwrite a UTF-8 text file in a SharePoint document
    library by path.

    content is always treated as text (UTF-8 encoded before upload) -- this
    tool cannot create a real .docx/.xlsx/.pptx/.pdf or other binary
    document. To upload an already-generated local file's real bytes, use
    sharepoint_upload_file with that file's path instead.
    """
    try:
        suffix = Path(file_path.strip()).suffix.lower()
        if suffix in _BINARY_ONLY_EXTENSIONS:
            return _error(
                f"'{file_path}' names a binary document format, but "
                "sharepoint_upload_text_file only writes text content -- "
                "uploading it here would produce a mislabeled, broken file. "
                "If you already have this file on disk (e.g. as a task "
                "output), use sharepoint_upload_file with its local path to "
                "upload the real binary content instead."
            )
        result = _graph_request(
            "PUT",
            _drive_content_path(site_id, file_path, drive_id),
            extra_headers={"Content-Type": "text/plain; charset=utf-8"},
            data=content.encode("utf-8"),
        )
        if not isinstance(result, dict) or not result.get("id"):
            raise RuntimeError("SharePoint did not confirm the upload completed")
        return _success(item=_caller_safe_drive_item(result))
    except Exception as e:
        logger.error(
            "Error uploading SharePoint text file %s for site %s: %s",
            file_path,
            site_id,
            e,
        )
        return _error(str(e))


@mcp.tool()
def sharepoint_upload_file(
    local_file_path: str,
    site_id: str,
    remote_path: str = "",
    drive_id: str | None = None,
    mime_type: str = "",
) -> str:
    """Upload a local file's real bytes to a SharePoint document library --
    use this (not sharepoint_upload_text_file) for a PDF, image, Office
    document, or any other binary content.

    local_file_path: path to a file already on disk (e.g. in the task
    workspace). Must be inside an allowed directory.
    remote_path: the document library path to upload to, overwriting any
    existing file there; defaults to the local file's own name at the
    library root.

    Limited to files up to 4 MB (a resumable upload session for larger
    files, like onedrive_upload_file supports, is not yet implemented here).
    """
    try:
        local_path = _resolve_upload_file_path(local_file_path)
        resolved_remote_path = remote_path.strip() or local_path.name
        content_path = _drive_content_path(site_id, resolved_remote_path, drive_id)
        resolved_mime_type = mime_type.strip() or _guess_mime_type(resolved_remote_path)
        if resolved_mime_type is None:
            resolved_mime_type = (
                _guess_mime_type(local_path.name) or "application/octet-stream"
            )

        # Open once and read the size from that same descriptor (rather than
        # a separate Path.stat() before opening) so the size check and the
        # bytes actually uploaded are bound to one file description -- a
        # symlink swapped in between a separate stat() and open() could
        # otherwise upload different content than what was size-checked.
        try:
            fh_ctx = local_path.open("rb")
        except OSError as e:
            # str(OSError) embeds the absolute host path (e.g. "[Errno 13]
            # Permission denied: '/full/host/path'"), which the generic
            # `except Exception` below would otherwise leak straight into
            # the caller/LLM-facing error message.
            logger.warning("Failed to open upload file %s: %s", local_path, e)
            raise ValueError("Could not read the file at the given path") from e
        with fh_ctx as fh:
            file_size = os.fstat(fh.fileno()).st_size
            if file_size == 0:
                raise ValueError(f"File is empty: {local_file_path}")
            if file_size > _SIMPLE_UPLOAD_MAX_BYTES:
                raise ValueError(
                    f"File is {file_size} bytes, over the "
                    f"{_SIMPLE_UPLOAD_MAX_BYTES // 1_000_000} MB limit "
                    "sharepoint_upload_file currently supports"
                )
            content = fh.read(_SIMPLE_UPLOAD_MAX_BYTES + 1)
        if len(content) > _SIMPLE_UPLOAD_MAX_BYTES:
            raise RuntimeError("the local file grew during upload")

        result = _graph_request(
            "PUT",
            content_path,
            extra_headers={"Content-Type": resolved_mime_type},
            data=content,
            timeout=_BINARY_UPLOAD_TIMEOUT_SECONDS,
        )
        if not isinstance(result, dict) or not result.get("id"):
            raise RuntimeError("SharePoint did not confirm the upload completed")
        return _success(item=_caller_safe_drive_item(result))
    except Exception as e:
        if isinstance(e, (OSError, ValueError, RuntimeError, PermissionError)):
            logger.error(
                "Error uploading SharePoint file %s for site %s: %s",
                local_file_path,
                site_id,
                e,
            )
        else:
            logger.exception(
                "Unexpected error uploading SharePoint file %s for site %s",
                local_file_path,
                site_id,
            )
        return _error(str(e))


@mcp.tool()
def sharepoint_list_lists(site_id: str, top: int = 50) -> str:
    """List the SharePoint lists (e.g. custom lists, document metadata
    lists) in a site."""
    try:
        result = _graph_request(
            "GET",
            f"/sites/{_site_segment(site_id)}/lists",
            params={"$top": max(1, min(top, 200))},
        )
        return _success(lists=result.get("value", []))
    except Exception as e:
        logger.error("Error listing SharePoint lists for site %s: %s", site_id, e)
        return _error(str(e))


@mcp.tool()
def sharepoint_list_list_items(site_id: str, list_id: str, top: int = 50) -> str:
    """List items in a SharePoint list, including each item's column values.

    list_id accepts either the list's Graph id or its display name."""
    try:
        result = _graph_request(
            "GET",
            f"/sites/{_site_segment(site_id)}/lists/{url_path_id(list_id, 'list_id')}/items",
            params={"$top": max(1, min(top, 200)), "$expand": "fields"},
        )
        return _success(items=result.get("value", []))
    except Exception as e:
        logger.error(
            "Error listing SharePoint list items for site %s list %s: %s",
            site_id,
            list_id,
            e,
        )
        return _error(str(e))


@mcp.tool()
def sharepoint_create_list_item(site_id: str, list_id: str, fields_json: str) -> str:
    """Create an item in a SharePoint list.

    fields_json is a JSON object string of column-name to value pairs, e.g.
    '{"Title": "New task", "Status": "Not Started"}'. Column names and
    valid values are list-specific; use sharepoint_list_list_items or the
    list's own settings to discover them."""
    try:
        fields = _parse_fields_json(fields_json)
        result = _graph_request(
            "POST",
            f"/sites/{_site_segment(site_id)}/lists/{url_path_id(list_id, 'list_id')}/items",
            body={"fields": fields},
        )
        return _success(item=result)
    except Exception as e:
        logger.error(
            "Error creating SharePoint list item for site %s list %s: %s",
            site_id,
            list_id,
            e,
        )
        return _error(str(e))


@mcp.tool()
def sharepoint_update_list_item(
    site_id: str, list_id: str, item_id: str, fields_json: str
) -> str:
    """Update an existing SharePoint list item's column values.

    fields_json is a JSON object string of the column-name to value pairs
    to change; unset columns are left untouched."""
    try:
        fields = _parse_fields_json(fields_json)
        if not fields:
            raise ValueError("fields_json must contain at least one field to update")
        result = _graph_request(
            "PATCH",
            f"/sites/{_site_segment(site_id)}/lists/{url_path_id(list_id, 'list_id')}"
            f"/items/{url_path_id(item_id, 'item_id')}/fields",
            body=fields,
        )
        return _success(item=result)
    except Exception as e:
        logger.error(
            "Error updating SharePoint list item %s for site %s list %s: %s",
            item_id,
            site_id,
            list_id,
            e,
        )
        return _error(str(e))


@mcp.tool()
def sharepoint_delete_list_item(site_id: str, list_id: str, item_id: str) -> str:
    """Delete an item from a SharePoint list."""
    try:
        _graph_request(
            "DELETE",
            f"/sites/{_site_segment(site_id)}/lists/{url_path_id(list_id, 'list_id')}"
            f"/items/{url_path_id(item_id, 'item_id')}",
        )
        return _success(message="List item deleted successfully")
    except Exception as e:
        logger.error(
            "Error deleting SharePoint list item %s for site %s list %s: %s",
            item_id,
            site_id,
            list_id,
            e,
        )
        return _error(str(e))


if __name__ == "__main__":
    mcp.run()
