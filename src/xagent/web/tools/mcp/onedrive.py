import base64
import json
import logging
import os
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
    """Upload or overwrite a UTF-8 text file in OneDrive by path."""
    try:
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
