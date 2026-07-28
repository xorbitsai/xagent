import logging
from typing import Any
from urllib.parse import quote

from mcp.server.fastmcp import FastMCP

from . import meta_graph
from .meta_graph import (
    GraphAPIError,
)
from .meta_graph import bounded_limit as _bounded_limit
from .meta_graph import error_response as _error
from .meta_graph import graph_error_response as _graph_error
from .meta_graph import graph_request as _graph_request
from .meta_graph import is_public_image_url as _is_public_image_url
from .meta_graph import success_response as _success
from .utils import setup_proxy_env

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("facebook-mcp")

setup_proxy_env()

mcp = FastMCP("facebook-mcp")
requests = meta_graph.requests  # exposed for test monkeypatching


def _page_path(page_id: str, suffix: str) -> str:
    if not page_id or not str(page_id).strip():
        raise ValueError("page_id is required")
    return f"/{quote(str(page_id).strip(), safe='')}/{suffix}"


def _post_path(post_id: str, suffix: str) -> str:
    if not post_id or not str(post_id).strip():
        raise ValueError("post_id is required")
    return f"/{quote(str(post_id).strip(), safe='')}/{suffix}"


def _normalize_page(page: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": page.get("id"),
        "name": page.get("name"),
        "category": page.get("category"),
        "tasks": page.get("tasks", []),
        "has_access_token": bool(page.get("access_token")),
    }


def _list_pages_with_tokens() -> list[dict[str, Any]]:
    result = _graph_request(
        "GET",
        "/me/accounts",
        params={"fields": "id,name,category,tasks,access_token"},
    )
    pages = result.get("data") if isinstance(result, dict) else None
    if not isinstance(pages, list):
        return []
    return [page for page in pages if isinstance(page, dict)]


def _page_access_token(page_id: str) -> str:
    pages = _list_pages_with_tokens()
    for page in pages:
        if str(page.get("id")) == str(page_id):
            token = page.get("access_token")
            if token:
                return str(token)
            raise ValueError(f"Page {page_id} did not include an access token")
    raise ValueError(f"Page {page_id} is not accessible to the connected user")


@mcp.tool()
def facebook_auth_status() -> str:
    """Check whether the injected Meta access token is usable."""
    try:
        me = _graph_request("GET", "/me", params={"fields": "id,name,email"})
        return _success(
            authenticated=True,
            user={
                "id": me.get("id"),
                "name": me.get("name"),
                "email": me.get("email"),
            },
        )
    except GraphAPIError as e:
        logger.error("Error checking Facebook auth status: %s", e)
        return _graph_error(e)
    except Exception as e:
        logger.error("Error checking Facebook auth status: %s", e)
        return _error(str(e))


@mcp.tool()
def facebook_list_pages() -> str:
    """List Facebook Pages accessible to the connected Meta account."""
    try:
        pages = [_normalize_page(page) for page in _list_pages_with_tokens()]
        return _success(pages=pages)
    except GraphAPIError as e:
        logger.error("Error listing Facebook Pages: %s", e)
        return _graph_error(e)
    except Exception as e:
        logger.error("Error listing Facebook Pages: %s", e)
        return _error(str(e))


@mcp.tool()
def facebook_list_page_posts(page_id: str, limit: int = 10) -> str:
    """List recent posts for a Facebook Page by page_id, including like/comment/share counts."""
    try:
        page_token = _page_access_token(page_id)
        result = _graph_request(
            "GET",
            _page_path(page_id, "feed"),
            token=page_token,
            params={
                "fields": (
                    "id,message,created_time,permalink_url,full_picture,status_type,"
                    "likes.limit(0).summary(true),comments.limit(0).summary(true),shares"
                ),
                "limit": _bounded_limit(limit),
            },
        )
        return _success(
            posts=result.get("data", []),
            next_link=result.get("paging", {}).get("next"),
        )
    except GraphAPIError as e:
        logger.error("Error listing Facebook Page posts for %s: %s", page_id, e)
        return _graph_error(e)
    except Exception as e:
        logger.error("Error listing Facebook Page posts for %s: %s", page_id, e)
        return _error(str(e))


@mcp.tool()
def facebook_list_post_comments(page_id: str, post_id: str, limit: int = 10) -> str:
    """List comments left by fans on a Facebook Page post by post_id."""
    try:
        page_token = _page_access_token(page_id)
        result = _graph_request(
            "GET",
            _post_path(post_id, "comments"),
            token=page_token,
            params={
                "fields": "id,message,created_time,from",
                "filter": "stream",
                "limit": _bounded_limit(limit),
            },
        )
        return _success(
            comments=result.get("data", []),
            next_link=result.get("paging", {}).get("next"),
        )
    except GraphAPIError as e:
        logger.error("Error listing comments for post %s: %s", post_id, e)
        return _graph_error(e)
    except Exception as e:
        logger.error("Error listing comments for post %s: %s", post_id, e)
        return _error(str(e))


@mcp.tool()
def facebook_publish_text_post(page_id: str, message: str) -> str:
    """Publish a text post to a Facebook Page by page_id."""
    try:
        if not message.strip():
            raise ValueError("message is required")
        page_token = _page_access_token(page_id)
        result = _graph_request(
            "POST",
            _page_path(page_id, "feed"),
            token=page_token,
            data={"message": message},
        )
        return _success(post_id=result.get("id"))
    except GraphAPIError as e:
        logger.error("Error publishing Facebook Page text post for %s: %s", page_id, e)
        return _graph_error(e)
    except Exception as e:
        logger.error("Error publishing Facebook Page text post for %s: %s", page_id, e)
        return _error(str(e))


@mcp.tool()
def facebook_publish_image_post(
    page_id: str,
    image_url: str,
    caption: str | None = None,
    published: bool = True,
) -> str:
    """Publish an image post to a Facebook Page using a public image URL."""
    try:
        if not _is_public_image_url(image_url):
            raise ValueError("image_url must be a public http or https URL")
        page_token = _page_access_token(page_id)
        data = {
            "url": image_url,
            "published": "true" if published else "false",
        }
        if caption:
            data["caption"] = caption

        result = _graph_request(
            "POST",
            _page_path(page_id, "photos"),
            token=page_token,
            data=data,
        )
        return _success(photo_id=result.get("id"), post_id=result.get("post_id"))
    except GraphAPIError as e:
        logger.error("Error publishing Facebook Page image post for %s: %s", page_id, e)
        return _graph_error(e)
    except Exception as e:
        logger.error("Error publishing Facebook Page image post for %s: %s", page_id, e)
        return _error(str(e))


if __name__ == "__main__":
    mcp.run()
