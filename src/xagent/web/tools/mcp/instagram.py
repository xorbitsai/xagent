import logging
import time
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
logger = logging.getLogger("instagram-mcp")

setup_proxy_env()

mcp = FastMCP("instagram-mcp")
requests = meta_graph.requests  # exposed for test monkeypatching

LINKED_ACCOUNT_FIELDS = (
    "id,name,category,tasks,access_token,"
    "instagram_business_account{id,username,name,profile_picture_url}"
)
PROFILE_FIELDS = (
    "id,username,name,biography,profile_picture_url,followers_count,"
    "follows_count,media_count,website"
)
MEDIA_FIELDS = (
    "id,caption,media_type,media_url,permalink,timestamp,username,thumbnail_url"
)
MEDIA_CONTAINER_STATUS_FIELDS = "status_code,status"
MEDIA_CONTAINER_MAX_POLLS = 30
MEDIA_CONTAINER_POLL_INTERVAL_SECONDS = 1.0
_MEDIA_CONTAINER_READY_STATUSES = frozenset({"FINISHED", "PUBLISHED"})
_MEDIA_CONTAINER_FAILED_STATUSES = frozenset({"ERROR", "EXPIRED"})


def _graph_path(object_id: str, suffix: str | None = None) -> str:
    if not object_id.strip():
        raise ValueError("object id is required")
    path = f"/{quote(object_id.strip(), safe='')}"
    if suffix:
        path = f"{path}/{suffix}"
    return path


def _page_summary(page: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": page.get("id"),
        "name": page.get("name"),
        "category": page.get("category"),
        "tasks": page.get("tasks", []),
        "has_access_token": bool(page.get("access_token")),
    }


def _linked_account(page: dict[str, Any]) -> dict[str, Any] | None:
    instagram_account = page.get("instagram_business_account")
    if not isinstance(instagram_account, dict):
        return None
    return {
        "page": _page_summary(page),
        "instagram_account": {
            "id": instagram_account.get("id"),
            "username": instagram_account.get("username"),
            "name": instagram_account.get("name"),
            "profile_picture_url": instagram_account.get("profile_picture_url"),
        },
    }


def _list_pages_with_instagram_accounts() -> list[dict[str, Any]]:
    result = _graph_request(
        "GET",
        "/me/accounts",
        params={"fields": LINKED_ACCOUNT_FIELDS},
    )
    pages = result.get("data") if isinstance(result, dict) else None
    if not isinstance(pages, list):
        return []
    accounts: list[dict[str, Any]] = []
    for page in pages:
        if not isinstance(page, dict):
            continue
        account = _linked_account(page)
        if account:
            accounts.append(account)
    return accounts


def _wait_for_media_container(container_id: str) -> None:
    for attempt in range(MEDIA_CONTAINER_MAX_POLLS):
        status = _graph_request(
            "GET",
            _graph_path(container_id),
            params={"fields": MEDIA_CONTAINER_STATUS_FIELDS},
        )
        status_code = str(status.get("status_code") or "").upper()
        if status_code in _MEDIA_CONTAINER_READY_STATUSES:
            return
        if status_code in _MEDIA_CONTAINER_FAILED_STATUSES:
            raise ValueError(
                f"Instagram media processing failed for container {container_id}"
            )
        if attempt + 1 < MEDIA_CONTAINER_MAX_POLLS:
            time.sleep(MEDIA_CONTAINER_POLL_INTERVAL_SECONDS)
    raise TimeoutError(
        f"Instagram media processing timed out for container {container_id}"
    )


@mcp.tool()
def instagram_auth_status() -> str:
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
        logger.error("Error checking Instagram auth status: %s", e)
        return _graph_error(e)
    except Exception as e:
        logger.error("Error checking Instagram auth status: %s", e)
        return _error(str(e))


@mcp.tool()
def instagram_list_linked_accounts() -> str:
    """List Facebook Pages that have linked Instagram professional accounts."""
    try:
        return _success(accounts=_list_pages_with_instagram_accounts())
    except GraphAPIError as e:
        logger.error("Error listing Instagram linked accounts: %s", e)
        return _graph_error(e)
    except Exception as e:
        logger.error("Error listing Instagram linked accounts: %s", e)
        return _error(str(e))


@mcp.tool()
def instagram_get_profile(instagram_account_id: str) -> str:
    """Get profile information for an Instagram professional account."""
    try:
        profile = _graph_request(
            "GET",
            _graph_path(instagram_account_id),
            params={"fields": PROFILE_FIELDS},
        )
        return _success(profile=profile)
    except GraphAPIError as e:
        logger.error(
            "Error getting Instagram profile for %s: %s", instagram_account_id, e
        )
        return _graph_error(e)
    except Exception as e:
        logger.error(
            "Error getting Instagram profile for %s: %s", instagram_account_id, e
        )
        return _error(str(e))


@mcp.tool()
def instagram_list_media(instagram_account_id: str, limit: int = 10) -> str:
    """List recent media for an Instagram professional account."""
    try:
        result = _graph_request(
            "GET",
            _graph_path(instagram_account_id, "media"),
            params={"fields": MEDIA_FIELDS, "limit": _bounded_limit(limit)},
        )
        return _success(
            media=result.get("data", []),
            next_link=result.get("paging", {}).get("next"),
        )
    except GraphAPIError as e:
        logger.error(
            "Error listing Instagram media for %s: %s", instagram_account_id, e
        )
        return _graph_error(e)
    except Exception as e:
        logger.error(
            "Error listing Instagram media for %s: %s", instagram_account_id, e
        )
        return _error(str(e))


@mcp.tool()
def instagram_publish_image(
    instagram_account_id: str,
    image_url: str | None = None,
    caption: str | None = None,
    file_path: str | None = None,
) -> str:
    """Create and publish an Instagram image from a URL or workspace FileRef.

    Provide exactly one of image_url and file_path. The Xagent host resolves a
    registered file_id (or file:<id>) supplied as file_path into a short-lived
    provider URL before this connector runs.
    """
    resolved_image_url: str | None = None
    try:
        url_value = (
            image_url if isinstance(image_url, str) and image_url.strip() else None
        )
        file_value = (
            file_path if isinstance(file_path, str) and file_path.strip() else None
        )
        if (url_value is None) == (file_value is None):
            raise ValueError("Provide exactly one of image_url or file_path")
        if file_value is not None:
            raise ValueError(
                "file_path must be a registered file_id or file:<id> resolved by "
                "the Xagent host"
            )
        if url_value is None or not _is_public_image_url(url_value):
            raise ValueError("image_url must be a public http or https URL")
        resolved_image_url = url_value
        data = {"image_url": resolved_image_url}
        if caption:
            data["caption"] = caption

        container = _graph_request(
            "POST",
            _graph_path(instagram_account_id, "media"),
            data=data,
        )
        container_id = container.get("id")
        if not container_id:
            raise ValueError("Instagram media container response did not include id")

        _wait_for_media_container(str(container_id))
        published = _graph_request(
            "POST",
            _graph_path(instagram_account_id, "media_publish"),
            data={"creation_id": str(container_id)},
        )
        return _success(container_id=container_id, media_id=published.get("id"))
    except GraphAPIError as e:
        if resolved_image_url:
            e.sensitive_values.add(resolved_image_url)
            e.sensitive_values.add(resolved_image_url.replace("/", r"\/"))
        logger.error(
            "Instagram Graph API rejected image publishing for %s",
            instagram_account_id,
        )
        return _graph_error(e)
    except Exception as e:
        logger.error(
            "Error publishing Instagram image for %s: %s", instagram_account_id, e
        )
        return _error(str(e))


if __name__ == "__main__":
    mcp.run()
