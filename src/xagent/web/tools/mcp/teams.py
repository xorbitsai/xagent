import json
import logging
import os
from typing import Any
from urllib.parse import quote

import requests
from mcp.server.fastmcp import FastMCP

from .utils import setup_proxy_env

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("teams-mcp")

setup_proxy_env()

mcp = FastMCP("teams-mcp")

GRAPH_BASE_URL = "https://graph.microsoft.com/v1.0"
DEFAULT_TIMEOUT_SECONDS = 30


def _success(**payload: Any) -> str:
    return json.dumps({"status": "success", **payload}, ensure_ascii=False)


def _error(message: str, *, details: Any = None) -> str:
    payload: dict[str, Any] = {"status": "error", "message": message}
    if details is not None:
        payload["details"] = details
    return json.dumps(payload, ensure_ascii=False)


def _content_type(value: str) -> str:
    normalized = value.strip().lower()
    if normalized not in {"text", "html"}:
        raise ValueError("content_type must be either 'text' or 'html'")
    return normalized


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
    extra_headers: dict[str, str] | None = None,
) -> Any:
    response = requests.request(
        method=method,
        url=f"{GRAPH_BASE_URL}{path}",
        headers=_graph_headers(extra_headers),
        params=params,
        json=body,
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

    if response.status_code == 204 or not response.content:
        return {}
    return response.json()


def _message_body(content: str, content_type: str) -> dict[str, Any]:
    return {"body": {"contentType": _content_type(content_type), "content": content}}


@mcp.tool()
def auth_status() -> str:
    """Check whether the injected Microsoft Graph token is usable."""
    try:
        me = _graph_request(
            "GET",
            "/me",
            params={"$select": "id,displayName,userPrincipalName,mail"},
        )
        return _success(
            authenticated=True,
            user={
                "id": me.get("id"),
                "displayName": me.get("displayName"),
                "userPrincipalName": me.get("userPrincipalName"),
                "mail": me.get("mail"),
            },
        )
    except Exception as e:
        logger.error("Error checking Teams auth status: %s", e)
        return _error(str(e))


@mcp.tool()
def get_current_user() -> str:
    """Get the current Microsoft 365 user profile from Microsoft Graph."""
    try:
        me = _graph_request(
            "GET",
            "/me",
            params={
                "$select": (
                    "id,displayName,givenName,surname,userPrincipalName,mail,"
                    "jobTitle,department,mobilePhone,officeLocation"
                )
            },
        )
        return _success(user=me)
    except Exception as e:
        logger.error("Error getting current Teams user: %s", e)
        return _error(str(e))


@mcp.tool()
def list_teams(top: int = 25) -> str:
    """List the Microsoft Teams that the current user has joined."""
    try:
        result = _graph_request(
            "GET",
            "/me/joinedTeams",
            params={"$top": max(1, min(top, 100))},
        )
        return _success(teams=result.get("value", []))
    except Exception as e:
        logger.error("Error listing Teams: %s", e)
        return _error(str(e))


@mcp.tool()
def list_channels(team_id: str, top: int = 50) -> str:
    """List channels for a Microsoft Team by team_id."""
    try:
        result = _graph_request(
            "GET",
            f"/teams/{quote(team_id, safe='')}/channels",
            params={"$top": max(1, min(top, 200))},
        )
        return _success(channels=result.get("value", []))
    except Exception as e:
        logger.error("Error listing channels for team %s: %s", team_id, e)
        return _error(str(e))


@mcp.tool()
def list_team_members(team_id: str, top: int = 50) -> str:
    """List members of a Microsoft Team by team_id."""
    try:
        result = _graph_request(
            "GET",
            f"/teams/{quote(team_id, safe='')}/members",
            params={"$top": max(1, min(top, 200))},
        )
        return _success(members=result.get("value", []))
    except Exception as e:
        logger.error("Error listing team members for %s: %s", team_id, e)
        return _error(str(e))


@mcp.tool()
def get_channel_messages(team_id: str, channel_id: str, top: int = 20) -> str:
    """Get recent messages from a team channel."""
    try:
        result = _graph_request(
            "GET",
            (
                f"/teams/{quote(team_id, safe='')}/channels/"
                f"{quote(channel_id, safe='')}/messages"
            ),
            params={"$top": max(1, min(top, 50))},
        )
        return _success(
            messages=result.get("value", []),
            next_link=result.get("@odata.nextLink"),
        )
    except Exception as e:
        logger.error(
            "Error getting channel messages for team %s channel %s: %s",
            team_id,
            channel_id,
            e,
        )
        return _error(str(e))


@mcp.tool()
def send_channel_message(
    team_id: str,
    channel_id: str,
    message: str,
    content_type: str = "text",
) -> str:
    """Send a message to a Microsoft Teams channel."""
    try:
        result = _graph_request(
            "POST",
            (
                f"/teams/{quote(team_id, safe='')}/channels/"
                f"{quote(channel_id, safe='')}/messages"
            ),
            body=_message_body(message, content_type),
        )
        return _success(message=result)
    except Exception as e:
        logger.error(
            "Error sending channel message for team %s channel %s: %s",
            team_id,
            channel_id,
            e,
        )
        return _error(str(e))


@mcp.tool()
def list_chats(top: int = 20) -> str:
    """List chats for the current Microsoft 365 user."""
    try:
        result = _graph_request(
            "GET",
            "/me/chats",
            params={"$top": max(1, min(top, 100))},
        )
        return _success(chats=result.get("value", []))
    except Exception as e:
        logger.error("Error listing chats: %s", e)
        return _error(str(e))


@mcp.tool()
def get_chat_messages(chat_id: str, top: int = 20) -> str:
    """Get recent messages from a Microsoft Teams chat."""
    try:
        result = _graph_request(
            "GET",
            f"/chats/{quote(chat_id, safe='')}/messages",
            params={"$top": max(1, min(top, 50))},
        )
        return _success(
            messages=result.get("value", []),
            next_link=result.get("@odata.nextLink"),
        )
    except Exception as e:
        logger.error("Error getting chat messages for %s: %s", chat_id, e)
        return _error(str(e))


@mcp.tool()
def send_chat_message(
    chat_id: str,
    message: str,
    content_type: str = "text",
) -> str:
    """Send a message to a Microsoft Teams chat."""
    try:
        result = _graph_request(
            "POST",
            f"/chats/{quote(chat_id, safe='')}/messages",
            body=_message_body(message, content_type),
        )
        return _success(message=result)
    except Exception as e:
        logger.error("Error sending chat message for %s: %s", chat_id, e)
        return _error(str(e))


if __name__ == "__main__":
    mcp.run()
