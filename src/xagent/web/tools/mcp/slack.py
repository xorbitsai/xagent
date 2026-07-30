import json
import logging
import os
from typing import Any

import requests
from mcp.server.fastmcp import FastMCP

from .utils import setup_proxy_env

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("slack-mcp")

# Ensure standard proxy environment variables are set to prevent hanging requests
setup_proxy_env()

mcp = FastMCP("slack-mcp")

SLACK_BASE_URL = "https://slack.com/api"
DEFAULT_TIMEOUT_SECONDS = 30
MAX_PAGES = 20


def _success(**payload: Any) -> str:
    return json.dumps({"status": "success", **payload}, ensure_ascii=False)


def _error(message: str) -> str:
    return json.dumps({"status": "error", "message": message}, ensure_ascii=False)


def _headers() -> dict[str, str]:
    access_token = os.environ.get("SLACK_ACCESS_TOKEN")
    if not access_token:
        raise ValueError("SLACK_ACCESS_TOKEN environment variable is missing")
    return {"Authorization": f"Bearer {access_token}"}


def _request(
    method: str,
    path: str,
    *,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Call a Slack Web API method.

    Slack always answers with HTTP 200, even on failure — success/failure is
    only signalled by the body's "ok" boolean plus an "error" code, so error
    detection here is on that field, not on raise_for_status().
    """
    response = requests.request(
        method=method,
        url=f"{SLACK_BASE_URL}/{path}",
        headers=_headers(),
        params=params,
        timeout=DEFAULT_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    payload: dict[str, Any] = response.json()
    if not payload.get("ok"):
        raise RuntimeError(payload.get("error") or "Unknown Slack API error")
    return payload


@mcp.tool()
def slack_list_channels(exclude_archived: bool = True) -> str:
    """
    List public channels in the connected Slack workspace (id, name, is_archived).
    Use this to resolve a channel name to an id before posting, or to find the
    right channel when the user names one imprecisely.
    """
    try:
        channels: list[dict[str, Any]] = []
        cursor: str | None = None
        for _ in range(MAX_PAGES):
            params: dict[str, Any] = {
                "types": "public_channel",
                "exclude_archived": exclude_archived,
                "limit": 200,
            }
            if cursor:
                params["cursor"] = cursor
            result = _request("GET", "conversations.list", params=params)
            channels.extend(
                {
                    "id": channel.get("id"),
                    "name": channel.get("name"),
                    "is_archived": channel.get("is_archived", False),
                }
                for channel in result.get("channels", [])
            )
            cursor = (result.get("response_metadata") or {}).get("next_cursor")
            if not cursor:
                break
        return _success(channels=channels)
    except Exception as e:
        logger.error(f"Error listing Slack channels: {e}")
        return _error(str(e))


@mcp.tool()
def slack_post_message(channel: str, text: str) -> str:
    """
    Post a message to a Slack channel.
    channel: a channel id (e.g. "C0123456789") or name (e.g. "#incidents" or "incidents").
    text: the message body (plain text or Slack mrkdwn).
    """
    try:
        result = _request(
            "POST",
            "chat.postMessage",
            params={"channel": channel, "text": text},
        )
        return _success(channel=result.get("channel"), ts=result.get("ts"))
    except Exception as e:
        logger.error(f"Error posting Slack message to {channel}: {e}")
        return _error(str(e))


if __name__ == "__main__":
    mcp.run()
