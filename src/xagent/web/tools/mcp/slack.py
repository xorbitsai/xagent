import json
import logging
import os
import re
import time
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
MAX_CHANNELS = 1000
# conversations.list is Tier-2 rate-limited (~20 req/min); on a 429 with a
# small Retry-After we wait once and retry rather than failing the page.
MAX_RETRY_AFTER_SECONDS = 30

# Slack channel/user/group ids are uppercase alphanumerics with a letter
# prefix; channel *names* are forced lowercase by Slack, so this can't
# misclassify a name as an id.
_SLACK_ID_PATTERN = re.compile(r"^[A-Z][A-Z0-9]{5,}$")


def _success(**payload: Any) -> str:
    return json.dumps({"status": "success", **payload}, ensure_ascii=False)


def _error(message: str) -> str:
    return json.dumps({"status": "error", "message": message}, ensure_ascii=False)


def _headers() -> dict[str, str]:
    access_token = os.environ.get("SLACK_ACCESS_TOKEN")
    if not access_token:
        raise ValueError("SLACK_ACCESS_TOKEN environment variable is missing")
    return {"Authorization": f"Bearer {access_token}"}


def _normalize_channel(channel: str) -> str:
    """Accept a channel id ("C0123456789") or a name with or without the
    leading "#"; bare names get the "#" prepended so all three documented
    forms reach the API in a shape it accepts."""
    value = channel.strip()
    if value.startswith("#") or _SLACK_ID_PATTERN.match(value):
        return value
    return f"#{value}"


def _request(
    method: str,
    path: str,
    *,
    params: dict[str, Any] | None = None,
    json_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Call a Slack Web API method.

    Slack always answers with HTTP 200, even on failure — success/failure is
    only signalled by the body's "ok" boolean plus an "error" code, so error
    detection here is on that field, not on raise_for_status().

    Write calls pass json_data rather than params: message text can exceed
    URL length limits, and query strings are commonly logged in plaintext by
    proxies/load balancers, which would leak message content.

    Rate limiting is the one non-200 case worth special handling: on a 429
    with a small Retry-After, wait once and retry instead of failing.
    """
    for attempt in (0, 1):
        response = requests.request(
            method=method,
            url=f"{SLACK_BASE_URL}/{path}",
            headers=_headers(),
            params=params,
            json=json_data,
            timeout=DEFAULT_TIMEOUT_SECONDS,
        )
        if response.status_code == 429 and attempt == 0:
            try:
                retry_after = int(response.headers.get("Retry-After", "0"))
            except ValueError:
                retry_after = 0
            if 0 < retry_after <= MAX_RETRY_AFTER_SECONDS:
                time.sleep(retry_after)
                continue
        break
    response.raise_for_status()
    payload: dict[str, Any] = response.json()
    if not payload.get("ok"):
        raise RuntimeError(payload.get("error") or "Unknown Slack API error")
    return payload


@mcp.tool()
def slack_list_channels(
    exclude_archived: bool = True,
    name_contains: str = "",
    limit: int = MAX_CHANNELS,
) -> str:
    """
    List public channels in the connected Slack workspace (id, name, is_archived).
    Use this to resolve a channel name to an id before posting.
    name_contains: optional case-insensitive substring filter on the channel
    name — pass it when looking for a specific channel in a large workspace
    instead of listing everything.
    limit: maximum channels to return (default 1000).
    The response includes truncated=true when there were more channels than
    returned (more pages, the limit was hit, or a page failed mid-way — a
    partial list is returned rather than discarded).
    """
    channels: list[dict[str, Any]] = []
    needle = name_contains.strip().lower()
    max_channels = max(1, min(int(limit), MAX_CHANNELS))
    cursor: str | None = None
    truncated = False
    pages_scanned = 0
    try:
        for _ in range(MAX_PAGES):
            params: dict[str, Any] = {
                "types": "public_channel",
                # Slack expects a lowercase "true"/"false" string; requests
                # would otherwise serialize the Python bool as "True"/"False",
                # which Slack's parser doesn't recognize as a boolean.
                "exclude_archived": "true" if exclude_archived else "false",
                "limit": 200,
            }
            if cursor:
                params["cursor"] = cursor
            try:
                result = _request("GET", "conversations.list", params=params)
            except Exception as page_exc:
                if not pages_scanned:
                    raise
                # A mid-pagination failure (e.g. a rate limit that outlived
                # the single retry) must not discard the pages already
                # fetched — return the partial list with a marker instead.
                # This can legitimately be an empty list when name_contains
                # filtered out everything scanned so far.
                logger.warning(f"Slack channel pagination stopped early: {page_exc}")
                return _success(channels=channels, truncated=True, error=str(page_exc))
            pages_scanned += 1
            raw_channels = result.get("channels") or []
            for index, channel in enumerate(raw_channels):
                name = str(channel.get("name") or "")
                if needle and needle not in name.lower():
                    continue
                channels.append(
                    {
                        "id": channel.get("id"),
                        "name": channel.get("name"),
                        "is_archived": channel.get("is_archived", False),
                    }
                )
                if len(channels) >= max_channels:
                    # Only truncated if there is actually more data: more raw
                    # results left in this page, or another page pending —
                    # hitting the limit on the very last item of the very
                    # last page is not truncation.
                    more_in_page = index + 1 < len(raw_channels)
                    more_pages = bool(
                        (result.get("response_metadata") or {}).get("next_cursor")
                    )
                    return _success(
                        channels=channels, truncated=more_in_page or more_pages
                    )
            cursor = (result.get("response_metadata") or {}).get("next_cursor")
            if not cursor:
                break
        else:
            # MAX_PAGES exhausted with a cursor still pending.
            truncated = bool(cursor)
        return _success(channels=channels, truncated=truncated)
    except Exception as e:
        logger.error(f"Error listing Slack channels: {e}")
        return _error(str(e))


@mcp.tool()
def slack_post_message(channel: str, text: str) -> str:
    """
    Post a message to a Slack channel.
    channel: a channel id (e.g. "C0123456789") or name (e.g. "#incidents" or
    "incidents" — a bare name is normalized to "#incidents").
    text: the message body (plain text or Slack mrkdwn).
    """
    try:
        result = _request(
            "POST",
            "chat.postMessage",
            json_data={"channel": _normalize_channel(channel), "text": text},
        )
        return _success(channel=result.get("channel"), ts=result.get("ts"))
    except Exception as e:
        logger.error(f"Error posting Slack message to {channel}: {e}")
        return _error(str(e))


if __name__ == "__main__":
    mcp.run()
