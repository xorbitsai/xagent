import json
import logging
import os
import time
from typing import Any
from urllib.parse import quote

import requests
from mcp.server.fastmcp import FastMCP

from .utils import setup_proxy_env

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("intercom-mcp")

setup_proxy_env()

mcp = FastMCP("intercom-mcp")

# No region prefix: Intercom auto-routes each request to the workspace's
# actual hosting region (US/EU/AU) based on the access token, so this one
# host covers every region without per-region configuration.
INTERCOM_BASE_URL = "https://api.intercom.io"
# Pinned so a future default-version bump on Intercom's side can't silently
# change response shapes underneath these tools. Re-check this against
# Intercom's API changelog (developers.intercom.com/docs/references) whenever
# 2.11 is scheduled for deprecation, and bump deliberately rather than
# reactively once a version stops resolving.
INTERCOM_API_VERSION = "2.11"
DEFAULT_TIMEOUT_SECONDS = 30
# Conversation/contact endpoints are rate-limited per-app; on a 429 with a
# small Retry-After we wait once and retry rather than failing outright,
# mirroring the same bounded-retry policy as the Slack sibling module.
MAX_RETRY_AFTER_SECONDS = 30
# An HTML gateway error page (or similar) landing in an unstructured error
# body must not be forwarded to the LLM/logs verbatim and unbounded, same
# reasoning and cap as the Zoom sibling module.
MAX_ERROR_RESPONSE_TEXT_CHARS = 1000

# Keyed by access token, defense-in-depth rather than an active optimization:
# every MCP tool call currently spawns and tears down a fresh subprocess
# (mcp_adapter._execute_mcp_call -> create_session), so in practice a process
# only ever sees one token and this dict never holds more than one entry.
# Keying by token still costs nothing and keeps this safe against a future
# pooled/reused-subprocess architecture, where a bare global would leak one
# user's admin identity into another user's replies.
_admin_id_cache: dict[str, str] = {}


def _success(**payload: Any) -> str:
    return json.dumps({"status": "success", **payload}, ensure_ascii=False)


def _error(message: str) -> str:
    return json.dumps({"status": "error", "message": message}, ensure_ascii=False)


def _headers() -> dict[str, str]:
    token = os.environ.get("INTERCOM_ACCESS_TOKEN")
    if not token:
        raise ValueError("INTERCOM_ACCESS_TOKEN environment variable is missing")
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Intercom-Version": INTERCOM_API_VERSION,
    }


def _request(
    method: str,
    path: str,
    *,
    body: dict[str, Any] | None = None,
) -> Any:
    for attempt in (0, 1):
        response = requests.request(
            method=method,
            url=f"{INTERCOM_BASE_URL}{path}",
            headers=_headers(),
            json=body,
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

    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        response_text = response.text.strip()
        if len(response_text) > MAX_ERROR_RESPONSE_TEXT_CHARS:
            response_text = response_text[:MAX_ERROR_RESPONSE_TEXT_CHARS] + (
                "... [truncated]"
            )
        message = str(exc)
        if response_text:
            message = f"{message} - {response_text}"
        raise RuntimeError(message) from exc

    if response.status_code == 204 or not response.content:
        return {}
    return response.json()


def _current_admin_id() -> str:
    """Resolve the admin id behind the connected access token, cached per token.

    Replying to a conversation as an admin requires an admin_id, but the OAuth
    connection only gives us a token -- not the identity it was created for.
    """
    token = os.environ.get("INTERCOM_ACCESS_TOKEN")
    if not token:
        raise ValueError("INTERCOM_ACCESS_TOKEN environment variable is missing")
    if token in _admin_id_cache:
        return _admin_id_cache[token]
    me = _request("GET", "/me")
    admin_id = me.get("id")
    if not admin_id:
        raise RuntimeError("Could not resolve the Intercom admin id for this token")
    _admin_id_cache[token] = str(admin_id)
    return _admin_id_cache[token]


def _contact_summary(contact: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": contact.get("id"),
        "name": contact.get("name"),
        "email": contact.get("email"),
        "phone": contact.get("phone"),
        "role": contact.get("role"),
        "last_seen_at": contact.get("last_seen_at"),
    }


def _conversation_summary(conversation: dict[str, Any]) -> dict[str, Any]:
    source = conversation.get("source") or {}
    return {
        "id": conversation.get("id"),
        "state": conversation.get("state"),
        "title": conversation.get("title"),
        "subject": source.get("subject"),
        "preview": source.get("body"),
        "created_at": conversation.get("created_at"),
        "updated_at": conversation.get("updated_at"),
    }


@mcp.tool()
def intercom_search_contacts(query: str, limit: int = 10) -> str:
    """
    Search Intercom contacts whose name or email contains `query`.
    """
    try:
        if not query.strip():
            raise ValueError("query must not be blank")
        body = {
            "query": {
                "operator": "OR",
                "value": [
                    {"field": "name", "operator": "~", "value": query},
                    {"field": "email", "operator": "~", "value": query},
                ],
            },
            "pagination": {"per_page": max(1, min(limit, 100))},
        }
        result = _request("POST", "/contacts/search", body=body)
        contacts = [_contact_summary(c) for c in result.get("data", [])]
        return _success(
            contacts=contacts, total=result.get("total_count", len(contacts))
        )
    except Exception as e:
        logger.error(f"Error searching contacts: {e}")
        return _error(str(e))


@mcp.tool()
def intercom_get_contact(contact_id: str) -> str:
    """
    Get an Intercom contact by id.
    """
    try:
        contact = _request("GET", f"/contacts/{quote(contact_id, safe='')}")
        return _success(contact=_contact_summary(contact))
    except Exception as e:
        logger.error(f"Error getting contact: {e}")
        return _error(str(e))


@mcp.tool()
def intercom_list_conversations(state: str = "open", limit: int = 20) -> str:
    """
    List Intercom conversations. `state` is one of "open", "closed", "snoozed",
    or "all". Returns at most `limit` conversations (max 100), most recently
    updated first. `has_more` is true when the search matched more
    conversations than were returned.
    """
    try:
        valid_states = {"open", "closed", "snoozed", "all"}
        if state not in valid_states:
            raise ValueError(f"state must be one of {sorted(valid_states)}")

        # `query` is a required field on /conversations/search (unlike
        # GET /conversations, which has no filter and a different
        # pagination/response shape) -- for "all" we still filter, on a
        # predicate every real conversation satisfies, rather than switch
        # endpoints just to express "no filter".
        if state == "all":
            query: dict[str, Any] = {
                "field": "created_at",
                "operator": ">",
                "value": 0,
            }
        else:
            query = {"field": "state", "operator": "=", "value": state}

        per_page = max(1, min(limit, 100))
        body: dict[str, Any] = {
            "query": query,
            "pagination": {"per_page": per_page},
            # `sort` is a common structure documented for both
            # /contacts/search and /conversations/search (Intercom docs,
            # "Pagination & Sorting (Search)"), not contacts-only.
            "sort": {"field": "updated_at", "order": "descending"},
        }

        result = _request("POST", "/conversations/search", body=body)
        conversations = [
            _conversation_summary(c) for c in result.get("conversations", [])
        ]
        # Intercom's search pagination carries a "next" cursor object under
        # "pages" only when more results remain past this page.
        has_more = bool((result.get("pages") or {}).get("next"))
        return _success(
            conversations=conversations,
            total_count=result.get("total_count", len(conversations)),
            has_more=has_more,
        )
    except Exception as e:
        logger.error(f"Error listing conversations: {e}")
        return _error(str(e))


@mcp.tool()
def intercom_get_conversation(conversation_id: str) -> str:
    """
    Get a full Intercom conversation by id, including the message thread
    (conversation_parts).
    """
    try:
        conversation = _request(
            "GET", f"/conversations/{quote(conversation_id, safe='')}"
        )
        parts = (conversation.get("conversation_parts") or {}).get(
            "conversation_parts", []
        )
        return _success(
            conversation=_conversation_summary(conversation),
            parts=[
                {
                    "id": part.get("id"),
                    "author": (part.get("author") or {}).get("name"),
                    "body": part.get("body"),
                    "created_at": part.get("created_at"),
                }
                for part in parts
            ],
        )
    except Exception as e:
        logger.error(f"Error getting conversation: {e}")
        return _error(str(e))


@mcp.tool()
def intercom_reply_to_conversation(conversation_id: str, body: str) -> str:
    """
    Reply to a customer on an Intercom conversation, as the connected admin.
    """
    try:
        if not body.strip():
            raise ValueError("body must not be blank")
        reply = _request(
            "POST",
            f"/conversations/{quote(conversation_id, safe='')}/reply",
            body={
                "message_type": "comment",
                "type": "admin",
                "admin_id": _current_admin_id(),
                "body": body,
            },
        )
        return _success(conversation=_conversation_summary(reply))
    except Exception as e:
        logger.error(f"Error replying to conversation: {e}")
        return _error(str(e))


@mcp.tool()
def intercom_add_internal_note(conversation_id: str, body: str) -> str:
    """
    Add an internal note to an Intercom conversation. Internal notes are only
    visible to teammates, never to the customer.
    """
    try:
        if not body.strip():
            raise ValueError("body must not be blank")
        reply = _request(
            "POST",
            f"/conversations/{quote(conversation_id, safe='')}/reply",
            body={
                "message_type": "note",
                "type": "admin",
                "admin_id": _current_admin_id(),
                "body": body,
            },
        )
        return _success(conversation=_conversation_summary(reply))
    except Exception as e:
        logger.error(f"Error adding internal note: {e}")
        return _error(str(e))


@mcp.tool()
def intercom_close_conversation(conversation_id: str) -> str:
    """
    Close an Intercom conversation.
    """
    try:
        conversation = _request(
            "POST",
            f"/conversations/{quote(conversation_id, safe='')}/parts",
            body={
                "message_type": "close",
                "type": "admin",
                "admin_id": _current_admin_id(),
            },
        )
        return _success(conversation=_conversation_summary(conversation))
    except Exception as e:
        logger.error(f"Error closing conversation: {e}")
        return _error(str(e))


if __name__ == "__main__":
    mcp.run()
