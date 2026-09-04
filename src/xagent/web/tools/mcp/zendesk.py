import json
import logging
import re
import socket
import time
from collections.abc import Callable
from os import environ
from typing import Any

import requests
from mcp.server.fastmcp import FastMCP

from ....config import get_tool_max_output_length
from ....core.utils.security import (
    PrivateNetworkHostError,
    redact_sensitive_text,
    reject_private_network_host,
)
from ...utils.graphql_errors import truncate_error_text
from .utils import (
    clamp_limit,
    resolve_id_from_url,
    setup_proxy_env,
    url_path_id,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("zendesk-mcp")

# Ensure standard proxy environment variables are set to prevent hanging requests
setup_proxy_env()

mcp = FastMCP("zendesk-mcp")

# A shared Session (HTTP keep-alive / connection pooling) rather than a bare
# requests.request() per call -- most benefit accrues to this module's own
# 429-retry (a second request to the same host right after the first), and
# to any single tool call that happens to make more than one request; a
# fresh connection per call is otherwise a fixed cost this avoids for free.
_session = requests.Session()

DEFAULT_TIMEOUT_SECONDS = 30
MAX_LIMIT = 100
MAX_RETRY_AFTER_SECONDS = 30

# Only a DNS *label* (no dots, scheme, port, or slashes) is ever accepted, so
# the string itself can never name a host outside "*.zendesk.com" -- but a
# perfectly legitimate hostname can still be rebound by DNS to a private/
# internal address at request time (this is orthogonal to who chose the
# hostname string), so _base_url() below still resolves and checks every
# address, same defense-in-depth posthog.py's _base_url() applies to its own
# two hardcoded-enum hostnames.
_SUBDOMAIN_PATTERN = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")

# Zendesk's own documented ceilings on how deep a result window can be
# paginated (page * per_page must stay under this) -- past it Zendesk itself
# returns an opaque HTTP error rather than a clean "no more results". Checked
# client-side so a caller mechanically incrementing `page` gets a predictable
# empty page instead of that error. The two search endpoints have different
# documented ceilings -- the unified /search.json is capped at 1,000 results,
# while /users/search.json goes to 10,000 -- so they are NOT interchangeable
# despite both being "search" endpoints.
_MAX_SEARCH_RESULT_WINDOW = 1000
_MAX_USER_SEARCH_RESULT_WINDOW = 10000

# Accepts a ticket/user/organization id pasted as a Zendesk agent UI URL
# (e.g. "https://acme.zendesk.com/agent/tickets/123") in addition to a bare
# numeric id -- an agent copying a link out of the Zendesk UI is a common
# enough input shape that requiring the bare id only would otherwise surface
# as an opaque Pydantic ValidationError.
_TICKET_URL_ID_PATTERN = re.compile(r"/agent/tickets/(\d+)")
_USER_URL_ID_PATTERN = re.compile(r"/agent/users/(\d+)")
_ORGANIZATION_URL_ID_PATTERN = re.compile(r"/agent/organizations/(\d+)")


def _success(**payload: Any) -> str:
    return json.dumps({"status": "success", **payload}, ensure_ascii=False)


def _error(message: str) -> str:
    return json.dumps({"status": "error", "message": message}, ensure_ascii=False)


def _auth() -> tuple[str, str]:
    """Zendesk's Basic-Auth-with-API-token scheme: the username is the
    agent's email address suffixed with the literal "/token", and the
    password is the API token itself (generated self-serve, Admin Center ->
    Apps and integrations -> APIs -> Zendesk API -> Add API token -- no
    review). Zendesk's own docs mark this auth method "(deprecated)" in
    favor of OAuth, but state it remains fully supported with no
    announced removal date; OAuth access tokens for a private (non-
    marketplace) client would need the same review-free bar but add a
    full authorization-code exchange this module doesn't otherwise need.
    """
    # Stripped like ZENDESK_SUBDOMAIN below: a value that's only whitespace
    # (e.g. a trailing newline from pasting the credential into a
    # connect-flow form) is not a usable credential and should be treated
    # as missing here rather than sent to Zendesk as a malformed Basic Auth
    # username/password, matching mixpanel.py's identical stripping.
    email = environ.get("ZENDESK_EMAIL", "").strip()
    api_token = environ.get("ZENDESK_API_TOKEN", "").strip()
    if not email:
        raise ValueError("ZENDESK_EMAIL environment variable is missing")
    if not api_token:
        raise ValueError("ZENDESK_API_TOKEN environment variable is missing")
    return (f"{email}/token", api_token)


def _base_url() -> str:
    subdomain = environ.get("ZENDESK_SUBDOMAIN", "").strip().lower()
    if not subdomain:
        raise ValueError("ZENDESK_SUBDOMAIN environment variable is missing")
    if not _SUBDOMAIN_PATTERN.match(subdomain):
        raise ValueError(
            "ZENDESK_SUBDOMAIN must be a single DNS label (letters, digits, "
            "and hyphens only, no leading/trailing hyphen) -- pass just the "
            "subdomain, e.g. 'acme' for acme.zendesk.com, not a full URL"
        )
    hostname = f"{subdomain}.zendesk.com"
    try:
        resolved = socket.getaddrinfo(
            hostname, 443, socket.AF_UNSPEC, socket.SOCK_STREAM
        )
        for *_, sockaddr in resolved:
            reject_private_network_host(str(sockaddr[0]))
    except PrivateNetworkHostError as exc:
        raise ValueError(f"ZENDESK_SUBDOMAIN is not allowed: {exc}") from exc
    except OSError as exc:
        raise ValueError(f"Zendesk host could not be resolved: {exc}") from exc
    return f"https://{hostname}/api/v2"


def _ensure_configured() -> None:
    """Raise the usual config error if the subdomain/credentials are
    missing or invalid, for a code path (the search-window-ceiling guards)
    that returns before ever reaching _request(), which normally does this
    validation as a side effect of building the request."""
    _base_url()
    _auth()


def _clamp_limit(limit: int) -> int:
    return clamp_limit(limit, max_limit=MAX_LIMIT)


def _require_non_blank(value: str | None, field_name: str) -> str:
    if not value or not value.strip():
        raise ValueError(f"{field_name} must not be blank")
    return value.strip()


# Zendesk silently truncates a comment body past 64KiB with no error
# reported -- a write that "succeeds" but stores less than what was sent.
# Rejecting locally, before the request, turns that into a clear local
# error instead of a silent content mismatch discovered later.
_MAX_COMMENT_BODY_BYTES = 65536


def _require_comment_body(value: str | None, field_name: str) -> str:
    value = _require_non_blank(value, field_name)
    byte_length = len(value.encode("utf-8"))
    if byte_length > _MAX_COMMENT_BODY_BYTES:
        raise ValueError(
            f"{field_name} is {byte_length} bytes, over Zendesk's "
            f"{_MAX_COMMENT_BODY_BYTES}-byte comment body limit -- Zendesk "
            "truncates rather than rejecting an oversized body, silently "
            "storing less than what was sent, so this is rejected locally "
            "instead"
        )
    return value


def _clean_tags(tags: list[str]) -> list[str]:
    """Strip whitespace and drop empty entries from a caller-supplied tag
    list before sending it to Zendesk -- FastMCP's schema only validates
    that this is a list of strings, not that each one is meaningful, and an
    LLM caller is exactly the kind of source likely to pass "vip " (trailing
    space, silently failing to match the canonical "vip" tag already used
    elsewhere in the account) or an empty string left over from a
    trailing-comma split done upstream of this tool."""
    return [t.strip() for t in tags if t.strip()]


def _resolve_tags(tags: list[str] | None) -> list[str] | None:
    """Resolve a caller-supplied tags param into the value to send in the
    request body, or None if the field should be omitted entirely (tags
    not provided).

    Distinguishes three states: None (omit), an explicit empty list []
    (this connector's documented "clear all tags" signal), and a
    non-empty list (cleaned via _clean_tags, but rejected if cleaning
    reduces it to nothing). A *non-empty* input that _clean_tags reduces
    to [] -- e.g. ["  ", ""], exactly the kind of LLM sloppiness
    _clean_tags exists to absorb -- must not collapse into the same wire
    request as an explicit clear: Zendesk's tag update is a full replace,
    so doing so would destructively wipe an existing tag set on a ticket
    the caller never asked to untag, with no undo available in this
    connector."""
    if tags is None:
        return None
    if not tags:
        return []
    cleaned = _clean_tags(tags)
    if not cleaned:
        raise ValueError(
            "tags contained no usable values (all entries were blank) -- "
            "pass an empty list [] to explicitly clear tags instead"
        )
    return cleaned


def _unwrap(result: Any, key: str) -> Any:
    """Pull a Zendesk response's single-object envelope (e.g. {"ticket":
    {...}}) out by its key, falling back to the raw payload if it isn't
    shaped that way."""
    return result.get(key, result) if isinstance(result, dict) else result


def _extract_error_detail(response: requests.Response) -> str | None:
    """Pull the human-readable message out of a Zendesk error body.

    Zendesk's error shape is inconsistent across endpoints: a plain string
    "error" with a separate "description" (e.g. RecordNotFound), or a
    nested {"error": {"title", "message"}} object (e.g. some auth
    failures). Returns None so the caller falls back to the raw response
    text in that case.
    """
    try:
        payload = response.json()
    except ValueError:
        return None
    if not isinstance(payload, dict):
        return None
    description = payload.get("description")
    if isinstance(description, str) and description:
        return description
    error = payload.get("error")
    if isinstance(error, str) and error:
        return error
    if isinstance(error, dict):
        message = error.get("message") or error.get("title")
        if isinstance(message, str) and message:
            return message
    return None


def _request(
    method: str,
    path: str,
    *,
    params: dict[str, Any] | None = None,
    json_data: dict[str, Any] | None = None,
) -> Any:
    url = f"{_base_url()}{path}"
    try:
        for attempt in (0, 1):
            response = _session.request(
                method=method,
                url=url,
                auth=_auth(),
                params=params,
                json=json_data,
                timeout=DEFAULT_TIMEOUT_SECONDS,
                # A redirect response is never followed with Basic Auth
                # credentials still attached: Zendesk's documented API
                # doesn't redirect, so a 3xx here is either a
                # misconfiguration or a host trying to relay the
                # credentials elsewhere.
                allow_redirects=False,
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
    except requests.RequestException as exc:
        # A connection/timeout/proxy failure's message can itself embed
        # sensitive data -- e.g. a ProxyError echoing the ambient
        # HTTPS_PROXY URL, which may carry embedded user:pass@ credentials
        # (setup_proxy_env() exports whatever the OS has configured).
        raise RuntimeError(redact_sensitive_text(str(exc))) from exc

    if 300 <= response.status_code < 400:
        raise RuntimeError(
            f"Zendesk returned an unexpected redirect (HTTP {response.status_code}); "
            "refusing to follow it with credentials attached"
        )
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        message = str(exc)
        detail = _extract_error_detail(response)
        if detail is None:
            detail = truncate_error_text(response.text.strip())
        if detail:
            message = f"{message} - {redact_sensitive_text(detail)}"
        raise RuntimeError(message) from exc

    if response.status_code == 204 or not response.content:
        return {}
    try:
        return response.json()
    except ValueError as exc:
        raise RuntimeError(
            f"Zendesk returned a 2xx response with a non-JSON body: {exc}"
        ) from exc


def _extract_list_field(
    payload: dict[str, Any], list_key: str, limit: int
) -> tuple[list[Any], list[Any]]:
    """Validate and slice the named list field out of a Zendesk list
    response -- the payload-shape check and slicing shared by both the
    cursor- and offset-paginated response shapes below; only how "more
    pages" and the resume cursor are derived from the result differs
    between them. Returns (page, all_items) since both callers need the
    unsliced length to detect a page[size] overflow."""
    if not isinstance(payload, dict):
        raise ValueError(
            f"Expected a JSON object from Zendesk, got {type(payload).__name__}"
        )
    items = payload.get(list_key) or []
    if not isinstance(items, list):
        raise ValueError(
            f'Expected Zendesk\'s "{list_key}" field to be a list, got '
            f"{type(items).__name__}"
        )
    return items[:limit], items


def _cursor_page(
    payload: dict[str, Any], list_key: str, limit: int
) -> tuple[list[Any], bool, str | None]:
    """Slice one page of a cursor-paginated Zendesk list response.

    Zendesk's cursor pagination (page[size]/page[after]) reports more-pages
    via meta.has_more and the next cursor via meta.after_cursor -- mirrors
    posthog.py's _paginated_results, adapted to this response shape.
    """
    page, items = _extract_list_field(payload, list_key, limit)
    meta = payload.get("meta") or {}
    has_more = bool(meta.get("has_more")) or len(items) > limit
    after_cursor = meta.get("after_cursor")
    # Deliberately not "and bool(page)": an empty page with has_more=true
    # (e.g. Zendesk reporting more results over a window that just emptied
    # out from concurrent deletions) is exactly the case with the least to
    # fall back on -- it must raise too, not slip through as a silent
    # has_more=true/after_cursor=null dead end.
    cursor_expected = has_more
    if cursor_expected and not after_cursor:
        # Zendesk's own cursor-pagination contract guarantees an
        # after_cursor whenever has_more is true. A response that violates
        # that (e.g. more rows than the page[size] this call requested, but
        # no cursor to resume from) is defensive-only territory this
        # connector cannot safely paper over -- failing loudly beats
        # silently handing back a has_more=true/after_cursor=null response
        # the caller can never advance past.
        raise RuntimeError(
            "Zendesk reported more results are available but did not "
            "return a resume cursor -- this violates its own documented "
            "cursor-pagination contract"
        )
    return page, has_more, after_cursor if cursor_expected else None


def _offset_page(
    payload: dict[str, Any], list_key: str, limit: int
) -> tuple[list[Any], bool]:
    """Slice one page of an offset-paginated Zendesk response (search.json/
    users/search.json only support offset pagination, not the cursor style
    every other list endpoint here uses)."""
    page, items = _extract_list_field(payload, list_key, limit)
    has_more = bool(payload.get("next_page")) or len(items) > limit
    return page, has_more


def _past_search_window(page: int, max_results: int, ceiling: int) -> bool:
    """True once the requested page's *end* offset crosses Zendesk's
    documented search-window ceiling for the endpoint being called (1,000
    for /search.json, 10,000 for /users/search.json -- see
    _MAX_SEARCH_RESULT_WINDOW / _MAX_USER_SEARCH_RESULT_WINDOW), past which
    Zendesk itself returns an opaque HTTP error rather than a clean empty
    page. Checked against the window's end (page * max_results), not its
    start, so a `limit` that doesn't evenly divide the ceiling still
    catches a page that straddles it instead of letting it through to
    error out remotely (e.g. limit=30, page=34 spans results 991-1020 of a
    1,000-result window)."""
    return page * max_results > ceiling


def _list_offset_paginated(
    path: str,
    list_key: str,
    summary_fn: Callable[[dict[str, Any]], dict[str, Any]],
    params: dict[str, Any],
    limit: int,
    extra_fields_fn: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> str:
    """Shared body for the offset-paginated search tools (zendesk_search,
    zendesk_search_users) -- symmetric to _list_cursor_paginated below, so
    both pagination families in this file build their own truncation-aware
    flat envelope and cap it themselves, rather than routing through the
    generic success_with_capped_dict and patching has_more onto an
    already-capped payload afterward. That patch-after-the-fact approach
    can itself push the payload back over the cap -- either by flipping
    "false" to the longer "true", or worse, by re-adding a "has_more" key
    that success_with_capped_dict's own phase-2 key-dropping had already
    discarded to fit -- defeating the exact invariant the capping
    subsystem exists to enforce.

    extra_fields_fn, if given, is called once with the raw Zendesk response
    to compute constant fields (e.g. zendesk_search's "count") that ride
    alongside the paginated list rather than being paginated themselves.
    """
    result = _request("GET", path, params=params)
    items, has_more = _offset_page(result, list_key, limit)
    summaries = [summary_fn(item) for item in items]
    extra_fields = extra_fields_fn(result) if extra_fields_fn else {}

    def _build(page: list[dict[str, Any]], truncated: bool) -> str:
        return _success(
            **{list_key: page},
            **extra_fields,
            # See _list_cursor_paginated's identical comment: a truncated
            # page must report has_more=True regardless of what Zendesk's
            # own response said, since the caller can't yet see the items
            # this call dropped for size.
            has_more=has_more or truncated,
            truncated=truncated,
        )

    response = _build(summaries, False)
    max_output_length = get_tool_max_output_length()
    if len(response) <= max_output_length:
        return response

    while len(response) > max_output_length and summaries:
        summaries = summaries[: len(summaries) // 2]
        response = _build(summaries, True)
    return response


def _list_cursor_paginated(
    path: str,
    list_key: str,
    summary_fn: Callable[[dict[str, Any]], dict[str, Any]],
    limit: int,
    after_cursor: str | None,
) -> str:
    """Shared body for every cursor-paginated "list X" tool below (tickets,
    ticket comments, users, organizations) -- only the path, response key,
    and per-item summarizer differ between them."""
    max_results = _clamp_limit(limit)
    params: dict[str, Any] = {
        "page[size]": max_results,
        # Some cursor-paginated endpoints (users, organizations) omit
        # meta.has_more entirely unless this is explicitly requested,
        # which _cursor_page then reads as "no more pages" and silently
        # drops the next cursor along with every item past this one.
        # Passing it on every endpoint (not just the ones known to need
        # it) is a harmless no-op where it's already the default.
        "include_boundary_indicators": "true",
    }
    if after_cursor:
        params["page[after]"] = after_cursor
    result = _request("GET", path, params=params)
    items, has_more, next_cursor = _cursor_page(result, list_key, max_results)
    summaries = [summary_fn(item) for item in items]

    def _build(page: list[dict[str, Any]], cursor: str | None, truncated: bool) -> str:
        return _success(
            **{list_key: page},
            # A truncated page must report has_more=True even if Zendesk's
            # own meta said this was the last page: the items dropped by
            # truncation are still unread, so a caller that trusts has_more
            # alone (a natural reading of that field) must not stop here.
            has_more=has_more or truncated,
            after_cursor=cursor,
            truncated=truncated,
        )

    response = _build(summaries, next_cursor, False)
    max_output_length = get_tool_max_output_length()
    if len(response) <= max_output_length:
        return response

    # A shrunk page must never keep Zendesk's real next-page cursor: that
    # cursor marks the end of the *full* page Zendesk returned, so pairing
    # a locally truncated page with it would make the next call resume
    # past the untrimmed items -- silently dropping them for good, not
    # just deferring them. On truncation, the cursor handed back is
    # instead this call's own *input* cursor (or None for the first page)
    # -- Zendesk still considers the full untruncated page consumed, so
    # the only way to recover the trimmed items is retrying that same
    # starting point with a smaller `limit`, mirroring hubspot.py's
    # _paged_list.
    fallback_cursor = after_cursor or None
    while len(response) > max_output_length and summaries:
        summaries = summaries[: len(summaries) // 2]
        response = _build(summaries, fallback_cursor, True)
    return response


def _ticket_summary(ticket: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": ticket.get("id"),
        "subject": ticket.get("subject"),
        "status": ticket.get("status"),
        "priority": ticket.get("priority"),
        "requester_id": ticket.get("requester_id"),
        "assignee_id": ticket.get("assignee_id"),
        "group_id": ticket.get("group_id"),
        "tags": ticket.get("tags"),
        "created_at": ticket.get("created_at"),
        "updated_at": ticket.get("updated_at"),
    }


def _user_summary(user: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": user.get("id"),
        "name": user.get("name"),
        "email": user.get("email"),
        "role": user.get("role"),
        "organization_id": user.get("organization_id"),
        "created_at": user.get("created_at"),
    }


def _organization_summary(org: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": org.get("id"),
        "name": org.get("name"),
        "domain_names": org.get("domain_names"),
        "created_at": org.get("created_at"),
    }


def _comment_summary(comment: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": comment.get("id"),
        "author_id": comment.get("author_id"),
        "body": comment.get("plain_body") or comment.get("body"),
        "public": comment.get("public"),
        "created_at": comment.get("created_at"),
    }


def _search_result_summary(result: dict[str, Any]) -> dict[str, Any]:
    result_type = result.get("result_type")
    if result_type == "ticket":
        return {"result_type": result_type, **_ticket_summary(result)}
    if result_type == "user":
        return {"result_type": result_type, **_user_summary(result)}
    if result_type == "organization":
        return {"result_type": result_type, **_organization_summary(result)}
    return {
        "result_type": result_type,
        "id": result.get("id"),
        "name": result.get("name") or result.get("subject"),
    }


def _resolve_path_id(value: str, pattern: re.Pattern[str], field_name: str) -> str:
    """Accept a bare id or a full Zendesk agent UI URL (e.g.
    "https://acme.zendesk.com/agent/tickets/123"), then percent-encode the
    result for safe interpolation into a URL path segment."""
    return url_path_id(resolve_id_from_url(str(value), pattern, field_name), field_name)


def _find_comment_event(result: Any) -> dict[str, Any] | None:
    """Pull the newly created Comment event out of a ticket-update
    response's audit trail (PUT /tickets/{id}.json returns {"ticket":
    ..., "audit": {"events": [...]}} -- the audit, not the ticket object
    itself, is where a just-added comment's own id/body/public actually
    show up), so reply/note tools can confirm exactly what was posted --
    including `public`, the one field that distinguishes a customer-visible
    reply from an internal note."""
    if not isinstance(result, dict):
        return None
    events = (result.get("audit") or {}).get("events") or []
    for event in events:
        if isinstance(event, dict) and event.get("type") == "Comment":
            return _comment_summary(event)
    return None


def _add_comment(ticket_id: str, body: str, public: bool) -> str:
    """Post a comment and return the already-capped success envelope (not
    a dict for the caller to wrap) -- a comment body can be up to 64KiB
    (Zendesk's own limit, enforced by _require_comment_body above), well
    past a configured output cap, and this is the only mutation response
    in this file that echoes caller-supplied free text back verbatim, so
    it needs its own truncation handling rather than relying on the list
    tools' page-shrinking (there's no list here to shrink)."""
    body = _require_comment_body(body, "body")
    result = _request(
        "PUT",
        f"/tickets/{_resolve_path_id(ticket_id, _TICKET_URL_ID_PATTERN, 'ticket_id')}.json",
        json_data={"ticket": {"comment": {"body": body, "public": public}}},
    )
    ticket = _unwrap(result, "ticket")
    ticket_summary = _ticket_summary(ticket) if isinstance(ticket, dict) else ticket
    comment = _find_comment_event(result)

    def _build(comment_value: dict[str, Any] | None) -> str:
        return _success(ticket=ticket_summary, comment=comment_value)

    response = _build(comment)
    max_output_length = get_tool_max_output_length()
    if len(response) <= max_output_length or not isinstance(comment, dict):
        return response

    comment_body = comment.get("body")
    if not isinstance(comment_body, str) or not comment_body:
        return response

    # Truncate the echoed body rather than dropping the comment entirely:
    # id/public/created_at still confirm what was posted and -- crucially
    # -- whether it went out publicly, even once the body preview is cut.
    truncated_comment = dict(comment)
    while len(response) > max_output_length and comment_body:
        comment_body = comment_body[: len(comment_body) // 2]
        truncated_comment["body"] = comment_body
        truncated_comment["body_truncated"] = True
        response = _build(truncated_comment)
    return response


@mcp.tool()
def zendesk_search(query: str, limit: int = 25, page: int = 1) -> str:
    """
    Unified search across tickets, users, and organizations using Zendesk's
    search syntax, e.g. "type:ticket status:open priority:urgent" or
    "type:user email:jane@example.com".
    limit: max results to return (default 25, hard cap 100).
    page: 1-based page number; pass the previous page + 1 to continue. If a
    call is truncated for size (`truncated: true` in the response,
    `has_more` forced true), the trimmed items are not recoverable by
    retrying the same `page` with a smaller `limit` -- Zendesk's `page`
    means a different item range once `limit` changes. Instead, re-run the
    search with a smaller `limit` starting from `page=1` and page through
    normally; that re-partitions the same result set into pieces small
    enough to avoid truncation.
    """
    try:
        query = _require_non_blank(query, "query")
        max_results = _clamp_limit(limit)
        page = max(1, page)
        if _past_search_window(page, max_results, _MAX_SEARCH_RESULT_WINDOW):
            # Past Zendesk's own documented result-window ceiling: Zendesk
            # itself would answer with an opaque HTTP error here, so return
            # a clean, predictable "no more results" instead of forwarding
            # a caller mechanically incrementing `page` into that error.
            # Validate config here too (normally _request's job) -- this
            # branch never reaches _request, so a misconfigured
            # subdomain/credential would otherwise be masked as "no
            # results" instead of surfacing as the usual config error.
            _ensure_configured()
            return _success(results=[], count=None, has_more=False, truncated=False)
        return _list_offset_paginated(
            "/search.json",
            "results",
            _search_result_summary,
            {"query": query, "per_page": max_results, "page": page},
            max_results,
            extra_fields_fn=lambda result: {"count": result.get("count")},
        )
    except Exception as e:
        # The query can carry end-user PII (the tool's own docstring
        # example is an email address) -- logged length-bounded rather than
        # verbatim, matching this file's redaction posture for response
        # bodies elsewhere.
        logger.error(f"Error searching Zendesk for query of length {len(query)}: {e}")
        return _error(str(e))


@mcp.tool()
def zendesk_list_tickets(limit: int = 25, after_cursor: str | None = None) -> str:
    """
    List tickets in Zendesk's default order. Does not include archived
    tickets -- Zendesk excludes them from this endpoint regardless of age;
    there is no tool in this connector for browsing archived tickets. For
    a filtered view (by status, priority, assignee, etc.) use
    zendesk_search instead, e.g. query="type:ticket status:open".
    limit: max tickets to return (default 25, hard cap 100).
    after_cursor: pass the previous call's own after_cursor to fetch the
    next page; omit for the first page.
    """
    try:
        return _list_cursor_paginated(
            "/tickets.json", "tickets", _ticket_summary, limit, after_cursor
        )
    except Exception as e:
        logger.error(f"Error listing Zendesk tickets: {e}")
        return _error(str(e))


@mcp.tool()
def zendesk_get_ticket(ticket_id: str) -> str:
    """
    Get a Zendesk ticket by id -- a bare numeric id, or a full ticket URL
    copied from the Zendesk agent UI.
    """
    try:
        path_id = _resolve_path_id(ticket_id, _TICKET_URL_ID_PATTERN, "ticket_id")
        result = _request("GET", f"/tickets/{path_id}.json")
        return _success(ticket=_ticket_summary(_unwrap(result, "ticket")))
    except Exception as e:
        logger.error(f"Error fetching Zendesk ticket {ticket_id}: {e}")
        return _error(str(e))


@mcp.tool()
def zendesk_create_ticket(
    subject: str,
    comment: str,
    requester_email: str | None = None,
    requester_name: str | None = None,
    priority: str | None = None,
    tags: list[str] | None = None,
) -> str:
    """
    Create a new Zendesk ticket.
    subject: the ticket's subject line.
    comment: the ticket's initial (public) comment/description. Max 64KiB
    (Zendesk's own limit) -- rejected locally rather than silently
    truncated.
    requester_email: optional email of the end user this ticket is on
    behalf of; defaults to the connected agent if omitted.
    requester_name: the requester's display name. Ignored by Zendesk if
    requester_email already matches an existing user, but REQUIRED if it
    doesn't -- Zendesk rejects an unrecognized requester_email with no
    requester_name. Must not be passed without requester_email.
    priority: optional, one of "low", "normal", "high", "urgent".
    tags: optional list of tags -- a non-empty list with only blank/
    whitespace entries is rejected rather than silently sent as no tags.
    """
    try:
        subject = _require_non_blank(subject, "subject")
        comment = _require_comment_body(comment, "comment")
        ticket: dict[str, Any] = {"subject": subject, "comment": {"body": comment}}
        # Blank-after-strip is treated the same as not provided (matching
        # status/priority elsewhere in this function) rather than sent to
        # Zendesk verbatim -- a whitespace-only requester_name would
        # otherwise silently defeat the "name is required for a new
        # requester" rule this parameter exists to satisfy.
        requester_email = (requester_email or "").strip() or None
        requester_name = (requester_name or "").strip() or None
        if requester_email:
            requester: dict[str, Any] = {"email": requester_email}
            if requester_name:
                requester["name"] = requester_name
            ticket["requester"] = requester
        elif requester_name:
            raise ValueError("requester_name requires requester_email")
        if priority:
            ticket["priority"] = priority
        tags_value = _resolve_tags(tags)
        if tags_value is not None:
            ticket["tags"] = tags_value
        result = _request("POST", "/tickets.json", json_data={"ticket": ticket})
        return _success(ticket=_ticket_summary(_unwrap(result, "ticket")))
    except Exception as e:
        logger.error(f"Error creating Zendesk ticket: {e}")
        return _error(str(e))


@mcp.tool()
def zendesk_update_ticket(
    ticket_id: str,
    status: str | None = None,
    priority: str | None = None,
    tags: list[str] | None = None,
    assignee_id: int | None = None,
    group_id: int | None = None,
) -> str:
    """
    Update a ticket's status, priority, tags, assignee, and/or group. Only
    the fields explicitly provided (not None) are changed. Use
    zendesk_reply_to_ticket or zendesk_add_internal_note to add a comment
    instead.
    ticket_id: a bare numeric id, or a full ticket URL copied from the
    Zendesk agent UI.
    status: optional, one of "new", "open", "pending", "hold", "solved",
    "closed" -- an empty string is treated the same as leaving it unset
    (there is no valid "clear the status" value).
    priority: optional, one of "low", "normal", "high", "urgent" -- an empty
    string is treated the same as leaving it unset, for the same reason.
    tags: optional list of tags -- replaces the ticket's existing tags
    entirely (pass an empty list to clear them), it does not add to them.
    A non-empty list with only blank/whitespace entries is rejected rather
    than silently treated the same as an explicit clear.
    assignee_id: optional user id (from zendesk_get_user/zendesk_list_users)
    to reassign the ticket to.
    group_id: optional group id to route the ticket to.
    """
    try:
        fields: dict[str, Any] = {}
        if status:
            fields["status"] = status
        if priority:
            fields["priority"] = priority
        tags_value = _resolve_tags(tags)
        if tags_value is not None:
            fields["tags"] = tags_value
        if assignee_id is not None:
            fields["assignee_id"] = assignee_id
        if group_id is not None:
            fields["group_id"] = group_id
        if not fields:
            raise ValueError(
                "at least one of status/priority/tags/assignee_id/group_id "
                "must be provided"
            )
        result = _request(
            "PUT",
            f"/tickets/{_resolve_path_id(ticket_id, _TICKET_URL_ID_PATTERN, 'ticket_id')}.json",
            json_data={"ticket": fields},
        )
        return _success(ticket=_ticket_summary(_unwrap(result, "ticket")))
    except Exception as e:
        logger.error(f"Error updating Zendesk ticket {ticket_id}: {e}")
        return _error(str(e))


@mcp.tool()
def zendesk_delete_ticket(ticket_id: str) -> str:
    """
    Delete a Zendesk ticket. This is a soft delete: Zendesk moves the
    ticket to its "Deleted tickets" area rather than destroying it
    immediately, and it can still be restored there (outside this
    connector) for a limited time.
    ticket_id: a bare numeric id, or a full ticket URL copied from the
    Zendesk agent UI.
    """
    try:
        path_id = _resolve_path_id(ticket_id, _TICKET_URL_ID_PATTERN, "ticket_id")
        _request("DELETE", f"/tickets/{path_id}.json")
        return _success(ticket_id=path_id)
    except Exception as e:
        logger.error(f"Error deleting Zendesk ticket {ticket_id}: {e}")
        return _error(str(e))


@mcp.tool()
def zendesk_list_ticket_comments(
    ticket_id: str, limit: int = 25, after_cursor: str | None = None
) -> str:
    """
    List the comment thread on a ticket, oldest first (the first comment is
    the ticket's original description).
    ticket_id: a bare numeric id, or a full ticket URL copied from the
    Zendesk agent UI.
    limit: max comments to return (default 25, hard cap 100).
    after_cursor: pass the previous call's own after_cursor to fetch the
    next page; omit for the first page.
    """
    try:
        path_id = _resolve_path_id(ticket_id, _TICKET_URL_ID_PATTERN, "ticket_id")
        path = f"/tickets/{path_id}/comments.json"
        return _list_cursor_paginated(
            path, "comments", _comment_summary, limit, after_cursor
        )
    except Exception as e:
        logger.error(f"Error listing comments for Zendesk ticket {ticket_id}: {e}")
        return _error(str(e))


@mcp.tool()
def zendesk_reply_to_ticket(ticket_id: str, body: str) -> str:
    """
    Reply to a Zendesk ticket as a public comment -- visible to the
    requester/end user. Returns the created comment (id, body, and
    public=true) alongside the updated ticket, so the caller can confirm
    exactly what was posted.
    ticket_id: a bare numeric id, or a full ticket URL copied from the
    Zendesk agent UI.
    body: max 64KiB (Zendesk's own limit) -- rejected locally rather than
    silently truncated.
    """
    try:
        return _add_comment(ticket_id, body, public=True)
    except Exception as e:
        logger.error(f"Error replying to Zendesk ticket {ticket_id}: {e}")
        return _error(str(e))


@mcp.tool()
def zendesk_add_internal_note(ticket_id: str, body: str) -> str:
    """
    Add an internal note to a Zendesk ticket. Internal notes are only
    visible to agents, never to the requester/end user. Returns the created
    comment (id, body, and public=false) alongside the updated ticket, so
    the caller can confirm the note did not go out publicly.
    ticket_id: a bare numeric id, or a full ticket URL copied from the
    Zendesk agent UI.
    body: max 64KiB (Zendesk's own limit) -- rejected locally rather than
    silently truncated.
    """
    try:
        return _add_comment(ticket_id, body, public=False)
    except Exception as e:
        logger.error(f"Error adding internal note to Zendesk ticket {ticket_id}: {e}")
        return _error(str(e))


@mcp.tool()
def zendesk_list_users(limit: int = 25, after_cursor: str | None = None) -> str:
    """
    List all users (agents and end users) in this Zendesk account.
    limit: max users to return (default 25, hard cap 100).
    after_cursor: pass the previous call's own after_cursor to fetch the
    next page; omit for the first page.
    """
    try:
        return _list_cursor_paginated(
            "/users.json", "users", _user_summary, limit, after_cursor
        )
    except Exception as e:
        logger.error(f"Error listing Zendesk users: {e}")
        return _error(str(e))


@mcp.tool()
def zendesk_get_user(user_id: str) -> str:
    """
    Get a Zendesk user by id.
    user_id: a bare numeric id, or a full user URL copied from the Zendesk
    agent UI.
    """
    try:
        path_id = _resolve_path_id(user_id, _USER_URL_ID_PATTERN, "user_id")
        result = _request("GET", f"/users/{path_id}.json")
        return _success(user=_user_summary(_unwrap(result, "user")))
    except Exception as e:
        logger.error(f"Error fetching Zendesk user {user_id}: {e}")
        return _error(str(e))


@mcp.tool()
def zendesk_search_users(query: str, limit: int = 25, page: int = 1) -> str:
    """
    Search users by name, email, or external_id, e.g. "jane@example.com".
    limit: max results to return (default 25, hard cap 100).
    page: 1-based page number; pass the previous page + 1 to continue. If a
    call is truncated for size (`truncated: true` in the response,
    `has_more` forced true), the trimmed users are not recoverable by
    retrying the same `page` with a smaller `limit` -- see zendesk_search's
    identical note for why; re-run with a smaller `limit` from `page=1`
    instead.
    """
    try:
        query = _require_non_blank(query, "query")
        max_results = _clamp_limit(limit)
        page = max(1, page)
        if _past_search_window(page, max_results, _MAX_USER_SEARCH_RESULT_WINDOW):
            # Past Zendesk's own documented result-window ceiling for THIS
            # endpoint -- 10,000, not the unified /search.json's 1,000 (see
            # _MAX_USER_SEARCH_RESULT_WINDOW). See zendesk_search's
            # identical guard for the config-validation/response-shape
            # rationale.
            _ensure_configured()
            return _success(users=[], has_more=False, truncated=False)
        return _list_offset_paginated(
            "/users/search.json",
            "users",
            _user_summary,
            {"query": query, "per_page": max_results, "page": page},
            max_results,
        )
    except Exception as e:
        # The query can carry end-user PII (name/email/external_id) --
        # logged length-bounded rather than verbatim, matching
        # zendesk_search's redaction posture.
        logger.error(
            f"Error searching Zendesk users for query of length {len(query)}: {e}"
        )
        return _error(str(e))


@mcp.tool()
def zendesk_list_organizations(limit: int = 25, after_cursor: str | None = None) -> str:
    """
    List all organizations in this Zendesk account.
    limit: max organizations to return (default 25, hard cap 100).
    after_cursor: pass the previous call's own after_cursor to fetch the
    next page; omit for the first page.
    """
    try:
        return _list_cursor_paginated(
            "/organizations.json",
            "organizations",
            _organization_summary,
            limit,
            after_cursor,
        )
    except Exception as e:
        logger.error(f"Error listing Zendesk organizations: {e}")
        return _error(str(e))


@mcp.tool()
def zendesk_get_organization(organization_id: str) -> str:
    """
    Get a Zendesk organization by id.
    organization_id: a bare numeric id, or a full organization URL copied
    from the Zendesk agent UI.
    """
    try:
        path_id = _resolve_path_id(
            organization_id, _ORGANIZATION_URL_ID_PATTERN, "organization_id"
        )
        result = _request("GET", f"/organizations/{path_id}.json")
        return _success(
            organization=_organization_summary(_unwrap(result, "organization"))
        )
    except Exception as e:
        logger.error(f"Error fetching Zendesk organization {organization_id}: {e}")
        return _error(str(e))


if __name__ == "__main__":
    mcp.run()
