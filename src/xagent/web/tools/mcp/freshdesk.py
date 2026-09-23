"""Freshdesk MCP connector, backed by the Freshdesk REST API.

Scope note: the tools here are written against the published REST contract
(developers.freshdesk.com/api) and covered by mocked tests. They have not yet
been exercised against a live tenant, so the response *shapes* are documented
rather than observed -- the end-to-end check against a real Freshdesk account
is still outstanding on xorbitsai/xagent-saas#1409, and anything this module
assumes about an envelope is a documentation claim until then.

Why REST rather than Freshdesk's own remote MCP endpoint: that endpoint is
metered separately and generously little (100 actions per account per month on
Growth, against 100 REST calls per minute), and bridging to it would put the
per-tenant URL and a static credential on a shared catalog row, which the
catalog has no per-user shape for. Going through REST keeps this connector on
the existing stdio/api_key path, where the subdomain and key are ordinary
per-user encrypted env.
"""

import json
import logging
import os
import re
import socket
from typing import Any

import requests
from mcp.server.fastmcp import FastMCP

from ....config import get_tool_max_output_length
from ....core.tools.core.web_content import get_trusted_proxy_url
from ....core.utils.security import (
    PrivateNetworkHostError,
    redact_sensitive_text,
    reject_private_network_host,
)
from ...utils.graphql_errors import truncate_error_text
from .utils import clamp_limit, setup_proxy_env, success_with_capped_dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("freshdesk-mcp")

# Ensure standard proxy environment variables are set to prevent hanging requests
setup_proxy_env()

mcp = FastMCP("freshdesk-mcp")

# One module-level session, configured exactly as zendesk.py configures its
# own -- that connector is this one's closest analogue (helpdesk, subdomain
# label, fixed vendor domain, Basic auth), so it is the right thing to copy.
_session = requests.Session()
# trust_env=False so requests never falls back to an ambient/OS-native proxy
# that get_trusted_proxy_url() does not police. A proxy does its own DNS
# resolution for the real connection, so an ambient one silently bypasses the
# private-network check _base_url() performs on the addresses *this* process
# resolved -- the exact hole that check exists to close. It also disables
# .netrc auto-auth, which a redirect would otherwise use to attach someone
# else's credentials to the target host.
_session.trust_env = False
# trust_env=False also turns off requests' own REQUESTS_CA_BUNDLE/
# CURL_CA_BUNDLE lookup, so it is re-applied explicitly: an operator opting
# into a trusted egress proxy is the textbook TLS-intercepting corporate proxy
# with an internal CA, which would otherwise fail closed with an opaque
# SSLError.
_ca_bundle = os.environ.get("REQUESTS_CA_BUNDLE") or os.environ.get("CURL_CA_BUNDLE")
if _ca_bundle:
    _session.verify = _ca_bundle

# The only host this connector ever talks to. Freshdesk serves each tenant at
# <subdomain>.freshdesk.com and does not support custom domains for
# programmatic access, so the host is composed from a validated label rather
# than accepted as a URL.
#
# That is NOT the whole SSRF story, and an earlier revision of this file
# wrongly said it was. Composing the host only settles who chose the *string*;
# a perfectly legitimate hostname can still be rebound by DNS to a private or
# internal address at request time, which is orthogonal. zendesk.py -- the
# closest analogue in this repo, same label-plus-fixed-domain shape -- makes
# exactly that point and resolves and checks every address anyway, as
# posthog.py does for its two hardcoded hostnames. _base_url() below follows
# them.
FRESHDESK_DOMAIN = "freshdesk.com"

# A DNS label: 1-63 chars, alphanumeric, internal hyphens only. Deliberately
# stricter than DNS itself (no underscores, ASCII only) -- a Freshdesk
# subdomain is chosen from their signup form, which is narrower still, and
# anything this rejects is a user error rather than a tenant we are locking
# out. Anchored with \A/\Z rather than ^/$ because $ also matches before a
# trailing newline, which would let "acme\n" through into the Host header.
SUBDOMAIN_PATTERN = re.compile(r"\A[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\Z")

DEFAULT_TIMEOUT_SECONDS = 30
# Freshdesk's own list page-size cap (developers.freshdesk.com/api: per_page
# defaults to 30 and maxes out at 100).
MAX_PER_PAGE = 100
DEFAULT_PER_PAGE = 30

# The filter/search endpoint is a different, narrower contract from the list
# endpoints: Freshdesk fixes it at 30 results per page and refuses pages past
# 10, so one query can reach at most 300 tickets however it is paged.
SEARCH_PAGE_SIZE = 30
MAX_SEARCH_PAGE = 10
# "Query string must be enclosed between a pair of double quotes and can have
# up to 512 characters" -- the quotes are inside the budget.
MAX_QUERY_LENGTH = 512

# Freshdesk's built-in ticket statuses. NOT an exhaustive set: a helpdesk can
# define custom statuses, which are assigned instance-specific numeric values
# above these, so this connector cannot know which values are legal for a given
# tenant. It is therefore used to *name* the defaults in an error message, never
# to reject an unrecognized value -- rejecting would make this connector
# narrower than the API it fronts and break every helpdesk with a custom status.
STATUS_OPEN = 2
MIN_TICKET_STATUS = 2
TICKET_STATUSES = {2: "Open", 3: "Pending", 4: "Resolved", 5: "Closed"}

# Priority, unlike status, is a fixed four-value field with no customization,
# so it is validated as a closed set.
PRIORITY_LOW = 1
TICKET_PRIORITIES = {1: "Low", 2: "Medium", 3: "High", 4: "Urgent"}


# A Freshdesk URL carries the caller's search terms -- a contact email, a
# ticket query -- in its query string, and a urllib3 RequestException embeds
# the full URL in its message. redact_sensitive_text only knows
# credential-shaped keys (api_key/token/...), not `query=`, so the query string
# is scrubbed wholesale before any exception text is logged or returned to the
# model. Copied from zendesk.py, which closed the same hole.
_QUERY_STRING_PATTERN = re.compile(r"\?[^\s\"'()<>]+")
# Only redact when the "?" is immediately preceded, within the same
# whitespace-delimited token, by something path-shaped (containing a "/"), so
# ordinary prose ending in a question mark is left alone.
_PATH_LIKE_TOKEN_PATTERN = re.compile(r"[^\s\"'()<>]*$")


def _scrub_query_strings(text: str) -> str:
    def _redact(match: re.Match[str]) -> str:
        preceding_token = _PATH_LIKE_TOKEN_PATTERN.search(text[: match.start()])
        token = preceding_token.group(0) if preceding_token else ""
        return "?<query redacted>" if "/" in token else match.group(0)

    return _QUERY_STRING_PATTERN.sub(_redact, text)


def _sanitize_exception_text(exc: BaseException) -> str:
    return _scrub_query_strings(redact_sensitive_text(str(exc)))


def _success(**payload: Any) -> str:
    return json.dumps({"status": "success", **payload}, ensure_ascii=False)


def _error(message: str) -> str:
    return json.dumps({"status": "error", "message": message}, ensure_ascii=False)


def _subdomain() -> str:
    """Return the validated tenant subdomain label.

    Normalizes case and surrounding whitespace first: DNS labels are
    case-insensitive and a copy-pasted value routinely carries both, so
    normalizing keeps a recoverable input from becoming an error -- while
    everything that could change *which host* is contacted (a dot, slash,
    colon, credential marker, or query/fragment separator) is rejected,
    since the label is interpolated into the host position below.
    """
    raw = (os.environ.get("FRESHDESK_SUBDOMAIN") or "").strip().lower()
    if not raw:
        raise ValueError("FRESHDESK_SUBDOMAIN environment variable is missing or empty")
    if not SUBDOMAIN_PATTERN.match(raw):
        raise ValueError(
            "FRESHDESK_SUBDOMAIN must be just the tenant label -- the 'acme' in "
            f"acme.{FRESHDESK_DOMAIN}, not a full hostname or URL"
        )
    return raw


def _base_url() -> str:
    """Compose the tenant's API root, refusing one that resolves privately.

    The label check in _subdomain() settles which *name* is contacted; this
    settles which *address*. Both are needed -- see FRESHDESK_DOMAIN's comment.
    """
    hostname = f"{_subdomain()}.{FRESHDESK_DOMAIN}"
    try:
        resolved = socket.getaddrinfo(
            hostname, 443, socket.AF_UNSPEC, socket.SOCK_STREAM
        )
        for *_, sockaddr in resolved:
            reject_private_network_host(str(sockaddr[0]))
    except PrivateNetworkHostError as exc:
        raise ValueError(f"FRESHDESK_SUBDOMAIN is not allowed: {exc}") from exc
    except OSError as exc:
        raise ValueError(f"Freshdesk host could not be resolved: {exc}") from exc
    return f"https://{hostname}/api/v2"


def _api_key() -> str:
    # Stripped, not a bare os.environ.get(): a stray leading/trailing newline
    # or space in the injected key would otherwise produce a malformed Basic
    # Auth header rather than the clear "missing" error below.
    api_key = (os.environ.get("FRESHDESK_API_KEY") or "").strip()
    if not api_key:
        raise ValueError("FRESHDESK_API_KEY environment variable is missing or empty")
    return api_key


def _extract_error_detail(response: requests.Response) -> str | None:
    """Pull a human-readable message out of a Freshdesk error body.

    Freshdesk's 400 bodies put a generic "Validation failed" in ``description``
    and the actionable part in ``errors`` (a list of {field, message, code}),
    so both are joined rather than taking the first key that matches -- a
    caller told only "Validation failed" has nothing to fix. Returns None when
    the body is not JSON or not an object, so the caller falls back to the raw
    response text.
    """
    try:
        payload = response.json()
    except ValueError:
        return None
    if not isinstance(payload, dict):
        return None

    parts: list[str] = []
    for key in ("description", "message"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            parts.append(value)
            break

    errors = payload.get("errors")
    if isinstance(errors, (list, dict)) and errors:
        parts.append(json.dumps(errors, ensure_ascii=False))
    elif isinstance(errors, str) and errors:
        parts.append(errors)

    return ": ".join(parts) if parts else None


def _send(
    method: str,
    path: str,
    *,
    params: dict[str, Any] | None = None,
    json_data: Any = None,
) -> requests.Response:
    """Issue one request and translate transport/HTTP failures.

    Split out from ``_request`` so the paginated helpers can read the
    ``Link`` response header, which the parsed body does not carry.
    """
    try:
        proxy_url = get_trusted_proxy_url()
    except PrivateNetworkHostError as exc:
        raise type(exc)(redact_sensitive_text(str(exc))) from exc
    proxies = {"http": proxy_url, "https": proxy_url} if proxy_url else {}

    try:
        response = _session.request(
            method=method,
            url=f"{_base_url()}{path}",
            proxies=proxies,
            # Never follow a redirect with the Basic Auth credential attached.
            # Freshdesk's documented API does not redirect, so a 3xx is either
            # a misconfiguration or a host trying to relay the request -- and a
            # 307/308 would replay the body (ticket text, contact search terms)
            # to wherever Location points.
            allow_redirects=False,
            # Freshdesk authenticates with HTTP Basic using the API key as the
            # username and an ignored password ("X" by the vendor's own
            # convention) -- not a bearer token. Note this differs from their
            # remote MCP endpoint, which takes a bare `Authorization: <key>`.
            auth=(_api_key(), "X"),
            headers={"Content-Type": "application/json"},
            # Drop "" as well as None: every list filter is optional, and an
            # LLM tool-call passing e.g. status="" to mean "no filter" would
            # otherwise become a real `?status=` query param.
            params={
                k: v for k, v in (params or {}).items() if v is not None and v != ""
            },
            json=json_data,
            timeout=DEFAULT_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        # A connection/timeout/proxy failure's message can itself embed
        # sensitive data -- e.g. a ProxyError echoing the ambient HTTPS_PROXY
        # URL, which may carry embedded user:pass@ credentials
        # (setup_proxy_env() exports whatever the OS has configured).
        raise RuntimeError(
            f"Freshdesk request failed: {truncate_error_text(_sanitize_exception_text(exc))}"
        ) from exc

    if 300 <= response.status_code < 400:
        raise RuntimeError(
            f"Freshdesk returned an unexpected redirect (HTTP {response.status_code}); "
            "refusing to follow it with credentials attached"
        )

    if response.status_code >= 400:
        detail = _extract_error_detail(response)
        if detail is None:
            detail = response.text.strip()
        # The response body is host-controlled content -- if it echoes request
        # headers (a misconfigured proxy/WAF error page), redact the Basic Auth
        # credential before it reaches logs or the LLM's context.
        detail = truncate_error_text(
            _scrub_query_strings(redact_sensitive_text(detail))
        )
        # 429 is the one error a caller can act on, and Freshdesk puts the wait
        # in Retry-After. Surfacing it turns "rate limited" into a decision the
        # caller can actually make.
        retry_after = response.headers.get("Retry-After")
        suffix = (
            f" (retry after {retry_after}s)"
            if response.status_code == 429 and retry_after
            else ""
        )
        # A detail-less 404 means the edge answered, not the tenant:
        # freshdesk.com is wildcard-resolved, so an unknown subdomain returns a
        # body-less 404 from every path (verified against a nonexistent
        # subdomain). Only when there is no detail -- a real Freshdesk 404
        # carries a description, and replacing it would bury the real cause.
        if not detail and response.status_code == 404:
            detail = (
                "no response body -- if this happens for every request, "
                f"check that FRESHDESK_SUBDOMAIN ({_subdomain()!r}) names "
                "an existing Freshdesk account"
            )
        # The credential hint is APPENDED rather than substituted: a real
        # Freshdesk 401/403 carries a message, so gating this on an empty
        # detail (as an earlier revision did) meant it never fired in
        # practice, which is the only case it was written for.
        if response.status_code in (401, 403):
            credential_hint = (
                "check that FRESHDESK_API_KEY is current and that the key's "
                "agent has permission for this operation"
            )
            detail = f"{detail} -- {credential_hint}" if detail else credential_hint
        raise RuntimeError(
            f"Freshdesk API error (status {response.status_code}){suffix}"
            + (f": {detail}" if detail else "")
        )

    return response


def _body(response: requests.Response) -> Any:
    if response.status_code == 204 or not response.content:
        return {}
    try:
        return response.json()
    except ValueError as exc:
        raise RuntimeError(
            f"Freshdesk returned a 2xx response with a non-JSON body: {exc}"
        ) from exc


def _request(
    method: str,
    path: str,
    *,
    params: dict[str, Any] | None = None,
    json_data: Any = None,
) -> Any:
    return _body(_send(method, path, params=params, json_data=json_data))


def _has_next_page(response: requests.Response) -> bool:
    """Whether Freshdesk says another page exists.

    Freshdesk signals this with an RFC 5988 ``Link: <...>; rel="next"`` header
    and returns no total count on list endpoints. Reading the header is the
    only exact answer: inferring from ``len(items) == per_page`` reports a
    phantom next page whenever the last page happens to be exactly full, and
    an LLM following that signal issues one pointless call per list.
    """
    return 'rel="next"' in (response.headers.get("Link") or "")


def _paged_list(
    path: str,
    field: str,
    *,
    page: int,
    per_page: int,
    filters: dict[str, Any] | None = None,
) -> str:
    """Fetch one page of a bare-array list endpoint and build its envelope.

    The four list tools differ only in path, response field and filters, so
    the pagination clamping, array validation, Link-header reading and output
    capping live here once rather than being repeated with four chances to
    drift apart.
    """
    response = _send(
        "GET",
        path,
        params={
            **(filters or {}),
            "page": max(1, int(page)),
            "per_page": _clamp_per_page(per_page),
        },
    )
    items = _validated_list(_body(response), path)
    return _success_with_capped_list(
        field, {field: items, "has_more": _has_next_page(response)}
    )


def _clamp_per_page(per_page: int) -> int:
    return clamp_limit(per_page, max_limit=MAX_PER_PAGE)


def _validated_dict(result: Any, endpoint: str) -> dict[str, Any]:
    """Return ``result`` if it's a dict, else raise.

    Every tool body is inside a ``try/except Exception -> _error(str(e))``
    block, so raising here (rather than returning a union every call site
    would have to narrow) reaches the caller's own error envelope unchanged.
    """
    if not isinstance(result, dict):
        raise RuntimeError(f"Freshdesk returned an unexpected response for {endpoint}")
    return result


def _validated_list(result: Any, endpoint: str) -> list[Any]:
    """Return ``result`` if it's a list, else raise.

    Freshdesk's list endpoints return a bare JSON array rather than an
    envelope object (the search endpoints differ and are validated as dicts
    via ``_validated_dict``).
    """
    if not isinstance(result, list):
        raise RuntimeError(f"Freshdesk returned an unexpected response for {endpoint}")
    return result


def _success_with_capped_list(list_field: str, payload: dict[str, Any]) -> str:
    """Build a success payload from a page of Freshdesk objects, halving
    ``payload[list_field]`` until it fits the platform's output limit.

    Mirrors chartmogul.py's helper of the same name, including its explicit
    "data lost" message when halving actually ran: the dropped objects belong
    to the page this one call already fetched, so they are not recoverable by
    asking for the next page -- only by re-running with a smaller per_page.
    Freshdesk makes this more likely than most: a ticket carries its full
    description, so a page of 100 can be very large.
    """
    max_output_length = get_tool_max_output_length()
    items = payload.get(list_field) or []

    def _build(items: list[Any], truncated: bool, halved: bool) -> str:
        extra: dict[str, Any] = {}
        if halved:
            extra["message"] = (
                f"Returned {len(items)} {list_field} out of the full page; the "
                "rest did not fit the output size limit and cannot be recovered "
                "by fetching the next page (a smaller per_page avoids this)."
            )
        return _success(
            **{**payload, list_field: items, "truncated": truncated, **extra}
        )

    halved = False
    response = _build(items, False, halved)
    while len(response) > max_output_length and items:
        items = items[: len(items) // 2]
        halved = True
        response = _build(items, True, halved)
    if halved and len(response) > max_output_length:
        # Even an empty item list didn't buy back enough room -- drop the
        # added "message" text itself as a last resort, matching chartmogul's
        # identical fallback.
        response = _build(items, True, False)
    return response


def _coerce_int(value: Any, field_name: str) -> int:
    """Coerce to int, refusing anything that would silently change meaning.

    ``int()`` truncates: ``int(9.7)`` is 9, so a fractional value used to be
    accepted as a different, valid-looking number. ``bool`` is an ``int``
    subclass, so ``True`` would otherwise pass as 1.
    """
    if isinstance(value, bool):
        raise RuntimeError(f"{field_name} must be an integer, got {value!r}")
    if isinstance(value, float) and not value.is_integer():
        raise RuntimeError(f"{field_name} must be a whole number, got {value!r}")
    try:
        return int(value)
    except (TypeError, ValueError):
        raise RuntimeError(f"{field_name} must be an integer, got {value!r}") from None


def _clean_tags(tags: list[str]) -> list[str]:
    """Strip each tag and drop the blanks.

    FastMCP validates only that this is a list of strings, and an LLM caller
    is exactly the source likely to pass "vip " -- which silently fails to
    match the canonical "vip" already on the account -- or an empty string
    left over from a split upstream of this tool.
    """
    return [t.strip() for t in tags if isinstance(t, str) and t.strip()]


def _resolve_tags(tags: list[str] | None) -> list[str] | None:
    """Resolve the caller's tags into what to send, or None to omit the field.

    Three states, and the middle one matters: None omits, an explicit empty
    list clears, and a non-empty list is cleaned. A non-empty list that cleans
    down to nothing -- ["  ", ""], the sloppiness _clean_tags absorbs -- must
    NOT collapse into the same request as an explicit clear. Freshdesk replaces
    the whole tag list, so that would wipe the tags off a ticket nobody asked
    to untag, with no undo in this connector. Ported from zendesk.py, which
    replaces its tag list the same way.
    """
    if tags is None:
        return None
    if not tags:
        return []
    cleaned = _clean_tags(tags)
    if not cleaned:
        raise RuntimeError(
            "tags contained no usable values (all entries were blank) -- pass "
            "an empty list [] to explicitly clear tags instead"
        )
    return cleaned


def _positive_id(value: Any, field_name: str) -> int:
    """Coerce and bounds-check an object id.

    Interpolated into the request path, so it is validated as a positive
    integer here rather than trusted: an LLM passing "12 OR 1=1", a float, or
    a path fragment would otherwise be pasted straight into the URL.
    ``field_name`` is echoed so a bad contact id does not report a complaint
    about a ticket id.
    """
    parsed = _coerce_int(value, field_name)
    if parsed <= 0:
        raise RuntimeError(f"{field_name} must be a positive integer, got {parsed}")
    return parsed


def _validated_priority(value: Any) -> int | None:
    """Validate priority against its closed set.

    Rejecting locally turns an LLM's plausible-but-wrong value into a message
    naming the legal ones instead of Freshdesk's generic "Validation failed".
    Safe to close because priority is not customizable.
    """
    if value is None:
        return None
    parsed = _coerce_int(value, "priority")
    if parsed not in TICKET_PRIORITIES:
        legal = ", ".join(f"{k} ({v})" for k, v in sorted(TICKET_PRIORITIES.items()))
        raise RuntimeError(f"priority must be one of: {legal}; got {parsed}")
    return parsed


def _validated_status(value: Any) -> int | None:
    """Bound-check a ticket status without closing the set.

    A helpdesk can define custom statuses with instance-specific numeric
    values above the four built-ins, so an unrecognized value is forwarded to
    Freshdesk to accept or reject -- this connector has no way to know a given
    tenant's legal set, and rejecting one would break every helpdesk that
    defines one.

    Values below the first real status are still refused: 0 and 1 are not
    statuses on any tenant, and 1 in particular is the trap worth catching --
    it is a legal *priority*, so an LLM reaches for it.
    """
    if value is None:
        return None
    parsed = _coerce_int(value, "status")
    if parsed < MIN_TICKET_STATUS:
        legal = ", ".join(f"{k} ({v})" for k, v in sorted(TICKET_STATUSES.items()))
        raise RuntimeError(
            f"status must be {MIN_TICKET_STATUS} or greater; got {parsed}. "
            f"The built-in statuses are {legal}; a custom status uses a "
            "higher, helpdesk-specific value."
        )
    return parsed


# ---------------------------------------------------------------------------
# Tickets
# ---------------------------------------------------------------------------


@mcp.tool()
def freshdesk_list_tickets(
    filter_name: str | None = None,
    updated_since: str | None = None,
    include: str | None = None,
    order_by: str | None = None,
    order_type: str | None = None,
    page: int = 1,
    per_page: int = DEFAULT_PER_PAGE,
) -> str:
    """
    List tickets in the Freshdesk helpdesk, newest first by default.

    ONLY TICKETS CREATED IN THE PAST 30 DAYS ARE RETURNED unless you pass
    updated_since. This is Freshdesk's own default, and nothing in the
    response distinguishes "no older tickets exist" from "older tickets were
    not looked at" -- has_more goes false at the end of the 30-day window. To
    cover anything older, pass updated_since explicitly, or use
    freshdesk_search_tickets, which is not windowed this way -- though it
    excludes archived tickets, so on a tenant with archiving enabled neither
    path reaches everything old.

    This endpoint offers only Freshdesk's canned views, not arbitrary
    filtering: use freshdesk_search_tickets for conditions such as
    "open tickets assigned to X".

    filter_name: one of Freshdesk's predefined views -- "new_and_my_open",
    "watching", "spam", "deleted". Omit for the default view, which excludes
    spam and deleted tickets.
    updated_since: ISO 8601 timestamp, e.g. "2026-09-01T00:00:00Z"; returns
    only tickets updated at or after it, and lifts the 30-day default window
    described above.
    include: comma-separated side-loads, e.g. "description,requester,stats".
    On accounts created after 2018-11-30 the ticket body is NOT returned by
    this endpoint unless you pass include=description -- without it the
    tickets come back with metadata only. Each side-load enlarges every ticket
    in the page, which makes size truncation more likely, so pair
    include=description with a smaller per_page.
    order_by: "created_at", "due_by", "updated_at" or "status".
    order_type: "asc" or "desc".
    page: 1-based page number. Freshdesk serves at most 300 pages (30,000
    tickets) from this endpoint; narrow with updated_since to reach past that.
    per_page: tickets per page (default 30, hard cap 100).

    The response carries has_more, taken from Freshdesk's own Link header, so
    it is exact rather than inferred. If truncated is true the page was
    trimmed to fit the output limit and the dropped tickets are NOT on the
    next page -- re-run with a smaller per_page to see them.
    """
    try:
        return _paged_list(
            "/tickets",
            "tickets",
            page=page,
            per_page=per_page,
            filters={
                "filter": filter_name,
                "updated_since": updated_since,
                "include": include,
                "order_by": order_by,
                "order_type": order_type,
            },
        )
    except Exception as exc:
        logger.error("Error in freshdesk_list_tickets: %s", exc)
        return _error(str(exc))


@mcp.tool()
def freshdesk_get_ticket(ticket_id: int, include: str | None = None) -> str:
    """
    Fetch one ticket by its numeric id, including its description.

    ticket_id: the ticket's numeric id (the number in its Freshdesk URL).
    include: comma-separated side-loads, e.g. "conversations,requester,stats".
    Note that "conversations" returns only the OLDEST ten ("up to ten
    conversations sorted by created_at in ascending order"), so on a long
    thread it shows the start, not the latest exchange -- use
    freshdesk_list_ticket_conversations to reach the rest.
    """
    try:
        ticket = _validated_dict(
            _request(
                "GET",
                f"/tickets/{_positive_id(ticket_id, 'ticket_id')}",
                params={"include": include},
            ),
            f"/tickets/{ticket_id}",
        )
        return success_with_capped_dict("ticket", ticket)
    except Exception as exc:
        logger.error("Error in freshdesk_get_ticket: %s", exc)
        return _error(str(exc))


@mcp.tool()
def freshdesk_search_tickets(query: str, page: int = 1) -> str:
    """
    Search tickets with Freshdesk's filter query language.

    query: the filter expression WITHOUT the enclosing double quotes this
    tool adds, e.g. status:2 AND priority:4, or
    agent_id:123 AND status:2, or
    created_at:>'2026-09-01'. String values take single quotes, numbers and
    booleans take none. Combine with AND/OR and parentheses. Freshdesk caps
    the expression at 512 characters.

    Queryable fields include agent_id, group_id, priority, status, tag,
    type, created_at, updated_at, due_by and custom fields.

    page: 1-based page number. Freshdesk fixes this endpoint at 30 results
    per page and refuses pages beyond 10, so at most 300 tickets are
    reachable for one query -- narrow the query rather than paging further.

    Archived tickets are NOT included in the results, so this is not a
    complete substitute for the 30-day window on freshdesk_list_tickets when a
    tenant has archiving enabled.

    The response carries total, Freshdesk's own count of matches, which can
    exceed the number reachable through paging.
    """
    try:
        expression = (query or "").strip()
        # A caller that already wrapped the expression is the common mistake,
        # and the old code turned it into '""status:2""' -- syntactically fine
        # for Freshdesk, and always empty. Strip exactly one surrounding pair,
        # then refuse any double quote that remains: Freshdesk quotes string
        # values with single quotes, so an interior double quote cannot be a
        # legitimate part of the expression and guessing at its intent would
        # silently change the query.
        if len(expression) >= 2 and expression[0] == '"' and expression[-1] == '"':
            expression = expression[1:-1].strip()
        if not expression:
            raise RuntimeError("query must not be empty")
        if '"' in expression:
            raise RuntimeError(
                "query must not contain a double quote; Freshdesk quotes "
                "string values with single quotes, e.g. tag:'urgent'"
            )
        # Freshdesk caps the quoted query at 512 characters, and the two
        # quotes added below count toward it -- so the budget for the caller's
        # expression is 510. Rejecting locally beats spending a request to be
        # told, and the message can say by how much it overran.
        quoted_length = len(expression) + 2
        if quoted_length > MAX_QUERY_LENGTH:
            raise RuntimeError(
                f"query is {quoted_length} characters once quoted, over "
                f"Freshdesk's {MAX_QUERY_LENGTH}-character limit; shorten it by "
                f"{quoted_length - MAX_QUERY_LENGTH}"
            )
        page_number = max(1, int(page))
        if page_number > MAX_SEARCH_PAGE:
            raise RuntimeError(
                f"Freshdesk refuses search pages beyond {MAX_SEARCH_PAGE} "
                f"(asked for {page_number}); narrow the query instead"
            )
        # The expression must reach Freshdesk wrapped in double quotes; requests
        # percent-encodes them. Quotes are added here rather than asked of the
        # caller because an LLM passing a pre-quoted string would otherwise
        # produce a doubly-quoted, always-empty search.
        response = _send(
            "GET",
            "/search/tickets",
            params={"query": f'"{expression}"', "page": page_number},
        )
        payload = _validated_dict(_body(response), "/search/tickets")
        total = payload.get("total")
        results = payload.get("results")
        if not isinstance(results, list):
            raise RuntimeError(
                "Freshdesk returned an unexpected response for /search/tickets"
            )
        return _success_with_capped_list(
            "results",
            {
                "results": results,
                "total": total,
                "page": page_number,
                # This endpoint sends no Link header, but it does return an
                # exact total in the same response, so use that and fall back
                # to the full-page heuristic only when it is missing or not a
                # number. The heuristic alone reported a phantom page whenever
                # the last page was exactly full. Either way page 10 is
                # Freshdesk's ceiling, so nothing beyond it is reachable.
                "has_more": page_number < MAX_SEARCH_PAGE
                and (
                    page_number * SEARCH_PAGE_SIZE < total
                    if isinstance(total, int) and not isinstance(total, bool)
                    else len(results) == SEARCH_PAGE_SIZE
                ),
            },
        )
    except Exception as exc:
        logger.error("Error in freshdesk_search_tickets: %s", exc)
        return _error(str(exc))


@mcp.tool()
def freshdesk_create_ticket(
    subject: str,
    description: str,
    email: str | None = None,
    requester_id: int | None = None,
    phone: str | None = None,
    name: str | None = None,
    status: int = STATUS_OPEN,
    priority: int = PRIORITY_LOW,
    responder_id: int | None = None,
    group_id: int | None = None,
    tags: list[str] | None = None,
    cc_emails: list[str] | None = None,
) -> str:
    """
    Create a ticket on behalf of a requester.

    subject: the ticket's subject line.
    description: the ticket body; Freshdesk renders it as HTML.
    email / requester_id / phone: how the requester is identified. At least
    one is required -- Freshdesk creates a new contact for an unknown email
    or phone.
    name: the requester's name. Freshdesk makes this MANDATORY when phone is
    given without email ("If the phone number is set and the email address is
    not, then the name attribute is mandatory"), and otherwise uses it to name
    the contact it creates for an unknown email -- so it is worth passing
    whenever the requester may be new, not only on the phone-only path.
    status: 2 Open (default), 3 Pending, 4 Resolved, 5 Closed.
    priority: 1 Low (default), 2 Medium, 3 High, 4 Urgent.
    responder_id: the agent to assign; omit to leave unassigned.
    group_id: the group to route to.
    tags: tags to set on the new ticket.
    cc_emails: addresses to copy on the ticket's email notifications.
    """
    try:
        if not (subject or "").strip():
            raise RuntimeError("subject must not be empty")
        if not (description or "").strip():
            raise RuntimeError("description must not be empty")

        # Normalize once and use the normalized values everywhere below: the
        # guards used to read stripped values while the payload sent the raw
        # ones, so a whitespace-only email was judged absent and still sent
        # (the "" filter below does not catch "   ").
        email = (email or "").strip() or None
        phone = (phone or "").strip() or None
        name = (name or "").strip() or None
        subject = subject.strip()
        description = description.strip()
        has_email = email is not None
        has_phone = phone is not None
        if requester_id is None and not has_email and not has_phone:
            raise RuntimeError(
                "one of email, requester_id or phone is required to identify "
                "the requester"
            )
        # Freshdesk rejects a phone-only create that carries no name. Catching
        # it here names the missing argument; forwarding it spends a request to
        # be told "Validation failed".
        if has_phone and not has_email and requester_id is None and name is None:
            raise RuntimeError(
                "name is required when creating a ticket from a phone number "
                "without an email address"
            )
        # Body ids get the same validation as path ids: 0 and negatives are
        # not valid Freshdesk object ids, and forwarding one spends a request
        # to be told so.
        for field_name, field_value in (
            ("requester_id", requester_id),
            ("responder_id", responder_id),
            ("group_id", group_id),
        ):
            if field_value is not None:
                _positive_id(field_value, field_name)
        payload: dict[str, Any] = {
            "subject": subject,
            "description": description,
            "status": _validated_status(status),
            "priority": _validated_priority(priority),
        }
        optional = {
            "email": email,
            "requester_id": requester_id,
            "phone": phone,
            "name": name,
            "responder_id": responder_id,
            "group_id": group_id,
            "tags": _resolve_tags(tags),
            "cc_emails": cc_emails,
        }
        payload.update({k: v for k, v in optional.items() if v not in (None, "", [])})
        ticket = _validated_dict(
            _request("POST", "/tickets", json_data=payload), "/tickets"
        )
        return success_with_capped_dict("ticket", ticket)
    except Exception as exc:
        logger.error("Error in freshdesk_create_ticket: %s", exc)
        return _error(str(exc))


@mcp.tool()
def freshdesk_update_ticket(
    ticket_id: int,
    status: int | None = None,
    priority: int | None = None,
    responder_id: int | None = None,
    group_id: int | None = None,
    tags: list[str] | None = None,
) -> str:
    """
    Update a ticket's status, priority, assignee, group or tags.

    Only the fields passed are changed; omitted fields are left alone.

    ticket_id: the ticket's numeric id.
    status: 2 Open, 3 Pending, 4 Resolved, 5 Closed.
    priority: 1 Low, 2 Medium, 3 High, 4 Urgent.
    responder_id: the agent to assign the ticket to.
    group_id: the group to route the ticket to.
    tags: REPLACES the ticket's entire tag list rather than adding to it --
    Freshdesk has no "append a tag" operation, so read the ticket first with
    freshdesk_get_ticket and pass the existing tags plus the new one, or the
    others are removed.
    """
    try:
        payload: dict[str, Any] = {}
        validated_status = _validated_status(status)
        if validated_status is not None:
            payload["status"] = validated_status
        validated_priority = _validated_priority(priority)
        if validated_priority is not None:
            payload["priority"] = validated_priority
        if responder_id is not None:
            payload["responder_id"] = _positive_id(responder_id, "responder_id")
        if group_id is not None:
            payload["group_id"] = _positive_id(group_id, "group_id")
        # An explicit empty list is a real instruction ("clear the tags") and
        # must survive, unlike None which means "leave them alone".
        resolved_tags = _resolve_tags(tags)
        if resolved_tags is not None:
            payload["tags"] = resolved_tags
        if not payload:
            raise RuntimeError(
                "pass at least one of status, priority, responder_id, group_id "
                "or tags to update"
            )
        ticket = _validated_dict(
            _request(
                "PUT",
                f"/tickets/{_positive_id(ticket_id, 'ticket_id')}",
                json_data=payload,
            ),
            f"/tickets/{ticket_id}",
        )
        return success_with_capped_dict("ticket", ticket)
    except Exception as exc:
        logger.error("Error in freshdesk_update_ticket: %s", exc)
        return _error(str(exc))


# ---------------------------------------------------------------------------
# Conversations
# ---------------------------------------------------------------------------


@mcp.tool()
def freshdesk_list_ticket_conversations(
    ticket_id: int,
    page: int = 1,
    per_page: int = DEFAULT_PER_PAGE,
) -> str:
    """
    List a ticket's conversation entries -- replies and notes -- oldest first.

    ticket_id: the ticket's numeric id.
    page: 1-based page number.
    per_page: entries per page (default 30, hard cap 100).

    Each entry carries `private`: true marks an internal note that the
    requester cannot see, false a reply that was sent to them.
    """
    try:
        return _paged_list(
            f"/tickets/{_positive_id(ticket_id, 'ticket_id')}/conversations",
            "conversations",
            page=page,
            per_page=per_page,
        )
    except Exception as exc:
        logger.error("Error in freshdesk_list_ticket_conversations: %s", exc)
        return _error(str(exc))


@mcp.tool()
def freshdesk_reply_to_ticket(
    ticket_id: int,
    body: str,
    cc_emails: list[str] | None = None,
    bcc_emails: list[str] | None = None,
) -> str:
    """
    Post a public reply to a ticket. THE REQUESTER IS EMAILED THIS TEXT.

    Use freshdesk_add_note_to_ticket for anything the customer should not
    see -- a reply cannot be unsent.

    ticket_id: the ticket's numeric id.
    body: the reply content; Freshdesk renders it as HTML.
    cc_emails / bcc_emails: additional recipients for this reply only.
    """
    try:
        if not (body or "").strip():
            raise RuntimeError("body must not be empty")
        payload: dict[str, Any] = {"body": body}
        if cc_emails:
            payload["cc_emails"] = cc_emails
        if bcc_emails:
            payload["bcc_emails"] = bcc_emails
        reply = _validated_dict(
            _request(
                "POST",
                f"/tickets/{_positive_id(ticket_id, 'ticket_id')}/reply",
                json_data=payload,
            ),
            f"/tickets/{ticket_id}/reply",
        )
        return success_with_capped_dict("reply", reply)
    except Exception as exc:
        logger.error("Error in freshdesk_reply_to_ticket: %s", exc)
        return _error(str(exc))


@mcp.tool()
def freshdesk_add_note_to_ticket(
    ticket_id: int,
    body: str,
    private: bool = True,
    notify_emails: list[str] | None = None,
) -> str:
    """
    Add a note to a ticket. Private by default.

    ticket_id: the ticket's numeric id.
    body: the note content; Freshdesk renders it as HTML.
    private: true (the default) keeps the note internal to agents. Passing
    false makes it visible to the requester in the ticket's portal view --
    which is a disclosure, so pass it deliberately rather than to "share
    context".
    notify_emails: agents to email about this note.
    """
    try:
        if not (body or "").strip():
            raise RuntimeError("body must not be empty")
        payload: dict[str, Any] = {"body": body, "private": bool(private)}
        if notify_emails:
            payload["notify_emails"] = notify_emails
        note = _validated_dict(
            _request(
                "POST",
                f"/tickets/{_positive_id(ticket_id, 'ticket_id')}/notes",
                json_data=payload,
            ),
            f"/tickets/{ticket_id}/notes",
        )
        return success_with_capped_dict("note", note)
    except Exception as exc:
        logger.error("Error in freshdesk_add_note_to_ticket: %s", exc)
        return _error(str(exc))


# ---------------------------------------------------------------------------
# Contacts and agents
# ---------------------------------------------------------------------------


@mcp.tool()
def freshdesk_get_contact(contact_id: int) -> str:
    """
    Fetch one contact by its numeric id.

    contact_id: the contact's numeric id, as carried by a ticket's
    requester_id.
    """
    try:
        contact = _validated_dict(
            _request("GET", f"/contacts/{_positive_id(contact_id, 'contact_id')}"),
            f"/contacts/{contact_id}",
        )
        return success_with_capped_dict("contact", contact)
    except Exception as exc:
        logger.error("Error in freshdesk_get_contact: %s", exc)
        return _error(str(exc))


@mcp.tool()
def freshdesk_search_contacts(
    term: str | None = None,
    email: str | None = None,
    phone: str | None = None,
    mobile: str | None = None,
    company_id: int | None = None,
    page: int = 1,
    per_page: int = DEFAULT_PER_PAGE,
) -> str:
    """
    Find contacts by name keyword or by an exact field match.

    term: a name fragment, matched by Freshdesk's autocomplete. Use it only
    when you do not have a precise value -- it returns a short ranked
    shortlist whose objects carry little more than id and name, and it is not
    a reliable way to look someone up by email address. For an email, use the
    email filter below, which is exact and authoritative.
    email / phone / mobile / company_id: exact-match filters, returning full
    contact objects.

    term cannot be combined with the exact filters; pass either one term or
    any number of filters. With neither, this lists contacts in Freshdesk's
    default order.

    page / per_page: pagination for the filtered listing (per_page default
    30, hard cap 100). They do not apply to a term search, which returns
    Freshdesk's own ranked shortlist in one response.
    """
    try:
        keyword = (term or "").strip()
        filters = {
            "email": email,
            "phone": phone,
            "mobile": mobile,
            "company_id": company_id,
        }
        active_filters = {k: v for k, v in filters.items() if v not in (None, "")}
        if keyword and active_filters:
            raise RuntimeError(
                "pass either term or the exact filters "
                f"({', '.join(sorted(active_filters))}), not both"
            )
        if keyword:
            # Freshdesk's documented keyword endpoint. The structured
            # /search/contacts filter API is marked BETA with no published
            # query grammar, so it is deliberately not built on here.
            contacts = _validated_list(
                _request("GET", "/contacts/autocomplete", params={"term": keyword}),
                "/contacts/autocomplete",
            )
            return _success_with_capped_list(
                "contacts", {"contacts": contacts, "has_more": False}
            )
        return _paged_list(
            "/contacts",
            "contacts",
            page=page,
            per_page=per_page,
            filters=active_filters,
        )
    except Exception as exc:
        logger.error("Error in freshdesk_search_contacts: %s", exc)
        return _error(str(exc))


@mcp.tool()
def freshdesk_list_agents(
    email: str | None = None,
    state: str | None = None,
    page: int = 1,
    per_page: int = DEFAULT_PER_PAGE,
) -> str:
    """
    List the helpdesk's agents, for resolving who to assign a ticket to.

    email: exact-match filter on an agent's email.
    state: "fulltime" or "occasional".
    page: 1-based page number.
    per_page: agents per page (default 30, hard cap 100).

    Each agent's `id` is what freshdesk_update_ticket takes as responder_id.
    """
    try:
        return _paged_list(
            "/agents",
            "agents",
            page=page,
            per_page=per_page,
            filters={"email": email, "state": state},
        )
    except Exception as exc:
        logger.error("Error in freshdesk_list_agents: %s", exc)
        return _error(str(exc))


if __name__ == "__main__":
    mcp.run()
