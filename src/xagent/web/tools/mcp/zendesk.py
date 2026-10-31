import hashlib
import json
import logging
import re
import socket
import time
from collections.abc import Callable
from os import environ
from typing import Any

from mcp.server.fastmcp import FastMCP

from ....config import get_tool_max_output_length, get_zendesk_marketplace_headers
from ....core.tools.core.web_content import get_trusted_proxy_url
from ....core.utils import public_requests as requests
from ....core.utils.security import (
    PrivateNetworkHostError,
    redact_sensitive_text,
    reject_private_network_host,
)
from ...utils.graphql_errors import truncate_error_text
from .utils import (
    clamp_limit,
    resolve_id_from_url,
    url_path_id,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("zendesk-mcp")


mcp = FastMCP("zendesk-mcp")


class _OutcomeUnknownError(RuntimeError):
    """A request whose effect on Zendesk is unknown: the connection failed, or
    Zendesk answered 5xx. A write may still have been applied."""


# A shared Session (HTTP keep-alive / connection pooling) rather than a bare
# requests.request() per call -- most benefit accrues to this module's own
# 429-retry (a second request to the same host right after the first), and
# to any single tool call that happens to make more than one request; a
# fresh connection per call is otherwise a fixed cost this avoids for free.
_session = requests.Session()
# trust_env=False so requests never falls back to an ambient/OS-native proxy
# source get_trusted_proxy_url() doesn't police: a trust_env=True Session
# (requests' default) still calls urllib.request.getproxies(), which falls
# back *past* HTTP_PROXY/HTTPS_PROXY/ALL_PROXY to the OS's own proxy
# configuration once no env var is set. A proxy performs its own DNS
# resolution for the real connection, so any ambient proxy silently bypasses
# _base_url()'s private-network check of the addresses *this process*
# resolved -- the exact DNS-rebinding-via-proxy hole that check exists to
# close. Same intent as posthog.py/magento.py, the two other connectors
# carrying that check -- though applied once here (at import, to this
# module-level _session) rather than per-call (their _make_request()
# builds a fresh Session and re-reads these env vars on every call): each
# MCP tool call runs in its own fresh subprocess (see
# mcp_adapter._execute_mcp_call), so "at import" and "at the one call this
# process will ever make" are the same moment here, same as it already is
# for the rest of this module-level _session's configuration.
# trust_env=False also disables requests' own REQUESTS_CA_BUNDLE/
# CURL_CA_BUNDLE lookup, so it is re-applied explicitly: an operator
# opting into a trusted egress proxy (XAGENT_TRUSTED_EGRESS_PROXY=1) is
# the textbook TLS-intercepting corporate proxy with an internal CA,
# which would otherwise fail closed with an opaque SSLError. .netrc
# auto-auth stays disabled since this connector always sends its own
# Bearer token.
_session.trust_env = False
_ca_bundle = environ.get("REQUESTS_CA_BUNDLE") or environ.get("CURL_CA_BUNDLE")
if _ca_bundle:
    _session.verify = _ca_bundle

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
# paginated -- checked client-side (see _past_search_window) so a page that
# cannot contain a single valid result gets a predictable empty response
# instead of an opaque remote error. The two search endpoints have
# different documented ceilings -- the unified /search.json is capped at
# 1,000 results, while /users/search.json goes to 10,000 -- so they are NOT
# interchangeable despite both being "search" endpoints.
_MAX_SEARCH_RESULT_WINDOW = 1000
_MAX_USER_SEARCH_RESULT_WINDOW = 10000
# Help Center article search has its own documented ceiling, separate
# from the two Support search endpoints above: "up to a maximum of 1000
# results" (developer.zendesk.com/api-reference/help_center/
# help-center-api/help_center_search/).
_MAX_ARTICLE_SEARCH_RESULT_WINDOW = 1000

# Accepts a ticket/user/organization id pasted as a Zendesk agent UI URL
# (e.g. "https://acme.zendesk.com/agent/tickets/123") in addition to a bare
# numeric id -- an agent copying a link out of the Zendesk UI is a common
# enough input shape that requiring the bare id only would otherwise surface
# as an opaque Pydantic ValidationError.
_TICKET_URL_ID_PATTERN = re.compile(r"/agent/tickets/(\d+)")
_USER_URL_ID_PATTERN = re.compile(r"/agent/users/(\d+)")
_ORGANIZATION_URL_ID_PATTERN = re.compile(r"/agent/organizations/(\d+)")
# Help Center article URLs: the public page (/hc/en-us/articles/123-slug)
# and the Knowledge admin editor (/knowledge/articles/123/en-us).
_ARTICLE_URL_ID_PATTERN = re.compile(r"/articles/(\d+)")
# Admin Center's macro editor (/admin/workspaces/agent-workspace/macros/123).
_MACRO_URL_ID_PATTERN = re.compile(r"/macros/(\d+)")


def _success(**payload: Any) -> str:
    return json.dumps({"status": "success", **payload}, ensure_ascii=False)


def _error(message: str) -> str:
    return json.dumps({"status": "error", "message": message}, ensure_ascii=False)


def _access_token() -> str:
    """The OAuth access token the platform injects for this connection.

    Zendesk OAuth grants are per account: the token is only valid on the
    subdomain it was issued for, which the platform injects alongside it as
    ZENDESK_SUBDOMAIN from the same stored grant.
    """
    # Stripped like ZENDESK_SUBDOMAIN below: a whitespace-only value is not
    # a usable credential and is treated as missing.
    token = environ.get("ZENDESK_ACCESS_TOKEN", "").strip()
    if not token:
        raise ValueError(
            "ZENDESK_ACCESS_TOKEN environment variable is missing; reconnect "
            "Zendesk to authorize this connector"
        )
    return token


def _headers() -> dict[str, str]:
    # The Marketplace settings reach this subprocess through the catalog
    # row's launch_config.static_env; unset, they add nothing.
    return {
        "Authorization": f"Bearer {_access_token()}",
        **get_zendesk_marketplace_headers(),
    }


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
    _access_token()


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


def _blank_to_none(value: str | None) -> str | None:
    """Normalize a blank (empty or whitespace-only) optional string to
    None, so callers can treat "not provided" and "provided as blank" the
    same way -- e.g. status/priority have no valid "clear the field" value
    in Zendesk, so a blank one must be dropped rather than sent verbatim."""
    return (value or "").strip() or None


# Zendesk's closed enums for the two ticket fields this connector writes.
# Validated locally (case-insensitively) so a typo surfaces as a clear local
# error naming the accepted values, instead of a remote 422 whose body only
# says the value is invalid -- the same posture intercom.py takes for its
# analogous `state` parameter.
#
# "closed" is deliberately excluded from the settable set: Zendesk's status
# field can *read* as "closed", but reaching it is automation-only (a set
# number of days after "solved") -- a direct PUT trying to set it is
# rejected remotely, so allowing it through local validation would only
# trade one confusing error for another.
_TICKET_STATUSES = frozenset({"new", "open", "pending", "hold", "solved"})
_TICKET_PRIORITIES = frozenset({"low", "normal", "high", "urgent"})


def _require_one_of(value: str, allowed: frozenset[str], field_name: str) -> str:
    normalized = value.lower()
    if normalized not in allowed:
        raise ValueError(
            f"{field_name} must be one of {sorted(allowed)}, got {value!r}"
        )
    return normalized


# Any query string in text that came back from `requests`: both
# requests.HTTPError (via response.url) and connection-level exceptions (via
# urllib3's "Max retries exceeded with url: /path?query=...") echo the full
# request URL, and the search tools' `query` parameter -- documented with an
# email-address example -- rides in that query string. redact_sensitive_text
# only knows credential-shaped keys (api_key/token/...), not `query=`, so the
# query string is scrubbed wholesale before any exception text is logged or
# returned.
_QUERY_STRING_PATTERN = re.compile(r"\?[^\s\"'()<>]+")
# A bare "?" alone doesn't mean a URL query string -- an ordinary sentence
# like "...is this a duplicate ticket? Contact support." would otherwise get
# mangled. Only redact when the "?" is immediately preceded (within the same
# whitespace-delimited token) by something path-shaped -- containing a "/",
# as any real URL or path does -- leaving plain prose untouched.
_PATH_LIKE_TOKEN_PATTERN = re.compile(r"[^\s\"'()<>]*$")


def _scrub_query_strings(text: str) -> str:
    def _redact(match: re.Match[str]) -> str:
        preceding_token = _PATH_LIKE_TOKEN_PATTERN.search(text[: match.start()])
        token = preceding_token.group(0) if preceding_token else ""
        return "?<query redacted>" if "/" in token else match.group(0)

    return _QUERY_STRING_PATTERN.sub(_redact, text)


def _sanitize_exception_text(exc: BaseException) -> str:
    return _scrub_query_strings(redact_sensitive_text(str(exc)))


_OVERSIZED_ITEMS_MESSAGE = (
    "Every item in this page was individually too large to fit the output "
    "size limit, so none could be returned. Retrying with a smaller `limit` "
    "may surface different items if more exist, but cannot shrink an "
    "individually oversized item."
)
_OVERSIZED_ITEMS_MESSAGE_SHORT = (
    "Every item in this page was too large to fit; retrying will not help."
)


def _finalize_capped(response: str, max_output_length: int) -> str:
    """Last line of defense for every size-capped response builder in this
    file: the halving loops above it shrink *content*, but the fixed
    envelope around that content (status/has_more/cursor keys) has a floor
    they cannot shrink below. An operator can configure
    XAGENT_TOOL_MAX_OUTPUT_LENGTH under that floor, and the platform's
    output filter then truncates the oversized string at a raw character
    boundary -- handing the caller invalid JSON. A short, valid error
    envelope is strictly better than that, and names the actual cause.

    The error envelope itself is shrunk in stages rather than returned
    unconditionally: a cap small enough to reject the full response can
    also be too small for the detailed explanation, so each stage below is
    tried only if the previous one still doesn't fit."""
    if len(response) <= max_output_length:
        return response
    detailed = _error(
        "response cannot fit the configured output cap of "
        f"{max_output_length} characters even after truncation; raise "
        "XAGENT_TOOL_MAX_OUTPUT_LENGTH"
    )
    if len(detailed) <= max_output_length:
        return detailed
    minimal = _error("output cap too small")
    if len(minimal) <= max_output_length:
        return minimal
    # Below the size of any valid JSON error envelope this function can
    # construct -- there is nothing left to shrink. XAGENT_TOOL_MAX_OUTPUT_LENGTH
    # would need to be set below roughly 20 characters to reach this.
    return json.dumps({"status": "error"}, ensure_ascii=False)


def _error_capped(message: str) -> str:
    """Every success path in this file is capped before it's returned; an
    error message built from Zendesk's own (redacted) response detail
    deserves the same guarantee -- an oversized error is still an oversized
    string for the platform's output filter to mangle into invalid JSON."""
    return _finalize_capped(_error(message), get_tool_max_output_length())


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


def _extract_field_reasons(details: Any) -> str | None:
    """Join RecordInvalid's per-field reasons into one readable string.

    A 422 RecordInvalid body's "description" is a generic
    "Record validation errors" -- the actual reason (e.g. "Status: closed
    is not valid for ticket update", or a duplicate-email message) lives in
    "details", keyed by field name (or "base") with a list of
    {"description": ...} entries.
    """
    if not isinstance(details, dict):
        return None
    reasons: list[str] = []
    for entries in details.values():
        if not isinstance(entries, list):
            continue
        for entry in entries:
            text = entry.get("description") if isinstance(entry, dict) else entry
            if isinstance(text, str) and text:
                reasons.append(text)
    if not reasons:
        return None
    # A validation error can name arbitrarily many fields (e.g. a bulk-style
    # payload with every field invalid) -- bounded the same way the
    # unstructured response-text fallback below already is, so this can't
    # be the thing that pushes a raised error past the output cap.
    return truncate_error_text("; ".join(reasons))


def _extract_error_detail(body: str) -> str | None:
    """Pull the human-readable message out of a Zendesk error body.

    Zendesk's error shape is inconsistent across endpoints: a plain string
    "error" with a separate "description" (e.g. RecordNotFound), or a
    nested {"error": {"title", "message"}} object (e.g. some auth
    failures). Falls back to the raw response text when neither a message
    nor field-level details are present.
    """
    try:
        payload = json.loads(body)
    except ValueError:
        return None
    if not isinstance(payload, dict):
        return None
    message = None
    description = payload.get("description")
    if isinstance(description, str) and description:
        message = description
    else:
        error = payload.get("error")
        if isinstance(error, str) and error:
            message = error
        elif isinstance(error, dict):
            nested = error.get("message") or error.get("title")
            if isinstance(nested, str) and nested:
                message = nested

    field_reasons = _extract_field_reasons(payload.get("details"))
    if message and field_reasons:
        return f"{message}: {field_reasons}"
    return field_reasons or message


def _request(
    method: str,
    path: str,
    *,
    params: dict[str, Any] | None = None,
    json_data: dict[str, Any] | None = None,
) -> Any:
    # An ambient HTTP(S) proxy makes the *proxy* perform DNS resolution for
    # the real connection, not this process -- silently bypassing
    # _base_url()'s private-network validation, which only checks the
    # addresses this process itself resolves. get_trusted_proxy_url()
    # raises unless the proxy is explicitly marked trusted to enforce its
    # own private-range egress policy (XAGENT_TRUSTED_EGRESS_PROXY=1),
    # rather than silently trusting ambient or OS-native proxies.
    # Paired with the module session's trust_env=False, "no
    # proxy" here is an actual guarantee for the call, not just for the env
    # vars this function reads.
    try:
        proxy_url = get_trusted_proxy_url()
    except PrivateNetworkHostError as exc:
        raise type(exc)(redact_sensitive_text(str(exc))) from exc
    proxies = {"http": proxy_url, "https": proxy_url} if proxy_url else {}

    url = f"{_base_url()}{path}"
    try:
        for attempt in (0, 1):
            response = _session.request(
                method=method,
                url=url,
                headers=_headers(),
                params=params,
                json=json_data,
                timeout=DEFAULT_TIMEOUT_SECONDS,
                proxies=proxies,
                # A redirect response is never followed with the Bearer
                # token still attached: Zendesk's documented API
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
        # A connection/timeout/proxy failure's message can embed sensitive
        # data two ways: a ProxyError echoing a proxy URL with user:pass@
        # credentials, and urllib3's "Max retries exceeded with url:
        # /search.json?query=..." echoing the request's own query string
        # (which for the search tools can carry end-user PII).
        raise _OutcomeUnknownError(_sanitize_exception_text(exc)) from exc

    if 300 <= response.status_code < 400:
        raise RuntimeError(
            f"Zendesk returned an unexpected redirect (HTTP {response.status_code}); "
            "refusing to follow it with credentials attached"
        )
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        # Deliberately NOT str(exc): requests formats an HTTPError as
        # "<status> ... for url: <full url>", and the full URL includes the
        # query string -- for the search tools, the caller's raw query
        # (documented with an email-address example). Status + Zendesk's
        # own (redacted) error detail is everything the caller can act on.
        message = f"Zendesk returned HTTP {response.status_code}"
        # A provider error body could echo the credential back. It is removed
        # from the raw body once, before any parsing or truncation could cut
        # it into a prefix that no longer matches.
        body = response.text.replace(_access_token(), "[redacted]")
        detail = _extract_error_detail(body)
        if detail is None:
            detail = truncate_error_text(body.strip())
        if detail:
            detail = _scrub_query_strings(redact_sensitive_text(detail))
            message = f"{message} - {detail}"
        # Neither heals on retry, so say what does.
        if response.status_code == 401:
            message = (
                f"{message}. The Zendesk authorization expired or was revoked; "
                "reconnect Zendesk and try again."
            )
        elif response.status_code == 403:
            message = (
                f"{message}. The connected Zendesk user's role or the scopes "
                "granted to this connection do not allow this request; "
                "reconnecting Zendesk requests the scopes again."
            )
        if response.status_code >= 500:
            raise _OutcomeUnknownError(message) from exc
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
    """Slice one page of an offset-paginated Zendesk response (the search
    endpoints and the side conversation endpoints only support offset
    pagination, not the cursor style the other list endpoints here use)."""
    page, items = _extract_list_field(payload, list_key, limit)
    has_more = bool(payload.get("next_page")) or len(items) > limit
    return page, has_more


def _past_search_window(page: int, max_results: int, ceiling: int) -> bool:
    """True once the requested page *starts* at or past Zendesk's documented
    search-window ceiling for the endpoint being called (1,000 for
    /search.json and /help_center/articles/search.json, 10,000 for
    /users/search.json -- see _MAX_SEARCH_RESULT_WINDOW /
    _MAX_ARTICLE_SEARCH_RESULT_WINDOW / _MAX_USER_SEARCH_RESULT_WINDOW), i.e. the
    page cannot contain a single result Zendesk would serve.

    Deliberately the window's *start*, not its end: a page that merely
    straddles the ceiling (limit=30, page=34 spans 991-1020 of a 1,000
    window) still holds results 991-1000, and rejecting it locally would
    silently discard them with has_more=false -- unrecoverable through this
    tool. Forwarding it instead has a strictly better worst case: Zendesk
    either serves the valid tail, or answers 422 (documented for pages past
    the limit), which _request surfaces as a visible, structured error
    rather than silent data loss."""
    return (page - 1) * max_results >= ceiling


def _list_offset_paginated(
    path: str,
    list_key: str,
    summary_fn: Callable[[dict[str, Any]], dict[str, Any]],
    params: dict[str, Any],
    limit: int,
    extra_fields_fn: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> str:
    """Shared body for the offset-paginated tools (zendesk_search,
    zendesk_search_users, zendesk_search_articles, and the two side
    conversation tools) -- symmetric to _list_cursor_paginated below, so
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

    def _build(
        page: list[dict[str, Any]], truncated: bool, message: str | None = None
    ) -> str:
        return _success(
            **{list_key: page},
            **extra_fields,
            # See _list_cursor_paginated's identical comment: a truncated
            # page must report has_more=True regardless of what Zendesk's
            # own response said, since the caller can't yet see the items
            # this call dropped for size.
            has_more=has_more or truncated,
            truncated=truncated,
            **({"message": message} if message else {}),
        )

    response = _build(summaries, False)
    max_output_length = get_tool_max_output_length()
    if len(response) <= max_output_length:
        return response

    while len(response) > max_output_length and summaries:
        summaries = summaries[: len(summaries) // 2]
        response = _build(summaries, True)
    if not summaries:
        # Collapsed all the way to zero: even the single largest item didn't
        # fit alone, so has_more=true here is NOT the usual "retry with a
        # smaller limit" situation -- a smaller limit cannot shrink an
        # individually oversized item. Say so (mirrors hubspot.py's
        # _paged_list) -- but never fall through to the bare has_more=true
        # envelope silently just because the explanation didn't fit: without
        # it, this is indistinguishable from ordinary truncation and drives
        # the caller into a retry loop that can never make progress. Try a
        # shorter explanation before giving up on explaining at all.
        for message in (_OVERSIZED_ITEMS_MESSAGE, _OVERSIZED_ITEMS_MESSAGE_SHORT):
            candidate = _build([], True, message)
            if len(candidate) <= max_output_length:
                return candidate
        return _finalize_capped(
            _error(
                "every item in this page was too large to fit the "
                "configured output cap; raise XAGENT_TOOL_MAX_OUTPUT_LENGTH"
            ),
            max_output_length,
        )
    return _finalize_capped(response, max_output_length)


def _list_cursor_paginated(
    path: str,
    list_key: str,
    summary_fn: Callable[[dict[str, Any]], dict[str, Any]],
    limit: int,
    after_cursor: str | None,
) -> str:
    """Shared body for every cursor-paginated "list X" tool below (tickets,
    ticket comments, users, organizations, macros) -- only the path, response key,
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

    def _build(
        page: list[dict[str, Any]],
        cursor: str | None,
        truncated: bool,
        message: str | None = None,
    ) -> str:
        return _success(
            **{list_key: page},
            # A truncated page must report has_more=True even if Zendesk's
            # own meta said this was the last page: the items dropped by
            # truncation are still unread, so a caller that trusts has_more
            # alone (a natural reading of that field) must not stop here.
            has_more=has_more or truncated,
            after_cursor=cursor,
            truncated=truncated,
            **({"message": message} if message else {}),
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
    if not summaries:
        # Collapsed to zero: the single largest item didn't fit alone, so
        # "retry the same cursor with a smaller limit" cannot help -- a
        # smaller limit doesn't shrink an individually oversized item. Say
        # so (hubspot.py's _paged_list pattern) -- but never fall through to
        # the bare has_more=true envelope silently just because the
        # explanation didn't fit: without it, this is indistinguishable from
        # ordinary truncation and drives the caller into a retry loop that
        # can never make progress. Try a shorter explanation before giving
        # up on explaining at all.
        for message in (_OVERSIZED_ITEMS_MESSAGE, _OVERSIZED_ITEMS_MESSAGE_SHORT):
            candidate = _build([], fallback_cursor, True, message)
            if len(candidate) <= max_output_length:
                return candidate
        return _finalize_capped(
            _error(
                "every item in this page was too large to fit the "
                "configured output cap; raise XAGENT_TOOL_MAX_OUTPUT_LENGTH"
            ),
            max_output_length,
        )
    return _finalize_capped(response, max_output_length)


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


def _ticket_detail(ticket: dict[str, Any]) -> dict[str, Any]:
    """_ticket_summary plus the fields only worth the size cost for a
    single ticket -- the caller asked about (or just created/updated)
    exactly one ticket, so its actual text is the point, unlike a list or
    search result where dozens of tickets share one output cap. Mirrors
    linear.py's _ISSUE_SUMMARY_FIELDS/_ISSUE_DETAIL_FIELDS split.

    custom_fields is deliberately excluded: unlike the fields below, it's
    unbounded and shaped entirely by the caller's own account
    configuration, not a fixed Zendesk schema -- the truncation cost/benefit
    is too unpredictable to include unconditionally here."""
    return {
        **_ticket_summary(ticket),
        "description": ticket.get("description"),
        "organization_id": ticket.get("organization_id"),
        "type": ticket.get("type"),
        "via": ticket.get("via"),
    }


def _build_ticket_detail_response(
    ticket: dict[str, Any], max_output_length: int
) -> str:
    """Shared by zendesk_get_ticket/zendesk_create_ticket/
    zendesk_update_ticket: description can be up to 64KiB, Zendesk's own
    limit on a comment body (description *is* the ticket's first comment),
    well past a typical output cap. Shrinks description the same way
    _add_comment already shrinks an oversized comment body, rather than
    letting one long ticket fail the whole call outright via
    _finalize_capped's blunter last-resort fallback."""
    return _build_detail_response(
        "ticket", _ticket_detail(ticket), "description", max_output_length
    )


def _build_detail_response(
    key: str, detail: dict[str, Any], text_field: str, max_output_length: int
) -> str:
    """Return one object under `key`, halving its single long free-text
    field (marking `<text_field>_truncated`) until the response fits, and
    only then falling back to _finalize_capped's error."""
    response = _success(**{key: detail})
    text = detail.get(text_field)
    while len(response) > max_output_length and isinstance(text, str) and text:
        text = text[: len(text) // 2]
        detail = {**detail, text_field: text, f"{text_field}_truncated": True}
        response = _success(**{key: detail})
    return _finalize_capped(response, max_output_length)


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


def _article_summary(article: dict[str, Any]) -> dict[str, Any]:
    """A Help Center search hit: the search response's own `snippet`
    (the matched passage) instead of the HTML body, which only
    zendesk_get_article returns -- a page of 25 full article bodies would
    otherwise be truncated away almost every time."""
    return {
        "id": article.get("id"),
        "title": article.get("title"),
        "html_url": article.get("html_url"),
        "locale": article.get("locale"),
        "section_id": article.get("section_id"),
        "snippet": article.get("snippet"),
        "updated_at": article.get("updated_at"),
    }


def _article_detail(article: dict[str, Any]) -> dict[str, Any]:
    """A single article, with its HTML body as Zendesk stores it."""
    return {
        "id": article.get("id"),
        "title": article.get("title"),
        "html_url": article.get("html_url"),
        "locale": article.get("locale"),
        "section_id": article.get("section_id"),
        "author_id": article.get("author_id"),
        "draft": article.get("draft"),
        "label_names": article.get("label_names"),
        "body": article.get("body"),
        "created_at": article.get("created_at"),
        "updated_at": article.get("updated_at"),
    }


def _macro_summary(macro: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": macro.get("id"),
        "title": macro.get("title"),
        "description": macro.get("description"),
        "updated_at": macro.get("updated_at"),
    }


def _macro_detail(macro: dict[str, Any]) -> dict[str, Any]:
    return {
        **_macro_summary(macro),
        # Always true in zendesk_list_macros (it lists /macros/active), but
        # a macro fetched by id can be inactive.
        "active": macro.get("active"),
        "actions": macro.get("actions"),
        "restriction": macro.get("restriction"),
        "created_at": macro.get("created_at"),
    }


def _side_conversation_summary(conversation: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": conversation.get("id"),
        "subject": conversation.get("subject"),
        "state": conversation.get("state"),
        "preview_text": conversation.get("preview_text"),
        "participants": conversation.get("participants"),
        "created_at": conversation.get("created_at"),
        "updated_at": conversation.get("updated_at"),
        "message_added_at": conversation.get("message_added_at"),
    }


def _side_conversation_event_summary(event: dict[str, Any]) -> dict[str, Any]:
    """One event in a side conversation's history. A message event carries
    its plain-text body, falling back to html_body when that is empty (then
    `body_is_html` is true, so a caller can tell markup from plain text); a
    state change carries `updates` and no message."""
    message = event.get("message")
    return {
        "id": event.get("id"),
        "type": event.get("type"),
        "created_at": event.get("created_at"),
        "actor": event.get("actor"),
        "updates": event.get("updates"),
        "message": (
            {
                "subject": message.get("subject"),
                "from": message.get("from"),
                "to": message.get("to"),
                "body": message.get("body") or message.get("html_body"),
                "body_is_html": not message.get("body")
                and bool(message.get("html_body")),
                "attachments": message.get("attachments"),
            }
            if isinstance(message, dict)
            else None
        ),
    }


# Macro actions zendesk_apply_macro can turn into one ticket update under
# the connector's tickets:write scope. Every other action (side
# conversations, CCs, followers, custom fields, ...) is refused before
# anything is saved: the apply preview silently leaves out actions it
# can't express as ticket fields, so it can't be trusted to show them.
# custom_status_id is deliberately absent: a macro that sets a custom ticket
# status is refused, while one that sets a status category ("status")
# applies, and Zendesk picks that category's default custom status.
_MACRO_FIELD_ACTIONS = ("status", "priority", "assignee_id", "group_id")
_MACRO_TAG_ACTIONS = frozenset({"current_tags", "remove_tags", "set_tags"})
_MACRO_COMMENT_ACTIONS = frozenset({"comment_value", "comment_value_html"})
_MACRO_COMMENT_MODE_ACTION = "comment_mode_is_public"
_ALLOWED_MACRO_ACTIONS = (
    frozenset(_MACRO_FIELD_ACTIONS)
    | _MACRO_TAG_ACTIONS
    | _MACRO_COMMENT_ACTIONS
    | {_MACRO_COMMENT_MODE_ACTION}
)


def _macro_comment_is_public(actions: list[dict[str, Any]]) -> bool:
    """The macro's comment visibility. Without a comment_mode_is_public
    action the comment is public, the agent UI's default reply mode. An
    unrecognized value is refused rather than guessed: a wrong guess posts
    an internal note to the customer."""
    for action in actions:
        if action.get("field") != _MACRO_COMMENT_MODE_ACTION:
            continue
        value = action.get("value")
        normalized = str(value).strip().lower()
        if normalized in ("true", "false"):
            return normalized == "true"
        raise ValueError(
            f"the macro sets {_MACRO_COMMENT_MODE_ACTION} to {value!r}, which "
            "is not true or false; nothing was saved"
        )
    return True


def _build_apply_macro_response(payload: dict[str, Any]) -> str:
    """Cap an apply-macro response, shrinking in order of what the caller
    loses least: the reported comment body, then the saved ticket's summary
    (down to its id), then the change details. A saved result is never
    turned into an error, down to the smallest success envelope: an error
    invites a retry, and a retry of a save posts the comment again."""
    max_output_length = get_tool_max_output_length()
    response = _success(**payload)
    comment = payload["changes"].get("comment")
    body = comment.get("body") if isinstance(comment, dict) else None
    while (
        not payload["dry_run"]
        and len(response) > max_output_length
        and isinstance(body, str)
        and body
    ):
        body = body[: len(body) // 2]
        payload = {
            **payload,
            "truncated": True,
            "changes": {
                **payload["changes"],
                "comment": {**comment, "body": body, "body_truncated": True},
            },
        }
        response = _success(**payload)
    if len(response) > max_output_length and isinstance(payload.get("ticket"), dict):
        payload = {
            **payload,
            "truncated": True,
            "ticket": {"id": payload["ticket"].get("id")},
        }
        response = _success(**payload)
    if len(response) > max_output_length and payload["dry_run"]:
        # Never hand out a preview_token for changes the caller can't see:
        # it would let them be saved unreviewed.
        return _error_capped(
            "this macro's preview is too large to show within the output "
            "limit, so it can't be approved here; nothing was saved. Raise "
            "XAGENT_TOOL_MAX_OUTPUT_LENGTH or apply it in the Zendesk agent UI."
        )
    if len(response) > max_output_length:
        payload = {**payload, "truncated": True, "changes": {"truncated": True}}
        response = _success(**payload)
    if len(response) > max_output_length and not payload["dry_run"]:
        # The save is committed: keep shrinking to the smallest success
        # envelope rather than let the cap turn it into an error.
        for fallback in (
            {
                "dry_run": False,
                "ticket_id": payload.get("ticket_id"),
                "truncated": True,
            },
            {"dry_run": False, "truncated": True},
            {"dry_run": False},
            {},
        ):
            response = _success(**fallback)
            if len(response) <= max_output_length:
                break
    return _finalize_capped(response, max_output_length)


def _resolve_path_id(
    value: int | str, pattern: re.Pattern[str], field_name: str
) -> str:
    """Accept a bare id (int or str) or a full Zendesk agent UI URL (e.g.
    "https://acme.zendesk.com/agent/tickets/123"), then percent-encode the
    result for safe interpolation into a URL path segment."""
    return url_path_id(resolve_id_from_url(str(value), pattern, field_name), field_name)


def _find_comment_event(result: Any, public: bool) -> dict[str, Any] | None:
    """Pull the newly created Comment event out of a ticket-update
    response's audit trail (PUT /tickets/{id}.json returns {"ticket":
    ..., "audit": {"events": [...]}} -- the audit, not the ticket object
    itself, is where a just-added comment's own id/body/public actually
    show up), so reply/note tools can confirm exactly what was posted --
    including `public`, the one field that distinguishes a customer-visible
    reply from an internal note.

    Each audit covers a single update and this connector posts exactly one
    comment per update, so the Comment event is normally unique. The
    `public` match is cheap insurance for the residual case (an
    account-configured integration adding its own comment into the same
    audit): a candidate whose visibility differs from what was sent can
    never be the one this call created, so it's skipped rather than risk
    labeling an internal note as public."""
    if not isinstance(result, dict):
        return None
    events = (result.get("audit") or {}).get("events") or []
    for event in events:
        if (
            isinstance(event, dict)
            and event.get("type") == "Comment"
            and event.get("public") == public
        ):
            return _comment_summary(event)
    return None


def _add_comment(ticket_id: int | str, body: str, public: bool) -> str:
    """Post a comment and return the already-capped success envelope (not
    a dict for the caller to wrap) -- a comment body can be up to 64KiB
    (Zendesk's own limit, enforced by _require_comment_body above), well
    past a configured output cap, and this is the only mutation response
    in this file that echoes caller-supplied free text back verbatim, so
    it needs its own truncation handling rather than relying on the list
    tools' page-shrinking (there's no list here to shrink).

    Two things can be oversized, and they are shrunk in order of how much
    the caller loses: first the echoed comment body (a preview; id/public
    still confirm what was posted), then the ticket summary (collapsed to
    just its id -- the correlation key -- since a long subject or a large
    tag set is fixed-size content no halving of the *comment* can offset).
    A top-level `truncated` flag reports either cut."""
    body = _require_comment_body(body, "body")
    result = _request(
        "PUT",
        f"/tickets/{_resolve_path_id(ticket_id, _TICKET_URL_ID_PATTERN, 'ticket_id')}.json",
        json_data={"ticket": {"comment": {"body": body, "public": public}}},
    )
    ticket = _unwrap(result, "ticket")
    ticket_summary = _ticket_summary(ticket) if isinstance(ticket, dict) else ticket
    comment = _find_comment_event(result, public)

    def _build(
        ticket_value: Any, comment_value: dict[str, Any] | None, truncated: bool
    ) -> str:
        return _success(ticket=ticket_value, comment=comment_value, truncated=truncated)

    response = _build(ticket_summary, comment, False)
    max_output_length = get_tool_max_output_length()
    if len(response) <= max_output_length:
        return response

    truncated_comment = dict(comment) if isinstance(comment, dict) else comment
    comment_body = (
        truncated_comment.get("body") if isinstance(truncated_comment, dict) else None
    )
    while (
        len(response) > max_output_length
        and isinstance(truncated_comment, dict)
        and isinstance(comment_body, str)
        and comment_body
    ):
        comment_body = comment_body[: len(comment_body) // 2]
        truncated_comment["body"] = comment_body
        truncated_comment["body_truncated"] = True
        response = _build(ticket_summary, truncated_comment, True)

    if len(response) > max_output_length and isinstance(ticket_summary, dict):
        ticket_summary = {"id": ticket_summary.get("id")}
        response = _build(ticket_summary, truncated_comment, True)
    return _finalize_capped(response, max_output_length)


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
            # This page's *start* is past the window (see
            # _past_search_window's docstring) -- it cannot contain a
            # single valid result, so return a clean, predictable "no more
            # results" instead of forwarding a caller mechanically
            # incrementing `page` into a request that could only error.
            # Validate config here too (normally _request's job) -- this
            # branch never reaches _request, so a misconfigured
            # subdomain/credential would otherwise be masked as "no
            # results" instead of surfacing as the usual config error.
            _ensure_configured()
            return _finalize_capped(
                _success(results=[], count=None, has_more=False, truncated=False),
                get_tool_max_output_length(),
            )
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
        # `query or ""`: a None query fails _require_non_blank inside the
        # try, and len(None) here would turn the error handler itself into
        # an uncaught TypeError.
        logger.error(
            f"Error searching Zendesk for query of length {len(query or '')}: {e}"
        )
        return _error_capped(str(e))


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
        return _error_capped(str(e))


@mcp.tool()
def zendesk_get_ticket(ticket_id: int | str) -> str:
    """
    Get a Zendesk ticket by id.
    ticket_id: a bare numeric id, or a full ticket URL copied from the
    Zendesk agent UI.
    """
    try:
        path_id = _resolve_path_id(ticket_id, _TICKET_URL_ID_PATTERN, "ticket_id")
        result = _request("GET", f"/tickets/{path_id}.json")
        return _build_ticket_detail_response(
            _unwrap(result, "ticket"), get_tool_max_output_length()
        )
    except Exception as e:
        logger.error(f"Error fetching Zendesk ticket {ticket_id}: {e}")
        return _error_capped(str(e))


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
    priority: optional, one of "low", "normal", "high", "urgent"
    (case-insensitive) -- any other value is rejected locally.
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
        requester_email = _blank_to_none(requester_email)
        requester_name = _blank_to_none(requester_name)
        if requester_email:
            requester: dict[str, Any] = {"email": requester_email}
            if requester_name:
                requester["name"] = requester_name
            ticket["requester"] = requester
        elif requester_name:
            raise ValueError("requester_name requires requester_email")
        priority = _blank_to_none(priority)
        if priority:
            ticket["priority"] = _require_one_of(
                priority, _TICKET_PRIORITIES, "priority"
            )
        tags_value = _resolve_tags(tags)
        if tags_value is not None:
            ticket["tags"] = tags_value
        result = _request("POST", "/tickets.json", json_data={"ticket": ticket})
        return _build_ticket_detail_response(
            _unwrap(result, "ticket"), get_tool_max_output_length()
        )
    except Exception as e:
        logger.error(f"Error creating Zendesk ticket: {e}")
        return _error_capped(str(e))


@mcp.tool()
def zendesk_update_ticket(
    ticket_id: int | str,
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
    status: optional, one of "new", "open", "pending", "hold", "solved"
    (case-insensitive) -- "closed" is not settable directly; Zendesk
    reaches it automatically some time after "solved". A blank (empty or
    whitespace-only) string is treated the same as leaving it unset (there
    is no valid "clear the status" value); any other non-blank value is
    rejected locally.
    priority: optional, one of "low", "normal", "high", "urgent"
    (case-insensitive) -- a blank string is treated the same as leaving it
    unset, for the same reason; any other non-blank value is rejected
    locally.
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
        status = _blank_to_none(status)
        if status:
            fields["status"] = _require_one_of(status, _TICKET_STATUSES, "status")
        priority = _blank_to_none(priority)
        if priority:
            fields["priority"] = _require_one_of(
                priority, _TICKET_PRIORITIES, "priority"
            )
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
        return _build_ticket_detail_response(
            _unwrap(result, "ticket"), get_tool_max_output_length()
        )
    except Exception as e:
        logger.error(f"Error updating Zendesk ticket {ticket_id}: {e}")
        return _error_capped(str(e))


@mcp.tool()
def zendesk_delete_ticket(ticket_id: int | str) -> str:
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
        # DELETE returns no body for Zendesk to echo an integer id from, but
        # every other tool's ticket_id field is a real int. _resolve_path_id
        # doesn't itself validate that path_id is numeric (it only rejects
        # blank/whitespace-padded input) -- but the DELETE above already
        # succeeded against a real Zendesk ticket, and Zendesk's own
        # /tickets/{id}.json route 404s on a non-numeric id before this line
        # is ever reached, so int() here can't raise in practice. It's also
        # inside this same try block, so even a hypothetical failure here
        # would route through the normal error path below, not crash.
        return _success(ticket_id=int(path_id))
    except Exception as e:
        logger.error(f"Error deleting Zendesk ticket {ticket_id}: {e}")
        return _error_capped(str(e))


@mcp.tool()
def zendesk_list_ticket_comments(
    ticket_id: int | str, limit: int = 25, after_cursor: str | None = None
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
        return _error_capped(str(e))


@mcp.tool()
def zendesk_reply_to_ticket(ticket_id: int | str, body: str) -> str:
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
        return _error_capped(str(e))


@mcp.tool()
def zendesk_add_internal_note(ticket_id: int | str, body: str) -> str:
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
        return _error_capped(str(e))


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
        return _error_capped(str(e))


@mcp.tool()
def zendesk_get_user(user_id: int | str) -> str:
    """
    Get a Zendesk user by id.
    user_id: a bare numeric id, or a full user URL copied from the Zendesk
    agent UI.
    """
    try:
        path_id = _resolve_path_id(user_id, _USER_URL_ID_PATTERN, "user_id")
        result = _request("GET", f"/users/{path_id}.json")
        response = _success(user=_user_summary(_unwrap(result, "user")))
        return _finalize_capped(response, get_tool_max_output_length())
    except Exception as e:
        logger.error(f"Error fetching Zendesk user {user_id}: {e}")
        return _error_capped(str(e))


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
            return _finalize_capped(
                _success(users=[], has_more=False, truncated=False),
                get_tool_max_output_length(),
            )
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
            f"Error searching Zendesk users for query of length {len(query or '')}: {e}"
        )
        return _error_capped(str(e))


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
        return _error_capped(str(e))


@mcp.tool()
def zendesk_get_organization(organization_id: int | str) -> str:
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
        response = _success(
            organization=_organization_summary(_unwrap(result, "organization"))
        )
        return _finalize_capped(response, get_tool_max_output_length())
    except Exception as e:
        logger.error(f"Error fetching Zendesk organization {organization_id}: {e}")
        return _error_capped(str(e))


@mcp.tool()
def zendesk_search_articles(query: str, limit: int = 25, page: int = 1) -> str:
    """
    Search Help Center (Zendesk Knowledge) articles by keyword, e.g.
    "reset password". Returns each article's title, URL, locale, section,
    and the matched snippet; use zendesk_get_article for the full body.
    limit: max articles to return (default 25, hard cap 100).
    page: 1-based page number; pass the previous page + 1 to continue. If a
    call is truncated for size (`truncated: true` in the response,
    `has_more` forced true), re-run the search with a smaller `limit` from
    `page=1` -- see zendesk_search's identical note for why.
    """
    try:
        query = _require_non_blank(query, "query")
        max_results = _clamp_limit(limit)
        page = max(1, page)
        if _past_search_window(page, max_results, _MAX_ARTICLE_SEARCH_RESULT_WINDOW):
            # See zendesk_search's identical guard.
            _ensure_configured()
            return _finalize_capped(
                _success(results=[], count=None, has_more=False, truncated=False),
                get_tool_max_output_length(),
            )
        return _list_offset_paginated(
            "/help_center/articles/search.json",
            "results",
            _article_summary,
            {"query": query, "per_page": max_results, "page": page},
            max_results,
            extra_fields_fn=lambda result: {"count": result.get("count")},
        )
    except Exception as e:
        # Logged length-bounded like zendesk_search: a query can carry
        # end-user text.
        logger.error(
            "Error searching Zendesk Help Center articles for query of length "
            f"{len(query or '')}: {e}"
        )
        return _error_capped(str(e))


@mcp.tool()
def zendesk_get_article(article_id: int | str) -> str:
    """
    Get a Help Center article by id, including its HTML body. A very long
    body is cut to fit the output limit and marked `body_truncated: true`;
    the cut can fall inside an HTML tag.
    article_id: a bare numeric id, or the article's Help Center or
    Knowledge URL.
    """
    try:
        path_id = _resolve_path_id(article_id, _ARTICLE_URL_ID_PATTERN, "article_id")
        result = _request("GET", f"/help_center/articles/{path_id}.json")
        return _build_detail_response(
            "article",
            _article_detail(_unwrap(result, "article")),
            "body",
            get_tool_max_output_length(),
        )
    except Exception as e:
        logger.error(f"Error fetching Zendesk Help Center article {article_id}: {e}")
        return _error_capped(str(e))


@mcp.tool()
def zendesk_list_macros(limit: int = 25, after_cursor: str | None = None) -> str:
    """
    List the active macros the connected agent can use, with each macro's
    title and description. Use zendesk_get_macro for what a macro does.
    limit: max macros to return (default 25, hard cap 100).
    after_cursor: pass the previous call's own after_cursor to fetch the
    next page; omit for the first page.
    """
    try:
        return _list_cursor_paginated(
            "/macros/active.json", "macros", _macro_summary, limit, after_cursor
        )
    except Exception as e:
        logger.error(f"Error listing Zendesk macros: {e}")
        return _error_capped(str(e))


@mcp.tool()
def zendesk_get_macro(macro_id: int | str) -> str:
    """
    Get a macro by id, including its actions (the ticket fields it sets and
    the comment it adds) and which groups or users it is restricted to.
    macro_id: a bare numeric id (from zendesk_list_macros), or the macro's
    Admin Center URL.
    """
    try:
        path_id = _resolve_path_id(macro_id, _MACRO_URL_ID_PATTERN, "macro_id")
        result = _request("GET", f"/macros/{path_id}.json")
        # Not shrunk like an article body: a macro with only some of its
        # actions would misdescribe what applying it does.
        response = _success(macro=_macro_detail(_unwrap(result, "macro")))
        return _finalize_capped(response, get_tool_max_output_length())
    except Exception as e:
        logger.error(f"Error fetching Zendesk macro {macro_id}: {e}")
        return _error_capped(str(e))


@mcp.tool()
def zendesk_list_side_conversations(
    ticket_id: int | str, limit: int = 25, page: int = 1
) -> str:
    """
    List the side conversations on a ticket: the separate threads agents
    start with people outside the ticket (other teams, vendors). Returns
    each one's subject, state, preview text, and participants; use
    zendesk_get_side_conversation for its messages.
    ticket_id: a bare numeric id, or a full ticket URL copied from the
    Zendesk agent UI. Zendesk answers HTTP 403, not 404, for a ticket that
    doesn't exist.
    limit: max side conversations to return (default 25, hard cap 100).
    page: 1-based page number; pass the previous page + 1 to continue. If a
    call is truncated for size (`truncated: true`, `has_more` forced true),
    re-run with a smaller `limit` from `page=1` instead -- see
    zendesk_search's identical note for why.
    """
    try:
        path_id = _resolve_path_id(ticket_id, _TICKET_URL_ID_PATTERN, "ticket_id")
        max_results = _clamp_limit(limit)
        return _list_offset_paginated(
            f"/tickets/{path_id}/side_conversations.json",
            "side_conversations",
            _side_conversation_summary,
            # Offset paging only: this endpoint rejects page[size] with
            # "page must be an integer".
            {"per_page": max_results, "page": max(1, page)},
            max_results,
            extra_fields_fn=lambda result: {"count": result.get("count")},
        )
    except Exception as e:
        logger.error(
            f"Error listing side conversations on Zendesk ticket {ticket_id}: {e}"
        )
        return _error_capped(str(e))


@mcp.tool()
def zendesk_get_side_conversation(
    ticket_id: int | str,
    side_conversation_id: str,
    limit: int = 25,
    page: int = 1,
) -> str:
    """
    Get one side conversation on a ticket with a page of its history: each
    message's sender, recipients, subject, body (plain text, or the HTML
    body when the plain text is empty; `body_is_html` is true in that
    case), and attachments, plus state changes. If the response is too
    large, events are dropped (`truncated: true`, `has_more` forced true)
    while the side conversation itself is kept; re-run with a smaller
    `limit` from `page=1` to read them.
    ticket_id: a bare numeric id, or a full ticket URL copied from the
    Zendesk agent UI. Zendesk answers HTTP 403, not 404, for a ticket that
    doesn't exist.
    side_conversation_id: the `id` from zendesk_list_side_conversations (a
    UUID string, not a number).
    limit: max history events to return (default 25, hard cap 100).
    page: 1-based page number of the history; pass the previous page + 1 to
    continue.
    """
    try:
        ticket_path_id = _resolve_path_id(
            ticket_id, _TICKET_URL_ID_PATTERN, "ticket_id"
        )
        conversation_path_id = url_path_id(side_conversation_id, "side_conversation_id")
        path = f"/tickets/{ticket_path_id}/side_conversations/{conversation_path_id}"
        conversation = _unwrap(_request("GET", f"{path}.json"), "side_conversation")
        detail = {
            **_side_conversation_summary(conversation),
            "state_updated_at": conversation.get("state_updated_at"),
        }
        max_results = _clamp_limit(limit)
        return _list_offset_paginated(
            f"{path}/events.json",
            "events",
            _side_conversation_event_summary,
            # Offset paging only: like the side conversation list, this
            # endpoint rejects page[size] with "page must be an integer".
            {"per_page": max_results, "page": max(1, page)},
            max_results,
            # Fixed fields ride alongside the page and are never trimmed;
            # only the events list shrinks to fit the output cap.
            extra_fields_fn=lambda result: {
                "side_conversation": detail,
                "count": result.get("count"),
            },
        )
    except Exception as e:
        logger.error(
            f"Error fetching side conversation {side_conversation_id} on Zendesk "
            f"ticket {ticket_id}: {e}"
        )
        return _error_capped(str(e))


@mcp.tool()
def zendesk_apply_macro(
    ticket_id: int | str,
    macro_id: int | str,
    dry_run: bool = True,
    preview_token: str | None = None,
) -> str:
    """
    Apply a macro to a ticket the way the Zendesk agent UI does: set the
    fields the macro sets and add its comment, with placeholders such as
    {{ticket.requester.first_name}} filled in. Two steps: first preview
    (the default, `dry_run: true`), which saves nothing and returns the
    changes and a `preview_token`; then save with dry_run=false and that
    same preview_token. The save is refused, with nothing saved, if the
    ticket or the macro changed since that preview -- preview again and
    re-confirm. A save is also refused if the macro renders differently
    than in the preview (e.g. the requester's details changed).
    Only macros whose actions set status, priority, assignee, group,
    tags, or add a comment can be applied. A macro with any other action
    (for example a side conversation, CCs, a custom field, or a custom
    ticket status) is refused before anything is saved, listing the
    actions it can't apply, and so is an inactive macro.
    The comment is public unless the macro marks it internal. In the
    reported changes, `body_is_html` marks a comment body that is HTML.
    ticket_id: a bare numeric id, or a full ticket URL copied from the
    Zendesk agent UI.
    macro_id: a bare numeric id (from zendesk_list_macros), or the macro's
    Admin Center URL.
    preview_token: required to save; the `preview_token` from a preview of
    this macro on this ticket. It binds the exact write that preview showed
    (ticket state, macro version, and the rendered changes); it is not a
    record of a human approval, so show the preview to the user before
    saving.
    """
    try:
        if isinstance(preview_token, str):
            preview_token = preview_token.strip()
        if not dry_run and not preview_token:
            raise ValueError(
                "to save, pass the preview_token from a preview (dry_run=true) "
                "of this macro on this ticket; nothing was saved"
            )
        ticket_path_id = _resolve_path_id(
            ticket_id, _TICKET_URL_ID_PATTERN, "ticket_id"
        )
        macro_path_id = _resolve_path_id(macro_id, _MACRO_URL_ID_PATTERN, "macro_id")
        macro = _unwrap(_request("GET", f"/macros/{macro_path_id}.json"), "macro")
        if macro.get("active") is False:
            raise ValueError("this macro is inactive; nothing was saved")
        actions = [a for a in macro.get("actions") or [] if isinstance(a, dict)]
        fields = {str(action.get("field")) for action in actions}
        unsupported = sorted(fields - _ALLOWED_MACRO_ACTIONS)
        if unsupported:
            raise ValueError(
                "this macro has actions this connector can't apply: "
                f"{', '.join(unsupported)}; nothing was saved. Apply it in "
                "the Zendesk agent UI instead."
            )
        public = _macro_comment_is_public(actions)
        if not fields - {_MACRO_COMMENT_MODE_ACTION}:
            raise ValueError("this macro has no actions to apply")

        # The ticket-scoped preview, not GET /macros/{id}/apply: only this
        # one renders placeholders, and it carries the final tag set and the
        # updated_at the save is pinned to.
        preview = _request(
            "GET", f"/tickets/{ticket_path_id}/macros/{macro_path_id}/apply.json"
        )
        preview_ticket = (
            (preview.get("result") or {}).get("ticket")
            if isinstance(preview, dict)
            else None
        )
        if not isinstance(preview_ticket, dict):
            raise RuntimeError("Zendesk returned no preview of this macro's result")

        changes: dict[str, Any] = {}
        update: dict[str, Any] = {}
        for field in _MACRO_FIELD_ACTIONS:
            if field in fields:
                if field not in preview_ticket:
                    raise RuntimeError(
                        f"Zendesk's preview of this macro has no {field}; "
                        "nothing was saved"
                    )
                changes[field] = update[field] = preview_ticket.get(field)
        if fields & _MACRO_TAG_ACTIONS:
            # The whole final set: a single-ticket update ignores
            # additional_tags/remove_tags. safe_update below keeps this from
            # overwriting tags someone changed after the preview.
            tags = preview_ticket.get("tags")
            if not isinstance(tags, list):
                raise RuntimeError(
                    "Zendesk's preview of this macro has no tag list; nothing was saved"
                )
            changes["tags"] = update["tags"] = tags
        if fields & _MACRO_COMMENT_ACTIONS:
            comment = preview_ticket.get("comment") or {}
            html_body = comment.get("html_body")
            body = comment.get("body")
            if html_body:
                update["comment"] = {"html_body": html_body, "public": public}
            elif body:
                update["comment"] = {"body": body, "public": public}
            else:
                raise RuntimeError(
                    "Zendesk's preview of this macro has an empty comment; "
                    "nothing was saved"
                )
            changes["comment"] = {
                "body": html_body or body,
                "body_is_html": bool(html_body),
                "public": public,
            }

        preview_stamp = preview_ticket.get("updated_at")
        if not preview_stamp:
            raise RuntimeError(
                "Zendesk's preview has no updated_at to pin a save to; nothing "
                "was saved"
            )
        # Binds a save to exactly what was previewed: the ticket and its
        # state, and the macro and its version. The digest also binds the
        # exact write (rendered comment, its visibility, fields and tags), so
        # a macro edited mid-call or a placeholder rendering differently
        # can't change what is saved.
        write_digest = hashlib.sha256(
            json.dumps(update, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()[:32]
        token = "|".join(
            str(part)
            for part in (
                preview_ticket.get("id"),
                preview_stamp,
                macro.get("id"),
                macro.get("updated_at"),
                write_digest,
            )
        )
        payload: dict[str, Any] = {
            "dry_run": dry_run,
            "ticket_id": preview_ticket.get("id"),
            "macro": {"id": macro.get("id"), "title": macro.get("title")},
            "changes": changes,
        }
        if dry_run:
            payload["preview_token"] = token
            return _build_apply_macro_response(payload)

        if token != preview_token:
            raise ValueError(
                "the ticket or the macro changed since the preview this "
                "preview_token came from, or it is from another ticket or "
                "macro; nothing was saved. Preview again and confirm the new "
                "changes."
            )
        # safe_update pinned to the approved preview: Zendesk answers 409 and
        # saves nothing if the ticket changes between this check and the
        # save. It also makes a retry after a successful save fail instead of
        # posting the comment twice, since the save itself moves updated_at.
        try:
            result = _request(
                "PUT",
                f"/tickets/{ticket_path_id}.json",
                json_data={
                    "ticket": {
                        **update,
                        "safe_update": True,
                        "updated_stamp": preview_stamp,
                    }
                },
            )
        except _OutcomeUnknownError as exc:
            raise RuntimeError(
                f"{exc}. The save may have been applied: check the ticket with "
                "zendesk_get_ticket and zendesk_list_ticket_comments before "
                "applying this macro again."
            ) from exc
        ticket = _unwrap(result, "ticket")
        payload["ticket"] = (
            _ticket_summary(ticket) if isinstance(ticket, dict) else ticket
        )
        if "comment" in update:
            event = _find_comment_event(result, public)
            # The body is already in changes.comment; don't report it twice.
            payload["comment"] = (
                {k: v for k, v in event.items() if k != "body"} if event else None
            )
        return _build_apply_macro_response(payload)
    except Exception as e:
        logger.error(
            f"Error applying Zendesk macro {macro_id} to ticket {ticket_id}: {e}"
        )
        return _error_capped(str(e))


if __name__ == "__main__":
    mcp.run()
