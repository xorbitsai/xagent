import hashlib
import json
import logging
import re
import socket
import time
from collections.abc import Callable
from os import environ
from typing import Any, NoReturn

import requests
from mcp.server.fastmcp import FastMCP

from ....config import get_tool_max_output_length
from ....core.utils.security import (
    PrivateNetworkHostError,
    redact_sensitive_text,
    reject_private_network_host,
)
from ...utils.graphql_errors import graphql_errors_message, truncate_error_text
from .utils import clamp_limit, setup_proxy_env, success_with_capped_dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("shopify-mcp")

# Ensure standard proxy environment variables are set to prevent hanging requests
setup_proxy_env()

mcp = FastMCP("shopify-mcp")

DEFAULT_TIMEOUT_SECONDS = 30
MAX_LIMIT = 100
# Shopify releases a new Admin API version quarterly (YYYY-01/04/07/10) and
# supports each one for about a year; pinned so a version bump on Shopify's
# side can't silently change response shapes underneath these tools.
# Re-check this against shopify.dev/docs/api/admin-graphql whenever this
# version is scheduled for retirement, and bump deliberately.
SHOPIFY_API_VERSION = "2026-07"
MAX_RETRY_AFTER_SECONDS = 30
SHOPIFY_REQUIRED_ADMIN_SCOPES = frozenset(
    {"write_products", "write_orders", "read_customers"}
)
SHOPIFY_OPTIONAL_ADMIN_SCOPES = frozenset({"read_all_orders"})
# The host currently starts a fresh stdio process for every tool invocation, so
# this cache only deduplicates checks within one invocation. Keeping it keyed by
# credential still makes it safe if stdio sessions are pooled in the future.
_scope_cache: tuple[str, frozenset[str]] | None = None

_READ_CAPABILITY_SCOPE_GROUPS = {
    "products": frozenset({"read_products", "write_products"}),
    "orders": frozenset({"read_orders", "write_orders"}),
    "customers": frozenset({"read_customers", "write_customers"}),
}
_WRITE_CAPABILITY_SCOPES = frozenset({"write_products", "write_orders"})

# Only a DNS *label* (no dots, scheme, port, or slashes) is ever accepted, so
# the string itself can never name a host outside "*.myshopify.com" -- but a
# legitimate hostname can still be rebound by DNS to a private/internal
# address at request time (orthogonal to who chose the hostname string), so
# _graphql_url() below still resolves and checks every address, same
# defense-in-depth posthog.py's _base_url() applies to its own two
# hardcoded-enum hostnames.
_SUBDOMAIN_PATTERN = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")

# [0-9], not \d -- \d matches any Unicode decimal digit in a str pattern
# (e.g. Arabic-Indic), which would let a lookalike id through this format
# check even though Shopify's own numeric ids are always ASCII.
_GID_PATTERN = re.compile(r"^gid://shopify/[A-Za-z]+/[0-9]+$")

_PRODUCT_STATUSES = frozenset({"ACTIVE", "ARCHIVED", "DRAFT", "UNLISTED"})


def _success(*, _errors: list[Any] | None = None, **payload: Any) -> str:
    body: dict[str, Any] = {"status": "success", **payload}
    if _errors:
        # A genuine partial GraphQL success (one sub-field failed, others
        # resolved) -- surface it in the result instead of only the server
        # log, matching linear.py's identical warnings contract. Routed
        # through _errors_detail (not graphql_errors_message directly) for
        # the same reason every other consumer of a GraphQL response's
        # top-level "errors" value in this module is: Shopify doesn't
        # always send a list there.
        body["warnings"] = [_errors_detail(_errors)]
    return json.dumps(body, ensure_ascii=False)


def _safe_text(value: Any) -> str:
    # Redact the complete value before truncating. Reversing this order can
    # leave a recognizable token prefix when the boundary splits the token.
    text = str(value)
    token = environ.get("SHOPIFY_ACCESS_TOKEN", "")
    if token:
        text = text.replace(token, "[REDACTED]")
    return truncate_error_text(redact_sensitive_text(text))


def _log_failure(operation: str, exc: BaseException) -> None:
    # Do not use exc_info here: an arbitrary requests exception may embed the
    # credential-bearing request/headers in its own representation.
    logger.error(
        "Shopify %s failed (%s): %s",
        operation,
        type(exc).__name__,
        _safe_text(exc),
    )


def _error(message: str) -> str:
    return json.dumps(
        {"status": "error", "message": _safe_text(message)}, ensure_ascii=False
    )


def _indeterminate(message: str) -> str:
    """Return an explicit unsafe-to-retry mutation outcome."""
    return json.dumps(
        {
            "status": "indeterminate",
            "retryable": False,
            "mutation_may_have_completed": True,
            "message": _safe_text(message),
        },
        ensure_ascii=False,
    )


def _mutation_indeterminate(message: str) -> str:
    """Log and return an unsafe-to-retry mutation outcome."""
    safe_message = _safe_text(message)
    logger.error("Shopify mutation outcome indeterminate: %s", safe_message)
    return _indeterminate(safe_message)


def _headers() -> dict[str, str]:
    access_token = environ.get("SHOPIFY_ACCESS_TOKEN")
    if not access_token:
        raise ValueError("SHOPIFY_ACCESS_TOKEN environment variable is missing")
    if (
        access_token != access_token.strip()
        or not 16 <= len(access_token) <= 512
        or any(ord(char) < 33 or ord(char) > 126 for char in access_token)
    ):
        raise ValueError(
            "SHOPIFY_ACCESS_TOKEN must be 16-512 printable ASCII characters "
            "without surrounding whitespace"
        )
    return {
        "X-Shopify-Access-Token": access_token,
        "Content-Type": "application/json",
    }


def _graphql_url() -> str:
    subdomain = environ.get("SHOPIFY_STORE_DOMAIN", "").strip().lower()
    if not subdomain:
        raise ValueError("SHOPIFY_STORE_DOMAIN environment variable is missing")
    if not _SUBDOMAIN_PATTERN.match(subdomain):
        raise ValueError(
            "SHOPIFY_STORE_DOMAIN must be a single DNS label (letters, digits, "
            "and hyphens only, no leading/trailing hyphen) -- pass just the "
            "store name, e.g. 'acme' for acme.myshopify.com, not a full URL"
        )
    hostname = f"{subdomain}.myshopify.com"
    try:
        resolved = socket.getaddrinfo(
            hostname, 443, socket.AF_UNSPEC, socket.SOCK_STREAM
        )
        for *_, sockaddr in resolved:
            reject_private_network_host(str(sockaddr[0]))
    except PrivateNetworkHostError as exc:
        raise ValueError(f"SHOPIFY_STORE_DOMAIN is not allowed: {exc}") from exc
    except OSError as exc:
        raise ValueError(f"Shopify host could not be resolved: {exc}") from exc
    return f"https://{hostname}/admin/api/{SHOPIFY_API_VERSION}/graphql.json"


def _scopes_url() -> str:
    return _graphql_url().split("/admin/api/", 1)[0] + "/admin/oauth/access_scopes.json"


def _require_non_blank(value: str, field_name: str) -> str:
    if not value or not value.strip():
        raise ValueError(f"{field_name} must not be blank")
    return value


def _gid(resource: str, value: str) -> str:
    """Normalize a caller-supplied id to Shopify's global id (gid) form.

    Every Shopify Admin GraphQL id argument is typed `ID!` and expects the
    full "gid://shopify/<Resource>/<numeric id>" string, but a caller (or an
    LLM copying an id from a list result) is just as likely to pass the
    bare numeric id -- accept either rather than making every tool's
    docstring explain the gid format. A full gid is only accepted for the
    matching resource type -- e.g. passing an Order's gid to a tool that
    expects a product_id is rejected here instead of being sent to Shopify
    as-is, which would otherwise either 404 or (worse, since ids aren't
    scoped per-type in this check) resolve to the wrong object type.
    """
    text = str(value).strip()
    if text.startswith("gid://shopify/"):
        if _GID_PATTERN.match(text) and text.split("/")[-2] == resource:
            return text
        raise ValueError(
            f"{resource.lower()}_id must be a gid://shopify/{resource}/... "
            f"string (or a bare numeric id), got {value!r}"
        )
    # str.isdigit() also accepts non-ASCII Unicode decimal digits (e.g.
    # Arabic-Indic "١٢٣"), which Shopify's numeric id would never actually
    # contain -- a stricter ASCII-only check here means a lookalike value
    # gets this function's clear local error instead of an opaque failure
    # from Shopify after being forwarded as-is.
    if text.isascii() and text.isdigit():
        return f"gid://shopify/{resource}/{text}"
    raise ValueError(
        f"{resource.lower()}_id must be numeric or a gid://shopify/{resource}/... "
        f"string, got {value!r}"
    )


def _user_errors_message(user_errors: list[dict[str, Any]]) -> str:
    """Join a mutation's userErrors array into one message.

    Every write mutation in this module returns `userErrors { field message
    }` as its primary error channel (a non-empty list means the write did
    not happen, even on an otherwise-200 GraphQL response) -- `field` is a
    path array (e.g. ["title"]) for a nested input, not a plain string.
    """
    parts = []
    for err in user_errors:
        field = err.get("field")
        # field is a path array (e.g. ["variants", 0, "price"]) that can mix
        # strings with integer array indices -- ".".join() requires every
        # element to already be a str, so an index entry raises TypeError
        # without the str() conversion.
        field_path = ".".join(map(str, field)) if isinstance(field, list) else field
        message = err.get("message") or "unknown error"
        parts.append(f"{field_path}: {message}" if field_path else message)
    return "; ".join(parts) if parts else "Shopify reported a validation error"


def _errors_detail(errors_field: Any) -> str:
    """Render a GraphQL response's top-level "errors" value as text.

    Per the GraphQL spec this is always a list of error objects, but
    Shopify's own auth-failure responses (e.g. a 401 for an invalid access
    token) put a plain string here instead -- "[API] Invalid API key or
    access token (...)" -- and a malformed/non-conforming backend could in
    principle put a bare dict. graphql_errors_message assumes a list and
    iterates whatever it's given: over a str that walks it one character at
    a time (producing a mangled "a; p; i" message), and over a dict that
    walks its keys, silently dropping the actual diagnostic text in its
    values. Both are handled here before ever reaching that helper.

    The result is truncated and redacted here, once, rather than leaving
    every call site responsible for remembering to do both -- this is the
    single choke point every top-level "errors" value passes through
    before becoming user- or log-facing text (a raised RuntimeError, a
    logged warning, or a tool's "warnings"/error message), so a caller
    that forgot either step would otherwise let an unbounded or
    credential-bearing error body straight through, same class of issue
    already fixed for the raw-response-body fallback text elsewhere in
    this module.
    """
    if isinstance(errors_field, str):
        text = errors_field
    elif isinstance(errors_field, list):
        text = graphql_errors_message(errors_field)
    else:
        text = str(errors_field)
    return _safe_text(text)


def _split_tags(tags: str) -> list[str]:
    return [t.strip() for t in tags.split(",") if t.strip()]


def _throttle_wait_seconds(status_code: int, payload: Any) -> float | None:
    """Return how long to wait before retrying, or None if this response
    doesn't signal throttling.

    Shopify's GraphQL cost-based throttling can surface as either an HTTP
    429 or an HTTP 200 whose body carries a "Throttled" error (the request
    cost more "leaky bucket" points than were available) -- checked
    defensively for both shapes since Shopify's own docs and real-world
    responses aren't fully consistent on which one to expect. `payload` is
    the response body already parsed once by the caller (or None if it
    wasn't valid JSON) -- shared with `_graphql`'s own data-extraction
    parse rather than decoding the same body a second time.
    """
    throttled = status_code == 429
    if not throttled and isinstance(payload, dict):
        raw_errors = payload.get("errors")
        errors = raw_errors if isinstance(raw_errors, list) else []
        for entry in errors:
            if not isinstance(entry, dict):
                continue
            if str(entry.get("message", "")).strip().lower() == "throttled":
                throttled = True
                break
            extensions = entry.get("extensions")
            if isinstance(extensions, dict) and extensions.get("code") == "THROTTLED":
                throttled = True
                break
    if not throttled:
        return None

    # Shopify's own rate-limit docs recommend a flat one-second backoff;
    # when extensions.cost.throttleStatus is present, compute a more
    # precise wait (how long until enough "bucket" capacity restores to
    # cover the query that was just rejected) instead of guessing.
    if isinstance(payload, dict):
        extensions = payload.get("extensions")
        cost = extensions.get("cost") if isinstance(extensions, dict) else None
        throttle_status = cost.get("throttleStatus") if isinstance(cost, dict) else None
        if not isinstance(cost, dict) or not isinstance(throttle_status, dict):
            return 1.0
        requested = cost.get("requestedQueryCost")
        available = throttle_status.get("currentlyAvailable")
        restore_rate = throttle_status.get("restoreRate")
        if (
            isinstance(requested, (int, float))
            and isinstance(available, (int, float))
            and isinstance(restore_rate, (int, float))
            and restore_rate > 0
        ):
            return max(1.0, (requested - available) / restore_rate)
    return 1.0


class MutationOutcomeIndeterminate(RuntimeError):
    """The request may have reached Shopify, so retrying could duplicate a write."""


def _raise_response_shape(message: str, *, mutation: bool) -> NoReturn:
    if mutation:
        raise MutationOutcomeIndeterminate(message)
    raise RuntimeError(message)


def _granted_admin_scopes() -> set[str]:
    """Read the scopes granted to the current custom-app token."""
    try:
        response = requests.get(
            _scopes_url(),
            headers=_headers(),
            timeout=DEFAULT_TIMEOUT_SECONDS,
            allow_redirects=False,
        )
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Could not validate Shopify Admin API scopes: {_safe_text(exc)}"
        ) from exc
    if 300 <= response.status_code < 400:
        raise RuntimeError(
            "Shopify scope validation returned an unexpected redirect "
            f"(HTTP {response.status_code})"
        )
    if response.status_code >= 400:
        raise RuntimeError(
            f"Shopify scope validation failed (HTTP {response.status_code})"
        )
    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError("Shopify scope validation returned non-JSON data") from exc
    entries = payload.get("access_scopes") if isinstance(payload, dict) else None
    if not isinstance(entries, list):
        raise RuntimeError("Shopify scope validation returned an invalid response")
    return {
        entry["handle"]
        for entry in entries
        if isinstance(entry, dict) and isinstance(entry.get("handle"), str)
    }


def _require_admin_scopes(*accepted_scopes: str) -> None:
    """Require at least one equivalent scope, caching by credential identity."""
    global _scope_cache

    token = environ.get("SHOPIFY_ACCESS_TOKEN", "")
    domain = environ.get("SHOPIFY_STORE_DOMAIN", "")
    cache_key = hashlib.sha256(f"{domain}\0{token}".encode()).hexdigest()
    if _scope_cache is None or _scope_cache[0] != cache_key:
        _scope_cache = (cache_key, frozenset(_granted_admin_scopes()))
    granted = _scope_cache[1]
    if not granted.intersection(accepted_scopes):
        raise ValueError(
            "Shopify token requires one of these Admin API scopes: "
            + ", ".join(accepted_scopes)
        )


@mcp.tool()
def shopify_validate_connection() -> str:
    """Validate the token and its Admin API scopes before using tools.

    A token is valid when it can read products, orders, and customers. The
    response separately reports whether every write tool is available.
    read_all_orders is optional and expands order history beyond 60 days.
    """
    try:
        granted = _granted_admin_scopes()
        missing_read_capabilities = sorted(
            resource
            for resource, alternatives in _READ_CAPABILITY_SCOPE_GROUPS.items()
            if not alternatives.intersection(granted)
        )
        if missing_read_capabilities:
            return _error(
                "Shopify token cannot read required resources: "
                + ", ".join(missing_read_capabilities)
            )
        missing_write_scopes = sorted(_WRITE_CAPABILITY_SCOPES - granted)
        return _success(
            connection_valid=True,
            read_capable=True,
            write_capable=not missing_write_scopes,
            missing_write_scopes=missing_write_scopes,
            read_scope_alternatives={
                resource: sorted(alternatives)
                for resource, alternatives in _READ_CAPABILITY_SCOPE_GROUPS.items()
            },
            optional_scopes=sorted(SHOPIFY_OPTIONAL_ADMIN_SCOPES),
            granted_optional_scopes=sorted(SHOPIFY_OPTIONAL_ADMIN_SCOPES & granted),
        )
    except Exception as exc:
        _log_failure("connection validation", exc)
        return _error(str(exc))


def _graphql(
    query: str,
    variables: dict[str, Any] | None = None,
    *,
    mutation: bool = False,
) -> tuple[dict[str, Any], list[Any]]:
    """Run one GraphQL query/mutation against this store's Admin API
    endpoint.

    Returns (data, errors). Every query/mutation in this module selects
    exactly one top-level field, so if that field comes back null alongside
    a non-empty "errors" array there is nothing usable to return. Reads fail;
    mutations retain that response so the dispatched write can be reported as
    indeterminate. If at least one top-level field is non-null, it's a genuine
    partial success (e.g. a nested
    sub-field's resolver failed): errors is returned alongside data so the
    caller can surface it as a warning rather than only logging it. Mirrors
    linear.py's `_graphql`, adapted for Shopify's cost-based throttling
    instead of Linear's rate-limit-header scheme.
    """
    # Resolved once and reused across both attempts (not re-resolved inside
    # the loop) -- a throttled request already pays the retry's sleep cost;
    # repeating the DNS resolution + private-IP check on the retry would
    # just be duplicated work for the same store, not additional safety.
    url = _graphql_url()
    try:
        for attempt in (0, 1):
            response = requests.post(
                url,
                headers=_headers(),
                json={"query": query, "variables": variables or {}},
                timeout=DEFAULT_TIMEOUT_SECONDS,
                allow_redirects=False,
            )
            # Parsed once per attempt and carried past the loop for reuse by
            # the status-code branches below -- every prior version of this
            # function parsed the same (non-retried) response body a second
            # time in the success/error branch, doubling the JSON-decode
            # cost of every call on the common, non-throttled path.
            try:
                payload = response.json()
            except ValueError:
                payload = None
            if attempt == 0:
                wait_seconds = _throttle_wait_seconds(response.status_code, payload)
                if wait_seconds is not None and wait_seconds <= MAX_RETRY_AFTER_SECONDS:
                    time.sleep(wait_seconds)
                    continue
            break
    except requests.RequestException as exc:
        message = _safe_text(exc)
        if mutation:
            raise MutationOutcomeIndeterminate(
                "Shopify did not provide a mutation response. The write may have "
                "completed; verify store state before any manual retry. "
                f"Transport detail: {message}"
            ) from exc
        raise RuntimeError(message) from exc

    if 300 <= response.status_code < 400:
        raise RuntimeError(
            f"Shopify returned an unexpected redirect (HTTP {response.status_code}); "
            "refusing to follow it with credentials attached"
        )
    if response.status_code >= 400:
        detail: str | None = None
        if isinstance(payload, dict) and payload.get("errors"):
            detail = _errors_detail(payload["errors"])
        if detail is None:
            detail = truncate_error_text(response.text.strip())
        message = (
            f"Shopify API error (status {response.status_code}): {_safe_text(detail)}"
        )
        if mutation and response.status_code >= 500:
            raise MutationOutcomeIndeterminate(
                message
                + "; Shopify returned a server error after receiving the mutation"
            )
        raise RuntimeError(message)

    if payload is None:
        detail = _safe_text(response.text.strip())
        _raise_response_shape(
            f"Shopify API returned a non-JSON response: {detail}",
            mutation=mutation,
        )
    if not isinstance(payload, dict):
        _raise_response_shape(
            "Shopify API returned an unexpected (non-object) response body",
            mutation=mutation,
        )

    data = payload.get("data")
    if not isinstance(data, dict):
        _raise_response_shape(
            "Shopify API response has an invalid top-level data field",
            mutation=mutation,
        )
    if len(data) > 1:
        _raise_response_shape(
            f"Shopify API response had {len(data)} top-level fields "
            f"({sorted(data)}), but this module's error handling assumes exactly one",
            mutation=mutation,
        )
    raw_errors = payload.get("errors", [])
    if not isinstance(raw_errors, list) or not all(
        isinstance(entry, dict) for entry in raw_errors
    ):
        _raise_response_shape(
            "Shopify API response has an invalid top-level errors field",
            mutation=mutation,
        )
    errors = raw_errors
    if errors:
        message = _errors_detail(errors)
        if all(value is None for value in data.values()):
            if mutation:
                return data, errors
            raise RuntimeError(message)
        logger.warning(
            f"Shopify GraphQL partial error (data still returned): {message}"
        )
    return data, errors


def _extract_connection(
    data: dict[str, Any],
    field_name: str,
    summary_fn: Callable[[dict[str, Any]], dict[str, Any]],
) -> tuple[list[tuple[dict[str, Any], str | None]], bool, str | None]:
    """Return summarized nodes paired with their own Relay edge cursors."""
    connection = data.get(field_name)
    if not isinstance(connection, dict):
        raise RuntimeError(
            f"Shopify API response has an invalid {field_name} connection"
        )
    edges = connection.get("edges")
    page_info = connection.get("pageInfo")
    if not isinstance(page_info, dict):
        raise RuntimeError(
            f"Shopify API response has an invalid {field_name}.pageInfo object"
        )
    end_cursor = page_info.get("endCursor")
    has_next_page = page_info.get("hasNextPage")
    if not isinstance(has_next_page, bool):
        raise RuntimeError(
            f"Shopify API response has an invalid {field_name}.pageInfo.hasNextPage"
        )
    has_more = has_next_page
    if has_more and not end_cursor:
        raise RuntimeError(
            "Shopify returned hasNextPage=true without a continuation cursor; "
            "pagination stopped to prevent an infinite loop"
        )
    if end_cursor is not None and not isinstance(end_cursor, str):
        raise RuntimeError(
            f"Shopify API response has an invalid {field_name}.pageInfo.endCursor"
        )

    # Some mocked/legacy responses use nodes instead of the requested edges.
    # Support that shape only when nodes is itself a valid list; without edge
    # cursors, local output truncation will fail closed rather than guess one.
    if edges is None:
        nodes = connection.get("nodes")
        if not isinstance(nodes, list):
            raise RuntimeError(
                f"Shopify API response has an invalid {field_name}.nodes list"
            )
        if not all(isinstance(node, dict) for node in nodes):
            raise RuntimeError(
                f"Shopify API response has an invalid node in {field_name}.nodes"
            )
        return [(summary_fn(node), None) for node in nodes], has_more, end_cursor
    if not isinstance(edges, list):
        raise RuntimeError(
            f"Shopify API response has an invalid {field_name}.edges list"
        )

    items: list[tuple[dict[str, Any], str | None]] = []
    for edge in edges:
        if not isinstance(edge, dict) or not isinstance(edge.get("node"), dict):
            continue
        cursor = edge.get("cursor")
        items.append(
            (summary_fn(edge["node"]), cursor if isinstance(cursor, str) else None)
        )
    return items, has_more, end_cursor


def _list_connection(
    root_field: str,
    selection: str,
    limit: int,
    query: str,
    after: str,
    summary_fn: Callable[[dict[str, Any]], dict[str, Any]],
) -> str:
    """Shared body for every paginated "list X" tool below (products,
    orders, customers, collections) -- only the field name, per-item GraphQL
    selection, and summarizer differ between them."""
    max_results = clamp_limit(limit, max_limit=MAX_LIMIT)
    args = ["first: $first"]
    signature = ["$first: Int!"]
    variables: dict[str, Any] = {"first": max_results}
    if query:
        args.append("query: $query")
        signature.append("$query: String!")
        variables["query"] = query
    if after:
        args.append("after: $after")
        signature.append("$after: String!")
        variables["after"] = after
    data, errors = _graphql(
        f"query({', '.join(signature)}) {{ {root_field}({', '.join(args)}) {{"
        f" edges {{ cursor node {{ {selection} }} }}"
        " pageInfo { hasNextPage endCursor } } }",
        variables,
    )
    entries, has_more, next_cursor = _extract_connection(data, root_field, summary_fn)
    return _success_paginated(root_field, entries, has_more, next_cursor, after, errors)


def _success_paginated(
    field_name: str,
    entries: list[tuple[dict[str, Any], str | None]],
    has_more: bool,
    next_cursor: str | None,
    input_after: str,
    errors: list[Any],
) -> str:
    """Build a bounded page whose cursor follows the last returned item."""

    def _build(
        page: list[dict[str, Any]], more: bool, cursor: str | None, truncated: bool
    ) -> str:
        return _success(
            **{field_name: page},
            truncated=truncated,
            has_more=more,
            after_cursor=cursor,
            _errors=errors,
        )

    if has_more and (not next_cursor or next_cursor == input_after):
        return _error(
            "Shopify returned a pagination cursor that did not advance; "
            "pagination stopped to prevent an infinite loop"
        )

    items = [item for item, _cursor in entries]
    response = _build(items, has_more, next_cursor if has_more else None, False)
    max_output_length = get_tool_max_output_length()
    if len(response) <= max_output_length:
        return response

    for count in range(len(entries) - 1, 0, -1):
        cursor = entries[count - 1][1]
        if not cursor or cursor == input_after:
            continue
        response = _build(items[:count], True, cursor, True)
        if len(response) <= max_output_length:
            return response

    # A single record can itself exceed the output budget. Return its stable
    # identity plus an explicit truncation marker, and still advance by the
    # edge cursor so later records remain reachable.
    if entries:
        first_item, first_cursor = entries[0]
        if first_cursor and first_cursor != input_after:
            compact = {
                "id": str(first_item.get("id", ""))[:256] or None,
                "truncated": True,
            }
            response = _build([compact], True, first_cursor, True)
            if len(response) <= max_output_length:
                return response

    return _error(
        "Shopify page exceeds the output limit and has no safe continuation cursor"
    )


def _run_mutation(
    mutation: str,
    variables: dict[str, Any],
    mutation_field: str,
    object_key: str,
    summary_fn: Callable[[dict[str, Any]], dict[str, Any]],
) -> str:
    """Shared body for every write tool below (create/update product,
    update order): run the mutation, then apply the same
    result-missing/userErrors/success discriminator each one needs.

    A missing, null, or non-object mutation projection covers two ambiguous
    shapes: the field resolved to null, or a malformed response omitted it.
    Either way there is nothing usable to return after dispatch, so this
    reports an unsafe-to-retry indeterminate outcome instead of an empty
    success or an ordinary retryable-looking error
    (Shopify's write mutations here use an empty `userErrors` list as their
    success signal instead of Linear's boolean `success` field, but an
    absent result must still be treated as indeterminate, not vacuously
    "no errors").
    """
    try:
        data, errors = _graphql(mutation, variables, mutation=True)
    except MutationOutcomeIndeterminate as exc:
        return _mutation_indeterminate(str(exc))
    try:
        result = data.get(mutation_field)
        if not isinstance(result, dict):
            detail = f": {_errors_detail(errors)}" if errors else ""
            return _mutation_indeterminate(
                f"Shopify returned no valid {mutation_field} projection after the "
                f"mutation request{detail}"
            )
        user_errors = result.get("userErrors")
        if not isinstance(user_errors, list) or not all(
            isinstance(entry, dict) for entry in user_errors
        ):
            return _mutation_indeterminate(
                f"Shopify returned an invalid userErrors projection for {mutation_field}"
            )
        if user_errors:
            # userErrors alone can omit useful context a top-level GraphQL
            # `errors` entry carries (e.g. a query-level access-scope warning
            # attached to the same response) -- fold both in, mirroring
            # linear.py's `_mutation_failure_message`, which does the same for
            # Linear's boolean `success` discriminator.
            message = _user_errors_message(user_errors)
            if errors:
                message = f"{message} ({_errors_detail(errors)})"
            return _error(message)
        object_value = result.get(object_key)
        if not isinstance(object_value, dict):
            # userErrors is empty, but the object itself is also null -- e.g. an
            # access-scope error on one selected field null-propagated up to the
            # whole object, with the real cause only in the top-level `errors`
            # this response still carries. Reporting this as success with an
            # all-null object would hide that entirely.
            detail = f": {_errors_detail(errors)}" if errors else ""
            return _mutation_indeterminate(
                f"Shopify returned no valid {object_key} projection after "
                f"{mutation_field}{detail}"
            )
        return _success_capped(object_key, summary_fn(object_value), errors)
    except Exception as exc:
        return _mutation_indeterminate(
            f"Shopify could not validate the {mutation_field} response after "
            f"dispatch ({type(exc).__name__}): {_safe_text(exc)}"
        )


def _success_capped(field_name: str, value: dict[str, Any], errors: list[Any]) -> str:
    """Build a single-object success response, shrinking `value` if it
    doesn't fit the platform's output-size cap.

    Every list tool below goes through `_success_paginated`'s size-capping
    on the way out, but a single get/create/update result can be just as
    unbounded -- e.g. `_product_summary`'s `tags` (Shopify allows up to 250)
    or `_order_summary`'s `note` (up to ~5000 chars) -- and was previously
    returned via a bare `_success()` call with no cap at all, so an
    oversized single object hit the same hard-truncated-into-broken-JSON
    failure mode the pagination helper exists to prevent. Reuses
    `success_with_capped_dict` (utils.py) for the actual shrinking rather
    than reimplementing its halving logic locally -- unlike the pagination
    case, there's no cursor to invalidate here, so the generic dict-capping
    helper's usual contract applies unmodified.
    """
    response = _success(**{field_name: value}, _errors=errors)
    max_output_length = get_tool_max_output_length()
    if len(response) <= max_output_length:
        return response

    capped = success_with_capped_dict(field_name, value)
    if not errors:
        return capped
    payload = json.loads(capped)
    warning = _errors_detail(errors)
    payload["warnings"] = [warning]
    payload["warnings_truncated"] = False
    with_warnings = json.dumps(payload, ensure_ascii=False)
    if len(with_warnings) <= max_output_length:
        return with_warnings

    payload["warnings_truncated"] = True
    payload["truncated"] = True
    marker = "... [truncated]"
    low, high = 0, len(warning)
    while low < high:
        middle = (low + high + 1) // 2
        payload["warnings"] = [warning[:middle] + marker]
        if len(json.dumps(payload, ensure_ascii=False)) <= max_output_length:
            low = middle
        else:
            high = middle - 1
    payload["warnings"] = [warning[:low] + marker]
    while len(json.dumps(payload, ensure_ascii=False)) > max_output_length:
        value = payload.get(field_name)
        if not isinstance(value, dict) or not value:
            break
        keys = list(value)
        payload[field_name] = {key: value[key] for key in keys[: len(keys) // 2]}
    return json.dumps(payload, ensure_ascii=False)


def _shop_summary(shop: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": shop.get("name"),
        "domain": shop.get("myshopifyDomain"),
        "email": shop.get("email"),
        "currency": shop.get("currencyCode"),
        "timezone": shop.get("ianaTimezone"),
    }


_PRODUCT_FIELDS = (
    "id title handle status vendor productType tags totalInventory createdAt updatedAt"
)


def _product_summary(product: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": product.get("id"),
        "title": product.get("title"),
        "handle": product.get("handle"),
        "status": product.get("status"),
        "vendor": product.get("vendor"),
        "product_type": product.get("productType"),
        "tags": product.get("tags"),
        "total_inventory": product.get("totalInventory"),
        "created_at": product.get("createdAt"),
        "updated_at": product.get("updatedAt"),
    }


_ORDER_FIELDS = (
    "id name email displayFinancialStatus displayFulfillmentStatus"
    " totalPriceSet { shopMoney { amount currencyCode } } tags note createdAt"
)


def _order_summary(order: dict[str, Any]) -> dict[str, Any]:
    total_price_set = order.get("totalPriceSet")
    if not isinstance(total_price_set, dict):
        raise RuntimeError("Shopify API response has an invalid order.totalPriceSet")
    total_price = total_price_set.get("shopMoney")
    if not isinstance(total_price, dict):
        raise RuntimeError(
            "Shopify API response has an invalid order.totalPriceSet.shopMoney"
        )
    return {
        "id": order.get("id"),
        "name": order.get("name"),
        "email": order.get("email"),
        "financial_status": order.get("displayFinancialStatus"),
        "fulfillment_status": order.get("displayFulfillmentStatus"),
        "total_price": total_price.get("amount"),
        "currency": total_price.get("currencyCode"),
        "tags": order.get("tags"),
        "note": order.get("note"),
        "created_at": order.get("createdAt"),
    }


_CUSTOMER_FIELDS = (
    "id firstName lastName defaultEmailAddress { emailAddress }"
    " defaultPhoneNumber { phoneNumber } numberOfOrders tags createdAt"
)


def _customer_summary(customer: dict[str, Any]) -> dict[str, Any]:
    email_address = customer.get("defaultEmailAddress")
    if email_address is not None and not isinstance(email_address, dict):
        raise RuntimeError(
            "Shopify API response has an invalid customer.defaultEmailAddress"
        )
    phone_number = customer.get("defaultPhoneNumber")
    if phone_number is not None and not isinstance(phone_number, dict):
        raise RuntimeError(
            "Shopify API response has an invalid customer.defaultPhoneNumber"
        )
    return {
        "id": customer.get("id"),
        "first_name": customer.get("firstName"),
        "last_name": customer.get("lastName"),
        "email": email_address.get("emailAddress") if email_address else None,
        "phone": phone_number.get("phoneNumber") if phone_number else None,
        "number_of_orders": customer.get("numberOfOrders"),
        "tags": customer.get("tags"),
        "created_at": customer.get("createdAt"),
    }


_COLLECTION_FIELDS = "id title handle productsCount { count }"


def _collection_summary(collection: dict[str, Any]) -> dict[str, Any]:
    products_count = collection.get("productsCount")
    if not isinstance(products_count, dict):
        raise RuntimeError(
            "Shopify API response has an invalid collection.productsCount"
        )
    return {
        "id": collection.get("id"),
        "title": collection.get("title"),
        "handle": collection.get("handle"),
        "products_count": products_count.get("count"),
    }


@mcp.tool()
def shopify_get_shop() -> str:
    """
    Get basic info about the connected Shopify store (name, domain,
    currency, timezone). Use this to verify the connection instead of
    asking the user for their store's details.
    """
    try:
        data, errors = _graphql(
            "query { shop { name myshopifyDomain email currencyCode ianaTimezone } }"
        )
        shop = data.get("shop")
        if not isinstance(shop, dict) or not shop:
            return _error("Shopify did not return shop info")
        return _success_capped("shop", _shop_summary(shop), errors)
    except Exception as exc:
        _log_failure("shop lookup", exc)
        return _error(str(exc))


@mcp.tool()
def shopify_list_products(query: str = "", limit: int = 25, after: str = "") -> str:
    """
    List/search products.
    query: optional Shopify search-syntax filter, e.g. "status:active" or
    "title:*shirt*". Leave empty to list all products.
    limit: max products to return (default 25, hard cap 100).
    after: pass the previous call's own after_cursor to fetch the next
    page; omit for the first page.
    """
    try:
        _require_admin_scopes("read_products", "write_products")
        return _list_connection(
            "products", _PRODUCT_FIELDS, limit, query, after, _product_summary
        )
    except Exception as exc:
        _log_failure("product listing", exc)
        return _error(str(exc))


@mcp.tool()
def shopify_get_product(product_id: str) -> str:
    """
    Get a Shopify product by id (numeric id or full gid://shopify/Product/...).
    """
    try:
        _require_admin_scopes("read_products", "write_products")
        data, errors = _graphql(
            f"query($id: ID!) {{ product(id: $id) {{ {_PRODUCT_FIELDS} }} }}",
            {"id": _gid("Product", product_id)},
        )
        product = data.get("product")
        if not isinstance(product, dict) or not product:
            return _error(f"Product '{product_id}' not found")
        return _success_capped("product", _product_summary(product), errors)
    except Exception as exc:
        _log_failure("product lookup", exc)
        return _error(str(exc))


@mcp.tool()
def shopify_create_product(
    title: str,
    description: str = "",
    vendor: str = "",
    product_type: str = "",
    tags: str = "",
    status: str = "DRAFT",
) -> str:
    """
    Create a new product.
    title: the product's name.
    description: optional HTML description.
    vendor: optional supplier/brand name.
    product_type: optional category/classification.
    tags: optional comma-separated tags.
    status: one of "ACTIVE", "ARCHIVED", "DRAFT", "UNLISTED" (default
    "DRAFT" -- Shopify creates products unavailable to customers by
    default; note that "ACTIVE" alone does not add it to a sales channel,
    which this connector has no tool for).
    """
    try:
        _require_admin_scopes("write_products")
        _require_non_blank(title, "title")
        if status not in _PRODUCT_STATUSES:
            return _error(
                f"status must be one of {sorted(_PRODUCT_STATUSES)}, got {status!r}"
            )
        product_input: dict[str, Any] = {"title": title, "status": status}
        if description:
            product_input["descriptionHtml"] = description
        if vendor:
            product_input["vendor"] = vendor
        if product_type:
            product_input["productType"] = product_type
        if tags:
            product_input["tags"] = _split_tags(tags)
        return _run_mutation(
            "mutation($product: ProductCreateInput!) { productCreate(product: $product)"
            f" {{ product {{ {_PRODUCT_FIELDS} }} userErrors {{ field message }} }} }}",
            {"product": product_input},
            "productCreate",
            "product",
            _product_summary,
        )
    except Exception as exc:
        _log_failure("product creation", exc)
        return _error(str(exc))


@mcp.tool()
def shopify_update_product(
    product_id: str,
    title: str | None = None,
    description: str | None = None,
    vendor: str | None = None,
    product_type: str | None = None,
    tags: str | None = None,
    status: str | None = None,
) -> str:
    """
    Update an existing product. Only the fields explicitly provided (not
    None) are changed -- pass an empty string for description/vendor/
    product_type to clear that field (title cannot be cleared to blank).
    status: optional, one of "ACTIVE", "ARCHIVED", "DRAFT", "UNLISTED".
    """
    try:
        _require_admin_scopes("write_products")
        product_input: dict[str, Any] = {"id": _gid("Product", product_id)}
        if title is not None:
            _require_non_blank(title, "title")
            product_input["title"] = title
        if description is not None:
            product_input["descriptionHtml"] = description
        if vendor is not None:
            product_input["vendor"] = vendor
        if product_type is not None:
            product_input["productType"] = product_type
        if tags is not None:
            product_input["tags"] = _split_tags(tags)
        if status is not None:
            if status not in _PRODUCT_STATUSES:
                return _error(
                    f"status must be one of {sorted(_PRODUCT_STATUSES)}, got {status!r}"
                )
            product_input["status"] = status
        if len(product_input) == 1:
            return _error("at least one field to update must be provided")

        return _run_mutation(
            "mutation($product: ProductUpdateInput!) { productUpdate(product: $product)"
            f" {{ product {{ {_PRODUCT_FIELDS} }} userErrors {{ field message }} }} }}",
            {"product": product_input},
            "productUpdate",
            "product",
            _product_summary,
        )
    except Exception as exc:
        _log_failure("product update", exc)
        return _error(str(exc))


@mcp.tool()
def shopify_list_orders(query: str = "", limit: int = 25, after: str = "") -> str:
    """
    List/search orders. Shopify's Admin API only returns orders from
    roughly the last 60 days by default unless this connection has been
    granted the read_all_orders scope.
    query: optional Shopify search-syntax filter, e.g.
    "financial_status:paid" or "fulfillment_status:unfulfilled".
    limit: max orders to return (default 25, hard cap 100).
    after: pass the previous call's own after_cursor to fetch the next
    page; omit for the first page.
    """
    try:
        _require_admin_scopes("read_orders", "write_orders")
        return _list_connection(
            "orders", _ORDER_FIELDS, limit, query, after, _order_summary
        )
    except Exception as exc:
        _log_failure("order listing", exc)
        return _error(str(exc))


@mcp.tool()
def shopify_get_order(order_id: str) -> str:
    """
    Get a Shopify order by id (numeric id or full gid://shopify/Order/...).
    """
    try:
        _require_admin_scopes("read_orders", "write_orders")
        data, errors = _graphql(
            f"query($id: ID!) {{ order(id: $id) {{ {_ORDER_FIELDS} }} }}",
            {"id": _gid("Order", order_id)},
        )
        order = data.get("order")
        if not isinstance(order, dict) or not order:
            return _error(f"Order '{order_id}' not found")
        return _success_capped("order", _order_summary(order), errors)
    except Exception as exc:
        _log_failure("order lookup", exc)
        return _error(str(exc))


@mcp.tool()
def shopify_update_order(
    order_id: str, tags: str | None = None, note: str | None = None
) -> str:
    """
    Update an order's tags and/or note -- this tool does not touch payment,
    fulfillment, or customer-contact fields. Only the fields explicitly
    provided (not None) are changed -- pass an empty string to clear tags
    or the note entirely.
    tags: optional comma-separated tags, replacing the order's existing tags.
    note: optional internal note text.
    """
    try:
        _require_admin_scopes("write_orders")
        order_input: dict[str, Any] = {"id": _gid("Order", order_id)}
        if tags is not None:
            order_input["tags"] = _split_tags(tags)
        if note is not None:
            order_input["note"] = note
        if len(order_input) == 1:
            return _error("at least one of tags/note must be provided")

        return _run_mutation(
            "mutation($input: OrderInput!) { orderUpdate(input: $input)"
            f" {{ order {{ {_ORDER_FIELDS} }} userErrors {{ field message }} }} }}",
            {"input": order_input},
            "orderUpdate",
            "order",
            _order_summary,
        )
    except Exception as exc:
        _log_failure("order update", exc)
        return _error(str(exc))


@mcp.tool()
def shopify_list_customers(query: str = "", limit: int = 25, after: str = "") -> str:
    """
    List/search customers.
    query: optional Shopify search-syntax filter, e.g.
    "email:jane@example.com" or "tag:vip".
    limit: max customers to return (default 25, hard cap 100).
    after: pass the previous call's own after_cursor to fetch the next
    page; omit for the first page.
    """
    try:
        _require_admin_scopes("read_customers", "write_customers")
        return _list_connection(
            "customers", _CUSTOMER_FIELDS, limit, query, after, _customer_summary
        )
    except Exception as exc:
        _log_failure("customer listing", exc)
        return _error(str(exc))


@mcp.tool()
def shopify_get_customer(customer_id: str) -> str:
    """
    Get a Shopify customer by id (numeric id or full gid://shopify/Customer/...).
    """
    try:
        _require_admin_scopes("read_customers", "write_customers")
        data, errors = _graphql(
            f"query($id: ID!) {{ customer(id: $id) {{ {_CUSTOMER_FIELDS} }} }}",
            {"id": _gid("Customer", customer_id)},
        )
        customer = data.get("customer")
        if not isinstance(customer, dict) or not customer:
            return _error(f"Customer '{customer_id}' not found")
        return _success_capped("customer", _customer_summary(customer), errors)
    except Exception as exc:
        _log_failure("customer lookup", exc)
        return _error(str(exc))


@mcp.tool()
def shopify_list_collections(query: str = "", limit: int = 25, after: str = "") -> str:
    """
    List collections (product groupings) -- id, title, handle, and how many
    products each contains.
    query: optional Shopify search-syntax filter, e.g. "title:*sale*".
    Leave empty to list all collections.
    limit: max collections to return (default 25, hard cap 100).
    after: pass the previous call's own after_cursor to fetch the next
    page; omit for the first page.
    """
    try:
        _require_admin_scopes("read_products", "write_products")
        return _list_connection(
            "collections", _COLLECTION_FIELDS, limit, query, after, _collection_summary
        )
    except Exception as exc:
        _log_failure("collection listing", exc)
        return _error(str(exc))


if __name__ == "__main__":
    mcp.run()
