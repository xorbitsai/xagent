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

# Only a DNS *label* (no dots, scheme, port, or slashes) is ever accepted, so
# the string itself can never name a host outside "*.myshopify.com" -- but a
# legitimate hostname can still be rebound by DNS to a private/internal
# address at request time (orthogonal to who chose the hostname string), so
# _graphql_url() below still resolves and checks every address, same
# defense-in-depth posthog.py's _base_url() applies to its own two
# hardcoded-enum hostnames.
_SUBDOMAIN_PATTERN = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")

_GID_PATTERN = re.compile(r"^gid://shopify/[A-Za-z]+/\d+$")

_PRODUCT_STATUSES = frozenset({"ACTIVE", "ARCHIVED", "DRAFT"})


def _success(*, _errors: list[Any] | None = None, **payload: Any) -> str:
    body: dict[str, Any] = {"status": "success", **payload}
    if _errors:
        # A genuine partial GraphQL success (one sub-field failed, others
        # resolved) -- surface it in the result instead of only the server
        # log, matching linear.py's identical warnings contract.
        body["warnings"] = [graphql_errors_message(_errors)]
    return json.dumps(body, ensure_ascii=False)


def _error(message: str) -> str:
    return json.dumps({"status": "error", "message": message}, ensure_ascii=False)


def _headers() -> dict[str, str]:
    access_token = environ.get("SHOPIFY_ACCESS_TOKEN")
    if not access_token:
        raise ValueError("SHOPIFY_ACCESS_TOKEN environment variable is missing")
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


def _clamp_limit(limit: int) -> int:
    return clamp_limit(limit, max_limit=MAX_LIMIT)


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


def _mutation_failure_message(
    user_errors: list[dict[str, Any]], errors: list[Any]
) -> str:
    """`userErrors` alone can omit useful context a top-level GraphQL
    `errors` entry carries (e.g. a query-level access-scope warning attached
    to the same response) -- fold both in, mirroring linear.py's
    `_mutation_failure_message`, which does the same for Linear's boolean
    `success` discriminator."""
    message = _user_errors_message(user_errors)
    if errors:
        message = f"{message} ({graphql_errors_message(errors)})"
    return message


def _split_tags(tags: str) -> list[str]:
    return [t.strip() for t in tags.split(",") if t.strip()]


def _throttle_wait_seconds(response: requests.Response) -> float | None:
    """Return how long to wait before retrying, or None if this response
    doesn't signal throttling.

    Shopify's GraphQL cost-based throttling can surface as either an HTTP
    429 or an HTTP 200 whose body carries a "Throttled" error (the request
    cost more "leaky bucket" points than were available) -- checked
    defensively for both shapes since Shopify's own docs and real-world
    responses aren't fully consistent on which one to expect. The response
    body is parsed at most once (shared between the throttle check and the
    wait-time calculation) rather than twice.
    """
    # Parsed regardless of status code -- a 429 can carry the same
    # errors/extensions.cost body a 200 throttle response does, and this
    # value feeds the wait-time calculation below either way.
    try:
        payload = response.json()
    except ValueError:
        payload = None

    throttled = response.status_code == 429
    if not throttled and isinstance(payload, dict):
        for entry in payload.get("errors") or []:
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
        cost = (payload.get("extensions") or {}).get("cost") or {}
        throttle_status = cost.get("throttleStatus") or {}
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


def _graphql(
    query: str, variables: dict[str, Any] | None = None
) -> tuple[dict[str, Any], list[Any]]:
    """Run one GraphQL query/mutation against this store's Admin API
    endpoint.

    Returns (data, errors). Every query/mutation in this module selects
    exactly one top-level field, so if that field comes back null alongside
    a non-empty "errors" array there is nothing usable to return -- that is
    treated as a hard failure and raises instead. If at least one top-level
    field is non-null, it's a genuine partial success (e.g. a nested
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
            if attempt == 0:
                wait_seconds = _throttle_wait_seconds(response)
                if wait_seconds is not None and wait_seconds <= MAX_RETRY_AFTER_SECONDS:
                    time.sleep(wait_seconds)
                    continue
            break
    except requests.RequestException as exc:
        raise RuntimeError(redact_sensitive_text(str(exc))) from exc

    if 300 <= response.status_code < 400:
        raise RuntimeError(
            f"Shopify returned an unexpected redirect (HTTP {response.status_code}); "
            "refusing to follow it with credentials attached"
        )
    if response.status_code >= 400:
        detail: str | None = None
        try:
            payload = response.json()
            if isinstance(payload, dict) and payload.get("errors"):
                errors_field = payload["errors"]
                # Shopify's own auth-failure responses (e.g. a 401 for an
                # invalid access token) put a plain string here -- "[API]
                # Invalid API key or access token (...)" -- not the GraphQL
                # {"errors": [{"message": ...}]} shape graphql_errors_message
                # expects. Iterating a str with that helper walks it one
                # character at a time instead of raising, producing a
                # mangled "a; p; i" message instead of a clear error.
                detail = (
                    errors_field
                    if isinstance(errors_field, str)
                    else graphql_errors_message(errors_field)
                )
        except ValueError:
            pass
        if detail is None:
            detail = truncate_error_text(response.text.strip())
        raise RuntimeError(
            f"Shopify API error (status {response.status_code}): "
            f"{redact_sensitive_text(detail)}"
        )

    try:
        payload = response.json()
    except ValueError:
        detail = truncate_error_text(response.text.strip())
        raise RuntimeError(
            f"Shopify API returned a non-JSON response: {detail}"
        ) from None
    if not isinstance(payload, dict):
        raise RuntimeError(
            "Shopify API returned an unexpected (non-object) response body"
        )

    data = payload.get("data") or {}
    if len(data) > 1:
        raise RuntimeError(
            f"Shopify API response had {len(data)} top-level fields "
            f"({sorted(data)}), but this module's error handling assumes exactly one"
        )
    errors = payload.get("errors") or []
    if errors:
        message = graphql_errors_message(errors)
        if all(value is None for value in data.values()):
            raise RuntimeError(message)
        logger.warning(
            f"Shopify GraphQL partial error (data still returned): {message}"
        )
    return data, errors


def _extract_connection(
    data: dict[str, Any],
    field_name: str,
    summary_fn: Callable[[dict[str, Any]], dict[str, Any]],
) -> tuple[list[dict[str, Any]], bool, str | None]:
    """Pull nodes + cursor-pagination info out of a Relay-style connection
    field (`{ nodes { ... } pageInfo { hasNextPage endCursor } }`), shared
    by every list tool below."""
    connection = data.get(field_name) or {}
    nodes = connection.get("nodes") or []
    page_info = connection.get("pageInfo") or {}
    has_more = bool(page_info.get("hasNextPage"))
    after_cursor = page_info.get("endCursor") if has_more else None
    # A connection's individual nodes can be null (e.g. a node that failed
    # to resolve due to permissions or a backend error) even when the list
    # itself is present -- summary_fn assumes a dict, so skip null entries
    # rather than letting one bad node crash the whole page.
    return (
        [summary_fn(node) for node in nodes if node is not None],
        has_more,
        after_cursor,
    )


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
    max_results = _clamp_limit(limit)
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
        f" nodes {{ {selection} }} pageInfo {{ hasNextPage endCursor }} }} }}",
        variables,
    )
    items, has_more, next_cursor = _extract_connection(data, root_field, summary_fn)
    return success_with_capped_dict(
        root_field,
        {root_field: items, "has_more": has_more, "after_cursor": next_cursor},
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

    `result.get(mutation_field)` being falsy covers two distinct failure
    shapes the same way: the mutation field resolved to null (Shopify's own
    signal that the input's id/lookup didn't resolve to anything), or --
    defensively -- a malformed response where the field was omitted
    entirely with no errors at all. Either way there is nothing usable to
    return, so this fails closed instead of reporting an empty object as a
    success, matching linear.py's `if not result.get("success")` check
    (Shopify's write mutations here use an empty `userErrors` list as their
    success signal instead of Linear's boolean `success` field, but an
    absent result must still be treated as failure, not vacuously "no
    errors").
    """
    data, errors = _graphql(mutation, variables)
    result = data.get(mutation_field)
    if not result:
        return _error(f"Shopify returned no result for {mutation_field}")
    user_errors = result.get("userErrors") or []
    if user_errors:
        return _error(_mutation_failure_message(user_errors, errors))
    return _success(
        **{object_key: summary_fn(result.get(object_key) or {})}, _errors=errors
    )


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
    total_price = (order.get("totalPriceSet") or {}).get("shopMoney") or {}
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
    return {
        "id": customer.get("id"),
        "first_name": customer.get("firstName"),
        "last_name": customer.get("lastName"),
        "email": (customer.get("defaultEmailAddress") or {}).get("emailAddress"),
        "phone": (customer.get("defaultPhoneNumber") or {}).get("phoneNumber"),
        "number_of_orders": customer.get("numberOfOrders"),
        "tags": customer.get("tags"),
        "created_at": customer.get("createdAt"),
    }


_COLLECTION_FIELDS = "id title handle productsCount { count }"


def _collection_summary(collection: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": collection.get("id"),
        "title": collection.get("title"),
        "handle": collection.get("handle"),
        "products_count": (collection.get("productsCount") or {}).get("count"),
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
        return _success(shop=data.get("shop") or {}, _errors=errors)
    except Exception as e:
        logger.error(f"Error fetching Shopify shop info: {e}", exc_info=True)
        return _error(str(e))


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
        return _list_connection(
            "products", _PRODUCT_FIELDS, limit, query, after, _product_summary
        )
    except Exception as e:
        logger.error(f"Error listing Shopify products: {e}", exc_info=True)
        return _error(str(e))


@mcp.tool()
def shopify_get_product(product_id: str) -> str:
    """
    Get a Shopify product by id (numeric id or full gid://shopify/Product/...).
    """
    try:
        data, errors = _graphql(
            f"query($id: ID!) {{ product(id: $id) {{ {_PRODUCT_FIELDS} }} }}",
            {"id": _gid("Product", product_id)},
        )
        product = data.get("product")
        if not product:
            return _error(f"Product '{product_id}' not found")
        return _success(product=_product_summary(product), _errors=errors)
    except Exception as e:
        logger.error(f"Error fetching Shopify product {product_id}: {e}", exc_info=True)
        return _error(str(e))


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
    status: one of "ACTIVE", "ARCHIVED", "DRAFT" (default "DRAFT" -- Shopify
    creates products unpublished by default; publish it separately once
    it's ready).
    """
    try:
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
    except Exception as e:
        logger.error(f"Error creating Shopify product: {e}", exc_info=True)
        return _error(str(e))


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
    status: optional, one of "ACTIVE", "ARCHIVED", "DRAFT".
    """
    try:
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
    except Exception as e:
        logger.error(f"Error updating Shopify product {product_id}: {e}", exc_info=True)
        return _error(str(e))


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
        return _list_connection(
            "orders", _ORDER_FIELDS, limit, query, after, _order_summary
        )
    except Exception as e:
        logger.error(f"Error listing Shopify orders: {e}", exc_info=True)
        return _error(str(e))


@mcp.tool()
def shopify_get_order(order_id: str) -> str:
    """
    Get a Shopify order by id (numeric id or full gid://shopify/Order/...).
    """
    try:
        data, errors = _graphql(
            f"query($id: ID!) {{ order(id: $id) {{ {_ORDER_FIELDS} }} }}",
            {"id": _gid("Order", order_id)},
        )
        order = data.get("order")
        if not order:
            return _error(f"Order '{order_id}' not found")
        return _success(order=_order_summary(order), _errors=errors)
    except Exception as e:
        logger.error(f"Error fetching Shopify order {order_id}: {e}", exc_info=True)
        return _error(str(e))


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
    except Exception as e:
        logger.error(f"Error updating Shopify order {order_id}: {e}", exc_info=True)
        return _error(str(e))


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
        return _list_connection(
            "customers", _CUSTOMER_FIELDS, limit, query, after, _customer_summary
        )
    except Exception as e:
        logger.error(f"Error listing Shopify customers: {e}", exc_info=True)
        return _error(str(e))


@mcp.tool()
def shopify_get_customer(customer_id: str) -> str:
    """
    Get a Shopify customer by id (numeric id or full gid://shopify/Customer/...).
    """
    try:
        data, errors = _graphql(
            f"query($id: ID!) {{ customer(id: $id) {{ {_CUSTOMER_FIELDS} }} }}",
            {"id": _gid("Customer", customer_id)},
        )
        customer = data.get("customer")
        if not customer:
            return _error(f"Customer '{customer_id}' not found")
        return _success(customer=_customer_summary(customer), _errors=errors)
    except Exception as e:
        logger.error(
            f"Error fetching Shopify customer {customer_id}: {e}", exc_info=True
        )
        return _error(str(e))


@mcp.tool()
def shopify_list_collections(limit: int = 25, after: str = "") -> str:
    """
    List collections (product groupings) -- id, title, handle, and how many
    products each contains.
    limit: max collections to return (default 25, hard cap 100).
    after: pass the previous call's own after_cursor to fetch the next
    page; omit for the first page.
    """
    try:
        return _list_connection(
            "collections", _COLLECTION_FIELDS, limit, "", after, _collection_summary
        )
    except Exception as e:
        logger.error(f"Error listing Shopify collections: {e}", exc_info=True)
        return _error(str(e))


if __name__ == "__main__":
    mcp.run()
