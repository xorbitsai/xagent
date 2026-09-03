import json
import logging
import re
import socket
from collections.abc import Callable
from os import environ
from typing import Any
from urllib.parse import urlsplit

import requests
from mcp.server.fastmcp import FastMCP

from ....core.utils.security import (
    PrivateNetworkHostError,
    redact_sensitive_text,
    reject_private_network_host,
)
from ...utils.graphql_errors import truncate_error_text
from .utils import (
    clamp_limit,
    require_clean_identifier,
    setup_proxy_env,
    success_with_capped_dict,
    url_path_id,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("magento-mcp")

# Ensure standard proxy environment variables are set to prevent hanging requests
setup_proxy_env()

mcp = FastMCP("magento-mcp")

DEFAULT_TIMEOUT_SECONDS = 30
MAX_LIMIT = 100

_PRODUCT_STATUSES = frozenset({1, 2})  # 1 = Enabled, 2 = Disabled
_PRODUCT_VISIBILITIES = frozenset({1, 2, 3, 4})  # Not Visible/Catalog/Search/Both
_STORE_CODE_PATTERN = re.compile(r"^[A-Za-z0-9_]+$")


def _success(**payload: Any) -> str:
    return json.dumps({"status": "success", **payload}, ensure_ascii=False)


def _error(message: str) -> str:
    return json.dumps({"status": "error", "message": message}, ensure_ascii=False)


def _headers() -> dict[str, str]:
    """Magento's "Integration" access token, generated self-serve (Admin ->
    System -> Extensions -> Integrations -> Add New Integration, choose the
    resources it can access, Activate) -- no marketplace review, since this
    never leaves the merchant's own store.

    Sent as a plain Bearer token. Note: Magento 2.4.4+ disabled using an
    Integration's token this way by default (security hardening against a
    never-expiring bearer credential) -- the store admin must set Stores ->
    Configuration -> Services -> OAuth -> Consumer Settings -> "Allow OAuth
    Access Tokens to be used as standalone Bearer tokens" to Yes (or
    `bin/magento config:set oauth/consumer/enable_integration_as_bearer 1`)
    for this to work on 2.4.4+. The alternative -- OAuth 1.0a request
    signing with the integration's consumer key/secret -- has no such
    prerequisite but is a materially different, much higher-complexity
    auth scheme this module doesn't implement; a 401 from an otherwise
    correctly-configured integration is the signal to check this setting.
    """
    token = environ.get("MAGENTO_ACCESS_TOKEN")
    if not token:
        raise ValueError("MAGENTO_ACCESS_TOKEN environment variable is missing")
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


def _base_url() -> str:
    """Validate and return the store's bare origin (scheme + host[:port]).

    Unlike Shopify/Zendesk (always a "*.myshopify.com"/"*.zendesk.com"
    subdomain), Magento/Adobe Commerce is commonly self-hosted at an
    arbitrary domain -- MAGENTO_BASE_URL is a full user-supplied origin, so
    (mirroring posthog.py's POSTHOG_HOST handling) it's validated as a bare
    https origin with no embedded credentials/path/query/fragment, then
    resolved and checked against every address it resolves to before this
    connector will ever send its Bearer token there -- the primary SSRF
    defense here, not just defense-in-depth, since (again unlike PostHog)
    there is no small fixed-hostname allowlist to lean on first.
    """
    raw = environ.get("MAGENTO_BASE_URL", "").strip()
    if not raw:
        raise ValueError("MAGENTO_BASE_URL environment variable is missing")
    if "://" not in raw:
        raw = f"https://{raw}"
    parsed = urlsplit(raw)
    if parsed.scheme.lower() != "https":
        raise ValueError("MAGENTO_BASE_URL must be an https:// URL")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("MAGENTO_BASE_URL must not contain embedded credentials")
    if parsed.path.strip("/") or parsed.query or parsed.fragment:
        raise ValueError(
            "MAGENTO_BASE_URL must be a bare origin (scheme and host[:port] "
            "only), not a URL with a path, query, or fragment"
        )
    hostname = parsed.hostname
    if not hostname:
        raise ValueError("MAGENTO_BASE_URL must include a hostname")
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError(f"MAGENTO_BASE_URL has an invalid port: {exc}") from exc

    try:
        resolved = socket.getaddrinfo(
            hostname, port or 443, socket.AF_UNSPEC, socket.SOCK_STREAM
        )
        for *_, sockaddr in resolved:
            reject_private_network_host(str(sockaddr[0]))
    except PrivateNetworkHostError as exc:
        raise ValueError(f"MAGENTO_BASE_URL is not allowed: {exc}") from exc
    except OSError as exc:
        raise ValueError(f"MAGENTO_BASE_URL could not be resolved: {exc}") from exc

    origin = f"https://{hostname}"
    if port:
        origin += f":{port}"
    return origin


def _api_base_url() -> str:
    """MAGENTO_STORE_CODE scopes every call to one store view (a Magento
    installation with multiple storefronts/languages can have several) --
    left unset, /rest/V1/ resolves to the default store view, which is the
    right behavior for the overwhelmingly common single-store-view install
    and requires no extra setup from those users."""
    store_code = environ.get("MAGENTO_STORE_CODE", "").strip()
    if not store_code:
        return f"{_base_url()}/rest/V1"
    # Magento store codes are themselves restricted to this shape (letters,
    # digits, underscores); enforcing it here means a misconfigured env var
    # (e.g. one containing "/" or "?") fails with a clear error instead of
    # silently splicing extra path segments or query params into every
    # request this connector makes.
    if not _STORE_CODE_PATTERN.match(store_code):
        raise ValueError(
            "MAGENTO_STORE_CODE must contain only letters, digits, and "
            f"underscores, got {store_code!r}"
        )
    return f"{_base_url()}/rest/{store_code}/V1"


def _clamp_limit(limit: int) -> int:
    return clamp_limit(limit, max_limit=MAX_LIMIT)


def _require_non_blank(value: str, field_name: str) -> str:
    if not value or not value.strip():
        raise ValueError(f"{field_name} must not be blank")
    return value


def _validate_choice(
    value: int, allowed: frozenset[int], field_name: str
) -> str | None:
    """Return an error message if value isn't in allowed, else None -- shared
    by every status/visibility check below so the wording and comparison
    can't drift between the list/create/update tools that all repeat it."""
    if value not in allowed:
        return f"{field_name} must be one of {sorted(allowed)}, got {value!r}"
    return None


def _extract_error_detail(response: requests.Response) -> str | None:
    """Pull the human-readable message out of a Magento error body.

    Magento's exception responses are {"message": "...", "parameters": ...}
    where the message is a template with %-prefixed placeholders --
    %resources/%fieldName-style named ones (parameters is a dict) or
    positional %1/%2 ones (parameters is a list), depending on which
    exception raised it. Substituted here so the LLM sees the actual values
    instead of a raw template. Both the named and positional forms
    substitute longest-placeholder-first (by key length, or by descending
    index for the positional form) rather than in whatever order the
    parameters happened to arrive in -- otherwise substituting a shorter
    placeholder first (e.g. "%1" or "%field") can also match inside a
    longer one that hasn't been substituted yet ("%10", "%fieldName"),
    corrupting it. Returns None so the caller falls back to the raw
    response text when the body isn't in this shape at all.
    """
    try:
        payload = response.json()
    except ValueError:
        return None
    if not isinstance(payload, dict):
        return None
    message = payload.get("message")
    if not isinstance(message, str) or not message:
        return None
    parameters = payload.get("parameters")
    if isinstance(parameters, dict):
        for key, value in sorted(parameters.items(), key=lambda item: -len(item[0])):
            message = message.replace(f"%{key}", str(value))
    elif isinstance(parameters, list):
        for index, value in reversed(list(enumerate(parameters, start=1))):
            message = message.replace(f"%{index}", str(value))
    return message


def _request(
    method: str,
    path: str,
    *,
    params: dict[str, Any] | None = None,
    json_data: dict[str, Any] | None = None,
) -> Any:
    url = f"{_api_base_url()}{path}"
    try:
        response = requests.request(
            method=method,
            url=url,
            headers=_headers(),
            params=params,
            json=json_data,
            timeout=DEFAULT_TIMEOUT_SECONDS,
            # A redirect response is never followed with the Bearer header
            # still attached: a self-hosted store redirecting an API call
            # (e.g. an http->https or www-canonicalization rule misapplied
            # to /rest/) is either a misconfiguration or a host trying to
            # relay the credential elsewhere.
            allow_redirects=False,
        )
    except requests.RequestException as exc:
        # A connection/timeout/proxy failure's message can itself embed
        # sensitive data -- e.g. a ProxyError echoing the ambient
        # HTTPS_PROXY URL, which may carry embedded user:pass@ credentials
        # (setup_proxy_env() exports whatever the OS has configured).
        raise RuntimeError(redact_sensitive_text(str(exc))) from exc

    if 300 <= response.status_code < 400:
        raise RuntimeError(
            f"Magento returned an unexpected redirect (HTTP {response.status_code}); "
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
            f"Magento returned a 2xx response with a non-JSON body: {exc}"
        ) from exc


def _search_criteria_params(
    filters: list[tuple[str, str, str]], page_size: int, current_page: int
) -> dict[str, Any]:
    """Build Magento's searchCriteria query params.

    Each filter gets its *own* filter_groups entry (never grouped together
    into one). Magento's searchCriteria ORs the filters *within* a single
    filter_groups entry and ANDs *across* filter_groups entries -- the
    reverse of what putting them in one group would suggest -- and every
    caller here wants every given filter satisfied simultaneously (e.g.
    sku_like AND status), never "either". filters: list of (field,
    condition_type, value) tuples, e.g. [("sku", "like", "%shirt%")].
    """
    params: dict[str, Any] = {
        "searchCriteria[pageSize]": page_size,
        "searchCriteria[currentPage]": current_page,
    }
    for index, (field, condition_type, value) in enumerate(filters):
        prefix = f"searchCriteria[filter_groups][{index}][filters][0]"
        params[f"{prefix}[field]"] = field
        params[f"{prefix}[value]"] = value
        params[f"{prefix}[condition_type]"] = condition_type
    return params


def _paginated_result(
    payload: Any, page_size: int, current_page: int
) -> tuple[list[Any], bool]:
    """Slice a Magento searchCriteria list response.

    Magento's SearchResultsInterface responses are {"items": [...],
    "search_criteria": {...}, "total_count": N} -- has_more is computed
    from total_count/pageSize rather than an extra "next page" signal,
    since Magento's own response carries no such flag.
    """
    if not isinstance(payload, dict):
        raise ValueError(
            f"Expected a JSON object from Magento, got {type(payload).__name__}"
        )
    items = payload.get("items") or []
    if not isinstance(items, list):
        raise ValueError(
            f'Expected Magento\'s "items" field to be a list, got {type(items).__name__}'
        )
    total_count = payload.get("total_count")
    has_more = isinstance(total_count, int) and current_page * page_size < total_count
    return items, has_more


def _list_search(
    path: str,
    result_key: str,
    filters: list[tuple[str, str, str]],
    limit: int,
    page: int,
    summary_fn: Callable[[dict[str, Any]], dict[str, Any]],
    extra_params: dict[str, Any] | None = None,
) -> str:
    """Shared body for every "list/search X" tool below (products, orders,
    customers): clamp paging, build searchCriteria params, call the
    endpoint, slice the response, and wrap it -- only the path, result key,
    filters, and summarizer differ between them."""
    max_results = _clamp_limit(limit)
    current_page = max(1, page)
    params = _search_criteria_params(filters, max_results, current_page)
    if extra_params:
        params.update(extra_params)
    result = _request("GET", path, params=params)
    items, has_more = _paginated_result(result, max_results, current_page)
    return success_with_capped_dict(
        result_key,
        {
            result_key: [summary_fn(item) for item in items],
            "has_more": has_more,
            "next_page": current_page + 1 if has_more else None,
        },
    )


def _product_summary(product: dict[str, Any]) -> dict[str, Any]:
    return {
        "sku": product.get("sku"),
        "name": product.get("name"),
        "price": product.get("price"),
        "status": product.get("status"),
        "visibility": product.get("visibility"),
        "type_id": product.get("type_id"),
        "attribute_set_id": product.get("attribute_set_id"),
        "created_at": product.get("created_at"),
        "updated_at": product.get("updated_at"),
    }


def _order_summary(order: dict[str, Any]) -> dict[str, Any]:
    return {
        "entity_id": order.get("entity_id"),
        "increment_id": order.get("increment_id"),
        "state": order.get("state"),
        "status": order.get("status"),
        "customer_email": order.get("customer_email"),
        "grand_total": order.get("grand_total"),
        "currency": order.get("order_currency_code"),
        "created_at": order.get("created_at"),
        "updated_at": order.get("updated_at"),
    }


def _customer_summary(customer: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": customer.get("id"),
        "email": customer.get("email"),
        "firstname": customer.get("firstname"),
        "lastname": customer.get("lastname"),
        "group_id": customer.get("group_id"),
        "created_at": customer.get("created_at"),
    }


def _category_summary(category: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": category.get("id"),
        "parent_id": category.get("parent_id"),
        "name": category.get("name"),
        "is_active": category.get("is_active"),
        "position": category.get("position"),
        "level": category.get("level"),
        "product_count": category.get("product_count"),
        "children_data": [
            _category_summary(child) for child in category.get("children_data") or []
        ],
    }


@mcp.tool()
def magento_list_products(
    sku_like: str = "", status: int | None = None, limit: int = 25, page: int = 1
) -> str:
    """
    Search/list products.
    sku_like: optional substring to match against SKU (wrapped in % wildcards
    server-side, e.g. "shirt" matches any SKU containing "shirt").
    status: optional filter, 1 (Enabled) or 2 (Disabled).
    limit: max products to return (default 25, hard cap 100).
    page: 1-based page number; pass the previous page + 1 to continue.
    """
    try:
        if status is not None:
            if err := _validate_choice(status, _PRODUCT_STATUSES, "status"):
                return _error(err)
        filters: list[tuple[str, str, str]] = []
        if sku_like:
            filters.append(("sku", "like", f"%{sku_like}%"))
        if status is not None:
            filters.append(("status", "eq", str(status)))
        return _list_search(
            "/products", "products", filters, limit, page, _product_summary
        )
    except Exception as e:
        logger.error(f"Error listing Magento products: {e}")
        return _error(str(e))


@mcp.tool()
def magento_get_product(sku: str) -> str:
    """
    Get a Magento product by SKU (Magento's REST API identifies products by
    SKU, not a separate numeric id).
    """
    try:
        result = _request("GET", f"/products/{url_path_id(sku, 'sku')}")
        return _success(product=_product_summary(result))
    except Exception as e:
        logger.error(f"Error fetching Magento product {sku}: {e}")
        return _error(str(e))


@mcp.tool()
def magento_create_product(
    sku: str,
    name: str,
    price: float,
    attribute_set_id: int = 4,
    type_id: str = "simple",
    status: int = 1,
    visibility: int = 4,
) -> str:
    """
    Create a new (simple) product.
    sku: the product's unique SKU.
    name: the product's display name.
    price: the product's price.
    attribute_set_id: the attribute set this product uses (default 4, the
    stock "Default" set in most installs -- if product creation fails with
    an attribute-set error, look up the correct id via the store's admin
    under Catalog -> Attributes -> Attribute Set).
    type_id: the product type (default "simple"; other Magento product
    types -- configurable, bundle, grouped, virtual, downloadable -- need
    additional fields this tool does not set).
    status: 1 (Enabled) or 2 (Disabled).
    visibility: 1 (Not Visible Individually), 2 (Catalog), 3 (Search), or
    4 (Catalog and Search).
    """
    try:
        # sku only ever goes into the JSON body here (never a URL path
        # segment, unlike get/update product's sku), so it's validated with
        # require_clean_identifier rather than url_path_id -- but that
        # still means (consistent with get/update) a whitespace-padded
        # value like " SHIRT-1 " is rejected here too, rather than silently
        # creating a product a later get/update of the same string would
        # then reject as invalid.
        require_clean_identifier(sku, "sku")
        _require_non_blank(name, "name")
        if err := _validate_choice(status, _PRODUCT_STATUSES, "status"):
            return _error(err)
        if err := _validate_choice(visibility, _PRODUCT_VISIBILITIES, "visibility"):
            return _error(err)
        product_input = {
            "sku": sku,
            "name": name,
            "price": price,
            "attribute_set_id": attribute_set_id,
            "type_id": type_id,
            "status": status,
            "visibility": visibility,
        }
        result = _request("POST", "/products", json_data={"product": product_input})
        return _success(product=_product_summary(result))
    except Exception as e:
        logger.error(f"Error creating Magento product {sku}: {e}")
        return _error(str(e))


@mcp.tool()
def magento_update_product(
    sku: str,
    name: str | None = None,
    price: float | None = None,
    status: int | None = None,
    visibility: int | None = None,
) -> str:
    """
    Update an existing product. Only the fields explicitly provided (not
    None) are changed.
    status: optional, 1 (Enabled) or 2 (Disabled).
    visibility: optional, 1 (Not Visible Individually), 2 (Catalog),
    3 (Search), or 4 (Catalog and Search).
    """
    try:
        encoded_sku = url_path_id(sku, "sku")
        product_input: dict[str, Any] = {"sku": sku}
        fields_provided = False
        if name is not None:
            _require_non_blank(name, "name")
            product_input["name"] = name
            fields_provided = True
        if price is not None:
            product_input["price"] = price
            fields_provided = True
        if status is not None:
            if err := _validate_choice(status, _PRODUCT_STATUSES, "status"):
                return _error(err)
            product_input["status"] = status
            fields_provided = True
        if visibility is not None:
            if err := _validate_choice(visibility, _PRODUCT_VISIBILITIES, "visibility"):
                return _error(err)
            product_input["visibility"] = visibility
            fields_provided = True
        if not fields_provided:
            return _error("at least one field to update must be provided")

        result = _request(
            "PUT", f"/products/{encoded_sku}", json_data={"product": product_input}
        )
        return _success(product=_product_summary(result))
    except Exception as e:
        logger.error(f"Error updating Magento product {sku}: {e}")
        return _error(str(e))


@mcp.tool()
def magento_list_orders(status: str = "", limit: int = 25, page: int = 1) -> str:
    """
    Search/list orders, most recent first.
    status: optional order status to filter by, e.g. "processing",
    "complete", "pending", "canceled" (a store's exact status values are
    configurable, so these are the common defaults, not a fixed enum).
    limit: max orders to return (default 25, hard cap 100).
    page: 1-based page number; pass the previous page + 1 to continue.
    """
    try:
        filters: list[tuple[str, str, str]] = []
        if status:
            filters.append(("status", "eq", status))
        extra_params = {
            "searchCriteria[sortOrders][0][field]": "created_at",
            "searchCriteria[sortOrders][0][direction]": "DESC",
        }
        return _list_search(
            "/orders", "orders", filters, limit, page, _order_summary, extra_params
        )
    except Exception as e:
        logger.error(f"Error listing Magento orders: {e}")
        return _error(str(e))


@mcp.tool()
def magento_get_order(order_id: int) -> str:
    """
    Get a Magento order by its numeric entity id (from magento_list_orders;
    not the customer-facing increment_id like "000000123").
    """
    try:
        result = _request("GET", f"/orders/{order_id}")
        return _success(order=_order_summary(result))
    except Exception as e:
        logger.error(f"Error fetching Magento order {order_id}: {e}")
        return _error(str(e))


@mcp.tool()
def magento_add_order_comment(
    order_id: int,
    comment: str,
    status: str = "",
    notify_customer: bool = False,
    visible_to_customer: bool = True,
) -> str:
    """
    Add a status-history comment to an order -- Magento has no separate
    "internal note" concept; visible_to_customer controls whether this
    entry appears in the customer's own order history.
    order_id: an order's numeric entity id, from magento_list_orders.
    comment: the comment text.
    status: optional order status to set alongside the comment (leave
    empty to keep the order's current status).
    notify_customer: if true, emails the customer this comment/status change.
    visible_to_customer: if false, this is an internal-only note.
    """
    try:
        _require_non_blank(comment, "comment")
        status_history: dict[str, Any] = {
            "comment": comment,
            "is_customer_notified": notify_customer,
            "is_visible_on_front": visible_to_customer,
        }
        if status:
            status_history["status"] = status
        result = _request(
            "POST",
            f"/orders/{order_id}/comments",
            json_data={"statusHistory": status_history},
        )
        return _success(added=bool(result))
    except Exception as e:
        logger.error(f"Error adding comment to Magento order {order_id}: {e}")
        return _error(str(e))


@mcp.tool()
def magento_list_customers(email_like: str = "", limit: int = 25, page: int = 1) -> str:
    """
    Search/list customer accounts.
    email_like: optional substring to match against email (wrapped in %
    wildcards server-side).
    limit: max customers to return (default 25, hard cap 100).
    page: 1-based page number; pass the previous page + 1 to continue.
    """
    try:
        filters: list[tuple[str, str, str]] = []
        if email_like:
            filters.append(("email", "like", f"%{email_like}%"))
        return _list_search(
            "/customers/search", "customers", filters, limit, page, _customer_summary
        )
    except Exception as e:
        logger.error(f"Error listing Magento customers: {e}")
        return _error(str(e))


@mcp.tool()
def magento_get_customer(customer_id: int) -> str:
    """
    Get a Magento customer by numeric id.
    """
    try:
        result = _request("GET", f"/customers/{customer_id}")
        return _success(customer=_customer_summary(result))
    except Exception as e:
        logger.error(f"Error fetching Magento customer {customer_id}: {e}")
        return _error(str(e))


@mcp.tool()
def magento_get_category_tree(
    root_category_id: int | None = None, depth: int | None = None
) -> str:
    """
    Get the category tree (id, name, active/position, product_count, and
    nested children_data), starting from root_category_id (the store's
    absolute root if omitted) down to depth levels (unlimited if omitted).
    """
    try:
        params: dict[str, Any] = {}
        if root_category_id is not None:
            params["rootCategoryId"] = root_category_id
        if depth is not None:
            params["depth"] = depth
        result = _request("GET", "/categories", params=params)
        return success_with_capped_dict("category", _category_summary(result))
    except Exception as e:
        logger.error(f"Error fetching Magento category tree: {e}")
        return _error(str(e))


@mcp.tool()
def magento_get_category(category_id: int) -> str:
    """
    Get one category's own fields (id, name, active/position). Magento's
    plain single-category lookup does not reliably populate product_count
    or children_data the way the tree endpoint does (those are computed by
    the tree-building service, not a plain load) -- use
    magento_get_category_tree with root_category_id=category_id and
    depth=1 instead if you need this category's children or product count.
    """
    try:
        result = _request("GET", f"/categories/{category_id}")
        return _success(category=_category_summary(result))
    except Exception as e:
        logger.error(f"Error fetching Magento category {category_id}: {e}")
        return _error(str(e))


if __name__ == "__main__":
    mcp.run()
