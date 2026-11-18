"""Azure Cosmos DB (NoSQL API) MCP connector, read-only, key authenticated.

Backed by Microsoft's ``azure-cosmos`` Python SDK (synchronous client: every
tool call runs in its own short-lived stdio subprocess, so there is nothing to
gain from the async client).

Configuration, all three required: ``COSMOS_ENDPOINT`` (the account URI from
the portal, ``https://<account>.documents.azure.com:443/``), ``COSMOS_KEY``
(an account key; a read-only key is enough and recommended) and
``COSMOS_DATABASE``. Container tools are pinned to ``COSMOS_DATABASE``; the
account key itself is account-wide, so "one database only" is enforced here.

Read-only by construction: the only SDK calls made are ``list_databases``,
``DatabaseProxy.read``, ``list_containers``, ``query_items`` and ``read_item``.
Cosmos DB's SQL dialect has no data-modifying statements, and a query must
still start with ``SELECT``.

The endpoint is user supplied and this module runs on the web host, so:

* only ``https://<one label>.documents.azure.{com,us,cn}`` is accepted (public,
  Azure Government and Azure operated by 21Vianet), default port only;
* the SDK's regional endpoint discovery is turned off, so every request goes to
  the validated endpoint, and the SDK does not follow redirects;
* every request additionally passes ``_GuardedTransport``, which re-checks the
  host and sends through the public-only ``public_requests`` session (resolves,
  rejects private/link-local addresses and connects to the address it checked).

The key, the endpoint's query string and raw SDK exception text never reach a
tool result or a log line; failures are mapped to ``error_code`` values.
"""

import json
import logging
import os
import re
from itertools import islice
from typing import Any, Iterable
from urllib.parse import urlsplit

from azure.core.exceptions import (
    AzureError,
    ServiceRequestError,
    ServiceRequestTimeoutError,
    ServiceResponseTimeoutError,
)
from azure.core.pipeline.transport import RequestsTransport
from azure.cosmos import CosmosClient
from azure.cosmos.exceptions import CosmosHttpResponseError
from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from ....core.utils import public_requests
from ....core.utils.security import redact_sensitive_text
from .utils import clamp_limit

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cosmos-mcp")
# azure-core logs request URLs and headers at INFO; keep the SDK quiet.
logging.getLogger("azure").setLevel(logging.WARNING)

mcp = FastMCP("cosmos-mcp")

READ_ONLY = ToolAnnotations(readOnlyHint=True)

# Account host suffixes: public cloud, Azure Government, Azure operated by
# 21Vianet (Microsoft Learn, "Compare Azure Government and global Azure" and
# docs.azure.cn). Germany's cloud is retired and not listed.
ALLOWED_HOST_SUFFIXES = (
    "documents.azure.com",
    "documents.azure.us",
    "documents.azure.cn",
)
_HOST_PATTERN = re.compile(
    r"\A[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.(?:"
    + "|".join(re.escape(s) for s in ALLOWED_HOST_SUFFIXES)
    + r")\Z"
)
# Cosmos resource ids may not contain / \ ? # (Cosmos DB resource-id rules).
_ID_FORBIDDEN = re.compile(r"[/\\?#\x00-\x1f\x7f]")
_PARAM_NAME = re.compile(r"\A@?([A-Za-z_][A-Za-z0-9_]{0,63})\Z")
_SELECT = re.compile(r"\A\s*SELECT\b", re.IGNORECASE)

# Bounds on a single call. The SDK defaults (65 s read timeout, up to 9
# throttle retries) are far too generous for a call the model is waiting on.
CONNECTION_TIMEOUT_SECONDS = 10
READ_TIMEOUT_SECONDS = 20
TOTAL_TIMEOUT_SECONDS = 45
RETRY_TOTAL = 2
RETRY_BACKOFF_MAX_SECONDS = 4
THROTTLE_RETRY_TOTAL = 3
THROTTLE_BACKOFF_MAX_SECONDS = 8

DEFAULT_PAGE_SIZE = 25
MAX_PAGE_SIZE = 100
MAX_LIST_ITEMS = 100
MAX_QUERY_CHARS = 4096
MAX_PARAMETERS = 50
MAX_CONTINUATION_TOKEN_CHARS = 32768
# Cap on the serialized ``items`` / ``item`` of a result; the envelope around
# them is small. Over the cap the tool truncates and says so.
MAX_RESULT_BYTES = 60_000
MAX_OMITTED_IDS = 50

_REQUEST_CHARGE_HEADER = "x-ms-request-charge"
_RETRY_AFTER_HEADER = "x-ms-retry-after-ms"
_KEY_AUTH_DISABLED_MARKER = "local authorization is disabled"


class CosmosConfigError(ValueError):
    """A problem with the connector's own input; the message is ours, safe to show."""


def _success(**payload: Any) -> str:
    return json.dumps({"status": "success", **payload}, ensure_ascii=False, default=str)


def _error(error_code: str, message: str, **extra: Any) -> str:
    return json.dumps(
        {"status": "error", "error_code": error_code, "message": message, **extra},
        ensure_ascii=False,
    )


def _env(name: str) -> str:
    value = (os.environ.get(name) or "").strip()
    if not value:
        raise CosmosConfigError(f"{name} environment variable is missing or empty")
    return value


def validate_endpoint(raw: str) -> str:
    """Return the normalized ``https://<host>/`` or raise ``CosmosConfigError``.

    Accepts only ``https`` with a host of one label under an Azure Cosmos DB
    account suffix, port 443 (or none), no userinfo, path, query or fragment.
    IP literals fail the host pattern.
    """
    value = raw.strip()
    if not value.isascii() or not value.isprintable() or " " in value:
        raise CosmosConfigError("COSMOS_ENDPOINT is not a valid URL")
    try:
        parts = urlsplit(value)
        port = parts.port
    except ValueError:
        raise CosmosConfigError("COSMOS_ENDPOINT is not a valid URL") from None
    if parts.scheme != "https":
        raise CosmosConfigError("COSMOS_ENDPOINT must be an https URL")
    if "@" in parts.netloc or parts.username is not None or parts.password is not None:
        raise CosmosConfigError("COSMOS_ENDPOINT must not contain credentials")
    if "?" in value or "#" in value or parts.query or parts.fragment:
        raise CosmosConfigError("COSMOS_ENDPOINT must not contain a query or fragment")
    if parts.path not in ("", "/"):
        raise CosmosConfigError("COSMOS_ENDPOINT must not contain a path")
    if port not in (None, 443):
        raise CosmosConfigError("COSMOS_ENDPOINT must use the default https port 443")
    host = (parts.hostname or "").lower()
    if not _HOST_PATTERN.match(host):
        raise CosmosConfigError(
            "COSMOS_ENDPOINT must be an Azure Cosmos DB account URI such as "
            "https://<account>.documents.azure.com:443/ (also *.documents.azure.us "
            "and *.documents.azure.cn)"
        )
    return f"https://{host}/"


class _GuardedTransport(RequestsTransport):  # type: ignore[no-any-unimported]
    """Azure Core's requests transport with the host re-checked on every request.

    The session is ``public_requests.Session``: it resolves the host, rejects
    non-public addresses and connects to the address it checked. The host check
    here means a request to anything but an allowed Cosmos host (a discovered
    regional endpoint, a redirect target) is refused before it is sent.
    """

    def __init__(self) -> None:
        super().__init__(session=public_requests.Session(), session_owner=True)

    def send(self, request: Any, **kwargs: Any) -> Any:
        parts = urlsplit(str(request.url))
        validate_endpoint(f"{parts.scheme}://{parts.netloc}/")
        return super().send(request, **kwargs)


def _client() -> CosmosClient:
    endpoint = validate_endpoint(_env("COSMOS_ENDPOINT"))
    key = _env("COSMOS_KEY")
    return CosmosClient(
        endpoint,
        credential=key,
        transport=_GuardedTransport(),
        # No regional endpoint discovery: all traffic stays on the endpoint
        # validated above (the account metadata can name other hosts).
        enable_endpoint_discovery=False,
        connection_timeout=CONNECTION_TIMEOUT_SECONDS,
        read_timeout=READ_TIMEOUT_SECONDS,
        timeout=TOTAL_TIMEOUT_SECONDS,
        retry_total=RETRY_TOTAL,
        retry_backoff_max=RETRY_BACKOFF_MAX_SECONDS,
        retry_throttle_total=THROTTLE_RETRY_TOTAL,
        retry_throttle_backoff_max=THROTTLE_BACKOFF_MAX_SECONDS,
    )


def _database_name() -> str:
    return _check_id(_env("COSMOS_DATABASE"), "COSMOS_DATABASE")


def _check_id(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > 255
        or _ID_FORBIDDEN.search(value)
    ):
        raise CosmosConfigError(
            f"{name} must be a non-empty name of at most 255 characters without "
            "/ \\ ? # or control characters"
        )
    return value


def _query_error_details(exc: CosmosHttpResponseError) -> str | None:
    """Best-effort: the service's own error messages for a rejected query.

    A 400 body looks like ``Message: {"errors":[{"message":"Syntax error ..."}]}``
    (shape inferred from service behaviour, not documented). Returns ``None``
    when it does not parse, so no raw text is ever passed through.
    """
    text = str(getattr(exc, "message", "") or "")
    start = text.find("{")
    if start < 0:
        return None
    try:
        body, _ = json.JSONDecoder().raw_decode(text[start:])
        messages = [str(e["message"]) for e in body["errors"] if isinstance(e, dict)]
    except (ValueError, KeyError, TypeError):
        return None
    if not messages:
        return None
    return redact_sensitive_text(" ".join(messages))[:300]


def _map_error(exc: Exception, target: str) -> str:
    """Map an exception to an actionable envelope without echoing its text."""
    if isinstance(exc, CosmosConfigError):
        return _error("invalid_configuration", str(exc))
    if isinstance(exc, CosmosHttpResponseError):
        status = exc.status_code
        text = f"{getattr(exc, 'message', '')} {exc}".lower()
        logger.error("Cosmos request failed: HTTP %s on %s", status, target)
        if _KEY_AUTH_DISABLED_MARKER in text and status in (401, 403):
            return _error(
                "key_auth_disabled",
                "This account has key-based authentication disabled "
                "(disableLocalAuth), so account keys cannot be used.",
                http_status=status,
            )
        if status == 401:
            return _error(
                "authentication_failed",
                "Cosmos DB rejected the key (401). Check that COSMOS_KEY is a "
                "valid key for this account and was copied in full.",
                http_status=401,
            )
        if status == 403:
            return _error(
                "forbidden",
                "Cosmos DB refused the request (403). The account's firewall or "
                "network settings most likely block this service's IP address, "
                "or public network access is disabled.",
                http_status=403,
            )
        if status == 404:
            return _error(
                "not_found",
                f"Not found (404): check that {target} exists in the configured "
                "database.",
                http_status=404,
            )
        if status == 429:
            retry_after = (getattr(exc, "headers", None) or {}).get(_RETRY_AFTER_HEADER)
            return _error(
                "throttled",
                "Cosmos DB throttled the request (429): the container's "
                "throughput is exhausted. Retry later or lower the page size.",
                http_status=429,
                **({"retry_after_ms": retry_after} if retry_after else {}),
            )
        if status == 400:
            details = _query_error_details(exc)
            return _error(
                "bad_request",
                "Cosmos DB rejected the request (400)."
                + (f" {details}" if details else ""),
                http_status=400,
            )
        if status == 408:
            return _error("timeout", "The request to Cosmos DB timed out.")
        if status is not None and status >= 500:
            return _error(
                "service_unavailable",
                f"Cosmos DB is unavailable ({status}). Retry later.",
                http_status=status,
            )
        return _error(
            "request_failed",
            f"Cosmos DB returned HTTP {status}.",
            http_status=status,
        )
    if isinstance(exc, (ServiceRequestTimeoutError, ServiceResponseTimeoutError)):
        logger.error("Cosmos request timed out on %s", target)
        return _error("timeout", "The request to Cosmos DB timed out.")
    if isinstance(exc, ServiceRequestError):
        logger.error("Cosmos endpoint unreachable on %s", target)
        return _error(
            "unreachable",
            "Could not reach the Cosmos DB endpoint. Check COSMOS_ENDPOINT and "
            "network access.",
        )
    logger.error("Cosmos call failed on %s: %s", target, type(exc).__name__)
    if isinstance(exc, AzureError):
        return _error("request_failed", "The Cosmos DB request failed.")
    return _error("internal_error", "The Cosmos DB request could not be completed.")


def _serialized_size(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False, default=str).encode("utf-8"))


def _fit_items(items: list[Any]) -> tuple[list[Any], list[Any]]:
    """Split ``items`` into the prefix that fits ``MAX_RESULT_BYTES`` and the rest."""
    kept: list[Any] = []
    used = 2
    for index, item in enumerate(items):
        size = _serialized_size(item) + 1
        if used + size > MAX_RESULT_BYTES:
            return kept, items[index:]
        kept.append(item)
        used += size
    return kept, []


def _parameters(raw: dict[str, Any] | None) -> list[dict[str, Any]]:
    if raw is None:
        return []
    if not isinstance(raw, dict) or len(raw) > MAX_PARAMETERS:
        raise CosmosConfigError(
            f"parameters must be an object of at most {MAX_PARAMETERS} named values"
        )
    result = []
    for name, value in raw.items():
        match = _PARAM_NAME.match(str(name))
        if not match:
            raise CosmosConfigError(
                "parameter names must look like @name (letters, digits, underscore)"
            )
        result.append({"name": f"@{match.group(1)}", "value": value})
    return result


def _charge_recorder() -> tuple[list[float], Any]:
    """A ``response_hook`` that collects the request charge of every response."""
    charges: list[float] = []

    def record(headers: Any, _body: Any) -> None:
        try:
            charges.append(float(headers.get(_REQUEST_CHARGE_HEADER)))
        except (TypeError, ValueError):
            pass

    return charges, record


def _bounded(iterable: Iterable[Any]) -> tuple[list[Any], bool]:
    items = list(islice(iterable, MAX_LIST_ITEMS + 1))
    return items[:MAX_LIST_ITEMS], len(items) > MAX_LIST_ITEMS


@mcp.tool(annotations=READ_ONLY)
def cosmos_list_databases() -> str:
    """
    List the databases of the connected Cosmos DB account (account-wide; the
    other tools only work on the configured database).
    """
    try:
        database_name = _database_name()
        client = _client()
        try:
            databases, truncated = _bounded(
                client.list_databases(max_item_count=MAX_LIST_ITEMS + 1)
            )
        finally:
            client.close()
        return _success(
            databases=[d.get("id") for d in databases],
            configured_database=database_name,
            truncated=truncated,
        )
    except Exception as e:
        return _map_error(e, "the account")


@mcp.tool(annotations=READ_ONLY)
def cosmos_list_containers() -> str:
    """
    List the containers of the configured database with their partition key
    paths. Also confirms that the endpoint, key, network access and database
    name all work (use it first if any other Cosmos call fails).
    """
    try:
        database_name = _database_name()
        client = _client()
        try:
            database = client.get_database_client(database_name)
            database.read()  # one metadata read: the database must exist
            containers, truncated = _bounded(
                database.list_containers(max_item_count=MAX_LIST_ITEMS + 1)
            )
        finally:
            client.close()
        return _success(
            database=database_name,
            containers=[
                {
                    "id": c.get("id"),
                    "partition_key_paths": (c.get("partitionKey") or {}).get("paths"),
                }
                for c in containers
            ],
            truncated=truncated,
        )
    except Exception as e:
        return _map_error(e, "the configured database")


@mcp.tool(annotations=READ_ONLY)
def cosmos_query_container(
    container: str,
    query: str,
    parameters: dict[str, Any] | None = None,
    partition_key: str
    | int
    | float
    | bool
    | list[str | int | float | bool]
    | None = None,
    max_item_count: int = DEFAULT_PAGE_SIZE,
    continuation_token: str | None = None,
) -> str:
    """
    Run a read-only SQL query (must start with SELECT) against one container
    of the configured database and return ONE page of results; it never reads
    the whole result set.
    container: container name (see cosmos_list_containers).
    query: Cosmos DB SQL, e.g. "SELECT c.id, c.status FROM c WHERE c.status = @s".
    parameters: values for @names used in the query, e.g. {"@s": "open"}.
    partition_key: optional; restricts the query to one logical partition
    (a list for hierarchical keys). Without it the query fans out across all
    partitions, which costs more request units.
    max_item_count: page size, 1-100 (default 25).
    continuation_token: the continuation_token of the previous page to get the
    next one; omit for the first page.
    Cross-partition queries with ORDER BY, aggregates (COUNT, SUM...) or DISTINCT
    do not support continuation tokens (Microsoft's azure-cosmos README): narrow
    them with partition_key or a filter instead of paging.
    Returns items, request_charge (RU), continuation_token (null when there is
    no further page) and truncated. When the page is larger than the size cap,
    items are cut: truncated is true, omitted_count says how many were dropped
    and omitted_ids lists their ids; select fewer fields or lower
    max_item_count instead of paging on.
    """
    try:
        database_name = _database_name()
        container = _check_id(container, "container")
        if not isinstance(query, str) or not _SELECT.match(query):
            raise CosmosConfigError("query must be a SQL query starting with SELECT")
        if len(query) > MAX_QUERY_CHARS:
            raise CosmosConfigError(
                f"query must be at most {MAX_QUERY_CHARS} characters"
            )
        sql_parameters = _parameters(parameters)
        page_size = clamp_limit(max_item_count, max_limit=MAX_PAGE_SIZE)
        if continuation_token is not None and (
            not continuation_token.isprintable()
            or len(continuation_token) > MAX_CONTINUATION_TOKEN_CHARS
        ):
            raise CosmosConfigError("continuation_token is not valid")

        charges, record_charge = _charge_recorder()

        client = _client()
        try:
            container_client = client.get_database_client(
                database_name
            ).get_container_client(container)
            query_kwargs: dict[str, Any] = {
                "query": query,
                "parameters": sql_parameters or None,
                "max_item_count": page_size,
                "response_hook": record_charge,
            }
            if partition_key is None:
                query_kwargs["enable_cross_partition_query"] = True
            else:
                query_kwargs["partition_key"] = partition_key
            pager: Any = container_client.query_items(**query_kwargs).by_page(
                continuation_token or None
            )
            page = list(next(pager, iter(())))
            next_token = pager.continuation_token
        finally:
            client.close()

        kept, omitted = _fit_items(page)
        extra: dict[str, Any] = {}
        if omitted:
            extra = {
                "omitted_count": len(omitted),
                "omitted_ids": [
                    i["id"] for i in omitted if isinstance(i, dict) and "id" in i
                ][:MAX_OMITTED_IDS],
            }
        return _success(
            database=database_name,
            container=container,
            items=kept,
            count=len(kept),
            max_item_count=page_size,
            request_charge=round(sum(charges), 2),
            continuation_token=next_token or None,
            has_more=bool(next_token),
            truncated=bool(omitted),
            **extra,
        )
    except Exception as e:
        return _map_error(e, "the container")


@mcp.tool(annotations=READ_ONLY)
def cosmos_read_item(
    container: str,
    id: str,
    partition_key: str | int | float | bool | list[str | int | float | bool] | None,
) -> str:
    """
    Read one item by id and partition key from a container of the configured
    database (a point read: the cheapest read Cosmos DB offers).
    container: container name (see cosmos_list_containers).
    id: the item's id.
    partition_key: the item's partition key value (a list for hierarchical keys).
    An item over the size cap comes back as item_preview (the start of its JSON)
    with truncated true.
    """
    try:
        database_name = _database_name()
        container = _check_id(container, "container")
        item_id = _check_id(id, "id")
        charges, record_charge = _charge_recorder()

        client = _client()
        try:
            item = (
                client.get_database_client(database_name)
                .get_container_client(container)
                .read_item(
                    item_id, partition_key=partition_key, response_hook=record_charge
                )
            )
        finally:
            client.close()
        charge = round(sum(charges), 2)
        size = _serialized_size(item)
        if size > MAX_RESULT_BYTES:
            preview = json.dumps(item, ensure_ascii=False, default=str)[
                : MAX_RESULT_BYTES // 4
            ]
            return _success(
                database=database_name,
                container=container,
                item=None,
                item_preview=preview,
                size_bytes=size,
                request_charge=charge,
                truncated=True,
            )
        return _success(
            database=database_name,
            container=container,
            item=item,
            request_charge=charge,
            truncated=False,
        )
    except Exception as e:
        return _map_error(e, "the item")


if __name__ == "__main__":
    mcp.run()
