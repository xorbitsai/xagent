"""Azure Cosmos DB (NoSQL API) MCP connector, read-only, key authenticated.

Backed by Microsoft's ``azure-cosmos`` Python SDK (synchronous client: every
tool call runs in its own short-lived stdio subprocess, so there is nothing to
gain from the async client).

Configuration, all three required: ``COSMOS_ENDPOINT`` (the account URI from
the portal, ``https://<account>.documents.azure.com:443/``), ``COSMOS_KEY``
(an account key; a read-only key is enough and recommended) and
``COSMOS_DATABASE``. The container tools (list containers, query, read item)
are pinned to ``COSMOS_DATABASE`` and take no database argument. The account
key itself is account-wide, and ``cosmos_list_databases`` lists the whole
account.

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
tool result or a log line; failures are mapped to ``error_code`` values. Every
response is capped as a whole against ``get_tool_max_output_length()``, so the
paging fields at its end survive the platform's output filter.
"""

import json
import logging
import os
import re
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from itertools import islice
from typing import Any
from urllib.parse import urlsplit

from azure.core.exceptions import (
    AzureError,
    ServiceRequestError,
    ServiceRequestTimeoutError,
    ServiceResponseTimeoutError,
)
from azure.core.pipeline.transport import RequestsTransport
from azure.cosmos import CosmosClient
from azure.cosmos.exceptions import CosmosClientTimeoutError, CosmosHttpResponseError
from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from ....config import get_tool_max_output_length
from ....core.utils import public_requests
from ....core.utils.security import redact_sensitive_text
from .utils import clamp_limit, success_with_capped_dict

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
# Characters Cosmos DB resource ids may not contain: / \ ? # (Microsoft Learn,
# ``Resource.Id`` in the SDK reference); control characters are refused too.
_ID_FORBIDDEN = re.compile(r"[/\\?#\x00-\x1f\x7f]")
_PARAM_NAME = re.compile(r"\A@?([A-Za-z_][A-Za-z0-9_]{0,63})\Z")
_SELECT = re.compile(r"\A\s*SELECT\b", re.IGNORECASE)
_STRING_LITERAL = re.compile(r"'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"")
# Clauses that make the Python SDK merge partial results on the client, so
# the token of the last backend response cannot resume a cross-partition query.
_NON_RESUMABLE = re.compile(
    r"\b(?:ORDER\s+BY|GROUP\s+BY|DISTINCT|TOP|OFFSET)\b"
    r"|\b(?:COUNT|SUM|AVG|MIN|MAX)\s*\(",
    re.IGNORECASE,
)

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
# Limits from "Azure Cosmos DB service quotas": database and container names
# 255 characters, item ids 1,023 bytes.
MAX_NAME_CHARS = 255
MAX_ITEM_ID_BYTES = 1023
MAX_OMITTED_IDS = 50

_REQUEST_CHARGE_HEADER = "x-ms-request-charge"
_RETRY_AFTER_HEADER = "x-ms-retry-after-ms"
_KEY_AUTH_DISABLED_MARKER = "local authorization is disabled"


class CosmosConfigError(ValueError):
    """A problem with the connector's own input; the message is ours, safe to show."""


class BlockedDestinationError(Exception):
    """A request was refused because its host is not an allowed Cosmos DB host."""


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
    if "@" in parts.netloc:
        raise CosmosConfigError("COSMOS_ENDPOINT must not contain credentials")
    if "?" in value or "#" in value:
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
        try:
            validate_endpoint(f"{parts.scheme}://{parts.netloc}/")
        except CosmosConfigError:
            logger.warning(
                "Refused a request to a host outside the Cosmos DB allowlist: %s",
                parts.hostname,
            )
            raise BlockedDestinationError from None
        return super().send(request, **kwargs)


def _client() -> CosmosClient:
    endpoint = validate_endpoint(_env("COSMOS_ENDPOINT"))
    key = _env("COSMOS_KEY")
    transport = _GuardedTransport()
    try:
        return CosmosClient(
            endpoint,
            credential=key,
            transport=transport,
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
    except BaseException:
        # The constructor does network I/O (the account read); do not leak the
        # transport's session when it fails.
        transport.close()
        raise


@contextmanager
def _open_client() -> Iterator[CosmosClient]:
    client = _client()
    try:
        yield client
    finally:
        client.close()


def _database_name() -> str:
    return _check_name(_env("COSMOS_DATABASE"), "COSMOS_DATABASE")


def _check_name(value: Any, name: str) -> str:
    """A database or container name (255 characters at most)."""
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > MAX_NAME_CHARS
        or value in (".", "..")
        or _ID_FORBIDDEN.search(value)
    ):
        raise CosmosConfigError(
            f"{name} must be a non-empty name of at most {MAX_NAME_CHARS} characters, "
            "not '.' or '..', without / \\ ? # or control characters"
        )
    return value


def _check_item_id(value: Any) -> str:
    """An item id (1,023 bytes at most). ``.`` and ``..`` are dot segments that
    URL normalization removes from the request path, so they could never match
    the signed resource."""
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value.encode("utf-8")) > MAX_ITEM_ID_BYTES
        or value in (".", "..")
        or _ID_FORBIDDEN.search(value)
    ):
        raise CosmosConfigError(
            f"id must be a non-empty item id of at most {MAX_ITEM_ID_BYTES} bytes, "
            "not '.' or '..', without / \\ ? # or control characters"
        )
    return value


def _short(value: str) -> str:
    return value if len(value) <= 80 else value[:77] + "..."


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


def _map_error(exc: Exception, target: str, not_found: str) -> str:
    """Map an exception to an actionable envelope without echoing its text.

    ``target`` names the call for the log; ``not_found`` is the full 404
    message for this call site.
    """
    if isinstance(exc, CosmosConfigError):
        return _error("invalid_configuration", str(exc))
    if isinstance(exc, BlockedDestinationError):
        return _error(
            "blocked_destination",
            "The request was refused because it targeted a host that is not an "
            "Azure Cosmos DB account host.",
        )
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
            return _error("not_found", not_found, http_status=404)
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
    if isinstance(
        exc,
        (
            CosmosClientTimeoutError,
            ServiceRequestTimeoutError,
            ServiceResponseTimeoutError,
        ),
    ):
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


def _success_with_capped_list(
    field: str, payload: dict[str, Any], hint: str | None = None
) -> str:
    """A success payload, halving ``payload[field]`` until the *whole* response
    fits ``get_tool_max_output_length()``.

    The platform's output filter slices an oversized response blindly, which
    would drop the paging fields that follow the list, so the cap covers the
    entire JSON. Halved entries belong to the page already fetched and are not
    on the next page: their ids are returned (``omitted_ids``) so a caller can
    still fetch each one, even when one record alone is larger than the limit.
    The paging fields (``continuation_token``, ``has_more``) always survive,
    and the token already points past the page, so a retry makes progress.
    """
    max_output_length = max(64, get_tool_max_output_length())
    items = payload[field]
    rest = {k: v for k, v in payload.items() if k != "truncated"}
    already_truncated = bool(payload.get("truncated"))

    def build(kept: list[Any]) -> str:
        if len(kept) == len(items):
            return _success(**rest, truncated=already_truncated)
        omitted = items[len(kept) :]
        ids = [i["id"] for i in omitted if isinstance(i, dict) and "id" in i]
        return _success(
            **{**rest, field: kept},
            truncated=True,
            omitted_count=len(omitted),
            omitted_ids=ids[:MAX_OMITTED_IDS],
            message=hint
            or (
                f"Returned {len(kept)} of {len(items)} {field}; the rest did not "
                "fit the output size limit and are not on the next page. Fetch "
                "each omitted id with cosmos_read_item, select fewer fields, or "
                "lower max_item_count."
            ),
        )

    kept = items
    response = build(kept)
    while len(response) > max_output_length and kept:
        kept = kept[: len(kept) // 2]
        response = build(kept)
    if len(response) > max_output_length:
        # Even the envelope plus ids does not fit (a very small configured
        # limit): keep it valid JSON and say how many were left out.
        response = _success(
            **{**rest, field: []}, truncated=True, omitted_count=len(items)
        )
    return response


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


def _is_resumable_across_partitions(query: str) -> bool:
    """Whether a continuation token can resume this query across partitions.

    Microsoft Learn: in the Python SDK aggregate cross-partition queries
    (sorting, counting, distinct) do not support continuation tokens, and
    GROUP BY / DISTINCT cannot use them at the service. TOP and OFFSET are
    merged client-side the same way (read from the SDK source). A best-effort
    check on the query text with string literals removed.
    """
    return not _NON_RESUMABLE.search(_STRING_LITERAL.sub("''", query))


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
        with _open_client() as client:
            databases, truncated = _bounded(
                client.list_databases(
                    max_item_count=MAX_LIST_ITEMS + 1, timeout=TOTAL_TIMEOUT_SECONDS
                )
            )
        return _success_with_capped_list(
            "databases",
            {
                "databases": [d.get("id") for d in databases],
                "configured_database": database_name,
                "truncated": truncated,
            },
            hint="The list did not fit the output size limit.",
        )
    except Exception as e:
        return _map_error(
            e,
            "the account",
            "The Cosmos DB account was not found (404). Check COSMOS_ENDPOINT.",
        )


@mcp.tool(annotations=READ_ONLY)
def cosmos_list_containers() -> str:
    """
    List the containers of the configured database with their partition key
    paths. Also confirms that the endpoint, key, network access and database
    name all work (use it first if any other Cosmos call fails).
    """
    database_name = "the configured database"
    try:
        database_name = _database_name()
        with _open_client() as client:
            database = client.get_database_client(database_name)
            # One metadata read: the database must exist.
            database.read(timeout=TOTAL_TIMEOUT_SECONDS)
            containers, truncated = _bounded(
                database.list_containers(
                    max_item_count=MAX_LIST_ITEMS + 1, timeout=TOTAL_TIMEOUT_SECONDS
                )
            )
        return _success_with_capped_list(
            "containers",
            {
                "database": database_name,
                "containers": [
                    {
                        "id": c.get("id"),
                        "partition_key_paths": (c.get("partitionKey") or {}).get(
                            "paths"
                        ),
                    }
                    for c in containers
                ],
                "truncated": truncated,
            },
            hint="The list did not fit the output size limit.",
        )
    except Exception as e:
        return _map_error(
            e,
            "the configured database",
            f"Database {_short(database_name)!r} was not found (404). "
            "Check COSMOS_DATABASE.",
        )


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
    (a list for hierarchical keys). Omit it (null) for a query across all
    partitions, which costs more request units.
    max_item_count: page size, 1-100 (default 25).
    continuation_token: the continuation_token of the previous page to get the
    next one; omit for the first page.
    Returns items, request_charge (RU), continuation_token (null on the last
    page), has_more and truncated. When the response is larger than the output
    size limit, items are cut: truncated is true, omitted_count says how many
    were dropped and omitted_ids lists their ids (fetch them with
    cosmos_read_item, or select fewer fields); the continuation_token still
    points past this page.
    A query across partitions (no partition_key) that uses ORDER BY, GROUP BY,
    DISTINCT, TOP, OFFSET or an aggregate (COUNT, SUM, AVG, MIN, MAX) cannot be
    resumed with a token: the result has no continuation_token, resumable is
    false and may_be_incomplete says whether the page was full; a token passed
    with such a query is rejected. Narrow it with partition_key or a filter, or
    use a streamable query (SELECT ... WHERE ...) to page.
    """
    container_name = "the container"
    try:
        database_name = _database_name()
        container_name = _check_name(container, "container")
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
        resumable = partition_key is not None or _is_resumable_across_partitions(query)
        if continuation_token and not resumable:
            raise CosmosConfigError(
                "this query cannot be resumed with a continuation token across "
                "partitions (ORDER BY, GROUP BY, DISTINCT, TOP, OFFSET or an "
                "aggregate): pass partition_key or narrow the query"
            )

        charges, record_charge = _charge_recorder()
        with _open_client() as client:
            container_client = client.get_database_client(
                database_name
            ).get_container_client(container_name)
            query_kwargs: dict[str, Any] = {
                "query": query,
                "parameters": sql_parameters or None,
                "max_item_count": page_size,
                "response_hook": record_charge,
                "timeout": TOTAL_TIMEOUT_SECONDS,
            }
            if partition_key is None:
                query_kwargs["enable_cross_partition_query"] = True
            else:
                query_kwargs["partition_key"] = partition_key
            pager: Any = container_client.query_items(**query_kwargs).by_page(
                continuation_token or None
            )
            try:
                page = list(next(pager))
                next_token = pager.continuation_token
            except StopIteration:
                # No more results. The pager still holds the token the caller
                # sent, which must not be echoed back.
                page, next_token = [], None
        if next_token == continuation_token:
            next_token = None  # no progress: never invite a loop
        if not resumable:
            next_token = None

        payload: dict[str, Any] = {
            "database": database_name,
            "container": container_name,
            "items": page,
            "max_item_count": page_size,
            "request_charge": round(sum(charges), 2),
            "continuation_token": next_token or None,
            "has_more": bool(next_token),
        }
        if not resumable:
            payload["resumable"] = False
            payload["may_be_incomplete"] = len(page) >= page_size
        return _success_with_capped_list("items", payload)
    except Exception as e:
        return _map_error(
            e,
            "the container",
            f"Container {_short(container_name)!r} or the configured database was "
            "not found (404).",
        )


@mcp.tool(annotations=READ_ONLY)
def cosmos_read_item(
    container: str,
    id: str,
    partition_key: str | int | float | bool | list[str | int | float | bool],
) -> str:
    """
    Read one item by id and partition key from a container of the configured
    database (a point read: the cheapest read Cosmos DB offers).
    container: container name (see cosmos_list_containers).
    id: the item's id.
    partition_key: the item's partition key value (a list for hierarchical
    keys). Required; an item with a null partition key cannot be read here.
    An item larger than the output size limit is trimmed field by field and
    truncated is true.
    """
    container_name = "the container"
    item_id = "the item"
    try:
        database_name = _database_name()
        container_name = _check_name(container, "container")
        item_id = _check_item_id(id)
        charges, record_charge = _charge_recorder()
        with _open_client() as client:
            item = (
                client.get_database_client(database_name)
                .get_container_client(container_name)
                .read_item(
                    item_id,
                    partition_key=partition_key,
                    response_hook=record_charge,
                    timeout=TOTAL_TIMEOUT_SECONDS,
                )
            )
        return success_with_capped_dict(
            "item",
            item,
            critical_fields={
                "database": database_name,
                "container": container_name,
                "request_charge": round(sum(charges), 2),
            },
        )
    except Exception as e:
        return _map_error(
            e,
            "the item",
            f"Item {_short(item_id)!r} was not found in container "
            f"{_short(container_name)!r} (404). Check the id and the partition key "
            "value, and that the container exists in the configured database.",
        )


if __name__ == "__main__":
    mcp.run()
