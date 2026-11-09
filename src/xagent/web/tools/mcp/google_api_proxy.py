"""An httplib2-compatible transport for googleapiclient on Composio grants.

A Composio-backed grant holds Composio's ids, not a Google token, so no
``Credentials(token=...)`` can be built. This transport sends every request
googleapiclient builds through Composio's Proxy Execute endpoint
(``composio_proxy.proxy_request``), which adds the user's token, and rebuilds
the ``(response, content)`` pair googleapiclient expects.

Pass an instance as ``http=`` to ``googleapiclient.discovery.build`` and pass
``static_discovery=True`` as well: the discovery document comes from
googleapiclient's bundled copy and is never fetched through the proxy.

A batch request is one ``multipart/mixed`` body, which the proxy does not
carry. ``execute_batch`` sends the batch's requests one at a time instead,
with bounded concurrency and an overall deadline, and hands the results to
the batch's own callbacks in the order the requests were added.
"""

from __future__ import annotations

import http.client
import json
import os
from collections.abc import Collection, Mapping
from concurrent.futures import ThreadPoolExecutor, wait
from typing import Any
from urllib.parse import parse_qsl, urlsplit

from googleapiclient.errors import HttpError  # type: ignore[import-not-found]

from .composio_proxy import (
    COMPOSIO_CONNECTED_ACCOUNT_ID_ENV,
    DEFAULT_TIMEOUT_SECONDS,
    proxy_request,
)

# How many batch requests may be in flight at once. A Gmail message search
# asks for 10 messages by default, so 8 keeps the common case to two waves
# without opening a connection per request when a caller asks for many more.
DEFAULT_BATCH_CONCURRENCY = 8
# The whole batch must finish inside this budget. It matches the per-call
# proxy timeout: without it a batch of N requests against an unresponsive
# proxy could hold the tool open for N/workers times the timeout.
DEFAULT_BATCH_DEADLINE_SECONDS = float(DEFAULT_TIMEOUT_SECONDS)

# The only request headers forwarded to the API. Both change what Google
# returns or how it reads the body. Content-Length and User-Agent describe
# the request we no longer make, Accept-Encoding would ask Google to compress
# a body Composio has to relay, and Composio injects the Authorization header
# itself.
_FORWARDED_HEADERS = ("accept", "content-type")

_JSON_CONTENT_TYPE = "application/json; charset=UTF-8"


def uses_composio_proxy() -> bool:
    """Whether this process serves a Composio-backed grant: it is launched
    with Composio's ids instead of a Google token."""
    return bool(os.environ.get(COMPOSIO_CONNECTED_ACCOUNT_ID_ENV))


class Response(dict[str, str]):
    """The part of ``httplib2.Response`` that googleapiclient reads."""

    def __init__(self, status: int, headers: Mapping[str, str]) -> None:
        super().__init__({"status": str(status), **headers})
        self.status = status
        self.reason = http.client.responses.get(status, "")
        self.version = 11
        self.fromcache = False


def _content(response_data: Any) -> tuple[bytes, str]:
    """The response body and content type for one proxied answer."""
    if response_data is None:
        return b"", _JSON_CONTENT_TYPE
    if isinstance(response_data, bytes):
        return response_data, "application/octet-stream"
    if isinstance(response_data, str):
        return response_data.encode("utf-8"), "text/plain; charset=UTF-8"
    return json.dumps(response_data).encode("utf-8"), _JSON_CONTENT_TYPE


class GoogleApiProxyHttp:
    """An httplib2-compatible ``http`` for googleapiclient on a Composio grant.

    ``allowed_hosts`` are the API's own hostnames. They are checked before
    anything is sent, so a request can never take Composio's injected token to
    a host this connector does not call.
    """

    def __init__(
        self,
        *,
        allowed_hosts: Collection[str],
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        if isinstance(allowed_hosts, str):
            raise TypeError(
                "allowed_hosts must be a collection of hostnames, not a str"
            )
        self._allowed_hosts = frozenset(host.lower() for host in allowed_hosts)
        self._timeout = timeout

    def request(
        self,
        uri: str,
        method: str = "GET",
        body: str | bytes | None = None,
        headers: Mapping[str, str] | None = None,
        redirections: int = 5,
        connection_type: Any = None,
    ) -> tuple[Response, bytes]:
        """Send one googleapiclient request through Composio.

        ``redirections`` and ``connection_type`` are accepted because
        httplib2's callers pass them; neither applies here. Composio follows
        no redirect for us, so a 3xx reaches googleapiclient as a 3xx.
        """
        request_headers = headers or {}
        parsed = urlsplit(uri)
        endpoint = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        query = parse_qsl(parsed.query, keep_blank_values=True)
        forwarded = {
            name: value
            for name, value in request_headers.items()
            if name.lower() in _FORWARDED_HEADERS
        }
        if _replays_a_get(request_headers):
            # googleapiclient turns a GET whose URI is too long into a POST
            # with the query form-encoded in the body. The proxy carries a
            # query as parameters and a body only as JSON, so restore the GET
            # instead of sending a form body it cannot carry.
            method = "GET"
            query.extend(parse_qsl(_form_text(body), keep_blank_values=True))
            forwarded = {
                name: value
                for name, value in forwarded.items()
                if name.lower() != "content-type"
            }
            proxied_body = None
        else:
            proxied_body = _json_body(body)
        answer = proxy_request(
            method,
            endpoint,
            allowed_hosts=self._allowed_hosts,
            query=query,
            headers=forwarded,
            body=proxied_body,
            timeout=self._timeout,
        )
        content, content_type = _content(answer.data)
        return (
            Response(
                answer.status,
                {
                    "content-type": content_type,
                    "content-length": str(len(content)),
                },
            ),
            content,
        )

    def close(self) -> None:
        """httplib2 closes its connection pool here; this transport has none."""


def _json_body(body: str | bytes | None) -> Any:
    """The request body as the JSON value the proxy's ``body`` takes."""
    # googleapiclient serializes JSON bodies to text. Media uploads pass
    # bytes, which the proxy cannot carry as JSON; refusing loudly beats
    # sending them as a JSON string.
    if body is None:
        return None
    if isinstance(body, (bytes, bytearray)):
        raise ValueError("the proxy transport only carries JSON request bodies")
    return json.loads(body)


def _replays_a_get(headers: Mapping[str, str]) -> bool:
    """Whether googleapiclient rewrote this request from a GET to a POST."""
    return any(
        name.lower() == "x-http-method-override" and value.upper() == "GET"
        for name, value in headers.items()
    )


def _form_text(body: str | bytes | None) -> str:
    if body is None:
        return ""
    if isinstance(body, str):
        return body
    return body.decode("utf-8")


def execute_batch(
    batch: Any,
    *,
    max_workers: int = DEFAULT_BATCH_CONCURRENCY,
    deadline: float = DEFAULT_BATCH_DEADLINE_SECONDS,
) -> None:
    """Run one ``BatchHttpRequest`` as individual proxied requests.

    ``BatchHttpRequest.execute`` posts a single ``multipart/mixed`` body,
    which the proxy does not carry. Each request the batch holds is sent on
    its own instead, with bounded concurrency, and its ``(request_id,
    response, exception)`` reaches the same per-request and global callbacks a
    native batch calls, in the order the requests were added.

    There is no credential refresh: a proxied request carries no OAuth
    credentials of ours, so a 401 is the provider's answer and reaches the
    callback as an error.

    A request still unanswered at ``deadline`` reaches its callback as a
    ``TimeoutError``; the batch's own threads are abandoned rather than
    awaited, so the caller returns inside its budget.
    """
    # BatchHttpRequest keeps its requests, their order and its callbacks in
    # these attributes and googleapiclient has no public accessor for them.
    order: list[str] = list(batch._order)
    if not order:
        return
    requests = batch._requests

    def run(request_id: str) -> tuple[Any, bytes] | BaseException:
        request = requests[request_id]
        http = request.http
        if http is None:
            return ValueError("Missing a valid http object.")
        try:
            answer: tuple[Any, bytes] = http.request(
                request.uri,
                method=request.method,
                body=request.body,
                headers=request.headers,
            )
        except Exception as exc:
            # A proxied call fails on its own: Composio may refuse one
            # request while the others are fine. Native batches only see
            # provider errors here; a transport error would have aborted them.
            return exc
        return answer

    outcomes: dict[str, tuple[Any, bytes] | BaseException] = {}
    executor = ThreadPoolExecutor(
        max_workers=max(1, min(max_workers, len(order))),
        thread_name_prefix="google-api-proxy",
    )
    try:
        futures = {request_id: executor.submit(run, request_id) for request_id in order}
        _, pending = wait(futures.values(), timeout=deadline)
        for future in pending:
            future.cancel()
        for request_id, future in futures.items():
            if future in pending:
                outcomes[request_id] = TimeoutError(
                    f"this request did not finish within {deadline:g}s "
                    "(the batch is bounded so the tool returns in time)"
                )
            else:
                outcomes[request_id] = future.result()
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    for request_id in order:
        request = requests[request_id]
        outcome = outcomes[request_id]
        response: Any = None
        exception: BaseException | None = None
        if isinstance(outcome, BaseException):
            exception = outcome
        else:
            raw_response, content = outcome
            try:
                if raw_response.status >= 300:
                    raise HttpError(raw_response, content, uri=request.uri)
                response = request.postproc(raw_response, content)
            except HttpError as exc:
                exception = exc
        callback = batch._callbacks[request_id]
        if callback is not None:
            callback(request_id, response, exception)
        if batch._callback is not None:
            batch._callback(request_id, response, exception)
