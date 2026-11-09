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
carry. ``execute_batch`` sends the batch's requests individually instead, with
bounded concurrency, a cap on the requests one batch may hold, and an overall
deadline, and hands the results to the batch's own callbacks in the order the
requests were added.

A binary answer (``ProxyResponse.binary_data``) is not in ``data``: Composio
stores the file and returns a signed URL. The transport downloads that URL and
returns its bytes, with the file's own content type and length.
"""

from __future__ import annotations

import http.client
import json
from collections.abc import Collection, Mapping
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qsl, urlsplit

from googleapiclient.errors import HttpError  # type: ignore[import-not-found]

from ....core.utils import public_requests as requests
from .composio_proxy import (
    DEFAULT_TIMEOUT_SECONDS,
    ComposioProxyError,
    ProxyResponse,
    check_json_body_size,
    proxy_request,
)

# How many batch requests may be in flight at once. A Gmail message search
# asks for 10 messages by default, so 8 keeps the common case to two waves and
# caps how many sessions are open at once when a caller asks for many more;
# each proxied call opens its own session.
DEFAULT_BATCH_CONCURRENCY = 8
# The whole batch must finish inside this budget. It matches the per-call
# proxy timeout: without it a batch of N requests against an unresponsive
# proxy could hold the tool open for N/workers times the timeout.
DEFAULT_BATCH_DEADLINE_SECONDS = float(DEFAULT_TIMEOUT_SECONDS)
# One tool call must not fan out into an unbounded number of proxy calls:
# the concurrency above limits only how many are in flight, not how many are
# made. 100 is Google's own per-batch request limit, so the fallback stays on
# the same scale as the native batch it replaces.
MAX_BATCH_REQUESTS = 100

# The only request headers forwarded to the API. Both change what Google
# returns or how it reads the body. Content-Length and User-Agent describe
# the request we no longer make, Accept-Encoding would ask Google to compress
# a body Composio has to relay, and Composio injects the Authorization header
# itself.
_FORWARDED_HEADERS = ("accept", "content-type")

_JSON_CONTENT_TYPE = "application/json; charset=UTF-8"


class Response(dict[str, str]):
    """The part of ``httplib2.Response`` that googleapiclient reads."""

    def __init__(self, status: int, headers: Mapping[str, str] | None = None) -> None:
        normalized = {
            str(name).lower(): str(value) for name, value in (headers or {}).items()
        }
        normalized["status"] = str(status)
        super().__init__(normalized)
        self.status = status
        self.reason = http.client.responses.get(status, "")
        self.version = 11
        self.fromcache = False


def _content(response_data: Any) -> tuple[bytes, str]:
    """The response body and content type for one proxied JSON answer."""
    if response_data is None:
        return b"", _JSON_CONTENT_TYPE
    if isinstance(response_data, str):
        return response_data.encode("utf-8"), "text/plain; charset=UTF-8"
    return json.dumps(response_data).encode("utf-8"), _JSON_CONTENT_TYPE


def _binary_content(
    binary_data: Mapping[str, Any], *, timeout: float
) -> tuple[bytes, str]:
    """The bytes behind a binary answer's signed URL, and their content type.

    Composio uploads a binary upstream body to its own storage and returns
    that signed URL in ``binary_data``; the URL expires, so it must not be
    kept, and it must not appear in an error either (it is a bearer-like
    credential for the file). ``public_requests`` keeps the download off a
    non-public destination.
    """
    url = binary_data.get("url")
    if not isinstance(url, str) or not url.startswith("https://"):
        raise ComposioProxyError(
            "the proxied binary response has no usable download URL"
        )
    try:
        response = requests.get(url, timeout=timeout)
    except Exception:
        # Type only: the exception's text can name the signed URL.
        raise ComposioProxyError(
            "the proxied binary response could not be downloaded"
        ) from None
    if not 200 <= response.status_code < 300:
        raise ComposioProxyError(
            "the proxied binary response could not be downloaded "
            f"(HTTP {response.status_code})"
        )
    content_type = binary_data.get("content_type")
    if not isinstance(content_type, str) or not content_type:
        content_type = "application/octet-stream"
    return response.content, content_type


def _body_and_type(answer: ProxyResponse, *, timeout: float) -> tuple[bytes, str]:
    """The body and content type of one provider answer.

    Gmail answers an empty success with 204, so a 2xx other than 204 that
    carries neither a body nor a file is an incomplete answer for this
    transport. Failing that one request is better than handing an empty body
    to a caller that reads it as a JSON document. The shared client leaves
    that judgment to each API: Salesforce, for one, reads an empty answer as
    an empty object.
    """
    if answer.binary_data is not None:
        return _binary_content(answer.binary_data, timeout=timeout)
    if answer.ok and answer.status != 204 and answer.data is None:
        raise ComposioProxyError(
            "Composio proxy returned an answer without the provider's data"
        )
    return _content(answer.data)


@dataclass(frozen=True)
class _PreparedRequest:
    """One googleapiclient request translated into a proxy call."""

    method: str
    endpoint: str
    query: list[tuple[str, str]]
    headers: dict[str, str]
    body: Any


class GoogleApiProxyHttp:
    """An httplib2-compatible ``http`` for googleapiclient on a Composio grant.

    ``allowed_hosts`` are the API's own hostnames. They are checked before
    anything is sent, so a request can never take Composio's injected token to
    a host this connector does not call.

    ``extra_headers`` names further request headers an API needs forwarded
    (for example a Drive resource key); every other header stays behind.

    ``max_body_bytes`` overrides the shared JSON body limit for this API, when
    a stage measurement shows its real limit differs.
    """

    def __init__(
        self,
        *,
        allowed_hosts: Collection[str],
        extra_headers: Collection[str] = (),
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        max_body_bytes: int | None = None,
    ) -> None:
        if isinstance(allowed_hosts, str):
            raise TypeError(
                "allowed_hosts must be a collection of hostnames, not a str"
            )
        self._allowed_hosts = frozenset(host.lower() for host in allowed_hosts)
        self._forwarded_headers = frozenset(_FORWARDED_HEADERS) | {
            header.lower() for header in extra_headers
        }
        self._timeout = timeout
        self._max_body_bytes = max_body_bytes

    def _prepare(
        self,
        uri: str,
        method: str,
        body: str | bytes | None,
        headers: Mapping[str, str] | None,
    ) -> _PreparedRequest:
        request_headers = headers or {}
        parsed = urlsplit(uri)
        endpoint = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        query = parse_qsl(parsed.query, keep_blank_values=True)
        forwarded = {
            name: value
            for name, value in request_headers.items()
            if name.lower() in self._forwarded_headers
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
        return _PreparedRequest(
            method=method,
            endpoint=endpoint,
            query=query,
            headers=forwarded,
            body=proxied_body,
        )

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
        httplib2's callers pass them; neither applies here. This transport
        does not follow a redirect: the status and the provider's headers
        (including ``location``) are returned, so the caller can follow it.
        """
        prepared = self._prepare(uri, method, body, headers)
        answer = proxy_request(
            prepared.method,
            prepared.endpoint,
            allowed_hosts=self._allowed_hosts,
            query=prepared.query,
            headers=prepared.headers,
            body=prepared.body,
            timeout=self._timeout,
            max_body_bytes=self._max_body_bytes,
        )
        content, content_type = _body_and_type(answer, timeout=self._timeout)
        response_headers = {
            str(name).lower(): str(value)
            for name, value in (answer.headers or {}).items()
        }
        response_headers["content-type"] = content_type
        response_headers["content-length"] = str(len(content))
        return Response(answer.status, response_headers), content

    def check_request(
        self,
        uri: str,
        method: str = "GET",
        body: str | bytes | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        """Refuse one request the proxy would reject for its JSON body size,
        without sending it.

        Sending several requests in one tool call (one message each) checks
        every request first, so an oversized later one cannot leave the
        earlier ones already delivered.
        """
        prepared = self._prepare(uri, method, body, headers)
        check_json_body_size(
            prepared.method,
            prepared.endpoint,
            query=prepared.query,
            headers=prepared.headers,
            body=prepared.body,
            max_body_bytes=self._max_body_bytes,
        )

    def close(self) -> None:
        """httplib2 closes its connection pool here; this transport has none."""


def precheck_body_size(request: Any) -> None:
    """Refuse one googleapiclient request the proxy would reject for its body
    size, before it is sent.

    A request on the native transport is left alone, so the native path is
    unchanged.
    """
    http = getattr(request, "http", None)
    if not isinstance(http, GoogleApiProxyHttp):
        return
    http.check_request(
        request.uri,
        method=request.method,
        body=request.body,
        headers=request.headers,
    )


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
    return (body.decode("utf-8") if isinstance(body, bytes) else body) or ""


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

    A batch above ``MAX_BATCH_REQUESTS`` is refused before any of it is sent,
    so one tool call cannot fan out into an unbounded number of proxy calls.

    A request still unanswered at ``deadline`` reaches its callback as a
    ``TimeoutError``; the batch's own threads are abandoned rather than
    awaited, so the caller returns inside its budget.
    """
    # BatchHttpRequest keeps its requests, their order and its callbacks in
    # these attributes and googleapiclient has no public accessor for them. A
    # release that renames one must fail with a readable message rather than
    # an AttributeError on every proxied batch.
    missing = [
        name
        for name in ("_order", "_requests", "_callbacks", "_callback")
        if not hasattr(batch, name)
    ]
    if missing:
        raise RuntimeError(
            "this google-api-python-client does not expose the batch "
            f"attributes the proxy fallback reads (missing {', '.join(missing)})"
        )
    order: list[str] = list(batch._order)
    if not order:
        return
    if len(order) > MAX_BATCH_REQUESTS:
        raise ValueError(
            f"this batch has {len(order)} requests, above the "
            f"{MAX_BATCH_REQUESTS} request limit for one proxied batch; "
            "ask for fewer at a time"
        )
    batch_requests = batch._requests

    def run(request_id: str) -> tuple[Any, bytes] | BaseException:
        request = batch_requests[request_id]
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
            # ``pending`` is a snapshot taken when ``wait`` returned: a
            # request can finish between that snapshot and this loop, and
            # its result must not be replaced by a made-up timeout.
            if future.cancelled() or (future in pending and not future.done()):
                outcomes[request_id] = TimeoutError(
                    f"this request did not finish within {deadline:g}s "
                    "(the batch is bounded so the tool returns in time)"
                )
            else:
                outcomes[request_id] = future.result()
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    for request_id in order:
        request = batch_requests[request_id]
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
            except Exception as exc:
                # A response that cannot be turned into a result fails that
                # request alone; the other callbacks have not run yet and must
                # still run.
                exception = exc
        callback = batch._callbacks[request_id]
        if callback is not None:
            callback(request_id, response, exception)
        if batch._callback is not None:
            batch._callback(request_id, response, exception)
