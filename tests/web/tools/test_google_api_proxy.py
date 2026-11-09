"""The googleapiclient proxy transport for Composio-backed Google grants.

Composio is faked here; no request leaves the process.
"""

from __future__ import annotations

import json
import threading
import time
from typing import Any

import pytest
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from xagent.web.tools.mcp import composio_proxy, google_api_proxy

API_KEY = "test-composio-api-key-value"
ACCOUNT_ID = "ca_gmail_account_1"
COMPOSIO_USER = "xagent-0123456789abcdef"
GMAIL_HOSTS = ("gmail.googleapis.com",)


class _Response:
    def __init__(self, status_code: int, payload: Any = None) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> Any:
        if self._payload is None:
            raise ValueError("no JSON body")
        return self._payload


class FakeProxy:
    """Answers by endpoint substring; records every proxy payload."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.by_endpoint: list[tuple[str, _Response]] = []
        self.default = _Response(200, {"status": 200, "data": {}})
        self.lock = threading.Lock()

    def answer(self, needle: str, status: int, data: Any = None) -> None:
        self.by_endpoint.append(
            (needle, _Response(200, {"status": status, "data": data}))
        )

    def answer_with(self, status_code: int, payload: Any) -> None:
        self.default = _Response(status_code, payload)

    def request(self, **kwargs: Any) -> _Response:
        assert kwargs["url"] == composio_proxy.COMPOSIO_PROXY_URL
        payload = kwargs["json"]
        with self.lock:
            self.calls.append(payload)
        for needle, response in self.by_endpoint:
            if needle in payload["endpoint"]:
                return response
        return self.default


@pytest.fixture
def proxy(monkeypatch) -> FakeProxy:
    monkeypatch.setenv(composio_proxy.COMPOSIO_API_KEY_ENV, API_KEY)
    monkeypatch.setenv(composio_proxy.COMPOSIO_CONNECTED_ACCOUNT_ID_ENV, ACCOUNT_ID)
    monkeypatch.setenv(composio_proxy.COMPOSIO_USER_ID_ENV, COMPOSIO_USER)
    fake = FakeProxy()
    monkeypatch.setattr(composio_proxy.requests, "request", fake.request)
    return fake


def _http() -> google_api_proxy.GoogleApiProxyHttp:
    return google_api_proxy.GoogleApiProxyHttp(allowed_hosts=GMAIL_HOSTS)


def _secrets_in(text: str) -> list[str]:
    return [value for value in (API_KEY, ACCOUNT_ID, COMPOSIO_USER) if value in text]


def _gmail_service() -> Any:
    return build("gmail", "v1", http=_http(), static_discovery=True)


def test_a_request_is_mapped_to_one_proxy_call(proxy):
    http = _http()

    http.request(
        "https://gmail.googleapis.com/gmail/v1/users/me/messages"
        "?maxResults=10&labelIds=INBOX&labelIds=STARRED",
        method="POST",
        body='{"raw": "abc"}',
        headers={"content-type": "application/json", "accept": "application/json"},
    )

    (call,) = proxy.calls
    assert call == {
        "endpoint": "https://gmail.googleapis.com/gmail/v1/users/me/messages",
        "method": "POST",
        "connected_account_id": ACCOUNT_ID,
        "user_id": COMPOSIO_USER,
        "parameters": [
            {"name": "maxResults", "value": "10", "type": "query"},
            {"name": "labelIds", "value": "INBOX", "type": "query"},
            {"name": "labelIds", "value": "STARRED", "type": "query"},
            {"name": "content-type", "value": "application/json", "type": "header"},
            {"name": "accept", "value": "application/json", "type": "header"},
        ],
        "body": {"raw": "abc"},
    }


def test_a_get_googleapiclient_rewrote_to_a_post_is_sent_as_a_get(proxy):
    """A GET whose URI is too long is sent as a POST with the query
    form-encoded in the body. The proxy carries a query as parameters and a
    body only as JSON, so the query goes back on the URI."""
    long_query = "x" * 2200

    _gmail_service().users().messages().list(userId="me", q=long_query).execute()

    (call,) = proxy.calls
    assert call["method"] == "GET"
    assert "body" not in call
    assert {"name": "q", "value": long_query, "type": "query"} in call["parameters"]
    assert all(item["name"] != "content-type" for item in call["parameters"])


def test_only_headers_that_change_the_answer_are_forwarded(proxy):
    """Content-Length and User-Agent describe a request we no longer make,
    Accept-Encoding would ask for a body the proxy has to relay, and the
    token is Composio's to inject."""
    http = _http()

    http.request(
        "https://gmail.googleapis.com/gmail/v1/users/me/messages/send",
        method="POST",
        body='{"raw": "abc"}',
        headers={
            "content-type": "application/json",
            "accept": "application/json",
            "content-length": "13",
            "accept-encoding": "gzip, deflate",
            "user-agent": "(gzip)",
            "authorization": "Bearer must-not-travel",
        },
    )

    (call,) = proxy.calls
    forwarded = {item["name"] for item in call["parameters"]}
    assert forwarded == {"content-type", "accept"}
    assert "must-not-travel" not in json.dumps(call)


def test_a_named_extra_header_is_forwarded(proxy):
    """A connector can name the headers its API needs; everything else,
    including a token, still stays behind."""
    http = google_api_proxy.GoogleApiProxyHttp(
        allowed_hosts=GMAIL_HOSTS,
        extra_headers=("X-Goog-Drive-Resource-Keys",),
    )

    http.request(
        "https://gmail.googleapis.com/gmail/v1/users/me/labels",
        headers={
            "X-Goog-Drive-Resource-Keys": "key-1/key-2",
            "authorization": "Bearer must-not-travel",
        },
    )

    (call,) = proxy.calls
    assert {
        "name": "X-Goog-Drive-Resource-Keys",
        "value": "key-1/key-2",
        "type": "header",
    } in call["parameters"]
    assert all(item["name"] != "authorization" for item in call["parameters"])


def test_a_response_is_mapped_back_to_status_and_content(proxy):
    proxy.answer_with(200, {"status": 200, "data": {"id": "m1"}})

    response, content = _http().request(
        "https://gmail.googleapis.com/gmail/v1/users/me/messages/m1"
    )

    assert response.status == 200
    assert response.reason == "OK"
    assert response["content-type"] == "application/json; charset=UTF-8"
    assert json.loads(content) == {"id": "m1"}


def test_a_no_content_answer_has_an_empty_body(proxy):
    proxy.answer_with(200, {"status": 204, "data": None})

    response, content = _http().request(
        "https://gmail.googleapis.com/gmail/v1/users/me/labels/xyz",
        method="DELETE",
    )

    assert response.status == 204
    assert content == b""


@pytest.mark.parametrize(
    "data, expected, content_type",
    [
        ("%PDF-1.4 bytes", b"%PDF-1.4 bytes", "text/plain; charset=UTF-8"),
        (b"%PDF-1.4 bytes", b"%PDF-1.4 bytes", "application/octet-stream"),
        (None, b"", "application/json; charset=UTF-8"),
    ],
    ids=["text", "bytes", "empty"],
)
def test_a_non_json_answer_is_returned_as_bytes(proxy, data, expected, content_type):
    """A later media download reuses this transport; its answer may not be
    JSON."""
    proxy.answer_with(200, {"status": 200, "data": data})

    response, content = _http().request(
        "https://gmail.googleapis.com/gmail/v1/users/me/messages/m1/attachments/x"
    )

    assert content == expected
    assert response["content-type"] == content_type


def test_an_upstream_error_keeps_its_status_and_body(proxy):
    error = {"error": {"code": 404, "message": "Requested entity was not found."}}
    proxy.answer_with(200, {"status": 404, "data": error})

    response, content = _http().request(
        "https://gmail.googleapis.com/gmail/v1/users/me/labels/missing"
    )

    assert response.status == 404
    assert response.reason == "Not Found"
    assert json.loads(content) == error


def test_a_redirect_is_handed_back_not_followed(proxy):
    proxy.answer_with(200, {"status": 302, "data": None})

    response, _ = _http().request(
        "https://gmail.googleapis.com/gmail/v1/users/me/labels"
    )

    assert response.status == 302


def test_an_upstream_error_becomes_the_googleapiclient_error(proxy):
    """googleapiclient, not the transport, turns the status into HttpError."""
    proxy.answer_with(
        200,
        {
            "status": 404,
            "data": {
                "error": {
                    "code": 404,
                    "message": "Requested entity was not found.",
                    "errors": [{"message": "nf", "domain": "global"}],
                }
            },
        },
    )

    with pytest.raises(HttpError) as raised:
        _gmail_service().users().labels().get(userId="me", id="x").execute()

    message = str(raised.value)
    assert "HttpError 404" in message
    assert "Requested entity was not found." in message


@pytest.mark.parametrize(
    "status, slug",
    [(401, "InvalidAPIKey"), (404, "ConnectedAccount_NotFound"), (429, None)],
)
def test_a_composio_refusal_raises_and_names_no_secret(proxy, status, slug):
    echoed = f"account {ACCOUNT_ID} of user {COMPOSIO_USER} key {API_KEY}"
    error: dict[str, Any] = {"message": echoed}
    if slug:
        error["slug"] = slug
    proxy.answer_with(status, {"error": error})

    with pytest.raises(composio_proxy.ComposioProxyError) as raised:
        _http().request("https://gmail.googleapis.com/gmail/v1/users/me/labels")

    assert raised.value.status == status
    if slug:
        assert slug in str(raised.value)
    assert _secrets_in(str(raised.value)) == []
    assert _secrets_in(repr(raised.value.args)) == []


@pytest.mark.parametrize(
    "uri",
    [
        "http://gmail.googleapis.com/gmail/v1/users/me/labels",
        "https://attacker.example/gmail/v1/users/me/labels",
        "https://gmail.googleapis.com.attacker.example/gmail/v1",
        "https://user@gmail.googleapis.com/gmail/v1",
        "https://gmail.googleapis.com:8443/gmail/v1",
    ],
    ids=["http", "other-host", "lookalike-host", "credentials", "port"],
)
def test_an_endpoint_off_the_allowed_hosts_is_refused_before_sending(proxy, uri):
    with pytest.raises(ValueError):
        _http().request(uri)

    assert proxy.calls == []


def test_allowed_hosts_given_as_one_string_is_refused():
    with pytest.raises(TypeError):
        google_api_proxy.GoogleApiProxyHttp(allowed_hosts="gmail.googleapis.com")


@pytest.mark.parametrize("body", [b"\x00\x01", "not json"], ids=["bytes", "not-json"])
def test_a_body_that_is_not_json_is_refused_before_sending(proxy, body):
    with pytest.raises(ValueError):
        _http().request(
            "https://gmail.googleapis.com/gmail/v1/users/me/messages",
            method="POST",
            body=body,
        )

    assert proxy.calls == []


def _added_batch(service: Any, ids: list[str], callback) -> Any:
    batch = service.new_batch_http_request(callback=callback)
    for message_id in ids:
        batch.add(
            service.users()
            .messages()
            .get(
                userId="me",
                id=message_id,
                format="metadata",
                metadataHeaders=["Subject", "From"],
            )
        )
    return batch


def test_a_batch_calls_the_same_callbacks_in_request_order(proxy):
    """The native batch calls every callback in the order its requests were
    added; the fallback must not reorder them by completion."""
    proxy.answer("/messages/m1", 200, {"id": "m1"})
    proxy.answer("/messages/m2", 404, {"error": {"code": 404, "message": "nf"}})
    proxy.answer("/messages/m3", 200, {"id": "m3"})
    seen: list[tuple[str, Any, Any]] = []

    def callback(request_id, response, exception):
        seen.append((request_id, response, exception))

    google_api_proxy.execute_batch(
        _added_batch(_gmail_service(), ["m1", "m2", "m3"], callback)
    )

    assert [item[0] for item in seen] == ["1", "2", "3"]
    assert seen[0][1] == {"id": "m1"}
    assert seen[1][1] is None
    assert "HttpError 404" in str(seen[1][2])
    assert seen[2][1] == {"id": "m3"}
    assert all(item[2] is None for item in (seen[0], seen[2]))


def test_a_batch_request_keeps_the_query_googleapiclient_built(proxy):
    proxy.answer("/messages/m1", 200, {"id": "m1"})

    google_api_proxy.execute_batch(
        _added_batch(_gmail_service(), ["m1"], lambda *_args: None)
    )

    (call,) = proxy.calls
    assert call["endpoint"] == (
        "https://gmail.googleapis.com/gmail/v1/users/me/messages/m1"
    )
    assert call["method"] == "GET"
    assert call["parameters"] == [
        {"name": "format", "value": "metadata", "type": "query"},
        {"name": "metadataHeaders", "value": "Subject", "type": "query"},
        {"name": "metadataHeaders", "value": "From", "type": "query"},
        {"name": "alt", "value": "json", "type": "query"},
        {"name": "accept", "value": "application/json", "type": "header"},
    ]


def test_a_batch_runs_requests_concurrently_but_bounded(monkeypatch):
    monkeypatch.setenv(composio_proxy.COMPOSIO_API_KEY_ENV, API_KEY)
    monkeypatch.setenv(composio_proxy.COMPOSIO_CONNECTED_ACCOUNT_ID_ENV, ACCOUNT_ID)
    monkeypatch.setenv(composio_proxy.COMPOSIO_USER_ID_ENV, COMPOSIO_USER)
    in_flight = 0
    peak = 0
    lock = threading.Lock()

    def slow(**kwargs: Any) -> _Response:
        nonlocal in_flight, peak
        with lock:
            in_flight += 1
            peak = max(peak, in_flight)
        time.sleep(0.05)
        with lock:
            in_flight -= 1
        return _Response(200, {"status": 200, "data": {}})

    monkeypatch.setattr(composio_proxy.requests, "request", slow)

    batch = _added_batch(
        _gmail_service(), [str(number) for number in range(8)], lambda *_args: None
    )

    google_api_proxy.execute_batch(batch, max_workers=2)

    assert peak <= 2, "bounded concurrency must cap the in-flight requests"
    assert peak > 1, "the requests must run concurrently"


def test_a_request_unanswered_at_the_deadline_reaches_its_callback(monkeypatch):
    monkeypatch.setenv(composio_proxy.COMPOSIO_API_KEY_ENV, API_KEY)
    monkeypatch.setenv(composio_proxy.COMPOSIO_CONNECTED_ACCOUNT_ID_ENV, ACCOUNT_ID)
    monkeypatch.setenv(composio_proxy.COMPOSIO_USER_ID_ENV, COMPOSIO_USER)
    seen: list[tuple[str, Any, Any]] = []

    def slow(**kwargs: Any) -> _Response:
        time.sleep(0.5)
        return _Response(200, {"status": 200, "data": {}})

    monkeypatch.setattr(composio_proxy.requests, "request", slow)

    def callback(request_id, response, exception):
        seen.append((request_id, response, exception))

    batch = _added_batch(_gmail_service(), ["m1"], callback)
    started = time.monotonic()

    google_api_proxy.execute_batch(batch, deadline=0.05)

    assert time.monotonic() - started < 0.4, "the batch must return at its deadline"
    (request_id, response, exception) = seen[0]
    assert request_id == "1"
    assert response is None
    assert isinstance(exception, TimeoutError)


def test_one_composio_failure_does_not_lose_the_other_requests(proxy):
    proxy.answer("/messages/m1", 200, {"id": "m1"})
    proxy.answer_with(
        500,
        {"error": {"message": f"account {ACCOUNT_ID} failed"}},
    )
    seen: list[tuple[str, Any, Any]] = []

    def callback(request_id, response, exception):
        seen.append((request_id, response, exception))

    google_api_proxy.execute_batch(
        _added_batch(_gmail_service(), ["m1", "m2"], callback)
    )

    assert seen[0][1] == {"id": "m1"}
    assert isinstance(seen[1][2], composio_proxy.ComposioProxyError)
    assert _secrets_in(str(seen[1][2])) == []


def test_a_batch_with_no_requests_sends_nothing(proxy):
    service = _gmail_service()

    google_api_proxy.execute_batch(service.new_batch_http_request())

    assert proxy.calls == []


class _Download:
    """The public_requests response shape a binary download returns."""

    def __init__(self, status_code: int, content: bytes) -> None:
        self.status_code = status_code
        self.content = content


def test_a_binary_answer_is_downloaded_from_its_signed_url(proxy, monkeypatch):
    """Composio does not relay the bytes: it stores the file and answers
    ``data=None`` plus ``binary_data``. The transport must return the bytes,
    not an empty body that a downloader would treat as a finished 0-byte
    file."""
    seen: dict[str, Any] = {}

    def download(url: str, timeout: float) -> _Download:
        seen["url"] = url
        seen["timeout"] = timeout
        return _Download(200, b"%PDF-1.4 real bytes")

    monkeypatch.setattr(google_api_proxy.requests, "get", download)
    proxy.answer_with(
        200,
        {
            "status": 200,
            "data": None,
            "binary_data": {
                "url": "https://files.composio.dev/signed",
                "content_type": "application/pdf",
                "size": 18,
            },
            "headers": {"x-goog-generation": "7"},
        },
    )

    response, content = _http().request(
        "https://gmail.googleapis.com/gmail/v1/users/me/messages/m1/attachments/a"
    )

    assert seen["url"] == "https://files.composio.dev/signed"
    assert content == b"%PDF-1.4 real bytes"
    assert response["content-type"] == "application/pdf"
    assert response["content-length"] == str(len(content))
    # The provider's own response headers survive too.
    assert response["x-goog-generation"] == "7"


def test_a_binary_answer_without_a_usable_url_is_refused(proxy, monkeypatch):
    def download(url: str, timeout: float) -> _Download:
        raise AssertionError("no download without a usable URL")

    monkeypatch.setattr(google_api_proxy.requests, "get", download)
    proxy.answer_with(
        200,
        {
            "status": 200,
            "data": None,
            "binary_data": {"content_type": "application/pdf", "size": 3},
        },
    )

    with pytest.raises(composio_proxy.ComposioProxyError) as raised:
        _http().request("https://gmail.googleapis.com/gmail/v1/users/me/messages/m1")

    assert "download URL" in str(raised.value)


def test_a_failed_binary_download_names_no_signed_url(proxy, monkeypatch):
    def download(url: str, timeout: float) -> _Download:
        return _Download(403, b"denied")

    monkeypatch.setattr(google_api_proxy.requests, "get", download)
    proxy.answer_with(
        200,
        {
            "status": 200,
            "data": None,
            "binary_data": {
                "url": "https://files.composio.dev/signed-secret",
                "content_type": "application/pdf",
                "size": 3,
            },
        },
    )

    with pytest.raises(composio_proxy.ComposioProxyError) as raised:
        _http().request("https://gmail.googleapis.com/gmail/v1/users/me/messages/m1")

    assert "HTTP 403" in str(raised.value)
    assert "files.composio.dev" not in str(raised.value)
    assert "signed-secret" not in str(raised.value)


def test_a_redirect_keeps_its_location_header(proxy):
    """A 3xx is handed back with the provider's Location so a caller can
    follow it; the transport does not silently follow or drop it."""
    proxy.answer_with(
        200,
        {
            "status": 302,
            "data": None,
            "headers": {
                "Location": "https://drive.usercontent.google.com/download/x",
                "X-Goog-Request-Id": "r1",
            },
        },
    )

    response, content = _http().request(
        "https://gmail.googleapis.com/gmail/v1/users/me/messages/m1/attachments/a"
    )

    assert response.status == 302
    assert response["location"] == "https://drive.usercontent.google.com/download/x"
    assert response["x-goog-request-id"] == "r1"
    assert content == b""


def test_a_batch_above_the_request_cap_is_refused_before_sending(proxy):
    ids = [str(number) for number in range(google_api_proxy.MAX_BATCH_REQUESTS + 1)]

    with pytest.raises(ValueError, match="request limit"):
        google_api_proxy.execute_batch(
            _added_batch(_gmail_service(), ids, lambda *_args: None)
        )

    assert proxy.calls == []


def test_a_request_that_finishes_after_the_deadline_snapshot_keeps_its_result(
    proxy, monkeypatch
):
    """``wait`` returns a snapshot of pending futures; a request can finish
    between that snapshot and the classification, and its result must not be
    replaced by a made-up TimeoutError."""
    real_wait = google_api_proxy.wait

    def delayed_wait(futures, timeout=None):
        done, pending = real_wait(futures, timeout=timeout)
        # Let a request in the pending snapshot finish before classifying.
        time.sleep(0.3)
        return done, pending

    def slow(**kwargs: Any) -> _Response:
        time.sleep(0.1)
        return _Response(200, {"status": 200, "data": {"id": "m1"}})

    monkeypatch.setattr(google_api_proxy, "wait", delayed_wait)
    monkeypatch.setattr(composio_proxy.requests, "request", slow)
    seen: list[tuple[str, Any, Any]] = []

    def callback(request_id, response, exception):
        seen.append((request_id, response, exception))

    batch = _added_batch(_gmail_service(), ["m1"], callback)
    google_api_proxy.execute_batch(batch, deadline=0.02)

    assert seen[0][1] == {"id": "m1"}
    assert seen[0][2] is None


def test_a_redirect_reaches_the_caller_with_its_location(proxy):
    """A caller (for example a media download in #3648) can read the
    Location off the HttpError's response."""
    proxy.answer_with(
        200,
        {
            "status": 302,
            "data": None,
            "headers": {"Location": "https://drive.usercontent.google.com/download/x"},
        },
    )

    with pytest.raises(HttpError) as raised:
        _gmail_service().users().labels().get(userId="me", id="x").execute()

    assert raised.value.resp.status == 302
    assert (
        raised.value.resp["location"]
        == "https://drive.usercontent.google.com/download/x"
    )


def test_a_late_answer_after_the_deadline_is_not_delivered_twice(monkeypatch):
    monkeypatch.setenv(composio_proxy.COMPOSIO_API_KEY_ENV, API_KEY)
    monkeypatch.setenv(composio_proxy.COMPOSIO_CONNECTED_ACCOUNT_ID_ENV, ACCOUNT_ID)
    monkeypatch.setenv(composio_proxy.COMPOSIO_USER_ID_ENV, COMPOSIO_USER)
    seen: list[tuple[str, Any, Any]] = []

    def slow(**kwargs: Any) -> _Response:
        time.sleep(0.3)
        return _Response(200, {"status": 200, "data": {"id": "m1"}})

    monkeypatch.setattr(composio_proxy.requests, "request", slow)

    def callback(request_id, response, exception):
        seen.append((request_id, response, exception))

    batch = _added_batch(_gmail_service(), ["m1"], callback)
    google_api_proxy.execute_batch(batch, deadline=0.02)
    time.sleep(0.4)

    assert len(seen) == 1
    assert seen[0][1] is None
    assert isinstance(seen[0][2], TimeoutError)
