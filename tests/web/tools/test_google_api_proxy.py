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

from tests.web.tools.conftest import (
    ACCOUNT_ID,
    API_KEY,
    COMPOSIO_USER,
    GMAIL_HOSTS,
    _Response,
)
from xagent.web.tools.mcp import composio_proxy, google_api_proxy


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
        "?maxResults=10&labelIds=INBOX",
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
            {"name": "content-type", "value": "application/json", "type": "header"},
            {"name": "accept", "value": "application/json", "type": "header"},
        ],
        "body": {"raw": "abc"},
    }


def test_a_query_name_that_repeats_in_the_uri_is_refused_before_sending(proxy):
    """Composio's proxy appears to keep one value of a repeated query
    parameter, so a request that would lose a value is refused here."""
    with pytest.raises(ValueError) as raised:
        _http().request(
            "https://gmail.googleapis.com/gmail/v1/users/me/messages"
            "?labelIds=INBOX&labelIds=STARRED"
        )

    assert "labelIds" in str(raised.value)
    assert proxy.calls == []


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


def test_a_long_get_whose_query_repeats_a_name_is_refused_before_sending(proxy):
    """The form body googleapiclient moves the query of a long GET into keeps
    a repeated parameter as its pairs; restoring the GET must not send them,
    because the proxy appears to keep one value of a repeated name."""
    with pytest.raises(ValueError) as raised:
        _gmail_service().users().messages().list(
            userId="me", q="x" * 2200, labelIds=["INBOX", "STARRED"]
        ).execute()

    assert "labelIds" in str(raised.value)
    assert proxy.calls == []


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
    assert response["content-type"] == "application/json; charset=UTF-8"


@pytest.mark.parametrize(
    "answer",
    [{"status": 200}, {"status": 200, "data": None}],
    ids=["absent", "null"],
)
def test_a_200_without_a_body_is_refused(proxy, answer):
    """Gmail answers an empty success with 204, so a 200 with neither a body
    nor a file is an incomplete answer and fails this request here. The
    shared client leaves that rule to each API: Salesforce (#3669) reads the
    same answer as ``{}``."""
    proxy.answer_with(200, answer)

    with pytest.raises(composio_proxy.ComposioProxyError) as raised:
        _http().request("https://gmail.googleapis.com/gmail/v1/users/me/labels")

    assert "without the provider's data" in str(raised.value)


def test_a_text_answer_is_returned_as_bytes(proxy):
    """A later media download reuses this transport; its answer may not be
    JSON."""
    proxy.answer_with(200, {"status": 200, "data": "%PDF-1.4 bytes"})

    response, content = _http().request(
        "https://gmail.googleapis.com/gmail/v1/users/me/messages/m1/attachments/x"
    )

    assert content == b"%PDF-1.4 bytes"
    assert response["content-type"] == "text/plain; charset=UTF-8"


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
    """One metadata read per id, the shape a Composio grant asks for: no
    ``metadataHeaders``, whose repeated name the proxy appears to collapse."""
    batch = service.new_batch_http_request(callback=callback)
    for message_id in ids:
        batch.add(
            service.users()
            .messages()
            .get(userId="me", id=message_id, format="metadata")
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


def test_a_200_with_a_null_body_fails_only_its_request(proxy):
    """Composio answers ``{"status": 200, "data": null}`` for an empty 2xx
    body. That is no body at all, not a 0-byte JSON document: it must reach
    its own callback as a Composio error, not escape and lose the batch."""
    proxy.answer("/messages/m1", 200, {"id": "m1"})
    proxy.answer("/messages/m2", 200, None)
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
    assert isinstance(seen[1][2], composio_proxy.ComposioProxyError)
    assert "without the provider's data" in str(seen[1][2])
    assert seen[2][1] == {"id": "m3"}


def test_a_postproc_error_fails_only_its_request(proxy):
    """``postproc`` runs after every request has been sent. An error there
    used to escape and leave the remaining callbacks uncalled."""
    proxy.answer("/messages/m1", 200, {"id": "m1"})
    proxy.answer("/messages/m3", 200, {"id": "m3"})
    seen: list[tuple[str, Any, Any]] = []

    def callback(request_id, response, exception):
        seen.append((request_id, response, exception))

    def postproc(resp, content):
        raise ValueError("not a result")

    batch = _added_batch(_gmail_service(), ["m1", "m2", "m3"], callback)
    batch._requests["2"].postproc = postproc

    google_api_proxy.execute_batch(batch)

    assert [item[0] for item in seen] == ["1", "2", "3"]
    assert seen[0][1] == {"id": "m1"}
    assert seen[1][1] is None
    assert isinstance(seen[1][2], ValueError)
    assert seen[2][1] == {"id": "m3"}


def test_a_callback_that_raises_mid_batch_is_not_swallowed(proxy):
    """A native batch lets a callback's exception propagate; the fallback
    does the same instead of hiding a connector's own bug and silently
    dropping that request."""
    proxy.answer("/messages/m1", 200, {"id": "m1"})
    proxy.answer("/messages/m3", 200, {"id": "m3"})
    seen: list[str] = []

    def callback(request_id, response, exception):
        seen.append(request_id)
        if request_id == "2":
            raise RuntimeError("the callback itself failed")

    with pytest.raises(RuntimeError, match="the callback itself failed"):
        google_api_proxy.execute_batch(
            _added_batch(_gmail_service(), ["m1", "m2", "m3"], callback)
        )

    assert seen == ["1", "2"]


def test_a_batch_with_no_requests_sends_nothing(proxy):
    service = _gmail_service()

    google_api_proxy.execute_batch(service.new_batch_http_request())

    assert proxy.calls == []


class _Download:
    """The public_requests response shape a binary download returns."""

    def __init__(
        self,
        status_code: int,
        content: bytes,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self.content = content
        self.headers = headers or {}


def test_a_binary_answer_is_downloaded_from_its_signed_url(proxy, monkeypatch):
    """Composio does not relay the bytes: it stores the file and answers
    ``data=None`` plus ``binary_data``. The transport must return the bytes,
    not an empty body that a downloader would treat as a finished 0-byte
    file."""
    seen: dict[str, Any] = {}

    def download(url: str, timeout: float, **kwargs: Any) -> _Download:
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
                "size": 19,
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
    def download(url: str, timeout: float, **kwargs: Any) -> _Download:
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
    def download(url: str, timeout: float, **kwargs: Any) -> _Download:
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


def test_a_short_binary_answer_is_refused(proxy, monkeypatch):
    """A signed URL whose object is shorter than the size Composio declared
    must not reach a downloader as a complete file."""
    monkeypatch.setattr(
        google_api_proxy.requests,
        "get",
        lambda url, timeout, **kwargs: _Download(200, b"short"),
    )
    proxy.answer_with(
        200,
        {
            "status": 200,
            "data": None,
            "binary_data": {
                "url": "https://files.composio.dev/signed-secret",
                "content_type": "application/pdf",
                "size": 100,
            },
        },
    )

    with pytest.raises(composio_proxy.ComposioProxyError) as raised:
        _http().request("https://gmail.googleapis.com/gmail/v1/users/me/messages/m1")

    text = str(raised.value)
    assert "truncated" in text
    assert "100" in text and "5" in text
    assert "files.composio.dev" not in text
    assert "signed-secret" not in text


def test_a_binary_answer_with_a_conflicting_content_length_is_refused(
    proxy, monkeypatch
):
    """The signed URL's own Content-Length is checked too, so a stale
    ``binary_data.size`` cannot hide a short object."""
    monkeypatch.setattr(
        google_api_proxy.requests,
        "get",
        lambda url, timeout, **kwargs: _Download(
            200, b"short", headers={"Content-Length": "100"}
        ),
    )
    proxy.answer_with(
        200,
        {
            "status": 200,
            "data": None,
            "binary_data": {
                "url": "https://files.composio.dev/signed",
                "content_type": "application/pdf",
                "size": 5,
            },
        },
    )

    with pytest.raises(composio_proxy.ComposioProxyError) as raised:
        _http().request("https://gmail.googleapis.com/gmail/v1/users/me/messages/m1")

    assert "truncated" in str(raised.value)


def test_a_binary_download_asks_for_no_content_encoding(proxy, monkeypatch):
    """``requests`` would otherwise ask for gzip/deflate and decode the body,
    while Content-Length stays the encoded length."""
    seen: dict[str, Any] = {}

    def download(url: str, timeout: float, **kwargs: Any) -> _Download:
        seen.update(kwargs)
        return _Download(200, b"%PDF-1.4 real bytes")

    monkeypatch.setattr(google_api_proxy.requests, "get", download)
    proxy.answer_with(
        200,
        {
            "status": 200,
            "binary_data": {
                "url": "https://files.composio.dev/signed",
                "content_type": "application/pdf",
                "size": 19,
            },
        },
    )

    _http().request("https://gmail.googleapis.com/gmail/v1/users/me/messages/m1")

    assert seen["headers"]["Accept-Encoding"] == "identity"


def test_a_compressed_binary_answer_is_not_read_as_truncated(proxy, monkeypatch):
    """When the signed URL compresses anyway, its Content-Length is the
    encoded length: the decoded body is compared with Composio's size only."""
    # 19 decoded bytes served as 12 gzip bytes.
    monkeypatch.setattr(
        google_api_proxy.requests,
        "get",
        lambda url, timeout, **kwargs: _Download(
            200,
            b"%PDF-1.4 real bytes",
            headers={"Content-Length": "12", "Content-Encoding": "gzip"},
        ),
    )
    proxy.answer_with(
        200,
        {
            "status": 200,
            "binary_data": {
                "url": "https://files.composio.dev/signed",
                "content_type": "application/pdf",
                "size": 19,
            },
        },
    )

    _, content = _http().request(
        "https://gmail.googleapis.com/gmail/v1/users/me/messages/m1"
    )

    assert content == b"%PDF-1.4 real bytes"


def test_a_compressed_binary_answer_is_still_checked_against_composios_size(
    proxy, monkeypatch
):
    """``binary_data.size`` is the decoded file's size, so it still guards a
    compressed answer."""
    monkeypatch.setattr(
        google_api_proxy.requests,
        "get",
        lambda url, timeout, **kwargs: _Download(
            200,
            b"short",
            headers={"Content-Length": "5", "Content-Encoding": "gzip"},
        ),
    )
    proxy.answer_with(
        200,
        {
            "status": 200,
            "binary_data": {
                "url": "https://files.composio.dev/signed",
                "content_type": "application/pdf",
                "size": 100,
            },
        },
    )

    with pytest.raises(composio_proxy.ComposioProxyError) as raised:
        _http().request("https://gmail.googleapis.com/gmail/v1/users/me/messages/m1")

    assert "truncated" in str(raised.value)


def test_a_partial_binary_answer_is_refused(proxy, monkeypatch):
    """This transport never asks for a range, so a 206 is not the whole
    file even when its length looks usable."""
    monkeypatch.setattr(
        google_api_proxy.requests,
        "get",
        lambda url, timeout, **kwargs: _Download(206, b"part"),
    )
    proxy.answer_with(
        200,
        {
            "status": 200,
            "data": None,
            "binary_data": {
                "url": "https://files.composio.dev/signed",
                "content_type": "application/pdf",
                "size": 100,
            },
        },
    )

    with pytest.raises(composio_proxy.ComposioProxyError) as raised:
        _http().request("https://gmail.googleapis.com/gmail/v1/users/me/messages/m1")

    assert "206" in str(raised.value)
    assert "partial" in str(raised.value)


def test_a_binary_answer_with_an_unusable_size_is_refused(proxy, monkeypatch):
    """A size that is present but not a byte count is not silently skipped."""
    monkeypatch.setattr(
        google_api_proxy.requests,
        "get",
        lambda url, timeout, **kwargs: _Download(200, b"bytes"),
    )
    proxy.answer_with(
        200,
        {
            "status": 200,
            "data": None,
            "binary_data": {
                "url": "https://files.composio.dev/signed",
                "content_type": "application/pdf",
                "size": "all of them",
            },
        },
    )

    with pytest.raises(composio_proxy.ComposioProxyError) as raised:
        _http().request("https://gmail.googleapis.com/gmail/v1/users/me/messages/m1")

    assert "unusable size" in str(raised.value)


def test_a_media_answer_relayed_as_a_parsed_object_is_refused(proxy):
    """A JSON body Composio parsed cannot be re-serialized into the file's
    exact bytes, so a media read refuses it instead of returning changed
    content as success."""
    proxy.answer_with(200, {"status": 200, "data": {"b": 2, "a": "é"}})

    with pytest.raises(composio_proxy.ComposioProxyError) as raised:
        _http().request(
            "https://gmail.googleapis.com/gmail/v1/users/me/messages/m1/"
            "attachments/a?alt=media"
        )

    assert "parsed object" in str(raised.value)
    assert "exact bytes" in str(raised.value)


def test_a_media_error_answer_relayed_as_an_object_keeps_its_status(proxy):
    """A failed media read is a provider error, not a body-fidelity case:
    its status and JSON error text must still reach googleapiclient."""
    proxy.answer_with(
        200,
        {
            "status": 404,
            "data": {"error": {"code": 404, "message": "File not found."}},
        },
    )

    response, content = _http().request(
        "https://gmail.googleapis.com/gmail/v1/users/me/messages/m1/"
        "attachments/a?alt=media"
    )

    assert response.status == 404
    assert b"File not found." in content


def test_a_media_answer_relayed_as_text_is_still_returned(proxy):
    proxy.answer_with(200, {"status": 200, "data": "plain text"})

    response, content = _http().request(
        "https://gmail.googleapis.com/gmail/v1/users/me/messages/m1/"
        "attachments/a?alt=media"
    )

    assert response.status == 200
    assert content == b"plain text"
    assert response["content-type"].startswith("text/plain")


def test_a_json_api_answer_relayed_as_an_object_is_still_returned(proxy):
    """The parsed-object refusal applies only to a media read; a normal JSON
    call's dict answer is exactly what the caller expects."""
    proxy.answer_with(200, {"status": 200, "data": {"b": 2, "a": "é"}})

    response, content = _http().request(
        "https://gmail.googleapis.com/gmail/v1/users/me/labels"
    )

    assert response.status == 200
    assert json.loads(content) == {"b": 2, "a": "é"}


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


def test_a_batch_object_without_the_private_attributes_is_refused_clearly(proxy):
    """``pyproject.toml`` sets only a lower bound for
    google-api-python-client, and the fallback reads private attributes. A
    release that renames one must not surface as an opaque AttributeError on
    every search."""

    class OpaqueBatch:
        pass

    with pytest.raises(RuntimeError, match="missing _order"):
        google_api_proxy.execute_batch(OpaqueBatch())

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


def test_a_binary_answer_without_a_data_key_is_downloaded(proxy, monkeypatch):
    """Composio's schema makes ``data`` optional, so a file answer may carry
    only ``binary_data``; it must still download instead of becoming an
    empty body."""
    seen: dict[str, Any] = {}

    def download(url: str, timeout: float, **kwargs: Any) -> _Download:
        seen["url"] = url
        return _Download(200, b"%PDF-1.4 no data key")

    monkeypatch.setattr(google_api_proxy.requests, "get", download)
    proxy.answer_with(
        200,
        {
            "status": 200,
            "binary_data": {
                "url": "https://files.composio.dev/signed",
                "content_type": "application/pdf",
                "size": 20,
            },
        },
    )

    response, content = _http().request(
        "https://gmail.googleapis.com/gmail/v1/users/me/messages/m1/attachments/a"
    )

    assert seen["url"] == "https://files.composio.dev/signed"
    assert content == b"%PDF-1.4 no data key"
    assert response["content-type"] == "application/pdf"
    assert response["content-length"] == str(len(content))


def test_a_redirect_without_a_data_key_keeps_its_location(proxy):
    """A bodyless 3xx is a normal provider answer; its Location must reach
    the caller even when the answer has no ``data`` member."""
    proxy.answer_with(
        200,
        {
            "status": 302,
            "headers": {"Location": "https://drive.usercontent.google.com/download/x"},
        },
    )

    response, content = _http().request(
        "https://gmail.googleapis.com/gmail/v1/users/me/messages/m1/attachments/a"
    )

    assert response.status == 302
    assert response["location"] == "https://drive.usercontent.google.com/download/x"
    assert content == b""


def test_an_upstream_error_without_a_data_key_still_reaches_the_caller(proxy):
    """A 304 or a bodyless upstream error keeps the provider's status; it is
    googleapiclient that turns it into an ``HttpError``."""
    proxy.answer_with(200, {"status": 304})

    with pytest.raises(HttpError) as raised:
        _gmail_service().users().labels().get(userId="me", id="x").execute()

    assert raised.value.resp.status == 304


def test_a_malformed_binary_answer_is_not_a_zero_byte_success(proxy, monkeypatch):
    """``binary_data`` is an object in Composio's schema; a string used to be
    dropped silently and the answer became a successful 0-byte body."""

    def download(*args: Any, **kwargs: Any) -> _Download:
        raise AssertionError("no download for a malformed envelope")

    monkeypatch.setattr(google_api_proxy.requests, "get", download)
    proxy.answer_with(
        200,
        {
            "status": 200,
            "data": None,
            "binary_data": "https://files.composio.dev/signed",
        },
    )

    with pytest.raises(composio_proxy.ComposioProxyError) as raised:
        _http().request("https://gmail.googleapis.com/gmail/v1/users/me/messages/m1")

    assert "binary" in str(raised.value)


def test_a_download_that_raises_names_no_signed_url(proxy, monkeypatch):
    """The signed URL is a bearer-like credential; an exception from the
    download client can name it, so only a generic message is raised."""

    def download(url: str, timeout: float, **kwargs: Any) -> _Download:
        raise RuntimeError(f"boom {url}")

    monkeypatch.setattr(google_api_proxy.requests, "get", download)
    proxy.answer_with(
        200,
        {
            "status": 200,
            "binary_data": {
                "url": "https://files.composio.dev/signed-secret",
                "content_type": "application/pdf",
                "size": 3,
            },
        },
    )

    with pytest.raises(composio_proxy.ComposioProxyError) as raised:
        _http().request("https://gmail.googleapis.com/gmail/v1/users/me/messages/m1")

    text = str(raised.value)
    assert "files.composio.dev" not in text
    assert "signed-secret" not in text
    assert raised.value.__cause__ is None


def test_the_transport_can_override_the_shared_body_limit(proxy):
    """The 4.5 MB default is an inferred bound; a stage measurement that
    finds a different per-API limit must be usable without changing the
    shared constant."""
    http = google_api_proxy.GoogleApiProxyHttp(
        allowed_hosts=GMAIL_HOSTS, max_body_bytes=200
    )
    body = json.dumps({"raw": "x" * 500})

    with pytest.raises(ValueError, match="200 byte limit"):
        http.request(
            "https://gmail.googleapis.com/gmail/v1/users/me/messages/send",
            method="POST",
            body=body,
            headers={"content-type": "application/json"},
        )
    with pytest.raises(ValueError, match="200 byte limit"):
        http.check_request(
            "https://gmail.googleapis.com/gmail/v1/users/me/messages/send",
            method="POST",
            body=body,
            headers={"content-type": "application/json"},
        )

    assert proxy.calls == []
