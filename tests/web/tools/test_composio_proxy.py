"""Provider-neutral Composio Proxy Execute client (#3645).

The Salesforce, Gmail and Drive runtimes (#3646, #3647, #3648) send their API
calls through this client on a Composio-backed grant. Composio is faked here;
no request leaves the process.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from xagent.web.tools.mcp import composio_proxy

API_KEY = "test-composio-api-key-value"
ACCOUNT_ID = "ca_proxy_account_1"
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
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.response: _Response | Exception = _Response(
            200, {"data": {"ok": True}, "status": 200}
        )

    def request(self, **kwargs: Any) -> _Response:
        self.calls.append(kwargs)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


@pytest.fixture
def proxy(monkeypatch) -> FakeProxy:
    monkeypatch.setenv(composio_proxy.COMPOSIO_API_KEY_ENV, API_KEY)
    monkeypatch.setenv(composio_proxy.COMPOSIO_CONNECTED_ACCOUNT_ID_ENV, ACCOUNT_ID)
    monkeypatch.setenv(composio_proxy.COMPOSIO_USER_ID_ENV, COMPOSIO_USER)
    fake = FakeProxy()
    monkeypatch.setattr(composio_proxy.requests, "request", fake.request)
    return fake


def _secrets_in(text: str) -> list[str]:
    return [value for value in (API_KEY, ACCOUNT_ID, COMPOSIO_USER) if value in text]


def test_a_call_is_sent_as_one_proxy_execute_request(proxy):
    response = composio_proxy.proxy_request(
        "post",
        "https://gmail.googleapis.com/gmail/v1/users/me/labels",
        allowed_hosts=GMAIL_HOSTS,
        query=[("alt", "json"), ("fields", "id,name")],
        headers={"X-Goog-Request-Reason": "test"},
        body={"name": "Receipts"},
        timeout=12,
    )

    assert (response.status, response.data) == (200, {"ok": True})
    (call,) = proxy.calls
    assert call["method"] == "POST"
    assert call["url"] == composio_proxy.COMPOSIO_PROXY_URL
    assert call["headers"]["x-api-key"] == API_KEY
    assert call["timeout"] == 12
    assert call["json"] == {
        "endpoint": "https://gmail.googleapis.com/gmail/v1/users/me/labels",
        "method": "POST",
        "connected_account_id": ACCOUNT_ID,
        "user_id": COMPOSIO_USER,
        "parameters": [
            {"name": "alt", "value": "json", "type": "query"},
            {"name": "fields", "value": "id,name", "type": "query"},
            {"name": "X-Goog-Request-Reason", "value": "test", "type": "header"},
        ],
        "body": {"name": "Receipts"},
    }


def test_a_path_is_left_for_composio_to_resolve_against_the_account(proxy):
    """Salesforce sends org-relative paths; Composio knows the org's host."""
    composio_proxy.proxy_request(
        "GET",
        "/services/data/v62.0/query",
        allowed_hosts=(),
        query=[("q", "SELECT Id FROM Account")],
    )

    (call,) = proxy.calls
    assert call["json"]["endpoint"] == "/services/data/v62.0/query"
    assert "body" not in call["json"]


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://gmail.googleapis.com/gmail/v1/users/me/labels",
        "gmail.googleapis.com/gmail/v1/users/me/labels",
        "//gmail.googleapis.com/gmail/v1/users/me/labels",
        "",
        "https://attacker.example/gmail/v1/users/me/labels",
        "https://gmail.googleapis.com.attacker.example/gmail/v1",
        "https://user@gmail.googleapis.com/gmail/v1",
        "https://gmail.googleapis.com:8443/gmail/v1",
        "/\\evil.example/gmail/v1",
        "/gmail/v1/users/me labels",
        "/gmail/v1/users/me\nlabels",
        "/gmail/v1/users/me\x7f",
    ],
    ids=[
        "http",
        "no-scheme",
        "scheme-relative",
        "empty",
        "other-host",
        "lookalike-host",
        "credentials",
        "port",
        "backslash-path",
        "space-in-path",
        "control-in-path",
        "delete-in-path",
    ],
)
def test_an_endpoint_off_the_allowed_hosts_is_refused_before_sending(proxy, endpoint):
    """Composio adds the user's token to the call, so it must not go elsewhere."""
    with pytest.raises(ValueError):
        composio_proxy.proxy_request("GET", endpoint, allowed_hosts=GMAIL_HOSTS)

    assert proxy.calls == []


def test_an_upstream_error_is_returned_not_raised(proxy):
    """The caller maps its own API's errors; this one is Salesforce's array."""
    errors = [{"errorCode": "INVALID_FIELD", "message": "No such column 'Foo'"}]
    proxy.response = _Response(200, {"data": errors, "status": 400})

    response = composio_proxy.proxy_request(
        "GET", "/services/data/v62.0/query", allowed_hosts=()
    )

    assert (response.status, response.data) == (400, errors)
    assert not response.ok


@pytest.mark.parametrize("status", [401, 404, 429, 500])
def test_a_composio_refusal_raises_its_own_error_without_secrets(proxy, status):
    echoed = f"connected account {ACCOUNT_ID} of user {COMPOSIO_USER} key {API_KEY}"
    proxy.response = _Response(
        status,
        {"error": {"slug": "ConnectedAccount_NotFound", "message": echoed}},
    )

    with pytest.raises(composio_proxy.ComposioProxyError) as raised:
        composio_proxy.proxy_request(
            "GET", "/services/data/v62.0/sobjects", allowed_hosts=()
        )

    error = raised.value
    assert error.status == status
    assert error.slug == "ConnectedAccount_NotFound"
    assert "ConnectedAccount_NotFound" in str(error)
    assert _secrets_in(str(error)) == []
    assert _secrets_in(repr(error.args)) == []


@pytest.mark.parametrize(
    "payload",
    [None, ["not", "an", "object"], {"data": {}}, {"data": {}, "status": "200"}],
    ids=["not-json", "list", "no-status", "text-status"],
)
def test_an_answer_that_is_not_a_proxy_result_is_a_composio_error(proxy, payload):
    proxy.response = _Response(200, payload)

    with pytest.raises(composio_proxy.ComposioProxyError):
        composio_proxy.proxy_request(
            "GET", "/services/data/v62.0/sobjects", allowed_hosts=()
        )


def test_an_unreachable_composio_names_no_secret(proxy):
    proxy.response = ConnectionError(
        f"cannot reach {composio_proxy.COMPOSIO_PROXY_URL} for {ACCOUNT_ID}"
    )

    with pytest.raises(composio_proxy.ComposioProxyError) as raised:
        composio_proxy.proxy_request(
            "GET", "/services/data/v62.0/sobjects", allowed_hosts=()
        )

    assert raised.value.status is None
    assert _secrets_in(str(raised.value)) == []
    # The connection error's text is not chained in either.
    assert raised.value.__cause__ is None


@pytest.mark.parametrize(
    "missing",
    [
        composio_proxy.COMPOSIO_API_KEY_ENV,
        composio_proxy.COMPOSIO_CONNECTED_ACCOUNT_ID_ENV,
        composio_proxy.COMPOSIO_USER_ID_ENV,
    ],
)
def test_a_missing_launch_variable_is_named_and_nothing_is_sent(
    proxy, monkeypatch, missing
):
    monkeypatch.delenv(missing)

    with pytest.raises(ValueError, match=missing):
        composio_proxy.proxy_request(
            "GET", "/services/data/v62.0/sobjects", allowed_hosts=()
        )

    assert proxy.calls == []


def test_redact_hides_every_launch_secret_in_nested_values(proxy):
    value = {
        "message": f"account {ACCOUNT_ID}",
        "items": [f"user {COMPOSIO_USER}", {"key": API_KEY}, 3],
    }

    redacted = composio_proxy.redact(value)

    assert _secrets_in(repr(redacted)) == []
    assert redacted["items"][2] == 3


@pytest.mark.parametrize(
    "endpoint,allowed",
    [
        ("https://Gmail.GoogleAPIs.com/gmail/v1/users/me/profile", GMAIL_HOSTS),
        (
            "https://gmail.googleapis.com/gmail/v1/users/me/profile",
            ("Gmail.GoogleAPIs.com",),
        ),
    ],
    ids=["endpoint-case", "allowed-case"],
)
def test_an_allowed_host_matches_regardless_of_case(proxy, endpoint, allowed):
    composio_proxy.proxy_request("GET", endpoint, allowed_hosts=allowed)

    assert len(proxy.calls) == 1


def test_a_long_composio_message_is_cut_after_redaction(proxy):
    """The key straddles the cut: cutting first would leave its prefix."""
    prefix = "Composio proxy request failed with HTTP 500: "
    kept = 8
    padding = "x" * (composio_proxy.MAX_ERROR_TEXT_CHARS - len(prefix) - kept)
    proxy.response = _Response(500, {"error": {"message": padding + API_KEY}})

    with pytest.raises(composio_proxy.ComposioProxyError) as raised:
        composio_proxy.proxy_request(
            "GET", "/services/data/v62.0/sobjects", allowed_hosts=()
        )

    text = str(raised.value)
    assert text.startswith(prefix + padding)
    assert text.endswith("...[truncated]")
    assert len(text) == composio_proxy.MAX_ERROR_TEXT_CHARS + len("...[truncated]")
    assert API_KEY[:kept] not in text


def test_a_query_mapping_sends_each_name_and_value(proxy):
    composio_proxy.proxy_request(
        "GET",
        "/services/data/v62.0/query",
        allowed_hosts=(),
        query={"q": "SELECT Id FROM Account", "id": "x"},
    )

    (call,) = proxy.calls
    assert call["json"]["parameters"] == [
        {"name": "q", "value": "SELECT Id FROM Account", "type": "query"},
        {"name": "id", "value": "x", "type": "query"},
    ]


def test_allowed_hosts_given_as_one_string_is_refused(proxy):
    with pytest.raises(TypeError):
        composio_proxy.proxy_request(
            "GET",
            "https://gmail.googleapis.com/gmail/v1/users/me/profile",
            allowed_hosts="gmail.googleapis.com",
        )

    assert proxy.calls == []


@pytest.mark.parametrize(
    "response",
    [
        _Response(200, {"data": {}, "status": True}),
        _Response(502, {"error": "Bad gateway"}),
        _Response(503, None),
    ],
    ids=["boolean-status", "error-not-an-object", "no-json-body"],
)
def test_odd_composio_answers_are_composio_errors(proxy, response):
    proxy.response = response

    with pytest.raises(composio_proxy.ComposioProxyError) as raised:
        composio_proxy.proxy_request(
            "GET", "/services/data/v62.0/sobjects", allowed_hosts=()
        )

    if response.status_code != 200:
        assert raised.value.status == response.status_code
        assert raised.value.slug is None


def test_a_body_over_the_proxy_limit_is_refused_before_sending(proxy, monkeypatch):
    """A request bigger than the proxy accepts is refused here, with both
    sizes, instead of failing at Composio's edge as an opaque 413."""
    monkeypatch.setattr(composio_proxy, "MAX_JSON_BODY_BYTES", 200)

    with pytest.raises(ValueError) as raised:
        composio_proxy.proxy_request(
            "POST",
            "/services/data/v62.0/sobjects/Account",
            allowed_hosts=(),
            body={"Name": "x" * 500},
        )

    assert "too large to send through Composio" in str(raised.value)
    assert "200 byte limit" in str(raised.value)
    assert proxy.calls == []


def test_the_body_limit_counts_the_whole_proxy_payload(proxy, monkeypatch):
    """The limit covers the envelope too, not just the provider's body."""
    body = {"Name": "x" * 50}
    monkeypatch.setattr(composio_proxy, "MAX_JSON_BODY_BYTES", 10_000_000)

    composio_proxy.proxy_request(
        "POST", "/services/data/v62.0/sobjects/Account", allowed_hosts=(), body=body
    )
    (call,) = proxy.calls
    payload_size = len(json.dumps(call["json"]))

    monkeypatch.setattr(composio_proxy, "MAX_JSON_BODY_BYTES", payload_size)
    composio_proxy.proxy_request(
        "POST", "/services/data/v62.0/sobjects/Account", allowed_hosts=(), body=body
    )
    monkeypatch.setattr(composio_proxy, "MAX_JSON_BODY_BYTES", payload_size - 1)

    with pytest.raises(ValueError):
        composio_proxy.proxy_request(
            "POST",
            "/services/data/v62.0/sobjects/Account",
            allowed_hosts=(),
            body=body,
        )

    assert len(proxy.calls) == 2
