"""Provider-neutral Composio Proxy Execute client (#3645).

The Salesforce, Gmail and Drive runtimes (#3646, #3647, #3648) send their API
calls through this client on a Composio-backed grant. Composio is faked here;
no request leaves the process.
"""

from __future__ import annotations

import base64
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


def test_a_response_keeps_the_provider_headers_and_binary_data(proxy):
    """#3648 needs the provider's headers (a 3xx's Location) and Composio's
    binary envelope; both are part of the answer and must not be dropped."""
    headers = {
        "location": "https://files.example/signed",
        "x-goog-request-id": "r1",
    }
    binary_data = {
        "url": "https://files.example/signed",
        "content_type": "application/pdf",
        "size": 3,
    }
    proxy.response = _Response(
        200,
        {
            "status": 302,
            "data": None,
            "headers": headers,
            "binary_data": binary_data,
        },
    )

    response = composio_proxy.proxy_request(
        "GET",
        "https://gmail.googleapis.com/gmail/v1/users/me/messages/m1/attachments/a",
        allowed_hosts=GMAIL_HOSTS,
    )

    assert response.status == 302
    assert response.headers == headers
    assert response.binary_data == binary_data


def test_a_no_content_answer_may_omit_data(proxy):
    proxy.response = _Response(200, {"status": 204})

    response = composio_proxy.proxy_request(
        "DELETE",
        "https://gmail.googleapis.com/gmail/v1/users/me/labels/x",
        allowed_hosts=GMAIL_HOSTS,
    )

    assert (response.status, response.data) == (204, None)


def test_an_error_that_echoes_the_request_body_hides_the_message(proxy):
    """A Gmail message's body and attachments travel base64 in ``raw``; a
    vendor error echoing the body must not put them in the tool result."""
    raw = base64.urlsafe_b64encode(b"Subject: private\n\nLEAK-ME body").decode()
    proxy.response = _Response(
        400,
        {"error": {"message": f"invalid body {json.dumps({'raw': raw})}"}},
    )

    with pytest.raises(composio_proxy.ComposioProxyError) as raised:
        composio_proxy.proxy_request(
            "POST",
            "https://gmail.googleapis.com/gmail/v1/users/me/messages/send",
            allowed_hosts=GMAIL_HOSTS,
            body={"raw": raw},
        )

    text = str(raised.value)
    assert raw not in text
    assert "[redacted]" in text


def test_an_error_that_echoes_a_provider_token_is_redacted(proxy):
    """Composio can echo an upstream Authorization header; its value is not
    one of our launch variables, so it is redacted by shape."""
    token = "ya29.PROVIDER-SECRET-TOKEN-123"
    proxy.response = _Response(
        401,
        {"error": {"message": f"upstream said authorization: Bearer {token}"}},
    )

    with pytest.raises(composio_proxy.ComposioProxyError) as raised:
        composio_proxy.proxy_request(
            "GET", "/services/data/v62.0/sobjects", allowed_hosts=()
        )

    text = str(raised.value)
    assert token not in text
    assert "Bearer [redacted]" in text


def test_a_multiline_query_value_is_passed_through(proxy):
    """M2: a Salesforce SOQL/SOSL query or a Gmail ``q`` can be multi-line.
    The value is Composio's to encode, so it must not be refused here; only
    names, headers and the endpoint are checked for control characters."""
    soql = "SELECT Id, Name\nFROM Account\nWHERE Name = 'Acme'"

    response = composio_proxy.proxy_request(
        "GET",
        "/services/data/v62.0/query",
        allowed_hosts=(),
        query=[("q", soql)],
    )

    assert response.status == 200
    (call,) = proxy.calls
    assert call["json"]["parameters"] == [{"name": "q", "value": soql, "type": "query"}]


@pytest.mark.parametrize(
    "query, headers",
    [
        ([("q\nX-Injected", "1")], None),
        (None, {"accept": "application/json\nX-Injected: 1"}),
        (None, {"accept": "application/json\rX-Injected: 1"}),
        (None, {"accept": "application/json\r\nX-Injected: 1"}),
    ],
    ids=["query-name-lf", "header-lf", "header-cr", "header-crlf"],
)
def test_a_control_character_in_a_query_name_or_header_is_refused_before_sending(
    proxy, query, headers
):
    with pytest.raises(ValueError, match="control character"):
        composio_proxy.proxy_request(
            "GET",
            "/services/data/v62.0/query",
            allowed_hosts=(),
            query=query or (),
            headers=headers,
        )

    assert proxy.calls == []


def test_an_absolute_endpoint_with_a_control_character_is_refused(proxy):
    with pytest.raises(ValueError, match="control character"):
        composio_proxy.proxy_request(
            "GET",
            "https://gmail.googleapis.com/gmail/v1/users/me/labels\nX: 1",
            allowed_hosts=GMAIL_HOSTS,
        )

    assert proxy.calls == []


def test_a_per_call_body_limit_overrides_the_shared_default(proxy):
    """#3669's Salesforce may have a different real limit; the shared default
    stays, but a connector can pass its own."""
    with pytest.raises(ValueError, match="200 byte limit"):
        composio_proxy.proxy_request(
            "POST",
            "/services/data/v62.0/sobjects/Account",
            allowed_hosts=(),
            body={"Name": "x" * 500},
            max_body_bytes=200,
        )

    assert proxy.calls == []


def test_a_successful_answer_without_data_or_binary_data_is_a_composio_error(proxy):
    """A truncated 2xx used to reach a callback and fail opaquely there; with
    neither a body nor a file there is nothing to return."""
    proxy.response = _Response(200, {"status": 200})

    with pytest.raises(composio_proxy.ComposioProxyError) as raised:
        composio_proxy.proxy_request(
            "GET",
            "https://gmail.googleapis.com/gmail/v1/users/me/labels",
            allowed_hosts=GMAIL_HOSTS,
        )

    assert "without the provider's data" in str(raised.value)


def test_a_redirect_without_data_is_still_a_provider_answer(proxy):
    """Composio's 200 schema only requires ``status``: ``data`` is optional,
    so a 3xx with no body must reach the caller with its Location instead of
    being refused as an incomplete answer."""
    location = "https://drive.usercontent.google.com/download/x"
    proxy.response = _Response(200, {"status": 302, "headers": {"location": location}})

    response = composio_proxy.proxy_request(
        "GET",
        "https://gmail.googleapis.com/gmail/v1/users/me/messages/m1/attachments/a",
        allowed_hosts=GMAIL_HOSTS,
    )

    assert response.status == 302
    assert response.data is None
    assert response.headers == {"location": location}


def test_a_binary_answer_may_omit_data(proxy):
    binary_data = {
        "url": "https://files.composio.dev/signed",
        "content_type": "application/pdf",
        "size": 3,
    }
    proxy.response = _Response(200, {"status": 200, "binary_data": binary_data})

    response = composio_proxy.proxy_request(
        "GET",
        "https://gmail.googleapis.com/gmail/v1/users/me/messages/m1/attachments/a",
        allowed_hosts=GMAIL_HOSTS,
    )

    assert response.status == 200
    assert response.data is None
    assert response.binary_data == binary_data


def test_an_upstream_error_may_omit_data(proxy):
    proxy.response = _Response(200, {"status": 404})

    response = composio_proxy.proxy_request(
        "GET",
        "https://gmail.googleapis.com/gmail/v1/users/me/labels/missing",
        allowed_hosts=GMAIL_HOSTS,
    )

    assert response.status == 404
    assert response.data is None
    assert not response.ok


@pytest.mark.parametrize(
    "answer, fragment",
    [
        (
            {"status": 200, "data": None, "binary_data": "https://files/signed"},
            "binary",
        ),
        (
            {"status": 200, "data": None, "binary_data": ["https://files/signed"]},
            "binary",
        ),
        (
            {"status": 302, "data": None, "headers": ["location", "https://example/x"]},
            "headers",
        ),
        (
            {"status": 302, "data": None, "headers": "location: https://example/x"},
            "headers",
        ),
    ],
    ids=["binary-str", "binary-list", "headers-list", "headers-str"],
)
def test_a_malformed_envelope_field_is_a_composio_error(proxy, answer, fragment):
    """A non-object field used to be dropped silently, which turned a
    malformed file answer into a successful 0-byte body and lost a 3xx's
    Location."""
    proxy.response = _Response(200, answer)

    with pytest.raises(composio_proxy.ComposioProxyError) as raised:
        composio_proxy.proxy_request(
            "GET",
            "https://gmail.googleapis.com/gmail/v1/users/me/messages/m1",
            allowed_hosts=GMAIL_HOSTS,
        )

    assert fragment in str(raised.value)


def test_an_error_that_echoes_a_truncated_body_hides_the_message(proxy):
    """A vendor that cuts the echoed body off leaves the ``"raw"`` member
    unterminated; the base64 prefix still carries the MIME headers and the
    start of the body, so it must be redacted to the end of the echo."""
    raw = base64.urlsafe_b64encode(
        b"Subject: TRUNCATED-ECHO-SUBJECT\r\n\r\n" + b"BODY-MARKER-TRUNCATED" * 100
    ).decode()
    truncated = raw[:600]
    proxy.response = _Response(
        400, {"error": {"message": f'invalid body {{"raw": "{truncated}'}}
    )

    with pytest.raises(composio_proxy.ComposioProxyError) as raised:
        composio_proxy.proxy_request(
            "POST",
            "https://gmail.googleapis.com/gmail/v1/users/me/messages/send",
            allowed_hosts=GMAIL_HOSTS,
            body={"raw": raw},
        )

    text = str(raised.value)
    assert truncated not in text
    assert "[redacted]" in text


@pytest.mark.parametrize(
    "message, secret",
    [
        ("bad grant refresh_token=1//0gSECRET-REFRESH", "1//0gSECRET-REFRESH"),
        ("bad grant client_secret=GOCSPX-SECRET-CLIENT", "GOCSPX-SECRET-CLIENT"),
        ('bad body {"refresh_token": "1//0gSECRET-JSON"}', "1//0gSECRET-JSON"),
        ('bad body {"client_secret": "GOCSPX-SECRET-JSON"}', "GOCSPX-SECRET-JSON"),
    ],
    ids=["refresh-form", "client-form", "refresh-json", "client-json"],
)
def test_an_error_that_names_a_non_bearer_credential_is_redacted(
    proxy, message, secret
):
    """A vendor can name the grant's refresh token or client secret; neither
    is one of our launch variables, so they are matched by member name."""
    proxy.response = _Response(401, {"error": {"message": message}})

    with pytest.raises(composio_proxy.ComposioProxyError) as raised:
        composio_proxy.proxy_request(
            "GET", "/services/data/v62.0/sobjects", allowed_hosts=()
        )

    text = str(raised.value)
    assert secret not in text
    assert "[redacted]" in text


def test_an_error_that_echoes_the_decoded_message_hides_its_content(proxy):
    """A vendor can echo the decoded request instead of the JSON body it was
    sent; the mail body and the caller-written headers then appear in
    clear."""
    raw = base64.urlsafe_b64encode(
        b"To: a@example.com\r\nFrom: me\r\nSubject: DECODED-ECHO-SUBJECT\r\n"
        b"\r\nDECODED-BODY-MARKER\r\n"
    ).decode()
    proxy.response = _Response(
        400,
        {
            "error": {
                "message": (
                    "invalid message Subject: DECODED-ECHO-SUBJECT "
                    "body DECODED-BODY-MARKER"
                )
            }
        },
    )

    with pytest.raises(composio_proxy.ComposioProxyError) as raised:
        composio_proxy.proxy_request(
            "POST",
            "https://gmail.googleapis.com/gmail/v1/users/me/messages/send",
            allowed_hosts=GMAIL_HOSTS,
            body={"raw": raw},
        )

    text = str(raised.value)
    assert "DECODED-ECHO-SUBJECT" not in text
    assert "DECODED-BODY-MARKER" not in text
