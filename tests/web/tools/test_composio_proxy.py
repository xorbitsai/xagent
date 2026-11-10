"""Provider-neutral Composio Proxy Execute client (#3645).

The Salesforce, Gmail and Drive runtimes (#3646, #3647, #3648) send their API
calls through this client on a Composio-backed grant. Composio is faked here;
no request leaves the process.
"""

from __future__ import annotations

import base64
import json
import random
import time
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


def test_a_request_with_a_body_cuts_only_after_the_secret_is_redacted(proxy):
    """The decoded-body pass truncates the text before it runs; the launch
    secrets must already be gone when that happens, or a key across the cut
    would keep its prefix."""
    prefix = "Composio proxy request failed with HTTP 400: "
    kept = 8
    padding = "x" * (composio_proxy.MAX_ERROR_TEXT_CHARS - len(prefix) - kept)
    proxy.response = _Response(
        400, {"error": {"message": padding + API_KEY + "y" * 2000}}
    )

    with pytest.raises(composio_proxy.ComposioProxyError) as raised:
        composio_proxy.proxy_request(
            "POST",
            "https://gmail.googleapis.com/gmail/v1/users/me/messages/send",
            allowed_hosts=GMAIL_HOSTS,
            body={"name": "a body that keeps the decoded pass in play"},
        )

    text = str(raised.value)
    assert API_KEY[:kept] not in text
    assert API_KEY not in text


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
    """A Salesforce SOQL/SOSL query or a Gmail ``q`` can be multi-line. The
    value is Composio's to encode, so it must not be refused here; only names,
    headers and the endpoint are checked for control characters."""
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
        (None, {"accept\nX-Injected": "1"}),
        (None, {"accept": "application/json\nX-Injected: 1"}),
        (None, {"accept": "application/json\rX-Injected: 1"}),
        (None, {"accept": "application/json\r\nX-Injected: 1"}),
    ],
    ids=["query-name-lf", "header-name-lf", "header-lf", "header-cr", "header-crlf"],
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


@pytest.mark.parametrize(
    "answer",
    [{"status": 200}, {"status": 200, "data": None}],
    ids=["absent", "null"],
)
def test_a_successful_answer_without_data_is_returned_as_no_body(proxy, answer):
    """Composio's schema makes ``data`` optional and nullable. Whether an
    empty 2xx is a complete answer is the connector's own API's rule:
    Salesforce (#3669) reads it as ``{}``, while Gmail refuses it in its
    transport. The shared client only reports that there is no body."""
    proxy.response = _Response(200, answer)

    response = composio_proxy.proxy_request(
        "GET",
        "https://gmail.googleapis.com/gmail/v1/users/me/labels",
        allowed_hosts=GMAIL_HOSTS,
    )

    assert response.status == 200
    assert response.data is None


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


def test_a_decoded_fragment_across_the_cut_is_redacted_whole(proxy):
    """A decoded body line can cross the cut. Cutting the text before the
    per-line pass used to keep the line's visible prefix."""
    line = "BODY-MARKER-" + "x" * 1188
    raw = base64.urlsafe_b64encode(b"Subject: x\r\n\r\n" + line.encode()).decode()
    proxy.response = _Response(400, {"error": {"message": f"invalid body {line}"}})

    with pytest.raises(composio_proxy.ComposioProxyError) as raised:
        composio_proxy.proxy_request(
            "POST",
            "https://gmail.googleapis.com/gmail/v1/users/me/messages/send",
            allowed_hosts=GMAIL_HOSTS,
            body={"raw": raw},
        )

    text = str(raised.value)
    assert line[:80] not in text
    assert "x" * 100 not in text


def test_a_decoded_fragment_that_contains_a_token_is_redacted_whole(proxy):
    """Redacting the token first rewrites the fragment, so the decoded line
    used to survive with only the token masked. The line must be gone in the
    form redaction leaves it too."""
    line = "please read the Bearer sk-secret-token-value section in this mail"
    raw = base64.urlsafe_b64encode(b"Subject: x\r\n\r\n" + line.encode()).decode()
    proxy.response = _Response(400, {"error": {"message": f"invalid body {line}"}})

    with pytest.raises(composio_proxy.ComposioProxyError) as raised:
        composio_proxy.proxy_request(
            "POST",
            "https://gmail.googleapis.com/gmail/v1/users/me/messages/send",
            allowed_hosts=GMAIL_HOSTS,
            body={"raw": raw},
        )

    text = str(raised.value)
    assert "please read the" not in text
    assert "section in this mail" not in text
    assert "[redacted]" in text


def test_a_short_decoded_fragment_near_the_cut_leaves_one_marker(proxy):
    """Cutting the text once in the body pass and again in the error made the
    tail read ``...[truncated...[truncated]`` when a short fragment near the
    cut shortened the kept text."""
    prefix = "Composio proxy request failed with HTTP 400: "
    fragment = "DECODED-FRAGMENT-1234"
    message = prefix + "z" * (979 - len(prefix)) + fragment + "y" * 500
    raw = base64.urlsafe_b64encode(
        b"Subject: x\r\n\r\n" + f"{fragment}\r\n{'p' * 100}".encode()
    ).decode()
    proxy.response = _Response(400, {"error": {"message": message[len(prefix) :]}})

    with pytest.raises(composio_proxy.ComposioProxyError) as raised:
        composio_proxy.proxy_request(
            "POST",
            "https://gmail.googleapis.com/gmail/v1/users/me/messages/send",
            allowed_hosts=GMAIL_HOSTS,
            body={"raw": raw},
        )

    text = str(raised.value)
    assert fragment not in text
    assert text.endswith("...[truncated]")
    assert text.count("truncat") == 1


def test_a_large_decoded_body_is_redacted_from_a_truncated_error_text(proxy):
    """Only the part of the decoded content that can reach the kept text is
    scanned. A 4.5 MB message decodes to tens of thousands of lines, and each
    one used to run a ``replace`` over the whole untruncated text. The lines
    that do show must still be gone, including one across the cut and one
    whose token redaction rewrote."""
    cross_cut = "CROSS-CUT-MARKER-" + "x" * 1188
    token_line = "please read the Bearer sk-secret-token-value section in this mail"
    lines = [f"attachment-line-{number:05d}-{'x' * 40}" for number in range(20000)]
    raw = base64.urlsafe_b64encode(
        b"Subject: long body\r\n\r\n"
        + f"{cross_cut}\r\n{token_line}\r\n".encode()
        + "\r\n".join(lines).encode()
    ).decode()
    # The error text repeats some of the decoded content, not the whole
    # message, so the body fragment itself is not what redacts it.
    proxy.response = _Response(
        400,
        {
            "error": {
                "message": "invalid body "
                + token_line
                + " "
                + cross_cut
                + " "
                + "y" * 1_000_000
            }
        },
    )

    started = time.monotonic()
    with pytest.raises(composio_proxy.ComposioProxyError) as raised:
        composio_proxy.proxy_request(
            "POST",
            "https://gmail.googleapis.com/gmail/v1/users/me/messages/send",
            allowed_hosts=GMAIL_HOSTS,
            body={"raw": raw},
        )
    elapsed = time.monotonic() - started

    text = str(raised.value)
    assert "please read the" not in text
    assert "section in this mail" not in text
    assert cross_cut[:80] not in text
    assert len(text) == composio_proxy.MAX_ERROR_TEXT_CHARS + len("...[truncated]")
    assert elapsed < 1.0, "the decoded pass must not scan the whole message per line"


def test_a_long_recipient_header_does_not_pull_the_body_into_view(proxy):
    """A group send keeps its recipients on one very long ``To:`` line. When
    the vendor echoes the decoded message, redacting that line shortened the
    text in front of the body, which then started after the kept window: the
    body used to be pulled into the error as a whole."""
    recipients = ",".join(f"user{index:02d}@example.com" for index in range(70))
    body_lines = [
        f"BODY-LINE-{number}-" + marker * 40 for number, marker in enumerate("bcd")
    ]
    decoded = (
        f"To: {recipients}\r\n"
        "Subject: group send\r\n"
        "MIME-Version: 1.0\r\n"
        "\r\n" + "\r\n".join(body_lines)
    )
    raw = base64.urlsafe_b64encode(decoded.encode()).decode()
    proxy.response = _Response(400, {"error": {"message": "invalid body " + decoded}})

    with pytest.raises(composio_proxy.ComposioProxyError) as raised:
        composio_proxy.proxy_request(
            "POST",
            "https://gmail.googleapis.com/gmail/v1/users/me/messages/send",
            allowed_hosts=GMAIL_HOSTS,
            body={"raw": raw},
        )

    text = str(raised.value)
    assert "user00@example.com" not in text
    assert "user69@example.com" not in text
    assert all(line not in text for line in body_lines)
    assert "[redacted]" in text


def test_a_truncated_decoded_echo_does_not_pull_later_lines_into_view(proxy):
    """The vendor can echo only the start of the decoded body. Redacting the
    echoed lines shortened the text, and lines that were never echoed into
    the kept window before were pulled in as whole lines."""
    lines = [f"SECRET-BODY-LINE-{number:03d}-" + "x" * 22 for number in range(100)]
    raw = base64.urlsafe_b64encode(
        b"Subject: x\r\n\r\n" + "\r\n".join(lines).encode()
    ).decode()
    proxy.response = _Response(
        400, {"error": {"message": "invalid body " + "\r\n".join(lines[:40])}}
    )

    with pytest.raises(composio_proxy.ComposioProxyError) as raised:
        composio_proxy.proxy_request(
            "POST",
            "https://gmail.googleapis.com/gmail/v1/users/me/messages/send",
            allowed_hosts=GMAIL_HOSTS,
            body={"raw": raw},
        )

    text = str(raised.value)
    assert all(line not in text for line in lines)
    assert "[redacted]" in text


def test_a_normalised_decoded_echo_does_not_pull_later_lines_into_view(proxy):
    """The vendor can echo the decoded body with LF line endings. The whole
    body then no longer matches and each line is replaced on its own, which
    shortens the text and used to pull the later lines into view."""
    lines = [f"SECRET-BODY-LINE-{number:03d}-" + "x" * 22 for number in range(100)]
    raw = base64.urlsafe_b64encode(
        b"Subject: x\r\n\r\n" + "\r\n".join(lines).encode()
    ).decode()
    proxy.response = _Response(
        400, {"error": {"message": "invalid body " + "\n".join(lines)}}
    )

    with pytest.raises(composio_proxy.ComposioProxyError) as raised:
        composio_proxy.proxy_request(
            "POST",
            "https://gmail.googleapis.com/gmail/v1/users/me/messages/send",
            allowed_hosts=GMAIL_HOSTS,
            body={"raw": raw},
        )

    text = str(raised.value)
    assert all(line not in text for line in lines)
    assert len(text) == composio_proxy.MAX_ERROR_TEXT_CHARS + len("...[truncated]")


def test_a_decoded_body_of_repeated_short_lines_is_looked_at_once(proxy):
    """A body of identical short lines (an attachment's base64 padding, for
    one) yields tens of thousands of the same fragment. One ``find`` and one
    ``redact`` per line made this take over a second; each distinct fragment
    must be looked at once."""
    decoded = "Subject: perf\r\n\r\n" + "\r\n".join(["x" * 8] * 20000) + "\r\n"
    raw = base64.urlsafe_b64encode(decoded.encode()).decode()
    proxy.response = _Response(400, {"error": {"message": "invalid body " + decoded}})

    started = time.monotonic()
    with pytest.raises(composio_proxy.ComposioProxyError) as raised:
        composio_proxy.proxy_request(
            "POST",
            "https://gmail.googleapis.com/gmail/v1/users/me/messages/send",
            allowed_hosts=GMAIL_HOSTS,
            body={"raw": raw},
        )
    elapsed = time.monotonic() - started

    text = str(raised.value)
    assert "x" * 8 not in text
    assert elapsed < 0.5, "the pass must look at each distinct fragment once"


def test_a_body_of_same_prefix_lines_is_looked_up_by_length(proxy):
    """The lookup's worst shape: thousands of distinct lines that all open
    with the same 32 characters, and a vendor that echoes only the first 100
    of them. Every position inside a matched line used to try every fragment
    that shared that opening -- about 2.5 million ``startswith`` calls for
    this echo. A bucket must cost one lookup per distinct length, not one per
    fragment."""
    lines = ["X" * 894 + f"{number:06d}" for number in range(2800)]
    decoded = "Subject: many\r\n\r\n" + "\r\n".join(lines)
    raw = base64.urlsafe_b64encode(decoded.encode()).decode()
    assert len(raw) < composio_proxy.MAX_JSON_BODY_BYTES
    proxy.response = _Response(
        400, {"error": {"message": "invalid body " + "\n".join(lines[:100])}}
    )

    started = time.monotonic()
    with pytest.raises(composio_proxy.ComposioProxyError) as raised:
        composio_proxy.proxy_request(
            "POST",
            "https://gmail.googleapis.com/gmail/v1/users/me/messages/send",
            allowed_hosts=GMAIL_HOSTS,
            body={"raw": raw},
        )
    elapsed = time.monotonic() - started

    text = str(raised.value)
    assert lines[0][:200] not in text
    assert "X" * 100 not in text
    assert elapsed < 1.0, "a same-prefix bucket must be looked up by length"


@pytest.mark.parametrize("prefix_len", [45, 76])
def test_a_partial_echo_of_same_prefix_lines_is_looked_up_by_length(proxy, prefix_len):
    """The same bucket when the vendor echoes only the start of the decoded
    body: the whole-body fragment no longer matches and every line is looked
    up on its own. 20k lines sharing a 45- or 76-character opening used to
    cost a scan over the whole bucket at every position."""
    lines = ["A" * prefix_len + f"{number:05d}" + "y" * 20 for number in range(20000)]
    decoded = "Subject: perf\r\n\r\n" + "\r\n".join(lines)
    raw = base64.urlsafe_b64encode(decoded.encode()).decode()
    proxy.response = _Response(
        400, {"error": {"message": "invalid body " + decoded[:6000]}}
    )

    started = time.monotonic()
    with pytest.raises(composio_proxy.ComposioProxyError) as raised:
        composio_proxy.proxy_request(
            "POST",
            "https://gmail.googleapis.com/gmail/v1/users/me/messages/send",
            allowed_hosts=GMAIL_HOSTS,
            body={"raw": raw},
        )
    elapsed = time.monotonic() - started

    text = str(raised.value)
    assert not [line for line in lines if line in text]
    assert elapsed < 1.0, "a same-prefix bucket must be looked up by length"


@pytest.mark.parametrize("header_len", [900, 1000, 1001, 1200])
def test_a_fragment_that_starts_inside_a_long_match_keeps_no_tail(proxy, header_len):
    """A vendor's echo can put the start of a body line inside a very long
    ``To:`` value and let the line run past the value's end. A match longer
    than the kept text used to be skipped whole, which left the rest of that
    line in view; the scan now starts at the last kept-text worth of
    characters of the match, so a line that overlaps its end is merged into
    one region. 900 and 1000 are the controls that the old scan already
    covered."""
    header = "A" * header_len
    tail = "EDGE-TAIL-SECRET-" + "z" * 280
    body_line = "A" * 500 + tail
    decoded = f"To: {header}\r\nSubject: edge\r\nMIME-Version: 1.0\r\n\r\n{body_line}"
    raw = base64.urlsafe_b64encode(decoded.encode()).decode()
    proxy.response = _Response(
        400, {"error": {"message": "invalid body " + header + tail}}
    )

    with pytest.raises(composio_proxy.ComposioProxyError) as raised:
        composio_proxy.proxy_request(
            "POST",
            "https://gmail.googleapis.com/gmail/v1/users/me/messages/send",
            allowed_hosts=GMAIL_HOSTS,
            body={"raw": raw},
        )

    text = str(raised.value)
    assert "EDGE-TAIL-SECRET-" not in text
    assert "z" * 100 not in text


def test_a_body_of_distinct_lengths_echoed_from_its_tail_is_bounded(proxy):
    """A body whose lines share one opening *and* have 1000 different lengths,
    echoed from its end: every match is just over the kept text, so the scan
    walks the full window at every one of them. The same body echoed from its
    start cost 0.8 s here and its end cost 12.7 s; the work the pass may
    spend is counted, so the tail echo is bounded like the others."""
    lines = ["A" * 32 + "A" * index + "z" for index in range(1000)]
    decoded = "Subject: w\r\n\r\n" + "\r\n".join(lines)
    raw = base64.urlsafe_b64encode(decoded.encode()).decode()
    proxy.response = _Response(
        400, {"error": {"message": "invalid body " + "\r\n".join(lines[-100:])}}
    )

    started = time.monotonic()
    with pytest.raises(composio_proxy.ComposioProxyError) as raised:
        composio_proxy.proxy_request(
            "POST",
            "https://gmail.googleapis.com/gmail/v1/users/me/messages/send",
            allowed_hosts=GMAIL_HOSTS,
            body={"raw": raw},
        )
    elapsed = time.monotonic() - started

    text = str(raised.value)
    assert not [line for line in lines if line in text]
    assert elapsed < 1.0, "distinct lengths must not multiply the work per match"


@pytest.mark.parametrize("line_count", [1000, 1600])
def test_a_body_of_long_lines_with_distinct_lengths_is_bounded(proxy, line_count):
    """The same shape with lines just over the kept text (1000..2599
    characters, 2.0/3.8 MB of raw), so every match costs the full scan window
    rather than its own length. It took 41 s and 77 s here; the counted work
    has to cut it short with the rest of the text hidden, and no line may
    stay visible."""
    lines = ["A" * (1000 + index - 12) + f"{index:012d}" for index in range(line_count)]
    decoded = "Subject: w\r\n\r\n" + "\r\n".join(lines)
    raw = base64.urlsafe_b64encode(decoded.encode()).decode()
    assert len(raw) < composio_proxy.MAX_JSON_BODY_BYTES
    proxy.response = _Response(
        400, {"error": {"message": "invalid body " + "\r\n".join(lines[:100])}}
    )

    started = time.monotonic()
    with pytest.raises(composio_proxy.ComposioProxyError) as raised:
        composio_proxy.proxy_request(
            "POST",
            "https://gmail.googleapis.com/gmail/v1/users/me/messages/send",
            allowed_hosts=GMAIL_HOSTS,
            body={"raw": raw},
        )
    elapsed = time.monotonic() - started

    text = str(raised.value)
    assert not [line for line in lines[:100] if line in text]
    assert elapsed < 1.0, (
        "a long match must not turn distinct lengths into unbounded work"
    )


def test_a_pass_that_runs_out_of_its_work_hides_the_rest(proxy, monkeypatch):
    """The work allowance is a counted bound, not a clock: when the next
    comparison does not fit, everything after it is dropped behind one
    marker. The marker after a 3000-character match is what a growing region
    would otherwise leave in view."""
    monkeypatch.setattr(composio_proxy, "_MAX_REDACTION_WORK", 2000, raising=False)
    message = "invalid body " + "A" * 3000 + "AFTER-THE-MATCH-SECRET"
    decoded = "Subject: w\r\n\r\n" + "A" * 3000
    raw = base64.urlsafe_b64encode(decoded.encode()).decode()
    proxy.response = _Response(400, {"error": {"message": message}})

    with pytest.raises(composio_proxy.ComposioProxyError) as raised:
        composio_proxy.proxy_request(
            "POST",
            "https://gmail.googleapis.com/gmail/v1/users/me/messages/send",
            allowed_hosts=GMAIL_HOSTS,
            body={"raw": raw},
        )

    text = str(raised.value)
    assert "AFTER-THE-MATCH-SECRET" not in text
    assert "A" * 100 not in text


def test_a_pass_with_enough_work_keeps_the_text_after_a_long_match(proxy):
    """The control for the allowance above: with the shipped budget the same
    message is scanned, the match is replaced, and the text after it stays."""
    message = "invalid body " + "A" * 3000 + "AFTER-THE-MATCH-KEPT"
    decoded = "Subject: w\r\n\r\n" + "A" * 3000
    raw = base64.urlsafe_b64encode(decoded.encode()).decode()
    proxy.response = _Response(400, {"error": {"message": message}})

    with pytest.raises(composio_proxy.ComposioProxyError) as raised:
        composio_proxy.proxy_request(
            "POST",
            "https://gmail.googleapis.com/gmail/v1/users/me/messages/send",
            allowed_hosts=GMAIL_HOSTS,
            body={"raw": raw},
        )

    text = str(raised.value)
    assert "AFTER-THE-MATCH-KEPT" in text
    assert "A" * 100 not in text


def test_a_growing_region_that_spends_the_work_hides_the_rest(proxy, monkeypatch):
    """A run that a fragment keeps extending walks its region until the work
    allowance is gone, then takes the rest of the text as one region. The
    later marker must not survive; the shorter allowance only makes the bound
    deterministic, it must never leave fragment text in view."""
    monkeypatch.setattr(composio_proxy, "_MAX_REDACTION_WORK", 1_000_000, raising=False)
    message = "invalid body " + "A" * 40_000 + "\r\nLATE-BODY-SECRET-0123456789"
    decoded = "Subject: w\r\n\r\n" + "A" * 2000 + "\r\n" + "A" * 1001
    raw = base64.urlsafe_b64encode(decoded.encode()).decode()
    proxy.response = _Response(400, {"error": {"message": message}})

    started = time.monotonic()
    with pytest.raises(composio_proxy.ComposioProxyError) as raised:
        composio_proxy.proxy_request(
            "POST",
            "https://gmail.googleapis.com/gmail/v1/users/me/messages/send",
            allowed_hosts=GMAIL_HOSTS,
            body={"raw": raw},
        )
    elapsed = time.monotonic() - started

    text = str(raised.value)
    assert "LATE-BODY-SECRET-0123456789" not in text
    assert "A" * 100 not in text
    assert elapsed < 1.0


@pytest.mark.parametrize("rewrite", ["lf", "crlf", "double-blank", "indent"])
def test_random_bodies_of_distinct_lengths_stay_bounded(proxy, rewrite):
    """A fixed-seed sweep over the shapes the pass used to be slow on: bodies
    whose lines share a 40-character opening and have random lengths, echoed
    with the line endings and indentation a vendor may add. None of the
    lines may stay visible, and each pass has to stay inside the bound."""
    rng = random.Random(20261110)
    for case, (low, high, count) in enumerate(
        ((200, 4000, 120), (900, 1100, 300), (64, 5000, 200))
    ):
        widths = [rng.randint(low, high) for _ in range(count)]
        lines = [
            "Q" * 40 + f"{case}{index:04d}" + "d" * (width - 46)
            for index, width in enumerate(widths)
        ]
        decoded = "Subject: random\r\n\r\n" + "\r\n".join(lines)
        raw = base64.urlsafe_b64encode(decoded.encode()).decode()
        echo = {
            "lf": decoded.replace("\r\n", "\n"),
            "crlf": decoded.replace("\n", "\r\n"),
            "double-blank": decoded.replace("\n", "\n\n"),
            "indent": "\n".join("  " + line for line in decoded.split("\n")),
        }[rewrite]
        proxy.response = _Response(400, {"error": {"message": "invalid body " + echo}})

        started = time.monotonic()
        with pytest.raises(composio_proxy.ComposioProxyError) as raised:
            composio_proxy.proxy_request(
                "POST",
                "https://gmail.googleapis.com/gmail/v1/users/me/messages/send",
                allowed_hosts=GMAIL_HOSTS,
                body={"raw": raw},
            )
        elapsed = time.monotonic() - started

        text = str(raised.value)
        assert not [line for line in lines if line in text]
        assert elapsed < 1.0, f"{rewrite} case {case} took {elapsed:.2f}s"


@pytest.mark.parametrize("line_count", [100, 200])
def test_a_hand_written_body_of_long_lines_echoed_with_lf_is_bounded(proxy, line_count):
    """A hand-written MIME body whose lines are all just over the kept text
    (5,001..5,600 characters) and all a different length, echoed with LF
    endings. Every line matches near its start, so the scan walks its whole
    window at each one and the lookup tries every distinct length that shares
    the opening: 100 and 200 of them cost 12.2 s and 25.0 s here. The counted
    allowance has to cut both, and no line may stay visible."""
    lines = ["A" * (length - 1) + "z" for length in range(5001, 5001 + line_count)]
    decoded = (
        "To: a@example.com\r\nSubject: bench\r\nMIME-Version: 1.0\r\n"
        'Content-Type: text/plain; charset="utf-8"\r\n'
        "Content-Transfer-Encoding: 7bit\r\n\r\n" + "\r\n".join(lines)
    )
    raw = base64.urlsafe_b64encode(decoded.encode()).decode()
    assert len(raw) < composio_proxy.MAX_JSON_BODY_BYTES
    proxy.response = _Response(
        400, {"error": {"message": "invalid body " + decoded.replace("\r\n", "\n")}}
    )

    started = time.monotonic()
    with pytest.raises(composio_proxy.ComposioProxyError) as raised:
        composio_proxy.proxy_request(
            "POST",
            "https://gmail.googleapis.com/gmail/v1/users/me/messages/send",
            allowed_hosts=GMAIL_HOSTS,
            body={"raw": raw},
        )
    elapsed = time.monotonic() - started

    text = str(raised.value)
    assert not [line for line in lines if line in text]
    assert elapsed < 1.0, "distinct long lengths must not multiply the work"


def test_a_long_run_echoed_over_long_lines_stays_bounded(proxy):
    """The same hand-written body at 600 lines, a 5,000-character ``To`` value
    and an echo that is one 4 MB run of ``A``. The run keeps growing the
    region over the ``To`` match while every line length shares the opening,
    which cost 8.6 s here. The allowance has to cut it, and neither the run
    nor a line of the body may stay visible."""
    lines = ["A" * (length - 1) + "z" for length in range(5001, 5601)]
    decoded = (
        "To: " + "A" * 5000 + "\r\nSubject: bench\r\nMIME-Version: 1.0\r\n"
        'Content-Type: text/plain; charset="utf-8"\r\n'
        "Content-Transfer-Encoding: 7bit\r\n\r\n" + "\r\n".join(lines)
    )
    raw = base64.urlsafe_b64encode(decoded.encode()).decode()
    assert len(raw) < composio_proxy.MAX_JSON_BODY_BYTES
    proxy.response = _Response(
        400, {"error": {"message": "invalid body " + "A" * 4_000_000}}
    )

    started = time.monotonic()
    with pytest.raises(composio_proxy.ComposioProxyError) as raised:
        composio_proxy.proxy_request(
            "POST",
            "https://gmail.googleapis.com/gmail/v1/users/me/messages/send",
            allowed_hosts=GMAIL_HOSTS,
            body={"raw": raw},
        )
    elapsed = time.monotonic() - started

    text = str(raised.value)
    assert "A" * 100 not in text
    assert not [line for line in lines if line in text]
    assert elapsed < 1.0, "a growing run must not multiply the work"
