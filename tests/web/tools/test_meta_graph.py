import json
import logging
from unittest.mock import Mock

import pytest
import requests

from xagent.web.tools.mcp import composio_proxy, meta_graph


class MockResponse:
    def __init__(self, json_data=None, text="", status_code=200):
        self._json_data = json_data or {}
        self.text = text
        self.status_code = status_code
        self.content = text.encode("utf-8") if text else b"{}"

    def json(self):
        return self._json_data

    def raise_for_status(self):
        if self.status_code >= 400:
            error = requests.HTTPError(f"HTTP {self.status_code}")
            error.response = self
            raise error


def test_graph_request_redacts_env_and_request_tokens(monkeypatch):
    monkeypatch.setenv("META_ACCESS_TOKEN", "user-token")

    def request(method, url, **kwargs):
        assert method == "GET"
        assert url == "https://graph.facebook.com/v25.0/page-1/feed"
        assert kwargs["headers"]["Authorization"] == "Bearer page-token"
        return MockResponse(
            {"error": {"message": "Bad Page token page-token user-token"}},
            text="Bad Page token page-token user-token",
            status_code=400,
        )

    monkeypatch.setattr(meta_graph.requests, "request", request)

    try:
        meta_graph.graph_request("GET", "/page-1/feed", token="page-token")
    except meta_graph.GraphAPIError as exc:
        result = json.loads(meta_graph.graph_error_response(exc))
    else:
        raise AssertionError("expected GraphAPIError")

    serialized = json.dumps(result)
    assert "page-token" not in serialized
    assert "user-token" not in serialized
    assert "[redacted]" in serialized


def test_response_error_text_truncates_large_body():
    response = MockResponse(text="x" * 1200, status_code=502)

    assert meta_graph.response_error_text(response) == (
        "x" * meta_graph.MAX_ERROR_RESPONSE_TEXT_CHARS + "... [truncated]"
    )


def test_graph_request_sends_json_body_with_json_content_type(monkeypatch):
    monkeypatch.setenv("META_ACCESS_TOKEN", "user-token")
    seen = {}

    def request(**kwargs):
        seen.update(kwargs)
        return MockResponse({"messages": [{"id": "wamid.1"}]}, text="{}")

    monkeypatch.setattr(meta_graph.requests, "request", request)

    result = meta_graph.graph_request(
        "POST", "/pn-1/messages", json_body={"type": "text", "text": {"body": "hi"}}
    )

    assert result == {"messages": [{"id": "wamid.1"}]}
    assert seen["json"] == {"type": "text", "text": {"body": "hi"}}
    assert "data" not in seen
    assert seen["headers"] == {
        "Authorization": "Bearer user-token",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }


def test_graph_request_sends_no_json_kwarg_for_form_posts(monkeypatch):
    """Existing form-encoded callers (Facebook/Instagram/Meta Ads) must keep
    getting no `json` kwarg at all -- graph_request builds kwargs
    conditionally rather than always passing `json=` and relying on
    `requests`' own `if not data and json is not None` precedence."""
    monkeypatch.setenv("META_ACCESS_TOKEN", "user-token")
    seen = {}

    def request(**kwargs):
        seen.update(kwargs)
        return MockResponse({"id": "post-1"}, text="{}")

    monkeypatch.setattr(meta_graph.requests, "request", request)

    meta_graph.graph_request("POST", "/page-1/feed", data={"message": "hi"})

    assert "json" not in seen
    assert seen["data"] == {"message": "hi"}
    assert seen["headers"]["Content-Type"] == "application/x-www-form-urlencoded"


def test_graph_request_omits_data_and_json_kwargs_for_plain_get(monkeypatch):
    """A bare GET (neither data nor json_body) must get neither kwarg at
    all -- pinned directly against meta_graph.graph_request rather than only
    indirectly through each connector's own auth_status test."""
    monkeypatch.setenv("META_ACCESS_TOKEN", "user-token")
    seen = {}

    def request(**kwargs):
        seen.update(kwargs)
        return MockResponse({"id": "u1"}, text="{}")

    monkeypatch.setattr(meta_graph.requests, "request", request)

    meta_graph.graph_request("GET", "/me", params={"fields": "id"})

    assert "data" not in seen
    assert "json" not in seen
    assert "Content-Type" not in seen["headers"]


def test_graph_request_rejects_data_and_json_body_together(monkeypatch):
    monkeypatch.setenv("META_ACCESS_TOKEN", "user-token")
    request = Mock()
    monkeypatch.setattr(meta_graph.requests, "request", request)

    with pytest.raises(ValueError, match="either json_body or form/multipart data"):
        meta_graph.graph_request("POST", "/x", data={"a": 1}, json_body={"b": 2})

    request.assert_not_called()


def test_graph_request_rejects_files_and_json_body_together(monkeypatch):
    monkeypatch.setenv("META_ACCESS_TOKEN", "user-token")
    request = Mock()
    monkeypatch.setattr(meta_graph.requests, "request", request)

    with pytest.raises(ValueError, match="either json_body or form/multipart data"):
        meta_graph.graph_request(
            "POST", "/x", json_body={"a": 1}, files={"source": object()}
        )

    request.assert_not_called()


def test_graph_request_sends_multipart_without_content_type(monkeypatch):
    monkeypatch.setenv("META_ACCESS_TOKEN", "user-token")
    seen = {}

    def request(**kwargs):
        seen.update(kwargs)
        return MockResponse({"id": "photo-1"}, text="{}")

    monkeypatch.setattr(meta_graph.requests, "request", request)
    source = ("image.png", object(), "image/png")

    meta_graph.graph_request(
        "POST", "/page-1/photos", data={"caption": "hello"}, files={"source": source}
    )

    assert seen["data"] == {"caption": "hello"}
    assert seen["files"] == {"source": source}
    assert "Content-Type" not in seen["headers"]


def test_graph_headers_omits_content_type_by_default():
    assert meta_graph.graph_headers("tok") == {
        "Authorization": "Bearer tok",
        "Accept": "application/json",
    }


def test_graph_headers_sets_given_content_type():
    assert meta_graph.graph_headers("tok", content_type="application/json") == {
        "Authorization": "Bearer tok",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }


def test_graph_path_quotes_and_appends_suffix():
    assert meta_graph.graph_path("waba/../evil", "phone_numbers") == (
        "/waba%2F..%2Fevil/phone_numbers"
    )
    assert meta_graph.graph_path("pn-1") == "/pn-1"


def test_graph_path_requires_object_id():
    with pytest.raises(ValueError, match="object id is required"):
        meta_graph.graph_path("   ")


def test_auth_status_success(monkeypatch):
    monkeypatch.setenv("META_ACCESS_TOKEN", "user-token")
    monkeypatch.setattr(
        meta_graph.requests,
        "request",
        lambda **kwargs: MockResponse({"id": "u1", "name": "Alice"}),
    )

    result = json.loads(meta_graph.auth_status(logging.getLogger(__name__), "Test"))

    assert result == {
        "status": "success",
        "authenticated": True,
        "user": {"id": "u1", "name": "Alice", "email": None},
    }


def test_auth_status_surfaces_graph_error(monkeypatch):
    monkeypatch.setenv("META_ACCESS_TOKEN", "user-token")
    body = {"error": {"message": "bad token", "code": 190}}
    monkeypatch.setattr(
        meta_graph.requests,
        "request",
        lambda **kwargs: MockResponse(body, text=json.dumps(body), status_code=401),
    )

    result = json.loads(meta_graph.auth_status(logging.getLogger(__name__), "Test"))

    assert result["status"] == "error"
    assert result["details"]["error"]["code"] == 190


def test_auth_status_without_token_returns_error(monkeypatch):
    monkeypatch.delenv("META_ACCESS_TOKEN", raising=False)
    request = Mock()
    monkeypatch.setattr(meta_graph.requests, "request", request)

    result = json.loads(meta_graph.auth_status(logging.getLogger(__name__), "Test"))

    assert result["status"] == "error"
    assert "META_ACCESS_TOKEN" in result["message"]
    request.assert_not_called()


COMPOSIO_PROXY_URL = "https://backend.composio.dev/api/v3.1/tools/execute/proxy"


@pytest.fixture()
def composio_env(monkeypatch):
    """The env a Composio-backed Facebook grant launches with (#3531)."""
    monkeypatch.delenv("META_ACCESS_TOKEN", raising=False)
    monkeypatch.setenv("COMPOSIO_API_KEY", "composio-key")
    monkeypatch.setenv("COMPOSIO_CONNECTED_ACCOUNT_ID", "ca_123")
    monkeypatch.setenv("COMPOSIO_USER_ID", "actor-hmac")
    # Page tokens seen in proxied responses are remembered process-wide; start
    # each test without the previous test's tokens.
    monkeypatch.setattr(meta_graph, "_proxy_response_secrets", set(), raising=False)


def _proxy_response(data=None, *, status=200):
    body = {"data": data, "status": status, "headers": {}}
    return MockResponse(body, text=json.dumps(body))


def _capture_proxy(monkeypatch, *responses):
    calls = []
    pending = list(responses)

    def request(**kwargs):
        calls.append(kwargs)
        return pending.pop(0)

    monkeypatch.setattr(meta_graph.requests, "request", request)
    return calls


def test_proxy_get_maps_params_to_query_and_sends_no_authorization(
    monkeypatch, composio_env
):
    calls = _capture_proxy(monkeypatch, _proxy_response({"id": "fb-user-1"}))

    result = meta_graph.graph_request(
        "GET", "/me", params={"fields": "id,name", "limit": 5}, timeout=12
    )

    assert result == {"id": "fb-user-1"}
    [call] = calls
    assert call["method"] == "POST"
    assert call["url"] == COMPOSIO_PROXY_URL
    assert call["headers"]["x-api-key"] == "composio-key"
    assert "Authorization" not in call["headers"]
    assert call["timeout"] == 12
    assert call["json"] == {
        "endpoint": "https://graph.facebook.com/v25.0/me",
        "method": "GET",
        "connected_account_id": "ca_123",
        "user_id": "actor-hmac",
        "parameters": [
            {"name": "fields", "value": "id,name", "type": "query"},
            {"name": "limit", "value": "5", "type": "query"},
        ],
    }


def test_proxy_posts_one_accepted_request(monkeypatch, composio_env):
    _capture_proxy(monkeypatch, _proxy_response({"id": "fb-user-1"}))

    token = composio_proxy.begin_request_count()
    try:
        meta_graph.graph_request("GET", "/me")
    finally:
        accepted = composio_proxy.end_request_count(token)

    assert accepted == 1


def test_a_refused_proxy_call_accepts_nothing(monkeypatch, composio_env):
    body = {"error": {"message": "rate limited", "slug": "TooManyRequests"}}
    _capture_proxy(
        monkeypatch, MockResponse(body, text=json.dumps(body), status_code=429)
    )

    token = composio_proxy.begin_request_count()
    try:
        with pytest.raises(meta_graph.ComposioProxyError):
            meta_graph.graph_request("GET", "/me")
    finally:
        accepted = composio_proxy.end_request_count(token)

    assert accepted == 0


@pytest.mark.parametrize(
    "kwargs, path",
    [
        ({"data": {"message": "hi"}}, "/page-1/feed"),
        ({"json_body": {"type": "text", "text": {"body": "hi"}}}, "/pn-1/messages"),
    ],
)
def test_proxy_sends_form_or_json_data_as_body(monkeypatch, composio_env, kwargs, path):
    """graph_request takes a body as form data or as JSON; the proxy carries
    either as its JSON ``body`` and must not drop one."""
    calls = _capture_proxy(monkeypatch, _proxy_response({"id": "x"}))

    meta_graph.graph_request("POST", path, **kwargs)

    payload = calls[0]["json"]
    assert payload["method"] == "POST"
    assert payload["body"] == next(iter(kwargs.values()))
    assert "data" not in calls[0]


def test_proxy_sends_page_token_as_access_token_query_parameter(
    monkeypatch, composio_env
):
    calls = _capture_proxy(monkeypatch, _proxy_response({"data": []}))

    meta_graph.graph_request("GET", "/page-1/feed", token="page-token")

    call = calls[0]
    assert {
        "name": "access_token",
        "value": "page-token",
        "type": "query",
    } in call["json"]["parameters"]
    assert "page-token" not in json.dumps(call["headers"])
    assert "Authorization" not in call["headers"]


def test_proxy_upstream_graph_error_keeps_the_graph_error_body(
    monkeypatch, composio_env
):
    graph_body = {"error": {"message": "(#10) no permission", "code": 10}}
    _capture_proxy(monkeypatch, _proxy_response(graph_body, status=403))

    with pytest.raises(meta_graph.GraphAPIError) as caught:
        meta_graph.graph_request("GET", "/page-1/feed", token="page-token")

    assert caught.value.details == graph_body
    assert "403" in str(caught.value)


@pytest.mark.parametrize("http_status", [401, 404, 429, 500])
def test_proxy_composio_failure_is_distinct_and_redacted(
    monkeypatch, composio_env, http_status
):
    """A failure from Composio itself (bad key, unknown account, rate limit,
    its own outage) is not a Graph error and carries no secret. The message
    echoes request values, as the 404 a live call gave for an unknown
    account did."""
    body = {
        "error": {
            "message": 'Connected account "ca_123" (actor-hmac) not found; '
            "key composio-key, token page-token",
            "code": 10,
            "slug": "ConnectedAccount_ResourceNotFound",
            "status": http_status,
        }
    }
    _capture_proxy(
        monkeypatch,
        MockResponse(body, text=json.dumps(body), status_code=http_status),
    )

    with pytest.raises(meta_graph.GraphAPIError) as caught:
        meta_graph.graph_request("GET", "/page-1/feed", token="page-token")

    result = json.loads(meta_graph.graph_error_response(caught.value))
    serialized = json.dumps(result)
    assert "Composio" in result["message"]
    assert str(http_status) in result["message"]
    # Not a Graph error body, so callers that inspect Graph error codes
    # (the Facebook engagement fallback) do not mistake it for one.
    assert "error" not in (result.get("details") or {})
    for secret in ("composio-key", "ca_123", "actor-hmac", "page-token"):
        assert secret not in serialized
        assert secret not in str(caught.value)


def test_proxy_redacts_before_truncating_a_long_error(monkeypatch, composio_env):
    """A secret straddling the truncation point must not leave a prefix.

    The error message is a 45-character prefix plus this text, so the secret
    starts at index 995 and the 1000-character cut falls inside it."""
    message = "x" * 950 + "SECRETPAGETOKEN"
    body = {"error": {"message": message, "status": 400}}
    _capture_proxy(
        monkeypatch, MockResponse(body, text=json.dumps(body), status_code=400)
    )
    monkeypatch.setattr(meta_graph, "_proxy_response_secrets", {"SECRETPAGETOKEN"})

    with pytest.raises(meta_graph.GraphAPIError) as caught:
        meta_graph.graph_request("GET", "/me")

    assert "SECRE" not in str(caught.value)
    assert "SECRE" not in json.dumps(caught.value.details)


class _NonJsonResponse(MockResponse):
    def json(self):
        raise ValueError("not JSON")


@pytest.mark.parametrize("http_status", [200, 502])
def test_proxy_non_json_response_is_a_composio_error(
    monkeypatch, composio_env, http_status
):
    """An HTML error page or an empty body from Composio, on success or
    failure status alike, is reported as a Composio failure, not a crash."""
    _capture_proxy(
        monkeypatch,
        _NonJsonResponse(text="<html>Bad gateway</html>", status_code=http_status),
    )

    with pytest.raises(meta_graph.ComposioProxyError) as caught:
        meta_graph.graph_request("GET", "/me")

    assert "Composio" in str(caught.value)


def test_proxy_redacts_page_tokens_returned_by_earlier_calls(monkeypatch, composio_env):
    accounts = {
        "data": [
            {"id": "page-1", "access_token": "page-one-token"},
            {"id": "page-2", "access_token": "page-two-token"},
        ]
    }
    leaked = {"error": {"message": "conflict with page-two-token", "code": 100}}
    _capture_proxy(
        monkeypatch,
        _proxy_response(accounts),
        _proxy_response(leaked, status=400),
    )

    meta_graph.graph_request("GET", "/me/accounts")
    with pytest.raises(meta_graph.GraphAPIError) as caught:
        meta_graph.graph_request("GET", "/page-1/feed", token="page-one-token")

    serialized = meta_graph.graph_error_response(caught.value)
    assert "page-two-token" not in serialized
    assert "page-two-token" not in str(caught.value)


def test_proxy_drops_the_page_token_graph_echoes_into_paging_links(
    monkeypatch, composio_env
):
    """Graph copies a query-parameter token into paging.next; in proxy mode
    that is the Page token, and next_link reaches the model."""
    next_link = (
        "https://graph.facebook.com/v25.0/page-1/feed"
        "?access_token=page-token&fields=id&limit=1&after=CURSOR"
    )
    _capture_proxy(
        monkeypatch,
        _proxy_response(
            {
                "data": [{"id": "post-1"}],
                "paging": {"cursors": {"after": "CURSOR"}, "next": next_link},
            }
        ),
    )

    result = meta_graph.graph_request("GET", "/page-1/feed", token="page-token")

    assert result["paging"]["next"] == (
        "https://graph.facebook.com/v25.0/page-1/feed?fields=id&limit=1&after=CURSOR"
    )
    assert "page-token" not in json.dumps(result)


@pytest.mark.parametrize("proxied", [False, True])
def test_graph_request_refuses_a_base_url_outside_the_graph_hosts(monkeypatch, proxied):
    """The token (or the Composio key) is never sent to any other host."""
    if proxied:
        monkeypatch.setenv("COMPOSIO_API_KEY", "composio-key")
        monkeypatch.setenv("COMPOSIO_CONNECTED_ACCOUNT_ID", "ca_123")
        monkeypatch.setenv("COMPOSIO_USER_ID", "actor-hmac")
    else:
        monkeypatch.delenv("COMPOSIO_CONNECTED_ACCOUNT_ID", raising=False)
        monkeypatch.setenv("META_ACCESS_TOKEN", "user-token")
    request = Mock()
    monkeypatch.setattr(meta_graph.requests, "request", request)

    for base_url in (
        "https://attacker.example/v25.0",
        "http://graph.instagram.com/v25.0",
        "https://graph.instagram.com/v25.0/",
        "https://graph.instagram.com/v24.0",
        "https://graph.instagram.com.evil/v25.0",
        "https://graph.facebook.com.evil/v25.0",
    ):
        with pytest.raises(ValueError, match="base_url"):
            meta_graph.graph_request("GET", "/me", base_url=base_url)
    request.assert_not_called()


def test_graph_request_accepts_the_instagram_host_natively(monkeypatch):
    """The allowlist admits exactly the two Graph hosts; a later loosening
    of the exact match is caught by the near misses above."""
    monkeypatch.delenv("COMPOSIO_CONNECTED_ACCOUNT_ID", raising=False)
    monkeypatch.setenv("META_ACCESS_TOKEN", "user-token")
    request = Mock(return_value=MockResponse({"id": "1"}, text="{}"))
    monkeypatch.setattr(meta_graph.requests, "request", request)

    meta_graph.graph_request("GET", "/me", base_url="https://graph.instagram.com/v25.0")

    assert request.call_args.kwargs["url"] == "https://graph.instagram.com/v25.0/me"


@pytest.mark.parametrize("serves", [True, False], ids=["composio", "native"])
def test_uses_composio_proxy_defers_to_the_shared_client(monkeypatch, serves):
    """The grant check has one definition, in composio_proxy; meta_graph must
    not keep a second copy that can drift from it."""
    monkeypatch.setattr(composio_proxy, "serves_composio_grant", lambda: serves)

    assert meta_graph.uses_composio_proxy() is serves
