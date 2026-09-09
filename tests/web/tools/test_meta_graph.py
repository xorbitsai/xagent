import json

import requests

from xagent.web.tools.mcp import meta_graph


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
    assert seen["data"] is None
    assert seen["headers"] == {
        "Authorization": "Bearer user-token",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }


def test_graph_request_sends_null_json_for_form_posts(monkeypatch):
    """Existing form-encoded callers must keep sending a null json body (the
    Facebook/Instagram tests assert `json=None` in their exact call shape) --
    `requests` only substitutes a JSON body `if not data and json is not
    None`, so this never changes what actually goes over the wire."""
    monkeypatch.setenv("META_ACCESS_TOKEN", "user-token")
    seen = {}

    def request(**kwargs):
        seen.update(kwargs)
        return MockResponse({"id": "post-1"}, text="{}")

    monkeypatch.setattr(meta_graph.requests, "request", request)

    meta_graph.graph_request("POST", "/page-1/feed", data={"message": "hi"})

    assert seen["json"] is None
    assert seen["data"] == {"message": "hi"}
    assert seen["headers"]["Content-Type"] == "application/x-www-form-urlencoded"


def test_graph_request_rejects_data_and_json_body_together(monkeypatch):
    monkeypatch.setenv("META_ACCESS_TOKEN", "user-token")
    called = False

    def request(**kwargs):
        nonlocal called
        called = True
        return MockResponse()

    monkeypatch.setattr(meta_graph.requests, "request", request)

    try:
        meta_graph.graph_request("POST", "/x", data={"a": 1}, json_body={"b": 2})
    except ValueError as exc:
        assert "either data or json_body" in str(exc)
    else:
        raise AssertionError("expected ValueError")
    assert called is False
