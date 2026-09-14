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


def test_pagination_fields_reports_has_more_when_next_link_present():
    result = {
        "data": [{"id": "1"}],
        "paging": {
            "cursors": {"before": "b", "after": "cursor-1"},
            "next": "https://graph.facebook.com/next",
        },
    }

    assert meta_graph.pagination_fields(result) == {
        "next_link": "https://graph.facebook.com/next",
        "after_cursor": "cursor-1",
        "has_more": True,
    }


def test_pagination_fields_reports_no_more_on_last_page():
    """The Graph API can still return a cursors.after value on the last
    page -- has_more must key off paging.next, not the mere presence of a
    cursor, or callers would loop forever passing a cursor that yields no
    further rows."""
    result = {
        "data": [{"id": "1"}],
        "paging": {"cursors": {"before": "b", "after": "cursor-last"}},
    }

    assert meta_graph.pagination_fields(result) == {
        "next_link": None,
        "after_cursor": "cursor-last",
        "has_more": False,
    }


def test_pagination_fields_handles_missing_paging():
    assert meta_graph.pagination_fields({"data": []}) == {
        "next_link": None,
        "after_cursor": None,
        "has_more": False,
    }
