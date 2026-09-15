import json
from unittest.mock import Mock

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


def test_get_permissions_maps_permission_to_status(monkeypatch):
    monkeypatch.setenv("META_ACCESS_TOKEN", "user-token")
    monkeypatch.setattr(
        meta_graph.requests,
        "request",
        Mock(
            return_value=MockResponse(
                {
                    "data": [
                        {"permission": "pages_show_list", "status": "granted"},
                        {"permission": "ads_read", "status": "declined"},
                    ]
                }
            )
        ),
    )

    assert meta_graph.get_permissions() == {
        "pages_show_list": "granted",
        "ads_read": "declined",
    }


def test_get_permissions_treats_non_list_data_as_empty(monkeypatch):
    monkeypatch.setenv("META_ACCESS_TOKEN", "user-token")
    monkeypatch.setattr(
        meta_graph.requests,
        "request",
        Mock(return_value=MockResponse({"data": None})),
    )

    assert meta_graph.get_permissions() == {}


def test_missing_permissions_flags_ungranted_and_absent_permissions():
    granted = {"pages_show_list": "granted", "ads_read": "declined"}

    assert meta_graph.missing_permissions(
        ["pages_show_list", "ads_read", "pages_manage_posts"], granted
    ) == ["ads_read", "pages_manage_posts"]
