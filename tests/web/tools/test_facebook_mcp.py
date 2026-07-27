import json
from unittest.mock import Mock

import requests

from xagent.web.tools.mcp import facebook


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


def _payload(result: str):
    return json.loads(result)


def test_auth_status_uses_injected_meta_token(monkeypatch):
    monkeypatch.setenv("META_ACCESS_TOKEN", "user-token")
    mock_request = Mock(
        return_value=MockResponse({"id": "user-1", "name": "Alice Meta"})
    )
    monkeypatch.setattr(facebook.requests, "request", mock_request)

    result = _payload(facebook.facebook_auth_status())

    assert result == {
        "status": "success",
        "authenticated": True,
        "user": {"id": "user-1", "name": "Alice Meta", "email": None},
    }
    mock_request.assert_called_once_with(
        method="GET",
        url="https://graph.facebook.com/v25.0/me",
        headers={
            "Authorization": "Bearer user-token",
            "Accept": "application/json",
        },
        params={"fields": "id,name,email"},
        data=None,
        timeout=30,
    )


def test_list_pages_hides_page_access_tokens(monkeypatch):
    monkeypatch.setenv("META_ACCESS_TOKEN", "user-token")
    mock_request = Mock(
        return_value=MockResponse(
            {
                "data": [
                    {
                        "id": "page-1",
                        "name": "Launch Page",
                        "category": "Software",
                        "tasks": ["CREATE_CONTENT"],
                        "access_token": "page-token",
                    }
                ]
            }
        )
    )
    monkeypatch.setattr(facebook.requests, "request", mock_request)

    result = _payload(facebook.facebook_list_pages())

    assert result == {
        "status": "success",
        "pages": [
            {
                "id": "page-1",
                "name": "Launch Page",
                "category": "Software",
                "tasks": ["CREATE_CONTENT"],
                "has_access_token": True,
            }
        ],
    }
    assert mock_request.call_args.kwargs["url"] == (
        "https://graph.facebook.com/v25.0/me/accounts"
    )
    assert mock_request.call_args.kwargs["params"] == {
        "fields": "id,name,category,tasks,access_token"
    }


def test_list_pages_treats_non_list_data_as_empty(monkeypatch):
    monkeypatch.setenv("META_ACCESS_TOKEN", "user-token")
    monkeypatch.setattr(
        facebook.requests,
        "request",
        Mock(return_value=MockResponse({"data": None})),
    )

    result = _payload(facebook.facebook_list_pages())

    assert result == {"status": "success", "pages": []}


def test_list_page_posts_uses_page_access_token(monkeypatch):
    monkeypatch.setenv("META_ACCESS_TOKEN", "user-token")

    def request(method, url, **kwargs):
        if url.endswith("/me/accounts"):
            return MockResponse(
                {
                    "data": [
                        {
                            "id": "page-1",
                            "name": "Launch Page",
                            "access_token": "page-token",
                        }
                    ]
                }
            )
        assert method == "GET"
        assert url == "https://graph.facebook.com/v25.0/page-1/feed"
        assert kwargs["headers"]["Authorization"] == "Bearer page-token"
        assert kwargs["params"] == {
            "fields": (
                "id,message,created_time,permalink_url,full_picture,status_type,"
                "likes.summary(true),comments.summary(true),shares"
            ),
            "limit": 5,
        }
        return MockResponse(
            {
                "data": [{"id": "post-1", "message": "hello"}],
                "paging": {"next": "https://graph.facebook.com/next"},
            }
        )

    monkeypatch.setattr(facebook.requests, "request", Mock(side_effect=request))

    result = _payload(facebook.facebook_list_page_posts("page-1", limit=5))

    assert result == {
        "status": "success",
        "posts": [{"id": "post-1", "message": "hello"}],
        "next_link": "https://graph.facebook.com/next",
    }


def test_list_post_comments_uses_page_access_token(monkeypatch):
    monkeypatch.setenv("META_ACCESS_TOKEN", "user-token")

    def request(method, url, **kwargs):
        if url.endswith("/me/accounts"):
            return MockResponse(
                {
                    "data": [
                        {
                            "id": "page-1",
                            "name": "Launch Page",
                            "access_token": "page-token",
                        }
                    ]
                }
            )
        assert method == "GET"
        assert url == "https://graph.facebook.com/v25.0/page-1_post-1/comments"
        assert kwargs["headers"]["Authorization"] == "Bearer page-token"
        assert kwargs["params"] == {
            "fields": "id,message,created_time,from",
            "filter": "stream",
            "limit": 5,
        }
        return MockResponse(
            {
                "data": [
                    {
                        "id": "comment-1",
                        "message": "Great news!",
                        "from": {"id": "fan-1", "name": "A Fan"},
                    }
                ],
                "paging": {"next": "https://graph.facebook.com/next"},
            }
        )

    monkeypatch.setattr(facebook.requests, "request", Mock(side_effect=request))

    result = _payload(
        facebook.facebook_list_post_comments("page-1", "page-1_post-1", limit=5)
    )

    assert result == {
        "status": "success",
        "comments": [
            {
                "id": "comment-1",
                "message": "Great news!",
                "from": {"id": "fan-1", "name": "A Fan"},
            }
        ],
        "next_link": "https://graph.facebook.com/next",
    }


def test_publish_text_post_uses_page_access_token_and_message_payload(monkeypatch):
    monkeypatch.setenv("META_ACCESS_TOKEN", "user-token")

    def request(method, url, **kwargs):
        if url.endswith("/me/accounts"):
            return MockResponse(
                {"data": [{"id": "page-1", "access_token": "page-token"}]}
            )
        assert method == "POST"
        assert url == "https://graph.facebook.com/v25.0/page-1/feed"
        assert kwargs["headers"] == {
            "Authorization": "Bearer page-token",
            "Accept": "application/json",
            "Content-Type": "application/x-www-form-urlencoded",
        }
        assert kwargs["data"] == {"message": "Launch update"}
        return MockResponse({"id": "page-1_post-1"})

    monkeypatch.setattr(facebook.requests, "request", Mock(side_effect=request))

    result = _payload(facebook.facebook_publish_text_post("page-1", "Launch update"))

    assert result == {
        "status": "success",
        "post_id": "page-1_post-1",
    }


def test_publish_image_post_uses_public_url_payload(monkeypatch):
    monkeypatch.setenv("META_ACCESS_TOKEN", "user-token")

    def request(method, url, **kwargs):
        if url.endswith("/me/accounts"):
            return MockResponse(
                {"data": [{"id": "page-1", "access_token": "page-token"}]}
            )
        assert method == "POST"
        assert url == "https://graph.facebook.com/v25.0/page-1/photos"
        assert kwargs["headers"]["Authorization"] == "Bearer page-token"
        assert kwargs["data"] == {
            "url": "https://example.com/image.png",
            "published": "true",
            "caption": "Launch visual",
        }
        return MockResponse({"id": "photo-1", "post_id": "page-1_post-1"})

    monkeypatch.setattr(facebook.requests, "request", Mock(side_effect=request))

    result = _payload(
        facebook.facebook_publish_image_post(
            "page-1", "https://example.com/image.png", caption="Launch visual"
        )
    )

    assert result == {
        "status": "success",
        "photo_id": "photo-1",
        "post_id": "page-1_post-1",
    }


def test_graph_api_errors_are_structured_and_redact_tokens(monkeypatch):
    monkeypatch.setenv("META_ACCESS_TOKEN", "user-token")
    monkeypatch.setattr(
        facebook.requests,
        "request",
        Mock(
            return_value=MockResponse(
                {
                    "error": {
                        "message": "Bad OAuth token user-token",
                        "code": 190,
                    }
                },
                text='{"error":{"message":"Bad OAuth token user-token","code":190}}',
                status_code=400,
            )
        ),
    )

    result = _payload(facebook.facebook_auth_status())

    assert result["status"] == "error"
    assert "user-token" not in json.dumps(result)
    assert "[redacted]" in json.dumps(result)
    assert result["details"]["error"]["code"] == 190


def test_graph_api_error_message_truncates_large_response_text(monkeypatch):
    monkeypatch.setenv("META_ACCESS_TOKEN", "user-token")
    long_response_text = "x" * 1200
    monkeypatch.setattr(
        facebook.requests,
        "request",
        Mock(return_value=MockResponse(text=long_response_text, status_code=502)),
    )

    result = _payload(facebook.facebook_auth_status())

    assert result["status"] == "error"
    assert "... [truncated]" in result["message"]
    assert "x" * 1001 not in result["message"]


def test_page_token_is_redacted_from_publish_errors(monkeypatch):
    monkeypatch.setenv("META_ACCESS_TOKEN", "user-token")

    def request(method, url, **kwargs):
        if url.endswith("/me/accounts"):
            return MockResponse(
                {"data": [{"id": "page-1", "access_token": "page-token"}]}
            )
        return MockResponse(
            {
                "error": {
                    "message": "Bad Page token page-token",
                    "code": 190,
                }
            },
            text='{"error":{"message":"Bad Page token page-token","code":190}}',
            status_code=400,
        )

    monkeypatch.setattr(facebook.requests, "request", Mock(side_effect=request))

    result = _payload(facebook.facebook_publish_text_post("page-1", "Launch update"))

    serialized = json.dumps(result)
    assert result["status"] == "error"
    assert "page-token" not in serialized
    assert "[redacted]" in serialized
    assert result["details"]["error"]["code"] == 190


def test_publish_image_post_rejects_non_public_url(monkeypatch):
    monkeypatch.setenv("META_ACCESS_TOKEN", "user-token")
    mock_request = Mock()
    monkeypatch.setattr(facebook.requests, "request", mock_request)

    result = _payload(
        facebook.facebook_publish_image_post("page-1", "/tmp/local-image.png")
    )

    assert result == {
        "status": "error",
        "message": "image_url must be a public http or https URL",
    }
    mock_request.assert_not_called()
