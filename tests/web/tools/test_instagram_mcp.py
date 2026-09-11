import json
from unittest.mock import Mock

import requests

from xagent.web.tools.mcp import instagram


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
    monkeypatch.setattr(instagram.requests, "request", mock_request)

    result = _payload(instagram.instagram_auth_status())

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
        json=None,
        timeout=30,
    )


def test_list_linked_accounts_returns_pages_with_instagram_accounts(monkeypatch):
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
                        "instagram_business_account": {
                            "id": "ig-1",
                            "username": "launch",
                            "name": "Launch",
                            "profile_picture_url": "https://example.com/avatar.jpg",
                        },
                    },
                    {
                        "id": "page-2",
                        "name": "No Instagram",
                        "access_token": "page-token-2",
                    },
                ]
            }
        )
    )
    monkeypatch.setattr(instagram.requests, "request", mock_request)

    result = _payload(instagram.instagram_list_linked_accounts())

    assert result == {
        "status": "success",
        "accounts": [
            {
                "page": {
                    "id": "page-1",
                    "name": "Launch Page",
                    "category": "Software",
                    "tasks": ["CREATE_CONTENT"],
                    "has_access_token": True,
                },
                "instagram_account": {
                    "id": "ig-1",
                    "username": "launch",
                    "name": "Launch",
                    "profile_picture_url": "https://example.com/avatar.jpg",
                },
            }
        ],
    }
    assert mock_request.call_args.kwargs["url"] == (
        "https://graph.facebook.com/v25.0/me/accounts"
    )
    assert mock_request.call_args.kwargs["params"] == {
        "fields": (
            "id,name,category,tasks,access_token,"
            "instagram_business_account{id,username,name,profile_picture_url}"
        )
    }
    assert "page-token" not in json.dumps(result)


def test_list_linked_accounts_treats_non_list_data_as_empty(monkeypatch):
    monkeypatch.setenv("META_ACCESS_TOKEN", "user-token")
    monkeypatch.setattr(
        instagram.requests,
        "request",
        Mock(return_value=MockResponse({"data": None})),
    )

    result = _payload(instagram.instagram_list_linked_accounts())

    assert result == {"status": "success", "accounts": []}


def test_get_profile_reads_selected_instagram_account(monkeypatch):
    monkeypatch.setenv("META_ACCESS_TOKEN", "user-token")
    mock_request = Mock(
        return_value=MockResponse(
            {
                "id": "ig-1",
                "username": "launch",
                "name": "Launch",
                "biography": "Build notes",
                "followers_count": 10,
                "media_count": 2,
            }
        )
    )
    monkeypatch.setattr(instagram.requests, "request", mock_request)

    result = _payload(instagram.instagram_get_profile("ig-1"))

    assert result["status"] == "success"
    assert result["profile"]["username"] == "launch"
    mock_request.assert_called_once_with(
        method="GET",
        url="https://graph.facebook.com/v25.0/ig-1",
        headers={
            "Authorization": "Bearer user-token",
            "Accept": "application/json",
        },
        params={
            "fields": (
                "id,username,name,biography,profile_picture_url,followers_count,"
                "follows_count,media_count,website"
            )
        },
        data=None,
        json=None,
        timeout=30,
    )


def test_list_media_reads_recent_instagram_media(monkeypatch):
    monkeypatch.setenv("META_ACCESS_TOKEN", "user-token")
    mock_request = Mock(
        return_value=MockResponse(
            {
                "data": [{"id": "media-1", "caption": "hello"}],
                "paging": {"next": "https://graph.facebook.com/next"},
            }
        )
    )
    monkeypatch.setattr(instagram.requests, "request", mock_request)

    result = _payload(instagram.instagram_list_media("ig-1", limit=5))

    assert result == {
        "status": "success",
        "media": [{"id": "media-1", "caption": "hello"}],
        "next_link": "https://graph.facebook.com/next",
    }
    assert mock_request.call_args.kwargs["url"] == (
        "https://graph.facebook.com/v25.0/ig-1/media"
    )
    assert mock_request.call_args.kwargs["params"] == {
        "fields": (
            "id,caption,media_type,media_url,permalink,timestamp,username,thumbnail_url"
        ),
        "limit": 5,
    }


def test_publish_image_creates_container_then_publishes(monkeypatch):
    monkeypatch.setenv("META_ACCESS_TOKEN", "user-token")

    def request(method, url, **kwargs):
        if url == "https://graph.facebook.com/v25.0/ig-1/media":
            assert method == "POST"
            assert kwargs["headers"] == {
                "Authorization": "Bearer user-token",
                "Accept": "application/json",
                "Content-Type": "application/x-www-form-urlencoded",
            }
            assert kwargs["data"] == {
                "image_url": "https://example.com/image.png",
                "caption": "Launch visual",
            }
            return MockResponse({"id": "container-1"})
        assert url == "https://graph.facebook.com/v25.0/ig-1/media_publish"
        assert method == "POST"
        assert kwargs["data"] == {"creation_id": "container-1"}
        return MockResponse({"id": "media-1"})

    monkeypatch.setattr(instagram.requests, "request", Mock(side_effect=request))

    result = _payload(
        instagram.instagram_publish_image(
            "ig-1", "https://example.com/image.png", caption="Launch visual"
        )
    )

    assert result == {
        "status": "success",
        "container_id": "container-1",
        "media_id": "media-1",
    }


def test_publish_image_rejects_non_public_url(monkeypatch):
    monkeypatch.setenv("META_ACCESS_TOKEN", "user-token")
    mock_request = Mock()
    monkeypatch.setattr(instagram.requests, "request", mock_request)

    result = _payload(instagram.instagram_publish_image("ig-1", "/tmp/image.png"))

    assert result == {
        "status": "error",
        "message": "image_url must be a public http or https URL",
    }
    mock_request.assert_not_called()


def test_graph_api_errors_are_structured_and_redact_tokens(monkeypatch):
    monkeypatch.setenv("META_ACCESS_TOKEN", "user-token")
    monkeypatch.setattr(
        instagram.requests,
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

    result = _payload(instagram.instagram_auth_status())

    assert result["status"] == "error"
    assert "user-token" not in json.dumps(result)
    assert "[redacted]" in json.dumps(result)
    assert result["details"]["error"]["code"] == 190


def test_graph_api_error_message_truncates_large_response_text(monkeypatch):
    monkeypatch.setenv("META_ACCESS_TOKEN", "user-token")
    long_response_text = "x" * 1200
    monkeypatch.setattr(
        instagram.requests,
        "request",
        Mock(return_value=MockResponse(text=long_response_text, status_code=502)),
    )

    result = _payload(instagram.instagram_auth_status())

    assert result["status"] == "error"
    assert "... [truncated]" in result["message"]
    assert "x" * 1001 not in result["message"]
