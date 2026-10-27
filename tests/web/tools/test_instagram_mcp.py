import json
from unittest.mock import Mock

import pytest
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
        if url == "https://graph.facebook.com/v25.0/container-1":
            assert method == "GET"
            assert kwargs["params"] == {"fields": "status_code,status"}
            return MockResponse({"status_code": "FINISHED"})
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


def test_publish_image_waits_for_container_processing(monkeypatch):
    monkeypatch.setenv("META_ACCESS_TOKEN", "user-token")
    statuses = iter(["IN_PROGRESS", "IN_PROGRESS", "FINISHED"])
    sleep = Mock()
    monkeypatch.setattr(instagram.time, "sleep", sleep)

    def request(method, url, **kwargs):
        if url.endswith("/ig-1/media"):
            return MockResponse({"id": "container-1"})
        if url.endswith("/container-1"):
            return MockResponse({"status_code": next(statuses)})
        return MockResponse({"id": "media-1"})

    monkeypatch.setattr(instagram.requests, "request", Mock(side_effect=request))

    result = _payload(
        instagram.instagram_publish_image("ig-1", "https://example.com/image.jpg")
    )

    assert result["status"] == "success"
    assert sleep.call_count == 2


def test_publish_image_stops_when_container_processing_fails(monkeypatch):
    monkeypatch.setenv("META_ACCESS_TOKEN", "user-token")
    requests_seen = []

    def request(method, url, **kwargs):
        requests_seen.append(url)
        if url.endswith("/ig-1/media"):
            return MockResponse({"id": "container-1"})
        return MockResponse(
            {
                "status_code": "ERROR",
                "status": "Media width is unsupported",
            }
        )

    monkeypatch.setattr(instagram.requests, "request", Mock(side_effect=request))

    result = _payload(
        instagram.instagram_publish_image("ig-1", "https://example.com/image.jpg")
    )

    assert result == {
        "status": "error",
        "message": (
            "Instagram media processing failed for container container-1: "
            "Media width is unsupported"
        ),
    }
    assert not any(url.endswith("/media_publish") for url in requests_seen)


def test_publish_image_timeout_includes_reusable_container_id(monkeypatch):
    monkeypatch.setenv("META_ACCESS_TOKEN", "user-token")
    monkeypatch.setattr(instagram, "MEDIA_CONTAINER_MAX_POLLS", 2)
    sleep = Mock()
    monkeypatch.setattr(instagram.time, "sleep", sleep)

    def request(method, url, **kwargs):
        if url.endswith("/ig-1/media"):
            return MockResponse({"id": "container-timeout"})
        return MockResponse({"status_code": "IN_PROGRESS"})

    monkeypatch.setattr(instagram.requests, "request", Mock(side_effect=request))

    result = _payload(
        instagram.instagram_publish_image("ig-1", "https://example.com/image.jpg")
    )

    assert result == {
        "status": "error",
        "message": (
            "Instagram media processing timed out for container container-timeout"
        ),
    }
    sleep.assert_called_once_with(instagram.MEDIA_CONTAINER_POLL_INTERVAL_SECONDS)


def test_media_container_polling_honors_wall_clock_deadline(monkeypatch):
    graph_request = Mock(return_value={"status_code": "IN_PROGRESS"})
    sleep = Mock()
    monotonic = Mock(side_effect=[0.0, 0.0, 60.0])
    monkeypatch.setattr(instagram, "_graph_request", graph_request)
    monkeypatch.setattr(instagram.time, "sleep", sleep)
    monkeypatch.setattr(instagram.time, "monotonic", monotonic)

    with pytest.raises(TimeoutError, match="container-deadline"):
        instagram._wait_for_media_container("container-deadline")

    graph_request.assert_called_once_with(
        "GET",
        "/container-deadline",
        params={"fields": "status_code,status"},
        timeout=instagram.meta_graph.DEFAULT_TIMEOUT_SECONDS,
    )
    sleep.assert_not_called()


def test_publish_image_redacts_graph_error_while_polling(monkeypatch, caplog):
    monkeypatch.setenv("META_ACCESS_TOKEN", "user-token")
    signed_url = "https://api.example.test/media?ticket=temporary-secret"
    escaped_url = signed_url.replace("/", r"\/")
    requests_seen = []

    def request(method, url, **kwargs):
        requests_seen.append(url)
        if url.endswith("/ig-1/media"):
            return MockResponse({"id": "container-1"})
        body = {"error": {"message": f"Could not fetch {signed_url}", "code": 9004}}
        return MockResponse(
            body,
            text=f'{{"error":{{"message":"Could not fetch {escaped_url}"}}}}',
            status_code=400,
        )

    monkeypatch.setattr(instagram.requests, "request", Mock(side_effect=request))

    result = _payload(instagram.instagram_publish_image("ig-1", signed_url))

    serialized = json.dumps(result)
    assert result["status"] == "error"
    assert signed_url not in serialized
    assert escaped_url not in serialized
    assert signed_url not in caplog.text
    assert escaped_url not in caplog.text
    assert result["details"]["container_id"] == "container-1"
    assert "[redacted]" in caplog.text
    assert not any(url.endswith("/media_publish") for url in requests_seen)


def test_publish_image_requires_exactly_one_source(monkeypatch):
    mock_request = Mock()
    monkeypatch.setattr(instagram.requests, "request", mock_request)

    missing = _payload(instagram.instagram_publish_image("ig-1"))
    duplicate = _payload(
        instagram.instagram_publish_image(
            "ig-1",
            "https://example.com/image.jpg",
            file_path="file:31218a9f-1497-4216-9f75-1fc8d51368c4",
        )
    )

    expected = {
        "status": "error",
        "message": "Provide exactly one of image_url or file_path",
    }
    assert missing == expected
    assert duplicate == expected
    mock_request.assert_not_called()


def test_publish_image_file_ref_requires_host_resolution(monkeypatch):
    mock_request = Mock()
    monkeypatch.setattr(instagram.requests, "request", mock_request)

    result = _payload(
        instagram.instagram_publish_image(
            "ig-1", file_path="file:31218a9f-1497-4216-9f75-1fc8d51368c4"
        )
    )

    assert result["status"] == "error"
    assert "resolved by the Xagent host" in result["message"]
    mock_request.assert_not_called()


def test_publish_error_redacts_temporary_image_url(monkeypatch, caplog):
    monkeypatch.setenv("META_ACCESS_TOKEN", "user-token")
    signed_url = "https://api.example.test/media?ticket=temporary-secret"
    escaped_url = signed_url.replace("/", r"\/")
    body = {"error": {"message": f"Could not fetch {signed_url}", "code": 9004}}
    monkeypatch.setattr(
        instagram.requests,
        "request",
        Mock(
            return_value=MockResponse(
                body,
                text=f'{{"error":{{"message":"Could not fetch {escaped_url}"}}}}',
                status_code=400,
            )
        ),
    )

    result = _payload(instagram.instagram_publish_image("ig-1", signed_url))

    assert result["status"] == "error"
    assert signed_url not in json.dumps(result)
    assert escaped_url not in json.dumps(result)
    assert signed_url not in caplog.text
    assert escaped_url not in caplog.text
    assert "[redacted]" in json.dumps(result)


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
