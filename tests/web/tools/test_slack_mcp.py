import json
from unittest.mock import Mock

import pytest
import requests

from xagent.web.tools.mcp import slack


class MockResponse:
    def __init__(self, json_data=None, status_code=200):
        self._json_data = json_data if json_data is not None else {}
        self.status_code = status_code

    def json(self):
        return self._json_data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} Client Error", response=self)


@pytest.fixture(autouse=True)
def _credentials(monkeypatch):
    monkeypatch.setenv("SLACK_ACCESS_TOKEN", "xoxb-test-token")


def test_headers_require_access_token(monkeypatch):
    monkeypatch.delenv("SLACK_ACCESS_TOKEN")

    with pytest.raises(ValueError, match="SLACK_ACCESS_TOKEN"):
        slack._headers()


def test_headers_include_bearer_token():
    assert slack._headers() == {"Authorization": "Bearer xoxb-test-token"}


def test_request_raises_on_slack_ok_false(monkeypatch):
    monkeypatch.setattr(
        slack.requests,
        "request",
        Mock(return_value=MockResponse({"ok": False, "error": "invalid_auth"})),
    )

    with pytest.raises(RuntimeError, match="invalid_auth"):
        slack._request("GET", "auth.test")


def test_request_raises_generic_message_when_error_field_absent(monkeypatch):
    monkeypatch.setattr(
        slack.requests,
        "request",
        Mock(return_value=MockResponse({"ok": False})),
    )

    with pytest.raises(RuntimeError, match="Unknown Slack API error"):
        slack._request("GET", "auth.test")


def test_request_raises_on_http_error_status(monkeypatch):
    monkeypatch.setattr(
        slack.requests,
        "request",
        Mock(return_value=MockResponse({}, status_code=500)),
    )

    with pytest.raises(requests.HTTPError):
        slack._request("GET", "auth.test")


def test_list_channels_returns_flat_list(monkeypatch):
    monkeypatch.setattr(
        slack.requests,
        "request",
        Mock(
            return_value=MockResponse(
                {
                    "ok": True,
                    "channels": [
                        {"id": "C1", "name": "general", "is_archived": False},
                        {"id": "C2", "name": "incidents", "is_archived": False},
                    ],
                }
            )
        ),
    )

    result = json.loads(slack.slack_list_channels())

    assert result["status"] == "success"
    assert result["channels"] == [
        {"id": "C1", "name": "general", "is_archived": False},
        {"id": "C2", "name": "incidents", "is_archived": False},
    ]


def test_list_channels_passes_exclude_archived_and_types(monkeypatch):
    mock_request = Mock(return_value=MockResponse({"ok": True, "channels": []}))
    monkeypatch.setattr(slack.requests, "request", mock_request)

    slack.slack_list_channels(exclude_archived=False)

    params = mock_request.call_args.kwargs["params"]
    assert params["exclude_archived"] == "false"
    assert params["types"] == "public_channel"
    assert "cursor" not in params


def test_list_channels_follows_pagination(monkeypatch):
    mock_request = Mock(
        side_effect=[
            MockResponse(
                {
                    "ok": True,
                    "channels": [{"id": "C1", "name": "general", "is_archived": False}],
                    "response_metadata": {"next_cursor": "page-2"},
                }
            ),
            MockResponse(
                {
                    "ok": True,
                    "channels": [
                        {"id": "C2", "name": "incidents", "is_archived": False}
                    ],
                    "response_metadata": {"next_cursor": ""},
                }
            ),
        ]
    )
    monkeypatch.setattr(slack.requests, "request", mock_request)

    result = json.loads(slack.slack_list_channels())

    assert result["status"] == "success"
    assert [c["id"] for c in result["channels"]] == ["C1", "C2"]
    assert mock_request.call_count == 2
    second_call_params = mock_request.call_args_list[1].kwargs["params"]
    assert second_call_params["cursor"] == "page-2"


def test_list_channels_returns_error_payload_on_failure(monkeypatch):
    monkeypatch.setattr(
        slack.requests,
        "request",
        Mock(return_value=MockResponse({"ok": False, "error": "missing_scope"})),
    )

    result = json.loads(slack.slack_list_channels())

    assert result["status"] == "error"
    assert "missing_scope" in result["message"]


def test_post_message_returns_channel_and_ts(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(
            {"ok": True, "channel": "C0123", "ts": "1753900000.000100"}
        )
    )
    monkeypatch.setattr(slack.requests, "request", mock_request)

    result = json.loads(slack.slack_post_message("C0123", "Incident detected"))

    assert result["status"] == "success"
    assert result["channel"] == "C0123"
    assert result["ts"] == "1753900000.000100"
    call_kwargs = mock_request.call_args.kwargs
    assert call_kwargs["url"].endswith("/chat.postMessage")
    assert call_kwargs["json"] == {"channel": "C0123", "text": "Incident detected"}


def test_post_message_accepts_channel_name(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse({"ok": True, "channel": "C0123", "ts": "1.1"})
    )
    monkeypatch.setattr(slack.requests, "request", mock_request)

    json.loads(slack.slack_post_message("#incidents", "hello"))

    assert mock_request.call_args.kwargs["json"]["channel"] == "#incidents"


def test_post_message_returns_error_payload_on_channel_not_found(monkeypatch):
    monkeypatch.setattr(
        slack.requests,
        "request",
        Mock(return_value=MockResponse({"ok": False, "error": "channel_not_found"})),
    )

    result = json.loads(slack.slack_post_message("C_missing", "hi"))

    assert result["status"] == "error"
    assert "channel_not_found" in result["message"]


def test_post_message_returns_error_payload_on_not_in_channel(monkeypatch):
    monkeypatch.setattr(
        slack.requests,
        "request",
        Mock(return_value=MockResponse({"ok": False, "error": "not_in_channel"})),
    )

    result = json.loads(slack.slack_post_message("C0123", "hi"))

    assert result["status"] == "error"
    assert "not_in_channel" in result["message"]
