import json
from unittest.mock import Mock

import pytest
import requests

from xagent.web.tools.mcp import slack


class MockResponse:
    def __init__(self, json_data=None, status_code=200, headers=None):
        self._json_data = json_data if json_data is not None else {}
        self.status_code = status_code
        self.headers = headers or {}

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


@pytest.mark.parametrize(
    ("raw", "normalized"),
    [
        ("C0123456789", "C0123456789"),  # channel id, untouched
        ("#incidents", "#incidents"),  # already hash-prefixed
        ("incidents", "#incidents"),  # bare name gets the hash
        (" incidents ", "#incidents"),  # whitespace trimmed first
    ],
)
def test_normalize_channel(raw, normalized):
    assert slack._normalize_channel(raw) == normalized


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


def test_request_retries_once_on_rate_limit_with_small_retry_after(monkeypatch):
    mock_request = Mock(
        side_effect=[
            MockResponse({}, status_code=429, headers={"Retry-After": "2"}),
            MockResponse({"ok": True, "channels": []}),
        ]
    )
    sleep = Mock()
    monkeypatch.setattr(slack.requests, "request", mock_request)
    monkeypatch.setattr(slack.time, "sleep", sleep)

    result = slack._request("GET", "conversations.list")

    assert result["ok"] is True
    assert mock_request.call_count == 2
    sleep.assert_called_once_with(2)


def test_request_does_not_retry_on_large_retry_after(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse({}, status_code=429, headers={"Retry-After": "120"})
    )
    sleep = Mock()
    monkeypatch.setattr(slack.requests, "request", mock_request)
    monkeypatch.setattr(slack.time, "sleep", sleep)

    with pytest.raises(requests.HTTPError):
        slack._request("GET", "conversations.list")

    assert mock_request.call_count == 1
    sleep.assert_not_called()


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
    assert result["truncated"] is False


@pytest.mark.parametrize(("value", "serialized"), [(True, "true"), (False, "false")])
def test_list_channels_serializes_exclude_archived_lowercase(
    monkeypatch, value, serialized
):
    mock_request = Mock(return_value=MockResponse({"ok": True, "channels": []}))
    monkeypatch.setattr(slack.requests, "request", mock_request)

    slack.slack_list_channels(exclude_archived=value)

    params = mock_request.call_args.kwargs["params"]
    assert params["exclude_archived"] == serialized
    assert params["types"] == "public_channel"
    assert "cursor" not in params


def test_list_channels_filters_by_name_contains(monkeypatch):
    monkeypatch.setattr(
        slack.requests,
        "request",
        Mock(
            return_value=MockResponse(
                {
                    "ok": True,
                    "channels": [
                        {"id": "C1", "name": "general", "is_archived": False},
                        {"id": "C2", "name": "prod-incidents", "is_archived": False},
                    ],
                }
            )
        ),
    )

    result = json.loads(slack.slack_list_channels(name_contains="INCIDENT"))

    assert result["status"] == "success"
    assert [c["id"] for c in result["channels"]] == ["C2"]


def test_list_channels_stops_at_limit_and_flags_truncation(monkeypatch):
    monkeypatch.setattr(
        slack.requests,
        "request",
        Mock(
            return_value=MockResponse(
                {
                    "ok": True,
                    "channels": [
                        {"id": f"C{i}", "name": f"chan-{i}", "is_archived": False}
                        for i in range(5)
                    ],
                    "response_metadata": {"next_cursor": "more"},
                }
            )
        ),
    )

    result = json.loads(slack.slack_list_channels(limit=3))

    assert result["status"] == "success"
    assert len(result["channels"]) == 3
    assert result["truncated"] is True


def test_list_channels_limit_matching_last_item_of_last_page_is_not_truncated(
    monkeypatch,
):
    """Hitting the limit exactly on the last channel of the last page (no
    more raw results in this page, no next_cursor) means there is nothing
    left — truncated must be False, not a false positive from the limit
    check alone."""
    monkeypatch.setattr(
        slack.requests,
        "request",
        Mock(
            return_value=MockResponse(
                {
                    "ok": True,
                    "channels": [
                        {"id": f"C{i}", "name": f"chan-{i}", "is_archived": False}
                        for i in range(3)
                    ],
                    "response_metadata": {"next_cursor": ""},
                }
            )
        ),
    )

    result = json.loads(slack.slack_list_channels(limit=3))

    assert result["status"] == "success"
    assert len(result["channels"]) == 3
    assert result["truncated"] is False


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
    assert result["truncated"] is False
    assert mock_request.call_count == 2
    second_call_params = mock_request.call_args_list[1].kwargs["params"]
    assert second_call_params["cursor"] == "page-2"


def test_list_channels_returns_partial_results_when_a_later_page_fails(monkeypatch):
    """conversations.list is Tier-2 rate-limited; a failure on page N must not
    discard the N-1 pages already fetched."""
    mock_request = Mock(
        side_effect=[
            MockResponse(
                {
                    "ok": True,
                    "channels": [{"id": "C1", "name": "general", "is_archived": False}],
                    "response_metadata": {"next_cursor": "page-2"},
                }
            ),
            MockResponse({"ok": False, "error": "ratelimited"}),
        ]
    )
    monkeypatch.setattr(slack.requests, "request", mock_request)

    result = json.loads(slack.slack_list_channels())

    assert result["status"] == "success"
    assert [c["id"] for c in result["channels"]] == ["C1"]
    assert result["truncated"] is True
    assert "ratelimited" in result["error"]


def test_list_channels_returns_partial_results_when_filter_matched_nothing_yet(
    monkeypatch,
):
    """A name_contains filter that matched nothing on page 1 must not be
    confused with "no page fetched yet": a page-2 failure should still
    return the (empty) partial result rather than raising."""
    mock_request = Mock(
        side_effect=[
            MockResponse(
                {
                    "ok": True,
                    "channels": [{"id": "C1", "name": "general", "is_archived": False}],
                    "response_metadata": {"next_cursor": "page-2"},
                }
            ),
            MockResponse({"ok": False, "error": "ratelimited"}),
        ]
    )
    monkeypatch.setattr(slack.requests, "request", mock_request)

    result = json.loads(
        slack.slack_list_channels(name_contains="no-such-channel-matches")
    )

    assert result["status"] == "success"
    assert result["channels"] == []
    assert result["truncated"] is True
    assert "ratelimited" in result["error"]


def test_list_channels_returns_error_payload_when_first_page_fails(monkeypatch):
    monkeypatch.setattr(
        slack.requests,
        "request",
        Mock(return_value=MockResponse({"ok": False, "error": "missing_scope"})),
    )

    result = json.loads(slack.slack_list_channels())

    assert result["status"] == "error"
    assert "missing_scope" in result["message"]


def test_list_channels_tolerates_missing_channels_key(monkeypatch):
    monkeypatch.setattr(
        slack.requests,
        "request",
        Mock(return_value=MockResponse({"ok": True})),
    )

    result = json.loads(slack.slack_list_channels())

    assert result["status"] == "success"
    assert result["channels"] == []


def test_list_channels_reports_missing_token_through_the_tool(monkeypatch):
    monkeypatch.delenv("SLACK_ACCESS_TOKEN")

    result = json.loads(slack.slack_list_channels())

    assert result["status"] == "error"
    assert "SLACK_ACCESS_TOKEN" in result["message"]


def test_post_message_returns_channel_and_ts(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(
            {"ok": True, "channel": "C0123", "ts": "1753900000.000100"}
        )
    )
    monkeypatch.setattr(slack.requests, "request", mock_request)

    result = json.loads(slack.slack_post_message("C0123456789", "Incident detected"))

    assert result["status"] == "success"
    assert result["channel"] == "C0123"
    assert result["ts"] == "1753900000.000100"
    call_kwargs = mock_request.call_args.kwargs
    assert call_kwargs["url"].endswith("/chat.postMessage")
    assert call_kwargs["json"] == {
        "channel": "C0123456789",
        "text": "Incident detected",
    }


@pytest.mark.parametrize(
    ("raw", "sent"), [("#incidents", "#incidents"), ("incidents", "#incidents")]
)
def test_post_message_normalizes_channel_names(monkeypatch, raw, sent):
    mock_request = Mock(
        return_value=MockResponse({"ok": True, "channel": "C0123", "ts": "1.1"})
    )
    monkeypatch.setattr(slack.requests, "request", mock_request)

    json.loads(slack.slack_post_message(raw, "hello"))

    assert mock_request.call_args.kwargs["json"]["channel"] == sent


@pytest.mark.parametrize("error_code", ["channel_not_found", "not_in_channel"])
def test_post_message_returns_error_payload_on_slack_error(monkeypatch, error_code):
    monkeypatch.setattr(
        slack.requests,
        "request",
        Mock(return_value=MockResponse({"ok": False, "error": error_code})),
    )

    result = json.loads(slack.slack_post_message("C0123456789", "hi"))

    assert result["status"] == "error"
    assert error_code in result["message"]


def test_post_message_reports_http_error_through_the_tool(monkeypatch):
    monkeypatch.setattr(
        slack.requests,
        "request",
        Mock(return_value=MockResponse({}, status_code=500)),
    )

    result = json.loads(slack.slack_post_message("C0123456789", "hi"))

    assert result["status"] == "error"
    assert "500" in result["message"]


def test_post_message_reports_missing_token_through_the_tool(monkeypatch):
    monkeypatch.delenv("SLACK_ACCESS_TOKEN")

    result = json.loads(slack.slack_post_message("C0123456789", "hi"))

    assert result["status"] == "error"
    assert "SLACK_ACCESS_TOKEN" in result["message"]


def test_post_message_includes_thread_ts_when_provided(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse({"ok": True, "channel": "C0123", "ts": "1.2"})
    )
    monkeypatch.setattr(slack.requests, "request", mock_request)

    slack.slack_post_message("C0123456789", "reply", thread_ts="1700000000.000100")

    assert mock_request.call_args.kwargs["json"] == {
        "channel": "C0123456789",
        "text": "reply",
        "thread_ts": "1700000000.000100",
    }


def test_post_message_omits_thread_ts_when_absent(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse({"ok": True, "channel": "C0123", "ts": "1.2"})
    )
    monkeypatch.setattr(slack.requests, "request", mock_request)

    slack.slack_post_message("C0123456789", "hello")

    assert "thread_ts" not in mock_request.call_args.kwargs["json"]


# ---------------------------------------------------------------------------
# _resolve_channel_id
# ---------------------------------------------------------------------------


def test_resolve_channel_id_passes_through_ids():
    assert slack._resolve_channel_id("C0123456789") == "C0123456789"


def test_resolve_channel_id_looks_up_name(monkeypatch):
    monkeypatch.setattr(
        slack.requests,
        "request",
        Mock(
            return_value=MockResponse(
                {
                    "ok": True,
                    "channels": [
                        {"id": "C1", "name": "general"},
                        {"id": "C2", "name": "incidents"},
                    ],
                }
            )
        ),
    )

    assert slack._resolve_channel_id("#incidents") == "C2"
    assert slack._resolve_channel_id("incidents") == "C2"


def test_resolve_channel_id_raises_when_not_found(monkeypatch):
    monkeypatch.setattr(
        slack.requests,
        "request",
        Mock(return_value=MockResponse({"ok": True, "channels": []})),
    )

    with pytest.raises(ValueError, match="Could not resolve channel"):
        slack._resolve_channel_id("no-such-channel")


def test_resolve_channel_id_rejects_empty_name():
    with pytest.raises(ValueError, match="channel must not be empty"):
        slack._resolve_channel_id("#")


# ---------------------------------------------------------------------------
# slack_get_channel_history
# ---------------------------------------------------------------------------


def test_get_channel_history_returns_messages_and_pagination(monkeypatch):
    monkeypatch.setattr(
        slack.requests,
        "request",
        Mock(
            return_value=MockResponse(
                {
                    "ok": True,
                    "messages": [
                        {
                            "ts": "1.1",
                            "user": "U1",
                            "text": "hello",
                            "thread_ts": "1.1",
                            "reply_count": 2,
                        }
                    ],
                    "has_more": True,
                    "response_metadata": {"next_cursor": "page-2"},
                }
            )
        ),
    )

    result = json.loads(slack.slack_get_channel_history("C0123456789"))

    assert result["status"] == "success"
    assert result["messages"] == [
        {
            "ts": "1.1",
            "user": "U1",
            "text": "hello",
            "thread_ts": "1.1",
            "reply_count": 2,
        }
    ]
    assert result["has_more"] is True
    assert result["next_cursor"] == "page-2"


def test_get_channel_history_passes_time_bounds_and_cursor(monkeypatch):
    mock_request = Mock(return_value=MockResponse({"ok": True, "messages": []}))
    monkeypatch.setattr(slack.requests, "request", mock_request)

    slack.slack_get_channel_history(
        "C0123456789", limit=10, oldest="1.0", latest="2.0", cursor="abc"
    )

    params = mock_request.call_args.kwargs["params"]
    assert params == {
        "channel": "C0123456789",
        "limit": 10,
        "oldest": "1.0",
        "latest": "2.0",
        "cursor": "abc",
    }


def test_get_channel_history_reports_error_payload(monkeypatch):
    monkeypatch.setattr(
        slack.requests,
        "request",
        Mock(return_value=MockResponse({"ok": False, "error": "missing_scope"})),
    )

    result = json.loads(slack.slack_get_channel_history("C0123456789"))

    assert result["status"] == "error"
    assert "missing_scope" in result["message"]


# ---------------------------------------------------------------------------
# slack_get_thread_replies
# ---------------------------------------------------------------------------


def test_get_thread_replies_returns_messages(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(
            {
                "ok": True,
                "messages": [
                    {"ts": "1.1", "user": "U1", "text": "parent"},
                    {"ts": "1.2", "user": "U2", "text": "reply"},
                ],
                "has_more": False,
            }
        )
    )
    monkeypatch.setattr(slack.requests, "request", mock_request)

    result = json.loads(slack.slack_get_thread_replies("C0123456789", "1.1"))

    assert result["status"] == "success"
    assert result["messages"] == [
        {"ts": "1.1", "user": "U1", "text": "parent"},
        {"ts": "1.2", "user": "U2", "text": "reply"},
    ]
    call_params = mock_request.call_args.kwargs["params"]
    assert call_params["channel"] == "C0123456789"
    assert call_params["ts"] == "1.1"


def test_get_thread_replies_reports_error_payload(monkeypatch):
    monkeypatch.setattr(
        slack.requests,
        "request",
        Mock(return_value=MockResponse({"ok": False, "error": "thread_not_found"})),
    )

    result = json.loads(slack.slack_get_thread_replies("C0123456789", "1.1"))

    assert result["status"] == "error"
    assert "thread_not_found" in result["message"]


# ---------------------------------------------------------------------------
# slack_get_channel_info
# ---------------------------------------------------------------------------


def test_get_channel_info_returns_topic_and_metadata(monkeypatch):
    monkeypatch.setattr(
        slack.requests,
        "request",
        Mock(
            return_value=MockResponse(
                {
                    "ok": True,
                    "channel": {
                        "id": "C0123456789",
                        "name": "incidents",
                        "topic": {"value": "Prod incidents"},
                        "purpose": {"value": "Track prod incidents"},
                        "is_archived": False,
                        "is_private": False,
                        "is_im": False,
                        "is_mpim": False,
                        "num_members": 12,
                    },
                }
            )
        ),
    )

    result = json.loads(slack.slack_get_channel_info("C0123456789"))

    assert result == {
        "status": "success",
        "id": "C0123456789",
        "name": "incidents",
        "topic": "Prod incidents",
        "purpose": "Track prod incidents",
        "is_archived": False,
        "is_private": False,
        "is_im": False,
        "is_mpim": False,
        "num_members": 12,
    }


def test_get_channel_info_reports_error_payload(monkeypatch):
    monkeypatch.setattr(
        slack.requests,
        "request",
        Mock(return_value=MockResponse({"ok": False, "error": "channel_not_found"})),
    )

    result = json.loads(slack.slack_get_channel_info("C0123456789"))

    assert result["status"] == "error"
    assert "channel_not_found" in result["message"]


# ---------------------------------------------------------------------------
# slack_list_direct_messages
# ---------------------------------------------------------------------------


def test_list_direct_messages_separates_dms_and_group_dms(monkeypatch):
    monkeypatch.setattr(
        slack.requests,
        "request",
        Mock(
            return_value=MockResponse(
                {
                    "ok": True,
                    "channels": [
                        {"id": "D1", "user": "U1", "is_mpim": False},
                        {"id": "G1", "name": "mpdm-a--b-1", "is_mpim": True},
                    ],
                }
            )
        ),
    )

    result = json.loads(slack.slack_list_direct_messages())

    assert result["status"] == "success"
    assert result["conversations"] == [
        {"id": "D1", "is_group_dm": False, "user": "U1"},
        {"id": "G1", "is_group_dm": True, "name": "mpdm-a--b-1"},
    ]
    assert result["truncated"] is False


def test_list_direct_messages_stops_at_limit_and_flags_truncation(monkeypatch):
    monkeypatch.setattr(
        slack.requests,
        "request",
        Mock(
            return_value=MockResponse(
                {
                    "ok": True,
                    "channels": [
                        {"id": f"D{i}", "user": f"U{i}", "is_mpim": False}
                        for i in range(5)
                    ],
                    "response_metadata": {"next_cursor": "more"},
                }
            )
        ),
    )

    result = json.loads(slack.slack_list_direct_messages(limit=3))

    assert result["status"] == "success"
    assert len(result["conversations"]) == 3
    assert result["truncated"] is True


def test_list_direct_messages_reports_error_payload(monkeypatch):
    monkeypatch.setattr(
        slack.requests,
        "request",
        Mock(return_value=MockResponse({"ok": False, "error": "missing_scope"})),
    )

    result = json.loads(slack.slack_list_direct_messages())

    assert result["status"] == "error"
    assert "missing_scope" in result["message"]


# ---------------------------------------------------------------------------
# slack_search_messages
# ---------------------------------------------------------------------------


def test_search_messages_matches_channel_history(monkeypatch):
    monkeypatch.setattr(
        slack.requests,
        "request",
        Mock(
            return_value=MockResponse(
                {
                    "ok": True,
                    "messages": [
                        {"ts": "1.1", "user": "U1", "text": "the deploy failed"},
                        {"ts": "1.2", "user": "U2", "text": "unrelated"},
                    ],
                }
            )
        ),
    )

    result = json.loads(
        slack.slack_search_messages(
            "C0123456789", "deploy", include_thread_replies=False
        )
    )

    assert result["status"] == "success"
    assert result["matches"] == [
        {"ts": "1.1", "user": "U1", "text": "the deploy failed"}
    ]
    assert result["truncated"] is False


def test_search_messages_searches_thread_replies(monkeypatch):
    mock_request = Mock(
        side_effect=[
            MockResponse(
                {
                    "ok": True,
                    "messages": [
                        {
                            "ts": "1.1",
                            "user": "U1",
                            "text": "kickoff",
                            "thread_ts": "1.1",
                            "reply_count": 1,
                        }
                    ],
                }
            ),
            MockResponse(
                {
                    "ok": True,
                    "messages": [
                        {"ts": "1.1", "user": "U1", "text": "kickoff"},
                        {"ts": "1.2", "user": "U2", "text": "the deploy failed"},
                    ],
                }
            ),
        ]
    )
    monkeypatch.setattr(slack.requests, "request", mock_request)

    result = json.loads(slack.slack_search_messages("C0123456789", "deploy"))

    assert result["status"] == "success"
    assert result["matches"] == [
        {"ts": "1.2", "user": "U2", "text": "the deploy failed", "thread_ts": "1.1"}
    ]
    assert mock_request.call_count == 2


def test_search_messages_rejects_empty_query():
    result = json.loads(slack.slack_search_messages("C0123456789", "   "))

    assert result["status"] == "error"
    assert "query" in result["message"]


def test_search_messages_reports_error_payload(monkeypatch):
    monkeypatch.setattr(
        slack.requests,
        "request",
        Mock(return_value=MockResponse({"ok": False, "error": "channel_not_found"})),
    )

    result = json.loads(slack.slack_search_messages("C0123456789", "deploy"))

    assert result["status"] == "error"
    assert "channel_not_found" in result["message"]


# ---------------------------------------------------------------------------
# slack_add_reaction / slack_remove_reaction
# ---------------------------------------------------------------------------


def test_add_reaction_sends_expected_payload(monkeypatch):
    mock_request = Mock(return_value=MockResponse({"ok": True}))
    monkeypatch.setattr(slack.requests, "request", mock_request)

    result = json.loads(
        slack.slack_add_reaction("C0123456789", "1700000000.000100", ":thumbsup:")
    )

    assert result["status"] == "success"
    assert result["emoji_name"] == "thumbsup"
    call_kwargs = mock_request.call_args.kwargs
    assert call_kwargs["url"].endswith("/reactions.add")
    assert call_kwargs["json"] == {
        "channel": "C0123456789",
        "timestamp": "1700000000.000100",
        "name": "thumbsup",
    }


def test_remove_reaction_sends_expected_payload(monkeypatch):
    mock_request = Mock(return_value=MockResponse({"ok": True}))
    monkeypatch.setattr(slack.requests, "request", mock_request)

    result = json.loads(
        slack.slack_remove_reaction("C0123456789", "1700000000.000100", "thumbsup")
    )

    assert result["status"] == "success"
    call_kwargs = mock_request.call_args.kwargs
    assert call_kwargs["url"].endswith("/reactions.remove")
    assert call_kwargs["json"] == {
        "channel": "C0123456789",
        "timestamp": "1700000000.000100",
        "name": "thumbsup",
    }


def test_add_reaction_reports_error_payload(monkeypatch):
    monkeypatch.setattr(
        slack.requests,
        "request",
        Mock(return_value=MockResponse({"ok": False, "error": "already_reacted"})),
    )

    result = json.loads(slack.slack_add_reaction("C0123456789", "1.1", "thumbsup"))

    assert result["status"] == "error"
    assert "already_reacted" in result["message"]


# ---------------------------------------------------------------------------
# slack_upload_file
# ---------------------------------------------------------------------------


def test_upload_file_rejects_path_outside_allowed_dirs(tmp_path, monkeypatch):
    outside_file = tmp_path / "outside.txt"
    outside_file.write_text("secret")
    monkeypatch.setenv("XAGENT_SLACK_FILE_ALLOWED_DIRS", str(tmp_path / "workspace"))

    result = json.loads(slack.slack_upload_file("C0123456789", str(outside_file)))

    assert result["status"] == "error"
    assert "outside allowed directories" in result["message"]


def test_upload_file_rejects_missing_file(tmp_path, monkeypatch):
    monkeypatch.setenv("XAGENT_SLACK_FILE_ALLOWED_DIRS", str(tmp_path))

    result = json.loads(
        slack.slack_upload_file("C0123456789", str(tmp_path / "missing.txt"))
    )

    assert result["status"] == "error"
    assert "File not found" in result["message"]


def test_upload_file_completes_upload_flow(tmp_path, monkeypatch):
    local_file = tmp_path / "report.txt"
    local_file.write_text("incident report")
    monkeypatch.setenv("XAGENT_SLACK_FILE_ALLOWED_DIRS", str(tmp_path))

    mock_request = Mock(
        side_effect=[
            MockResponse(
                {
                    "ok": True,
                    "upload_url": "https://files.slack.com/upload/v1/abc",
                    "file_id": "F123",
                }
            ),
            MockResponse({"ok": True, "files": [{"id": "F123"}]}),
        ]
    )
    monkeypatch.setattr(slack.requests, "request", mock_request)
    mock_post = Mock(return_value=MockResponse({}, status_code=200))
    monkeypatch.setattr(slack.requests, "post", mock_post)

    result = json.loads(
        slack.slack_upload_file("C0123456789", str(local_file), title="Incident report")
    )

    assert result["status"] == "success"
    assert result["file_id"] == "F123"
    assert result["filename"] == "report.txt"

    init_call = mock_request.call_args_list[0]
    assert init_call.kwargs["url"].endswith("/files.getUploadURLExternal")
    assert init_call.kwargs["params"]["filename"] == "report.txt"

    assert mock_post.call_args.args[0] == "https://files.slack.com/upload/v1/abc"

    complete_call = mock_request.call_args_list[1]
    assert complete_call.kwargs["url"].endswith("/files.completeUploadExternal")
    assert complete_call.kwargs["json"] == {
        "files": [{"id": "F123", "title": "Incident report"}],
        "channel_id": "C0123456789",
    }


def test_upload_file_reports_error_when_upload_url_missing(tmp_path, monkeypatch):
    local_file = tmp_path / "report.txt"
    local_file.write_text("incident report")
    monkeypatch.setenv("XAGENT_SLACK_FILE_ALLOWED_DIRS", str(tmp_path))

    monkeypatch.setattr(
        slack.requests,
        "request",
        Mock(return_value=MockResponse({"ok": True})),
    )

    result = json.loads(slack.slack_upload_file("C0123456789", str(local_file)))

    assert result["status"] == "error"
    assert "upload URL" in result["message"]


def test_search_messages_opted_out_thread_replies_is_not_truncation(monkeypatch):
    """Threaded parents left unscanned because the caller passed
    include_thread_replies=False are an explicit opt-out, not missing
    coverage — truncated must stay False."""
    monkeypatch.setattr(
        slack.requests,
        "request",
        Mock(
            return_value=MockResponse(
                {
                    "ok": True,
                    "messages": [
                        {
                            "ts": "1.1",
                            "user": "U1",
                            "text": "the deploy failed",
                            "thread_ts": "1.1",
                            "reply_count": 3,
                        }
                    ],
                }
            )
        ),
    )

    result = json.loads(
        slack.slack_search_messages(
            "C0123456789", "deploy", include_thread_replies=False
        )
    )

    assert result["status"] == "success"
    assert len(result["matches"]) == 1
    assert result["truncated"] is False


def test_search_messages_failed_thread_fetch_flags_truncation(monkeypatch):
    """A thread whose replies could not be fetched (e.g. a rate limit that
    outlived the single retry) is genuinely missing coverage — the partial
    result must be flagged truncated rather than silently complete."""
    mock_request = Mock(
        side_effect=[
            MockResponse(
                {
                    "ok": True,
                    "messages": [
                        {
                            "ts": "1.1",
                            "user": "U1",
                            "text": "kickoff",
                            "thread_ts": "1.1",
                            "reply_count": 1,
                        }
                    ],
                }
            ),
            MockResponse({"ok": False, "error": "ratelimited"}),
        ]
    )
    monkeypatch.setattr(slack.requests, "request", mock_request)

    result = json.loads(slack.slack_search_messages("C0123456789", "deploy"))

    assert result["status"] == "success"
    assert result["matches"] == []
    assert result["truncated"] is True
