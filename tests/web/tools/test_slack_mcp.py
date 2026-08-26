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

    @property
    def ok(self):
        return self.status_code < 400

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
        ("G0123456789", "G0123456789"),  # private channel / mpim id, untouched
        ("D0123456789", "D0123456789"),  # DM id, untouched
        ("U0123456789", "U0123456789"),  # user id, untouched — DM by user id
        ("B0123456789", "B0123456789"),  # bot id, untouched
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

    with pytest.raises(slack._SlackAPIError, match="invalid_auth"):
        slack._request("GET", "auth.test")


def test_request_translates_missing_scope_for_every_endpoint(monkeypatch):
    """missing_scope means the same thing everywhere Slack can return it, so
    _request translates it once at the lowest level rather than leaving
    every individual tool to leak the bare code."""
    monkeypatch.setattr(
        slack.requests,
        "request",
        Mock(return_value=MockResponse({"ok": False, "error": "missing_scope"})),
    )

    with pytest.raises(slack._SlackMissingScopeError, match="reconnect") as exc_info:
        slack._request("GET", "auth.test")
    assert isinstance(exc_info.value, slack._SlackActionableError)


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
    """missing_scope is translated into an actionable reconnect message by
    _request itself, so this reaches slack_list_channels — a tool that
    isn't wrapped by _request_requiring_membership at all — the same way
    it reaches every other tool in this file."""
    monkeypatch.setattr(
        slack.requests,
        "request",
        Mock(return_value=MockResponse({"ok": False, "error": "missing_scope"})),
    )

    result = json.loads(slack.slack_list_channels())

    assert result["status"] == "error"
    assert "missing_scope" in result["message"]
    assert "reconnect" in result["message"]


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


def test_post_message_sends_a_bare_user_id_unchanged(monkeypatch):
    """Regression: a prior narrowing of the shared id pattern (for
    _resolve_channel_id's channel-only prefixes) must not also make
    _normalize_channel treat a user id as a bare name and prefix it with
    "#" — chat.postMessage accepts a user id directly to open/post into a
    1:1 DM, and "#U0123456789" would 404."""
    mock_request = Mock(
        return_value=MockResponse({"ok": True, "channel": "D0123", "ts": "1.1"})
    )
    monkeypatch.setattr(slack.requests, "request", mock_request)

    json.loads(slack.slack_post_message("U0123456789", "hello"))

    assert mock_request.call_args.kwargs["json"]["channel"] == "U0123456789"


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


def test_post_message_reports_actionable_error_when_not_a_member(monkeypatch):
    """A private channel/DM chat.postMessage can't fall back on
    chat:write.public — the caller must get the same slack_join_channel
    guidance as the read tools, not a bare not_in_channel code."""
    monkeypatch.setattr(
        slack.requests,
        "request",
        Mock(return_value=MockResponse({"ok": False, "error": "not_in_channel"})),
    )

    result = json.loads(slack.slack_post_message("G0123456789", "hi"))

    assert result["status"] == "error"
    assert "slack_join_channel" in result["message"]


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
# _request_requiring_membership
# ---------------------------------------------------------------------------


def test_request_requiring_membership_passes_through_on_success(monkeypatch):
    mock_request = Mock(return_value=MockResponse({"ok": True, "messages": []}))
    monkeypatch.setattr(slack.requests, "request", mock_request)

    result = slack._request_requiring_membership(
        "GET", "conversations.history", params={"channel": "C0123456789"}
    )

    assert result == {"ok": True, "messages": []}
    assert mock_request.call_count == 1


def test_request_requiring_membership_reraises_unrelated_errors(monkeypatch):
    """channel_not_found isn't in the default not_a_member_codes (only
    conversations.replies/reactions.* need the wider set) — this must raise
    the raw _SlackAPIError, not get rewritten into the actionable
    _SlackNotAMemberError, so a widening regression can't hide behind a
    substring match on the (still-present) error code."""
    monkeypatch.setattr(
        slack.requests,
        "request",
        Mock(return_value=MockResponse({"ok": False, "error": "channel_not_found"})),
    )

    with pytest.raises(slack._SlackAPIError, match="channel_not_found"):
        slack._request_requiring_membership(
            "GET", "conversations.history", params={"channel": "C0123456789"}
        )


def test_request_requiring_membership_raises_actionable_error_without_auto_joining(
    monkeypatch,
):
    """Joining changes the channel's visible member list, so this must never
    happen silently on the caller's behalf — a not_in_channel failure should
    surface an actionable message pointing at slack_join_channel (which the
    calling agent should only invoke once the user has agreed), not attempt
    conversations.join itself."""
    mock_request = Mock(
        return_value=MockResponse({"ok": False, "error": "not_in_channel"})
    )
    monkeypatch.setattr(slack.requests, "request", mock_request)

    with pytest.raises(RuntimeError, match="slack_join_channel"):
        slack._request_requiring_membership(
            "GET", "conversations.history", params={"channel": "C0123456789"}
        )

    # Exactly one call: the original read attempt, and nothing else — no
    # conversations.join call was made on the caller's behalf.
    assert mock_request.call_count == 1


def test_request_requiring_membership_treats_no_permission_as_actionable(monkeypatch):
    """no_permission is documented identically across every endpoint this
    wrapper covers as "make sure your app is a member of the conversation"
    (verified against Slack's own API reference for conversations.history,
    conversations.replies, chat.postMessage, reactions.add/remove, and
    files.completeUploadExternal) — it must be actionable everywhere, not
    just the endpoints that happen to also document not_in_channel."""
    monkeypatch.setattr(
        slack.requests,
        "request",
        Mock(return_value=MockResponse({"ok": False, "error": "no_permission"})),
    )

    with pytest.raises(RuntimeError, match="slack_join_channel"):
        slack._request_requiring_membership(
            "GET", "conversations.history", params={"channel": "C0123456789"}
        )


@pytest.mark.parametrize(
    ("caller", "kwargs"),
    [
        (lambda: slack.slack_get_channel_history("C0123456789"), {}),
        (lambda: slack.slack_get_thread_replies("C0123456789", "1.1"), {}),
        (lambda: slack.slack_post_message("C0123456789", "hi"), {}),
    ],
)
def test_no_permission_reports_actionable_error_across_read_and_write_tools(
    monkeypatch, caller, kwargs
):
    monkeypatch.setattr(
        slack.requests,
        "request",
        Mock(return_value=MockResponse({"ok": False, "error": "no_permission"})),
    )

    result = json.loads(caller())

    assert result["status"] == "error"
    assert "slack_join_channel" in result["message"]


def test_add_reaction_reports_actionable_error_for_no_permission(monkeypatch):
    monkeypatch.setattr(
        slack.requests,
        "request",
        Mock(return_value=MockResponse({"ok": False, "error": "no_permission"})),
    )

    result = json.loads(slack.slack_add_reaction("C0123456789", "1.1", "thumbsup"))

    assert result["status"] == "error"
    assert "slack_join_channel" in result["message"]


def test_upload_file_reports_actionable_error_for_no_permission(tmp_path, monkeypatch):
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
            MockResponse({"ok": False, "error": "no_permission"}),
        ]
    )
    monkeypatch.setattr(slack.requests, "request", mock_request)
    monkeypatch.setattr(
        slack.requests, "post", Mock(return_value=MockResponse({}, status_code=200))
    )

    result = json.loads(slack.slack_upload_file("C0123456789", str(local_file)))

    assert result["status"] == "error"
    assert "slack_join_channel" in result["message"]


def test_request_requiring_membership_hedges_channel_not_found_for_invalid_raw_id(
    monkeypatch,
):
    """channel_not_found for conversations.replies/reactions.* is genuinely
    ambiguous when the caller passed a raw channel id directly:
    _resolve_channel_id never validates a syntactically-valid id via
    conversations.list (only names go through that lookup), so this could
    be a real not-a-member case or simply a wrong/deleted id. The message
    must not assert the membership explanation as fact."""
    monkeypatch.setattr(
        slack.requests,
        "request",
        Mock(return_value=MockResponse({"ok": False, "error": "channel_not_found"})),
    )

    result = json.loads(slack.slack_add_reaction("C0000000009", "1.1", "thumbsup"))

    assert result["status"] == "error"
    assert "doesn't exist" in result["message"]
    assert "slack_join_channel" in result["message"]


def test_post_message_does_not_treat_dm_channel_not_found_as_actionable(monkeypatch):
    """Unlike conversations.replies/reactions.*, chat.postMessage's
    channel_not_found is NOT widened to actionable: Slack's own docs say a
    DM target returns exactly this code when the app lacks permission to
    open that DM — a case slack_join_channel (which only joins channels,
    never DMs) cannot fix, so treating it as "go call slack_join_channel"
    would be actively misleading."""
    monkeypatch.setattr(
        slack.requests,
        "request",
        Mock(return_value=MockResponse({"ok": False, "error": "channel_not_found"})),
    )

    result = json.loads(slack.slack_post_message("U0123456789", "hi"))

    assert result["status"] == "error"
    assert "slack_join_channel" not in result["message"]
    assert "channel_not_found" in result["message"]


# ---------------------------------------------------------------------------
# slack_join_channel
# ---------------------------------------------------------------------------


async def test_join_channel_is_discoverable_via_mcp_list_tools():
    """Every direct-call test below exercises the Python function, not the
    MCP registration — a regression in the @mcp.tool() decorator, the
    exposed tool name, or the generated input schema could leave those
    tests green while an agent can't discover or correctly invoke this
    tool at all."""
    tools = await slack.mcp.list_tools()
    tool = next(t for t in tools if t.name == "slack_join_channel")

    assert tool.inputSchema["required"] == ["channel"]
    assert tool.inputSchema["properties"]["channel"]["type"] == "string"
    # The docstring is what the calling agent actually reads to decide
    # whether it's safe to call this tool without asking first — confirm
    # the real one made it through registration, not an empty/stub one.
    assert "explicitly confirmed" in (tool.description or "")


def test_join_channel_sends_expected_payload(monkeypatch):
    mock_request = Mock(return_value=MockResponse({"ok": True}))
    monkeypatch.setattr(slack.requests, "request", mock_request)

    result = json.loads(slack.slack_join_channel("C0123456789"))

    assert result == {
        "status": "success",
        "channel": "C0123456789",
        "already_member": False,
    }
    call_kwargs = mock_request.call_args.kwargs
    assert call_kwargs["url"].endswith("/conversations.join")
    assert call_kwargs["json"] == {"channel": "C0123456789"}


def test_join_channel_resolves_bare_name(monkeypatch):
    mock_request = Mock(
        side_effect=[
            MockResponse(
                {
                    "ok": True,
                    "channels": [{"id": "C1", "name": "incidents"}],
                }
            ),
            MockResponse({"ok": True}),
        ]
    )
    monkeypatch.setattr(slack.requests, "request", mock_request)

    result = json.loads(slack.slack_join_channel("incidents"))

    assert result == {"status": "success", "channel": "C1", "already_member": False}
    join_call = mock_request.call_args_list[1]
    assert join_call.kwargs["json"] == {"channel": "C1"}


def test_join_channel_reports_already_member_via_response_metadata_warnings(
    monkeypatch,
):
    """conversations.join succeeds even when the bot already had membership
    (flagged via response_metadata.warnings) — the caller must be able to
    tell that apart from a fresh join. Sets only response_metadata.warnings
    (no top-level "warning") so this actually exercises that field alone —
    see the sibling test below for the top-level field."""
    monkeypatch.setattr(
        slack.requests,
        "request",
        Mock(
            return_value=MockResponse(
                {
                    "ok": True,
                    "response_metadata": {"warnings": ["already_in_channel"]},
                }
            )
        ),
    )

    result = json.loads(slack.slack_join_channel("C0123456789"))

    assert result == {
        "status": "success",
        "channel": "C0123456789",
        "already_member": True,
    }


def test_join_channel_reports_already_member_via_top_level_warning(monkeypatch):
    """Slack can also carry the already-in-channel signal in a top-level
    "warning" string rather than response_metadata.warnings — a real
    response that only sets this field must still be recognized, not just
    one that happens to set both."""
    monkeypatch.setattr(
        slack.requests,
        "request",
        Mock(
            return_value=MockResponse(
                {
                    "ok": True,
                    "warning": "already_in_channel",
                }
            )
        ),
    )

    result = json.loads(slack.slack_join_channel("C0123456789"))

    assert result == {
        "status": "success",
        "channel": "C0123456789",
        "already_member": True,
    }


def test_join_channel_reports_error_for_private_channel(monkeypatch):
    monkeypatch.setattr(
        slack.requests,
        "request",
        Mock(
            return_value=MockResponse(
                {"ok": False, "error": "method_not_supported_for_channel_type"}
            )
        ),
    )

    result = json.loads(slack.slack_join_channel("G0123456789"))

    assert result["status"] == "error"
    assert "method_not_supported_for_channel_type" in result["message"]


def test_join_channel_reports_error_for_private_channel_by_name(monkeypatch):
    """Same as the id-based case above, but through the name-resolution
    path (_resolve_channel_id's "private_channel" type in conversations.list
    covers private channels too, unlike slack_list_channels)."""
    mock_request = Mock(
        side_effect=[
            MockResponse(
                {
                    "ok": True,
                    "channels": [{"id": "G0123456789", "name": "leadership-private"}],
                }
            ),
            MockResponse(
                {"ok": False, "error": "method_not_supported_for_channel_type"}
            ),
        ]
    )
    monkeypatch.setattr(slack.requests, "request", mock_request)

    result = json.loads(slack.slack_join_channel("leadership-private"))

    assert result["status"] == "error"
    assert "method_not_supported_for_channel_type" in result["message"]


def test_join_channel_reports_actionable_error_for_missing_scope(monkeypatch):
    """A user who hasn't reconnected the Slack app since channels:join was
    added would hit missing_scope here — the error must tell them to
    reconnect, not surface the bare Slack code."""
    monkeypatch.setattr(
        slack.requests,
        "request",
        Mock(return_value=MockResponse({"ok": False, "error": "missing_scope"})),
    )

    result = json.loads(slack.slack_join_channel("C0123456789"))

    assert result["status"] == "error"
    assert "reconnect" in result["message"]


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


def test_get_channel_history_reports_actionable_error_when_not_a_member(monkeypatch):
    """Never auto-joins on the caller's behalf — the error must point the
    agent at slack_join_channel (which only fires with the user's OK) rather
    than silently retrying."""
    mock_request = Mock(
        return_value=MockResponse({"ok": False, "error": "not_in_channel"})
    )
    monkeypatch.setattr(slack.requests, "request", mock_request)

    result = json.loads(slack.slack_get_channel_history("C0123456789"))

    assert result["status"] == "error"
    assert "slack_join_channel" in result["message"]
    assert mock_request.call_count == 1


def test_get_channel_history_does_not_treat_channel_not_found_as_actionable(
    monkeypatch,
):
    """conversations.history documents not_in_channel distinctly from
    channel_not_found (unlike conversations.replies/reactions.*), so a
    genuine channel_not_found here must stay a plain error — widening the
    default code set to match the replies/reactions endpoints would make a
    truly-missing channel get misreported as a membership problem."""
    monkeypatch.setattr(
        slack.requests,
        "request",
        Mock(return_value=MockResponse({"ok": False, "error": "channel_not_found"})),
    )

    result = json.loads(slack.slack_get_channel_history("C0123456789"))

    assert result["status"] == "error"
    assert "slack_join_channel" not in result["message"]
    assert "channel_not_found" in result["message"]


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


def test_get_thread_replies_reports_actionable_error_when_not_a_member(monkeypatch):
    """conversations.replies doesn't document not_in_channel — it documents
    channel_not_found instead, which is ambiguous (bad id vs. a real
    channel hidden from a non-member). Since _resolve_channel_id already
    confirmed the channel exists, this must still get the actionable
    message rather than an opaque channel_not_found."""
    mock_request = Mock(
        return_value=MockResponse({"ok": False, "error": "channel_not_found"})
    )
    monkeypatch.setattr(slack.requests, "request", mock_request)

    result = json.loads(slack.slack_get_thread_replies("C0123456789", "1.1"))

    assert result["status"] == "error"
    assert "slack_join_channel" in result["message"]
    assert mock_request.call_count == 1


def test_get_thread_replies_treats_not_in_channel_as_actionable_too(monkeypatch):
    """Undocumented for this endpoint, but handled defensively in case
    Slack's real behavior differs from its docs."""
    monkeypatch.setattr(
        slack.requests,
        "request",
        Mock(return_value=MockResponse({"ok": False, "error": "not_in_channel"})),
    )

    result = json.loads(slack.slack_get_thread_replies("C0123456789", "1.1"))

    assert result["status"] == "error"
    assert "slack_join_channel" in result["message"]


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
    """channel_not_found for conversations.info genuinely means a bad/
    deleted channel id (unlike conversations.replies/reactions.*, which
    overload it for membership too) — Slack documents no_permission, not
    channel_not_found, for the non-member case (see the test below), so
    this must stay a plain, non-actionable error."""
    monkeypatch.setattr(
        slack.requests,
        "request",
        Mock(return_value=MockResponse({"ok": False, "error": "channel_not_found"})),
    )

    result = json.loads(slack.slack_get_channel_info("C0123456789"))

    assert result["status"] == "error"
    assert "channel_not_found" in result["message"]
    assert "slack_join_channel" not in result["message"]


def test_get_channel_info_reports_actionable_error_when_not_a_member(monkeypatch):
    """conversations.info documents no_permission (not channel_not_found)
    for a caller that isn't a member of a private channel — this must get
    the same actionable message every other membership-gated tool in this
    file gives, not a bare Slack error code."""
    monkeypatch.setattr(
        slack.requests,
        "request",
        Mock(return_value=MockResponse({"ok": False, "error": "no_permission"})),
    )

    result = json.loads(slack.slack_get_channel_info("C0123456789"))

    assert result["status"] == "error"
    assert "slack_join_channel" in result["message"]


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


def test_search_messages_dedupes_thread_broadcast_replies(monkeypatch):
    """A thread-broadcast reply is surfaced both in conversations.history
    (the main channel timeline) and again in conversations.replies for its
    thread — a match on that message must appear once, not twice."""
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
                        },
                        {
                            "ts": "1.2",
                            "user": "U2",
                            "text": "the deploy failed",
                            "thread_ts": "1.1",
                        },
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
        {"ts": "1.2", "user": "U2", "text": "the deploy failed"}
    ]


def test_search_messages_flags_truncation_for_a_thread_with_over_200_replies(
    monkeypatch,
):
    """conversations.replies is fetched with a flat limit=200 per thread and
    has_more/next_cursor are not followed — a thread with more replies than
    that must be flagged truncated rather than silently reported complete,
    even when nothing failed and every threaded parent was attempted."""
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
                            "reply_count": 250,
                        }
                    ],
                }
            ),
            MockResponse(
                {
                    "ok": True,
                    "has_more": True,
                    "messages": [{"ts": "1.1", "user": "U1", "text": "kickoff"}],
                }
            ),
        ]
    )
    monkeypatch.setattr(slack.requests, "request", mock_request)

    result = json.loads(slack.slack_search_messages("C0123456789", "deploy"))

    assert result["status"] == "success"
    assert result["matches"] == []
    assert result["truncated"] is True


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


def test_search_messages_reports_actionable_error_when_not_a_member(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse({"ok": False, "error": "not_in_channel"})
    )
    monkeypatch.setattr(slack.requests, "request", mock_request)

    result = json.loads(
        slack.slack_search_messages(
            "C0123456789", "deploy", include_thread_replies=False
        )
    )

    assert result["status"] == "error"
    assert "slack_join_channel" in result["message"]
    assert mock_request.call_count == 1


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


def test_add_reaction_reports_actionable_error_when_not_a_member(monkeypatch):
    """Unlike chat.postMessage, reactions.add has no chat:write.public-style
    exception — it always requires channel membership. It also doesn't
    document not_in_channel (it documents channel_not_found instead), so
    this must get the actionable message from that code, same as the read
    tools."""
    monkeypatch.setattr(
        slack.requests,
        "request",
        Mock(return_value=MockResponse({"ok": False, "error": "channel_not_found"})),
    )

    result = json.loads(slack.slack_add_reaction("C0123456789", "1.1", "thumbsup"))

    assert result["status"] == "error"
    assert "slack_join_channel" in result["message"]


def test_remove_reaction_reports_actionable_error_when_not_a_member(monkeypatch):
    """reactions.remove has the same undocumented-not_in_channel behavior as
    reactions.add — mirrored here since slack_remove_reaction previously had
    no membership test at all."""
    monkeypatch.setattr(
        slack.requests,
        "request",
        Mock(return_value=MockResponse({"ok": False, "error": "channel_not_found"})),
    )

    result = json.loads(slack.slack_remove_reaction("C0123456789", "1.1", "thumbsup"))

    assert result["status"] == "error"
    assert "slack_join_channel" in result["message"]


# ---------------------------------------------------------------------------
# slack_upload_file
# ---------------------------------------------------------------------------


def test_upload_file_rejects_path_outside_allowed_dirs(tmp_path, monkeypatch):
    outside_file = tmp_path / "outside.txt"
    outside_file.write_text("secret")
    allowed_dir = tmp_path / "workspace"
    monkeypatch.setenv("XAGENT_SLACK_FILE_ALLOWED_DIRS", str(allowed_dir))

    result = json.loads(slack.slack_upload_file("C0123456789", str(outside_file)))

    assert result["status"] == "error"
    assert "outside the allowed upload directories" in result["message"]
    # The absolute host path must not leak into the caller/LLM-facing
    # message — only into the server-side log.
    assert str(outside_file) not in result["message"]
    assert str(allowed_dir) not in result["message"]


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


def test_upload_file_reports_actionable_error_when_not_a_member(tmp_path, monkeypatch):
    """files.completeUploadExternal requires channel membership the same way
    conversations.history does — this must get the same slack_join_channel
    guidance as the read tools, not a bare not_in_channel code."""
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
            MockResponse({"ok": False, "error": "not_in_channel"}),
        ]
    )
    monkeypatch.setattr(slack.requests, "request", mock_request)
    monkeypatch.setattr(
        slack.requests, "post", Mock(return_value=MockResponse({}, status_code=200))
    )

    result = json.loads(slack.slack_upload_file("C0123456789", str(local_file)))

    assert result["status"] == "error"
    assert "slack_join_channel" in result["message"]


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
    assert result["error"] == "ratelimited"


def test_search_messages_surfaces_actionable_thread_join_failure(monkeypatch):
    """A thread-replies fetch that fails because the bot isn't a member of
    the channel must not be reduced to `truncated: true` with no
    explanation — the actionable message from _request_requiring_membership has to
    reach the caller, not just a server-side log line.

    Uses not_in_channel rather than channel_not_found: the history call
    just above already succeeded for this channel_id, which proves
    membership, so channel_not_found from conversations.replies is no
    longer actionable here (see
    test_search_messages_thread_channel_not_found_is_not_actionable_once_membership_proven)
    — not_in_channel exercises the same "actionable failure reaches the
    caller" behavior without relying on that now-fixed misclassification."""
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
            MockResponse({"ok": False, "error": "not_in_channel"}),
        ]
    )
    monkeypatch.setattr(slack.requests, "request", mock_request)

    result = json.loads(slack.slack_search_messages("C0123456789", "deploy"))

    assert result["status"] == "success"
    assert result["truncated"] is True
    assert "slack_join_channel" in result["error"]


def test_search_messages_thread_channel_not_found_is_not_actionable_once_membership_proven(
    monkeypatch,
):
    """channel_not_found from conversations.replies is ambiguous in
    general (see _PATHS_HIDING_CHANNEL_FROM_NON_MEMBERS's module comment),
    but inside slack_search_messages the conversations.history call just
    above always succeeds for this same channel_id first — which proves
    membership — before the thread loop ever runs. A later channel_not_found
    there therefore cannot mean "not a member"; it must stay a plain,
    non-actionable error (most likely the thread was deleted, or the
    channel was archived mid-scan), and — unlike a genuine actionable
    failure — must not stop the scan: the next collected thread still gets
    attempted."""
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
                        },
                        {
                            "ts": "1.2",
                            "user": "U1",
                            "text": "the deploy failed",
                            "thread_ts": "1.2",
                            "reply_count": 1,
                        },
                    ],
                }
            ),
            MockResponse({"ok": False, "error": "channel_not_found"}),
            MockResponse({"ok": True, "messages": [{"ts": "1.2", "user": "U1"}]}),
        ]
    )
    monkeypatch.setattr(slack.requests, "request", mock_request)

    result = json.loads(slack.slack_search_messages("C0123456789", "deploy"))

    assert result["status"] == "success"
    assert "slack_join_channel" not in result.get("error", "")
    assert "channel_not_found" in result["error"]
    # 1 history call + both thread attempts — the loop must not stop after
    # the first thread's channel_not_found, since it isn't actionable here.
    assert mock_request.call_count == 3


def test_search_messages_thread_no_permission_is_not_actionable_once_membership_proven(
    monkeypatch,
):
    """no_permission is overloaded by Slack the same way channel_not_found
    is (see the module comment on _AMBIGUOUS_NOT_A_MEMBER_CODES) — once the
    conversations.history call just above already proved membership on
    this same channel_id, a later no_permission on conversations.replies
    can no longer mean "not a member" either, and must get the same
    non-actionable, scan-continues treatment as channel_not_found does in
    the sibling test above."""
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
                        },
                        {
                            "ts": "1.2",
                            "user": "U1",
                            "text": "the deploy failed",
                            "thread_ts": "1.2",
                            "reply_count": 1,
                        },
                    ],
                }
            ),
            MockResponse({"ok": False, "error": "no_permission"}),
            MockResponse({"ok": True, "messages": [{"ts": "1.2", "user": "U1"}]}),
        ]
    )
    monkeypatch.setattr(slack.requests, "request", mock_request)

    result = json.loads(slack.slack_search_messages("C0123456789", "deploy"))

    assert result["status"] == "success"
    assert "slack_join_channel" not in result.get("error", "")
    assert "no_permission" in result["error"]
    # 1 history call + both thread attempts — the loop must not stop after
    # the first thread's no_permission, since it isn't actionable here.
    assert mock_request.call_count == 3


def test_search_messages_actionable_thread_error_survives_a_later_unrelated_one(
    monkeypatch,
):
    """An actionable missing_scope/reconnect failure on one thread must not
    be masked by an earlier, less useful failure (e.g. a rate limit) on a
    different thread in the same search — the actionable guidance has to
    win even though it wasn't the first failure seen.
    (Ordered transient-then-actionable deliberately: the reverse order is
    covered by test_search_messages_stops_thread_scan_after_channel_wide_membership_failure,
    which also asserts the scan stops there instead of continuing. Uses
    missing_scope rather than channel_not_found so this test's intent —
    actionable-wins-over-earlier-transient — doesn't depend on the
    proven-membership channel_not_found case covered separately above.)"""
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
                        },
                        {
                            "ts": "1.2",
                            "user": "U1",
                            "text": "kickoff again",
                            "thread_ts": "1.2",
                            "reply_count": 1,
                        },
                    ],
                }
            ),
            MockResponse({"ok": False, "error": "ratelimited"}),
            MockResponse({"ok": False, "error": "missing_scope"}),
        ]
    )
    monkeypatch.setattr(slack.requests, "request", mock_request)

    result = json.loads(slack.slack_search_messages("C0123456789", "deploy"))

    assert result["status"] == "success"
    assert result["truncated"] is True
    assert "reconnect" in result["error"]
    assert mock_request.call_count == 3


def test_search_messages_stops_thread_scan_after_channel_wide_membership_failure(
    monkeypatch,
):
    """A not-a-member failure discovered on one thread is a channel-wide bot
    state, not specific to that thread — every remaining threaded parent
    would fail identically, so the scan must stop there instead of
    repeating the same doomed conversations.replies call for each of the
    (here, 3) collected threads.

    Uses not_in_channel rather than channel_not_found — see
    test_search_messages_surfaces_actionable_thread_join_failure for why."""
    mock_request = Mock(
        side_effect=[
            MockResponse(
                {
                    "ok": True,
                    "messages": [
                        {
                            "ts": f"1.{i}",
                            "user": "U1",
                            "text": "kickoff",
                            "thread_ts": f"1.{i}",
                            "reply_count": 1,
                        }
                        for i in range(3)
                    ],
                }
            ),
            MockResponse({"ok": False, "error": "not_in_channel"}),
        ]
    )
    monkeypatch.setattr(slack.requests, "request", mock_request)

    result = json.loads(slack.slack_search_messages("C0123456789", "deploy"))

    assert result["status"] == "success"
    assert result["truncated"] is True
    assert "slack_join_channel" in result["error"]
    # 1 history call + exactly 1 thread attempt — the other 2 collected
    # threaded parents must never be attempted once the first one reveals
    # a channel-wide membership problem.
    assert mock_request.call_count == 2


def test_search_messages_stops_thread_scan_after_missing_scope_failure(monkeypatch):
    """missing_scope is connection-wide, not even channel-specific — it's at
    least as certain to recur on every remaining thread as a not-a-member
    failure is, so the scan must stop there too. Exercises the shared
    _SlackActionableError base (not a hardcoded _SlackNotAMemberError
    check), since _SlackMissingScopeError is a sibling subclass, not that
    one specifically."""
    mock_request = Mock(
        side_effect=[
            MockResponse(
                {
                    "ok": True,
                    "messages": [
                        {
                            "ts": f"1.{i}",
                            "user": "U1",
                            "text": "kickoff",
                            "thread_ts": f"1.{i}",
                            "reply_count": 1,
                        }
                        for i in range(3)
                    ],
                }
            ),
            MockResponse({"ok": False, "error": "missing_scope"}),
        ]
    )
    monkeypatch.setattr(slack.requests, "request", mock_request)

    result = json.loads(slack.slack_search_messages("C0123456789", "deploy"))

    assert result["status"] == "success"
    assert result["truncated"] is True
    assert "reconnect" in result["error"]
    assert mock_request.call_count == 2


def test_search_messages_history_page_failure_still_scans_collected_threads(
    monkeypatch,
):
    """A page-2+ history failure must not skip scanning threaded parents
    already collected from page 1 — only the pages after the failure are
    lost, not the thread replies for messages already seen."""
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
                    "response_metadata": {"next_cursor": "page-2"},
                }
            ),
            MockResponse({"ok": False, "error": "ratelimited"}),
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
    assert result["truncated"] is True
    assert result["error"] == "ratelimited"


def test_search_messages_skips_thread_scan_after_channel_wide_membership_error(
    monkeypatch,
):
    """Unlike a transient history-page failure (which still scans
    already-collected threads, per the test above), a not-a-member failure
    is a channel-wide bot state — every conversations.replies call for this
    same channel would fail identically, so the thread loop must not even
    attempt them."""
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
                    "response_metadata": {"next_cursor": "page-2"},
                }
            ),
            MockResponse({"ok": False, "error": "not_in_channel"}),
        ]
    )
    monkeypatch.setattr(slack.requests, "request", mock_request)

    result = json.loads(slack.slack_search_messages("C0123456789", "deploy"))

    assert result["status"] == "success"
    assert result["truncated"] is True
    assert "slack_join_channel" in result["error"]
    # Exactly 2 calls: the failing page-2 history call, then nothing else —
    # no conversations.replies attempt for the threaded parent from page 1.
    assert mock_request.call_count == 2


def test_search_messages_history_page_failure_preserves_partial_matches(monkeypatch):
    """A mid-pagination failure on the top-level history scan (page 2+) must
    not discard matches already found on page 1 — mirrors the partial-result
    pattern slack_list_channels/slack_list_direct_messages already use."""
    mock_request = Mock(
        side_effect=[
            MockResponse(
                {
                    "ok": True,
                    "messages": [
                        {"ts": "1.1", "user": "U1", "text": "the deploy failed"}
                    ],
                    "response_metadata": {"next_cursor": "page-2"},
                }
            ),
            MockResponse({"ok": False, "error": "ratelimited"}),
        ]
    )
    monkeypatch.setattr(slack.requests, "request", mock_request)

    result = json.loads(
        slack.slack_search_messages(
            "C0123456789", "deploy", include_thread_replies=False
        )
    )

    assert result["status"] == "success"
    assert result["matches"] == [
        {"ts": "1.1", "user": "U1", "text": "the deploy failed"}
    ]
    assert result["truncated"] is True
    assert result["error"] == "ratelimited"


# ---------------------------------------------------------------------------
# F9 — the Slack-ID short-circuit must not accept user/bot ids as channels
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("C0123456789", True),  # public channel
        ("G0123456789", True),  # private channel / mpim
        ("D0123456789", True),  # 1:1 DM
        ("U0123456789", False),  # user id — must NOT short-circuit
        ("B0123456789", False),  # bot id — must NOT short-circuit
    ],
)
def test_slack_id_pattern_restricted_to_conversation_prefixes(value, expected):
    assert bool(slack._SLACK_ID_PATTERN.match(value)) is expected


def test_resolve_channel_id_does_not_short_circuit_a_user_id(monkeypatch):
    """A user id (e.g. copied from slack_list_direct_messages' "user" field)
    must go through name resolution rather than being treated as an
    already-resolved channel id — it will fail with a clear "could not
    resolve" error instead of an opaque Slack channel_not_found."""
    monkeypatch.setattr(
        slack.requests,
        "request",
        Mock(return_value=MockResponse({"ok": True, "channels": []})),
    )

    with pytest.raises(ValueError, match="Could not resolve channel"):
        slack._resolve_channel_id("U0123456789")


# ---------------------------------------------------------------------------
# F6 — _allowed_file_dirs must strip whitespace around comma-separated dirs
# ---------------------------------------------------------------------------


def test_allowed_file_dirs_strips_whitespace_around_entries(tmp_path, monkeypatch):
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()
    monkeypatch.setenv("XAGENT_SLACK_FILE_ALLOWED_DIRS", f"  {dir_a} ,{dir_b}  ")

    result = slack._allowed_file_dirs()

    assert result == [dir_a.resolve(), dir_b.resolve()]


# ---------------------------------------------------------------------------
# F7 — slack_upload_file must reject empty files
# ---------------------------------------------------------------------------


def test_upload_file_rejects_empty_file(tmp_path, monkeypatch):
    empty_file = tmp_path / "empty.txt"
    empty_file.write_text("")
    monkeypatch.setenv("XAGENT_SLACK_FILE_ALLOWED_DIRS", str(tmp_path))

    result = json.loads(slack.slack_upload_file("C0123456789", str(empty_file)))

    assert result["status"] == "error"
    assert "empty" in result["message"].lower()


# ---------------------------------------------------------------------------
# F3 — thread-fetch budget is consumed even when the request fails
# ---------------------------------------------------------------------------


def test_search_messages_thread_budget_consumed_on_repeated_failures(monkeypatch):
    """MAX_SEARCH_THREADS bounds total conversations.replies *attempts*, not
    just successes — a channel with more threaded parents than the budget,
    all of which fail, must not be retried past the budget."""
    threaded_history = MockResponse(
        {
            "ok": True,
            "messages": [
                {
                    "ts": f"1.{i}",
                    "user": "U1",
                    "text": "kickoff",
                    "thread_ts": f"1.{i}",
                    "reply_count": 1,
                }
                for i in range(slack.MAX_SEARCH_THREADS + 5)
            ],
        }
    )
    failing_reply = MockResponse({"ok": False, "error": "ratelimited"})
    mock_request = Mock(side_effect=[threaded_history] + [failing_reply] * 100)
    monkeypatch.setattr(slack.requests, "request", mock_request)

    result = json.loads(slack.slack_search_messages("C0123456789", "deploy"))

    assert result["status"] == "success"
    assert result["truncated"] is True
    # 1 history call + exactly MAX_SEARCH_THREADS reply attempts, not one per
    # threaded parent (there are 5 more parents than the budget allows).
    assert mock_request.call_count == 1 + slack.MAX_SEARCH_THREADS


# ---------------------------------------------------------------------------
# F4 — slack_search_messages limit must not be an unbounded pass-through
# ---------------------------------------------------------------------------


def test_search_messages_limit_is_clamped(monkeypatch):
    monkeypatch.setattr(
        slack.requests,
        "request",
        Mock(
            return_value=MockResponse(
                {
                    "ok": True,
                    "messages": [
                        {"ts": "1.1", "user": "U1", "text": "the deploy failed"}
                    ],
                }
            )
        ),
    )

    result = json.loads(
        slack.slack_search_messages(
            "C0123456789", "deploy", limit=10**9, include_thread_replies=False
        )
    )

    assert result["status"] == "success"
    assert result["matches"] == [
        {"ts": "1.1", "user": "U1", "text": "the deploy failed"}
    ]


# ---------------------------------------------------------------------------
# F5 — slack_list_direct_messages must tolerate a mid-pagination failure
# ---------------------------------------------------------------------------


def test_list_direct_messages_returns_partial_results_when_a_later_page_fails(
    monkeypatch,
):
    mock_request = Mock(
        side_effect=[
            MockResponse(
                {
                    "ok": True,
                    "channels": [{"id": "D1", "user": "U1", "is_mpim": False}],
                    "response_metadata": {"next_cursor": "page-2"},
                }
            ),
            MockResponse({"ok": False, "error": "ratelimited"}),
        ]
    )
    monkeypatch.setattr(slack.requests, "request", mock_request)

    result = json.loads(slack.slack_list_direct_messages())

    assert result["status"] == "success"
    assert result["conversations"] == [{"id": "D1", "is_group_dm": False, "user": "U1"}]
    assert result["truncated"] is True
    assert "ratelimited" in result["error"]


def test_list_direct_messages_raises_when_first_page_fails(monkeypatch):
    monkeypatch.setattr(
        slack.requests,
        "request",
        Mock(return_value=MockResponse({"ok": False, "error": "missing_scope"})),
    )

    result = json.loads(slack.slack_list_direct_messages())

    assert result["status"] == "error"
    assert "missing_scope" in result["message"]


# ---------------------------------------------------------------------------
# S1 — slack_remove_reaction error path (parity with slack_add_reaction)
# ---------------------------------------------------------------------------


def test_remove_reaction_reports_error_payload(monkeypatch):
    monkeypatch.setattr(
        slack.requests,
        "request",
        Mock(return_value=MockResponse({"ok": False, "error": "no_reaction"})),
    )

    result = json.loads(slack.slack_remove_reaction("C0123456789", "1.1", "thumbsup"))

    assert result["status"] == "error"
    assert "no_reaction" in result["message"]
