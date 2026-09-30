from __future__ import annotations

import json
import time
from unittest.mock import Mock

import pytest

from xagent.web.tools.mcp import slack

ALLOWED = "C0123456789"
DENIED = "C9876543210"


@pytest.fixture(autouse=True)
def _slack_token(monkeypatch):
    monkeypatch.setenv("SLACK_ACCESS_TOKEN", "test-token")


def _install_policy(monkeypatch, *, channel_ids=(ALLOWED,), expires_at=None) -> None:
    monkeypatch.setenv(
        slack._CHANNEL_ACCESS_POLICY_ENV_VAR,
        json.dumps(
            {
                "version": 2,
                "channel_ids": list(channel_ids),
                "expires_at": expires_at or time.time() + 300,
                "capabilities": ["read"],
            }
        ),
    )


@pytest.mark.parametrize(
    "payload",
    [
        "not-json",
        "{}",
        '{"version":1,"channel_ids":[],"expires_at":9999999999}',
        '{"version":2,"channel_ids":[],"expires_at":NaN,"capabilities":["read"]}',
        '{"version":true,"channel_ids":[],"expires_at":9999999999}',
        '{"version":2,"channel_ids":[],"expires_at":9999999999,"capabilities":[]}',
        '{"version":2,"channel_ids":[],"expires_at":9999999999,"capabilities":["read","write"]}',
        '{"version":2,"channel_ids":[],"expires_at":9999999999,"capabilities":[true]}',
    ],
)
def test_malformed_policy_denies_listing_before_slack_api(monkeypatch, payload) -> None:
    monkeypatch.setenv(slack._CHANNEL_ACCESS_POLICY_ENV_VAR, payload)
    request = Mock()
    monkeypatch.setattr(slack.requests, "request", request)

    result = json.loads(slack.slack_list_channels())

    assert result["status"] == "error"
    assert "policy is unavailable or expired" in result["message"]
    request.assert_not_called()


def test_expired_policy_denies_direct_channel_before_slack_api(monkeypatch) -> None:
    _install_policy(monkeypatch, expires_at=time.time() - 1)
    request = Mock()
    monkeypatch.setattr(slack.requests, "request", request)

    result = json.loads(slack.slack_get_channel_history(ALLOWED))

    assert result["status"] == "error"
    request.assert_not_called()


def test_policy_expiring_between_pages_denies_instead_of_returning_partial_data(
    monkeypatch,
) -> None:
    _install_policy(monkeypatch, expires_at=200.0)
    monkeypatch.setattr(
        slack, "_policy_now", Mock(side_effect=[100.0, 100.0, 100.0, 201.0])
    )
    response = Mock(
        status_code=200,
        json=Mock(
            return_value={
                "ok": True,
                "channels": [{"id": ALLOWED, "name": "allowed"}],
                "response_metadata": {"next_cursor": "page-2"},
            }
        ),
        raise_for_status=Mock(),
    )
    request = Mock(return_value=response)
    monkeypatch.setattr(slack.requests, "request", request)

    result = json.loads(slack.slack_list_channels())

    assert result["status"] == "error"
    assert "policy is unavailable or expired" in result["message"]
    assert request.call_count == 1


@pytest.mark.parametrize(
    ("operation", "args"),
    [
        (slack.slack_join_channel, (DENIED,)),
        (slack.slack_post_message, (DENIED, "hello")),
        (slack.slack_get_channel_history, (DENIED,)),
        (slack.slack_get_thread_replies, (DENIED, "1.0")),
        (slack.slack_get_channel_info, (DENIED,)),
        (slack.slack_search_messages, (DENIED, "needle")),
        (slack.slack_add_reaction, (DENIED, "1.0", "eyes")),
        (slack.slack_remove_reaction, (DENIED, "1.0", "eyes")),
        (slack.slack_upload_file, (DENIED, "/does/not/matter")),
    ],
)
def test_every_channel_operation_denies_out_of_policy_before_api(
    monkeypatch, operation, args
) -> None:
    _install_policy(monkeypatch)
    request = Mock()
    post = Mock()
    monkeypatch.setattr(slack.requests, "request", request)
    monkeypatch.setattr(slack.requests, "post", post)

    result = json.loads(operation(*args))

    assert result["status"] == "error"
    assert "does not allow" in result["message"]
    request.assert_not_called()
    post.assert_not_called()


@pytest.mark.parametrize(
    ("operation", "args"),
    [
        (slack.slack_join_channel, (ALLOWED,)),
        (slack.slack_post_message, (ALLOWED, "hello")),
        (slack.slack_add_reaction, (ALLOWED, "1.0", "eyes")),
        (slack.slack_remove_reaction, (ALLOWED, "1.0", "eyes")),
        (slack.slack_upload_file, (ALLOWED, "/does/not/matter")),
    ],
)
def test_read_only_policy_denies_every_mutation_before_io(
    monkeypatch, operation, args
) -> None:
    _install_policy(monkeypatch)
    request = Mock()
    post = Mock()
    monkeypatch.setattr(slack.requests, "request", request)
    monkeypatch.setattr(slack.requests, "post", post)

    result = json.loads(operation(*args))

    assert result == {
        "status": "error",
        "message": "Slack runtime policy does not allow this operation",
    }
    request.assert_not_called()
    post.assert_not_called()


def test_name_resolves_to_id_before_policy_check(monkeypatch) -> None:
    _install_policy(monkeypatch)
    request = Mock(
        side_effect=[
            Mock(
                status_code=200,
                json=Mock(
                    return_value={
                        "ok": True,
                        "channels": [{"id": ALLOWED, "name": "incident"}],
                    }
                ),
                raise_for_status=Mock(),
            ),
            Mock(
                status_code=200,
                json=Mock(return_value={"ok": True, "messages": []}),
                raise_for_status=Mock(),
            ),
        ]
    )
    monkeypatch.setattr(slack.requests, "request", request)

    result = json.loads(slack.slack_get_channel_history("#incident"))

    assert result["status"] == "success"
    assert request.call_count == 2
    assert request.call_args_list[1].kwargs["params"]["channel"] == ALLOWED


def test_name_resolving_outside_policy_never_reads_history(monkeypatch) -> None:
    _install_policy(monkeypatch)
    response = Mock(
        status_code=200,
        json=Mock(
            return_value={
                "ok": True,
                "channels": [{"id": DENIED, "name": "incident"}],
            }
        ),
        raise_for_status=Mock(),
    )
    request = Mock(return_value=response)
    monkeypatch.setattr(slack.requests, "request", request)

    result = json.loads(slack.slack_get_channel_history("incident"))

    assert result["status"] == "error"
    assert request.call_count == 1
    assert request.call_args.kwargs["params"]["types"] == (
        "public_channel,private_channel"
    )


@pytest.mark.parametrize(
    ("operation", "result_key", "types"),
    [
        (slack.slack_list_channels, "channels", "public_channel"),
        (slack.slack_list_direct_messages, "conversations", "im,mpim"),
    ],
)
def test_listings_filter_out_of_policy_conversations(
    monkeypatch, operation, result_key, types
) -> None:
    _install_policy(monkeypatch)
    response = Mock(
        status_code=200,
        json=Mock(
            return_value={
                "ok": True,
                "channels": [
                    {"id": ALLOWED, "name": "allowed"},
                    {"id": DENIED, "name": "denied"},
                ],
            }
        ),
        raise_for_status=Mock(),
    )
    request = Mock(return_value=response)
    monkeypatch.setattr(slack.requests, "request", request)

    result = json.loads(operation())

    assert [item["id"] for item in result[result_key]] == [ALLOWED]
    assert request.call_args.kwargs["params"]["types"] == types


def test_policy_absence_preserves_legacy_user_id_posting(monkeypatch) -> None:
    response = Mock(
        status_code=200,
        json=Mock(return_value={"ok": True, "channel": "D111111", "ts": "1.0"}),
        raise_for_status=Mock(),
    )
    request = Mock(return_value=response)
    monkeypatch.setattr(slack.requests, "request", request)

    result = json.loads(slack.slack_post_message("U0123456789", "hello"))

    assert result["status"] == "success"
    assert request.call_args.kwargs["json"]["channel"] == "U0123456789"
