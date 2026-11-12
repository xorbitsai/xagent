"""Slack's smoke V2 Tier 1 checks (#3571), run end to end.

The declaration in ``registry.py`` runs through the real engine, and every
check calls the real tool function in ``web/tools/mcp/slack.py``. Only
Slack's Web API is replaced, by an in-memory workspace laid out like the QA
workspace's seed: one public channel whose topic is set, and one thread in
it, a parent message with one reply, posted after the bots joined and the
topic was set.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from collections.abc import Callable
from typing import Any

import pytest

from xagent.web.connector_checks.engine import run_checks
from xagent.web.connector_checks.registry import CONNECTOR_CHECKS
from xagent.web.connector_checks.results import RunResult
from xagent.web.tools.mcp import slack

from .fakes import FakeAdapter, FakeTool, listed, rows, targets

CHANNEL = "C0SEED00001"
OTHER_CHANNEL = "C0GENERAL01"
CHANNEL_NAME = "xagent-smoke-seed"
TOPIC = "xagent smoke seed topic"
THREAD = "1762905600.000100"
THREAD_TEXT = "xagent smoke seed thread"
REPLY = "1762905601.000200"
REPLY_TEXT = "xagent smoke seed reply"
BOT_JOINED = "1762905500.000100"
FIXTURES = {
    "slack.seed.channel_id": CHANNEL,
    "slack.seed.thread_ts": THREAD,
    "slack.seed.reply_ts": REPLY,
}
# Slack allows a new non-Marketplace installation one call a minute to each.
_THROTTLED = ("conversations.history", "conversations.replies")


class _Response:
    status_code = 200
    headers: dict[str, str] = {}

    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._payload


class Workspace:
    """An in-memory Slack workspace answering the requests slack.py sends."""

    def __init__(self) -> None:
        self.revoked = False
        self.bot_is_member = True
        self.channels: dict[str, dict[str, Any]] = {
            CHANNEL: self._channel(CHANNEL, CHANNEL_NAME, TOPIC),
            OTHER_CHANNEL: self._channel(OTHER_CHANNEL, "general", ""),
        }
        # Top-level messages per channel, oldest first: the bots joined and
        # the topic was set before the seed posted its thread.
        self.messages: dict[str, list[dict[str, Any]]] = {
            CHANNEL: [
                {
                    "ts": BOT_JOINED,
                    "user": "U0BOT000001",
                    "subtype": "channel_join",
                    "text": "<@U0BOT000001> has joined the channel",
                },
                {
                    "ts": "1762905510.000100",
                    "user": "U0SEED00001",
                    "subtype": "channel_topic",
                    "text": f"set the channel topic: {TOPIC}",
                },
                {
                    "ts": THREAD,
                    "user": "U0SEED00001",
                    "text": THREAD_TEXT,
                    "thread_ts": THREAD,
                    "reply_count": 1,
                },
            ],
            OTHER_CHANNEL: [],
        }
        self.replies: dict[str, list[dict[str, Any]]] = {
            THREAD: [
                {
                    "ts": REPLY,
                    "user": "U0SEED00001",
                    "text": REPLY_TEXT,
                    "thread_ts": THREAD,
                }
            ]
        }
        self.calls: list[tuple[str, str]] = []

    @staticmethod
    def _channel(channel_id: str, name: str, topic: str) -> dict:
        return {
            "id": channel_id,
            "name": name,
            "is_archived": False,
            "is_private": False,
            "is_im": False,
            "is_mpim": False,
            "topic": {"value": topic},
            "purpose": {"value": ""},
        }

    def request(self, *, method: str, url: str, **kwargs: Any) -> _Response:
        api = url.removeprefix(f"{slack.SLACK_BASE_URL}/")
        self.calls.append((method, api))
        return _Response(self._answer(api, kwargs.get("params") or {}))

    def _answer(self, api: str, params: dict[str, Any]) -> dict[str, Any]:
        if self.revoked:
            return {"ok": False, "error": "token_revoked"}
        if api == "conversations.list":
            assert params["types"] == "public_channel", params
            return {
                "ok": True,
                "channels": list(self.channels.values()),
                "response_metadata": {"next_cursor": ""},
            }
        channel = params["channel"]
        if channel not in self.channels:
            return {"ok": False, "error": "channel_not_found"}
        # Slack documents not_in_channel for conversations.history; the fake
        # gates no other method on membership.
        if (
            api == "conversations.history"
            and channel == CHANNEL
            and not self.bot_is_member
        ):
            return {"ok": False, "error": "not_in_channel"}
        if api == "conversations.info":
            return {"ok": True, "channel": self.channels[channel]}
        if api == "conversations.history":
            return self._history(channel, params)
        if api == "conversations.replies":
            parents = [m for m in self.messages[channel] if m["ts"] == params["ts"]]
            if not parents:
                return {"ok": False, "error": "thread_not_found"}
            return {
                "ok": True,
                "messages": [*parents, *self.replies.get(params["ts"], [])],
                "has_more": False,
            }
        raise AssertionError(api)

    def _history(self, channel: str, params: dict[str, Any]) -> dict[str, Any]:
        # Newest first; ``latest`` and ``oldest`` are exclusive by default.
        found = [
            message
            for message in reversed(self.messages[channel])
            if float(message["ts"]) < float(params.get("latest") or "inf")
            and float(message["ts"]) > float(params.get("oldest") or 0)
        ]
        limit = int(params["limit"])
        return {"ok": True, "messages": found[:limit], "has_more": len(found) > limit}

    def post(self, ts: str, text: str, *, thread: str | None = None) -> None:
        message = {"ts": ts, "user": "U0OTHER0001", "text": text}
        if thread is None:
            self.messages[CHANNEL].append(message)
        else:
            self.replies[thread].append({**message, "thread_ts": thread})
            # Slack counts a parent's replies as they are posted.
            (parent,) = [m for m in self.messages[CHANNEL] if m["ts"] == thread]
            parent["reply_count"] += 1


def _tool(name: str) -> FakeTool:
    function: Callable[..., str] = getattr(slack, name)

    def respond(args: dict[str, Any]) -> Any:
        return {
            "content": [{"type": "text", "text": function(**args)}],
            "structured_content": None,
            "is_error": False,
        }

    return FakeTool(name, respond=respond)


# Every tool the server registers, so the session lists what production does
# without a channel access policy.
_TOOLS = (
    "slack_join_channel",
    "slack_list_channels",
    "slack_post_message",
    "slack_get_channel_history",
    "slack_get_thread_replies",
    "slack_get_channel_info",
    "slack_list_direct_messages",
    "slack_search_messages",
    "slack_add_reaction",
    "slack_remove_reaction",
    "slack_upload_file",
)


@pytest.fixture()
def workspace(monkeypatch: pytest.MonkeyPatch) -> Workspace:
    workspace = Workspace()
    monkeypatch.setenv("SLACK_ACCESS_TOKEN", "test-token")
    monkeypatch.delenv(slack._CHANNEL_ACCESS_POLICY_ENV_VAR, raising=False)
    monkeypatch.setattr(slack.requests, "request", workspace.request)
    return workspace


def _adapter(fixtures: dict[str, Any] = FIXTURES) -> FakeAdapter:
    tools = [_tool(name) for name in _TOOLS]
    return FakeAdapter(targets("slack"), {"slack": listed(*tools)}, fixtures=fixtures)


async def _run(adapter: FakeAdapter) -> RunResult:
    return await run_checks(adapter, check_timeout=5.0, total_timeout=60.0)


def _check_rows(result: RunResult) -> dict[str, tuple[Any, ...]]:
    # The session row carries no check id.
    return {row[1] or "session": row[2:] for row in rows(result) if row[0] == "slack"}


_CHECKS = (
    "slack.l3_probe",
    "slack.find_seed_channel",
    "slack.read_seed_history",
    "slack.read_seed_thread",
    "slack.read_seed_channel",
)
_PASSED = {
    "session": ("pass", None, None),
    **{check: ("pass", None, None) for check in _CHECKS},
}


# The declaration


def test_slack_checks_the_tools_that_carry_its_calls() -> None:
    checks = CONNECTOR_CHECKS["slack"]

    assert [check.id for check in checks.reads] == list(_CHECKS)
    assert checks.writes == ()
    assert checks.sweep is None
    assert checks.checked_tools() == {
        "slack_list_channels",
        "slack_get_channel_history",
        "slack_get_thread_replies",
        "slack_get_channel_info",
    }
    assert sorted(exemption.tool for exemption in checks.exemptions) == [
        "slack_add_reaction",
        "slack_join_channel",
        "slack_list_direct_messages",
        "slack_post_message",
        "slack_remove_reaction",
        "slack_search_messages",
        "slack_upload_file",
    ]


def test_each_exemption_names_the_issue_that_decides_it() -> None:
    reasons = {e.tool: e.reason for e in CONNECTOR_CHECKS["slack"].exemptions}

    for tool in ("slack_join_channel", "slack_add_reaction", "slack_remove_reaction"):
        assert reasons[tool].endswith("decided in #3743"), tool
    for tool in ("slack_search_messages", "slack_list_direct_messages"):
        assert reasons[tool].endswith("decided in #3571"), tool


# A run against the seeded workspace


async def test_every_check_passes_on_the_seeded_workspace_and_writes_nothing(
    workspace: Workspace,
) -> None:
    result = await _run(_adapter())

    assert _check_rows(result) == _PASSED
    assert result.complete
    assert {method for method, _ in workspace.calls} == {"GET"}


async def test_a_run_calls_each_throttled_method_once(workspace: Workspace) -> None:
    await _run(_adapter())

    called = Counter(api for _, api in workspace.calls)
    assert {api: called[api] for api in _THROTTLED} == {api: 1 for api in _THROTTLED}


async def test_every_check_passes_under_the_bot_grants_read_policy(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    # What a Toby run starts Slack with: a read-only policy that admits the
    # public seed channel (xorbitsai/xagent-saas#2553).
    monkeypatch.setenv(
        slack._CHANNEL_ACCESS_POLICY_ENV_VAR,
        json.dumps(
            {
                "version": 2,
                "channel_ids": [CHANNEL],
                "expires_at": time.time() + 300,
                "capabilities": ["read"],
            }
        ),
    )

    result = await _run(_adapter())

    assert _check_rows(result) == _PASSED


async def test_later_messages_in_the_seed_channel_break_no_check(
    workspace: Workspace,
) -> None:
    # Another bot joins after the seed, someone posts, and the thread grows.
    workspace.post("1762909200.000100", "<@U0BOT000002> has joined the channel")
    workspace.post("1762909300.000100", "an unrelated message")
    workspace.post("1762909400.000100", "a later reply", thread=THREAD)

    result = await _run(_adapter())

    assert _check_rows(result) == _PASSED


async def test_without_the_seeds_fixtures_only_the_probe_runs(
    workspace: Workspace,
) -> None:
    result = await _run(_adapter({}))

    checks = _check_rows(result)
    assert checks.pop("slack.l3_probe") == ("pass", None, None)
    assert checks.pop("session") == ("pass", None, None)
    assert {outcome[:2] for outcome in checks.values()} == {("skip", "fixture_missing")}


async def test_a_revoked_grant_fails_every_check(workspace: Workspace) -> None:
    workspace.revoked = True

    result = await _run(_adapter())

    checks = _check_rows(result)
    assert checks.pop("session") == ("pass", None, None)
    assert set(checks) == set(_CHECKS)
    assert {outcome for outcome, _, _ in checks.values()} == {"fail"}


async def test_a_bot_outside_the_seed_channel_fails_the_history_read(
    workspace: Workspace,
) -> None:
    workspace.bot_is_member = False

    result = await _run(_adapter())

    outcome, _, _ = _check_rows(result)["slack.read_seed_history"]
    assert outcome == "fail"


@pytest.mark.parametrize(
    ("key", "value", "check"),
    [
        ("slack.seed.channel_id", OTHER_CHANNEL, "slack.find_seed_channel"),
        ("slack.seed.channel_id", OTHER_CHANNEL, "slack.read_seed_channel"),
        ("slack.seed.thread_ts", "1762905599.000100", "slack.read_seed_history"),
        ("slack.seed.thread_ts", "1762905599.000100", "slack.read_seed_thread"),
        ("slack.seed.reply_ts", "1762905602.000300", "slack.read_seed_thread"),
    ],
)
async def test_a_seeded_object_that_drifted_fails_its_read(
    workspace: Workspace, key: str, value: str, check: str
) -> None:
    result = await _run(_adapter({**FIXTURES, key: value}))

    outcome, failure_class, _ = _check_rows(result)[check]
    assert (outcome, failure_class) == ("fail", "assertion_failed")


@pytest.mark.parametrize(
    ("drift", "check"),
    [
        (
            lambda w: w.channels[CHANNEL].update(name="renamed"),
            "slack.find_seed_channel",
        ),
        (
            lambda w: w.channels[CHANNEL]["topic"].update(value="changed"),
            "slack.read_seed_channel",
        ),
        (
            lambda w: w.messages[CHANNEL][-1].update(text="edited"),
            "slack.read_seed_history",
        ),
        (
            lambda w: w.messages[CHANNEL][-1].pop("reply_count"),
            "slack.read_seed_history",
        ),
        (
            lambda w: w.replies[THREAD][0].update(text="edited"),
            "slack.read_seed_thread",
        ),
    ],
)
async def test_seeded_content_that_changed_fails_its_read(
    workspace: Workspace, drift: Callable[[Workspace], None], check: str
) -> None:
    drift(workspace)

    result = await _run(_adapter())

    outcome, failure_class, _ = _check_rows(result)[check]
    assert (outcome, failure_class) == ("fail", "assertion_failed")


async def test_a_seed_channel_whose_name_matches_twice_fails_the_lookup(
    workspace: Workspace,
) -> None:
    workspace.channels["C0SEED00002"] = Workspace._channel(
        "C0SEED00002", f"{CHANNEL_NAME}-copy", ""
    )

    result = await _run(_adapter())

    outcome, failure_class, _ = _check_rows(result)["slack.find_seed_channel"]
    assert (outcome, failure_class) == ("fail", "assertion_failed")
