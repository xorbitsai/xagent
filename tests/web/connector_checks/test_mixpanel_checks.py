"""Smoke V2 check of Mixpanel (#3573).

The declaration is pinned in ``test_registry.py`` (``_TIER_2_READS``). Mixpanel's
Free plan refuses the Query API and answers the user-profile (Engage) call, so
the one check is Engage and the connector has left ``LEGACY_CONNECTORS``, whose
L3 probe is a Query API call. This runs the declared call against the real tool
with only the vendor request replaced, and judges what the tool returns with the
check's own assertions.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from xagent.web.connector_checks.assertions import judge_result
from xagent.web.connector_checks.declarations import ReadCheck
from xagent.web.connector_checks.registry import (
    _MIXPANEL_QUERY_API_NEEDS_A_PAID_PLAN,
    CONNECTOR_CHECKS,
    LEGACY_CONNECTORS,
)
from xagent.web.connector_checks.results import FailureClass
from xagent.web.tools.mcp import mixpanel

CHECK_ID = "mixpanel.query_engage"
# The three Query API tools the Free plan was seen to refuse.
REFUSED_ON_FREE = (
    "mixpanel_list_event_names",
    "mixpanel_get_top_events",
    "mixpanel_list_funnels",
)


def _check() -> ReadCheck:
    check = next(c for c in CONNECTOR_CHECKS["mixpanel"].reads if c.id == CHECK_ID)
    assert isinstance(check, ReadCheck)
    return check


def _judge(output: str) -> tuple[FailureClass | None, str | None]:
    check = _check()
    raw = {
        "content": [{"type": "text", "text": output}],
        "structured_content": None,
        "is_error": False,
    }
    return judge_result(raw, check.payload, check.assertions)


def _engage(results: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "page": 0,
        "page_size": 1000,
        "results": results,
        "session_id": "abc",
        "status": "ok",
        "total": len(results),
    }


def _run(monkeypatch: pytest.MonkeyPatch, answer: dict[str, Any]) -> tuple[str, list]:
    calls: list[tuple[Any, ...]] = []

    def request(method: str, host: str, path: str, **kwargs: Any) -> dict[str, Any]:
        calls.append((method, host, path, kwargs.get("form_data")))
        return answer

    monkeypatch.setattr(mixpanel, "_request", request)
    check = _check()
    return getattr(mixpanel, check.tool)(**check.arguments), calls


# --- what is declared ------------------------------------------------------------


def test_mixpanel_declares_one_read_and_no_l3_probe() -> None:
    checks = CONNECTOR_CHECKS["mixpanel"]

    assert [c.id for c in checks.reads] == [CHECK_ID]
    assert not checks.writes


def test_mixpanel_has_left_the_legacy_list() -> None:
    assert "mixpanel" not in LEGACY_CONNECTORS


def test_the_tools_the_free_plan_refuses_say_so() -> None:
    reasons = {e.tool: e.reason for e in CONNECTOR_CHECKS["mixpanel"].exemptions}

    for tool in REFUSED_ON_FREE:
        assert reasons[tool] == _MIXPANEL_QUERY_API_NEEDS_A_PAID_PLAN


# --- what a check accepts ----------------------------------------------------------


@pytest.mark.parametrize(
    "results", [[], [{"$distinct_id": "u1", "$properties": {"$name": "A"}}]]
)
def test_an_engage_output_satisfies_the_declaration(
    monkeypatch: pytest.MonkeyPatch, results: list[dict[str, Any]]
) -> None:
    output, calls = _run(monkeypatch, _engage(results))

    assert _judge(output) == (None, None)
    assert [(m, p) for m, _, p, _ in calls] == [("POST", "/api/query/engage")]


# --- what a check rejects ----------------------------------------------------------


def test_a_check_fails_on_a_tool_error_envelope() -> None:
    # The Free plan's refusal, a rejected credential or a missing project id: the
    # tool reports it in its own envelope with the MCP error flag unset.
    output = json.dumps(
        {"status": "error", "message": "402 Your plan does not allow API calls"}
    )

    assert _judge(output) == (FailureClass.ASSERTION_FAILED, "status_success")


def test_a_check_fails_when_the_profile_list_is_missing() -> None:
    output = json.dumps({"status": "success", "truncated": False})

    assert _judge(output) == (FailureClass.ASSERTION_FAILED, "has_keys")


@pytest.mark.parametrize(
    "engage",
    [
        {"page": 0, "page_size": 1000, "results": "none", "total": 0},
        {"page": 0, "page_size": 1000, "results": [], "total": "0"},
    ],
)
def test_a_check_fails_on_another_shape(engage: dict[str, Any]) -> None:
    output = json.dumps({"status": "success", "engage": engage, "truncated": False})

    failure, _ = _judge(output)

    assert failure is FailureClass.ASSERTION_FAILED
