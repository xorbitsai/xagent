"""The engine runs today's L3 probes with the probe route's verdicts.

Each case feeds the same listing and the same tool behaviour to the probe
route's own call path (``_call_declared_probe`` in
``services/connector_probe.py``) and to the engine running the read check
derived from the probe (``registry.l3_read_check``), and compares the two.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import pytest

from xagent.web.connector_checks.declarations import ConnectorChecks
from xagent.web.connector_checks.engine import run_checks
from xagent.web.connector_checks.identity import ConnectorTarget, SessionPurpose
from xagent.web.connector_checks.registry import (
    CONNECTOR_CHECKS,
    connector_checks_for,
    l3_read_check,
)
from xagent.web.connector_probes import (
    CONNECTOR_PROBES,
    UNPROBEABLE_CONNECTORS,
    ConnectorProbe,
)
from xagent.web.services import connector_probe
from xagent.web.services.connector_probe import (
    ConnectorProbeOutcome,
    ConnectorProbeResult,
    _call_declared_probe,
    _declared_probe,
)
from xagent.web.services.mcp_connector_session import (
    ConnectorToolsOutcome,
    ConnectorToolsResult,
)

from .fakes import FakeAdapter, FakeTool, failed, listed, rows

# A probe outcome and the class its V2 check row carries.
_V2_CLASS = {
    ConnectorProbeOutcome.OK: None,
    ConnectorProbeOutcome.PROBE_FAILED: "call_failed",
    ConnectorProbeOutcome.PROBE_ASSERTION_FAILED: "assertion_failed",
    ConnectorProbeOutcome.PROBE_TOOL_MISSING: "tool_missing",
    ConnectorProbeOutcome.TIMED_OUT: "timed_out",
}
# The one reason the two name differently: the call's own bound.
_V2_REASON = {"probe_call": "check_call"}

PROBED = sorted(CONNECTOR_PROBES)


def _text(*objects: Any) -> dict[str, Any]:
    return {
        "content": [
            {
                "type": "text",
                "text": item if isinstance(item, str) else json.dumps(item),
            }
            for item in objects
        ],
        "structured_content": None,
        "is_error": False,
    }


def _keys(probe: ConnectorProbe) -> dict[str, Any]:
    return {key: "value" for key in probe.expect_keys}


# Each case builds the tool the listing offers for the probe, or None for a
# listing without it.
_CASES: dict[str, Callable[[ConnectorProbe], FakeTool | None]] = {
    "success": lambda probe: FakeTool(probe.tool, _text({"status": "success"})),
    "success_with_keys": lambda probe: FakeTool(
        probe.tool, _text({"status": "success", **_keys(probe)})
    ),
    "keys_without_status": lambda probe: FakeTool(probe.tool, _text(_keys(probe))),
    "status_error": lambda probe: FakeTool(
        probe.tool, _text({"status": "error", "message": "token expired"})
    ),
    "flagged_error": lambda probe: FakeTool(
        probe.tool, {**_text({"status": "success"}), "is_error": True}
    ),
    "plain_text": lambda probe: FakeTool(probe.tool, _text("Error: unauthorized")),
    "deeply_nested": lambda probe: FakeTool(probe.tool, _text("[" * 100000)),
    "two_items": lambda probe: FakeTool(
        probe.tool, _text({"status": "success"}, {"status": "success"})
    ),
    "array": lambda probe: FakeTool(probe.tool, _text([{"status": "success"}])),
    "not_a_mapping": lambda probe: FakeTool(probe.tool, None),
    "raises": lambda probe: FakeTool(probe.tool, error=RuntimeError("boom")),
    "times_out": lambda probe: FakeTool(probe.tool, delay=1.0),
    "tool_missing": lambda probe: None,
}

_TIMEOUT = 0.02


def _derived(app_id: str) -> Callable[[str], ConnectorChecks | None]:
    check = l3_read_check(app_id)
    assert check is not None
    return lambda requested: ConnectorChecks(requested, reads=(check,))


def _result_factory(app_id: str) -> Callable[..., ConnectorProbeResult]:
    def result(
        outcome: ConnectorProbeOutcome, *, tool: str | None, reason: str | None
    ) -> ConnectorProbeResult:
        return ConnectorProbeResult(outcome, 1, app_id, tool, reason)

    return result


async def _probe_route(
    app_id: str, listing: ConnectorToolsResult
) -> ConnectorProbeResult:
    return await _call_declared_probe(
        listing,
        CONNECTOR_PROBES[app_id],
        app_id=app_id,
        result=_result_factory(app_id),
    )


@pytest.mark.parametrize("case", sorted(_CASES))
@pytest.mark.parametrize("app_id", PROBED)
async def test_a_derived_check_reaches_the_probe_routes_verdict(
    app_id: str, case: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        connector_probe, "get_mcp_tool_init_timeout_seconds", lambda: _TIMEOUT
    )
    # Each path gets its own tool: a tool records its calls.
    v1_tool, v2_tool = (_CASES[case](CONNECTOR_PROBES[app_id]) for _ in range(2))
    v1_listing = listed() if v1_tool is None else listed(v1_tool)
    v2_listing = listed() if v2_tool is None else listed(v2_tool)

    probe = await _probe_route(app_id, v1_listing)
    adapter = FakeAdapter([ConnectorTarget(app_id)], {app_id: v2_listing})
    # Both paths take their call bound from the same patched timeout.
    result = await run_checks(adapter, declarations=_derived(app_id))

    session, check = rows(result)
    assert session == (app_id, None, "pass", None, None)
    expected_reason = _V2_REASON.get(probe.reason or "", probe.reason)
    assert check == (
        app_id,
        f"{app_id}.l3_probe",
        "pass" if probe.outcome is ConnectorProbeOutcome.OK else "fail",
        _V2_CLASS[probe.outcome],
        expected_reason,
    )
    assert adapter.opened == [(app_id, SessionPurpose.CALL_TOOLS)]


@pytest.mark.parametrize(
    "outcome", [item for item in ConnectorToolsOutcome if item.value != "ok"]
)
async def test_a_failed_listing_is_the_session_rows_verdict(
    outcome: ConnectorToolsOutcome,
) -> None:
    listing = failed(outcome, "listing_reason")

    probe = await _probe_route("gmail", listing)
    adapter = FakeAdapter([ConnectorTarget("gmail")], {"gmail": listing})
    result = await run_checks(adapter, declarations=_derived("gmail"))

    assert rows(result) == [
        ("gmail", None, "fail", probe.outcome.value, "listing_reason"),
        ("gmail", "gmail.l3_probe", "skip", "session_unavailable", None),
    ]


@pytest.mark.parametrize("app_id", sorted(UNPROBEABLE_CONNECTORS))
@pytest.mark.parametrize("listing_ok", [True, False])
async def test_an_unprobeable_connector_lists_and_records_its_reason(
    app_id: str, listing_ok: bool
) -> None:
    listing = (
        listed(FakeTool("any_tool"))
        if listing_ok
        else failed(ConnectorToolsOutcome.FAILED_TO_START, "initialize")
    )
    reason = UNPROBEABLE_CONNECTORS[app_id]
    probe = _declared_probe(app_id, _result_factory(app_id))
    assert isinstance(probe, ConnectorProbeResult)
    assert (probe.outcome, probe.reason) == (ConnectorProbeOutcome.UNPROBEABLE, reason)

    adapter = FakeAdapter([ConnectorTarget(app_id)], {app_id: listing})
    result = await run_checks(adapter, declarations=_derived(app_id))

    session_verdict = (
        ("pass", None, None)
        if listing_ok
        else ("fail", "failed_to_start", "initialize")
    )
    assert rows(result) == [
        (app_id, None, *session_verdict),
        (app_id, f"{app_id}.l3_probe", "unprobeable", None, reason),
    ]
    assert adapter.opened == [(app_id, SessionPurpose.LIST_ONLY)]


async def test_an_app_without_a_probe_lists_and_is_not_declared() -> None:
    probe = _declared_probe("acme", _result_factory("acme"))
    assert isinstance(probe, ConnectorProbeResult)
    assert probe.outcome is ConnectorProbeOutcome.NOT_DECLARED

    adapter = FakeAdapter([ConnectorTarget("acme")], {"acme": listed()})
    result = await run_checks(adapter)

    assert rows(result) == [
        ("acme", None, "pass", None, None),
        ("acme", "acme.l3_probe", "skip", "not_declared", None),
    ]
    assert adapter.opened == [("acme", SessionPurpose.LIST_ONLY)]


@pytest.mark.parametrize(
    "app_id", [app_id for app_id in PROBED if app_id not in CONNECTOR_CHECKS]
)
def test_the_default_declarations_run_each_probe_as_its_l3_check(app_id: str) -> None:
    check = l3_read_check(app_id)
    assert check is not None

    assert connector_checks_for(app_id) == ConnectorChecks(app_id, reads=(check,))


async def test_the_engine_runs_the_registry_by_default() -> None:
    tool = FakeTool("slack_list_channels")
    adapter = FakeAdapter([ConnectorTarget("slack")], {"slack": listed(tool)})

    result = await run_checks(adapter)

    assert rows(result)[1] == ("slack", "slack.l3_probe", "pass", None, None)
    assert tool.calls == [dict(CONNECTOR_PROBES["slack"].arguments)]
