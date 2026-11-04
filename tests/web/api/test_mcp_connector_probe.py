"""POST /api/mcp/servers/{id}/probe: the L3 probe route.

The route takes only the server id, gates on the probe allowlist, and maps
the service's outcome onto the same status and code shapes as
GET /servers/{id}/tools. The service itself is covered in
tests/web/services/test_connector_probe.py.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from xagent.web.api.mcp import probe_mcp_server
from xagent.web.services.connector_probe import (
    ConnectorProbeBusy,
    ConnectorProbeOutcome,
    ConnectorProbeResult,
)

from .conftest import _admin_headers, client

RUNNER = "xagent.web.services.connector_probe.run_connector_probe"


def _user(username: str = "monitor") -> SimpleNamespace:
    return SimpleNamespace(id=1, username=username, email=None, is_admin=False)


@pytest.fixture()
def allow_monitor(monkeypatch):
    monkeypatch.setenv("XAGENT_CONNECTOR_PROBE_USERS", "monitor")


def _runner(result: ConnectorProbeResult | BaseException):
    calls: list[dict] = []

    async def run(db, *, user, server_id):
        calls.append({"db": db, "user": user, "server_id": server_id})
        if isinstance(result, BaseException):
            raise result
        return result

    run.calls = calls  # type: ignore[attr-defined]
    return run


def test_route_accepts_no_tool_or_arguments():
    """The declaration in code is the only source of what gets called."""
    parameters = set(inspect.signature(probe_mcp_server).parameters)
    assert parameters == {"server_id", "current_user", "db"}


@pytest.mark.asyncio
async def test_rejects_callers_outside_the_allowlist(monkeypatch):
    monkeypatch.setenv("XAGENT_CONNECTOR_PROBE_USERS", "monitor")
    runner = _runner(ConnectorProbeResult(ConnectorProbeOutcome.OK, 7, "github"))
    monkeypatch.setattr(RUNNER, runner)
    admin = SimpleNamespace(id=2, username="admin", email=None, is_admin=True)

    with pytest.raises(HTTPException) as exc_info:
        await probe_mcp_server(7, admin, db=object())

    assert exc_info.value.status_code == 403
    assert exc_info.value.detail["code"] == "probe_forbidden"
    assert runner.calls == []


@pytest.mark.asyncio
async def test_rejects_everyone_when_unset(monkeypatch):
    monkeypatch.delenv("XAGENT_CONNECTOR_PROBE_USERS", raising=False)
    runner = _runner(ConnectorProbeResult(ConnectorProbeOutcome.OK, 7, "github"))
    monkeypatch.setattr(RUNNER, runner)

    with pytest.raises(HTTPException) as exc_info:
        await probe_mcp_server(7, _user(), db=object())

    assert exc_info.value.status_code == 403
    assert runner.calls == []


@pytest.mark.asyncio
async def test_ok_answers_the_payload(allow_monitor, monkeypatch):
    runner = _runner(
        ConnectorProbeResult(
            ConnectorProbeOutcome.OK, 7, "github", "github_get_current_user"
        )
    )
    monkeypatch.setattr(RUNNER, runner)
    db = object()
    user = _user()

    body = await probe_mcp_server(7, user, db=db)

    assert body == {
        "code": "ok",
        "outcome": "ok",
        "server_id": 7,
        "app_id": "github",
        "tool": "github_get_current_user",
    }
    assert runner.calls == [{"db": db, "user": user, "server_id": 7}]


@pytest.mark.parametrize(
    ("result", "status", "code"),
    [
        (
            ConnectorProbeResult(ConnectorProbeOutcome.NOT_FOUND, 7),
            404,
            "mcp_server_not_found",
        ),
        (
            ConnectorProbeResult(ConnectorProbeOutcome.NOT_DECLARED, 7, "custom"),
            404,
            "probe_not_declared",
        ),
        (
            ConnectorProbeResult(
                ConnectorProbeOutcome.UNPROBEABLE, 7, "notion", None, "covered_by_l2"
            ),
            400,
            "unprobeable",
        ),
        (
            ConnectorProbeResult(
                ConnectorProbeOutcome.NOT_CONNECTED,
                7,
                "github",
                "github_get_current_user",
                "oauth_token_required",
            ),
            400,
            "oauth_token_required",
        ),
        (
            ConnectorProbeResult(
                ConnectorProbeOutcome.UNSUPPORTED, 7, "stripe", "t", "sandboxed"
            ),
            400,
            "unsupported",
        ),
        (
            ConnectorProbeResult(
                ConnectorProbeOutcome.FAILED_TO_START, 7, "github", "t", "initialize"
            ),
            502,
            "mcp_tools_unavailable",
        ),
        (
            ConnectorProbeResult(
                ConnectorProbeOutcome.TIMED_OUT, 7, "github", "t", "probe_call"
            ),
            502,
            "mcp_tools_unavailable",
        ),
        (
            ConnectorProbeResult(
                ConnectorProbeOutcome.PROBE_TOOL_MISSING, 7, "github", "t"
            ),
            502,
            "probe_tool_missing",
        ),
        (
            ConnectorProbeResult(
                ConnectorProbeOutcome.PROBE_FAILED, 7, "github", "t", "tool_error"
            ),
            502,
            "probe_failed",
        ),
        (
            ConnectorProbeResult(
                ConnectorProbeOutcome.PROBE_ASSERTION_FAILED,
                7,
                "github",
                "t",
                "status_not_success",
            ),
            502,
            "probe_assertion_failed",
        ),
    ],
)
@pytest.mark.asyncio
async def test_failures_answer_with_the_outcome_code(
    allow_monitor, monkeypatch, result, status, code
):
    monkeypatch.setattr(RUNNER, _runner(result))

    with pytest.raises(HTTPException) as exc_info:
        await probe_mcp_server(7, _user(), db=object())

    assert exc_info.value.status_code == status
    detail = exc_info.value.detail
    assert detail["code"] == code
    assert detail["outcome"] == result.outcome.value
    assert detail["server_id"] == 7
    if result.reason is not None and result.reason != code:
        assert detail["reason"] == result.reason


@pytest.mark.asyncio
async def test_busy_answers_429(allow_monitor, monkeypatch):
    monkeypatch.setattr(RUNNER, _runner(ConnectorProbeBusy()))

    with pytest.raises(HTTPException) as exc_info:
        await probe_mcp_server(7, _user(), db=object())

    assert exc_info.value.status_code == 429
    assert exc_info.value.detail["code"] == "probe_busy"


@pytest.mark.asyncio
async def test_unexpected_errors_are_opaque(allow_monitor, monkeypatch):
    monkeypatch.setattr(RUNNER, _runner(RuntimeError("token ghp_secret leaked")))

    with pytest.raises(HTTPException) as exc_info:
        await probe_mcp_server(7, _user(), db=object())

    assert exc_info.value.status_code == 500
    assert "ghp_secret" not in str(exc_info.value.detail)


@pytest.mark.usefixtures("_test_db")
def test_route_is_a_post_gated_on_the_probe_allowlist(monkeypatch):
    """The method, the path and the dependency wiring, through the real app."""
    monkeypatch.delenv("XAGENT_CONNECTOR_PROBE_USERS", raising=False)
    headers = _admin_headers()

    posted = client.post("/api/mcp/servers/1/probe", headers=headers)
    assert posted.status_code == 403, posted.text
    assert posted.json()["detail"]["code"] == "probe_forbidden"

    assert client.get("/api/mcp/servers/1/probe", headers=headers).status_code == 405
