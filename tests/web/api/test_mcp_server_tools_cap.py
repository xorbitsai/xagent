"""GET /api/mcp/servers/{id}/tools: the route's own concurrency cap (#3265).

Each call starts the server and lists its tools, so the route admits at most
``_MCP_SERVER_TOOLS_MAX_INFLIGHT`` loads per event loop and answers 503 to the
rest without starting anything. A slot is given back however the load ends.
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from typing import Any, Callable

import pytest
from fastapi import HTTPException

import xagent.web.api.mcp as mcp_module
from xagent.core.tools.adapters.vibe.mcp_adapter import (
    MCPFailurePhase,
    MCPLoadResult,
    MCPServerLoadFailure,
)
from xagent.web.api.mcp import get_mcp_server_tools
from xagent.web.services.mcp_connector_session import (
    ConnectorToolsOutcome,
    ConnectorToolsResult,
)

LISTER = "xagent.web.services.mcp_connector_session.list_connector_tools"
CAP = 2


async def _wait_until(predicate: Callable[[], bool], *, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError(f"condition not met within {timeout}s")
        await asyncio.sleep(0)


def _ok(server_id: int) -> ConnectorToolsResult:
    return ConnectorToolsResult(
        ConnectorToolsOutcome.OK,
        server_id,
        "srv",
        load_result=MCPLoadResult(
            tools=(SimpleNamespace(name="a", description="A"),),
            loaded_servers=("srv",),
            failures=(),
        ),
    )


class _Lister:
    """Stands in for list_connector_tools: counts starts and the peak number
    running at once, and holds each load until the test releases it."""

    def __init__(self) -> None:
        self.started = 0
        self.active = 0
        self.peak = 0
        self.gates: list[asyncio.Event] = []
        self.outcomes: list[Any] = []

    async def __call__(self, db, *, user, server_id, name_prefix):
        self.started += 1
        self.active += 1
        self.peak = max(self.peak, self.active)
        gate = asyncio.Event()
        self.gates.append(gate)
        try:
            await gate.wait()
            outcome = self.outcomes.pop(0) if self.outcomes else _ok(server_id)
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome
        finally:
            self.active -= 1


@pytest.fixture()
def lister(monkeypatch) -> _Lister:
    monkeypatch.setattr(mcp_module, "_MCP_SERVER_TOOLS_MAX_INFLIGHT", CAP)
    fake = _Lister()
    monkeypatch.setattr(LISTER, fake)
    return fake


def _call(server_id: int = 1) -> "asyncio.Task[dict]":
    user = SimpleNamespace(id=1, is_admin=False)
    return asyncio.ensure_future(get_mcp_server_tools(server_id, user, object()))


async def _call_over_cap(lister: _Lister, server_id: int) -> HTTPException:
    """Make one call while every slot is held and return its rejection.

    Fails at once, rather than hanging on the held fake, if the call started
    a load instead of being rejected.
    """
    before = lister.started
    task = _call(server_id)
    await _wait_until(lambda: task.done() or lister.started > before)
    assert lister.started == before, "a call beyond the cap started a load"
    error = task.exception()
    assert isinstance(error, HTTPException)
    return error


async def _drain(lister: _Lister, tasks: list) -> None:
    for gate in lister.gates:
        gate.set()
    await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.timeout(10)
async def test_calls_beyond_the_cap_get_503_without_starting_a_load(lister):
    tasks = [_call(server_id) for server_id in range(1, CAP + 1)]
    try:
        await _wait_until(lambda: lister.started == CAP)

        error = await _call_over_cap(lister, 99)

        assert error.status_code == 503
        assert error.detail["code"] == "mcp_tools_busy"
        assert lister.peak == CAP
    finally:
        await _drain(lister, tasks)

    responses = [task.result() for task in tasks]
    assert [response["tool_count"] for response in responses] == [1] * CAP


@pytest.mark.asyncio
@pytest.mark.timeout(10)
@pytest.mark.parametrize(
    "ending, expected_status",
    [
        ("finished", None),
        ("timed_out", 502),
        ("raised", 500),
        ("cancelled", None),
    ],
)
async def test_a_slot_is_given_back_however_the_load_ends(
    lister, ending, expected_status
):
    if ending == "timed_out":
        lister.outcomes.append(
            ConnectorToolsResult(
                ConnectorToolsOutcome.TIMED_OUT,
                1,
                "srv",
                reason="initialize",
                load_result=MCPLoadResult(
                    tools=(),
                    loaded_servers=(),
                    failures=(
                        MCPServerLoadFailure(
                            server_name="srv",
                            phase=MCPFailurePhase.INITIALIZE,
                            error_type="TimeoutError",
                        ),
                    ),
                ),
            )
        )
    elif ending == "raised":
        lister.outcomes.append(RuntimeError("boom"))

    tasks = [_call(server_id) for server_id in range(1, CAP + 1)]
    try:
        await _wait_until(lambda: lister.started == CAP)
        first = tasks[0]
        if ending == "cancelled":
            first.cancel()
        else:
            lister.gates[0].set()
        await asyncio.gather(first, return_exceptions=True)

        if ending == "cancelled":
            assert first.cancelled()
        elif expected_status is None:
            assert first.result()["tool_count"] == 1
        else:
            assert isinstance(first.exception(), HTTPException)
            assert first.exception().status_code == expected_status

        assert lister.active == CAP - 1
        tasks.append(_call(CAP + 1))
        await _wait_until(lambda: lister.started == CAP + 1)
        assert (await _call_over_cap(lister, 99)).status_code == 503
        assert lister.peak == CAP
    finally:
        await _drain(lister, tasks)


@pytest.mark.asyncio
@pytest.mark.timeout(10)
async def test_rejected_call_takes_no_slot(lister):
    tasks = [_call(server_id) for server_id in range(1, CAP + 1)]
    try:
        await _wait_until(lambda: lister.started == CAP)
        for _ in range(3):
            assert (await _call_over_cap(lister, 99)).status_code == 503

        lister.gates[0].set()
        await asyncio.gather(tasks[0], return_exceptions=True)
        tasks.append(_call(CAP + 1))
        await _wait_until(lambda: lister.started == CAP + 1)
        assert (await _call_over_cap(lister, 100)).status_code == 503
    finally:
        await _drain(lister, tasks)
