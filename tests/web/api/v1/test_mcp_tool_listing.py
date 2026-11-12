"""Integration tests for listing a connector's tools through /v1."""

import asyncio
import gc
from types import SimpleNamespace
from typing import Any

import pytest
import sqlalchemy as sa
from fastapi import HTTPException
from sqlalchemy.exc import OperationalError

from xagent.core.tools.adapters.vibe.mcp_adapter import (
    MCPFailurePhase,
    MCPLoadResult,
    MCPServerLoadFailure,
)
from xagent.web.api import mcp as web_mcp
from xagent.web.api.v1 import mcp_servers as routes
from xagent.web.models.database import get_engine
from xagent.web.services import connector_team_scope, mcp_connector_session
from xagent.web.services.mcp_connector_session import (
    _CREDENTIAL_REASONS,
    ConnectorToolsOutcome,
    ConnectorToolsResult,
)

from ..conftest import client
from .test_mcp_servers import (
    CONNECTOR_URL,
    Holder,
    _error,
    _holder,
    _link,
    _seed,
    _share_with_team,
)

pytestmark = pytest.mark.usefixtures("_test_db")

LOADER = "xagent.core.tools.adapters.vibe.mcp_adapter.load_mcp_tools_as_agent_tools"
SECRET = "SECRET-diagnostic-text"


@pytest.fixture(autouse=True)
def _no_team_hooks():
    connector_team_scope.set_connector_team_hooks()
    yield
    connector_team_scope.set_connector_team_hooks()


@pytest.fixture
def bob() -> Holder:
    return _holder("bob")


@pytest.fixture
def carol() -> Holder:
    return _holder("carol")


def _tools(holder: Holder, server_id: int):
    return client.get(f"/v1/mcp-servers/{server_id}/tools", headers=holder.headers)


def _tool(name: str = "search", description: str = "Search things") -> Any:
    return SimpleNamespace(name=name, description=description)


def _loaded(*tools: Any, failures: tuple[Any, ...] = ()) -> MCPLoadResult:
    return MCPLoadResult(
        tools=tuple(tools), loaded_servers=("x",) if tools else (), failures=failures
    )


def _failure(phase: MCPFailurePhase, attempts: int = 1) -> MCPServerLoadFailure:
    return MCPServerLoadFailure(
        server_name="x", phase=phase, error_type="SecretLeakingError", attempts=attempts
    )


def _stub_loader(monkeypatch, result: Any = None, error: Exception | None = None):
    """Replace the network boundary of the tool loader; return its calls."""
    calls: list[dict[str, Any]] = []

    async def load(connections, **kwargs):
        calls.append({"connections": connections, **kwargs})
        if error is not None:
            raise error
        return result

    monkeypatch.setattr(LOADER, load)
    return calls


def _stub_session_listing(monkeypatch, result: ConnectorToolsResult):
    calls: list[int] = []

    async def listing(db, **kwargs):
        calls.append(kwargs["server_id"])
        return result

    monkeypatch.setattr(mcp_connector_session, "list_connector_tools", listing)
    return calls


def _stub_web_function(
    monkeypatch, *, result: Any = None, error: Exception | None = None
):
    calls: list[dict[str, Any]] = []

    async def web_tools(server_id, current_user, db):
        calls.append({"server_id": server_id, "user_id": int(current_user.id)})
        if error is not None:
            raise error
        return result

    monkeypatch.setattr(web_mcp, "get_mcp_server_tools", web_tools)
    return calls


WEB_OK = {
    "server_name": "n",
    "tool_count": 1,
    "tools": [{"name": "server_1_search", "description": "Search"}],
}


# ===== success =====


def test_listing_returns_the_tools_a_task_would_get(bob, monkeypatch):
    server_id = _seed("records", owner=bob)
    calls = _stub_loader(monkeypatch, _loaded(_tool("a"), _tool("b", "Second")))

    resp = _tools(bob, server_id)

    assert resp.status_code == 200, resp.text
    assert resp.json() == {
        "id": server_id,
        "tools": [
            {"name": "a", "description": "Search things"},
            {"name": "b", "description": "Second"},
        ],
        "failures": [],
    }
    assert len(calls) == 1
    assert calls[0]["name_prefix"] == f"server_{server_id}_"
    assert calls[0]["connections"]["records"]["url"] == CONNECTOR_URL


def test_partial_failures_are_listed_without_diagnostics(bob, monkeypatch):
    server_id = _seed("records", owner=bob)
    _stub_loader(
        monkeypatch,
        _loaded(_tool(), failures=(_failure(MCPFailurePhase.LIST_TOOLS, 2),)),
    )

    body = _tools(bob, server_id).json()

    assert body["failures"] == [{"phase": "list_tools", "attempts": 2}]
    assert "SecretLeakingError" not in str(body)
    assert "server_name" not in str(body)


# ===== which connectors may be listed =====


def _own_disabled(bob, carol):
    return _seed("off", owner=bob, is_active=False), 409, "disabled"


def _team_shared(bob, carol):
    server_id = _seed("shared", owner=carol)
    _share_with_team(server_id)
    return server_id, 409, "team_shared"


@pytest.mark.parametrize("cell", ["own_disabled", "team_shared"])
def test_precheck_refuses_before_any_listing(bob, carol, monkeypatch, cell):
    server_id, status, reason = {
        "own_disabled": _own_disabled,
        "team_shared": _team_shared,
    }[cell](bob, carol)
    web_calls = _stub_web_function(monkeypatch, result=WEB_OK)

    error = _error(_tools(bob, server_id), status, "connector_unavailable")

    assert error["details"] == {"reason": reason}
    assert error["message"] == (
        "This connector cannot be started from the API right now; "
        "details.reason says why."
    )
    assert web_calls == []


@pytest.mark.parametrize("cell", ["someone_elses", "missing"])
def test_precheck_hides_connectors_the_holder_cannot_see(bob, carol, monkeypatch, cell):
    server_id = _seed("carols", owner=carol) if cell == "someone_elses" else 987654
    web_calls = _stub_web_function(monkeypatch, result=WEB_OK)

    error = _error(_tools(bob, server_id), 404, "connector_not_found")

    assert "details" not in error
    assert web_calls == []


@pytest.mark.parametrize(
    ("transport", "url"),
    [
        ("stdio", None),
        ("websocket", "wss://mcp.example.test/ws"),
        ("oauth", None),
    ],
)
def test_only_remote_transports_can_be_listed(bob, monkeypatch, transport, url):
    server_id = _seed("local", owner=bob, transport=transport, url=url)
    web_calls = _stub_web_function(monkeypatch, result=WEB_OK)

    error = _error(_tools(bob, server_id), 400, "transport_not_supported")

    assert error["message"] == (
        "Listing tools through the API supports only streamable_http and sse "
        "connectors."
    )
    assert web_calls == []


def test_a_team_shared_local_connector_is_refused_for_its_transport(
    bob, carol, monkeypatch
):
    server_id = _seed("shared-local", owner=carol, transport="stdio", url=None)
    _share_with_team(server_id)
    web_calls = _stub_web_function(monkeypatch, result=WEB_OK)

    _error(_tools(bob, server_id), 400, "transport_not_supported")

    assert web_calls == []


def test_a_failing_visibility_hook_is_a_plain_500(bob, carol, monkeypatch):
    server_id = _seed("carols", owner=carol)
    web_calls = _stub_web_function(monkeypatch, result=WEB_OK)

    def failing(_db, _user_id):
        raise RuntimeError(SECRET)

    connector_team_scope.set_connector_team_hooks(visibility=failing)

    resp = _tools(bob, server_id)

    _error(resp, 500, "internal_error")
    assert SECRET not in resp.text and web_calls == []


def test_a_failing_precheck_query_is_a_plain_500(bob, monkeypatch):
    server_id = _seed("records", owner=bob)
    web_calls = _stub_web_function(monkeypatch, result=WEB_OK)
    engine = get_engine()

    def fail(conn, cursor, statement, parameters, context, executemany):
        if "user_mcpservers" in statement:
            raise OperationalError(statement, parameters, Exception(SECRET))

    sa.event.listen(engine, "before_cursor_execute", fail)
    try:
        resp = _tools(bob, server_id)
    finally:
        sa.event.remove(engine, "before_cursor_execute", fail)

    _error(resp, 500, "internal_error")
    assert SECRET not in resp.text and web_calls == []


def test_a_connector_linked_by_the_holder_and_shared_is_listed_as_own(
    bob, carol, monkeypatch
):
    server_id = _seed("records", owner=carol)
    _link(bob, server_id, is_owner=False)
    _share_with_team(server_id)
    _stub_loader(monkeypatch, _loaded(_tool()))

    assert _tools(bob, server_id).status_code == 200


# ===== translating the web function's failures =====


def _web_tools_error(monkeypatch, bob, error: HTTPException):
    server_id = _seed("records", owner=bob)
    _stub_web_function(monkeypatch, error=error)
    return _tools(bob, server_id)


def _credential_row(reason: str):
    return ConnectorToolsResult(
        ConnectorToolsOutcome.NOT_CONNECTED, 1, "n", reason=reason
    )


@pytest.mark.parametrize("reason", sorted(_CREDENTIAL_REASONS))
def test_credential_failures_are_authorization_failures(bob, monkeypatch, reason):
    assert len(_CREDENTIAL_REASONS) == 6
    server_id = _seed("records", owner=bob)
    _stub_session_listing(monkeypatch, _credential_row(reason))

    error = _error(_tools(bob, server_id), 409, "mcp_oauth_authorization_failed")

    assert error["details"] == {"reason": reason}
    assert error["message"] == (
        "Authorization can only be completed in the xagent web app, not with an "
        "API key. Open Connectors and select this connector; if it offers "
        "Connect, choose it, otherwise ask the connector's owner."
    )


_OTHER_400 = {
    "unsupported": (
        ConnectorToolsResult(ConnectorToolsOutcome.UNSUPPORTED, 1, "n", reason="x"),
        "unsupported",
    ),
    "invalid_configuration": (
        ConnectorToolsResult(
            ConnectorToolsOutcome.FAILED_TO_START,
            1,
            "n",
            reason="invalid_configuration",
            diagnostic={"code": "invalid_configuration", "message": SECRET},
        ),
        "invalid_configuration",
    ),
    "code_that_is_not_a_plain_word": (
        ConnectorToolsResult(
            ConnectorToolsOutcome.FAILED_TO_START,
            1,
            "n",
            reason="x",
            diagnostic={"code": "Bad Code!" + SECRET, "message": SECRET},
        ),
        "unknown",
    ),
    "code_that_is_not_a_string": (
        ConnectorToolsResult(
            ConnectorToolsOutcome.FAILED_TO_START,
            1,
            "n",
            reason="x",
            diagnostic={"code": 7, "message": SECRET},
        ),
        "unknown",
    ),
    "platform_client_rejected_for_a_provider": (
        ConnectorToolsResult(
            ConnectorToolsOutcome.NOT_CONNECTED,
            1,
            "n",
            reason="client_rejected:google",
            diagnostic={"code": "client_rejected:google", "message": SECRET},
        ),
        "client_rejected",
    ),
}


@pytest.mark.parametrize("cell", list(_OTHER_400))
def test_other_rejections_are_connector_unavailable(bob, monkeypatch, cell):
    result, reason = _OTHER_400[cell]
    server_id = _seed("records", owner=bob)
    _stub_session_listing(monkeypatch, result)

    resp = _tools(bob, server_id)

    error = _error(resp, 409, "connector_unavailable")
    assert error["details"] == {"reason": reason}
    assert SECRET not in resp.text


def test_a_non_dict_rejection_is_connector_unavailable(bob, monkeypatch):
    resp = _web_tools_error(monkeypatch, bob, HTTPException(400, SECRET))

    error = _error(resp, 409, "connector_unavailable")
    assert error["details"] == {"reason": "unknown"}
    assert SECRET not in resp.text


_BAD_GATEWAY = {
    "failure_reason_from_the_start": (
        ConnectorToolsResult(
            ConnectorToolsOutcome.FAILED_TO_START, 1, "n", reason="config_load_failed"
        ),
        "config_load_failed",
    ),
    "failure_reason_that_is_not_a_plain_word": (
        ConnectorToolsResult(
            ConnectorToolsOutcome.FAILED_TO_START, 1, "n", reason="Bad " + SECRET
        ),
        "no_tools",
    ),
    "failure_reason_with_a_trailing_line_break": (
        ConnectorToolsResult(
            ConnectorToolsOutcome.FAILED_TO_START, 1, "n", reason="config_load_failed\n"
        ),
        "no_tools",
    ),
}


@pytest.mark.parametrize("cell", list(_BAD_GATEWAY))
def test_a_connector_that_cannot_start_is_a_bad_gateway(bob, monkeypatch, cell):
    result, reason = _BAD_GATEWAY[cell]
    server_id = _seed("records", owner=bob)
    _stub_session_listing(monkeypatch, result)

    resp = _tools(bob, server_id)

    error = _error(resp, 502, "connector_tools_unavailable")
    assert error["details"] == {"reason": reason, "failures": []}
    assert (
        error["message"] == "The connector could not be started or returned no tools."
    )
    assert SECRET not in resp.text


def test_a_connector_that_returns_no_tools_is_a_bad_gateway(bob, monkeypatch):
    server_id = _seed("records", owner=bob)
    _stub_loader(monkeypatch, _loaded(failures=(_failure(MCPFailurePhase.LIST_TOOLS),)))

    resp = _tools(bob, server_id)

    error = _error(resp, 502, "connector_tools_unavailable")
    assert error["details"] == {
        "reason": "no_tools",
        "failures": [{"phase": "list_tools", "attempts": 1}],
    }
    assert "SecretLeakingError" not in resp.text


@pytest.mark.parametrize(
    "error",
    [
        HTTPException(502, SECRET),
        HTTPException(500, SECRET),
        HTTPException(503, {"code": "x", "message": SECRET}),
    ],
    ids=["bad_gateway_text", "internal_text", "unexpected_status"],
)
def test_other_web_failures_are_plain_500s(bob, monkeypatch, error):
    resp = _web_tools_error(monkeypatch, bob, error)

    _error(resp, 500, "internal_error")
    assert SECRET not in resp.text


def test_a_web_404_after_the_precheck_is_not_found(bob, monkeypatch):
    resp = _web_tools_error(
        monkeypatch, bob, HTTPException(404, "MCP server not found")
    )

    _error(resp, 404, "connector_not_found")


def test_a_loader_that_raises_is_a_plain_500(bob, monkeypatch):
    server_id = _seed("records", owner=bob)
    _stub_loader(monkeypatch, error=RuntimeError(SECRET))

    resp = _tools(bob, server_id)

    _error(resp, 500, "internal_error")
    assert SECRET not in resp.text


def test_an_oauth_connector_the_holder_has_not_authorized(bob, monkeypatch):
    server_id = _seed(
        "oauth-records",
        owner=bob,
        auth={
            "type": "mcp_oauth",
            "resource": CONNECTOR_URL,
            "issuer": "https://auth.example.test",
            "client_id": "client-1",
            "scope": "records.read",
        },
    )
    calls = _stub_loader(monkeypatch, _loaded(_tool()))

    resp = _tools(bob, server_id)

    _error(resp, 409, "mcp_oauth_authorization_failed")
    assert calls == []


# ===== the listing gate =====

BUSY = "Too many connector tool listings are running; retry shortly."


def _use_timeout(monkeypatch, seconds: float) -> None:
    monkeypatch.setattr(routes, "get_mcp_tool_init_timeout_seconds", lambda: seconds)


def _recording_gates(monkeypatch) -> list[asyncio.Semaphore]:
    real = routes._v1_tool_listing_gate
    seen: list[asyncio.Semaphore] = []

    def gate():
        seen.append(real())
        return seen[-1]

    monkeypatch.setattr(routes, "_v1_tool_listing_gate", gate)
    return seen


def _gate_cell_web_gate_full(bob, monkeypatch):
    monkeypatch.setattr(
        web_mcp, "_mcp_connection_test_gate", lambda: asyncio.Semaphore(0)
    )
    _use_timeout(monkeypatch, 0.05)
    calls = _stub_web_function(monkeypatch, result=WEB_OK)
    server_id = _seed("records", owner=bob)

    assert _tools(bob, server_id).status_code == 200
    assert len(calls) == 1


def _gate_cell_own_gate_full(bob, monkeypatch):
    monkeypatch.setattr(routes, "_v1_tool_listing_gate", lambda: asyncio.Semaphore(0))
    _use_timeout(monkeypatch, 0.05)
    calls = _stub_web_function(monkeypatch, result=WEB_OK)
    server_id = _seed("records", owner=bob)

    error = _error(_tools(bob, server_id), 429, "rate_limited")

    assert error["message"] == BUSY
    assert calls == []


def _gate_cell_normal_call(bob, monkeypatch):
    gates = _recording_gates(monkeypatch)
    _stub_web_function(monkeypatch, result=WEB_OK)
    server_id = _seed("records", owner=bob)

    assert _tools(bob, server_id).status_code == 200

    assert [g._value for g in gates] == [4]
    assert routes._v1_tool_listing_loads == set()


def _gate_cell_web_function_error(bob, monkeypatch):
    gates = _recording_gates(monkeypatch)
    _stub_web_function(monkeypatch, error=HTTPException(404, "MCP server not found"))
    server_id = _seed("records", owner=bob)

    _error(_tools(bob, server_id), 404, "connector_not_found")

    assert [g._value for g in gates] == [4]
    assert routes._v1_tool_listing_loads == set()


async def _cancel_a_listing(monkeypatch, timeout: float) -> dict[str, Any]:
    """Start a listing that blocks, cancel its request, report what happened."""
    _use_timeout(monkeypatch, timeout)
    entered, release = asyncio.Event(), asyncio.Event()
    state: dict[str, Any] = {"load_cancelled": False}

    async def blocked(server_id, user_id):
        entered.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            state["load_cancelled"] = True
            raise
        return WEB_OK

    monkeypatch.setattr(routes, "_list_saved_connector_tools", blocked)
    request = asyncio.create_task(routes._run_tool_listing(1, 1))
    await asyncio.wait_for(entered.wait(), 5)
    gate = routes._v1_tool_listing_gate()
    request.cancel()
    with pytest.raises(asyncio.CancelledError):
        await request
    await asyncio.sleep(0)
    state["gate"] = gate
    state["value_after_cancel"] = gate._value
    state["loads_after_cancel"] = set(routes._v1_tool_listing_loads)
    release.set()
    if state["loads_after_cancel"]:
        await asyncio.wait_for(asyncio.wait(state["loads_after_cancel"]), 5)
    await asyncio.sleep(0)
    state["value_at_end"] = gate._value
    state["loads_at_end"] = set(routes._v1_tool_listing_loads)
    return state


def _gate_cell_cancelled_with_a_deadline(bob, monkeypatch):
    state = asyncio.run(_cancel_a_listing(monkeypatch, 5))

    assert state["value_after_cancel"] == 3
    assert state["load_cancelled"] is False
    assert len(state["loads_after_cancel"]) == 1
    assert state["value_at_end"] == 4
    assert state["loads_at_end"] == set()


def _gate_cell_cancelled_without_a_deadline(bob, monkeypatch):
    state = asyncio.run(_cancel_a_listing(monkeypatch, 0))

    assert state["load_cancelled"] is True
    assert state["value_after_cancel"] == 4
    assert state["loads_at_end"] == set()


async def _abandon_a_failing_listing(monkeypatch) -> dict[str, Any]:
    """Cancel a request whose background listing then fails; report the loop."""
    _use_timeout(monkeypatch, 5)
    entered, release = asyncio.Event(), asyncio.Event()
    unretrieved: list[str] = []
    asyncio.get_running_loop().set_exception_handler(
        lambda _loop, context: unretrieved.append(context["message"])
    )

    async def failing(server_id, user_id):
        entered.set()
        await release.wait()
        raise RuntimeError(SECRET)

    monkeypatch.setattr(routes, "_list_saved_connector_tools", failing)
    request = asyncio.create_task(routes._run_tool_listing(1, 1))
    await asyncio.wait_for(entered.wait(), 5)
    gate = routes._v1_tool_listing_gate()
    request.cancel()
    with pytest.raises(asyncio.CancelledError):
        await request
    loads = set(routes._v1_tool_listing_loads)
    release.set()
    await asyncio.wait_for(asyncio.wait(loads), 5)
    await asyncio.sleep(0)
    del loads, request
    gc.collect()
    await asyncio.sleep(0)
    return {"unretrieved": unretrieved, "gate_value": gate._value}


def test_an_abandoned_listing_that_fails_has_its_error_retrieved(monkeypatch):
    state = asyncio.run(_abandon_a_failing_listing(monkeypatch))

    assert state["unretrieved"] == []
    assert state["gate_value"] == 4
    assert routes._v1_tool_listing_loads == set()


def _gate_cell_one_gate_per_loop(bob, monkeypatch):
    async def gate():
        return routes._v1_tool_listing_gate()

    assert asyncio.run(gate()) is not asyncio.run(gate())


_GATE_CELLS = {
    name.removeprefix("_gate_cell_"): fn
    for name, fn in sorted(globals().items())
    if name.startswith("_gate_cell_")
}


@pytest.mark.parametrize("cell", list(_GATE_CELLS))
def test_tool_listing_has_its_own_gate(bob, monkeypatch, cell):
    assert len(_GATE_CELLS) == 7
    _GATE_CELLS[cell](bob, monkeypatch)
