"""``find_app_connection``: resolver-driven connect card and server-side resume."""

from __future__ import annotations

from typing import Any

import pytest

from xagent.core.tools import connector_resolver
from xagent.core.tools.adapters.vibe.find_app_connection_tool import (
    FindAppConnectionTool,
    create_find_app_connection_tool,
)
from xagent.core.tools.adapters.vibe.interaction_types import (
    INTERACTION_TYPE_ALIASES,
    INTERACTION_TYPES,
)
from xagent.core.tools.connector_resolver import (
    ConnectorConnectionState,
    ConnectorResolution,
    ConnectorResolveRequest,
)
from xagent.core.tools.user_interaction import (
    ToolInteractionSettlement,
    tool_result_waits_for_user,
    user_interaction_resume_callable,
)


class FakeResolver:
    def __init__(
        self,
        resolution: ConnectorResolution | None = None,
        state: ConnectorConnectionState | None = None,
    ) -> None:
        self.resolution = resolution
        self.state = state
        self.requests: list[ConnectorResolveRequest] = []
        self.state_calls: list[dict[str, Any]] = []

    async def resolve(self, request: ConnectorResolveRequest) -> ConnectorResolution:
        self.requests.append(request)
        assert self.resolution is not None
        return self.resolution

    async def connection_state(
        self, *, interaction_id: str, user_id: int | None, task_id: str | None
    ) -> ConnectorConnectionState | None:
        self.state_calls.append(
            {"interaction_id": interaction_id, "user_id": user_id, "task_id": task_id}
        )
        return self.state


def _tool(resolver: FakeResolver) -> FindAppConnectionTool:
    return FindAppConnectionTool(resolver=resolver, user_id=7, task_id="t-1")


ARGS = {"app_name": "Cosmos", "purpose": "Count orders"}
CONNECTOR = {"name": "Azure Cosmos DB", "fields": [{"label": "Endpoint"}]}


@pytest.mark.asyncio
async def test_needs_connection_returns_a_tool_authored_waiting_result() -> None:
    resolver = FakeResolver(
        ConnectorResolution("needs_connection", connector=CONNECTOR)
    )
    result = await _tool(resolver).run_json_async(ARGS)

    assert tool_result_waits_for_user(result)
    [interaction] = result["interactions"]
    assert interaction["type"] == "connect_connector"
    assert interaction["connector"] == CONNECTOR
    assert interaction["live_widget"] is True
    # The id the resolver was given is the id the user is waiting on.
    assert result["interaction_id"] == resolver.requests[0].interaction_id
    assert resolver.requests[0].user_id == 7
    assert resolver.requests[0].purpose == "Count orders"


def test_connect_connector_is_not_model_facing() -> None:
    assert "connect_connector" not in INTERACTION_TYPES
    assert "connect_connector" not in INTERACTION_TYPE_ALIASES


@pytest.mark.asyncio
async def test_already_connected_succeeds_without_waiting() -> None:
    resolver = FakeResolver(ConnectorResolution("connected", result={"server": "db"}))
    result = await _tool(resolver).run_json_async(ARGS)

    assert not tool_result_waits_for_user(result)
    assert result["success"] is True
    assert result["status"] == "connected"
    assert result["server"] == "db"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "resolution",
    [
        ConnectorResolution("not_found"),
        # A needs_connection with nothing to render must not strand the run.
        ConnectorResolution("needs_connection", connector=None),
    ],
)
async def test_not_found_is_a_plain_result(resolution: ConnectorResolution) -> None:
    result = await _tool(FakeResolver(resolution)).run_json_async(ARGS)

    assert not tool_result_waits_for_user(result)
    assert result["status"] == "not_found"
    assert "not available" in result["message"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("state", "expected"),
    [
        (
            ConnectorConnectionState("connected", result={"server": "db"}),
            {"success": True, "status": "connected", "server": "db"},
        ),
        (ConnectorConnectionState("skipped"), {"status": "skipped"}),
    ],
)
async def test_resume_settles_from_server_state(
    state: ConnectorConnectionState, expected: dict[str, Any]
) -> None:
    resolver = FakeResolver(state=state)
    resume = user_interaction_resume_callable(_tool(resolver))
    assert resume is not None

    settlement = await resume(interaction_id="i-1", response="ignored")

    assert isinstance(settlement, ToolInteractionSettlement)
    assert settlement.status == "succeeded"
    assert expected.items() <= settlement.result.items()
    assert resolver.state_calls == [
        {"interaction_id": "i-1", "user_id": 7, "task_id": "t-1"}
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "state", [None, ConnectorConnectionState("pending")], ids=["unknown", "pending"]
)
async def test_resume_returns_none_so_the_model_replans(state) -> None:
    assert (
        await _tool(FakeResolver(state=state)).resume_user_interaction(
            interaction_id="i-1", response="done"
        )
        is None
    )


@pytest.mark.asyncio
async def test_resume_ignores_the_reply_text() -> None:
    """A reply claiming success must not settle anything; a secret in it is
    never read."""

    pending = FakeResolver(state=ConnectorConnectionState("pending"))
    assert (
        await _tool(pending).resume_user_interaction(
            interaction_id="i-1", response="connected! key=SECRET-123"
        )
        is None
    )
    connected = FakeResolver(state=ConnectorConnectionState("connected"))
    settlement = await _tool(connected).resume_user_interaction(
        interaction_id="i-1", response="SECRET-123"
    )
    assert settlement is not None
    assert "SECRET-123" not in repr(settlement)


@pytest.fixture
def _clean_hook():
    connector_resolver.set_connector_resolver_hook(None)
    yield
    connector_resolver.set_connector_resolver_hook(None)


class _Config:
    def get_user_id(self) -> int:
        return 7

    def get_task_id(self) -> str:
        return "t-1"


@pytest.mark.asyncio
async def test_tool_is_offered_only_when_a_resolver_is_registered(_clean_hook) -> None:
    assert await create_find_app_connection_tool(_Config()) == []  # type: ignore[arg-type]

    resolver = FakeResolver()
    connector_resolver.set_connector_resolver_hook(resolver)
    assert connector_resolver.has_connector_resolver_hook()
    [tool] = await create_find_app_connection_tool(_Config())  # type: ignore[arg-type]
    assert tool.name == "find_app_connection"
