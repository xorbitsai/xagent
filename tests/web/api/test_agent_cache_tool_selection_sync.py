"""A reused AgentService follows its agent's current tool selection.

The selection is fixed on the cached ``WebToolConfig`` when the runtime is
built. These tests drive the real ``get_agent_for_task`` cache-hit path with a
real ``WebToolConfig`` and pin that a turn whose setup snapshot carries another
selection applies it (and rebuilds tools once), while an unchanged selection
leaves the cached runtime alone.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from xagent.core.tools.adapters.vibe.selection_spec import ToolSelectionSpec
from xagent.web.models.agent import AgentStatus
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.user import User
from xagent.web.services.agent_service_manager import (
    AgentServiceManager,
    _spec_wants_mcp,
)
from xagent.web.services.llm_utils import AgentRuntimeFields
from xagent.web.services.task_setup_snapshot import (
    RuntimeUserFields,
    TaskSetupSnapshot,
    _TaskFields,
)
from xagent.web.tools.config import WebToolConfig

TASK_ID = 42
STALE_MCP_CONFIGS = [{"name": "stale"}]


def _snapshot(tool_categories: list[str] | None) -> TaskSetupSnapshot:
    return TaskSetupSnapshot(
        task=_TaskFields(
            id=TASK_ID,
            user_id=1,
            status=TaskStatus.WAITING_FOR_USER,
            agent_id=7,
            agent_config=None,
            model_name=None,
            compact_model_name=None,
            execution_mode="flash",
            agent_type="standard",
        ),
        runtime_user=RuntimeUserFields(id=1, is_admin=False),
        has_reconstructable_history=False,
        task_pattern="single_call",
        task_llm=None,
        task_fast_llm=None,
        task_vision_llm=None,
        task_compact_llm=None,
        agent=AgentRuntimeFields(
            id=7,
            name="selection-agent",
            status=AgentStatus.PUBLISHED,
            instructions="be terse",
        ),
        agent_config={
            "llms": (None, None, None, None),
            "execution_mode": "flash",
            "instructions": "be terse",
            "skills": [],
            "knowledge_bases": [],
            "tool_categories": tool_categories,
        },
        excluded_agent_id=7,
    )


class _CachedAgent:
    """The cached AgentService, reduced to what a cache hit touches."""

    def __init__(self, tool_config: WebToolConfig) -> None:
        self.tool_config = tool_config
        self.invalidate_calls = 0
        # Waiting for the user: between turns, not executing.
        self.execution_status: dict[str, Any] | None = {
            "is_running": False,
            "is_resumable": True,
        }

    def invalidate_tools(self) -> None:
        self.invalidate_calls += 1

    def get_execution_status(self, execution_id: str) -> dict[str, Any] | None:
        return self.execution_status

    def revoke_memory(self, **kwargs: Any) -> None:
        pass

    def apply_memory_policy(self, **kwargs: Any) -> None:
        pass


def _warm(built_with: list[str] | None) -> tuple[AgentServiceManager, _CachedAgent]:
    """A manager holding a runtime built for an agent with ``built_with``."""
    spec = ToolSelectionSpec.from_raw(tool_categories=built_with)
    tool_config = WebToolConfig(
        db=None,
        request=None,
        user_id=1,
        tool_selection_spec=spec,
        include_mcp_tools=_spec_wants_mcp(spec),
    )
    tool_config._cached_mcp_configs = list(STALE_MCP_CONFIGS)
    agent = _CachedAgent(tool_config)
    manager = AgentServiceManager()
    manager._agents[TASK_ID] = agent
    manager._agent_owner_ids[TASK_ID] = 1
    manager._agent_scope_fingerprints[TASK_ID] = None
    return manager, agent


async def _next_turn(
    manager: AgentServiceManager, tool_categories: list[str] | None
) -> Any:
    task_row = Task(
        id=TASK_ID,
        user_id=1,
        title="selection task",
        description="x",
        status=TaskStatus.WAITING_FOR_USER,
        agent_id=7,
        agent_type="standard",
    )
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = task_row
    return await manager.get_agent_for_task(
        task_id=TASK_ID,
        db=db,
        user=User(id=1, username="owner", password_hash="hash", is_admin=False),
        task_setup_snapshot=_snapshot(tool_categories),
        connector_runtime_turn_id=None,
    )


@pytest.mark.asyncio
async def test_a_warm_runtime_gets_a_server_its_agent_gained_between_turns() -> None:
    manager, agent = _warm(["basic"])
    tool_config = agent.tool_config
    # Built without MCP: the config gate answers before any lookup.
    tool_config._cached_mcp_configs = None
    assert await tool_config.get_mcp_server_configs() == []

    result = await _next_turn(manager, ["basic", "mcp:store"])

    assert result is agent
    assert tool_config.get_tool_selection_spec() == ToolSelectionSpec.from_raw(
        tool_categories=["basic", "mcp:store"]
    )
    assert tool_config.get_tool_selection_spec().mcp_servers == frozenset({"store"})
    assert agent.invalidate_calls == 1
    # The gate is open now: the next tool build loads the server's config.
    loaded = [{"name": "store", "config": {}}]
    tool_config._load_mcp_server_configs = AsyncMock(return_value=loaded)  # type: ignore[method-assign]
    assert await tool_config.get_mcp_server_configs() == loaded


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("built_with", "this_turn"),
    [
        (["basic", "mcp:store"], ["basic", "mcp:store"]),
        # Stored in another order, or with a repeated entry: the same selection.
        (["basic", "mcp:store", "file"], ["file", "mcp:store", "basic", "basic"]),
        (["basic", "mcp"], ["mcp", "basic"]),
        (["basic"], ["basic"]),
        (None, None),
        ([], []),
    ],
)
async def test_an_unchanged_selection_leaves_the_cached_runtime_alone(
    built_with: list[str] | None, this_turn: list[str] | None
) -> None:
    manager, agent = _warm(built_with)
    spec_before = agent.tool_config.get_tool_selection_spec()

    result = await _next_turn(manager, this_turn)

    assert result is agent
    assert agent.invalidate_calls == 0
    assert agent.tool_config.get_tool_selection_spec() is spec_before
    assert agent.tool_config._cached_mcp_configs == STALE_MCP_CONFIGS


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("built_with", "this_turn"),
    [
        (["basic", "mcp"], ["basic"]),
        (["basic", "mcp:store", "mcp:other"], ["basic", "mcp:other"]),
        (["basic", "file"], ["basic"]),
        (["basic", "mcp"], []),
    ],
)
async def test_a_narrowed_selection_is_honoured_on_the_next_turn(
    built_with: list[str], this_turn: list[str]
) -> None:
    manager, agent = _warm(built_with)
    tool_config = agent.tool_config

    await _next_turn(manager, this_turn)

    expected = ToolSelectionSpec.from_raw(tool_categories=this_turn)
    assert tool_config.get_tool_selection_spec() == expected
    assert agent.invalidate_calls == 1
    assert tool_config._cached_mcp_configs is None
    if not _spec_wants_mcp(expected):
        # Nothing may load a server the agent no longer selects.
        tool_config._load_mcp_server_configs = AsyncMock(  # type: ignore[method-assign]
            side_effect=AssertionError("MCP configs must not be loaded")
        )
        assert await tool_config.get_mcp_server_configs() == []


@pytest.mark.asyncio
async def test_two_turns_with_the_same_new_selection_rebuild_once() -> None:
    manager, agent = _warm(["basic"])

    await _next_turn(manager, ["basic", "mcp:store"])
    await _next_turn(manager, ["mcp:store", "basic"])

    assert agent.invalidate_calls == 1


@pytest.mark.asyncio
async def test_a_lookup_during_an_executing_run_does_not_swap_its_selection() -> None:
    manager, agent = _warm(["basic"])
    spec_before = agent.tool_config.get_tool_selection_spec()
    agent.execution_status = {"is_running": True, "is_resumable": False}

    await _next_turn(manager, ["basic", "mcp:store"])

    assert agent.tool_config.get_tool_selection_spec() is spec_before
    assert agent.invalidate_calls == 0
    assert agent.tool_config._cached_mcp_configs == STALE_MCP_CONFIGS

    # Once that run is over, the next turn's lookup applies it.
    agent.execution_status = None
    await _next_turn(manager, ["basic", "mcp:store"])

    assert agent.tool_config.get_tool_selection_spec().mcp_servers == frozenset(
        {"store"}
    )
    assert agent.invalidate_calls == 1


def test_a_lookup_without_a_snapshot_keeps_the_selection() -> None:
    manager, agent = _warm(["basic"])
    spec_before = agent.tool_config.get_tool_selection_spec()

    manager._sync_tool_selection(TASK_ID, None, actor_marked=False)

    assert agent.tool_config.get_tool_selection_spec() is spec_before
    assert agent.invalidate_calls == 0
    assert agent.tool_config._cached_mcp_configs == STALE_MCP_CONFIGS
