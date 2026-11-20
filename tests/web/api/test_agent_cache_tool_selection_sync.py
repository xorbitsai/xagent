"""A reused AgentService follows its task's current tool selection.

The selection is fixed on the cached ``WebToolConfig`` when the runtime is
built. Every test here builds the runtime through the real
``get_agent_for_task`` cache-miss path (the tool factory and the service are
stand-ins; the tool config is a real ``WebToolConfig`` built from what the
manager hands the factory), then looks the task up again on the cache-hit
path. A lookup that carries another selection applies it and rebuilds tools
once; one that carries the same selection, however it was derived, leaves the
cached runtime alone.
"""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import replace
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from xagent.core.agent.service import AgentService
from xagent.core.model.chat.basic.base import BaseLLM
from xagent.core.tools.adapters.vibe.selection_spec import ToolSelectionSpec
from xagent.web.models.task import TaskStatus
from xagent.web.models.user import User
from xagent.web.services.agent_service_manager import (
    AgentServiceManager,
    _include_mcp_tools,
    _spec_wants_mcp,
)
from xagent.web.services.llm_utils import AgentRuntimeFields
from xagent.web.services.mcp_runtime import MCPBuiltinOAuthActorPolicy
from xagent.web.services.task_runtime import (
    MCP_RUNTIME_AUTHORIZATION_POLICY_REQUIRED_KEY,
)
from xagent.web.services.task_setup_snapshot import (
    RuntimeUserFields,
    TaskModelOverride,
    TaskSetupSnapshot,
    _TaskFields,
    apply_task_model_override,
)
from xagent.web.services.workforce_runtime import WorkforceTaskRuntime
from xagent.web.tools.config import WebToolConfig

TASK_ID = 42
STALE_MCP_CONFIGS = [{"name": "stale"}]
WAITING = {"is_running": False, "is_resumable": True}
RUNNING = {"is_running": True, "is_resumable": False}
LOADER = "xagent.web.services.agent_service_manager.load_task_setup_snapshot_sync"
NO_ROW = object()  # what a snapshot load reads for a task that is gone


class _Model(BaseLLM):
    @property
    def abilities(self) -> list[str]:
        return ["chat"]

    @property
    def model_name(self) -> str:
        return "selected"

    @property
    def supports_thinking_mode(self) -> bool:
        return False

    async def chat(self, messages, **kwargs):  # type: ignore[override]
        return ""


# The models the agent resolved, as in the neighbouring tests: a snapshot that
# carries none makes the build fall back to a default model read from the
# environment (``OPENAI_API_KEY`` and friends), which CI does not have.
MODEL = _Model()


def _snapshot(
    tool_categories: list[str] | None, *, actor_marked: bool = False
) -> TaskSetupSnapshot:
    return TaskSetupSnapshot(
        task=_TaskFields(
            id=TASK_ID,
            user_id=1,
            status=TaskStatus.PENDING,
            agent_id=7,
            agent_config=(
                {MCP_RUNTIME_AUTHORIZATION_POLICY_REQUIRED_KEY: True}
                if actor_marked
                else {}
            ),
            model_name=None,
            compact_model_name=None,
            execution_mode="flash",
            agent_type="standard",
        ),
        runtime_user=RuntimeUserFields(id=1, is_admin=False),
        has_reconstructable_history=False,
        task_pattern="single_call",
        task_llm=MODEL,
        task_fast_llm=MODEL,
        task_vision_llm=MODEL,
        task_compact_llm=MODEL,
        agent=AgentRuntimeFields(
            id=7,
            name="selection-agent",
            status="published",
            instructions="be terse",
            agent_creator_user_id=1,
        ),
        agent_config={
            "llms": (MODEL, MODEL, MODEL, MODEL),
            "execution_mode": "flash",
            "instructions": "be terse",
            "skills": [],
            "knowledge_bases": [],
            "tool_categories": tool_categories,
        },
        excluded_agent_id=7,
    )


class _Runtime:
    """Drives ``get_agent_for_task``; the first lookup builds, later ones reuse."""

    def __init__(self) -> None:
        self.manager = AgentServiceManager()
        self.services: list[Any] = []
        self.snapshot_loads = 0

    async def lookup(
        self,
        snapshot: TaskSetupSnapshot | None,
        *,
        loaded: TaskSetupSnapshot | Exception | object | None = None,
        policy: MCPBuiltinOAuthActorPolicy | None = None,
    ) -> Any:
        """Look the task up with ``snapshot`` (``None``: without one, as the
        reply and A2A resume paths do); ``loaded`` is what a snapshot load reads."""
        runtime = self

        class Service(MagicMock):
            def _get_child_mock(self, **kw: Any) -> MagicMock:
                return MagicMock(**kw)

            def get_execution_status(self, execution_id: str) -> dict[str, Any]:
                return dict(self.execution_status)

            def revoke_memory(self, **kwargs: Any) -> None:
                AgentService.revoke_memory(self, **kwargs)

            def __init__(self, **kwargs: Any) -> None:
                super().__init__()
                self.execution_status = {"is_running": False, "is_resumable": False}
                self.tool_config = kwargs["tool_config"]
                self.memory = kwargs["memory"]
                self.memory_enabled = kwargs["memory_enabled"]
                self.memory_available = kwargs["memory_available"]
                self.memory_availability_reason = kwargs["memory_availability_reason"]
                self.execution_metadata = dict(kwargs["execution_metadata"])
                self.agent = SimpleNamespace(memory_store=self.memory)
                self._execution_adapter = None
                self.workspace = None
                runtime.services.append(self)

        async def create_tools(*args: Any, **kwargs: Any) -> Any:
            # What ``create_default_tools`` builds the tool config from.
            spec = kwargs["tool_selection_spec"]
            return [], WebToolConfig(
                db=None,
                request=None,
                user_id=1,
                tool_selection_spec=spec,
                include_mcp_tools=_include_mcp_tools(
                    spec, force=kwargs["force_mcp_tools"]
                ),
            )

        def load(*args: Any, **kwargs: Any) -> TaskSetupSnapshot:
            runtime.snapshot_loads += 1
            if isinstance(loaded, Exception):
                raise loaded
            if loaded is NO_ROW:
                return None
            return loaded or snapshot or _snapshot(["basic"])

        db = MagicMock()
        db.query.return_value.filter.return_value.first.return_value = MagicMock(
            status=TaskStatus.PENDING
        )
        module = "xagent.web.services.agent_service_manager"
        with ExitStack() as stack:
            for target in (
                patch(LOADER, new=load),
                patch.object(self.manager, "_load_persisted_conversation_history"),
                patch.object(
                    self.manager, "_load_persisted_execution_context", new=AsyncMock()
                ),
                patch(f"{module}.create_task_tracer", return_value=MagicMock()),
                patch(f"{module}.create_default_tools", new=create_tools),
                patch(
                    "xagent.web.sandbox_manager.get_sandbox_manager", return_value=None
                ),
                patch(f"{module}.AgentService", new=Service),
            ):
                stack.enter_context(target)
            return await self.manager.get_agent_for_task(
                task_id=TASK_ID,
                db=db,
                user=User(id=1, username="owner", password_hash="hash", is_admin=False),
                task_setup_snapshot=snapshot,
                mcp_runtime_authorization_policy=policy,
            )


async def _built(
    snapshot: TaskSetupSnapshot, **lookup: Any
) -> tuple[_Runtime, Any, WebToolConfig]:
    """A runtime built from ``snapshot`` whose run now waits for the user."""
    runtime = _Runtime()
    service = await runtime.lookup(snapshot, **lookup)
    service.execution_status = dict(WAITING)
    service.tool_config._cached_mcp_configs = list(STALE_MCP_CONFIGS)
    return runtime, service, service.tool_config


def _assert_left_alone(service: Any, spec_before: Any) -> None:
    assert service.tool_config.get_tool_selection_spec() is spec_before
    service.invalidate_tools.assert_not_called()
    assert service.tool_config._cached_mcp_configs == STALE_MCP_CONFIGS


def _categories(tool_config: WebToolConfig) -> frozenset[str]:
    return frozenset(tool_config.get_tool_selection_spec().categories)


@pytest.mark.asyncio
async def test_a_warm_runtime_gets_a_server_its_agent_gained_between_turns() -> None:
    runtime, service, tool_config = await _built(_snapshot(["basic"]))
    # Built without MCP: the config gate answers before any lookup.
    tool_config._cached_mcp_configs = None
    assert await tool_config.get_mcp_server_configs() == []

    result = await runtime.lookup(_snapshot(["basic", "mcp:store"]))

    assert result is service
    assert tool_config.get_tool_selection_spec() == ToolSelectionSpec.from_raw(
        tool_categories=["basic", "mcp:store"]
    )
    assert tool_config.get_tool_selection_spec().mcp_servers == frozenset({"store"})
    service.invalidate_tools.assert_called_once_with()
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
    runtime, service, tool_config = await _built(_snapshot(built_with))
    spec_before = tool_config.get_tool_selection_spec()

    assert await runtime.lookup(_snapshot(this_turn)) is service

    _assert_left_alone(service, spec_before)


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
    runtime, service, tool_config = await _built(_snapshot(built_with))

    await runtime.lookup(_snapshot(this_turn))

    expected = ToolSelectionSpec.from_raw(tool_categories=this_turn)
    assert tool_config.get_tool_selection_spec() == expected
    service.invalidate_tools.assert_called_once_with()
    assert tool_config._cached_mcp_configs is None
    if not _spec_wants_mcp(expected):
        # Nothing may load a server the agent no longer selects.
        tool_config._load_mcp_server_configs = AsyncMock(  # type: ignore[method-assign]
            side_effect=AssertionError("MCP configs must not be loaded")
        )
        assert await tool_config.get_mcp_server_configs() == []


@pytest.mark.asyncio
async def test_two_turns_with_the_same_new_selection_rebuild_once() -> None:
    runtime, service, _ = await _built(_snapshot(["basic"]))

    await runtime.lookup(_snapshot(["basic", "mcp:store"]))
    await runtime.lookup(_snapshot(["mcp:store", "basic"]))

    service.invalidate_tools.assert_called_once_with()


@pytest.mark.asyncio
async def test_a_lookup_during_a_running_turn_does_not_swap_its_selection() -> None:
    runtime, service, tool_config = await _built(_snapshot(["basic"]))
    spec_before = tool_config.get_tool_selection_spec()
    service.execution_status = dict(RUNNING)

    await runtime.lookup(_snapshot(["basic", "mcp:store"]))

    _assert_left_alone(service, spec_before)

    # Once that turn is over, the next lookup applies it.
    service.execution_status = dict(WAITING)
    await runtime.lookup(_snapshot(["basic", "mcp:store"]))

    assert tool_config.get_tool_selection_spec().mcp_servers == frozenset({"store"})
    service.invalidate_tools.assert_called_once_with()


# --- The same snapshot is a no-op whatever went into deriving the selection ---


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_categories", [["basic"], ["basic", "mcp:store"], None])
async def test_an_actor_marked_build_and_its_reuse_derive_the_same_selection(
    tool_categories: list[str] | None,
) -> None:
    policy = MCPBuiltinOAuthActorPolicy(resource_owner_key="actor:alice")
    snapshot = _snapshot(tool_categories, actor_marked=True)
    runtime, service, tool_config = await _built(snapshot, policy=policy)
    spec_before = tool_config.get_tool_selection_spec()
    # The forced MCP load of an actor-marked task, as the build set it.
    assert tool_config._include_mcp_tools

    assert await runtime.lookup(snapshot, policy=policy) is service

    _assert_left_alone(service, spec_before)
    assert tool_config._include_mcp_tools


@pytest.mark.asyncio
async def test_an_override_build_and_its_reuse_derive_the_same_selection() -> None:
    override = TaskModelOverride(
        llm=_Model(), excluded_tool_categories=frozenset({"mcp", "image"})
    )
    snapshot = apply_task_model_override(
        _snapshot(["basic", "mcp", "image", "vision"]), override
    )
    runtime, service, tool_config = await _built(snapshot)
    spec_before = tool_config.get_tool_selection_spec()
    assert _categories(tool_config) == {"basic"}

    assert await runtime.lookup(snapshot) is service

    _assert_left_alone(service, spec_before)


@pytest.mark.asyncio
async def test_an_actor_marked_override_that_excludes_mcp_stays_without_it() -> None:
    policy = MCPBuiltinOAuthActorPolicy(resource_owner_key="actor:alice")
    override = TaskModelOverride(
        llm=_Model(), excluded_tool_categories=frozenset({"mcp"})
    )
    snapshot = apply_task_model_override(
        _snapshot(["basic", "mcp"], actor_marked=True), override
    )
    runtime, service, tool_config = await _built(snapshot, policy=policy)
    spec_before = tool_config.get_tool_selection_spec()
    assert not _spec_wants_mcp(spec_before) and not tool_config._include_mcp_tools

    assert await runtime.lookup(snapshot, policy=policy) is service

    _assert_left_alone(service, spec_before)
    assert not tool_config._include_mcp_tools


# --- A waiting run keeps the override it was built with ---


@pytest.mark.asyncio
@pytest.mark.parametrize("carried", ["snapshot", "none"])
async def test_a_lookup_without_the_override_keeps_what_the_override_excluded(
    carried: str,
) -> None:
    # Pause, resume and a reply reach a waiting override run with a plain
    # snapshot, or with none. The selection they derive must not hand back
    # the categories the override took out.
    selected = ["basic", "mcp", "image", "vision"]
    override = TaskModelOverride(
        llm=_Model(), excluded_tool_categories=frozenset({"mcp", "image"})
    )
    runtime, service, tool_config = await _built(
        apply_task_model_override(_snapshot(selected), override)
    )
    spec_before = tool_config.get_tool_selection_spec()

    plain = _snapshot(selected)
    if carried == "snapshot":
        assert await runtime.lookup(plain) is service
    else:
        assert await runtime.lookup(None, loaded=plain) is service

    _assert_left_alone(service, spec_before)
    assert _categories(tool_config) == {"basic"}
    assert not tool_config._include_mcp_tools


@pytest.mark.asyncio
async def test_a_change_under_a_carried_override_is_applied_without_its_exclusions() -> (
    None
):
    override = TaskModelOverride(
        llm=_Model(), excluded_tool_categories=frozenset({"mcp"})
    )
    runtime, service, tool_config = await _built(
        apply_task_model_override(_snapshot(["basic", "mcp", "vision"]), override)
    )

    # The agent gained a server; the lookup carries no override.
    gained = ["basic", "mcp", "vision", "mcp:store"]
    assert await runtime.lookup(_snapshot(gained)) is service

    spec = tool_config.get_tool_selection_spec()
    # The plain "mcp" and the vision tools stay out; the named server comes in.
    assert _categories(tool_config) == {"basic"}
    assert spec.mcp_servers == frozenset({"store"})
    assert spec.scoped_mcp_servers() == frozenset({"store"})
    service.invalidate_tools.assert_called_once_with()


@pytest.mark.asyncio
async def test_an_override_that_no_longer_applies_leaves_the_selection_alone() -> None:
    override = TaskModelOverride(
        llm=_Model(), excluded_tool_categories=frozenset({"mcp"})
    )
    runtime, service, tool_config = await _built(
        apply_task_model_override(_snapshot(["basic", "mcp"]), override)
    )
    spec_before = tool_config.get_tool_selection_spec()

    # The agent went back to the default tools: there is no list left to
    # exclude from, and the unrestricted selection must not replace a narrowed one.
    assert await runtime.lookup(_snapshot(None)) is service

    _assert_left_alone(service, spec_before)


# --- A lookup without a setup snapshot: the reply and A2A resume paths ---


@pytest.mark.asyncio
async def test_a_lookup_without_a_snapshot_loads_the_current_selection() -> None:
    runtime, service, tool_config = await _built(_snapshot(["basic"]))
    loads_before = runtime.snapshot_loads

    current = _snapshot(["basic", "mcp:store"])
    assert await runtime.lookup(None, loaded=current) is service

    assert runtime.snapshot_loads == loads_before + 1
    assert tool_config.get_tool_selection_spec().mcp_servers == frozenset({"store"})
    assert tool_config._include_mcp_tools
    service.invalidate_tools.assert_called_once_with()


@pytest.mark.asyncio
async def test_a_snapshot_that_cannot_be_loaded_leaves_the_selection_alone() -> None:
    runtime, service, tool_config = await _built(_snapshot(["basic"]))
    spec_before = tool_config.get_tool_selection_spec()

    assert await runtime.lookup(None, loaded=RuntimeError("database gone")) is service

    _assert_left_alone(service, spec_before)


@pytest.mark.asyncio
async def test_a_snapshot_load_that_finds_no_task_leaves_the_selection_alone() -> None:
    runtime, service, tool_config = await _built(_snapshot(["basic"]))
    spec_before = tool_config.get_tool_selection_spec()

    assert await runtime.lookup(None, loaded=NO_ROW) is service

    _assert_left_alone(service, spec_before)


@pytest.mark.asyncio
async def test_a_lookup_without_a_snapshot_loads_none_under_a_running_turn() -> None:
    runtime, service, tool_config = await _built(_snapshot(["basic"]))
    spec_before = tool_config.get_tool_selection_spec()
    loads_before = runtime.snapshot_loads
    service.execution_status = dict(RUNNING)

    await runtime.lookup(None, loaded=_snapshot(["basic", "mcp:store"]))

    assert runtime.snapshot_loads == loads_before
    _assert_left_alone(service, spec_before)


# --- What the sync does not follow ---


@pytest.mark.asyncio
async def test_a_workforce_task_keeps_the_selection_it_was_built_with() -> None:
    # Its spec names the worker agents, next to a build-time allowed_agent_ids
    # the sync does not refresh; the two must not come apart.
    def workforce(tool_categories: list[str], workers: list[int]) -> TaskSetupSnapshot:
        return replace(
            _snapshot(tool_categories),
            workforce_runtime=WorkforceTaskRuntime(
                workforce_run_id=1,
                workforce_id=1,
                snapshot={},
                allowed_agent_ids=workers,
                agent_tool_overrides={},
                worker_tool_names={f"agent_{worker}" for worker in workers},
                manager_system_prompt=None,
                manager_agent_id=7,
            ),
        )

    runtime, service, tool_config = await _built(workforce(["basic"], [11]))
    spec_before = tool_config.get_tool_selection_spec()

    assert await runtime.lookup(workforce(["basic", "mcp:store"], [11, 12])) is service

    _assert_left_alone(service, spec_before)
