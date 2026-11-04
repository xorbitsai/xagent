"""The tool factory and the runner resolve one spill directory per task.

A spilled tool result is written by the tool layer into the directory its
tool set's SpillTarget names, and registered by the ExecutionContext the
runner builds, which looks for the file under its own workspace path. The
two sides build their TaskWorkspace objects separately: the factory from
the tool config's workspace config (after AgentService rewrites its task_id
to the service id), and the runner from AgentService.workspace_base_dir and
the execution adapter's workspace id. If the two ever named different
directories, every spill would reach the model as "unavailable".

These tests drive both production paths from one AgentService --
_ensure_tools_initialized for the factory, and the execution adapter's
runner and its _build_context for the context -- and assert that two
distinct workspace objects still name the same spill directory.
"""

from __future__ import annotations

from typing import Any

import pytest

from xagent.core.agent.service import AgentService
from xagent.core.tools.adapters.vibe.config import ToolConfig
from xagent.core.tools.adapters.vibe.output_filter_wrapper import (
    OutputFilteredToolWrapper,
)
from xagent.core.tools.adapters.vibe.sandboxed_tool.sandbox_config import (
    extract_bound_method_target,
)
from xagent.core.tools.tool_result_spill import (
    SPILL_READ_TOOL_NAME,
    spill_dir_for_workspace,
)
from xagent.core.workspace import TaskWorkspace

SERVICE_ID = "web_task_spill_dir"
DB_TASK_ID = "4242"


class _ServiceToolConfig(ToolConfig):
    """A tool config with the workspace-config contract AgentService reads.

    AgentService looks for, and rewrites, a ``_workspace_config`` dict; the
    factory reads the workspace config through ``get_workspace_config``.
    The production web tool config keeps both as one dict, and so does
    this one, so the task_id rewrite reaches the factory the same way.
    """

    def __init__(self, workspace_config: dict[str, Any]) -> None:
        super().__init__({"workspace": workspace_config})
        self._workspace_config = self.workspace_config


def _factory_workspace(tools: list[Any]) -> TaskWorkspace:
    readers = [tool for tool in tools if tool.name == SPILL_READ_TOOL_NAME]
    assert len(readers) == 1, [tool.name for tool in tools]
    target = extract_bound_method_target(readers[0]._target)
    assert target is not None
    instance, _ = target
    workspace = instance.workspace
    assert isinstance(workspace, TaskWorkspace)
    return workspace


async def _runner_context_and_workspace(service: AgentService) -> tuple[Any, Any]:
    """Build the execution context the way AgentService's runs do.

    The execution adapter derives the execution id and the workspace id and
    starts its runner with no base_dir, so the runner falls back to the
    workspace_base_dir the adapter was built with; _build_context is the
    runner method that turns those into the context's workspace path.
    """
    adapter = service._build_execution_adapter()
    runner, _ = adapter._build_runner()
    execution_id = str(adapter.config.current_task_id or adapter.config.service_id)
    return await runner._build_context(
        task="spill directory check",
        execution_id=execution_id,
        user_id=None,
        session_id=None,
        workspace_id=adapter._workspace_id(execution_id),
        allowed_external_dirs=adapter.config.allowed_external_dirs,
        base_dir=None,
        metadata=None,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "initial_task_id",
    [SERVICE_ID, "task-id-before-the-service-rewrites-it"],
    ids=["task-id-equals-service-id", "task-id-rewritten-to-service-id"],
)
async def test_the_factory_and_the_runner_resolve_one_spill_directory(
    tmp_path, initial_task_id
):
    base_dir = str(tmp_path)
    tool_config = _ServiceToolConfig({"base_dir": base_dir, "task_id": initial_task_id})
    service = AgentService(
        name="spill-directory-service",
        id=SERVICE_ID,
        task_id=DB_TASK_ID,
        tool_config=tool_config,
        enable_workspace=True,
        workspace_base_dir=base_dir,
    )

    await service._ensure_tools_initialized()
    context, runner_workspace = await _runner_context_and_workspace(service)

    factory_workspace = _factory_workspace(service.tools)
    # Two workspace objects built separately; one shared object would only
    # pin the spelling of the layout, not two paths agreeing on it.
    assert isinstance(runner_workspace, TaskWorkspace)
    assert factory_workspace is not runner_workspace
    # The directory is keyed on the service id, not on the task id the tool
    # config started with and not on the execution id the runner runs under.
    expected = spill_dir_for_workspace(tmp_path.resolve() / SERVICE_ID)
    wrappers = [
        tool
        for tool in service.tools
        if isinstance(tool, OutputFilteredToolWrapper)
        and tool._spill_target is not None
    ]
    assert wrappers
    assert {wrapper._spill_target.spill_dir for wrapper in wrappers} == {expected}
    assert context._spill_dir() == expected
    assert spill_dir_for_workspace(factory_workspace.workspace_dir) == expected
    assert spill_dir_for_workspace(runner_workspace.workspace_dir) == expected
    if initial_task_id == SERVICE_ID:
        assert service.workspace is not None
        assert spill_dir_for_workspace(service.workspace.workspace_dir) == expected
