from __future__ import annotations

import inspect

import pytest

from xagent.core.tools.adapters.vibe.browser_use import create_browser_tools


@pytest.mark.asyncio
async def test_browser_tools_share_runtime_task_session_after_setup() -> None:
    tools = create_browser_tools(task_id="workspace-task")

    for tool in tools:
        setup = getattr(tool, "setup", None)
        if not callable(setup):
            continue
        result = setup(task_id="runtime-task")
        if inspect.isawaitable(result):
            await result

    session_tools = [
        tool
        for tool in tools
        if getattr(tool, "name", "") != "browser_list_sessions"
        and hasattr(tool, "_task_id")
    ]

    assert session_tools
    assert {tool._task_id for tool in session_tools} == {"runtime-task"}
