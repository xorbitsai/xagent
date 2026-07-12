from __future__ import annotations

import inspect

import pytest

from xagent.core.tools.adapters.vibe.browser_use import (
    BrowserScreenshotTool,
    BrowserTaskSessionMixin,
    create_browser_tools,
)
from xagent.core.workspace import TaskWorkspace


def test_browser_debug_tools_are_opt_in() -> None:
    default_names = {tool.name for tool in create_browser_tools(task_id="task")}
    debug_names = {
        tool.name
        for tool in create_browser_tools(
            task_id="task",
            include_debug_tools=True,
        )
    }

    assert "browser_list_sessions" not in default_names
    assert "browser_list_sessions" in debug_names


@pytest.mark.asyncio
async def test_browser_task_session_mixin_defaults_to_no_task() -> None:
    tool = BrowserTaskSessionMixin()

    assert tool._task_id is None

    await tool.setup(task_id=None)

    assert tool._task_id is None


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


def test_browser_task_session_mixin_defaults_to_step_scoped_session() -> None:
    tool = BrowserTaskSessionMixin()
    tool._task_id = "task-412"

    args = tool._with_default_session(
        {"url": "poster.html", "_xagent_step_id": "render english"}
    )

    assert args["session_id"] == "task-412:render_english"
    assert "_xagent_step_id" not in args


def test_browser_task_session_mixin_keeps_explicit_session() -> None:
    tool = BrowserTaskSessionMixin()
    tool._task_id = "task-412"

    args = tool._with_default_session(
        {
            "url": "poster.html",
            "session_id": "custom-session",
            "_xagent_step_id": "render_english",
        }
    )

    assert args["session_id"] == "custom-session"
    assert "_xagent_step_id" not in args


@pytest.mark.asyncio
async def test_browser_screenshot_returns_registered_file_ref(
    tmp_path, monkeypatch
) -> None:
    def mock_create_record(self, file_id, file_path, db_session=None):
        path_str = str(file_path)
        resolved_str = str(file_path.resolve())
        self._recently_registered_files[path_str] = file_id
        self._recently_registered_files[resolved_str] = file_id
        self._file_id_to_path[file_id] = file_path

    monkeypatch.setattr(TaskWorkspace, "_create_file_record", mock_create_record)

    async def fake_browser_screenshot(**kwargs):
        return {
            "success": True,
            "session_id": kwargs["session_id"],
            "screenshot": (
                "data:image/png;base64,"
                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAA"
                "DElEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
            ),
            "format": "png",
            "full_page": True,
            "wait_for_lazy_load": False,
            "message": "ok",
            "error": "",
        }

    monkeypatch.setattr(
        "xagent.core.tools.adapters.vibe.browser_use.browser_screenshot",
        fake_browser_screenshot,
    )

    workspace = TaskWorkspace("test_task", str(tmp_path))
    tool = BrowserScreenshotTool(task_id="task-412", workspace=workspace)

    result = await tool.run_json_async(
        {
            "full_page": True,
            "output_filename": "poster_en.png",
            "_xagent_step_id": "render_english",
        }
    )

    assert result["success"] is True
    assert result["screenshot"] == "output/poster_en.png"
    assert result["file_id"]
    assert result["file_ref"]["file_id"] == result["file_id"]
    assert result["file_ref"]["relative_path"] == "output/poster_en.png"
    assert result["markdown_link"] == (f"[poster_en.png](file:{result['file_id']})")
