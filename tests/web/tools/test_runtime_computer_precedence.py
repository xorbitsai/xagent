from types import SimpleNamespace

import pytest

import xagent.web.api.chat as chat_module
from xagent.core.task_runtime import (
    TaskRuntimeContext,
    TaskRuntimeContribution,
    merge_task_runtime_contributions,
)
from xagent.core.tools.adapters.vibe import browser_tools as browser_tools_module
from xagent.core.tools.adapters.vibe.base import ToolCategory
from xagent.core.tools.adapters.vibe.config import ToolConfig
from xagent.core.tools.adapters.vibe.factory import ToolFactory, ToolRegistry
from xagent.web.api.chat import create_default_tools


@pytest.mark.asyncio
async def test_create_default_tools_prefers_out_of_tree_computer_runtime(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    runtime_tool = SimpleNamespace(
        name="computer",
        metadata=SimpleNamespace(category=ToolCategory.OTHER),
    )
    conflicting_runtime_tool = SimpleNamespace(
        name="computer",
        metadata=SimpleNamespace(category=ToolCategory.OTHER),
    )

    class _FakeToolConfig(ToolConfig):
        def __init__(self, **_kwargs):
            super().__init__({})
            self.runtime_contribution = TaskRuntimeContribution()
            self.runtime_workspace = None

        def get_workspace_config(self):
            return {"task_id": "web_task_11"}

        def set_task_runtime_contribution(self, contribution) -> None:
            self.runtime_contribution = contribution

        def get_task_runtime_contribution(self):
            return self.runtime_contribution

        def set_task_runtime_workspace(self, workspace) -> None:
            self.runtime_workspace = workspace

        def get_task_runtime_workspace(self):
            return self.runtime_workspace

        def get_browser_tools_enabled(self) -> bool:
            return True

    async def create_registered_tools(config):
        return await browser_tools_module.create_browser_tools(config)

    async def build_runtime(_context):
        return merge_task_runtime_contributions(
            {
                "out_of_tree_browser": TaskRuntimeContribution(
                    tools=(runtime_tool,),
                    environment="Control the selected browser.",
                    preferred_input_modalities=("image",),
                ),
                "second_browser": TaskRuntimeContribution(
                    tools=(conflicting_runtime_tool,),
                    environment="Control a different browser.",
                ),
            }
        )

    monkeypatch.setattr("xagent.web.tools.config.WebToolConfig", _FakeToolConfig)
    monkeypatch.setattr(
        "xagent.web.models.database.get_session_local",
        lambda: object(),
    )
    monkeypatch.setattr(
        ToolRegistry,
        "create_registered_tools",
        create_registered_tools,
    )
    monkeypatch.setattr(
        ToolFactory,
        "create_workspace",
        lambda _config: SimpleNamespace(id="workspace"),
    )
    monkeypatch.setattr(chat_module, "build_task_runtime", build_runtime)
    monkeypatch.setattr(
        chat_module,
        "registered_task_extensions",
        lambda: ("out_of_tree_browser",),
    )

    tools, config = await create_default_tools(
        None,
        user=SimpleNamespace(id=7, is_admin=False),
        task_id="web_task_11",
        task_runtime_context=TaskRuntimeContext(
            task_id=11,
            user_id=7,
            source="internal",
            session_factory=lambda: object(),
        ),
    )

    assert tools == [runtime_tool]
    assert config.runtime_contribution.tools == (runtime_tool,)
    assert config.runtime_contribution.environment == "Control the selected browser."
    assert "Dropping task runtime extension 'second_browser'" in caplog.text
