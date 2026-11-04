"""Knowledge-base authoring tools must stay off the chat tool registry.

``create_knowledge_base_from_file`` / ``create_knowledge_base_from_url``
exist for the agent-builder chat, which instantiates them directly. Any
registry creator for them would mount them on every chat agent whose
selection admits the ``knowledge`` category (or on every unconfigured
agent), where a model can hallucinate calls to them (issue #2219).
"""

from typing import Any

import pytest

from xagent.core.tools.adapters.vibe.factory import ToolRegistry
from xagent.core.tools.adapters.vibe.file_ingestion_tool import (
    CreateKnowledgeBaseFromFileTool,
)
from xagent.core.tools.adapters.vibe.web_ingestion_tool import (
    CreateKnowledgeBaseFromUrlTool,
)

KB_AUTHORING_TOOL_NAMES = frozenset(
    {"create_knowledge_base_from_file", "create_knowledge_base_from_url"}
)


KB_AUTHORING_MODULES = frozenset(
    {
        CreateKnowledgeBaseFromFileTool.__module__,
        CreateKnowledgeBaseFromUrlTool.__module__,
    }
)


def _registered_creators() -> list[Any]:
    ToolRegistry._import_tool_modules()
    return [creator for creator, _cats, _gate in ToolRegistry._tool_creators]


def test_kb_authoring_modules_register_no_creator() -> None:
    creators = _registered_creators()
    assert not [c for c in creators if c.__module__ in KB_AUTHORING_MODULES]
    names = {c.__name__ for c in creators}
    assert "create_file_ingestion_tools" not in names
    assert "create_web_ingestion_tools" not in names


def test_kb_search_creator_stays_registered() -> None:
    assert "create_knowledge_tools" in {c.__name__ for c in _registered_creators()}


@pytest.mark.parametrize(
    "tool_cls", [CreateKnowledgeBaseFromFileTool, CreateKnowledgeBaseFromUrlTool]
)
def test_authoring_tool_classes_keep_their_names_and_sync_contract(tool_cls) -> None:
    # The builder-chat mounting itself is pinned by
    # tests/web/api/test_websocket_builder_chat.py.
    tool = tool_cls(user_id=1)
    assert tool.name in KB_AUTHORING_TOOL_NAMES
    with pytest.raises(NotImplementedError, match="Only supports async execution."):
        tool.run_json_sync({})
