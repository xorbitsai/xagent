"""Delegated sub-agents inherit the parent task's excluded media model ids."""

import os
import tempfile
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from xagent.core.tools.adapters.vibe.agent_tool import (
    AgentTool,
    create_agent_tools,
    get_published_agents_tools,
)
from xagent.core.tools.adapters.vibe.selection_spec import ToolSelectionSpec
from xagent.web.models.agent import Agent, AgentStatus
from xagent.web.models.database import Base
from xagent.web.models.model import Model
from xagent.web.models.user import User
from xagent.web.tools.config import WebToolConfig

pytestmark = pytest.mark.usefixtures("no_resolvable_default_llm")


@pytest.fixture
def published_agent_db() -> Any:
    temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    temp_db.close()
    engine = create_engine(f"sqlite:///{temp_db.name}")
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    with session_factory() as db:
        user = User(username="exclusion-owner", password_hash="x", is_admin=False)
        db.add(user)
        db.commit()
        db.refresh(user)
        model = Model(
            model_id="chat-model-id",
            category="llm",
            model_provider="openai",
            model_name="gpt-4",
            api_key="test-api-key",
            base_url="https://api.openai.com/v1",
            temperature=0.7,
            abilities=["chat"],
        )
        db.add(model)
        db.commit()
        db.refresh(model)
        agent = Agent(
            user_id=user.id,
            name="Delegate",
            status=AgentStatus.PUBLISHED,
            models={"general": model.id},
        )
        db.add(agent)
        db.commit()
        db.refresh(agent)
        ids = (user.id, agent.id)
    try:
        yield session_factory, ids[0], ids[1]
    finally:
        engine.dispose()
        try:
            os.remove(temp_db.name)
        except OSError:
            pass


def _agent_tools(tools: list[Any]) -> list[AgentTool]:
    assert tools and all(isinstance(tool, AgentTool) for tool in tools)
    return tools


def _parent_config(
    session_factory: Any, user_id: int, excluded: frozenset[str]
) -> WebToolConfig:
    return WebToolConfig(
        db=None,
        request=None,
        db_factory=session_factory,
        user_id=user_id,
        task_id="parent-task",
        workspace_config={"task_id": "parent-task"},
        include_mcp_tools=False,
        tool_selection_spec=ToolSelectionSpec.from_raw(tool_categories=None),
        excluded_model_ids=excluded,
    )


def test_get_published_agents_tools_passes_excluded_ids_to_each_tool(
    published_agent_db: Any,
) -> None:
    session_factory, user_id, _ = published_agent_db

    tools = get_published_agents_tools(
        session_factory=session_factory,
        user_id=user_id,
        excluded_model_ids=frozenset({"image-a"}),
    )

    assert all(t._excluded_model_ids == {"image-a"} for t in _agent_tools(tools))


def test_get_published_agents_tools_defaults_to_no_exclusion(
    published_agent_db: Any,
) -> None:
    session_factory, user_id, _ = published_agent_db

    tools = get_published_agents_tools(session_factory=session_factory, user_id=user_id)

    assert all(t._excluded_model_ids == frozenset() for t in _agent_tools(tools))


@pytest.mark.asyncio
@pytest.mark.parametrize("excluded", [frozenset({"image-a", "tts-b"}), frozenset()])
async def test_create_agent_tools_reads_the_parent_task_exclusion(
    published_agent_db: Any, excluded: frozenset[str]
) -> None:
    session_factory, user_id, _ = published_agent_db
    config = _parent_config(session_factory, user_id, excluded)
    try:
        tools = await create_agent_tools(config)
    finally:
        config.close()

    assert all(t._excluded_model_ids == excluded for t in _agent_tools(tools))


@pytest.mark.asyncio
@pytest.mark.parametrize("excluded", [frozenset({"image-a"}), frozenset()])
async def test_delegated_child_tool_config_carries_the_exclusion(
    published_agent_db: Any, excluded: frozenset[str]
) -> None:
    session_factory, user_id, agent_id = published_agent_db
    tools = get_published_agents_tools(
        session_factory=session_factory,
        user_id=user_id,
        task_id="parent-task",
        excluded_model_ids=excluded,
    )
    tool = next(t for t in _agent_tools(tools) if t._agent_id == agent_id)

    with (
        patch(
            "xagent.web.services.llm_utils.UserAwareModelStorage"
        ) as mock_storage_class,
        patch("xagent.core.agent.service.AgentService") as mock_agent_service_class,
        patch("xagent.core.memory.in_memory.InMemoryMemoryStore"),
    ):
        mock_storage_class.return_value = Mock()
        mock_agent_service_class.return_value.execute_task = AsyncMock(
            return_value={"output": "done", "file_outputs": []}
        )
        await tool.run_json_async({"task": "draw something"})

    child_config = mock_agent_service_class.call_args.kwargs["tool_config"]
    assert child_config.get_excluded_model_ids() == excluded
