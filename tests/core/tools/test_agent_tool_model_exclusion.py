"""Delegated sub-agents inherit the parent task's excluded media model ids."""

from pathlib import Path
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
from xagent.web.models.user import User, UserModel
from xagent.web.tools.config import WebToolConfig

pytestmark = pytest.mark.usefixtures("no_resolvable_default_llm")

IMAGE_MODEL_ID = "image-a"


@pytest.fixture
def published_agent_db(tmp_path: Path) -> Any:
    engine = create_engine(f"sqlite:///{tmp_path / 'exclusion.db'}")
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
            tool_categories=None,
            models={"general": model.id},
        )
        db.add(agent)
        db.add(
            Agent(
                user_id=user.id,
                name="Sibling",
                status=AgentStatus.PUBLISHED,
                models={"general": model.id},
            )
        )
        image = Model(
            model_id=IMAGE_MODEL_ID,
            category="image",
            model_provider="openai",
            model_name="gpt-image-1",
            api_key="test-api-key",
            base_url="https://api.openai.com/v1",
            abilities=["generate"],
        )
        db.add(image)
        db.commit()
        db.refresh(agent)
        db.refresh(image)
        db.add(UserModel(user_id=user.id, model_id=image.id, is_owner=True))
        db.commit()
        ids = (user.id, agent.id)
    try:
        yield session_factory, ids[0], ids[1]
    finally:
        engine.dispose()


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
        excluded_model_ids=frozenset({IMAGE_MODEL_ID}),
    )

    assert all(t._excluded_model_ids == {IMAGE_MODEL_ID} for t in _agent_tools(tools))


def test_get_published_agents_tools_defaults_to_no_exclusion(
    published_agent_db: Any,
) -> None:
    session_factory, user_id, _ = published_agent_db

    tools = get_published_agents_tools(session_factory=session_factory, user_id=user_id)

    assert all(t._excluded_model_ids == frozenset() for t in _agent_tools(tools))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "excluded", [frozenset({IMAGE_MODEL_ID, "tts-b"}), frozenset()]
)
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
@pytest.mark.parametrize("excluded", [frozenset({IMAGE_MODEL_ID}), frozenset()])
async def test_delegated_child_drops_the_excluded_media_model(
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

    seen: dict[str, Any] = {}

    async def run_child(**_: Any) -> dict[str, Any]:
        # The child config is closed after the run, so read it while it runs.
        child_config = mock_agent_service_class.call_args.kwargs["tool_config"]
        seen["image_models"] = set(child_config.get_image_models())
        seen["grandchildren"] = await create_agent_tools(child_config)
        return {"output": "done", "file_outputs": []}

    with (
        patch(
            "xagent.web.services.llm_utils.UserAwareModelStorage"
        ) as mock_storage_class,
        patch("xagent.core.agent.service.AgentService") as mock_agent_service_class,
    ):
        mock_storage_class.return_value = Mock()
        mock_agent_service_class.return_value.execute_task = AsyncMock(
            side_effect=run_child
        )
        await tool.run_json_async({"task": "draw something"})

    # The empty-set parametrization shows the seeded model is otherwise visible.
    assert seen["image_models"] == {IMAGE_MODEL_ID} - excluded
    assert all(
        t._excluded_model_ids == excluded for t in _agent_tools(seen["grandchildren"])
    )
