import tempfile

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from xagent.core.tools.adapters.vibe.agent_tool import (
    AgentTool,
    create_create_agent_tool,
    create_list_agents_tool,
    create_update_agent_tool,
    get_published_agents_tools,
)
from xagent.core.tools.adapters.vibe.agent_tool_names import gen_agent_tool_name
from xagent.core.tools.adapters.vibe.config import ToolConfig
from xagent.web.models.agent import Agent, AgentStatus
from xagent.web.models.database import Base
from xagent.web.models.user import User
from xagent.web.tools.config import WebToolConfig


def _create_session() -> tuple[Session, str, sessionmaker]:
    temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    temp_db.close()
    db_url = f"sqlite:///{temp_db.name}"
    engine = create_engine(db_url)
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return SessionLocal(), temp_db.name, SessionLocal


class _Request:
    def __init__(self, user: User):
        self.user = user


def _tool_name(agent: Agent) -> str:
    return gen_agent_tool_name(agent.id, agent.name)


def test_delegation_config_defaults_are_noop() -> None:
    config = ToolConfig({})

    assert config.get_allowed_agent_ids() is None
    assert config.get_agent_tool_overrides() == {}
    assert config.get_enable_global_agent_tools() is True
    assert config.get_allow_cross_user_agent_ids() is False
    assert config.get_parent_task_id() is None
    assert config.get_parent_tracer() is None
    assert config.get_agent_call_stack() == []


def test_web_delegation_config_defaults_are_noop() -> None:
    db, db_path, SessionLocal = _create_session()
    try:
        user = User(username="config-owner", password_hash="x", is_admin=False)
        db.add(user)
        db.commit()
        db.refresh(user)

        config = WebToolConfig(db=db, request=_Request(user), user_id=user.id)

        assert config.get_allowed_agent_ids() is None
        assert config.get_agent_tool_overrides() == {}
        assert config.get_enable_global_agent_tools() is True
        assert config.get_allow_cross_user_agent_ids() is False
        assert config.get_parent_task_id() is None
        assert config.get_parent_tracer() is None
        assert config.get_agent_call_stack() == []
    finally:
        db.close()
        try:
            import os

            os.remove(db_path)
        except OSError:
            pass


def test_non_owner_cannot_see_other_users_published_agent_tools() -> None:
    db, db_path, SessionLocal = _create_session()
    try:
        owner = User(username="owner", password_hash="x", is_admin=False)
        other_user = User(username="other", password_hash="x", is_admin=False)
        db.add_all([owner, other_user])
        db.commit()
        db.refresh(owner)
        db.refresh(other_user)

        published_agent = Agent(
            user_id=owner.id,
            name="Owner Published Agent",
            status=AgentStatus.PUBLISHED,
        )
        db.add(published_agent)
        db.commit()

        tools_for_other = get_published_agents_tools(
            session_factory=SessionLocal, user_id=2
        )
        tool_names = {tool.name for tool in tools_for_other}

        assert _tool_name(published_agent) not in tool_names
    finally:
        db.close()
        try:
            import os

            os.remove(db_path)
        except OSError:
            pass


def test_owner_sees_only_own_published_agents_not_drafts() -> None:
    db, db_path, SessionLocal = _create_session()
    try:
        owner = User(username="owner2", password_hash="x", is_admin=False)
        db.add(owner)
        db.commit()
        db.refresh(owner)

        published_agent = Agent(
            user_id=owner.id,
            name="Owner Published Agent",
            status=AgentStatus.PUBLISHED,
        )
        draft_agent = Agent(
            user_id=owner.id,
            name="Owner Draft Agent",
            status=AgentStatus.DRAFT,
        )
        db.add_all([published_agent, draft_agent])
        db.commit()

        tools_for_owner = get_published_agents_tools(
            session_factory=SessionLocal, user_id=1
        )
        tool_names = {tool.name for tool in tools_for_owner}

        assert _tool_name(published_agent) in tool_names
        assert _tool_name(draft_agent) not in tool_names
    finally:
        db.close()
        try:
            import os

            os.remove(db_path)
        except OSError:
            pass


def test_published_agent_tool_name_romanizes_non_ascii_names() -> None:
    db, db_path, SessionLocal = _create_session()
    try:
        owner = User(username="owner_non_ascii", password_hash="x", is_admin=False)
        db.add(owner)
        db.commit()
        db.refresh(owner)

        published_agent = Agent(
            user_id=owner.id,
            name="中文整理助手",
            status=AgentStatus.PUBLISHED,
        )
        db.add(published_agent)
        db.commit()
        db.refresh(published_agent)

        tools = get_published_agents_tools(
            session_factory=SessionLocal, user_id=owner.id
        )

        assert {tool.name for tool in tools} == {
            f"agent_zhong_wen_zheng_li_zhu_shou__a{published_agent.id}"
        }
    finally:
        db.close()
        try:
            import os

            os.remove(db_path)
        except OSError:
            pass


def test_allowed_agent_ids_include_only_selected_published_user_agents() -> None:
    db, db_path, SessionLocal = _create_session()
    try:
        owner = User(username="owner3", password_hash="x", is_admin=False)
        other_user = User(username="other3", password_hash="x", is_admin=False)
        db.add_all([owner, other_user])
        db.commit()
        db.refresh(owner)
        db.refresh(other_user)

        selected_published = Agent(
            user_id=owner.id,
            name="Selected Published Agent",
            status=AgentStatus.PUBLISHED,
        )
        selected_draft = Agent(
            user_id=owner.id,
            name="Selected Draft Agent",
            status=AgentStatus.DRAFT,
        )
        unselected_published = Agent(
            user_id=owner.id,
            name="Unselected Published Agent",
            status=AgentStatus.PUBLISHED,
        )
        other_users_agent = Agent(
            user_id=other_user.id,
            name="Other Users Agent",
            status=AgentStatus.PUBLISHED,
        )
        db.add_all(
            [
                selected_published,
                selected_draft,
                unselected_published,
                other_users_agent,
            ]
        )
        db.commit()

        tools = get_published_agents_tools(
            session_factory=SessionLocal,
            user_id=owner.id,
            allowed_agent_ids=[
                selected_published.id,
                selected_draft.id,
                other_users_agent.id,
            ],
        )
        tool_names = {tool.name for tool in tools}

        assert _tool_name(selected_published) in tool_names
        assert _tool_name(selected_draft) not in tool_names
        assert _tool_name(unselected_published) not in tool_names
        assert _tool_name(other_users_agent) not in tool_names
    finally:
        db.close()
        try:
            import os

            os.remove(db_path)
        except OSError:
            pass


def test_allowed_agent_ids_can_cross_users_only_when_enabled() -> None:
    db, db_path, SessionLocal = _create_session()
    try:
        owner = User(username="owner_cross", password_hash="x", is_admin=False)
        runner = User(username="runner_cross", password_hash="x", is_admin=False)
        db.add_all([owner, runner])
        db.commit()
        db.refresh(owner)
        db.refresh(runner)

        published_agent = Agent(
            user_id=owner.id,
            name="Shared Workforce Worker",
            status=AgentStatus.PUBLISHED,
        )
        db.add(published_agent)
        db.commit()
        db.refresh(published_agent)

        blocked_tools = get_published_agents_tools(
            session_factory=SessionLocal,
            user_id=runner.id,
            allowed_agent_ids=[published_agent.id],
            enable_global_agent_tools=False,
        )
        assert _tool_name(published_agent) not in {tool.name for tool in blocked_tools}

        allowed_tools = get_published_agents_tools(
            session_factory=SessionLocal,
            user_id=runner.id,
            allowed_agent_ids=[published_agent.id],
            enable_global_agent_tools=False,
            allow_cross_user_agent_ids=True,
        )
        assert _tool_name(published_agent) in {tool.name for tool in allowed_tools}
    finally:
        db.close()
        try:
            import os

            os.remove(db_path)
        except OSError:
            pass


@pytest.mark.asyncio
async def test_agent_tool_execution_enforces_owner_visibility() -> None:
    db, db_path, SessionLocal = _create_session()
    try:
        owner = User(username="exec_owner", password_hash="x", is_admin=False)
        runner = User(username="exec_runner", password_hash="x", is_admin=False)
        db.add_all([owner, runner])
        db.commit()
        db.refresh(owner)
        db.refresh(runner)

        published_agent = Agent(
            user_id=owner.id,
            name="Private Worker",
            status=AgentStatus.PUBLISHED,
        )
        db.add(published_agent)
        db.commit()
        db.refresh(published_agent)

        tool = AgentTool(
            agent_id=published_agent.id,
            agent_name=published_agent.name,
            agent_description="Private worker",
            session_factory=SessionLocal,
            user_id=runner.id,
        )

        result = await tool.run_json_async({"task": "run private worker"})

        assert result["response"] == f"Error: Agent {published_agent.id} not found"
    finally:
        db.close()
        try:
            import os

            os.remove(db_path)
        except OSError:
            pass


@pytest.mark.asyncio
async def test_agent_tool_execution_enforces_target_allowed_agent_ids() -> None:
    db, db_path, SessionLocal = _create_session()
    try:
        owner = User(username="allow_owner", password_hash="x", is_admin=False)
        db.add(owner)
        db.commit()
        db.refresh(owner)

        allowed_agent = Agent(
            user_id=owner.id,
            name="Allowed Worker",
            status=AgentStatus.PUBLISHED,
        )
        blocked_agent = Agent(
            user_id=owner.id,
            name="Blocked Worker",
            status=AgentStatus.PUBLISHED,
        )
        db.add_all([allowed_agent, blocked_agent])
        db.commit()
        db.refresh(allowed_agent)
        db.refresh(blocked_agent)

        tool = AgentTool(
            agent_id=blocked_agent.id,
            agent_name=blocked_agent.name,
            agent_description="Blocked worker",
            session_factory=SessionLocal,
            user_id=owner.id,
            target_allowed_agent_ids=[allowed_agent.id],
        )

        result = await tool.run_json_async({"task": "run blocked worker"})

        assert result["response"] == f"Error: Agent {blocked_agent.id} not found"
    finally:
        db.close()
        try:
            import os

            os.remove(db_path)
        except OSError:
            pass


@pytest.mark.asyncio
async def test_agent_tool_execution_allows_cross_user_only_with_target_allowlist() -> (
    None
):
    db, db_path, SessionLocal = _create_session()
    try:
        owner = User(username="cross_owner", password_hash="x", is_admin=False)
        runner = User(username="cross_runner", password_hash="x", is_admin=False)
        db.add_all([owner, runner])
        db.commit()
        db.refresh(owner)
        db.refresh(runner)

        published_agent = Agent(
            user_id=owner.id,
            name="Cross User Worker",
            status=AgentStatus.PUBLISHED,
        )
        db.add(published_agent)
        db.commit()
        db.refresh(published_agent)

        blocked_tool = AgentTool(
            agent_id=published_agent.id,
            agent_name=published_agent.name,
            agent_description="Cross user worker",
            session_factory=SessionLocal,
            user_id=runner.id,
            target_allowed_agent_ids=[published_agent.id],
        )
        blocked_result = await blocked_tool.run_json_async(
            {"task": "run cross user worker"}
        )
        assert (
            blocked_result["response"] == f"Error: Agent {published_agent.id} not found"
        )

        allowed_tool = AgentTool(
            agent_id=published_agent.id,
            agent_name=published_agent.name,
            agent_description="Cross user worker",
            session_factory=SessionLocal,
            user_id=runner.id,
            target_allowed_agent_ids=[published_agent.id],
            target_allow_cross_user_agent_ids=True,
        )
        allowed_result = await allowed_tool.run_json_async(
            {"task": "run cross user worker"}
        )
        assert allowed_result["response"] == (
            f"Error: No valid model configured for agent {published_agent.name}"
        )
    finally:
        db.close()
        try:
            import os

            os.remove(db_path)
        except OSError:
            pass


@pytest.mark.asyncio
async def test_delegation_allowed_agent_ids_do_not_block_current_worker_execution() -> (
    None
):
    db, db_path, SessionLocal = _create_session()
    try:
        owner = User(username="nested_owner", password_hash="x", is_admin=False)
        db.add(owner)
        db.commit()
        db.refresh(owner)

        worker = Agent(
            user_id=owner.id,
            name="Nested Restricted Worker",
            status=AgentStatus.PUBLISHED,
        )
        db.add(worker)
        db.commit()
        db.refresh(worker)

        tools = get_published_agents_tools(
            session_factory=SessionLocal,
            user_id=owner.id,
            allowed_agent_ids=[worker.id],
            enable_global_agent_tools=False,
            agent_tool_overrides={
                worker.id: {
                    "allowed_agent_ids": [],
                    "enable_global_agent_tools": False,
                }
            },
        )

        assert len(tools) == 1
        result = await tools[0].run_json_async({"task": "run nested restricted worker"})
        assert result["response"] == (
            f"Error: No valid model configured for agent {worker.name}"
        )
    finally:
        db.close()
        try:
            import os

            os.remove(db_path)
        except OSError:
            pass


def test_global_agent_tools_can_be_disabled_without_allowed_workers() -> None:
    db, db_path, SessionLocal = _create_session()
    try:
        owner = User(username="owner4", password_hash="x", is_admin=False)
        db.add(owner)
        db.commit()
        db.refresh(owner)

        published_agent = Agent(
            user_id=owner.id,
            name="Published Agent",
            status=AgentStatus.PUBLISHED,
        )
        db.add(published_agent)
        db.commit()

        tools = get_published_agents_tools(
            session_factory=SessionLocal,
            user_id=owner.id,
            enable_global_agent_tools=False,
        )

        assert tools == []
    finally:
        db.close()
        try:
            import os

            os.remove(db_path)
        except OSError:
            pass


def test_agent_call_stack_prevents_recursive_agent_tools() -> None:
    db, db_path, SessionLocal = _create_session()
    try:
        owner = User(username="owner5", password_hash="x", is_admin=False)
        db.add(owner)
        db.commit()
        db.refresh(owner)

        active_agent = Agent(
            user_id=owner.id,
            name="Active Agent",
            status=AgentStatus.PUBLISHED,
        )
        other_agent = Agent(
            user_id=owner.id,
            name="Other Agent",
            status=AgentStatus.PUBLISHED,
        )
        db.add_all([active_agent, other_agent])
        db.commit()
        db.refresh(active_agent)
        db.refresh(other_agent)

        tools = get_published_agents_tools(
            session_factory=SessionLocal,
            user_id=owner.id,
            agent_call_stack=[active_agent.id],
        )
        tool_names = {tool.name for tool in tools}

        assert _tool_name(active_agent) not in tool_names
        assert _tool_name(other_agent) in tool_names
    finally:
        db.close()
        try:
            import os

            os.remove(db_path)
        except OSError:
            pass


def test_worker_overrides_inject_selected_agent_tool_metadata() -> None:
    db, db_path, SessionLocal = _create_session()
    try:
        owner = User(username="owner6", password_hash="x", is_admin=False)
        db.add(owner)
        db.commit()
        db.refresh(owner)

        worker = Agent(
            user_id=owner.id,
            name="Writer Agent",
            description="General writer",
            status=AgentStatus.PUBLISHED,
        )
        global_agent = Agent(
            user_id=owner.id,
            name="Global Agent",
            status=AgentStatus.PUBLISHED,
        )
        db.add_all([worker, global_agent])
        db.commit()
        db.refresh(worker)

        tools = get_published_agents_tools(
            session_factory=SessionLocal,
            user_id=owner.id,
            allowed_agent_ids=[worker.id],
            enable_global_agent_tools=False,
            agent_tool_overrides={
                worker.id: {
                    "tool_name": "call_workforce_worker_1_writer",
                    "description": "Write the workforce report.",
                    "extra_system_prompt": "Focus on the assigned writing task.",
                    "allowed_agent_ids": [],
                    "enable_global_agent_tools": False,
                }
            },
        )

        assert [tool.name for tool in tools] == ["call_workforce_worker_1_writer"]
        assert tools[0].description == "Write the workforce report."
    finally:
        db.close()
        try:
            import os

            os.remove(db_path)
        except OSError:
            pass


@pytest.mark.asyncio
async def test_restricted_agent_config_hides_agent_management_tools() -> None:
    class RestrictedConfig:
        def get_enable_agent_tools(self) -> bool:
            return True

        def get_enable_global_agent_tools(self) -> bool:
            return False

    config = RestrictedConfig()

    assert await create_create_agent_tool(config) == []  # type: ignore[arg-type]
    assert await create_update_agent_tool(config) == []  # type: ignore[arg-type]
    assert await create_list_agents_tool(config) == []  # type: ignore[arg-type]
