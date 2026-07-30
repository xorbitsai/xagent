import asyncio
import os
import tempfile
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import xagent.core.tools.adapters.vibe.agent_tool as mod
from xagent.core.tools.adapters.vibe.agent_tool import AgentTool
from xagent.web.models.agent import Agent, AgentStatus
from xagent.web.models.database import Base
from xagent.web.models.model import Model
from xagent.web.models.user import User
from xagent.web.services.llm_utils import UserAwareModelStorage


class _Stop(Exception):
    """Halt the run before the sub-agent executes."""


class _DelegatedQuery:
    def __init__(self, agent):
        self._agent = agent

    def filter(self, *_args):
        return self

    def first(self):
        return self._agent


class _DelegatedSession:
    def __init__(self, agent):
        self._agent = agent

    def query(self, *_args):
        return _DelegatedQuery(self._agent)

    def commit(self):
        return None

    def close(self):
        return None


class _FailingCloseConfig:
    def close(self):
        raise ValueError("cleanup sentinel")


def _delegated_agent_tool() -> AgentTool:
    return AgentTool(
        agent_id=1,
        agent_name="Delegated",
        agent_description="d",
        session_factory=lambda: _DelegatedSession(
            SimpleNamespace(
                id=1,
                name="Delegated",
                instructions=None,
                knowledge_bases=None,
                skills=None,
                tool_categories=[],
                models={"general": 1},
                execution_mode=None,
            )
        ),
        user_id=1,
        tool_name="delegated",
        tool_description="d",
    )


def _patch_delegated_runtime(monkeypatch, execute_task):
    import xagent.core.agent.service as service_module
    import xagent.core.tools.adapters.vibe.agent_model_resolution as resolution

    class FakeAgentService:
        workspace = None

        def __init__(self, **_kwargs):
            return None

        async def execute_task(self, **_kwargs):
            return await execute_task()

    monkeypatch.setattr(mod, "WebToolConfig", lambda **_kwargs: _FailingCloseConfig())
    monkeypatch.setattr(service_module, "AgentService", FakeAgentService)
    monkeypatch.setattr(
        resolution,
        "resolve_agent_model_llms",
        lambda *_args: (object(), None, None, None),
    )


@pytest.mark.asyncio
async def test_agent_tool_maps_successful_body_cleanup_failure_to_boundary_error(
    monkeypatch,
):
    async def execute_task():
        return {"output": "completed"}

    _patch_delegated_runtime(monkeypatch, execute_task)
    tool = _delegated_agent_tool()
    monkeypatch.setattr(tool, "_create_child_execution_tracer", lambda **_kwargs: None)

    result = await tool.run_json_async({"task": "run"})

    assert result["response"].endswith("Tool runtime cleanup could not be completed.")


@pytest.mark.asyncio
async def test_agent_tool_preserves_body_failure_when_cleanup_also_fails(
    monkeypatch, caplog
):
    primary = RuntimeError("body sentinel")

    async def execute_task():
        raise primary

    _patch_delegated_runtime(monkeypatch, execute_task)
    tool = _delegated_agent_tool()
    monkeypatch.setattr(tool, "_create_child_execution_tracer", lambda **_kwargs: None)

    result = await tool.run_json_async({"task": "run"})

    assert result["response"].endswith("body sentinel")
    assert "Failed to close delegated agent tool runtime after execution" in caplog.text


@pytest.mark.asyncio
async def test_agent_tool_preserves_cancelled_error_identity_when_cleanup_fails(
    monkeypatch,
):
    primary = asyncio.CancelledError("cancelled sentinel")

    async def execute_task():
        raise primary

    _patch_delegated_runtime(monkeypatch, execute_task)
    tool = _delegated_agent_tool()
    monkeypatch.setattr(tool, "_create_child_execution_tracer", lambda **_kwargs: None)

    with pytest.raises(asyncio.CancelledError) as caught:
        await tool.run_json_async({"task": "run"})

    assert caught.value is primary


def _create_factory() -> tuple[sessionmaker, str]:
    temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    temp_db.close()
    db_url = f"sqlite:///{temp_db.name}"
    engine = create_engine(db_url)
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return SessionLocal, temp_db.name


def test_agent_tool_does_not_share_a_live_session_with_child_config(monkeypatch):
    """The child WebToolConfig must be built with a factory, never a live session."""
    SessionLocal, db_path = _create_factory()
    try:
        seed = SessionLocal()
        try:
            user = User(username="iso_owner", password_hash="x", is_admin=False)
            seed.add(user)
            seed.commit()
            seed.refresh(user)

            model = Model(
                model_id="general-model",
                model_provider="openai",
                model_name="General Model",
                api_key="x",
            )
            seed.add(model)
            seed.commit()
            seed.refresh(model)

            agent = Agent(
                user_id=user.id,
                name="Iso Worker",
                status=AgentStatus.PUBLISHED,
                models={"general": model.id},
            )
            seed.add(agent)
            seed.commit()
            seed.refresh(agent)

            agent_id = agent.id
            user_id = user.id
        finally:
            seed.close()

        # Make model resolution succeed so we reach the WebToolConfig build.
        monkeypatch.setattr(
            UserAwareModelStorage,
            "get_llm_by_name_with_access",
            lambda self, model_id, uid: object(),
        )

        captured: dict = {}

        def spy(*args, **kwargs):
            captured["db"] = kwargs.get("db")
            captured["db_factory"] = kwargs.get("db_factory")
            raise _Stop()

        monkeypatch.setattr(mod, "WebToolConfig", spy)

        tool = AgentTool(
            agent_id=agent_id,
            agent_name="Iso Worker",
            agent_description="d",
            session_factory=SessionLocal,
            user_id=user_id,
            tool_name="t",
            tool_description="d",
        )

        try:
            asyncio.run(tool.run_json_async({"task": "hi"}))
        except _Stop:
            pass

        assert captured["db"] is None
        assert captured["db_factory"] is SessionLocal
    finally:
        try:
            os.remove(db_path)
        except OSError:
            pass
