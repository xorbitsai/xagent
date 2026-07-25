import asyncio
import os
import tempfile

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
            captured["computer_runtime_kind"] = kwargs.get("computer_runtime_kind")
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
            computer_runtime_kind="desktop_relay",
        )

        try:
            asyncio.run(tool.run_json_async({"task": "hi"}))
        except _Stop:
            pass

        assert captured["db"] is None
        assert captured["db_factory"] is SessionLocal
        assert captured["computer_runtime_kind"] == "desktop_relay"
    finally:
        try:
            os.remove(db_path)
        except OSError:
            pass
