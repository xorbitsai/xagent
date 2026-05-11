import tempfile

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from xagent.core.tools.adapters.vibe.agent_tool import (
    create_agent_tools,
    get_published_agents_tools,
)
from xagent.web.models.agent import Agent, AgentStatus
from xagent.web.models.database import Base
from xagent.web.models.user import User
from xagent.web.tools.config import WebToolConfig


def _create_session() -> tuple[Session, str]:
    temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    temp_db.close()
    db_url = f"sqlite:///{temp_db.name}"
    engine = create_engine(db_url)
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return SessionLocal(), temp_db.name


def test_non_owner_cannot_see_other_users_published_agent_tools() -> None:
    db, db_path = _create_session()
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

        tools_for_other = get_published_agents_tools(db=db, user_id=2)
        tool_names = {tool.name for tool in tools_for_other}

        assert "call_agent_owner_published_agent" not in tool_names
    finally:
        db.close()
        try:
            import os

            os.remove(db_path)
        except OSError:
            pass


def test_owner_sees_only_own_published_agents_not_drafts() -> None:
    db, db_path = _create_session()
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

        tools_for_owner = get_published_agents_tools(db=db, user_id=1)
        tool_names = {tool.name for tool in tools_for_owner}

        assert "call_agent_owner_published_agent" in tool_names
        assert "call_agent_owner_draft_agent" not in tool_names
    finally:
        db.close()
        try:
            import os

            os.remove(db_path)
        except OSError:
            pass


def test_allowed_agent_ids_include_only_selected_published_user_agents() -> None:
    db, db_path = _create_session()
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
            db=db,
            user_id=owner.id,
            allowed_agent_ids=[
                selected_published.id,
                selected_draft.id,
                other_users_agent.id,
            ],
        )
        tool_names = {tool.name for tool in tools}

        assert "call_agent_selected_published_agent" in tool_names
        assert "call_agent_selected_draft_agent" not in tool_names
        assert "call_agent_unselected_published_agent" not in tool_names
        assert "call_agent_other_users_agent" not in tool_names
    finally:
        db.close()
        try:
            import os

            os.remove(db_path)
        except OSError:
            pass


@pytest.mark.asyncio
async def test_create_agent_tools_treats_empty_delegate_allowlist_as_unrestricted() -> (
    None
):
    db, db_path = _create_session()
    try:
        owner = User(username="owner4", password_hash="x", is_admin=False)
        db.add(owner)
        db.commit()
        db.refresh(owner)

        published_agent = Agent(
            user_id=owner.id,
            name="Default Published Agent",
            status=AgentStatus.PUBLISHED,
        )
        db.add(published_agent)
        db.commit()

        config = WebToolConfig(db=db, request=None, user_id=owner.id, user=owner)
        config._delegate_agent_ids = []

        tools = await create_agent_tools(config)
        tool_names = {tool.name for tool in tools}

        assert "call_agent_default_published_agent" in tool_names
    finally:
        db.close()
        try:
            import os

            os.remove(db_path)
        except OSError:
            pass
