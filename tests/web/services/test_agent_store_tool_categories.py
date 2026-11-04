"""Tool-categories persistence semantics in AgentStore (issue #944).

``None`` and ``[]`` mean different things at runtime
(``ToolSelectionSpec.from_raw``): ``None`` is the legacy "unconfigured"
value that keeps the full default tool set, ``[]`` is explicitly zero
tools. The store must preserve that distinction on writes while response
payloads keep rendering ``None`` as ``[]``.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from xagent.web.models.database import Base
from xagent.web.models.user import User
from xagent.web.services.agent_store import (
    AgentStore,
    clean_tool_categories,
    normalize_tool_categories,
)


@pytest.fixture()
def db() -> Iterator[Session]:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture()
def user_id(db: Session) -> int:
    user = User(username="store_user", password_hash="x", is_admin=False)
    db.add(user)
    db.commit()
    db.refresh(user)
    return int(user.id)


def test_normalize_preserves_none() -> None:
    assert normalize_tool_categories(None) is None


def test_normalize_keeps_empty_list() -> None:
    assert normalize_tool_categories([]) == []


def test_normalize_strips_unassignable_categories() -> None:
    assert normalize_tool_categories(["file", "agent", "other"]) == ["file"]


def test_clean_coerces_none_for_responses() -> None:
    assert clean_tool_categories(None) == []
    assert clean_tool_categories(["file", "agent"]) == ["file"]


def test_add_agent_omitted_tool_categories_persists_null(
    db: Session, user_id: int
) -> None:
    agent = AgentStore(db).add_agent(
        user_id=user_id,
        name="unconfigured",
        description=None,
        instructions=None,
    )
    db.commit()
    db.refresh(agent)
    assert agent.tool_categories is None


def test_add_agent_empty_tool_categories_persists_empty_list(
    db: Session, user_id: int
) -> None:
    agent = AgentStore(db).add_agent(
        user_id=user_id,
        name="zero-tools",
        description=None,
        instructions=None,
        tool_categories=[],
    )
    db.commit()
    db.refresh(agent)
    assert agent.tool_categories == []


def test_update_agent_fields_preserves_none_tool_categories(
    db: Session, user_id: int
) -> None:
    store = AgentStore(db)
    agent = store.add_agent(
        user_id=user_id,
        name="update-to-unconfigured",
        description=None,
        instructions=None,
        tool_categories=["file"],
    )
    db.commit()

    updated = store.update_agent_fields(
        user_id, int(agent.id), {"tool_categories": None}
    )
    assert updated is not None
    assert updated.tool_categories is None


def test_update_agent_fields_keeps_empty_tool_categories(
    db: Session, user_id: int
) -> None:
    store = AgentStore(db)
    agent = store.add_agent(
        user_id=user_id,
        name="update-to-zero-tools",
        description=None,
        instructions=None,
        tool_categories=["file"],
    )
    db.commit()

    updated = store.update_agent_fields(user_id, int(agent.id), {"tool_categories": []})
    assert updated is not None
    assert updated.tool_categories == []


def test_update_agent_fields_strips_unassignable_tool_categories(
    db: Session, user_id: int
) -> None:
    store = AgentStore(db)
    agent = store.add_agent(
        user_id=user_id,
        name="update-strips-unassignable",
        description=None,
        instructions=None,
        tool_categories=[],
    )
    db.commit()

    updated = store.update_agent_fields(
        user_id, int(agent.id), {"tool_categories": ["file", "agent", "other"]}
    )
    assert updated is not None
    assert updated.tool_categories == ["file"]


def test_agent_response_dict_renders_null_as_empty_list(
    db: Session, user_id: int
) -> None:
    store = AgentStore(db)
    agent = store.add_agent(
        user_id=user_id,
        name="unconfigured-response",
        description=None,
        instructions=None,
    )
    db.commit()
    db.refresh(agent)
    assert store.agent_to_response_dict(agent)["tool_categories"] == []
