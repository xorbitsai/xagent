"""Tests for the race-safe get-or-create used when instantiating a workforce
template's worker agents (`_get_or_create_quick_access_worker_agent` in
`xagent.web.services.workforce_creator`).

A plain select-then-insert here would race: two concurrent callers (e.g. a
double-clicked "Use" on a workforce template) can both miss the initial
SELECT and then collide on `uq_agents_user_id_template_id_quick_access`
(the (user_id, template_id, quick-access-origin) unique index) at INSERT.
The resolver must recover by re-reading the winner's row instead of letting
the IntegrityError propagate as an unhandled 500.
"""

import asyncio
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from xagent.web.models import Agent, Base, User
from xagent.web.models.agent import AgentOrigin, AgentStatus
from xagent.web.services import workforce_creator
from xagent.web.services.workforce_creator import (
    _find_quick_access_worker_agent,
    _get_or_create_quick_access_worker_agent,
)


@pytest.fixture()
def db_session():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    session = sessionmaker(bind=engine)()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)
        engine.dispose()


def _create_user(db) -> User:
    user = User(username="race-user", password_hash="hash")
    db.add(user)
    db.flush()
    return user


class _FakeTemplateManager:
    async def get_template(self, template_id: str) -> dict:
        return {
            "name": "GA Analyzer",
            "agent_config": {
                "instructions": "You are the GA Analyzer.",
                "execution_mode": "balanced",
                "skills": [],
                "tool_categories": [],
            },
        }


def test_no_collision_creates_and_returns_a_new_agent(db_session) -> None:
    user = _create_user(db_session)

    resolved = asyncio.run(
        _get_or_create_quick_access_worker_agent(
            db_session,
            _FakeTemplateManager(),
            user_id=user.id,
            template_id="ga_analyzer",
        )
    )

    assert resolved.origin == AgentOrigin.TEMPLATE_QUICK_ACCESS.value
    assert resolved.status == AgentStatus.PUBLISHED
    assert resolved.template_id == "ga_analyzer"


def test_repeat_call_reuses_the_same_agent(db_session) -> None:
    user = _create_user(db_session)
    template_manager = _FakeTemplateManager()

    first = asyncio.run(
        _get_or_create_quick_access_worker_agent(
            db_session, template_manager, user_id=user.id, template_id="ga_analyzer"
        )
    )
    second = asyncio.run(
        _get_or_create_quick_access_worker_agent(
            db_session, template_manager, user_id=user.id, template_id="ga_analyzer"
        )
    )

    assert first.id == second.id
    assert (
        db_session.query(Agent)
        .filter(Agent.origin == AgentOrigin.TEMPLATE_QUICK_ACCESS.value)
        .count()
        == 1
    )


def test_race_retry_reuses_row_committed_by_concurrent_winner(db_session) -> None:
    """Simulates the exact TOCTOU window a double-clicked "Use" can hit:
    the initial SELECT misses the row (as if a concurrent request hadn't
    committed yet), then the real INSERT collides with a row that a
    concurrent winner already committed - the resolver must recover by
    re-reading that row rather than raising IntegrityError.
    """
    user = _create_user(db_session)

    competitor = Agent(
        user_id=user.id,
        name="GA Analyzer",
        instructions="from the concurrent winner",
        execution_mode="balanced",
        template_id="ga_analyzer",
        origin=AgentOrigin.TEMPLATE_QUICK_ACCESS.value,
        status=AgentStatus.PUBLISHED,
    )
    db_session.add(competitor)
    db_session.commit()

    call_count = {"n": 0}

    def _miss_once_then_delegate(db, *, user_id, template_id):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return None
        return _find_quick_access_worker_agent(
            db, user_id=user_id, template_id=template_id
        )

    with patch.object(
        workforce_creator,
        "_find_quick_access_worker_agent",
        side_effect=_miss_once_then_delegate,
    ):
        resolved = asyncio.run(
            _get_or_create_quick_access_worker_agent(
                db_session,
                _FakeTemplateManager(),
                user_id=user.id,
                template_id="ga_analyzer",
            )
        )

    assert resolved.id == competitor.id
    # One missed pre-check, one successful retry lookup after the real
    # INSERT hit the real unique-constraint violation.
    assert call_count["n"] == 2
    assert (
        db_session.query(Agent)
        .filter(Agent.origin == AgentOrigin.TEMPLATE_QUICK_ACCESS.value)
        .count()
        == 1
    )


def test_gives_up_with_409_if_retries_are_exhausted(db_session) -> None:
    """If every retry's re-lookup keeps missing the row (a pathological
    case, e.g. persistent contention), the resolver must surface a clear
    409 rather than retrying forever or letting IntegrityError escape.
    """
    from fastapi import HTTPException

    user = _create_user(db_session)

    competitor = Agent(
        user_id=user.id,
        name="GA Analyzer",
        instructions="from the concurrent winner",
        execution_mode="balanced",
        template_id="ga_analyzer",
        origin=AgentOrigin.TEMPLATE_QUICK_ACCESS.value,
        status=AgentStatus.PUBLISHED,
    )
    db_session.add(competitor)
    db_session.commit()

    with patch.object(
        workforce_creator,
        "_find_quick_access_worker_agent",
        return_value=None,
    ):
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(
                _get_or_create_quick_access_worker_agent(
                    db_session,
                    _FakeTemplateManager(),
                    user_id=user.id,
                    template_id="ga_analyzer",
                )
            )

    assert exc_info.value.status_code == 409
