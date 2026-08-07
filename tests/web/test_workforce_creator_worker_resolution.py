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
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from xagent.web.models import Agent, Base, User
from xagent.web.models.agent import AgentOrigin, AgentStatus
from xagent.web.services import workforce_creator
from xagent.web.services.agent_store import AgentStore
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


def test_creates_agent_with_description_and_models_from_template(db_session) -> None:
    """The /task quick-access resolver (`_spec_from_template`) populates
    description/models/suggested_prompts from the template; this resolver
    used to hardcode them all to empty even though it writes the exact same
    (user_id, template_id, quick-access-origin) row - so whichever flow ran
    first silently won, permanently, for every field the other flow would
    have set (PR #1127 re-review, F2).

    `knowledge_bases` is the deliberate exception: the /task path validates
    template KBs per-user (`_validate_agent_knowledge_bases`) before
    attaching them, and this path has no equivalent check - so it must NOT
    forward them raw (that would silently attach dangling KB ids instead of
    failing loudly the way /task does)."""

    class _RichFakeTemplateManager:
        async def get_template(self, template_id: str) -> dict:
            return {
                "name": "GA Analyzer",
                "descriptions": {"en": "Explains GA4 trends."},
                "agent_config": {
                    "instructions": "You are the GA Analyzer.",
                    "execution_mode": "balanced",
                    "skills": [],
                    "tool_categories": [],
                    "models": {"general": "gpt-4o"},
                    "knowledge_bases": ["kb-1"],
                    "suggested_prompts": ["Summarize this week's traffic"],
                },
            }

    user = _create_user(db_session)

    resolved = asyncio.run(
        _get_or_create_quick_access_worker_agent(
            db_session,
            _RichFakeTemplateManager(),
            user_id=user.id,
            template_id="ga_analyzer",
        )
    )

    assert resolved.description == "Explains GA4 trends."
    assert resolved.models == {"general": "gpt-4o"}
    assert resolved.suggested_prompts == ["Summarize this week's traffic"]
    # Never forwarded without per-user validation - see docstring.
    assert resolved.knowledge_bases == []


def test_reuse_raises_specific_error_when_existing_agent_is_unpublished(
    db_session,
) -> None:
    """A user can unpublish their quick-access agent for a template a
    workforce also depends on (a normal, supported action). Reusing it
    as-is would pass this resolver only to fail downstream in
    create_workforce_worker's require_published=True check with a generic
    400 that the frontend renders as "please retry" - which can never
    succeed, since every retry resolves to the same unpublished agent
    (PR #1127 re-review, F1). The resolver must raise a specific,
    actionable error instead of returning the unpublished agent.
    """
    from fastapi import HTTPException

    from xagent.web.services.workforce_creator import (
        WORKFORCE_WORKER_UNPUBLISHED_CODE,
    )

    user = _create_user(db_session)
    unpublished = Agent(
        user_id=user.id,
        name="GA Analyzer",
        instructions="You are the GA Analyzer.",
        execution_mode="balanced",
        template_id="ga_analyzer",
        origin=AgentOrigin.TEMPLATE_QUICK_ACCESS.value,
        status=AgentStatus.DRAFT,
    )
    db_session.add(unpublished)
    db_session.commit()

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            _get_or_create_quick_access_worker_agent(
                db_session,
                _FakeTemplateManager(),
                user_id=user.id,
                template_id="ga_analyzer",
            )
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail["code"] == WORKFORCE_WORKER_UNPUBLISHED_CODE
    assert exc_info.value.detail["params"]["agent_name"] == "GA Analyzer"
    assert "GA Analyzer" in exc_info.value.detail["message"]
    # Never silently republished or replaced.
    db_session.refresh(unpublished)
    assert unpublished.status == AgentStatus.DRAFT
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

    Also stages a stand-in for the manager agent / Workforce that
    `create_workforce_from_template` flushes into the *same* session before
    ever calling this resolver, and asserts it survives the retry. The
    resolver recovers via a SAVEPOINT (`db.begin_nested()`) specifically so
    a collision only unwinds its own failed insert; a regression to a
    plain `db.rollback()` would pass every other assertion in this test
    while silently discarding that staged-but-uncommitted outer row too.
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

    # Stand-in for the manager agent create_workforce_from_template flushes
    # into its outer transaction before resolving any worker agents -
    # staged (flushed) but deliberately never committed here, matching the
    # caller's real sequencing.
    outer_manager = Agent(
        user_id=user.id,
        name="Growth Marketing Manager",
        instructions="orchestrator",
        execution_mode="think",
        origin=AgentOrigin.WORKFORCE_GENERATED_MANAGER.value,
        status=AgentStatus.PUBLISHED,
    )
    db_session.add(outer_manager)
    db_session.flush()
    outer_manager_id = outer_manager.id

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

    # The staged outer row must still be visible in-session after the
    # SAVEPOINT recovery, and must still be there once the caller's outer
    # transaction actually commits.
    assert (
        db_session.query(Agent).filter(Agent.id == outer_manager_id).first() is not None
    )
    db_session.commit()
    assert (
        db_session.query(Agent).filter(Agent.id == outer_manager_id).first() is not None
    )


def test_race_retry_recovers_from_a_genuine_name_collision(db_session) -> None:
    """A concurrent, unrelated agent create (different origin, no
    template_id) can win a race for the exact name
    resolve_unique_agent_name just certified as free, in the window
    between that check and our own INSERT - a real (user_id, name)
    collision, not this template's (user_id, template_id) quick-access
    race. The resolver must recognize this via is_agent_name_unique_violation
    and retry (the next resolve_unique_agent_name call picks a fresh name),
    not misdiagnose it as the quick-access race - whose re-select would
    find nothing here and burn every retry on a 409 that retrying can
    never fix.

    A first attempt at this test tried to simulate the collision by
    inserting the competing row from inside the mocked
    resolve_unique_agent_name call - but that call happens *inside* the
    resolver's own `db.begin_nested()`, so the SAVEPOINT rollback undid
    that competing row right along with the failed insert, and the retry
    then succeeded with the original, undisambiguated name (proving the
    retry mechanism works, but not testing this branch's classification
    at all). Raising a pre-built IntegrityError directly is the reliable
    way to exercise the classification logic without depending on
    SAVEPOINT/rollback timing.
    """
    user = _create_user(db_session)
    real_add_agent = AgentStore.add_agent
    call_count = {"n": 0}

    def _fail_once_with_name_violation(self, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise IntegrityError(
                "INSERT INTO agents (...)",
                {},
                Exception("UNIQUE constraint failed: agents.user_id, agents.name"),
            )
        return real_add_agent(self, **kwargs)

    with patch.object(AgentStore, "add_agent", _fail_once_with_name_violation):
        resolved = asyncio.run(
            _get_or_create_quick_access_worker_agent(
                db_session,
                _FakeTemplateManager(),
                user_id=user.id,
                template_id="ga_analyzer",
            )
        )

    assert call_count["n"] == 2
    assert resolved.origin == AgentOrigin.TEMPLATE_QUICK_ACCESS.value
    assert resolved.template_id == "ga_analyzer"
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
