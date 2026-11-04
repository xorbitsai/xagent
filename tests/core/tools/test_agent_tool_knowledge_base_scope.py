"""A delegated sub-agent resolves its declared knowledge bases against the
team that owns the agent being delegated to -- the same answer a directly
run agent gets -- not against whoever happens to be running the delegation.

Each test drives the real ``AgentTool`` delegation path against a real
SQLite database and captures the ``WebToolConfig`` it builds, stopping the
run right after that point.

Workforce delegation reaches a worker through the run snapshot's allowlist
without re-checking the runner's current access to it, so the governing
team is only granted on that branch while the runner can still run the
agent directly; otherwise the delegated run gets no knowledge tools at all.
"""

from __future__ import annotations

from typing import Any, Optional

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import xagent.core.tools.adapters.vibe.agent_tool as agent_tool_module
from xagent.core.tools.adapters.vibe import knowledge_tools
from xagent.core.tools.adapters.vibe.agent_tool import AgentTool
from xagent.core.tools.adapters.vibe.document_search import KnowledgeSearchTool
from xagent.web.models import Base, User
from xagent.web.models.agent import Agent, AgentStatus
from xagent.web.services import agent_team_scope
from xagent.web.services.agent_team_scope import AgentTeamScope
from xagent.web.services.workforce_access import (
    WorkforcePolicy,
    set_workforce_policy,
)

T1 = 101
T2 = 202


class _Stop(Exception):
    """Halt the delegated run right after its WebToolConfig is captured."""


@pytest.fixture(autouse=True)
def _reset_access_hooks():
    yield
    agent_team_scope.set_agent_team_scope_hook(None)
    set_workforce_policy(WorkforcePolicy())


def _seed(
    *,
    team_id: Optional[int],
    knowledge_bases: Optional[list[str]],
    visibility: str = "team",
    status: AgentStatus = AgentStatus.PUBLISHED,
) -> tuple[Any, int, int, int]:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(bind=engine)
    with session_factory() as db:
        creator = User(username="kb-agent-creator", password_hash="hash")
        runner = User(username="kb-delegation-runner", password_hash="hash")
        db.add_all([creator, runner])
        db.flush()
        agent = Agent(
            user_id=creator.id,
            team_id=team_id,
            visibility=visibility,
            name="KB Agent",
            instructions="answer from the knowledge base",
            status=status,
            execution_mode="balanced",
            models={"general": 1},
            knowledge_bases=knowledge_bases,
            skills=[],
            tool_categories=["knowledge"],
        )
        db.add(agent)
        db.commit()
        return session_factory, int(agent.id), int(creator.id), int(runner.id)


async def _delegate(
    monkeypatch: pytest.MonkeyPatch,
    *,
    session_factory: Any,
    agent_id: int,
    runner_id: int,
    workforce: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import xagent.core.tools.adapters.vibe.agent_model_resolution as resolution

    monkeypatch.setattr(
        resolution,
        "resolve_agent_model_llms",
        lambda *_args: (object(), None, None, None),
    )

    captured: dict[str, Any] = {}
    real_web_tool_config_cls = agent_tool_module.WebToolConfig

    class _CapturingWebToolConfig(real_web_tool_config_cls):  # type: ignore[misc,valid-type]
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            captured["instance"] = self

    monkeypatch.setattr(agent_tool_module, "WebToolConfig", _CapturingWebToolConfig)

    import xagent.core.agent.service as service_module

    class _StoppingAgentService:
        def __init__(self, **_kwargs: Any) -> None:
            return None

        async def execute_task(self, **_kwargs: Any) -> Any:
            raise _Stop()

    monkeypatch.setattr(service_module, "AgentService", _StoppingAgentService)

    extra: dict[str, Any] = {}
    if workforce:
        # The shape a workforce run gives its worker tools: an explicit
        # allowlist from the run snapshot plus cross-user execution, which
        # skips the live ownership predicate in the delegation query.
        extra = {
            "target_allowed_agent_ids": [agent_id],
            "target_allow_cross_user_agent_ids": True,
        }
    tool = AgentTool(
        agent_id=agent_id,
        agent_name="KB Agent",
        agent_description="delegated",
        session_factory=session_factory,
        user_id=runner_id,
        tool_name="kb_agent",
        tool_description="delegated",
        **extra,
    )
    monkeypatch.setattr(tool, "_create_child_execution_tracer", lambda **_kwargs: None)

    result = await tool.run_json_async({"task": "run"})
    return captured, result


async def _search_kwargs(
    monkeypatch: pytest.MonkeyPatch, tool_config: Any
) -> dict[str, Any]:
    """Run the knowledge search tool built from ``tool_config`` through the
    real compatibility facade, intercepting only the innermost core
    function, and return what reached it."""
    from xagent.core.tools.core import document_search as core_document_search

    reached: dict[str, Any] = {}

    async def _fake_search_impl(tool_args: Any, **kwargs: Any) -> Any:
        reached.update(kwargs)
        return core_document_search.KnowledgeSearchResult(results=[], summary="ok")

    monkeypatch.setattr(
        core_document_search, "_search_knowledge_base_impl", _fake_search_impl
    )
    tools = await knowledge_tools._create_knowledge_tools_impl(tool_config)
    search_tool = next(t for t in tools if isinstance(t, KnowledgeSearchTool))
    await search_tool.run_json_async({"query": "hello"})
    return reached


@pytest.mark.asyncio
async def test_team_agent_delegation_resolves_against_the_agents_team(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_factory, agent_id, creator_id, _ = _seed(
        team_id=T1, knowledge_bases=["kb1"]
    )

    captured, _ = await _delegate(
        monkeypatch,
        session_factory=session_factory,
        agent_id=agent_id,
        runner_id=creator_id,
    )

    tool_config = captured["instance"]
    assert tool_config.get_knowledge_base_team_id() == T1
    assert tool_config.get_agent_creator_user_id() == creator_id
    assert tool_config.get_declared_knowledge_bases() == ["kb1"]
    # Connector visibility stays closed on this path: the two decisions are
    # carried by separate fields and differ here by construction.
    assert tool_config._connector_team_id is None

    reached = await _search_kwargs(monkeypatch, tool_config)
    assert reached["governing_team_id"] == T1
    assert reached["agent_creator_user_id"] == creator_id
    assert reached["declared_knowledge_bases"] == ["kb1"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("team_id", "knowledge_bases"),
    [
        # A personal agent has no governing team to resolve against.
        (None, ["kb1"]),
        # No stored declaration at all: a governing team would open the
        # "search everything visible" branch onto the team's whole set,
        # which the direct-run path never does for this agent (it builds no
        # knowledge tool for it), so no team is granted.
        (T1, None),
    ],
)
async def test_delegation_grants_no_governing_team_without_team_and_declaration(
    monkeypatch: pytest.MonkeyPatch,
    team_id: Optional[int],
    knowledge_bases: Optional[list[str]],
) -> None:
    session_factory, agent_id, creator_id, _ = _seed(
        team_id=team_id, knowledge_bases=knowledge_bases
    )

    captured, _ = await _delegate(
        monkeypatch,
        session_factory=session_factory,
        agent_id=agent_id,
        runner_id=creator_id,
    )

    tool_config = captured["instance"]
    assert tool_config.get_knowledge_base_team_id() is None
    assert tool_config.get_allowed_collections() == knowledge_bases


class _VisibleAgentPolicy(WorkforcePolicy):
    def __init__(self, agent_id: int) -> None:
        self._agent_id = agent_id

    def get_visible_agent_ids(self, db: Any, user: Any, purpose: str) -> set[int]:
        return {self._agent_id}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "access", ["team_member", "team_member_draft_agent", "published_and_visible"]
)
async def test_workforce_delegation_with_current_access_uses_agents_team(
    monkeypatch: pytest.MonkeyPatch, access: str
) -> None:
    # A draft agent is admitted only through the team-visible ownership
    # branch, never through the published-and-visible one.
    session_factory, agent_id, creator_id, runner_id = _seed(
        team_id=T1,
        knowledge_bases=["kb1"],
        status=(
            AgentStatus.DRAFT
            if access == "team_member_draft_agent"
            else AgentStatus.PUBLISHED
        ),
    )
    if access in ("team_member", "team_member_draft_agent"):
        agent_team_scope.set_agent_team_scope_hook(
            lambda _db, user_id: AgentTeamScope(team_id=T1, is_team_admin=False)
        )
    else:
        # Not a member of the agent's team, but the agent is published and
        # visible to the runner through the application's visibility
        # policy -- the second way the direct-run path admits an agent.
        set_workforce_policy(_VisibleAgentPolicy(agent_id))

    captured, _ = await _delegate(
        monkeypatch,
        session_factory=session_factory,
        agent_id=agent_id,
        runner_id=runner_id,
        workforce=True,
    )

    tool_config = captured["instance"]
    assert tool_config.get_knowledge_base_team_id() == T1
    # The agent's creator, never the runner.
    assert tool_config.get_agent_creator_user_id() == creator_id
    assert tool_config.get_allowed_collections() == ["kb1"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("agent_team_id", "runner_scope", "visibility"),
    [
        # The runner left the agent's team after the workforce run started.
        (T1, None, "team"),
        # The agent was moved to a team the runner has never belonged to.
        (T2, AgentTeamScope(team_id=T1, is_team_admin=False), "team"),
        # The agent was narrowed to team admins; the runner is not one.
        (T1, AgentTeamScope(team_id=T1, is_team_admin=False), "admins"),
    ],
)
async def test_workforce_delegation_without_current_access_gets_no_knowledge_tools(
    monkeypatch: pytest.MonkeyPatch,
    agent_team_id: int,
    runner_scope: Optional[AgentTeamScope],
    visibility: str,
) -> None:
    session_factory, agent_id, _, runner_id = _seed(
        team_id=agent_team_id, knowledge_bases=["kb1"], visibility=visibility
    )
    agent_team_scope.set_agent_team_scope_hook(
        lambda _db, user_id: runner_scope if user_id == runner_id else None
    )

    captured, _ = await _delegate(
        monkeypatch,
        session_factory=session_factory,
        agent_id=agent_id,
        runner_id=runner_id,
        workforce=True,
    )

    tool_config = captured["instance"]
    assert tool_config.get_knowledge_base_team_id() is None
    assert tool_config.get_allowed_collections() == []
    assert await knowledge_tools._create_knowledge_tools_impl(tool_config) == []


@pytest.mark.asyncio
async def test_workforce_delegation_fails_when_the_access_check_cannot_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_factory, agent_id, _, runner_id = _seed(team_id=T1, knowledge_bases=["kb1"])

    def _broken_scope(_db: Any, _user_id: int) -> AgentTeamScope:
        raise RuntimeError("team scope unavailable")

    agent_team_scope.set_agent_team_scope_hook(_broken_scope)

    captured, result = await _delegate(
        monkeypatch,
        session_factory=session_factory,
        agent_id=agent_id,
        runner_id=runner_id,
        workforce=True,
    )

    # A failed access check is never read as "has access": the delegation
    # fails before any tool config is built.
    assert result["success"] is False
    assert "team scope unavailable" in str(result)
    assert "instance" not in captured
