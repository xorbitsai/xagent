"""Delegated sub-agents build their ``WebToolConfig`` with
``connector_team_id=None`` (``agent_tool.py``), a deliberate scoping
decision -- delegated identity is restored from a persisted workforce
snapshot with no re-authorization at consumption, so opening team connector
visibility there would grant a team's connectors off a snapshot nobody
re-checks. This test exercises the real construction path (not a
reconstruction of it) and pins the consequence: a delegated agent whose
tool categories name a server that exists only through a team grant
resolves to an empty MCP tool set, silently, rather than raising or
executing the team's tools.
"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import xagent.core.tools.adapters.vibe.agent_tool as agent_tool_module
from xagent.core.tools.adapters.vibe.agent_tool import AgentTool
from xagent.web.models import Base, MCPServer, User
from xagent.web.models.agent import Agent, AgentStatus
from xagent.web.services import connector_team_scope

T1 = 101


class _Stop(Exception):
    """Halt the delegated run right after its WebToolConfig is captured."""


@pytest.fixture(autouse=True)
def _reset_hooks():
    yield
    connector_team_scope.set_connector_team_hooks()


@pytest.mark.asyncio
async def test_delegated_spec_naming_team_only_server_resolves_empty(monkeypatch):
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(bind=engine)

    with session_factory() as seed_db:
        owner = User(username="delegated-owner", password_hash="hash")
        seed_db.add(owner)
        seed_db.flush()
        # No personal UserMCPServer link for owner -- reachable only through
        # the team hook installed below.
        team_server = MCPServer(
            name="team-only-delegated-probe",
            managed="external",
            transport="streamable_http",
            url="https://example.com/mcp",
        )
        seed_db.add(team_server)
        seed_db.flush()
        agent = Agent(
            user_id=owner.id,
            name="Delegated Agent",
            instructions="be terse",
            status=AgentStatus.PUBLISHED,
            execution_mode="balanced",
            models={"general": 1},
            knowledge_bases=[],
            skills=[],
            tool_categories=["mcp"],
        )
        seed_db.add(agent)
        seed_db.commit()
        agent_id = int(agent.id)
        owner_id = int(owner.id)
        team_server_id = int(team_server.id)

    # A team hook that would grant the team-only server to team T1 -- proves
    # the empty result below is because of the delegated scoping decision,
    # not because no hook is installed at all.
    connector_team_scope.set_connector_team_hooks(
        team_visibility=lambda db, *, team_id: (
            {"mcp": {team_server_id}, "custom_api": set()}
            if team_id == T1
            else {"mcp": set(), "custom_api": set()}
        )
    )

    import xagent.core.tools.adapters.vibe.agent_model_resolution as resolution

    monkeypatch.setattr(
        resolution,
        "resolve_agent_model_llms",
        lambda *_args: (object(), None, None, None),
    )

    captured: dict[str, object] = {}
    real_web_tool_config_cls = agent_tool_module.WebToolConfig

    class _CapturingWebToolConfig(real_web_tool_config_cls):
        def __init__(self, **kwargs):
            captured["connector_team_id"] = kwargs.get("connector_team_id", "OMITTED")
            super().__init__(**kwargs)
            captured["instance"] = self

    monkeypatch.setattr(agent_tool_module, "WebToolConfig", _CapturingWebToolConfig)

    import xagent.core.agent.service as service_module

    class _StoppingAgentService:
        def __init__(self, **_kwargs):
            return None

        async def execute_task(self, **_kwargs):
            # The WebToolConfig is already built and captured by this point
            # (agent_tool.py constructs it before handing off to
            # AgentService); stop here rather than running a real turn.
            raise _Stop()

    monkeypatch.setattr(service_module, "AgentService", _StoppingAgentService)

    tool = AgentTool(
        agent_id=agent_id,
        agent_name="Delegated Agent",
        agent_description="delegated",
        session_factory=session_factory,
        user_id=owner_id,
        tool_name="delegated",
        tool_description="delegated",
    )
    monkeypatch.setattr(tool, "_create_child_execution_tracer", lambda **_kwargs: None)

    result = await tool.run_json_async({"task": "run"})

    # The delegated run itself fails closed (the stop sentinel surfaces as a
    # classified failure) -- that is incidental to this test, which cares
    # about the WebToolConfig construction captured along the way.
    assert result["success"] is False

    assert captured["connector_team_id"] is None
    tool_config = captured["instance"]
    assert tool_config._connector_team_id is None

    configs = await tool_config.get_mcp_server_configs()
    assert configs == []
