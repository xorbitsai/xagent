"""At run time only the runner's own OAuth grant is used on a team connector.

A team agent's connector is visible to everyone who runs the agent, but the
bearer token sent to it comes from a grant the runner made for themselves. A
runner without one gets an authorization-required diagnostic, never another
member's token, and a task that did not come from the runner's own web session,
personal SDK key or own trigger never uses a member-made grant at all.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any
from urllib.parse import parse_qs, urlparse

import pytest
from sqlalchemy import text

from tests.web.api.test_mcp_oauth_team_member import _discovery, _request
from tests.web.services.test_team_member_grant_task_gate import (  # noqa: F401
    _MCP_URL,
    _MEMBER_TOKEN,
    _TEAM_ID,
    _build_tools_and_load_mcp,
    _task,
    _team_connector_fixture,
    _user,
)
from xagent.core.utils.encryption import encrypt_value
from xagent.web.api import mcp as mcp_api
from xagent.web.api.mcp import (
    MCPOAuthConnectRequest,
    connect_mcp_oauth,
    mcp_oauth_callback,
)
from xagent.web.models.database import Base, get_db, get_engine, init_db
from xagent.web.models.mcp import UserMCPServer
from xagent.web.models.mcp_oauth import MCPOAuthClient, MCPOAuthGrant
from xagent.web.models.user import User
from xagent.web.services import connector_team_scope
from xagent.web.services.agent_team_scope import (
    AgentTeamScope,
    set_agent_team_scope_hook,
)

_CREATOR_TOKEN = "creator-own-access-token"


@pytest.fixture()
def gate_db(tmp_path) -> Iterator[Any]:
    init_db(db_url=f"sqlite:///{tmp_path / 'member_grant_runtime.db'}")
    db = next(get_db())
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=get_engine())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "runner",
    [
        "member_with_grant",
        "member_without_grant",
        "creator",
        "other_member_without_grant",
        "member_with_grant_widget_task",
    ],
)
async def test_runtime_uses_only_runner_own_grant(gate_db, runner) -> None:
    ids = _team_connector_fixture(gate_db)
    server_id = ids["server_id"]
    member_id = ids["member_id"]
    other_member = _user(gate_db, "gate-other-member")
    creator_id = int(
        gate_db.execute(
            text("SELECT user_id FROM agents WHERE id = :id"),
            {"id": ids["agent_id"]},
        ).scalar_one()
    )
    # The creator holds the connector through a personal association and
    # their own grant.
    gate_db.add(
        UserMCPServer(
            user_id=creator_id,
            mcpserver_id=server_id,
            is_owner=True,
            is_active=True,
        )
    )
    client = gate_db.query(MCPOAuthClient).one()
    gate_db.add(
        MCPOAuthGrant(
            mcp_server_id=server_id,
            user_id=creator_id,
            mcp_oauth_client_id=client.id,
            resource_owner_key=f"xagent:user:{creator_id}",
            issuer=client.issuer,
            resource="https://mcp.gate.test/mcp",
            scope="",
            access_token=encrypt_value(_CREATOR_TOKEN),
            status="active",
        )
    )
    gate_db.commit()

    runner_id = {
        "member_with_grant": member_id,
        "member_with_grant_widget_task": member_id,
        "member_without_grant": int(other_member.id),
        "other_member_without_grant": int(other_member.id),
        "creator": creator_id,
    }[runner]
    if runner == "member_without_grant":
        # A member who never authorized: drop the grant the fixture made.
        gate_db.query(MCPOAuthGrant).filter(MCPOAuthGrant.user_id == member_id).delete()
        gate_db.commit()
        runner_id = member_id
    source = "widget" if runner == "member_with_grant_widget_task" else "internal"

    def _team_visibility(db, *, team_id: int) -> dict[str, set[int]]:
        if team_id == _TEAM_ID:
            return {"mcp": {server_id}, "custom_api": set()}
        return {"mcp": set(), "custom_api": set()}

    set_agent_team_scope_hook(
        lambda db, user_id: AgentTeamScope(team_id=_TEAM_ID, is_team_admin=False)
    )
    connector_team_scope.set_connector_team_hooks(team_visibility=_team_visibility)
    try:
        task = _task(gate_db, runner_id, source=source, agent_id=ids["agent_id"])
        configs, diagnostics, _ = await _build_tools_and_load_mcp(
            gate_db, task=task, member_id=runner_id
        )
    finally:
        set_agent_team_scope_hook(None)
        connector_team_scope.set_connector_team_hooks()

    payload = json.dumps(configs, default=str)
    if runner == "member_with_grant":
        assert diagnostics == []
        assert _MEMBER_TOKEN in payload
        assert _CREATOR_TOKEN not in payload
    elif runner == "creator":
        assert _CREATOR_TOKEN in payload
        assert _MEMBER_TOKEN not in payload
    elif runner == "member_with_grant_widget_task":
        # The grant exists; the task's source is what withholds it.
        assert [d.get("code") for d in diagnostics] == [
            "grant_not_allowed_for_task_source"
        ]
        assert _MEMBER_TOKEN not in payload
        assert _CREATOR_TOKEN not in payload
    else:
        assert [d.get("code") for d in diagnostics] == ["authorization_required"]
        assert _MEMBER_TOKEN not in payload
        assert _CREATOR_TOKEN not in payload


_LOGIN_TOKEN = "member-login-access-token"


@pytest.mark.asyncio
async def test_grant_from_member_login_is_used_by_web_chat_but_not_widget_task(
    gate_db, monkeypatch
) -> None:
    ids = _team_connector_fixture(gate_db)
    server_id = ids["server_id"]
    member_id = ids["member_id"]
    # Start from a member with no grant: the only grant used below is the one
    # the member login writes.
    gate_db.query(MCPOAuthGrant).filter(MCPOAuthGrant.user_id == member_id).delete()
    gate_db.commit()

    async def fake_discover(*_args: Any, **_kwargs: Any) -> Any:
        return _discovery(_MCP_URL)

    async def fake_register(_server: Any, *, redirect_uri: str) -> Any:
        return SimpleNamespace(
            client_id="member-login-client", token_endpoint_auth_method="none"
        )

    async def fake_exchange(**_kwargs: Any) -> dict[str, str]:
        return {"access_token": _LOGIN_TOKEN, "token_type": "Bearer"}

    monkeypatch.setattr(mcp_api, "discover_mcp_oauth_metadata", fake_discover)
    monkeypatch.setattr(mcp_api, "register_mcp_oauth_public_client", fake_register)
    monkeypatch.setattr(mcp_api, "_exchange_mcp_oauth_code", fake_exchange)

    def _member_visibility(db: Any, user_id: int) -> dict[str, set[int]]:
        if int(user_id) != member_id:
            return {"mcp": set(), "custom_api": set(), "member_oauth_mcp": set()}
        return {
            "mcp": {server_id},
            "custom_api": set(),
            "member_oauth_mcp": {server_id},
        }

    def _team_visibility(db: Any, *, team_id: int) -> dict[str, set[int]]:
        if team_id == _TEAM_ID:
            return {"mcp": {server_id}, "custom_api": set()}
        return {"mcp": set(), "custom_api": set()}

    set_agent_team_scope_hook(
        lambda db, user_id: AgentTeamScope(team_id=_TEAM_ID, is_team_admin=False)
    )
    connector_team_scope.set_connector_team_hooks(
        visibility=_member_visibility, team_visibility=_team_visibility
    )
    try:
        member = gate_db.query(User).filter(User.id == member_id).one()
        # The member holds no association row, so this is the member login.
        assert (
            gate_db.query(UserMCPServer)
            .filter(UserMCPServer.user_id == member_id)
            .count()
            == 0
        )
        started = await connect_mcp_oauth(
            server_id,
            MCPOAuthConnectRequest(redirect_after="/tools"),
            member,
            gate_db,
        )
        assert started.status_code == 303
        state = parse_qs(urlparse(started.headers["location"]).query)["state"][0]
        gate_db.expire_all()
        finished = await mcp_oauth_callback(
            _request(f"/api/mcp/oauth/callback?code=auth-code&state={state}"),
            gate_db,
        )
        assert "mcp_oauth_success=1" in finished.headers["location"]
        gate_db.expire_all()
        grant = gate_db.query(MCPOAuthGrant).one()
        assert int(grant.user_id) == member_id
        assert grant.resource_owner_key == f"xagent:user:{member_id}"

        web_task = _task(
            gate_db, member_id, source="internal", agent_id=ids["agent_id"]
        )
        widget_task = _task(
            gate_db, member_id, source="widget", agent_id=ids["agent_id"]
        )
        web_configs, web_diagnostics, _ = await _build_tools_and_load_mcp(
            gate_db, task=web_task, member_id=member_id
        )
        widget_configs, widget_diagnostics, _ = await _build_tools_and_load_mcp(
            gate_db, task=widget_task, member_id=member_id
        )
    finally:
        set_agent_team_scope_hook(None)
        connector_team_scope.set_connector_team_hooks()

    assert web_diagnostics == []
    assert _LOGIN_TOKEN in json.dumps(web_configs, default=str)
    assert [d.get("code") for d in widget_diagnostics] == [
        "grant_not_allowed_for_task_source"
    ]
    assert _LOGIN_TOKEN not in json.dumps(widget_configs, default=str)
