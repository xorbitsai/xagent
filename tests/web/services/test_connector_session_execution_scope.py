"""The connector session under a task's execution scope (#3742).

A caller that holds a persisted, leased task (the Toby connector checks,
xorbitsai/xagent-saas#2553) passes the task's ``ExecutionScope`` along with
its execution identity. The session hands the scope to the task's config
unchanged, so an embedder's Slack actor runtime grant resolver can check it
against the task the identity names, on the first connection and on every
call's refresh. Without a scope the resolver keeps receiving None.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from tests.web.services.test_mcp_connector_session import (
    LOADER,
    SANDBOX_MANAGER,
    _comparable,
    _connect_chrome,
    _Loader,
)
from xagent.core.execution_scope import ExecutionScope
from xagent.core.tools.adapters.vibe.mcp_adapter import MCPLoadResult
from xagent.core.tools.adapters.vibe.mcp_tools import create_mcp_tools
from xagent.web.builtin_mcp_registry import seed_builtin_oauth_and_public_mcp_apps
from xagent.web.mcp_apps import ensure_builtin_oauth_server_definition
from xagent.web.models.database import Base
from xagent.web.models.mcp import MCPServer, UserMCPServer
from xagent.web.models.user import User
from xagent.web.services.mcp_connector_session import (
    ConnectorToolsOutcome,
    list_connector_tools,
)
from xagent.web.services.mcp_runtime import (
    MCPActorAuthorizationPolicy,
    MCPActorExecutionIdentity,
)
from xagent.web.services.slack_actor_runtime import (
    SLACK_ACTOR_RUNTIME_REFRESH_KEY,
    SlackActorRuntimeGrant,
    SlackActorRuntimeGrantRequest,
    SlackChannelAccessPolicy,
    set_slack_actor_runtime_grant_resolver,
)
from xagent.web.tools.config import (
    TokenRequest,
    WebToolConfig,
    set_oauth_token_resolver_hook,
)

OWNER = "toby:slack:T1:U1"
IDENTITY = MCPActorExecutionIdentity(
    task_id=7, run_id="run-7", turn_id="turn-7", lease_attempt_id="attempt-7"
)


def _policy() -> MCPActorAuthorizationPolicy:
    return MCPActorAuthorizationPolicy(resource_owner_key=OWNER)


@pytest.fixture()
def db_user(tmp_path):
    """The seeded catalog plus one plain user, as in the session tests."""
    engine = create_engine(
        f"sqlite:///{tmp_path / 'connector-scope.db'}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    with engine.begin() as connection:
        seed_builtin_oauth_and_public_mcp_apps(connection)
    db = sessionmaker(bind=engine, autoflush=False, autocommit=False)()
    user = User(username="monitor", password_hash="x", is_admin=False)
    db.add(user)
    db.commit()
    db.refresh(user)
    yield db, user
    db.close()
    engine.dispose()


@pytest.fixture(autouse=True)
def _no_sandbox_manager(monkeypatch):
    monkeypatch.setattr(SANDBOX_MANAGER, lambda: None)


@pytest.fixture(autouse=True)
def _reset_resolvers():
    yield
    set_slack_actor_runtime_grant_resolver(None)
    set_oauth_token_resolver_hook(None)


class _GrantResolver:
    """An embedder's Slack grant resolver: records each request and answers
    every one with the same read-only bot grant."""

    def __init__(self) -> None:
        self.requests: list[SlackActorRuntimeGrantRequest] = []
        self._grant = SlackActorRuntimeGrant(
            access_token="bot-token",
            channel_access=SlackChannelAccessPolicy(
                channel_ids=frozenset({"C0123456789"}),
                expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
                capabilities=frozenset({"read"}),
            ),
        )

    def __call__(self, request: SlackActorRuntimeGrantRequest) -> Any:
        self.requests.append(request)
        return self._grant


def _connect_catalog(db, user: User, app_id: str) -> MCPServer:
    """The canonical catalog row, linked the way the connect routes link it
    (no ownership), with no credential stored for the user."""
    server = ensure_builtin_oauth_server_definition(db, app_id=app_id)
    db.commit()
    db.add(
        UserMCPServer(
            user_id=user.id, mcpserver_id=server.id, is_owner=False, is_active=True
        )
    )
    db.commit()
    return server


async def _list_with_the_actor_identity(
    db, user: User, server: MCPServer, **keywords: Any
):
    return await list_connector_tools(
        db,
        user=user,
        server_id=server.id,
        name_prefix="mcp_",
        mcp_runtime_authorization_policy=_policy(),
        mcp_actor_execution_identity=IDENTITY,
        **keywords,
    )


@pytest.mark.parametrize("scoped", [True, False])
@pytest.mark.asyncio
async def test_slack_actor_grant_gets_the_given_scope_for_the_connection_and_each_call(
    db_user, monkeypatch, scoped
):
    db, user = db_user
    server = _connect_catalog(db, user, "slack")
    resolver = _GrantResolver()
    set_slack_actor_runtime_grant_resolver(resolver)
    loader = _Loader()
    monkeypatch.setattr(LOADER, loader)
    scope = ExecutionScope(sandbox_key_suffix="probe-task") if scoped else None

    result = await _list_with_the_actor_identity(
        db, user, server, **({"execution_scope": scope} if scoped else {})
    )

    assert result.outcome is ConnectorToolsOutcome.OK
    ((connections, _),) = loader.calls
    connection = connections[server.name]
    assert connection["env"]["SLACK_ACCESS_TOKEN"] == "bot-token"
    # A tool call resolves the grant again before it runs (mcp_adapter's
    # _invocation_connection); the refresh is what it calls.
    refreshed = await connection[SLACK_ACTOR_RUNTIME_REFRESH_KEY]()
    assert refreshed is not None
    assert refreshed["env"]["SLACK_ACCESS_TOKEN"] == "bot-token"

    first, per_call = resolver.requests
    # The same object, not an equal copy: an embedder may key on identity or
    # on payload the scope's equality does not compare.
    assert first.scope is scope
    assert per_call.scope is scope
    assert first.execution_identity == per_call.execution_identity == IDENTITY
    assert first.resource_owner_key == per_call.resource_owner_key == OWNER


@pytest.mark.asyncio
async def test_slack_on_the_actor_grant_starts_like_the_task(db_user, monkeypatch):
    db, user = db_user
    server = _connect_catalog(db, user, "slack")
    resolver = _GrantResolver()
    set_slack_actor_runtime_grant_resolver(resolver)
    scope = ExecutionScope(sandbox_key_suffix="probe-task")
    task_loader = _Loader(MCPLoadResult(tools=(), loaded_servers=(), failures=()))
    monkeypatch.setattr(LOADER, task_loader)
    await create_mcp_tools(
        WebToolConfig(
            db=db,
            request=None,
            user=user,
            user_id=user.id,
            mcp_runtime_authorization_policy=_policy(),
            mcp_actor_execution_identity=IDENTITY,
            execution_scope=scope,
        )
    )
    ((task_connections, task_kwargs),) = task_loader.calls

    loader = _Loader()
    monkeypatch.setattr(LOADER, loader)
    result = await _list_with_the_actor_identity(
        db, user, server, execution_scope=scope
    )

    assert result.outcome is ConnectorToolsOutcome.OK
    ((connections, kwargs),) = loader.calls
    assert _comparable(connections[server.name]) == _comparable(
        task_connections[server.name]
    )
    assert kwargs["connector_refs"] == task_kwargs["connector_refs"]
    # One resolution for the task's build and one for the session's.
    assert len(resolver.requests) == 2
    assert all(request.scope is scope for request in resolver.requests)


@pytest.mark.asyncio
async def test_token_resolver_hook_gets_the_scope_without_an_actor_policy(db_user):
    db, user = db_user
    server = _connect_catalog(db, user, "github")
    requests: list[TokenRequest] = []

    def hook(request: TokenRequest) -> None:
        requests.append(request)
        return None

    set_oauth_token_resolver_hook(hook)
    scope = ExecutionScope(sandbox_key_suffix="probe-task")

    result = await list_connector_tools(
        db, user=user, server_id=server.id, name_prefix="mcp_", execution_scope=scope
    )

    assert result.outcome is ConnectorToolsOutcome.NOT_CONNECTED
    assert requests
    assert all(request.scope is scope for request in requests)


@pytest.mark.asyncio
async def test_a_scope_does_not_lift_the_execution_scoped_refusal(db_user, monkeypatch):
    db, user = db_user
    server = _connect_chrome(db, user)
    loader = _Loader()
    monkeypatch.setattr(LOADER, loader)

    result = await list_connector_tools(
        db,
        user=user,
        server_id=server.id,
        name_prefix="mcp_",
        mcp_runtime_authorization_policy=_policy(),
        mcp_actor_execution_identity=IDENTITY,
        execution_scope=ExecutionScope(sandbox_key_suffix="probe-task"),
    )

    assert result.outcome is ConnectorToolsOutcome.UNSUPPORTED
    assert result.reason == "execution_scoped"
    assert loader.calls == []
