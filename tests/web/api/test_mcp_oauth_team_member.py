"""A team member authorizes a shared OAuth connector with their own account.

The member holds no ``user_mcpservers`` row, so every flow here is a
``team_member`` flow: the flow row is the lifecycle anchor and the grant is
stored under the member. These tests drive the real route functions with a
stubbed user-keyed visibility hook, a stubbed discovery, and a stubbed token
exchange.
"""

from __future__ import annotations

import re
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import sessionmaker
from starlette.requests import Request

from xagent.core.utils.encryption import encrypt_value
from xagent.web.api import mcp as mcp_api
from xagent.web.api.mcp import (
    MCPOAuthConnectRequest,
    MCPServerUpdate,
    connect_mcp_oauth,
    mcp_oauth_callback,
    update_mcp_server,
)
from xagent.web.models import MCPOAuthClient, MCPOAuthFlowState, MCPOAuthGrant
from xagent.web.models.database import Base
from xagent.web.models.mcp import MCPServer, UserMCPServer
from xagent.web.models.user import User
from xagent.web.services import connector_team_scope
from xagent.web.services.mcp_oauth import select_mcp_oauth_grants

URL = "https://mcp.example.com/mcp"
ISSUER = "https://auth.example.com"


def _request(path: str) -> Request:
    parsed = urlparse(path)
    state = parse_qs(parsed.query).get("state", [None])[0]
    headers = []
    if state:
        headers.append(
            (
                b"cookie",
                (
                    f"{mcp_api.MCP_OAUTH_STATE_COOKIE}="
                    f"{mcp_api._mcp_oauth_state_cookie_value(state)}"
                ).encode(),
            )
        )
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": parsed.path,
            "query_string": parsed.query.encode(),
            "headers": headers,
        }
    )


def _discovery(resource: str = URL) -> SimpleNamespace:
    return SimpleNamespace(
        resource=resource,
        scopes=("records.read",),
        protected_resource=SimpleNamespace(authorization_servers=(ISSUER,)),
        authorization_server=SimpleNamespace(
            issuer=ISSUER,
            authorization_endpoint=f"{ISSUER}/authorize",
            token_endpoint=f"{ISSUER}/token",
            registration_endpoint=f"{ISSUER}/register",
            client_id_metadata_document_supported=False,
            raw={"issuer": ISSUER, "revocation_endpoint": f"{ISSUER}/revoke"},
        ),
    )


class Hook:
    """A user-keyed visibility hook answering for the member only."""

    def __init__(self, member_id: int, server_id: int) -> None:
        self.member_id = member_id
        self.visible: set[int] = {server_id}
        self.open: set[int] = {server_id}
        self.calls = 0
        self.fail_from_call: int | None = None
        self.end_transaction_from_call: int | None = None
        self.read_marker = False
        # (visible, open) answered from the second call on.
        self.from_second_call: tuple[set[int], set[int]] | None = None

    def __call__(self, db, user_id: int) -> dict[str, set[int]]:
        self.calls += 1
        if self.read_marker:
            db.execute(text("SELECT 'hook_read'"))
        if self.fail_from_call is not None and self.calls >= self.fail_from_call:
            raise RuntimeError("hook unavailable")
        if (
            self.end_transaction_from_call is not None
            and self.calls >= self.end_transaction_from_call
        ):
            db.commit()
        if self.from_second_call is not None and self.calls >= 2:
            visible, open_ = self.from_second_call
            return {
                "mcp": set(visible),
                "custom_api": set(),
                "member_oauth_mcp": set(open_),
            }
        if int(user_id) != self.member_id:
            return {"mcp": set(), "custom_api": set(), "member_oauth_mcp": set()}
        return {
            "mcp": set(self.visible),
            "custom_api": set(),
            "member_oauth_mcp": set(self.open),
        }


class Env:
    def __init__(self, tmp_path, monkeypatch) -> None:
        self.engine = create_engine(
            f"sqlite:///{tmp_path / 'team-member.db'}",
            connect_args={"check_same_thread": False},
        )
        Base.metadata.create_all(self.engine)
        self.session_factory = sessionmaker(
            bind=self.engine, autoflush=False, autocommit=False
        )
        self.db = self.session_factory()
        self.creator = self._user("creator")
        self.member = self._user("member")
        self.other = self._user("other")
        self.server = self._server()
        # The edit helpers use this id rather than ``self.server``: that object
        # is the route session's own row, and touching it after the route rolled
        # back would reload it before the edit commits.
        self.server_id = int(self.server.id)
        for user, owner in ((self.creator, True), (self.other, False)):
            self.db.add(
                UserMCPServer(
                    user_id=user.id,
                    mcpserver_id=self.server.id,
                    is_owner=owner,
                    is_active=True,
                )
            )
        self.db.commit()
        self.hook = Hook(int(self.member.id), int(self.server.id))
        connector_team_scope.set_connector_team_hooks(visibility=self.hook)

        self.discovery_calls = 0
        self.register_calls = 0
        self.discovery_resource = URL
        self.on_register = None
        self.revoked: list[object] = []
        self.on_exchange = None

        async def fake_discover(*_args, **_kwargs):
            self.discovery_calls += 1
            return _discovery(self.discovery_resource)

        async def fake_register(_server, *, redirect_uri):
            self.register_calls += 1
            if self.on_register is not None:
                self.on_register()
            return SimpleNamespace(
                client_id="dynamic-client", token_endpoint_auth_method="none"
            )

        async def fake_exchange(**_kwargs):
            if self.on_exchange is not None:
                self.on_exchange()
            return {
                "access_token": "member-access-token",
                "token_type": "Bearer",
                "scope": "records.read",
            }

        async def record_revoke(snapshot):
            self.revoked.append(snapshot)

        monkeypatch.setattr(mcp_api, "discover_mcp_oauth_metadata", fake_discover)
        monkeypatch.setattr(mcp_api, "register_mcp_oauth_public_client", fake_register)
        monkeypatch.setattr(mcp_api, "_exchange_mcp_oauth_code", fake_exchange)
        monkeypatch.setattr(
            mcp_api, "_revoke_mcp_oauth_issued_token_externally", record_revoke
        )

    def _user(self, name: str) -> User:
        user = User(username=name, password_hash="x", is_admin=False)
        self.db.add(user)
        self.db.commit()
        self.db.refresh(user)
        return user

    def _server(self, **auth_overrides) -> MCPServer:
        auth = {
            "type": "mcp_oauth",
            "resource": URL,
            "issuer": ISSUER,
            "scope": "records.read",
            "client_secret": encrypt_value("shared-secret"),
        }
        auth.update(auth_overrides)
        server = MCPServer.from_config(
            {
                "name": "records",
                "managed": "external",
                "transport": "streamable_http",
                "url": URL,
                "auth": {"type": "mcp_oauth"},
            }
        )
        server.auth = auth
        self.db.add(server)
        self.db.commit()
        self.db.refresh(server)
        return server

    def edit_server(self, **fields) -> None:
        """Commit a definition change from a separate session."""
        with self.session_factory() as other_db:
            row = other_db.get(MCPServer, self.server_id)
            for key, value in fields.items():
                setattr(row, key, value)
            other_db.commit()

    def edit_auth(self, **changes) -> None:
        with self.session_factory() as other_db:
            row = other_db.get(MCPServer, self.server_id)
            auth = dict(row.auth)
            auth.update(changes)
            row.auth = auth
            other_db.commit()

    async def connect(self, user: User | None = None, *, server_id: int | None = None):
        self.db.expire_all()  # a request starts on a fresh session
        return await connect_mcp_oauth(
            server_id if server_id is not None else self.server.id,
            MCPOAuthConnectRequest(redirect_after="/tools"),
            user or self.member,
            self.db,
        )

    async def connect_state(self, user: User | None = None) -> str:
        response = await self.connect(user)
        assert response.status_code == 303
        return parse_qs(urlparse(response.headers["location"]).query)["state"][0]

    async def callback(self, state: str):
        self.db.expire_all()  # a request starts on a fresh session
        return await mcp_oauth_callback(
            _request(f"/api/mcp/oauth/callback?code=auth-code&state={state}"),
            self.db,
        )

    def default_key(self, user: User) -> str:
        return f"xagent:user:{user.id}"

    def add_member_grant(self) -> MCPOAuthGrant:
        client = MCPOAuthClient(
            mcp_server_id=self.server.id,
            issuer=ISSUER,
            authorization_endpoint=f"{ISSUER}/authorize",
            token_endpoint=f"{ISSUER}/token",
            client_id="dynamic-client",
            token_endpoint_auth_method="none",
            redirect_uri=mcp_api._default_mcp_oauth_redirect_uri(),
        )
        self.db.add(client)
        self.db.flush()
        grant = MCPOAuthGrant(
            mcp_server_id=self.server.id,
            user_id=self.member.id,
            mcp_oauth_client_id=client.id,
            resource_owner_key=self.default_key(self.member),
            issuer=ISSUER,
            resource=URL,
            scope="records.read",
            access_token=encrypt_value("member-token"),
            status="active",
        )
        self.db.add(grant)
        self.db.commit()
        return grant

    def selected_member_grants(self) -> list:
        self.db.expire_all()
        server = self.db.get(MCPServer, self.server.id)
        return select_mcp_oauth_grants(
            self.db,
            server_id=int(server.id),
            user_id=int(self.member.id),
            auth_config=dict(server.auth),
            resource_owner_key=self.default_key(self.member),
            allow_team_member_grants=True,
        )

    def close(self) -> None:
        self.db.close()
        self.engine.dispose()


@pytest.fixture()
def env(tmp_path, monkeypatch):
    environment = Env(tmp_path, monkeypatch)
    try:
        yield environment
    finally:
        connector_team_scope.set_connector_team_hooks()
        environment.close()


def _error_code(response) -> str:
    return parse_qs(urlparse(response.headers["location"]).query)["mcp_oauth_error"][0]


def _http_error(excinfo) -> tuple[int, object]:
    return excinfo.value.status_code, excinfo.value.detail


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scope_has_server", [True, False], ids=["scope_has_server", "scope_lacks_server"]
)
@pytest.mark.parametrize("association", ["active", "inactive"])
async def test_association_holder_never_takes_team_path(
    env, association, scope_has_server
):
    holder = env.other
    env.hook.member_id = int(holder.id)
    if not scope_has_server:
        env.hook.visible = set()
        env.hook.open = set()
    if association == "inactive":
        env.db.query(UserMCPServer).filter(UserMCPServer.user_id == holder.id).update(
            {UserMCPServer.is_active: False}
        )
        env.db.commit()

    if association == "inactive":
        with pytest.raises(HTTPException) as excinfo:
            await env.connect(holder)
        assert _http_error(excinfo) == (404, "MCP server not found")
        assert env.db.query(MCPOAuthFlowState).count() == 0
    else:
        response = await env.connect(holder)
        assert response.status_code == 303
        flow = env.db.query(MCPOAuthFlowState).one()
        assert flow.access_path == "association"
        assert flow.association_lifecycle_generation is not None
    # The overlay is never consulted for a caller who holds a row.
    assert env.hook.calls == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case",
    [
        "visible_open",
        "visible_not_open",
        "not_visible",
        "no_hook",
        "hook_raises_precheck",
        "visible_stdio",
        "visible_builtin_oauth",
    ],
)
async def test_team_member_connect_admission(env, case):
    server_id = int(env.server.id)
    if case == "visible_not_open":
        env.hook.open = set()
    elif case == "not_visible":
        env.hook.visible = set()
        env.hook.open = set()
    elif case == "no_hook":
        connector_team_scope.set_connector_team_hooks()
    elif case == "hook_raises_precheck":
        env.hook.fail_from_call = 1
    elif case in ("visible_stdio", "visible_builtin_oauth"):
        if case == "visible_stdio":
            other = MCPServer.from_config(
                {
                    "name": "local-records",
                    "managed": "external",
                    "transport": "stdio",
                    "command": "records-mcp",
                }
            )
        else:
            other = MCPServer.from_config(
                {
                    "name": "hosted-records",
                    "managed": "external",
                    "transport": "streamable_http",
                    "url": URL,
                    "auth": {"type": "oauth", "provider": "google"},
                }
            )
        env.db.add(other)
        env.db.commit()
        env.db.refresh(other)
        server_id = int(other.id)
        env.hook.visible = {server_id}
        env.hook.open = {server_id}

    expected = {
        "visible_open": 303,
        "visible_not_open": 403,
        "not_visible": 404,
        "no_hook": 404,
        "hook_raises_precheck": 503,
        "visible_stdio": 400,
        "visible_builtin_oauth": 400,
    }[case]
    if expected == 303:
        response = await env.connect(server_id=server_id)
        assert response.status_code == 303
        flow = env.db.query(MCPOAuthFlowState).one()
        assert flow.access_path == "team_member"
        assert flow.association_lifecycle_generation is None
        return
    with pytest.raises(HTTPException) as excinfo:
        await env.connect(server_id=server_id)
    status_code, detail = _http_error(excinfo)
    assert status_code == expected
    if case == "visible_not_open":
        assert detail == {
            "code": "member_authorization_not_enabled",
            "message": "This connector is not open to member authorization",
        }
    if case == "hook_raises_precheck":
        assert detail == "Connector team scope is unavailable."
    assert env.db.query(MCPOAuthFlowState).count() == 0
    assert env.discovery_calls == 0


@pytest.mark.asyncio
async def test_team_member_callback_stores_grant_for_flow_user(env):
    state = await env.connect_state()
    # The callback takes no signed-in user: whoever's browser finishes the
    # popup, the grant belongs to the flow's user.
    response = await env.callback(state)
    assert "mcp_oauth_success=1" in response.headers["location"]
    grant = env.db.query(MCPOAuthGrant).one()
    assert grant.user_id == env.member.id
    assert grant.resource_owner_key == env.default_key(env.member)
    assert (
        env.db.query(MCPOAuthGrant)
        .filter(MCPOAuthGrant.user_id == env.other.id)
        .count()
        == 0
    )
    assert env.db.query(MCPOAuthFlowState).one().consumed_at is not None
    assert env.revoked == []


@pytest.mark.asyncio
async def test_team_member_rows_use_default_owner_key(env):
    state = await env.connect_state()
    flow = env.db.query(MCPOAuthFlowState).one()
    assert flow.resource_owner_key == f"xagent:user:{env.member.id}"
    await env.callback(state)
    env.db.expire_all()
    assert env.db.query(MCPOAuthGrant).one().resource_owner_key == (
        f"xagent:user:{env.member.id}"
    )


@pytest.mark.asyncio
async def test_team_member_callback_for_deleted_flow_is_400_without_redirect(env):
    state = await env.connect_state()
    env.db.query(MCPOAuthFlowState).delete()
    env.db.commit()
    with pytest.raises(HTTPException) as excinfo:
        await env.callback(state)
    assert _http_error(excinfo) == (
        400,
        {"code": "invalid_state", "message": "Invalid OAuth state"},
    )


@pytest.mark.asyncio
async def test_team_member_callback_preflight_rejects_lost_access(env):
    state = await env.connect_state()
    env.hook.open = set()
    response = await env.callback(state)
    assert _error_code(response) == "invalid_state"
    assert env.db.query(MCPOAuthGrant).count() == 0
    # No code was exchanged, so there is nothing to revoke.
    assert env.revoked == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case",
    [
        "membership_removed",
        "link_removed",
        "association_created",
        "url_changed",
        "no_longer_open",
        "hook_raises",
    ],
)
async def test_team_member_callback_fails_closed_when_access_lost(env, case):
    state = await env.connect_state()
    server_id = int(env.server.id)
    member_id = int(env.member.id)

    def lose_access() -> None:
        if case == "membership_removed":
            env.hook.visible = set()
            env.hook.open = set()
        elif case == "link_removed":
            env.hook.visible = {server_id + 100}
            env.hook.open = {server_id + 100}
        elif case == "association_created":
            with env.session_factory() as other_db:
                other_db.add(
                    UserMCPServer(
                        user_id=member_id,
                        mcpserver_id=server_id,
                        is_owner=False,
                        is_active=True,
                    )
                )
                other_db.commit()
        elif case == "url_changed":
            env.edit_server(url="https://mcp.example.com/elsewhere")
        elif case == "no_longer_open":
            env.hook.open = set()
        elif case == "hook_raises":
            env.hook.fail_from_call = env.hook.calls + 1

    env.on_exchange = lose_access
    response = await env.callback(state)
    expected_code = (
        "token_exchange_failed" if case == "hook_raises" else "invalid_state"
    )
    assert _error_code(response) == expected_code
    env.db.expire_all()
    assert env.db.query(MCPOAuthGrant).count() == 0
    assert len(env.revoked) == 1


def _statement_log(env):
    log: list[str] = []

    def record(_conn, _cursor, statement, *_rest):
        if statement.startswith("BEGIN IMMEDIATE"):
            log.append("BEGIN IMMEDIATE")
        elif "hook_read" in statement:
            log.append("hook_read")
        else:
            match = re.search(r"\bFROM (\w+)", statement)
            if match and statement.lstrip().upper().startswith("SELECT"):
                log.append(match.group(1))

    event.listen(env.engine, "before_cursor_execute", record)
    return log


def _tail_from_last_begin(log: list[str]) -> list[str]:
    start = len(log) - 1 - log[::-1].index("BEGIN IMMEDIATE")
    return log[start:]


@pytest.mark.asyncio
async def test_team_member_lifecycle_lock_order(env):
    env.hook.read_marker = True
    log = _statement_log(env)

    state = await env.connect_state()
    persist = _tail_from_last_begin(log)
    ordered = [
        name
        for name in persist
        if name in ("mcp_servers", "users", "user_mcpservers", "hook_read")
    ]
    assert ordered[:4] == ["mcp_servers", "users", "user_mcpservers", "hook_read"]

    log.clear()
    await env.callback(state)
    lock_phase = _tail_from_last_begin(log)
    ordered = [
        name
        for name in lock_phase
        if name
        in (
            "mcp_servers",
            "users",
            "user_mcpservers",
            "mcp_oauth_flow_states",
            "hook_read",
        )
    ]
    assert ordered[:5] == [
        "mcp_servers",
        "users",
        "user_mcpservers",
        "mcp_oauth_flow_states",
        "hook_read",
    ]


_OTHER_HOST_CALLBACK = "https://evil.example.org/api/mcp/oauth/callback"


def _other_path_callback() -> str:
    return mcp_api._default_mcp_oauth_redirect_uri() + "/elsewhere"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case",
    [
        "configured_redirect_uri_other_host",
        "configured_redirect_uri_other_path",
        "configured_redirect_uri_default",
        "redirect_uri_unconfigured",
        "association_holder_configured_redirect_uri_other_host",
        "prm_resource_equals_url",
        "prm_resource_trailing_slash",
        "prm_resource_other_host",
        "prm_resource_other_path",
        "url_changed_before_persist",
        "auth_changed_before_persist",
        "binding_snapshot_taken_before_registration_rollback",
    ],
)
async def test_team_member_connect_requires_resource_equal_to_url(
    env, case, monkeypatch
):
    user = env.member
    expect_status = 303
    expect_code = None
    clients_before = env.db.query(MCPOAuthClient).count()

    if case == "configured_redirect_uri_other_host":
        env.edit_auth(redirect_uri=_OTHER_HOST_CALLBACK)
        expect_status, expect_code = 400, "redirect_uri_not_allowed"
    elif case == "configured_redirect_uri_other_path":
        env.edit_auth(redirect_uri=_other_path_callback())
        expect_status, expect_code = 400, "redirect_uri_not_allowed"
    elif case == "configured_redirect_uri_default":
        env.edit_auth(redirect_uri=mcp_api._default_mcp_oauth_redirect_uri())
    elif case == "association_holder_configured_redirect_uri_other_host":
        env.edit_auth(redirect_uri=_OTHER_HOST_CALLBACK)
        user = env.creator
    elif case == "prm_resource_trailing_slash":
        env.discovery_resource = URL + "/"
    elif case == "prm_resource_other_host":
        env.discovery_resource = "https://other.example.org/mcp"
        expect_status, expect_code = 400, "resource_mismatch"
    elif case == "prm_resource_other_path":
        env.discovery_resource = "https://mcp.example.com/other"
        expect_status, expect_code = 400, "resource_mismatch"
    elif case == "url_changed_before_persist":
        env.on_register = lambda: env.edit_server(url="https://mcp.example.com/v2")
        expect_status, expect_code = 409, "oauth_lifecycle_changed"
    elif case == "auth_changed_before_persist":
        # Committed between discovery and the flow write.
        env.on_register = lambda: env.edit_auth(issuer="https://auth.example.org")
        expect_status, expect_code = 409, "oauth_lifecycle_changed"
    elif case == "binding_snapshot_taken_before_registration_rollback":
        # The registration step rolls the session back first, so a binding read
        # after it would already see this edit and let the flow through.
        env.on_register = lambda: env.edit_auth(
            resource_metadata_url=(
                "https://mcp.example.com/.well-known/oauth-protected-resource/other"
            )
        )
        expect_status, expect_code = 409, "oauth_lifecycle_changed"

    if case == "binding_snapshot_taken_before_registration_rollback":
        binding_reads: list[object] = []
        real_binding = mcp_api._member_grant_binding

        def counting_binding(server):
            binding_reads.append(server)
            return real_binding(server)

        monkeypatch.setattr(mcp_api, "_member_grant_binding", counting_binding)
        # The shared discovery step, which the association path also runs,
        # does not take the member binding.
        await mcp_api._discover_mcp_oauth_connect(env.server)
        assert binding_reads == []

    if expect_status == 303:
        response = await env.connect(user)
        assert response.status_code == 303
        return

    with pytest.raises(HTTPException) as excinfo:
        await env.connect(user)
    status_code, detail = _http_error(excinfo)
    assert status_code == expect_status
    assert detail["code"] == expect_code
    assert env.db.query(MCPOAuthFlowState).count() == 0
    if expect_code == "redirect_uri_not_allowed":
        assert detail["message"] == (
            "Member authorization requires the connector to use this "
            "deployment's OAuth callback"
        )
        assert env.discovery_calls == 0
        assert env.register_calls == 0
        assert env.db.query(MCPOAuthClient).count() == clients_before
    if expect_code == "resource_mismatch":
        assert detail["message"] == (
            "MCP OAuth protected resource does not match the connector URL"
        )
        assert env.register_calls == 0
        assert env.db.query(MCPOAuthClient).count() == clients_before


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["hook_raises", "hook_ends_transaction"])
async def test_team_member_connect_hook_failure_under_lock_is_503(env, case):
    # Call 1 is the preflight; call 2 is the one made under the row locks.
    if case == "hook_raises":
        env.hook.fail_from_call = 2
    else:
        env.hook.end_transaction_from_call = 2
    with pytest.raises(HTTPException) as excinfo:
        await env.connect()
    assert _http_error(excinfo) == (503, "Connector team scope is unavailable.")
    assert env.hook.calls == 2
    assert env.db.query(MCPOAuthFlowState).count() == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case",
    [
        "owner_demoted",
        "owner_left_team",
        "owner_deleted_association",
        "former_owner_edits_url",
    ],
)
async def test_owner_changes_close_new_logins_but_keep_existing_grants(env, case):
    env.add_member_grant()
    assert len(env.selected_member_grants()) == 1

    if case == "former_owner_edits_url":
        update_mcp_server(
            env.server.id,
            MCPServerUpdate(config={"url": "https://mcp.example.com/moved"}),
            current_user=env.creator,
            db=env.db,
        )
        env.db.expire_all()
        assert env.db.query(MCPOAuthGrant).count() == 0
        return

    if case == "owner_deleted_association":
        env.db.query(UserMCPServer).filter(
            UserMCPServer.user_id == env.creator.id
        ).delete()
        env.db.commit()
    # The embedder stops opening the connector; the member still sees it.
    env.hook.open = set()

    with pytest.raises(HTTPException) as excinfo:
        await env.connect()
    assert _http_error(excinfo)[0] == 403
    assert len(env.selected_member_grants()) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case", ["no_longer_open", "no_longer_visible", "association_created"]
)
async def test_team_member_connect_rechecks_access_under_lock(env, case):
    server_id = int(env.server.id)
    member_id = int(env.member.id)
    if case == "no_longer_open":
        # The pre-check (call 1) says open; the check under the locks says not.
        env.hook.from_second_call = ({server_id}, set())
    elif case == "no_longer_visible":
        env.hook.from_second_call = (set(), set())
    else:

        def add_association() -> None:
            with env.session_factory() as other_db:
                other_db.add(
                    UserMCPServer(
                        user_id=member_id,
                        mcpserver_id=server_id,
                        is_owner=False,
                        is_active=True,
                    )
                )
                other_db.commit()

        env.on_register = add_association

    with pytest.raises(HTTPException) as excinfo:
        await env.connect()
    status_code, detail = _http_error(excinfo)
    assert status_code == 409
    assert detail["code"] == "oauth_lifecycle_changed"
    env.db.expire_all()
    assert env.db.query(MCPOAuthFlowState).count() == 0


@pytest.mark.asyncio
async def test_team_member_connect_keeps_this_callback_after_redirect_uri_edit(env):
    # ``redirect_uri`` is not part of the member binding, so an edit committed
    # after discovery does not fail the connect. The authorization code still
    # goes to this deployment's callback: the flow uses the value checked
    # before discovery, not the edited one.
    env.on_register = lambda: env.edit_auth(redirect_uri=_OTHER_HOST_CALLBACK)
    response = await env.connect()
    assert response.status_code == 303
    callback = mcp_api._default_mcp_oauth_redirect_uri()
    query = parse_qs(urlparse(response.headers["location"]).query)
    assert query["redirect_uri"] == [callback]
    env.db.expire_all()
    flow = env.db.query(MCPOAuthFlowState).one()
    client = env.db.get(MCPOAuthClient, flow.mcp_oauth_client_id)
    assert client.redirect_uri == callback


@pytest.mark.asyncio
async def test_team_member_callback_preflight_rejects_association_holder(env):
    state = await env.connect_state()
    exchanged: list[bool] = []
    env.on_exchange = lambda: exchanged.append(True)
    with env.session_factory() as other_db:
        other_db.add(
            UserMCPServer(
                user_id=env.member.id,
                mcpserver_id=env.server.id,
                is_owner=False,
                is_active=True,
            )
        )
        other_db.commit()
    response = await env.callback(state)
    assert _error_code(response) == "invalid_state"
    # Refused before the code was exchanged, so nothing to revoke.
    assert exchanged == []
    assert env.revoked == []
    assert env.db.query(MCPOAuthGrant).count() == 0


@pytest.mark.asyncio
async def test_team_member_callback_preflight_hook_failure_fails_before_exchange(env):
    state = await env.connect_state()
    exchanged: list[bool] = []
    env.on_exchange = lambda: exchanged.append(True)
    # The next hook call is the callback preflight, before any code exchange.
    env.hook.fail_from_call = env.hook.calls + 1
    response = await env.callback(state)
    assert _error_code(response) == "invalid_state"
    location = response.headers["location"].replace("+", " ")
    assert "Team connector access could not be checked" in location
    assert exchanged == []
    assert env.revoked == []
    env.db.expire_all()
    assert env.db.query(MCPOAuthGrant).count() == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["access_path_changed", "client_changed"])
async def test_team_member_callback_requires_the_exact_flow_row(env, change):
    state = await env.connect_state()

    def edit_flow_row() -> None:
        with env.session_factory() as other_db:
            flow = other_db.query(MCPOAuthFlowState).one()
            if change == "access_path_changed":
                flow.access_path = "association"
            else:
                other_client = MCPOAuthClient(
                    mcp_server_id=env.server_id,
                    issuer=ISSUER,
                    authorization_endpoint=f"{ISSUER}/authorize",
                    token_endpoint=f"{ISSUER}/token",
                    client_id="another-client",
                    token_endpoint_auth_method="none",
                    redirect_uri=mcp_api._default_mcp_oauth_redirect_uri(),
                )
                other_db.add(other_client)
                other_db.flush()
                flow.mcp_oauth_client_id = other_client.id
            other_db.commit()

    env.on_exchange = edit_flow_row
    response = await env.callback(state)
    assert _error_code(response) == "invalid_state"
    env.db.expire_all()
    assert env.db.query(MCPOAuthGrant).count() == 0
    assert len(env.revoked) == 1
