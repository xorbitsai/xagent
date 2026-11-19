"""Actor-private remote MCP servers (#3896).

Several actors run as one shared executing user. The embedding application
links each actor's custom remote MCP server to that user and declares, through
``set_actor_private_mcp_servers_hook``, which actor owns it. The engine shows a
mapped server only to its owner's runs and uses only that owner's credential.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

import httpx
import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker

from xagent.core.tools.adapters.vibe.connector_runtime import ConnectorRuntimeError
from xagent.core.utils.encryption import encrypt_value
from xagent.web.api import mcp as mcp_api
from xagent.web.api.mcp import MCPOAuthConnectRequest
from xagent.web.models import MCPOAuthFlowState
from xagent.web.models.database import Base
from xagent.web.models.mcp import MCPServer, UserMCPServer
from xagent.web.models.mcp_oauth import MCPOAuthClient, MCPOAuthGrant
from xagent.web.models.public_mcp import PublicMCPApp
from xagent.web.models.user import User
from xagent.web.services import mcp_oauth as mcp_oauth_service
from xagent.web.services.actor_private_mcp import (
    actor_private_mcp_servers_hook_installed,
    resolve_actor_private_mcp_servers_or_raise,
    set_actor_private_mcp_servers_hook,
)
from xagent.web.services.connector_runtime import _load_visible_runtime_connectors
from xagent.web.services.connector_team_scope import (
    ConnectorHookSessionBoundaryError,
)
from xagent.web.services.mcp_connector_session import find_visible_server
from xagent.web.services.mcp_runtime import MCPActorAuthorizationPolicy
from xagent.web.tools.config import (
    ResolvedToken,
    WebToolConfig,
    set_oauth_token_resolver_hook,
)

OWNER_A = "embedding:workspace:actor-a"
OWNER_B = "embedding:workspace:actor-b"
OWNER_C = "embedding:workspace:actor-c"
ISSUER = "https://issuer.example"


def _oauth_auth(url: str) -> dict:
    return {
        "type": "mcp_oauth",
        "resource": url,
        "issuer": ISSUER,
        "client_id": "private-client",
        "scope": "read",
    }


def _add_server(db, user, *, name: str, url: str, auth: dict, transport: str):
    server = MCPServer.from_config(
        {
            "name": name,
            "managed": "external",
            "transport": transport,
            "url": url,
            "auth": auth,
        }
    )
    db.add(server)
    db.flush()
    db.add(
        UserMCPServer(
            user_id=int(user.id),
            mcpserver_id=int(server.id),
            is_owner=True,
            is_active=True,
        )
    )
    return server


def _add_grant(db, user, server, *, owner: str, token: str) -> None:
    client = (
        db.query(MCPOAuthClient)
        .filter(MCPOAuthClient.mcp_server_id == int(server.id))
        .one_or_none()
    )
    if client is None:
        client = MCPOAuthClient(
            mcp_server_id=int(server.id),
            issuer=ISSUER,
            authorization_endpoint=f"{ISSUER}/authorize",
            token_endpoint=f"{ISSUER}/token",
            client_id="private-client",
            token_endpoint_auth_method="none",
            redirect_uri="https://xagent.example/callback",
        )
        db.add(client)
        db.flush()
    db.add(
        MCPOAuthGrant(
            mcp_server_id=int(server.id),
            user_id=int(user.id),
            mcp_oauth_client_id=int(client.id),
            resource_owner_key=owner,
            issuer=ISSUER,
            resource=server.url,
            scope="read",
            access_token=encrypt_value(token),
            expires_at=datetime(2099, 1, 1, tzinfo=timezone.utc),
            status="active",
        )
    )


@pytest.fixture()
def seeded(tmp_path):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'actor-private.db'}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    db = session_factory()
    user = User(username="shared-executing-user", password_hash="x", is_admin=False)
    db.add(user)
    db.commit()
    db.refresh(user)

    url_a = "https://a.example/mcp"
    url_b = "https://b.example/mcp"
    url_w = "https://w.example/mcp"
    server_a = _add_server(
        db,
        user,
        name="private-a",
        url=url_a,
        auth=_oauth_auth(url_a),
        transport="streamable_http",
    )
    server_b = _add_server(
        db,
        user,
        name="private-b",
        url=url_b,
        auth=_oauth_auth(url_b),
        transport="sse",
    )
    server_w = _add_server(
        db,
        user,
        name="workspace-w",
        url=url_w,
        auth=_oauth_auth(url_w),
        transport="streamable_http",
    )
    _add_grant(db, user, server_a, owner=OWNER_A, token="a-token")
    _add_grant(db, user, server_a, owner=f"xagent:user:{user.id}", token="ws-a-token")
    _add_grant(db, user, server_b, owner=OWNER_B, token="b-token")
    _add_grant(db, user, server_w, owner=f"xagent:user:{user.id}", token="ws-token")
    db.commit()

    mapping = {int(server_a.id): OWNER_A, int(server_b.id): OWNER_B}
    calls: list[int] = []

    def hook(hook_db, *, user_id: int):
        calls.append(user_id)
        return dict(mapping)

    set_actor_private_mcp_servers_hook(hook)
    try:
        yield SimpleNamespace(
            db=db,
            user=user,
            a=server_a,
            b=server_b,
            w=server_w,
            mapping=mapping,
            calls=calls,
            session_factory=session_factory,
            workspace_dir=str(tmp_path / "workspace"),
        )
    finally:
        set_actor_private_mcp_servers_hook(None)
        set_oauth_token_resolver_hook(None)
        db.close()
        engine.dispose()


def _policy(owner: str = OWNER_A) -> MCPActorAuthorizationPolicy:
    return MCPActorAuthorizationPolicy(resource_owner_key=owner)


def _config(seeded, *, policy, mcp_auth_context=None, runtime_view=None):
    config = WebToolConfig(
        db=seeded.db,
        db_factory=seeded.session_factory,
        request=None,
        user=seeded.user,
        user_id=int(seeded.user.id),
        workspace_config={"base_dir": seeded.workspace_dir, "task_id": "1"},
        mcp_auth_context=mcp_auth_context,
        mcp_runtime_authorization_policy=policy,
    )
    if runtime_view is not None:
        config._connector_runtime_view = runtime_view
    return config


def _by_name(configs: list[dict]) -> dict[str, dict]:
    return {config["name"]: config for config in configs}


# --- The hook -----------------------------------------------------------------


def test_without_a_hook_nothing_is_private(seeded) -> None:
    set_actor_private_mcp_servers_hook(None)

    assert actor_private_mcp_servers_hook_installed() is False
    assert (
        resolve_actor_private_mcp_servers_or_raise(
            seeded.db, user_id=int(seeded.user.id)
        )
        == {}
    )


def test_the_hook_is_asked_for_the_executing_user(seeded) -> None:
    answer = resolve_actor_private_mcp_servers_or_raise(
        seeded.db, user_id=int(seeded.user.id)
    )

    assert answer == seeded.mapping
    assert seeded.calls == [int(seeded.user.id)]


@pytest.mark.parametrize(
    "answer",
    [
        None,
        [1, 2],
        {"1": OWNER_A},
        {True: OWNER_A},
        {1: 7},
        {1: ""},
        {1: f" {OWNER_A} "},
        {1: "xagent:user:1"},
        {1: "xagent:user:999"},
    ],
)
def test_a_malformed_answer_fails_closed(seeded, answer) -> None:
    set_actor_private_mcp_servers_hook(lambda db, *, user_id: answer)

    with pytest.raises(ConnectorRuntimeError) as error:
        resolve_actor_private_mcp_servers_or_raise(
            seeded.db, user_id=int(seeded.user.id)
        )

    assert error.value.status_code == 503


def test_a_failing_hook_fails_closed_and_leaves_the_session_usable(seeded) -> None:
    def hook(db, *, user_id):
        db.add(User(id=user_id, username="flush-poison-dup", password_hash="x"))
        db.flush()  # primary-key collision with the seeded user: fails on SQLite too

    set_actor_private_mcp_servers_hook(hook)

    with pytest.raises(ConnectorRuntimeError) as error:
        resolve_actor_private_mcp_servers_or_raise(
            seeded.db, user_id=int(seeded.user.id)
        )

    assert isinstance(error.value.__cause__, SQLAlchemyError)
    # Without the session restore this raises PendingRollbackError.
    assert seeded.db.query(MCPServer).count() == 3


def test_a_poisoned_session_behind_a_malformed_answer_is_restored(seeded) -> None:
    def hook(db, *, user_id):
        try:
            db.add(User(id=user_id, username="flush-poison-dup", password_hash="x"))
            db.flush()
        except SQLAlchemyError:
            pass
        # Nothing propagates from the hook: the validator rejects this answer.
        return {1: f"xagent:user:{user_id}"}

    set_actor_private_mcp_servers_hook(hook)

    with pytest.raises(ConnectorRuntimeError) as error:
        resolve_actor_private_mcp_servers_or_raise(
            seeded.db, user_id=int(seeded.user.id)
        )

    assert isinstance(error.value.__cause__, ValueError)
    assert seeded.db.query(MCPServer).count() == 3


# --- Visibility -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_run_sees_only_its_own_private_servers(seeded) -> None:
    configs = _by_name(
        await _config(seeded, policy=_policy(OWNER_A)).get_mcp_server_configs()
    )

    assert set(configs) == {"private-a", "workspace-w"}


@pytest.mark.asyncio
async def test_a_run_without_a_policy_sees_no_private_server(seeded) -> None:
    configs = _by_name(await _config(seeded, policy=None).get_mcp_server_configs())

    assert set(configs) == {"workspace-w"}


@pytest.mark.asyncio
async def test_an_actor_without_private_servers_sees_none(seeded) -> None:
    configs = _by_name(
        await _config(seeded, policy=_policy(OWNER_C)).get_mcp_server_configs()
    )

    assert set(configs) == {"workspace-w"}


@pytest.mark.asyncio
async def test_a_single_server_load_respects_the_owner(seeded) -> None:
    config = _config(seeded, policy=_policy(OWNER_A))

    assert await config.load_mcp_server_config(int(seeded.b.id)) is None
    loaded = await config.load_mcp_server_config(int(seeded.a.id))
    assert loaded is not None and loaded["name"] == "private-a"


@pytest.mark.asyncio
async def test_a_failing_hook_fails_the_mcp_load_closed(seeded) -> None:
    def hook(db, *, user_id):
        raise RuntimeError("store unavailable")

    set_actor_private_mcp_servers_hook(hook)

    with pytest.raises(ConnectorRuntimeError):
        await _config(seeded, policy=_policy(OWNER_A)).get_mcp_server_configs()


def test_the_connector_session_agrees_with_the_loader(seeded) -> None:
    def find(server, policy):
        return find_visible_server(
            seeded.db,
            user_id=int(seeded.user.id),
            server_id=int(server.id),
            connector_team_id=None,
            mcp_runtime_authorization_policy=policy,
        )

    assert find(seeded.a, None) is None
    assert find(seeded.a, _policy(OWNER_B)) is None
    assert find(seeded.a, _policy(OWNER_A)) is not None
    assert find(seeded.w, None) is not None


def test_runtime_connector_views_never_include_private_servers(seeded) -> None:
    visible = _load_visible_runtime_connectors(seeded.db, user_id=int(seeded.user.id))

    mcp_ids = {ref.connector_id for ref in visible if ref.connector_type == "mcp"}
    assert mcp_ids == {int(seeded.w.id)}


# --- Credentials ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_private_server_uses_its_owners_grant(seeded) -> None:
    configs = _by_name(
        await _config(seeded, policy=_policy(OWNER_A)).get_mcp_server_configs()
    )

    assert configs["private-a"]["config"]["headers"]["Authorization"] == (
        "Bearer a-token"
    )
    assert configs["workspace-w"]["config"]["headers"]["Authorization"] == (
        "Bearer ws-token"
    )


@pytest.mark.asyncio
async def test_a_private_server_never_falls_back_to_the_workspace_grant(
    seeded,
) -> None:
    seeded.db.query(MCPOAuthGrant).filter(
        MCPOAuthGrant.resource_owner_key == OWNER_A
    ).delete()
    seeded.db.commit()

    configs = _by_name(
        await _config(seeded, policy=_policy(OWNER_A)).get_mcp_server_configs()
    )

    assert configs["private-a"]["transport"] == "unavailable"
    assert "ws-a-token" not in str(configs)


@pytest.mark.asyncio
async def test_a_missing_grant_diagnostic_does_not_name_the_owner(seeded) -> None:
    seeded.db.query(MCPOAuthGrant).filter(
        MCPOAuthGrant.resource_owner_key == OWNER_A
    ).delete()
    seeded.db.commit()
    config = _config(seeded, policy=_policy(OWNER_A))

    configs = await config.get_mcp_server_configs()

    assert OWNER_A not in str(configs)
    assert OWNER_A not in str(config._mcp_oauth_diagnostics)


@pytest.mark.asyncio
@pytest.mark.parametrize("owner", [OWNER_A, "xagent:user:{user_id}"])
async def test_task_supplied_authorization_is_refused(seeded, owner) -> None:
    owner = owner.format(user_id=seeded.user.id)
    configs = _by_name(
        await _config(
            seeded,
            policy=_policy(OWNER_A),
            mcp_auth_context={str(seeded.a.id): {"resource_owner_key": owner}},
        ).get_mcp_server_configs()
    )

    assert configs["private-a"]["transport"] == "unavailable"
    assert configs["private-a"]["config"]["reason"] == "config_load_failed"


@pytest.mark.asyncio
async def test_a_runtime_selector_is_refused(seeded) -> None:
    configs = _by_name(
        await _config(
            seeded,
            policy=_policy(OWNER_A),
            runtime_view={
                f"mcp:{seeded.a.id}": {"auth_selector": {"resource_owner_key": OWNER_A}}
            },
        ).get_mcp_server_configs()
    )

    assert configs["private-a"]["transport"] == "unavailable"
    assert configs["private-a"]["config"]["reason"] == "config_load_failed"


@pytest.mark.asyncio
async def test_the_token_resolver_is_not_asked_for_a_private_server(seeded) -> None:
    providers: list[str] = []

    async def resolver(request) -> ResolvedToken | None:
        providers.append(request.provider)
        return ResolvedToken(access_token="resolver-token")

    set_oauth_token_resolver_hook(resolver)
    configs = _by_name(
        await _config(seeded, policy=_policy(OWNER_A)).get_mcp_server_configs()
    )

    assert configs["private-a"]["config"]["headers"]["Authorization"] == (
        "Bearer a-token"
    )
    assert "private-a" not in providers


@pytest.mark.asyncio
async def test_runtime_fields_are_dropped_for_a_private_server(seeded) -> None:
    seeded.a.runtime_bindings = [
        {
            "source": {"input_type": "secrets", "key": "authorization"},
            "target": {"target_type": "transport_headers", "key": "Authorization"},
        }
    ]
    seeded.a.allow_delegated_authorization = True
    seeded.db.commit()

    configs = _by_name(
        await _config(
            seeded,
            policy=_policy(OWNER_A),
            runtime_view={
                f"mcp:{seeded.a.id}": {
                    "context": {"tenant": "t1"},
                    "secrets": {"authorization": "Bearer delegated-token"},
                    "auth_selector": {},
                }
            },
        ).get_mcp_server_configs()
    )

    private = configs["private-a"]
    assert private["config"]["headers"]["Authorization"] == "Bearer a-token"
    assert "delegated-token" not in str(private)
    for key in (
        "runtime_input_schema",
        "runtime_bindings",
        "allow_delegated_authorization",
        "connector_runtime",
    ):
        assert key not in private


@pytest.mark.asyncio
async def test_a_static_key_comes_from_the_encrypted_auth(seeded) -> None:
    seeded.a.auth = {"type": "bearer", "bearer_token": encrypt_value("static-a")}
    seeded.db.commit()

    configs = _by_name(
        await _config(seeded, policy=_policy(OWNER_A)).get_mcp_server_configs()
    )

    assert configs["private-a"]["config"]["headers"]["Authorization"] == (
        "Bearer static-a"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["stdio", "websocket"])
async def test_a_private_server_must_use_a_remote_http_transport(
    seeded, transport
) -> None:
    seeded.a.transport = transport
    if transport == "stdio":
        seeded.a.command = "python"
        seeded.a.url = None
    else:
        seeded.a.url = "wss://a.example/mcp"
    seeded.db.commit()

    configs = _by_name(
        await _config(seeded, policy=_policy(OWNER_A)).get_mcp_server_configs()
    )

    assert configs["private-a"]["transport"] == "unavailable"
    assert configs["private-a"]["config"]["reason"] == "invalid_configuration"


# --- OAuth start by server id -------------------------------------------------------


@pytest.fixture()
def vendor(monkeypatch):
    url = "https://a.example/mcp"
    prm = "https://a.example/.well-known/oauth-protected-resource"
    as_metadata = f"{ISSUER}/.well-known/oauth-authorization-server"
    registration = f"{ISSUER}/register"

    def handler(request: httpx.Request) -> httpx.Response:
        target = str(request.url)
        if request.method == "GET" and target == url:
            return httpx.Response(
                401, headers={"WWW-Authenticate": f'Bearer resource_metadata="{prm}"'}
            )
        if request.method == "GET" and target == prm:
            return httpx.Response(
                200, json={"resource": url, "authorization_servers": [ISSUER]}
            )
        if request.method == "GET" and target == as_metadata:
            return httpx.Response(
                200,
                json={
                    "issuer": ISSUER,
                    "authorization_endpoint": f"{ISSUER}/authorize",
                    "token_endpoint": f"{ISSUER}/token",
                    "registration_endpoint": registration,
                    "code_challenge_methods_supported": ["S256"],
                    "response_types_supported": ["code"],
                    "token_endpoint_auth_methods_supported": ["none"],
                },
            )
        if request.method == "POST" and target == registration:
            return httpx.Response(
                201,
                json={
                    "client_id": "dynamic-client",
                    "token_endpoint_auth_method": "none",
                },
            )
        return httpx.Response(404)

    monkeypatch.setattr(
        mcp_oauth_service,
        "create_mcp_oauth_http_client",
        lambda **_kwargs: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    monkeypatch.setenv("XAGENT_PUBLIC_API_BASE_URL", "https://api.xagent.test/")


@pytest.mark.asyncio
async def test_oauth_start_stores_the_flow_under_the_owner(seeded, vendor) -> None:
    seeded.db.query(MCPOAuthClient).delete()
    seeded.db.commit()

    response = await mcp_api.connect_mcp_oauth_server_for_owner(
        int(seeded.a.id),
        MCPOAuthConnectRequest(redirect_after="/settings/mcp"),
        seeded.user,
        seeded.db,
        resource_owner_key=OWNER_A,
        accept="application/json",
    )
    seeded.db.commit()

    assert response.status_code == 200
    authorization_url = json.loads(response.body)["authorization_url"]
    # The row's auth names a pre-registered client, so no dynamic
    # registration happens.
    assert parse_qs(urlparse(authorization_url).query)["client_id"] == [
        "private-client"
    ]
    flow = seeded.db.query(MCPOAuthFlowState).one()
    assert flow.resource_owner_key == OWNER_A
    assert flow.mcp_server_id == int(seeded.a.id)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("server_attr", "owner"),
    [("a", OWNER_B), ("b", OWNER_A), ("w", OWNER_A)],
)
async def test_oauth_start_refuses_a_server_the_owner_does_not_hold(
    seeded, server_attr, owner
) -> None:
    server = getattr(seeded, server_attr)

    with pytest.raises(HTTPException) as error:
        await mcp_api.connect_mcp_oauth_server_for_owner(
            int(server.id),
            MCPOAuthConnectRequest(redirect_after="/settings/mcp"),
            seeded.user,
            seeded.db,
            resource_owner_key=owner,
            accept="application/json",
        )

    assert error.value.status_code == 404
    assert seeded.db.query(MCPOAuthFlowState).count() == 0


@pytest.mark.asyncio
async def test_oauth_start_requires_the_hook(seeded) -> None:
    set_actor_private_mcp_servers_hook(None)

    with pytest.raises(HTTPException) as error:
        await mcp_api.connect_mcp_oauth_server_for_owner(
            int(seeded.a.id),
            MCPOAuthConnectRequest(redirect_after="/settings/mcp"),
            seeded.user,
            seeded.db,
            resource_owner_key=OWNER_A,
            accept="application/json",
        )

    assert error.value.status_code == 404


@pytest.mark.asyncio
async def test_oauth_start_refuses_a_catalog_server_even_when_mapped(seeded) -> None:
    url = "https://a.example/mcp"
    seeded.db.add(
        PublicMCPApp(
            app_id="catalog-remote",
            name="Catalog Remote",
            transport="streamable_http",
            launch_config={"url": url, "auth": _oauth_auth(url)},
            is_visible_in_connector=True,
        )
    )
    seeded.a.auth = {**_oauth_auth(url), "app_id": "catalog-remote"}
    seeded.db.commit()

    with pytest.raises(HTTPException) as error:
        await mcp_api.connect_mcp_oauth_server_for_owner(
            int(seeded.a.id),
            MCPOAuthConnectRequest(redirect_after="/settings/mcp"),
            seeded.user,
            seeded.db,
            resource_owner_key=OWNER_A,
            accept="application/json",
        )

    assert error.value.status_code == 404
    assert seeded.db.query(MCPOAuthFlowState).count() == 0


def _ghost_app_id(url: str) -> dict:
    return {**_oauth_auth(url), "app_id": "ghost"}


def _empty_app_id(url: str) -> dict:
    return {**_oauth_auth(url), "app_id": ""}


def _missing_provenance(url: str) -> dict:
    return {
        **_oauth_auth(url),
        "builtin_provenance": {"registry": "xagent", "app_id": "missing"},
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "shape", ["ghost_app_id", "empty_app_id", "missing_provenance", "name_collision"]
)
async def test_one_rule_refuses_a_claimed_row_at_oauth_start_and_in_the_builder(
    seeded, shape
) -> None:
    url = "https://a.example/mcp"
    if shape == "name_collision":
        seeded.db.add(
            PublicMCPApp(
                app_id="catalog-remote",
                name="Catalog Remote",
                transport="streamable_http",
                launch_config={"url": url, "auth": _oauth_auth(url)},
                is_visible_in_connector=True,
            )
        )
        seeded.a.name = "catalog-remote"
    else:
        seeded.a.auth = {
            "ghost_app_id": _ghost_app_id,
            "empty_app_id": _empty_app_id,
            "missing_provenance": _missing_provenance,
        }[shape](url)
    seeded.db.commit()

    with pytest.raises(HTTPException) as error:
        await mcp_api.connect_mcp_oauth_server_for_owner(
            int(seeded.a.id),
            MCPOAuthConnectRequest(redirect_after="/settings/mcp"),
            seeded.user,
            seeded.db,
            resource_owner_key=OWNER_A,
            accept="application/json",
        )
    assert error.value.status_code == 404
    assert seeded.db.query(MCPOAuthFlowState).count() == 0

    configs = await _config(seeded, policy=_policy(OWNER_A)).get_mcp_server_configs()
    config = next(c for c in configs if c["name"] == seeded.a.name)
    assert config["transport"] == "unavailable"
    assert config["config"]["reason"] == "invalid_configuration"


@pytest.mark.asyncio
async def test_plaintext_headers_are_never_sent_for_a_static_key(seeded) -> None:
    seeded.a.auth = {"type": "bearer", "bearer_token": encrypt_value("static-a")}
    seeded.a.headers = {"Authorization": "Bearer plain-header", "X-Api-Key": "hdr"}
    seeded.db.commit()

    configs = _by_name(
        await _config(seeded, policy=_policy(OWNER_A)).get_mcp_server_configs()
    )

    assert configs["private-a"]["config"]["headers"] == {
        "Authorization": "Bearer static-a"
    }


@pytest.mark.asyncio
async def test_an_oauth_server_sends_only_the_owners_bearer(seeded) -> None:
    seeded.a.headers = {"X-Api-Key": "hdr"}
    seeded.db.commit()

    configs = _by_name(
        await _config(seeded, policy=_policy(OWNER_A)).get_mcp_server_configs()
    )

    assert configs["private-a"]["config"]["headers"] == {
        "Authorization": "Bearer a-token"
    }


@pytest.mark.asyncio
async def test_a_mapped_catalog_row_does_not_run_as_a_private_server(seeded) -> None:
    url = "https://a.example/mcp"
    seeded.db.add(
        PublicMCPApp(
            app_id="catalog-remote",
            name="Catalog Remote",
            transport="streamable_http",
            launch_config={"url": url, "auth": _oauth_auth(url)},
            is_visible_in_connector=True,
        )
    )
    seeded.a.auth = {**_oauth_auth(url), "app_id": "catalog-remote"}
    seeded.db.commit()

    configs = _by_name(
        await _config(seeded, policy=_policy(OWNER_A)).get_mcp_server_configs()
    )

    assert configs["private-a"]["transport"] == "unavailable"
    assert configs["private-a"]["config"]["reason"] == "invalid_configuration"


@pytest.mark.asyncio
async def test_oauth_start_refuses_a_hook_that_ends_the_callers_transaction(
    seeded,
) -> None:
    def hook(db, *, user_id):
        db.commit()
        return dict(seeded.mapping)

    set_actor_private_mcp_servers_hook(hook)

    with pytest.raises(ConnectorHookSessionBoundaryError):
        await mcp_api.connect_mcp_oauth_server_for_owner(
            int(seeded.a.id),
            MCPOAuthConnectRequest(redirect_after="/settings/mcp"),
            seeded.user,
            seeded.db,
            resource_owner_key=OWNER_A,
            accept="application/json",
        )
    assert seeded.db.query(MCPOAuthFlowState).count() == 0


@pytest.mark.asyncio
async def test_an_api_key_comes_only_from_the_encrypted_auth(seeded) -> None:
    seeded.a.auth = {
        "type": "api_key",
        "api_key_name": "X-Api-Key",
        "api_key_value": encrypt_value("enc-key"),
    }
    seeded.a.headers = {"X-Api-Key": "plain-key"}
    seeded.db.commit()

    configs = _by_name(
        await _config(seeded, policy=_policy(OWNER_A)).get_mcp_server_configs()
    )

    assert configs["private-a"]["config"]["headers"] == {"X-Api-Key": "enc-key"}


@pytest.mark.asyncio
async def test_oauth_start_refuses_a_server_that_could_never_run(seeded) -> None:
    seeded.a.transport = "websocket"
    seeded.a.url = "wss://a.example/mcp"
    seeded.db.commit()

    with pytest.raises(HTTPException) as error:
        await mcp_api.connect_mcp_oauth_server_for_owner(
            int(seeded.a.id),
            MCPOAuthConnectRequest(redirect_after="/settings/mcp"),
            seeded.user,
            seeded.db,
            resource_owner_key=OWNER_A,
            accept="application/json",
        )

    assert error.value.status_code == 404
    assert seeded.db.query(MCPOAuthFlowState).count() == 0
