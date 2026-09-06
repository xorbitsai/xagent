"""PostgreSQL release gates for concurrent OAuth callback cleanup."""

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timedelta, timezone
from http.cookies import SimpleCookie
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

import pytest
from sqlalchemy import text
from sqlalchemy.orm import sessionmaker

from tests.shared.postgres_disposable import disposable_database_factory
from xagent.core.utils.encryption import encrypt_value
from xagent.web import mcp_apps
from xagent.web.api import auth as auth_api
from xagent.web.api import mcp as mcp_api
from xagent.web.models import MCPOAuthClient, MCPOAuthFlowState, MCPOAuthGrant
from xagent.web.models.actor_oauth_flow import ActorOAuthFlowState
from xagent.web.models.database import Base
from xagent.web.models.mcp import MCPServer, UserMCPServer
from xagent.web.models.oauth_provider import OAuthProvider
from xagent.web.models.public_mcp import PublicMCPApp
from xagent.web.models.user import User
from xagent.web.models.user_oauth import UserOAuth
from xagent.web.tools import config as web_tools_config
from xagent.web.tools.config import WebToolConfig

pytestmark = pytest.mark.postgresql

TEST_BUILTIN_APP_ID = "calendar"
TEST_BUILTIN_EXECUTION = {
    "name": "Google Calendar",
    "transport": "oauth",
    "provider_name": "custom",
    "oauth_scopes": [],
    "launch_config": {"command": "calendar"},
}


class _Response:
    status_code = 200

    def __init__(self, data: dict[str, object]) -> None:
        self._data = data

    def json(self) -> dict[str, object]:
        return self._data


def _provider() -> SimpleNamespace:
    return SimpleNamespace(
        client_id=encrypt_value("client-id"),
        client_secret=encrypt_value("client-secret"),
        auth_url="https://provider.example/authorize",
        token_url="https://provider.example/token",
        userinfo_url="",
        redirect_uri="https://xagent.example/api/auth/custom/callback",
        default_scopes=[],
        user_id_path="id",
        email_path="email",
    )


@pytest.fixture
def postgresql_engine(monkeypatch):
    registry_lookup = mcp_apps.get_builtin_execution_fields_and_optional_scopes

    def test_registry(app_id: str):
        if app_id == TEST_BUILTIN_APP_ID:
            return TEST_BUILTIN_EXECUTION, []
        if app_id == "drive":
            return {
                "name": "Drive",
                "transport": "oauth",
                "provider_name": "other",
                "oauth_scopes": [],
                "launch_config": {"command": "drive"},
            }, []
        return registry_lookup(app_id)

    monkeypatch.setattr(
        mcp_apps, "get_builtin_execution_fields_and_optional_scopes", test_registry
    )

    # The shared factory skips only when this explicit release-gate URL is absent.
    with disposable_database_factory("xagent_actor_oauth") as make:
        yield make("callback_claim")


def test_concurrent_callbacks_exchange_and_persist_only_once(
    postgresql_engine, monkeypatch
) -> None:
    tables = [
        User.__table__,
        PublicMCPApp.__table__,
        MCPServer.__table__,
        UserMCPServer.__table__,
        ActorOAuthFlowState.__table__,
        UserOAuth.__table__,
    ]
    Base.metadata.create_all(postgresql_engine, tables=tables)
    factory = sessionmaker(bind=postgresql_engine, autoflush=False, autocommit=False)
    with factory() as db:
        user = User(username="workspace-account", password_hash="hash")
        app = PublicMCPApp(
            app_id="calendar",
            name="Google Calendar",
            transport="oauth",
            provider_name="custom",
            launch_config={"command": "calendar"},
            is_visible_in_connector=True,
        )
        server = MCPServer(
            name="Google Calendar",
            managed="external",
            transport="oauth",
            auth={"app_id": "calendar", "provider": "custom"},
        )
        db.add_all([user, app, server])
        db.flush()
        db.add(
            UserMCPServer(
                user_id=user.id,
                mcpserver_id=server.id,
                is_owner=False,
                is_active=True,
            )
        )
        db.commit()
        start = auth_api.start_builtin_oauth_for_resource_owner(
            provider="custom",
            app_id="calendar",
            user=user,
            resource_owner_key="toby:slack:41:UALICE",
            db=db,
            db_provider=_provider(),
        )
        db.commit()

    state = parse_qs(urlparse(start.headers["location"]).query)["state"][0]
    cookies = SimpleCookie()
    cookies.load(start.headers["set-cookie"])
    cookie_name, morsel = next(iter(cookies.items()))
    request = SimpleNamespace(
        query_params={"state": state, "code": "provider-code"},
        cookies={cookie_name: morsel.value},
    )

    exchange_entered = threading.Event()
    release_exchange = threading.Event()
    exchange_count = 0
    exchange_lock = threading.Lock()

    def post(*_args, **_kwargs):
        nonlocal exchange_count
        with exchange_lock:
            exchange_count += 1
        exchange_entered.set()
        assert release_exchange.wait(timeout=10)
        return _Response({"access_token": "actor-token", "scope": "profile.read"})

    monkeypatch.setattr(auth_api.requests, "post", post)

    def callback() -> int:
        with factory() as callback_db:
            return auth_api.generic_oauth_callback(
                "custom", request, callback_db, _provider()
            ).status_code

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(callback) for _ in range(2)]
        assert exchange_entered.wait(timeout=10)
        completed, _pending = wait(futures, timeout=10, return_when=FIRST_COMPLETED)
        assert len(completed) == 1
        assert next(iter(completed)).result() == 400
        release_exchange.set()
        statuses = sorted(future.result(timeout=10) for future in futures)

    assert statuses == [200, 400]
    assert exchange_count == 1
    with factory() as db:
        assert db.query(ActorOAuthFlowState).count() == 0
        row = db.query(UserOAuth).one()
        assert row.resource_owner_key == "toby:slack:41:UALICE"
        assert row.access_token == "actor-token"


def test_disconnect_during_actor_exchange_rejects_credential(
    postgresql_engine, monkeypatch
) -> None:
    tables = [
        User.__table__,
        PublicMCPApp.__table__,
        MCPServer.__table__,
        UserMCPServer.__table__,
        ActorOAuthFlowState.__table__,
        UserOAuth.__table__,
    ]
    Base.metadata.create_all(postgresql_engine, tables=tables)
    factory = sessionmaker(bind=postgresql_engine, autoflush=False, autocommit=False)
    with factory() as db:
        user = User(username="workspace-account", password_hash="hash")
        app = PublicMCPApp(
            app_id="calendar",
            name="Google Calendar",
            transport="oauth",
            provider_name="custom",
            launch_config={"command": "calendar"},
            is_visible_in_connector=True,
        )
        server = MCPServer(
            name="Google Calendar",
            managed="external",
            transport="oauth",
            auth={"app_id": "calendar", "provider": "custom"},
        )
        db.add_all([user, app, server])
        db.flush()
        db.add(
            UserMCPServer(
                user_id=user.id,
                mcpserver_id=server.id,
                is_owner=False,
                is_active=True,
            )
        )
        db.commit()
        user_id = int(user.id)
        server_id = int(server.id)
        start = auth_api.start_builtin_oauth_for_resource_owner(
            provider="custom",
            app_id="calendar",
            user=user,
            resource_owner_key="toby:slack:41:UALICE",
            db=db,
            db_provider=_provider(),
        )
        db.commit()

    state = parse_qs(urlparse(start.headers["location"]).query)["state"][0]
    cookies = SimpleCookie()
    cookies.load(start.headers["set-cookie"])
    cookie_name, morsel = next(iter(cookies.items()))
    request = SimpleNamespace(
        query_params={"state": state, "code": "provider-code"},
        cookies={cookie_name: morsel.value},
    )

    exchange_entered = threading.Event()
    release_exchange = threading.Event()

    def post(*_args, **_kwargs):
        exchange_entered.set()
        assert release_exchange.wait(timeout=10)
        return _Response({"access_token": "actor-token", "scope": "profile.read"})

    monkeypatch.setattr(auth_api.requests, "post", post)

    def callback() -> int:
        with factory() as callback_db:
            return auth_api.generic_oauth_callback(
                "custom", request, callback_db, _provider()
            ).status_code

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(callback)
        try:
            assert exchange_entered.wait(timeout=10)
            with factory() as disconnect_db:
                link = (
                    disconnect_db.query(UserMCPServer)
                    .filter(
                        UserMCPServer.user_id == user_id,
                        UserMCPServer.mcpserver_id == server_id,
                    )
                    .one()
                )
                disconnect_db.delete(link)
                disconnect_db.commit()
        finally:
            release_exchange.set()
        assert future.result(timeout=10) == 400

    with factory() as db:
        assert (
            db.query(UserMCPServer)
            .filter(
                UserMCPServer.user_id == user_id,
                UserMCPServer.mcpserver_id == server_id,
            )
            .count()
            == 0
        )
        assert db.query(UserOAuth).count() == 0


def test_independent_actor_flows_replace_one_credential(
    postgresql_engine, monkeypatch
) -> None:
    tables = [
        User.__table__,
        PublicMCPApp.__table__,
        MCPServer.__table__,
        UserMCPServer.__table__,
        ActorOAuthFlowState.__table__,
        UserOAuth.__table__,
    ]
    Base.metadata.create_all(postgresql_engine, tables=tables)
    factory = sessionmaker(bind=postgresql_engine, autoflush=False, autocommit=False)
    with factory() as db:
        user = User(username="workspace-account", password_hash="hash")
        app = PublicMCPApp(
            app_id="calendar",
            name="Google Calendar",
            transport="oauth",
            provider_name="custom",
            launch_config={"command": "calendar"},
            is_visible_in_connector=True,
        )
        server = MCPServer(
            name="Google Calendar",
            managed="external",
            transport="oauth",
            auth={"app_id": "calendar", "provider": "custom"},
        )
        db.add_all([user, app, server])
        db.flush()
        db.add(
            UserMCPServer(
                user_id=user.id,
                mcpserver_id=server.id,
                is_owner=False,
                is_active=True,
            )
        )
        db.commit()
        starts = [
            auth_api.start_builtin_oauth_for_resource_owner(
                provider="custom",
                app_id="calendar",
                user=user,
                resource_owner_key="toby:slack:41:UALICE",
                db=db,
                db_provider=_provider(),
            )
            for _ in range(2)
        ]
        db.commit()

    requests = []
    for index, start in enumerate(starts):
        state = parse_qs(urlparse(start.headers["location"]).query)["state"][0]
        cookies = SimpleCookie()
        cookies.load(start.headers["set-cookie"])
        cookie_name, morsel = next(iter(cookies.items()))
        requests.append(
            SimpleNamespace(
                query_params={"state": state, "code": f"provider-code-{index}"},
                cookies={cookie_name: morsel.value},
            )
        )

    exchange_barrier = threading.Barrier(2)

    def exchange(*_args, **_kwargs) -> _Response:
        exchange_barrier.wait(timeout=10)
        return _Response({"access_token": "actor-token", "scope": "profile.read"})

    monkeypatch.setattr(auth_api.requests, "post", exchange)

    def callback(request) -> int:
        with factory() as callback_db:
            return auth_api.generic_oauth_callback(
                "custom", request, callback_db, _provider()
            ).status_code

    with ThreadPoolExecutor(max_workers=2) as executor:
        statuses = sorted(executor.map(callback, requests))

    assert statuses == [200, 200]
    with factory() as db:
        rows = db.query(UserOAuth).all()
        assert len(rows) == 1
        assert rows[0].resource_owner_key == "toby:slack:41:UALICE"


def _start_actor_flow(db, user, *, app_id: str, provider: str):
    response = auth_api.start_builtin_oauth_for_resource_owner(
        provider=provider,
        app_id=app_id,
        user=user,
        resource_owner_key="toby:slack:41:UALICE",
        db=db,
        db_provider=_provider(),
    )
    db.commit()
    state = parse_qs(urlparse(response.headers["location"]).query)["state"][0]
    cookies = SimpleCookie()
    cookies.load(response.headers["set-cookie"])
    cookie_name, morsel = next(iter(cookies.items()))
    return SimpleNamespace(
        query_params={"state": state, "code": f"{app_id}-code"},
        cookies={cookie_name: morsel.value},
    )


def _seed_actor_app(db, user, *, app_id: str, provider: str) -> None:
    execution, _optional_scopes = (
        mcp_apps.get_builtin_execution_fields_and_optional_scopes(app_id)
    )
    assert execution is not None
    assert execution["provider_name"] == provider
    app = PublicMCPApp(
        app_id=app_id,
        name=execution["name"],
        transport=execution["transport"],
        provider_name=provider,
        oauth_scopes=execution["oauth_scopes"],
        launch_config=execution["launch_config"],
        is_visible_in_connector=True,
    )
    server = MCPServer(
        name=execution["name"],
        managed="external",
        transport=execution["transport"],
        auth={"app_id": app_id, "provider": provider},
    )
    db.add_all([app, server])
    db.flush()
    db.add(
        UserMCPServer(
            user_id=user.id,
            mcpserver_id=server.id,
            is_owner=False,
            is_active=True,
        )
    )


def test_concurrent_actor_flows_for_distinct_apps_preserve_both_credentials(
    postgresql_engine, monkeypatch
) -> None:
    tables = [
        User.__table__,
        PublicMCPApp.__table__,
        MCPServer.__table__,
        UserMCPServer.__table__,
        ActorOAuthFlowState.__table__,
        UserOAuth.__table__,
    ]
    Base.metadata.create_all(postgresql_engine, tables=tables)
    factory = sessionmaker(bind=postgresql_engine, autoflush=False, autocommit=False)
    with factory() as db:
        user = User(username="workspace-account", password_hash="hash")
        db.add(user)
        db.flush()
        _seed_actor_app(db, user, app_id="calendar", provider="custom")
        _seed_actor_app(db, user, app_id="drive", provider="other")
        db.commit()
        flows = [
            (
                "custom",
                _start_actor_flow(db, user, app_id="calendar", provider="custom"),
            ),
            ("other", _start_actor_flow(db, user, app_id="drive", provider="other")),
        ]

    exchange_barrier = threading.Barrier(2)

    def exchange_token(*_args, **_kwargs):
        # Rendezvous before the callback locks the shared user row; waiting
        # after credential deletion would deadlock the serialized writers.
        exchange_barrier.wait(timeout=30)
        return _Response({"access_token": "actor-token", "scope": "profile.read"})

    monkeypatch.setattr(auth_api.requests, "post", exchange_token)

    def callback(provider_name: str, request) -> int:
        with factory() as callback_db:
            return auth_api.generic_oauth_callback(
                provider_name, request, callback_db, _provider()
            ).status_code

    with ThreadPoolExecutor(max_workers=2) as executor:
        statuses = sorted(
            future.result(timeout=60)
            for future in [
                executor.submit(callback, provider_name, request)
                for provider_name, request in flows
            ]
        )

    assert statuses == [200, 200]
    with factory() as db:
        assert {
            (row.provider, row.resource_owner_key) for row in db.query(UserOAuth).all()
        } == {
            ("calendar", "toby:slack:41:UALICE"),
            ("drive", "toby:slack:41:UALICE"),
        }


def _seed_actor_runtime_credential(postgresql_engine):
    Base.metadata.create_all(
        postgresql_engine,
        tables=[User.__table__, UserOAuth.__table__],
    )
    factory = sessionmaker(
        bind=postgresql_engine,
        autoflush=False,
        autocommit=False,
    )
    with factory() as db:
        user = User(username="actor-runtime-user", password_hash="hash")
        db.add(user)
        db.flush()
        account = UserOAuth(
            user_id=int(user.id),
            provider="calendar",
            resource_owner_key="toby:slack:41:UALICE",
            access_token="expired-token",
            refresh_token="rotating-refresh-token",
            expires_at=datetime.now(timezone.utc) - timedelta(minutes=1),
            provider_user_id="provider-user",
        )
        db.add(account)
        db.commit()
        return factory, int(user.id), int(account.id)


def _actor_runtime_config(factory, db, user: User) -> WebToolConfig:
    return WebToolConfig(
        db=db,
        db_factory=factory,
        request=None,
        user=user,
        user_id=int(user.id),
        workspace_config={"base_dir": "/tmp", "task_id": "1"},
    )


@pytest.mark.asyncio
async def test_actor_refresh_row_lock_wait_keeps_event_loop_responsive(
    postgresql_engine,
    monkeypatch,
) -> None:
    factory, user_id, account_id = _seed_actor_runtime_credential(postgresql_engine)
    lock_acquired = threading.Event()
    release_lock = threading.Event()

    def hold_credential_lock() -> None:
        with factory() as db:
            db.query(UserOAuth).filter(
                UserOAuth.id == account_id
            ).with_for_update().one()
            lock_acquired.set()
            assert release_lock.wait(timeout=5)
            db.rollback()

    lock_thread = threading.Thread(target=hold_credential_lock)
    lock_thread.start()
    assert lock_acquired.wait(timeout=5)

    async def refresh_exact_actor(_db, account, _provider_name):
        account.access_token = "refreshed-token"
        account.expires_at = datetime.now(timezone.utc) + timedelta(hours=1)
        return True

    monkeypatch.setattr(
        web_tools_config,
        "refresh_oauth_token_if_needed",
        refresh_exact_actor,
    )

    release_timer = threading.Timer(0.8, release_lock.set)
    release_timer.start()
    resolver = None
    try:
        with factory() as db:
            user = db.get(User, user_id)
            assert user is not None
            resolver = asyncio.create_task(
                _actor_runtime_config(
                    factory, db, user
                )._resolve_legacy_oauth_access_token(
                    provider_name="custom",
                    app_id="calendar",
                    resource_owner_key="toby:slack:41:UALICE",
                )
            )
            await asyncio.sleep(0.1)
            loop_remained_responsive = not release_lock.is_set()
            result = await resolver
    finally:
        release_lock.set()
        release_timer.cancel()
        if resolver is not None and not resolver.done():
            await resolver
        lock_thread.join(timeout=5)

    assert not lock_thread.is_alive()
    assert loop_remained_responsive
    assert result.access_token == "refreshed-token"


@pytest.mark.asyncio
async def test_actor_refresh_reloads_replacement_credential_after_gate(
    postgresql_engine,
    monkeypatch,
) -> None:
    factory, user_id, original_account_id = _seed_actor_runtime_credential(
        postgresql_engine
    )
    replacement_account_ids: list[int] = []

    def replace_credential() -> None:
        with factory() as db:
            original = db.get(UserOAuth, original_account_id)
            assert original is not None
            db.delete(original)
            db.flush()
            replacement = UserOAuth(
                user_id=user_id,
                provider="calendar",
                resource_owner_key="toby:slack:41:UALICE",
                access_token="winner-token",
                refresh_token="winner-refresh-token",
                expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
                provider_user_id="provider-user",
            )
            db.add(replacement)
            db.commit()
            replacement_account_ids.append(int(replacement.id))

    class ReplacingLock:
        async def acquire(self) -> bool:
            await asyncio.to_thread(replace_credential)
            return True

        def release(self) -> None:
            return None

    monkeypatch.setattr(
        web_tools_config,
        "_actor_oauth_refresh_lock",
        lambda *_args: ReplacingLock(),
    )

    with factory() as db:
        user = db.get(User, user_id)
        assert user is not None
        result = await _actor_runtime_config(
            factory, db, user
        )._resolve_legacy_oauth_access_token(
            provider_name="custom",
            app_id="calendar",
            resource_owner_key="toby:slack:41:UALICE",
        )

    assert result.access_token == "winner-token"
    assert replacement_account_ids
    assert replacement_account_ids[0] != original_account_id


@pytest.mark.asyncio
async def test_concurrent_postgresql_actor_refresh_uses_one_rotating_token(
    postgresql_engine,
    monkeypatch,
) -> None:
    factory, user_id, account_id = _seed_actor_runtime_credential(postgresql_engine)
    OAuthProvider.__table__.create(postgresql_engine)
    with factory() as db:
        db.add(
            OAuthProvider(
                provider_name="custom",
                name="Custom",
                client_id="client-id",
                client_secret="client-secret",
                auth_url="https://provider.example/authorize",
                token_url="https://provider.example/token",
            )
        )
        db.commit()

    original_query = web_tools_config.scoped_user_oauth_query
    query_barrier = threading.Barrier(2)

    def synchronized_query(db, *, user_id, resource_owner_key):
        query = original_query(
            db,
            user_id=user_id,
            resource_owner_key=resource_owner_key,
        )
        if resource_owner_key is not None:
            query_barrier.wait(timeout=5)
        return query

    monkeypatch.setattr(
        web_tools_config,
        "scoped_user_oauth_query",
        synchronized_query,
    )

    class Response:
        status_code = 200

        def json(self) -> dict[str, object]:
            return {
                "access_token": "winner-token",
                "refresh_token": "winner-refresh-token",
                "expires_in": 3600,
            }

    post_calls = 0
    post_lock = threading.Lock()

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def post(self, *_args, **_kwargs):
            nonlocal post_calls
            with post_lock:
                post_calls += 1
            return Response()

    monkeypatch.setattr(web_tools_config.httpx, "AsyncClient", Client)

    def resolve():
        with factory() as db:
            user = db.get(User, user_id)
            assert user is not None
            return asyncio.run(
                _actor_runtime_config(
                    factory, db, user
                )._resolve_legacy_oauth_access_token(
                    provider_name="custom",
                    app_id="calendar",
                    resource_owner_key="toby:slack:41:UALICE",
                )
            )

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(resolve) for _ in range(2)]
        results = [future.result(timeout=10) for future in futures]

    assert post_calls == 1
    assert [result.access_token for result in results] == ["winner-token"] * 2
    with factory() as db:
        persisted = db.get(UserOAuth, account_id)
        assert persisted is not None
        assert persisted.access_token == "winner-token"
        assert persisted.refresh_token == "winner-refresh-token"


def test_mcp_callback_and_delete_do_not_deadlock(
    postgresql_engine, monkeypatch
) -> None:
    tables = [
        User.__table__,
        MCPServer.__table__,
        UserMCPServer.__table__,
        MCPOAuthClient.__table__,
        MCPOAuthFlowState.__table__,
        MCPOAuthGrant.__table__,
    ]
    Base.metadata.create_all(postgresql_engine, tables=tables)
    factory = sessionmaker(bind=postgresql_engine, autoflush=False, autocommit=False)
    state = "callback-delete-deadlock"
    with factory() as db:
        user = User(username="mcp-workspace-account", password_hash="hash")
        server = MCPServer(
            name="remote-notes",
            managed="external",
            transport="streamable_http",
            url="https://mcp.example.com/mcp",
            auth={"type": "mcp_oauth"},
        )
        db.add_all([user, server])
        db.flush()
        client = MCPOAuthClient(
            mcp_server_id=server.id,
            issuer="https://auth.example.com",
            authorization_endpoint="https://auth.example.com/authorize",
            token_endpoint="https://auth.example.com/token",
            client_id="client-id",
            token_endpoint_auth_method="none",
            redirect_uri="https://xagent.example/api/mcp/oauth/callback",
        )
        db.add(client)
        db.flush()
        association = UserMCPServer(
            user_id=user.id,
            mcpserver_id=server.id,
            is_owner=True,
            is_active=True,
        )
        db.add(association)
        db.flush()
        db.add(
            MCPOAuthFlowState(
                state=state,
                mcp_server_id=server.id,
                user_id=user.id,
                association_lifecycle_generation=association.lifecycle_generation,
                mcp_oauth_client_id=client.id,
                resource_owner_key=f"xagent:user:{user.id}",
                issuer=client.issuer,
                resource="https://mcp.example.com/mcp",
                scope="records.read",
                code_verifier=encrypt_value("verifier"),
                redirect_after="/settings/mcp",
                expires_at=datetime.now(timezone.utc) + timedelta(minutes=10),
            )
        )
        db.commit()
        user_id = int(user.id)
        server_id = int(server.id)

    request = SimpleNamespace(
        query_params={"state": state, "code": "provider-code"},
        cookies={
            mcp_api.MCP_OAUTH_STATE_COOKIE: mcp_api._mcp_oauth_state_cookie_value(state)
        },
    )
    callback_has_flow_lock = threading.Event()
    delete_has_association_lock = threading.Event()
    release_callback = threading.Event()
    real_upsert = mcp_api._upsert_mcp_oauth_grant
    real_actor_state = mcp_api._has_actor_owned_mcp_oauth_state

    async def exchange_code(**_kwargs):
        return {
            "access_token": "mcp-token",
            "token_type": "Bearer",
            "scope": "records.read",
        }

    def pause_after_flow_lock(*args, **kwargs):
        callback_has_flow_lock.set()
        assert release_callback.wait(timeout=10)
        return real_upsert(*args, **kwargs)

    def observe_delete_lock(*args, **kwargs):
        delete_has_association_lock.set()
        return real_actor_state(*args, **kwargs)

    monkeypatch.setattr(mcp_api, "_exchange_mcp_oauth_code", exchange_code)
    monkeypatch.setattr(mcp_api, "_upsert_mcp_oauth_grant", pause_after_flow_lock)
    monkeypatch.setattr(
        mcp_api,
        "_has_actor_owned_mcp_oauth_state",
        observe_delete_lock,
    )

    def callback():
        with factory() as db:
            return asyncio.run(mcp_api.mcp_oauth_callback(request, db))

    def delete():
        with factory() as db:
            user = db.get(User, user_id)
            assert user is not None
            return asyncio.run(mcp_api.delete_mcp_server(server_id, user, db))

    with ThreadPoolExecutor(max_workers=2) as executor:
        callback_future = executor.submit(callback)
        assert callback_has_flow_lock.wait(timeout=10)
        delete_future = executor.submit(delete)
        assert not delete_has_association_lock.wait(timeout=0.2)
        release_callback.set()
        callback_response = callback_future.result(timeout=10)
        assert delete_has_association_lock.wait(timeout=10)
        assert delete_future.result(timeout=10) is None

    callback_query = parse_qs(urlparse(callback_response.headers["location"]).query)
    assert callback_query["mcp_oauth_success"] == ["1"]
    with factory() as db:
        assert db.query(MCPOAuthFlowState).count() == 0
        assert db.query(MCPOAuthGrant).count() == 0
        assert db.query(UserMCPServer).count() == 0


def test_connect_sweep_skips_flow_locked_by_delete(
    postgresql_engine, monkeypatch
) -> None:
    tables = [
        User.__table__,
        MCPServer.__table__,
        UserMCPServer.__table__,
        MCPOAuthClient.__table__,
        MCPOAuthFlowState.__table__,
    ]
    Base.metadata.create_all(postgresql_engine, tables=tables)
    factory = sessionmaker(bind=postgresql_engine, autoflush=False, autocommit=False)

    with factory() as db:
        connect_user = User(username="connect-user", password_hash="hash")
        delete_user = User(username="delete-user", password_hash="hash")
        connect_server = MCPServer(
            name="connect-target",
            managed="external",
            transport="streamable_http",
            url="https://connect.example.com/mcp",
            auth={
                "type": "mcp_oauth",
                "resource": "https://connect.example.com/mcp",
                "issuer": "https://auth.example.com",
                "scope": "records.read",
                "client_id": "client-id",
                "redirect_uri": "https://xagent.example/api/mcp/oauth/callback",
                "token_endpoint_auth_method": "none",
            },
        )
        delete_server = MCPServer(
            name="delete-target",
            managed="external",
            transport="streamable_http",
            url="https://delete.example.com/mcp",
            auth={"type": "mcp_oauth"},
        )
        db.add_all([connect_user, delete_user, connect_server, delete_server])
        db.flush()

        connect_client = MCPOAuthClient(
            mcp_server_id=connect_server.id,
            issuer="https://auth.example.com",
            authorization_endpoint="https://auth.example.com/authorize",
            token_endpoint="https://auth.example.com/token",
            client_id="client-id",
            token_endpoint_auth_method="none",
            redirect_uri="https://xagent.example/api/mcp/oauth/callback",
        )
        delete_client = MCPOAuthClient(
            mcp_server_id=delete_server.id,
            issuer="https://auth.example.com",
            authorization_endpoint="https://auth.example.com/authorize",
            token_endpoint="https://auth.example.com/token",
            client_id="client-id",
            token_endpoint_auth_method="none",
            redirect_uri="https://xagent.example/api/mcp/oauth/callback",
        )
        db.add_all([connect_client, delete_client])
        db.flush()
        db.add_all(
            [
                UserMCPServer(
                    user_id=connect_user.id,
                    mcpserver_id=connect_server.id,
                    is_owner=True,
                    is_active=True,
                ),
                UserMCPServer(
                    user_id=delete_user.id,
                    mcpserver_id=delete_server.id,
                    is_owner=True,
                    is_active=True,
                ),
                MCPOAuthFlowState(
                    state="delete-locked-stale-flow",
                    mcp_server_id=delete_server.id,
                    user_id=delete_user.id,
                    mcp_oauth_client_id=delete_client.id,
                    resource_owner_key=f"xagent:user:{delete_user.id}",
                    issuer="https://auth.example.com",
                    resource="https://delete.example.com/mcp",
                    scope="records.read",
                    code_verifier=encrypt_value("verifier"),
                    expires_at=(
                        datetime.now(timezone.utc)
                        - mcp_api.MCP_OAUTH_FLOW_STATE_RETENTION
                        - timedelta(hours=1)
                    ),
                ),
            ]
        )
        db.commit()
        connect_user_id = int(connect_user.id)
        connect_server_id = int(connect_server.id)

    async def discover(*_args, **_kwargs):
        return SimpleNamespace(
            resource="https://connect.example.com/mcp",
            scopes=("records.read",),
            authorization_server=SimpleNamespace(
                issuer="https://auth.example.com",
                authorization_endpoint="https://auth.example.com/authorize",
                token_endpoint="https://auth.example.com/token",
                raw={"issuer": "https://auth.example.com"},
            ),
        )

    monkeypatch.setattr(mcp_api, "discover_mcp_oauth_metadata", discover)

    with factory() as delete_db, factory() as connect_db:
        deleted = (
            delete_db.query(MCPOAuthFlowState)
            .filter(MCPOAuthFlowState.state == "delete-locked-stale-flow")
            .delete(synchronize_session=False)
        )
        assert deleted == 1

        connect_db.execute(text("SET LOCAL lock_timeout = '1s'"))
        connect_user = connect_db.get(User, connect_user_id)
        assert connect_user is not None
        response = asyncio.run(
            mcp_api.connect_mcp_oauth(
                connect_server_id,
                mcp_api.MCPOAuthConnectRequest(redirect_after="/settings/mcp"),
                connect_user,
                connect_db,
                accept=None,
            )
        )
        assert response.status_code == 303
        delete_db.rollback()

    with factory() as db:
        states = {row.state for row in db.query(MCPOAuthFlowState).all()}
        assert "delete-locked-stale-flow" in states
        assert len(states) == 2
