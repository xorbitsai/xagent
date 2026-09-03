from __future__ import annotations

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

import pytest
from fastapi import HTTPException
from sqlalchemy import event
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker
from starlette.requests import Request

from tests.shared.postgres_disposable import disposable_database_factory
from xagent.core.utils.encryption import decrypt_value, encrypt_value
from xagent.web.api import mcp as mcp_api
from xagent.web.models.database import Base
from xagent.web.models.mcp import MCPServer, UserMCPServer
from xagent.web.models.mcp_oauth import MCPOAuthClient, MCPOAuthFlowState, MCPOAuthGrant
from xagent.web.models.public_mcp import PublicMCPApp
from xagent.web.models.user import User
from xagent.web.models.user_oauth import UserOAuth

pytestmark = pytest.mark.postgresql

POSTGRES_TABLES = [
    User.__table__,
    PublicMCPApp.__table__,
    MCPServer.__table__,
    UserMCPServer.__table__,
    UserOAuth.__table__,
    MCPOAuthClient.__table__,
    MCPOAuthGrant.__table__,
    MCPOAuthFlowState.__table__,
]


@pytest.fixture
def postgresql_engine():
    with disposable_database_factory("xagent_mcp_teardown") as make:
        yield make("owner_race")


@pytest.fixture
def postgresql_context(postgresql_engine):
    Base.metadata.create_all(postgresql_engine, tables=POSTGRES_TABLES)
    factory = sessionmaker(
        bind=postgresql_engine,
        autoflush=False,
        autocommit=False,
    )
    return postgresql_engine, factory


def _seed(factory) -> tuple[int, int, int]:
    with factory() as db:
        user = User(username="workspace-account", password_hash="hash")
        expected = PublicMCPApp(
            app_id="legacy-mail",
            name="Legacy Mail",
            transport="oauth",
            provider_name="legacy-provider",
            oauth_scopes=[],
            launch_config={},
            is_visible_in_connector=True,
        )
        other = PublicMCPApp(
            app_id="other-mail",
            name="Other Mail",
            transport="oauth",
            provider_name="other-provider",
            oauth_scopes=[],
            launch_config={},
            is_visible_in_connector=True,
        )
        server = MCPServer(
            name="Legacy Mail",
            managed="external",
            transport="oauth",
            auth=None,
        )
        db.add_all([user, expected, other, server])
        db.flush()
        db.add_all(
            [
                UserMCPServer(
                    user_id=user.id,
                    mcpserver_id=server.id,
                    is_owner=True,
                    is_active=True,
                ),
                UserOAuth(
                    user_id=user.id,
                    provider="legacy-provider",
                    access_token=encrypt_value("legacy-token"),
                ),
                UserOAuth(
                    user_id=user.id,
                    provider="other-provider",
                    access_token=encrypt_value("other-token"),
                ),
            ]
        )
        db.commit()
        return int(user.id), int(expected.id), int(server.id)


def _oauth_callback_request(state: str) -> Request:
    cookie = (
        f"{mcp_api.MCP_OAUTH_STATE_COOKIE}="
        f"{mcp_api._mcp_oauth_state_cookie_value(state)}"
    )
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/api/mcp/oauth/callback",
            "query_string": f"code=code-123&state={state}".encode(),
            "headers": [(b"cookie", cookie.encode())],
        }
    )


def _seed_oauth_lifecycle(factory) -> tuple[int, int, int, int, int, int]:
    with factory() as db:
        user = User(username="oauth-owner", password_hash="hash")
        other_user = User(username="oauth-other", password_hash="hash")
        app = PublicMCPApp(
            app_id="remote-notes",
            name="Remote Notes",
            transport="streamable_http",
            provider_name=None,
            oauth_scopes=[],
            launch_config={
                "url": "https://mcp.example/mcp",
                "auth": {"type": "mcp_oauth"},
            },
            is_visible_in_connector=True,
        )
        server = MCPServer(
            name="remote-notes",
            managed="external",
            transport="streamable_http",
            url="https://mcp.example/mcp",
            auth={
                "type": "mcp_oauth",
                "client_id": "client-id",
                "client_secret": "client-secret",
                "resource": "https://mcp.example/mcp",
                "issuer": "https://auth.example",
                "scope": "notes.read",
                "redirect_uri": "https://xagent.example/api/mcp/oauth/callback",
                "token_endpoint_auth_method": "client_secret_post",
            },
        )
        db.add_all([user, other_user, app, server])
        db.flush()
        db.add_all(
            [
                UserMCPServer(
                    user_id=user.id,
                    mcpserver_id=server.id,
                    is_owner=True,
                    is_active=True,
                ),
                UserMCPServer(
                    user_id=other_user.id,
                    mcpserver_id=server.id,
                    is_owner=False,
                    is_active=True,
                ),
            ]
        )
        client = MCPOAuthClient(
            mcp_server_id=server.id,
            issuer="https://auth.example",
            authorization_endpoint="https://auth.example/authorize",
            token_endpoint="https://auth.example/token",
            client_id="client-id",
            client_secret=encrypt_value("client-secret"),
            token_endpoint_auth_method="client_secret_post",
            redirect_uri="https://xagent.example/api/mcp/oauth/callback",
            metadata_json={"revocation_endpoint": "https://auth.example/revoke"},
        )
        db.add(client)
        db.flush()
        flow = MCPOAuthFlowState(
            state="old-flow-state",
            mcp_server_id=server.id,
            user_id=user.id,
            mcp_oauth_client_id=client.id,
            resource_owner_key=f"xagent:user:{user.id}",
            issuer="https://auth.example",
            resource="https://mcp.example/mcp",
            scope="notes.read",
            code_verifier=encrypt_value("verifier"),
            redirect_after="/mcp",
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=10),
        )
        db.add(flow)
        db.commit()
        return (
            int(user.id),
            int(other_user.id),
            int(app.id),
            int(server.id),
            int(client.id),
            int(flow.id),
        )


def _run_oauth_teardown(factory, *, user_id: int, app_pk: int, server_id: int) -> None:
    with factory() as teardown_db:
        current_user = teardown_db.get(User, user_id)
        assert current_user is not None
        asyncio.run(
            mcp_api.teardown_mcp_app_server(
                server_id,
                app_id="remote-notes",
                expected_catalog_app_id=app_pk,
                expected_provider_name=None,
                current_user=current_user,
                db=teardown_db,
            )
        )


def _run_oauth_callback(factory, state: str = "old-flow-state"):
    with factory() as callback_db:
        return asyncio.run(
            mcp_api.mcp_oauth_callback(_oauth_callback_request(state), callback_db)
        )


@pytest.mark.parametrize("mutation", ["delete", "rename-reassign", "provider-drift"])
def test_owner_mutation_between_preflight_and_teardown_fails_closed(
    postgresql_context, mutation: str
) -> None:
    _, factory = postgresql_context
    user_id, expected_pk, server_id = _seed(factory)
    barrier = threading.Barrier(2)

    def preflight_then_teardown() -> int:
        with factory() as teardown_db:
            current_user = teardown_db.get(User, user_id)
            assert current_user is not None
            # This is the SaaS preflight boundary: exact app and immutable row
            # identity are read before a second Session mutates the catalog.
            expected = teardown_db.get(PublicMCPApp, expected_pk)
            assert expected is not None and expected.app_id == "legacy-mail"
            barrier.wait(timeout=10)
            barrier.wait(timeout=10)
            try:
                asyncio.run(
                    mcp_api.teardown_mcp_app_server(
                        server_id,
                        app_id="legacy-mail",
                        expected_catalog_app_id=expected_pk,
                        expected_provider_name="legacy-provider",
                        current_user=current_user,
                        db=teardown_db,
                    )
                )
            except HTTPException as exc:
                return exc.status_code
            raise AssertionError("teardown unexpectedly accepted a changed owner")

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(preflight_then_teardown)
        barrier.wait(timeout=10)
        with factory() as mutation_db:
            expected = mutation_db.get(PublicMCPApp, expected_pk)
            assert expected is not None
            if mutation == "delete":
                mutation_db.delete(expected)
            elif mutation == "provider-drift":
                expected.provider_name = "replacement-provider"
            else:
                expected.name = "Moved Legacy Mail"
                other = (
                    mutation_db.query(PublicMCPApp)
                    .filter(PublicMCPApp.app_id == "other-mail")
                    .one()
                )
                other.name = "Legacy Mail"
            mutation_db.commit()
        barrier.wait(timeout=10)
        assert future.result(timeout=10) == 403

    with factory() as verify_db:
        assert verify_db.get(MCPServer, server_id) is not None
        assert (
            verify_db.query(UserMCPServer)
            .filter(UserMCPServer.mcpserver_id == server_id)
            .count()
            == 1
        )
        assert {row.provider for row in verify_db.query(UserOAuth).all()} == {
            "legacy-provider",
            "other-provider",
        }


def test_catalog_mutation_waits_while_teardown_holds_identity_locks(
    postgresql_context, monkeypatch: pytest.MonkeyPatch
) -> None:
    postgresql_engine, factory = postgresql_context
    user_id, expected_pk, server_id = _seed(factory)
    identity_checked = threading.Event()
    release_teardown = threading.Event()
    mutation_sent = threading.Event()
    mutation_committed = threading.Event()
    mutation_thread_id: list[int] = []
    real_gate = mcp_api._locked_catalog_app_for_server

    def gated_identity(*args, **kwargs):
        answer = real_gate(*args, **kwargs)
        identity_checked.set()
        assert release_teardown.wait(timeout=10)
        return answer

    monkeypatch.setattr(mcp_api, "_locked_catalog_app_for_server", gated_identity)

    def observe_catalog_update(
        _connection, _cursor, statement, _parameters, _context, _executemany
    ) -> None:
        if (
            mutation_thread_id
            and threading.get_ident() == mutation_thread_id[0]
            and statement.lstrip().startswith("UPDATE public_mcp_apps")
        ):
            mutation_sent.set()

    event.listen(postgresql_engine, "before_cursor_execute", observe_catalog_update)

    def teardown() -> None:
        with factory() as teardown_db:
            current_user = teardown_db.get(User, user_id)
            assert current_user is not None
            asyncio.run(
                mcp_api.teardown_mcp_app_server(
                    server_id,
                    app_id="legacy-mail",
                    expected_catalog_app_id=expected_pk,
                    expected_provider_name="legacy-provider",
                    current_user=current_user,
                    db=teardown_db,
                )
            )

    def rename() -> None:
        mutation_thread_id.append(threading.get_ident())
        with factory() as mutation_db:
            app = mutation_db.get(PublicMCPApp, expected_pk)
            assert app is not None
            app.name = "Renamed After Teardown"
            mutation_db.commit()
        mutation_committed.set()

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            teardown_future = executor.submit(teardown)
            assert identity_checked.wait(timeout=10)
            rename_future = executor.submit(rename)
            assert mutation_sent.wait(timeout=10)
            assert not mutation_committed.wait(timeout=0.25)
            release_teardown.set()
            teardown_future.result(timeout=10)
            rename_future.result(timeout=10)
    finally:
        release_teardown.set()
        event.remove(
            postgresql_engine,
            "before_cursor_execute",
            observe_catalog_update,
        )

    assert mutation_committed.is_set()
    with factory() as verify_db:
        assert verify_db.get(MCPServer, server_id) is None
        assert (
            verify_db.query(UserOAuth).filter_by(provider="legacy-provider").count()
            == 0
        )


def test_concurrent_association_insert_serializes_with_parent_teardown(
    postgresql_context, monkeypatch: pytest.MonkeyPatch
) -> None:
    postgresql_engine, factory = postgresql_context
    user_id, expected_pk, server_id = _seed(factory)
    with factory() as setup_db:
        other_user = User(username="concurrent-account", password_hash="hash")
        setup_db.add(other_user)
        setup_db.commit()
        other_user_id = int(other_user.id)

    identity_checked = threading.Event()
    release_teardown = threading.Event()
    association_insert_sent = threading.Event()
    insert_finished = threading.Event()
    real_gate = mcp_api._locked_catalog_app_for_server

    def gated_identity(*args, **kwargs):
        answer = real_gate(*args, **kwargs)
        identity_checked.set()
        assert release_teardown.wait(timeout=10)
        return answer

    monkeypatch.setattr(mcp_api, "_locked_catalog_app_for_server", gated_identity)

    def observe_association_insert(
        _connection, _cursor, statement, _parameters, _context, _executemany
    ) -> None:
        if statement.lstrip().startswith("INSERT INTO user_mcpservers"):
            association_insert_sent.set()

    event.listen(postgresql_engine, "before_cursor_execute", observe_association_insert)

    def teardown() -> None:
        with factory() as teardown_db:
            current_user = teardown_db.get(User, user_id)
            assert current_user is not None
            asyncio.run(
                mcp_api.teardown_mcp_app_server(
                    server_id,
                    app_id="legacy-mail",
                    expected_catalog_app_id=expected_pk,
                    expected_provider_name="legacy-provider",
                    current_user=current_user,
                    db=teardown_db,
                )
            )

    def insert_association() -> str:
        with factory() as insert_db:
            insert_db.add(
                UserMCPServer(
                    user_id=other_user_id,
                    mcpserver_id=server_id,
                    is_owner=False,
                    is_active=True,
                )
            )
            try:
                insert_db.commit()
            except IntegrityError:
                insert_db.rollback()
                return "foreign-key-rejected"
            finally:
                insert_finished.set()
        return "committed"

    with ThreadPoolExecutor(max_workers=2) as executor:
        try:
            teardown_future = executor.submit(teardown)
            assert identity_checked.wait(timeout=10)
            insert_future = executor.submit(insert_association)
            assert association_insert_sent.wait(timeout=10)
            assert not insert_finished.wait(timeout=0.25)
            release_teardown.set()
            teardown_future.result(timeout=10)
            assert insert_future.result(timeout=10) == "foreign-key-rejected"
        finally:
            release_teardown.set()
            event.remove(
                postgresql_engine,
                "before_cursor_execute",
                observe_association_insert,
            )

    with factory() as verify_db:
        assert verify_db.get(MCPServer, server_id) is None
        assert (
            verify_db.query(UserMCPServer)
            .filter(UserMCPServer.mcpserver_id == server_id)
            .count()
            == 0
        )


def test_callback_cannot_resurrect_grant_after_teardown_and_fast_reconnect(
    postgresql_context, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, factory = postgresql_context
    user_id, _, app_pk, server_id, client_id, old_flow_id = _seed_oauth_lifecycle(
        factory
    )
    exchange_started = threading.Event()
    release_exchange = threading.Event()
    revocations: list[mcp_api._MCPOAuthRevocationSnapshot] = []

    async def exchange_after_teardown(**_kwargs):
        exchange_started.set()
        assert release_exchange.wait(timeout=10)
        return {
            "access_token": "new-access-token",
            "refresh_token": "new-refresh-token",
            "token_type": "Bearer",
            "scope": "notes.read",
        }

    async def observe_revocation(snapshot) -> None:
        revocations.append(snapshot)

    monkeypatch.setattr(mcp_api, "_exchange_mcp_oauth_code", exchange_after_teardown)
    monkeypatch.setattr(
        mcp_api, "_revoke_mcp_oauth_snapshot_externally", observe_revocation
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        callback_future = executor.submit(_run_oauth_callback, factory)
        assert exchange_started.wait(timeout=10)
        teardown_future = executor.submit(
            _run_oauth_teardown,
            factory,
            user_id=user_id,
            app_pk=app_pk,
            server_id=server_id,
        )
        teardown_future.result(timeout=10)

        # A new lifecycle may reconnect while the old provider request is still
        # returning. The old callback must require its exact deleted flow, not
        # merely accept the new active association.
        with factory() as reconnect_db:
            reconnect_db.add(
                UserMCPServer(
                    user_id=user_id,
                    mcpserver_id=server_id,
                    is_owner=False,
                    is_active=True,
                )
            )
            reconnect_db.add(
                MCPOAuthFlowState(
                    state="new-flow-state",
                    mcp_server_id=server_id,
                    user_id=user_id,
                    mcp_oauth_client_id=client_id,
                    resource_owner_key=f"xagent:user:{user_id}",
                    issuer="https://auth.example",
                    resource="https://mcp.example/mcp",
                    scope="notes.read",
                    code_verifier=encrypt_value("new-verifier"),
                    expires_at=datetime.now(timezone.utc) + timedelta(minutes=10),
                )
            )
            reconnect_db.commit()

        release_exchange.set()
        response = callback_future.result(timeout=10)

    query = parse_qs(urlparse(response.headers["location"]).query)
    assert query["mcp_oauth_error"] == ["invalid_state"]
    assert len(revocations) == 1
    assert decrypt_value(revocations[0].access_token) == "new-access-token"
    assert decrypt_value(str(revocations[0].refresh_token)) == "new-refresh-token"
    with factory() as verify_db:
        assert verify_db.get(MCPServer, server_id) is not None
        assert verify_db.get(MCPOAuthFlowState, old_flow_id) is None
        assert verify_db.query(MCPOAuthGrant).filter_by(user_id=user_id).count() == 0
        assert (
            verify_db.query(MCPOAuthFlowState)
            .filter_by(user_id=user_id, state="new-flow-state")
            .count()
            == 1
        )


def test_connect_cannot_create_flow_after_teardown_wins_during_discovery(
    postgresql_context, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, factory = postgresql_context
    user_id, _, app_pk, server_id, _, _ = _seed_oauth_lifecycle(factory)
    discovery_started = threading.Event()
    release_discovery = threading.Event()

    async def discovery_after_teardown(_server, _auth_config):
        discovery_started.set()
        assert release_discovery.wait(timeout=10)
        return SimpleNamespace(
            resource="https://mcp.example/mcp",
            scopes=("notes.read",),
            protected_resource=SimpleNamespace(
                authorization_servers=("https://auth.example",),
            ),
            authorization_server=SimpleNamespace(
                issuer="https://auth.example",
                authorization_endpoint="https://auth.example/authorize",
                token_endpoint="https://auth.example/token",
                registration_endpoint=None,
                client_id_metadata_document_supported=False,
                raw={"issuer": "https://auth.example"},
            ),
        )

    monkeypatch.setattr(
        mcp_api, "_discover_mcp_oauth_for_server", discovery_after_teardown
    )

    def connect() -> int:
        with factory() as connect_db:
            current_user = connect_db.get(User, user_id)
            assert current_user is not None
            try:
                asyncio.run(
                    mcp_api.connect_mcp_oauth(
                        server_id,
                        mcp_api.MCPOAuthConnectRequest(),
                        current_user,
                        connect_db,
                    )
                )
            except HTTPException as exc:
                return exc.status_code
        raise AssertionError("OAuth connect unexpectedly persisted a flow")

    with ThreadPoolExecutor(max_workers=2) as executor:
        connect_future = executor.submit(connect)
        assert discovery_started.wait(timeout=10)
        teardown_future = executor.submit(
            _run_oauth_teardown,
            factory,
            user_id=user_id,
            app_pk=app_pk,
            server_id=server_id,
        )
        teardown_future.result(timeout=10)
        release_discovery.set()
        assert connect_future.result(timeout=10) == 409

    with factory() as verify_db:
        assert verify_db.get(MCPServer, server_id) is not None
        assert verify_db.query(UserMCPServer).filter_by(user_id=user_id).count() == 0
        assert (
            verify_db.query(MCPOAuthFlowState).filter_by(user_id=user_id).count() == 0
        )


def test_callback_persistence_serializes_before_teardown_cleanup(
    postgresql_context, monkeypatch: pytest.MonkeyPatch
) -> None:
    postgresql_engine, factory = postgresql_context
    user_id, _, app_pk, server_id, _, _ = _seed_oauth_lifecycle(factory)
    grant_staged = threading.Event()
    release_callback = threading.Event()
    teardown_lock_sent = threading.Event()
    teardown_thread_id: list[int] = []
    real_upsert = mcp_api._upsert_mcp_oauth_grant

    async def exchange_immediately(**_kwargs):
        return {
            "access_token": "producer-first-access",
            "refresh_token": "producer-first-refresh",
            "token_type": "Bearer",
            "scope": "notes.read",
        }

    def gated_upsert(*args, **kwargs):
        grant = real_upsert(*args, **kwargs)
        grant_staged.set()
        assert release_callback.wait(timeout=10)
        return grant

    def observe_teardown_server_lock(
        _connection, _cursor, statement, _parameters, _context, _executemany
    ) -> None:
        normalized = " ".join(statement.split())
        if (
            teardown_thread_id
            and threading.get_ident() == teardown_thread_id[0]
            and "FROM mcp_servers" in normalized
            and "FOR UPDATE" in normalized
        ):
            teardown_lock_sent.set()

    monkeypatch.setattr(mcp_api, "_exchange_mcp_oauth_code", exchange_immediately)
    monkeypatch.setattr(mcp_api, "_upsert_mcp_oauth_grant", gated_upsert)
    event.listen(
        postgresql_engine, "before_cursor_execute", observe_teardown_server_lock
    )

    def teardown() -> None:
        teardown_thread_id.append(threading.get_ident())
        _run_oauth_teardown(
            factory, user_id=user_id, app_pk=app_pk, server_id=server_id
        )

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            callback_future = executor.submit(_run_oauth_callback, factory)
            assert grant_staged.wait(timeout=10)
            teardown_future = executor.submit(teardown)
            assert teardown_lock_sent.wait(timeout=10)
            assert not teardown_future.done()
            release_callback.set()
            callback_response = callback_future.result(timeout=10)
            teardown_future.result(timeout=10)
    finally:
        release_callback.set()
        event.remove(
            postgresql_engine,
            "before_cursor_execute",
            observe_teardown_server_lock,
        )

    query = parse_qs(urlparse(callback_response.headers["location"]).query)
    assert query["mcp_oauth_success"] == ["1"]
    with factory() as verify_db:
        assert verify_db.get(MCPServer, server_id) is not None
        assert verify_db.query(MCPOAuthGrant).filter_by(user_id=user_id).count() == 0
        assert (
            verify_db.query(MCPOAuthFlowState).filter_by(user_id=user_id).count() == 0
        )
        assert verify_db.query(UserMCPServer).filter_by(user_id=user_id).count() == 0
