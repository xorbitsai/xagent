"""PostgreSQL release gate for MCP OAuth producer lifecycle fencing."""

from __future__ import annotations

import asyncio
import threading
import time
from datetime import timedelta
from types import SimpleNamespace
from typing import Any
from urllib.parse import parse_qs, urlparse

import pytest
from fastapi import HTTPException
from sqlalchemy import event, text
from sqlalchemy.orm import sessionmaker
from starlette.requests import Request

from tests.shared.postgres_disposable import disposable_database_factory
from xagent.core.utils.encryption import encrypt_value
from xagent.web.api import mcp as mcp_api
from xagent.web.models import MCPOAuthClient, MCPOAuthFlowState, MCPOAuthGrant
from xagent.web.models.database import Base
from xagent.web.models.mcp import MCPServer, UserMCPServer
from xagent.web.models.public_mcp import PublicMCPApp
from xagent.web.models.user import User
from xagent.web.services import connector_team_scope

pytestmark = pytest.mark.postgresql


@pytest.fixture()
def postgresql_engine():
    with disposable_database_factory("xagent_mcp_oauth_lifecycle") as make:
        engine = make("producer_fence")
        Base.metadata.create_all(
            engine,
            tables=[
                User.__table__,
                MCPServer.__table__,
                UserMCPServer.__table__,
                PublicMCPApp.__table__,
                MCPOAuthClient.__table__,
                MCPOAuthGrant.__table__,
                MCPOAuthFlowState.__table__,
            ],
        )
        yield engine


def _seed_lifecycle(factory, *, flow_consumed: bool = True):
    with factory() as db:
        user = User(username="postgres-oauth-alice", password_hash="x")
        other_user = User(username="postgres-oauth-bob", password_hash="x")
        server = MCPServer.from_config(
            {
                "name": "postgres-oauth-records",
                "managed": "external",
                "transport": "streamable_http",
                "url": "https://mcp.example.com/mcp",
                "auth": {"type": "mcp_oauth"},
            }
        )
        app = PublicMCPApp(
            app_id="postgres-oauth-records",
            name="Postgres OAuth Records",
            transport="streamable_http",
        )
        other_app = PublicMCPApp(
            app_id="postgres-other-app",
            name="Postgres Other App",
            transport="streamable_http",
        )
        db.add_all([user, other_user, server, app, other_app])
        db.flush()
        association = UserMCPServer(
            user_id=user.id,
            mcpserver_id=server.id,
            is_owner=True,
            is_active=True,
        )
        other_association = UserMCPServer(
            user_id=other_user.id,
            mcpserver_id=server.id,
            is_owner=False,
            is_active=True,
        )
        client = MCPOAuthClient(
            mcp_server_id=server.id,
            issuer="https://auth.example.com",
            authorization_endpoint="https://auth.example.com/authorize",
            token_endpoint="https://auth.example.com/token",
            client_id="postgres-client",
            token_endpoint_auth_method="none",
            redirect_uri="https://xagent.example.com/api/mcp/oauth/callback",
        )
        db.add_all([association, other_association, client])
        db.flush()
        flow = MCPOAuthFlowState(
            state="postgres-flow-state",
            mcp_server_id=server.id,
            user_id=user.id,
            association_lifecycle_generation=association.lifecycle_generation,
            mcp_oauth_client_id=client.id,
            resource_owner_key=f"xagent:user:{user.id}",
            issuer="https://auth.example.com",
            resource="https://mcp.example.com/mcp",
            scope="records.read",
            code_verifier=encrypt_value("postgres-verifier"),
            expires_at=mcp_api._utc_now() + timedelta(minutes=10),
            consumed_at=mcp_api._utc_now() if flow_consumed else None,
        )
        db.add(flow)
        db.commit()
        return {
            "user_id": int(user.id),
            "other_user_id": int(other_user.id),
            "server_id": int(server.id),
            "client_id": int(client.id),
            "flow_id": int(flow.id),
            "generation": association.lifecycle_generation,
            "catalog_generation": app.generation,
            "catalog_id": int(app.id),
            "other_catalog_id": int(other_app.id),
        }


def _discovery():
    return SimpleNamespace(
        resource="https://mcp.example.com/mcp",
        scopes=("records.read",),
        authorization_server=SimpleNamespace(
            issuer="https://auth.example.com",
            authorization_endpoint="https://auth.example.com/authorize",
            token_endpoint="https://auth.example.com/token",
            registration_endpoint="https://auth.example.com/register",
            raw={},
        ),
    )


def _callback_request(state: str) -> Request:
    path = f"/api/mcp/oauth/callback?code=auth-code&state={state}"
    parsed = urlparse(path)
    cookie = (
        f"{mcp_api.MCP_OAUTH_STATE_COOKIE}="
        f"{mcp_api._mcp_oauth_state_cookie_value(state)}"
    )
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": parsed.path,
            "query_string": parsed.query.encode(),
            "headers": [(b"cookie", cookie.encode())],
        }
    )


def _identities(seed):
    association_identity = mcp_api._MCPOAuthAssociationIdentity(
        server_id=seed["server_id"],
        user_id=seed["user_id"],
        lifecycle_generation=seed["generation"],
    )
    flow_identity = mcp_api._MCPOAuthFlowIdentity(
        id=seed["flow_id"],
        state="postgres-flow-state",
        server_id=seed["server_id"],
        user_id=seed["user_id"],
        client_id=seed["client_id"],
        association_lifecycle_generation=seed["generation"],
    )
    return association_identity, flow_identity


def test_disconnect_first_replacement_cannot_receive_stale_callback_grant(
    postgresql_engine, monkeypatch
) -> None:
    factory = sessionmaker(bind=postgresql_engine, autoflush=False, autocommit=False)
    seed = _seed_lifecycle(factory)
    association_identity, flow_identity = _identities(seed)

    teardown_locked = threading.Event()
    allow_teardown = threading.Event()
    producer_started = threading.Event()
    producer_finished = threading.Event()
    disconnect_errors: list[BaseException] = []
    producer_results: list[object] = []

    def gated_team_delete(*args, **kwargs):
        teardown_locked.set()
        assert allow_teardown.wait(timeout=5)
        return SimpleNamespace(
            blocked_reason=None,
            team_owned=False,
            authorized=False,
            delete_definition=False,
        )

    monkeypatch.setattr(
        connector_team_scope, "delete_team_connector", gated_team_delete
    )

    def disconnect() -> None:
        try:
            with factory() as disconnect_db:
                asyncio.run(
                    mcp_api.delete_mcp_server(
                        seed["server_id"],
                        current_user=disconnect_db.get(User, seed["user_id"]),
                        db=disconnect_db,
                    )
                )
        except BaseException as exc:  # pragma: no cover - assertion reports it
            disconnect_errors.append(exc)

    def producer() -> None:
        with factory() as producer_db:
            producer_started.set()
            producer_results.append(
                mcp_api._lock_active_mcp_oauth_lifecycle(
                    producer_db,
                    association_identity=association_identity,
                    flow_identity=flow_identity,
                )
            )
            producer_db.rollback()
        producer_finished.set()

    disconnect_thread = threading.Thread(target=disconnect)
    disconnect_thread.start()
    assert teardown_locked.wait(timeout=5)
    producer_thread = threading.Thread(target=producer)
    producer_thread.start()
    assert producer_started.wait(timeout=2)
    assert not producer_finished.wait(timeout=0.2)
    allow_teardown.set()
    disconnect_thread.join(timeout=5)
    producer_thread.join(timeout=5)
    assert disconnect_errors == []
    assert producer_finished.is_set()
    assert producer_results == [None]

    with factory() as disconnect_db:
        replacement_association = UserMCPServer(
            user_id=seed["user_id"],
            mcpserver_id=seed["server_id"],
            is_owner=False,
            is_active=True,
        )
        disconnect_db.add(replacement_association)
        disconnect_db.flush()
        replacement_flow = MCPOAuthFlowState(
            state="postgres-flow-state",
            mcp_server_id=seed["server_id"],
            user_id=seed["user_id"],
            association_lifecycle_generation=(
                replacement_association.lifecycle_generation
            ),
            mcp_oauth_client_id=seed["client_id"],
            resource_owner_key="replacement-owner",
            issuer="https://auth.example.com",
            resource="https://mcp.example.com/mcp",
            scope="records.read",
            code_verifier=encrypt_value("replacement-verifier"),
            expires_at=mcp_api._utc_now() + timedelta(minutes=10),
        )
        disconnect_db.add(replacement_flow)
        disconnect_db.commit()
        replacement_flow_id = int(replacement_flow.id)
        assert replacement_association.lifecycle_generation != seed["generation"]

    with factory() as producer_db:
        assert (
            mcp_api._lock_active_mcp_oauth_lifecycle(
                producer_db,
                association_identity=association_identity,
                flow_identity=flow_identity,
            )
            is None
        )
        producer_db.rollback()

    with factory() as verify_db:
        assert verify_db.query(MCPOAuthGrant).count() == 0
        assert verify_db.query(MCPOAuthFlowState).one().id == replacement_flow_id
        assert (
            verify_db.query(UserMCPServer)
            .filter(UserMCPServer.mcpserver_id == seed["server_id"])
            .count()
            == 2
        )


def test_producer_first_holds_lifecycle_locks_until_grant_commit(
    postgresql_engine, monkeypatch
) -> None:
    factory = sessionmaker(bind=postgresql_engine, autoflush=False, autocommit=False)
    seed = _seed_lifecycle(factory)
    association_identity, flow_identity = _identities(seed)
    disconnect_started = threading.Event()
    disconnect_lock_attempted = threading.Event()
    disconnect_finished = threading.Event()
    disconnect_errors: list[BaseException] = []

    monkeypatch.setattr(
        connector_team_scope,
        "delete_team_connector",
        lambda *args, **kwargs: SimpleNamespace(
            blocked_reason=None,
            team_owned=False,
            authorized=False,
            delete_definition=False,
        ),
    )

    def disconnect() -> None:
        try:
            with factory() as disconnect_db:
                disconnect_started.set()
                asyncio.run(
                    mcp_api.delete_mcp_server(
                        seed["server_id"],
                        current_user=disconnect_db.get(User, seed["user_id"]),
                        db=disconnect_db,
                    )
                )
        except BaseException as exc:  # pragma: no cover - assertion reports it
            disconnect_errors.append(exc)
        finally:
            disconnect_finished.set()

    with factory() as producer_db:
        lifecycle = mcp_api._lock_active_mcp_oauth_lifecycle(
            producer_db,
            association_identity=association_identity,
            flow_identity=flow_identity,
        )
        assert lifecycle is not None
        locked_flow = lifecycle[2]
        assert locked_flow is not None
        original_lock = mcp_api._lock_active_mcp_oauth_lifecycle

        def record_disconnect_lock_attempt(*args, **kwargs):
            disconnect_lock_attempted.set()
            return original_lock(*args, **kwargs)

        monkeypatch.setattr(
            mcp_api,
            "_lock_active_mcp_oauth_lifecycle",
            record_disconnect_lock_attempt,
        )
        disconnect_thread = threading.Thread(target=disconnect)
        disconnect_thread.start()
        assert disconnect_started.wait(timeout=2)
        assert disconnect_lock_attempted.wait(timeout=2)
        assert not disconnect_finished.wait(timeout=0.2)
        mcp_api._upsert_mcp_oauth_grant(
            producer_db,
            flow_state=locked_flow,
            token_data={
                "access_token": "postgres-issued-token",
                "token_type": "Bearer",
                "scope": "records.read",
            },
        )
        producer_db.commit()

    disconnect_thread.join(timeout=5)
    assert disconnect_finished.is_set()
    assert disconnect_errors == []
    with factory() as verify_db:
        assert verify_db.query(MCPOAuthGrant).count() == 0
        assert verify_db.query(MCPOAuthFlowState).count() == 0
        assert (
            verify_db.query(UserMCPServer)
            .filter(UserMCPServer.mcpserver_id == seed["server_id"])
            .count()
            == 1
        )


def _allow_app_teardown(monkeypatch) -> None:
    monkeypatch.setattr(
        connector_team_scope,
        "delete_team_connector",
        lambda *args, **kwargs: SimpleNamespace(
            blocked_reason=None,
            team_owned=False,
            authorized=False,
            delete_definition=False,
        ),
    )


@pytest.mark.parametrize("replaced", ["catalog", "association"])
def test_app_teardown_rejects_preexisting_replacement_generation(
    postgresql_engine, monkeypatch, replaced
) -> None:
    factory = sessionmaker(bind=postgresql_engine, autoflush=False, autocommit=False)
    seed = _seed_lifecycle(factory)
    _allow_app_teardown(monkeypatch)
    with factory() as mutation_db:
        if replaced == "catalog":
            mutation_db.query(PublicMCPApp).filter_by(id=seed["catalog_id"]).delete()
            replacement = PublicMCPApp(
                app_id="postgres-oauth-records",
                name="Postgres OAuth Records",
                transport="streamable_http",
            )
        else:
            mutation_db.query(UserMCPServer).filter_by(
                user_id=seed["user_id"], mcpserver_id=seed["server_id"]
            ).delete()
            replacement = UserMCPServer(
                user_id=seed["user_id"],
                mcpserver_id=seed["server_id"],
                is_owner=True,
                is_active=True,
            )
        mutation_db.add(replacement)
        mutation_db.commit()

    with factory() as teardown_db, pytest.raises(mcp_api.HTTPException) as exc:
        asyncio.run(
            mcp_api.teardown_mcp_app_server(
                seed["server_id"],
                app_id="postgres-oauth-records",
                expected_provider_name=None,
                expected_catalog_generation=seed["catalog_generation"],
                expected_association_generation=seed["generation"],
                current_user=teardown_db.get(User, seed["user_id"]),
                db=teardown_db,
            )
        )
    assert exc.value.status_code == (403 if replaced == "catalog" else 404)


@pytest.mark.parametrize("replaced", ["catalog", "association"])
def test_app_teardown_serializes_later_replacement_with_lock_evidence(
    postgresql_engine, monkeypatch, replaced
) -> None:
    factory = sessionmaker(bind=postgresql_engine, autoflush=False, autocommit=False)
    seed = _seed_lifecycle(factory)
    _allow_app_teardown(monkeypatch)
    identity_locked = threading.Event()
    release_teardown = threading.Event()
    mutation_sent = threading.Event()
    different_row_update_returned = threading.Event()
    replacement_generation: list[object] = []
    teardown_pid: list[int] = []
    mutation_pid: list[int] = []
    mutation_thread_id: list[int] = []
    errors: list[BaseException] = []
    real_owner_check = mcp_api._locked_catalog_app_for_server

    def hold_identity(*args, **kwargs):
        result = real_owner_check(*args, **kwargs)
        identity_locked.set()
        assert release_teardown.wait(timeout=10)
        return result

    def observe_mutation(_conn, _cursor, statement, _params, _context, _many):
        normalized = " ".join(statement.split())
        expected = (
            "UPDATE public_mcp_apps"
            if replaced == "catalog"
            else "DELETE FROM user_mcpservers"
        )
        if (
            mutation_thread_id
            and threading.get_ident() == mutation_thread_id[0]
            and normalized.startswith(expected)
        ):
            mutation_sent.set()

    def observe_mutation_return(_conn, _cursor, statement, _params, _context, _many):
        if (
            replaced == "catalog"
            and mutation_thread_id
            and threading.get_ident() == mutation_thread_id[0]
            and " ".join(statement.split()).startswith("UPDATE public_mcp_apps")
        ):
            different_row_update_returned.set()

    monkeypatch.setattr(mcp_api, "_locked_catalog_app_for_server", hold_identity)
    event.listen(postgresql_engine, "before_cursor_execute", observe_mutation)
    event.listen(postgresql_engine, "after_cursor_execute", observe_mutation_return)

    def teardown() -> None:
        try:
            with factory() as teardown_db:
                teardown_pid.append(teardown_db.scalar(text("SELECT pg_backend_pid()")))
                asyncio.run(
                    mcp_api.teardown_mcp_app_server(
                        seed["server_id"],
                        app_id="postgres-oauth-records",
                        expected_provider_name=None,
                        expected_catalog_generation=seed["catalog_generation"],
                        expected_association_generation=seed["generation"],
                        current_user=teardown_db.get(User, seed["user_id"]),
                        db=teardown_db,
                    )
                )
        except BaseException as exc:  # pragma: no cover - assertion reports it
            errors.append(exc)

    def replace() -> None:
        try:
            mutation_thread_id.append(threading.get_ident())
            with factory() as mutation_db:
                mutation_pid.append(mutation_db.scalar(text("SELECT pg_backend_pid()")))
                if replaced == "catalog":
                    other = mutation_db.get(PublicMCPApp, seed["other_catalog_id"])
                    other.name = "Mutation Proving Table Share Lock"
                    mutation_db.commit()
                    mutation_db.query(PublicMCPApp).filter_by(
                        id=seed["catalog_id"]
                    ).delete()
                    replacement = PublicMCPApp(
                        app_id="postgres-oauth-records",
                        name="Postgres OAuth Records",
                        transport="streamable_http",
                    )
                else:
                    mutation_db.query(UserMCPServer).filter_by(
                        user_id=seed["user_id"], mcpserver_id=seed["server_id"]
                    ).delete()
                    replacement = UserMCPServer(
                        user_id=seed["user_id"],
                        mcpserver_id=seed["server_id"],
                        is_owner=True,
                        is_active=True,
                    )
                mutation_db.add(replacement)
                mutation_db.commit()
                replacement_generation.append(
                    replacement.generation
                    if replaced == "catalog"
                    else replacement.lifecycle_generation
                )
        except BaseException as exc:  # pragma: no cover - assertion reports it
            errors.append(exc)

    teardown_thread = threading.Thread(target=teardown)
    mutation_thread = threading.Thread(target=replace)
    try:
        teardown_thread.start()
        assert identity_locked.wait(timeout=10)
        mutation_thread.start()
        assert mutation_sent.wait(timeout=10)
        deadline = time.monotonic() + 10
        blockers: list[int] = []
        while time.monotonic() < deadline and teardown_pid[0] not in blockers:
            if replaced == "catalog" and different_row_update_returned.is_set():
                break
            with postgresql_engine.connect() as observer:
                blockers = list(
                    observer.scalar(
                        text("SELECT pg_blocking_pids(:pid)"),
                        {"pid": mutation_pid[0]},
                    )
                    or []
                )
        assert not different_row_update_returned.is_set()
        assert teardown_pid[0] in blockers
        release_teardown.set()
        teardown_thread.join(timeout=10)
        mutation_thread.join(timeout=10)
    finally:
        release_teardown.set()
        event.remove(postgresql_engine, "before_cursor_execute", observe_mutation)
        event.remove(postgresql_engine, "after_cursor_execute", observe_mutation_return)

    assert not teardown_thread.is_alive() and not mutation_thread.is_alive()
    assert errors == []
    old = seed["catalog_generation"] if replaced == "catalog" else seed["generation"]
    assert replacement_generation[0] != old


def test_real_callback_producer_blocks_disconnect_until_grant_commit(
    postgresql_engine, monkeypatch
) -> None:
    factory = sessionmaker(bind=postgresql_engine, autoflush=False, autocommit=False)
    seed = _seed_lifecycle(factory, flow_consumed=False)
    producer_at_upsert = threading.Event()
    allow_producer_commit = threading.Event()
    disconnect_lock_attempted = threading.Event()
    disconnect_finished = threading.Event()
    callback_results: list[Any] = []
    callback_errors: list[BaseException] = []
    disconnect_errors: list[BaseException] = []

    async def exchange_code(**kwargs):
        return {
            "access_token": "postgres-callback-token",
            "token_type": "Bearer",
            "scope": "records.read",
        }

    original_upsert = mcp_api._upsert_mcp_oauth_grant

    def gated_upsert(*args, **kwargs):
        producer_at_upsert.set()
        assert allow_producer_commit.wait(timeout=5)
        return original_upsert(*args, **kwargs)

    original_lock = mcp_api._lock_active_mcp_oauth_lifecycle

    def record_disconnect_lock_attempt(*args, **kwargs):
        if threading.current_thread().name == "postgres-real-disconnect":
            disconnect_lock_attempted.set()
        return original_lock(*args, **kwargs)

    monkeypatch.setattr(mcp_api, "_exchange_mcp_oauth_code", exchange_code)
    monkeypatch.setattr(mcp_api, "_upsert_mcp_oauth_grant", gated_upsert)
    monkeypatch.setattr(
        mcp_api,
        "_lock_active_mcp_oauth_lifecycle",
        record_disconnect_lock_attempt,
    )
    monkeypatch.setattr(
        connector_team_scope,
        "delete_team_connector",
        lambda *args, **kwargs: SimpleNamespace(
            blocked_reason=None,
            team_owned=False,
            authorized=False,
            delete_definition=False,
        ),
    )

    def callback() -> None:
        try:
            with factory() as callback_db:
                callback_results.append(
                    asyncio.run(
                        mcp_api.mcp_oauth_callback(
                            _callback_request("postgres-flow-state"), callback_db
                        )
                    )
                )
        except BaseException as exc:  # pragma: no cover - assertion reports it
            callback_errors.append(exc)

    def disconnect() -> None:
        try:
            with factory() as disconnect_db:
                asyncio.run(
                    mcp_api.delete_mcp_server(
                        seed["server_id"],
                        current_user=disconnect_db.get(User, seed["user_id"]),
                        db=disconnect_db,
                    )
                )
        except BaseException as exc:  # pragma: no cover - assertion reports it
            disconnect_errors.append(exc)
        finally:
            disconnect_finished.set()

    callback_thread = threading.Thread(target=callback, name="postgres-real-callback")
    callback_thread.start()
    assert producer_at_upsert.wait(timeout=5)
    disconnect_thread = threading.Thread(
        target=disconnect, name="postgres-real-disconnect"
    )
    disconnect_thread.start()
    assert disconnect_lock_attempted.wait(timeout=5)
    assert not disconnect_finished.wait(timeout=0.2)
    allow_producer_commit.set()
    callback_thread.join(timeout=5)
    disconnect_thread.join(timeout=5)

    assert not callback_thread.is_alive()
    assert not disconnect_thread.is_alive()
    assert callback_errors == []
    assert disconnect_errors == []
    assert len(callback_results) == 1
    assert callback_results[0].status_code == 307
    with factory() as verify_db:
        assert verify_db.query(MCPOAuthGrant).count() == 0
        assert verify_db.query(MCPOAuthFlowState).count() == 0
        assert (
            verify_db.query(UserMCPServer)
            .filter(UserMCPServer.mcpserver_id == seed["server_id"])
            .count()
            == 1
        )


def test_connect_rejects_delete_during_discovery(
    postgresql_engine, monkeypatch
) -> None:
    factory = sessionmaker(bind=postgresql_engine, autoflush=False, autocommit=False)
    seed = _seed_lifecycle(factory, flow_consumed=False)
    delete_locked = threading.Event()
    allow_delete = threading.Event()
    connect_lock_attempted = threading.Event()
    connect_finished = threading.Event()
    connect_results: list[Any] = []
    connect_errors: list[BaseException] = []
    delete_errors: list[BaseException] = []

    async def discover(*args, **kwargs):
        return _discovery()

    async def register(*args, **kwargs):
        return SimpleNamespace(
            client_id="postgres-dynamic-client",
            token_endpoint_auth_method="none",
        )

    def gated_team_delete(*args, **kwargs):
        delete_locked.set()
        assert allow_delete.wait(timeout=5)
        return SimpleNamespace(
            blocked_reason=None,
            team_owned=False,
            authorized=False,
            delete_definition=False,
        )

    original_lock = mcp_api._lock_active_mcp_oauth_lifecycle

    def record_connect_lock(*args, **kwargs):
        if threading.current_thread().name == "postgres-connect":
            connect_lock_attempted.set()
        return original_lock(*args, **kwargs)

    monkeypatch.setattr(mcp_api, "_discover_mcp_oauth_for_server", discover)
    monkeypatch.setattr(mcp_api, "register_mcp_oauth_public_client", register)
    monkeypatch.setattr(
        mcp_api,
        "_lock_active_mcp_oauth_lifecycle",
        record_connect_lock,
    )
    monkeypatch.setattr(
        connector_team_scope,
        "delete_team_connector",
        gated_team_delete,
    )

    def delete() -> None:
        try:
            with factory() as delete_db:
                asyncio.run(
                    mcp_api.delete_mcp_server(
                        seed["server_id"],
                        current_user=delete_db.get(User, seed["user_id"]),
                        db=delete_db,
                    )
                )
        except BaseException as exc:  # pragma: no cover - assertion reports it
            delete_errors.append(exc)

    def connect() -> None:
        try:
            with factory() as connect_db:
                connect_results.append(
                    asyncio.run(
                        mcp_api.connect_mcp_oauth(
                            seed["server_id"],
                            mcp_api.MCPOAuthConnectRequest(
                                redirect_after="/settings/mcp"
                            ),
                            current_user=connect_db.get(User, seed["user_id"]),
                            db=connect_db,
                            accept="application/json",
                        )
                    )
                )
        except BaseException as exc:  # pragma: no cover - assertion reports it
            connect_errors.append(exc)
        finally:
            connect_finished.set()

    delete_thread = threading.Thread(target=delete, name="postgres-delete")
    delete_thread.start()
    assert delete_locked.wait(timeout=5)
    connect_thread = threading.Thread(target=connect, name="postgres-connect")
    connect_thread.start()
    assert connect_lock_attempted.wait(timeout=5)
    assert not connect_finished.wait(timeout=0.2)
    allow_delete.set()
    delete_thread.join(timeout=5)
    connect_thread.join(timeout=5)

    assert not delete_thread.is_alive()
    assert not connect_thread.is_alive()
    assert delete_errors == []
    assert connect_results == []
    assert len(connect_errors) == 1
    assert isinstance(connect_errors[0], HTTPException)
    assert connect_errors[0].status_code == 409
    with factory() as verify_db:
        assert (
            verify_db.query(MCPOAuthFlowState)
            .filter(MCPOAuthFlowState.user_id == seed["user_id"])
            .count()
            == 0
        )
        assert (
            verify_db.query(UserMCPServer)
            .filter(UserMCPServer.mcpserver_id == seed["server_id"])
            .count()
            == 1
        )


def test_callback_rejects_deactivation_during_exchange(
    postgresql_engine, monkeypatch
) -> None:
    factory = sessionmaker(bind=postgresql_engine, autoflush=False, autocommit=False)
    seed = _seed_lifecycle(factory, flow_consumed=False)
    exchange_started = threading.Event()
    allow_exchange = threading.Event()
    callback_results: list[Any] = []
    callback_errors: list[BaseException] = []

    async def exchange_code(**kwargs):
        exchange_started.set()
        assert allow_exchange.wait(timeout=5)
        return {
            "access_token": "deactivated-callback-token",
            "token_type": "Bearer",
            "scope": "records.read",
        }

    monkeypatch.setattr(mcp_api, "_exchange_mcp_oauth_code", exchange_code)

    def callback() -> None:
        try:
            with factory() as callback_db:
                callback_results.append(
                    asyncio.run(
                        mcp_api.mcp_oauth_callback(
                            _callback_request("postgres-flow-state"), callback_db
                        )
                    )
                )
        except BaseException as exc:  # pragma: no cover - assertion reports it
            callback_errors.append(exc)

    callback_thread = threading.Thread(target=callback, name="postgres-callback")
    callback_thread.start()
    assert exchange_started.wait(timeout=5)
    with factory() as deactivate_db:
        response = mcp_api.update_mcp_server(
            seed["server_id"],
            mcp_api.MCPServerUpdate(is_active=False),
            current_user=deactivate_db.get(User, seed["user_id"]),
            db=deactivate_db,
        )
        assert response.is_active is False
    allow_exchange.set()
    callback_thread.join(timeout=5)

    assert not callback_thread.is_alive()
    assert callback_errors == []
    assert len(callback_results) == 1
    callback_response = callback_results[0]
    assert callback_response.status_code == 307
    callback_query = parse_qs(urlparse(callback_response.headers["location"]).query)
    assert callback_query["mcp_oauth_error"] == ["invalid_state"]
    with factory() as verify_db:
        assert verify_db.query(MCPOAuthGrant).count() == 0
        association = (
            verify_db.query(UserMCPServer)
            .filter(
                UserMCPServer.user_id == seed["user_id"],
                UserMCPServer.mcpserver_id == seed["server_id"],
            )
            .one()
        )
        assert association.is_active is False


def test_trusted_reconnect_and_disconnect_use_one_lock_order(
    postgresql_engine, monkeypatch
) -> None:
    factory = sessionmaker(bind=postgresql_engine, autoflush=False, autocommit=False)
    seed = _seed_lifecycle(factory, flow_consumed=False)
    with factory() as setup_db:
        association = (
            setup_db.query(UserMCPServer)
            .filter(
                UserMCPServer.user_id == seed["user_id"],
                UserMCPServer.mcpserver_id == seed["server_id"],
            )
            .one()
        )
        association.is_owner = False
        association.can_delete = True
        association.is_active = False
        setup_db.query(MCPOAuthFlowState).delete()
        setup_db.commit()

    discovery_started = threading.Event()
    allow_discovery = threading.Event()
    delete_locked = threading.Event()
    allow_delete = threading.Event()
    connect_finished = threading.Event()
    connect_errors: list[BaseException] = []
    disconnect_errors: list[BaseException] = []

    def ensure_catalog(db, app_id):
        assert app_id == "postgres-oauth-records"
        return db.get(MCPServer, seed["server_id"]), {"id": app_id}

    async def discover(*args, **kwargs):
        discovery_started.set()
        assert allow_discovery.wait(timeout=10)
        return _discovery()

    async def register(*args, **kwargs):
        return SimpleNamespace(
            client_id="postgres-dynamic-client",
            token_endpoint_auth_method="none",
        )

    def gated_team_delete(*args, **kwargs):
        delete_locked.set()
        assert allow_delete.wait(timeout=10)
        return SimpleNamespace(
            blocked_reason=None,
            team_owned=False,
            authorized=False,
            delete_definition=False,
        )

    monkeypatch.setattr(mcp_api, "_ensure_catalog_mcp_oauth_server", ensure_catalog)
    monkeypatch.setattr(mcp_api, "_discover_mcp_oauth_for_server", discover)
    monkeypatch.setattr(mcp_api, "register_mcp_oauth_public_client", register)
    monkeypatch.setattr(
        connector_team_scope, "delete_team_connector", gated_team_delete
    )

    def connect() -> None:
        try:
            with factory() as connect_db:
                asyncio.run(
                    mcp_api.connect_mcp_oauth_app_for_owner(
                        "postgres-oauth-records",
                        mcp_api.MCPOAuthConnectRequest(redirect_after="/settings/mcp"),
                        current_user=connect_db.get(User, seed["user_id"]),
                        db=connect_db,
                        resource_owner_key="toby:slack:workspace:alice",
                        accept="application/json",
                    )
                )
                connect_db.commit()
        except BaseException as exc:  # pragma: no cover - assertion reports it
            connect_errors.append(exc)
        finally:
            connect_finished.set()

    def disconnect() -> None:
        try:
            with factory() as disconnect_db:
                asyncio.run(
                    mcp_api.delete_mcp_server(
                        seed["server_id"],
                        current_user=disconnect_db.get(User, seed["user_id"]),
                        db=disconnect_db,
                    )
                )
        except BaseException as exc:  # pragma: no cover - assertion reports it
            disconnect_errors.append(exc)

    connect_thread = threading.Thread(target=connect, name="postgres-owner-connect")
    disconnect_thread = threading.Thread(
        target=disconnect, name="postgres-owner-disconnect"
    )
    try:
        connect_thread.start()
        assert discovery_started.wait(timeout=10)
        disconnect_thread.start()
        assert delete_locked.wait(timeout=10)
        allow_discovery.set()
        assert not connect_finished.wait(timeout=0.2)
        allow_delete.set()
        connect_thread.join(timeout=10)
        disconnect_thread.join(timeout=10)
    finally:
        allow_discovery.set()
        allow_delete.set()

    assert not connect_thread.is_alive()
    assert not disconnect_thread.is_alive()
    assert disconnect_errors == []
    assert len(connect_errors) == 1
    assert isinstance(connect_errors[0], HTTPException)
    assert connect_errors[0].status_code == 409
    with factory() as verify_db:
        assert (
            verify_db.query(UserMCPServer)
            .filter(
                UserMCPServer.user_id == seed["user_id"],
                UserMCPServer.mcpserver_id == seed["server_id"],
            )
            .count()
            == 0
        )
        assert verify_db.query(MCPOAuthFlowState).count() == 0
