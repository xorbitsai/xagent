"""PostgreSQL release gate for MCP OAuth producer lifecycle fencing."""

from __future__ import annotations

import asyncio
import threading
from datetime import timedelta
from types import SimpleNamespace
from typing import Any
from urllib.parse import urlparse

import pytest
from sqlalchemy.orm import sessionmaker
from starlette.requests import Request

from tests.shared.postgres_disposable import disposable_database_factory
from xagent.core.utils.encryption import encrypt_value
from xagent.web.api import mcp as mcp_api
from xagent.web.models import MCPOAuthClient, MCPOAuthFlowState, MCPOAuthGrant
from xagent.web.models.database import Base
from xagent.web.models.mcp import MCPServer, UserMCPServer
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
        db.add_all([user, other_user, server])
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
        }


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

    def gated_team_delete(*args):
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
        lambda *args: SimpleNamespace(
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
        lambda *args: SimpleNamespace(
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
