from __future__ import annotations

import asyncio
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from xagent.core.utils.encryption import encrypt_value
from xagent.db.sqlite import apply_sqlite_concurrency_pragmas
from xagent.web.api import mcp as mcp_api
from xagent.web.models.database import Base
from xagent.web.models.mcp import MCPServer, UserMCPServer
from xagent.web.models.mcp_oauth import (
    MCPOAuthClient,
    MCPOAuthFlowState,
    MCPOAuthGrant,
)
from xagent.web.models.public_mcp import PublicMCPApp
from xagent.web.models.user import User
from xagent.web.models.user_oauth import UserOAuth


@pytest.fixture
def db() -> Session:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    @event.listens_for(engine, "connect")
    def enable_foreign_keys(dbapi_connection, _connection_record) -> None:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session
    engine.dispose()


def _user(db: Session, name: str = "workspace-account") -> User:
    user = User(username=name, password_hash="hash", is_admin=False)
    db.add(user)
    db.flush()
    return user


def _app(
    db: Session,
    *,
    app_id: str,
    name: str,
    transport: str,
    provider: str | None = None,
    launch_config: dict | None = None,
) -> PublicMCPApp:
    app = PublicMCPApp(
        app_id=app_id,
        name=name,
        transport=transport,
        provider_name=provider,
        oauth_scopes=[],
        launch_config=launch_config or {},
        is_visible_in_connector=True,
    )
    db.add(app)
    db.flush()
    return app


def _associate(db: Session, *, user: User, server: MCPServer) -> UserMCPServer:
    association = UserMCPServer(
        user_id=user.id,
        mcpserver_id=server.id,
        is_owner=True,
        is_active=True,
    )
    db.add(association)
    db.flush()
    return association


def _remote_oauth_state(
    db: Session, *, user: User, app_id: str = "remote-notes"
) -> tuple[PublicMCPApp, MCPServer, MCPOAuthClient]:
    app = _app(
        db,
        app_id=app_id,
        name="Remote Notes",
        transport="streamable_http",
        launch_config={
            "url": "https://mcp.example/mcp",
            "auth": {"type": "mcp_oauth"},
        },
    )
    server = MCPServer.from_config(
        {
            "name": app_id,
            "managed": "external",
            "transport": "streamable_http",
            "url": "https://mcp.example/mcp",
            "auth": {
                "type": "mcp_oauth",
                "resource": "https://mcp.example/mcp",
            },
        }
    )
    db.add(server)
    db.flush()
    _associate(db, user=user, server=server)
    client = MCPOAuthClient(
        mcp_server_id=server.id,
        issuer="https://auth.example",
        authorization_endpoint="https://auth.example/authorize",
        token_endpoint="https://auth.example/token",
        client_id="client-id",
        client_secret=encrypt_value("client-secret"),
        token_endpoint_auth_method="client_secret_post",
        redirect_uri="https://xagent.example/callback",
        metadata_json={"revocation_endpoint": "https://auth.example/revoke"},
    )
    db.add(client)
    db.flush()
    db.add_all(
        [
            MCPOAuthGrant(
                mcp_server_id=server.id,
                user_id=user.id,
                mcp_oauth_client_id=client.id,
                resource_owner_key=f"xagent:user:{user.id}",
                issuer="https://auth.example",
                resource="https://mcp.example/mcp",
                scope="notes.read",
                access_token=encrypt_value("access-token"),
                refresh_token=encrypt_value("refresh-token"),
                status="active",
            ),
            MCPOAuthFlowState(
                state=f"state-{app_id}",
                mcp_server_id=server.id,
                user_id=user.id,
                mcp_oauth_client_id=client.id,
                resource_owner_key=f"xagent:user:{user.id}",
                issuer="https://auth.example",
                resource="https://mcp.example/mcp",
                scope="notes.read",
                code_verifier=encrypt_value("verifier"),
                expires_at=datetime.now(timezone.utc) + timedelta(minutes=10),
            ),
        ]
    )
    db.commit()
    return app, server, client


@pytest.mark.asyncio
async def test_builtin_oauth_teardown_deletes_exact_credential_and_last_server(
    db: Session,
) -> None:
    user = _user(db)
    app = _app(
        db,
        app_id="calendar",
        name="Calendar",
        transport="oauth",
        provider="custom-calendar",
    )
    server = MCPServer(
        name="Calendar",
        managed="external",
        transport="oauth",
        auth={"app_id": "calendar", "provider": "custom-calendar"},
    )
    db.add(server)
    db.flush()
    _associate(db, user=user, server=server)
    db.add_all(
        [
            UserOAuth(
                user_id=user.id,
                provider="custom-calendar",
                access_token=encrypt_value("provider-token"),
            ),
            UserOAuth(
                user_id=user.id,
                provider="unrelated",
                access_token=encrypt_value("keep-token"),
            ),
        ]
    )
    db.commit()
    app_pk, server_id = int(app.id), int(server.id)

    await mcp_api.teardown_mcp_app_server(
        server_id,
        app_id="calendar",
        expected_catalog_app_id=app_pk,
        expected_provider_name="custom-calendar",
        current_user=user,
        db=db,
    )

    assert db.get(MCPServer, server_id) is None
    assert db.query(UserMCPServer).count() == 0
    assert {row.provider for row in db.query(UserOAuth).all()} == {"unrelated"}


@pytest.mark.asyncio
async def test_remote_oauth_last_user_cascades_client_secret_grant_and_flow(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    user = _user(db)
    app, server, client = _remote_oauth_state(db, user=user)
    app_pk, server_id, client_id = int(app.id), int(server.id), int(client.id)
    observed: list[tuple[int, int, int]] = []

    async def observe_after_commit(snapshot) -> None:
        observed.append(
            (
                db.query(MCPServer).count(),
                db.query(MCPOAuthClient).count(),
                snapshot.grant_id,
            )
        )

    monkeypatch.setattr(
        mcp_api, "_revoke_mcp_oauth_snapshot_externally", observe_after_commit
    )

    await mcp_api.teardown_mcp_app_server(
        server_id,
        app_id="remote-notes",
        expected_catalog_app_id=app_pk,
        expected_provider_name=None,
        current_user=user,
        db=db,
    )

    assert observed and observed[0][:2] == (0, 0)
    assert db.get(MCPServer, server_id) is None
    assert db.get(MCPOAuthClient, client_id) is None
    assert db.query(MCPOAuthGrant).count() == 0
    assert db.query(MCPOAuthFlowState).count() == 0
    assert db.query(UserMCPServer).count() == 0


@pytest.mark.asyncio
async def test_final_server_delete_failure_rolls_back_every_cleanup_for_retry(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    user = _user(db)
    app, server, client = _remote_oauth_state(db, user=user)
    app_pk, server_id, client_id = int(app.id), int(server.id), int(client.id)
    failures = {"remaining": 1}

    def fail_once(_mapper, _connection, target) -> None:
        if int(target.id) == server_id and failures["remaining"]:
            failures["remaining"] -= 1
            raise RuntimeError("simulated final delete failure")

    event.listen(MCPServer, "before_delete", fail_once)
    try:
        with pytest.raises(HTTPException) as exc_info:
            await mcp_api.teardown_mcp_app_server(
                server_id,
                app_id="remote-notes",
                expected_catalog_app_id=app_pk,
                expected_provider_name=None,
                current_user=user,
                db=db,
            )
        assert exc_info.value.status_code == 500

        assert db.get(MCPServer, server_id) is not None
        assert db.get(MCPOAuthClient, client_id) is not None
        assert db.query(MCPOAuthGrant).count() == 1
        assert db.query(MCPOAuthFlowState).count() == 1
        assert db.query(UserMCPServer).count() == 1

        async def ignore_revoke(_snapshot) -> None:
            return None

        monkeypatch.setattr(
            mcp_api, "_revoke_mcp_oauth_snapshot_externally", ignore_revoke
        )
        await mcp_api.teardown_mcp_app_server(
            server_id,
            app_id="remote-notes",
            expected_catalog_app_id=app_pk,
            expected_provider_name=None,
            current_user=user,
            db=db,
        )
    finally:
        event.remove(MCPServer, "before_delete", fail_once)

    assert db.get(MCPServer, server_id) is None
    assert db.get(MCPOAuthClient, client_id) is None
    assert db.query(MCPOAuthGrant).count() == 0
    assert db.query(MCPOAuthFlowState).count() == 0
    assert db.query(UserMCPServer).count() == 0


@pytest.mark.asyncio
async def test_recreated_catalog_owner_fails_closed_before_cleanup(db: Session) -> None:
    user = _user(db)
    app = _app(
        db,
        app_id="legacy-mail",
        name="Legacy Mail",
        transport="oauth",
        provider="legacy-provider",
    )
    server = MCPServer(
        name="Legacy Mail", managed="external", transport="oauth", auth=None
    )
    db.add(server)
    db.flush()
    _associate(db, user=user, server=server)
    credential = UserOAuth(
        user_id=user.id,
        provider="legacy-provider",
        access_token=encrypt_value("must-survive"),
    )
    db.add(credential)
    _app(
        db,
        app_id="pk-keeper",
        name="PK Keeper",
        transport="stdio",
    )
    db.commit()
    expected_pk, server_id = int(app.id), int(server.id)

    db.delete(app)
    db.commit()
    replacement = _app(
        db,
        app_id="legacy-mail",
        name="Legacy Mail",
        transport="oauth",
        provider="replacement-provider",
    )
    db.commit()
    assert int(replacement.id) != expected_pk

    with pytest.raises(HTTPException) as exc_info:
        await mcp_api.teardown_mcp_app_server(
            server_id,
            app_id="legacy-mail",
            expected_catalog_app_id=expected_pk,
            expected_provider_name="legacy-provider",
            current_user=user,
            db=db,
        )

    assert exc_info.value.status_code == 403
    assert db.get(MCPServer, server_id) is not None
    assert db.query(UserMCPServer).count() == 1
    assert db.query(UserOAuth).filter_by(provider="legacy-provider").count() == 1


@pytest.mark.asyncio
async def test_provider_drift_fails_closed_before_cleanup(db: Session) -> None:
    user = _user(db)
    app = _app(
        db,
        app_id="calendar",
        name="Calendar",
        transport="oauth",
        provider="original-provider",
    )
    server = MCPServer(
        name="Calendar",
        managed="external",
        transport="oauth",
        auth={"app_id": "calendar", "provider": "original-provider"},
    )
    db.add(server)
    db.flush()
    _associate(db, user=user, server=server)
    db.add_all(
        [
            UserOAuth(
                user_id=user.id,
                provider="original-provider",
                access_token=encrypt_value("original-secret"),
            ),
            UserOAuth(
                user_id=user.id,
                provider="replacement-provider",
                access_token=encrypt_value("replacement-secret"),
            ),
        ]
    )
    db.commit()
    app_pk, server_id = int(app.id), int(server.id)

    app.provider_name = "replacement-provider"
    db.commit()

    with pytest.raises(HTTPException) as exc_info:
        await mcp_api.teardown_mcp_app_server(
            server_id,
            app_id="calendar",
            expected_catalog_app_id=app_pk,
            expected_provider_name="original-provider",
            current_user=user,
            db=db,
        )

    assert exc_info.value.status_code == 403
    assert db.get(MCPServer, server_id) is not None
    assert db.query(UserMCPServer).count() == 1
    assert {row.provider for row in db.query(UserOAuth).all()} == {
        "original-provider",
        "replacement-provider",
    }


@pytest.mark.asyncio
async def test_multi_user_teardown_deletes_all_current_user_grant_statuses(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    user = _user(db)
    other_user = _user(db, "other-account")
    app, server, client = _remote_oauth_state(db, user=user)
    app_pk, server_id = int(app.id), int(server.id)
    _associate(db, user=other_user, server=server)
    db.add_all(
        [
            MCPOAuthGrant(
                mcp_server_id=server_id,
                user_id=user.id,
                mcp_oauth_client_id=client.id,
                resource_owner_key=f"xagent:user:{user.id}",
                issuer="https://auth.example",
                resource="https://mcp.example/mcp",
                scope="notes.revoked",
                access_token=encrypt_value("revoked-access-token"),
                refresh_token=encrypt_value("revoked-refresh-token"),
                status="revoked",
            ),
            MCPOAuthGrant(
                mcp_server_id=server_id,
                user_id=user.id,
                mcp_oauth_client_id=client.id,
                resource_owner_key=f"xagent:user:{user.id}",
                issuer="https://auth.example",
                resource="https://mcp.example/mcp",
                scope="notes.inactive",
                access_token=encrypt_value("inactive-access-token"),
                refresh_token=encrypt_value("inactive-refresh-token"),
                status="inactive",
            ),
            MCPOAuthGrant(
                mcp_server_id=server_id,
                user_id=other_user.id,
                mcp_oauth_client_id=client.id,
                resource_owner_key=f"xagent:user:{other_user.id}",
                issuer="https://auth.example",
                resource="https://mcp.example/mcp",
                scope="notes.read",
                access_token=encrypt_value("other-access-token"),
                refresh_token=encrypt_value("other-refresh-token"),
                status="active",
            ),
        ]
    )
    db.commit()
    active_grant_id = int(
        db.query(MCPOAuthGrant.id).filter_by(user_id=user.id, status="active").scalar()
    )
    revoked_grant_ids: list[int] = []

    async def observe_revoke(snapshot) -> None:
        revoked_grant_ids.append(snapshot.grant_id)

    monkeypatch.setattr(
        mcp_api, "_revoke_mcp_oauth_snapshot_externally", observe_revoke
    )

    await mcp_api.teardown_mcp_app_server(
        server_id,
        app_id="remote-notes",
        expected_catalog_app_id=app_pk,
        expected_provider_name=None,
        current_user=user,
        db=db,
    )

    assert revoked_grant_ids == [active_grant_id]
    assert db.get(MCPServer, server_id) is not None
    assert db.get(MCPOAuthClient, int(client.id)) is not None
    assert db.query(MCPOAuthGrant).filter_by(user_id=user.id).count() == 0
    assert db.query(MCPOAuthGrant).filter_by(user_id=other_user.id).count() == 1
    assert db.query(MCPOAuthFlowState).filter_by(user_id=user.id).count() == 0
    assert db.query(UserMCPServer).filter_by(user_id=user.id).count() == 0
    assert db.query(UserMCPServer).filter_by(user_id=other_user.id).count() == 1


@pytest.mark.asyncio
async def test_unexpected_teardown_failure_logs_no_exception_detail(
    db: Session,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    user = _user(db)
    db.commit()
    secret_detail = "raw upstream detail access_token=audit-secret"

    def fail_lock(_db: Session) -> None:
        raise RuntimeError(secret_detail)

    monkeypatch.setattr(mcp_api, "_lock_catalog_for_app_teardown", fail_lock)
    with caplog.at_level(logging.ERROR, logger=mcp_api.logger.name):
        with pytest.raises(HTTPException) as exc_info:
            await mcp_api.teardown_mcp_app_server(
                42,
                app_id="safe-app-id",
                expected_catalog_app_id=7,
                expected_provider_name=None,
                current_user=user,
                db=db,
            )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to delete MCP server"
    assert "safe-app-id" in caplog.text
    assert "server 42" in caplog.text
    assert secret_detail not in caplog.text
    assert "audit-secret" not in caplog.text


@pytest.mark.asyncio
async def test_sqlite_teardown_rejects_without_discarding_caller_writes(
    db: Session,
) -> None:
    user = _user(db)
    app = _app(
        db,
        app_id="calendar",
        name="Calendar",
        transport="oauth",
        provider="calendar-provider",
    )
    server = MCPServer(
        name="Calendar",
        managed="external",
        transport="oauth",
        auth={"app_id": "calendar", "provider": "calendar-provider"},
    )
    db.add(server)
    db.flush()
    _associate(db, user=user, server=server)
    db.commit()
    db.refresh(user)
    app_pk, server_id = int(app.id), int(server.id)
    pending = UserOAuth(
        user_id=user.id,
        provider="unrelated-pending-write",
        access_token=encrypt_value("must-remain-pending"),
    )
    db.add(pending)

    with pytest.raises(HTTPException) as exc_info:
        await mcp_api.teardown_mcp_app_server(
            server_id,
            app_id="calendar",
            expected_catalog_app_id=app_pk,
            expected_provider_name="calendar-provider",
            current_user=user,
            db=db,
        )

    assert exc_info.value.status_code == 500
    assert pending in db.new
    with db.no_autoflush:
        assert db.get(MCPServer, server_id) is not None
        assert db.query(UserMCPServer).filter_by(mcpserver_id=server_id).count() == 1


@pytest.mark.parametrize("mutation", ["provider-drift", "delete-recreate"])
def test_sqlite_catalog_mutation_waits_for_teardown_write_fence(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    engine = create_engine(
        f"sqlite:///{tmp_path / 'teardown-race.db'}",
        connect_args={"check_same_thread": False},
    )
    apply_sqlite_concurrency_pragmas(engine, busy_timeout_ms=10_000)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    with factory() as seed_db:
        user = _user(seed_db)
        app = _app(
            seed_db,
            app_id="calendar",
            name="Calendar",
            transport="oauth",
            provider="original-provider",
        )
        _app(seed_db, app_id="pk-keeper", name="PK Keeper", transport="stdio")
        server = MCPServer(
            name="Calendar",
            managed="external",
            transport="oauth",
            auth={"app_id": "calendar", "provider": "original-provider"},
        )
        seed_db.add(server)
        seed_db.flush()
        _associate(seed_db, user=user, server=server)
        seed_db.add_all(
            [
                UserOAuth(
                    user_id=user.id,
                    provider="original-provider",
                    access_token=encrypt_value("delete-me"),
                ),
                UserOAuth(
                    user_id=user.id,
                    provider="replacement-provider",
                    access_token=encrypt_value("keep-me"),
                ),
            ]
        )
        seed_db.commit()
        user_id, app_pk, server_id = int(user.id), int(app.id), int(server.id)

    identity_checked = threading.Event()
    release_teardown = threading.Event()
    mutation_sent = threading.Event()
    mutation_finished = threading.Event()
    mutation_thread_id: list[int] = []
    real_gate = mcp_api._locked_catalog_app_for_server

    def gated_identity(*args, **kwargs):
        result = real_gate(*args, **kwargs)
        identity_checked.set()
        assert release_teardown.wait(timeout=10)
        return result

    monkeypatch.setattr(mcp_api, "_locked_catalog_app_for_server", gated_identity)

    expected_prefix = (
        "UPDATE public_mcp_apps"
        if mutation == "provider-drift"
        else "DELETE FROM public_mcp_apps"
    )

    def observe_mutation(
        _connection, _cursor, statement, _parameters, _context, _executemany
    ) -> None:
        if (
            mutation_thread_id
            and threading.get_ident() == mutation_thread_id[0]
            and statement.lstrip().startswith(expected_prefix)
        ):
            mutation_sent.set()

    event.listen(engine, "before_cursor_execute", observe_mutation)

    def teardown() -> None:
        with factory() as teardown_db:
            current_user = teardown_db.get(User, user_id)
            assert current_user is not None
            asyncio.run(
                mcp_api.teardown_mcp_app_server(
                    server_id,
                    app_id="calendar",
                    expected_catalog_app_id=app_pk,
                    expected_provider_name="original-provider",
                    current_user=current_user,
                    db=teardown_db,
                )
            )

    def mutate_catalog() -> None:
        mutation_thread_id.append(threading.get_ident())
        try:
            with factory() as mutation_db:
                expected = mutation_db.get(PublicMCPApp, app_pk)
                assert expected is not None
                if mutation == "provider-drift":
                    expected.provider_name = "replacement-provider"
                    mutation_db.commit()
                else:
                    mutation_db.delete(expected)
                    mutation_db.commit()
                    mutation_db.add(
                        PublicMCPApp(
                            app_id="calendar",
                            name="Calendar Replacement",
                            transport="oauth",
                            provider_name="replacement-provider",
                            oauth_scopes=[],
                            launch_config={},
                            is_visible_in_connector=True,
                        )
                    )
                    mutation_db.commit()
        finally:
            mutation_finished.set()

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            teardown_future = executor.submit(teardown)
            assert identity_checked.wait(timeout=10)
            mutation_future = executor.submit(mutate_catalog)
            assert mutation_sent.wait(timeout=10)
            assert not mutation_finished.wait(timeout=0.25)
            release_teardown.set()
            teardown_future.result(timeout=10)
            mutation_future.result(timeout=10)
    finally:
        release_teardown.set()
        event.remove(engine, "before_cursor_execute", observe_mutation)

    with factory() as verify_db:
        assert verify_db.get(MCPServer, server_id) is None
        assert (
            verify_db.query(UserOAuth).filter_by(provider="original-provider").count()
            == 0
        )
        assert (
            verify_db.query(UserOAuth)
            .filter_by(provider="replacement-provider")
            .count()
            == 1
        )
    engine.dispose()
