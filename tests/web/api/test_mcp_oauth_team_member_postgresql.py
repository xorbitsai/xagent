"""PostgreSQL release gate for the team-member OAuth lock order."""

from __future__ import annotations

import asyncio
import re
import threading
import time
from datetime import timedelta
from types import SimpleNamespace
from urllib.parse import urlparse

import pytest
from sqlalchemy import event, text
from sqlalchemy.orm import sessionmaker
from starlette.requests import Request

from tests.shared.async_waits import DB_PROGRESS_TIMEOUT
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

URL = "https://mcp.example.com/mcp"
ISSUER = "https://auth.example.com"
# How long a call must stay blocked before it counts as "waiting on the lock".
BLOCKED_WINDOW_SECONDS = 1.0


@pytest.fixture()
def postgresql_engine():
    with disposable_database_factory("xagent_mcp_team_member") as make:
        engine = make("lock_order")
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


@pytest.fixture(autouse=True)
def _reset_hooks():
    yield
    connector_team_scope.set_connector_team_hooks()


def _seed(factory):
    with factory() as db:
        member = User(username="pg-team-member", password_hash="x")
        server = MCPServer.from_config(
            {
                "name": "records",
                "managed": "external",
                "transport": "streamable_http",
                "url": URL,
                "auth": {"type": "mcp_oauth"},
            }
        )
        server.auth = {"type": "mcp_oauth", "client_secret": encrypt_value("s")}
        db.add_all([member, server])
        db.flush()
        client = MCPOAuthClient(
            mcp_server_id=server.id,
            issuer=ISSUER,
            authorization_endpoint=f"{ISSUER}/authorize",
            token_endpoint=f"{ISSUER}/token",
            client_id="pg-client",
            token_endpoint_auth_method="none",
            redirect_uri="https://xagent.example.com/api/mcp/oauth/callback",
        )
        db.add(client)
        db.flush()
        flow = MCPOAuthFlowState(
            state="pg-team-flow",
            mcp_server_id=server.id,
            user_id=member.id,
            access_path="team_member",
            mcp_oauth_client_id=client.id,
            resource_owner_key=f"xagent:user:{member.id}",
            issuer=ISSUER,
            resource=URL,
            scope="records.read",
            code_verifier=encrypt_value("verifier"),
            expires_at=mcp_api._utc_now() + timedelta(minutes=10),
        )
        db.add(flow)
        db.commit()
        ids = {
            "member_id": int(member.id),
            "server_id": int(server.id),
            "client_id": int(client.id),
            "flow_id": int(flow.id),
        }
    connector_team_scope.set_connector_team_hooks(
        visibility=lambda _db, user_id: {
            "mcp": {ids["server_id"]},
            "custom_api": set(),
            "member_oauth_mcp": {ids["server_id"]},
        }
    )
    return ids


def _prepared() -> mcp_api._PreparedMCPOAuthConnect:
    discovery = SimpleNamespace(
        authorization_server=SimpleNamespace(
            issuer=ISSUER,
            authorization_endpoint=f"{ISSUER}/authorize",
            token_endpoint=f"{ISSUER}/token",
            registration_endpoint=None,
            raw={},
        ),
    )
    return mcp_api._PreparedMCPOAuthConnect(
        discovered=mcp_api._MCPOAuthConnectDiscovery(
            auth_config={},
            discovery=discovery,
            redirect_uri="https://xagent.example.com/api/mcp/oauth/callback",
            selected_issuer=ISSUER,
        ),
        client_id="pg-client",
        client_secret=None,
        token_endpoint_auth_method="none",
        registration_lookup_hash=None,
        selected_scope="records.read",
        selected_resource=URL,
    )


def _connect_call(factory, ids):
    with factory() as db:
        server = db.get(MCPServer, ids["server_id"])
        binding = mcp_api._member_grant_binding(server)
        db.rollback()
        return mcp_api._persist_team_member_mcp_oauth_connect_flow(
            db,
            server_id=ids["server_id"],
            user_id=ids["member_id"],
            prepared=_prepared(),
            binding_at_discovery=binding,
            redirect_after="/tools",
        )


def _callback_lock_call(factory, ids):
    with factory() as db:
        identity = mcp_api._MCPOAuthTeamFlowIdentity(
            id=ids["flow_id"],
            state="pg-team-flow",
            server_id=ids["server_id"],
            user_id=ids["member_id"],
            client_id=ids["client_id"],
        )
        locked = mcp_api._lock_team_member_mcp_oauth_lifecycle(
            db, flow_identity=identity, resource=URL
        )
        db.rollback()
        return locked is not None


@pytest.mark.parametrize(
    ("name", "call"),
    [("connect_persist", _connect_call), ("callback_lock", _callback_lock_call)],
    ids=["connect_persist", "callback_lock"],
)
def test_users_row_share_lock_blocks_on_exclusive_holder(
    postgresql_engine, name, call
) -> None:
    factory = sessionmaker(bind=postgresql_engine, autoflush=False, autocommit=False)
    ids = _seed(factory)

    holder = factory()
    holder.execute(
        text("SELECT id FROM users WHERE id = :id FOR NO KEY UPDATE"),
        {"id": ids["member_id"]},
    )
    results: list[object] = []
    errors: list[BaseException] = []

    def run() -> None:
        try:
            results.append(call(factory, ids))
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    try:
        worker.join(timeout=BLOCKED_WINDOW_SECONDS)
        assert worker.is_alive(), (
            f"{name} finished while another transaction held the user row"
        )
    finally:
        holder.commit()
        holder.close()
    worker.join(timeout=DB_PROGRESS_TIMEOUT)
    assert not worker.is_alive()
    assert errors == []
    assert results and results[0]


def test_team_member_paths_take_locks_in_order(postgresql_engine) -> None:
    factory = sessionmaker(bind=postgresql_engine, autoflush=False, autocommit=False)
    ids = _seed(factory)
    statements: list[str] = []

    def record(_conn, _cursor, statement, *_rest):
        normalized = " ".join(statement.split())
        match = re.search(
            r"FROM (\w+) .*?(FOR (?:NO KEY )?(?:UPDATE|SHARE))\b", normalized
        )
        if match and normalized.upper().startswith("SELECT"):
            statements.append(f"{match.group(1)} {match.group(2)}")

    event.listen(postgresql_engine, "before_cursor_execute", record)
    try:
        assert _callback_lock_call(factory, ids)
        callback_order = list(statements)
        statements.clear()
        assert _connect_call(factory, ids) is not None
        connect_order = [item for item in statements if "mcp_oauth_clients" not in item]
    finally:
        event.remove(postgresql_engine, "before_cursor_execute", record)

    assert callback_order == [
        "mcp_servers FOR UPDATE",
        "users FOR SHARE",
        "mcp_oauth_flow_states FOR UPDATE",
    ]
    assert connect_order[:2] == ["mcp_servers FOR UPDATE", "users FOR SHARE"]


def _callback_request(state: str) -> Request:
    parsed = urlparse(f"/api/mcp/oauth/callback?code=auth-code&state={state}")
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


# Far longer than the callback and the delete need; only a frozen event loop
# reaches it.
FREEZE_WATCHDOG_SECONDS = 20.0


def test_member_callback_does_not_freeze_the_event_loop(
    postgresql_engine, monkeypatch
) -> None:
    """The callback and a connector delete share one event loop.

    The delete runs its SQL on the loop thread and waits for the connector row
    lock the callback holds. If the callback's worker thread returned with its
    transaction still open, the loop could never run the commit and both would
    wait forever. The watchdog is a separate thread because a frozen loop
    cannot time itself out.
    """
    factory = sessionmaker(bind=postgresql_engine, autoflush=False, autocommit=False)
    ids = _seed(factory)
    with factory() as db:
        owner = User(username="pg-connector-owner", password_hash="x")
        db.add(owner)
        db.flush()
        db.add(
            UserMCPServer(
                user_id=owner.id,
                mcpserver_id=ids["server_id"],
                is_owner=True,
                can_delete=True,
                is_active=True,
            )
        )
        db.commit()
        owner_id = int(owner.id)

    locks_held = threading.Event()
    calls: list[int] = []

    def visibility(_db, user_id):
        calls.append(user_id)
        if len(calls) >= 2:
            # Second call is the one made under the row locks.
            locks_held.set()
            time.sleep(0.3)
        return {
            "mcp": {ids["server_id"]},
            "custom_api": set(),
            "member_oauth_mcp": {ids["server_id"]},
        }

    connector_team_scope.set_connector_team_hooks(visibility=visibility)

    async def fake_exchange(**_kwargs):
        return {"access_token": "member-token", "token_type": "Bearer"}

    monkeypatch.setattr(mcp_api, "_exchange_mcp_oauth_code", fake_exchange)

    finished = threading.Event()
    timed_out = threading.Event()

    def watchdog() -> None:
        if finished.wait(FREEZE_WATCHDOG_SECONDS):
            return
        timed_out.set()
        with postgresql_engine.connect() as admin:
            admin.execute(
                text(
                    "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                    "WHERE datname = current_database() AND pid <> pg_backend_pid()"
                )
            )

    threading.Thread(target=watchdog, daemon=True).start()

    async def scenario():
        with factory() as callback_db, factory() as delete_db:
            callback = asyncio.create_task(
                mcp_api.mcp_oauth_callback(
                    _callback_request("pg-team-flow"), callback_db
                )
            )
            # Wait off the loop for the callback to hold its locks.
            await asyncio.get_running_loop().run_in_executor(
                None, locks_held.wait, FREEZE_WATCHDOG_SECONDS
            )
            delete = mcp_api.delete_mcp_server(
                ids["server_id"],
                current_user=delete_db.get(User, owner_id),
                db=delete_db,
            )
            return await asyncio.gather(callback, delete, return_exceptions=True)

    try:
        results = asyncio.run(scenario())
    finally:
        finished.set()
    assert not timed_out.is_set(), "the event loop froze on the connector lock"
    callback_result, delete_result = results
    assert not isinstance(callback_result, BaseException), callback_result
    assert not isinstance(delete_result, BaseException), delete_result
    assert "mcp_oauth_success=1" in callback_result.headers["location"]


def _assert_no_row_lock_left_behind(engine, server_id: int) -> None:
    """Checked from other connections while the worker's session is still open."""
    with engine.connect() as other:
        other.execute(
            text("SELECT id FROM mcp_servers WHERE id = :id FOR UPDATE NOWAIT"),
            {"id": server_id},
        )
        other.rollback()
    with engine.connect() as other:
        open_writers = other.execute(
            text(
                "SELECT count(*) FROM pg_stat_activity "
                "WHERE datname = current_database() "
                "AND pid <> pg_backend_pid() AND backend_xid IS NOT NULL"
            )
        ).scalar_one()
        other.rollback()
    assert open_writers == 0


@pytest.mark.parametrize(
    "case",
    [
        "store_false_url_changed",
        "store_false_not_open",
        "store_upsert_raises",
        "persist_none_not_open",
        "persist_binding_raises",
    ],
)
def test_member_oauth_worker_calls_leave_no_row_lock_behind(
    postgresql_engine, monkeypatch, case
) -> None:
    factory = sessionmaker(bind=postgresql_engine, autoflush=False, autocommit=False)
    ids = _seed(factory)
    server_id = ids["server_id"]
    member_id = ids["member_id"]

    if "not_open" in case:
        connector_team_scope.set_connector_team_hooks(
            visibility=lambda _db, user_id: {
                "mcp": {server_id},
                "custom_api": set(),
                "member_oauth_mcp": set(),
            }
        )
    if case == "store_false_url_changed":
        with factory() as editor:
            editor.query(MCPServer).filter(MCPServer.id == server_id).update(
                {MCPServer.url: "https://mcp.example.com/moved"}
            )
            editor.commit()
    if case == "store_upsert_raises":

        def boom(*_args, **_kwargs):
            raise RuntimeError("write failed")

        monkeypatch.setattr(mcp_api, "_upsert_mcp_oauth_grant", boom)

    identity = mcp_api._MCPOAuthTeamFlowIdentity(
        id=ids["flow_id"],
        state="pg-team-flow",
        server_id=server_id,
        user_id=member_id,
        client_id=ids["client_id"],
    )

    async def run(db):
        if case.startswith("store_"):
            call = asyncio.to_thread(
                mcp_api._store_team_member_mcp_oauth_grant,
                db,
                flow_identity=identity,
                resource=URL,
                token_data={"access_token": "t", "token_type": "Bearer"},
            )
        else:
            server = db.get(MCPServer, server_id)
            binding = mcp_api._member_grant_binding(server)
            db.rollback()
            if case == "persist_binding_raises":

                def boom(_server):
                    raise RuntimeError("binding failed")

                monkeypatch.setattr(mcp_api, "_member_grant_binding", boom)
            call = asyncio.to_thread(
                mcp_api._persist_team_member_mcp_oauth_connect_flow,
                db,
                server_id=server_id,
                user_id=member_id,
                prepared=_prepared(),
                binding_at_discovery=binding,
                redirect_after="/tools",
            )
        try:
            return await call
        except RuntimeError as exc:
            return exc

    with factory() as db:
        result = asyncio.run(run(db))
        # The worker's session is still open: its transaction must already
        # be over.
        _assert_no_row_lock_left_behind(postgresql_engine, server_id)
    if case.startswith("store_false") or case == "persist_none_not_open":
        assert result in (False, None)
    else:
        assert isinstance(result, RuntimeError)
