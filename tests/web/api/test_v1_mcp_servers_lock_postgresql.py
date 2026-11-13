"""Real-PostgreSQL coverage for the row locks behind ``PATCH /v1/mcp-servers/{id}``.

``FOR UPDATE`` is a no-op on SQLite, so the suites that run there can count lock
calls but cannot show that a second connection really waits, nor which lock
strength reaches the server. This file runs the update through the service the
route calls, on a real PostgreSQL database with two connections:

* one connection plays another writer that holds the connector's definition row
  and has not committed;
* the other runs the update and is expected to wait for it.

The SQL the engine actually sent is read back and counted by table name. Only
statements whose main table is ``mcp_servers`` and which carry ``FOR NO KEY
UPDATE`` count as definition-row locks. A change of ``url`` also makes the web
function lock rows of ``mcp_oauth_flow_states`` and ``mcp_oauth_grants`` with a
plain ``FOR UPDATE``; counting every locking statement would include them.

Like the other ``*_postgresql.py`` suites, the database comes from
``tests/shared/postgres_disposable.py``, which reads ``XAGENT_TEST_POSTGRES_URL``
and skips the test when it is not set.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import re
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker

from tests.shared.async_waits import DB_PROGRESS_TIMEOUT
from tests.shared.postgres_disposable import disposable_database_factory
from xagent.web.api.v1.errors import V1ApiError
from xagent.web.api.v1.mcp_servers import _reject_invalid_update
from xagent.web.models.database import Base
from xagent.web.models.mcp import MCPServer, UserMCPServer
from xagent.web.models.mcp_oauth import MCPOAuthClient, MCPOAuthGrant
from xagent.web.models.user import User
from xagent.web.schemas.v1 import V1McpServerUpdateRequest
from xagent.web.services import connector_team_scope, mcp_server_management
from xagent.web.services.mcp_server_management import McpServerManagementRuntime

pytestmark = pytest.mark.postgresql

ISSUER = "https://auth.example.test"
STORED_URL = "https://mcp.example.test/mcp"
MOVED_URL = "https://moved.example.test/mcp"
OAUTH_AUTH = {
    "type": "mcp_oauth",
    "client_id": "web-client",
    "scope": "read",
    "issuer": ISSUER,
    "resource_metadata_url": "https://mcp.example.test/.well-known/resource",
}
MASKED = "********"
LOCK_TEXT = "FOR NO KEY UPDATE"
DEFINITION_ROW_READ = re.compile(r"\bFROM mcp_servers\b")

# Which update the statements being recorded belong to. ``asyncio.to_thread``
# copies the context, so the worker thread of the update under test sees it and
# the other connection, running in the test thread, does not.
_RECORDING: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "recording", default=False
)


@dataclass
class _Statement:
    text: str
    started: float
    seconds: float = 0.0


@dataclass
class _Recorded:
    statements: list[_Statement] = field(default_factory=list)

    def definition_locks(self) -> list[int]:
        """Positions of the definition-row lock statements, by table name."""
        return [
            index
            for index, item in enumerate(self.statements)
            if DEFINITION_ROW_READ.search(item.text) and LOCK_TEXT in item.text
        ]

    def positions(self, pattern: str) -> list[int]:
        return [
            index
            for index, item in enumerate(self.statements)
            if re.search(pattern, item.text)
        ]


@contextlib.contextmanager
def _recording(session_factory):
    engine = session_factory.kw["bind"]
    recorded = _Recorded()
    pending: list[_Statement] = []

    def before(_conn, _cursor, statement, _parameters, _context, _executemany):
        if _RECORDING.get():
            item = _Statement(statement, time.monotonic())
            pending.append(item)
            recorded.statements.append(item)

    def after(_conn, _cursor, _statement, _parameters, _context, _executemany):
        if _RECORDING.get() and pending:
            item = pending.pop()
            item.seconds = time.monotonic() - item.started

    sa.event.listen(engine, "before_cursor_execute", before)
    sa.event.listen(engine, "after_cursor_execute", after)
    try:
        yield recorded
    finally:
        sa.event.remove(engine, "before_cursor_execute", before)
        sa.event.remove(engine, "after_cursor_execute", after)


@pytest.fixture()
def session_factory(monkeypatch):
    with disposable_database_factory("xagent_v1_mcp_lock") as make_database:
        engine = make_database("v1_update")
        Base.metadata.create_all(bind=engine)
        factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        # The service opens its own sessions; point them at this database.
        monkeypatch.setattr(mcp_server_management, "get_session_local", lambda: factory)
        connector_team_scope.set_connector_team_hooks()
        yield factory
        connector_team_scope.set_connector_team_hooks()


def _user(db, name: str) -> int:
    user = User(username=name, password_hash="x", is_admin=False)
    db.add(user)
    db.flush()
    return int(user.id)


@pytest.fixture()
def world(session_factory) -> SimpleNamespace:
    """bob owns a connector that has a stored request header and an OAuth setup."""
    with session_factory() as db:
        bob = _user(db, "bob")
        carol = _user(db, "carol")
        dave = _user(db, "dave")
        server = MCPServer(
            name="locked-connector",
            managed="external",
            transport="streamable_http",
            url=STORED_URL,
            headers={"X-Key": "original-value"},
            auth=OAUTH_AUTH,
        )
        db.add(server)
        db.flush()
        db.add(
            UserMCPServer(
                user_id=bob,
                mcpserver_id=server.id,
                is_owner=True,
                can_delete=False,
                is_active=True,
            )
        )
        client = MCPOAuthClient(
            mcp_server_id=server.id,
            issuer=ISSUER,
            authorization_endpoint=f"{ISSUER}/authorize",
            token_endpoint=f"{ISSUER}/token",
            client_id="web-client",
            redirect_uri="https://xagent.example.test/callback",
        )
        db.add(client)
        db.flush()
        ids = SimpleNamespace(
            bob=bob,
            carol=carol,
            dave=dave,
            server=int(server.id),
            client=int(client.id),
        )
        db.commit()
    return ids


def _grant(world, user_id: int) -> MCPOAuthGrant:
    return MCPOAuthGrant(
        mcp_server_id=world.server,
        user_id=user_id,
        mcp_oauth_client_id=world.client,
        lookup_hash=uuid.uuid4().hex,
        resource_owner_key=f"xagent:user:{user_id}",
        issuer=ISSUER,
        resource=STORED_URL,
        scope=uuid.uuid4().hex,
        access_token="encrypted",
        status="active",
        expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
    )


def _hold_definition_row(db, world) -> None:
    """Take the row lock a writer in the middle of a transaction would hold."""
    row = (
        db.query(MCPServer).filter(MCPServer.id == world.server).with_for_update().one()
    )
    assert row.id == world.server


def _run_update(world, body: dict) -> object:
    """The update a holder's PATCH runs, on its own thread and event loop."""
    request = V1McpServerUpdateRequest(**body)

    async def update():
        _RECORDING.set(True)
        return await McpServerManagementRuntime().update_server(
            user_id=world.bob,
            key_prefix="test-key",
            server_id=world.server,
            request=request,
            validate=_reject_invalid_update,
        )

    try:
        return asyncio.run(update())
    except V1ApiError as exc:
        return exc


def _wait_until_a_backend_waits_for_a_lock(session_factory) -> None:
    """Return once some connection of this database is blocked on a lock."""
    deadline = time.monotonic() + DB_PROGRESS_TIMEOUT
    while time.monotonic() < deadline:
        with session_factory() as probe:
            waiting = probe.execute(
                sa.text(
                    "SELECT count(*) FROM pg_stat_activity "
                    "WHERE datname = current_database() "
                    "AND wait_event_type = 'Lock'"
                )
            ).scalar_one()
        if waiting:
            return
        time.sleep(0.05)
    raise AssertionError("no connection ever waited for a lock")


def _update_waits_for(
    session_factory, world, body: dict, release, recorded: _Recorded
) -> object:
    """Start the update, prove it is blocked, run ``release``, return its result.

    Some definition-row lock statement must have started before ``release`` ran
    and finished after it, so the wait was on that lock and not on a later one.
    """
    with ThreadPoolExecutor(max_workers=1) as executor:
        future: Future = executor.submit(_run_update, world, body)
        try:
            _wait_until_a_backend_waits_for_a_lock(session_factory)
            assert not future.done(), "the update finished while the row was held"
            released_at = time.monotonic()
        finally:
            release()
        result = future.result(timeout=DB_PROGRESS_TIMEOUT)
    assert any(
        recorded.statements[index].started
        < released_at
        < recorded.statements[index].started + recorded.statements[index].seconds
        for index in recorded.definition_locks()
    ), "no definition-row lock statement spanned the moment the row was released"
    return result


def _stored(session_factory, world) -> MCPServer:
    with session_factory() as db:
        row = db.query(MCPServer).filter(MCPServer.id == world.server).one()
        db.expunge(row)
        return row


def test_a_headers_only_update_merges_the_value_another_writer_committed(
    session_factory, world
) -> None:
    """Another writer holds the definition row and changes ``X-Key``. A request
    that resends the masked ``X-Key`` and adds ``X-Extra`` waits for it, and what
    is stored afterwards is the ``X-Key`` that writer committed plus the new
    ``X-Extra``. It takes the definition row lock once, in the web function:
    the service has no lock of its own to take for a body without ``url`` or
    ``auth``."""
    writer = session_factory()
    try:
        _hold_definition_row(writer, world)
        writer.execute(
            sa.update(MCPServer)
            .where(MCPServer.id == world.server)
            .values(headers={"X-Key": "committed-by-the-other-writer"})
        )
        with _recording(session_factory) as recorded:
            result = _update_waits_for(
                session_factory,
                world,
                {"headers": {"X-Key": MASKED, "X-Extra": "added"}},
                writer.commit,
                recorded,
            )
    finally:
        writer.close()

    assert not isinstance(result, Exception), result
    assert _stored(session_factory, world).headers == {
        "X-Key": "committed-by-the-other-writer",
        "X-Extra": "added",
    }
    assert len(recorded.definition_locks()) == 1, [
        item.text for item in recorded.statements
    ]


def test_a_url_change_waits_for_a_concurrent_authorization_and_is_refused(
    session_factory, world
) -> None:
    """Another writer holds the definition row and, in the same transaction, gives
    carol a link and an active grant. A request from the owner that moves ``url``
    waits, then sees that grant and is refused with the url unchanged. The service
    takes the only definition-row lock, so the grant query runs after the wait;
    the web function is never reached."""
    writer = session_factory()
    try:
        _hold_definition_row(writer, world)
        writer.add(
            UserMCPServer(
                user_id=world.carol,
                mcpserver_id=world.server,
                is_owner=False,
                can_delete=False,
                is_active=True,
            )
        )
        writer.add(_grant(world, world.carol))
        writer.flush()
        with _recording(session_factory) as recorded:
            result = _update_waits_for(
                session_factory, world, {"url": MOVED_URL}, writer.commit, recorded
            )
    finally:
        writer.close()

    assert isinstance(result, V1ApiError), result
    assert result.http_status == 409
    assert result.code.value == "connector_not_editable"
    assert _stored(session_factory, world).url == STORED_URL
    locks = recorded.definition_locks()
    assert len(locks) == 1, [item.text for item in recorded.statements]
    grant_reads = recorded.positions(r"\bFROM mcp_oauth_grants\b")
    assert grant_reads and min(grant_reads) > locks[0], (
        "the grants were read before the definition row was locked"
    )


def test_a_url_change_locks_the_definition_row_twice_without_waiting_on_itself(
    session_factory, world
) -> None:
    """Only dave has a team-path authorization, so the change is allowed. The
    service locks the definition row and the web function locks it again in the
    same transaction. PostgreSQL never makes a transaction wait for a row lock
    it already holds, while a second lock taken on another connection would
    block and keep the update from finishing within ``DB_PROGRESS_TIMEOUT``;
    so the update finishing shows that the second statement did not wait. The
    change is stored and dave's grant is removed. The locks on the flow and
    grant tables that the web function takes on a ``url`` change are not
    definition-row locks."""
    with session_factory() as db:
        db.add(_grant(world, world.dave))
        db.commit()

    with _recording(session_factory) as recorded:
        with ThreadPoolExecutor(max_workers=1) as executor:
            result = executor.submit(_run_update, world, {"url": MOVED_URL}).result(
                timeout=DB_PROGRESS_TIMEOUT
            )

    assert not isinstance(result, Exception), result
    assert _stored(session_factory, world).url == MOVED_URL
    with session_factory() as db:
        assert (
            db.query(MCPOAuthGrant).filter_by(mcp_server_id=world.server).count() == 0
        )
    locks = recorded.definition_locks()
    assert len(locks) == 2, [item.text for item in recorded.statements]
    for table in ("mcp_oauth_flow_states", "mcp_oauth_grants"):
        other = [
            index
            for index in recorded.positions(rf"\bFROM {table}\b")
            if "FOR UPDATE" in recorded.statements[index].text
            and LOCK_TEXT not in recorded.statements[index].text
        ]
        assert len(other) == 1, (table, other)
        assert other[0] not in locks


_SCENARIOS = (
    test_a_headers_only_update_merges_the_value_another_writer_committed,
    test_a_url_change_waits_for_a_concurrent_authorization_and_is_refused,
    test_a_url_change_locks_the_definition_row_twice_without_waiting_on_itself,
)


def test_the_suite_has_three_scenarios() -> None:
    assert len(_SCENARIOS) == 3
    in_module = [
        name
        for name, value in globals().items()
        if name.startswith("test_") and callable(value) and value in _SCENARIOS
    ]
    assert len(in_module) == 3
