"""Integration tests for the /v1/* personal management auth dependency.

Drives /v1/me to verify each personal-key failure path returns the
stable ``{"error": {"code": "invalid_api_key", ...}}`` envelope.

Test plumbing (client, _test_db fixture, auth helpers) is shared via
``tests/web/api/conftest.py``.
"""

import asyncio
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import bcrypt
import pytest
from fastapi.security import HTTPAuthorizationCredentials
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool

from xagent.core.utils.api_key import BCRYPT_COST
from xagent.web.models.agent import Agent, AgentOrigin
from xagent.web.models.user_api_key import UserApiKey

from ..conftest import (
    _admin_headers,
    _direct_db_session,
    _install_one_slot_queue_pool,
    client,
)

# Opt this file into the shared conftest ``_test_db`` fixture. See the
# note in test_agent_api_keys.py for why we use ``usefixtures`` with a
# string name rather than importing the fixture directly.
pytestmark = pytest.mark.usefixtures("_test_db")


def _create_agent_and_key() -> tuple[int, str, str]:
    """Helper: create an agent + generate its first API key.

    Returns: (agent_id, full_key, key_prefix)
    """
    headers = _admin_headers()
    agent_resp = client.post(
        "/api/agents",
        headers=headers,
        json={
            "name": "v1 auth test agent",
            "description": "for /v1/* auth tests",
            "instructions": "test",
            "execution_mode": "balanced",
        },
    )
    assert agent_resp.status_code == 200, agent_resp.text
    agent_id = agent_resp.json()["id"]

    key_resp = client.post(f"/api/agents/{agent_id}/api-key", headers=headers)
    assert key_resp.status_code == 200, key_resp.text
    body = key_resp.json()
    return agent_id, body["full_key"], body["key_prefix"]


def _create_personal_key() -> tuple[str, str]:
    """Helper: create a personal management key for the admin user."""
    headers = _admin_headers()
    key_resp = client.post("/api/me/personal-keys", headers=headers)
    assert key_resp.status_code == 200, key_resp.text
    body = key_resp.json()
    return body["full_key"], body["key_prefix"]


def _mark_generated_manager(agent_id: int) -> None:
    db = _direct_db_session()
    try:
        db.query(Agent).filter(Agent.id == agent_id).update(
            {"origin": AgentOrigin.WORKFORCE_GENERATED_MANAGER.value}
        )
        db.commit()
    finally:
        db.close()


# ===== happy path =====


def test_valid_personal_key_returns_me_response():
    """A freshly generated personal key authenticates /v1/me."""
    full_key, prefix = _create_personal_key()

    resp = client.get("/v1/me", headers={"Authorization": f"Bearer {full_key}"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["principal_type"] == "user"
    assert body["user_id"] > 0
    # admin fixture: username="admin", email="admin@example.com" -- the two
    # differ, so this pins that each field carries its own value.
    assert body["username"] == "admin"
    assert body["email"] == "admin@example.com"
    assert body["key_prefix"] == prefix


def test_agent_runtime_key_cannot_authenticate_me():
    """Runtime keys are not accepted by management identity endpoints."""
    _agent_id, full_key, _prefix = _create_agent_and_key()
    resp = client.get("/v1/me", headers={"Authorization": f"Bearer {full_key}"})
    _assert_invalid_api_key(resp)


# ===== failure paths -- all must return the same envelope =====


def _assert_invalid_api_key(resp) -> None:
    """Every auth failure should respond with the same shape."""
    assert resp.status_code == 401, resp.text
    body = resp.json()
    assert body == {
        "error": {
            "code": "invalid_api_key",
            "message": body["error"]["message"],  # message is free text
        }
    }
    # Ensure no internal SQL message or raw exception slipped into message
    msg = body["error"]["message"]
    assert "bcrypt" not in msg.lower()
    assert "sqlalchemy" not in msg.lower()


def test_missing_authorization_header_returns_401():
    resp = client.get("/v1/me")
    _assert_invalid_api_key(resp)


def test_malformed_authorization_header_returns_401():
    resp = client.get("/v1/me", headers={"Authorization": "Bearer not_a_key"})
    _assert_invalid_api_key(resp)


def test_wrong_brand_prefix_returns_401():
    resp = client.get(
        "/v1/me", headers={"Authorization": "Bearer sk_ABCDEF_" + "x" * 32}
    )
    _assert_invalid_api_key(resp)


def test_unknown_prefix_returns_401():
    """A well-formed key with a prefix that's never been issued."""
    fake_key = "xag_personal_ZZZZZZ_" + "x" * 32
    resp = client.get("/v1/me", headers={"Authorization": f"Bearer {fake_key}"})
    _assert_invalid_api_key(resp)


def test_known_prefix_wrong_secret_returns_401():
    """Prefix is real but the secret doesn't bcrypt-match."""
    full_key, _prefix = _create_personal_key()
    # Replace just the secret half with a different (but well-formed) value
    parts = full_key.split("_")
    parts[3] = "y" * 32
    wrong_key = "_".join(parts)
    resp = client.get("/v1/me", headers={"Authorization": f"Bearer {wrong_key}"})
    _assert_invalid_api_key(resp)


def test_revoked_key_returns_401():
    """Once DELETE rotates / revokes, the old key must stop working."""
    full_key, prefix = _create_personal_key()
    admin = _admin_headers()
    keys = client.get("/api/me/personal-keys", headers=admin)
    assert keys.status_code == 200
    key_id = next(row["id"] for row in keys.json() if row["key_prefix"] == prefix)
    revoke = client.delete(f"/api/me/personal-keys/{key_id}", headers=admin)
    assert revoke.status_code == 200
    assert revoke.json()["revoked"] is True

    resp = client.get("/v1/me", headers={"Authorization": f"Bearer {full_key}"})
    _assert_invalid_api_key(resp)


def _set_key_expiry(prefix: str, expires_at) -> None:
    """Force a personal key's ``expires_at`` to a fixed value via direct DB
    write, bypassing HTTP (the create endpoint leaves it null)."""
    db = _direct_db_session()
    try:
        db.query(UserApiKey).filter(UserApiKey.key_prefix == prefix).update(
            {"expires_at": expires_at}
        )
        db.commit()
    finally:
        db.close()


def test_expired_key_with_naive_expiry_returns_401_not_500():
    """An expired key must yield 401, even when ``expires_at`` reads back
    naive (as ``DateTime(timezone=True)`` does on SQLite).

    Comparing a naive ``expires_at`` against an aware ``now`` raises
    TypeError -- which would surface as a 500. The auth dep normalizes
    to aware UTC first, so the expiry check stays a clean 401.
    """
    full_key, prefix = _create_personal_key()
    naive_past = (datetime.now(timezone.utc) - timedelta(days=1)).replace(tzinfo=None)
    _set_key_expiry(prefix, naive_past)

    resp = client.get("/v1/me", headers={"Authorization": f"Bearer {full_key}"})
    _assert_invalid_api_key(resp)


def test_unexpired_key_with_naive_future_expiry_authenticates():
    """A future, naive ``expires_at`` must not be misread as expired."""
    full_key, prefix = _create_personal_key()
    naive_future = (datetime.now(timezone.utc) + timedelta(days=1)).replace(tzinfo=None)
    _set_key_expiry(prefix, naive_future)

    resp = client.get("/v1/me", headers={"Authorization": f"Bearer {full_key}"})
    assert resp.status_code == 200, resp.text


def test_generated_manager_key_returns_401():
    agent_id, full_key, _prefix = _create_agent_and_key()
    _mark_generated_manager(agent_id)

    resp = client.post(
        "/v1/chat/tasks",
        headers={"Authorization": f"Bearer {full_key}"},
        json={
            "agent_id": agent_id,
            "message": {"role": "user", "content": "hello"},
        },
    )

    _assert_invalid_api_key(resp)


@pytest.mark.asyncio
async def test_runtime_key_auth_runs_sql_and_bcrypt_off_event_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Runtime-key auth owns its Session and bcrypt work in one DB worker."""

    from xagent.web.api.v1 import deps

    agent_id, full_key, prefix = _create_agent_and_key()
    event_loop_thread = threading.get_ident()
    query_threads: list[int] = []
    bcrypt_threads: list[int] = []
    original_load = deps._load_runtime_key_record
    original_verify = deps.verify_api_key

    def recording_load(key_prefix: str):  # type: ignore[no-untyped-def]
        query_threads.append(threading.get_ident())
        return original_load(key_prefix)

    def recording_verify(raw: str, key_hash: str) -> bool:
        bcrypt_threads.append(threading.get_ident())
        return original_verify(raw, key_hash)

    monkeypatch.setattr(deps, "_load_runtime_key_record", recording_load)
    monkeypatch.setattr(deps, "verify_api_key", recording_verify)

    principal = await deps.get_principal_from_api_key(
        HTTPAuthorizationCredentials(scheme="Bearer", credentials=full_key)
    )

    assert principal.agent is not None
    assert principal.agent.id == agent_id
    assert principal.key.key_prefix == prefix
    assert query_threads and query_threads[0] != event_loop_thread
    assert bcrypt_threads == query_threads


@pytest.mark.asyncio
async def test_runtime_key_auth_pool_wait_keeps_event_loop_responsive(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A saturated auth pool cannot stop unrelated asyncio work."""

    from xagent.web.api.v1 import deps

    engine = create_engine(
        f"sqlite:///{tmp_path / 'auth-pool.db'}",
        connect_args={"check_same_thread": False},
        poolclass=QueuePool,
        pool_size=1,
        max_overflow=0,
        pool_timeout=30,
    )
    event_loop_thread = threading.get_ident()
    worker_started = threading.Event()
    held_connection = engine.connect()

    def waiting_resolve(_credentials):  # type: ignore[no-untyped-def]
        worker_started.set()
        assert threading.get_ident() != event_loop_thread
        with engine.connect() as connection:
            connection.execute(text("SELECT 1")).scalar_one()
        return deps.ApiKeyPrincipal(
            key=deps.RuntimeApiKeySnapshot(key_prefix="ABC123"),
            agent=deps.AgentPrincipalSnapshot(
                id=1,
                user_id=2,
                execution_mode="balanced",
                status="published",
                origin="user",
            ),
        )

    monkeypatch.setattr(
        deps,
        "_resolve_principal_from_credentials",
        waiting_resolve,
    )
    auth = asyncio.create_task(deps.get_principal_from_api_key(None))
    try:
        assert await asyncio.to_thread(worker_started.wait, 30)
        await asyncio.sleep(0)
        assert not auth.done()
    finally:
        held_connection.close()
        try:
            principal = await asyncio.wait_for(auth, timeout=30)
        finally:
            engine.dispose()

    assert principal.agent is not None


@pytest.mark.asyncio
async def test_runtime_key_auth_cancellation_drains_worker_before_propagating(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancellation cannot abandon auth work with a live worker Session."""

    from xagent.web.api.v1 import deps

    _agent_id, full_key, _prefix = _create_agent_and_key()
    worker_started = threading.Event()
    allow_worker = threading.Event()
    worker_finished = threading.Event()
    original_load = deps._load_runtime_key_record

    def blocking_load(key_prefix: str):  # type: ignore[no-untyped-def]
        worker_started.set()
        assert allow_worker.wait(timeout=2)
        try:
            return original_load(key_prefix)
        finally:
            worker_finished.set()

    monkeypatch.setattr(deps, "_load_runtime_key_record", blocking_load)
    auth = asyncio.create_task(
        deps.get_principal_from_api_key(
            HTTPAuthorizationCredentials(scheme="Bearer", credentials=full_key)
        )
    )
    await asyncio.to_thread(worker_started.wait, 2)
    auth.cancel()
    allow_worker.set()

    with pytest.raises(asyncio.CancelledError):
        await auth
    assert worker_finished.is_set()


@pytest.mark.asyncio
async def test_personal_key_auth_runs_sql_and_bcrypt_off_event_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Personal-key auth follows the same worker-owned snapshot boundary."""

    from xagent.web.api.v1 import deps

    full_key, prefix = _create_personal_key()
    event_loop_thread = threading.get_ident()
    query_threads: list[int] = []
    bcrypt_threads: list[int] = []
    original_load = deps._load_personal_key_record
    original_verify = deps.verify_api_key

    def recording_load(key_prefix: str):  # type: ignore[no-untyped-def]
        query_threads.append(threading.get_ident())
        return original_load(key_prefix)

    def recording_verify(raw: str, key_hash: str) -> bool:
        bcrypt_threads.append(threading.get_ident())
        return original_verify(raw, key_hash)

    monkeypatch.setattr(deps, "_load_personal_key_record", recording_load)
    monkeypatch.setattr(deps, "verify_api_key", recording_verify)

    user, key = await deps.get_user_from_personal_key(
        HTTPAuthorizationCredentials(scheme="Bearer", credentials=full_key)
    )

    assert user.username == "admin"
    assert key.key_prefix == prefix
    assert query_threads and query_threads[0] != event_loop_thread
    assert bcrypt_threads == query_threads


def test_paused_key_returns_401():
    """A paused (not revoked) key is rejected identically to a revoked one.

    Pausing goes through the multi-key admin service directly here (no
    HTTP endpoint under test) -- the point is that ``get_agent_from_api_key``
    treats ``paused_at`` the same as ``revoked_at``, with the same opaque
    401 envelope.
    """
    agent_id, full_key, prefix = _create_agent_and_key()
    db = _direct_db_session()
    try:
        from xagent.web.models.agent_api_key import AgentApiKey

        db.query(AgentApiKey).filter(AgentApiKey.key_prefix == prefix).update(
            {"paused_at": datetime.now(timezone.utc)}
        )
        db.commit()
    finally:
        db.close()

    resp = client.post(
        "/v1/chat/tasks",
        headers={"Authorization": f"Bearer {full_key}"},
        json={
            "agent_id": agent_id,
            "message": {"role": "user", "content": "hello"},
        },
    )

    _assert_invalid_api_key(resp)


def _usage_snapshot(prefix: str):
    from xagent.web.models.agent_api_key import AgentApiKey

    db = _direct_db_session()
    try:
        row = db.query(AgentApiKey).filter(AgentApiKey.key_prefix == prefix).one()
        return row.last_used_at, row.usage_month, row.usage_month_calls
    finally:
        db.close()


@pytest.mark.asyncio
async def test_record_key_usage_skips_revoked_key():
    """Direct-call test for the defense-in-depth guard in ``record_key_usage``.

    Both real call sites (create task / append message) are gated by
    ``get_agent_from_api_key``, which already excludes revoked/paused keys
    before either ever runs -- so no HTTP-level test can reach the guard
    in ``record_key_usage``'s own WHERE clause. Call it directly instead
    to prove the guard itself works, independent of caller discipline.
    """
    from xagent.web.api.v1.deps import record_key_usage
    from xagent.web.models.agent_api_key import AgentApiKey

    _agent_id, _full_key, prefix = _create_agent_and_key()
    db = _direct_db_session()
    try:
        db.query(AgentApiKey).filter(AgentApiKey.key_prefix == prefix).update(
            {"revoked_at": datetime.now(timezone.utc)}
        )
        db.commit()
    finally:
        db.close()

    before = _usage_snapshot(prefix)
    await record_key_usage(prefix)
    assert _usage_snapshot(prefix) == before


@pytest.mark.asyncio
async def test_record_key_usage_skips_paused_key():
    from xagent.web.api.v1.deps import record_key_usage
    from xagent.web.models.agent_api_key import AgentApiKey

    _agent_id, _full_key, prefix = _create_agent_and_key()
    db = _direct_db_session()
    try:
        db.query(AgentApiKey).filter(AgentApiKey.key_prefix == prefix).update(
            {"paused_at": datetime.now(timezone.utc)}
        )
        db.commit()
    finally:
        db.close()

    before = _usage_snapshot(prefix)
    await record_key_usage(prefix)
    assert _usage_snapshot(prefix) == before


@pytest.mark.asyncio
async def test_record_key_usage_updates_active_key():
    """Sanity counterpart: the guard doesn't block a legitimately active key."""
    from xagent.web.api.v1.deps import record_key_usage

    _agent_id, _full_key, prefix = _create_agent_and_key()

    await record_key_usage(prefix)

    last_used_at, usage_month, usage_month_calls = _usage_snapshot(prefix)
    assert last_used_at is not None
    assert usage_month == datetime.now(timezone.utc).strftime("%Y-%m")
    assert usage_month_calls == 1


@pytest.mark.asyncio
async def test_record_key_usage_pool_wait_keeps_event_loop_responsive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Best-effort metering may wait for a slot, but never on the event loop."""

    from xagent.web.api.v1.deps import record_key_usage

    _agent_id, _full_key, prefix = _create_agent_and_key()
    engine = _install_one_slot_queue_pool(monkeypatch, pool_timeout=0.15)
    held_connection = engine.connect()

    async def invoke_usage() -> None:
        await record_key_usage(prefix)

    from tests.web.pool_contention_shared import GUARD_TIMEOUT, gated_pool_checkout

    try:
        with gated_pool_checkout(engine) as gate:
            usage_task = asyncio.create_task(invoke_usage())
            try:
                await gate.wait_until_contending()
                assert not usage_task.done()
            finally:
                held_connection.close()
                gate.let_through()
                await asyncio.wait_for(usage_task, timeout=GUARD_TIMEOUT)
    finally:
        engine.dispose()


# ===== timing oracle defense =====


def test_unknown_prefix_performs_the_same_bcrypt_work_as_wrong_secret():
    """Both HTTP rejection paths perform one real bcrypt check at equal cost."""
    full_key, _prefix = _create_personal_key()
    parts = full_key.split("_")
    parts[3] = "z" * 32
    wrong_secret_key = "_".join(parts)

    with patch.object(bcrypt, "checkpw", wraps=bcrypt.checkpw) as checkpw:
        resp1 = client.get(
            "/v1/me", headers={"Authorization": f"Bearer {wrong_secret_key}"}
        )
        assert resp1.status_code == 401
        checkpw.assert_called_once()
        real_password, real_hash = checkpw.call_args.args
        assert real_password == wrong_secret_key.encode()
        assert int(real_hash.split(b"$")[2]) == BCRYPT_COST
        checkpw.reset_mock()

        fake_key = "xag_personal_ZZZZZZ_" + "x" * 32
        resp2 = client.get("/v1/me", headers={"Authorization": f"Bearer {fake_key}"})
        assert resp2.status_code == 401
        assert resp2.json() == resp1.json()
        checkpw.assert_called_once()
        _dummy_password, dummy_hash = checkpw.call_args.args
        assert int(dummy_hash.split(b"$")[2]) == BCRYPT_COST


# ===== /v1/* internal_error envelope (catch-all) =====


def test_internal_exception_returns_v1_envelope_not_fastapi_detail():
    """Non-V1ApiError exceptions on /v1/* must still match SDK contract.

    If an upstream layer (db.query, bcrypt, dependency) raises an
    unexpected exception, the response MUST be the stable
    ``{"error": {"code": "internal_error", "message": ...}}`` shape --
    not FastAPI's default ``{"detail": "Internal Server Error"}``,
    which would break SDK clients that key off ``body.error.code``.

    We force the failure by patching ``parse_api_key`` (called inside
    the auth dep) to raise a RuntimeError. That gets the request past
    the FastAPI routing layer but blows up inside our handler chain
    BEFORE V1ApiError is raised, exercising the generic Exception
    branch of the global handler.
    """
    secret_internal_msg = "secret-internal-detail-do-not-leak"
    with patch(
        "xagent.web.api.v1.deps.parse_api_key",
        side_effect=RuntimeError(secret_internal_msg),
    ):
        resp = client.get(
            "/v1/me",
            headers={"Authorization": "Bearer xag_personal_ABCDEF_" + "x" * 32},
        )

    # Must be 500 in the V1 envelope, not 500 with FastAPI's detail key.
    assert resp.status_code == 500, resp.text
    body = resp.json()
    assert body == {
        "error": {
            "code": "internal_error",
            "message": "Internal server error.",
        }
    }
    # Sanity: no internal exception message leaks into the response
    assert secret_internal_msg not in resp.text
    # Sanity: NOT the default FastAPI {"detail": ...} shape
    assert "detail" not in body
