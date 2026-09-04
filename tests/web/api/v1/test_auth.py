"""Integration tests for the /v1/* personal management auth dependency.

Drives /v1/me to verify each personal-key failure path returns the
stable ``{"error": {"code": "invalid_api_key", ...}}`` envelope.

Test plumbing (client, _test_db fixture, auth helpers) is shared via
``tests/web/api/conftest.py``.
"""

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor, wait
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import bcrypt
import pytest
from fastapi.security import HTTPAuthorizationCredentials
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool

from xagent.core.utils.api_key import (
    SHA256_HASH_PREFIX,
    hash_api_key,
    verify_api_key,
)
from xagent.web.models.agent import Agent, AgentOrigin
from xagent.web.models.agent_api_key import AgentApiKey
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


def _replace_key_hash(key_model, prefix: str, key_hash: str) -> None:  # type: ignore[no-untyped-def]
    db = _direct_db_session()
    try:
        db.query(key_model).filter(key_model.key_prefix == prefix).update(
            {"key_hash": key_hash}
        )
        db.commit()
    finally:
        db.close()


def _load_key_hash(key_model, prefix: str) -> str:  # type: ignore[no-untyped-def]
    db = _direct_db_session()
    try:
        return str(
            db.query(key_model.key_hash).filter(key_model.key_prefix == prefix).scalar()
        )
    finally:
        db.close()


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


def test_new_personal_key_uses_sha256_verifier():
    full_key, prefix = _create_personal_key()
    stored_hash = _load_key_hash(UserApiKey, prefix)

    assert stored_hash.startswith(SHA256_HASH_PREFIX)
    assert verify_api_key(full_key, stored_hash) is True


def test_legacy_personal_key_migrates_after_successful_authentication():
    full_key, prefix = _create_personal_key()
    legacy_hash = bcrypt.hashpw(full_key.encode(), bcrypt.gensalt(rounds=4)).decode()
    _replace_key_hash(UserApiKey, prefix, legacy_hash)

    resp = client.get("/v1/me", headers={"Authorization": f"Bearer {full_key}"})

    assert resp.status_code == 200, resp.text
    migrated_hash = _load_key_hash(UserApiKey, prefix)
    assert migrated_hash == hash_api_key(full_key)


def test_legacy_personal_key_wrong_secret_does_not_migrate():
    full_key, prefix = _create_personal_key()
    legacy_hash = bcrypt.hashpw(full_key.encode(), bcrypt.gensalt(rounds=4)).decode()
    _replace_key_hash(UserApiKey, prefix, legacy_hash)
    parts = full_key.split("_")
    parts[-1] = "z" * 32

    resp = client.get(
        "/v1/me",
        headers={"Authorization": f"Bearer {'_'.join(parts)}"},
    )

    _assert_invalid_api_key(resp)
    assert _load_key_hash(UserApiKey, prefix) == legacy_hash


def test_legacy_personal_key_migration_session_failure_does_not_reject_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Even Session construction for best-effort migration cannot escape."""
    from xagent.web.api.v1 import deps

    full_key, prefix = _create_personal_key()
    legacy_hash = bcrypt.hashpw(full_key.encode(), bcrypt.gensalt(rounds=4)).decode()
    _replace_key_hash(UserApiKey, prefix, legacy_hash)
    session_local = deps.get_session_local()

    class MigrationFailingSessionFactory:
        def __call__(self):  # type: ignore[no-untyped-def]
            return session_local()

        def begin(self):  # type: ignore[no-untyped-def]
            raise RuntimeError("migration Session factory unavailable")

    monkeypatch.setattr(
        deps,
        "get_session_local",
        lambda: MigrationFailingSessionFactory(),
    )

    resp = client.get("/v1/me", headers={"Authorization": f"Bearer {full_key}"})

    assert resp.status_code == 200, resp.text
    assert _load_key_hash(UserApiKey, prefix) == legacy_hash


def test_legacy_runtime_key_migrates_after_successful_authentication():
    from xagent.web.api.v1.deps import (
        _resolve_principal_from_credentials,
        _upgrade_api_key_hash_isolated,
    )

    agent_id, full_key, prefix = _create_agent_and_key()
    legacy_hash = bcrypt.hashpw(full_key.encode(), bcrypt.gensalt(rounds=4)).decode()
    _replace_key_hash(AgentApiKey, prefix, legacy_hash)

    principal = _resolve_principal_from_credentials(
        HTTPAuthorizationCredentials(scheme="Bearer", credentials=full_key)
    )

    assert principal.agent is not None
    assert principal.agent.id == agent_id
    assert _load_key_hash(AgentApiKey, prefix) == hash_api_key(full_key)

    second_principal = _resolve_principal_from_credentials(
        HTTPAuthorizationCredentials(scheme="Bearer", credentials=full_key)
    )
    assert second_principal.agent is not None
    assert second_principal.agent.id == agent_id

    # A concurrent worker holding the old snapshot cannot overwrite the
    # winner's digest with a stale compare-and-swap.
    different_key = full_key[:-1] + ("A" if full_key[-1] != "A" else "B")
    _upgrade_api_key_hash_isolated(
        AgentApiKey,
        key_prefix=prefix,
        raw_key=different_key,
        observed_hash=legacy_hash,
    )
    assert _load_key_hash(AgentApiKey, prefix) == hash_api_key(full_key)


def test_concurrent_wrong_legacy_secrets_verify_without_serializing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Legacy verification stays concurrent and precedes migration coordination."""
    from xagent.web.api.v1 import deps
    from xagent.web.api.v1.errors import V1ApiError

    _agent_id, full_key, prefix = _create_agent_and_key()
    legacy_hash = bcrypt.hashpw(full_key.encode(), bcrypt.gensalt(rounds=4)).decode()
    _replace_key_hash(AgentApiKey, prefix, legacy_hash)
    wrong_key = full_key[:-1] + ("A" if full_key[-1] != "A" else "B")
    legacy_checks = 0
    active_checks = 0
    peak_checks = 0
    count_lock = threading.Lock()
    verification_barrier = threading.Barrier(8)
    original_verify = deps.verify_api_key_with_timing

    def recording_verify(raw: str, stored_hash: str):  # type: ignore[no-untyped-def]
        nonlocal active_checks, legacy_checks, peak_checks
        if stored_hash.startswith("$2"):
            with count_lock:
                legacy_checks += 1
                active_checks += 1
                peak_checks = max(peak_checks, active_checks)
            try:
                verification_barrier.wait(timeout=2)
                return original_verify(raw, stored_hash)
            finally:
                with count_lock:
                    active_checks -= 1
        return original_verify(raw, stored_hash)

    monkeypatch.setattr(deps, "verify_api_key_with_timing", recording_verify)
    credentials = HTTPAuthorizationCredentials(
        scheme="Bearer",
        credentials=wrong_key,
    )

    def authenticate(_index: int) -> bool:
        try:
            deps._resolve_principal_from_credentials(credentials)
        except V1ApiError:
            return False
        return True

    with ThreadPoolExecutor(max_workers=8) as pool:
        authenticated = list(pool.map(authenticate, range(8)))

    assert authenticated == [False] * 8
    assert legacy_checks == 8
    assert peak_checks == 8
    assert _load_key_hash(AgentApiKey, prefix) == legacy_hash


def test_concurrent_valid_legacy_auth_does_not_wait_for_migration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only one migration runs while other verified requests return promptly."""
    from xagent.web.api.v1 import deps

    agent_id, full_key, prefix = _create_agent_and_key()
    legacy_hash = bcrypt.hashpw(full_key.encode(), bcrypt.gensalt(rounds=4)).decode()
    _replace_key_hash(AgentApiKey, prefix, legacy_hash)
    verification_barrier = threading.Barrier(8)
    migration_started = threading.Event()
    release_migration = threading.Event()
    original_verify = deps.verify_api_key_with_timing
    original_upgrade = deps._upgrade_api_key_hash_isolated
    migration_calls = 0
    count_lock = threading.Lock()

    def synchronized_verify(raw: str, stored_hash: str):  # type: ignore[no-untyped-def]
        verification_barrier.wait(timeout=2)
        return original_verify(raw, stored_hash)

    def blocking_upgrade(*args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal migration_calls
        with count_lock:
            migration_calls += 1
        migration_started.set()
        assert release_migration.wait(timeout=2)
        return original_upgrade(*args, **kwargs)

    monkeypatch.setattr(deps, "verify_api_key_with_timing", synchronized_verify)
    monkeypatch.setattr(deps, "_upgrade_api_key_hash_isolated", blocking_upgrade)
    credentials = HTTPAuthorizationCredentials(
        scheme="Bearer",
        credentials=full_key,
    )
    pool = ThreadPoolExecutor(max_workers=8)
    futures = [
        pool.submit(deps._resolve_principal_from_credentials, credentials)
        for _ in range(8)
    ]
    try:
        assert migration_started.wait(timeout=2)
        completed, _pending = wait(futures, timeout=1)
        completed_before_migration = len(completed)
    finally:
        release_migration.set()
        pool.shutdown(wait=True)

    principals = [future.result() for future in futures]
    assert completed_before_migration == 7
    assert migration_calls == 1
    assert all(
        principal.agent is not None and principal.agent.id == agent_id
        for principal in principals
    )
    assert _load_key_hash(AgentApiKey, prefix) == hash_api_key(full_key)


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
    """Prefix is real but the secret doesn't match the stored verifier."""
    full_key, _prefix = _create_personal_key()
    # Replace just the secret half with a different (but well-formed) value
    parts = full_key.split("_")
    parts[3] = "y" * 32
    wrong_key = "_".join(parts)
    resp = client.get("/v1/me", headers={"Authorization": f"Bearer {wrong_key}"})
    _assert_invalid_api_key(resp)


def test_malformed_bcrypt_verifier_uses_dummy_padding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A malformed ``$2`` value has not paid bcrypt's timing floor."""
    from xagent.web.api.v1 import deps

    full_key, prefix = _create_personal_key()
    _replace_key_hash(
        UserApiKey,
        prefix,
        "$2b$12$this-is-not-a-valid-bcrypt-hash",
    )
    dummy_calls = 0

    def record_dummy() -> None:
        nonlocal dummy_calls
        dummy_calls += 1

    monkeypatch.setattr(deps, "verify_dummy", record_dummy)

    resp = client.get("/v1/me", headers={"Authorization": f"Bearer {full_key}"})

    _assert_invalid_api_key(resp)
    assert dummy_calls == 1


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


def test_generated_manager_key_returns_401_with_dummy_padding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xagent.web.api.v1 import deps

    agent_id, full_key, _prefix = _create_agent_and_key()
    _mark_generated_manager(agent_id)
    dummy_calls = 0

    def record_dummy() -> None:
        nonlocal dummy_calls
        dummy_calls += 1

    monkeypatch.setattr(deps, "verify_dummy", record_dummy)

    resp = client.post(
        "/v1/chat/tasks",
        headers={"Authorization": f"Bearer {full_key}"},
        json={
            "agent_id": agent_id,
            "message": {"role": "user", "content": "hello"},
        },
    )

    _assert_invalid_api_key(resp)
    assert dummy_calls == 1


@pytest.mark.asyncio
async def test_runtime_key_auth_runs_sql_and_verification_off_event_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Runtime-key auth owns its Session and verification in one DB worker."""

    from xagent.web.api.v1 import deps

    agent_id, full_key, prefix = _create_agent_and_key()
    event_loop_thread = threading.get_ident()
    query_threads: list[int] = []
    verification_threads: list[int] = []
    original_load = deps._load_runtime_key_record
    original_verify = deps.verify_api_key_with_timing

    def recording_load(key_prefix: str):  # type: ignore[no-untyped-def]
        query_threads.append(threading.get_ident())
        return original_load(key_prefix)

    def recording_verify(raw: str, key_hash: str):  # type: ignore[no-untyped-def]
        verification_threads.append(threading.get_ident())
        return original_verify(raw, key_hash)

    monkeypatch.setattr(deps, "_load_runtime_key_record", recording_load)
    monkeypatch.setattr(deps, "verify_api_key_with_timing", recording_verify)

    principal = await deps.get_principal_from_api_key(
        HTTPAuthorizationCredentials(scheme="Bearer", credentials=full_key)
    )

    assert principal.agent is not None
    assert principal.agent.id == agent_id
    assert principal.key.key_prefix == prefix
    assert query_threads and query_threads[0] != event_loop_thread
    assert verification_threads == query_threads


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
        pool_timeout=1,
    )
    worker_started = threading.Event()
    held_connection = engine.connect()

    def waiting_resolve(_credentials):  # type: ignore[no-untyped-def]
        worker_started.set()
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
    await asyncio.to_thread(worker_started.wait, 2)

    ticks = 0

    async def ticker() -> None:
        nonlocal ticks
        for _ in range(5):
            await asyncio.sleep(0.01)
            ticks += 1

    try:
        await ticker()
        assert ticks == 5
        assert auth.done() is False
    finally:
        held_connection.close()

    principal = await auth
    assert principal.agent is not None
    engine.dispose()


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
async def test_personal_key_auth_runs_sql_and_verification_off_event_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Personal-key auth follows the same worker-owned snapshot boundary."""

    from xagent.web.api.v1 import deps

    full_key, prefix = _create_personal_key()
    event_loop_thread = threading.get_ident()
    query_threads: list[int] = []
    verification_threads: list[int] = []
    original_load = deps._load_personal_key_record
    original_verify = deps.verify_api_key_with_timing

    def recording_load(key_prefix: str):  # type: ignore[no-untyped-def]
        query_threads.append(threading.get_ident())
        return original_load(key_prefix)

    def recording_verify(raw: str, key_hash: str):  # type: ignore[no-untyped-def]
        verification_threads.append(threading.get_ident())
        return original_verify(raw, key_hash)

    monkeypatch.setattr(deps, "_load_personal_key_record", recording_load)
    monkeypatch.setattr(deps, "verify_api_key_with_timing", recording_verify)

    user, key = await deps.get_user_from_personal_key(
        HTTPAuthorizationCredentials(scheme="Bearer", credentials=full_key)
    )

    assert user.username == "admin"
    assert key.key_prefix == prefix
    assert query_threads and query_threads[0] != event_loop_thread
    assert verification_threads == query_threads


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

    usage_task = asyncio.create_task(invoke_usage())
    started_at = asyncio.get_running_loop().time()
    ticker_task = asyncio.create_task(asyncio.sleep(0.02))
    try:
        await ticker_task
        assert asyncio.get_running_loop().time() - started_at < 0.08
    finally:
        held_connection.close()
        await usage_task
        engine.dispose()


# ===== timing oracle defense =====


def test_unknown_prefix_and_wrong_sha_secret_each_use_dummy_padding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both fast failure branches deterministically route through the dummy."""
    from xagent.web.api.v1 import deps

    full_key, _prefix = _create_personal_key()
    parts = full_key.split("_")
    parts[3] = "z" * 32
    wrong_secret_key = "_".join(parts)
    dummy_calls = 0

    def record_dummy() -> None:
        nonlocal dummy_calls
        dummy_calls += 1

    monkeypatch.setattr(deps, "verify_dummy", record_dummy)
    resp1 = client.get(
        "/v1/me", headers={"Authorization": f"Bearer {wrong_secret_key}"}
    )
    assert resp1.status_code == 401
    assert dummy_calls == 1

    fake_key = "xag_personal_ZZZZZZ_" + "x" * 32
    resp2 = client.get("/v1/me", headers={"Authorization": f"Bearer {fake_key}"})
    assert resp2.status_code == 401
    assert dummy_calls == 2


# ===== /v1/* internal_error envelope (catch-all) =====


def test_internal_exception_returns_v1_envelope_not_fastapi_detail():
    """Non-V1ApiError exceptions on /v1/* must still match SDK contract.

    If an upstream layer (db.query, verifier, dependency) raises an
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
