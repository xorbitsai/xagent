"""Concurrency-boundary tests for the V1 agent-management runtime."""

from __future__ import annotations

import asyncio
import logging
import threading
from datetime import datetime, timezone

import pytest
from sqlalchemy import event, text
from sqlalchemy.orm import Session

from tests.web.pool_contention_shared import (
    CONTENTION_POOL_TIMEOUT,
    GUARD_TIMEOUT,
    assert_pool_checkout_off_loop,
    gated_pool_checkout,
)
from xagent.web.models.agent import Agent
from xagent.web.models.agent_api_key import AgentApiKey
from xagent.web.models.user import User
from xagent.web.services.agent_management import (
    AgentCreateSpec,
    AgentManagementRuntime,
    _RuntimeKeyDeliveryOutcome,
)
from xagent.web.services.api_keys import (
    AgentApiKeyService,
    RuntimeKeyReceipt,
)

from ..conftest import (
    _admin_headers,
    _direct_db_session,
    _install_one_slot_queue_pool,
)

pytestmark = pytest.mark.usefixtures("_test_db")


# Cross-thread handshake deadline.
#
# Every ``Event.wait`` and ``asyncio.wait_for`` below is a rendezvous, not a
# latency assertion: the other side releases it the moment it reaches the
# point under test. The deadline exists only so that a genuine deadlock fails
# one test instead of hanging the suite, so it is sized for the worst CI
# machine rather than for the expected one.
#
# It has to be generous because the code these handshakes straddle is
# deliberately expensive. Runtime-key delivery hashes with bcrypt at
# ``BCRYPT_COST`` 12 -- ~100ms per draw on idle commodity hardware, up to
# ``PREFIX_COLLISION_RETRIES`` draws -- and then commits. CI runs this suite
# as ``pytest -n 4`` on a 4-vCPU runner that is simultaneously driving a
# Docker daemon, so the 2s budget these waits used to carry (~9x headroom on
# an idle workstation) was routinely missed there. Oversubscribing an
# 18-core machine ~8x reproduces the exact CI failures: ``assert False`` on a
# handshake wait, ``TimeoutError`` on a settlement wait.
_HANDSHAKE_TIMEOUT = 30.0

# Deliberately short, and deliberately not the constant above: this probe
# waits on something that must *not* happen (a second fence acquisition while
# the first is still held), so the wait is pure cost -- a longer one would
# only slow the suite down without strengthening the assertion.
_NEGATIVE_PROBE_TIMEOUT = 0.1


def _admin_identity() -> tuple[int, bool]:
    _admin_headers()
    db = _direct_db_session()
    try:
        user = db.query(User).filter(User.username == "admin").one()
        return int(user.id), bool(user.is_admin)
    finally:
        db.close()


def _create_spec(
    name: str,
    *,
    generate_runtime_key: bool = True,
    knowledge_bases: tuple[str, ...] = (),
    tool_categories: tuple[str, ...] = (),
) -> AgentCreateSpec:
    return AgentCreateSpec(
        name=name,
        description=None,
        instructions="Be useful.",
        execution_mode="balanced",
        models=None,
        knowledge_bases=knowledge_bases,
        skills=(),
        tool_categories=tool_categories,
        suggested_prompts=(),
        generate_runtime_key=generate_runtime_key,
    )


async def _cancel_after_runtime_key_commit(
    operation: asyncio.Task[object],
    committed: threading.Event,
    release: threading.Event,
) -> None:
    assert await asyncio.to_thread(committed.wait, _HANDSHAKE_TIMEOUT)
    operation.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(operation, timeout=_HANDSHAKE_TIMEOUT)


@pytest.mark.asyncio
async def test_create_and_rotate_hash_without_holding_pool_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """bcrypt is worker-owned and runs before either write transaction opens."""

    from xagent.core.utils import api_key

    user_id, is_admin = _admin_identity()
    engine = _install_one_slot_queue_pool(monkeypatch, pool_timeout=0.5)
    event_loop_thread = threading.get_ident()
    hash_observations: list[tuple[int, int]] = []
    sql_threads: list[int] = []
    original_hashpw = api_key.bcrypt.hashpw

    @event.listens_for(engine, "before_cursor_execute")
    def record_sql_thread(*_args) -> None:  # type: ignore[no-untyped-def]
        sql_threads.append(threading.get_ident())

    def recording_hashpw(*args, **kwargs):  # type: ignore[no-untyped-def]
        hash_observations.append((threading.get_ident(), engine.pool.checkedout()))
        return original_hashpw(*args, **kwargs)

    monkeypatch.setattr(api_key.bcrypt, "hashpw", recording_hashpw)
    runtime = AgentManagementRuntime()

    created = await runtime.create_agent(
        user_id=user_id,
        is_admin=is_admin,
        spec=_create_spec("bcrypt boundary"),
    )
    rotated = await runtime.rotate_agent_runtime_key(
        user_id=user_id,
        agent_id=created.agent.id,
    )

    assert rotated is not None
    assert len(hash_observations) == 2
    assert all(thread_id != event_loop_thread for thread_id, _ in hash_observations)
    assert all(checked_out == 0 for _, checked_out in hash_observations)
    assert sql_threads
    assert all(thread_id != event_loop_thread for thread_id in sql_threads)
    engine.dispose()


@pytest.mark.asyncio
async def test_list_pool_wait_does_not_block_event_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user_id, _is_admin = _admin_identity()
    engine = _install_one_slot_queue_pool(
        monkeypatch, pool_timeout=CONTENTION_POOL_TIMEOUT
    )
    held_connection = engine.connect()
    runtime = AgentManagementRuntime()
    try:
        with gated_pool_checkout(engine) as gate:
            listing = asyncio.create_task(runtime.list_agents(user_id=user_id))
            try:
                await gate.wait_until_contending()
                assert not listing.done()
            finally:
                held_connection.close()
                gate.let_through()
                result = await asyncio.wait_for(listing, timeout=GUARD_TIMEOUT)
        assert result == ()
    finally:
        held_connection.close()
        engine.dispose()


@pytest.mark.asyncio
async def test_list_cache_io_never_holds_database_pool_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Team-scope SQL and list SQL both release the pool before cache I/O."""

    from xagent.web.services import agent_store

    user_id, _is_admin = _admin_identity()
    engine = _install_one_slot_queue_pool(monkeypatch, pool_timeout=0.5)
    cache_get_entered = threading.Event()
    allow_cache_get = threading.Event()
    cache_set_entered = threading.Event()
    allow_cache_set = threading.Event()
    cache_observations: list[tuple[str, int]] = []

    def db_backed_team_scope(db, resolved_user_id: int):  # type: ignore[no-untyped-def]
        assert (
            db.query(User.id).filter(User.id == resolved_user_id).scalar()
            == resolved_user_id
        )
        return None

    def gated_cache_get(_key: str):  # type: ignore[no-untyped-def]
        cache_observations.append(("get", engine.pool.checkedout()))
        cache_get_entered.set()
        assert allow_cache_get.wait(timeout=_HANDSHAKE_TIMEOUT)
        return None

    def gated_cache_set(_key: str, _value):  # type: ignore[no-untyped-def]
        cache_observations.append(("set", engine.pool.checkedout()))
        cache_set_entered.set()
        assert allow_cache_set.wait(timeout=_HANDSHAKE_TIMEOUT)

    def probe_pool() -> None:
        with engine.connect() as connection:
            assert connection.execute(text("SELECT 1")).scalar_one() == 1

    monkeypatch.setattr(agent_store, "get_agent_team_scope", db_backed_team_scope)
    monkeypatch.setattr(agent_store, "cache_get", gated_cache_get)
    monkeypatch.setattr(agent_store, "cache_set", gated_cache_set)
    listing = asyncio.create_task(AgentManagementRuntime().list_agents(user_id=user_id))

    try:
        assert await asyncio.to_thread(cache_get_entered.wait, _HANDSHAKE_TIMEOUT)
        assert cache_observations == [("get", 0)]
        await asyncio.to_thread(probe_pool)
        allow_cache_get.set()

        assert await asyncio.to_thread(cache_set_entered.wait, _HANDSHAKE_TIMEOUT)
        assert cache_observations == [("get", 0), ("set", 0)]
        await asyncio.to_thread(probe_pool)
        allow_cache_set.set()

        assert await listing == ()
    finally:
        allow_cache_get.set()
        allow_cache_set.set()
        if not listing.done():
            listing.cancel()
        await asyncio.gather(listing, return_exceptions=True)
        engine.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["create", "rotate"])
async def test_write_pool_wait_does_not_block_event_loop(
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    from xagent.web.services import agent_management

    user_id, is_admin = _admin_identity()
    runtime = AgentManagementRuntime()
    rotate_target = await runtime.create_agent(
        user_id=user_id,
        is_admin=is_admin,
        spec=_create_spec("pool-wait rotate target", generate_runtime_key=False),
    )
    engine = _install_one_slot_queue_pool(monkeypatch, pool_timeout=0.5)
    held_connection = engine.connect()
    candidate_ready = threading.Event()
    original_generate = agent_management.generate_api_key

    def recording_generate(*args, **kwargs):  # type: ignore[no-untyped-def]
        candidate = original_generate(*args, **kwargs)
        candidate_ready.set()
        return candidate

    monkeypatch.setattr(agent_management, "generate_api_key", recording_generate)
    if operation == "create":
        write = asyncio.create_task(
            runtime.create_agent(
                user_id=user_id,
                is_admin=is_admin,
                spec=_create_spec("pool-wait create"),
            )
        )
    else:
        write = asyncio.create_task(
            runtime.rotate_agent_runtime_key(
                user_id=user_id,
                agent_id=rotate_target.agent.id,
            )
        )

    try:
        assert await asyncio.to_thread(candidate_ready.wait, _HANDSHAKE_TIMEOUT)
        for _ in range(4):
            await asyncio.sleep(0.01)
        assert write.done() is False
    finally:
        held_connection.close()

    assert await write is not None
    engine.dispose()


@pytest.mark.asyncio
async def test_cancelled_pool_wait_drains_worker_before_propagating(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user_id, _is_admin = _admin_identity()
    engine = _install_one_slot_queue_pool(monkeypatch, pool_timeout=0.5)
    held_connection = engine.connect()
    runtime = AgentManagementRuntime()
    worker_started = threading.Event()
    original_list = runtime._list_agents_sync

    def recording_list(*, user_id: int):  # type: ignore[no-untyped-def]
        worker_started.set()
        return original_list(user_id=user_id)

    monkeypatch.setattr(runtime, "_list_agents_sync", recording_list)
    listing = asyncio.create_task(runtime.list_agents(user_id=user_id))

    assert await asyncio.to_thread(worker_started.wait, _HANDSHAKE_TIMEOUT)
    listing.cancel()
    held_connection.close()

    with pytest.raises(asyncio.CancelledError):
        await listing
    assert engine.pool.checkedout() == 0
    engine.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("from_template", [False, True])
async def test_cancel_after_committed_create_revokes_the_undelivered_exact_key(
    monkeypatch: pytest.MonkeyPatch,
    from_template: bool,
) -> None:
    """The generated V1 key is revoked when delivery is cancelled after commit."""

    from xagent.web.services import agent_management

    user_id, is_admin = _admin_identity()
    committed = threading.Event()
    release = threading.Event()
    captured: dict[str, int | str] = {}
    original_create = (
        agent_management.AgentManagementService.create_agent_with_optional_key
    )

    def pause_after_commit(service, *args, **kwargs):  # type: ignore[no-untyped-def]
        agent, response = original_create(service, *args, **kwargs)
        assert response is not None
        captured["agent_id"] = int(agent.id)
        captured["prefix"] = response.key_prefix
        committed.set()
        assert release.wait(timeout=_HANDSHAKE_TIMEOUT)
        return agent, response

    monkeypatch.setattr(
        agent_management.AgentManagementService,
        "create_agent_with_optional_key",
        pause_after_commit,
    )
    if from_template:

        class TemplateManagerStub:
            async def get_template(self, template_id: str):  # type: ignore[no-untyped-def]
                assert template_id == "runtime-key-template"
                return {"name": "runtime key template", "agent_config": {}}

        operation: asyncio.Task[object] = asyncio.create_task(
            AgentManagementRuntime(
                template_manager=TemplateManagerStub()
            ).create_agent_from_template(
                user_id=user_id,
                is_admin=is_admin,
                template_id="runtime-key-template",
                name=None,
                description=None,
                instructions=None,
                execution_mode=None,
                models=None,
                knowledge_bases=None,
                skills=None,
                tool_categories=None,
                suggested_prompts=None,
                generate_runtime_key=True,
            )
        )
    else:
        operation = asyncio.create_task(
            AgentManagementRuntime().create_agent(
                user_id=user_id,
                is_admin=is_admin,
                spec=_create_spec("cancel committed create"),
            )
        )

    await _cancel_after_runtime_key_commit(operation, committed, release)

    db = _direct_db_session()
    try:
        row = (
            db.query(AgentApiKey)
            .filter(
                AgentApiKey.agent_id == captured["agent_id"],
                AgentApiKey.key_prefix == captured["prefix"],
            )
            .one()
        )
        assert row.revoked_at is not None
    finally:
        db.close()


@pytest.mark.asyncio
async def test_cancel_after_committed_rotation_revokes_the_undelivered_exact_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xagent.web.services import agent_management

    user_id, is_admin = _admin_identity()
    runtime = AgentManagementRuntime()
    target = await runtime.create_agent(
        user_id=user_id,
        is_admin=is_admin,
        spec=_create_spec("cancel committed rotation"),
    )
    db = _direct_db_session()
    try:
        db.add(
            AgentApiKey(
                agent_id=target.agent.id,
                key_prefix="SECOND",
                key_hash="second-active-key-hash",
            )
        )
        db.commit()
        previous_keys = (
            db.query(AgentApiKey)
            .filter(
                AgentApiKey.agent_id == target.agent.id,
                AgentApiKey.revoked_at.is_(None),
            )
            .all()
        )
        previous_key_ids = {int(row.id) for row in previous_keys}
        assert len(previous_key_ids) == 2
    finally:
        db.close()
    committed = threading.Event()
    release = threading.Event()
    captured: dict[str, str] = {}
    original_rotate = agent_management.AgentManagementService.generate_agent_runtime_key

    def pause_after_commit(service, *args, **kwargs):  # type: ignore[no-untyped-def]
        response = original_rotate(service, *args, **kwargs)
        assert response is not None
        captured["prefix"] = response.key_prefix
        committed.set()
        assert release.wait(timeout=_HANDSHAKE_TIMEOUT)
        return response

    monkeypatch.setattr(
        agent_management.AgentManagementService,
        "generate_agent_runtime_key",
        pause_after_commit,
    )
    operation: asyncio.Task[object] = asyncio.create_task(
        runtime.rotate_agent_runtime_key(user_id=user_id, agent_id=target.agent.id)
    )
    await _cancel_after_runtime_key_commit(operation, committed, release)

    db = _direct_db_session()
    try:
        rows = {
            int(row.id): row
            for row in db.query(AgentApiKey)
            .filter(AgentApiKey.agent_id == target.agent.id)
            .all()
        }
        assert all(rows[key_id].revoked_at is None for key_id in previous_key_ids)
        undelivered = next(
            row for row in rows.values() if row.key_prefix == captured["prefix"]
        )
        assert undelivered.revoked_at is not None
    finally:
        db.close()


@pytest.mark.asyncio
async def test_later_rotation_fence_prevents_stale_compensation_from_restoring_old_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xagent.web.services import agent_management

    user_id, is_admin = _admin_identity()
    runtime = AgentManagementRuntime()
    target = await runtime.create_agent(
        user_id=user_id,
        is_admin=is_admin,
        spec=_create_spec("later rotation fence"),
    )
    db = _direct_db_session()
    try:
        original_key = (
            db.query(AgentApiKey)
            .filter(
                AgentApiKey.agent_id == target.agent.id,
                AgentApiKey.revoked_at.is_(None),
            )
            .one()
        )
        original_key_id = int(original_key.id)
    finally:
        db.close()

    committed = threading.Event()
    release = threading.Event()
    captured: dict[str, str] = {}
    original_rotate = agent_management.AgentManagementService.generate_agent_runtime_key
    rotate_calls = 0

    def pause_first_after_commit(service, *args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal rotate_calls
        rotate_calls += 1
        response = original_rotate(service, *args, **kwargs)
        assert response is not None
        if rotate_calls == 1:
            captured["first_prefix"] = response.key_prefix
            committed.set()
            assert release.wait(timeout=_HANDSHAKE_TIMEOUT)
        return response

    monkeypatch.setattr(
        agent_management.AgentManagementService,
        "generate_agent_runtime_key",
        pause_first_after_commit,
    )
    operation: asyncio.Task[object] = asyncio.create_task(
        runtime.rotate_agent_runtime_key(
            user_id=user_id,
            agent_id=target.agent.id,
        )
    )
    assert await asyncio.to_thread(committed.wait, _HANDSHAKE_TIMEOUT)

    later = await AgentManagementRuntime().rotate_agent_runtime_key(
        user_id=user_id,
        agent_id=target.agent.id,
    )
    assert later is not None

    operation.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(operation, timeout=_HANDSHAKE_TIMEOUT)

    db = _direct_db_session()
    try:
        rows = {
            row.key_prefix: row
            for row in db.query(AgentApiKey)
            .filter(AgentApiKey.agent_id == target.agent.id)
            .all()
        }
        original = next(row for row in rows.values() if int(row.id) == original_key_id)
        assert original.revoked_at is not None
        assert rows[captured["first_prefix"]].revoked_at is not None
        assert rows[later.key_prefix].revoked_at is None
    finally:
        db.close()


@pytest.mark.asyncio
async def test_sqlite_runtime_key_transition_fence_serializes_transactions() -> None:
    """The transition fence must serialize writers before either snapshots keys."""

    from xagent.web.models.database import get_session_local
    from xagent.web.services.api_keys import acquire_runtime_key_transition_fence

    user_id, is_admin = _admin_identity()
    target = await AgentManagementRuntime().create_agent(
        user_id=user_id,
        is_admin=is_admin,
        spec=_create_spec("sqlite transition fence", generate_runtime_key=False),
    )
    observer = _direct_db_session()
    try:
        original_updated_at = observer.get(Agent, target.agent.id).updated_at
    finally:
        observer.close()
    SessionLocal = get_session_local()
    first_acquired = threading.Event()
    release_first = threading.Event()
    second_started = threading.Event()
    second_acquired = threading.Event()

    def hold_first_fence() -> None:
        with SessionLocal() as db:
            assert acquire_runtime_key_transition_fence(db, target.agent.id)
            first_acquired.set()
            assert release_first.wait(timeout=_HANDSHAKE_TIMEOUT)
            db.commit()

    def acquire_second_fence() -> None:
        assert first_acquired.wait(timeout=_HANDSHAKE_TIMEOUT)
        with SessionLocal() as db:
            second_started.set()
            assert acquire_runtime_key_transition_fence(db, target.agent.id)
            second_acquired.set()
            db.commit()

    first = asyncio.create_task(asyncio.to_thread(hold_first_fence))
    assert await asyncio.to_thread(first_acquired.wait, _HANDSHAKE_TIMEOUT)
    second = asyncio.create_task(asyncio.to_thread(acquire_second_fence))
    assert await asyncio.to_thread(second_started.wait, _HANDSHAKE_TIMEOUT)
    try:
        assert not await asyncio.to_thread(
            second_acquired.wait, _NEGATIVE_PROBE_TIMEOUT
        )
    finally:
        release_first.set()

    await asyncio.wait_for(first, timeout=_HANDSHAKE_TIMEOUT)
    await asyncio.wait_for(second, timeout=_HANDSHAKE_TIMEOUT)
    assert second_acquired.is_set()
    observer = _direct_db_session()
    try:
        assert observer.get(Agent, target.agent.id).updated_at == original_updated_at
    finally:
        observer.close()


@pytest.mark.asyncio
async def test_paused_later_state_blocks_runtime_key_compensation() -> None:
    """Compensation must not replace an operator's later pause decision."""

    user_id, is_admin = _admin_identity()
    target = await AgentManagementRuntime().create_agent(
        user_id=user_id,
        is_admin=is_admin,
        spec=_create_spec("paused compensation fence"),
    )

    db = _direct_db_session()
    try:
        previous_key = (
            db.query(AgentApiKey)
            .filter(
                AgentApiKey.agent_id == target.agent.id,
                AgentApiKey.revoked_at.is_(None),
            )
            .one()
        )
        previous_key_id = int(previous_key.id)
        key_service = AgentApiKeyService(db)
        key_service.rotate_key_for_runtime_delivery(
            target.agent.id,
            candidate=("xag_PAUSED_" + "p" * 32, "PAUSED", "paused-hash"),
        )
        receipt = key_service.runtime_key_receipt
        assert receipt is not None
        paused_at = datetime.now(timezone.utc)
        new_key = db.get(AgentApiKey, receipt.key_id)
        assert new_key is not None
        new_key.paused_at = paused_at
        db.commit()
    finally:
        db.close()

    result = AgentManagementRuntime._compensate_runtime_key_sync(receipt)

    assert result.new_key_revoked == 0
    assert result.prior_keys_restored == 0
    db = _direct_db_session()
    try:
        previous_key = db.get(AgentApiKey, previous_key_id)
        new_key = db.get(AgentApiKey, receipt.key_id)
        assert previous_key is not None
        assert new_key is not None
        assert previous_key.revoked_at is not None
        assert new_key.revoked_at is None
        assert new_key.paused_at is not None
    finally:
        db.close()


def test_runtime_key_session_operation_wraps_success_with_service_receipt() -> None:
    """The Session boundary owns the common detached success envelope."""

    expected = object()
    receipt = RuntimeKeyReceipt(key_id=1, agent_id=2, key_prefix="ABC123")

    def operation(service) -> object:  # type: ignore[no-untyped-def]
        service.runtime_key_receipt = receipt
        return expected

    outcome = AgentManagementRuntime._run_runtime_key_session_operation(operation)

    assert outcome.result is expected
    assert outcome.receipt is receipt
    assert outcome.error is None
    assert outcome.traceback is None


@pytest.mark.asyncio
async def test_rotation_mapping_failure_restores_the_previous_active_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A post-commit worker failure leaves the last delivered key usable."""

    from xagent.web.services import agent_management

    user_id, is_admin = _admin_identity()
    runtime = AgentManagementRuntime()
    target = await runtime.create_agent(
        user_id=user_id,
        is_admin=is_admin,
        spec=_create_spec("rotation mapping restore"),
    )
    db = _direct_db_session()
    try:
        previous_key = (
            db.query(AgentApiKey)
            .filter(
                AgentApiKey.agent_id == target.agent.id,
                AgentApiKey.revoked_at.is_(None),
            )
            .one()
        )
        previous_key_id = int(previous_key.id)
    finally:
        db.close()

    mapping_error = RuntimeError("runtime key mapping failed")
    monkeypatch.setattr(
        agent_management,
        "_runtime_key_snapshot",
        lambda _response: (_ for _ in ()).throw(mapping_error),
    )

    with pytest.raises(RuntimeError) as exc_info:
        await runtime.rotate_agent_runtime_key(
            user_id=user_id,
            agent_id=target.agent.id,
        )
    assert exc_info.value is mapping_error

    db = _direct_db_session()
    try:
        rows = (
            db.query(AgentApiKey)
            .filter(AgentApiKey.agent_id == target.agent.id)
            .order_by(AgentApiKey.id)
            .all()
        )
        assert len(rows) == 2
        assert (
            next(row for row in rows if int(row.id) == previous_key_id).revoked_at
            is None
        )
        assert (
            next(row for row in rows if int(row.id) != previous_key_id).revoked_at
            is not None
        )
    finally:
        db.close()


@pytest.mark.asyncio
async def test_compensation_cancellation_is_not_swallowed_after_worker_settles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = AgentManagementRuntime()
    started = threading.Event()
    release = threading.Event()

    def delayed_compensation(_receipt: RuntimeKeyReceipt) -> None:
        started.set()
        assert release.wait(timeout=_HANDSHAKE_TIMEOUT)

    monkeypatch.setattr(runtime, "_compensate_runtime_key_sync", delayed_compensation)
    compensation = asyncio.create_task(
        runtime._compensate_runtime_key(
            RuntimeKeyReceipt(key_id=1, agent_id=2, key_prefix="ABC123")
        )
    )
    assert await asyncio.to_thread(started.wait, _HANDSHAKE_TIMEOUT)
    compensation.cancel()
    release.set()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(compensation, timeout=_HANDSHAKE_TIMEOUT)


@pytest.mark.asyncio
async def test_delivery_error_keeps_its_existing_cause_when_compensation_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = AgentManagementRuntime()
    upstream_error = ValueError("response mapper root cause")
    delivery_error = RuntimeError("response mapping failed")
    delivery_error.__cause__ = upstream_error
    compensation_error = RuntimeError("compensation database unavailable")
    receipt = RuntimeKeyReceipt(key_id=1, agent_id=2, key_prefix="ABC123")

    async def fail_compensation(_receipt: RuntimeKeyReceipt) -> BaseException:
        return compensation_error

    monkeypatch.setattr(runtime, "_compensate_runtime_key", fail_compensation)

    with pytest.raises(RuntimeError) as exc_info:
        await runtime._run_runtime_key_delivery(
            lambda: _RuntimeKeyDeliveryOutcome(
                result=None,
                receipt=receipt,
                error=delivery_error,
                traceback=delivery_error.__traceback__,
            )
        )

    assert exc_info.value is delivery_error
    assert exc_info.value.__cause__ is upstream_error
    assert any(
        "compensation database unavailable" in note for note in exc_info.value.__notes__
    )


@pytest.mark.asyncio
async def test_runtime_key_delivery_keeps_worker_error_as_cancellation_cause() -> None:
    started = threading.Event()
    release = threading.Event()
    worker_error = RuntimeError("post-commit mapper failed")

    def operation() -> _RuntimeKeyDeliveryOutcome[object]:
        started.set()
        assert release.wait(timeout=_HANDSHAKE_TIMEOUT)
        return _RuntimeKeyDeliveryOutcome(
            result=None,
            receipt=None,
            error=worker_error,
            traceback=worker_error.__traceback__,
        )

    caller = asyncio.create_task(
        AgentManagementRuntime()._run_runtime_key_delivery(operation)
    )
    assert await asyncio.to_thread(started.wait, _HANDSHAKE_TIMEOUT)
    caller.cancel()
    release.set()

    with pytest.raises(asyncio.CancelledError) as exc_info:
        await asyncio.wait_for(caller, timeout=_HANDSHAKE_TIMEOUT)

    assert exc_info.value.__cause__ is worker_error


@pytest.mark.asyncio
async def test_runtime_key_delivery_keeps_process_control_over_cancellation() -> None:
    class WorkerShutdown(BaseException):
        pass

    started = threading.Event()
    release = threading.Event()
    shutdown = WorkerShutdown("controlled worker shutdown")

    def operation() -> _RuntimeKeyDeliveryOutcome[object]:
        started.set()
        assert release.wait(timeout=_HANDSHAKE_TIMEOUT)
        return _RuntimeKeyDeliveryOutcome(
            result=None,
            receipt=None,
            error=shutdown,
            traceback=shutdown.__traceback__,
        )

    caller = asyncio.create_task(
        AgentManagementRuntime()._run_runtime_key_delivery(operation)
    )
    assert await asyncio.to_thread(started.wait, _HANDSHAKE_TIMEOUT)
    caller.cancel()
    release.set()

    with pytest.raises(WorkerShutdown) as exc_info:
        await asyncio.wait_for(caller, timeout=_HANDSHAKE_TIMEOUT)

    assert exc_info.value is shutdown


def test_already_revoked_runtime_key_compensation_is_an_audited_noop(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="xagent.web.services.agent_management")
    result = AgentManagementRuntime._compensate_runtime_key_sync(
        RuntimeKeyReceipt(key_id=999999, agent_id=2, key_prefix="ABC123")
    )

    assert result.new_key_revoked == 0
    assert result.prior_keys_restored == 0
    assert "new_key_revoked=0" in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("persist_before_error", [True, False])
async def test_commit_ambiguity_preserves_the_exact_receipt_for_compensation(
    monkeypatch: pytest.MonkeyPatch,
    persist_before_error: bool,
) -> None:
    user_id, is_admin = _admin_identity()
    ambiguous = RuntimeError("commit response lost")
    original_commit = Session.commit
    commit_calls = 0

    def commit_then_raise(session: Session) -> None:
        nonlocal commit_calls
        commit_calls += 1
        if persist_before_error:
            original_commit(session)
        else:
            session.rollback()
        if commit_calls == 1:
            raise ambiguous
        original_commit(session)

    monkeypatch.setattr(Session, "commit", commit_then_raise)
    runtime = AgentManagementRuntime()
    with pytest.raises(RuntimeError) as exc_info:
        await runtime.create_agent(
            user_id=user_id,
            is_admin=is_admin,
            spec=_create_spec(f"ambiguous {persist_before_error}"),
        )

    assert exc_info.value is ambiguous
    monkeypatch.undo()
    db = _direct_db_session()
    try:
        rows = db.query(AgentApiKey).all()
        if persist_before_error:
            assert len(rows) == 1
            assert rows[0].revoked_at is not None
        else:
            assert rows == []
    finally:
        db.close()


@pytest.mark.asyncio
async def test_post_commit_mapping_failure_revokes_the_runtime_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xagent.web.services import agent_management

    user_id, is_admin = _admin_identity()
    captured: dict[str, int | str] = {}
    mapping_error = RuntimeError("response mapping failed")
    original_create = (
        agent_management.AgentManagementService.create_agent_with_optional_key
    )

    def record_key(service, *args, **kwargs):  # type: ignore[no-untyped-def]
        agent, response = original_create(service, *args, **kwargs)
        assert response is not None
        captured["agent_id"] = int(agent.id)
        captured["prefix"] = response.key_prefix
        return agent, response

    monkeypatch.setattr(
        agent_management.AgentManagementService,
        "create_agent_with_optional_key",
        record_key,
    )
    monkeypatch.setattr(
        agent_management.AgentStore,
        "agent_to_response_dict",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(mapping_error),
    )
    with pytest.raises(RuntimeError) as exc_info:
        await AgentManagementRuntime().create_agent(
            user_id=user_id,
            is_admin=is_admin,
            spec=_create_spec("mapping failure"),
        )

    assert exc_info.value is mapping_error
    db = _direct_db_session()
    try:
        row = (
            db.query(AgentApiKey)
            .filter(
                AgentApiKey.agent_id == captured["agent_id"],
                AgentApiKey.key_prefix == captured["prefix"],
            )
            .one()
        )
        assert row.revoked_at is not None
    finally:
        db.close()


@pytest.mark.asyncio
async def test_post_commit_session_close_failure_revokes_the_runtime_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xagent.web.services import agent_management

    user_id, is_admin = _admin_identity()
    captured: dict[str, int | str] = {}
    close_error = RuntimeError("worker session close failed")
    original_create = (
        agent_management.AgentManagementService.create_agent_with_optional_key
    )
    original_close = Session.close
    close_calls = 0

    def record_key(service, *args, **kwargs):  # type: ignore[no-untyped-def]
        agent, response = original_create(service, *args, **kwargs)
        assert response is not None
        captured["agent_id"] = int(agent.id)
        captured["prefix"] = response.key_prefix
        return agent, response

    def close_once(session: Session) -> None:
        nonlocal close_calls
        original_close(session)
        close_calls += 1
        if close_calls == 1:
            raise close_error

    monkeypatch.setattr(
        agent_management.AgentManagementService,
        "create_agent_with_optional_key",
        record_key,
    )
    monkeypatch.setattr(Session, "close", close_once)
    with pytest.raises(RuntimeError) as exc_info:
        await AgentManagementRuntime().create_agent(
            user_id=user_id,
            is_admin=is_admin,
            spec=_create_spec("session close failure"),
        )

    assert exc_info.value is close_error
    monkeypatch.undo()
    db = _direct_db_session()
    try:
        row = (
            db.query(AgentApiKey)
            .filter(
                AgentApiKey.agent_id == captured["agent_id"],
                AgentApiKey.key_prefix == captured["prefix"],
            )
            .one()
        )
        assert row.revoked_at is not None
    finally:
        db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("operation_kind", ["create", "rotate"])
@pytest.mark.parametrize("failure_stage", ["commit", "refresh"])
async def test_cancelled_receipt_failure_is_not_replaced_by_session_close_failure(
    monkeypatch: pytest.MonkeyPatch,
    operation_kind: str,
    failure_stage: str,
) -> None:
    """Keep post-commit delivery evidence through a failing Session.close()."""

    user_id, is_admin = _admin_identity()
    runtime = AgentManagementRuntime()
    agent_id: int | None = None
    if operation_kind == "rotate":
        created = await runtime.create_agent(
            user_id=user_id,
            is_admin=is_admin,
            spec=_create_spec(
                "close failure rotate target", generate_runtime_key=False
            ),
        )
        agent_id = created.agent.id

    primary_error = RuntimeError(f"{operation_kind} {failure_stage} failed")
    close_error = RuntimeError("worker session close failed")
    original_commit = Session.commit
    original_refresh = Session.refresh
    original_close = Session.close
    close_started = threading.Event()
    release_close = threading.Event()
    commit_calls = 0
    refresh_calls = 0
    close_calls = 0

    def fail_after_commit(session: Session) -> None:
        nonlocal commit_calls
        commit_calls += 1
        original_commit(session)
        if failure_stage == "commit" and commit_calls == 1:
            raise primary_error

    def fail_after_commit_refresh(session: Session, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        nonlocal refresh_calls
        refresh_calls += 1
        if failure_stage == "refresh" and refresh_calls == 1:
            raise primary_error
        original_refresh(session, *args, **kwargs)

    def close_after_primary_failure(session: Session) -> None:
        nonlocal close_calls
        close_calls += 1
        original_close(session)
        if close_calls == 1:
            close_started.set()
            assert release_close.wait(timeout=_HANDSHAKE_TIMEOUT)
            raise close_error

    monkeypatch.setattr(Session, "commit", fail_after_commit)
    monkeypatch.setattr(Session, "refresh", fail_after_commit_refresh)
    monkeypatch.setattr(Session, "close", close_after_primary_failure)

    if operation_kind == "create":
        operation: asyncio.Task[object] = asyncio.create_task(
            runtime.create_agent(
                user_id=user_id,
                is_admin=is_admin,
                spec=_create_spec(f"close failure {failure_stage}"),
            )
        )
    else:
        assert agent_id is not None
        operation = asyncio.create_task(
            runtime.rotate_agent_runtime_key(user_id=user_id, agent_id=agent_id)
        )

    assert await asyncio.to_thread(close_started.wait, _HANDSHAKE_TIMEOUT)
    operation.cancel()
    release_close.set()

    with pytest.raises(asyncio.CancelledError) as exc_info:
        await asyncio.wait_for(operation, timeout=_HANDSHAKE_TIMEOUT)

    assert exc_info.value.__cause__ is primary_error
    monkeypatch.undo()
    db = _direct_db_session()
    try:
        if agent_id is None:
            row = db.query(AgentApiKey).one()
        else:
            row = (
                db.query(AgentApiKey)
                .filter(AgentApiKey.agent_id == agent_id)
                .order_by(AgentApiKey.id.desc())
                .first()
            )
            assert row is not None
        assert row.revoked_at is not None
    finally:
        db.close()


@pytest.mark.asyncio
async def test_runtime_key_opt_out_and_not_found_rotation_need_no_compensation() -> (
    None
):
    user_id, is_admin = _admin_identity()
    runtime = AgentManagementRuntime()

    created = await runtime.create_agent(
        user_id=user_id,
        is_admin=is_admin,
        spec=_create_spec("no runtime key", generate_runtime_key=False),
    )

    assert created.api_key is None
    assert (
        await runtime.rotate_agent_runtime_key(user_id=user_id, agent_id=999999) is None
    )


@pytest.mark.asyncio
async def test_compensation_keeps_the_event_loop_responsive_while_pool_is_held(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xagent.web.services import agent_management

    user_id, is_admin = _admin_identity()
    engine = _install_one_slot_queue_pool(
        monkeypatch, pool_timeout=CONTENTION_POOL_TIMEOUT
    )
    committed = threading.Event()
    release = threading.Event()
    original_create = (
        agent_management.AgentManagementService.create_agent_with_optional_key
    )

    def pause_after_commit(service, *args, **kwargs):  # type: ignore[no-untyped-def]
        result = original_create(service, *args, **kwargs)
        committed.set()
        assert release.wait(timeout=_HANDSHAKE_TIMEOUT)
        return result

    monkeypatch.setattr(
        agent_management.AgentManagementService,
        "create_agent_with_optional_key",
        pause_after_commit,
    )
    operation: asyncio.Task[object] = asyncio.create_task(
        AgentManagementRuntime().create_agent(
            user_id=user_id,
            is_admin=is_admin,
            spec=_create_spec("pool compensation"),
        )
    )
    assert await asyncio.to_thread(committed.wait, _HANDSHAKE_TIMEOUT)
    held_connection = engine.connect()
    try:
        # Cancellation may swallow a worker failure. Assert the checkout's
        # thread outside the worker as well as observing the parked operation.
        with assert_pool_checkout_off_loop(engine), gated_pool_checkout(engine) as gate:
            operation.cancel()
            release.set()
            try:
                await gate.wait_until_contending()
                assert not operation.done()
            finally:
                held_connection.close()
                gate.let_through()
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(operation, timeout=_HANDSHAKE_TIMEOUT)
    finally:
        release.set()
        held_connection.close()
        assert engine.pool.checkedout() == 0
        engine.dispose()


@pytest.mark.asyncio
async def test_cancellation_stays_primary_when_runtime_key_compensation_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xagent.web.services import agent_management

    user_id, is_admin = _admin_identity()
    runtime = AgentManagementRuntime()
    committed = threading.Event()
    release = threading.Event()
    compensation_error = RuntimeError("revocation database unavailable")
    original_create = (
        agent_management.AgentManagementService.create_agent_with_optional_key
    )

    def pause_after_commit(service, *args, **kwargs):  # type: ignore[no-untyped-def]
        result = original_create(service, *args, **kwargs)
        committed.set()
        assert release.wait(timeout=_HANDSHAKE_TIMEOUT)
        return result

    monkeypatch.setattr(
        agent_management.AgentManagementService,
        "create_agent_with_optional_key",
        pause_after_commit,
    )
    monkeypatch.setattr(
        runtime,
        "_compensate_runtime_key_sync",
        lambda _receipt: (_ for _ in ()).throw(compensation_error),
    )
    operation: asyncio.Task[object] = asyncio.create_task(
        runtime.create_agent(
            user_id=user_id,
            is_admin=is_admin,
            spec=_create_spec("compensation failure"),
        )
    )
    assert await asyncio.to_thread(committed.wait, _HANDSHAKE_TIMEOUT)
    operation.cancel()
    release.set()

    with pytest.raises(asyncio.CancelledError) as exc_info:
        await asyncio.wait_for(operation, timeout=_HANDSHAKE_TIMEOUT)

    assert exc_info.value.__cause__ is compensation_error


@pytest.mark.asyncio
async def test_compensation_process_control_wins_over_caller_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xagent.web.services import agent_management

    class WorkerShutdown(BaseException):
        pass

    user_id, is_admin = _admin_identity()
    runtime = AgentManagementRuntime()
    committed = threading.Event()
    release = threading.Event()
    shutdown = WorkerShutdown("controlled worker shutdown")
    original_create = (
        agent_management.AgentManagementService.create_agent_with_optional_key
    )

    def pause_after_commit(service, *args, **kwargs):  # type: ignore[no-untyped-def]
        result = original_create(service, *args, **kwargs)
        committed.set()
        assert release.wait(timeout=_HANDSHAKE_TIMEOUT)
        return result

    monkeypatch.setattr(
        agent_management.AgentManagementService,
        "create_agent_with_optional_key",
        pause_after_commit,
    )
    monkeypatch.setattr(
        runtime,
        "_compensate_runtime_key_sync",
        lambda _receipt: (_ for _ in ()).throw(shutdown),
    )
    operation: asyncio.Task[object] = asyncio.create_task(
        runtime.create_agent(
            user_id=user_id,
            is_admin=is_admin,
            spec=_create_spec("compensation shutdown"),
        )
    )
    assert await asyncio.to_thread(committed.wait, _HANDSHAKE_TIMEOUT)
    operation.cancel()
    release.set()

    with pytest.raises(WorkerShutdown) as exc_info:
        await asyncio.wait_for(operation, timeout=_HANDSHAKE_TIMEOUT)

    assert exc_info.value is shutdown


@pytest.mark.asyncio
async def test_template_and_kb_materialization_have_no_outer_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xagent.web.services import agent_management

    user_id, is_admin = _admin_identity()
    engine = _install_one_slot_queue_pool(monkeypatch, pool_timeout=0.5)
    observations: list[tuple[str, int]] = []

    class TemplateManagerStub:
        async def get_template(self, template_id: str):  # type: ignore[no-untyped-def]
            observations.append(("template", engine.pool.checkedout()))
            assert template_id == "qa"
            return {
                "name": "Q&A",
                "descriptions": {"en": "Answers questions."},
                "agent_config": {
                    "instructions": "Answer clearly.",
                    "knowledge_bases": ["visible-kb"],
                    "tool_categories": ["knowledge"],
                },
            }

    async def no_missing_kbs(*args, **kwargs):  # type: ignore[no-untyped-def]
        observations.append(("knowledge_base", engine.pool.checkedout()))
        return []

    monkeypatch.setattr(
        agent_management,
        "find_missing_knowledge_bases",
        no_missing_kbs,
    )
    runtime = AgentManagementRuntime(template_manager=TemplateManagerStub())

    created = await runtime.create_agent_from_template(
        user_id=user_id,
        is_admin=is_admin,
        template_id="qa",
        name="Detached template",
        description=None,
        instructions=None,
        execution_mode=None,
        models=None,
        knowledge_bases=None,
        skills=None,
        tool_categories=None,
        suggested_prompts=None,
        generate_runtime_key=False,
    )

    assert created.agent.name == "Detached template"
    assert observations == [("template", 0), ("knowledge_base", 0)]
    engine.dispose()


@pytest.mark.asyncio
async def test_unique_prefix_collision_retries_whole_create_transaction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xagent.web.services import agent_management

    user_id, is_admin = _admin_identity()
    runtime = AgentManagementRuntime()
    existing = await runtime.create_agent(
        user_id=user_id,
        is_admin=is_admin,
        spec=_create_spec("existing key owner", generate_runtime_key=False),
    )

    colliding_prefix = "ABC123"
    db = _direct_db_session()
    try:
        db.add(
            AgentApiKey(
                agent_id=existing.agent.id,
                key_prefix=colliding_prefix,
                key_hash="hash-1",
            )
        )
        db.commit()
    finally:
        db.close()

    candidates = iter(
        [
            ("xag_ABC123_" + "a" * 32, colliding_prefix, "hash-1"),
            ("xag_XYZ789_" + "b" * 32, "XYZ789", "hash-2"),
        ]
    )
    monkeypatch.setattr(
        agent_management,
        "generate_api_key",
        lambda *_args, **_kwargs: next(candidates),
    )

    created = await runtime.create_agent(
        user_id=user_id,
        is_admin=is_admin,
        spec=_create_spec("collision retry"),
    )

    assert created.api_key is not None
    assert created.api_key.key_prefix == "XYZ789"
    db = _direct_db_session()
    try:
        rows = db.query(Agent).filter(Agent.name == "collision retry").all()
        assert len(rows) == 1
    finally:
        db.close()


@pytest.mark.asyncio
async def test_unique_prefix_collision_retries_whole_rotate_transaction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xagent.web.services import agent_management

    user_id, is_admin = _admin_identity()
    runtime = AgentManagementRuntime()
    owner = await runtime.create_agent(
        user_id=user_id,
        is_admin=is_admin,
        spec=_create_spec("rotate target", generate_runtime_key=False),
    )
    blocker = await runtime.create_agent(
        user_id=user_id,
        is_admin=is_admin,
        spec=_create_spec("prefix owner", generate_runtime_key=False),
    )

    db = _direct_db_session()
    try:
        db.add(
            AgentApiKey(
                agent_id=blocker.agent.id,
                key_prefix="ABC123",
                key_hash="hash-1",
            )
        )
        db.commit()
    finally:
        db.close()

    candidates = iter(
        [
            ("xag_ABC123_" + "a" * 32, "ABC123", "hash-1"),
            ("xag_XYZ789_" + "b" * 32, "XYZ789", "hash-2"),
        ]
    )
    monkeypatch.setattr(
        agent_management,
        "generate_api_key",
        lambda *_args, **_kwargs: next(candidates),
    )

    rotated = await runtime.rotate_agent_runtime_key(
        user_id=user_id,
        agent_id=owner.agent.id,
    )

    assert rotated is not None
    assert rotated.key_prefix == "XYZ789"
