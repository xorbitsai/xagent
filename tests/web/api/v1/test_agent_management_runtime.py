"""Concurrency-boundary tests for the V1 agent-management runtime."""

from __future__ import annotations

import asyncio
import threading

import pytest
from sqlalchemy import event, text

from xagent.web.models.agent import Agent
from xagent.web.models.agent_api_key import AgentApiKey
from xagent.web.models.user import User
from xagent.web.services.agent_management import (
    AgentCreateSpec,
    AgentManagementRuntime,
)

from ..conftest import (
    _admin_headers,
    _direct_db_session,
    _install_one_slot_queue_pool,
)

pytestmark = pytest.mark.usefixtures("_test_db")


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
    engine = _install_one_slot_queue_pool(monkeypatch, pool_timeout=0.5)
    held_connection = engine.connect()
    runtime = AgentManagementRuntime()
    listing = asyncio.create_task(runtime.list_agents(user_id=user_id))

    ticks = 0

    async def ticker() -> None:
        nonlocal ticks
        for _ in range(4):
            await asyncio.sleep(0.01)
            ticks += 1

    try:
        await ticker()
        assert ticks == 4
        assert listing.done() is False
    finally:
        held_connection.close()

    assert await listing == ()
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
        assert allow_cache_get.wait(timeout=2)
        return None

    def gated_cache_set(_key: str, _value):  # type: ignore[no-untyped-def]
        cache_observations.append(("set", engine.pool.checkedout()))
        cache_set_entered.set()
        assert allow_cache_set.wait(timeout=2)

    def probe_pool() -> None:
        with engine.connect() as connection:
            assert connection.execute(text("SELECT 1")).scalar_one() == 1

    monkeypatch.setattr(agent_store, "get_agent_team_scope", db_backed_team_scope)
    monkeypatch.setattr(agent_store, "cache_get", gated_cache_get)
    monkeypatch.setattr(agent_store, "cache_set", gated_cache_set)
    listing = asyncio.create_task(AgentManagementRuntime().list_agents(user_id=user_id))

    try:
        assert await asyncio.to_thread(cache_get_entered.wait, 2)
        assert cache_observations == [("get", 0)]
        await asyncio.to_thread(probe_pool)
        allow_cache_get.set()

        assert await asyncio.to_thread(cache_set_entered.wait, 2)
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
        assert await asyncio.to_thread(candidate_ready.wait, 2)
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

    assert await asyncio.to_thread(worker_started.wait, 1)
    listing.cancel()
    held_connection.close()

    with pytest.raises(asyncio.CancelledError):
        await listing
    assert engine.pool.checkedout() == 0
    engine.dispose()


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
