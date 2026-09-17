"""A worker admits no execution until its runtime and event transport are ready."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from sqlalchemy import text

from xagent.web import worker
from xagent.web.models.database import Base, get_engine, init_db


def test_worker_schema_requires_deployed_revision_without_migrating(tmp_path):
    init_db(db_url=f"sqlite:///{tmp_path / 'worker.db'}")
    worker.validate_worker_schema()
    with get_engine().begin() as connection:
        connection.execute(text("DELETE FROM alembic_version"))
    with pytest.raises(RuntimeError, match="run migrations"):
        worker.validate_worker_schema()
    with get_engine().connect() as connection:
        assert (
            connection.execute(
                text("SELECT COUNT(*) FROM alembic_version")
            ).scalar_one()
            == 0
        )
    Base.metadata.drop_all(bind=get_engine())


@pytest.mark.asyncio
@pytest.mark.parametrize("bridge_fails", [False, True])
async def test_runtime_readiness_and_shutdown_order(monkeypatch, bridge_fails):
    monkeypatch.setenv("XAGENT_SHARED_TASK_EXECUTION_ENABLED", "true")
    monkeypatch.setenv("XAGENT_TASK_EXECUTION_ROLE", "worker")
    monkeypatch.setenv("XAGENT_REDIS_URL", "redis://localhost:6379/0")
    from cryptography.fernet import Fernet

    monkeypatch.setenv("ENCRYPTION_KEY", Fernet.generate_key().decode())
    order = []
    for name in (
        "configure_db",
        "validate_worker_schema",
        "validate_interaction_rollout_at_startup",
        "register_local_browser_runtime",
        "register_execution_scope_snapshot_loader",
        "initialize_langfuse",
        "shutdown_task_runtime_hook_executor",
        "unregister_local_browser_runtime",
        "flush_langfuse",
    ):
        monkeypatch.setattr(worker, name, Mock())
    manager = SimpleNamespace(
        start_accepting=lambda: order.append("accept"),
        shutdown=AsyncMock(side_effect=lambda: order.append("drain")),
    )
    monkeypatch.setattr(worker, "background_task_manager", manager)
    monkeypatch.setattr(
        worker,
        "close_task_coordinators",
        AsyncMock(side_effect=lambda: order.append("coordinator_drain")),
    )
    monkeypatch.setattr(
        worker, "create_skill_manager", lambda: SimpleNamespace(initialize=AsyncMock())
    )
    monkeypatch.setattr(
        worker,
        "create_template_manager",
        lambda: SimpleNamespace(initialize=AsyncMock()),
    )
    monkeypatch.setattr(worker, "get_sandbox_manager", lambda: None)

    from xagent.web.services import chrome_mcp_runtime

    pool = SimpleNamespace(
        close_all=AsyncMock(side_effect=lambda: order.append("chrome_drain"))
    )
    monkeypatch.setattr(chrome_mcp_runtime, "_chrome_pool", pool)
    monkeypatch.setattr(chrome_mcp_runtime, "_chrome_pool_manager", object())

    async def start_bridge():
        order.append("bridge")
        if bridge_fails:
            raise ConnectionError("bridge unavailable")

    monkeypatch.setattr(worker, "start_task_event_bridge", start_bridge)

    async def stop_dispatch():
        order.append("stop_claims")

    async def stop_bridge():
        order.append("stop_bridge")

    async def stop_heartbeat():
        order.append("heartbeat_idle")

    async def recovery(**kwargs):
        await asyncio.Event().wait()

    monkeypatch.setattr(worker, "stop_task_command_dispatcher", stop_dispatch)
    monkeypatch.setattr(worker, "stop_task_event_bridge", stop_bridge)
    monkeypatch.setattr(worker, "wait_for_heartbeat_manager_idle", stop_heartbeat)
    monkeypatch.setattr(worker, "run_task_lease_recovery_loop", recovery)
    monkeypatch.setattr(
        worker,
        "start_task_command_dispatcher",
        lambda executor: order.append("dispatch"),
    )

    async def initialize_host():
        order.append("host_hooks")

    stop = asyncio.Event()
    stop.set()
    if bridge_fails:
        with pytest.raises(ConnectionError):
            await worker.run_worker(initialize_host=initialize_host, stop=stop)
        assert "accept" not in order
        assert "dispatch" not in order
    else:
        await worker.run_worker(initialize_host=initialize_host, stop=stop)
        assert (
            order.index("host_hooks")
            < order.index("bridge")
            < order.index("accept")
            < order.index("dispatch")
        )
    assert (
        order.index("stop_claims")
        < order.index("coordinator_drain")
        < order.index("drain")
        < order.index("heartbeat_idle")
        < order.index("stop_bridge")
        < order.index("chrome_drain")
    )

    pool.close_all.assert_awaited_once()
    assert chrome_mcp_runtime._chrome_pool is None
    assert chrome_mcp_runtime._chrome_pool_manager is None
