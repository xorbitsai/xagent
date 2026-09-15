import asyncio
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID

import pytest

from xagent.core.tools.adapters.vibe.sandboxed_tool.chrome_session import (
    ChromeSessionContractError,
)
from xagent.web.services import chrome_mcp_runtime
from xagent.web.services.actor_mcp_runtime import (
    ActorMCPStdioConnectionIdentity,
    ActorMCPStdioSessionIdentity,
)
from xagent.web.services.chrome_mcp_runtime import (
    _create_chrome_sandbox,
    bind_chrome_execution_scope,
)
from xagent.web.services.mcp_runtime import (
    MCPActorExecutionIdentity,
)


def _identity() -> ActorMCPStdioSessionIdentity:
    return ActorMCPStdioSessionIdentity(
        execution=MCPActorExecutionIdentity(
            task_id=7,
            run_id="run-one",
            turn_id="turn-one",
            lease_attempt_id="attempt-one",
        ),
        connection=ActorMCPStdioConnectionIdentity(
            user_id=11,
            resource_owner_key="toby:owner",
            app_id="chrome-devtools",
            catalog_app_generation=UUID("11111111-1111-4111-8111-111111111111"),
            lifecycle_generation=UUID("22222222-2222-4222-8222-222222222222"),
        ),
    )


def _connection():
    return {
        "transport": "stdio",
        "command": "npx",
        "args": [],
        "env": {"XAGENT_MCP_CALLER_ID": "11"},
    }


def test_same_identity_has_one_opaque_scope_and_strips_host_env():
    identity = _identity()
    first, first_connection = bind_chrome_execution_scope(
        "chrome-devtools", _connection(), identity
    )
    second, second_connection = bind_chrome_execution_scope(
        "chrome-devtools", _connection(), identity
    )

    assert first == second
    assert first.key == identity.key
    assert len(first.digest) == 64
    assert identity.connection.resource_owner_key not in repr(first)
    assert identity.execution.run_id not in repr(first)
    assert first_connection["env"] == {}
    assert second_connection["env"] == {}


@pytest.mark.parametrize(
    ("part", "field", "value"),
    [
        ("execution", "task_id", 8),
        ("execution", "run_id", "run-two"),
        ("execution", "turn_id", "turn-two"),
        ("execution", "lease_attempt_id", "attempt-two"),
        ("connection", "user_id", 12),
        ("connection", "resource_owner_key", "toby:other"),
        (
            "connection",
            "catalog_app_generation",
            UUID("33333333-3333-4333-8333-333333333333"),
        ),
        (
            "connection",
            "lifecycle_generation",
            UUID("44444444-4444-4444-8444-444444444444"),
        ),
    ],
)
def test_every_identity_dimension_changes_scope(part, field, value):
    identity = _identity()
    changed = replace(
        identity, **{part: replace(getattr(identity, part), **{field: value})}
    )
    connection = _connection()
    connection["env"]["XAGENT_MCP_CALLER_ID"] = str(changed.connection.user_id)

    original, _ = bind_chrome_execution_scope(
        "chrome-devtools", _connection(), identity
    )
    changed_scope, _ = bind_chrome_execution_scope(
        "chrome-devtools", connection, changed
    )

    assert original.key != changed_scope.key
    assert original.digest != changed_scope.digest


@pytest.mark.parametrize("identity", [None, object()])
def test_execution_scoped_chrome_requires_exact_identity(identity):
    with pytest.raises(ChromeSessionContractError, match="exact session identity"):
        bind_chrome_execution_scope("chrome-devtools", _connection(), identity)


def test_execution_scoped_chrome_rejects_identity_subclass():
    class DerivedIdentity(ActorMCPStdioSessionIdentity):
        pass

    identity = _identity()
    derived = DerivedIdentity(identity.execution, identity.connection)

    with pytest.raises(ChromeSessionContractError, match="exact session identity"):
        bind_chrome_execution_scope("chrome-devtools", _connection(), derived)


def test_rejects_identity_inside_child_connection():
    connection = _connection()
    connection["actor_stdio_session_identity"] = _identity()
    with pytest.raises(ChromeSessionContractError, match="host-only side channel"):
        bind_chrome_execution_scope("chrome-devtools", connection, _identity())


def test_rejects_wrong_app_and_unexpected_child_env():
    with pytest.raises(ChromeSessionContractError, match="canonical Chrome app"):
        bind_chrome_execution_scope("ordinary-server", _connection(), _identity())
    wrong_identity = replace(
        _identity(),
        connection=replace(_identity().connection, app_id="other-app"),
    )
    with pytest.raises(ChromeSessionContractError, match="does not match"):
        bind_chrome_execution_scope("chrome-devtools", _connection(), wrong_identity)
    connection = _connection()
    connection["env"]["SECRET"] = "must-not-cross"
    with pytest.raises(ChromeSessionContractError, match="unexpected environment"):
        bind_chrome_execution_scope("chrome-devtools", connection, _identity())


def test_chrome_identity_user_mismatch_fails_closed():
    connection = _connection()
    connection["env"]["XAGENT_MCP_CALLER_ID"] = "999"
    with pytest.raises(ChromeSessionContractError, match="caller identity"):
        bind_chrome_execution_scope("chrome-devtools", connection, _identity())


@pytest.mark.asyncio
async def test_dedicated_sandbox_uses_only_opaque_scope_and_attaches(monkeypatch):
    events = []
    manager = SimpleNamespace(
        get_or_create_lease_provider=AsyncMock(),
        attach_provider=AsyncMock(return_value=True),
        delete_durable_sandbox_strict=AsyncMock(),
    )
    provider = SimpleNamespace(primary_sandbox=object())
    manager.get_or_create_lease_provider.side_effect = lambda *args: (
        events.append("backend-create") or provider
    )
    lifecycle = SimpleNamespace(
        backend_lifecycle_digest="b" * 64,
        mark_ready=AsyncMock(side_effect=lambda: events.append("ready")),
        renew=AsyncMock(),
        delete=AsyncMock(),
        defer_unknown_create=AsyncMock(),
    )
    coordinator = SimpleNamespace(
        register=AsyncMock(
            side_effect=lambda **kwargs: events.append("register") or lifecycle
        )
    )
    monkeypatch.setattr(chrome_mcp_runtime, "get_sandbox_manager", lambda: manager)
    monkeypatch.setattr(
        chrome_mcp_runtime,
        "get_chrome_lifecycle_coordinator",
        lambda: coordinator,
    )

    scope, _ = bind_chrome_execution_scope(
        "chrome-devtools", _connection(), _identity()
    )
    handle = await _create_chrome_sandbox(scope)
    await handle.before_backend()
    await handle.delete()

    coordinator.register.assert_awaited_once_with(
        scope_digest=scope.digest,
        task_id=7,
        run_id="run-one",
        lease_attempt_id="attempt-one",
        turn_id="turn-one",
    )
    manager.get_or_create_lease_provider.assert_awaited_once_with(
        "chrome-execution", "b" * 64
    )
    manager.attach_provider.assert_awaited_once_with(
        "chrome-execution", "b" * 64, provider
    )
    lifecycle.mark_ready.assert_awaited_once()
    lifecycle.renew.assert_awaited_once()
    lifecycle.delete.assert_awaited_once()
    assert events == ["register", "backend-create", "ready"]


@pytest.mark.asyncio
async def test_backend_create_failure_tombstones_before_returning(monkeypatch):
    manager = SimpleNamespace(
        get_or_create_lease_provider=AsyncMock(
            side_effect=RuntimeError("backend create unknown")
        ),
        attach_provider=AsyncMock(),
    )
    lifecycle = SimpleNamespace(
        backend_lifecycle_digest="b" * 64,
        mark_ready=AsyncMock(),
        renew=AsyncMock(),
        delete=AsyncMock(),
        defer_unknown_create=AsyncMock(),
    )
    coordinator = SimpleNamespace(register=AsyncMock(return_value=lifecycle))
    monkeypatch.setattr(chrome_mcp_runtime, "get_sandbox_manager", lambda: manager)
    monkeypatch.setattr(
        chrome_mcp_runtime,
        "get_chrome_lifecycle_coordinator",
        lambda: coordinator,
    )
    scope, _ = bind_chrome_execution_scope(
        "chrome-devtools", _connection(), _identity()
    )

    with pytest.raises(ChromeSessionContractError, match="creation failed"):
        await _create_chrome_sandbox(scope)

    lifecycle.defer_unknown_create.assert_awaited_once()
    lifecycle.delete.assert_not_awaited()
    lifecycle.mark_ready.assert_not_awaited()
    manager.attach_provider.assert_not_awaited()


@pytest.mark.asyncio
async def test_cancelled_backend_create_drains_unknown_tombstone(monkeypatch):
    create_started = asyncio.Event()
    tombstoned = asyncio.Event()

    async def create(*_args):
        create_started.set()
        await asyncio.Event().wait()

    async def defer_unknown_create():
        await asyncio.sleep(0)
        tombstoned.set()

    manager = SimpleNamespace(
        get_or_create_lease_provider=AsyncMock(side_effect=create),
        attach_provider=AsyncMock(),
    )
    lifecycle = SimpleNamespace(
        backend_lifecycle_digest="b" * 64,
        mark_ready=AsyncMock(),
        renew=AsyncMock(),
        delete=AsyncMock(),
        defer_unknown_create=AsyncMock(side_effect=defer_unknown_create),
    )
    coordinator = SimpleNamespace(register=AsyncMock(return_value=lifecycle))
    monkeypatch.setattr(chrome_mcp_runtime, "get_sandbox_manager", lambda: manager)
    monkeypatch.setattr(
        chrome_mcp_runtime,
        "get_chrome_lifecycle_coordinator",
        lambda: coordinator,
    )
    scope, _ = bind_chrome_execution_scope(
        "chrome-devtools", _connection(), _identity()
    )

    task = asyncio.create_task(_create_chrome_sandbox(scope))
    await create_started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert tombstoned.is_set()
    lifecycle.defer_unknown_create.assert_awaited_once()
    lifecycle.delete.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_at", ["attach", "ready"])
async def test_post_create_failure_strictly_compensates(monkeypatch, failure_at):
    provider = SimpleNamespace(primary_sandbox=object())
    manager = SimpleNamespace(
        get_or_create_lease_provider=AsyncMock(return_value=provider),
        attach_provider=AsyncMock(return_value=True),
    )
    lifecycle = SimpleNamespace(
        backend_lifecycle_digest="b" * 64,
        mark_ready=AsyncMock(),
        renew=AsyncMock(),
        delete=AsyncMock(),
        defer_unknown_create=AsyncMock(),
    )
    if failure_at == "attach":
        manager.attach_provider.side_effect = RuntimeError("attach failed")
    else:
        lifecycle.mark_ready.side_effect = RuntimeError("ready failed")
    coordinator = SimpleNamespace(register=AsyncMock(return_value=lifecycle))
    monkeypatch.setattr(chrome_mcp_runtime, "get_sandbox_manager", lambda: manager)
    monkeypatch.setattr(
        chrome_mcp_runtime,
        "get_chrome_lifecycle_coordinator",
        lambda: coordinator,
    )
    scope, _ = bind_chrome_execution_scope(
        "chrome-devtools", _connection(), _identity()
    )

    with pytest.raises(ChromeSessionContractError):
        await _create_chrome_sandbox(scope)

    lifecycle.delete.assert_awaited_once()
    lifecycle.defer_unknown_create.assert_not_awaited()


@pytest.mark.asyncio
async def test_pool_manager_change_fails_closed_and_shutdown_drains(monkeypatch):
    first_manager = object()
    second_manager = object()
    pool = AsyncMock()
    chrome_mcp_runtime._chrome_pool_manager = first_manager
    chrome_mcp_runtime._chrome_pool = pool
    monkeypatch.setattr(
        chrome_mcp_runtime, "get_sandbox_manager", lambda: second_manager
    )

    with pytest.raises(ChromeSessionContractError, match="manager changed"):
        chrome_mcp_runtime.get_chrome_execution_session_pool()

    await chrome_mcp_runtime.shutdown_chrome_execution_session_pool()
    pool.close_all.assert_awaited_once()
    assert chrome_mcp_runtime._chrome_pool is None
    assert chrome_mcp_runtime._chrome_pool_manager is None


@pytest.mark.asyncio
async def test_shutdown_resets_pool_even_when_drain_fails():
    pool = AsyncMock()
    pool.close_all.side_effect = RuntimeError("drain failed")
    chrome_mcp_runtime._chrome_pool_manager = object()
    chrome_mcp_runtime._chrome_pool = pool

    with pytest.raises(RuntimeError, match="drain failed"):
        await chrome_mcp_runtime.shutdown_chrome_execution_session_pool()

    assert chrome_mcp_runtime._chrome_pool is None
    assert chrome_mcp_runtime._chrome_pool_manager is None


@pytest.mark.asyncio
async def test_recovery_runs_immediately_then_periodically_until_stopped(monkeypatch):
    events: list[str] = []
    loop_started = asyncio.Event()

    async def sweep_once():
        events.append("immediate-sweep")
        return 0

    async def run_sweep_loop():
        events.append("periodic-loop")
        loop_started.set()
        await asyncio.Event().wait()

    coordinator = SimpleNamespace(
        sweep_once=AsyncMock(side_effect=sweep_once),
        run_sweep_loop=run_sweep_loop,
    )
    app = SimpleNamespace(state=SimpleNamespace())
    monkeypatch.setattr(chrome_mcp_runtime, "get_sandbox_manager", object)
    monkeypatch.setattr(
        chrome_mcp_runtime,
        "get_chrome_lifecycle_coordinator",
        lambda: coordinator,
    )

    task = await chrome_mcp_runtime.start_chrome_lifecycle_recovery(app)
    await asyncio.wait_for(loop_started.wait(), timeout=1)
    assert task is app.state.chrome_lifecycle_recovery_task
    assert events == ["immediate-sweep", "periodic-loop"]

    await chrome_mcp_runtime.stop_chrome_lifecycle_recovery(app)
    assert task.cancelled()
    assert app.state.chrome_lifecycle_recovery_task is None


@pytest.mark.asyncio
async def test_initial_database_unknown_does_not_disable_periodic_recovery(monkeypatch):
    periodic_started = asyncio.Event()

    async def run_sweep_loop():
        periodic_started.set()
        await asyncio.Event().wait()

    coordinator = SimpleNamespace(
        sweep_once=AsyncMock(side_effect=RuntimeError("database unavailable")),
        run_sweep_loop=run_sweep_loop,
    )
    app = SimpleNamespace(state=SimpleNamespace())
    monkeypatch.setattr(chrome_mcp_runtime, "get_sandbox_manager", object)
    monkeypatch.setattr(
        chrome_mcp_runtime,
        "get_chrome_lifecycle_coordinator",
        lambda: coordinator,
    )

    task = await chrome_mcp_runtime.start_chrome_lifecycle_recovery(app)
    await asyncio.wait_for(periodic_started.wait(), timeout=1)

    coordinator.sweep_once.assert_awaited_once()
    assert task is not None and not task.done()
    await chrome_mcp_runtime.stop_chrome_lifecycle_recovery(app)
