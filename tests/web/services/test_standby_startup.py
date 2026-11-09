"""Standby through the real startup and shutdown handlers of the web app.

Pins what ``test_standby.py`` cannot: that ``startup_event`` defers every write,
claim path and singleton (not just the database half), that activation runs
them in a normal start's order, that a late activation failure undoes what the
early steps opened, that a standby never cleans up shared sandboxes, and that
its database binding is read-only until activation.
"""

from __future__ import annotations

import asyncio
import importlib
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import sqlalchemy as sa

from tests.integration.test_auto_migration_startup import (  # noqa: F401
    unregister_startup_task_extensions,
)
from tests.integration.test_auto_migration_startup import (
    _patch_channel_modules_disabled,
)
from xagent.config import STANDBY as STANDBY_ENV
from xagent.web.models import database as database_module
from xagent.web.services.release_hold import release_hold
from xagent.web.services.standby import StandbyPhase, standby

app_module = importlib.import_module("xagent.web.app")


@pytest.fixture(autouse=True)
def clean_state():
    standby.reset()
    yield
    standby.reset()
    release_hold.reopen()


class _Manager:
    async def initialize(self) -> None:
        return None

    async def list_skills(self) -> list[str]:
        return []

    async def list_templates(self) -> list[str]:
        return []


class _Sandbox:
    def __init__(self, events: list[str]) -> None:
        self._events = events
        self.cleanup = AsyncMock(side_effect=lambda: events.append("sandbox_cleanup"))

    async def _resolve_backend_probe(self) -> bool:
        return False

    async def warmup(self) -> None:
        self._events.append("sandbox_warmup")


def _patch_startup(monkeypatch: pytest.MonkeyPatch, events: list[str]) -> _Sandbox:
    """Record every startup side effect that standby must postpone."""
    _patch_channel_modules_disabled(monkeypatch)
    sandbox = _Sandbox(events)
    monkeypatch.setenv(STANDBY_ENV, "1")
    monkeypatch.setenv("LANCEDB_AUTO_MIGRATE", "false")
    monkeypatch.setattr(app_module, "init_db", lambda: events.append("init_db"))
    monkeypatch.setattr(
        app_module, "init_db_standby", lambda: events.append("bind_readonly")
    )
    monkeypatch.setattr(
        app_module,
        "complete_standby_db_init",
        lambda: events.append("init_db_deferred"),
    )

    async def admissions(_app) -> None:
        events.append("admissions")

    monkeypatch.setattr(app_module, "run_host_startup_admissions", admissions)
    monkeypatch.setattr(
        "xagent.web.services.trace_database.get_trace_database_runtime",
        lambda: events.append("trace_db"),
    )
    monkeypatch.setattr(
        app_module, "register_local_browser_runtime", lambda: events.append("browser")
    )
    for name in (
        "start_file_storage_startup_sync_task",
        "start_trigger_dispatcher_task",
        "start_task_lease_recovery_task",
        "start_task_auto_resume_task",
        "start_uploaded_file_recovery_task",
        "start_orphan_upload_gc_task",
        "start_retention_purge_task",
        "start_task_cleanup_retry_task",
        "start_composio_reconcile_task",
        "start_runtime_performance_monitor",
        "start_app_process_diagnostics",
    ):
        monkeypatch.setattr(app_module, name, lambda _a, n=name: events.append(n))
    monkeypatch.setattr(
        app_module,
        "start_temp_file_cleanup_task",
        lambda _a: events.append("start_temp_file_cleanup_task"),
    )
    monkeypatch.setattr(
        "xagent.web.services.task_command_transport.start_task_command_dispatcher",
        lambda _e: events.append("command_dispatcher"),
    )
    monkeypatch.setattr("xagent.skills.utils.create_skill_manager", lambda: _Manager())
    monkeypatch.setattr(
        "xagent.templates.utils.create_template_manager", lambda: _Manager()
    )
    monkeypatch.setattr(
        "xagent.web.dynamic_memory_store.admit_memory_storage_at_startup",
        lambda: events.append("memory_admission"),
    )
    monkeypatch.setattr(
        "xagent.web.sandbox_manager.get_sandbox_manager", lambda: sandbox
    )

    async def readiness(_mgr) -> None:
        events.append("sandbox_readiness")

    monkeypatch.setattr(
        "xagent.web.sandbox_manager.check_sandbox_static_readiness", readiness
    )
    monkeypatch.setattr(
        "xagent.providers.vector_store.lancedb.get_connection_from_env",
        lambda: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "xagent.core.tools.core.RAG_tools.LanceDB.schema_manager.check_table_needs_migration",
        lambda _c, _t: False,
    )
    return sandbox


@pytest.mark.usefixtures("unregister_startup_task_extensions")
@pytest.mark.asyncio
async def test_startup_event_defers_every_write_and_singleton_until_activation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    sandbox = _patch_startup(monkeypatch, events)
    monkeypatch.setattr(
        "xagent.web.app.get_shared_task_execution_enabled", lambda: False
    )

    await app_module.startup_event()

    # Reads and warmup only; closed, sealed hold.
    assert events == [
        "bind_readonly",
        "browser",
        "sandbox_readiness",
        "start_runtime_performance_monitor",
        "start_app_process_diagnostics",
    ]
    assert release_hold.closed and release_hold.sealed
    assert standby.phase is StandbyPhase.STANDBY
    sandbox.cleanup.assert_not_awaited()

    await standby.activate()
    after = events[5:]
    expected_order = [
        "init_db_deferred",
        "admissions",
        "trace_db",
        "start_task_lease_recovery_task",
        "memory_admission",
        "sandbox_cleanup",
        "sandbox_warmup",
        "command_dispatcher",
        "start_composio_reconcile_task",
    ]
    positions = [after.index(name) for name in expected_order]
    assert positions == sorted(positions), after
    assert "init_db" not in events
    assert not release_hold.closed and standby.phase is StandbyPhase.ACTIVE


@pytest.mark.usefixtures("unregister_startup_task_extensions")
@pytest.mark.asyncio
async def test_late_activation_failure_reseals_and_stops_started_loops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    _patch_startup(monkeypatch, events)
    monkeypatch.setattr(
        "xagent.web.app.get_shared_task_execution_enabled", lambda: False
    )
    stopped: list[str] = []

    async def stop_loops() -> None:
        stopped.append("stopped")

    monkeypatch.setattr(app_module, "_stop_runtime_loops", stop_loops)

    monkeypatch.setattr(
        "xagent.web.dynamic_memory_store.admit_memory_storage_at_startup",
        lambda: (_ for _ in ()).throw(RuntimeError("memory admission failed")),
    )
    await app_module.startup_event()
    with pytest.raises(RuntimeError, match="memory admission failed"):
        await standby.activate()

    assert standby.phase is StandbyPhase.FAILED
    # "open admission" had already run: the failure hook undid it.
    assert "start_task_lease_recovery_task" in events
    assert release_hold.sealed and release_hold.closed
    assert stopped == ["stopped"]


@pytest.mark.asyncio
@pytest.mark.parametrize("started_as_standby", [True, False])
async def test_shutdown_cleans_sandboxes_only_when_it_owned_them(
    monkeypatch: pytest.MonkeyPatch, started_as_standby: bool
) -> None:
    events: list[str] = []
    sandbox = _Sandbox(events)
    monkeypatch.setattr(
        "xagent.web.sandbox_manager.get_sandbox_manager", lambda: sandbox
    )
    monkeypatch.setattr(app_module, "flush_langfuse", lambda: None)
    for name in (
        "stop_orphan_upload_gc_task",
        "stop_composio_reconcile_task",
        "stop_retention_purge_task",
        "stop_task_cleanup_retry_task",
        "stop_uploaded_file_recovery_task",
        "stop_task_auto_resume_task",
        "stop_task_lease_recovery_task",
        "stop_runtime_performance_monitor",
        "stop_app_process_diagnostics",
    ):
        monkeypatch.setattr(app_module, name, AsyncMock())
    if started_as_standby:
        standby.begin()

    await app_module.shutdown_event()

    if started_as_standby:
        sandbox.cleanup.assert_not_awaited()
    else:
        sandbox.cleanup.assert_awaited_once()


def _sqlite_at_older_revision(tmp_path: Path) -> str:
    from alembic.script import ScriptDirectory

    from xagent.db.migration import create_alembic_config

    url = f"sqlite:///{tmp_path / 'db.sqlite'}"
    engine = sa.create_engine(url)
    script = ScriptDirectory.from_config(create_alembic_config(engine))
    heads = set(script.get_heads())
    older = next(r.revision for r in script.walk_revisions() if r.revision not in heads)
    with engine.begin() as conn:
        conn.execute(sa.text("CREATE TABLE alembic_version (version_num VARCHAR(32))"))
        conn.execute(sa.text("INSERT INTO alembic_version VALUES (:r)"), {"r": older})
    engine.dispose()
    return url


def test_standby_database_is_read_only_until_activation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    url = _sqlite_at_older_revision(tmp_path)
    monkeypatch.setattr(database_module, "_engine", None)
    monkeypatch.setattr(database_module, "_SessionLocal", None)
    monkeypatch.setattr(
        database_module, "_initialize_database_schema", lambda _engine: []
    )
    monkeypatch.setattr(
        "xagent.web.services.encryption_key_rotation.validate_stored_encryption_keys",
        lambda: None,
    )

    database_module.init_db_standby(url)
    try:
        with database_module.get_engine().begin() as conn:
            with pytest.raises(sa.exc.OperationalError, match="readonly"):
                conn.execute(sa.text("UPDATE alembic_version SET version_num = 'x'"))

        database_module.complete_standby_db_init()
        with database_module.get_engine().begin() as conn:
            conn.execute(sa.text("UPDATE alembic_version SET version_num = 'y'"))
    finally:
        database_module.get_engine().dispose()


@pytest.mark.asyncio
async def test_abort_stops_channels_and_the_shared_event_bridge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    channel = asyncio.create_task(asyncio.sleep(60))
    monkeypatch.setattr(app_module.app.state, "slack_task", channel, raising=False)
    monkeypatch.setattr(app_module, "_stop_runtime_loops", AsyncMock())
    cleanup_stop = threading.Event()
    monkeypatch.setattr(
        app_module.app.state, "temp_file_cleanup_stop", cleanup_stop, raising=False
    )
    monkeypatch.setattr(app_module, "get_shared_task_execution_enabled", lambda: True)
    reconciliation = AsyncMock()
    bridge = AsyncMock()
    monkeypatch.setattr(
        "xagent.web.api.websocket.manager.stop_stream_reconciliation", reconciliation
    )
    monkeypatch.setattr(
        "xagent.web.services.task_event_bridge.stop_task_event_bridge", bridge
    )

    await app_module._abort_standby_activation()

    assert channel.cancelled()
    assert cleanup_stop.is_set()
    reconciliation.assert_awaited_once()
    bridge.assert_awaited_once()
    assert release_hold.sealed
