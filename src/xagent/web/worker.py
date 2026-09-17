"""HTTP-free shared execution host: ``python -m xagent.web.worker``.

Embedding deployments can call run_worker(initialize_host=...) to register the
same trusted runtime providers, resolvers and admission checks as their web host.
"""

from __future__ import annotations

import asyncio
import logging
import signal
from collections.abc import Awaitable, Callable

from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from dotenv import load_dotenv

from ..config import (
    get_shared_task_execution_enabled,
    get_task_execution_role,
    get_task_lease_recovery_batch_size,
    get_task_lease_recovery_interval_seconds,
    validate_task_execution_host_config,
)
from ..core.tracing.langfuse import flush_langfuse, initialize_langfuse
from ..db.config import create_alembic_config
from ..skills.utils import create_skill_manager
from ..templates.utils import create_template_manager
from .models.database import configure_db, get_engine
from .sandbox_manager import check_sandbox_static_readiness, get_sandbox_manager
from .services.chrome_mcp_runtime import shutdown_chrome_execution_session_pool
from .services.execution_scope_snapshot import register_execution_scope_snapshot_loader
from .services.interaction_rollout import validate_interaction_rollout_at_startup
from .services.local_browser_runtime import (
    register_local_browser_runtime,
    unregister_local_browser_runtime,
)
from .services.task_command_execution import execute_durable_task_command
from .services.task_command_transport import (
    start_task_command_dispatcher,
    stop_task_command_dispatcher,
)
from .services.task_coordinator_runtime import close_task_coordinators
from .services.task_event_bridge import start_task_event_bridge, stop_task_event_bridge
from .services.task_execution import background_task_manager
from .services.task_lease_recovery import run_task_lease_recovery_loop
from .services.task_lease_service import wait_for_heartbeat_manager_idle
from .services.task_runtime import shutdown_task_runtime_hook_executor

logger = logging.getLogger(__name__)


def validate_worker_schema() -> None:
    """Inspect the deployed revision; worker replicas never execute migrations."""
    engine = get_engine()
    expected = set(
        ScriptDirectory.from_config(create_alembic_config(engine)).get_heads()
    )
    with engine.connect() as connection:
        actual = set(MigrationContext.configure(connection).get_current_heads())
    if actual != expected:
        raise RuntimeError(
            "Worker database schema does not match this release; run migrations before starting workers"
        )


async def run_worker(
    *,
    initialize_host: Callable[[], Awaitable[None]] | None = None,
    stop: asyncio.Event | None = None,
) -> None:
    validate_task_execution_host_config()
    if not get_shared_task_execution_enabled() or get_task_execution_role() != "worker":
        raise ValueError(
            "The worker entry point requires shared execution and the worker role"
        )
    configure_db()
    await asyncio.to_thread(validate_worker_schema)
    validate_interaction_rollout_at_startup()
    recovery = None
    sandbox = None
    idle_sweep = None
    stop = stop if stop is not None else asyncio.Event()
    try:
        if initialize_host is not None:
            await initialize_host()
        register_local_browser_runtime()
        register_execution_scope_snapshot_loader()
        initialize_langfuse()
        await create_skill_manager().initialize()
        await create_template_manager().initialize()
        sandbox = get_sandbox_manager()
        if sandbox is not None:
            await check_sandbox_static_readiness(sandbox)
            await sandbox.cleanup()
            await sandbox.warmup()
            from ..config import get_sandbox_idle_ttl

            if get_sandbox_idle_ttl() is not None:
                idle_sweep = asyncio.create_task(sandbox.run_idle_sweep_loop())
        await start_task_event_bridge()
        background_task_manager.start_accepting()
        recovery = asyncio.create_task(
            run_task_lease_recovery_loop(
                poll_interval_seconds=get_task_lease_recovery_interval_seconds(),
                batch_size=get_task_lease_recovery_batch_size(),
            )
        )
        start_task_command_dispatcher(execute_durable_task_command)
        logger.info("Shared task worker ready")
        await stop.wait()
    finally:
        # Stop every claim path first. Finalizers retain the bridge and runtime
        # resources until execution and its heartbeats have drained.
        await stop_task_command_dispatcher()
        if recovery is not None:
            recovery.cancel()
            await asyncio.gather(recovery, return_exceptions=True)
        await close_task_coordinators()
        await background_task_manager.shutdown()
        await wait_for_heartbeat_manager_idle()
        await stop_task_event_bridge()
        if idle_sweep is not None:
            idle_sweep.cancel()
            await asyncio.gather(idle_sweep, return_exceptions=True)
        shutdown_task_runtime_hook_executor()
        unregister_local_browser_runtime()
        try:
            await shutdown_chrome_execution_session_pool()
        except Exception:
            logger.exception("Failed to drain Chrome execution sessions")
        if sandbox is not None:
            await sandbox.cleanup()
        flush_langfuse()


async def _main() -> None:
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)
    try:
        await run_worker(stop=stop)
    finally:
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.remove_signal_handler(sig)


if __name__ == "__main__":
    load_dotenv()
    from .logging_config import setup_logging

    setup_logging(force=True)
    asyncio.run(_main())
