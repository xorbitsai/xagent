import asyncio
import json
import logging
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from typing import Any, cast

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware
from starlette.types import ASGIApp, Receive, Scope, Send

from ..config import (
    get_agent_runtime,
    get_background_job_sweep_interval_seconds,
    get_file_storage_startup_sync_enabled,
    get_gmail_watch_enabled,
    get_gmail_watch_renewal_interval_seconds,
    get_orphan_upload_sweep_interval_seconds,
    get_session_secret,
    get_task_lease_recovery_batch_size,
    get_task_lease_recovery_interval_seconds,
    get_taskless_upload_ttl_seconds,
    get_trigger_dispatcher_batch_size,
    get_trigger_dispatcher_enabled,
    get_trigger_dispatcher_interval_seconds,
    get_uploaded_file_recovery_batch_size,
    get_uploaded_file_recovery_interval_seconds,
    get_uploaded_file_recovery_stale_seconds,
    get_uploads_dir,
)
from ..core.execution_scope import (
    ExecutionScopeAuthorityError,
    ExecutionScopeResolverContractError,
)
from ..core.file_storage import StorageKeyScopeError
from ..core.tracing.langfuse import flush_langfuse, initialize_langfuse
from .api.a2a import router as a2a_router
from .api.admin_interaction_rollout import router as admin_interaction_rollout_router
from .api.admin_mcp import admin_mcp_router
from .api.admin_users import router as admin_users_router
from .api.agent_api_keys import router as agent_api_keys_router
from .api.agents import router as agents_router
from .api.auth import auth_router
from .api.channel import router as channel_router
from .api.chat import chat_router
from .api.cloud_storage import cloud_router
from .api.computer import computer_router
from .api.conversation_logs import router as conversation_logs_router
from .api.custom_api import custom_api_router
from .api.deployment_config import router as deployment_config_router
from .api.files import file_router
from .api.jobs import jobs_router
from .api.kb import kb_router
from .api.mcp import mcp_router
from .api.me import router as me_router
from .api.memory import MemoryManagementRouter
from .api.model import model_router
from .api.monitor import monitor_router
from .api.personal_api_keys import router as personal_api_keys_router
from .api.progress_ws import progress_ws_router
from .api.share import share_router
from .api.skill_hub import router as skill_hub_router
from .api.skills import router as skills_router
from .api.system import system_router
from .api.templates import router as templates_router
from .api.tools import tools_router
from .api.triggers import router as triggers_router
from .api.v1 import v1_router
from .api.v1.errors import V1ApiError, V1ErrorCode, v1_api_error_handler
from .api.websocket import ws_router
from .api.widget import widget_router
from .api.workforces import router as workforces_router
from .dynamic_memory_store import get_memory_store
from .logging_config import setup_logging
from .models.database import init_db
from .services.a2a_protocol import A2AApiError, a2a_api_error_handler, a2a_error
from .services.interaction_rollout import (
    get_interaction_rollout_policy,
    is_native_schema_ready,
    mark_native_schema_ready,
    validate_interaction_rollout_at_startup,
)
from .services.local_browser_runtime import (
    register_local_browser_runtime,
    unregister_local_browser_runtime,
)
from .services.ops_signals import (
    INTERACTION_ROLLOUT_SCHEMA_ABSENT,
    clear_degradation,
    register_degradation,
)
from .services.orphan_upload_gc import run_orphan_upload_gc_loop
from .services.skill_runtime import (
    SkillRuntimeSessionBoundaryError,
    skill_runtime_session_boundary_error_handler,
)
from .services.task_interaction_schema import interaction_requests_table_exists
from .services.task_lease_recovery import run_task_lease_recovery_loop
from .services.uploaded_file_recovery import (
    run_uploaded_file_compensation_recovery_loop,
)

# Configure logging when running under gunicorn/uwsgi (no __main__.py)
setup_logging()  # Uses XAGENT_LOG_LEVEL env var or defaults to INFO

logger = logging.getLogger(__name__)

__all__ = ["app"]


@contextmanager
def _startup_phase(name: str) -> Iterator[None]:
    """Log a begin/end pair with duration around a startup phase.

    A slow phase awaited inline with no logs makes a multi-minute stall look
    like a fast start. One line in, one line out per phase (never per
    loop/tick) makes the next slow start obvious. A failing phase still logs
    its end line before the error propagates.
    """
    logger.info("startup phase begin: %s", name)
    started = time.monotonic()
    try:
        yield
    except asyncio.CancelledError:
        # WHY: CancelledError is a BaseException, so the Exception handler
        # below misses it and the terminal line would be dropped.
        logger.error(
            "startup phase cancelled: %s (after %.2fs)",
            name,
            time.monotonic() - started,
        )
        raise
    except Exception:
        logger.error(
            "startup phase failed: %s (after %.2fs)", name, time.monotonic() - started
        )
        raise
    else:
        logger.info("startup phase done: %s (%.2fs)", name, time.monotonic() - started)


# Ensure web, uploads directory exists before configuring static files
uploads_dir = get_uploads_dir()
uploads_dir.mkdir(parents=True, exist_ok=True)


# FastAPI app creation here
app = FastAPI(
    title="xagent", description="The Agent Operating System", redirect_slashes=False
)

# Track background migration task for graceful shutdown cleanup.
_migration_task: asyncio.Task[None] | None = None
_file_storage_startup_sync_task: asyncio.Task[Any] | None = None
_trigger_dispatcher_task: asyncio.Task[Any] | None = None
_task_command_dispatcher_task: asyncio.Task[Any] | None = None
_sandbox_idle_sweep_task: asyncio.Task[None] | None = None

FILE_STORAGE_STARTUP_SYNC_EXEMPT_PATHS = frozenset({"/health", "/ready"})
FILE_STORAGE_STARTUP_SYNC_RETRY_INTERVAL_SECONDS = 5.0
FILE_STORAGE_STARTUP_SYNC_GATE_POLL_INTERVAL_SECONDS = 0.25


def run_startup_file_storage_sync() -> None:
    """Synchronize DB-registered local files into durable S3 storage."""
    if not get_file_storage_startup_sync_enabled():
        logger.info("Startup file storage sync is disabled")
        return

    from .services.startup_file_storage_sync import (
        sync_registered_files_to_durable_storage,
    )

    result = sync_registered_files_to_durable_storage()
    if result.failed:
        raise RuntimeError(
            f"Startup file storage sync failed for {result.failed} registered file(s)"
        )


async def _run_file_storage_startup_sync_with_retries(
    app_instance: FastAPI,
    *,
    retry_interval_seconds: float,
) -> None:
    while True:
        try:
            await asyncio.to_thread(run_startup_file_storage_sync)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            app_instance.state.file_storage_startup_sync_error = exc
            app_instance.state.file_storage_startup_sync_completed = False
            logger.error(
                "Startup file storage sync failed; retrying in %s seconds: %s",
                retry_interval_seconds,
                exc,
                exc_info=True,
            )
            await asyncio.sleep(retry_interval_seconds)
        else:
            app_instance.state.file_storage_startup_sync_error = None
            app_instance.state.file_storage_startup_sync_completed = True
            return


def start_file_storage_startup_sync_task(
    app_instance: FastAPI,
    *,
    retry_interval_seconds: float | None = None,
) -> asyncio.Task[Any] | None:
    """Start durable file storage startup sync without blocking app startup."""
    global _file_storage_startup_sync_task

    app_instance.state.file_storage_startup_sync_task = None
    app_instance.state.file_storage_startup_sync_error = None

    if not get_file_storage_startup_sync_enabled():
        logger.info("Startup file storage sync is disabled")
        app_instance.state.file_storage_startup_sync_completed = True
        return None

    resolved_retry_interval_seconds = (
        FILE_STORAGE_STARTUP_SYNC_RETRY_INTERVAL_SECONDS
        if retry_interval_seconds is None
        else retry_interval_seconds
    )
    task = asyncio.create_task(
        _run_file_storage_startup_sync_with_retries(
            app_instance,
            retry_interval_seconds=resolved_retry_interval_seconds,
        )
    )
    _file_storage_startup_sync_task = task
    app_instance.state.file_storage_startup_sync_task = task
    app_instance.state.file_storage_startup_sync_completed = False

    def _record_file_storage_sync_result(done_task: asyncio.Task[Any]) -> None:
        try:
            done_task.result()
        except asyncio.CancelledError:
            app_instance.state.file_storage_startup_sync_error = None
            app_instance.state.file_storage_startup_sync_completed = False
        except Exception as exc:  # noqa: BLE001
            app_instance.state.file_storage_startup_sync_error = exc
            app_instance.state.file_storage_startup_sync_completed = False
            logger.error("Startup file storage sync failed: %s", exc, exc_info=True)
        else:
            app_instance.state.file_storage_startup_sync_error = None
            app_instance.state.file_storage_startup_sync_completed = True

    task.add_done_callback(_record_file_storage_sync_result)
    logger.info("Started background startup file storage sync task")
    return task


async def _run_trigger_dispatcher(
    *,
    poll_interval_seconds: int,
    batch_size: int,
) -> None:
    from .models.database import get_session_local
    from .services.gmail_triggers import scan_due_gmail_watch_renewals
    from .services.triggers import (
        dispatch_pending_trigger_runs,
        scan_due_scheduled_triggers,
    )
    from .services.workforce_runtime import (
        WorkforceRunPauseTarget,
        pause_workforce_tasks_after_archive,
        reap_stale_preview_workforce_runs,
    )

    def _scan_due_scheduled_triggers_tick() -> int:
        # Scan in-process so scheduled triggers fire without a Celery
        # beat/worker. Idempotent with the Celery beat scan (unique run key +
        # next_run_at advanced in one committed txn), so running both is safe.
        SessionLocal = get_session_local()
        db = SessionLocal()
        try:
            return len(scan_due_scheduled_triggers(db))
        finally:
            db.close()

    def _reap_stale_preview_workforce_runs_tick() -> list[WorkforceRunPauseTarget]:
        # This is the third of three trigger-scan entrypoints (alongside the
        # Celery Beat and BackgroundJob-driven variants) -- runs in-process
        # for the same reason _scan_due_scheduled_triggers_tick does: a
        # deployment without Celery Beat must still reap abandoned
        # workforce-builder preview runs, not silently skip it.
        SessionLocal = get_session_local()
        db = SessionLocal()
        try:
            return reap_stale_preview_workforce_runs(db)
        finally:
            db.close()

    def _scan_due_gmail_watch_renewals_tick() -> int:
        SessionLocal = get_session_local()
        db = SessionLocal()
        try:
            return scan_due_gmail_watch_renewals(db)
        finally:
            db.close()

    def _sweep_gmail_provisioning_tick() -> int:
        from ..config import get_gmail_pubsub_project_id
        from .services.gmail_provisioning import sweep_gmail_provisioning

        if not get_gmail_pubsub_project_id():
            return 0
        SessionLocal = get_session_local()
        db = SessionLocal()
        try:
            return sweep_gmail_provisioning(db)
        finally:
            db.close()

    loop = asyncio.get_running_loop()
    next_gmail_watch_scan_at = 0.0
    next_preview_run_reap_at = 0.0
    while True:
        try:
            now = loop.time()
            if now >= next_gmail_watch_scan_at:
                try:
                    if get_gmail_watch_enabled():
                        renewed = await asyncio.to_thread(
                            _scan_due_gmail_watch_renewals_tick
                        )
                        if renewed:
                            logger.info(
                                "Trigger dispatcher renewed %s Gmail watch(es)",
                                renewed,
                            )
                        swept = await asyncio.to_thread(_sweep_gmail_provisioning_tick)
                        if swept:
                            logger.info(
                                "Trigger dispatcher retried %s Gmail registration(s)",
                                swept,
                            )
                finally:
                    next_gmail_watch_scan_at = (
                        now + get_gmail_watch_renewal_interval_seconds()
                    )

            processed = await asyncio.to_thread(_scan_due_scheduled_triggers_tick)
            if processed:
                logger.info(
                    "Trigger dispatcher processed %s due schedule(s)", processed
                )

            # Gated on its own, much coarser timer (matching the Gmail
            # watch-renewal gating above): the staleness threshold this
            # sweep acts on is hours-scale (get_workforce_preview_run_stale_
            # seconds, default 7200s), so checking on every dispatcher tick
            # (as low as a few seconds, get_trigger_dispatcher_interval_
            # seconds) is unnecessary load with no corresponding benefit.
            if now >= next_preview_run_reap_at:
                try:
                    reaped_pause_targets = await asyncio.to_thread(
                        _reap_stale_preview_workforce_runs_tick
                    )
                    if reaped_pause_targets:
                        await pause_workforce_tasks_after_archive(
                            reaped_pause_targets, reason="preview-reap"
                        )
                        logger.info(
                            "Trigger dispatcher paused %s orphaned preview "
                            "workforce run(s)",
                            len(reaped_pause_targets),
                        )
                finally:
                    next_preview_run_reap_at = (
                        now + get_background_job_sweep_interval_seconds()
                    )

            SessionLocal = get_session_local()
            db = SessionLocal()
            try:
                started = await dispatch_pending_trigger_runs(db, limit=batch_size)
                if started:
                    logger.info("Trigger dispatcher started %s trigger run(s)", started)
            finally:
                db.close()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.warning("Trigger dispatcher tick failed: %s", exc, exc_info=True)

        await asyncio.sleep(poll_interval_seconds)


def start_trigger_dispatcher_task(app_instance: FastAPI) -> asyncio.Task[Any] | None:
    """Start backend-side trigger execution dispatcher."""
    global _trigger_dispatcher_task

    app_instance.state.trigger_dispatcher_task = None
    if not get_trigger_dispatcher_enabled():
        logger.info("Trigger dispatcher is disabled")
        return None
    if os.getenv("PYTEST_CURRENT_TEST"):
        logger.info("Skipping trigger dispatcher (test environment)")
        return None

    poll_interval_seconds = get_trigger_dispatcher_interval_seconds()
    batch_size = get_trigger_dispatcher_batch_size()
    task = asyncio.create_task(
        _run_trigger_dispatcher(
            poll_interval_seconds=poll_interval_seconds,
            batch_size=batch_size,
        )
    )
    _trigger_dispatcher_task = task
    app_instance.state.trigger_dispatcher_task = task
    logger.info(
        "Started trigger dispatcher task (interval=%ss, batch_size=%s)",
        poll_interval_seconds,
        batch_size,
    )
    return task


def start_task_lease_recovery_task(
    app_instance: FastAPI,
) -> asyncio.Task[Any] | None:
    """Start automatic expired task-lease recovery for this backend process."""

    existing_task = cast(
        asyncio.Task[Any] | None,
        getattr(
            app_instance.state,
            "task_lease_recovery_task",
            None,
        ),
    )
    if existing_task is not None:
        if not existing_task.done():
            return existing_task
        try:
            failure = existing_task.exception()
        except asyncio.CancelledError:
            failure = None
        if failure is not None:
            logger.error(
                "Previous task lease recovery loop failed",
                exc_info=failure,
            )
        app_instance.state.task_lease_recovery_task = None

    if os.getenv("PYTEST_CURRENT_TEST"):
        logger.info("Skipping task lease recovery loop (test environment)")
        return None

    poll_interval_seconds = get_task_lease_recovery_interval_seconds()
    batch_size = get_task_lease_recovery_batch_size()
    task = asyncio.create_task(
        run_task_lease_recovery_loop(
            poll_interval_seconds=poll_interval_seconds,
            batch_size=batch_size,
        )
    )
    app_instance.state.task_lease_recovery_task = task
    logger.info(
        "Started task lease recovery loop (interval=%ss, batch_size=%s)",
        poll_interval_seconds,
        batch_size,
    )
    return task


async def stop_task_lease_recovery_task(app_instance: FastAPI) -> None:
    """Cancel and drain this process's task lease recovery loop."""

    task = getattr(app_instance.state, "task_lease_recovery_task", None)
    app_instance.state.task_lease_recovery_task = None
    if task is not None and not task.done():
        logger.info("Cancelling task lease recovery loop...")
        task.cancel()
    if task is not None:
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.error(
                "Task lease recovery loop stopped after failure",
                exc_info=exc,
            )


def start_uploaded_file_recovery_task(
    app_instance: FastAPI,
) -> asyncio.Task[Any] | None:
    """Start stale uploaded-file compensation recovery for this process."""

    existing_task = cast(
        asyncio.Task[Any] | None,
        getattr(app_instance.state, "uploaded_file_recovery_task", None),
    )
    if existing_task is not None:
        if not existing_task.done():
            return existing_task
        try:
            failure = existing_task.exception()
        except asyncio.CancelledError:
            failure = None
        if failure is not None:
            logger.error(
                "Previous uploaded-file recovery loop failed",
                exc_info=failure,
            )
        app_instance.state.uploaded_file_recovery_task = None

    if os.getenv("PYTEST_CURRENT_TEST"):
        logger.info("Skipping uploaded-file recovery loop (test environment)")
        return None

    poll_interval_seconds = get_uploaded_file_recovery_interval_seconds()
    stale_after_seconds = get_uploaded_file_recovery_stale_seconds()
    batch_size = get_uploaded_file_recovery_batch_size()
    task = asyncio.create_task(
        run_uploaded_file_compensation_recovery_loop(
            poll_interval_seconds=poll_interval_seconds,
            stale_after_seconds=stale_after_seconds,
            batch_size=batch_size,
        )
    )
    app_instance.state.uploaded_file_recovery_task = task
    logger.info(
        "Started uploaded-file recovery loop "
        "(interval=%ss, stale_after=%ss, batch_size=%s)",
        poll_interval_seconds,
        stale_after_seconds,
        batch_size,
    )
    return task


async def stop_uploaded_file_recovery_task(app_instance: FastAPI) -> None:
    """Cancel and drain this process's uploaded-file recovery loop."""

    task = getattr(app_instance.state, "uploaded_file_recovery_task", None)
    app_instance.state.uploaded_file_recovery_task = None
    if task is not None and not task.done():
        logger.info("Cancelling uploaded-file recovery loop...")
        task.cancel()
    if task is not None:
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.error(
                "Uploaded-file recovery loop stopped after failure",
                exc_info=exc,
            )


def start_orphan_upload_gc_task(
    app_instance: FastAPI,
) -> asyncio.Task[Any] | None:
    """Start the orphan task-less-upload GC loop for this process (#973).

    Runs in-app (like the uploaded-file recovery loop) so every supported
    deployment reaps abandoned task-less public uploads — the GC must not
    depend on an optional Celery worker while Gunicorn-only deployments keep
    accepting task-less uploads.
    """

    existing_task = cast(
        asyncio.Task[Any] | None,
        getattr(app_instance.state, "orphan_upload_gc_task", None),
    )
    if existing_task is not None:
        if not existing_task.done():
            return existing_task
        try:
            failure = existing_task.exception()
        except asyncio.CancelledError:
            failure = None
        if failure is not None:
            logger.error(
                "Previous orphan upload GC loop failed",
                exc_info=failure,
            )
        app_instance.state.orphan_upload_gc_task = None

    if os.getenv("PYTEST_CURRENT_TEST"):
        logger.info("Skipping orphan upload GC loop (test environment)")
        return None

    poll_interval_seconds = get_orphan_upload_sweep_interval_seconds()
    ttl_seconds = get_taskless_upload_ttl_seconds()
    task = asyncio.create_task(
        run_orphan_upload_gc_loop(
            poll_interval_seconds=poll_interval_seconds,
            ttl_seconds=ttl_seconds,
        )
    )
    app_instance.state.orphan_upload_gc_task = task
    logger.info(
        "Started orphan upload GC loop (interval=%ss, ttl=%ss)",
        poll_interval_seconds,
        ttl_seconds,
    )
    return task


async def stop_orphan_upload_gc_task(app_instance: FastAPI) -> None:
    """Cancel and drain this process's orphan upload GC loop."""

    task = getattr(app_instance.state, "orphan_upload_gc_task", None)
    app_instance.state.orphan_upload_gc_task = None
    if task is not None and not task.done():
        logger.info("Cancelling orphan upload GC loop...")
        task.cancel()
    if task is not None:
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.error(
                "Orphan upload GC loop stopped after failure",
                exc_info=exc,
            )


async def wait_for_file_storage_startup_sync(app_instance: FastAPI) -> None:
    """Wait until durable file storage startup sync has completed successfully."""
    while True:
        error = getattr(app_instance.state, "file_storage_startup_sync_error", None)
        if error is not None:
            raise error

        task = getattr(app_instance.state, "file_storage_startup_sync_task", None)
        if task is None:
            return

        if task.done():
            try:
                await task
            except Exception as exc:  # noqa: BLE001
                app_instance.state.file_storage_startup_sync_error = exc
                app_instance.state.file_storage_startup_sync_completed = False
                raise

            error = getattr(app_instance.state, "file_storage_startup_sync_error", None)
            if error is not None:
                raise error
            return

        try:
            await asyncio.wait_for(
                asyncio.shield(task),
                timeout=FILE_STORAGE_STARTUP_SYNC_GATE_POLL_INTERVAL_SECONDS,
            )
        except TimeoutError:
            continue
        except Exception as exc:  # noqa: BLE001
            app_instance.state.file_storage_startup_sync_error = exc
            app_instance.state.file_storage_startup_sync_completed = False
            raise


class FileStorageStartupSyncGateMiddleware:
    """Gate client traffic until durable file storage startup sync completes."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        scope_type = scope.get("type")
        if scope_type not in {"http", "websocket"}:
            await self.app(scope, receive, send)
            return

        path = str(scope.get("path") or "")
        if path in FILE_STORAGE_STARTUP_SYNC_EXEMPT_PATHS:
            await self.app(scope, receive, send)
            return
        if scope_type == "http" and str(scope.get("method") or "").upper() == "OPTIONS":
            await self.app(scope, receive, send)
            return

        app_instance = scope.get("app")
        if isinstance(app_instance, FastAPI):
            try:
                await wait_for_file_storage_startup_sync(app_instance)
            except Exception:
                if scope_type == "websocket":
                    await send(
                        {
                            "type": "websocket.close",
                            "code": 1013,
                            "reason": "Startup file storage sync failed",
                        }
                    )
                    return

                response = JSONResponse(
                    status_code=503,
                    content={"detail": "Startup file storage sync failed"},
                )
                await response(scope, receive, send)
                return

        await self.app(scope, receive, send)


@app.get("/health")
async def health_check() -> dict[str, Any]:
    """Health check endpoint for container probes.

    Active degraded-mode signals ride along for monitoring to alert on;
    the status stays "ok" so probes keep passing while degraded. Only
    signal names are exposed — /health is unauthenticated, and the detail
    strings describe security-relevant misconfiguration; those stay in the
    logs and the in-process registry.
    """
    from .services.ops_signals import active_degradations

    payload: dict[str, Any] = {"status": "ok"}
    degradations = active_degradations()
    if degradations:
        payload["degradations"] = sorted(degradations)
    return payload


@app.get("/ready")
async def readiness_check() -> JSONResponse:
    """Readiness check for startup file storage sync and, in native
    interaction-rollout mode, task_interaction_requests schema presence.

    The interaction-rollout segment below is a one-way latch: once the
    schema has been observed present, this endpoint never queries for it
    again for the lifetime of the process (the table is only ever added,
    never dropped, so a stale "present" reading cannot happen). In legacy
    or read mode the schema check and the database query it would trigger
    are skipped entirely -- zero added query. The frozen policy read
    above the mode check always runs regardless of mode, including its
    RuntimeError-if-uninitialized contract (see
    ``get_interaction_rollout_policy``'s docstring); that read is cheap
    (no I/O once the policy is frozen at startup) but it is not skipped.
    """
    task = getattr(app.state, "file_storage_startup_sync_task", None)
    error = getattr(app.state, "file_storage_startup_sync_error", None)

    if error is not None:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "detail": "Startup file storage sync failed"},
        )
    if task is not None and not task.done():
        return JSONResponse(
            status_code=503,
            content={
                "status": "starting",
                "detail": "Startup file storage sync running",
            },
        )

    policy = get_interaction_rollout_policy()
    if policy.mode == "native" and not is_native_schema_ready():
        # If the database itself is unreachable here (connection refused,
        # timeout), that exception propagates out of this route unhandled
        # and FastAPI turns it into a 500 -- asymmetric with the deliberate
        # 503 below for "database reachable, schema not yet migrated".
        # Accepted: a 500 still fails the readiness probe the same way a
        # 503 would, and folding "database unreachable" into that same
        # typed 503 would misreport a connectivity outage as a pending
        # migration.
        #
        # get_session_local is imported here rather than hoisted with the
        # policy/schema imports above: tests replace
        # database_module.get_session_local itself with a stub session
        # factory (see tests/web/test_interaction_rollout_observability.py),
        # and this import must re-resolve that name from the module on
        # every call for the replacement to take effect. A module-level
        # import would bind the original function once at process start
        # and keep calling it through that binding, silently defeating the
        # test's monkeypatch.
        from .models.database import get_session_local

        SessionLocal = get_session_local()
        db = SessionLocal()
        try:
            schema_present = interaction_requests_table_exists(db)
        finally:
            db.close()

        if not schema_present:
            # This endpoint is unauthenticated -- the detail below
            # deliberately does not name the missing table.
            register_degradation(
                INTERACTION_ROLLOUT_SCHEMA_ABSENT,
                "task_interaction_requests table not present",
            )
            return JSONResponse(
                status_code=503,
                content={
                    "status": "error",
                    "detail": "Interaction rollout schema not ready",
                },
            )

        mark_native_schema_ready()
        clear_degradation(INTERACTION_ROLLOUT_SCHEMA_ABSENT)

    return JSONResponse(status_code=200, content={"status": "ready"})


# Add global exception handler
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Handle request validation errors, especially those containing binary data.

    Path-aware response shape so the SDK envelope contract holds:

      - ``/v1/*``: rewrite to ``{"error": {"code": "invalid_input",
        "message": "..."}}`` so SDK clients can switch on
        ``body.error.code`` for 422 the same way they do for 401/404/409.
        FastAPI raises ``RequestValidationError`` before the endpoint
        runs, so the v1 endpoints themselves never see this -- the
        handler is the only place to translate.
      - ``/api/*`` and other paths: keep the existing
        ``{"detail": [sanitized_errors]}`` shape that the web UI and
        in-house clients already parse.
    """
    import traceback

    logger.error(f"Validation error in {request.url}: {str(exc)}")
    logger.error(f"Traceback: {traceback.format_exc()}")

    if request.url.path.startswith("/api/a2a/"):
        errors = exc.errors()
        first = errors[0] if errors else {}
        msg = first.get("msg") or "Invalid A2A request"
        loc = ".".join(str(part) for part in first.get("loc", []))
        return await a2a_api_error_handler(
            request,
            a2a_error(
                "invalid_argument",
                f"{msg} ({loc})" if loc else str(msg),
                status_code=400,
                details={"field": loc},
            ),
        )

    if request.url.path.startswith("/v1/"):
        # Take the first validation error as the human message; full
        # list isn't echoed to keep the response surface small and
        # avoid leaking internal field path patterns. Server log above
        # has the full detail for debugging.
        errors = exc.errors()
        first = errors[0] if errors else {}
        msg = first.get("msg") or "Invalid request body"
        loc = ".".join(str(p) for p in first.get("loc", []) if p not in (None, "body"))
        if loc:
            msg = f"{msg} ({loc})"
        return JSONResponse(
            status_code=422,
            content={"error": {"code": "invalid_input", "message": msg}},
        )

    # Sanitize error details to remove binary data and non-serializable objects
    sanitized_errors = []
    for error in exc.errors():
        sanitized_error = {}
        for key, value in error.items():
            # Try to serialize each value to check if it's JSON-serializable
            try:
                json.dumps(value)
                sanitized_error[key] = value
            except (TypeError, ValueError):
                # If not serializable, convert to string representation
                if key == "ctx" and isinstance(value, dict):
                    # Special handling for ctx dict - sanitize each sub-value
                    sanitized_ctx = {}
                    for ctx_key, ctx_value in value.items():
                        if isinstance(ctx_value, Exception):
                            sanitized_ctx[ctx_key] = str(ctx_value)
                        else:
                            try:
                                json.dumps(ctx_value)
                                sanitized_ctx[ctx_key] = ctx_value
                            except (TypeError, ValueError):
                                sanitized_ctx[ctx_key] = str(ctx_value)
                    sanitized_error[key] = sanitized_ctx
                else:
                    sanitized_error[key] = str(value)
        sanitized_errors.append(sanitized_error)

    return JSONResponse(
        status_code=422,
        content={"detail": sanitized_errors},
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> Any:
    """Global exception handler.

    For ``/v1/*`` paths (the SDK surface) we MUST return the stable
    ``{"error": {"code", "message"}}`` envelope -- anything else
    violates the contract documented in web/api/v1/errors.py and would
    confuse SDK clients that key off ``body.error.code``.

    For non-``/v1/*`` paths the original behavior is preserved: log
    the traceback and re-raise so FastAPI's default ``{"detail": ...}``
    handler runs (matching what /api/* callers and the web UI already
    expect).
    """
    import traceback

    logger.error(f"Unhandled exception in {request.url}: {exc}")
    logger.error(f"Traceback: {traceback.format_exc()}")

    if request.url.path.startswith("/api/a2a/"):
        logger.error("Unhandled A2A API error", exc_info=exc)
        return await a2a_api_error_handler(
            request,
            a2a_error(
                "internal",
                "Internal server error.",
                status_code=500,
            ),
        )

    if request.url.path.startswith("/v1/"):
        # Sanitize: never echo str(exc) -- it can leak SQL error
        # wording, table names, or storage backend identity. The full
        # traceback already went to the server log above.
        from fastapi.responses import JSONResponse

        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "internal_error",
                    "message": "Internal server error.",
                }
            },
        )

    # Non-/v1/* paths: original behavior unchanged. Re-raise so
    # FastAPI's default exception handling produces the {"detail": ...}
    # response the web UI / legacy clients depend on.
    raise exc


STORAGE_NAMESPACE_AUTHORITY_MESSAGE = "Storage namespace authority violation."


async def storage_namespace_authority_error_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    """Answer a storage namespace containment/authority fault once, app-wide.

    ``StorageKeyScopeError`` (a storage key outside the prefix its handle is
    bound to), ``ExecutionScopeAuthorityError`` (the scope resolver and a
    task's persisted snapshot disagree about the task's namespace) and
    ``ExecutionScopeResolverContractError`` (a resolver broke its return
    contract, or a persisted snapshot cannot be decoded at all) are permanent
    server-side configuration/authority faults. Registering them
    here instead of at each file endpoint keeps one classification for every
    route that touches durable storage -- and keeps them out of the retryable
    503 that ``DurableStorageOperationError`` maps to, since retrying a
    containment violation can never succeed.

    Status is 500: the fault is on the server, in its own namespace
    configuration or in a task's persisted scope, and there is nothing for the
    client to correct or retry. 503 would tell operators and clients to wait
    and retry a condition that is stable until someone changes the
    configuration, and no 4xx applies because the request itself is
    well-formed and authorized.

    The response body names the fault and nothing else. Scope segments,
    storage prefixes, and tenant identifiers can encode end-user identity, and
    ``str(exc)`` carries them; that detail belongs in the server-side log line
    below, which records the full traceback.

    ``/v1/*`` keeps the SDK's ``{"error": {"code", "message"}}`` envelope (see
    web/api/v1/errors.py) -- those endpoints reach durable storage while
    resolving a turn's attachments, and a bare ``{"detail": ...}`` there would
    break clients that switch on ``body.error.code``. Every other path gets
    FastAPI's default ``{"detail": ...}`` shape.

    Only HTTP scopes get a response. Starlette routes websocket exceptions
    through the same handler table, and websocket file attachments do reach
    durable storage, but a websocket connection cannot receive an HTTP
    response -- so those scopes re-raise and stay owned by the connection
    handler that established them.
    """
    logger.error(
        "Storage namespace authority violation for %s",
        request.url.path,
        exc_info=exc,
    )
    if request.scope.get("type") != "http":
        raise exc
    if request.url.path.startswith("/v1/"):
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": V1ErrorCode.INTERNAL_ERROR.value,
                    "message": STORAGE_NAMESPACE_AUTHORITY_MESSAGE,
                }
            },
        )
    return JSONResponse(
        status_code=500,
        content={"detail": STORAGE_NAMESPACE_AUTHORITY_MESSAGE},
    )


app.add_exception_handler(
    StorageKeyScopeError, storage_namespace_authority_error_handler
)
# Covers ExecutionScopeAbstentionMismatchError too: Starlette matches a
# handler by walking the raised exception's MRO.
app.add_exception_handler(
    ExecutionScopeAuthorityError, storage_namespace_authority_error_handler
)
# A resolver that broke its return contract, or a persisted snapshot that
# cannot be decoded, is the same permanent authority fault: nothing about the
# request can be corrected or retried into working.
app.add_exception_handler(
    ExecutionScopeResolverContractError, storage_namespace_authority_error_handler
)

# /v1/* SDK surface uses a stable {"error": {"code", "message"}} envelope
# distinct from FastAPI's default {"detail": "..."} shape used by /api/*.
# Typed V1ApiError raises pass through this handler so endpoints can
# choose their own HTTP status (401 / 404 / 409 / 429).
# See web/api/v1/errors.py for the contract.
app.add_exception_handler(V1ApiError, v1_api_error_handler)  # type: ignore[arg-type]
app.add_exception_handler(A2AApiError, a2a_api_error_handler)  # type: ignore[arg-type]
app.add_exception_handler(
    SkillRuntimeSessionBoundaryError,
    cast(Any, skill_runtime_session_boundary_error_handler),
)


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: "*" should not be used in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(SessionMiddleware, secret_key=get_session_secret(), same_site="lax")
app.add_middleware(FileStorageStartupSyncGateMiddleware)

current_dir = os.path.dirname(os.path.abspath(__file__))

# For static files
app.mount(
    "/uploads",
    StaticFiles(directory=str(uploads_dir)),
    name="uploads",
)

# memory management router with dynamic memory store
memory_router = MemoryManagementRouter(get_memory_store).get_router()

# API routers
app.include_router(auth_router)
app.include_router(chat_router)
app.include_router(cloud_router)
app.include_router(computer_router)
app.include_router(conversation_logs_router)
app.include_router(file_router)
app.include_router(jobs_router)
app.include_router(kb_router)
app.include_router(me_router)
app.include_router(personal_api_keys_router)
app.include_router(model_router)
app.include_router(ws_router)
app.include_router(monitor_router)
app.include_router(progress_ws_router)
app.include_router(memory_router)
app.include_router(mcp_router)
app.include_router(custom_api_router)
app.include_router(deployment_config_router)
app.include_router(tools_router)
app.include_router(admin_users_router)
app.include_router(admin_interaction_rollout_router)
app.include_router(admin_mcp_router)
app.include_router(skills_router)
app.include_router(skill_hub_router)
app.include_router(system_router)
app.include_router(templates_router)
app.include_router(agents_router)
app.include_router(agent_api_keys_router)
app.include_router(a2a_router)
app.include_router(triggers_router)
app.include_router(workforces_router)
app.include_router(channel_router, prefix="/api/channels", tags=["Channels"])
app.include_router(widget_router)
app.include_router(share_router)
# Public SDK surface, mounted under /v1. Auth via xag_* API key,
# error envelope {"error": {"code", "message"}}. See web/api/v1/.
app.include_router(v1_router)


# initial database and skill manager
@app.on_event("startup")
async def startup_event() -> None:
    global _migration_task
    logger.info("Agent runtime configured: %s", get_agent_runtime())
    validate_interaction_rollout_at_startup()
    with _startup_phase("database init"):
        init_db()

    # Keep built-in task-runtime providers scoped to the application lifespan.
    # Register even when disabled so task creation receives a precise 403
    # instead of an ambiguous "unknown extension" error.
    register_local_browser_runtime()

    # Reopen process-local task admission before any trigger, command, or
    # channel ingress can create background execution work for this lifespan.
    from .api.websocket import background_task_manager

    background_task_manager.start_accepting()

    start_file_storage_startup_sync_task(app)
    start_trigger_dispatcher_task(app)
    start_task_lease_recovery_task(app)
    start_uploaded_file_recovery_task(app)
    start_orphan_upload_gc_task(app)

    # Persisted ExecutionScope snapshots (workforce sub-tasks) keep a
    # sub-task scoped across process restarts. With no resolver registered
    # they are the sole answer; with one registered they are a
    # corroborating candidate (see
    # xagent.core.execution_scope.resolve_execution_scope).
    from ..core.execution_scope import execution_scope_resolver_registered
    from .services.execution_scope_snapshot import (
        register_execution_scope_snapshot_loader,
    )

    register_execution_scope_snapshot_loader()
    # This reflects only whether a resolver is registered at this point in
    # startup; it is not re-evaluated afterward. An embedder that registers
    # its resolver later (e.g. from its own startup hook running after this
    # one) leaves this line reporting "snapshot-only" even once a resolver
    # is in fact authoritative -- the mode itself is live and re-checked on
    # every resolution (see resolve_execution_scope), only this log line is
    # a startup-time snapshot of it.
    logger.info(
        "Execution scope authority mode at startup: %s",
        "resolver-authoritative (snapshot is a corroborating candidate)"
        if execution_scope_resolver_registered()
        else "snapshot-only (no resolver registered yet)",
    )

    from .services.trigger_rate_limit import warn_if_rate_limits_are_per_process

    warn_if_rate_limits_are_per_process()

    from .services.trigger_providers.gmail import (
        warn_if_gmail_oidc_verification_degraded,
        warn_if_gmail_watch_registration_degraded,
    )

    warn_if_gmail_oidc_verification_degraded()
    warn_if_gmail_watch_registration_degraded()

    initialize_langfuse()

    # Initialize skill manager
    from ..skills.utils import create_skill_manager

    skill_manager = create_skill_manager()
    with _startup_phase("skill manager init"):
        await skill_manager.initialize()
    app.state.skill_manager = skill_manager
    logger.info(
        f"Skill manager initialized with {len(await skill_manager.list_skills())} skills"
    )

    # Initialize template manager
    from ..templates.utils import create_template_manager

    template_manager = create_template_manager()
    with _startup_phase("template manager init"):
        await template_manager.initialize()
    app.state.template_manager = template_manager
    logger.info(
        f"Template manager initialized with {len(await template_manager.list_templates())} templates"
    )

    # Log memory store type (using dynamic manager)
    from .dynamic_memory_store import get_memory_store_manager

    manager = get_memory_store_manager()
    store_info = manager.get_store_info()

    if store_info["is_lancedb"]:
        logger.info("Using LanceDB memory store with vector search capabilities")
        logger.info(f"Embedding model ID: {store_info['embedding_model_id']}")
    else:
        logger.info("Using in-memory store (no vector search capabilities)")

    logger.info(
        f"Memory store similarity threshold: {store_info['similarity_threshold']}"
    )

    # Auto-migrate LanceDB tables if needed (for multi-tenancy support)
    # Controlled by LANCEDB_AUTO_MIGRATE environment variable (default: true)
    auto_migrate = os.getenv("LANCEDB_AUTO_MIGRATE", "true").lower() == "true"

    try:
        from ..core.tools.core.RAG_tools.LanceDB.schema_manager import (
            check_table_needs_migration,
        )
        from ..providers.vector_store.lancedb import get_connection_from_env

        conn = get_connection_from_env()

        # Check if any tables need migration
        needs_migration = False
        tables_to_check = [
            "chunks",
            "documents",
            "parses",
            "ingestion_runs",
            "prompt_templates",
        ]
        tables_need_migration_list = []

        for table_name in tables_to_check:
            if check_table_needs_migration(conn, table_name):
                logger.warning(
                    "Table '%s' needs migration (missing user_id field)",
                    table_name,
                )
                tables_need_migration_list.append(table_name)
                needs_migration = True

        # Check embeddings tables (use shared compat helper)
        if not needs_migration:
            try:
                from ..core.tools.core.RAG_tools.utils.lancedb_query_utils import (
                    list_embeddings_table_names,
                )

                for table_name in list_embeddings_table_names(conn):
                    if check_table_needs_migration(conn, table_name):
                        logger.warning(
                            "Table '%s' needs migration (missing user_id field)",
                            table_name,
                        )
                        tables_need_migration_list.append(table_name)
                        needs_migration = True
            except Exception as e:
                logger.warning("Could not check embeddings tables: %s", e)

        if needs_migration:
            if tables_need_migration_list:
                logger.warning(
                    "Tables requiring migration: %s",
                    ", ".join(tables_need_migration_list),
                )

            if auto_migrate:
                # Run migration in background to avoid blocking startup
                logger.info("=" * 60)
                logger.info("STARTING BACKGROUND LANCEDB MIGRATION")
                logger.info("=" * 60)
                logger.info(
                    "Tables requiring migration: %s",
                    ", ".join(tables_need_migration_list),
                )

                async def run_migration_background() -> None:
                    from ..migrations.lancedb.backfill_user_id import backfill_all

                    try:
                        result = await asyncio.to_thread(backfill_all, dry_run=False)
                        logger.info("=" * 60)
                        logger.info("BACKGROUND LANCEDB MIGRATION COMPLETED")
                        logger.info("=" * 60)
                        logger.info(
                            "Migration results: chunks=%s backfilled, embeddings=%s backfilled",
                            result.get("chunks", {}).get("backfilled", 0),
                            result.get("embeddings", {}).get("backfilled", 0),
                        )

                        # Log any skipped records
                        chunks_skipped = result.get("chunks", {}).get("skipped", 0)
                        embeddings_skipped = result.get("embeddings", {}).get(
                            "skipped", 0
                        )
                        if chunks_skipped > 0 or embeddings_skipped > 0:
                            logger.warning(
                                "Some records were skipped (no matching document): chunks=%s, embeddings=%s",
                                chunks_skipped,
                                embeddings_skipped,
                            )
                    except Exception as e:
                        logger.error("=" * 60)
                        logger.error("BACKGROUND LANCEDB MIGRATION FAILED")
                        logger.error("=" * 60)
                        logger.error("Error: %s", e, exc_info=True)
                        logger.warning(
                            "Some features may not work correctly. "
                            "Please run migration manually: python -m xagent.migrations.lancedb.backfill_user_id"
                        )

                # Start background task without awaiting, but keep a reference
                # so shutdown can cancel/await it gracefully.
                _migration_task = asyncio.create_task(run_migration_background())
            else:
                logger.warning(
                    "LANCEDB_AUTO_MIGRATE is disabled. "
                    "Migration will NOT run automatically. "
                    "To enable automatic migration, set LANCEDB_AUTO_MIGRATE=true. "
                    "To run migration manually: python -m xagent.migrations.lancedb.backfill_user_id"
                )
        else:
            logger.info("LanceDB tables are up to date, no migration needed")
    except Exception as e:
        logger.warning(
            "Could not check LanceDB migration status: %s. "
            "Application will continue, but some features may not work correctly.",
            e,
        )

    # Auto-fix file_id nullability and backfill documents table if needed
    # Controlled by LANCEDB_AUTO_MIGRATE environment variable (default: false)
    if auto_migrate:
        try:
            from ..providers.vector_store.lancedb import get_connection_from_env

            conn = get_connection_from_env()

            # Fix file_id nullability before any backfill (must run first since
            # the backfill reads the table and will crash if file_id is
            # non-nullable with null values)
            try:
                from ..migrations.lancedb.fix_file_id_nullable import (
                    fix_file_id_nullable,
                )

                fix_result = fix_file_id_nullable(dry_run=False, conn=conn)
                if fix_result.get("fixed"):
                    logger.info(
                        "Auto-fixed file_id column to nullable in documents table"
                    )
            except Exception as e:
                logger.warning("Could not fix file_id nullability: %s", e)

            # Check if documents table exists and needs backfill
            documents_table = None
            try:
                from ..core.tools.core.RAG_tools.LanceDB.schema_manager import (
                    _safe_close_table,
                )
                from ..core.tools.core.RAG_tools.utils.lancedb_query_utils import (
                    query_to_list,
                )

                documents_table = conn.open_table("documents")

                # Check for empty string file_id values
                empty_file_id_count = len(
                    query_to_list(
                        documents_table.search().where("file_id = ''").limit(1)
                    )
                )

                # Check for NULL user_id values
                null_user_id_count = len(
                    query_to_list(
                        documents_table.search().where("user_id IS NULL").limit(1)
                    )
                )

                if empty_file_id_count > 0 or null_user_id_count > 0:
                    logger.info("=" * 60)
                    logger.info("STARTING BACKGROUND DOCUMENTS TABLE BACKFILL")
                    logger.info("=" * 60)
                    if empty_file_id_count > 0:
                        logger.info("Found empty string file_id values to backfill")
                    if null_user_id_count > 0:
                        logger.info("Found NULL user_id values to backfill")

                    async def run_documents_backfill_background() -> None:
                        from ..migrations.lancedb.backfill_documents_file_id import (
                            backfill_all,
                        )

                        try:
                            result = await asyncio.to_thread(
                                backfill_all, dry_run=False, conn=conn
                            )
                            logger.info("=" * 60)
                            logger.info("DOCUMENTS TABLE BACKFILL COMPLETED")
                            logger.info("=" * 60)

                            file_id_result = result.get("file_id", {})
                            user_id_result = result.get("user_id", {})

                            if file_id_result.get("updated", 0) > 0:
                                logger.info(
                                    "file_id backfill: %d rows updated",
                                    file_id_result.get("updated", 0),
                                )
                            if user_id_result.get("updated", 0) > 0:
                                logger.info(
                                    "user_id backfill: %d rows updated",
                                    user_id_result.get("updated", 0),
                                )

                            if file_id_result.get("error"):
                                logger.warning(
                                    "file_id backfill error: %s",
                                    file_id_result.get("error"),
                                )
                            if user_id_result.get("error"):
                                logger.warning(
                                    "user_id backfill error: %s",
                                    user_id_result.get("error"),
                                )
                        except Exception as e:
                            logger.error("=" * 60)
                            logger.error("DOCUMENTS TABLE BACKFILL FAILED")
                            logger.error("=" * 60)
                            logger.error("Error: %s", e, exc_info=True)
                            logger.warning(
                                "Some features may not work correctly. "
                                "Please run backfill manually: python -m xagent.migrations.lancedb.backfill_documents_file_id"
                            )

                    # Start background task
                    _migration_task = asyncio.create_task(
                        run_documents_backfill_background()
                    )
                else:
                    logger.info("Documents table backfill not needed")
            except Exception as e:
                # Documents table might not exist yet
                logger.debug("Could not check documents table: %s", e)
            finally:
                _safe_close_table(documents_table)
        except Exception as e:
            logger.warning(
                "Could not check documents table backfill status: %s. "
                "Application will continue.",
                e,
            )

    # Periodic collection metadata rebuild to keep cache in sync
    async def run_metadata_rebuild_background() -> None:
        import os

        interval_hours = float(os.getenv("XAGENT_METADATA_REBUILD_INTERVAL_HOURS", "6"))
        interval_seconds = interval_hours * 3600
        while True:
            try:
                await asyncio.sleep(interval_seconds)
                from xagent.core.tools.core.RAG_tools.management.collection_manager import (
                    rebuild_collection_metadata,
                )

                await rebuild_collection_metadata()
                logger.info("Periodic collection metadata rebuild completed")
            except Exception as e:
                logger.warning("Collection metadata rebuild failed: %s", e)

    if not os.getenv("PYTEST_CURRENT_TEST"):
        app.state.metadata_rebuild_task = asyncio.create_task(
            run_metadata_rebuild_background()
        )
        logger.info(
            "Started background collection metadata rebuild task (interval=%sh)",
            os.getenv("XAGENT_METADATA_REBUILD_INTERVAL_HOURS", "6"),
        )
    else:
        logger.info("Skipping background metadata rebuild (test environment)")

    # Reconcile uploaded_files when auto migration is enabled.
    # Keep this under the same migration toggle for consistent startup behavior.
    # Run in background after app starts serving to avoid blocking startup.
    if auto_migrate:

        async def run_uploaded_file_reconcile_background() -> None:
            from .services.kb_file_service import reconcile_uploaded_files

            pending_migration_task = _migration_task
            if pending_migration_task and not pending_migration_task.done():
                try:
                    await pending_migration_task
                except Exception as e:  # noqa: BLE001
                    logger.warning(
                        "Documents table backfill did not complete before uploaded files reconcile: %s",
                        e,
                    )

            try:
                from ..migrations.lancedb.backfill_uploaded_file_links import (
                    backfill_all as backfill_uploaded_file_links,
                )

                link_result = await asyncio.to_thread(
                    backfill_uploaded_file_links, dry_run=False
                )
                logger.info("Uploaded file links backfill completed: %s", link_result)
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "Uploaded file links backfill failed before reconcile: %s",
                    e,
                    exc_info=True,
                )

            def _run_uploaded_file_reconcile() -> None:
                from .models.database import get_session_local

                session_local = get_session_local()
                db = session_local()
                try:
                    result = reconcile_uploaded_files(
                        db,
                        user_id=-1,
                        is_admin=True,
                        stale_ttl_hours=24 * 7,
                        delete_stale=False,
                    )
                    logger.info("Uploaded files reconcile completed: %s", result)
                finally:
                    db.close()

            try:
                await asyncio.to_thread(_run_uploaded_file_reconcile)
            except Exception as e:  # noqa: BLE001
                logger.warning("Uploaded files reconcile failed: %s", e)

        # Start background task without awaiting
        asyncio.create_task(run_uploaded_file_reconcile_background())
        logger.info("Started background uploaded files reconcile task")

        # Clean up orphaned temporary files from interrupted atomic replacements.
        # This walks the entire uploads tree inline during startup and can take
        # minutes on a large tree with no log output in between. Wrap it in a
        # startup phase so its duration is visible and a slow start is easy to
        # diagnose from the logs alone.
        try:
            from .api.kb import cleanup_orphaned_temp_files

            def _run_temp_file_cleanup() -> int:
                return cleanup_orphaned_temp_files()

            with _startup_phase("orphaned temp-file cleanup"):
                cleaned_count = await asyncio.to_thread(_run_temp_file_cleanup)
            if cleaned_count > 0:
                logger.info(
                    "Startup cleanup: removed %d orphaned temporary file(s)",
                    cleaned_count,
                )
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "Temporary file cleanup skipped due to error: %s",
                e,
            )

    # Warmup sandbox manager
    from .sandbox_manager import check_sandbox_static_readiness, get_sandbox_manager

    # WHY: the getter can construct the backend service and inventory
    # containers on a cold process, so a hang here needs its own phase line.
    with _startup_phase("sandbox manager init"):
        sandbox_mgr = get_sandbox_manager()
    if sandbox_mgr:
        # Readiness runs before cleanup/warmup and is deliberately not
        # wrapped in try/except: a static SANDBOX_VOLUMES/code-mount/
        # external-upload-dir conflict must fail startup outright rather
        # than surface later as a per-task SandboxRuntimeConflictError.
        # This also resolves and caches the backend-capability probe as a
        # side effect, so cleanup() below reads the cached value instead of
        # resolving it again.
        # cleanup() quiesce can be awaited inline for minutes with no logs.
        # Time each sub-phase so the next slow start names the exact one; the
        # quiesce summary breaks it down further.
        with _startup_phase("sandbox static readiness"):
            await check_sandbox_static_readiness(sandbox_mgr)
        with _startup_phase("sandbox cleanup"):
            await sandbox_mgr.cleanup()
        with _startup_phase("sandbox warmup"):
            await sandbox_mgr.warmup()
        logger.info("Sandbox manager initialized and warmed up")

        from ..config import get_sandbox_idle_ttl

        if get_sandbox_idle_ttl() is not None:
            global _sandbox_idle_sweep_task
            _sandbox_idle_sweep_task = asyncio.create_task(
                sandbox_mgr.run_idle_sweep_loop()
            )
            logger.info("Started sandbox idle sweep task")
    else:
        logger.info("Sandbox manager not available (disabled or init failed)")

    # Recover accepted-but-unfinished task commands only after the runtime,
    # skill/template managers, tracing, and sandbox services are ready.
    from .api.websocket import execute_durable_task_command
    from .services.task_command_transport import start_task_command_dispatcher

    global _task_command_dispatcher_task
    _task_command_dispatcher_task = start_task_command_dispatcher(
        execute_durable_task_command
    )
    app.state.task_command_dispatcher_task = _task_command_dispatcher_task
    logger.info("Started durable task command dispatcher")

    # Start configured chat channels.
    try:
        from .channels.feishu.bot import get_feishu_channel
        from .channels.slack.bot import get_slack_channel
        from .channels.telegram.bot import get_telegram_channel

        telegram_channel = get_telegram_channel()
        if telegram_channel.enabled:
            logger.info("Initializing Telegram channel manager...")
            app.state.telegram_task = asyncio.create_task(telegram_channel.start())
            logger.info("Telegram channel background task created successfully")

        feishu_channel = get_feishu_channel()
        if feishu_channel.enabled:
            logger.info("Initializing Feishu channel manager...")
            app.state.feishu_task = asyncio.create_task(feishu_channel.start())
            logger.info("Feishu channel background task created successfully")

        slack_channel = get_slack_channel()
        if slack_channel.enabled:
            logger.info("Initializing Slack channel manager...")
            app.state.slack_task = asyncio.create_task(slack_channel.start())
            logger.info("Slack channel background task created successfully")
    except Exception as e:
        logger.error(f"Failed to start chat channel managers: {e}", exc_info=True)


@app.on_event("shutdown")
async def shutdown_event() -> None:
    global \
        _file_storage_startup_sync_task, \
        _migration_task, \
        _task_command_dispatcher_task, \
        _trigger_dispatcher_task, \
        _sandbox_idle_sweep_task

    flush_langfuse()

    if _task_command_dispatcher_task is not None:
        from .services.task_command_transport import stop_task_command_dispatcher

        await stop_task_command_dispatcher()
    _task_command_dispatcher_task = None

    await stop_orphan_upload_gc_task(app)
    await stop_uploaded_file_recovery_task(app)
    await stop_task_lease_recovery_task(app)

    if _sandbox_idle_sweep_task and not _sandbox_idle_sweep_task.done():
        logger.info("Cancelling sandbox idle sweep task...")
        _sandbox_idle_sweep_task.cancel()
        with suppress(asyncio.CancelledError):
            await _sandbox_idle_sweep_task
    _sandbox_idle_sweep_task = None

    if _file_storage_startup_sync_task and not _file_storage_startup_sync_task.done():
        logger.info("Cancelling background startup file storage sync task...")
        _file_storage_startup_sync_task.cancel()
        with suppress(asyncio.CancelledError):
            await _file_storage_startup_sync_task
    _file_storage_startup_sync_task = None

    if _trigger_dispatcher_task and not _trigger_dispatcher_task.done():
        logger.info("Cancelling trigger dispatcher task...")
        _trigger_dispatcher_task.cancel()
        with suppress(asyncio.CancelledError):
            await _trigger_dispatcher_task
    _trigger_dispatcher_task = None

    if _migration_task and not _migration_task.done():
        logger.info("Cancelling background LanceDB migration task...")
        _migration_task.cancel()
        with suppress(asyncio.CancelledError):
            await _migration_task
    _migration_task = None

    # Cancel metadata rebuild background task
    if hasattr(app.state, "metadata_rebuild_task"):
        task = app.state.metadata_rebuild_task
        if task and not task.done():
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

    # Shutdown chat channels before draining task finalizers.
    try:
        if hasattr(app.state, "telegram_task"):
            app.state.telegram_task.cancel()
            logger.info("Cancelled Telegram polling task")
        if hasattr(app.state, "slack_task"):
            app.state.slack_task.cancel()
            logger.info("Cancelled Slack manager task")

        from .channels.feishu.bot import get_feishu_channel
        from .channels.slack.bot import get_slack_channel
        from .channels.telegram.bot import get_telegram_channel

        telegram_channel = get_telegram_channel()
        if telegram_channel.enabled:
            await telegram_channel.stop()
            logger.info("Telegram channel stopped successfully")

        feishu_channel = get_feishu_channel()
        await feishu_channel.stop()

        slack_channel = get_slack_channel()
        await slack_channel.stop()
    except Exception as e:
        logger.error("Failed to stop chat channels: %s", e, exc_info=True)

    # All producers are stopped. Drain task-owned finalizers and their shared
    # lease heartbeats before tearing down the sandboxes those tasks may use.
    from .api.websocket import background_task_manager
    from .services.task_lease_service import wait_for_heartbeat_manager_idle

    await background_task_manager.shutdown()
    await wait_for_heartbeat_manager_idle()

    from .services.task_runtime import shutdown_task_runtime_hook_executor

    shutdown_task_runtime_hook_executor()
    unregister_local_browser_runtime()

    # Shutdown all sandboxes
    from .sandbox_manager import get_sandbox_manager

    sandbox_mgr = get_sandbox_manager()
    if sandbox_mgr:
        await sandbox_mgr.cleanup()


from ..config import get_frontend_dist_dir  # noqa: E402

# Serve the built frontend static export (single-process / pip deployment).
# Registered last so the SPA catch-all only receives paths unmatched by the API
# routers above. If the export is absent (e.g. the multi-container Docker
# deployment where Next.js serves the frontend), the backend stays API-only.
from .frontend_static import mount_frontend  # noqa: E402

mount_frontend(app, get_frontend_dist_dir())


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
