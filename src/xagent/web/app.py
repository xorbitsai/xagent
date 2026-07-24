import asyncio
import json
import logging
import os
from contextlib import suppress
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware
from starlette.types import ASGIApp, Receive, Scope, Send

from ..config import (
    get_agent_runtime,
    get_file_storage_startup_sync_enabled,
    get_gmail_watch_enabled,
    get_gmail_watch_renewal_interval_seconds,
    get_session_secret,
    get_trigger_dispatcher_batch_size,
    get_trigger_dispatcher_enabled,
    get_trigger_dispatcher_interval_seconds,
    get_uploads_dir,
)
from ..core.tracing.langfuse import flush_langfuse, initialize_langfuse
from .api.a2a import router as a2a_router
from .api.admin_mcp import admin_mcp_router
from .api.admin_users import router as admin_users_router
from .api.agent_api_keys import router as agent_api_keys_router
from .api.agents import router as agents_router
from .api.auth import auth_router
from .api.channel import router as channel_router
from .api.chat import chat_router
from .api.cloud_storage import cloud_router
from .api.conversation_logs import router as conversation_logs_router
from .api.custom_api import custom_api_router
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
from .api.v1.errors import V1ApiError, v1_api_error_handler
from .api.websocket import ws_router
from .api.widget import widget_router
from .api.workforces import router as workforces_router
from .dynamic_memory_store import get_memory_store
from .logging_config import setup_logging
from .models.database import init_db
from .services.a2a_protocol import A2AApiError, a2a_api_error_handler, a2a_error

# Configure logging when running under gunicorn/uwsgi (no __main__.py)
setup_logging()  # Uses XAGENT_LOG_LEVEL env var or defaults to INFO

logger = logging.getLogger(__name__)


__all__ = ["app"]


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

    from .models.database import get_session_local
    from .services.startup_file_storage_sync import (
        sync_registered_files_to_durable_storage,
    )

    SessionLocal = get_session_local()
    db = SessionLocal()
    try:
        result = sync_registered_files_to_durable_storage(db)
        if result.failed:
            raise RuntimeError(
                "Startup file storage sync failed for "
                f"{result.failed} registered file(s)"
            )
    finally:
        db.close()


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
    """Readiness check that reflects startup file storage sync status."""
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


# /v1/* SDK surface uses a stable {"error": {"code", "message"}} envelope
# distinct from FastAPI's default {"detail": "..."} shape used by /api/*.
# Typed V1ApiError raises pass through this handler so endpoints can
# choose their own HTTP status (401 / 404 / 409 / 429).
# See web/api/v1/errors.py for the contract.
app.add_exception_handler(V1ApiError, v1_api_error_handler)  # type: ignore[arg-type]
app.add_exception_handler(A2AApiError, a2a_api_error_handler)  # type: ignore[arg-type]


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
app.include_router(tools_router)
app.include_router(admin_users_router)
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
    logger.info("Initializing database...")
    init_db()
    logger.info("Database initialized successfully")
    start_file_storage_startup_sync_task(app)
    start_trigger_dispatcher_task(app)

    # Persisted ExecutionScope snapshots (workforce sub-tasks) are preferred
    # over the embedder's resolver during per-task scope resolution.
    from .services.execution_scope_snapshot import (
        register_execution_scope_snapshot_loader,
    )

    register_execution_scope_snapshot_loader()

    from .services.trigger_rate_limit import warn_if_rate_limits_are_per_process

    warn_if_rate_limits_are_per_process()

    from .services.trigger_providers.gmail import (
        warn_if_gmail_oidc_verification_degraded,
    )

    warn_if_gmail_oidc_verification_degraded()

    initialize_langfuse()

    # Initialize skill manager
    from ..skills.utils import create_skill_manager

    skill_manager = create_skill_manager()
    await skill_manager.initialize()
    app.state.skill_manager = skill_manager
    logger.info(
        f"Skill manager initialized with {len(await skill_manager.list_skills())} skills"
    )

    # Initialize template manager
    from ..templates.utils import create_template_manager

    template_manager = create_template_manager()
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

        # Clean up orphaned temporary files from interrupted atomic replacements
        try:
            from .api.kb import cleanup_orphaned_temp_files

            def _run_temp_file_cleanup() -> int:
                return cleanup_orphaned_temp_files()

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
    from .sandbox_manager import get_sandbox_manager

    sandbox_mgr = get_sandbox_manager()
    if sandbox_mgr:
        await sandbox_mgr.cleanup()
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

    # Start Telegram and FeiShu channels if enabled
    try:
        from .channels.feishu.bot import get_feishu_channel
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
    except Exception as e:
        logger.error(f"Failed to start Telegram channel manager: {e}", exc_info=True)


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

    # Shutdown Telegram channel if enabled
    try:
        if hasattr(app.state, "telegram_task"):
            app.state.telegram_task.cancel()
            logger.info("Cancelled Telegram polling task")

        from .channels.feishu.bot import get_feishu_channel
        from .channels.telegram.bot import get_telegram_channel

        telegram_channel = get_telegram_channel()
        if telegram_channel.enabled:
            await telegram_channel.stop()
            logger.info("Telegram channel stopped successfully")

        feishu_channel = get_feishu_channel()
        await feishu_channel.stop()
    except Exception as e:
        logger.error("Failed to stop Telegram channel: %s", e, exc_info=True)

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
