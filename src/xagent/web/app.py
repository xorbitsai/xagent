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

from ..config import get_uploads_dir
from ..core.tracing.langfuse import flush_langfuse, initialize_langfuse
from .api.admin_mcp import admin_mcp_router
from .api.admin_users import router as admin_users_router
from .api.agents import router as agents_router
from .api.auth import auth_router
from .api.channel import router as channel_router
from .api.chat import chat_router
from .api.cloud_storage import cloud_router
from .api.custom_api import custom_api_router
from .api.files import file_router
from .api.kb import kb_router
from .api.mcp import mcp_router
from .api.memory import MemoryManagementRouter
from .api.model import model_router
from .api.monitor import monitor_router
from .api.progress_ws import progress_ws_router
from .api.skills import router as skills_router
from .api.system import system_router
from .api.templates import router as templates_router
from .api.tools import tools_router
from .api.v1 import v1_router
from .api.v1.errors import V1ApiError, v1_api_error_handler
from .api.websocket import ws_router
from .api.widget import widget_router
from .dynamic_memory_store import get_memory_store
from .logging_config import setup_logging
from .models.database import init_db
from .services.rag_storage_migration_service import RAGStorageMigrationService

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


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint for container probes."""
    return {"status": "ok"}


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


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: "*" should not be used in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
app.include_router(file_router)
app.include_router(kb_router)
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
app.include_router(system_router)
app.include_router(templates_router)
app.include_router(agents_router)
app.include_router(channel_router, prefix="/api/channels", tags=["Channels"])
app.include_router(widget_router)
# Public SDK surface, mounted under /v1. Auth via xag_* API key,
# error envelope {"error": {"code", "message"}}. See web/api/v1/.
app.include_router(v1_router)


# initial database and skill manager
@app.on_event("startup")
async def startup_event() -> None:
    global _migration_task
    logger.info("Agent runtime configured: v2")
    logger.info("Initializing database...")
    init_db()
    logger.info("Database initialized successfully")

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

    auto_migrate = os.getenv("LANCEDB_AUTO_MIGRATE", "true").lower() == "true"
    migration_service = RAGStorageMigrationService()
    _migration_task = await migration_service.start_background_migrations()

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
    else:
        logger.info("Sandbox manager not available (disabled or init failed)")

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
    global _migration_task

    flush_langfuse()

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


# Frontend is now served by Next.js at http://localhost:3000
# This backend only provides API endpoints


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
