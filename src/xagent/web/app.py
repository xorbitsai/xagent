import asyncio
import logging
import os
from contextlib import suppress

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from ..config import get_uploads_dir
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
from .api.text2sql import text2sql_router
from .api.tools import tools_router
from .api.websocket import ws_router
from .api.widget import widget_router
from .dynamic_memory_store import get_memory_store
from .logging_config import setup_logging
from .models.database import init_db

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
    """Handle request validation errors, especially those containing binary data"""
    import traceback

    logger.error(f"Validation error in {request.url}: {str(exc)}")
    logger.error(f"Traceback: {traceback.format_exc()}")

    # Sanitize error details to remove binary data and non-serializable objects
    sanitized_errors = []
    for error in exc.errors():
        sanitized_error = {}
        for key, value in error.items():
            # Try to serialize each value to check if it's JSON-serializable
            try:
                import json

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
async def global_exception_handler(request: Request, exc: Exception) -> None:
    """Global exception handler, ensuring all errors are recorded"""
    import traceback

    logger.error(f"Unhandled exception in {request.url}: {str(exc)}")
    logger.error(f"Traceback: {traceback.format_exc()}")
    # Re-raise the exception, let FastAPI handle it
    raise exc


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
app.include_router(text2sql_router)
app.include_router(tools_router)
app.include_router(admin_users_router)
app.include_router(admin_mcp_router)
app.include_router(skills_router)
app.include_router(system_router)
app.include_router(templates_router)
app.include_router(agents_router)
app.include_router(channel_router, prefix="/api/channels", tags=["Channels"])
app.include_router(widget_router)


# initial database and skill manager
@app.on_event("startup")
async def startup_event() -> None:
    global _migration_task
    logger.info("Initializing database...")
    init_db()
    logger.info("Database initialized successfully")

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

    if _migration_task and not _migration_task.done():
        logger.info("Cancelling background LanceDB migration task...")
        _migration_task.cancel()
        with suppress(asyncio.CancelledError):
            await _migration_task
    _migration_task = None

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
