import logging
import os

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .api.admin_users import router as admin_users_router
from .api.agents import router as agents_router
from .api.auth import auth_router
from .api.channel import router as channel_router
from .api.chat import chat_router
from .api.cloud_storage import cloud_router
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
from .config import UPLOADS_DIR
from .dynamic_memory_store import get_memory_store
from .logging_config import setup_logging
from .models.database import init_db

# Configure logging when running under gunicorn/uwsgi (no __main__.py)
setup_logging()  # Uses XAGENT_LOG_LEVEL env var or defaults to INFO

logger = logging.getLogger(__name__)

# 导出全局 memory store 供其他模块使用
__all__ = ["app"]

# 创建 FastAPI 应用
app = FastAPI(
    title="xagent", description="The Agent Operating System", redirect_slashes=False
)


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint for container probes."""
    return {"status": "ok"}


# 添加全局异常处理器
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
    """全局异常处理器，确保所有错误都被记录"""
    import traceback

    logger.error(f"Unhandled exception in {request.url}: {str(exc)}")
    logger.error(f"Traceback: {traceback.format_exc()}")
    # 重新抛出异常，让FastAPI默认处理
    raise exc


# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应该限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 获取当前目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 配置静态文件
app.mount(
    "/uploads",
    StaticFiles(directory=str(UPLOADS_DIR)),
    name="uploads",
)

# 创建 memory management router with dynamic memory store
memory_router = MemoryManagementRouter(get_memory_store).get_router()

# 注册API路由
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
app.include_router(text2sql_router)
app.include_router(tools_router)
app.include_router(admin_users_router)
app.include_router(skills_router)
app.include_router(system_router)
app.include_router(templates_router)
app.include_router(agents_router)
app.include_router(channel_router, prefix="/api/channels", tags=["Channels"])


# 初始化数据库
@app.on_event("startup")
async def startup_event() -> None:
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

    # Warmup sandbox manager
    from .sandbox_manager import get_sandbox_manager

    sandbox_mgr = get_sandbox_manager()
    if sandbox_mgr:
        await sandbox_mgr.cleanup()
        await sandbox_mgr.warmup()
        logger.info("Sandbox manager initialized and warmed up")
    else:
        logger.info("Sandbox manager not available (disabled or init failed)")

    # Start Telegram channel if enabled
    try:
        import asyncio

        from .channels.feishu.bot import get_feishu_channel
        from .channels.telegram.bot import get_telegram_channel

        telegram_channel = get_telegram_channel()
        if telegram_channel.enabled:
            logger.info("Initializing Telegram channel manager...")
            app.state.telegram_task = asyncio.create_task(telegram_channel.start())
            logger.info("Telegram channel background task created successfully")

        feishu_channel = get_feishu_channel()
        app.state.feishu_task = asyncio.create_task(feishu_channel.start())
    except Exception as e:
        logger.error(f"Failed to start Telegram channel manager: {e}", exc_info=True)


@app.on_event("shutdown")
async def shutdown_event() -> None:
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
        logger.error(f"Failed to stop Telegram channel: {e}", exc_info=True)

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
