import logging
from typing import Any, Generator

from sqlalchemy import Engine, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import NullPool, QueuePool

from ...config import (
    get_database_url,
    get_db_max_overflow,
    get_db_pool_size,
    get_db_pool_timeout_seconds,
)
from ...db.sqlite import apply_sqlite_concurrency_pragmas

_SessionLocal: sessionmaker[Session] | None = None

_engine: Engine | None = None

# Create base model class
# Mypy workaround: explicitly type Base as Any to avoid "variable not valid as type" error
Base: Any = declarative_base()


def get_db() -> Generator[Session, None, None]:
    """Get database session"""
    if _SessionLocal is None:
        raise RuntimeError("Session Local is not initialized. Call init_db() first.")
    db = _SessionLocal()
    try:
        yield db
    finally:
        db.close()


def release_db_connection_if_clean(db: Session | None) -> bool:
    """Return ``db``'s pooled connection by ending its (read-only) transaction.

    SQLAlchemy sessions hold their connection until commit/rollback/close, so
    a session that ran a SELECT and then awaits slow non-DB work (remote MCP
    initialization, sandbox startup, the agent run itself) pins a pool slot
    in ``idle in transaction`` for the whole wait (issue #889). Calling this
    before such an await releases the connection; the session stays usable
    and transparently re-acquires a connection on its next query.

    Only rolls back when the session has no pending ORM changes
    (new/dirty/deleted), so nothing uncommitted is ever discarded. Callers
    must not rely on it having released (hence the bool return): a session
    with pending writes keeps its connection.
    """
    if db is None:
        return False
    try:
        if db.new or db.dirty or db.deleted:
            return False
        if not db.in_transaction():
            return True
        db.rollback()
        return True
    except Exception:
        logging.getLogger(__name__).debug(
            "Failed to release DB connection", exc_info=True
        )
        return False


def get_session_local() -> sessionmaker[Session]:
    if _SessionLocal is None:
        raise RuntimeError("Session Local is not initialized. Call init_db() first.")
    return _SessionLocal


def get_engine() -> Engine:
    if _engine is None:
        raise RuntimeError("Engine is not initialized. Call init_db() first.")
    return _engine


def init_db(db_url: str | None = None) -> None:
    """Initialize database, create all tables and default users"""
    # Import all models to ensure they are registered with Base.metadata
    from . import (  # noqa: F401
        BackgroundJob,
        GmailWatchState,
        KBIngestTarget,
        MCPServer,
        Model,
        OAuthProvider,
        OidcConsumedToken,
        PublicMCPApp,
        SystemSetting,
        Task,
        TaskChatMessage,
        TaskConnectorRuntimeContext,
        TemplateStats,
        ToolConfig,
        ToolUsage,
        UploadedFile,
        User,
        UserApiKey,
        UserDefaultModel,
        UserIdentity,
        UserModel,
        UserSkill,
        UserSkillFile,
        UserTemplateRelation,
        Workforce,
        WorkforceAgent,
        WorkforceBuilderMessage,
        WorkforceRun,
    )
    from .agent import Agent  # noqa: F401
    from .sandbox import SandboxInfo, SandboxSnapshot  # noqa: F401

    global _SessionLocal
    global _engine

    # Database configuration
    if db_url is not None:
        database_url = db_url
    else:
        database_url = get_database_url()

    # Create database engine
    # For SQLite, use NullPool to prevent connection pool issues
    # For other databases, use QueuePool with timeout settings
    if "sqlite" in database_url:
        _engine = create_engine(
            database_url,
            connect_args={"check_same_thread": False},
            poolclass=NullPool,  # SQLite doesn't need connection pooling
        )
        # WAL + busy_timeout so concurrent writes (e.g. concurrent tool
        # execution) wait for the lock instead of failing with "database is
        # locked".
        apply_sqlite_concurrency_pragmas(_engine)
    else:
        _engine = create_engine(
            database_url,
            poolclass=QueuePool,
            pool_size=get_db_pool_size(),
            max_overflow=get_db_max_overflow(),
            pool_timeout=get_db_pool_timeout_seconds(),
            pool_recycle=3600,  # Recycle connections after 1 hour
            pool_pre_ping=True,  # Verify connections before using
        )

    # Create session factory
    _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_engine)

    # Try upgrade db to head first
    from ...db.migration import is_database_empty, try_upgrade_db

    should_seed_builtin_mcp_registry = is_database_empty(_engine)

    try_upgrade_db(_engine)

    # Create all tables
    Base.metadata.create_all(bind=_engine)

    if should_seed_builtin_mcp_registry:
        from ..builtin_mcp_registry import seed_builtin_oauth_and_public_mcp_apps

        with _engine.begin() as conn:
            seed_builtin_oauth_and_public_mcp_apps(conn)

    logger = logging.getLogger(__name__)
    logger.info("Database initialized. Waiting for first admin setup.")
