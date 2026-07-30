import logging
from typing import Any, Generator

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.engine import Connection
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import NullPool, QueuePool

from ...config import get_database_url, get_db_pool_kwargs
from ...db.sqlite import (
    apply_sqlite_concurrency_pragmas,
    ensure_sqlite_parent_directory,
)

_SessionLocal: sessionmaker[Session] | None = None

_engine: Engine | None = None
logger = logging.getLogger(__name__)

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


# ``new``/``dirty``/``deleted`` alone cannot prove a transaction is safe to
# roll back: an ORM flush empties them while the DML is still uncommitted,
# and Core/bulk DML through ``Session.execute()`` never touches them at all.
# Track "this transaction may have written" on every Session via class-level
# listeners:
#
#   - ``after_flush`` — ORM unit-of-work writes;
#   - ``do_orm_execute`` — every ``Session.execute()`` statement. Anything
#     that is not provably a SELECT (Core insert/update/delete, ``text()``
#     of any kind) sets the flag, deliberately conservative: an unrecognized
#     statement keeps the connection rather than risking a rollback of work.
#
# The flag lives in ``session.info`` and is cleared once the transaction
# actually ends. ``Session.close()`` without commit/rollback may leave it
# set — that fails closed (the helper just keeps the connection), never
# toward data loss. Out of contract: DML issued directly on
# ``session.connection()`` bypasses session events; don't do that on
# sessions handed to this helper.
_MAY_HAVE_WRITTEN_KEY = "xagent_txn_may_have_written"


@event.listens_for(Session, "after_flush")
def _mark_session_flushed(session: Session, _flush_context: Any) -> None:
    session.info[_MAY_HAVE_WRITTEN_KEY] = True


@event.listens_for(Session, "do_orm_execute")
def _mark_session_statement(orm_execute_state: Any) -> None:
    if not orm_execute_state.is_select:
        orm_execute_state.session.info[_MAY_HAVE_WRITTEN_KEY] = True


@event.listens_for(Session, "after_transaction_end")
def _clear_session_flushed(session: Session, transaction: Any) -> None:
    # Clear only when the ROOT transaction ends (commit, rollback, or
    # close). ``after_commit``/``after_rollback`` would fire for savepoint
    # (begin_nested) completion too, while the outer transaction — and its
    # uncommitted writes — is still open; clearing there would let the
    # helper roll those writes away.
    if transaction.parent is None:
        session.info[_MAY_HAVE_WRITTEN_KEY] = False


def release_db_connection_if_clean(db: Session | None) -> bool:
    """Return ``db``'s pooled connection by ending its (read-only) transaction.

    SQLAlchemy sessions hold their connection until commit/rollback/close, so
    a session that ran a SELECT and then awaits slow non-DB work (remote MCP
    initialization, sandbox startup, the agent run itself) pins a pool slot
    in ``idle in transaction`` for the whole wait (issue #889). Calling this
    before such an await releases the connection; the session stays usable
    and transparently re-acquires a connection on its next query.

    Only rolls back when the session has no pending ORM changes
    (new/dirty/deleted) and the current transaction executed nothing but
    SELECTs through the session (see ``_MAY_HAVE_WRITTEN_KEY`` above), so
    nothing uncommitted is ever discarded. Callers must not rely on it
    having released (hence the bool return): a session that may have
    written keeps its connection.
    """
    if db is None:
        return False
    try:
        if db.new or db.dirty or db.deleted:
            return False
        if db.info.get(_MAY_HAVE_WRITTEN_KEY, False):
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


def get_optional_session_local() -> sessionmaker[Session] | None:
    """Return the installed session factory without requiring database setup."""
    return _SessionLocal


def get_engine() -> Engine:
    if _engine is None:
        raise RuntimeError("Engine is not initialized. Call init_db() first.")
    return _engine


def _initialize_database_schema(engine: Engine) -> list[dict[str, Any]]:
    """Migrate, create, and seed the schema under one startup ownership lock."""

    from ...db.migration import (
        database_startup_lock,
        is_database_empty,
        try_upgrade_db,
    )
    from ..builtin_mcp_registry import (
        seed_builtin_oauth_and_public_mcp_apps,
        validate_builtin_public_mcp_apps,
    )

    with database_startup_lock(engine) as locked_connection:
        startup_bind: Engine | Connection = (
            locked_connection if locked_connection is not None else engine
        )
        should_seed_builtin_mcp_registry = is_database_empty(startup_bind)
        try_upgrade_db(
            engine,
            locked_connection=locked_connection,
        )
        Base.metadata.create_all(bind=startup_bind)

        if locked_connection is not None:
            if should_seed_builtin_mcp_registry:
                seed_builtin_oauth_and_public_mcp_apps(locked_connection)
            return validate_builtin_public_mcp_apps(locked_connection)

        with engine.begin() as connection:
            if should_seed_builtin_mcp_registry:
                seed_builtin_oauth_and_public_mcp_apps(connection)
            return validate_builtin_public_mcp_apps(connection)


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
        PublicMCPAppAudit,
        SystemSetting,
        Task,
        TaskChatMessage,
        TaskConnectorRuntimeContext,
        TaskExecutionCommand,
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
        database_url = ensure_sqlite_parent_directory(database_url)
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
            **get_db_pool_kwargs(),
        )

    # Create session factory
    _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_engine)

    builtin_mcp_mismatches = _initialize_database_schema(_engine)

    for mismatch in builtin_mcp_mismatches:
        logger.warning(
            "Built-in MCP catalog drift detected: app_id=%s "
            "mismatched_fields=%s canonical_hash=%s persisted_hash=%s",
            mismatch["app_id"],
            ",".join(mismatch["mismatched_fields"]),
            mismatch["canonical_hash"],
            mismatch["persisted_hash"],
        )

    logger.info("Database initialized. Waiting for first admin setup.")
