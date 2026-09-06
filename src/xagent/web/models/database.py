import logging
from pathlib import Path
from typing import Any, Generator

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.engine import Connection, make_url
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


def _sqlite_read_only_url(database_url: str) -> str:
    """Return a SQLite URI that refuses to create or modify its database.

    SQLite's default file connection creates a missing database before the
    first query. Audit commands need URI ``mode=ro`` semantics so a typo or
    stale path fails without leaving an empty artifact. In-memory databases
    have no filesystem state and therefore retain their original URL.
    """
    url = make_url(database_url)
    database = url.database
    if not database or database == ":memory:" or url.query.get("mode") == "memory":
        return database_url
    uri_database = (
        database
        if database.startswith("file:")
        else f"file:{Path(database).expanduser().resolve().as_posix()}"
    )
    return str(
        url.set(database=uri_database).update_query_dict({"mode": "ro", "uri": "true"})
    )


def get_db() -> Generator[Session, None, None]:
    """Get database session"""
    if _SessionLocal is None:
        raise RuntimeError(
            "Session Local is not initialized. Call configure_db() or init_db() first."
        )
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


# Hooks installed through ``services/connector_team_scope`` run on the
# endpoint's own live session, mid-request, often with row locks already
# held. A hook that ends that transaction releases those locks without the
# endpoint finding out. Counting how many times the root transaction has
# ended is what lets the seam notice: the count moving across a hook call
# is the whole signal.
#
# Same ``transaction.parent is None`` test, and for the same reason, as
# ``_clear_session_flushed`` directly above -- ``after_commit`` and
# ``after_rollback`` fire for savepoint completion too, and a savepoint
# completing is not the caller's transaction ending. Separate listener
# rather than an addition to that one: clearing the write flag and
# counting transaction ends are two jobs, and a session's write-flag
# behavior must not change because something else started counting.
_ROOT_TXN_END_COUNT_KEY = "xagent_root_txn_end_count"


@event.listens_for(Session, "after_transaction_end")
def _count_root_transaction_end(session: Session, transaction: Any) -> None:
    if transaction.parent is None:
        current = session.info.get(_ROOT_TXN_END_COUNT_KEY, 0)
        # The first listener in this module that reads a value back before
        # writing it, and ``session.info`` is reachable from an installed
        # hook. Without this, a non-count value left there would raise from
        # inside a transaction-end event -- including the one ``commit()``
        # fires, after the write is already durable. ``bool`` is excluded
        # because ``int(True)`` would otherwise pass as a count of 1;
        # ``root_transaction_end_count`` rejects it with the same predicate.
        if not isinstance(current, int) or isinstance(current, bool):
            current = 0
        session.info[_ROOT_TXN_END_COUNT_KEY] = current + 1


def root_transaction_end_count(db: Any) -> int | None:
    """How many root transactions have ended on ``db`` since it was created.

    ``None`` means the object cannot report a count at all: a duck-typed
    session has no ``info`` mapping, and ``web/tools/config.py`` documents
    that standalone embedders and unit tests supply those. A caller reads
    ``None`` as "cannot observe" and skips whatever comparison it was
    about to make, the same way ``_restore_session_after_hook_failure``
    skips a session with no ``rollback``.

    A value that is present but is not a count is a different thing and
    raises. The only way one gets there is something writing into
    ``session.info`` under this key, and a caller that cannot read the
    count it is about to compare has to refuse rather than quietly skip
    the comparison -- skipping is indistinguishable from passing.
    """
    info = getattr(db, "info", None)
    if info is None:
        return None
    value = info.get(_ROOT_TXN_END_COUNT_KEY, 0)
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError("connector hook session counter holds a non-count value")
    return value


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
        raise RuntimeError(
            "Session Local is not initialized. Call configure_db() or init_db() first."
        )
    return _SessionLocal


def get_optional_session_local() -> sessionmaker[Session] | None:
    """Return the installed session factory without requiring database setup."""
    return _SessionLocal


def get_engine() -> Engine:
    if _engine is None:
        raise RuntimeError(
            "Engine is not initialized. Call configure_db() or init_db() first."
        )
    return _engine


def configure_db(
    db_url: str | None = None,
    *,
    read_only: bool = False,
) -> None:
    """Bind the engine and session factory without initializing the schema.

    This is the database entry point for read-only operational commands that
    must inspect an already-deployed schema without running migrations,
    creating tables, or seeding application data. SQLite's write-oriented
    connection pragmas are deliberately omitted in ``read_only`` mode.
    """
    global _SessionLocal
    global _engine

    database_url = db_url if db_url is not None else get_database_url()
    if make_url(database_url).get_backend_name() == "sqlite":
        if read_only:
            database_url = _sqlite_read_only_url(database_url)
        else:
            database_url = ensure_sqlite_parent_directory(database_url)
        _engine = create_engine(
            database_url,
            connect_args={"check_same_thread": False},
            poolclass=NullPool,
            # A StatementError's default __str__ includes bound parameter
            # values -- without this, any `except Exception: logger.x(e)`
            # or `str(e)` anywhere in the app that touches a commit/execute
            # failure can put a live secret (an OAuth token, a password
            # hash) bound as a query parameter into a log or, worse, a
            # client-facing response.
            hide_parameters=True,
        )
        if not read_only:
            # WAL + busy_timeout let concurrent writes wait for the lock instead
            # of failing with "database is locked".
            apply_sqlite_concurrency_pragmas(_engine)
    else:
        engine_kwargs: dict[str, Any] = {
            "poolclass": QueuePool,
            **get_db_pool_kwargs(),
            # After the spread, not before: get_db_pool_kwargs() must never
            # be able to silently win this key and turn the guard off.
            "hide_parameters": True,
        }
        if read_only:
            # SQLAlchemy applies PostgreSQL's READ ONLY transaction mode when
            # each connection begins a transaction. This makes audit safety a
            # database-enforced property rather than a reconciler convention.
            engine_kwargs["execution_options"] = {"postgresql_readonly": True}
        _engine = create_engine(
            database_url,
            **engine_kwargs,
        )

    _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_engine)


def _initialize_database_schema(engine: Engine) -> list[dict[str, Any]]:
    """Migrate, create, and seed the schema under one startup ownership lock,
    and refuse to serve if the live ``taskstatus`` enum has drifted from
    ``TaskStatus`` (``check_task_status_enum_drift``, ``models/task.py``)."""

    from ...db.migration import (
        database_startup_lock,
        is_database_empty,
        try_upgrade_db,
    )
    from ..builtin_mcp_registry import (
        seed_builtin_oauth_and_public_mcp_apps,
        validate_builtin_public_mcp_apps,
    )
    from .task import check_task_status_enum_drift

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

        # Checked on both branches rather than only where locked_connection is
        # set: database_startup_lock yields None outright on any non-PostgreSQL
        # backend (see that function's own docstring), so PostgreSQL always
        # takes the locked branch and the other branch's check is a no-op there
        # -- but tying the check to "schema is ready" rather than to "which
        # branch we're on" means a future change to the lock's backend behavior
        # can't silently drop it.
        if locked_connection is not None:
            check_task_status_enum_drift(locked_connection)
            if should_seed_builtin_mcp_registry:
                seed_builtin_oauth_and_public_mcp_apps(locked_connection)
            return validate_builtin_public_mcp_apps(locked_connection)

        with engine.begin() as connection:
            check_task_status_enum_drift(connection)
            if should_seed_builtin_mcp_registry:
                seed_builtin_oauth_and_public_mcp_apps(connection)
            return validate_builtin_public_mcp_apps(connection)


def init_db(db_url: str | None = None) -> None:
    """Initialize database, create all tables and default users"""
    # Import all models to ensure they are registered with Base.metadata
    from . import (  # noqa: F401
        AutoModelCandidate,
        AutoModelConfig,
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
        TaskCommandTerminalEvent,
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

    configure_db(db_url)
    builtin_mcp_mismatches = _initialize_database_schema(get_engine())

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
