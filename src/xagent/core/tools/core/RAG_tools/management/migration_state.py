"""Cross-process embedding-migration freeze state for a KB collection."""

from __future__ import annotations

import asyncio
import functools
import logging
import time
from contextlib import asynccontextmanager, contextmanager
from contextvars import Context, ContextVar, copy_context
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Dict,
    Iterable,
    Iterator,
    Optional,
    TypeVar,
    Union,
    cast,
)

import pyarrow as pa  # type: ignore

from ..core.exceptions import CollectionMigratingError
from ..LanceDB.schema_manager import _safe_close_table
from ..storage.factory import get_metadata_store
from ..utils.lancedb_query_utils import list_table_names
from ..utils.string_utils import escape_lancedb_string

MIGRATION_TABLE = "collection_migration_state"
IDLE = "idle"
FROZEN = "frozen"
WRITE_MAX_RETRIES = 3

logger = logging.getLogger(__name__)

# Collections whose operation already passed its single admission check, and
# collections whose enclosing operation may still admit on its first guard call.
_ADMITTED: ContextVar[frozenset[str]] = ContextVar(
    "kb_admitted_collections", default=frozenset()
)
_PENDING: ContextVar[frozenset[str]] = ContextVar(
    "kb_pending_collections", default=frozenset()
)

_SCHEMA = pa.schema(
    [
        ("name", pa.string()),
        ("state", pa.string()),
        ("source_model_id", pa.string()),
        ("target_model_id", pa.string()),
        ("updated_at", pa.timestamp("us")),
    ]
)


@dataclass(frozen=True)
class CollectionMigrationState:
    """Migration freeze state stored in its own LanceDB table."""

    state: str = IDLE
    source_model_id: Optional[str] = None
    target_model_id: Optional[str] = None
    updated_at: Optional[datetime] = None

    def to_row(self) -> Dict[str, Any]:
        return {
            "state": self.state,
            "source_model_id": self.source_model_id,
            "target_model_id": self.target_model_id,
            "updated_at": self.updated_at,
        }


def _row_state(row: Dict[str, Any]) -> str:
    # Fail closed: a row we cannot interpret freezes the collection.
    state = row.get("state")
    if state in (IDLE, FROZEN):
        return str(state)
    return FROZEN


def _ensure_table(conn: Any) -> None:
    if MIGRATION_TABLE in list_table_names(conn):
        return
    try:
        conn.create_table(MIGRATION_TABLE, schema=_SCHEMA)
    except Exception:
        if MIGRATION_TABLE not in list_table_names(conn):
            raise


def _read_row(collection_name: str) -> Optional[Dict[str, Any]]:
    conn = get_metadata_store().get_raw_connection()
    if MIGRATION_TABLE not in list_table_names(conn):
        return None
    table = conn.open_table(MIGRATION_TABLE)
    try:
        where = f"name = '{escape_lancedb_string(collection_name)}'"
        rows = table.search().where(where).to_arrow().to_pylist()
    finally:
        _safe_close_table(table)
    return rows[0] if rows else None


def read_migration_state(collection_name: str) -> CollectionMigrationState:
    """Read the collection's state; missing table or row means idle."""
    row = _read_row(collection_name)
    if row is None:
        return CollectionMigrationState()
    return CollectionMigrationState(
        state=_row_state(row),
        source_model_id=row.get("source_model_id"),
        target_model_id=row.get("target_model_id"),
        updated_at=row.get("updated_at"),
    )


def write_migration_state(
    collection_name: str,
    state: CollectionMigrationState,
) -> CollectionMigrationState:
    """Store ``state`` for one collection; only this module writes the table."""
    if state.state not in (IDLE, FROZEN):
        raise ValueError(f"Unsupported migration state: {state.state!r}")
    stamped = replace(
        state,
        updated_at=datetime.now(timezone.utc).replace(tzinfo=None),
    )
    conn = get_metadata_store().get_raw_connection()
    row = {"name": collection_name, **stamped.to_row()}
    for attempt in range(WRITE_MAX_RETRIES - 1):
        try:
            _merge_row(conn, row)
            return stamped
        except Exception as exc:
            # Retries any error; the final attempt below re-raises.
            logger.warning(
                "Retrying migration state write for %s: %s", collection_name, exc
            )
            time.sleep(0.1 * (2**attempt))
    _merge_row(conn, row)
    return stamped


def _merge_row(conn: Any, row: dict[str, Any]) -> None:
    _ensure_table(conn)
    table = conn.open_table(MIGRATION_TABLE)
    try:
        table.merge_insert(
            ["name"]
        ).when_matched_update_all().when_not_matched_insert_all().execute([row])
    finally:
        _safe_close_table(table)


def clear_migration_state(collection_name: str) -> None:
    """Remove the collection's row; a missing table or row is a no-op."""
    conn = get_metadata_store().get_raw_connection()
    if MIGRATION_TABLE not in list_table_names(conn):
        return
    table = conn.open_table(MIGRATION_TABLE)
    try:
        table.delete(f"name = '{escape_lancedb_string(collection_name)}'")
    finally:
        _safe_close_table(table)


def _admit(collection: str) -> None:
    _ADMITTED.set(_ADMITTED.get() | {collection})


def _is_admitted(collection: str) -> bool:
    return collection in _ADMITTED.get()


@contextmanager
def pending_collection(*collections: str) -> Iterator[None]:
    """Open a scope whose first guard call per collection decides its admission."""
    token = _PENDING.set(_PENDING.get() | set(collections))
    admitted_token = _ADMITTED.set(_ADMITTED.get())
    try:
        yield
    finally:
        _ADMITTED.reset(admitted_token)
        _PENDING.reset(token)


@asynccontextmanager
async def admitted_async(collection: str) -> AsyncIterator[None]:
    """Check the freeze and mark the collection admitted for this scope."""
    if _is_admitted(collection):
        yield
        return
    await ensure_collection_writable_async(collection)
    token = _ADMITTED.set(_ADMITTED.get() | {collection})
    try:
        yield
    finally:
        _ADMITTED.reset(token)


async def admitted_context_async(collection: str) -> Context:
    """Check once, then return a context carrying the admission into a thread."""
    if not _is_admitted(collection):
        await ensure_collection_writable_async(collection)
    context = copy_context()
    context.run(_admit, collection)
    return context


F = TypeVar("F", bound=Callable[..., Awaitable[Any]])
S = TypeVar("S", bound=Callable[..., Any])
Names = Union[str, Iterable[str], None]


def _as_names(value: Names) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(value)


def admits_collection(
    get_name: Callable[..., Names],
) -> Callable[[F], F]:
    """Open a pending admission for the handler; its first guard call checks."""

    def decorate(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            names = _as_names(get_name(*args, **kwargs))
            if not names:
                return await func(*args, **kwargs)
            with pending_collection(*names):
                return await func(*args, **kwargs)

        return cast(F, wrapper)

    return decorate


def admits_collection_sync(
    get_name: Callable[..., Optional[str]],
) -> Callable[[S], S]:
    """Sync variant of :func:`admits_collection`, for job handlers."""

    def decorate(func: S) -> S:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            name = get_name(*args, **kwargs)
            if name is None:
                return func(*args, **kwargs)
            with pending_collection(name):
                return func(*args, **kwargs)

        return cast(S, wrapper)

    return decorate


def run_in_admitted_context(
    context: Context,
    fn: Callable[..., Any],
    /,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Run ``fn`` in a fresh copy of an admitted context; a Context cannot be shared."""
    return context.copy().run(fn, *args, **kwargs)


def _raise_if_frozen(collection_name: str, state: CollectionMigrationState) -> None:
    if state.state == FROZEN:
        raise CollectionMigratingError(
            f"Knowledge base '{collection_name}' is being upgraded to a new "
            "embedding model; writes are disabled until it finishes.",
            details={"collection": collection_name, "migration_state": state.state},
        )


def ensure_collection_writable(collection_name: str) -> None:
    """Raise before a write when the collection is frozen for migration."""
    if _is_admitted(collection_name):
        return
    _raise_if_frozen(collection_name, read_migration_state(collection_name))
    if collection_name in _PENDING.get():
        _ADMITTED.set(_ADMITTED.get() | {collection_name})


async def ensure_collection_writable_async(collection_name: str) -> None:
    """Async guard: runs the blocking state read off the event loop."""
    if _is_admitted(collection_name):
        return
    state = await asyncio.to_thread(read_migration_state, collection_name)
    _raise_if_frozen(collection_name, state)
    if collection_name in _PENDING.get():
        _ADMITTED.set(_ADMITTED.get() | {collection_name})
