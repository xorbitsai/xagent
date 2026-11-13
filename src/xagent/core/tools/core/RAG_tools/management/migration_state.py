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
from .write_leases import (
    LeaseLostError,
    LeaseSet,
    acquire_lease,
    acquire_lease_async,
    release_leases_async,
)

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
# (collection, lease_id) pairs this operation already holds, and the set that
# renews and releases them when the operation ends.
_LEASES: ContextVar[tuple[tuple[str, str], ...]] = ContextVar(
    "kb_write_leases", default=()
)
_LEASE_SET: ContextVar[Optional[LeaseSet]] = ContextVar("kb_lease_set", default=None)
# Every enclosing lease scope, innermost last: a nested scope must not hide a
# lease the outer one lost.
_LEASE_SETS: ContextVar[tuple[LeaseSet, ...]] = ContextVar("kb_lease_sets", default=())

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


def _read_row(
    collection_name: str, store: Optional[Any] = None
) -> Optional[Dict[str, Any]]:
    conn = (store or get_metadata_store()).get_raw_connection()
    if MIGRATION_TABLE not in list_table_names(conn):
        return None
    table = conn.open_table(MIGRATION_TABLE)
    try:
        where = f"name = '{escape_lancedb_string(collection_name)}'"
        rows = table.search().where(where).to_arrow().to_pylist()
    finally:
        _safe_close_table(table)
    return rows[0] if rows else None


def read_migration_state(
    collection_name: str, store: Optional[Any] = None
) -> CollectionMigrationState:
    """Read the collection's state; missing table or row means idle."""
    row = _read_row(collection_name, store)
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


def update_migration_state_if_frozen(
    collection_name: str,
    state: CollectionMigrationState,
) -> bool:
    """Write ``state`` only while the stored row is still frozen.

    Returns ``False`` when an unfreeze landed after the caller's read, so a
    migration step cannot put a frozen row back over it.
    """
    if state.state not in (IDLE, FROZEN):
        raise ValueError(f"Unsupported migration state: {state.state!r}")
    stamped = replace(
        state,
        updated_at=datetime.now(timezone.utc).replace(tzinfo=None),
    )
    conn = get_metadata_store().get_raw_connection()
    _ensure_table(conn)
    where = f"name = '{escape_lancedb_string(collection_name)}' AND state = '{FROZEN}'"
    for attempt in range(WRITE_MAX_RETRIES - 1):
        try:
            return _update_row(conn, where, stamped.to_row())
        except Exception as exc:
            logger.warning(
                "Retrying migration state update for %s: %s", collection_name, exc
            )
            time.sleep(0.1 * (2**attempt))
    return _update_row(conn, where, stamped.to_row())


def _update_row(conn: Any, where: str, values: Dict[str, Any]) -> bool:
    table = conn.open_table(MIGRATION_TABLE)
    try:
        return bool(table.update(where, values).rows_updated)
    finally:
        _safe_close_table(table)


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


def verify_active_leases(collection: Optional[str] = None) -> None:
    """Raise when the current scope's lease set is broken.

    Long writers call this at their commit points: a lease the migration has
    already seen expire means the migration may have started its window, so the
    operation must stop instead of writing behind it.
    """
    sets = _active_sets()
    if not sets:
        return
    try:
        for item in sets:
            item.verify()
    except LeaseLostError as exc:
        raise CollectionMigratingError(
            "Knowledge base write lease was lost; the operation must stop",
            details={"collection": collection, "error": str(exc)},
        ) from exc


def holds_lease(collection: str) -> bool:
    """Whether this scope already holds ``collection``'s lease."""
    return _holding_lease(collection)


@asynccontextmanager
async def holding_leases(
    *names: str, operation: str, store: Optional[Any] = None
) -> AsyncIterator[LeaseSet]:
    """Lease collections for a long operation that starts outside a request scope.

    The coordinator's delete and rename paths do their work after a guard that
    returned immediately, so they hold nothing the freeze can see. They wrap
    their whole body in this instead, and the leases then cover every write.
    """
    _verify_enclosing()
    lease_set = LeaseSet(store=store)
    set_token = _LEASE_SET.set(lease_set)
    sets_token = _LEASE_SETS.set(_LEASE_SETS.get() + (lease_set,))
    leases_token = _LEASES.set(_LEASES.get())
    try:
        for name in dict.fromkeys(names):
            if _holding_lease(name):
                # Reuse the enclosing lease: a fresh one would be a second
                # heartbeat whose loss the outer scope never sees.
                continue
            _record_lease(name, await acquire_lease_async(name, operation, store=store))
        yield lease_set
    finally:
        _LEASE_SETS.reset(sets_token)
        _LEASE_SET.reset(set_token)
        _LEASES.reset(leases_token)
        await release_leases_async(lease_set.stop())


@contextmanager
def leased_collections(*names: str, operation: str) -> Iterator[LeaseSet]:
    """Sync counterpart of :func:`holding_leases`."""
    _verify_enclosing()
    lease_set = LeaseSet()
    set_token = _LEASE_SET.set(lease_set)
    sets_token = _LEASE_SETS.set(_LEASE_SETS.get() + (lease_set,))
    leases_token = _LEASES.set(_LEASES.get())
    try:
        for name in dict.fromkeys(names):
            if _holding_lease(name):
                continue
            _record_lease(name, acquire_lease(name, operation))
        yield lease_set
    finally:
        _LEASE_SETS.reset(sets_token)
        _LEASE_SET.reset(set_token)
        _LEASES.reset(leases_token)
        lease_set.release()


def _active_sets() -> tuple[LeaseSet, ...]:
    """Every lease set visible to this scope, outermost first."""
    frame = _LEASE_SETS.get()
    current = _LEASE_SET.get()
    sets = list(frame)
    if current is not None and current not in sets:
        sets.append(current)
    return tuple(sets)


def _verify_enclosing() -> None:
    """Refuse to start a nested scope inside an already-broken one."""
    for item in _active_sets():
        try:
            item.verify()
        except LeaseLostError as exc:
            raise CollectionMigratingError(
                "Knowledge base write lease was lost before this step",
                details={"error": str(exc)},
            ) from exc


def _admit(collection: str) -> None:
    _ADMITTED.set(_ADMITTED.get() | {collection})


def _is_admitted(collection: str) -> bool:
    return collection in _ADMITTED.get()


def _holding_lease(collection: str) -> bool:
    return any(name == collection for name, _ in _LEASES.get())


def _record_lease(collection: str, lease_id: str) -> None:
    _LEASES.set(_LEASES.get() + ((collection, lease_id),))
    lease_set = _LEASE_SET.get()
    if lease_set is not None:
        lease_set.add(lease_id)


def _lease_for_guard(collection: str, *, store: Optional[Any] = None) -> Optional[str]:
    """Take the operation's lease before the freeze read, when one is due."""
    if collection not in _PENDING.get() or _holding_lease(collection):
        return None
    lease_id = acquire_lease(collection, "write-guard", store=store)
    _record_lease(collection, lease_id)
    return lease_id


async def _lease_for_guard_async(
    collection: str, *, store: Optional[Any] = None
) -> Optional[str]:
    if collection not in _PENDING.get() or _holding_lease(collection):
        return None
    lease_id = await acquire_lease_async(collection, "write-guard", store=store)
    _record_lease(collection, lease_id)
    return lease_id


@contextmanager
def pending_collection(
    *collections: str, store: Optional[Any] = None
) -> Iterator[None]:
    """Open a scope whose first guard call per collection decides its admission."""
    token = _PENDING.set(_PENDING.get() | set(collections))
    admitted_token = _ADMITTED.set(_ADMITTED.get())
    leases_token = _LEASES.set(_LEASES.get())
    lease_set = LeaseSet(store=store)
    set_token = _LEASE_SET.set(lease_set)
    try:
        yield
    finally:
        _LEASE_SET.reset(set_token)
        _LEASES.reset(leases_token)
        _ADMITTED.reset(admitted_token)
        _PENDING.reset(token)
        lease_set.release()


@asynccontextmanager
async def pending_collection_async(
    *collections: str, store: Optional[Any] = None
) -> AsyncIterator[None]:
    """``pending_collection`` for event-loop callers, releasing off the loop."""
    token = _PENDING.set(_PENDING.get() | set(collections))
    admitted_token = _ADMITTED.set(_ADMITTED.get())
    leases_token = _LEASES.set(_LEASES.get())
    lease_set = LeaseSet(store=store)
    set_token = _LEASE_SET.set(lease_set)
    try:
        yield
    finally:
        _LEASE_SET.reset(set_token)
        _LEASES.reset(leases_token)
        _ADMITTED.reset(admitted_token)
        _PENDING.reset(token)
        await release_leases_async(lease_set.stop())


@asynccontextmanager
async def admitted_async(collection: str) -> AsyncIterator[None]:
    """Check the freeze and mark the collection admitted for this scope.

    The lease goes in first: an operation that finds the collection frozen has
    already been visible to the migration, which is what closes the gap between
    checking and freezing.
    """
    if _is_admitted(collection):
        yield
        return
    leases_token = _LEASES.set(_LEASES.get())
    admitted_token = _ADMITTED.set(_ADMITTED.get())
    lease_set = LeaseSet()
    set_token = _LEASE_SET.set(lease_set)
    try:
        if not _holding_lease(collection):
            _record_lease(
                collection, await acquire_lease_async(collection, "admitted_async")
            )
        await ensure_collection_writable_async(collection)
        _ADMITTED.set(_ADMITTED.get() | {collection})
        try:
            yield
        finally:
            # Restore, don't subtract: an outer pending scope may have admitted
            # this name before us, and that admission belongs to the outer scope.
            _ADMITTED.reset(admitted_token)
    finally:
        _LEASE_SET.reset(set_token)
        _LEASES.reset(leases_token)
        # Stop renewing before awaiting: a cancelled release still lets the
        # rows expire instead of a heartbeat renewing them forever.
        await release_leases_async(lease_set.stop())


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
            token = _PENDING.set(_PENDING.get() | set(names))
            admitted_token = _ADMITTED.set(_ADMITTED.get())
            leases_token = _LEASES.set(_LEASES.get())
            lease_set = LeaseSet()
            set_token = _LEASE_SET.set(lease_set)
            try:
                return await func(*args, **kwargs)
            finally:
                _LEASE_SET.reset(set_token)
                _LEASES.reset(leases_token)
                _ADMITTED.reset(admitted_token)
                _PENDING.reset(token)
                # Async release: the sync one joins the heartbeat thread and
                # would block the event loop for up to its timeout.
                await release_leases_async(lease_set.stop())

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


def ensure_collection_writable(
    collection_name: str, *, store: Optional[Any] = None
) -> None:
    """Raise before a write when the collection is frozen for migration."""
    if _is_admitted(collection_name):
        return
    _lease_for_guard(collection_name, store=store)
    _raise_if_frozen(collection_name, read_migration_state(collection_name, store))
    if collection_name in _PENDING.get():
        _ADMITTED.set(_ADMITTED.get() | {collection_name})


async def ensure_collection_writable_async(
    collection_name: str, *, store: Optional[Any] = None
) -> None:
    """Async guard: runs the blocking state read off the event loop."""
    if _is_admitted(collection_name):
        return
    await _lease_for_guard_async(collection_name, store=store)
    state = await asyncio.to_thread(read_migration_state, collection_name, store)
    _raise_if_frozen(collection_name, state)
    if collection_name in _PENDING.get():
        _ADMITTED.set(_ADMITTED.get() | {collection_name})
