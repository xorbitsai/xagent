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
    Set,
    TypeVar,
    Union,
    cast,
)

import pyarrow as pa  # type: ignore

from xagent.core.utils.blocking import run_blocking

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
        ("owner", pa.string()),
        ("mode", pa.string()),
    ]
)


@dataclass(frozen=True)
class CollectionMigrationState:
    """Migration freeze state stored in its own LanceDB table."""

    state: str = IDLE
    source_model_id: Optional[str] = None
    target_model_id: Optional[str] = None
    updated_at: Optional[datetime] = None
    owner: Optional[str] = None
    mode: Optional[str] = None

    def to_row(self) -> Dict[str, Any]:
        return {
            "state": self.state,
            "source_model_id": self.source_model_id,
            "target_model_id": self.target_model_id,
            "updated_at": self.updated_at,
            "owner": self.owner,
            "mode": self.mode,
        }


def _row_state(row: Dict[str, Any]) -> str:
    # Fail closed: a row we cannot interpret freezes the collection.
    state = row.get("state")
    if state in (IDLE, FROZEN):
        return str(state)
    return FROZEN


def _ensure_table(conn: Any) -> None:
    if MIGRATION_TABLE in list_table_names(conn):
        _ensure_columns(conn)
        return
    try:
        conn.create_table(MIGRATION_TABLE, schema=_SCHEMA)
    except Exception:
        if MIGRATION_TABLE not in list_table_names(conn):
            raise


def _ensure_columns(conn: Any) -> None:
    """Add the columns a table created by an older version lacks."""
    table = conn.open_table(MIGRATION_TABLE)
    try:
        schema = table.schema
        # An earlier version added these as ``null``-typed, which rejects every
        # later string write; they can only hold nulls, so re-add them typed.
        broken = [f.name for f in schema if pa.types.is_null(f.type)]
        if broken:
            table.drop_columns(broken)
        present = set(schema.names) - set(broken)
        missing = [f for f in _SCHEMA if f.name not in present]
        if missing:
            table.add_columns({f.name: _null_cast(f.type) for f in missing})
    finally:
        _safe_close_table(table)


def _null_cast(field_type: Any) -> str:
    """A null of the column's own type: a string cast would break ``updated_at``."""
    if pa.types.is_timestamp(field_type):
        return "cast(null as timestamp)"
    return "cast(null as string)"


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
        owner=row.get("owner"),
        mode=row.get("mode"),
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


def read_raw_state(collection_name: str) -> Optional[Dict[str, Any]]:
    """The stored row as written, for a byte-exact restore."""
    row = _read_row(collection_name)
    return None if row is None else dict(row)


def restore_state(collection_name: str, row: Optional[Dict[str, Any]]) -> None:
    """Put ``row`` back exactly, or remove the collection's row when it was absent."""
    if row is None:
        clear_migration_state(collection_name)
        return
    conn = get_metadata_store().get_raw_connection()
    _merge_row(conn, dict(row))


def claim_freeze(
    collection_name: str,
    *,
    source_model_id: str,
    target_model_id: str,
    mode: str,
    owner: str,
    stale_after_seconds: float,
) -> str:
    """Freeze the collection for this run, or say why it cannot.

    Returns ``"claimed"`` for a fresh freeze, ``"resumed"`` for a window that is
    ours or whose owner stopped heartbeating, and ``"busy"`` otherwise. The write
    is conditional on the row read here, and the owner is re-read afterwards.
    """
    previous = _read_row(collection_name)
    requested = replace(
        CollectionMigrationState(
            state=FROZEN,
            source_model_id=source_model_id,
            target_model_id=target_model_id,
            mode=mode,
            owner=owner,
        ),
        updated_at=_now(),
    )
    conn = get_metadata_store().get_raw_connection()
    if previous is None:
        _ensure_table(conn)
        table = conn.open_table(MIGRATION_TABLE)
        try:
            table.merge_insert(["name"]).when_not_matched_insert_all().execute(
                [{"name": collection_name, **requested.to_row()}]
            )
        finally:
            _safe_close_table(table)
        outcome = "claimed"
    else:
        stored_state = previous.get("state")
        stored_owner = previous.get("owner")
        if stored_state == FROZEN:
            if stored_owner != owner and (
                _age_seconds(previous.get("updated_at")) < stale_after_seconds
            ):
                return "busy"
            outcome = "resumed"
        elif stored_state == IDLE:
            outcome = "claimed"
        else:
            return "busy"
        _ensure_table(conn)
        where = (
            f"name = '{escape_lancedb_string(collection_name)}' "
            f"AND state = '{stored_state}' AND {_owner_predicate(stored_owner)}"
        )
        for attempt in range(WRITE_MAX_RETRIES - 1):
            try:
                if not _update_row(conn, where, requested.to_row()):
                    return "busy"
                break
            except Exception as exc:
                logger.warning(
                    "Retrying migration state claim for %s: %s", collection_name, exc
                )
                time.sleep(0.1 * (2**attempt))
        else:
            if not _update_row(conn, where, requested.to_row()):
                return "busy"
    stored = read_migration_state(collection_name)
    if stored.state != FROZEN or stored.owner != owner:
        return "busy"
    return outcome


def heartbeat_freeze(collection_name: str, owner: str) -> bool:
    """Renew the window's row; ``False`` means this run no longer owns it."""
    conn = get_metadata_store().get_raw_connection()
    where = (
        f"name = '{escape_lancedb_string(collection_name)}' "
        f"AND state = '{FROZEN}' AND {_owner_predicate(owner)}"
    )
    return _update_row(conn, where, {"updated_at": _now()})


def _owner_predicate(owner: Optional[str]) -> str:
    if owner is None:
        return "owner IS NULL"
    return f"owner = '{escape_lancedb_string(owner)}'"


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _age_seconds(stamp: Any) -> float:
    if not isinstance(stamp, datetime):
        return 0.0
    current = datetime.now(timezone.utc)
    value = stamp if stamp.tzinfo else stamp.replace(tzinfo=timezone.utc)
    return max(0.0, (current - value).total_seconds())


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
    """Raise when any lease set in this operation is broken.

    Long writers call this at their commit points: a lease the migration has
    already seen expire means the migration may have started its window, so the
    operation must stop instead of writing behind it. Every enclosing set is
    checked, so a nested scope cannot hide a lease the outer one lost.
    """
    sets = _LEASE_SETS.get()
    if not sets:
        return
    if collection is not None and not holds_lease(collection):
        raise CollectionMigratingError(
            "No write lease is held for this collection; the operation must stop",
            details={"collection": collection},
        )
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
    return any(name == collection for name, _ in _LEASES.get())


def _push_scope(lease_set: LeaseSet) -> Any:
    return _LEASE_SETS.set(_LEASE_SETS.get() + (lease_set,))


def _current_set() -> Optional[LeaseSet]:
    sets = _LEASE_SETS.get()
    return sets[-1] if sets else None


def _verify_enclosing() -> None:
    """Refuse to start a nested scope inside an already-broken one."""
    for item in _LEASE_SETS.get():
        try:
            item.verify()
        except LeaseLostError as exc:
            raise CollectionMigratingError(
                "Knowledge base write lease was lost before this step",
                details={"error": str(exc)},
            ) from exc


@contextmanager
def _lease_scope(*, store: Optional[Any] = None) -> Iterator[LeaseSet]:
    """The one place a scope is opened: push the set, reset it, release it."""
    _verify_enclosing()
    lease_set = LeaseSet(store=store)
    sets_token = _push_scope(lease_set)
    leases_token = _LEASES.set(_LEASES.get())
    try:
        yield lease_set
    finally:
        _LEASE_SETS.reset(sets_token)
        _LEASES.reset(leases_token)
        lease_set.release()


@asynccontextmanager
async def _lease_scope_async(*, store: Optional[Any] = None) -> AsyncIterator[LeaseSet]:
    """``_lease_scope`` for event-loop callers: neither join nor I/O on the loop."""
    _verify_enclosing()
    lease_set = LeaseSet(store=store)
    sets_token = _push_scope(lease_set)
    leases_token = _LEASES.set(_LEASES.get())
    try:
        yield lease_set
    finally:
        _LEASE_SETS.reset(sets_token)
        _LEASES.reset(leases_token)
        # Stopping the heartbeat must not wait on an await: a second cancel
        # would leave it renewing a row nobody releases.
        lease_set.begin_stop()
        ids = lease_set.lease_ids
        try:
            await asyncio.shield(run_blocking(lease_set.stop))
        finally:
            await asyncio.shield(_release_tracked(ids, store))


def _take_lease(name: str, operation: str, store: Optional[Any]) -> None:
    """Take ``name``'s lease unless an enclosing scope already holds it."""
    if holds_lease(name):
        # A second lease would be a second heartbeat whose loss the outer scope
        # never sees; reuse what the outer scope already keeps alive.
        return
    _record_lease(name, acquire_lease(name, operation, store=store))


async def _take_lease_async(name: str, operation: str, store: Optional[Any]) -> None:
    if holds_lease(name):
        return
    task = asyncio.ensure_future(acquire_lease_async(name, operation, store=store))
    try:
        lease_id = await asyncio.shield(task)
    except asyncio.CancelledError:
        # The insert may land after the cancel; release whatever it produced so
        # the row cannot delay a migration for a full TTL.
        task.add_done_callback(functools.partial(_release_after_cancel, store=store))
        raise
    _record_lease(name, lease_id)


# Releases still in flight: the loop keeps only a weak reference to a task.
_RELEASE_TASKS: Set["asyncio.Future[Any]"] = set()


def _release_tracked(ids: Iterable[str], store: Optional[Any]) -> "asyncio.Future[Any]":
    task = asyncio.ensure_future(release_leases_async(list(ids), store=store))
    _RELEASE_TASKS.add(task)
    task.add_done_callback(_RELEASE_TASKS.discard)
    return task


def _release_after_cancel(
    task: "asyncio.Future[str]", store: Optional[Any] = None
) -> None:
    if task.cancelled() or task.exception() is not None:
        return
    try:
        _release_tracked([task.result()], store)
    except RuntimeError:  # pragma: no cover - loop gone with the request
        logger.warning("No loop to release lease %s after cancellation", task.result())


@asynccontextmanager
async def holding_leases(
    *names: str, operation: str, store: Optional[Any] = None
) -> AsyncIterator[LeaseSet]:
    """Lease collections for a long operation that starts outside a request scope.

    The coordinator's delete and rename paths do their work after a guard that
    returned immediately, so they hold nothing the freeze can see. They wrap
    their whole body in this instead, and the leases then cover every write.
    """
    async with _lease_scope_async(store=store) as lease_set:
        for name in dict.fromkeys(names):
            await _take_lease_async(name, operation, store)
        yield lease_set


@contextmanager
def leased_collections(*names: str, operation: str) -> Iterator[LeaseSet]:
    """Sync counterpart of :func:`holding_leases`."""
    with _lease_scope() as lease_set:
        for name in dict.fromkeys(names):
            _take_lease(name, operation, store=None)
        yield lease_set


def _admit(collection: str) -> None:
    _ADMITTED.set(_ADMITTED.get() | {collection})


def _is_admitted(collection: str) -> bool:
    return collection in _ADMITTED.get()


def _record_lease(collection: str, lease_id: str) -> None:
    _LEASES.set(_LEASES.get() + ((collection, lease_id),))
    lease_set = _current_set()
    if lease_set is not None:
        lease_set.add(lease_id)


def _lease_for_guard(collection: str, *, store: Optional[Any] = None) -> None:
    """Take the operation's lease before the freeze read, when one is due."""
    if collection not in _PENDING.get() or holds_lease(collection):
        return
    _record_lease(collection, acquire_lease(collection, "write-guard", store=store))


async def _lease_for_guard_async(
    collection: str, *, store: Optional[Any] = None
) -> None:
    if collection not in _PENDING.get() or holds_lease(collection):
        return
    await _take_lease_async(collection, "write-guard", store)


@contextmanager
def pending_collection(*collections: str) -> Iterator[None]:
    """Open a scope whose first guard call per collection decides its admission."""
    token = _PENDING.set(_PENDING.get() | set(collections))
    admitted_token = _ADMITTED.set(_ADMITTED.get())
    with _lease_scope():
        try:
            yield
        finally:
            _ADMITTED.reset(admitted_token)
            _PENDING.reset(token)


@asynccontextmanager
async def pending_collection_async(*collections: str) -> AsyncIterator[None]:
    """``pending_collection`` for event-loop callers, releasing off the loop."""
    token = _PENDING.set(_PENDING.get() | set(collections))
    admitted_token = _ADMITTED.set(_ADMITTED.get())
    async with _lease_scope_async():
        try:
            yield
        finally:
            _ADMITTED.reset(admitted_token)
            _PENDING.reset(token)


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
    async with _lease_scope_async():
        await _take_lease_async(collection, "admitted_async", None)
        await ensure_collection_writable_async(collection)
        # Taken after the guard: resetting restores the post-guard state, so an
        # admission the guard granted to the enclosing pending scope survives.
        admitted_token = _ADMITTED.set(_ADMITTED.get() | {collection})
        try:
            yield
        finally:
            _ADMITTED.reset(admitted_token)


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
            async with pending_collection_async(*names):
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
