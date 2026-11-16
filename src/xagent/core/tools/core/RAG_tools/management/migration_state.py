"""Cross-process embedding-migration freeze state for a KB collection."""

from __future__ import annotations

import asyncio
import functools
import logging
import time
import uuid
from contextlib import asynccontextmanager, contextmanager
from contextvars import Context, ContextVar, copy_context
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
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

from ..core.exceptions import CollectionMigratingError, CollectionNotFrozenError
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
# No collection can be named this, and the writers below refuse the name.
_INSERT_GUARD = "/insert-guard"
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
    ]
)


@dataclass(frozen=True)
class CollectionMigrationState:
    """Migration freeze state stored in its own LanceDB table.

    ``owner`` is the run holding the frozen row's window, ``None`` when no run
    does; only the window functions below write it.
    """

    state: str = IDLE
    source_model_id: Optional[str] = None
    target_model_id: Optional[str] = None
    updated_at: Optional[datetime] = None
    owner: Optional[str] = None

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
    if MIGRATION_TABLE not in list_table_names(conn):
        try:
            conn.create_table(MIGRATION_TABLE, schema=_SCHEMA)
            return
        except Exception:
            if MIGRATION_TABLE not in list_table_names(conn):
                raise
    if _has_owner_column(conn):
        return
    table = conn.open_table(MIGRATION_TABLE)
    try:
        table.add_columns([pa.field("owner", pa.string())])
    except Exception:
        # Of two racing upgrades LanceDB fails one; the column is there for it.
        if not _has_owner_column(conn):
            raise
    finally:
        _safe_close_table(table)


def _has_owner_column(conn: Any) -> bool:
    table = _open_listed_table(conn)
    try:
        return "owner" in table.schema.names
    finally:
        _safe_close_table(table)


def _open_listed_table(conn: Any) -> Any:
    for attempt in range(WRITE_MAX_RETRIES - 1):
        try:
            return conn.open_table(MIGRATION_TABLE)
        except ValueError:
            # Listed while another process is still committing its creation.
            time.sleep(0.1 * (2**attempt))
    return conn.open_table(MIGRATION_TABLE)


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
    )


def write_migration_state(
    collection_name: str,
    state: CollectionMigrationState,
) -> CollectionMigrationState:
    """Store ``state`` for a collection no run holds; only this module writes.

    Raises:
        CollectionMigratingError: If a run holds the collection's window.
    """
    _refuse_guard_name(collection_name)
    if state.state not in (IDLE, FROZEN):
        raise ValueError(f"Unsupported migration state: {state.state!r}")
    stamped = replace(
        state,
        updated_at=datetime.now(timezone.utc).replace(tzinfo=None),
        owner=None,
    )
    conn = get_metadata_store().get_raw_connection()
    row = {"name": collection_name, **stamped.to_row()}
    for attempt in range(WRITE_MAX_RETRIES - 1):
        try:
            written = _merge_row(conn, row)
            break
        except Exception as exc:
            # Retries any error; the final attempt below re-raises.
            logger.warning(
                "Retrying migration state write for %s: %s", collection_name, exc
            )
            time.sleep(0.1 * (2**attempt))
    else:
        written = _merge_row(conn, row)
    if not written:
        raise _held_error(collection_name)
    return stamped


def _update_row(conn: Any, where: str, values: Dict[str, Any]) -> bool:
    table = conn.open_table(MIGRATION_TABLE)
    try:
        return bool(table.update(where, values).rows_updated)
    finally:
        _safe_close_table(table)


def _merge_row(conn: Any, row: dict[str, Any]) -> bool:
    _ensure_table(conn)
    if _insert_new_row(conn, row):
        return True
    where = f"name = '{escape_lancedb_string(row['name'])}' AND owner IS NULL"
    return _update_row(conn, where, {k: v for k, v in row.items() if k != "name"})


def _insert_new_row(conn: Any, row: Dict[str, Any]) -> bool:
    """Insert ``row`` only if its collection has no row yet; whether it did.

    The same commit rewrites the guard row, so two racing inserts conflict in
    LanceDB and the one it re-runs finds the other's row; plain inserts never
    conflict, so both would land.
    """
    guard = {"name": _INSERT_GUARD, "state": IDLE, "updated_at": _next_stamp(None)}
    table = conn.open_table(MIGRATION_TABLE)
    try:
        if not table.search().where(f"name = '{_INSERT_GUARD}'").to_arrow().num_rows:
            table.add(pa.Table.from_pylist([guard], schema=_SCHEMA))
        result = (
            table.merge_insert(["name"])
            .when_matched_update_all(where=f"target.name = '{_INSERT_GUARD}'")
            .when_not_matched_insert_all()
            .execute(pa.Table.from_pylist([guard, row], schema=_SCHEMA))
        )
    finally:
        _safe_close_table(table)
    return bool(result.num_inserted_rows)


def clear_migration_state(collection_name: str) -> None:
    """Remove the collection's row unless a run holds it.

    A missing table or row is a no-op.

    Raises:
        CollectionMigratingError: If a run holds the collection's window.
    """
    _refuse_guard_name(collection_name)
    conn = get_metadata_store().get_raw_connection()
    if MIGRATION_TABLE not in list_table_names(conn):
        return
    _ensure_table(conn)
    table = conn.open_table(MIGRATION_TABLE)
    try:
        table.delete(
            f"name = '{escape_lancedb_string(collection_name)}' AND owner IS NULL"
        )
    finally:
        _safe_close_table(table)
    row = _read_row(collection_name)
    if row is not None and row.get("owner") is not None:
        raise _held_error(collection_name)


@dataclass
class MigrationWindow:
    """One run's hold on a frozen collection.

    ``updated_at`` is the row version this run wrote last: each owned write
    must still find it and advances it in place.
    """

    collection_name: str
    owner: str
    updated_at: datetime


def claim_migration_window(collection_name: str, *, holder: str) -> MigrationWindow:
    """Freeze a collection no run holds, for one run named after ``holder``.

    An absent or idle row is claimed; an idle row keeps its recorded
    transition, the rollback origin of the last switch.

    Raises:
        CollectionMigratingError: If the row is frozen, or changed after it was
            read; the details name its owner and when it was written.
    """
    _refuse_guard_name(collection_name)
    conn = get_metadata_store().get_raw_connection()
    _ensure_table(conn)
    owner = f"{holder}:{uuid.uuid4().hex}"
    row = _read_row(collection_name)
    stamp = _next_stamp(None if row is None else row.get("updated_at"))
    claim = {"state": FROZEN, "updated_at": stamp, "owner": owner}
    if row is None:
        claimed = _insert_new_row(conn, {"name": collection_name, **claim})
    elif _row_state(row) == IDLE:
        where = _version_is(
            collection_name, IDLE, row.get("owner"), row.get("updated_at")
        )
        claimed = _update_row(conn, where, claim)
    else:
        claimed = False
    if claimed:
        # Decided on the row read back, not on the count of a commit LanceDB re-ran.
        current = read_migration_state(collection_name)
        claimed = (current.owner, current.updated_at) == (owner, stamp)
    if not claimed:
        raise _held_error(collection_name)
    return MigrationWindow(collection_name, owner, stamp)


def take_over_migration_window(
    collection_name: str, observed: CollectionMigrationState, *, holder: str
) -> MigrationWindow:
    """Give the frozen row ``observed`` was read from to a new run.

    The caller states that the run holding it is gone; nothing here checks
    that. The row changes only while it is still exactly the one observed, and
    the collection row is fenced before this returns, so a binding write the
    previous run prepared can no longer land. If the fence fails, the row is
    already taken but no window is returned: take it over again.

    Raises:
        CollectionNotFrozenError: If ``observed`` is not a frozen row.
        CollectionMigratingError: If the row changed after it was observed.
    """
    _refuse_guard_name(collection_name)
    if observed.state != FROZEN:
        raise CollectionNotFrozenError(
            f"Collection '{collection_name}' is not frozen; claim it instead.",
            details={"collection": collection_name, "migration_state": observed.state},
        )
    conn = get_metadata_store().get_raw_connection()
    _ensure_table(conn)
    owner = f"{holder}:{uuid.uuid4().hex}"
    stamp = _next_stamp(observed.updated_at)
    where = _version_is(collection_name, FROZEN, observed.owner, observed.updated_at)
    if not _update_row(conn, where, {"updated_at": stamp, "owner": owner}):
        raise _held_error(collection_name)
    # collection_manager imports this module.
    from .collection_manager import fence_collection_binding_sync

    fence_collection_binding_sync(collection_name)
    return MigrationWindow(collection_name, owner, stamp)


def record_migration_transition(
    window: MigrationWindow,
    source_model_id: Optional[str],
    target_model_id: str,
) -> None:
    """Record the switch this run is about to apply, while it holds ``window``."""
    _write_owned(
        window,
        {"source_model_id": source_model_id, "target_model_id": target_model_id},
    )


def release_migration_window(window: MigrationWindow) -> None:
    """Unfreeze the collection; the recorded transition stays as rollback origin."""
    _write_owned(window, {"state": IDLE, "owner": None})


def _write_owned(window: MigrationWindow, values: Dict[str, Any]) -> None:
    stamp = _next_stamp(window.updated_at)
    where = _version_is(window.collection_name, FROZEN, window.owner, window.updated_at)
    conn = get_metadata_store().get_raw_connection()
    if not _update_row(conn, where, {**values, "updated_at": stamp}):
        raise CollectionNotFrozenError(
            f"Collection '{window.collection_name}' is no longer frozen by this "
            "run; read its migration state before acting on it.",
            details={
                "collection": window.collection_name,
                "owner": window.owner,
                "migration_state": "not-owned",
            },
        )
    window.updated_at = stamp


def _refuse_guard_name(collection_name: str) -> None:
    if collection_name == _INSERT_GUARD:
        raise ValueError(f"'{_INSERT_GUARD}' is reserved by the migration table.")


def _version_is(
    collection_name: str,
    state: str,
    owner: Optional[str],
    stamp: Optional[datetime],
) -> str:
    owner_sql = (
        "owner IS NULL"
        if owner is None
        else f"owner = '{escape_lancedb_string(owner)}'"
    )
    stamp_sql = (
        "updated_at IS NULL"
        if stamp is None
        else f"updated_at = timestamp '{stamp.strftime('%Y-%m-%d %H:%M:%S.%f')}'"
    )
    return (
        f"name = '{escape_lancedb_string(collection_name)}' "
        f"AND state = '{state}' AND {owner_sql} AND {stamp_sql}"
    )


def _next_stamp(previous: Optional[datetime]) -> datetime:
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    if previous is None or now > previous:
        return now
    # A clock that did not advance must still change the row version.
    return previous + timedelta(microseconds=1)


def _held_error(collection_name: str) -> CollectionMigratingError:
    current = read_migration_state(collection_name)
    return CollectionMigratingError(
        f"Knowledge base '{collection_name}' was not changed: its migration row "
        f"is {current.state}, owner {current.owner or 'unrecorded'}, written "
        f"{current.updated_at}.",
        details={
            "collection": collection_name,
            "migration_state": current.state,
            "owner": current.owner,
            "updated_at": current.updated_at,
        },
    )


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
