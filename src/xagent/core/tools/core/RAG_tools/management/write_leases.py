"""Short-lived write leases for operations admitted before a migration freeze.

The freeze guard admits an operation once per scope, and that admission lives in
process-local context variables, so a migration running in another process cannot
see a writer that was already admitted. Each admission records a lease row here
instead, released when the scope exits; a scope that outlives the TTL renews its
rows, and a crashed process leaves rows that expire.

Ordering is the point: the writer inserts its lease *before* it reads the freeze
state, and the migration writes the freeze *before* it reads leases. Either the
writer sees the freeze and refuses, or the migration sees the lease and waits.
"""

from __future__ import annotations

import logging
import os
import socket
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, List, Optional, Tuple

import pyarrow as pa  # type: ignore

from xagent.core.utils.blocking import run_blocking

from ..LanceDB.schema_manager import _safe_close_table
from ..storage.factory import get_metadata_store, get_vector_index_store
from ..utils.lancedb_query_utils import list_table_names
from ..utils.string_utils import escape_lancedb_string

LEASES_TABLE = "collection_write_leases"
# Short enough that a crashed operation stops delaying a migration quickly; a
# live scope renews its rows, so long ingests do not lose their lease.
DEFAULT_LEASE_TTL_SECONDS = 120.0
DEFAULT_WAIT_POLL_SECONDS = 1.0
COMPACT_EVERY_ACQUIRES = 200
# Writers and the migration compare wall clocks. Rows expire and are judged
# expired with this much slack, so a clock that runs ahead cannot make another
# host's live lease look dead; the cost is a slightly longer wait.
EXPIRY_SKEW_MARGIN_SECONDS = 5.0
LEASE_WRITE_RETRIES = 3
GC_EVERY_RENEWALS = 60

logger = logging.getLogger(__name__)

_SCHEMA = pa.schema(
    [
        ("name", pa.string()),
        ("lease_id", pa.string()),
        ("operation", pa.string()),
        ("owner", pa.string()),
        ("created_at", pa.timestamp("us")),
        ("expires_at", pa.timestamp("us")),
    ]
)

_acquire_count = 0
_acquire_lock = threading.Lock()
_compact_lock = threading.Lock()


@dataclass(frozen=True)
class WriteLease:
    """One admitted operation that may still be writing to a collection."""

    name: str
    lease_id: str
    operation: str
    owner: str
    created_at: Optional[datetime]
    expires_at: Optional[datetime]


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _store_connection(store: Optional[Any]) -> Any:
    return (store or get_metadata_store()).get_raw_connection()


def _ensure_table(conn: Any) -> None:
    if LEASES_TABLE in list_table_names(conn):
        return
    try:
        conn.create_table(LEASES_TABLE, schema=_SCHEMA)
    except Exception:
        if LEASES_TABLE not in list_table_names(conn):
            raise


def _compact() -> None:
    try:
        get_vector_index_store().trigger_reindex(LEASES_TABLE)
    except Exception:  # noqa: BLE001 - compaction must never fail a write
        logger.exception("Failed to compact %s", LEASES_TABLE)


def _maybe_compact() -> None:
    """Compact the lease table once every few hundred operations, off the caller.

    Rows are deleted on release, so the table only ever holds in-flight leases;
    what accumulates is fragments and versions from the add/delete/heartbeat
    writes. Compaction runs on a daemon thread so a slow merge never delays an
    admission.
    """
    global _acquire_count
    with _acquire_lock:
        _acquire_count += 1
        due = _acquire_count % COMPACT_EVERY_ACQUIRES == 0
    if not due or _compact_lock.locked():
        return
    with _compact_lock:
        threading.Thread(target=_compact, name="kb-lease-compact", daemon=True).start()


def acquire_lease(
    name: str,
    operation: str,
    *,
    ttl_seconds: float = DEFAULT_LEASE_TTL_SECONDS,
    store: Optional[Any] = None,
) -> str:
    """Record that this process may write ``name`` until it releases the lease.

    Raises on failure: an admission the migration cannot see must not proceed
    silently, the same way the freeze read fails closed.
    """
    conn = _store_connection(store)
    _ensure_table(conn)
    now = _now()
    lease_id = uuid.uuid4().hex
    row = {
        "name": name,
        "lease_id": lease_id,
        "operation": operation,
        "owner": f"{socket.gethostname()}:{os.getpid()}",
        "created_at": now,
        "expires_at": now + timedelta(seconds=max(1.0, float(ttl_seconds))),
    }
    table = conn.open_table(LEASES_TABLE)
    try:
        for attempt in range(LEASE_WRITE_RETRIES):
            try:
                table.add([row])
                break
            except Exception as exc:
                # A concurrent commit conflicts with ours; retry like the freeze
                # table's writer, and only then let the admission fail.
                if attempt == LEASE_WRITE_RETRIES - 1:
                    raise
                logger.warning("Retrying lease insert for %s: %s", name, exc)
                time.sleep(0.1 * (2**attempt))
    finally:
        _safe_close_table(table)
    _maybe_compact()
    return lease_id


def renew_leases(
    lease_ids: Iterable[str],
    *,
    ttl_seconds: float = DEFAULT_LEASE_TTL_SECONDS,
    store: Optional[Any] = None,
) -> Tuple[int, List[str]]:
    """Extend these leases; returns (renewed, lost ids).

    A lease whose row is gone, or whose expiry has already passed, is lost: it
    must not be revived, or a migration that saw it expire would never learn
    that its owner is still writing.
    """
    ids = [lease_id for lease_id in lease_ids if lease_id]
    if not ids:
        return 0, []
    conn = _store_connection(store)
    if LEASES_TABLE not in list_table_names(conn):
        return 0, list(ids)
    table = conn.open_table(LEASES_TABLE)
    renewed = 0
    lost: List[str] = []
    now = _now()
    try:
        for lease_id in ids:
            where = f"lease_id = '{escape_lancedb_string(lease_id)}'"
            rows = table.search().where(where).to_arrow().to_pylist()
            if not rows:
                lost.append(lease_id)
                continue
            expires_at = rows[0].get("expires_at")
            if not isinstance(expires_at, datetime) or expires_at <= now:
                lost.append(lease_id)
                continue
            target = now + timedelta(seconds=max(1.0, float(ttl_seconds)))
            if expires_at >= target:
                renewed += 1
                continue
            renewed += int(
                getattr(table.update(where, {"expires_at": target}), "rows_updated", 0)
                or 0
            )
        return renewed, lost
    finally:
        _safe_close_table(table)


def release_leases(lease_ids: Iterable[str], *, store: Optional[Any] = None) -> int:
    """Delete these leases; a missing table or row is a no-op."""
    ids = [lease_id for lease_id in lease_ids if lease_id]
    if not ids:
        return 0
    conn = _store_connection(store)
    if LEASES_TABLE not in list_table_names(conn):
        return 0
    table = conn.open_table(LEASES_TABLE)
    try:
        for lease_id in ids:
            table.delete(f"lease_id = '{escape_lancedb_string(lease_id)}'")
    finally:
        _safe_close_table(table)
    return len(ids)


def live_leases(name: str, *, store: Optional[Any] = None) -> List[WriteLease]:
    """The unexpired leases for ``name``; a missing table means none."""
    conn = _store_connection(store)
    if LEASES_TABLE not in list_table_names(conn):
        return []
    table = conn.open_table(LEASES_TABLE)
    try:
        where = f"name = '{escape_lancedb_string(name)}'"
        rows = table.search().where(where).to_arrow().to_pylist()
    finally:
        _safe_close_table(table)
    # Err towards alive: a lease is live until it is a margin past its expiry.
    cutoff = _now() - timedelta(seconds=EXPIRY_SKEW_MARGIN_SECONDS)
    live: List[WriteLease] = []
    for row in rows:
        expires_at = row.get("expires_at")
        if not isinstance(expires_at, datetime) or expires_at <= cutoff:
            continue
        live.append(
            WriteLease(
                name=str(row.get("name") or ""),
                lease_id=str(row.get("lease_id") or ""),
                operation=str(row.get("operation") or ""),
                owner=str(row.get("owner") or ""),
                created_at=row.get("created_at"),
                expires_at=expires_at,
            )
        )
    return live


def purge_expired_leases(
    name: Optional[str] = None, *, store: Optional[Any] = None
) -> int:
    """Delete expired rows, of ``name`` or of every collection; live rows stay.

    The delete repeats the expiry test in its own predicate, so a lease renewed
    between the read and the delete is not removed.
    """
    conn = _store_connection(store)
    if LEASES_TABLE not in list_table_names(conn):
        return 0
    table = conn.open_table(LEASES_TABLE)
    try:
        where = (
            f"name = '{escape_lancedb_string(name)}'"
            if name is not None
            else "lease_id IS NOT NULL"
        )
        rows = table.search().where(where).to_arrow().to_pylist()
        cutoff = _now() - timedelta(seconds=EXPIRY_SKEW_MARGIN_SECONDS)
        purged = 0
        for row in rows:
            lease_id = str(row.get("lease_id") or "")
            expires_at = row.get("expires_at")
            if not lease_id or not isinstance(expires_at, datetime):
                continue
            if expires_at > cutoff:
                continue
            lease_where = (
                f"lease_id = '{escape_lancedb_string(lease_id)}' "
                f"AND expires_at <= TIMESTAMP '{cutoff:%Y-%m-%d %H:%M:%S.%f}'"
            )
            try:
                result = table.delete(lease_where)
            except Exception as exc:  # noqa: BLE001 - a raced renewal keeps its row
                # Usually a renewal between the read and the delete; anything
                # else leaves an expired row behind, which is worth knowing.
                logger.warning("Could not purge lease %s: %s", lease_id, exc)
                continue
            purged += int(getattr(result, "num_deleted_rows", 0) or 0)
        return purged
    finally:
        _safe_close_table(table)


def wait_for_no_leases(
    name: str,
    *,
    timeout_seconds: float,
    poll_seconds: float = DEFAULT_WAIT_POLL_SECONDS,
    store: Optional[Any] = None,
) -> List[WriteLease]:
    """Poll until ``name`` has no live lease; returns what is left at the deadline."""
    deadline = time.monotonic() + max(0.0, float(timeout_seconds))
    while True:
        remaining = live_leases(name, store=store)
        if not remaining:
            purge_expired_leases(name, store=store)
            return []
        if time.monotonic() >= deadline:
            return remaining
        time.sleep(max(0.01, min(poll_seconds, max(0.01, deadline - time.monotonic()))))


async def acquire_lease_async(
    name: str,
    operation: str,
    *,
    ttl_seconds: float = DEFAULT_LEASE_TTL_SECONDS,
    store: Optional[Any] = None,
) -> str:
    return await run_blocking(
        acquire_lease, name, operation, ttl_seconds=ttl_seconds, store=store
    )


async def release_leases_async(
    lease_ids: Iterable[str], *, store: Optional[Any] = None
) -> None:
    """Release off the event loop; a failure is logged, never raised."""
    ids = list(lease_ids)
    try:
        await run_blocking(release_leases, ids, store=store)
    except Exception:  # noqa: BLE001 - the write already happened
        logger.exception("Failed to release write leases: %s", sorted(ids))


class LeaseLostError(RuntimeError):
    """A lease this operation held was purged; it must stop writing."""


class LeaseSet:
    """The leases one operation holds, kept alive for as long as it runs."""

    def __init__(
        self,
        *,
        ttl_seconds: float = DEFAULT_LEASE_TTL_SECONDS,
        store: Optional[Any] = None,
    ) -> None:
        self._ttl = ttl_seconds
        self._store = store
        self._ids: List[str] = []
        self._lost: List[str] = []
        self._renewals = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    @property
    def lease_ids(self) -> List[str]:
        with self._lock:
            return list(self._ids)

    @property
    def lost(self) -> List[str]:
        """Leases this scope could not renew: they are no longer held."""
        with self._lock:
            return list(self._lost)

    def add(self, lease_id: str) -> None:
        with self._lock:
            self._ids.append(lease_id)
            if self._thread is None:
                self._thread = threading.Thread(
                    target=self._renew_loop,
                    name="kb-write-lease-heartbeat",
                    daemon=True,
                )
                self._thread.start()

    def _renew_loop(self) -> None:
        interval = max(0.05, float(self._ttl) / 3.0)
        while not self._stop.wait(interval):
            ids = self.lease_ids
            try:
                _renewed, lost = renew_leases(
                    ids, ttl_seconds=self._ttl, store=self._store
                )
            except Exception:  # noqa: BLE001 - keep trying until the scope ends
                # A renewal we cannot perform is a lease we cannot prove we hold.
                logger.exception("Failed to renew write leases")
                lost = ids
            if lost:
                with self._lock:
                    for lease_id in lost:
                        if lease_id not in self._lost:
                            self._lost.append(lease_id)
                logger.error(
                    "Lost %d of %d write leases; the operation must stop",
                    len(lost),
                    len(ids),
                )
            self._renewals += 1
            if self._renewals % GC_EVERY_RENEWALS == 0:
                # Off this thread: GC must not delay the next renewal.
                threading.Thread(
                    target=self._gc, name="kb-lease-gc", daemon=True
                ).start()

    def _gc(self) -> None:
        try:
            purge_expired_leases(store=self._store)
        except Exception:  # noqa: BLE001 - GC must never fail a scope
            logger.exception("Failed to purge expired write leases")

    def verify(self) -> None:
        """Raise when this scope no longer holds every lease it took.

        Writers call this at their commit points: a lease lost to a stalled
        process would otherwise let it write while the migration believes the
        collection is quiet.
        """
        lost = self.lost
        if lost:
            raise LeaseLostError(
                f"Lost {len(lost)} write lease(s): {', '.join(sorted(lost))}"
            )

    def stop(self) -> List[str]:
        """Stop renewing and return the ids; safe to call more than once."""
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=5.0)
        return self.lease_ids

    def release(self) -> None:
        layer = self.stop()
        with self._lock:
            self._ids.clear()
        try:
            release_leases(layer, store=self._store)
        except Exception:  # noqa: BLE001 - the write already happened
            logger.exception("Failed to release write leases: %s", sorted(layer))
