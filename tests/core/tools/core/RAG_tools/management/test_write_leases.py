"""Write leases: admitted operations stay visible to a migration across processes."""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any

import pytest

from xagent.core.tools.core.RAG_tools.core.exceptions import CollectionMigratingError
from xagent.core.tools.core.RAG_tools.management import write_leases as leases
from xagent.core.tools.core.RAG_tools.management.migration_state import (
    FROZEN,
    CollectionMigrationState,
    admitted_async,
    ensure_collection_writable,
    ensure_collection_writable_async,
    pending_collection,
    write_migration_state,
)
from xagent.core.tools.core.RAG_tools.management.write_leases import (
    DEFAULT_LEASE_TTL_SECONDS,
    LEASES_TABLE,
    LeaseSet,
    acquire_lease,
    live_leases,
    purge_expired_leases,
    release_leases,
    renew_leases,
    wait_for_no_leases,
)


def _freeze(name: str = "kb") -> None:
    write_migration_state(name, CollectionMigrationState(state=FROZEN))


def _insert_lease(name: str, lease_id: str, expires_at: datetime) -> None:
    from xagent.core.tools.core.RAG_tools.LanceDB.schema_manager import (
        _safe_close_table,
    )
    from xagent.core.tools.core.RAG_tools.storage.factory import get_metadata_store

    conn = get_metadata_store().get_raw_connection()
    leases._ensure_table(conn)
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    table = conn.open_table(LEASES_TABLE)
    try:
        table.add(
            [
                {
                    "name": name,
                    "lease_id": lease_id,
                    "operation": "old",
                    "owner": "host:1",
                    "created_at": now,
                    "expires_at": expires_at,
                }
            ]
        )
    finally:
        _safe_close_table(table)


def test_a_lease_is_live_until_it_is_released() -> None:
    lease_id = acquire_lease("kb", "test")

    live = live_leases("kb")
    assert [item.lease_id for item in live] == [lease_id]
    assert live[0].operation == "test"

    assert release_leases([lease_id]) == 1
    assert live_leases("kb") == []


def test_the_default_ttl_is_short_enough_for_a_crashed_writer_to_clear() -> None:
    assert DEFAULT_LEASE_TTL_SECONDS <= 300


def test_expired_leases_are_not_live() -> None:
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    long_gone = timedelta(seconds=leases.EXPIRY_SKEW_MARGIN_SECONDS + 60)
    _insert_lease("kb", "expired", now - long_gone)
    _insert_lease("kb", "live", now + timedelta(minutes=5))

    assert [lease.lease_id for lease in live_leases("kb")] == ["live"]


def test_renewal_extends_a_lease() -> None:
    lease_id = acquire_lease("kb", "test", ttl_seconds=60)

    assert renew_leases([lease_id], ttl_seconds=600) == (1, [])
    assert live_leases("kb")[0].expires_at is not None


def test_renewal_never_shortens_a_lease() -> None:
    lease_id = acquire_lease("kb", "test", ttl_seconds=600)
    before = live_leases("kb")[0].expires_at

    assert renew_leases([lease_id], ttl_seconds=60) == (1, [])

    assert live_leases("kb")[0].expires_at == before


def test_purging_removes_only_expired_rows_of_that_collection() -> None:
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    long_gone = timedelta(seconds=leases.EXPIRY_SKEW_MARGIN_SECONDS + 60)
    _insert_lease("kb", "expired", now - long_gone)
    _insert_lease("kb", "live", now + timedelta(minutes=5))
    _insert_lease("other", "expired-other", now - long_gone)

    assert purge_expired_leases("kb") == 1

    assert [lease.lease_id for lease in live_leases("kb")] == ["live"]
    assert live_leases("other") == []  # expired, still present, ignored


def test_waits_return_what_is_left_at_the_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    acquire_lease("kb", "test")
    clock = {"now": 100.0}
    monkeypatch.setattr(leases.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(
        leases.time, "sleep", lambda seconds: clock.update(now=clock["now"] + seconds)
    )

    remaining = wait_for_no_leases("kb", timeout_seconds=2.0, poll_seconds=1.0)

    assert [item.operation for item in remaining] == ["test"]


def test_a_waiter_returns_once_the_lease_is_released(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lease_id = acquire_lease("kb", "test")
    clock = {"now": 100.0}
    monkeypatch.setattr(leases.time, "monotonic", lambda: clock["now"])

    def _sleep(seconds: float) -> None:
        clock["now"] += seconds
        release_leases([lease_id])

    monkeypatch.setattr(leases.time, "sleep", _sleep)

    assert wait_for_no_leases("kb", timeout_seconds=10.0, poll_seconds=1.0) == []


def test_the_guard_writes_the_lease_before_it_reads_the_freeze_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xagent.core.tools.core.RAG_tools.management import migration_state as state

    observed: list = []

    def _read(name: str, store=None):
        observed.append([lease.lease_id for lease in live_leases(name)])
        return CollectionMigrationState()

    monkeypatch.setattr(state, "read_migration_state", _read)

    with pending_collection("kb"):
        ensure_collection_writable("kb")

    assert observed and len(observed[0]) == 1


async def test_the_async_guard_writes_the_lease_before_it_reads_the_freeze_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xagent.core.tools.core.RAG_tools.management import migration_state as state

    observed: list = []

    def _read(name: str, store=None):
        observed.append([lease.lease_id for lease in live_leases(name)])
        return CollectionMigrationState()

    monkeypatch.setattr(state, "read_migration_state", _read)

    with pending_collection("kb"):
        await ensure_collection_writable_async("kb")

    assert observed and len(observed[0]) == 1


def test_a_frozen_collection_refuses_and_leaves_no_lease() -> None:
    _freeze("kb")

    with pytest.raises(CollectionMigratingError):
        with pending_collection("kb"):
            ensure_collection_writable("kb")

    assert live_leases("kb") == []


def test_one_lease_covers_every_guard_call_of_the_operation() -> None:
    with pending_collection("kb"):
        ensure_collection_writable("kb")
        ensure_collection_writable("kb")
        ensure_collection_writable("kb")
        assert len(live_leases("kb")) == 1

    assert live_leases("kb") == []


def test_a_rename_scope_leases_both_names() -> None:
    with pending_collection("old", "new"):
        ensure_collection_writable("old")
        ensure_collection_writable("new")
        assert len(live_leases("old")) == 1
        assert len(live_leases("new")) == 1

    assert live_leases("old") == []
    assert live_leases("new") == []


def test_a_scope_that_raises_releases_its_leases() -> None:
    with pytest.raises(RuntimeError):
        with pending_collection("kb"):
            ensure_collection_writable("kb")
            raise RuntimeError("boom")

    assert live_leases("kb") == []


def test_a_guard_outside_a_pending_scope_takes_no_lease() -> None:
    ensure_collection_writable("kb")

    assert live_leases("kb") == []


async def test_admitted_async_holds_a_lease_until_the_scope_exits() -> None:
    async with admitted_async("kb"):
        assert len(live_leases("kb")) == 1

    assert live_leases("kb") == []


async def test_admitted_async_takes_the_lease_before_it_reads_the_freeze_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xagent.core.tools.core.RAG_tools.management import migration_state as state

    observed: list = []

    def _read(name: str, store=None):
        observed.append([lease.lease_id for lease in live_leases(name)])
        return CollectionMigrationState()

    monkeypatch.setattr(state, "read_migration_state", _read)

    async with admitted_async("kb"):
        pass

    assert observed and len(observed[0]) == 1


def test_a_waiter_clears_expired_rows_it_finds_nothing_live_for() -> None:
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    _insert_lease(
        "kb", "expired", now - timedelta(seconds=leases.EXPIRY_SKEW_MARGIN_SECONDS + 60)
    )

    assert wait_for_no_leases("kb", timeout_seconds=0.0) == []

    from xagent.core.tools.core.RAG_tools.LanceDB.schema_manager import (
        _safe_close_table,
    )
    from xagent.core.tools.core.RAG_tools.storage.factory import get_metadata_store

    conn = get_metadata_store().get_raw_connection()
    table = conn.open_table(LEASES_TABLE)
    try:
        assert table.count_rows() == 0
    finally:
        _safe_close_table(table)


async def test_admitted_async_inside_a_pending_scope_keeps_one_lease() -> None:
    with pending_collection("kb"):
        async with admitted_async("kb"):
            assert len(live_leases("kb")) == 1

    assert live_leases("kb") == []


def test_the_heartbeat_extends_a_held_lease(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(leases, "EXPIRY_SKEW_MARGIN_SECONDS", 0.0)
    held = LeaseSet(ttl_seconds=0.5)
    try:
        held.add(acquire_lease("kb", "test", ttl_seconds=0.5))
        before = live_leases("kb")[0].expires_at
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            rows = live_leases("kb")
            if rows and rows[0].expires_at > before:
                break
            time.sleep(0.05)
        assert live_leases("kb")[0].expires_at > before
    finally:
        held.release()


def test_the_lease_table_is_compacted_periodically(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compacted: list = []
    store = object()
    monkeypatch.setattr(leases, "_compact", compacted.append)
    monkeypatch.setattr(leases, "_store_connection", lambda s: _RealConn(s))
    monkeypatch.setattr(leases, "_acquire_count", leases.COMPACT_EVERY_ACQUIRES - 1)

    acquire_lease("kb", "test", store=store)

    deadline = time.monotonic() + 3.0
    while not compacted and time.monotonic() < deadline:
        time.sleep(0.01)
    assert compacted == [store]


class _RealConn:
    """The default store's connection, whatever store the caller passed."""

    def __init__(self, store: Any) -> None:
        from xagent.core.tools.core.RAG_tools.storage.factory import (
            get_metadata_store,
        )

        self._conn = get_metadata_store().get_raw_connection()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._conn, name)


def _optimize_recorder(calls: list) -> Any:
    class _Table:
        def optimize(self, **kwargs: Any) -> None:
            calls.append(kwargs)

    class _Conn:
        def open_table(self, name: str) -> Any:
            calls.append(name)
            return _Table()

    return SimpleNamespace(get_raw_connection=lambda: _Conn())


def test_compaction_uses_the_lease_store_and_a_short_retention() -> None:
    calls: list = []

    leases._compact(_optimize_recorder(calls))

    assert calls == [
        LEASES_TABLE,
        {"cleanup_older_than": leases.COMPACT_RETENTION},
    ]
    assert leases.COMPACT_RETENTION < timedelta(hours=1)


def test_compaction_skips_while_another_one_runs() -> None:
    calls: list = []
    assert leases._compact_lock.acquire(blocking=False)
    try:
        leases._compact(_optimize_recorder(calls))
    finally:
        leases._compact_lock.release()

    assert calls == []


def test_a_lease_just_past_expiry_still_counts_as_live() -> None:
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    _insert_lease("kb", "soon", now - timedelta(seconds=1))

    assert [lease.lease_id for lease in live_leases("kb")] == ["soon"]


def test_purging_keeps_a_lease_renewed_between_the_read_and_the_delete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    _insert_lease(
        "kb", "expired", now - timedelta(seconds=leases.EXPIRY_SKEW_MARGIN_SECONDS + 60)
    )
    from xagent.core.tools.core.RAG_tools.storage.factory import get_metadata_store

    conn = get_metadata_store().get_raw_connection()
    table = conn.open_table(LEASES_TABLE)
    state = {"renewed": False}
    original_delete = table.delete

    def _delete(where: str):
        if not state["renewed"]:
            state["renewed"] = True
            # Another host renews its lease before our delete lands.
            table.update(
                "lease_id = 'expired'",
                {"expires_at": now + timedelta(minutes=5)},
            )
        return original_delete(where)

    class _Conn:
        def __getattr__(self, name: str):
            return getattr(conn, name)

        def open_table(self, name: str):
            if name == LEASES_TABLE:
                return _Instrumented(table, _delete)
            return conn.open_table(name)

    monkeypatch.setattr(leases, "_store_connection", lambda store: _Conn())

    purged = purge_expired_leases("kb")

    assert state["renewed"] is True
    assert purged == 0
    assert [lease.lease_id for lease in live_leases("kb")] == ["expired"]


class _Instrumented:
    def __init__(self, table, delete) -> None:
        self._table = table
        self._delete = delete

    def __getattr__(self, name: str):
        return getattr(self._table, name)

    def delete(self, where: str):
        return self._delete(where)


def test_a_scope_reports_a_lease_it_could_not_renew(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(leases, "renew_leases", lambda ids, **kwargs: (0, list(ids)))
    held = LeaseSet(ttl_seconds=0.2)
    try:
        held.add(acquire_lease("kb", "test"))
        deadline = time.monotonic() + 5.0
        while not held.lost and time.monotonic() < deadline:
            time.sleep(0.02)
        assert held.lost
        with pytest.raises(leases.LeaseLostError):
            held.verify()
    finally:
        held.release()


def test_a_scope_that_keeps_its_leases_verifies_clean() -> None:
    held = LeaseSet()
    try:
        held.add(acquire_lease("kb", "test"))
        held.verify()
    finally:
        held.release()


def test_a_lease_insert_conflict_is_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    from xagent.core.tools.core.RAG_tools.storage.factory import get_metadata_store

    table = get_metadata_store().get_raw_connection()
    leases._ensure_table(table)
    real_table = table.open_table(LEASES_TABLE)
    calls = {"n": 0}
    original_add = real_table.add

    def _add(rows):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("CommitConflict")
        return original_add(rows)

    monkeypatch.setattr(real_table, "add", _add)

    class _Conn:
        def __getattr__(self, name):
            return getattr(table, name)

        def open_table(self, name):
            return real_table

    monkeypatch.setattr(leases, "_store_connection", lambda store: _Conn())

    assert acquire_lease("kb", "test")
    assert calls["n"] == 2


async def test_admitted_async_hands_its_admission_back_to_the_outer_scope() -> None:
    with pending_collection("kb"):
        async with admitted_async("kb"):
            pass
        _freeze()
        # The outer scope still owns the admission the inner guard granted.
        ensure_collection_writable("kb")

    with pytest.raises(CollectionMigratingError):
        ensure_collection_writable("kb")


async def test_holding_leases_covers_the_work_and_never_leaks() -> None:
    from xagent.core.tools.core.RAG_tools.management.migration_state import (
        holding_leases,
    )

    async with holding_leases("kb", "other", operation="collection-delete"):
        assert {lease.name for lease in live_leases("kb")} == {"kb"}
        assert len(live_leases("other")) == 1

    assert live_leases("kb") == []
    assert live_leases("other") == []


async def test_holding_leases_releases_on_failure() -> None:
    from xagent.core.tools.core.RAG_tools.management.migration_state import (
        holding_leases,
    )

    with pytest.raises(RuntimeError):
        async with holding_leases("kb", operation="collection-rename"):
            raise RuntimeError("rename exploded")

    assert live_leases("kb") == []


def test_leased_collections_covers_sync_work_and_never_leaks() -> None:
    from xagent.core.tools.core.RAG_tools.management.migration_state import (
        leased_collections,
    )

    with leased_collections("kb", operation="document-delete"):
        assert len(live_leases("kb")) == 1

    assert live_leases("kb") == []


async def test_a_collection_delete_holds_its_lease_for_the_whole_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from unittest.mock import AsyncMock, MagicMock

    from xagent.core.tools.core.RAG_tools.kb.coordinator import KBCoordinator
    from xagent.core.tools.core.RAG_tools.storage.factory import get_metadata_store

    coordinator = KBCoordinator()
    store = get_metadata_store()

    async def _context(request):
        return SimpleNamespace(metadata_store=store)

    seen = []
    handle = MagicMock()
    handle.list_collection_documents = MagicMock(
        side_effect=lambda *args, **kwargs: seen.append(len(live_leases("kb"))) or []
    )
    handle.delete_collection_data = MagicMock(return_value={})
    handle.delete_collection_config = MagicMock()

    monkeypatch.setattr(coordinator, "get_context", _context)
    monkeypatch.setattr(coordinator, "_ensure_writable", AsyncMock())
    monkeypatch.setattr(coordinator, "open_collection", AsyncMock(return_value=handle))

    await coordinator.delete_collection("kb", user_id=1, is_admin=True)

    assert seen == [1]
    assert live_leases("kb") == []


async def test_a_collection_rename_leases_both_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from unittest.mock import AsyncMock, MagicMock

    from xagent.core.tools.core.RAG_tools.kb.coordinator import KBCoordinator
    from xagent.core.tools.core.RAG_tools.storage.factory import get_metadata_store

    coordinator = KBCoordinator()
    store = get_metadata_store()

    async def _context(request):
        return SimpleNamespace(metadata_store=store)

    during = []
    handle = MagicMock()

    def _rename_data(*args, **kwargs):
        during.append((len(live_leases("old")), len(live_leases("new"))))
        return []

    handle.rename_collection_data = MagicMock(side_effect=_rename_data)
    handle.rename_collection_status = AsyncMock(return_value=[])
    handle.rename_collection_metadata = AsyncMock()

    monkeypatch.setattr(coordinator, "get_context", _context)
    monkeypatch.setattr(coordinator, "_ensure_writable", AsyncMock())
    monkeypatch.setattr(coordinator, "open_collection", AsyncMock(return_value=handle))

    await coordinator.rename_collection("old", "new", user_id=1, is_admin=False)

    assert during == [(1, 1)]
    assert live_leases("old") == []
    assert live_leases("new") == []


def test_a_document_delete_holds_its_lease_for_the_whole_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from unittest.mock import MagicMock

    from xagent.core.tools.core.RAG_tools.management import collections as manager

    seen = []

    class _Store:
        def count_rows(self, **kwargs):
            seen.append(len(live_leases("kb")))
            return 0

        def __getattr__(self, name):
            return MagicMock()

    monkeypatch.setattr(manager, "get_vector_index_store", lambda: _Store())

    manager._delete_document_impl("kb", "doc-1", 7, False, coordinator=MagicMock())

    assert seen == [1]
    assert live_leases("kb") == []


async def test_admitted_async_does_not_leave_the_name_admitted() -> None:
    async with admitted_async("kb"):
        pass

    _freeze()
    with pytest.raises(CollectionMigratingError):
        ensure_collection_writable("kb")


def test_the_heartbeat_collects_expired_rows_of_other_collections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    purged = []
    monkeypatch.setattr(leases, "renew_leases", lambda ids, **kwargs: (len(ids), []))
    monkeypatch.setattr(
        leases, "purge_expired_leases", lambda name=None, **kwargs: purged.append(name)
    )
    monkeypatch.setattr(leases, "GC_EVERY_RENEWALS", 1)
    held = LeaseSet(ttl_seconds=0.2)
    try:
        held.add(acquire_lease("kb", "test"))
        deadline = time.monotonic() + 5.0
        while not purged and time.monotonic() < deadline:
            time.sleep(0.02)
        assert purged == [None]
    finally:
        held.release()


def test_compaction_does_not_delay_the_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(leases, "_compact", lambda store: time.sleep(1.0))
    monkeypatch.setattr(leases, "_acquire_count", leases.COMPACT_EVERY_ACQUIRES - 1)

    started = time.monotonic()
    acquire_lease("kb", "test")
    elapsed = time.monotonic() - started

    assert elapsed < 0.5


def test_renewing_an_expired_lease_reports_it_lost_instead_of_reviving_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    _insert_lease("kb", "expired", now - timedelta(seconds=60))

    renewed, lost = renew_leases(["expired"], ttl_seconds=120)

    assert renewed == 0
    assert lost == ["expired"]
    rows = live_leases("kb")
    assert rows == []


def test_one_failed_renewal_is_retried_instead_of_losing_the_leases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list = []
    real_renew = leases.renew_leases

    def _flaky(ids: Any, **kwargs: Any) -> Any:
        calls.append(len(ids))
        if len(calls) == 1:
            raise OSError("commit conflict")
        return real_renew(ids, **kwargs)

    monkeypatch.setattr(leases, "renew_leases", _flaky)
    held = LeaseSet(ttl_seconds=0.6)
    try:
        held.add(acquire_lease("kb", "test"))
        deadline = time.monotonic() + 5.0
        while len(calls) < 3 and time.monotonic() < deadline:
            time.sleep(0.02)
        assert len(calls) >= 3
        assert held.lost == []
        held.verify()
    finally:
        held.release()


def test_renewals_failing_past_the_ttl_fail_the_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        leases,
        "renew_leases",
        lambda ids, **kwargs: (_ for _ in ()).throw(OSError("gone")),
    )
    held = LeaseSet(ttl_seconds=0.2)
    try:
        held.add(acquire_lease("kb", "test"))
        time.sleep(0.4)
        assert held.lost == []
        with pytest.raises(leases.LeaseLostError):
            held.verify()
    finally:
        held.release()


def test_a_stalled_heartbeat_fails_the_check_at_the_local_deadline() -> None:
    held = LeaseSet(ttl_seconds=120)
    try:
        held.add(acquire_lease("kb", "test"))
        held.verify()
        held._alive_until = time.monotonic() - 1
        with pytest.raises(leases.LeaseLostError):
            held.verify()
    finally:
        held.release()


def test_a_set_that_owns_no_lease_passes_however_long_it_opens() -> None:
    held = LeaseSet(ttl_seconds=0.2)
    try:
        held._alive_until = time.monotonic() - 1
        held.verify()
        # Its own lifetime starts with the first row it owns.
        held.add(acquire_lease("kb", "test"))
        assert held._alive_until > time.monotonic()
        held.verify()
    finally:
        held.release()


def test_a_nested_scope_that_reuses_the_lease_keeps_passing_after_its_ttl() -> None:
    from xagent.core.tools.core.RAG_tools.management import migration_state

    with migration_state.leased_collections("kb", operation="outer"):
        with migration_state.leased_collections("kb", operation="inner") as inner:
            inner._alive_until = time.monotonic() - 1
            migration_state.verify_active_leases("kb")


def test_a_second_cancel_still_stops_the_heartbeat_and_releases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xagent.core.tools.core.RAG_tools.management import migration_state

    real = migration_state.acquire_lease_async

    async def _slow(name: str, operation: str, **kwargs: Any) -> str:
        await asyncio.sleep(0.2)
        return await real(name, operation, **kwargs)

    monkeypatch.setattr(migration_state, "acquire_lease_async", _slow)

    async def _run() -> None:
        async def _hold() -> None:
            async with migration_state.holding_leases("kb", operation="test"):
                pytest.fail("cancelled before admission")  # pragma: no cover

        task = asyncio.create_task(_hold())
        await asyncio.sleep(0.05)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(_run())
    deadline = time.monotonic() + 5.0
    while live_leases("kb") and time.monotonic() < deadline:
        time.sleep(0.02)
    assert live_leases("kb") == []


def test_a_cancelled_scope_stops_its_heartbeat_before_any_await(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xagent.core.tools.core.RAG_tools.management import migration_state

    async def _slow_blocking(fn: Any, *args: Any, **kwargs: Any) -> Any:
        await asyncio.sleep(0.5)
        return fn(*args, **kwargs)

    monkeypatch.setattr(migration_state, "run_blocking", _slow_blocking)
    opened: list = []

    async def _run() -> None:
        async def _hold() -> None:
            async with migration_state._lease_scope_async() as lease_set:
                opened.append(lease_set)
                await asyncio.sleep(10)

        task = asyncio.create_task(_hold())
        await asyncio.sleep(0)
        task.cancel()
        # A second cancel lands while the cleanup is still joining.
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert opened[-1]._stop.is_set()

    asyncio.run(_run())


def test_a_cancelled_acquire_releases_in_its_own_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xagent.core.tools.core.RAG_tools.management import migration_state

    seen: list = []
    marker = object()

    async def _release(ids: Any, store: Any = None) -> None:
        seen.append((list(ids), store))

    monkeypatch.setattr(migration_state, "release_leases_async", _release)

    async def _run() -> None:
        task: asyncio.Future = asyncio.get_running_loop().create_future()
        task.set_result("lease-id")
        migration_state._release_after_cancel(task, store=marker)
        await asyncio.sleep(0.05)

    asyncio.run(_run())

    assert seen == [(["lease-id"], marker)]


def test_the_release_tasks_are_kept_alive_until_they_run() -> None:
    from xagent.core.tools.core.RAG_tools.management import migration_state

    async def _run() -> None:
        task = migration_state._release_tracked(["not-a-lease"], None)
        assert task in migration_state._RELEASE_TASKS
        await task

    asyncio.run(_run())
    assert migration_state._RELEASE_TASKS == set()


def test_releases_use_the_store_the_lease_was_taken_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xagent.core.tools.core.RAG_tools.management import migration_state

    seen: list = []

    async def _spy(name: str, operation: str, **kwargs: Any) -> str:
        seen.append(kwargs.get("store"))
        return "lease-id"

    released: list = []

    async def _release(ids: Any, **kwargs: Any) -> None:
        released.append(kwargs.get("store"))

    monkeypatch.setattr(migration_state, "acquire_lease_async", _spy)
    monkeypatch.setattr(migration_state, "release_leases_async", _release)
    marker = object()

    async def _run() -> None:
        async with migration_state.holding_leases("kb", operation="test", store=marker):
            pass

    asyncio.run(_run())

    assert seen == [marker]
    assert released == [marker]


def test_a_released_set_fails_its_check_and_takes_no_more_leases() -> None:
    held = LeaseSet(ttl_seconds=120)
    held.add(acquire_lease("kb", "test"))
    held.verify()
    held.release()

    with pytest.raises(leases.LeaseLostError):
        held.verify()
    with pytest.raises(leases.LeaseLostError):
        held.add("late")


def test_work_outliving_its_scope_cannot_pass_the_commit_gate() -> None:
    from contextvars import copy_context

    from xagent.core.tools.core.RAG_tools.management import migration_state

    with pending_collection("kb"):
        ensure_collection_writable("kb")
        leftover = copy_context()

    with pytest.raises(CollectionMigratingError):
        leftover.run(migration_state.verify_active_leases, "kb")


def test_one_failed_release_still_releases_the_other_leases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ids = [acquire_lease(name, "rename") for name in ("a", "b", "c")]

    class _Flaky:
        def __init__(self, table: Any) -> None:
            self._table = table

        def delete(self, where: str) -> Any:
            if ids[0] in where:
                raise OSError("commit conflict")
            return self._table.delete(where)

        def __getattr__(self, name: str) -> Any:
            return getattr(self._table, name)

    real = leases._store_connection

    class _Conn:
        def __init__(self, conn: Any) -> None:
            self._conn = conn

        def open_table(self, name: str) -> Any:
            return _Flaky(self._conn.open_table(name))

        def __getattr__(self, name: str) -> Any:
            return getattr(self._conn, name)

    monkeypatch.setattr(leases, "_store_connection", lambda s: _Conn(real(s)))

    assert release_leases(ids) == 2
    assert live_leases("b") == [] and live_leases("c") == []
    assert [lease.lease_id for lease in live_leases("a")] == [ids[0]]


def test_a_renewal_does_not_revive_a_row_that_expired_after_the_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lease_id = acquire_lease("kb", "test")
    _expire_lease("kb", lease_id, -1)
    real = leases._store_connection

    class _Racing:
        def __init__(self, table: Any) -> None:
            self._table = table

        def update(self, where: str, values: Any) -> Any:
            # The row expires between the renewal's read and its write.
            _expire_lease("kb", lease_id, 30)
            return self._table.update(where, values)

        def __getattr__(self, name: str) -> Any:
            return getattr(self._table, name)

    class _Conn:
        def __init__(self, conn: Any) -> None:
            self._conn = conn

        def open_table(self, name: str) -> Any:
            return _Racing(self._conn.open_table(name))

        def __getattr__(self, name: str) -> Any:
            return getattr(self._conn, name)

    monkeypatch.setattr(leases, "_store_connection", lambda s: _Conn(real(s)))

    assert renew_leases([lease_id]) == (0, [lease_id])
    monkeypatch.setattr(leases, "_store_connection", real)
    assert live_leases("kb") == [] or live_leases("kb")[0].expires_at < datetime.now(
        timezone.utc
    ).replace(tzinfo=None)


def test_a_cancelled_acquire_releases_the_row_it_inserted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xagent.core.tools.core.RAG_tools.management import migration_state

    real = migration_state.acquire_lease_async

    async def _slow(name: str, operation: str, **kwargs: Any) -> str:
        await asyncio.sleep(0.2)
        return await real(name, operation, **kwargs)

    monkeypatch.setattr(migration_state, "acquire_lease_async", _slow)

    async def _run() -> None:
        async def _hold() -> None:
            async with migration_state.holding_leases("kb", operation="test"):
                pytest.fail("cancelled before admission")  # pragma: no cover

        task = asyncio.create_task(_hold())
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            await asyncio.sleep(0.05)
            if live_leases("kb") == [] and len(_inserted) > 0:
                break

    _inserted: list = []
    monkeypatch.setattr(
        migration_state,
        "release_leases_async",
        _tracking_release(migration_state.release_leases_async, _inserted),
    )
    asyncio.run(_run())

    assert _inserted
    assert live_leases("kb") == []


def _tracking_release(real: Any, seen: list) -> Any:
    async def _release(ids: Any, **kwargs: Any) -> Any:
        ids = list(ids)
        seen.extend(ids)
        return await real(ids, **kwargs)

    return _release


def test_a_lost_handler_lease_stops_a_nested_holding_leases_scope() -> None:
    from xagent.core.tools.core.RAG_tools.management import migration_state

    async def _run() -> None:
        async with migration_state.pending_collection_async("kb"):
            await migration_state.ensure_collection_writable_async("kb")
            async with migration_state.holding_leases("kb", operation="delete"):
                migration_state._LEASE_SETS.get()[0]._lost.append("gone")
                with pytest.raises(CollectionMigratingError):
                    migration_state.verify_active_leases("kb")

    asyncio.run(_run())


def test_a_lease_slightly_past_expiry_still_counts_as_held(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    _insert_lease("kb", "just-expired", now - timedelta(seconds=1))

    assert [lease.lease_id for lease in live_leases("kb")] == ["just-expired"]


def _mark_current_scope_lost() -> str:
    """Exactly the state a lost lease leaves: the set records the id."""
    from xagent.core.tools.core.RAG_tools.management import migration_state

    lease_set = migration_state._LEASE_SETS.get()[-1]
    lease_id = migration_state._LEASES.get()[-1][1]
    lease_set._lost.append(lease_id)
    return lease_id


@pytest.mark.parametrize("scenario", ["delete", "rename", "document-delete"])
def test_a_lost_outer_lease_stops_a_nested_lease_scope(scenario: str) -> None:
    from xagent.core.tools.core.RAG_tools.core.exceptions import (
        CollectionMigratingError,
    )
    from xagent.core.tools.core.RAG_tools.management import migration_state

    with pending_collection("kb"):
        ensure_collection_writable("kb")
        _mark_current_scope_lost()

        with pytest.raises(CollectionMigratingError):
            with migration_state.leased_collections("kb", operation=scenario):
                pytest.fail("a nested scope must not start")  # pragma: no cover

        async def _nested() -> None:
            async with migration_state.holding_leases("kb", operation=scenario):
                pytest.fail("a nested scope must not start")  # pragma: no cover

        with pytest.raises(CollectionMigratingError):
            asyncio.run(_nested())


def test_a_nested_scope_reuses_the_lease_instead_of_taking_a_second_one() -> None:
    from xagent.core.tools.core.RAG_tools.management import migration_state

    with pending_collection("kb"):
        ensure_collection_writable("kb")
        outer = {name: lease_id for name, lease_id in migration_state._LEASES.get()}

        with migration_state.leased_collections("kb", operation="inner"):
            inner = {name: lease_id for name, lease_id in migration_state._LEASES.get()}

    assert outer["kb"] == inner["kb"]


def test_verify_active_leases_checks_every_enclosing_scope() -> None:
    from xagent.core.tools.core.RAG_tools.core.exceptions import (
        CollectionMigratingError,
    )
    from xagent.core.tools.core.RAG_tools.management import migration_state

    with migration_state.leased_collections("kb", operation="outer") as outer:
        with migration_state.leased_collections("other", operation="inner"):
            outer._lost.append("lost-id")
            with pytest.raises(CollectionMigratingError):
                migration_state.verify_active_leases("other")


def test_a_lost_lease_fails_the_active_scope_check() -> None:
    from xagent.core.tools.core.RAG_tools.core.exceptions import (
        CollectionMigratingError,
    )
    from xagent.core.tools.core.RAG_tools.management import migration_state

    with migration_state.leased_collections("kb", operation="test"):
        _mark_current_scope_lost()
        with pytest.raises(CollectionMigratingError):
            migration_state.verify_active_leases("kb")


def test_a_lease_that_cannot_be_inserted_stops_the_operation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xagent.core.tools.core.RAG_tools.management import migration_state

    def _explode(*args: Any, **kwargs: Any) -> str:
        raise RuntimeError("lease table unavailable")

    monkeypatch.setattr(migration_state, "acquire_lease", _explode)
    with pytest.raises(RuntimeError, match="lease table unavailable"):
        with migration_state.leased_collections("kb", operation="test"):
            pytest.fail("must not run without its lease")  # pragma: no cover


def test_pending_collection_async_releases_its_lease() -> None:
    from xagent.core.tools.core.RAG_tools.management import migration_state

    async def _run() -> None:
        async with migration_state.pending_collection_async("kb"):
            await migration_state.ensure_collection_writable_async("kb")
            assert migration_state.holds_lease("kb")
        assert live_leases("kb") == []

    asyncio.run(_run())


def test_a_scope_whose_lease_was_lost_fails_its_check() -> None:
    from xagent.core.tools.core.RAG_tools.core.exceptions import (
        CollectionMigratingError,
    )
    from xagent.core.tools.core.RAG_tools.management import migration_state

    held = LeaseSet(ttl_seconds=0.2)
    # A row that does not exist: the first renewal finds it gone.
    held.add("not-a-real-lease")
    set_token = migration_state._LEASE_SETS.set((held,))
    leases_token = migration_state._LEASES.set((("kb", "not-a-real-lease"),))
    try:
        deadline = time.monotonic() + 5.0
        while not held.lost and time.monotonic() < deadline:
            time.sleep(0.02)
        with pytest.raises(CollectionMigratingError):
            migration_state.verify_active_leases("kb")
    finally:
        migration_state._LEASES.reset(leases_token)
        migration_state._LEASE_SETS.reset(set_token)
        held.release()


def _expire_lease(name: str, lease_id: str, seconds_ago: float) -> None:
    """Move an existing lease's expiry into the past."""
    from xagent.core.tools.core.RAG_tools.storage.factory import get_metadata_store

    conn = get_metadata_store().get_raw_connection()
    table = conn.open_table(LEASES_TABLE)
    edge = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(
        seconds=seconds_ago
    )
    table.update(f"lease_id = '{lease_id}'", {"expires_at": edge})


# --- real LanceDB: the lease table must work through the store's own API ---


def test_lease_lifecycle_on_a_real_table() -> None:
    lease_id = acquire_lease("kb", "document-ingestion")

    live = live_leases("kb")
    assert [lease.lease_id for lease in live] == [lease_id]
    assert live[0].operation == "document-ingestion"

    renewed, lost = renew_leases([lease_id])
    assert (renewed, lost) == (1, [])
    assert live_leases("kb")[0].expires_at > live[0].expires_at

    assert release_leases([lease_id]) == 1
    assert live_leases("kb") == []
    assert renew_leases([lease_id]) == (0, [lease_id])


def test_renewing_an_expired_row_counts_it_lost_on_a_real_table() -> None:
    lease_id = acquire_lease("kb", "document-ingestion")
    _expire_lease("kb", lease_id, 30)

    assert renew_leases([lease_id]) == (0, [lease_id])
    assert renew_leases([lease_id]) == (0, [lease_id])


def test_a_lease_a_margin_past_its_expiry_still_reads_as_live() -> None:
    lease_id = acquire_lease("kb", "document-ingestion")
    _expire_lease("kb", lease_id, 1)

    assert [lease.lease_id for lease in live_leases("kb")] == [lease_id]


def test_purge_removes_only_expired_leases_on_a_real_table() -> None:
    alive = acquire_lease("kb", "document-ingestion")
    stale = acquire_lease("other", "document-ingestion")
    _expire_lease("other", stale, 300)

    assert purge_expired_leases() == 1
    assert [lease.lease_id for lease in live_leases("kb")] == [alive]
    assert live_leases("other") == []


def test_the_lease_set_notices_a_real_lost_row() -> None:
    lease_set = LeaseSet(ttl_seconds=0.05)
    lease_id = acquire_lease("kb", "write-guard")
    lease_set.add(lease_id)
    release_leases([lease_id])

    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and not lease_set.lost:
        time.sleep(0.02)
    lease_set.stop()

    assert lease_set.lost == [lease_id]
    with pytest.raises(leases.LeaseLostError):
        lease_set.verify()
