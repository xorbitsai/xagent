"""Async LanceDB connections must be reachable from every event loop.

The stores that use them are process-wide singletons, while the codebase keeps
creating fresh event loops (``asyncio.run`` in the coordinator's sync wrappers,
the parse path, collection_manager). Guarding a per-instance ``_async_conn``
with an ``asyncio.Lock`` deadlocked that shape: the lock binds to whichever
loop first contended it, and the release path wakes that loop's future without
``call_soon_threadsafe``, so every other loop waits forever (#2200).

Every test here joins with a timeout and asserts the thread finished, so a
regression fails the suite instead of hanging CI.
"""

from __future__ import annotations

import asyncio
import os
import threading
from typing import Any, List
from unittest.mock import patch

import pytest

from xagent.providers.vector_store import lancedb as lancedb_module
from xagent.providers.vector_store.lancedb import (
    clear_connection_cache,
    get_async_connection_from_env,
)

JOIN_TIMEOUT = 5


@pytest.fixture(autouse=True)
def _isolate_pool(tmp_path, monkeypatch):
    monkeypatch.setenv("LANCEDB_DIR", str(tmp_path))
    clear_connection_cache()
    yield
    clear_connection_cache()


class _FakeAsyncConnection:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _slow_connect_async(delay: float = 0.2):
    """A connect_async whose latency is wide enough for openers to collide."""
    created: List[_FakeAsyncConnection] = []

    async def connect_async(_uri: str) -> _FakeAsyncConnection:
        await asyncio.sleep(delay)
        conn = _FakeAsyncConnection()
        created.append(conn)
        return conn

    return connect_async, created


def _run_in_own_loop(coro_factory, results: list, errors: list) -> threading.Thread:
    def target() -> None:
        try:
            results.append(asyncio.run(coro_factory()))
        except BaseException as exc:  # noqa: BLE001
            errors.append(f"{type(exc).__name__}: {exc}")

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    return thread


def test_concurrent_init_from_two_loops_does_not_deadlock() -> None:
    """Two threads, two loops, one uninitialized pool entry: both must return."""
    connect_async, created = _slow_connect_async()
    results: List[Any] = []
    errors: List[str] = []

    with patch.object(lancedb_module.lancedb, "connect_async", connect_async):
        threads = [
            _run_in_own_loop(get_async_connection_from_env, results, errors)
            for _ in range(2)
        ]
        for thread in threads:
            thread.join(timeout=JOIN_TIMEOUT)

    assert not [t for t in threads if t.is_alive()], (
        "a thread never returned from async connection init -- deadlock"
    )
    assert not errors, f"async connection init raised: {errors}"
    assert len(results) == 2
    assert results[0] is results[1], "both loops must share one pooled connection"
    # Without this the test would silently pass on plain sequential reuse,
    # never having put two openers in the window at all.
    assert len(created) == 2, (
        f"the two callers did not overlap ({len(created)} connect_async calls)"
    )
    # The opener that lost the insert race must not leak.
    leaked = [c for c in created if c is not results[0] and not c.closed]
    assert not leaked, f"{len(leaked)} superseded connections were left open"


def test_many_loops_reuse_one_pooled_connection() -> None:
    """A cached connection is handed to loops that had no part in creating it."""
    connect_async, created = _slow_connect_async(delay=0.0)
    results: List[Any] = []
    errors: List[str] = []

    with patch.object(lancedb_module.lancedb, "connect_async", connect_async):
        for _ in range(6):
            thread = _run_in_own_loop(get_async_connection_from_env, results, errors)
            thread.join(timeout=JOIN_TIMEOUT)
            assert not thread.is_alive()

    assert not errors, f"async connection init raised: {errors}"
    assert len(results) == 6
    assert all(conn is results[0] for conn in results)
    assert len(created) == 1, f"connect_async ran {len(created)} times, expected 1"


def test_clear_connection_cache_forces_a_fresh_async_connection() -> None:
    """``reset_rag_storage_for_tests`` relies on this to reset *all* state.

    Before the pool existed, the async connection went away with the store
    instance that `StorageFactory.reset_all()` dropped; now it outlives that,
    so clearing the cache is what forces the next connect. The pooled
    connection is deliberately *not* closed -- stores hold theirs across
    awaits, so closing here would strand an in-flight coroutine.
    """
    connect_async, created = _slow_connect_async(delay=0.0)
    results: List[Any] = []
    errors: List[str] = []

    with patch.object(lancedb_module.lancedb, "connect_async", connect_async):
        thread = _run_in_own_loop(get_async_connection_from_env, results, errors)
        thread.join(timeout=JOIN_TIMEOUT)
        assert not thread.is_alive()
        assert not errors and len(results) == 1

        clear_connection_cache()
        assert not created[0].closed, "a pooled connection must not be closed"

        thread = _run_in_own_loop(get_async_connection_from_env, results, errors)
        thread.join(timeout=JOIN_TIMEOUT)
        assert not thread.is_alive()

    assert not errors, f"async connection init raised: {errors}"
    assert len(created) == 2, "cache clear must force a fresh connect_async"
    assert results[1] is not results[0]


def test_store_async_methods_share_the_pool_across_loops() -> None:
    """The shape #2200 actually breaks: one store singleton, several loops."""
    from xagent.core.tools.core.RAG_tools.storage.lancedb_stores import (
        LanceDBVectorIndexStore,
    )

    connect_async, created = _slow_connect_async()
    results: List[Any] = []
    errors: List[str] = []
    first = LanceDBVectorIndexStore()
    second = LanceDBVectorIndexStore()

    with patch.object(lancedb_module.lancedb, "connect_async", connect_async):
        threads = [
            _run_in_own_loop(store._get_async_connection, results, errors)
            for store in (first, second, first)
        ]
        for thread in threads:
            thread.join(timeout=JOIN_TIMEOUT)

    assert not [t for t in threads if t.is_alive()], (
        "a store's async connection init never returned -- deadlock"
    )
    assert not errors, f"store async init raised: {errors}"
    assert len(results) == 3
    assert all(conn is results[0] for conn in results)
    assert not hasattr(first, "_async_lock")
    assert not hasattr(second, "_async_lock")


def test_real_async_connection_survives_a_second_event_loop() -> None:
    """The premise the pool rests on, checked against real lancedb.

    Every other test here fakes ``connect_async``, so none of them would notice
    if a lancedb upgrade started binding connections to their creating loop.
    This one does the crossing for real: build and write on one loop, then read
    and write the same connection from another after the first is gone.
    """
    import lancedb

    holder: dict = {}
    errors: List[str] = []

    async def create() -> int:
        conn = await lancedb.connect_async(os.environ["LANCEDB_DIR"])
        table = await conn.create_table("crossloop", data=[{"x": 1}])
        holder["conn"] = conn
        return await table.count_rows()

    async def reuse() -> int:
        table = await holder["conn"].open_table("crossloop")
        await table.add([{"x": 2}])
        return await table.count_rows()

    first: List[Any] = []
    thread = _run_in_own_loop(create, first, errors)
    thread.join(timeout=JOIN_TIMEOUT)
    assert not thread.is_alive()
    assert not errors, f"creating loop raised: {errors}"

    second: List[Any] = []
    thread = _run_in_own_loop(reuse, second, errors)
    thread.join(timeout=JOIN_TIMEOUT)
    assert not thread.is_alive(), "reusing the connection on a second loop hung"
    assert not errors, f"second loop raised: {errors}"
    assert first == [1] and second == [2]


def test_no_await_inside_the_pool_lock() -> None:
    """``_async_cache_lock`` is a threading lock, so awaiting under it hangs.

    A regression here cannot be caught at runtime: the blocked thread stops
    the event loop too, so ``asyncio.wait_for`` never gets to fire and the
    suite deadlocks instead of failing. Hence the static check.
    """
    import ast
    import inspect
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(get_async_connection_from_env)))

    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.With, ast.AsyncWith)):
            continue
        guards = {ast.unparse(item.context_expr) for item in node.items}
        if "_async_cache_lock" not in guards:
            continue
        for inner in ast.walk(node):
            if isinstance(inner, (ast.Await, ast.AsyncWith, ast.AsyncFor)):
                offenders.append(f"line {inner.lineno}: {ast.unparse(inner)[:60]}")

    assert not offenders, (
        "await under the threading lock would deadlock the loop: "
        + "; ".join(offenders)
    )


@pytest.mark.asyncio
async def test_concurrent_init_on_one_loop_shares_one_connection() -> None:
    """Gathered callers on a single loop converge on one pooled connection."""
    connect_async, created = _slow_connect_async(delay=0.2)

    with patch.object(lancedb_module.lancedb, "connect_async", connect_async):
        conns = await asyncio.wait_for(
            asyncio.gather(*(get_async_connection_from_env() for _ in range(5))),
            timeout=JOIN_TIMEOUT,
        )

    assert all(conn is conns[0] for conn in conns)
    survivors = [c for c in created if c is conns[0] or not c.closed]
    assert survivors == [conns[0]], "every superseded connection must be closed"


def test_failed_connect_leaves_the_pool_untouched() -> None:
    """A failed connect must propagate and cache nothing."""

    async def failing_connect(_uri: str) -> Any:
        raise RuntimeError("connect refused")

    results: List[Any] = []
    errors: List[str] = []

    with patch.object(lancedb_module.lancedb, "connect_async", failing_connect):
        thread = _run_in_own_loop(get_async_connection_from_env, results, errors)
        thread.join(timeout=JOIN_TIMEOUT)

    assert not thread.is_alive()
    assert errors == ["RuntimeError: connect refused"]
    assert not results
    assert not lancedb_module._async_connection_cache, "a failure must not cache"

    # The pool still works afterwards.
    connect_async, created = _slow_connect_async(delay=0.0)
    with patch.object(lancedb_module.lancedb, "connect_async", connect_async):
        thread = _run_in_own_loop(get_async_connection_from_env, results, errors)
        thread.join(timeout=JOIN_TIMEOUT)
    assert not thread.is_alive()
    assert len(results) == 1 and len(created) == 1


@pytest.mark.parametrize("env_value", [None, "", "   "])
def test_both_planes_agree_on_the_directory_without_the_env_var(
    env_value, monkeypatch
) -> None:
    """Sync and async must resolve the same dir, including with no env var.

    This is where the two planes used to diverge: the async side read
    ``LANCEDB_DIR`` directly with its own default while the sync side went
    through ``get_default_lancedb_dir()``. Every other test here sets the
    variable, so the unset and empty cases would otherwise go unchecked.
    """
    from xagent.providers.vector_store.lancedb import LanceDBConnectionManager

    if env_value is None:
        monkeypatch.delenv("LANCEDB_DIR", raising=False)
    else:
        monkeypatch.setenv("LANCEDB_DIR", env_value)

    manager = LanceDBConnectionManager()
    if env_value is not None and env_value.strip() == "":
        with pytest.raises(ValueError, match="is empty"):
            manager.resolve_dir_from_env()
        return

    resolved = manager.resolve_dir_from_env()
    assert resolved == manager._normalize_dirpath(manager.get_default_lancedb_dir())

    connect_async, created = _slow_connect_async(delay=0.0)
    results: List[Any] = []
    errors: List[str] = []
    with patch.object(lancedb_module.lancedb, "connect_async", connect_async):
        thread = _run_in_own_loop(get_async_connection_from_env, results, errors)
        thread.join(timeout=JOIN_TIMEOUT)

    assert not errors, f"async init raised: {errors}"
    assert list(lancedb_module._async_connection_cache) == [resolved], (
        "the async pool must key on the same dir the sync plane resolves"
    )
