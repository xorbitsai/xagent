"""Generation identity and the serving generation record."""

from __future__ import annotations

import asyncio

import pytest

from tests.shared.db_teardown import drop_all_tables
from xagent.web.models.database import get_db, get_engine, init_db
from xagent.web.services import generation
from xagent.web.services.generation import (
    current_generation,
    is_serving,
    serving_generation,
    set_serving_generation,
    start_allowed,
    tick_allowed,
)


@pytest.fixture()
def db(tmp_path, monkeypatch):
    monkeypatch.delenv(generation.GENERATION_ENV, raising=False)
    init_db(db_url=f"sqlite:///{tmp_path / 'generation.db'}")
    session = next(get_db())
    try:
        yield session
    finally:
        session.close()
        drop_all_tables(get_engine())


def test_generations_off_means_every_check_passes(db, monkeypatch) -> None:
    assert current_generation() is None
    assert is_serving()
    assert start_allowed("x")
    assert set_serving_generation("blue", None)
    # Another generation serving changes nothing while generations are off.
    assert is_serving()
    monkeypatch.setenv(generation.GENERATION_ENV, "  ")
    assert current_generation() is None
    assert is_serving()


def test_missing_row_passes_then_row_decides(db, monkeypatch) -> None:
    monkeypatch.setenv(generation.GENERATION_ENV, "green")
    assert serving_generation() is None
    assert is_serving()
    assert set_serving_generation("blue", None)
    assert serving_generation() == "blue"
    assert not is_serving()
    assert start_allowed("x")
    monkeypatch.setenv(generation.GENERATION_ENV, "blue")
    assert is_serving()
    assert start_allowed("x")


def test_compare_and_swap_never_overwrites_an_unexpected_value(db) -> None:
    assert set_serving_generation("blue", None)
    # Creating again while the row exists loses.
    assert not set_serving_generation("green", None)
    # A stale expectation loses and leaves the value alone.
    assert not set_serving_generation("green", "other")
    assert serving_generation() == "blue"
    assert set_serving_generation("green", "blue")
    assert serving_generation() == "green"
    assert not set_serving_generation("blue", "blue")
    with pytest.raises(ValueError):
        set_serving_generation("", "green")


def test_database_error_is_never_serving(db, monkeypatch) -> None:
    monkeypatch.setenv(generation.GENERATION_ENV, "green")

    def boom() -> str | None:
        raise RuntimeError("database down")

    monkeypatch.setattr(generation, "serving_generation", boom)
    assert not is_serving()
    # The start-time form only refuses on a definite answer.
    assert start_allowed("x")
    assert not asyncio.run(tick_allowed("x"))


def test_database_error_with_generations_off_still_passes(db, monkeypatch) -> None:
    def boom() -> str | None:
        raise RuntimeError("database down")

    monkeypatch.setattr(generation, "serving_generation", boom)
    assert is_serving()
    assert asyncio.run(tick_allowed("x"))


async def _run_briefly(coro) -> None:
    task = asyncio.create_task(coro)
    await asyncio.sleep(0.15)
    task.cancel()
    (outcome,) = await asyncio.gather(task, return_exceptions=True)
    # Cancellation is how the helper stops the loop; the sentinel is how the
    # fake tick stops it. Anything else is a real failure and must surface.
    if isinstance(outcome, BaseException) and not isinstance(
        outcome, asyncio.CancelledError
    ):
        if str(outcome) != "stop after the first tick":
            raise outcome


@pytest.mark.asyncio
@pytest.mark.parametrize("serving", [False, True])
async def test_lease_recovery_and_auto_resume_loops_honor_the_record(
    db, monkeypatch, serving
) -> None:
    from xagent.web.services import task_auto_resume, task_lease_recovery

    monkeypatch.setenv(generation.GENERATION_ENV, "green")
    assert set_serving_generation("green" if serving else "blue", None)
    calls: list[str] = []

    async def fake_recover(**_kw):
        calls.append("recover")
        return 0

    def fake_tick(*, now):
        calls.append("resume")
        raise RuntimeError("stop after the first tick")

    monkeypatch.setattr(
        task_lease_recovery, "recover_expired_task_leases_until_cutoff", fake_recover
    )
    monkeypatch.setattr(task_auto_resume, "run_auto_resume_tick", fake_tick)
    await _run_briefly(
        task_lease_recovery.run_task_lease_recovery_loop(
            poll_interval_seconds=0, batch_size=1
        )
    )
    await _run_briefly(
        task_auto_resume.run_task_auto_resume_loop(poll_interval_seconds=0)
    )
    # Each loop is guarded on its own, so each must be seen on its own.
    assert ("recover" in calls) is serving
    assert ("resume" in calls) is serving


@pytest.mark.asyncio
async def test_singleton_starters_start_dormant_when_another_generation_serves(
    db, monkeypatch
) -> None:
    from xagent.web import app as app_module
    from xagent.web.services import task_execution_host

    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setenv(generation.GENERATION_ENV, "green")
    assert set_serving_generation("blue", None)
    monkeypatch.setattr(task_execution_host, "consumes_task_commands", lambda: True)
    monkeypatch.setattr(app_module, "get_trigger_dispatcher_enabled", lambda: True)
    tasks = [
        app_module.start_trigger_dispatcher_task(app_module.app),
        app_module.start_task_lease_recovery_task(app_module.app),
        app_module.start_task_auto_resume_task(app_module.app),
    ]
    assert all(task is not None for task in tasks)
    assert not await tick_allowed("trigger-dispatcher")
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    app_module.app.state.trigger_dispatcher_task = None
    app_module.app.state.task_lease_recovery_task = None
    app_module.app.state.task_auto_resume_task = None


def test_generations_off_never_reads_the_database(monkeypatch) -> None:
    """Every host today: no query, no failure mode, no log on any tick."""
    monkeypatch.delenv(generation.GENERATION_ENV, raising=False)

    def forbidden() -> str | None:
        raise AssertionError("the record must not be read while generations are off")

    monkeypatch.setattr(generation, "serving_generation", forbidden)
    assert is_serving()
    assert start_allowed("x")
    assert asyncio.run(tick_allowed("x"))
    assert generation.is_serving_cached()
    assert generation.serving_status()["serving"] is True
    assert asyncio.run(generation.refresh_serving())


def test_generations_off_logs_nothing(monkeypatch, caplog) -> None:
    monkeypatch.delenv(generation.GENERATION_ENV, raising=False)
    monkeypatch.setattr(
        generation, "serving_generation", lambda: (_ for _ in ()).throw(RuntimeError())
    )
    with caplog.at_level("DEBUG", logger=generation.logger.name):
        is_serving()
        start_allowed("x")
        asyncio.run(tick_allowed("x"))
    assert caplog.records == []


def test_cached_serving_expires_and_read_failure_closes(monkeypatch) -> None:
    monkeypatch.setenv(generation.GENERATION_ENV, "green")
    record = {"value": "green"}

    def read() -> str | None:
        value = record["value"]
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(generation, "serving_generation", read)
    assert generation.is_serving_cached(max_age=0)
    record["value"] = RuntimeError("database down")
    assert not generation.is_serving_cached(max_age=0)
    assert generation.serving_status()["serving"] is None
    record["value"] = "blue"
    assert not generation.is_serving_cached(max_age=0)
    assert generation.serving_status()["serving"] is False


def test_cached_serving_rechecks_after_max_age(monkeypatch) -> None:
    monkeypatch.setenv(generation.GENERATION_ENV, "green")
    monkeypatch.setattr(generation, "_cached_at", float("-inf"))
    clock = {"now": 10.0}
    record = {"value": "green", "reads": 0}
    monkeypatch.setattr(generation.time, "monotonic", lambda: clock["now"])

    def read() -> str:
        record["reads"] += 1
        return record["value"]

    monkeypatch.setattr(generation, "serving_generation", read)
    assert generation.is_serving_cached()
    record["value"] = "blue"
    clock["now"] = 11.9
    assert generation.is_serving_cached()
    assert record["reads"] == 1
    clock["now"] = 12.0
    assert not generation.is_serving_cached()
    assert record["reads"] == 2


@pytest.mark.asyncio
async def test_refresh_wakes_dormant_waiter_and_rechecks_record(monkeypatch) -> None:
    monkeypatch.setenv(generation.GENERATION_ENV, "green")
    record = {"value": "blue"}
    monkeypatch.setattr(generation, "serving_generation", lambda: record["value"])
    waiter = asyncio.create_task(generation.wait_tick("test", 30))
    await asyncio.sleep(0)
    assert not waiter.done()
    record["value"] = "green"
    assert await generation.refresh_serving()
    await asyncio.wait_for(waiter, timeout=1)
    assert generation.serving_status()["serving"] is True


@pytest.mark.asyncio
async def test_refresh_between_failed_guard_and_wait_is_not_lost(monkeypatch) -> None:
    monkeypatch.setenv(generation.GENERATION_ENV, "green")
    record = {"value": "blue"}
    monkeypatch.setattr(generation, "serving_generation", lambda: record["value"])
    assert not await tick_allowed("test")
    record["value"] = "green"
    assert await generation.refresh_serving()
    await asyncio.wait_for(generation.wait_tick("test", 30), timeout=1)


@pytest.mark.asyncio
async def test_dormant_wait_preserves_shutdown_signal(monkeypatch) -> None:
    monkeypatch.setenv(generation.GENERATION_ENV, "green")
    monkeypatch.setattr(generation, "serving_generation", lambda: "blue")
    stop = asyncio.Event()
    waiter = asyncio.create_task(generation.wait_tick("test", 30, stop_event=stop))
    await asyncio.sleep(0)
    stop.set()
    await asyncio.wait_for(waiter, timeout=1)
