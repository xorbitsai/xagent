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
    assert not start_allowed("x")
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
async def test_singleton_starters_refuse_when_another_generation_serves(
    db, monkeypatch
) -> None:
    from xagent.web import app as app_module

    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setenv(generation.GENERATION_ENV, "green")
    assert set_serving_generation("blue", None)
    monkeypatch.setattr(app_module, "get_trigger_dispatcher_enabled", lambda: True)
    assert app_module.start_trigger_dispatcher_task(app_module.app) is None
    assert app_module.start_task_lease_recovery_task(app_module.app) is None
    assert app_module.start_task_auto_resume_task(app_module.app) is None


def test_generations_off_never_reads_the_database(monkeypatch) -> None:
    """Every host today: no query, no failure mode, no log on any tick."""
    monkeypatch.delenv(generation.GENERATION_ENV, raising=False)

    def forbidden() -> str | None:
        raise AssertionError("the record must not be read while generations are off")

    monkeypatch.setattr(generation, "serving_generation", forbidden)
    assert is_serving()
    assert start_allowed("x")
    assert asyncio.run(tick_allowed("x"))


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
