"""How the task-cleanup retry driver is attached to the application (#2587).

The driver's behaviour is tested in
``tests/web/services/test_task_cleanup_obligations.py``; what is pinned here is
that the starter really starts it, that the test-environment guard holds, and
that shutdown stops it.
"""

from __future__ import annotations

import asyncio
from importlib import import_module

import pytest
from fastapi import FastAPI


@pytest.fixture
def app_module():
    return import_module("xagent.web.app")


def test_the_test_environment_starts_nothing_unless_asked(app_module) -> None:
    app = FastAPI()

    assert app_module.start_task_cleanup_retry_task(app) is None
    assert getattr(app.state, "task_cleanup_retry_task", None) is None


@pytest.mark.asyncio
async def test_the_starter_runs_the_driver_and_shutdown_stops_it(
    app_module, monkeypatch
) -> None:
    import sqlalchemy as sa
    from sqlalchemy.orm import sessionmaker

    import xagent.web.models.database as database_module

    monkeypatch.setattr(
        database_module,
        "get_session_local",
        lambda: sessionmaker(bind=sa.create_engine("sqlite://")),
    )
    app = FastAPI()
    app.state.task_cleanup_retry_allowed_in_tests = True

    task = app_module.start_task_cleanup_retry_task(app)

    assert task is not None
    assert app.state.task_cleanup_retry_task is task
    # Starting twice returns the running loop rather than a second one.
    assert app_module.start_task_cleanup_retry_task(app) is task
    await asyncio.sleep(0)
    assert not task.done()

    await app_module.stop_task_cleanup_retry_task(app)

    assert task.done()
    assert getattr(app.state, "task_cleanup_retry_task", None) is None
