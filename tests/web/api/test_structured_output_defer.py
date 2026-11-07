"""Queries that read many task rows at once leave out the structured output
record column; the read of one task by primary key still returns it."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from contextlib import contextmanager

import pytest
from sqlalchemy import event
from sqlalchemy.exc import InvalidRequestError
from sqlalchemy.orm import Session, sessionmaker

from xagent.web.models.database import get_engine
from xagent.web.models.task import Task
from xagent.web.models.workforce import WorkforceRun
from xagent.web.services.workforce_runtime import cancel_active_workforce_runs

from .conftest import _direct_db_session
from .conftest import client as api_client
from .structured_output_defer_baseline import BASELINE
from .structured_output_defer_shared import (
    CALLERS,
    World,
    big_record,
    build_world,
    held_sessions,
)
from .v1.structured_output_shared import (
    COLUMN,
    bearer,
    create_agent_with_key,
    recorded_statements,
    task_selects,
)

pytestmark = pytest.mark.usefixtures("_test_db")


@contextmanager
def loaded_tasks() -> Iterator[list[Task]]:
    """Every Task instance loaded inside the block, kept alive for the caller.

    The sessions that loaded them are still open when the caller looks, so the
    instances are attached.
    """
    loaded: list[Task] = []

    def on_load(target: Task, _context) -> None:
        loaded.append(target)

    event.listen(Task, "load", on_load)
    try:
        yield loaded
    finally:
        event.remove(Task, "load", on_load)


def assert_record_not_read(statements: list[str], tasks: list[Task]) -> None:
    selects = task_selects(statements)
    assert selects, statements
    assert not [s for s in selects if COLUMN in s]
    assert tasks
    for task in tasks:
        with pytest.raises(InvalidRequestError):
            task.structured_output_record


class _HeldSession(Session):
    """A session whose ``close`` does nothing, so the instances it loaded stay
    attached after the ``with`` block that opened it ends."""

    def close(self) -> None:
        pass


@pytest.fixture
def stream_sessions_held(monkeypatch) -> Iterator[None]:
    """Make the stream reconciliation's own session outlive its ``with`` block,
    so the test can inspect the Task objects it returns."""
    opened: list[_HeldSession] = []

    def factory() -> sessionmaker[Session]:
        def make() -> _HeldSession:
            db = _HeldSession(bind=get_engine())
            opened.append(db)
            return db

        return make  # type: ignore[return-value]

    monkeypatch.setattr(
        "xagent.web.services.task_stream_snapshot.get_session_local", factory
    )
    yield
    for db in opened:
        Session.close(db)


@pytest.fixture
def world(_test_db) -> Iterator[World]:
    held_sessions.clear()
    yield build_world()
    for db in held_sessions:
        db.close()
    held_sessions.clear()


@pytest.mark.usefixtures("stream_sessions_held")
@pytest.mark.parametrize("name", sorted(CALLERS))
def test_a_query_that_reads_many_tasks_does_not_select_the_record_column(world, name):
    with recorded_statements() as statements, loaded_tasks() as loaded:
        text = CALLERS[name](world)

    assert_record_not_read(statements, loaded)
    assert text == BASELINE[name]


@pytest.mark.parametrize("name", ["tasks_user", "tasks_admin"])
def test_the_task_list_returns_every_row(world, name):
    assert len(json.loads(CALLERS[name](world))["tasks"]) == 3


def test_the_run_list_reads_its_tasks_in_one_select_without_the_record_column(world):
    with recorded_statements() as statements:
        runs = json.loads(CALLERS["runs"](world))

    preloads = [
        s
        for s in statements
        if re.search(r"\bFROM tasks\s+WHERE tasks\.id IN\b", s) is not None
    ]
    assert len(preloads) == 1
    assert COLUMN not in preloads[0]
    assert [item["task_title"] for item in runs["items"]] == [
        "task 2",
        "task 1",
        "task 0",
    ]


def test_archiving_a_workforce_cancels_its_runs_without_reading_the_record_column(
    world,
):
    db = _direct_db_session()
    try:
        with recorded_statements() as statements, loaded_tasks() as loaded:
            targets = cancel_active_workforce_runs(db, world.workforce_id)
            assert_record_not_read(statements, loaded)
        preloads = [
            s
            for s in statements
            if re.search(r"\bFROM tasks\s+WHERE tasks\.id IN\b", s) is not None
        ]
        assert len(preloads) == 1
        assert COLUMN not in preloads[0]
        # The runs are not returned; the same session hands back the same
        # instances, whose preloaded task is already there.
        runs = (
            db.query(WorkforceRun)
            .filter(WorkforceRun.id.in_(world.run_ids))
            .order_by(WorkforceRun.id)
            .all()
        )
        assert [run.task.title for run in runs] == ["task 0", "task 1", "task 2"]
        db.commit()
    finally:
        db.close()

    assert targets == []
    check = _direct_db_session()
    try:
        statuses = [
            run.status
            for run in check.query(WorkforceRun)
            .filter(WorkforceRun.id.in_(world.run_ids))
            .all()
        ]
    finally:
        check.close()
    assert statuses == ["cancelled"] * 3


def test_a_get_of_one_of_those_tasks_still_returns_its_record(world):
    agent_id, key = create_agent_with_key()
    db = _direct_db_session()
    try:
        db.query(Task).filter(Task.id == world.task_ids[0]).update(
            {Task.agent_id: agent_id}
        )
        db.commit()
    finally:
        db.close()

    with recorded_statements() as statements:
        response = api_client.get(
            f"/v1/chat/tasks/{world.task_ids[0]}", headers=bearer(key)
        )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["structured_output_status"] == "ready"
    assert body["structured_output"] == big_record("run-0")["output"]
    selects = task_selects(statements)
    assert selects and all(f"tasks.{COLUMN}" in s for s in selects)
