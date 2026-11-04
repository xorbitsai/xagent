"""The resume schedulers hand ``execute_resume_background`` the row's source.

Before this wiring both resume entry points left ``trusted_task_source``
unset, so the resume metadata carried ``task_source=None``. Combined with the
runner's overlay that closes a real failure mode: a user approving a pending
MCP write through the SDK reply or A2A resume had the checkpointed source
replaced, and the gate refused the replay it should have performed.

``None`` still has to mean "this caller knows nothing" rather than "this task
has no source" -- the runner overlay keeps the checkpointed value for a None
-- so the read here returns None only when there genuinely is no row value.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from xagent.web.models.agent import Agent
from xagent.web.models.database import get_session_local, init_db
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.user import User
from xagent.web.services import task_execution as task_execution_service
from xagent.web.services import task_resume
from xagent.web.services.task_lease_service import TaskLease


@pytest.fixture
def seeded_task(tmp_path: Any) -> int:
    init_db(db_url=f"sqlite:///{tmp_path / 'trusted-source.db'}")
    with get_session_local()() as db:
        user = User(username="owner", password_hash="unused")
        db.add(user)
        db.flush()
        agent = Agent(user_id=user.id, name="resume")
        db.add(agent)
        db.flush()
        task = Task(
            user_id=user.id,
            agent_id=agent.id,
            title="gated",
            input="hi",
            status=TaskStatus.WAITING_FOR_USER,
            source="slack",
        )
        db.add(task)
        db.commit()
        return int(task.id)


def test_trusted_source_is_read_from_the_task_row(seeded_task: int) -> None:
    assert task_resume._trusted_task_source_sync(seeded_task) == "slack"


def test_trusted_source_is_none_for_a_row_that_is_not_there(seeded_task: int) -> None:
    """A missing row is "I do not know", which the runner overlay skips."""

    assert task_resume._trusted_task_source_sync(seeded_task + 9999) is None


def test_trusted_source_is_none_for_a_legacy_null_source(seeded_task: int) -> None:
    with get_session_local()() as db:
        task = db.get(Task, seeded_task)
        task.source = None
        db.commit()

    assert task_resume._trusted_task_source_sync(seeded_task) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["slack", None])
async def test_reply_scheduler_forwards_the_trusted_source(
    source: str | None,
) -> None:
    manager = task_execution_service.BackgroundTaskManager()
    lease = TaskLease(task_id=4242, runner_id="runner-x", run_id="run-reply")
    gate = asyncio.Event()
    seen: list[dict[str, Any]] = []

    async def execute_resume_background(**kwargs: Any) -> None:
        seen.append(kwargs)
        await gate.wait()

    with (
        patch.object(task_execution_service, "background_task_manager", manager),
        patch.object(
            task_execution_service,
            "execute_resume_background",
            side_effect=execute_resume_background,
        ),
    ):
        await task_resume._schedule_waiting_reply_resume(
            task_id=4242,
            agent_service=MagicMock(),
            task_owner_user_id=1,
            task_lease=lease,
            heartbeat_stop=asyncio.Event(),
            heartbeat_task=asyncio.ensure_future(asyncio.sleep(0)),
            trusted_task_source=source,
        )
        try:
            await asyncio.wait_for(_until(lambda: bool(seen)), timeout=5)
            assert seen[0]["trusted_task_source"] == source
        finally:
            gate.set()
            await asyncio.wait_for(manager.resume_tasks[4242], timeout=5)


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["slack", None])
async def test_a2a_scheduler_forwards_the_trusted_source(source: str | None) -> None:
    manager = task_execution_service.BackgroundTaskManager()
    lease = TaskLease(task_id=4343, runner_id="runner-x", run_id="run-a2a")
    gate = asyncio.Event()
    seen: list[dict[str, Any]] = []

    async def execute_resume_background(**kwargs: Any) -> None:
        seen.append(kwargs)
        await gate.wait()

    with (
        patch(
            "xagent.web.services.task_execution.background_task_manager",
            manager,
        ),
        patch(
            "xagent.web.services.task_execution.execute_resume_background",
            new=AsyncMock(side_effect=execute_resume_background),
        ),
    ):
        await task_resume._schedule_waiting_a2a_resume(
            task_id=4343,
            agent_service=MagicMock(),
            task_owner_user_id=1,
            task_lease=lease,
            heartbeat_stop=asyncio.Event(),
            heartbeat_task=asyncio.ensure_future(asyncio.sleep(0)),
            resumable_status=TaskStatus.WAITING_FOR_USER,
            trusted_task_source=source,
        )
        try:
            await asyncio.wait_for(_until(lambda: bool(seen)), timeout=5)
            assert seen[0]["trusted_task_source"] == source
        finally:
            gate.set()
            await asyncio.wait_for(manager.resume_tasks[4343], timeout=5)


async def _until(predicate: Any) -> None:
    while not predicate():
        await asyncio.sleep(0)
