"""Persistent and resumed-task metadata must not carry the raw memory reason.

``AgentServiceMemoryPolicy.execution_metadata()`` is persisted in checkpoints
and traces, so it folds a trusted host resolver's arbitrary text up front. The
resume completion path applies the same idempotent caller-facing fold.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from xagent.web.models.database import Base, get_db, get_engine, init_db
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.user import User
from xagent.web.services import task_events
from xagent.web.services.agent_service_manager import (
    GENERIC_MEMORY_AVAILABILITY_REASON,
    MEMORY_AVAILABILITY_REASON_METADATA_KEY,
    MEMORY_AVAILABLE_METADATA_KEY,
    AgentServiceMemoryPolicy,
)
from xagent.web.services.task_execution import (
    background_task_manager,
    execute_resume_background,
)

RAW_REASON = "host resolver: tenant 42 over quota at shard eu-3"


@pytest.fixture()
def db_session(tmp_path):
    init_db(db_url=f"sqlite:///{tmp_path / 'resume_completion_metadata.db'}")
    db = next(get_db())
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=get_engine())


def _paused_task(db) -> tuple[User, Task]:
    owner = User(username="owner", password_hash="x")
    db.add(owner)
    db.commit()
    db.refresh(owner)
    task = Task(
        user_id=owner.id,
        title="t",
        description="d",
        status=TaskStatus.PAUSED,
        execution_mode="balanced",
        source="sdk",
    )
    db.add(task)
    db.commit()
    db.refresh(task)
    return owner, task


@pytest.mark.asyncio
async def test_resumed_completion_event_folds_the_memory_reason(
    db_session, monkeypatch: pytest.MonkeyPatch
) -> None:
    owner, task = _paused_task(db_session)
    task_id = int(task.id)
    execution_metadata = AgentServiceMemoryPolicy(
        memory=MagicMock(),
        memory_enabled=False,
        memory_available=False,
        memory_availability_reason=RAW_REASON,
    ).execution_metadata()
    assert (
        execution_metadata[MEMORY_AVAILABILITY_REASON_METADATA_KEY]
        == GENERIC_MEMORY_AVAILABILITY_REASON
    )

    result: dict[str, Any] = {
        "status": "completed",
        "success": True,
        "output": "done",
        "metadata": {**execution_metadata, "pattern": "react"},
        "agent_result": {"context": SimpleNamespace(messages=[])},
    }
    agent = MagicMock()
    agent.resume_execution_by_id = AsyncMock(return_value=result)
    sink = AsyncMock()
    monkeypatch.setattr(task_events, "_task_event_sink", sink)

    current = asyncio.current_task()
    assert current is not None
    background_task_manager.resume_tasks[task_id] = current
    try:
        await execute_resume_background(
            task_id=task_id,
            agent_service=agent,
            task_owner_user_id=int(owner.id),
        )
    finally:
        background_task_manager.resume_tasks.pop(task_id, None)

    completed = [
        call.args[0]
        for call in sink.call_args_list
        if isinstance(call.args[0], dict)
        and call.args[0].get("type") == "task_completed"
    ]
    assert len(completed) == 1
    metadata = completed[0]["metadata"]
    assert RAW_REASON not in repr(completed[0])
    assert metadata[MEMORY_AVAILABLE_METADATA_KEY] is False
    assert (
        metadata[MEMORY_AVAILABILITY_REASON_METADATA_KEY]
        == GENERIC_MEMORY_AVAILABILITY_REASON
    )
    assert metadata["pattern"] == "react"
    assert (
        result["metadata"][MEMORY_AVAILABILITY_REASON_METADATA_KEY]
        == GENERIC_MEMORY_AVAILABILITY_REASON
    )
