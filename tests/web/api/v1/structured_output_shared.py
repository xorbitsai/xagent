"""Helpers shared by the structured-output record tests: an SDK agent with a
key, task rows written straight to the database, and a recorder of the SELECT
statements one block of code sends."""

from __future__ import annotations

import re
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import event

from xagent.web.models.agent import Agent
from xagent.web.models.database import get_engine
from xagent.web.models.task import Task, TaskStatus

from ..conftest import _admin_headers, _direct_db_session, client

COLUMN = "structured_output_record"


def bearer(full_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {full_key}"}


def create_agent_with_key() -> tuple[int, str]:
    """An agent under the admin user and its API key: (agent_id, full_key)."""
    headers = _admin_headers()
    agent_resp = client.post(
        "/api/agents",
        headers=headers,
        json={
            "name": "structured output record agent",
            "description": "test",
            "instructions": "you are a test agent",
            "execution_mode": "balanced",
        },
    )
    assert agent_resp.status_code == 200, agent_resp.text
    agent_id = agent_resp.json()["id"]
    key_resp = client.post(f"/api/agents/{agent_id}/api-key", headers=headers)
    assert key_resp.status_code == 200, key_resp.text
    return agent_id, key_resp.json()["full_key"]


def insert_task(
    agent_id: int,
    *,
    status: TaskStatus = TaskStatus.COMPLETED,
    run_id: str | None = "run-1",
    record: Any = None,
    agent_config: Any = None,
    output: str | None = "final answer",
    error: str | None = None,
    title: str = "structured output task",
    moment: datetime | None = None,
) -> int:
    """One SDK task row, with its structured output record and agent_config."""
    db = _direct_db_session()
    try:
        agent = db.query(Agent).filter(Agent.id == agent_id).one()
        task = Task(
            user_id=int(agent.user_id),
            title=title,
            description=title,
            status=status,
            agent_id=agent_id,
            input="question",
            output=output,
            source="sdk",
            is_visible=False,
            run_id=run_id,
            agent_config=agent_config,
            error_message=error,
        )
        if moment is not None:
            task.created_at = moment
            task.updated_at = moment
        task.structured_output_record = record
        db.add(task)
        db.commit()
        return int(task.id)
    finally:
        db.close()


@contextmanager
def recorded_statements() -> Iterator[list[str]]:
    """Every SQL statement the test engine executes inside the block."""
    statements: list[str] = []

    def record(_conn, _cursor, statement, *_args) -> None:
        statements.append(statement)

    engine = get_engine()
    event.listen(engine, "before_cursor_execute", record)
    try:
        yield statements
    finally:
        event.remove(engine, "before_cursor_execute", record)


def task_selects(statements: list[str]) -> list[str]:
    """The statements that read rows from ``tasks`` into Python."""
    return [
        s
        for s in statements
        if s.lstrip().upper().startswith("SELECT")
        and not s.lstrip().upper().startswith("SELECT COUNT(")
        and re.search(r"\bFROM tasks\b", s) is not None
        and "tasks.title" in s
    ]


FORMAT_CONFIG = {"response_format": {"type": "json_schema", "name": "answer"}}
MOMENT = datetime(2026, 11, 1, 8, 30, tzinfo=timezone.utc)
RECORD = {
    "run_id": "run-1",
    "status": "ready",
    "output": {"zeta": 1, "alpha": [3, 2, 1]},
    "reason": None,
}

# Tasks whose GET response is fixed in structured_output_get_baseline.py:
# (status, run_id, record, agent_config, output, error).
BASELINE_TASKS: list[dict[str, Any]] = [
    {"status": TaskStatus.COMPLETED, "output": "final answer"},
    {
        "status": TaskStatus.COMPLETED,
        "output": "with a record",
        "record": RECORD,
        "agent_config": FORMAT_CONFIG,
    },
    {"status": TaskStatus.FAILED, "output": None, "error": "agent crashed"},
    {"status": TaskStatus.RUNNING, "output": None},
    {"status": TaskStatus.PENDING, "output": None, "run_id": None},
    {"status": TaskStatus.WAITING_FOR_USER, "output": None},
    {"status": TaskStatus.COMPLETED, "output": "\u00e9 \u4e2d\u6587 \U0001f600"},
]
NEW_KEYS = ("structured_output", "structured_output_status", "structured_output_reason")
