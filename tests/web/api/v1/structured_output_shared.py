"""Helpers shared by the structured-output record tests: an SDK agent with a
key, task rows written straight to the database, and a recorder of the SELECT
statements one block of code sends."""

from __future__ import annotations

import re
from collections.abc import Iterator
from contextlib import contextmanager
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
    title: str = "structured output task",
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
        )
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
