"""Three tasks that each hold a structured output record of about 2 MiB, and
a caller for each query that defers the record column, except the run
cancellation (the task list has two callers: its user and an admin)."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from tests.shared.auth_database import auth_db_override
from xagent.web.api.chat import chat_router
from xagent.web.api.conversation_logs import router as conversation_logs_router
from xagent.web.api.monitor import monitor_router
from xagent.web.api.workforces import router as workforces_router
from xagent.web.models.agent import Agent
from xagent.web.models.auth_database import get_auth_db
from xagent.web.models.database import get_db
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.user import User
from xagent.web.models.workforce import Workforce, WorkforceRun
from xagent.web.services.task_stream_snapshot import load_task_stream_snapshots

from .conftest import (
    _admin_headers,
    _direct_db_session,
    _override_get_db,
    _register_second_user,
)
from .conftest import client as api_client

MOMENT = datetime(2026, 11, 1, 8, 30, tzinfo=timezone.utc)
# The compact JSON size of the output is close to 2 MiB.
OUTPUT_SIZE = 2 * 1024 * 1024 - 256

# Sessions the application opened for the requests of ``http``; the test
# closes them, so that the objects they loaded can be inspected while attached.
held_sessions: list[Session] = []


def _holding_get_db() -> Iterator[Session]:
    db = next(get_db())
    held_sessions.append(db)
    yield db


app = FastAPI()
app.include_router(chat_router)
app.include_router(monitor_router)
app.include_router(conversation_logs_router)
app.include_router(workforces_router)
app.dependency_overrides[get_db] = _holding_get_db
app.dependency_overrides[get_auth_db] = auth_db_override(_override_get_db)
http = TestClient(app, raise_server_exceptions=False)


def big_record(run_id: str) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "status": "ready",
        "output": {"text": "x" * OUTPUT_SIZE},
        "reason": None,
    }


@dataclass(frozen=True)
class World:
    admin_headers: dict[str, str]
    user_headers: dict[str, str]
    task_ids: list[int]
    workforce_id: int
    run_ids: list[int]


def build_world() -> World:
    """Three completed SDK tasks of one user, each with a ~2 MiB record, each
    the task of an active run of one workforce of the admin."""
    admin_headers = _admin_headers()
    user_headers = _register_second_user()
    db = _direct_db_session()
    try:
        admin = db.query(User).filter(User.username == "admin").one()
        user = db.query(User).filter(User.username == "bob").one()
        response = api_client.post(
            "/api/agents",
            headers=admin_headers,
            json={
                "name": "manager",
                "description": "test",
                "instructions": "manage",
                "execution_mode": "balanced",
            },
        )
        assert response.status_code == 200, response.text
        manager = db.query(Agent).filter(Agent.id == response.json()["id"]).one()
        workforce = Workforce(
            owner_user_id=int(admin.id),
            scope_type="user",
            scope_id=str(admin.id),
            name="workforce",
            manager_agent_id=int(manager.id),
            status="active",
            created_at=MOMENT,
            updated_at=MOMENT,
        )
        db.add(workforce)
        db.flush()
        task_ids: list[int] = []
        run_ids: list[int] = []
        for index in range(3):
            moment = MOMENT + timedelta(minutes=index)
            task = Task(
                user_id=int(user.id),
                title=f"task {index}",
                description=f"description {index}",
                status=TaskStatus.COMPLETED,
                source="sdk",
                is_visible=False,
                run_id=f"run-{index}",
                output=f"answer {index}",
                created_at=moment,
                updated_at=moment,
            )
            task.structured_output_record = big_record(f"run-{index}")
            db.add(task)
            db.flush()
            run = WorkforceRun(
                workforce_id=int(workforce.id),
                task_id=int(task.id),
                user_id=int(admin.id),
                status="running",
                snapshot={"workforce": {"id": int(workforce.id)}},
                created_at=moment,
            )
            db.add(run)
            db.flush()
            task_ids.append(int(task.id))
            run_ids.append(int(run.id))
        db.commit()
        return World(admin_headers, user_headers, task_ids, int(workforce.id), run_ids)
    finally:
        db.close()


def text_of(path: str, headers: dict[str, str]) -> str:
    response = http.get(path, headers=headers)
    assert response.status_code == 200, response.text
    return response.text


# The response of each caller that returns one, by name.
CALLERS: dict[str, Callable[[World], str]] = {
    "tasks_user": lambda w: text_of(
        "/api/chat/tasks?per_page=3&include_hidden=true", w.user_headers
    ),
    "tasks_admin": lambda w: text_of(
        "/api/chat/tasks?per_page=3&include_hidden=true", w.admin_headers
    ),
    "monitor": lambda w: text_of("/api/monitor/agents", w.user_headers),
    "logs": lambda w: text_of("/api/conversation-logs?per_page=3", w.admin_headers),
    "runs": lambda w: text_of(
        f"/api/workforces/{w.workforce_id}/runs?size=3", w.admin_headers
    ),
    "stream": lambda w: json.dumps(
        load_task_stream_snapshots(w.task_ids), sort_keys=True, default=str
    ),
}
