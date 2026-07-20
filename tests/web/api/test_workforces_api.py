from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any

import pytest
from sqlalchemy import event

from xagent.web.api import workforces as workforces_api
from xagent.web.models.agent import Agent, AgentOrigin, AgentStatus
from xagent.web.models.database import get_engine
from xagent.web.models.task import Task, TaskStatus, TraceEvent
from xagent.web.models.user import User
from xagent.web.models.workforce import Workforce, WorkforceRun
from xagent.web.services.workforce_access import WorkforcePolicy, set_workforce_policy

from .conftest import (
    _admin_headers,
    _direct_db_session,
    _register_second_user,
    client,
)


@pytest.fixture(autouse=True)
def _db(_test_db: None) -> None:
    pass


@pytest.fixture(autouse=True)
def _reset_workforce_policy() -> None:
    set_workforce_policy(WorkforcePolicy())
    yield
    set_workforce_policy(WorkforcePolicy())


def _user_id(username: str = "admin") -> int:
    db = _direct_db_session()
    try:
        user = db.query(User).filter(User.username == username).first()
        assert user is not None
        return int(user.id)
    finally:
        db.close()


def _create_agent(
    user_id: int,
    name: str,
    status: AgentStatus,
    origin: str = AgentOrigin.USER.value,
) -> int:
    db = _direct_db_session()
    try:
        agent = Agent(
            user_id=user_id,
            name=name,
            description=f"{name} description",
            instructions=f"{name} instructions",
            execution_mode="balanced",
            origin=origin,
            status=status,
        )
        db.add(agent)
        db.commit()
        db.refresh(agent)
        return int(agent.id)
    finally:
        db.close()


def _create_published_agent(user_id: int, name: str) -> int:
    return _create_agent(user_id, name, AgentStatus.PUBLISHED)


def _create_workforce(
    headers: dict[str, str],
    *,
    name: str = "Support Workforce",
    worker_count: int = 1,
    canvas_layout: dict[str, Any] | None = None,
    username: str = "admin",
) -> dict[str, Any]:
    owner_id = _user_id(username)
    manager_agent_id = _create_published_agent(owner_id, f"{name} Manager")
    workers = []
    for index in range(worker_count):
        workers.append(
            {
                "source_type": "existing",
                "agent_id": _create_published_agent(
                    owner_id, f"{name} Worker {index + 1}"
                ),
                "alias": f"worker-{index + 1}",
                "assignment_instructions": f"Handle area {index + 1}",
                "enabled": True,
                "sort_order": index + 1,
                "canvas_position": {"x": 100 + index, "y": 200 + index},
            }
        )

    response = client.post(
        "/api/workforces",
        headers=headers,
        json={
            "name": name,
            "description": "Coordinates support work",
            "manager_agent_id": manager_agent_id,
            "manager_instructions": "Delegate and synthesize.",
            "canvas_layout": canvas_layout,
            "workers": workers,
        },
    )
    assert response.status_code == 200, response.text
    return response.json()


def _create_workforce_run(
    *,
    workforce_id: int,
    user_id: int,
    status: str,
    created_at: datetime,
    is_preview: bool = False,
    task_id: int | None = None,
    completed_at: datetime | None = None,
) -> int:
    db = _direct_db_session()
    try:
        run = WorkforceRun(
            workforce_id=workforce_id,
            task_id=task_id,
            user_id=user_id,
            status=status,
            is_preview=is_preview,
            snapshot={"workforce": {"id": workforce_id}},
            created_at=created_at,
            completed_at=completed_at,
        )
        db.add(run)
        db.commit()
        db.refresh(run)
        return int(run.id)
    finally:
        db.close()


def _create_task(user_id: int, *, title: str, description: str) -> int:
    db = _direct_db_session()
    try:
        task = Task(
            user_id=user_id,
            title=title,
            description=description,
            status=TaskStatus.COMPLETED,
            source="internal",
            is_visible=False,
        )
        db.add(task)
        db.commit()
        db.refresh(task)
        return int(task.id)
    finally:
        db.close()


def _create_task_workforce_run(
    *,
    workforce_id: int,
    user_id: int,
    title: str,
    description: str,
    status: TaskStatus,
    created_at: datetime,
) -> tuple[int, int]:
    db = _direct_db_session()
    try:
        task = Task(
            user_id=user_id,
            title=title,
            description=description,
            status=status,
            created_at=created_at,
        )
        db.add(task)
        db.flush()
        run = WorkforceRun(
            workforce_id=workforce_id,
            task_id=int(task.id),
            user_id=user_id,
            status=status.value,
            snapshot={"workforce": {"id": workforce_id}},
            created_at=created_at,
            completed_at=(
                created_at + timedelta(minutes=1)
                if status in {TaskStatus.COMPLETED, TaskStatus.FAILED}
                else None
            ),
        )
        db.add(run)
        db.commit()
        db.refresh(run)
        return int(run.id), int(task.id)
    finally:
        db.close()


class _VisibleAgentPolicy(WorkforcePolicy):
    def __init__(self, visible_agent_ids: set[int]) -> None:
        self.visible_agent_ids = visible_agent_ids

    def get_visible_agent_ids(
        self,
        db: Any,
        user: User,
        purpose: str,
    ) -> set[int]:
        del db, user
        assert purpose == "workforce_select"
        return self.visible_agent_ids


def test_workforce_endpoints_require_authentication() -> None:
    response = client.get("/api/workforces")
    assert response.status_code == 403


def test_agent_options_use_workforce_policy_and_only_published_agents() -> None:
    _admin_headers()
    bob_headers = _register_second_user()
    admin_id = _user_id("admin")
    bob_id = _user_id("bob")

    bob_published_id = _create_agent(
        bob_id,
        "Bob Published Worker",
        AgentStatus.PUBLISHED,
    )
    bob_draft_id = _create_agent(
        bob_id,
        "Bob Draft Worker",
        AgentStatus.DRAFT,
    )
    shared_published_id = _create_agent(
        admin_id,
        "Shared Published Worker",
        AgentStatus.PUBLISHED,
    )
    shared_draft_id = _create_agent(
        admin_id,
        "Shared Draft Worker",
        AgentStatus.DRAFT,
    )
    set_workforce_policy(_VisibleAgentPolicy({shared_published_id, shared_draft_id}))

    response = client.get("/api/workforces/agent-options", headers=bob_headers)
    assert response.status_code == 200, response.text
    options_by_id = {item["id"]: item for item in response.json()}

    assert bob_published_id in options_by_id
    assert shared_published_id in options_by_id
    assert bob_draft_id not in options_by_id
    assert shared_draft_id not in options_by_id

    assert options_by_id[bob_published_id]["access"] == "owner"
    assert options_by_id[bob_published_id]["readonly"] is False
    assert options_by_id[shared_published_id]["access"] == "policy"
    assert options_by_id[shared_published_id]["readonly"] is True
    assert options_by_id[shared_published_id]["can_edit"] is False


def test_generated_workforce_manager_agents_are_private_to_their_workforce() -> None:
    headers = _admin_headers()
    owner_id = _user_id("admin")
    reusable_manager_id = _create_agent(
        owner_id,
        "Reusable Manager",
        AgentStatus.PUBLISHED,
    )
    generated_manager_id = _create_agent(
        owner_id,
        "Generated Manager",
        AgentStatus.PUBLISHED,
        AgentOrigin.WORKFORCE_GENERATED_MANAGER.value,
    )

    reusable_response = client.post(
        "/api/workforces",
        headers=headers,
        json={
            "name": "Reusable Manager Workforce",
            "manager_agent_id": reusable_manager_id,
        },
    )
    assert reusable_response.status_code == 200, reusable_response.text

    generated_response = client.post(
        "/api/workforces",
        headers=headers,
        json={
            "name": "Generated Manager Workforce",
            "manager_agent_id": generated_manager_id,
        },
    )
    assert generated_response.status_code == 404


def test_update_workforce_keeps_existing_generated_manager_but_rejects_switching_to_one() -> (
    None
):
    headers = _admin_headers()
    owner_id = _user_id("admin")
    generated_manager_id = _create_agent(
        owner_id,
        "Generated Manager",
        AgentStatus.PUBLISHED,
        AgentOrigin.WORKFORCE_GENERATED_MANAGER.value,
    )
    reusable_manager_id = _create_agent(
        owner_id,
        "Reusable Manager",
        AgentStatus.PUBLISHED,
    )

    db = _direct_db_session()
    try:
        workforce = Workforce(
            owner_user_id=owner_id,
            scope_type="user",
            scope_id=str(owner_id),
            name="Generated Manager Workforce",
            manager_agent_id=generated_manager_id,
            status="draft",
        )
        db.add(workforce)
        db.commit()
        db.refresh(workforce)
        workforce_id = int(workforce.id)
    finally:
        db.close()

    keep_response = client.patch(
        f"/api/workforces/{workforce_id}",
        headers=headers,
        json={
            "name": "Renamed Generated Manager Workforce",
            "manager_agent_id": generated_manager_id,
        },
    )
    assert keep_response.status_code == 200, keep_response.text
    assert keep_response.json()["name"] == "Renamed Generated Manager Workforce"
    assert keep_response.json()["manager"]["id"] == generated_manager_id

    switch_to_reusable_response = client.patch(
        f"/api/workforces/{workforce_id}",
        headers=headers,
        json={"manager_agent_id": reusable_manager_id},
    )
    assert switch_to_reusable_response.status_code == 200, (
        switch_to_reusable_response.text
    )
    assert switch_to_reusable_response.json()["manager"]["id"] == reusable_manager_id

    switch_to_generated_response = client.patch(
        f"/api/workforces/{workforce_id}",
        headers=headers,
        json={"manager_agent_id": generated_manager_id},
    )
    assert switch_to_generated_response.status_code == 404


def test_workforce_detail_marks_generated_manager_readonly() -> None:
    headers = _admin_headers()
    owner_id = _user_id("admin")
    generated_manager_id = _create_agent(
        owner_id,
        "Generated Manager",
        AgentStatus.PUBLISHED,
        AgentOrigin.WORKFORCE_GENERATED_MANAGER.value,
    )

    db = _direct_db_session()
    try:
        workforce = Workforce(
            owner_user_id=owner_id,
            scope_type="user",
            scope_id=str(owner_id),
            name="Generated Manager Workforce",
            manager_agent_id=generated_manager_id,
            status="draft",
        )
        db.add(workforce)
        db.commit()
        db.refresh(workforce)
        workforce_id = int(workforce.id)
    finally:
        db.close()

    response = client.get(f"/api/workforces/{workforce_id}", headers=headers)
    assert response.status_code == 200, response.text
    manager = response.json()["manager"]
    assert manager["id"] == generated_manager_id
    assert manager["access"] == "owner"
    assert manager["readonly"] is True
    assert manager["can_edit"] is False
    assert manager["can_publish"] is False
    assert manager["can_delete"] is False


def test_workforce_detail_marks_policy_visible_agents_readonly() -> None:
    _admin_headers()
    bob_headers = _register_second_user()
    admin_id = _user_id("admin")

    shared_manager_id = _create_agent(
        admin_id,
        "Shared Manager",
        AgentStatus.PUBLISHED,
    )
    shared_worker_id = _create_agent(
        admin_id,
        "Shared Worker",
        AgentStatus.PUBLISHED,
    )
    set_workforce_policy(_VisibleAgentPolicy({shared_manager_id, shared_worker_id}))

    response = client.post(
        "/api/workforces",
        headers=bob_headers,
        json={
            "name": "Shared Agent Workforce",
            "manager_agent_id": shared_manager_id,
            "workers": [
                {
                    "source_type": "existing",
                    "agent_id": shared_worker_id,
                    "assignment_instructions": "Handle shared work",
                },
            ],
        },
    )
    assert response.status_code == 200, response.text
    workforce = response.json()

    detail_response = client.get(
        f"/api/workforces/{workforce['id']}",
        headers=bob_headers,
    )
    assert detail_response.status_code == 200, detail_response.text
    detail = detail_response.json()

    assert detail["manager"]["access"] == "policy"
    assert detail["manager"]["readonly"] is True
    assert detail["manager"]["can_edit"] is False
    assert detail["workers"][0]["agent"]["access"] == "policy"
    assert detail["workers"][0]["agent"]["readonly"] is True
    assert detail["workers"][0]["agent"]["can_edit"] is False


def test_create_list_get_and_cross_user_access_control() -> None:
    headers = _admin_headers()
    workforce = _create_workforce(headers)

    assert workforce["status"] == "draft"
    assert workforce["manager"]["name"] == "Support Workforce Manager"
    assert workforce["workers"][0]["agent"]["name"] == "Support Workforce Worker 1"
    assert "manager_instructions" not in workforce

    list_response = client.get("/api/workforces", headers=headers)
    assert list_response.status_code == 200
    list_payload = list_response.json()
    assert list_payload["total"] == 1
    assert list_payload["items"][0]["id"] == workforce["id"]

    detail_response = client.get(f"/api/workforces/{workforce['id']}", headers=headers)
    assert detail_response.status_code == 200
    assert detail_response.json()["id"] == workforce["id"]

    other_headers = _register_second_user()
    denied_response = client.get(
        f"/api/workforces/{workforce['id']}", headers=other_headers
    )
    assert denied_response.status_code == 403

    other_workforce = _create_workforce(
        other_headers,
        name="Other User Workforce",
        username="bob",
    )
    other_list_response = client.get("/api/workforces", headers=other_headers)
    assert other_list_response.status_code == 200
    other_list_payload = other_list_response.json()
    assert other_list_payload["total"] == 1
    assert other_list_payload["items"][0]["id"] == other_workforce["id"]


def test_manager_instructions_is_ignored_on_create_and_update() -> None:
    headers = _admin_headers()
    workforce = _create_workforce(headers)

    update_response = client.patch(
        f"/api/workforces/{workforce['id']}",
        headers=headers,
        json={"description": "Updated", "manager_instructions": "Legacy value"},
    )
    assert update_response.status_code == 200, update_response.text
    payload = update_response.json()
    assert payload["description"] == "Updated"
    assert "manager_instructions" not in payload


def test_list_workforces_paginates_visible_query_and_bulk_loads_last_runs() -> None:
    headers = _admin_headers()
    owner_id = _user_id()
    workforces = [
        _create_workforce(headers, name=f"Paged Workforce {index}")
        for index in range(3)
    ]
    now = datetime.now(timezone.utc)
    expected_latest_status: dict[int, str] = {}
    for index, workforce in enumerate(workforces):
        workforce_id = int(workforce["id"])
        _create_workforce_run(
            workforce_id=workforce_id,
            user_id=owner_id,
            status="failed",
            created_at=now + timedelta(minutes=index),
        )
        expected_latest_status[workforce_id] = "completed"
        _create_workforce_run(
            workforce_id=workforce_id,
            user_id=owner_id,
            status="completed",
            created_at=now + timedelta(minutes=10 + index),
        )

    workforce_run_selects: list[str] = []

    def track_workforce_run_queries(
        conn: Any,
        cursor: Any,
        statement: str,
        parameters: Any,
        context: Any,
        executemany: bool,
    ) -> None:
        del conn, cursor, parameters, context, executemany
        if "from workforce_runs" in statement.lower():
            workforce_run_selects.append(statement)

    event.listen(get_engine(), "before_cursor_execute", track_workforce_run_queries)
    try:
        response = client.get(
            "/api/workforces",
            headers=headers,
            params={"page": 1, "size": 2},
        )
    finally:
        event.remove(get_engine(), "before_cursor_execute", track_workforce_run_queries)

    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] == 3
    assert payload["pages"] == 2
    assert len(payload["items"]) == 2
    assert len(workforce_run_selects) == 1
    for item in payload["items"]:
        assert item["last_run"]["status"] == expected_latest_status[item["id"]]


def test_get_workforce_agent_execution_returns_only_requested_worker_trace() -> None:
    headers = _admin_headers()
    workforce = _create_workforce(headers, name="Agent Trace Workforce")
    owner_id = _user_id()
    now = datetime.now(timezone.utc)
    _, task_id = _create_task_workforce_run(
        workforce_id=int(workforce["id"]),
        user_id=owner_id,
        title="Agent trace run",
        description="Delegate the work",
        status=TaskStatus.COMPLETED,
        created_at=now,
    )

    db = _direct_db_session()
    try:
        db.add_all(
            [
                TraceEvent(
                    task_id=task_id,
                    build_id="agent_17_run",
                    event_id="worker-start",
                    event_type="react_task_start",
                    timestamp=now,
                    data={
                        "source": "xagent-agent-tool-child",
                        "worker_task_id": "agent_17_run",
                        "agent_id": 17,
                        "agent_name": "Editor Agent",
                        "worker_alias": "Editor",
                    },
                ),
                TraceEvent(
                    task_id=task_id,
                    build_id="agent_17_run",
                    event_id="worker-end",
                    event_type="react_task_end",
                    timestamp=now + timedelta(seconds=1),
                    data={
                        "source": "xagent-agent-tool-child",
                        "worker_task_id": "agent_17_run",
                        "agent_name": "Editor Agent",
                        "result": {"success": True},
                    },
                ),
                TraceEvent(
                    task_id=task_id,
                    event_id="delegation-end",
                    event_type="task_update_general",
                    timestamp=now + timedelta(seconds=2),
                    data={
                        "event_type": "workforce_delegation_end",
                        "worker_task_id": "agent_17_run",
                        "agent_id": 17,
                        "agent_name": "Editor Agent",
                        "worker_alias": "Editor",
                    },
                ),
                TraceEvent(
                    task_id=task_id,
                    build_id="agent_18_run",
                    event_id="other-worker",
                    event_type="react_task_start",
                    timestamp=now,
                    data={
                        "source": "xagent-agent-tool-child",
                        "worker_task_id": "agent_18_run",
                    },
                ),
            ]
        )
        db.commit()
    finally:
        db.close()

    response = client.get(
        f"/api/workforces/{workforce['id']}/runs/{task_id}/agent-executions/agent_17_run",
        headers=headers,
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["worker_task_id"] == "agent_17_run"
    assert payload["agent_name"] == "Editor Agent"
    assert payload["worker_alias"] == "Editor"
    assert payload["status"] == "completed"
    assert [event["event_id"] for event in payload["trace_events"]] == [
        "worker-start",
        "worker-end",
    ]


def test_get_workforce_agent_execution_derives_completed_without_summary() -> None:
    headers = _admin_headers()
    workforce = _create_workforce(headers, name="Derived Agent Trace Workforce")
    owner_id = _user_id()
    now = datetime.now(timezone.utc)
    _, task_id = _create_task_workforce_run(
        workforce_id=int(workforce["id"]),
        user_id=owner_id,
        title="Agent trace run",
        description="Delegate the work",
        status=TaskStatus.RUNNING,
        created_at=now,
    )

    db = _direct_db_session()
    try:
        db.add_all(
            [
                TraceEvent(
                    task_id=task_id,
                    build_id="agent_17_run",
                    event_id="worker-start",
                    event_type="react_task_start",
                    timestamp=now,
                    data={
                        "source": "xagent-agent-tool-child",
                        "worker_task_id": "agent_17_run",
                        "agent_name": "Editor Agent",
                    },
                ),
                TraceEvent(
                    task_id=task_id,
                    build_id="agent_17_run",
                    event_id="worker-end",
                    event_type="react_task_end",
                    timestamp=now + timedelta(seconds=1),
                    data={
                        "source": "xagent-agent-tool-child",
                        "worker_task_id": "agent_17_run",
                        "result": {"success": True},
                    },
                ),
            ]
        )
        db.commit()
    finally:
        db.close()

    response = client.get(
        f"/api/workforces/{workforce['id']}/runs/{task_id}/agent-executions/agent_17_run",
        headers=headers,
    )

    assert response.status_code == 200, response.text
    assert response.json()["status"] == "completed"


def test_get_workforce_agent_execution_marks_orphan_interrupted() -> None:
    headers = _admin_headers()
    workforce = _create_workforce(headers, name="Orphan Agent Trace Workforce")
    owner_id = _user_id()
    now = datetime.now(timezone.utc)
    _, task_id = _create_task_workforce_run(
        workforce_id=int(workforce["id"]),
        user_id=owner_id,
        title="Agent trace run",
        description="Delegate the work",
        status=TaskStatus.COMPLETED,
        created_at=now,
    )

    db = _direct_db_session()
    try:
        db.add(
            TraceEvent(
                task_id=task_id,
                build_id="agent_17_run",
                event_id="worker-start",
                event_type="react_task_start",
                timestamp=now,
                data={
                    "source": "xagent-agent-tool-child",
                    "worker_task_id": "agent_17_run",
                    "agent_name": "Editor Agent",
                },
            )
        )
        db.commit()
    finally:
        db.close()

    response = client.get(
        f"/api/workforces/{workforce['id']}/runs/{task_id}/agent-executions/agent_17_run",
        headers=headers,
    )

    assert response.status_code == 200, response.text
    assert response.json()["status"] == "interrupted"


@pytest.mark.parametrize(
    ("event_type", "result", "expected"),
    [
        ("dag_execute_end", {"success": True}, "completed"),
        ("dag_execute_end", {"success": False}, "failed"),
        ("trace_error", {}, "failed"),
    ],
)
def test_agent_execution_status_recognizes_dag_terminal_events(
    event_type: str,
    result: dict[str, Any],
    expected: str,
) -> None:
    assert (
        workforces_api._derive_agent_execution_status(
            [{"event_type": event_type, "data": {"result": result}}]
        )
        == expected
    )


def test_publish_unpublish_and_active_validation() -> None:
    headers = _admin_headers()
    empty_workforce = _create_workforce(headers, name="Empty Workforce", worker_count=0)

    invalid_publish = client.post(
        f"/api/workforces/{empty_workforce['id']}/publish",
        headers=headers,
    )
    assert invalid_publish.status_code == 400
    assert "at least one enabled worker" in invalid_publish.json()["detail"]

    workforce = _create_workforce(headers, name="Runnable Workforce")
    publish_response = client.post(
        f"/api/workforces/{workforce['id']}/publish",
        headers=headers,
    )
    assert publish_response.status_code == 200
    assert publish_response.json()["status"] == "active"

    unpublish_response = client.post(
        f"/api/workforces/{workforce['id']}/unpublish",
        headers=headers,
    )
    assert unpublish_response.status_code == 200
    assert unpublish_response.json()["status"] == "draft"


def test_worker_add_update_remove_and_active_rollback() -> None:
    headers = _admin_headers()
    workforce = _create_workforce(headers, name="Worker Edit Workforce")
    owner_id = _user_id()

    add_response = client.post(
        f"/api/workforces/{workforce['id']}/agents",
        headers=headers,
        json={
            "source_type": "existing",
            "agent_id": _create_published_agent(owner_id, "Additional Worker"),
            "alias": "extra",
            "assignment_instructions": "Handle overflow work",
            "enabled": True,
            "sort_order": 2,
        },
    )
    assert add_response.status_code == 200, add_response.text
    added_worker_id = add_response.json()["id"]

    update_response = client.patch(
        f"/api/workforces/{workforce['id']}/agents/{added_worker_id}",
        headers=headers,
        json={
            "alias": "overflow",
            "assignment_instructions": "Handle escalations",
            "canvas_position": {"x": 9, "y": 10},
        },
    )
    assert update_response.status_code == 200
    updated_worker = update_response.json()
    assert updated_worker["alias"] == "overflow"
    assert updated_worker["canvas_position"] == {"x": 9, "y": 10}

    invalid_sort_order = client.patch(
        f"/api/workforces/{workforce['id']}/agents/{added_worker_id}",
        headers=headers,
        json={"sort_order": None},
    )
    assert invalid_sort_order.status_code == 400

    detail_response = client.get(f"/api/workforces/{workforce['id']}", headers=headers)
    assert detail_response.status_code == 200
    added_worker = next(
        worker
        for worker in detail_response.json()["workers"]
        if worker["id"] == added_worker_id
    )
    assert added_worker["sort_order"] == 2

    delete_response = client.delete(
        f"/api/workforces/{workforce['id']}/agents/{added_worker_id}",
        headers=headers,
    )
    assert delete_response.status_code == 200
    assert delete_response.json() == {"status": "deleted"}

    publish_response = client.post(
        f"/api/workforces/{workforce['id']}/publish",
        headers=headers,
    )
    assert publish_response.status_code == 200
    only_worker_id = publish_response.json()["workers"][0]["id"]

    invalid_disable = client.patch(
        f"/api/workforces/{workforce['id']}/agents/{only_worker_id}",
        headers=headers,
        json={"enabled": False},
    )
    assert invalid_disable.status_code == 400

    detail_response = client.get(f"/api/workforces/{workforce['id']}", headers=headers)
    assert detail_response.status_code == 200
    assert detail_response.json()["workers"][0]["enabled"] is True


def test_archived_workforce_rejects_all_edit_boundaries() -> None:
    headers = _admin_headers()
    workforce = _create_workforce(headers, name="Archived Workforce")
    owner_id = _user_id()
    worker_id = workforce["workers"][0]["id"]

    archive_response = client.delete(
        f"/api/workforces/{workforce['id']}",
        headers=headers,
    )
    assert archive_response.status_code == 200

    patch_response = client.patch(
        f"/api/workforces/{workforce['id']}",
        headers=headers,
        json={"description": "updated"},
    )
    assert patch_response.status_code == 409

    add_response = client.post(
        f"/api/workforces/{workforce['id']}/agents",
        headers=headers,
        json={
            "source_type": "existing",
            "agent_id": _create_published_agent(owner_id, "Archived Late Worker"),
            "assignment_instructions": "Should not be added",
        },
    )
    assert add_response.status_code == 409

    update_worker_response = client.patch(
        f"/api/workforces/{workforce['id']}/agents/{worker_id}",
        headers=headers,
        json={"alias": "blocked"},
    )
    assert update_worker_response.status_code == 409

    remove_worker_response = client.delete(
        f"/api/workforces/{workforce['id']}/agents/{worker_id}",
        headers=headers,
    )
    assert remove_worker_response.status_code == 409


def test_from_prompt_creates_draft_workforce() -> None:
    headers = _admin_headers()
    _create_published_agent(_user_id(), "Research Worker")

    response = client.post(
        "/api/workforces/from-prompt",
        headers=headers,
        json={"prompt": "Create a research workforce for product analysis"},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["id"]
    assert payload["status"] == "draft"
    assert payload["manager"]["status"] == "published"
    assert "manager_instructions" not in payload

    db = _direct_db_session()
    try:
        manager_agent = db.get(Agent, payload["manager"]["id"])
        assert manager_agent is not None
        assert manager_agent.instructions
    finally:
        db.close()


def test_run_endpoint_delegates_to_run_service(monkeypatch: pytest.MonkeyPatch) -> None:
    headers = _admin_headers()
    workforce = _create_workforce(headers, name="Run Workforce")
    captured: dict[str, Any] = {}

    async def fake_start_workforce_run(
        db: Any,
        user: User,
        workforce_arg: Any,
        *,
        message: str,
        selected_file_ids: list[str] | None = None,
        execution_mode: str | None = None,
        is_preview: bool = False,
        is_visible: bool = True,
    ) -> Any:
        captured.update(
            {
                "user_id": int(user.id),
                "workforce_id": int(workforce_arg.id),
                "message": message,
                "selected_file_ids": selected_file_ids,
                "execution_mode": execution_mode,
                "is_preview": is_preview,
                "is_visible": is_visible,
            }
        )
        return SimpleNamespace(
            workforce_run=SimpleNamespace(id=99, status="pending"),
            task=SimpleNamespace(id=123),
        )

    monkeypatch.setattr(
        workforces_api,
        "start_workforce_run",
        fake_start_workforce_run,
    )

    response = client.post(
        f"/api/workforces/{workforce['id']}/runs",
        headers=headers,
        json={"message": "go", "files": ["file-1"], "execution_mode": "think"},
    )
    assert response.status_code == 200, response.text
    assert response.json() == {
        "workforce_run_id": 99,
        "task_id": 123,
        "status": "pending",
        "redirect_url": "/task/123",
    }
    assert captured == {
        "user_id": _user_id(),
        "workforce_id": workforce["id"],
        "message": "go",
        "selected_file_ids": ["file-1"],
        "execution_mode": "think",
        "is_preview": False,
        "is_visible": True,
    }


def test_list_workforce_runs_orders_paginates_and_excludes_previews() -> None:
    headers = _admin_headers()
    owner_id = _user_id()
    workforce = _create_workforce(headers, name="Runs History Workforce")
    workforce_id = int(workforce["id"])
    now = datetime.now(timezone.utc)

    task_id = _create_task(
        owner_id,
        title="Runs History Workforce: analyze",
        description="analyze " + "x" * 300,
    )
    oldest_run_id = _create_workforce_run(
        workforce_id=workforce_id,
        user_id=owner_id,
        status="completed",
        created_at=now,
        task_id=task_id,
        completed_at=now + timedelta(minutes=1),
    )
    preview_run_id = _create_workforce_run(
        workforce_id=workforce_id,
        user_id=owner_id,
        status="completed",
        created_at=now + timedelta(minutes=5),
        is_preview=True,
    )
    latest_run_id = _create_workforce_run(
        workforce_id=workforce_id,
        user_id=owner_id,
        status="running",
        created_at=now + timedelta(minutes=10),
    )

    response = client.get(f"/api/workforces/{workforce_id}/runs", headers=headers)
    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] == 2
    assert payload["pages"] == 1
    assert [item["id"] for item in payload["items"]] == [latest_run_id, oldest_run_id]

    oldest_item = payload["items"][1]
    assert oldest_item["task_id"] == task_id
    assert oldest_item["status"] == "completed"
    assert oldest_item["is_preview"] is False
    assert oldest_item["task_title"] == "Runs History Workforce: analyze"
    assert oldest_item["message"].endswith("...")
    assert len(oldest_item["message"]) == 203
    assert oldest_item["completed_at"] is not None

    latest_item = payload["items"][0]
    assert latest_item["task_id"] is None
    assert latest_item["task_title"] is None
    assert latest_item["message"] is None
    assert latest_item["completed_at"] is None

    with_previews = client.get(
        f"/api/workforces/{workforce_id}/runs",
        headers=headers,
        params={"include_preview": "true"},
    )
    assert with_previews.status_code == 200
    preview_payload = with_previews.json()
    assert preview_payload["total"] == 3
    assert [item["id"] for item in preview_payload["items"]] == [
        latest_run_id,
        preview_run_id,
        oldest_run_id,
    ]
    assert preview_payload["items"][1]["is_preview"] is True

    paged = client.get(
        f"/api/workforces/{workforce_id}/runs",
        headers=headers,
        params={"page": 2, "size": 1},
    )
    assert paged.status_code == 200
    paged_payload = paged.json()
    assert paged_payload["total"] == 2
    assert paged_payload["pages"] == 2
    assert [item["id"] for item in paged_payload["items"]] == [oldest_run_id]

    invalid = client.get(
        f"/api/workforces/{workforce_id}/runs",
        headers=headers,
        params={"page": 0},
    )
    assert invalid.status_code == 400

    # Preview runs are also excluded from the list endpoint's last_run.
    list_response = client.get(
        "/api/workforces",
        headers=headers,
        params={"search": "Runs History Workforce"},
    )
    assert list_response.status_code == 200
    assert list_response.json()["items"][0]["last_run"]["id"] == latest_run_id


def test_workforce_run_detail_and_access_control() -> None:
    headers = _admin_headers()
    owner_id = _user_id()
    workforce = _create_workforce(headers, name="Run Detail Workforce")
    workforce_id = int(workforce["id"])
    now = datetime.now(timezone.utc)
    run_id = _create_workforce_run(
        workforce_id=workforce_id,
        user_id=owner_id,
        status="completed",
        created_at=now,
    )

    detail = client.get(
        f"/api/workforces/{workforce_id}/runs/{run_id}", headers=headers
    )
    assert detail.status_code == 200
    detail_payload = detail.json()
    assert detail_payload["id"] == run_id
    assert detail_payload["snapshot"] == {"workforce": {"id": workforce_id}}

    missing = client.get(
        f"/api/workforces/{workforce_id}/runs/{run_id + 999}", headers=headers
    )
    assert missing.status_code == 404

    other_headers = _register_second_user()
    other_workforce = _create_workforce(
        other_headers,
        name="Other Runs Workforce",
        username="bob",
    )
    denied_list = client.get(
        f"/api/workforces/{workforce_id}/runs", headers=other_headers
    )
    assert denied_list.status_code == 403
    denied_detail = client.get(
        f"/api/workforces/{workforce_id}/runs/{run_id}", headers=other_headers
    )
    assert denied_detail.status_code == 403

    # A run id from another workforce is not addressable through this one.
    cross_lookup = client.get(
        f"/api/workforces/{other_workforce['id']}/runs/{run_id}",
        headers=other_headers,
    )
    assert cross_lookup.status_code == 404


def test_canvas_read_returns_nodes_edges_and_layout() -> None:
    headers = _admin_headers()
    workforce = _create_workforce(
        headers,
        name="Canvas Workforce",
        canvas_layout={"zoom": 0.8},
    )

    response = client.get(f"/api/workforces/{workforce['id']}/canvas", headers=headers)
    assert response.status_code == 200
    payload = response.json()
    assert payload["layout"] == {"zoom": 0.8}
    assert [node["type"] for node in payload["nodes"]] == ["human", "manager", "worker"]
    assert [edge["source"] for edge in payload["edges"]] == [
        "human",
        f"manager-{workforce['manager']['id']}",
    ]
