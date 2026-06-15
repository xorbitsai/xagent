from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any

import pytest
from sqlalchemy import event

from xagent.web.api import workforces as workforces_api
from xagent.web.models.agent import Agent, AgentOrigin, AgentStatus
from xagent.web.models.database import get_engine
from xagent.web.models.user import User
from xagent.web.models.workforce import Workforce, WorkforceBuilderMessage, WorkforceRun
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
) -> int:
    db = _direct_db_session()
    try:
        run = WorkforceRun(
            workforce_id=workforce_id,
            task_id=None,
            user_id=user_id,
            status=status,
            snapshot={"workforce": {"id": workforce_id}},
            created_at=created_at,
        )
        db.add(run)
        db.commit()
        db.refresh(run)
        return int(run.id)
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

    db = _direct_db_session()
    try:
        message = WorkforceBuilderMessage(
            workforce_id=workforce["id"],
            user_id=owner_id,
            role="assistant",
            content="Prepared patch.",
            proposed_patch={
                "summary": "Rename archived workforce.",
                "operations": [
                    {
                        "op": "update_workforce",
                        "fields": {"name": "Renamed Archived Workforce"},
                    }
                ],
                "warnings": [],
                "clarification": None,
            },
            status="proposed",
        )
        db.add(message)
        db.commit()
        db.refresh(message)
        message_id = int(message.id)
        proposed_patch = message.proposed_patch
    finally:
        db.close()

    apply_response = client.post(
        f"/api/workforces/{workforce['id']}/builder/apply",
        headers=headers,
        json={"message_id": message_id, "proposed_patch": proposed_patch},
    )
    assert apply_response.status_code == 409


def test_builder_propose_apply_requires_stored_patch_match() -> None:
    headers = _admin_headers()
    workforce = _create_workforce(headers, name="Builder Workforce")

    propose_response = client.post(
        f"/api/workforces/{workforce['id']}/builder/propose",
        headers=headers,
        json={"message": 'rename "Renamed Workforce"'},
    )
    assert propose_response.status_code == 200, propose_response.text
    propose_payload = propose_response.json()
    assert isinstance(propose_payload["assistant_message"], str)
    assert propose_payload["message_id"] == propose_payload["message"]["id"]
    assert propose_payload["message"]["status"] == "proposed"
    assert (
        propose_payload["message"]["proposed_patch"]
        == propose_payload["proposed_patch"]
    )

    patch = {
        "summary": "Rename Workforce.",
        "operations": [
            {
                "op": "update_workforce",
                "fields": {"name": "Renamed Workforce"},
            }
        ],
        "warnings": [],
        "clarification": None,
    }
    owner_id = _user_id()
    db = _direct_db_session()
    try:
        message = WorkforceBuilderMessage(
            workforce_id=workforce["id"],
            user_id=owner_id,
            role="assistant",
            content="I prepared 1 change for review.",
            proposed_patch=patch,
            status="proposed",
        )
        db.add(message)
        db.commit()
        db.refresh(message)
        message_id = int(message.id)
    finally:
        db.close()

    tampered_patch = {**patch, "summary": "tampered"}
    bad_apply = client.post(
        f"/api/workforces/{workforce['id']}/builder/apply",
        headers=headers,
        json={"message_id": message_id, "proposed_patch": tampered_patch},
    )
    assert bad_apply.status_code == 400
    assert "does not match" in bad_apply.json()["detail"]

    apply_response = client.post(
        f"/api/workforces/{workforce['id']}/builder/apply",
        headers=headers,
        json={"message_id": message_id, "proposed_patch": patch},
    )
    assert apply_response.status_code == 200, apply_response.text
    apply_payload = apply_response.json()
    assert apply_payload["message_id"] == message_id
    assert apply_payload["message"]["status"] == "applied"
    assert apply_payload["workforce"]["name"] == "Renamed Workforce"

    messages_response = client.get(
        f"/api/workforces/{workforce['id']}/builder/messages",
        headers=headers,
    )
    assert messages_response.status_code == 200
    message_roles = [item["role"] for item in messages_response.json()["items"]]
    assert message_roles.count("user") == 1
    assert message_roles.count("assistant") == 2


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

    messages_response = client.get(
        f"/api/workforces/{payload['id']}/builder/messages",
        headers=headers,
    )
    assert messages_response.status_code == 200
    assert len(messages_response.json()["items"]) == 2


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
        is_visible: bool = True,
    ) -> Any:
        captured.update(
            {
                "user_id": int(user.id),
                "workforce_id": int(workforce_arg.id),
                "message": message,
                "selected_file_ids": selected_file_ids,
                "execution_mode": execution_mode,
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
        "is_visible": True,
    }


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
