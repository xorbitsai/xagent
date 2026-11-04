"""Builder history reuses owner-scoped tasks and their persisted transcript."""

import asyncio

import pytest
from sqlalchemy.orm import Session

from tests.shared.postgres_disposable import disposable_database_factory
from xagent.web.api.agents import get_agent_preview_task
from xagent.web.models.agent import Agent
from xagent.web.models.chat_message import TaskChatMessage
from xagent.web.models.database import Base
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.user import User
from xagent.web.services import agent_team_scope
from xagent.web.services.chat_history_service import (
    DELIVERY_FAILED,
    DELIVERY_OUTCOME_UNKNOWN,
)

from .conftest import (
    _admin_headers,
    _direct_db_session,
    _register_second_user,
    client,
)

pytestmark = pytest.mark.usefixtures("_test_db")


def create_agent(headers, name="Preview agent"):
    response = client.post("/api/agents", headers=headers, json={"name": name})
    assert response.status_code == 200, response.text
    return response.json()


def create_preview(user_id, agent_id=None, **overrides):
    values = {
        "user_id": user_id,
        "title": "Sample",
        "description": "Original sample",
        "status": TaskStatus.COMPLETED,
        "is_visible": False,
        "agent_config": {
            "is_preview": True,
            "preview_agent_id": agent_id,
            "preview_config_key": "config-v1",
        },
    }
    values.update(overrides)
    with _direct_db_session() as db:
        task = Task(**values)
        db.add(task)
        db.flush()
        task_id = int(task.id)
        db.add(
            TaskChatMessage(
                task_id=task_id,
                user_id=user_id,
                role="user",
                message_type="message",
                content="Original sample",
                attachments=[
                    {"file_id": "sample-id", "name": "sample.csv", "size": 12}
                ],
            )
        )
        db.commit()
    return task_id


def test_latest_preview_restores_original_sample_not_followup():
    headers = _admin_headers()
    agent = create_agent(headers)
    url = f"/api/agents/{agent['id']}/preview-task"
    assert client.get(url, headers=headers).json() is None
    create_preview(agent["user_id"], agent["id"])
    latest = create_preview(agent["user_id"], agent["id"], status=TaskStatus.FAILED)
    with _direct_db_session() as db:
        db.add(
            TaskChatMessage(
                task_id=latest,
                user_id=agent["user_id"],
                role="user",
                message_type="message",
                content="Follow-up",
            )
        )
        db.commit()
    # Newer unrelated and visible tasks must not displace the latest preview.
    create_preview(agent["user_id"], agent["id"], is_visible=True)
    create_preview(agent["user_id"], agent["id"], agent_config={})
    response = client.get(url, headers=headers)
    assert response.status_code == 200
    assert response.json() == {
        "task_id": latest,
        "config_key": "config-v1",
        "message": "Original sample",
        "attachments": [{"file_id": "sample-id", "name": "sample.csv", "size": 12}],
    }


def test_attachment_only_sample_returns_empty_text_without_losing_files():
    headers = _admin_headers()
    agent = create_agent(headers)
    task_id = create_preview(agent["user_id"], agent["id"])
    with _direct_db_session() as db:
        sample = db.query(TaskChatMessage).filter_by(task_id=task_id).one()
        # Attachment-only turns persist an empty string, not SQL NULL.
        sample.content = ""
        db.commit()
    response = client.get(f"/api/agents/{agent['id']}/preview-task", headers=headers)
    assert response.status_code == 200
    assert response.json()["message"] == ""
    assert response.json()["attachments"] == [
        {"file_id": "sample-id", "name": "sample.csv", "size": 12}
    ]


@pytest.mark.parametrize("has_previous", [False, True])
def test_unsubmitted_preview_does_not_replace_previous_sample(has_previous):
    headers = _admin_headers()
    agent = create_agent(headers)
    previous = create_preview(agent["user_id"], agent["id"]) if has_previous else None
    empty = create_preview(agent["user_id"], agent["id"], status=TaskStatus.PENDING)
    with _direct_db_session() as db:
        # Task creation precedes file upload. A failed upload has no submitted
        # user turn; even an assistant-only diagnostic is not a retry sample.
        sample = db.query(TaskChatMessage).filter_by(task_id=empty).one()
        sample.role = "assistant"
        db.commit()
    response = client.get(f"/api/agents/{agent['id']}/preview-task", headers=headers)
    assert response.status_code == 200
    if previous is None:
        assert response.json() is None
    else:
        assert response.json()["task_id"] == previous
        assert response.json()["attachments"][0]["file_id"] == "sample-id"


def test_link_presave_preview_without_changing_runtime_identity():
    headers = _admin_headers()
    agent = create_agent(headers)
    task_id = create_preview(agent["user_id"])
    url = f"/api/agents/{agent['id']}/preview-task"
    for _ in range(2):
        response = client.put(url, headers=headers, json={"task_id": task_id})
        assert response.status_code == 204, response.text
    assert client.get(url, headers=headers).json()["task_id"] == task_id
    with _direct_db_session() as db:
        task = db.get(Task, task_id)
        assert task.agent_id is None
        assert task.agent_config == {
            "is_preview": True,
            "preview_agent_id": None,
            "preview_config_key": "config-v1",
            "preview_history_agent_id": agent["id"],
        }
    other = create_agent(headers, "Other")
    assert (
        client.put(
            f"/api/agents/{other['id']}/preview-task",
            headers=headers,
            json={"task_id": task_id},
        ).status_code
        == 409
    )


@pytest.mark.parametrize("overrides", [{"is_visible": True}, {"agent_config": {}}])
def test_cannot_link_normal_tasks(overrides):
    headers = _admin_headers()
    agent = create_agent(headers)
    task_id = create_preview(agent["user_id"], **overrides)
    assert (
        client.put(
            f"/api/agents/{agent['id']}/preview-task",
            headers=headers,
            json={"task_id": task_id},
        ).status_code
        == 404
    )


def test_cross_user_tasks_are_private_even_for_admin():
    admin = _admin_headers()
    bob = _register_second_user()
    admin_agent = create_agent(admin)
    bob_agent = create_agent(bob)
    admin_task = create_preview(admin_agent["user_id"], admin_agent["id"])
    bob_task = create_preview(bob_agent["user_id"], admin_agent["id"])
    admin_url = f"/api/agents/{admin_agent['id']}/preview-task"
    bob_url = f"/api/agents/{bob_agent['id']}/preview-task"
    assert client.get(admin_url, headers=admin).json()["task_id"] == admin_task
    assert client.get(admin_url, headers=bob).status_code == 404
    assert client.get(bob_url, headers=admin).status_code == 404
    assert (
        client.put(
            admin_url,
            headers=admin,
            json={"task_id": bob_task},
        ).status_code
        == 404
    )
    # The task belongs to the caller, but the target agent is not editable.
    assert (
        client.put(bob_url, headers=admin, json={"task_id": admin_task}).status_code
        == 404
    )


def test_runtime_preview_cannot_be_read_or_bound_as_another_agent():
    headers = _admin_headers()
    first = create_agent(headers, "First")
    second = create_agent(headers, "Second")
    task_id = create_preview(first["user_id"], first["id"])
    url = f"/api/agents/{second['id']}/preview-task"
    assert client.get(url, headers=headers).json() is None
    assert (
        client.put(url, headers=headers, json={"task_id": task_id}).status_code == 409
    )


def test_team_co_editors_only_see_and_bind_their_own_previews(monkeypatch):
    admin = _admin_headers()
    bob = _register_second_user()
    agent = create_agent(admin)
    bob_agent = create_agent(bob)
    user_ids = {agent["user_id"], bob_agent["user_id"]}
    monkeypatch.setattr(
        agent_team_scope,
        "_agent_team_scope_hook",
        lambda _db, uid: (
            agent_team_scope.AgentTeamScope(team_id=100, is_team_admin=False)
            if uid in user_ids
            else None
        ),
    )
    with _direct_db_session() as db:
        db.get(Agent, agent["id"]).team_id = 100
        db.commit()
    assert client.get(f"/api/agents/{agent['id']}", headers=bob).json()["can_edit"]
    url = f"/api/agents/{agent['id']}/preview-task"
    admin_task = create_preview(agent["user_id"], agent["id"])
    assert client.get(url, headers=bob).json() is None
    bob_task = create_preview(bob_agent["user_id"])
    assert client.put(url, headers=bob, json={"task_id": bob_task}).status_code == 204
    assert client.get(url, headers=admin).json()["task_id"] == admin_task
    assert client.get(url, headers=bob).json()["task_id"] == bob_task
    assert client.put(url, headers=bob, json={"task_id": admin_task}).status_code == 404


@pytest.mark.parametrize("delivery_status", [DELIVERY_FAILED, DELIVERY_OUTCOME_UNKNOWN])
def test_original_failed_or_unknown_sample_can_be_retried(delivery_status):
    headers = _admin_headers()
    agent = create_agent(headers)
    task_id = create_preview(agent["user_id"], agent["id"], status=TaskStatus.FAILED)
    with _direct_db_session() as db:
        sample = db.query(TaskChatMessage).filter_by(task_id=task_id).one()
        sample.delivery_status = delivery_status
        db.commit()
    response = client.get(f"/api/agents/{agent['id']}/preview-task", headers=headers)
    assert response.status_code == 200
    assert response.json()["message"] == "Original sample"
    assert response.json()["attachments"][0]["file_id"] == "sample-id"


def malformed_preview_configs(agent_id):
    return [
        {"is_preview": "abc", "preview_agent_id": agent_id},
        {"is_preview": "true", "preview_agent_id": agent_id},
        {"is_preview": True, "preview_agent_id": "abc"},
        {"is_preview": True, "preview_history_agent_id": "abc"},
        {"is_preview": True, "preview_agent_id": "9" * 100},
        {"is_preview": True, "preview_agent_id": {"id": agent_id}},
        {"is_preview": True, "preview_agent_id": [agent_id]},
        {"is_preview": True, "preview_agent_id": None},
    ]


def test_malformed_preview_metadata_does_not_break_history():
    headers = _admin_headers()
    agent = create_agent(headers)
    task_id = create_preview(agent["user_id"], agent["id"])
    for config in malformed_preview_configs(agent["id"]):
        create_preview(agent["user_id"], agent_config=config)
    response = client.get(f"/api/agents/{agent['id']}/preview-task", headers=headers)
    assert response.status_code == 200
    assert response.json()["task_id"] == task_id


@pytest.mark.postgresql
def test_postgresql_preview_query_ignores_malformed_metadata():
    with disposable_database_factory("preview_history") as make_database:
        engine = make_database("metadata")
        Base.metadata.create_all(engine)
        with Session(engine) as db:
            user = User(username="preview-history-test", password_hash="unused")
            db.add(user)
            db.flush()
            agent = Agent(user_id=user.id, name="Preview history")
            db.add(agent)
            db.flush()
            task = Task(
                user_id=user.id,
                title="Valid preview",
                is_visible=False,
                agent_config={"is_preview": True, "preview_agent_id": agent.id},
            )
            db.add(task)
            db.flush()
            db.add(
                TaskChatMessage(
                    task_id=task.id,
                    user_id=user.id,
                    role="user",
                    content="Sample",
                    message_type="message",
                )
            )
            for config in malformed_preview_configs(agent.id):
                malformed = Task(
                    user_id=user.id,
                    title="Malformed metadata",
                    is_visible=False,
                    agent_config=config,
                )
                db.add(malformed)
                db.flush()
                db.add(
                    TaskChatMessage(
                        task_id=malformed.id,
                        user_id=user.id,
                        role="user",
                        content="Malformed preview sample",
                        message_type="message",
                    )
                )
            db.add(
                Task(
                    user_id=user.id,
                    title="Upload failed before submission",
                    is_visible=False,
                    agent_config={"is_preview": True, "preview_agent_id": agent.id},
                )
            )
            db.commit()
            result = asyncio.run(get_agent_preview_task(agent.id, user, db))
            assert result is not None
            assert result.task_id == task.id
            assert result.message == "Sample"
            assert result.attachments == []
            # Binding metadata uses the same safe predicate as runtime metadata.
            task.agent_config = {
                "is_preview": True,
                "preview_history_agent_id": agent.id,
            }
            db.commit()
            assert (
                asyncio.run(get_agent_preview_task(agent.id, user, db)).task_id
                == task.id
            )


def test_legacy_preview_without_snapshot_is_not_current_config():
    headers = _admin_headers()
    agent = create_agent(headers)
    task_id = create_preview(
        agent["user_id"],
        agent_config={
            "is_preview": True,
            "preview_agent_id": agent["id"],
        },
    )
    response = client.get(f"/api/agents/{agent['id']}/preview-task", headers=headers)
    assert response.json()["task_id"] == task_id
    assert response.json()["config_key"] is None


def test_deleted_preview_is_not_returned():
    headers = _admin_headers()
    agent = create_agent(headers)
    task_id = create_preview(agent["user_id"], agent["id"])
    with _direct_db_session() as db:
        db.delete(db.get(Task, task_id))
        db.commit()
    assert (
        client.get(f"/api/agents/{agent['id']}/preview-task", headers=headers).json()
        is None
    )
