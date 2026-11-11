"""Real hosts: the storage-entry switch picks V2 per entry and rolls back safely."""

import json

import pytest

from tests.e2e.shared_execution_harness import (
    create_sdk_task,
    create_task,
    receive_event,
    restart_hosts,
    ws_url,
)

pytestmark = pytest.mark.e2e

ENTRIES = "XAGENT_EXECUTION_EVENT_STORAGE_ENTRIES"


@pytest.fixture(autouse=True)
def _only_sdk_enabled(monkeypatch):
    # Inherited by the hosts the shared_app fixture spawns after this.
    monkeypatch.setenv(ENTRIES, "sdk")


def _storage_version(task_id: int) -> int:
    from xagent.web.models.database import get_session_local
    from xagent.web.models.task import Task

    with get_session_local()() as db:
        return db.get(Task, task_id).conversation_storage_version


def test_only_enabled_entries_create_v2_tasks(shared_app):
    from websockets.sync.client import connect

    app = shared_app
    sdk_id, run_id, _ = create_sdk_task(app, "mixed sdk")
    web_id = create_task(app, "Mixed web", "Answer over websocket")
    assert _storage_version(sdk_id) == 2
    assert _storage_version(web_id) == 1

    with connect(ws_url(app, web_id), additional_headers=app.headers) as ws:
        receive_event(ws, "historical_data_complete")
        ws.send(json.dumps({"type": "execute_task"}))
        receive_event(ws, "task_completed")
    web = app.wait_task(web_id)
    sdk = app.wait_task(sdk_id, run_id=run_id)
    assert "Shared E2E answer" in web["output"]
    assert "Shared E2E answer" in sdk["output"]

    # Reconnecting replays the V1 task's history from its legacy storage.
    with connect(ws_url(app, web_id), additional_headers=app.headers) as ws:
        _, history = receive_event(ws, "historical_data_complete")
        assert "Shared E2E answer" in json.dumps(history)
    assert (_storage_version(sdk_id), _storage_version(web_id)) == (2, 1)


def test_switch_off_keeps_existing_v2_task_and_new_tasks_are_v1(shared_app):
    app = shared_app
    agent_id, headers = app.create_agent()
    task_id, _ = _sdk_task(app, agent_id, headers, "e2e:ask")
    assert _storage_version(task_id) == 2
    waiting = app.wait_task(task_id, status="waiting_for_user")

    # Roll back: every host restarts with the switch off.
    restart_hosts(app, "web", "worker", **{ENTRIES: ""})

    response = app.client.post(
        f"/v1/chat/tasks/{task_id}/reply",
        headers=headers,
        json={
            "agent_id": agent_id,
            "message": {"role": "user", "content": "e2e:answer"},
            "command_id": "e2e-rollback-reply",
        },
    )
    assert response.status_code == 202, response.text
    completed = app.wait_task(task_id)
    assert completed["run_id"] == waiting["run_id"]
    assert "Shared E2E answer" in completed["output"]
    assert _storage_version(task_id) == 2

    later_id, later_run = _sdk_task(app, agent_id, headers, "after rollback")
    assert _storage_version(later_id) == 1
    assert "Shared E2E answer" in app.wait_task(later_id, run_id=later_run)["output"]
    assert _storage_version(task_id) == 2


def _sdk_task(app, agent_id, headers, content) -> tuple[int, str]:
    response = app.client.post(
        "/v1/chat/tasks",
        headers=headers,
        json={"agent_id": agent_id, "message": {"role": "user", "content": content}},
    )
    assert response.status_code == 202, response.text
    accepted = response.json()
    return accepted["task_id"], accepted["run_id"]
