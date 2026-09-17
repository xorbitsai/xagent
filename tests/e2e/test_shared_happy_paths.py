"""Successful shared turns through each public conversation surface."""

import json

import pytest
from websockets.sync.client import connect

from tests.e2e.shared_execution_harness import receive_event

pytestmark = pytest.mark.e2e


def _chat_access(app, interface, tool_categories=None):
    agent_id, _ = app.create_agent(tool_categories=tool_categories)
    token = app.token
    prefix = "/api/chat"
    if interface != "owner":
        response = app.client.post(
            f"/api/agents/{agent_id}/publish", headers=app.headers
        )
        assert response.status_code == 200, response.text
        if interface == "widget":
            response = app.client.put(
                f"/api/agents/{agent_id}",
                headers=app.headers,
                json={"widget_enabled": True, "allowed_domains": ["*"]},
            )
            assert response.status_code == 200, response.text
            response = app.client.get(
                f"/api/agents/{agent_id}/widget-key", headers=app.headers
            )
            assert response.status_code == 200, response.text
            auth = {"widget_key": response.json()["widget_key"], "guest_id": "happy"}
        else:
            response = app.client.post(
                f"/api/agents/{agent_id}/share-link", headers=app.headers
            )
            assert response.status_code == 200, response.text
            auth = {"share_token": response.json()["share_token"]}
        response = app.client.post(f"/api/{interface}/auth", json=auth)
        assert response.status_code == 200, response.text
        token = response.json()["access_token"]
        prefix = f"/api/{interface}/chat"
    return agent_id, token, prefix


@pytest.mark.parametrize("interface", ["owner", "widget", "share"])
def test_agent_chat_reply_and_followup(shared_app, interface):
    app = shared_app
    agent_id, token, prefix = _chat_access(app, interface)
    response = app.client.post(
        f"{prefix}/task/create",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "title": "Happy conversation",
            "description": "e2e:ask",
            "agent_id": agent_id,
            "execution_mode": "balanced",
        },
    )
    assert response.status_code == 200, response.text
    task_id = response.json()["task_id"]
    ws_path = (
        f"/ws/chat/{task_id}" if interface == "owner" else f"{prefix}/ws/{task_id}"
    )
    url = (
        str(app.client.base_url).replace("http://", "ws://").rstrip("/")
        + ws_path
        + f"?token={token}"
    )
    with connect(url) as ws:
        receive_event(ws, "historical_data_complete")
        ws.send(
            json.dumps(
                {"type": "chat", "message": "e2e:ask", "client_message_id": "question"}
            )
        )
        receive_event(ws, "message_accepted")
        waiting = app.wait_task(task_id, status="waiting_for_user")
        ws.send(
            json.dumps(
                {"type": "chat", "message": "e2e:answer", "client_message_id": "answer"}
            )
        )
        receive_event(ws, "message_accepted")
        receive_event(ws, "task_completed")
        first = app.wait_task(task_id, run_id=waiting["run_id"])
        assert "Shared E2E answer" in first["output"]
        ws.send(
            json.dumps(
                {
                    "type": "chat",
                    "message": "Continue the conversation",
                    "client_message_id": "followup",
                }
            )
        )
        receive_event(ws, "message_accepted")
        receive_event(ws, "task_completed")
        second = app.wait_task(task_id)
        assert second["run_id"] != first["run_id"]
        assert "Shared E2E answer" in second["output"]
    calls = [
        json.loads(line)
        for path in app.root.glob("model-*.jsonl")
        for line in path.read_text().splitlines()
    ]
    assert {call["role"] for call in calls} == {"worker"}
    assert any(
        "Continue the conversation" in json.dumps(call["messages"]) for call in calls
    )
    with connect(url) as ws:
        _, history = receive_event(ws, "historical_data_complete")
        assert "Continue the conversation" in json.dumps(history)
        assert "Shared E2E answer" in json.dumps(history)


def test_a2a_blocking_send_and_context_continuation(shared_app):
    app = shared_app
    agent_id, headers = app.create_agent()
    response = app.client.post(f"/api/agents/{agent_id}/publish", headers=app.headers)
    assert response.status_code == 200, response.text
    headers["A2A-Version"] = "1.0"
    task_id = None
    first_run = None
    context_id = None
    for message_id in ["first", "followup"]:
        message = {
            "messageId": message_id,
            "role": "ROLE_USER",
            "parts": [{"text": "Happy first"}]
            if message_id == "first"
            else [{"data": {"question": "Happy followup"}}],
        }
        if task_id is not None:
            message["contextId"] = context_id
        response = app.client.post(
            f"/api/a2a/agents/{agent_id}/message:send",
            headers=headers,
            json={"message": message},
        )
        assert response.status_code == 200, response.text
        task = response.json()["task"]
        assert task["status"]["state"] == "TASK_STATE_COMPLETED"
        assert "Shared E2E answer" in json.dumps(task)
        if task_id is not None:
            assert int(task["id"]) != task_id
            assert task["contextId"] == context_id
        context_id = task["contextId"]
        task_id = int(task["id"])
        settled = app.wait_task(task_id)
        assert settled["run_id"] != first_run
        first_run = settled["run_id"]
        polled = app.client.get(
            f"/api/a2a/agents/{agent_id}/tasks/{task_id}", headers=headers
        )
        assert polled.status_code == 200, polled.text
        assert "Shared E2E answer" in polled.text


@pytest.mark.parametrize(
    "mode,route",
    [
        ("flash", ""),
        ("balanced", ""),
        ("think", ""),
        ("auto", "react"),
        ("auto", "final"),
        ("auto", "dag"),
    ],
)
def test_execution_modes_reach_final_result(shared_app, mode, route):
    app = shared_app
    response = app.client.post(
        "/api/chat/task/create",
        headers=app.headers,
        json={
            "title": "Mode happy path",
            "description": f"Answer the question in English e2e:route-{route}",
            "execution_mode": mode,
        },
    )
    assert response.status_code == 200, response.text
    task_id = response.json()["task_id"]
    url = (
        str(app.client.base_url).replace("http://", "ws://").rstrip("/")
        + f"/ws/chat/{task_id}?token={app.token}"
    )
    with connect(url) as ws:
        receive_event(ws, "historical_data_complete")
        ws.send(json.dumps({"type": "execute_task"}))
        _, events = receive_event(ws, "task_completed")
        assert "Shared E2E answer" in json.dumps(events)
    assert "Shared E2E answer" in app.wait_task(task_id)["output"]
    calls = [
        json.loads(line)
        for path in app.root.glob("model-*.jsonl")
        for line in path.read_text().splitlines()
    ]
    assert {call["role"] for call in calls} == {"worker"}
    if mode == "think" or route == "dag":
        assert any("generate_execution_plan" in call["tools"] for call in calls)
        assert any("assess_dag_completion" in call["tools"] for call in calls)
    if mode == "auto":
        assert any("select_execution_pattern" in call["tools"] for call in calls)


@pytest.mark.parametrize("interface", ["owner", "widget", "share"])
def test_chat_file_input_and_output(shared_app, interface):
    from xagent.web.models.database import get_session_local
    from xagent.web.models.uploaded_file import UploadedFile

    app = shared_app
    agent_id, token, prefix = _chat_access(app, interface, ["file"])

    headers = {"Authorization": f"Bearer {token}"}
    response = app.client.post(
        f"{prefix}/task/create",
        headers=headers,
        json={
            "title": "File happy path",
            "description": "e2e:files",
            "agent_id": agent_id,
            "execution_mode": "balanced",
        },
    )
    assert response.status_code == 200, response.text
    task_id = response.json()["task_id"]
    upload_url = (
        "/api/files/upload"
        if interface == "owner"
        else f"/api/{interface}/files/upload"
    )
    uploaded = app.client.post(
        upload_url,
        headers=headers,
        data={"task_type": "task", "task_id": str(task_id)},
        files={"file": ("source.txt", b"unique shared input\n", "text/plain")},
    )
    assert uploaded.status_code == 200, uploaded.text
    file_id = uploaded.json()["file_id"]
    ws_path = (
        f"/ws/chat/{task_id}" if interface == "owner" else f"{prefix}/ws/{task_id}"
    )
    url = (
        str(app.client.base_url).replace("http://", "ws://").rstrip("/")
        + ws_path
        + f"?token={token}"
    )
    with connect(url) as ws:
        receive_event(ws, "historical_data_complete")
        ws.send(
            json.dumps(
                {
                    "type": "chat",
                    "message": "e2e:files",
                    "files": [{"file_id": file_id}],
                    "client_message_id": "files",
                }
            )
        )
        receive_event(ws, "message_accepted")
        _, events = receive_event(ws, "task_completed")
    completed = app.wait_task(task_id)
    with get_session_local()() as db:
        output_id = (
            db.query(UploadedFile)
            .filter_by(task_id=str(task_id), filename="derived.txt")
            .one()
            .file_id
        )
    assert output_id in completed["output"]
    assert output_id in json.dumps(events)
    response = app.client.get(
        f"/api/files/public/download/{output_id}", params={"token": token}
    )
    assert response.status_code == 200, response.text
    assert response.content == b"UNIQUE SHARED INPUT\n"


def test_combined_host_runs_shared_task(shared_app):
    app = shared_app
    agent_id, headers = app.create_agent()
    app.close()
    app.processes.clear()
    app.pipes.clear()
    app.start("combined")
    response = app.client.post(
        "/v1/chat/tasks",
        headers=headers,
        json={
            "agent_id": agent_id,
            "message": {"role": "user", "content": "Combined happy path"},
        },
    )
    assert response.status_code == 202, response.text
    task_id = response.json()["task_id"]
    assert "Shared E2E answer" in app.wait_task(task_id)["output"]
    polled = app.client.get(f"/v1/chat/tasks/{task_id}", headers=headers)
    assert polled.status_code == 200, polled.text
    assert "Shared E2E answer" in polled.text
    calls = [
        json.loads(line)
        for path in app.root.glob("model-*.jsonl")
        for line in path.read_text().splitlines()
    ]
    assert {call["role"] for call in calls} == {"combined"}


def test_legacy_webhook_reaches_completed_task(shared_app):
    import time

    import bcrypt

    from xagent.web.models.database import get_session_local
    from xagent.web.models.trigger import AgentTrigger, TriggerRun

    app = shared_app
    agent_id, _ = app.create_agent()
    # This deprecated endpoint only accepts persisted pre-pipeline triggers.
    with get_session_local()() as db:
        trigger = AgentTrigger(
            user_id=app.user_id,
            agent_id=agent_id,
            name="Legacy happy path",
            type="webhook",
            enabled=True,
            webhook_token="legacy-happy",
            secret_hash=bcrypt.hashpw(b"legacy-secret", bcrypt.gensalt()).decode(),
        )
        db.add(trigger)
        db.commit()
    response = app.client.post(
        "/api/triggers/webhook/legacy-happy",
        headers={
            "x-xagent-trigger-secret": "legacy-secret",
            "x-xagent-event-id": "legacy-event",
        },
        json={"question": "Legacy happy path"},
    )
    assert response.status_code == 200, response.text
    assert response.headers["deprecation"] == "true"
    run_id = response.json()["trigger_run_id"]
    deadline = time.monotonic() + 30
    task_id = None
    while time.monotonic() < deadline:
        with get_session_local()() as db:
            task_id = db.get(TriggerRun, run_id).task_id
        if task_id:
            break
        time.sleep(0.03)
    assert task_id, app.diagnostics()
    assert "Shared E2E answer" in app.wait_task(task_id)["output"]
    with get_session_local()() as db:
        assert db.get(TriggerRun, run_id).status == "completed"
