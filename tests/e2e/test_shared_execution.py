"""Public interfaces run through the real shared command and Agent runtime."""

import json
import time

import pytest

from tests.e2e.shared_execution_harness import receive_event

pytestmark = pytest.mark.e2e


def test_sdk_create_append_and_poll_execute_only_in_worker(shared_app):
    app = shared_app
    agent_id, headers = app.create_agent()
    response = app.client.post(
        "/v1/chat/tasks",
        headers=headers,
        json={
            "agent_id": agent_id,
            "message": {"role": "user", "content": "shared first"},
        },
    )
    assert response.status_code == 202, response.text
    accepted = response.json()
    first = app.wait_task(accepted["task_id"], run_id=accepted["run_id"])
    assert first["run_id"] == accepted["run_id"]
    assert "Shared E2E answer" in first["output"]
    polled = app.client.get(f"/v1/chat/tasks/{accepted['task_id']}", headers=headers)
    assert polled.status_code == 200, polled.text
    assert polled.json()["status"] == "completed"
    _, other_headers = app.create_agent("Other agent")
    assert (
        app.client.get(
            f"/v1/chat/tasks/{accepted['task_id']}", headers=other_headers
        ).status_code
        == 404
    )
    assert app.client.get(f"/v1/chat/tasks/{accepted['task_id']}").status_code == 401
    events = app.client.get(
        f"/v1/chat/tasks/{accepted['task_id']}/events", headers=headers
    )
    assert events.status_code == 200, events.text
    assert "event: task.completed" in events.text
    assert "Shared E2E answer" in events.text
    response = app.client.post(
        f"/v1/chat/tasks/{accepted['task_id']}/messages",
        headers=headers,
        json={
            "agent_id": agent_id,
            "message": {"role": "user", "content": "shared second"},
        },
    )
    assert response.status_code == 202, response.text
    second = app.wait_task(accepted["task_id"], run_id=response.json()["run_id"])
    assert second["run_id"] != first["run_id"]
    calls = [
        json.loads(line)
        for path in app.root.glob("model-*.jsonl")
        for line in path.read_text().splitlines()
    ]
    assert calls
    assert {call["role"] for call in calls} == {"worker"}


def test_websocket_execute_stream_and_reconnect(shared_app):
    from websockets.sync.client import connect

    app = shared_app
    response = app.client.post(
        "/api/chat/task/create",
        headers=app.headers,
        json={
            "title": "Shared websocket",
            "description": "Answer over websocket",
            "execution_mode": "balanced",
        },
    )
    assert response.status_code == 200, response.text
    task_id = response.json()["task_id"]
    url = (
        str(app.client.base_url).replace("http://", "ws://").rstrip("/")
        + f"/ws/chat/{task_id}?token={app.token}"
    )
    with connect(url) as ws:
        receive_event(ws, "trace_event")
        ws.send(json.dumps({"type": "execute_task"}))
        _, events = receive_event(ws, "task_completed")
        assert any(e["type"] == "execution_started" for e in events)
        deltas = [e for e in events if e["type"] == "final_answer_delta"]
        assert deltas
        assert "".join(event["delta"] for event in deltas) == "Shared E2E answer"
    app.wait_task(task_id)
    from websockets.exceptions import ConnectionClosedError

    from tests.e2e.app_harness import build_access_token
    from xagent.web.models.database import get_session_local
    from xagent.web.models.user import User

    with get_session_local()() as db:
        other = User(username="other", password_hash="unused", is_admin=False)
        db.add(other)
        db.commit()
        foreign_token = build_access_token(username="other", user_id=other.id)
    with connect(url.replace(app.token, foreign_token)) as foreign:
        with pytest.raises(ConnectionClosedError) as denied:
            foreign.recv(timeout=10)
        assert denied.value.rcvd.code == 4003
    with connect(url) as ws:
        _, history = receive_event(ws, "historical_data_complete")
        assert "Shared E2E answer" in json.dumps(history)


@pytest.mark.parametrize("interface", ["sdk", "a2a"])
def test_waiting_reply_restores_checkpoint_in_new_worker(shared_app, interface):
    app = shared_app
    agent_id, headers = app.create_agent()
    if interface == "sdk":
        response = app.client.post(
            "/v1/chat/tasks",
            headers=headers,
            json={
                "agent_id": agent_id,
                "message": {"role": "user", "content": "e2e:ask"},
            },
        )
        assert response.status_code == 202, response.text
        task_id = response.json()["task_id"]
    else:
        published = app.client.post(
            f"/api/agents/{agent_id}/publish", headers=app.headers
        )
        assert published.status_code == 200, published.text
        headers["A2A-Version"] = "1.0"
        response = app.client.post(
            f"/api/a2a/agents/{agent_id}/message:send",
            headers=headers,
            json={
                "message": {
                    "messageId": "first",
                    "role": "ROLE_USER",
                    "parts": [{"text": "e2e:ask"}],
                },
                "configuration": {"returnImmediately": True},
            },
        )
        assert response.status_code == 200, response.text
        task_id = int(response.json()["task"]["id"])
    waiting = app.wait_task(task_id, status="waiting_for_user")
    worker, pipe = app.processes[-1], app.pipes[-1]
    pipe.send("stop")
    worker.join(30)
    assert worker.exitcode == 0, app.diagnostics()
    app.start("worker")
    if interface == "sdk":
        response = app.client.post(
            f"/v1/chat/tasks/{task_id}/reply",
            headers=headers,
            json={
                "agent_id": agent_id,
                "message": {"role": "user", "content": "e2e:answer"},
                "command_id": "e2e-reply",
            },
        )
        assert response.status_code == 202, response.text
    else:
        response = app.client.post(
            f"/api/a2a/agents/{agent_id}/message:send",
            headers=headers,
            json={
                "message": {
                    "messageId": "reply",
                    "taskId": str(task_id),
                    "role": "ROLE_USER",
                    "parts": [{"text": "e2e:answer"}],
                },
                "configuration": {"returnImmediately": True},
            },
        )
        assert response.status_code == 200, response.text
    completed = app.wait_task(task_id)
    assert completed["run_id"] == waiting["run_id"]
    assert "Shared E2E answer" in completed["output"]
    assert len(list(app.root.glob("model-*.jsonl"))) == 2
    if interface == "sdk":
        replay = app.client.post(
            f"/v1/chat/tasks/{task_id}/reply",
            headers=headers,
            json={
                "agent_id": agent_id,
                "message": {"role": "user", "content": "e2e:answer"},
                "command_id": "e2e-reply",
            },
        )
        assert replay.status_code == 202, replay.text
        calls = [
            line
            for path in app.root.glob("model-*.jsonl")
            for line in path.read_text().splitlines()
        ]
        assert len(calls) == 2


def test_trigger_webhook_and_test_reach_terminal_task(shared_app):
    from xagent.web.models.database import get_session_local
    from xagent.web.models.trigger import TriggerRun
    from xagent.web.services.trigger_providers import sign_webhook_payload

    app = shared_app
    agent_id, _ = app.create_agent()
    response = app.client.post(
        f"/api/agents/{agent_id}/triggers",
        headers=app.headers,
        json={"type": "webhook", "name": "Shared webhook"},
    )
    assert response.status_code == 200, response.text
    trigger = response.json()
    response = app.client.post(
        f"/api/agents/{agent_id}/triggers/{trigger['id']}/test",
        headers=app.headers,
        json={"payload": {"subject": "test"}},
    )
    assert response.status_code == 200, response.text
    app.wait_task(response.json()["trigger_run"]["task_id"])
    url = f"/api/triggers/callback/webhook/{trigger['callback_id']}"
    payload = b'{"subject":"shared webhook"}'
    assert app.client.post(url, content=payload).status_code == 401
    timestamp = str(int(time.time()))
    headers = {
        "x-xagent-timestamp": timestamp,
        "x-xagent-event-id": "shared-event",
        "x-xagent-signature": sign_webhook_payload(
            trigger["webhook_secret"], timestamp, payload
        ),
    }
    response = app.client.post(url, content=payload, headers=headers)
    assert response.status_code == 200, response.text
    run_id = response.json()["trigger_run_ids"][0]
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        with get_session_local()() as db:
            task_id = db.get(TriggerRun, run_id).task_id
        if task_id:
            break
        time.sleep(0.03)
    assert task_id
    app.wait_task(task_id)
    duplicate = app.client.post(url, content=payload, headers=headers)
    assert duplicate.status_code == 200, duplicate.text
    assert duplicate.json()["duplicates"] == 1
    with get_session_local()() as db:
        runs = db.query(TriggerRun).all()
        assert len(runs) == 2
        assert {run.status for run in runs} == {"completed"}


def test_a2a_cancel_reaches_running_worker(shared_app):
    app = shared_app
    agent_id, headers = app.create_agent()
    assert (
        app.client.post(
            f"/api/agents/{agent_id}/publish", headers=app.headers
        ).status_code
        == 200
    )
    headers["A2A-Version"] = "1.0"
    response = app.client.post(
        f"/api/a2a/agents/{agent_id}/message:send",
        headers=headers,
        json={
            "message": {
                "messageId": "gate",
                "role": "ROLE_USER",
                "parts": [{"text": "e2e:gate"}],
            },
            "configuration": {"returnImmediately": True},
        },
    )
    assert response.status_code == 200, response.text
    task_id = int(response.json()["task"]["id"])
    deadline = time.monotonic() + 30
    while not (app.root / "model-entered").exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert (app.root / "model-entered").exists(), app.diagnostics()
    response = app.client.post(
        f"/api/a2a/agents/{agent_id}/tasks/{task_id}:cancel", headers=headers
    )
    assert response.status_code == 200, response.text
    assert response.json()["status"]["state"] == "TASK_STATE_CANCELED"
    app.wait_task(task_id, status="failed")
    assert not (app.root / "model-release").exists()


@pytest.mark.parametrize(
    "interface",
    [
        "owner",
        "sdk",
        "widget",
        "share",
        "preview",
        "trigger",
        "trigger_webhook",
        "trigger_scheduled",
    ],
)
def test_workforce_manager_and_child_execute_in_worker(shared_app, interface):
    app = shared_app
    manager_id, _ = app.create_agent(tool_categories=["file"])
    worker_id, _ = app.create_agent("Shared child")
    for agent_id in [manager_id, worker_id]:
        response = app.client.post(
            f"/api/agents/{agent_id}/publish", headers=app.headers
        )
        assert response.status_code == 200, response.text
    response = app.client.post(
        "/api/workforces",
        headers=app.headers,
        json={
            "name": "Shared workforce",
            "manager_agent_id": manager_id,
            "manager_instructions": "Delegate and synthesize.",
            "workers": [
                {
                    "source_type": "existing",
                    "agent_id": worker_id,
                    "alias": "helper",
                    "assignment_instructions": "Handle the delegated question.",
                    "enabled": True,
                }
            ],
        },
    )
    assert response.status_code == 200, response.text
    workforce_id = response.json()["id"]
    with_files = interface in {"owner", "sdk", "widget", "share", "preview"}
    message = "e2e:delegate e2e:files" if with_files else "e2e:delegate"
    file_ids = []
    if interface in {"owner", "preview"}:
        uploaded = app.client.post(
            "/api/files/upload",
            headers=app.headers,
            data={"task_type": "task"},
            files={"file": ("source.txt", b"unique shared input\n", "text/plain")},
        )
        assert uploaded.status_code == 200, uploaded.text
        file_ids = [uploaded.json()["file_id"]]
    if interface == "owner":
        published = app.client.post(
            f"/api/workforces/{workforce_id}/publish", headers=app.headers
        )
        assert published.status_code == 200, published.text
        response = app.client.post(
            f"/api/workforces/{workforce_id}/runs",
            headers=app.headers,
            json={"message": message, "execution_mode": "balanced", "files": file_ids},
        )
    elif interface == "preview":
        response = app.client.post(
            "/api/workforces/preview/runs",
            headers=app.headers,
            json={
                "name": "Preview",
                "manager_agent_id": manager_id,
                "workers": [
                    {
                        "agent_id": worker_id,
                        "assignment_instructions": "Handle the question.",
                    }
                ],
                "message": message,
                "files": file_ids,
                "execution_mode": "balanced",
            },
        )
    else:
        published = app.client.post(
            f"/api/workforces/{workforce_id}/publish", headers=app.headers
        )
        assert published.status_code == 200, published.text
        if interface == "sdk":
            key = app.client.post(
                "/api/agent-api-keys",
                headers=app.headers,
                json={"workforce_id": workforce_id, "label": "SDK"},
            )
            assert key.status_code == 200, key.text
            uploaded = app.client.post(
                "/v1/chat/files",
                headers={"Authorization": f"Bearer {key.json()['full_key']}"},
                files=[
                    ("files", ("source.txt", b"unique shared input\n", "text/plain"))
                ],
            )
            assert uploaded.status_code == 200, uploaded.text
            file_ids = [uploaded.json()["files"][0]["file_id"]]
            response = app.client.post(
                f"/v1/workforces/{workforce_id}/runs",
                headers={"Authorization": f"Bearer {key.json()['full_key']}"},
                json={
                    "message": {"role": "user", "content": message, "files": file_ids}
                },
            )
        elif interface.startswith("trigger"):
            trigger = app.client.post(
                f"/api/workforces/{workforce_id}/triggers",
                headers=app.headers,
                json={
                    "type": "scheduled"
                    if interface == "trigger_scheduled"
                    else "webhook",
                    "name": "Workforce trigger",
                    **(
                        {"config": {"interval_seconds": 3600}}
                        if interface == "trigger_scheduled"
                        else {}
                    ),
                },
            )
            assert trigger.status_code == 200, trigger.text
            if interface == "trigger":
                response = app.client.post(
                    f"/api/workforces/{workforce_id}/triggers/{trigger.json()['id']}/test",
                    headers=app.headers,
                    json={"payload": {"subject": "e2e:delegate"}},
                )
            else:
                from datetime import datetime, timedelta, timezone

                from xagent.web.models.database import get_session_local
                from xagent.web.models.trigger import AgentTrigger, TriggerRun
                from xagent.web.services.trigger_providers import sign_webhook_payload

                config = trigger.json()
                if interface == "trigger_webhook":
                    payload = b'{"subject":"e2e:delegate"}'
                    timestamp = str(int(time.time()))
                    response = app.client.post(
                        f"/api/triggers/callback/webhook/{config['callback_id']}",
                        content=payload,
                        headers={
                            "x-xagent-timestamp": timestamp,
                            "x-xagent-event-id": "workforce-happy",
                            "x-xagent-signature": sign_webhook_payload(
                                config["webhook_secret"], timestamp, payload
                            ),
                        },
                    )
                    assert response.status_code == 200, response.text
                else:
                    with get_session_local()() as db:
                        row = db.get(AgentTrigger, config["id"])
                        row.next_run_at = datetime.now(timezone.utc) - timedelta(
                            seconds=5
                        )
                        row.prompt_template = "e2e:delegate"
                        db.commit()
                deadline = time.monotonic() + 30
                trigger_task_id = None
                while time.monotonic() < deadline:
                    with get_session_local()() as db:
                        run = (
                            db.query(TriggerRun)
                            .filter_by(trigger_id=config["id"])
                            .first()
                        )
                        if run is not None:
                            trigger_task_id = run.task_id
                    if trigger_task_id:
                        break
                    time.sleep(0.03)
                assert trigger_task_id, app.diagnostics()
        else:
            if interface == "widget":
                enabled = app.client.put(
                    f"/api/workforces/{workforce_id}/widget",
                    headers=app.headers,
                    json={"widget_enabled": True, "allowed_domains": ["*"]},
                )
                assert enabled.status_code == 200, enabled.text
                auth = app.client.post(
                    "/api/widget/auth",
                    json={
                        "guest_id": "e2e",
                        "widget_key": enabled.json()["widget_key"],
                    },
                )
            else:
                enabled = app.client.post(
                    f"/api/workforces/{workforce_id}/share-link", headers=app.headers
                )
                assert enabled.status_code == 200, enabled.text
                auth = app.client.post(
                    "/api/share/auth",
                    json={"share_token": enabled.json()["share_token"]},
                )
            assert auth.status_code == 200, auth.text
            uploaded = app.client.post(
                f"/api/{interface}/files/upload",
                headers={"Authorization": f"Bearer {auth.json()['access_token']}"},
                data={"task_type": "task"},
                files={"file": ("source.txt", b"unique shared input\n", "text/plain")},
            )
            assert uploaded.status_code == 200, uploaded.text
            file_ids = [uploaded.json()["file_id"]]
            response = app.client.post(
                f"/api/{interface}/chat/task/create",
                headers={"Authorization": f"Bearer {auth.json()['access_token']}"},
                json={"title": message, "description": message, "files": file_ids},
            )
    assert response.status_code == (202 if interface == "sdk" else 200), response.text
    body = response.json()
    task_id = (
        trigger_task_id
        if interface in {"trigger_webhook", "trigger_scheduled"}
        else (body["trigger_run"] if interface == "trigger" else body)["task_id"]
    )
    first = app.wait_task(task_id)
    if with_files:
        from xagent.web.models.database import get_session_local
        from xagent.web.models.uploaded_file import UploadedFile

        with get_session_local()() as db:
            output_id = (
                db.query(UploadedFile)
                .filter_by(task_id=str(task_id), filename="derived.txt")
                .one()
                .file_id
            )
        assert output_id in first["output"]
        file_token = (
            auth.json()["access_token"]
            if interface in {"widget", "share"}
            else app.token
        )
        downloaded = app.client.get(
            f"/api/files/public/download/{output_id}", params={"token": file_token}
        )
        assert downloaded.status_code == 200, downloaded.text
        assert downloaded.content == b"UNIQUE SHARED INPUT\n"
    if interface in {"owner", "sdk", "widget", "share", "preview"}:
        if interface == "sdk":
            sdk_headers = {"Authorization": f"Bearer {key.json()['full_key']}"}
            response = app.client.post(
                f"/v1/chat/tasks/{task_id}/messages",
                headers=sdk_headers,
                json={
                    "workforce_id": workforce_id,
                    "message": {"role": "user", "content": "e2e:ask"},
                },
            )
            assert response.status_code == 202, response.text
            waiting = app.wait_task(task_id, status="waiting_for_user")
            assert waiting["run_id"] != first["run_id"]
            response = app.client.post(
                f"/v1/chat/tasks/{task_id}/reply",
                headers=sdk_headers,
                json={
                    "workforce_id": workforce_id,
                    "message": {"role": "user", "content": "e2e:answer"},
                },
            )
            assert response.status_code == 202, response.text
            app.wait_task(task_id, run_id=waiting["run_id"])
            polled = app.client.get(f"/v1/chat/tasks/{task_id}", headers=sdk_headers)
            assert polled.status_code == 200, polled.text
            assert "Shared E2E answer" in polled.text
        else:
            from websockets.sync.client import connect

            token = (
                auth.json()["access_token"]
                if interface in {"widget", "share"}
                else app.token
            )
            ws_path = (
                f"/api/{interface}/chat/ws/{task_id}"
                if interface in {"widget", "share"}
                else f"/ws/chat/{task_id}"
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
                            "message": "e2e:ask",
                            "client_message_id": "workforce-question",
                        }
                    )
                )
                receive_event(ws, "message_accepted")
                waiting = app.wait_task(task_id, status="waiting_for_user")
                assert waiting["run_id"] != first["run_id"]
                ws.send(
                    json.dumps(
                        {
                            "type": "chat",
                            "message": "e2e:answer",
                            "client_message_id": "workforce-answer",
                        }
                    )
                )
                receive_event(ws, "message_accepted")
                receive_event(ws, "task_completed")
                app.wait_task(task_id, run_id=waiting["run_id"])
    from xagent.web.models.database import get_session_local
    from xagent.web.models.workforce import WorkforceRun

    with get_session_local()() as db:
        run = db.query(WorkforceRun).filter_by(task_id=task_id).one()
        assert run.status == "completed"
        assert run.is_preview == (interface == "preview")
        if interface.startswith("trigger"):
            from xagent.web.models.trigger import TriggerRun

            assert (
                db.query(TriggerRun).filter_by(task_id=task_id).one().status
                == "completed"
            )
    calls = [
        json.loads(line)
        for path in app.root.glob("model-*.jsonl")
        for line in path.read_text().splitlines()
    ]
    assert {call["role"] for call in calls} == {"worker"}
    assert any(
        "Answer the delegated question" in json.dumps(call["messages"])
        for call in calls
    )
    assert any(
        any(
            m.get("role") == "tool" and m.get("tool_call_id") == "e2e-delegation"
            for m in call["messages"]
        )
        for call in calls
    )


def test_websocket_pause_and_resume_with_two_workers(shared_app):
    from websockets.sync.client import connect

    app = shared_app
    app.start("worker")
    response = app.client.post(
        "/api/chat/task/create",
        headers=app.headers,
        json={
            "title": "Shared pause",
            "description": "e2e:gate",
            "execution_mode": "balanced",
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
        receive_event(ws, "execution_started")
        deadline = time.monotonic() + 30
        while not (app.root / "model-entered").exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert (app.root / "model-entered").exists(), app.diagnostics()
        ws.send(json.dumps({"type": "pause_task", "command_id": "pause-e2e"}))
        receive_event(ws, "task_command_accepted")
        app.wait_task(task_id, status="paused")
        # The original LLM request never completed. Resume must use its saved
        # runtime checkpoint, and either worker may win the next lease.
        (app.root / "model-release").touch()
        ws.send(json.dumps({"type": "resume_task", "command_id": "resume-e2e"}))
        receive_event(ws, "task_command_accepted")
        app.wait_task(task_id)
    calls = [
        json.loads(line)
        for path in app.root.glob("model-*.jsonl")
        for line in path.read_text().splitlines()
    ]
    assert len(calls) == 2
    assert {call["role"] for call in calls} == {"worker"}


def test_sdk_file_input_tool_output_and_download(shared_app):
    from xagent.web.models.database import get_session_local
    from xagent.web.models.uploaded_file import UploadedFile

    app = shared_app
    agent_id, headers = app.create_agent(tool_categories=["file"])
    uploaded = app.client.post(
        "/v1/chat/files",
        headers=headers,
        files=[("files", ("source.txt", b"unique shared input\n", "text/plain"))],
    )
    assert uploaded.status_code == 200, uploaded.text
    file_id = uploaded.json()["files"][0]["file_id"]
    response = app.client.post(
        "/v1/chat/tasks",
        headers=headers,
        json={
            "agent_id": agent_id,
            "message": {"role": "user", "content": "e2e:files", "files": [file_id]},
        },
    )
    assert response.status_code == 202, response.text
    task_id = response.json()["task_id"]
    completed = app.wait_task(task_id)
    with get_session_local()() as db:
        files = db.query(UploadedFile).filter_by(task_id=task_id).all()
        output = next(f for f in files if f.filename == "derived.txt")
        output_id = output.file_id
    assert output_id in completed["output"]
    downloaded = app.client.get(f"/api/files/download/{output_id}", headers=app.headers)
    assert downloaded.status_code == 200, downloaded.text
    assert downloaded.content == b"UNIQUE SHARED INPUT\n"
    assert app.client.get(f"/api/files/download/{output_id}").status_code == 403


@pytest.mark.parametrize("fault", ["web_exit", "worker_kill", "worker_sigterm"])
def test_host_loss_settles_without_duplicate_execution(shared_app, fault):
    app = shared_app
    agent_id, headers = app.create_agent()
    response = app.client.post(
        "/v1/chat/tasks",
        headers=headers,
        json={
            "agent_id": agent_id,
            "message": {"role": "user", "content": "e2e:gate"},
        },
    )
    assert response.status_code == 202, response.text
    accepted = response.json()
    task_id = accepted["task_id"]
    deadline = time.monotonic() + 30
    while not (app.root / "model-entered").exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert (app.root / "model-entered").exists(), app.diagnostics()
    index = 0 if fault == "web_exit" else 1
    process, pipe = app.processes[index], app.pipes[index]
    if fault == "worker_kill":
        process.kill()
        process.join(10)
        assert process.exitcode == -9
        app.processes.remove(process)
        app.start("worker")
    else:
        pipe.send("stop")
        process.join(30)
        assert process.exitcode == 0, app.diagnostics()
    if fault == "web_exit":
        (app.root / "model-release").touch()
        settled = app.wait_task(task_id)
        app.client.close()
        app.start("web")
        response = app.client.get(f"/v1/chat/tasks/{task_id}/events", headers=headers)
        assert response.status_code == 200, response.text
        assert "Shared E2E answer" in response.text
    elif fault == "worker_kill":
        from websockets.sync.client import connect

        app.wait_task(task_id, status="paused")
        calls = [
            line
            for path in app.root.glob("model-*.jsonl")
            for line in path.read_text().splitlines()
        ]
        assert len(calls) == 1
        (app.root / "model-release").touch()
        url = (
            str(app.client.base_url).replace("http://", "ws://").rstrip("/")
            + f"/ws/chat/{task_id}?token={app.token}"
        )
        with connect(url) as ws:
            receive_event(ws, "historical_data_complete")
            ws.send(json.dumps({"type": "resume_task", "command_id": "after-crash"}))
            receive_event(ws, "task_command_accepted")
            settled = app.wait_task(task_id)
    else:
        settled = app.wait_task(task_id, status="failed")
    assert settled["run_id"] == accepted["run_id"]
    calls = [
        json.loads(line)
        for path in app.root.glob("model-*.jsonl")
        for line in path.read_text().splitlines()
    ]
    assert len(calls) == (2 if fault == "worker_kill" else 1)
    assert {call["role"] for call in calls} == {"worker"}


def test_scheduled_dispatcher_runs_task_and_settles_trigger(shared_app):
    from datetime import datetime, timedelta, timezone

    from xagent.web.models.database import get_session_local
    from xagent.web.models.trigger import AgentTrigger, TriggerRun

    app = shared_app
    agent_id, _ = app.create_agent()
    response = app.client.post(
        f"/api/agents/{agent_id}/triggers",
        headers=app.headers,
        json={
            "type": "scheduled",
            "name": "Due trigger",
            "config": {"interval_seconds": 3600},
        },
    )
    assert response.status_code == 200, response.text
    trigger_id = response.json()["id"]
    with get_session_local()() as db:
        db.get(AgentTrigger, trigger_id).next_run_at = datetime.now(
            timezone.utc
        ) - timedelta(seconds=5)
        db.commit()
    deadline = time.monotonic() + 30
    task_id = None
    while time.monotonic() < deadline:
        with get_session_local()() as db:
            run = db.query(TriggerRun).filter_by(trigger_id=trigger_id).first()
            if run is not None:
                task_id = run.task_id
        if task_id:
            break
        time.sleep(0.03)
    assert task_id, app.diagnostics()
    app.wait_task(task_id)
    with get_session_local()() as db:
        run = db.query(TriggerRun).filter_by(trigger_id=trigger_id).one()
        assert run.status == "completed"


def test_a2a_stream_disconnect_and_resubscribe(shared_app):
    app = shared_app
    agent_id, headers = app.create_agent()
    published = app.client.post(f"/api/agents/{agent_id}/publish", headers=app.headers)
    assert published.status_code == 200, published.text
    headers["A2A-Version"] = "1.0"
    with app.client.stream(
        "POST",
        f"/api/a2a/agents/{agent_id}/message:stream",
        headers=headers,
        json={
            "message": {
                "messageId": "stream-start",
                "role": "ROLE_USER",
                "parts": [{"text": "e2e:gate"}],
            },
        },
    ) as stream:
        assert stream.status_code == 200, (
            stream.read().decode() if stream.status_code != 200 else ""
        )
        first = next(
            json.loads(line[5:])
            for line in stream.iter_lines()
            if line.startswith("data:")
        )
        task_id = int(first["task"]["id"])
    deadline = time.monotonic() + 30
    while not (app.root / "model-entered").exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert (app.root / "model-entered").exists(), app.diagnostics()
    with app.client.stream(
        "GET", f"/api/a2a/agents/{agent_id}/tasks/{task_id}:subscribe", headers=headers
    ) as stream:
        assert stream.status_code == 200, (
            stream.read().decode() if stream.status_code != 200 else ""
        )
        lines = stream.iter_lines()
        initial = next(
            json.loads(line[5:]) for line in lines if line.startswith("data:")
        )
        assert initial["task"]["id"] == str(task_id)
        (app.root / "model-release").touch()
        events = [json.loads(line[5:]) for line in lines if line.startswith("data:")]
    assert any(
        e.get("statusUpdate", {}).get("status", {}).get("state")
        == "TASK_STATE_COMPLETED"
        for e in events
    )
    artifacts = [e["artifactUpdate"] for e in events if "artifactUpdate" in e]
    assert artifacts and artifacts[-1]["lastChunk"]
    assert "Shared E2E answer" in json.dumps(artifacts)
    app.wait_task(task_id)
    fetched = app.client.get(
        f"/api/a2a/agents/{agent_id}/tasks/{task_id}", headers=headers
    )
    assert fetched.status_code == 200, fetched.text
    assert fetched.json()["status"]["state"] == "TASK_STATE_COMPLETED"


def test_websocket_running_message_is_delivered_once(shared_app):
    from websockets.sync.client import connect

    from xagent.web.models.chat_message import TaskChatMessage
    from xagent.web.models.database import get_session_local
    from xagent.web.models.task import Task
    from xagent.web.models.task_command import TaskExecutionCommand

    app = shared_app
    response = app.client.post(
        "/api/chat/task/create",
        headers=app.headers,
        json={
            "title": "Chat guidance",
            "description": "Question",
            "execution_mode": "balanced",
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
        ws.send(
            json.dumps(
                {"type": "chat", "message": "e2e:gate", "client_message_id": "opening"}
            )
        )
        receive_event(ws, "message_accepted")
        deadline = time.monotonic() + 30
        while not (app.root / "model-entered").exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert (app.root / "model-entered").exists(), app.diagnostics()
        with get_session_local()() as db:
            original_run_id = db.get(Task, task_id).run_id
        guidance = {
            "type": "chat",
            "message": "e2e:steer",
            "client_message_id": "guidance",
        }
        ws.send(json.dumps(guidance))
        receive_event(ws, "message_accepted")
        ws.send(json.dumps(guidance))
        receive_event(ws, "message_accepted")
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            with get_session_local()() as db:
                command = (
                    db.query(TaskExecutionCommand)
                    .filter_by(task_id=task_id, command_id="guidance")
                    .one()
                )
                command_status = command.status
            if command_status == "completed":
                break
            assert command_status != "failed", app.diagnostics()
            time.sleep(0.02)
        assert command_status == "completed", app.diagnostics()
        (app.root / "model-release").touch()
        receive_event(ws, "task_completed")
    app.wait_task(task_id, run_id=original_run_id)
    with get_session_local()() as db:
        assert (
            db.query(TaskChatMessage)
            .filter_by(task_id=task_id, role="user", content="e2e:steer")
            .count()
            == 1
        )
    calls = [
        json.loads(line)
        for path in app.root.glob("model-*.jsonl")
        for line in path.read_text().splitlines()
    ]
    assert any("e2e:steer" in json.dumps(call["messages"]) for call in calls)
    assert {call["role"] for call in calls} == {"worker"}
