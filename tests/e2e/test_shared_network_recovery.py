"""Disconnect only this application's real Redis sockets, then reconnect."""

import json
import time

import pytest
from websockets.sync.client import connect

from tests.e2e.shared_execution_harness import receive_event
from tests.e2e.tcp_link import ControllableTcpLink

pytestmark = pytest.mark.e2e


def test_redis_gap_preserves_result_and_rejects_unaccepted_start(shared_app):
    from xagent.web.models.database import get_session_local
    from xagent.web.models.task import Task

    app = shared_app
    # Restart these isolated hosts behind a controllable TCP link. The Redis
    # service itself remains available to other tests and applications.
    for pipe in app.pipes:
        pipe.send("stop")
    for process in app.processes:
        process.join(30)
        assert process.exitcode == 0, app.diagnostics()
    app.client.close()
    app.processes.clear()
    app.pipes.clear()
    proxy = ControllableTcpLink(app.environment["XAGENT_REDIS_URL"], default_port=6379)
    try:
        app.environment["XAGENT_REDIS_URL"] = proxy.url
        app.start_all("web", "worker")
        agent_id, headers = app.create_agent()
        request = {
            "agent_id": agent_id,
            "message": {"role": "user", "content": "e2e:gate"},
        }
        created = app.client.post("/v1/chat/tasks", headers=headers, json=request)
        assert created.status_code == 202, created.text
        task_id = created.json()["task_id"]
        url = (
            str(app.client.base_url).replace("http://", "ws://").rstrip("/")
            + f"/ws/chat/{task_id}?token={app.token}"
        )
        with connect(url) as ws:
            receive_event(ws, "historical_data_complete")
            deadline = time.monotonic() + 30
            while (
                not (app.root / "model-entered").exists()
                and time.monotonic() < deadline
            ):
                time.sleep(0.02)
            assert (app.root / "model-entered").exists(), app.diagnostics()
            proxy.cut()
            receive_event(ws, "stream_unavailable")
            rejected = app.client.post("/v1/chat/tasks", headers=headers, json=request)
            assert rejected.status_code >= 500, rejected.text
            with get_session_local()() as db:
                assert db.query(Task).count() == 1
            (app.root / "model-release").touch()
            completed = app.wait_task(task_id)
            assert "Shared E2E answer" in completed["output"]
            proxy.restore()
            receive_event(ws, "stream_resync_required")
        with connect(url) as ws:
            snapshot, _ = receive_event(ws, "historical_data_complete")
            assert snapshot["task_id"] == task_id
        events = app.client.get(f"/v1/chat/tasks/{task_id}/events", headers=headers)
        assert events.status_code == 200, events.text
        assert "Shared E2E answer" in events.text
        calls = [
            json.loads(line)
            for path in app.root.glob("model-*.jsonl")
            for line in path.read_text().splitlines()
        ]
        assert len(calls) == 1
    finally:
        proxy.restore()
        try:
            app.close()
        finally:
            proxy.close()
