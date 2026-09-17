"""Disconnect only this application's real Redis sockets, then reconnect."""

import json
import select
import socket
import socketserver
import threading
import time
from urllib.parse import urlsplit, urlunsplit

import pytest
from websockets.sync.client import connect

from tests.e2e.shared_execution_harness import receive_event

pytestmark = pytest.mark.e2e


class _RedisConnection(socketserver.BaseRequestHandler):
    def handle(self):
        proxy = self.server
        if not proxy.enabled.is_set():
            return
        upstream = socket.create_connection(proxy.upstream, timeout=3)
        upstream.settimeout(None)
        peers = [self.request, upstream]
        with proxy.lock:
            proxy.peers.update(peers)
        try:
            while proxy.enabled.is_set():
                ready, _, _ = select.select(peers, [], [], 0.1)
                for source in ready:
                    data = source.recv(65536)
                    if not data:
                        return
                    destination = upstream if source is self.request else self.request
                    destination.sendall(data)
        except (OSError, ValueError):
            pass  # A cut deliberately closes active sockets from another thread.
        finally:
            with proxy.lock:
                proxy.peers.difference_update(peers)
            upstream.close()


class _RedisProxy(socketserver.ThreadingTCPServer):
    daemon_threads = True

    def __init__(self, url):
        parsed = urlsplit(url)
        self.upstream = (parsed.hostname, parsed.port or 6379)
        self.enabled = threading.Event()
        self.enabled.set()
        self.lock = threading.Lock()
        self.peers = set()
        super().__init__(("127.0.0.1", 0), _RedisConnection)
        credentials = parsed.netloc.rpartition("@")[0]
        authority = f"127.0.0.1:{self.server_address[1]}"
        if credentials:
            authority = credentials + "@" + authority
        self.url = urlunsplit(parsed._replace(netloc=authority))
        self.thread = threading.Thread(target=self.serve_forever, daemon=True)
        self.thread.start()

    def cut(self):
        self.enabled.clear()
        with self.lock:
            for peer in self.peers:
                try:
                    peer.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass

    def close(self):
        self.cut()
        self.shutdown()
        self.thread.join(5)
        self.server_close()


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
    proxy = _RedisProxy(app.environment["XAGENT_REDIS_URL"])
    try:
        app.environment["XAGENT_REDIS_URL"] = proxy.url
        app.start("web")
        app.start("worker")
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
            proxy.enabled.set()
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
        proxy.enabled.set()
        try:
            app.close()
        finally:
            proxy.close()
