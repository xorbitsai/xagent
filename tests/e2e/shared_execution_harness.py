"""Real HTTP web host and standalone workers with only the LLM boundary replaced."""

from __future__ import annotations

import ast
import asyncio
import json
import multiprocessing
import os
import signal
import socket
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import pytest

from tests.e2e.app_harness import build_access_token


def _install_model_boundary(root: Path, role: str) -> None:
    from tests.e2e.scripted_llm import ScriptedLLM
    from xagent.web.api import chat
    from xagent.web.services import agent_service_manager

    class RecordingLLM(ScriptedLLM):
        def __init__(self):
            super().__init__([])

        async def chat(self, messages, **kwargs):
            with (root / f"model-{os.getpid()}.jsonl").open("a") as log:
                log.write(
                    json.dumps(
                        {
                            "role": role,
                            "messages": messages,
                            "tools": [
                                t["function"]["name"]
                                for t in kwargs.get("tools", []) or []
                            ],
                        }
                    )
                    + "\n"
                )
            text = json.dumps(messages)
            tool_names = {t["function"]["name"] for t in kwargs.get("tools", []) or []}
            if "generate_execution_plan" in tool_names:
                return {
                    "tool_calls": [
                        {
                            "id": "e2e-plan",
                            "function": {
                                "name": "generate_execution_plan",
                                "arguments": json.dumps(
                                    {
                                        "response_language": "English",
                                        "steps": [
                                            {
                                                "id": "answer",
                                                "task": "Answer the question in English",
                                                "dependencies": [],
                                                "description": "Produce the answer",
                                                "termination_condition": "The answer is ready",
                                                "completion_evidence": "A final answer",
                                                "tool_names": [],
                                            }
                                        ],
                                    }
                                ),
                            },
                        }
                    ]
                }
            if "assess_dag_completion" in tool_names:
                return {
                    "tool_calls": [
                        {
                            "id": "e2e-assess",
                            "function": {
                                "name": "assess_dag_completion",
                                "arguments": json.dumps(
                                    {
                                        "status": "completed",
                                        "reason": "The answer is ready",
                                        "answer": "Shared E2E answer",
                                        "missing_work": "",
                                        "replan_instruction": "",
                                    }
                                ),
                            },
                        }
                    ]
                }
            if any(
                t["function"]["name"] == "select_execution_pattern"
                for t in kwargs.get("tools", []) or []
            ):
                return {
                    "tool_calls": [
                        {
                            "id": "e2e-route",
                            "function": {
                                "name": "select_execution_pattern",
                                "arguments": json.dumps(
                                    {
                                        "action": "final_answer"
                                        if "e2e:route-final" in text
                                        else "plan_execute"
                                        if "e2e:route-dag" in text
                                        else "react",
                                        "reason": "Exercise the agent runtime",
                                        "answer": "Shared E2E answer"
                                        if "e2e:route-final" in text
                                        else "",
                                        "requires_current_or_external_facts": False,
                                        "existing_context_sufficient": True,
                                        "evidence_basis": "Test question",
                                        "missing_verification": "",
                                    }
                                ),
                            },
                        }
                    ]
                }
            delegated = [
                t["function"]["name"]
                for t in kwargs.get("tools", []) or []
                if t["function"]["name"].startswith("worker_")
            ]
            if (
                "e2e:delegate" in text
                and delegated
                and not any(m.get("role") == "tool" for m in messages)
            ):
                return {
                    "tool_calls": [
                        {
                            "id": "e2e-delegation",
                            "function": {
                                "name": delegated[0],
                                "arguments": json.dumps(
                                    {"task": "Answer the delegated question"}
                                ),
                            },
                        }
                    ]
                }
            results = {
                m.get("tool_call_id"): m.get("content")
                for m in messages
                if m.get("role") == "tool"
            }
            if "e2e:files" in text and "e2e:ask" not in text:
                if "e2e-read" not in results:
                    return {
                        "tool_calls": [
                            {
                                "id": "e2e-read",
                                "function": {
                                    "name": "read_file",
                                    "arguments": json.dumps(
                                        {"file_path": "source.txt"}
                                    ),
                                },
                            }
                        ]
                    }
                if "e2e-write" not in results:
                    return {
                        "tool_calls": [
                            {
                                "id": "e2e-write",
                                "function": {
                                    "name": "write_file",
                                    "arguments": json.dumps(
                                        {
                                            "file_path": "derived.txt",
                                            "content": results["e2e-read"]
                                            .removeprefix("Tool read_file returned: ")
                                            .upper(),
                                        }
                                    ),
                                },
                            }
                        ]
                    }
                written = ast.literal_eval(
                    results["e2e-write"].removeprefix("Tool write_file returned: ")
                )
                return "Shared E2E answer: " + written["markdown_link"]
            if "e2e:gate" in text:
                (root / "model-entered").touch()
                while not (root / "model-release").exists():
                    await asyncio.sleep(0.02)
            if "e2e:ask" in text and "e2e:answer" not in text:
                return {
                    "tool_calls": [
                        {
                            "id": "e2e-question",
                            "function": {
                                "name": "ask_user_question",
                                "arguments": json.dumps(
                                    {"message": "Which choice?", "interactions": []}
                                ),
                            },
                        }
                    ]
                }
            if any(
                t["function"]["name"] == "final_answer"
                for t in kwargs.get("tools", []) or []
            ):
                return {
                    "tool_calls": [
                        {
                            "id": "e2e-final",
                            "function": {
                                "name": "final_answer",
                                "arguments": json.dumps(
                                    {
                                        "response_language": "English",
                                        "answer": "Shared E2E answer",
                                    }
                                ),
                            },
                        }
                    ]
                }
            return "Shared E2E answer"

    agent_service_manager.create_default_llm = RecordingLLM
    chat.resolve_llms_from_names = lambda *a, **kw: (RecordingLLM(), None, None, None)


def _host(pipe, environment: dict[str, str], role: str, root: str) -> None:
    os.environ.update(environment)
    os.environ.pop("PYTEST_CURRENT_TEST", None)
    os.environ["XAGENT_TASK_EXECUTION_ROLE"] = role
    root_path = Path(root)
    # Each host's diagnostics remain available when a bounded wait fails.
    log = (root_path / f"{role}-{os.getpid()}.log").open("w")
    os.dup2(log.fileno(), 1)
    os.dup2(log.fileno(), 2)
    _install_model_boundary(root_path, role)
    if (root_path / "gmail-public-key.pem").exists():
        from tests.e2e.test_shared_gmail import install_gmail_network_boundary

        install_gmail_network_boundary(root_path)
    from xagent.web import sandbox_manager

    sandbox_manager.get_sandbox_manager = lambda: None
    if role in {"web", "combined"}:
        import uvicorn

        from xagent.web.app import app

        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
            listener.listen()
            server = uvicorn.Server(
                uvicorn.Config(app, log_level="warning", timeout_graceful_shutdown=20)
            )

            async def serve():
                serving = asyncio.create_task(server.serve(sockets=[listener]))
                while not server.started:
                    if serving.done():
                        await serving
                        raise RuntimeError("Web startup failed")
                    await asyncio.sleep(0.02)
                pipe.send({"port": port})
                await asyncio.to_thread(pipe.recv)
                server.should_exit = True
                await serving

            asyncio.run(serve())
    else:
        from xagent.web import worker
        from xagent.web.services.task_event_bridge import get_task_event_bridge

        async def consume():
            running = asyncio.create_task(worker._main())
            while True:
                if running.done():
                    await running
                    raise RuntimeError("Worker startup failed")
                try:
                    if get_task_event_bridge().ready.is_set():
                        break
                except ConnectionError:
                    pass
                await asyncio.sleep(0.02)
            pipe.send({"ready": True})
            await asyncio.to_thread(pipe.recv)
            os.kill(os.getpid(), signal.SIGTERM)
            await running

        asyncio.run(consume())


@dataclass
class SharedExecutionApp:
    root: Path
    environment: dict[str, str]
    token: str
    user_id: int
    processes: list[Any] = field(default_factory=list)
    pipes: list[Any] = field(default_factory=list)
    client: httpx.Client | None = None

    @property
    def headers(self):
        return {"Authorization": f"Bearer {self.token}"}

    def start(self, role: str):
        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe()
        process = context.Process(
            target=_host, args=(child, self.environment, role, str(self.root))
        )
        process.start()
        child.close()
        self.processes.append(process)
        self.pipes.append(parent)
        assert parent.poll(60), self.diagnostics()
        try:
            ready = parent.recv()
        except EOFError:
            pytest.fail(self.diagnostics())
        if role in {"web", "combined"}:
            self.client = httpx.Client(
                base_url=f"http://127.0.0.1:{ready['port']}", timeout=30
            )
        return process, parent

    def diagnostics(self):
        return "\n".join(
            f"{p.name}:\n{p.read_text()[-12000:]}" for p in self.root.glob("*.log")
        )

    def wait_task(
        self,
        task_id: int,
        *,
        status: str = "completed",
        run_id: str | None = None,
        timeout: float = 30,
    ):
        from xagent.web.models.database import get_session_local
        from xagent.web.models.task import Task

        deadline = time.monotonic() + timeout
        state = None
        while time.monotonic() < deadline:
            with get_session_local()() as db:
                task = db.get(Task, task_id)
                state = {
                    "status": task.status.value,
                    "run_id": task.run_id,
                    "runner_id": task.runner_id,
                    "output": task.output,
                    "error": task.error_message,
                }
            if (
                state["status"] == status
                and state["runner_id"] is None
                and (run_id is None or state["run_id"] == run_id)
            ):
                return state
            if state["status"] == "failed" and status != "failed":
                pytest.fail(f"Task failed: {state}\n{self.diagnostics()}")
            time.sleep(0.03)
        pytest.fail(f"Task did not reach {status}: {state}\n{self.diagnostics()}")

    def create_agent(self, name="Shared E2E agent", *, tool_categories=None):
        assert self.client is not None
        response = self.client.post(
            "/api/agents",
            headers=self.headers,
            json={
                "name": name,
                "instructions": "Answer the request.",
                "execution_mode": "balanced",
                "tool_categories": tool_categories or [],
            },
        )
        assert response.status_code == 200, response.text
        agent_id = response.json()["id"]
        response = self.client.post(
            f"/api/agents/{agent_id}/api-key", headers=self.headers
        )
        assert response.status_code == 200, response.text
        return agent_id, {"Authorization": f"Bearer {response.json()['full_key']}"}

    def close(self):
        if self.client is not None:
            self.client.close()
        for pipe in reversed(self.pipes):
            try:
                pipe.send("stop")
            except (BrokenPipeError, EOFError, OSError):
                pass
        failures = []
        for process in reversed(self.processes):
            process.join(30)
            if process.is_alive():
                failures.append(f"Host {process.pid} did not shut down")
                process.terminate()
                process.join(10)
            if process.exitcode != 0:
                failures.append(f"Host {process.pid} exited {process.exitcode}")
        for pipe in self.pipes:
            pipe.close()
        assert not failures, "\n".join(failures) + "\n" + self.diagnostics()


@pytest.fixture
def shared_app(tmp_path, monkeypatch):
    from xagent.web.models.database import get_engine, get_session_local, init_db
    from xagent.web.models.user import User

    root = tmp_path / "hosts"
    root.mkdir()
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'shared.db'}")
    monkeypatch.setenv("XAGENT_UPLOADS_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("LANGFUSE_TRACING_ENABLED", "false")
    monkeypatch.setenv("XAGENT_TASK_LEASE_TTL_SECONDS", "10")
    monkeypatch.setenv("XAGENT_TASK_LEASE_HEARTBEAT_SECONDS", "2")
    monkeypatch.setenv("XAGENT_TASK_LEASE_RECOVERY_INTERVAL_SECONDS", "1")
    monkeypatch.setenv("XAGENT_TRIGGER_DISPATCHER_INTERVAL_SECONDS", "1")
    monkeypatch.setenv("XAGENT_TRIGGER_DISPATCHER_STARTUP_JITTER_SECONDS", "0")
    monkeypatch.setenv("XAGENT_FILE_STORAGE_STARTUP_SYNC_ENABLED", "false")
    init_db()
    with get_session_local()() as db:
        user = User(username="e2e-owner", password_hash="unused", is_admin=True)
        db.add(user)
        db.commit()
        user_id = user.id
    app = SharedExecutionApp(
        root,
        dict(os.environ),
        build_access_token(username="e2e-owner", user_id=user_id),
        user_id,
    )
    try:
        app.start("web")
        app.start("worker")
        yield app
    finally:
        try:
            app.close()
        finally:
            get_engine().dispose()


def receive_event(ws, event_type):
    events = []
    deadline = time.monotonic() + 30
    for _ in range(200):
        remaining = deadline - time.monotonic()
        assert remaining > 0, events
        try:
            event = json.loads(ws.recv(timeout=remaining))
        except TimeoutError:
            pytest.fail(f"Missing {event_type}: {events}")
        events.append(event)
        if event["type"] == event_type or event.get("event_type") == event_type:
            return event, events
        assert event["type"] != "error", event
    pytest.fail(f"Missing {event_type}: {events}")
