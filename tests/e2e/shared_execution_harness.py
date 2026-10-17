"""Real HTTP web host and standalone workers with only the LLM boundary replaced."""

from __future__ import annotations

import ast
import asyncio
import fcntl
import json
import multiprocessing
import os
import signal
import socket
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import pytest

from tests.e2e.app_harness import build_access_token

# A provider fault the hosts' model boundary injects: ``{"kind": "unavailable"
# | "invalid_output", "remaining": int | null}``. ``remaining`` counts provider
# attempts (each tier-1 retry is one); ``null`` keeps the fault until the file
# is removed. Counting attempts rather than time makes the moment the provider
# "recovers" deterministic.
MODEL_FAULT_FILE = "model-fault.json"


def _take_model_fault(root: Path, tool_names: set[str]) -> str | None:
    """Consume one attempt of the current fault, if it applies to this call."""
    try:
        handle = (root / MODEL_FAULT_FILE).open("r+")
    except FileNotFoundError:
        return None
    with handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        spec = json.load(handle)
        kind = spec["kind"]
        # Unusable output only matters where the model must answer.
        if kind == "invalid_output" and "final_answer" not in tool_names:
            return None
        remaining = spec.get("remaining")
        if remaining is not None:
            if remaining <= 0:
                return None
            spec["remaining"] = remaining - 1
            handle.seek(0)
            handle.truncate()
            json.dump(spec, handle)
        return kind


def _install_model_boundary(root: Path, role: str) -> None:
    import openai

    from tests.e2e.scripted_llm import ScriptedLLM
    from xagent.web.api import chat
    from xagent.web.services import agent_service_manager

    class RecordingLLM(ScriptedLLM):
        def __init__(self):
            super().__init__([])

        async def chat(self, messages, **kwargs):
            tool_names = {t["function"]["name"] for t in kwargs.get("tools", []) or []}
            fault = _take_model_fault(root, tool_names)
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
                            "fault": fault,
                        }
                    )
                    + "\n"
                )
            if fault == "unavailable":
                # A real provider 5xx; the text must not read as a capacity,
                # quota or credential refusal, which are classified apart.
                request = httpx.Request(
                    "POST", "http://scripted-provider.invalid/v1/chat/completions"
                )
                raise openai.InternalServerError(
                    "Service Unavailable",
                    response=httpx.Response(503, request=request),
                    body=None,
                )
            if fault == "invalid_output":
                return {
                    "tool_calls": [
                        {
                            "id": "e2e-empty-final",
                            "function": {
                                "name": "final_answer",
                                "arguments": json.dumps(
                                    {"response_language": "English", "answer": ""}
                                ),
                            },
                        }
                    ]
                }
            text = json.dumps(messages)
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

    def boundary_llm():
        # Test-only knob: put the production tier-1 retry layer in front of
        # the scripted provider, so a fault is retried as a real one would be.
        llm = RecordingLLM()
        attempts = os.environ.get("E2E_MODEL_RETRY_ATTEMPTS")
        if not attempts:
            return llm
        from xagent.core.model.chat.basic.adapter import attach_chat_retry_wrapper

        return attach_chat_retry_wrapper(llm, max_retries=int(attempts))

    agent_service_manager.create_default_llm = boundary_llm
    chat.resolve_llms_from_names = lambda *a, **kw: (boundary_llm(), None, None, None)


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
    if (root_path / "event-read-acceptance").exists():
        import sqlalchemy as sa

        from tests.shared.execution_event_read_guard import reject_legacy_content
        from xagent.web.models.task import Task

        # Select V2 before INSERT only in these test processes; production
        # creation defaults and existing task versions are unchanged.
        def select_event_storage(_mapper, _connection, task):
            task.conversation_storage_version = 2

        sa.event.listen(Task, "before_insert", select_event_storage)
        sa.event.listen(
            sa.engine.Engine, "before_cursor_execute", reject_legacy_content
        )
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
        return self.start_all(role)[0]

    def start_all(self, *roles: str):
        """Spawn hosts together, then wait; each host pays a cold import."""
        context = multiprocessing.get_context("spawn")
        launched = []
        for role in roles:
            parent, child = context.Pipe()
            process = context.Process(
                target=_host, args=(child, self.environment, role, str(self.root))
            )
            process.start()
            child.close()
            self.processes.append(process)
            self.pipes.append(parent)
            launched.append((role, process, parent))
        for role, _, parent in launched:
            assert parent.poll(60), self.diagnostics()
            try:
                ready = parent.recv()
            except EOFError:
                pytest.fail(self.diagnostics())
            if role in {"web", "combined"}:
                if self.client is not None:
                    self.client.close()
                self.client = httpx.Client(
                    base_url=f"http://127.0.0.1:{ready['port']}", timeout=30
                )
        return [(process, parent) for _, process, parent in launched]

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


@contextmanager
def open_shared_app(
    tmp_path: Path,
    monkeypatch,
    *,
    database_url: str,
    host_database_url: str | None = None,
    event_read_acceptance: bool = False,
) -> Iterator[SharedExecutionApp]:
    """A web host and a worker on ``database_url``, closed on exit.

    The hosts reach the database through ``host_database_url`` when given;
    this process always uses ``database_url`` directly.
    """
    from xagent.web.models.database import get_engine, get_session_local, init_db
    from xagent.web.models.user import User

    root = tmp_path / "hosts"
    root.mkdir()
    if event_read_acceptance:
        (root / "event-read-acceptance").touch()
    monkeypatch.setenv("DATABASE_URL", database_url)
    monkeypatch.setenv("XAGENT_UPLOADS_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("LANGFUSE_TRACING_ENABLED", "false")
    monkeypatch.setenv("XAGENT_TASK_LEASE_TTL_SECONDS", "10")
    monkeypatch.setenv("XAGENT_TASK_LEASE_HEARTBEAT_SECONDS", "2")
    monkeypatch.setenv("XAGENT_TASK_LEASE_RECOVERY_INTERVAL_SECONDS", "1")
    monkeypatch.setenv("XAGENT_TRIGGER_DISPATCHER_INTERVAL_SECONDS", "1")
    monkeypatch.setenv("XAGENT_TRIGGER_DISPATCHER_STARTUP_JITTER_SECONDS", "0")
    monkeypatch.setenv("XAGENT_FILE_STORAGE_STARTUP_SYNC_ENABLED", "false")
    init_db()
    try:
        with get_session_local()() as db:
            user = User(username="e2e-owner", password_hash="unused", is_admin=True)
            db.add(user)
            db.commit()
            user_id = user.id
        environment = dict(os.environ)
        if host_database_url is not None:
            environment["DATABASE_URL"] = host_database_url
        app = SharedExecutionApp(
            root,
            environment,
            build_access_token(username="e2e-owner", user_id=user_id),
            user_id,
        )
        try:
            app.start_all("web", "worker")
            yield app
        finally:
            app.close()
    finally:
        get_engine().dispose()


@pytest.fixture
def shared_app(tmp_path, monkeypatch, request):
    with open_shared_app(
        tmp_path,
        monkeypatch,
        database_url=f"sqlite:///{tmp_path / 'shared.db'}",
        event_read_acceptance=getattr(request, "param", 1) == 2,
    ) as app:
        yield app


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


# ---------------------------------------------------------------- crash helpers

# Lease recovery takes the lease TTL (10s) plus a few polls; leave room for a
# cold worker start.
SETTLE_TIMEOUT = 60


def restart_hosts(app, *roles: str, **environment: str) -> None:
    """Replace every host, with ``environment`` applied to the new ones."""

    app.close()
    app.processes.clear()
    app.pipes.clear()
    app.client = None
    app.environment.update(environment)
    app.start_all(*roles)


def wait_for(predicate, app, what: str, timeout: float = SETTLE_TIMEOUT) -> Any:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.05)
    pytest.fail(f"Timed out waiting for {what}\n{app.diagnostics()}")


def kill_host(app, process) -> None:
    """SIGKILL one host and forget it."""
    process.kill()
    process.join(10)
    assert process.exitcode == -signal.SIGKILL
    index = app.processes.index(process)
    app.processes.pop(index)
    app.pipes.pop(index).close()


def model_calls(app) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for path in app.root.glob("model-*.jsonl")
        for line in path.read_text().splitlines()
    ]


def task_state(task_id: int) -> dict[str, Any]:
    from xagent.web.models.database import get_session_local
    from xagent.web.models.task import Task

    with get_session_local()() as db:
        task = db.get(Task, task_id)
        return {
            "status": task.status.value,
            "control_state": task.control_state,
            "run_id": task.run_id,
            "runner_id": task.runner_id,
            "state_version": task.state_version,
        }


def recovery_state(task_id: int) -> tuple[dict[str, Any] | None, list[str]]:
    """The task's recovery row and its event names, in order."""
    import sqlalchemy as sa

    from xagent.web.models.database import get_session_local
    from xagent.web.models.task_auto_recovery import (
        TaskAutoRecovery,
        TaskRecoveryEvent,
    )

    with get_session_local()() as db:
        row = db.get(TaskAutoRecovery, task_id)
        events = [
            event.event
            for event in db.scalars(
                sa.select(TaskRecoveryEvent)
                .where(TaskRecoveryEvent.task_id == task_id)
                .order_by(TaskRecoveryEvent.id)
            )
        ]
        if row is None:
            return None, events
        return {
            "state": row.state,
            "state_detail": row.state_detail,
            "reason": row.reason,
            "run_id": row.run_id,
            "total_resumes": row.total_resumes,
            "no_progress_resumes": row.no_progress_resumes,
        }, events


def auto_resume_commands(task_id: int) -> list[str]:
    from xagent.web.models.database import get_session_local
    from xagent.web.models.task_command import TaskExecutionCommand

    with get_session_local()() as db:
        return [
            command.command_id
            for command in db.query(TaskExecutionCommand)
            .filter(TaskExecutionCommand.task_id == task_id)
            .order_by(TaskExecutionCommand.id)
            if command.command_id.startswith("auto-resume:")
        ]


def ws_url(app, task_id: int) -> str:
    return (
        str(app.client.base_url).replace("http://", "ws://").rstrip("/")
        + f"/ws/chat/{task_id}?token={app.token}"
    )


def create_task(app, title: str, description: str) -> int:
    """An owner's internal balanced task, not yet executed."""
    response = app.client.post(
        "/api/chat/task/create",
        headers=app.headers,
        json={
            "title": title,
            "description": description,
            "execution_mode": "balanced",
        },
    )
    assert response.status_code == 200, response.text
    return response.json()["task_id"]


def create_sdk_task(app, content: str) -> tuple[int, str, dict[str, str]]:
    """An agent's SDK task: its id, run id and the caller's headers."""
    agent_id, headers = app.create_agent()
    response = app.client.post(
        "/v1/chat/tasks",
        headers=headers,
        json={"agent_id": agent_id, "message": {"role": "user", "content": content}},
    )
    assert response.status_code == 202, response.text
    accepted = response.json()
    return accepted["task_id"], accepted["run_id"], headers


def start_task(app, description: str = "e2e:gate") -> int:
    """An owner's internal balanced task, executed over the WebSocket.

    A gated description (``e2e:gate``) makes the first model call wait for
    ``model-release``; this then returns once that call has started.
    """
    from websockets.sync.client import connect

    task_id = create_task(app, "Auto resume", description)
    with connect(ws_url(app, task_id)) as ws:
        receive_event(ws, "historical_data_complete")
        ws.send(json.dumps({"type": "execute_task"}))
        receive_event(ws, "execution_started")
    if "e2e:gate" in description:
        wait_for(lambda: (app.root / "model-entered").exists(), app, "model call")
    return task_id


# Long enough for several automatic resumes with their backoff and jitter.
COLLECT_EVENTS_TIMEOUT = 45


def collect_events(app, ws, until) -> list[dict[str, Any]]:
    """Events up to and including the first one ``until`` accepts.

    Unlike :func:`receive_event` this tolerates ``error`` events: a failed
    model call may surface one before the run pauses.
    """
    events: list[dict[str, Any]] = []
    deadline = time.monotonic() + COLLECT_EVENTS_TIMEOUT
    while (remaining := deadline - time.monotonic()) > 0:
        try:
            event = json.loads(ws.recv(timeout=remaining))
        except TimeoutError:
            break
        events.append(event)
        if until(event):
            return events
    pytest.fail(f"Timed out collecting events: {events}\n{app.diagnostics()}")


# ---------------------------------------------------------------- model faults


def set_model_fault(app, kind: str, *, count: int | None) -> None:
    """Fail the next ``count`` provider attempts (``None``: until cleared).

    Change it only while no model call is in flight.
    """
    path = app.root / MODEL_FAULT_FILE
    staged = path.with_suffix(".tmp")
    staged.write_text(json.dumps({"kind": kind, "remaining": count}))
    os.replace(staged, path)


def clear_model_fault(app) -> None:
    (app.root / MODEL_FAULT_FILE).unlink(missing_ok=True)


def model_faults(app, kind: str | None = None) -> list[dict[str, Any]]:
    return [
        call
        for call in model_calls(app)
        if call.get("fault") is not None and kind in (None, call["fault"])
    ]


def answered_model_calls(app) -> list[dict[str, Any]]:
    return [call for call in model_calls(app) if call.get("fault") is None]


def task_detail(app, task_id: int) -> dict[str, Any]:
    response = app.client.get(f"/api/chat/task/{task_id}", headers=app.headers)
    assert response.status_code == 200, response.text
    return response.json()


def assistant_messages(task_id: int) -> list[tuple[str, str]]:
    """``(message_type, content)`` of the task's persisted assistant messages."""
    from xagent.web.models.chat_message import TaskChatMessage
    from xagent.web.models.database import get_session_local

    with get_session_local()() as db:
        return [
            (message.message_type, message.content)
            for message in db.query(TaskChatMessage)
            .filter(
                TaskChatMessage.task_id == task_id,
                TaskChatMessage.role == "assistant",
            )
            .order_by(TaskChatMessage.id)
        ]
