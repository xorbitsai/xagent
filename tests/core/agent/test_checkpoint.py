from __future__ import annotations

from typing import Any

import pytest

from xagent.core.agent import Agent, AgentRunner
from xagent.core.agent.checkpoint import (
    CHECKPOINT_EVENT_TYPE,
    CHECKPOINT_TYPE,
    LEGACY_CHECKPOINT_TYPES,
    CheckpointPersistenceError,
    TraceCheckpointStore,
)
from xagent.core.agent.trace import TraceEvent, TraceHandler, Tracer


class PersistentTraceBackend:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    async def trace_event(
        self,
        event_type: Any,
        *,
        task_id: str | None = None,
        data: dict[str, Any] | None = None,
        require_persisted: bool = False,
        **_: Any,
    ) -> str:
        event_id = f"event-{len(self.events) + 1}"
        self.events.append(
            {
                "event_id": event_id,
                "event_type": str(event_type),
                "task_id": task_id,
                "data": dict(data or {}),
                "require_persisted": require_persisted,
            }
        )
        return event_id

    async def load_latest_checkpoint(self, execution_id: str) -> dict[str, Any] | None:
        for event in reversed(self.events):
            data = event["data"]
            if (
                data.get("checkpoint_type") == CHECKPOINT_TYPE
                and data.get("root_execution_id") == execution_id
            ):
                return dict(data)
        return None


class BestEffortTraceBackend:
    async def trace_event(
        self,
        event_type: Any,
        *,
        task_id: str | None = None,
        data: dict[str, Any] | None = None,
    ) -> str:
        del event_type, task_id, data
        return "best-effort"


class NoneReturningCheckpointBackend:
    async def checkpoint(self, **_: Any) -> None:
        return None


class NoneReturningTraceEventBackend:
    async def trace_event(
        self,
        event_type: Any,
        *,
        task_id: str | None = None,
        data: dict[str, Any] | None = None,
        require_persisted: bool = False,
    ) -> None:
        del event_type, task_id, data, require_persisted
        return None


class LegacyCheckpointBackend:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    async def load_latest_checkpoint(self, execution_id: str) -> dict[str, Any]:
        legacy_type = next(iter(LEGACY_CHECKPOINT_TYPES))
        return {
            "checkpoint_type": legacy_type,
            "root_execution_id": execution_id,
            "snapshot": dict(self.payload),
        }


class FailingTraceHandler(TraceHandler):
    async def handle_event(self, _: TraceEvent) -> None:
        raise ValueError("checkpoint write failed: secret=supersecret")


class FakeLLM:
    async def chat(self, **_: Any) -> str:
        return "done"


class CheckpointingPattern:
    async def run(self, *, context: Any, runtime: Any, **_: Any) -> dict[str, Any]:
        context.add_assistant_message("done")
        await runtime.checkpoint(
            "final",
            context=context,
            pattern=self,
            status="completed",
        )
        return {"success": True, "output": "done"}


@pytest.mark.asyncio
async def test_trace_checkpoint_store_persists_full_snapshot_event() -> None:
    backend = PersistentTraceBackend()
    store = TraceCheckpointStore(backend)
    payload = {
        "type": "checkpoint",
        "label": "before_llm",
        "execution_id": "exec-checkpoint",
        "context": {"messages": []},
        "pattern": "ReActPattern",
        "pattern_state": {"current_iteration": 0},
        "status": "thinking",
    }

    event_id = await store.checkpoint(**payload)
    loaded = await store.load_latest_checkpoint("exec-checkpoint")

    assert event_id == "event-1"
    assert backend.events[0]["require_persisted"] is True
    assert backend.events[0]["data"]["checkpoint_type"] == CHECKPOINT_TYPE
    assert backend.events[0]["data"]["snapshot"] == payload
    assert loaded == payload


@pytest.mark.asyncio
async def test_required_trace_error_preserves_redacted_handler_cause() -> None:
    tracer = Tracer()
    tracer.add_handler(FailingTraceHandler())

    with pytest.raises(RuntimeError) as exc_info:
        await tracer.trace_event(
            CHECKPOINT_EVENT_TYPE,
            task_id="trace-cause",
            require_persisted=True,
        )

    message = str(exc_info.value)
    assert "First failure: ValueError: checkpoint write failed" in message
    assert "supersecret" not in message
    assert isinstance(exc_info.value.__cause__, ValueError)


@pytest.mark.asyncio
async def test_trace_checkpoint_store_reads_legacy_checkpoint_marker() -> None:
    payload = {
        "type": "checkpoint",
        "label": "before_llm",
        "execution_id": "legacy-exec",
        "context": {"messages": []},
    }
    store = TraceCheckpointStore(LegacyCheckpointBackend(payload))

    loaded = await store.load_latest_checkpoint("legacy-exec")

    assert loaded == payload


@pytest.mark.asyncio
async def test_trace_checkpoint_store_rejects_best_effort_trace_event() -> None:
    store = TraceCheckpointStore(BestEffortTraceBackend())

    with pytest.raises(CheckpointPersistenceError):
        await store.checkpoint(
            type="checkpoint",
            label="before_llm",
            execution_id="exec-best-effort",
        )


@pytest.mark.asyncio
async def test_trace_checkpoint_store_rejects_none_checkpoint_writer() -> None:
    store = TraceCheckpointStore(NoneReturningCheckpointBackend())

    with pytest.raises(CheckpointPersistenceError):
        await store.checkpoint(
            type="checkpoint",
            label="before_llm",
            execution_id="exec-none-writer",
        )


@pytest.mark.asyncio
async def test_trace_checkpoint_store_rejects_none_trace_event_writer() -> None:
    store = TraceCheckpointStore(NoneReturningTraceEventBackend())

    with pytest.raises(CheckpointPersistenceError):
        await store.checkpoint(
            type="checkpoint",
            label="before_llm",
            execution_id="exec-none-trace-event",
        )


@pytest.mark.asyncio
async def test_runner_can_use_trace_checkpoint_store_for_resume() -> None:
    backend = PersistentTraceBackend()
    store = TraceCheckpointStore(backend)
    agent = Agent(
        name="checkpointed",
        patterns=[CheckpointingPattern()],
        llm=FakeLLM(),
    )
    runner = AgentRunner(agent=agent, tracer=store)

    result = await runner.run(task="Say done", execution_id="exec-runner-store")
    loaded = await store.load_latest_checkpoint("exec-runner-store")

    assert result["success"] is True
    assert loaded is not None
    assert loaded["label"] == "final"
    assert loaded["context"]["messages"][-1]["content"] == "done"
