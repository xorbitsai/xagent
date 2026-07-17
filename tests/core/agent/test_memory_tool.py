"""Tests for the model-invocable ``store_memory`` tool."""

from typing import Any

import pytest

from xagent.core.agent.context.memory_tool import (
    StoreMemoryTool,
    build_store_memory_tool,
)
from xagent.core.memory.core import MemoryNote, MemoryResponse


class RecordingMemoryStore:
    def __init__(
        self,
        *,
        search_results: list[Any] | None = None,
        add_success: bool = True,
    ) -> None:
        self.search_results = search_results or []
        self.add_success = add_success
        self.searches: list[dict[str, Any]] = []
        self.added: list[MemoryNote] = []

    def search(self, **kwargs: Any) -> list[Any]:
        self.searches.append(kwargs)
        return list(self.search_results)

    def add(self, note: MemoryNote) -> MemoryResponse:
        self.added.append(note)
        if not self.add_success:
            return MemoryResponse(success=False, error="boom")
        return MemoryResponse(success=True, memory_id=f"mem-{len(self.added)}")


class TraceEventRecorder:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    async def trace_event(
        self,
        event_type: Any,
        *,
        task_id: str | None = None,
        step_id: str | None = None,
        data: dict[str, Any] | None = None,
        **_: Any,
    ) -> str:
        self.events.append(
            {
                "event_type": getattr(event_type, "value", str(event_type)),
                "task_id": task_id,
                "data": data or {},
            }
        )
        return str(len(self.events))


class FakeRuntime:
    def __init__(self, tracer: Any | None = None) -> None:
        self.tracer = tracer
        self.execution_id = "task-1"
        self.active_react_step_id = "react_step_1"


@pytest.mark.asyncio
async def test_store_memory_adds_note_with_metadata() -> None:
    store = RecordingMemoryStore()
    tool = StoreMemoryTool(memory_store=store, task="Fix the deploy pipeline")

    result = await tool.execute(
        content="User prefers reports in Chinese.", kind="user_preference"
    )

    assert result == {"success": True, "stored": True, "memory_id": "mem-1"}
    note = store.added[0]
    assert note.content == "User prefers reports in Chinese."
    assert note.category == "react_memory"
    assert note.metadata["task"] == "Fix the deploy pipeline"
    assert note.metadata["kind"] == "user_preference"
    assert note.metadata["source"] == "store_memory"
    assert tool.stored_count == 1


@pytest.mark.asyncio
async def test_store_memory_skips_duplicate() -> None:
    store = RecordingMemoryStore(search_results=[object()])
    tool = StoreMemoryTool(memory_store=store, task="task")

    result = await tool.execute(content="Same insight again.")

    assert result["success"] is True
    assert result["stored"] is False
    assert store.added == []
    assert store.searches[0]["filters"] == {"category": "react_memory"}
    assert tool.stored_count == 0


@pytest.mark.asyncio
async def test_store_memory_enforces_per_run_quota() -> None:
    store = RecordingMemoryStore()
    tool = StoreMemoryTool(memory_store=store, task="task", max_stores=2)

    assert (await tool.execute(content="First insight."))["success"] is True
    assert (await tool.execute(content="Second insight."))["success"] is True
    result = await tool.execute(content="Third insight.")

    assert result["success"] is False
    assert "limit" in result["error"]
    assert len(store.added) == 2


@pytest.mark.asyncio
async def test_store_memory_rejects_empty_content() -> None:
    store = RecordingMemoryStore()
    tool = StoreMemoryTool(memory_store=store, task="task")

    result = await tool.execute(content="   ")

    assert result["success"] is False
    assert store.added == []


@pytest.mark.asyncio
async def test_store_memory_normalizes_unknown_kind() -> None:
    store = RecordingMemoryStore()
    tool = StoreMemoryTool(memory_store=store, task="task")

    result = await tool.execute(content="An insight.", kind="made_up_kind")

    assert result["success"] is True
    assert store.added[0].metadata["kind"] == "domain_insight"


@pytest.mark.asyncio
async def test_store_memory_reports_add_failure() -> None:
    store = RecordingMemoryStore(add_success=False)
    tool = StoreMemoryTool(memory_store=store, task="task")

    result = await tool.execute(content="An insight.")

    assert result["success"] is False
    assert tool.stored_count == 0


@pytest.mark.asyncio
async def test_store_memory_emits_store_trace_events() -> None:
    tracer = TraceEventRecorder()
    store = RecordingMemoryStore()
    tool = StoreMemoryTool(
        memory_store=store,
        task="task",
        runtime=FakeRuntime(tracer=tracer),
    )

    await tool.execute(content="An insight.")

    event_types = [event["event_type"] for event in tracer.events]
    assert event_types == ["task_start_memory_store", "task_end_memory_store"]
    assert all(event["task_id"] == "task-1" for event in tracer.events)
    assert tracer.events[1]["data"]["storage_success"] is True
    assert tracer.events[1]["data"]["memory_id"] == "mem-1"


@pytest.mark.asyncio
async def test_store_memory_stores_when_dedup_search_fails() -> None:
    class BrokenSearchStore(RecordingMemoryStore):
        def search(self, **kwargs: Any) -> list[Any]:
            raise RuntimeError("search unavailable")

    store = BrokenSearchStore()
    tool = StoreMemoryTool(memory_store=store, task="task")

    result = await tool.execute(content="An insight.")

    assert result["success"] is True
    assert len(store.added) == 1


def test_build_store_memory_tool_requires_store() -> None:
    assert build_store_memory_tool(memory_store=None, task="task") is None
    tool = build_store_memory_tool(memory_store=RecordingMemoryStore(), task="task")
    assert isinstance(tool, StoreMemoryTool)
    assert tool.name == "store_memory"
