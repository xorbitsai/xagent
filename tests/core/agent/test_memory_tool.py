"""Tests for the model-invocable ``store_memory`` tool."""

from typing import Any

import pytest

from xagent.core.agent.context.memory_tool import (
    DeleteMemoryTool,
    SearchMemoryTool,
    StoreMemoryTool,
    UpdateMemoryTool,
    build_memory_tools,
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


class CrudMemoryStore(RecordingMemoryStore):
    """Recording store with get/update/delete for the CRUD tools."""

    def __init__(self, notes: dict[str, MemoryNote] | None = None) -> None:
        super().__init__()
        self.notes = dict(notes or {})
        self.updated: list[MemoryNote] = []
        self.deleted: list[str] = []

    def search(self, **kwargs: Any) -> list[Any]:
        self.searches.append(kwargs)
        return list(self.notes.values())

    def get(self, note_id: str) -> MemoryResponse:
        note = self.notes.get(note_id)
        if note is None:
            return MemoryResponse(success=False, error="Note not found")
        return MemoryResponse(success=True, memory_id=note_id, content=note)

    def update(self, note: MemoryNote) -> MemoryResponse:
        self.updated.append(note)
        self.notes[note.id] = note
        return MemoryResponse(success=True, memory_id=note.id)

    def delete(self, note_id: str) -> MemoryResponse:
        if note_id not in self.notes:
            return MemoryResponse(success=False, error="Note not found")
        del self.notes[note_id]
        self.deleted.append(note_id)
        return MemoryResponse(success=True, memory_id=note_id)


def _note(note_id: str, content: str) -> MemoryNote:
    return MemoryNote(id=note_id, content=content, category="react_memory")


@pytest.mark.asyncio
async def test_search_memory_returns_ids_and_content() -> None:
    store = CrudMemoryStore({"mem-1": _note("mem-1", "User prefers Chinese.")})
    tool = SearchMemoryTool(memory_store=store)

    result = await tool.execute(query="preferences")

    assert result["success"] is True
    assert result["count"] == 1
    assert result["memories"] == [
        {
            "id": "mem-1",
            "content": "User prefers Chinese.",
            "category": "react_memory",
        }
    ]
    assert store.searches[0]["query"] == "preferences"


@pytest.mark.asyncio
async def test_search_memory_rejects_empty_query_and_emits_trace() -> None:
    tracer = TraceEventRecorder()
    store = CrudMemoryStore()
    tool = SearchMemoryTool(memory_store=store, runtime=FakeRuntime(tracer=tracer))

    empty = await tool.execute(query="  ")
    ok = await tool.execute(query="anything")

    assert empty["success"] is False
    assert ok["success"] is True
    event_types = [event["event_type"] for event in tracer.events]
    assert event_types == [
        "task_start_memory_retrieve",
        "task_end_memory_retrieve",
    ]


@pytest.mark.asyncio
async def test_update_memory_replaces_content() -> None:
    store = CrudMemoryStore({"mem-1": _note("mem-1", "Old fact.")})
    tool = UpdateMemoryTool(memory_store=store, task="current task")

    result = await tool.execute(memory_id="mem-1", content="Corrected fact.")

    assert result == {"success": True, "memory_id": "mem-1"}
    assert store.updated[0].content == "Corrected fact."
    assert store.updated[0].metadata["updated_by_task"] == "current task"


@pytest.mark.asyncio
async def test_update_memory_reports_missing_note_and_bad_args() -> None:
    store = CrudMemoryStore()
    tool = UpdateMemoryTool(memory_store=store, task="task")

    missing = await tool.execute(memory_id="nope", content="New text.")
    bad = await tool.execute(memory_id="", content="")

    assert missing["success"] is False
    assert "not found" in missing["error"]
    assert bad["success"] is False
    assert store.updated == []


@pytest.mark.asyncio
async def test_delete_memory_removes_note() -> None:
    store = CrudMemoryStore({"mem-1": _note("mem-1", "Wrong fact.")})
    tool = DeleteMemoryTool(memory_store=store)

    result = await tool.execute(memory_id="mem-1")
    missing = await tool.execute(memory_id="mem-1")

    assert result == {"success": True, "memory_id": "mem-1"}
    assert store.deleted == ["mem-1"]
    assert missing["success"] is False


def test_build_memory_tools_composition() -> None:
    assert build_memory_tools(memory_store=None, task="task") == []
    tools = build_memory_tools(memory_store=CrudMemoryStore(), task="task")
    assert [tool.name for tool in tools] == [
        "store_memory",
        "search_memory",
        "update_memory",
        "delete_memory",
    ]


@pytest.mark.asyncio
async def test_update_memory_reports_note_without_content() -> None:
    class NoContentStore(CrudMemoryStore):
        def get(self, note_id: str) -> MemoryResponse:
            return MemoryResponse(success=True, memory_id=note_id, content=None)

    tool = UpdateMemoryTool(memory_store=NoContentStore(), task="task")

    result = await tool.execute(memory_id="mem-1", content="New text.")

    assert result["success"] is False
    assert "no stored note" in result["error"]


def test_build_memory_tools_stamps_guidance_flag_on_context() -> None:
    class Ctx:
        def __init__(self) -> None:
            self.metadata: dict[str, Any] = {}

    context = Ctx()
    tools = build_memory_tools(
        memory_store=RecordingMemoryStore(), task="task", context=context
    )

    assert tools
    assert context.metadata["memory_tools_enabled"] is True

    disabled_context = Ctx()
    assert (
        build_memory_tools(memory_store=None, task="task", context=disabled_context)
        == []
    )
    assert "memory_tools_enabled" not in disabled_context.metadata


@pytest.mark.asyncio
async def test_search_memory_coerces_invalid_limit() -> None:
    store = RecordingMemoryStore()
    tool = SearchMemoryTool(memory_store=store)

    result = await tool.execute(query="anything", limit="lots")  # type: ignore[arg-type]

    assert result["success"] is True
    assert store.searches[0]["k"] == 5
