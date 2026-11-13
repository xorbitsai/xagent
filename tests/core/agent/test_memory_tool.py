"""Tests for the model-invocable ``store_memory`` tool."""

import asyncio
from typing import Any, cast

import pytest

from xagent.core.agent.context.memory_tool import (
    DeleteMemoryTool,
    ForgetAllMyMemoriesTool,
    SearchMemoryTool,
    StoreMemoryTool,
    UpdateMemoryTool,
    build_memory_tools,
    build_store_memory_tool,
)
from xagent.core.agent.pattern.react.react import ReActPattern
from xagent.core.execution_scope import ExecutionScope, ExecutionScopeContext
from xagent.core.memory.core import MemoryNote, MemoryResponse
from xagent.core.memory.in_memory import InMemoryMemoryStore
from xagent.core.memory.lancedb import LanceDBMemoryStore
from xagent.core.model.embedding import BaseEmbedding
from xagent.web.user_isolated_memory import UserContext, UserIsolatedMemoryStore


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
    # The user's request text is conversation content and must not outlive a
    # purged conversation inside the memory store (#2753).
    assert note.metadata == {"kind": "user_preference", "source": "store_memory"}
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
async def test_store_memory_serializes_concurrent_quota_checks() -> None:
    store = RecordingMemoryStore()
    tool = StoreMemoryTool(memory_store=store, task="task", max_stores=1)

    results = await asyncio.gather(
        tool.execute(content="First concurrent insight."),
        tool.execute(content="Second concurrent insight."),
    )

    assert sum(result.get("stored") is True for result in results) == 1
    assert sum("limit" in result.get("error", "") for result in results) == 1
    assert len(store.added) == 1
    assert tool.stored_count == 1


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
        self.get_calls: list[str] = []

    def search(self, **kwargs: Any) -> list[Any]:
        self.searches.append(kwargs)
        return list(self.notes.values())

    def get(self, note_id: str) -> MemoryResponse:
        self.get_calls.append(note_id)
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
    assert tracer.events[-1]["data"]["success"] is True


@pytest.mark.asyncio
async def test_search_memory_reports_backend_failure() -> None:
    class BrokenSearchStore(CrudMemoryStore):
        def search(self, **kwargs: Any) -> list[Any]:
            raise RuntimeError("backend unavailable")

    tracer = TraceEventRecorder()
    tool = SearchMemoryTool(
        memory_store=BrokenSearchStore(), runtime=FakeRuntime(tracer=tracer)
    )

    result = await tool.execute(query="preferences")

    assert result == {"success": False, "error": "Failed to search memories."}
    assert tracer.events[-1]["data"]["success"] is False
    assert tracer.events[-1]["data"]["found"] is False


@pytest.mark.asyncio
async def test_search_memory_reports_terminal_lancedb_failure(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BrokenConnection:
        def open_table(self, _: str) -> Any:
            raise RuntimeError("memory table unavailable")

    base_store = LanceDBMemoryStore(
        db_dir=str(tmp_path),
        collection_name="terminal_failure",
    )
    monkeypatch.setattr(
        base_store._vector_store,
        "get_raw_connection",
        lambda: BrokenConnection(),
    )
    tool = SearchMemoryTool(
        memory_store=UserIsolatedMemoryStore(base_store),
    )

    result = await tool.execute(query="preferences")

    assert result == {"success": False, "error": "Failed to search memories."}


@pytest.mark.asyncio
async def test_update_memory_replaces_content() -> None:
    store = CrudMemoryStore({"mem-1": _note("mem-1", "Old fact.")})
    tool = UpdateMemoryTool(memory_store=store)

    result = await tool.execute(memory_id="mem-1", content="Corrected fact.")

    assert result == {"success": True, "memory_id": "mem-1"}
    assert store.updated[0].content == "Corrected fact."
    # No request text is stamped onto the rewritten note (#2753).
    assert store.updated[0].metadata == {}


@pytest.mark.asyncio
async def test_update_memory_initializes_missing_metadata() -> None:
    note = _note("mem-1", "Old fact.")
    note.metadata = None  # type: ignore[assignment]
    store = CrudMemoryStore({"mem-1": note})
    tool = UpdateMemoryTool(memory_store=store)

    result = await tool.execute(memory_id="mem-1", content="Corrected fact.")

    assert result == {"success": True, "memory_id": "mem-1"}
    assert store.updated[0].metadata == {}


@pytest.mark.asyncio
async def test_update_memory_reports_missing_note_and_bad_args() -> None:
    store = CrudMemoryStore()
    tool = UpdateMemoryTool(memory_store=store)

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

    assert result == {
        "success": True,
        "deleted": True,
        "verified": True,
        "memory_id": "mem-1",
    }
    assert store.deleted == ["mem-1"]
    assert store.get_calls == ["mem-1", "mem-1", "mem-1"]
    assert store.searches == []
    assert missing["success"] is False
    assert missing["deleted"] is False


@pytest.mark.asyncio
async def test_delete_memory_reports_lookup_delete_and_verification_failures() -> None:
    secret = "postgresql://admin:password@memory.internal/private"

    class LookupExceptionStore(CrudMemoryStore):
        def get(self, note_id: str) -> MemoryResponse:
            raise RuntimeError("lookup unavailable")

    class LookupFailureStore(CrudMemoryStore):
        def get(self, note_id: str) -> MemoryResponse:
            return MemoryResponse(success=False, error=secret)

    class DeleteFailureStore(CrudMemoryStore):
        def delete(self, note_id: str) -> MemoryResponse:
            return MemoryResponse(success=False, error=secret)

    class DeleteExceptionStore(CrudMemoryStore):
        def delete(self, note_id: str) -> MemoryResponse:
            raise RuntimeError("delete unavailable")

    class VerificationFailureStore(CrudMemoryStore):
        def get(self, note_id: str) -> MemoryResponse:
            self.get_calls.append(note_id)
            if len(self.get_calls) == 1:
                return super().get(note_id)
            return MemoryResponse(success=False, error=secret)

    class VerificationExceptionStore(CrudMemoryStore):
        def get(self, note_id: str) -> MemoryResponse:
            self.get_calls.append(note_id)
            if len(self.get_calls) == 1:
                return super().get(note_id)
            raise RuntimeError("verification unavailable")

    class DeleteNoOpStore(CrudMemoryStore):
        def delete(self, note_id: str) -> MemoryResponse:
            return MemoryResponse(success=True, memory_id=note_id)

    note = _note("mem-1", "Forget this.")
    lookup = await DeleteMemoryTool(
        memory_store=LookupExceptionStore({"mem-1": note})
    ).execute(memory_id="mem-1")
    lookup_failure = await DeleteMemoryTool(
        memory_store=LookupFailureStore({"mem-1": note})
    ).execute(memory_id="mem-1")
    delete = await DeleteMemoryTool(
        memory_store=DeleteFailureStore({"mem-1": note})
    ).execute(memory_id="mem-1")
    delete_exception = await DeleteMemoryTool(
        memory_store=DeleteExceptionStore({"mem-1": note})
    ).execute(memory_id="mem-1")
    verification = await DeleteMemoryTool(
        memory_store=VerificationFailureStore({"mem-1": note})
    ).execute(memory_id="mem-1")
    verification_exception = await DeleteMemoryTool(
        memory_store=VerificationExceptionStore({"mem-1": note})
    ).execute(memory_id="mem-1")
    still_exists = await DeleteMemoryTool(
        memory_store=DeleteNoOpStore({"mem-1": note})
    ).execute(memory_id="mem-1")

    assert lookup == {
        "success": False,
        "deleted": False,
        "error": "Failed to look up memory before deletion.",
    }
    assert lookup_failure == lookup
    assert delete == {
        "success": False,
        "deleted": None,
        "verified": False,
        "error": "Failed to delete memory.",
    }
    assert delete_exception == delete
    assert verification["success"] is False
    assert verification["deleted"] is None
    assert verification["verified"] is False
    assert "could not be verified" in verification["error"]
    assert verification_exception == verification
    assert still_exists == {
        "success": False,
        "deleted": False,
        "verified": False,
        "error": "Memory deletion could not be verified; the memory still exists.",
    }
    assert all(
        secret not in str(result)
        for result in (
            lookup_failure,
            delete,
            delete_exception,
            verification,
            verification_exception,
        )
    )


@pytest.mark.asyncio
async def test_delete_memory_fails_closed_on_success_without_content() -> None:
    class MissingContentStore(CrudMemoryStore):
        def get(self, note_id: str) -> MemoryResponse:
            return MemoryResponse(success=True, memory_id=note_id, content=None)

    store = MissingContentStore({"mem-1": _note("mem-1", "Forget this.")})
    result = await DeleteMemoryTool(memory_store=store).execute("mem-1")

    assert result == {
        "success": False,
        "deleted": False,
        "error": "Failed to look up memory before deletion.",
    }
    assert store.deleted == []


@pytest.mark.asyncio
async def test_delete_memory_rejects_other_owner_and_scope_ids() -> None:
    base_store = InMemoryMemoryStore()
    store = UserIsolatedMemoryStore(base_store)
    scope_a = ExecutionScope(memory_dimensions={"channel": "a"})
    scope_b = ExecutionScope(memory_dimensions={"channel": "b"})

    with UserContext(1), ExecutionScopeContext(scope_a):
        first = MemoryNote(content="User likes jasmine tea.")
        assert store.add(first).success
    with UserContext(1), ExecutionScopeContext(scope_b):
        similar = MemoryNote(content="User likes jasmine tea with ice.")
        assert store.add(similar).success
    with UserContext(2), ExecutionScopeContext(scope_a):
        other_owner = MemoryNote(content="User likes jasmine tea.")
        assert store.add(other_owner).success

    with UserContext(1), ExecutionScopeContext(scope_a):
        deleted = await DeleteMemoryTool(memory_store=store).execute(first.id)
        wrong_scope = await DeleteMemoryTool(memory_store=store).execute(similar.id)
        wrong_owner = await DeleteMemoryTool(memory_store=store).execute(other_owner.id)

    assert deleted["success"] is True
    assert wrong_scope == {
        "success": False,
        "deleted": False,
        "error": f"Memory '{similar.id}' not found or access denied.",
    }
    assert wrong_owner == {
        "success": False,
        "deleted": False,
        "error": f"Memory '{other_owner.id}' not found or access denied.",
    }
    assert not base_store.get(first.id).success
    assert base_store.get(similar.id).success
    assert base_store.get(other_owner.id).success


class _DeterministicEmbedding(BaseEmbedding):
    def __init__(self) -> None:
        self._dimension = 8

    def encode(self, text: Any, dimension: Any = None, instruct: Any = None) -> Any:
        if isinstance(text, str):
            return [0.1] * self._dimension
        return [[0.1] * self._dimension for _ in text]

    def get_dimension(self) -> int:
        return self._dimension

    @property
    def abilities(self) -> list[str]:
        return ["embed"]


@pytest.mark.asyncio
async def test_delete_memory_persists_after_lancedb_reopen(tmp_path: Any) -> None:
    scope = ExecutionScope(memory_dimensions={"channel": "a"})
    first_store = UserIsolatedMemoryStore(
        LanceDBMemoryStore(
            db_dir=str(tmp_path),
            collection_name="delete_reopen",
            embedding_model=_DeterministicEmbedding(),
        )
    )
    deleted_note = MemoryNote(content="User likes jasmine tea.")
    retained_note = MemoryNote(content="User likes jasmine tea with ice.")

    with UserContext(1), ExecutionScopeContext(scope):
        assert first_store.add(deleted_note).success
        assert first_store.add(retained_note).success
        result = await DeleteMemoryTool(memory_store=first_store).execute(
            deleted_note.id
        )
    assert result["success"] is True

    reopened_store = UserIsolatedMemoryStore(
        LanceDBMemoryStore(
            db_dir=str(tmp_path),
            collection_name="delete_reopen",
            embedding_model=_DeterministicEmbedding(),
        )
    )
    with UserContext(1), ExecutionScopeContext(scope):
        assert not reopened_store.get(deleted_note.id).success
        assert reopened_store.get(retained_note.id).success


_MEMBER_A = ExecutionScope(
    memory_dimensions={"client_application_id": "10", "end_user_id": "7"}
)
_MEMBER_B = ExecutionScope(
    memory_dimensions={"client_application_id": "10", "end_user_id": "8"}
)


def _seed_two_members(store: UserIsolatedMemoryStore) -> dict[str, str]:
    """Two members of one workspace under one machine owner, a second owner
    with member A's exact dimensions, and the owner's dimension-less note."""
    notes: dict[str, str] = {}
    for name, user_id, scope in (
        ("a1", 1, _MEMBER_A),
        ("a2", 1, _MEMBER_A),
        ("b", 1, _MEMBER_B),
        ("other_owner", 2, _MEMBER_A),
        ("owner_plain", 1, ExecutionScope(strict_memory_isolation=True)),
    ):
        with UserContext(user_id), ExecutionScopeContext(scope):
            note = MemoryNote(content=f"note {name}")
            assert store.add(note).success
            notes[name] = note.id
    return notes


@pytest.mark.asyncio
async def test_forget_all_my_memories_deletes_only_the_requesters_notes() -> None:
    base_store = InMemoryMemoryStore()
    store = UserIsolatedMemoryStore(base_store)
    notes = _seed_two_members(store)

    with UserContext(1), ExecutionScopeContext(_MEMBER_A):
        result = await ForgetAllMyMemoriesTool(memory_store=store).execute()

    assert result == {"success": True, "deleted_count": 2}
    remaining = {name for name, id_ in notes.items() if base_store.get(id_).success}
    assert remaining == {"b", "other_owner", "owner_plain"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scope",
    [ExecutionScope(strict_memory_isolation=True), None],
    ids=["creator_direct", "unscoped"],
)
async def test_forget_all_my_memories_refuses_without_scope_dimensions(
    scope: ExecutionScope | None,
) -> None:
    base_store = InMemoryMemoryStore()
    store = UserIsolatedMemoryStore(base_store)
    notes = _seed_two_members(store)

    with UserContext(1), ExecutionScopeContext(scope):
        result = await ForgetAllMyMemoriesTool(memory_store=store).execute()

    assert result["success"] is False
    assert result["deleted_count"] == 0
    assert "unavailable" in result["error"]
    assert all(base_store.get(id_).success for id_ in notes.values())


@pytest.mark.asyncio
async def test_forget_all_my_memories_reports_storage_failure() -> None:
    class FailingStore:
        def delete_current_scope(self) -> int:
            raise RuntimeError("disk full")

    result = await ForgetAllMyMemoriesTool(memory_store=FailingStore()).execute()

    assert result == {
        "success": False,
        "deleted_count": None,
        "error": "Failed to delete memories.",
    }


@pytest.mark.asyncio
async def test_forget_all_my_memories_on_lancedb(tmp_path: Any) -> None:
    base_store = LanceDBMemoryStore(
        db_dir=str(tmp_path),
        collection_name="forget_all",
        embedding_model=_DeterministicEmbedding(),
    )
    store = UserIsolatedMemoryStore(base_store)
    notes = _seed_two_members(store)

    with UserContext(1), ExecutionScopeContext(_MEMBER_A):
        result = await ForgetAllMyMemoriesTool(memory_store=store).execute()
        assert store.list_all() == []

    assert result == {"success": True, "deleted_count": 2}
    remaining = {name for name, id_ in notes.items() if base_store.get(id_).success}
    assert remaining == {"b", "other_owner", "owner_plain"}


def test_build_memory_tools_composition() -> None:
    assert build_memory_tools(memory_store=None, task="task") == []
    tools = build_memory_tools(memory_store=CrudMemoryStore(), task="task")
    assert [tool.name for tool in tools] == [
        "store_memory",
        "search_memory",
        "update_memory",
        "delete_memory",
        "forget_all_my_memories",
    ]
    assert "current user explicitly asks" in tools[3].description
    assert "explicitly asks you to forget everything" in tools[4].description


@pytest.mark.asyncio
async def test_update_memory_reports_note_without_content() -> None:
    class NoContentStore(CrudMemoryStore):
        def get(self, note_id: str) -> MemoryResponse:
            return MemoryResponse(success=True, memory_id=note_id, content=None)

    tool = UpdateMemoryTool(memory_store=NoContentStore())

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
async def test_react_uses_selected_custom_memory_tool_factory() -> None:
    class SkillManager:
        async def list_skills(self) -> list[dict[str, str]]:
            return [{"name": "memory-skill", "description": "Use memory"}]

        async def get_skill(self, _name: str) -> None:
            return None

    pattern = ReActPattern()
    store = InMemoryMemoryStore()
    custom_tool = object()
    calls: list[dict[str, Any]] = []

    factory_tools = [custom_tool]

    def factory(**kwargs: Any) -> list[Any]:
        calls.append(kwargs)
        return factory_tools

    pattern._memory_store = store
    pattern._memory_tools_factory = factory
    context = type("Context", (), {"metadata": {}})()
    runtime = FakeRuntime()

    tools = await pattern._with_context_tools(
        tools=[],
        context=context,
        task_text="remember this",
        runtime=cast(Any, runtime),
        skill_manager=SkillManager(),
        allowed_skills=None,
    )

    assert tools[0] is custom_tool
    assert tools[1].name == "load_skill"
    assert factory_tools == [custom_tool]
    assert context.metadata["memory_tools_enabled"] is True
    assert calls == [
        {
            "memory_store": store,
            "task": "remember this",
            "runtime": runtime,
            "context": context,
        }
    ]


@pytest.mark.asyncio
async def test_search_memory_coerces_invalid_limit() -> None:
    store = RecordingMemoryStore()
    tool = SearchMemoryTool(memory_store=store)

    result = await tool.execute(query="anything", limit="lots")  # type: ignore[arg-type]

    assert result["success"] is True
    assert store.searches[0]["k"] == 5
