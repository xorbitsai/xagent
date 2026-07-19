"""Model-invocable memory tools.

ReAct runs with an active memory store expose these tools so the model
manages memories itself during execution — storing valuable insights,
searching for more context, and correcting or removing stale entries —
instead of framework-driven memory pipeline steps.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Literal, get_args

from pydantic import BaseModel, Field

from ...memory.core import MemoryNote
from ..trace import (
    trace_memory_retrieve_end,
    trace_memory_retrieve_start,
    trace_memory_store_end,
    trace_memory_store_start,
)

logger = logging.getLogger(__name__)

STORE_MEMORY_TOOL_NAME = "store_memory"
SEARCH_MEMORY_TOOL_NAME = "search_memory"
UPDATE_MEMORY_TOOL_NAME = "update_memory"
DELETE_MEMORY_TOOL_NAME = "delete_memory"
MEMORY_TOOLS_METADATA_KEY = "memory_tools_enabled"
# Cap per ReAct run. Each DAG step runs its own ReActPattern, so a DAG
# task gets a fresh cap per step rather than one shared task-wide cap.
DEFAULT_MAX_STORES_PER_RUN = 5
# Cosine distance passed to search(similarity_threshold=...): lower = stricter
# dedup (fewer results count as duplicates).
DEFAULT_DEDUP_DISTANCE_THRESHOLD = 0.9
DEFAULT_SEARCH_LIMIT = 5
MAX_SEARCH_LIMIT = 20

MemoryKind = Literal[
    "user_preference",
    "failure_pattern",
    "success_pattern",
    "tool_usage",
    "strategy",
    "domain_insight",
]
_MEMORY_KINDS = frozenset(get_args(MemoryKind))

_STORE_MEMORY_DESCRIPTION = """Store a durable memory for future tasks.

Use this when you notice a UNIQUE, NON-OBVIOUS insight worth remembering, such as:
- A clear user preference or stable behavior pattern
- A non-obvious failure and how it was fixed
- A reusable strategy that is not routine
- A domain-specific insight that is hard to obtain otherwise

Do NOT store routine task completions, generic tool usage, common facts, or obvious strategies. Most tasks do not need any memory stored. Write the content as a self-contained statement understandable without this conversation."""


class StoreMemoryArgs(BaseModel):
    content: str = Field(
        description=(
            "Self-contained memory text. State the insight directly; do not "
            "reference 'this task' or 'the user said above'."
        )
    )
    kind: MemoryKind = Field(
        default="domain_insight",
        description="What kind of insight this memory captures.",
    )


class StoreMemoryTool:
    """Execution-scoped ``store_memory`` tool bound to a memory store."""

    name = STORE_MEMORY_TOOL_NAME
    description = _STORE_MEMORY_DESCRIPTION
    args_schema = StoreMemoryArgs

    def __init__(
        self,
        *,
        memory_store: Any,
        task: str,
        runtime: Any | None = None,
        category: str = "react_memory",
        max_stores: int = DEFAULT_MAX_STORES_PER_RUN,
        dedup_distance_threshold: float = DEFAULT_DEDUP_DISTANCE_THRESHOLD,
    ) -> None:
        self.memory_store = memory_store
        self.task = task
        self.runtime = runtime
        self.category = category
        self.max_stores = max_stores
        self.dedup_distance_threshold = dedup_distance_threshold
        self.stored_count = 0

    async def execute(
        self, content: str, kind: str = "domain_insight"
    ) -> dict[str, Any]:
        content = str(content or "").strip()
        if not content:
            return {
                "success": False,
                "error": "Memory content must be a non-empty string.",
            }
        if self.stored_count >= self.max_stores:
            return {
                "success": False,
                "error": (
                    f"Memory storage limit ({self.max_stores}) reached for this "
                    "task; do not store further memories."
                ),
            }
        if kind not in _MEMORY_KINDS:
            kind = "domain_insight"

        task_id = str(_runtime_attr(self.runtime, "execution_id") or "")
        step_id = _runtime_attr(self.runtime, "active_react_step_id")
        tracer = _runtime_attr(self.runtime, "tracer")

        if tracer is not None and task_id:
            await trace_memory_store_start(
                tracer,
                task_id,
                data={
                    "task": self.task[:200],
                    "memory_category": self.category,
                    "memory_kind": kind,
                    "source": STORE_MEMORY_TOOL_NAME,
                    "step_id": step_id,
                },
            )

        duplicate = await asyncio.to_thread(self._find_similar, content)
        if duplicate is not None:
            if tracer is not None and task_id:
                await trace_memory_store_end(
                    tracer,
                    task_id,
                    data={
                        "storage_success": False,
                        "decision": "duplicate_skipped",
                        "source": STORE_MEMORY_TOOL_NAME,
                        "step_id": step_id,
                    },
                )
            return {
                "success": True,
                "stored": False,
                "message": (
                    "A very similar memory already exists; skipped storing a duplicate."
                ),
            }

        memory_id = await asyncio.to_thread(self._add, content, kind)

        if tracer is not None and task_id:
            await trace_memory_store_end(
                tracer,
                task_id,
                data={
                    "storage_success": bool(memory_id),
                    "memory_id": memory_id,
                    "source": STORE_MEMORY_TOOL_NAME,
                    "step_id": step_id,
                },
            )

        if not memory_id:
            return {"success": False, "error": "Failed to store memory."}

        self.stored_count += 1
        return {"success": True, "stored": True, "memory_id": memory_id}

    def _find_similar(self, content: str) -> Any | None:
        search = getattr(self.memory_store, "search", None)
        if not callable(search):
            return None
        try:
            results = search(
                query=content,
                k=1,
                filters={"category": self.category},
                similarity_threshold=self.dedup_distance_threshold,
            )
        except Exception:
            logger.exception("store_memory dedup search failed; storing anyway")
            return None
        return results[0] if results else None

    def _add(self, content: str, kind: str) -> str | None:
        note = MemoryNote(
            content=content,
            category=self.category,
            metadata={
                "task": self.task,
                "kind": kind,
                "source": STORE_MEMORY_TOOL_NAME,
            },
        )
        try:
            response = self.memory_store.add(note)
        except Exception:
            logger.exception("store_memory failed to add memory")
            return None
        if not getattr(response, "success", False):
            return None
        return getattr(response, "memory_id", None)


class SearchMemoryArgs(BaseModel):
    query: str = Field(description="What to look for in stored memories.")
    limit: int = Field(
        default=DEFAULT_SEARCH_LIMIT,
        ge=1,
        le=MAX_SEARCH_LIMIT,
        description="Maximum number of memories to return.",
    )


class SearchMemoryTool:
    """Execution-scoped ``search_memory`` tool bound to a memory store."""

    name = SEARCH_MEMORY_TOOL_NAME
    description = (
        "Search stored memories from previous tasks. Use this when earlier "
        "insights, user preferences, or known failure patterns could help the "
        "current step. Results include each memory's id, which update_memory "
        "and delete_memory take."
    )
    args_schema = SearchMemoryArgs

    def __init__(self, *, memory_store: Any, runtime: Any | None = None) -> None:
        self.memory_store = memory_store
        self.runtime = runtime

    async def execute(
        self, query: str, limit: int = DEFAULT_SEARCH_LIMIT
    ) -> dict[str, Any]:
        query = str(query or "").strip()
        if not query:
            return {"success": False, "error": "query must be a non-empty string."}
        try:
            limit = int(limit)
        except (TypeError, ValueError):
            limit = DEFAULT_SEARCH_LIMIT
        limit = max(1, min(limit, MAX_SEARCH_LIMIT))

        task_id = str(_runtime_attr(self.runtime, "execution_id") or "")
        step_id = _runtime_attr(self.runtime, "active_react_step_id")
        tracer = _runtime_attr(self.runtime, "tracer")

        if tracer is not None and task_id:
            await trace_memory_retrieve_start(
                tracer,
                task_id=task_id,
                step_id=step_id,
                data={"query": query[:200], "source": SEARCH_MEMORY_TOOL_NAME},
            )

        memories = await asyncio.to_thread(self._search, query, limit)

        if tracer is not None and task_id:
            await trace_memory_retrieve_end(
                tracer,
                task_id=task_id,
                step_id=step_id,
                data={
                    "query": query[:200],
                    "memories_count": len(memories),
                    "found": bool(memories),
                    "source": SEARCH_MEMORY_TOOL_NAME,
                },
            )

        return {"success": True, "memories": memories, "count": len(memories)}

    def _search(self, query: str, limit: int) -> list[dict[str, Any]]:
        try:
            results = self.memory_store.search(query=query, k=limit)
        except Exception:
            logger.exception("search_memory failed")
            return []
        memories = []
        for note in results or []:
            memories.append(
                {
                    "id": getattr(note, "id", None),
                    "content": getattr(note, "content", ""),
                    "category": getattr(note, "category", "general"),
                }
            )
        return memories


class UpdateMemoryArgs(BaseModel):
    memory_id: str = Field(
        description="Id of the memory to update, as returned by search_memory."
    )
    content: str = Field(
        description="Corrected, self-contained memory text that replaces the "
        "old content."
    )


class UpdateMemoryTool:
    """Execution-scoped ``update_memory`` tool bound to a memory store."""

    name = UPDATE_MEMORY_TOOL_NAME
    description = (
        "Replace the content of a stored memory. Use this when a memory turns "
        "out to be outdated or contradicts what you observe now. Find the "
        "memory's id with search_memory first."
    )
    args_schema = UpdateMemoryArgs

    def __init__(self, *, memory_store: Any, task: str) -> None:
        self.memory_store = memory_store
        self.task = task

    async def execute(self, memory_id: str, content: str) -> dict[str, Any]:
        memory_id = str(memory_id or "").strip()
        content = str(content or "").strip()
        if not memory_id or not content:
            return {
                "success": False,
                "error": "memory_id and content must be non-empty strings.",
            }
        return await asyncio.to_thread(self._update, memory_id, content)

    def _update(self, memory_id: str, content: str) -> dict[str, Any]:
        try:
            existing = self.memory_store.get(memory_id)
            if not getattr(existing, "success", False):
                return {
                    "success": False,
                    "error": f"Memory '{memory_id}' not found.",
                }
            note = getattr(existing, "content", None)
            if note is None:
                return {
                    "success": False,
                    "error": f"Memory '{memory_id}' has no stored note to update.",
                }
            note.content = content
            if isinstance(note.metadata, dict):
                note.metadata["updated_by_task"] = self.task
            response = self.memory_store.update(note)
        except Exception:
            logger.exception("update_memory failed")
            return {"success": False, "error": "Failed to update memory."}
        if not getattr(response, "success", False):
            return {
                "success": False,
                "error": str(getattr(response, "error", None) or "Update failed."),
            }
        return {"success": True, "memory_id": memory_id}


class DeleteMemoryArgs(BaseModel):
    memory_id: str = Field(
        description="Id of the memory to delete, as returned by search_memory."
    )


class DeleteMemoryTool:
    """Execution-scoped ``delete_memory`` tool bound to a memory store."""

    name = DELETE_MEMORY_TOOL_NAME
    description = (
        "Permanently delete a stored memory. Use this only when a memory is "
        "plainly wrong or harmful and cannot be fixed by update_memory. Find "
        "the memory's id with search_memory first."
    )
    args_schema = DeleteMemoryArgs

    def __init__(self, *, memory_store: Any) -> None:
        self.memory_store = memory_store

    async def execute(self, memory_id: str) -> dict[str, Any]:
        memory_id = str(memory_id or "").strip()
        if not memory_id:
            return {"success": False, "error": "memory_id must be a non-empty string."}
        return await asyncio.to_thread(self._delete, memory_id)

    def _delete(self, memory_id: str) -> dict[str, Any]:
        try:
            response = self.memory_store.delete(memory_id)
        except Exception:
            logger.exception("delete_memory failed")
            return {"success": False, "error": "Failed to delete memory."}
        if not getattr(response, "success", False):
            return {
                "success": False,
                "error": str(getattr(response, "error", None) or "Delete failed."),
            }
        return {"success": True, "memory_id": memory_id}


def build_store_memory_tool(
    *,
    memory_store: Any | None,
    task: str,
    runtime: Any | None = None,
    category: str = "react_memory",
) -> StoreMemoryTool | None:
    """Create a ``store_memory`` tool, or None when no memory store is active."""

    if memory_store is None:
        return None
    return StoreMemoryTool(
        memory_store=memory_store,
        task=task,
        runtime=runtime,
        category=category,
    )


def build_memory_tools(
    *,
    memory_store: Any | None,
    task: str,
    runtime: Any | None = None,
    context: Any | None = None,
) -> list[Any]:
    """Create the memory tool set, or [] when no memory store is active.

    When ``context`` is given, mark it so the system context renders the
    memory-usage guidance alongside the tools (see
    ``MEMORY_TOOLS_METADATA_KEY``).
    """

    if memory_store is None:
        return []
    tools: list[Any] = [
        StoreMemoryTool(memory_store=memory_store, task=task, runtime=runtime),
        SearchMemoryTool(memory_store=memory_store, runtime=runtime),
        UpdateMemoryTool(memory_store=memory_store, task=task),
        DeleteMemoryTool(memory_store=memory_store),
    ]
    if tools and context is not None and hasattr(context, "metadata"):
        context.metadata[MEMORY_TOOLS_METADATA_KEY] = True
    return tools


def _runtime_attr(runtime: Any | None, name: str) -> Any | None:
    if runtime is None:
        return None
    return getattr(runtime, name, None)
