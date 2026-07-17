"""Model-invocable memory storage tool.

ReAct runs with an active memory store expose this tool so the model stores
valuable insights itself during execution, instead of a framework-driven
end-of-run memory-evaluation LLM call.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Literal

from pydantic import BaseModel, Field

from ...memory.core import MemoryNote
from ..trace import trace_memory_store_end, trace_memory_store_start

logger = logging.getLogger(__name__)

STORE_MEMORY_TOOL_NAME = "store_memory"
DEFAULT_MAX_STORES_PER_RUN = 5
DEFAULT_DEDUP_SIMILARITY_THRESHOLD = 0.9

MemoryKind = Literal[
    "user_preference",
    "failure_pattern",
    "success_pattern",
    "tool_usage",
    "strategy",
    "domain_insight",
]
_MEMORY_KINDS = frozenset(
    {
        "user_preference",
        "failure_pattern",
        "success_pattern",
        "tool_usage",
        "strategy",
        "domain_insight",
    }
)

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
        dedup_similarity_threshold: float = DEFAULT_DEDUP_SIMILARITY_THRESHOLD,
    ) -> None:
        self.memory_store = memory_store
        self.task = task
        self.runtime = runtime
        self.category = category
        self.max_stores = max_stores
        self.dedup_similarity_threshold = dedup_similarity_threshold
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
                similarity_threshold=self.dedup_similarity_threshold,
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


def _runtime_attr(runtime: Any | None, name: str) -> Any | None:
    if runtime is None:
        return None
    return getattr(runtime, name, None)
