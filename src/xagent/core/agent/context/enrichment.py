from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from typing import Any, cast

from ...agent.trace import (
    trace_memory_generate_end,
    trace_memory_generate_start,
    trace_memory_retrieve_end,
    trace_memory_retrieve_start,
    trace_memory_store_end,
    trace_memory_store_start,
)
from ...memory.core import MemoryNote
from ...user_context import current_user_id
from ..runtime import LLMCallInterrupted

logger = logging.getLogger(__name__)

MEMORY_CONTEXT_METADATA_KEY = "retrieved_memory_context"
RETRIEVED_MEMORIES_METADATA_KEY = "retrieved_memories"
SELECTED_SKILL_METADATA_KEY = "selected_skill"
SKILL_CONTEXT_METADATA_KEY = "selected_skill_context"
SKILL_SELECTION_ATTEMPTS_METADATA_KEY = "skill_selection_attempts"


class _RuntimeLLMProxy:
    def __init__(self, *, runtime: Any, llm: Any) -> None:
        self.runtime = runtime
        self.llm = llm

    async def chat(self, **kwargs: Any) -> Any:
        run_llm_call = getattr(self.runtime, "run_llm_call", None)
        if callable(run_llm_call):
            return await run_llm_call(self.llm, **kwargs)
        return await self.llm.chat(**kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.llm, name)


async def enrich_context_with_memory(
    *,
    context: Any,
    query: str,
    category: str,
    memory_store: Any | None,
    runtime: Any | None = None,
    similarity_threshold: float | None = None,
    include_general: bool = True,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Retrieve relevant v1-style memories and attach them to context metadata."""

    if memory_store is None or not query.strip():
        return []

    retrieved_by_category = context.metadata.setdefault(
        RETRIEVED_MEMORIES_METADATA_KEY, {}
    )
    if category in retrieved_by_category:
        cached = retrieved_by_category.get(category)
        return cached if isinstance(cached, list) else []

    task_id = str(
        _runtime_attr(runtime, "execution_id")
        or getattr(context, "execution_id", None)
        or ""
    )
    step_id = _runtime_attr(runtime, "active_react_step_id")
    tracer = _runtime_attr(runtime, "tracer")
    user_id = _current_user_id()

    if tracer is not None and task_id:
        await trace_memory_retrieve_start(
            tracer,
            task_id=task_id,
            step_id=step_id,
            data={"query": query[:200], "category": category},
        )

    memories = await asyncio.to_thread(
        _lookup_relevant_memories_with_context,
        memory_store,
        query,
        category,
        include_general,
        limit,
        similarity_threshold,
        user_id,
    )
    retrieved_by_category[category] = memories
    context.metadata[MEMORY_CONTEXT_METADATA_KEY] = _build_memory_context(
        context.metadata.get(MEMORY_CONTEXT_METADATA_KEY), query, memories
    )

    if tracer is not None and task_id:
        await trace_memory_retrieve_end(
            tracer,
            task_id=task_id,
            step_id=step_id,
            data={
                "query": query[:200],
                "category": category,
                "memories_count": len(memories),
                "found": bool(memories),
            },
        )

    logger.info(
        "Retrieved %s v2 memories for category=%s execution=%s",
        len(memories),
        category,
        getattr(context, "execution_id", None),
    )
    return memories


async def enrich_context_with_skill(
    *,
    context: Any,
    task: str,
    llm: Any | None,
    skill_manager: Any | None,
    runtime: Any | None = None,
    allowed_skills: list[str] | None = None,
) -> dict[str, Any] | None:
    """Select a skill and attach full skill guidance to context metadata."""

    if skill_manager is None or llm is None or not task.strip():
        return None
    existing = context.metadata.get(SELECTED_SKILL_METADATA_KEY)
    if isinstance(existing, dict):
        return existing
    attempt_key = _skill_selection_attempt_key(task, allowed_skills)
    attempts = context.metadata.setdefault(SKILL_SELECTION_ATTEMPTS_METADATA_KEY, {})
    if isinstance(attempts, dict) and attempt_key in attempts:
        attempted = attempts.get(attempt_key)
        return attempted if isinstance(attempted, dict) else None

    task_id = str(
        _runtime_attr(runtime, "execution_id")
        or getattr(context, "execution_id", None)
        or ""
    )
    tracer = _runtime_attr(runtime, "tracer")
    selected_skill = await skill_manager.select_skill(
        task=task,
        llm=_RuntimeLLMProxy(runtime=runtime, llm=llm) if runtime is not None else llm,
        tracer=tracer,
        task_id=task_id or None,
        allowed_skills=allowed_skills,
    )
    if not selected_skill:
        if isinstance(attempts, dict):
            attempts[attempt_key] = None
        return None

    selected_summary = {
        "name": selected_skill.get("name"),
        "description": selected_skill.get("description"),
        "when_to_use": selected_skill.get("when_to_use"),
    }
    if isinstance(attempts, dict):
        attempts[attempt_key] = selected_summary
    context.metadata[SELECTED_SKILL_METADATA_KEY] = selected_summary
    context.metadata[SKILL_CONTEXT_METADATA_KEY] = build_skill_context(selected_skill)
    logger.info(
        "Selected v2 skill %s for execution %s",
        selected_skill.get("name"),
        getattr(context, "execution_id", None),
    )
    return cast(dict[str, Any], selected_skill)


async def generate_and_store_react_memory(
    *,
    context: Any,
    task: str,
    result: dict[str, Any],
    iterations: int,
    llm: Any | None,
    memory_store: Any | None,
    runtime: Any | None = None,
) -> None:
    """Generate v1-style ReAct memory insights and store valuable memories."""

    if memory_store is None or llm is None or not task.strip():
        return

    task_id = str(
        _runtime_attr(runtime, "execution_id")
        or getattr(context, "execution_id", None)
        or ""
    )
    step_id = _runtime_attr(runtime, "active_react_step_id")
    tracer = _runtime_attr(runtime, "tracer")
    output = str(result.get("output") or result.get("response") or "")
    messages = [
        message.to_dict()
        for message in getattr(context, "messages", [])
        if not getattr(message, "hidden", False)
    ]

    if tracer is not None and task_id:
        await trace_memory_generate_start(
            tracer,
            task_id,
            data={
                "task": task[:200],
                "iterations": iterations,
                "result_length": len(output),
                "messages_count": len(messages),
                "step_id": step_id,
            },
        )

    insights = await _generate_react_memory_insights(
        task=task,
        output=output,
        iterations=iterations,
        messages=messages,
        llm=llm,
    )
    should_store = bool(insights and insights.get("should_store"))
    reason = str(insights.get("reason", "") if insights else "")

    if tracer is not None and task_id:
        await trace_memory_generate_end(
            tracer,
            task_id,
            data={
                "insights_generated": insights is not None,
                "should_store": should_store,
                "reason": reason,
                "step_id": step_id,
            },
        )

    if not should_store:
        if tracer is not None and task_id:
            await trace_memory_store_end(
                tracer,
                task_id,
                data={
                    "storage_success": False,
                    "decision": "not_worth_storing",
                    "reason": reason or "No valuable memory insight generated.",
                    "step_id": step_id,
                },
            )
        return

    if tracer is not None and task_id:
        await trace_memory_store_start(
            tracer,
            task_id,
            data={
                "task": task[:200],
                "memory_category": "react_memory",
                "step_id": step_id,
            },
        )

    assert insights is not None
    memory_id = await asyncio.to_thread(
        store_react_task_memory,
        memory_store=memory_store,
        task=task,
        result={
            "success": bool(result.get("success", True)),
            "output": output,
            "iterations": iterations,
            "history": messages[-10:] if len(messages) > 10 else messages,
        },
        tool_usage_insights=str(insights.get("tool_usage_insights", "")),
        reasoning_strategy=str(insights.get("reasoning_strategy", "")),
        classification={
            "user_preferences": insights.get("user_preferences", ""),
            "core_insight": insights.get("core_insight", ""),
            "failure_patterns": insights.get("failure_patterns", ""),
            "success_patterns": insights.get("success_patterns", ""),
        },
    )

    if tracer is not None and task_id:
        await trace_memory_store_end(
            tracer,
            task_id,
            data={
                "storage_success": bool(memory_id),
                "memory_id": memory_id,
                "reason": reason,
                "step_id": step_id,
            },
        )


def build_skill_context(skill: dict[str, Any]) -> str:
    name = str(skill.get("name") or "Unnamed Skill")
    content = str(skill.get("content") or "").strip()
    if not content:
        parts = [
            str(skill.get("description") or "").strip(),
            str(skill.get("when_to_use") or "").strip(),
        ]
        content = "\n\n".join(part for part in parts if part)
    return f"## Available Skill: {name}\n\n{content}".strip()


def latest_user_text(context: Any) -> str:
    for message in reversed(getattr(context, "messages", []) or []):
        if getattr(message, "role", None) == "user":
            return str(getattr(message, "content", "") or "")
    task = context.metadata.get("task") if hasattr(context, "metadata") else None
    return str(task or "")


async def _generate_react_memory_insights(
    *,
    task: str,
    output: str,
    iterations: int,
    messages: list[dict[str, Any]],
    llm: Any,
) -> dict[str, Any] | None:
    analysis_summary = f"""MEMORY STORAGE EVALUATION:

TASK: {task}
ITERATIONS: {iterations}

RESULT PREVIEW:
{output[:300]}{"..." if len(output) > 300 else ""}

Evaluate if this execution contains UNIQUE, NON-OBVIOUS insights that would be valuable for FUTURE tasks.

STORE ONLY IF it contains at least one of:
- Clear user preferences or stable behavior patterns
- Non-obvious failures and fixes
- Reusable strategies that are not routine
- Domain-specific insights hard to obtain otherwise

DO NOT STORE routine task completion, generic tool usage, common facts, or obvious strategies.

Respond with JSON:
{{
  "should_store": true/false,
  "reason": "specific storage or rejection reason",
  "core_insight": "essential learning if stored",
  "user_preferences": "observable user preferences",
  "failure_patterns": "non-obvious failure modes and solutions",
  "success_patterns": "non-obvious success patterns",
  "tool_usage_insights": "non-obvious tool usage insight",
  "reasoning_strategy": "reusable reasoning strategy"
}}"""
    working_messages = list(messages)
    working_messages.append({"role": "user", "content": analysis_summary})
    try:
        run_llm_call = getattr(llm, "run_llm_call", None)
        response = (
            await run_llm_call(messages=working_messages)
            if callable(run_llm_call)
            else await llm.chat(messages=working_messages)
        )
        content = (
            response.get("content", str(response))
            if isinstance(response, dict)
            else str(response)
        )
        return _parse_json_object(content)
    except LLMCallInterrupted:
        raise
    except Exception:
        logger.exception("Failed to generate v2 ReAct memory insights")
        return None


def _parse_json_object(content: str) -> dict[str, Any] | None:
    text = content.strip()
    for candidate in _json_candidates(text):
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        return payload if isinstance(payload, dict) else None
    return None


def _json_candidates(text: str) -> list[str]:
    candidates = [text]
    fence_start = text.find("```")
    if fence_start >= 0:
        content_start = text.find("\n", fence_start + 3)
        fence_end = text.find("```", content_start + 1)
        if content_start >= 0 and fence_end > content_start:
            candidates.append(text[content_start + 1 : fence_end].strip())

    object_start = text.find("{")
    object_end = text.rfind("}")
    if object_start >= 0 and object_end > object_start:
        candidates.append(text[object_start : object_end + 1].strip())

    return candidates


def _runtime_attr(runtime: Any | None, name: str) -> Any | None:
    if runtime is None:
        return None
    return getattr(runtime, name, None)


def _skill_selection_attempt_key(
    task: str,
    allowed_skills: list[str] | None,
) -> str:
    payload = json.dumps(
        {"task": task, "allowed_skills": sorted(allowed_skills or [])},
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _build_memory_context(
    existing_context: Any,
    query: str,
    memories: list[dict[str, Any]],
) -> str:
    if not memories:
        return str(existing_context or "")
    enhanced = enhance_goal_with_memory(query, memories)
    context_text = enhanced
    if query and enhanced.startswith(query):
        context_text = enhanced[len(query) :].lstrip()
    context_text = context_text.strip()
    if not context_text:
        context_text = enhanced
    if existing_context:
        existing = str(existing_context)
        if context_text in existing:
            return existing
        return f"{existing}\n\n{context_text}"
    return context_text


def _lookup_relevant_memories_with_context(
    memory_store: Any,
    query: str,
    category: str,
    include_general: bool,
    limit: int,
    similarity_threshold: float | None,
    user_id: Any | None,
) -> list[dict[str, Any]]:
    if user_id is not None:
        try:
            token = current_user_id.set(user_id)
            try:
                return lookup_relevant_memories(
                    memory_store,
                    query,
                    category,
                    include_general=include_general,
                    limit=limit,
                    similarity_threshold=similarity_threshold,
                )
            finally:
                current_user_id.reset(token)
        except Exception:
            logger.exception("Failed to retrieve memories with user context")
            return []

    return lookup_relevant_memories(
        memory_store,
        query,
        category,
        include_general=include_general,
        limit=limit,
        similarity_threshold=similarity_threshold,
    )


def _current_user_id() -> Any | None:
    return current_user_id.get()


def lookup_relevant_memories(
    memory_store: Any | None,
    query: str,
    category: str,
    *,
    include_general: bool = True,
    limit: int = 5,
    similarity_threshold: float | None = None,
) -> list[dict[str, Any]]:
    if memory_store is None:
        return []

    filters: dict[str, Any] = {}
    if category:
        filters["category"] = category
    search = getattr(memory_store, "search", None)
    if not callable(search):
        return []

    memories = search(
        query=query,
        k=limit,
        filters=filters or None,
        similarity_threshold=similarity_threshold,
    )
    if include_general and category != "general":
        memories.extend(
            search(
                query=query,
                k=limit,
                filters={"category": "general"},
                similarity_threshold=similarity_threshold,
            )
        )
    return [_memory_note_to_dict(memory) for memory in memories[:limit]]


def enhance_goal_with_memory(query: str, memories: list[dict[str, Any]]) -> str:
    if not memories:
        return query
    memory_lines = [
        f"- {str(memory.get('content') or '').strip()}"
        for memory in memories
        if str(memory.get("content") or "").strip()
    ]
    if not memory_lines:
        return query
    return f"{query}\n\nRelevant memory:\n" + "\n".join(memory_lines)


def store_react_task_memory(
    *,
    memory_store: Any | None,
    task: str,
    result: dict[str, Any],
    tool_usage_insights: str,
    reasoning_strategy: str,
    classification: dict[str, Any],
) -> str | None:
    if memory_store is None:
        return None

    content_parts = [
        f"Task: {task}",
        f"Result: {str(result.get('output') or '')}",
        f"Tool usage: {tool_usage_insights}",
        f"Reasoning strategy: {reasoning_strategy}",
        f"Core insight: {classification.get('core_insight') or ''}",
    ]
    note = MemoryNote(
        content="\n".join(part for part in content_parts if part.strip()),
        category="react_memory",
        metadata={
            "task": task,
            "success": bool(result.get("success", True)),
            "classification": classification,
        },
    )
    response = memory_store.add(note)
    return (
        getattr(response, "memory_id", None)
        if getattr(response, "success", False)
        else None
    )


def _memory_note_to_dict(memory: Any) -> dict[str, Any]:
    if hasattr(memory, "model_dump"):
        return cast(dict[str, Any], memory.model_dump())
    if isinstance(memory, dict):
        return memory
    return {
        "id": getattr(memory, "id", None),
        "content": getattr(memory, "content", ""),
        "category": getattr(memory, "category", "general"),
        "metadata": getattr(memory, "metadata", {}),
    }
