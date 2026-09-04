from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Literal, cast

from ...agent.trace import (
    trace_memory_retrieve_end,
    trace_memory_retrieve_start,
)
from ...user_context import current_user_id

logger = logging.getLogger(__name__)

MEMORY_CONTEXT_METADATA_KEY = "retrieved_memory_context"
RETRIEVED_MEMORIES_METADATA_KEY = "retrieved_memories"
SELECTED_SKILL_METADATA_KEY = "selected_skill"
SKILL_CONTEXT_METADATA_KEY = "selected_skill_context"
# True only when generate_image is registered and edit_image is not; a deployment
# with no image tools at all leaves it False.
IMAGE_EDIT_UNAVAILABLE_METADATA_KEY = "image_edit_unavailable"

DisplayMessageState = Literal["missing", "empty", "text"]
TOP_LEVEL_USER_REQUEST_METADATA_KEY = "_xagent_top_level_user_request"


@dataclass(frozen=True)
class TopLevelUserRequest:
    """Canonical execution and user-authored text for an independent request."""

    execution_text: str
    language_text: str
    display_state: DisplayMessageState
    has_pending_response: bool = False


@dataclass(frozen=True)
class PendingUserResponse:
    """Allowlisted context for one answer to a pending agent message."""

    answer: str
    question: str
    message_type: str


def pending_user_response(message: Any) -> PendingUserResponse | None:
    """Extract only language-relevant fields from a marked user message."""
    if getattr(message, "role", None) != "user":
        return None
    metadata = getattr(message, "metadata", None)
    marker = (
        metadata.get("response_to_waiting_for_user")
        if isinstance(metadata, dict)
        else None
    )
    if not isinstance(marker, dict):
        return None
    question = marker.get("question")
    if not isinstance(question, str) or not question.strip():
        return None
    raw_message_type = marker.get("message_type", "question")
    message_type = (
        raw_message_type.strip()
        if isinstance(raw_message_type, str) and raw_message_type.strip()
        else "question"
    )
    answer = getattr(message, "content", "")
    if not isinstance(answer, str):
        return None
    return PendingUserResponse(answer, question, message_type)


def _stored_top_level_user_request(context: Any) -> TopLevelUserRequest | None:
    metadata = getattr(context, "metadata", None)
    if not isinstance(metadata, dict):
        return None
    payload = metadata.get(TOP_LEVEL_USER_REQUEST_METADATA_KEY)
    if not isinstance(payload, dict):
        return None
    execution_text = payload.get("execution_text")
    language_text = payload.get("language_text")
    display_state = payload.get("display_state")
    if (
        not isinstance(execution_text, str)
        or not isinstance(language_text, str)
        or display_state not in {"missing", "empty", "text"}
    ):
        return None
    return TopLevelUserRequest(
        execution_text=execution_text,
        language_text=language_text,
        display_state=display_state,
    )


def _persist_top_level_user_request(context: Any, request: TopLevelUserRequest) -> None:
    metadata = getattr(context, "metadata", None)
    if not isinstance(metadata, dict):
        return
    metadata[TOP_LEVEL_USER_REQUEST_METADATA_KEY] = {
        "execution_text": request.execution_text,
        "language_text": request.language_text,
        "display_state": request.display_state,
    }


def hydrate_top_level_user_request(context: Any, root_context: Any) -> None:
    """Backfill a legacy child snapshot from its canonical root request."""
    if _stored_top_level_user_request(context) is not None:
        return
    root_request = _stored_top_level_user_request(root_context)
    if root_request is None:
        root_request = top_level_user_request(root_context)
    _persist_top_level_user_request(context, root_request)


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


def display_message_override(metadata: Any) -> str | None:
    """Return a supported display-message override, including an empty one.

    Missing keys and non-string values in directly constructed or restored
    contexts keep the execution-content fallback. The runner normalizes a
    present non-string value to an authoritative empty string at ingress.
    """
    if not isinstance(metadata, dict) or "display_message" not in metadata:
        return None
    display = metadata["display_message"]
    if not isinstance(display, str):
        return None
    return display.strip()


def top_level_user_request(
    context: Any,
    *,
    user_message_limit: int | None = None,
) -> TopLevelUserRequest:
    """Return and persist the latest independent top-level user request.

    ``user_message_limit`` freezes selection to a checkpointed prefix when a
    later waiting response has already been appended to the root context.
    """
    has_pending_response = False
    messages = list(getattr(context, "messages", []) or [])
    if user_message_limit is not None:
        user_messages = [
            message for message in messages if getattr(message, "role", None) == "user"
        ]
        messages = user_messages[: max(0, user_message_limit)]
    for message in reversed(messages):
        if getattr(message, "role", None) != "user" or getattr(
            message, "hidden", False
        ):
            continue
        metadata = getattr(message, "metadata", None)
        metadata = metadata if isinstance(metadata, dict) else {}
        if metadata.get("response_to_waiting_for_user"):
            has_pending_response = True
            continue
        if metadata.get("dag_step_id"):
            continue

        execution_text = str(getattr(message, "content", "") or "").strip()
        display_text = display_message_override(metadata)
        if display_text is None:
            if not execution_text:
                continue
            request = TopLevelUserRequest(
                execution_text=execution_text,
                language_text=execution_text,
                display_state="missing",
                has_pending_response=has_pending_response,
            )
        else:
            request = TopLevelUserRequest(
                execution_text=execution_text,
                language_text=display_text,
                display_state="text" if display_text else "empty",
                has_pending_response=has_pending_response,
            )
        _persist_top_level_user_request(context, request)
        return request

    stored = _stored_top_level_user_request(context)
    if stored is not None:
        return TopLevelUserRequest(
            execution_text=stored.execution_text,
            language_text=stored.language_text,
            display_state=stored.display_state,
            has_pending_response=has_pending_response,
        )

    metadata = getattr(context, "metadata", None)
    task = metadata.get("task") if isinstance(metadata, dict) else None
    task_text = str(task or "").strip()
    request = TopLevelUserRequest(
        execution_text=task_text,
        language_text=task_text,
        display_state="missing",
        has_pending_response=has_pending_response,
    )
    _persist_top_level_user_request(context, request)
    return request


def latest_user_text(context: Any, *, prefer_display: bool = False) -> str:
    """Return the latest user turn's text.

    ``prefer_display`` returns what the user actually typed instead of the
    runtime-augmented execution prompt; language anchors must use it, work
    anchors must not.
    """
    for message in reversed(getattr(context, "messages", []) or []):
        if getattr(message, "role", None) == "user":
            if prefer_display:
                metadata = getattr(message, "metadata", None)
                display = (
                    metadata.get("display_message")
                    if isinstance(metadata, dict)
                    else None
                )
                if isinstance(display, str) and display.strip():
                    return display
            return str(getattr(message, "content", "") or "")
    task = context.metadata.get("task") if hasattr(context, "metadata") else None
    return str(task or "")


def _runtime_attr(runtime: Any | None, name: str) -> Any | None:
    if runtime is None:
        return None
    return getattr(runtime, name, None)


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
    try:
        if user_id is not None:
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

        return lookup_relevant_memories(
            memory_store,
            query,
            category,
            include_general=include_general,
            limit=limit,
            similarity_threshold=similarity_threshold,
        )
    except Exception:
        logger.exception(
            "Failed to retrieve memories%s",
            " with user context" if user_id is not None else "",
        )
        return []


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
