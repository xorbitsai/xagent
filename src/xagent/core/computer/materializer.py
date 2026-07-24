from __future__ import annotations

import asyncio
import base64
from pathlib import Path
from typing import Any, Protocol

from ..context_ref import (
    CONTEXT_REFS_KEY,
    ContextReference,
    normalize_context_references,
)


class ContextReferenceResolver(Protocol):
    async def resolve_image(self, reference: ContextReference) -> str: ...


class WorkspaceContextReferenceResolver:
    """Resolves registered image FileRefs to ephemeral data URLs."""

    def __init__(self, workspace: Any) -> None:
        self.workspace = workspace

    async def resolve_image(self, reference: ContextReference) -> str:
        resolve_file_id = getattr(self.workspace, "resolve_file_id", None)
        if not callable(resolve_file_id):
            raise TypeError("workspace must expose resolve_file_id")
        path = resolve_file_id(reference.file_id)
        if path is None:
            raise FileNotFoundError(
                f"unable to resolve context FileRef {reference.file_id!r}"
            )
        mime_type = str(reference.safe_file_ref.get("mime_type") or "image/png")
        image_bytes = await asyncio.to_thread(Path(path).read_bytes)
        encoded = base64.b64encode(image_bytes).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"


def llm_supports_vision(llm: Any) -> bool:
    has_ability = getattr(llm, "has_ability", None)
    if callable(has_ability):
        try:
            return bool(has_ability("vision"))
        except (TypeError, ValueError):
            pass
    abilities = getattr(llm, "abilities", ())
    return "vision" in abilities if abilities is not None else False


async def materialize_messages(
    *,
    llm: Any,
    messages: list[dict[str, Any]],
    resolver: ContextReferenceResolver | None,
) -> list[dict[str, Any]]:
    """Materialize durable context refs without mutating persisted messages."""
    supports_vision = llm_supports_vision(llm)
    result: list[dict[str, Any]] = []
    deferred_user_messages: list[dict[str, Any]] = []

    async def flush_deferred() -> None:
        if deferred_user_messages:
            result.extend(deferred_user_messages)
            deferred_user_messages.clear()

    for index, message in enumerate(messages):
        copied = dict(message)
        raw_refs = copied.pop(CONTEXT_REFS_KEY, None)
        refs = normalize_context_references(raw_refs)
        if not refs:
            if copied.get("role") != "tool":
                await flush_deferred()
            result.append(copied)
            next_role = (
                messages[index + 1].get("role") if index + 1 < len(messages) else None
            )
            if copied.get("role") == "tool" and next_role != "tool":
                await flush_deferred()
            continue

        if not supports_vision:
            fallback = "\n".join(ref.compact_text() for ref in refs)
            content = str(copied.get("content") or "")
            copied["content"] = f"{content}\n{fallback}".strip()
            if copied.get("role") != "tool":
                await flush_deferred()
            result.append(copied)
            next_role = (
                messages[index + 1].get("role") if index + 1 < len(messages) else None
            )
            if copied.get("role") == "tool" and next_role != "tool":
                await flush_deferred()
            continue

        if resolver is None:
            raise RuntimeError(
                "vision context references require a configured resolver"
            )

        content_parts: list[dict[str, Any]] = []
        text = str(copied.get("content") or "")
        if copied.get("role") == "user" and text:
            content_parts.append({"type": "text", "text": text})
        elif copied.get("role") != "user":
            content_parts.append(
                {
                    "type": "text",
                    "text": "Image context for the preceding message.",
                }
            )
        for ref in refs:
            detail = "high" if ref.detail.value == "original" else ref.detail.value
            content_parts.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": await resolver.resolve_image(ref),
                        "detail": detail,
                    },
                }
            )

        if copied.get("role") == "user":
            copied["content"] = content_parts
            await flush_deferred()
            result.append(copied)
        else:
            result.append(copied)
            deferred_user_messages.append(
                {
                    "role": "user",
                    "content": content_parts,
                }
            )

        next_role = (
            messages[index + 1].get("role") if index + 1 < len(messages) else None
        )
        if next_role != "tool":
            await flush_deferred()

    await flush_deferred()
    return result


async def materialize_llm_kwargs(
    *,
    llm: Any,
    kwargs: dict[str, Any],
    resolver: ContextReferenceResolver | None,
) -> dict[str, Any]:
    messages = kwargs.get("messages")
    if not isinstance(messages, list):
        return kwargs
    materialized = dict(kwargs)
    materialized["messages"] = await materialize_messages(
        llm=llm,
        messages=messages,
        resolver=resolver,
    )
    return materialized
