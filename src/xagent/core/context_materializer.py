from __future__ import annotations

import asyncio
import base64
import logging
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Protocol

from .context_ref import (
    CONTEXT_REFS_KEY,
    ContextReference,
    normalize_context_references,
)

logger = logging.getLogger(__name__)

_IMAGE_CONTEXT_PLACEHOLDER = "Image context for the preceding message."


class ContextReferenceResolutionError(RuntimeError):
    """A durable context reference could not be safely materialized."""


class ContextReferenceResolver(Protocol):
    async def resolve_image(self, reference: ContextReference) -> str: ...


class WorkspaceContextReferenceResolver:
    """Resolve registered image FileRefs to short-lived provider data URLs."""

    def __init__(
        self,
        workspace: Any,
        *,
        cache_size: int = 8,
        cache_ttl_seconds: float = 300,
        max_image_bytes: int = 20 * 1024 * 1024,
    ) -> None:
        self.workspace = workspace
        self.cache_size = max(1, cache_size)
        self.cache_ttl_seconds = max(0, cache_ttl_seconds)
        self.max_image_bytes = max(1, max_image_bytes)
        self._cache: OrderedDict[str, tuple[float, str]] = OrderedDict()

    async def resolve_image(self, reference: ContextReference) -> str:
        cache_key = self._cache_key(reference)
        cached = self._cache.get(cache_key)
        now = time.monotonic()
        if cached is not None:
            created_at, data_url = cached
            if now - created_at <= self.cache_ttl_seconds:
                self._cache.move_to_end(cache_key)
                return data_url
            self._cache.pop(cache_key, None)

        resolve_file_id = getattr(self.workspace, "resolve_file_id", None)
        if not callable(resolve_file_id):
            raise ContextReferenceResolutionError(
                "workspace must expose resolve_file_id"
            )
        path = resolve_file_id(reference.file_id)
        if path is None:
            raise FileNotFoundError(
                f"unable to resolve context FileRef {reference.file_id!r}"
            )

        resolved_path = Path(path)
        file_size = await asyncio.to_thread(lambda: resolved_path.stat().st_size)
        if file_size > self.max_image_bytes:
            raise ContextReferenceResolutionError(
                f"context image exceeds {self.max_image_bytes} bytes"
            )
        mime_type = str(reference.safe_file_ref.get("mime_type") or "image/png")
        if not mime_type.startswith("image/"):
            raise ContextReferenceResolutionError(
                f"unsupported context image MIME type: {mime_type}"
            )

        image_bytes = await asyncio.to_thread(resolved_path.read_bytes)
        if len(image_bytes) > self.max_image_bytes:
            raise ContextReferenceResolutionError(
                f"context image exceeds {self.max_image_bytes} bytes"
            )
        encoded = base64.b64encode(image_bytes).decode("ascii")
        data_url = f"data:{mime_type};base64,{encoded}"
        self._cache[cache_key] = (now, data_url)
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return data_url

    def clear(self) -> None:
        self._cache.clear()

    @staticmethod
    def _cache_key(reference: ContextReference) -> str:
        digest = reference.metadata.get("sha256")
        if isinstance(digest, str) and digest:
            return f"{reference.file_id}:{digest}"
        return reference.file_id


def llm_supports_vision(llm: Any) -> bool:
    has_ability = getattr(llm, "has_ability", None)
    if callable(has_ability):
        try:
            return bool(has_ability("vision"))
        except (TypeError, ValueError):
            pass
    abilities = getattr(llm, "abilities", ())
    return "vision" in abilities if abilities is not None else False


def _append_text(content: Any, text: str) -> Any:
    if not text:
        return content
    if isinstance(content, list):
        copied = [dict(part) if isinstance(part, dict) else part for part in content]
        copied.append({"type": "text", "text": text})
        return copied
    current = str(content or "")
    return f"{current}\n{text}".strip()


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

    def flush_deferred() -> None:
        if deferred_user_messages:
            result.extend(deferred_user_messages)
            deferred_user_messages.clear()

    for index, message in enumerate(messages):
        copied = dict(message)
        refs = normalize_context_references(copied.pop(CONTEXT_REFS_KEY, None))
        role = copied.get("role")

        if not refs or not supports_vision or resolver is None:
            if refs:
                fallback = "\n".join(ref.compact_text() for ref in refs)
                copied["content"] = _append_text(copied.get("content"), fallback)
            if role != "tool":
                flush_deferred()
            result.append(copied)
            next_role = (
                messages[index + 1].get("role") if index + 1 < len(messages) else None
            )
            if role == "tool" and next_role != "tool":
                flush_deferred()
            continue

        text = copied.get("content")
        content_parts: list[dict[str, Any]] = []
        if role == "user":
            if isinstance(text, list):
                for part in text:
                    if isinstance(part, dict):
                        content_parts.append(dict(part))
                    elif str(part):
                        content_parts.append({"type": "text", "text": str(part)})
            elif str(text or ""):
                content_parts.append({"type": "text", "text": str(text)})
        else:
            content_parts.append({"type": "text", "text": _IMAGE_CONTEXT_PLACEHOLDER})

        unresolved: list[ContextReference] = []
        for reference in refs:
            try:
                url = await resolver.resolve_image(reference)
            except (
                ContextReferenceResolutionError,
                FileNotFoundError,
                OSError,
            ):
                logger.debug(
                    "Falling back to text for unresolved context ref %s",
                    reference.file_id,
                    exc_info=True,
                )
                unresolved.append(reference)
                continue

            image_url: dict[str, Any] = {"url": url}
            detail = (
                "high"
                if reference.detail.value == "original"
                else reference.detail.value
            )
            if detail != "auto":
                image_url["detail"] = detail
            content_parts.append(
                {
                    "type": "image_url",
                    "image_url": image_url,
                }
            )

        if unresolved:
            fallback = "\n".join(ref.compact_text() for ref in unresolved)
            content_parts.append({"type": "text", "text": fallback})

        if not any(part.get("type") == "image_url" for part in content_parts):
            copied["content"] = _append_text(
                copied.get("content"),
                "\n".join(ref.compact_text() for ref in unresolved),
            )
            if role != "tool":
                flush_deferred()
            result.append(copied)
        elif role == "user":
            flush_deferred()
            copied["content"] = content_parts
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
            flush_deferred()

    flush_deferred()
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
