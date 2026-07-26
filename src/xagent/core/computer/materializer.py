from __future__ import annotations

import asyncio
import base64
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Any, Protocol

from ...config import get_computer_max_live_frames
from ..context_ref import (
    CONTEXT_REFS_KEY,
    ContextReference,
    ContextReferencePurpose,
    normalize_context_references,
)

logger = logging.getLogger(__name__)

#: Text that introduces images attached to a non-user message, since providers
#: only accept image parts on a user turn.
_IMAGE_CONTEXT_PLACEHOLDER = "Image context for the preceding message."


class ContextReferenceResolver(Protocol):
    async def resolve_image(self, reference: ContextReference) -> str: ...


class WorkspaceContextReferenceResolver:
    """Resolves registered image FileRefs to ephemeral data URLs.

    Encoded images are cached per reference: the same screenshot would
    otherwise be re-read from disk and re-encoded on every model call for as
    long as it stays in the conversation.
    """

    def __init__(self, workspace: Any, *, cache_size: int = 8) -> None:
        self.workspace = workspace
        self.cache_size = max(1, cache_size)
        self._cache: OrderedDict[str, str] = OrderedDict()

    async def resolve_image(self, reference: ContextReference) -> str:
        cache_key = self._cache_key(reference)
        cached = self._cache.get(cache_key)
        if cached is not None:
            self._cache.move_to_end(cache_key)
            return cached

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
        data_url = f"data:{mime_type};base64,{encoded}"
        self._cache[cache_key] = data_url
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return data_url

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


def _live_observation_refs(
    messages: list[dict[str, Any]],
    max_live_frames: int,
) -> set[str]:
    """Identity keys of the observation frames that stay images this call.

    Superseded frames describe a page that no longer exists, and every frame
    kept alive costs a full image re-upload on every subsequent call. Only the
    newest few earn that, so older ones fall back to their text description.
    """
    observation_keys: list[str] = []
    for message in messages:
        for reference in normalize_context_references(
            message.get(CONTEXT_REFS_KEY, ())
        ):
            if reference.purpose is ContextReferencePurpose.OBSERVATION:
                observation_keys.append(reference.identity_key())
    if max_live_frames <= 0:
        return set()
    return set(observation_keys[-max_live_frames:])


async def materialize_messages(
    *,
    llm: Any,
    messages: list[dict[str, Any]],
    resolver: ContextReferenceResolver | None,
    max_live_frames: int | None = None,
) -> list[dict[str, Any]]:
    """Materialize durable context refs without mutating persisted messages."""
    supports_vision = llm_supports_vision(llm)
    live_refs = _live_observation_refs(
        messages,
        get_computer_max_live_frames() if max_live_frames is None else max_live_frames,
    )
    result: list[dict[str, Any]] = []
    deferred_user_messages: list[dict[str, Any]] = []

    async def flush_deferred() -> None:
        if deferred_user_messages:
            result.extend(deferred_user_messages)
            deferred_user_messages.clear()

    for index, message in enumerate(messages):
        copied = dict(message)
        raw_refs = copied.pop(CONTEXT_REFS_KEY, None)
        all_refs = normalize_context_references(raw_refs)
        refs = tuple(
            ref
            for ref in all_refs
            if ref.purpose is not ContextReferencePurpose.OBSERVATION
            or ref.identity_key() in live_refs
        )
        stale_refs = tuple(ref for ref in all_refs if ref not in refs)
        if stale_refs:
            # Keep the textual trace of a superseded frame so the transcript
            # still explains what the model was looking at.
            content = str(copied.get("content") or "")
            summary = "\n".join(ref.compact_text() for ref in stale_refs)
            copied["content"] = f"{content}\n{summary}".strip()
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
            content_parts.append({"type": "text", "text": _IMAGE_CONTEXT_PLACEHOLDER})
        unresolved: list[ContextReference] = []
        for ref in refs:
            detail = "high" if ref.detail.value == "original" else ref.detail.value
            try:
                url = await resolver.resolve_image(ref)
            except (FileNotFoundError, OSError):
                # Observation frames are pruned by retention and are not
                # reachable from another process. A missing image must degrade
                # to text, never fail the model call.
                logger.debug(
                    "Falling back to text for unresolved context ref %s",
                    ref.file_id,
                    exc_info=True,
                )
                unresolved.append(ref)
                continue
            image_url = {"url": url}
            if detail != "auto":
                # ``auto`` is the provider default. Omitting it preserves the
                # same semantics while avoiding OpenAI-compatible endpoints
                # that reject the explicitly supplied value.
                image_url["detail"] = detail
            content_parts.append(
                {
                    "type": "image_url",
                    "image_url": image_url,
                }
            )
        if unresolved:
            fallback = "\n".join(ref.compact_text() for ref in unresolved)
            if content_parts and content_parts[0].get("type") == "text":
                content_parts[0]["text"] = (
                    f"{content_parts[0]['text']}\n{fallback}".strip()
                )
            else:
                content_parts.insert(0, {"type": "text", "text": fallback})
        if not any(part.get("type") == "image_url" for part in content_parts):
            # Nothing left to attach: keep the message as plain text rather
            # than emitting a content-part list with no image in it.
            original = str(copied.get("content") or "")
            extra = [
                str(part.get("text") or "")
                for part in content_parts
                if part.get("type") == "text"
                and str(part.get("text") or "")
                not in {"", original, _IMAGE_CONTEXT_PLACEHOLDER}
            ]
            copied["content"] = "\n".join([original, *extra]).strip()
            if copied.get("role") != "tool":
                await flush_deferred()
            result.append(copied)
            next_role = (
                messages[index + 1].get("role") if index + 1 < len(messages) else None
            )
            if copied.get("role") == "tool" and next_role != "tool":
                await flush_deferred()
            continue

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
