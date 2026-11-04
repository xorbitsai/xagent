from __future__ import annotations

import asyncio
import base64
import logging
import time
from collections import OrderedDict
from io import BytesIO
from pathlib import Path
from typing import Any, NamedTuple, Protocol

from PIL import Image

from .context_ref import (
    CONTEXT_REFS_KEY,
    ContextReference,
    normalize_context_references,
)

logger = logging.getLogger(__name__)

_IMAGE_CONTEXT_PLACEHOLDER = "Image context for the preceding message."
_DEFAULT_CACHE_ENTRIES = 32
_DEFAULT_CACHE_BYTES = 32 * 1024 * 1024
_DEFAULT_CONTEXT_REF_TOKEN_BUDGET = 8_192
_MAX_CONTEXT_IMAGE_BYTES_PER_REQUEST = 32 * 1024 * 1024
_CONTEXT_REF_TOKEN_BUDGET_RATIO = 0.25
_SUPPORTED_IMAGE_MIME_TYPES = {
    "GIF": "image/gif",
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "WEBP": "image/webp",
}


class ContextReferenceResolutionError(RuntimeError):
    """A durable context reference could not be safely materialized."""


class _FileGenerationChanged(RuntimeError):
    """The resolved file changed while it was being materialized."""


class _FileGeneration(NamedTuple):
    path: str
    device: int
    inode: int
    size: int
    modified_ns: int


class _CacheEntry(NamedTuple):
    created_at: float
    data_url: str
    encoded_bytes: int


class _ReferencePosition(NamedTuple):
    message_index: int
    reference_index: int
    reference: ContextReference


class ContextReferenceResolver(Protocol):
    async def resolve_image(self, reference: ContextReference) -> str: ...


class WorkspaceContextReferenceResolver:
    """Resolve registered image FileRefs to short-lived provider data URLs."""

    def __init__(
        self,
        workspace: Any,
        *,
        cache_size: int = _DEFAULT_CACHE_ENTRIES,
        cache_ttl_seconds: float = 300,
        max_image_bytes: int = 20 * 1024 * 1024,
        max_cache_bytes: int = _DEFAULT_CACHE_BYTES,
    ) -> None:
        self.workspace = workspace
        self.cache_size = max(1, cache_size)
        self.cache_ttl_seconds = max(0, cache_ttl_seconds)
        self.max_image_bytes = max(1, max_image_bytes)
        self.max_cache_bytes = max(0, max_cache_bytes)
        self._cache: OrderedDict[str, _CacheEntry] = OrderedDict()
        self._cache_bytes = 0

    async def resolve_image(self, reference: ContextReference) -> str:
        declared_mime_type = str(reference.safe_file_ref.get("mime_type") or "").lower()
        declared_mime_type = declared_mime_type.split(";", 1)[0].strip()
        if declared_mime_type == "image/jpg":
            declared_mime_type = "image/jpeg"
        if declared_mime_type and declared_mime_type not in (
            _SUPPORTED_IMAGE_MIME_TYPES.values()
        ):
            raise ContextReferenceResolutionError(
                f"unsupported context image MIME type: {declared_mime_type}"
            )

        for attempt in range(2):
            resolved_path, generation = await asyncio.to_thread(
                self._resolve_generation,
                reference.file_id,
            )
            if generation.size > self.max_image_bytes:
                raise ContextReferenceResolutionError(
                    f"context image exceeds {self.max_image_bytes} bytes"
                )

            cache_key = self._cache_key(
                reference,
                generation,
                declared_mime_type,
            )
            now = time.monotonic()
            cached = self._cache.get(cache_key)
            if cached is not None:
                if now - cached.created_at <= self.cache_ttl_seconds:
                    self._cache.move_to_end(cache_key)
                    return cached.data_url
                self._cache.pop(cache_key, None)
                self._cache_bytes -= cached.encoded_bytes

            try:
                image_bytes = await asyncio.to_thread(
                    self._read_generation,
                    resolved_path,
                    generation,
                )
            except _FileGenerationChanged as exc:
                if attempt == 0:
                    continue
                raise ContextReferenceResolutionError(
                    "context image changed while it was being read"
                ) from exc

            if len(image_bytes) > self.max_image_bytes:
                raise ContextReferenceResolutionError(
                    f"context image exceeds {self.max_image_bytes} bytes"
                )
            mime_type = await asyncio.to_thread(
                self._validated_image_mime_type,
                image_bytes,
                declared_mime_type,
            )
            encoded = base64.b64encode(image_bytes).decode("ascii")
            data_url = f"data:{mime_type};base64,{encoded}"
            encoded_bytes = len(data_url)
            if encoded_bytes <= self.max_cache_bytes:
                replaced = self._cache.pop(cache_key, None)
                if replaced is not None:
                    self._cache_bytes -= replaced.encoded_bytes
                while self._cache and (
                    len(self._cache) >= self.cache_size
                    or self._cache_bytes + encoded_bytes > self.max_cache_bytes
                ):
                    _, evicted = self._cache.popitem(last=False)
                    self._cache_bytes -= evicted.encoded_bytes
                self._cache[cache_key] = _CacheEntry(
                    time.monotonic(),
                    data_url,
                    encoded_bytes,
                )
                self._cache_bytes += encoded_bytes
            return data_url

        raise ContextReferenceResolutionError("unable to read a stable context image")

    def _resolve_generation(
        self,
        file_id: str,
    ) -> tuple[Path, _FileGeneration]:
        resolve_file_id = getattr(self.workspace, "resolve_file_id_detached", None)
        if not callable(resolve_file_id):
            resolve_file_id = getattr(self.workspace, "resolve_file_id", None)
        if not callable(resolve_file_id):
            raise ContextReferenceResolutionError(
                "workspace must expose resolve_file_id"
            )
        path = resolve_file_id(file_id)
        if path is None:
            raise FileNotFoundError(f"unable to resolve context FileRef {file_id!r}")

        resolved_path = Path(path).resolve(strict=True)
        return resolved_path, self._generation(resolved_path)

    @staticmethod
    def _generation(path: Path) -> _FileGeneration:
        stat = path.stat()
        return _FileGeneration(
            path=str(path),
            device=stat.st_dev,
            inode=stat.st_ino,
            size=stat.st_size,
            modified_ns=stat.st_mtime_ns,
        )

    def _read_generation(
        self,
        path: Path,
        expected: _FileGeneration,
    ) -> bytes:
        if self._generation(path) != expected:
            raise _FileGenerationChanged
        image_bytes = path.read_bytes()
        if self._generation(path) != expected:
            raise _FileGenerationChanged
        return image_bytes

    @staticmethod
    def _validated_image_mime_type(
        image_bytes: bytes,
        declared_mime_type: str,
    ) -> str:
        try:
            with Image.open(BytesIO(image_bytes)) as image:
                image_format = str(image.format or "").upper()
                image.verify()
        except (
            Image.DecompressionBombError,
            OSError,
            SyntaxError,
            ValueError,
        ) as exc:
            raise ContextReferenceResolutionError(
                "context FileRef does not contain a valid supported image"
            ) from exc

        mime_type = _SUPPORTED_IMAGE_MIME_TYPES.get(image_format)
        if mime_type is None:
            raise ContextReferenceResolutionError(
                f"unsupported context image format: {image_format or 'unknown'}"
            )
        if declared_mime_type and declared_mime_type != mime_type:
            raise ContextReferenceResolutionError(
                "context image MIME type does not match its encoded format"
            )
        return mime_type

    @staticmethod
    def _cache_key(
        reference: ContextReference,
        generation: _FileGeneration,
        declared_mime_type: str,
    ) -> str:
        generation_key = ":".join(str(part) for part in generation)
        return f"{reference.file_id}:{declared_mime_type}:{generation_key}"


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


def _context_ref_fallback(
    reference: ContextReference,
    *,
    reason: str | None = None,
) -> str:
    compact = reference.compact_text()
    return f"{compact} {reason}" if reason else compact


def _context_ref_token_budget(llm: Any) -> int:
    context_window = getattr(llm, "context_window", None)
    if isinstance(context_window, int) and context_window > 0:
        return max(1, int(context_window * _CONTEXT_REF_TOKEN_BUDGET_RATIO))
    return _DEFAULT_CONTEXT_REF_TOKEN_BUDGET


def _prioritized_reference_positions(
    messages: list[dict[str, Any]],
    references_by_message: list[tuple[ContextReference, ...]],
) -> tuple[
    list[_ReferencePosition],
    list[_ReferencePosition],
    set[tuple[int, int]],
]:
    """Return unique current-turn refs first and older refs newest-first."""

    current_turn_start: int | None = None
    for message_index in range(len(messages) - 1, -1, -1):
        if messages[message_index].get("role") == "user":
            current_turn_start = message_index
            break

    current: list[_ReferencePosition] = []
    historical: list[_ReferencePosition] = []
    duplicates: set[tuple[int, int]] = set()
    seen_file_ids: set[str] = set()
    for message_index in range(len(messages) - 1, -1, -1):
        references = references_by_message[message_index]
        for reference_index in range(len(references) - 1, -1, -1):
            reference = references[reference_index]
            if reference.file_id in seen_file_ids:
                duplicates.add((message_index, reference_index))
                continue
            seen_file_ids.add(reference.file_id)
            position = _ReferencePosition(
                message_index,
                reference_index,
                reference,
            )
            if current_turn_start is not None and message_index >= current_turn_start:
                current.append(position)
            else:
                historical.append(position)
    return current, historical, duplicates


async def materialize_messages(
    *,
    llm: Any,
    messages: list[dict[str, Any]],
    resolver: ContextReferenceResolver | None,
) -> list[dict[str, Any]]:
    """Materialize durable context refs without mutating persisted messages."""

    supports_vision = llm_supports_vision(llm)
    references_by_message = [
        normalize_context_references(message.get(CONTEXT_REFS_KEY))
        for message in messages
    ]
    resolved_images: dict[tuple[int, int], str] = {}
    fallback_reasons: dict[tuple[int, int], str] = {}
    duplicate_positions: set[tuple[int, int]] = set()
    if supports_vision and resolver is not None:
        (
            current_positions,
            historical_positions,
            duplicate_positions,
        ) = _prioritized_reference_positions(messages, references_by_message)
        for message_index, reference_index in duplicate_positions:
            fallback_reasons[(message_index, reference_index)] = (
                "The same image is included with a more recent message."
            )
        token_budget = _context_ref_token_budget(llm)
        materialized_tokens = 0
        materialized_image_bytes = 0

        async def resolve(position: _ReferencePosition) -> str | None:
            try:
                return await resolver.resolve_image(position.reference)
            except (
                ContextReferenceResolutionError,
                FileNotFoundError,
                OSError,
            ):
                fallback_reasons[(position.message_index, position.reference_index)] = (
                    "The image could not be loaded."
                )
                logger.debug(
                    "Falling back to text for unresolved context ref %s",
                    position.reference.file_id,
                    exc_info=True,
                )
                return None

        for position in (*current_positions, *historical_positions):
            estimated_tokens = position.reference.estimated_tokens()
            if materialized_tokens + estimated_tokens > token_budget:
                fallback_reasons[(position.message_index, position.reference_index)] = (
                    "The image was omitted because this request contains more "
                    "images than the model can accept."
                )
                continue
            url = await resolve(position)
            if url is None:
                continue
            image_bytes = len(url.encode("utf-8"))
            if (
                materialized_image_bytes + image_bytes
                > _MAX_CONTEXT_IMAGE_BYTES_PER_REQUEST
            ):
                fallback_reasons[(position.message_index, position.reference_index)] = (
                    "The image was omitted because this request contains more "
                    "image data than the model can accept."
                )
                continue
            materialized_tokens += estimated_tokens
            materialized_image_bytes += image_bytes
            resolved_images[(position.message_index, position.reference_index)] = url

    result: list[dict[str, Any]] = []
    deferred_user_messages: list[dict[str, Any]] = []

    def flush_deferred() -> None:
        if deferred_user_messages:
            result.extend(deferred_user_messages)
            deferred_user_messages.clear()

    for index, message in enumerate(messages):
        copied = dict(message)
        copied.pop(CONTEXT_REFS_KEY, None)
        refs = references_by_message[index]
        role = copied.get("role")

        if not refs or not supports_vision or resolver is None:
            if refs:
                if not supports_vision:
                    reason = "This model cannot view the image directly."
                elif resolver is None:
                    reason = "The image is not available for direct viewing."
                else:
                    reason = None
                fallback = "\n".join(
                    _context_ref_fallback(ref, reason=reason) for ref in refs
                )
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

        unresolved: list[tuple[ContextReference, str | None]] = []
        for reference_index, reference in enumerate(refs):
            url = resolved_images.get((index, reference_index))
            if url is None:
                unresolved.append(
                    (
                        reference,
                        fallback_reasons.get((index, reference_index)),
                    )
                )
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
            fallback = "\n".join(
                _context_ref_fallback(ref, reason=reason) for ref, reason in unresolved
            )
            content_parts.append({"type": "text", "text": fallback})

        if not any(part.get("type") == "image_url" for part in content_parts):
            copied["content"] = _append_text(
                copied.get("content"),
                "\n".join(
                    _context_ref_fallback(ref, reason=reason)
                    for ref, reason in unresolved
                ),
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
