from __future__ import annotations

import json
import math
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .file_ref import sanitize_file_ref_for_context

CONTEXT_REFS_KEY = "_xagent_context_refs"

#: Reserved envelope field naming a scope whose earlier tool results this one
#: replaces. A ``computer`` observation describes a page that no longer exists
#: once the next action runs, so keeping every past element list in full would
#: grow the context without adding usable information.
SUPERSEDES_SCOPE_KEY = "_xagent_supersedes_scope"


def split_tool_result_supersedes_scope(result: Any) -> tuple[Any, str | None]:
    """Detach the supersede scope from a tool result before formatting."""
    if not isinstance(result, dict) or SUPERSEDES_SCOPE_KEY not in result:
        return result, None
    public_result = dict(result)
    scope = public_result.pop(SUPERSEDES_SCOPE_KEY)
    normalized = str(scope).strip() if scope is not None else ""
    return public_result, normalized or None


def _validated_metadata(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError("metadata must be an object")
    result = dict(value)
    try:
        serialized = json.dumps(
            result,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("metadata must be JSON serializable") from exc
    if len(serialized.encode("utf-8")) > 32_768:
        raise ValueError("metadata must be at most 32 KiB")
    normalized = serialized.lower()
    if "data:image/" in normalized and ";base64," in normalized:
        raise ValueError("metadata must not contain materialized image data")
    return result


class ContextReferencePurpose(str, Enum):
    OBSERVATION = "observation"
    ATTACHMENT = "attachment"
    ARTIFACT = "artifact"


class ImageDetail(str, Enum):
    AUTO = "auto"
    LOW = "low"
    HIGH = "high"
    ORIGINAL = "original"


class ContextReference(BaseModel):
    """Durable reference to non-text content used by a model call.

    The persisted representation is always a FileRef. Providers receive image
    bytes only after a runtime resolver materializes this reference just in time.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["image"] = "image"
    file_ref: dict[str, Any]
    purpose: ContextReferencePurpose = ContextReferencePurpose.ATTACHMENT
    frame_id: str | None = Field(default=None, max_length=256)
    detail: ImageDetail = ImageDetail.AUTO
    text_fallback: str | None = Field(default=None, max_length=16_000)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("file_ref", mode="before")
    @classmethod
    def _sanitize_file_ref(cls, value: Any) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise TypeError("file_ref must be a FileRef object")
        result = sanitize_file_ref_for_context(value)
        mime_type = str(result.get("mime_type") or "")
        if mime_type and not mime_type.startswith("image/"):
            raise ValueError("image context references require an image MIME type")
        return result

    @field_validator("frame_id")
    @classmethod
    def _normalize_frame_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("frame_id must not be empty")
        return normalized

    @field_validator("metadata", mode="before")
    @classmethod
    def _copy_metadata(cls, value: Any) -> dict[str, Any]:
        return _validated_metadata(value)

    @model_validator(mode="after")
    def _observation_requires_frame(self) -> "ContextReference":
        if self.purpose == ContextReferencePurpose.OBSERVATION and not self.frame_id:
            raise ValueError("observation context references require a frame_id")
        return self

    @property
    def file_id(self) -> str:
        return str(self.safe_file_ref["file_id"])

    @property
    def safe_file_ref(self) -> dict[str, Any]:
        return sanitize_file_ref_for_context(self.file_ref)

    def durable_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "type": self.type,
            "file_ref": self.safe_file_ref,
            "purpose": self.purpose.value,
            "detail": self.detail.value,
            "metadata": _validated_metadata(self.metadata),
        }
        if self.frame_id is not None:
            result["frame_id"] = self.frame_id
        if self.text_fallback is not None:
            result["text_fallback"] = self.text_fallback
        return result

    def identity_key(self) -> str:
        return json.dumps(
            self.durable_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    def compact_text(self) -> str:
        filename = self.safe_file_ref.get("filename") or "image"
        frame = f", frame={self.frame_id}" if self.frame_id else ""
        fallback = f": {self.text_fallback}" if self.text_fallback else ""
        return f"[image: {filename}, file_id={self.file_id}{frame}]{fallback}"

    def estimated_tokens(self) -> int:
        # Provider image-token accounting varies, but a flat constant makes
        # compaction believe a full-viewport screenshot is as cheap as a
        # thumbnail. When the reference records the viewport it was captured
        # at, the tile-based estimate that mainstream providers use is applied
        # instead. No binary or base64 payload is ever inspected.
        return self._image_tokens() + max(1, len(self.compact_text()) // 4)

    def _image_tokens(self) -> int:
        if self.detail is ImageDetail.LOW:
            return 85
        viewport = self.metadata.get("viewport")
        if not isinstance(viewport, dict):
            return (
                765 if self.detail in {ImageDetail.HIGH, ImageDetail.ORIGINAL} else 255
            )
        try:
            width = int(viewport.get("width") or 0)
            height = int(viewport.get("height") or 0)
        except (TypeError, ValueError):
            return 255
        if width <= 0 or height <= 0:
            return 255
        # Long side capped at 2048, short side at 768, then billed per 512px
        # tile plus a fixed base cost.
        scale = min(1.0, 2048 / max(width, height))
        width, height = int(width * scale), int(height * scale)
        scale = min(1.0, 768 / max(1, min(width, height)))
        width, height = int(width * scale), int(height * scale)
        tiles = math.ceil(max(1, width) / 512) * math.ceil(max(1, height) / 512)
        return 85 + 170 * tiles


def normalize_context_references(
    values: Any,
) -> tuple[ContextReference, ...]:
    if values is None:
        return ()
    if isinstance(values, ContextReference):
        return (values,)
    if not isinstance(values, (list, tuple)):
        raise TypeError("context_refs must be a list or tuple")
    return tuple(
        value
        if isinstance(value, ContextReference)
        else ContextReference.model_validate(value)
        for value in values
    )


def split_tool_result_context_references(
    result: Any,
) -> tuple[Any, tuple[ContextReference, ...]]:
    """Detach durable context refs from a tool result before model formatting.

    Tools can return ``_xagent_context_refs`` as a reserved internal envelope
    field. ExecutionContext stores those refs on the tool Message while keeping
    the public tool observation free of transport-only bookkeeping.
    """
    if not isinstance(result, dict) or CONTEXT_REFS_KEY not in result:
        return result, ()
    public_result = dict(result)
    raw_refs = public_result.pop(CONTEXT_REFS_KEY)
    return public_result, normalize_context_references(raw_refs)
