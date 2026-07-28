from __future__ import annotations

import json
import re
from enum import Enum
from pathlib import PureWindowsPath
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .file_ref import sanitize_file_ref_for_context

CONTEXT_REFS_KEY = "_xagent_context_refs"

_PATH_METADATA_PARTS = {
    "cwd",
    "dir",
    "directory",
    "directories",
    "home",
    "path",
    "paths",
}
_COMMON_PRIVATE_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9])/(?:Users|home|private|root|tmp|var/folders|workspace)"
    r"(?:/|(?=[\s)\]}>,'\";:]|$))"
)
_FILE_URI_PATH_RE = re.compile(r"(?i)(?<![A-Za-z0-9])file:/{1,3}\S+")
_WINDOWS_DRIVE_PATH_RE = re.compile(r"(?i)(?<![A-Za-z0-9])[A-Za-z]:[\\/][^\s\"'<>]+")
_UNC_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9:])(?:\\\\|//)[^\\/\s\"'<>]+[\\/][^\\/\s\"'<>]+"
)
_WINDOWS_ROOTED_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9\\/:])\\(?!\\)[^\\\s\"'<>]+(?:\\[^\\\s\"'<>]+)+"
)


def _contains_materialized_image(value: str) -> bool:
    normalized = value.lower()
    return "data:image/" in normalized and ";base64," in normalized


def _metadata_key_is_path_like(key: str) -> bool:
    parts = {part for part in re.split(r"[^a-z0-9]+", key.lower()) if part}
    return bool(parts & _PATH_METADATA_PARTS)


def _looks_like_absolute_filesystem_path(value: str) -> bool:
    stripped = value.strip()
    if not stripped:
        return False
    windows_path = PureWindowsPath(stripped)
    if stripped.startswith("/") or bool(windows_path.drive or windows_path.root):
        return True
    return any(
        pattern.search(value)
        for pattern in (
            _FILE_URI_PATH_RE,
            _COMMON_PRIVATE_PATH_RE,
            _WINDOWS_DRIVE_PATH_RE,
            _UNC_PATH_RE,
            _WINDOWS_ROOTED_PATH_RE,
        )
    )


def _validate_metadata_string(value: str) -> None:
    if _contains_materialized_image(value):
        raise ValueError("metadata must not contain materialized image data")
    if _looks_like_absolute_filesystem_path(value):
        raise ValueError("metadata must not contain absolute filesystem paths")


def _validate_metadata_tree(value: Any, *, key: str | None = None) -> None:
    if key is not None and _metadata_key_is_path_like(key):
        if value not in (None, "", [], {}):
            raise ValueError(f"metadata field {key!r} must not contain a path")

    if isinstance(value, dict):
        for child_key, child_value in value.items():
            if not isinstance(child_key, str):
                raise ValueError("metadata keys must be strings")
            _validate_metadata_string(child_key)
            _validate_metadata_tree(child_value, key=child_key)
        return
    if isinstance(value, list):
        for child in value:
            _validate_metadata_tree(child)
        return
    if isinstance(value, str):
        _validate_metadata_string(value)


def _validated_metadata(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError("metadata must be an object")
    result = dict(value)
    _validate_metadata_tree(result)
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
    normalized_value: object = json.loads(serialized)
    if not isinstance(normalized_value, dict):
        raise ValueError("metadata must be an object")
    normalized = cast(dict[str, Any], normalized_value)
    _validate_metadata_tree(normalized)
    return normalized


def _validated_text_fallback(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    if _contains_materialized_image(text):
        raise ValueError("text_fallback must not contain materialized image data")
    return text


class ImageDetail(str, Enum):
    AUTO = "auto"
    LOW = "low"
    HIGH = "high"
    ORIGINAL = "original"


class ContextReference(BaseModel):
    """Durable reference to non-text content used by a model call.

    The persisted representation contains only a registered FileRef and small,
    JSON-safe metadata. Provider payloads receive image bytes only after a
    runtime resolver materializes the reference just in time.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["image"] = "image"
    file_ref: dict[str, Any]
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

    @field_validator("metadata", mode="before")
    @classmethod
    def _copy_metadata(cls, value: Any) -> dict[str, Any]:
        return _validated_metadata(value)

    @field_validator("text_fallback", mode="before")
    @classmethod
    def _sanitize_text_fallback(cls, value: Any) -> str | None:
        return _validated_text_fallback(value)

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
            "detail": self.detail.value,
            "metadata": _validated_metadata(self.metadata),
        }
        text_fallback = _validated_text_fallback(self.text_fallback)
        if text_fallback is not None:
            result["text_fallback"] = text_fallback
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
        fallback = f": {self.text_fallback}" if self.text_fallback else ""
        return f"[image: {filename}, file_id={self.file_id}]{fallback}"

    def estimated_tokens(self) -> int:
        image_tokens = {
            ImageDetail.LOW: 85,
            ImageDetail.AUTO: 255,
            ImageDetail.HIGH: 765,
            ImageDetail.ORIGINAL: 765,
        }[self.detail]
        return image_tokens + max(1, len(self.compact_text()) // 4)


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

    Tools may return ``_xagent_context_refs`` as a reserved internal envelope
    field. ExecutionContext stores those refs on the tool Message while keeping
    the public tool observation free of transport-only bookkeeping.
    """

    if not isinstance(result, dict) or CONTEXT_REFS_KEY not in result:
        return result, ()
    public_result = dict(result)
    raw_refs = public_result.pop(CONTEXT_REFS_KEY)
    return public_result, normalize_context_references(raw_refs)
