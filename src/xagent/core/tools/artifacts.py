"""Helpers for exposing generated files as displayable tool artifacts."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterable

from ..file_ref import build_workspace_file_ref, guess_mime_type

logger = logging.getLogger(__name__)

GENERATED_ARTIFACT_EXTENSIONS = {
    ".csv",
    ".docx",
    ".gif",
    ".jpeg",
    ".jpg",
    ".mov",
    ".mp4",
    ".mpeg",
    ".mpg",
    ".pdf",
    ".png",
    ".pptx",
    ".svg",
    ".webp",
    ".webm",
    ".xls",
    ".xlsx",
}
SAFE_FILE_REF_KEYS = {
    "download_url",
    "file_id",
    "filename",
    "markdown_link",
    "markdown_ref",
    "mime_type",
    "preview_url",
    "relative_path",
    "size",
}
LOCAL_PATH_KEYS = {
    "absolute_path",
    "file_path",
    "image_path",
    "local_path",
    "output_dir",
    "output_path",
}

GeneratedArtifactSnapshot = dict[Path, tuple[int, int]]


def artifact_type_for_filename(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg"}:
        return "image"
    if suffix in {".mp4", ".mov", ".mpeg", ".mpg", ".webm"}:
        return "video"
    # Only OOXML .pptx maps to ``presentation`` — the frontend's inline
    # ``PptxPreviewRenderer`` (pptxviewjs) cannot render legacy binary
    # .ppt, so emitting ``presentation`` for .ppt would let an artifact
    # bypass the supported-format boundary on the consumer side. Legacy
    # .ppt falls through to the generic ``file`` kind, which renders as
    # a download link.
    if suffix == ".pptx":
        return "presentation"
    if suffix == ".docx":
        return "document"
    if suffix in {".csv", ".xls", ".xlsx"}:
        return "spreadsheet"
    return "file"


def build_inline_artifact(file_ref: dict[str, Any]) -> dict[str, str]:
    filename = str(file_ref.get("filename") or "artifact")
    return {
        "type": artifact_type_for_filename(filename),
        "file_id": str(file_ref.get("file_id") or ""),
        "filename": filename,
        "mime_type": str(file_ref.get("mime_type") or guess_mime_type(filename)),
        "display": "inline",
    }


def markdown_reference_for_artifact(artifact: dict[str, Any]) -> str | None:
    file_id = artifact.get("file_id")
    if not file_id:
        return None

    filename = str(artifact.get("filename") or "artifact")
    markdown_ref = f"file:{file_id}"
    artifact_type = str(
        artifact.get("type") or artifact_type_for_filename(filename)
    ).lower()
    if artifact_type == "image":
        return f"![{filename}]({markdown_ref})"
    return f"[{filename}]({markdown_ref})"


def format_tool_result_for_observation(tool_name: str, result: Any) -> str:
    """Format tool results for model-facing observations.

    The formatter may expose artifact usage conventions, but it stays transport
    neutral: concrete browser routes are a frontend/web concern.
    """
    if not isinstance(result, dict):
        return str(result)

    artifacts = result.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        return str(result)

    sanitized = sanitize_file_refs_for_observation(result)

    artifact_lines = _format_artifact_lines(artifacts)
    if not artifact_lines:
        return str(sanitized)

    return (
        f"Tool '{tool_name}' produced displayable artifact(s):\n"
        + "\n".join(artifact_lines)
        + "\nUse the Markdown/chat form in assistant messages. "
        + "When writing HTML for Xagent preview, reference the same file_id "
        + "through the file preview service instead of local filesystem paths. "
        + f"Sanitized result metadata: {sanitized}"
    )


def _format_artifact_lines(artifacts: list[Any]) -> list[str]:
    lines: list[str] = []
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            continue
        file_id = artifact.get("file_id")
        if not file_id:
            continue
        filename = artifact.get("filename") or "generated image"
        markdown_ref = markdown_reference_for_artifact(artifact)
        if not markdown_ref:
            continue
        artifact_type = str(artifact.get("type") or "").lower()
        markdown_label = (
            "Markdown/chat image" if artifact_type == "image" else "Markdown/chat file"
        )
        lines.append(
            "\n".join(
                [
                    f"- {filename}",
                    f"  file_id: {file_id}",
                    f"  {markdown_label}: {markdown_ref}",
                    "  HTML preview: use the file preview service for this file_id",
                ]
            )
        )
    return lines


def sanitize_file_refs_for_observation(value: Any) -> Any:
    """Return model/context-safe FileRef metadata without local paths."""
    return sanitize_tool_result_for_public_context(value)


def sanitize_tool_result_for_public_context(value: Any) -> Any:
    """Return tool result data safe for model/context exposure."""
    known_paths = _collect_known_local_paths(value)
    return _sanitize_tool_result_value(value, known_paths)


def _sanitize_tool_result_value(value: Any, known_paths: dict[str, str]) -> Any:
    if isinstance(value, list):
        return [_sanitize_tool_result_value(item, known_paths) for item in value]

    if isinstance(value, str):
        return _replace_known_local_paths(value, known_paths)

    if not isinstance(value, dict):
        return value

    return {
        key: _sanitize_tool_result_value(item, known_paths)
        for key, item in _safe_tool_result_items(value)
    }


def _safe_tool_result_items(value: dict[str, Any]) -> Iterable[tuple[str, Any]]:
    if _is_file_ref_like(value):
        return ((key, value[key]) for key in SAFE_FILE_REF_KEYS if key in value)
    return ((key, item) for key, item in value.items() if key not in LOCAL_PATH_KEYS)


def _is_file_ref_like(value: dict[str, Any]) -> bool:
    return (
        "file_id" in value
        and "filename" in value
        and ("file_path" in value or "relative_path" in value or "mime_type" in value)
    )


def _collect_known_local_paths(value: Any) -> dict[str, str]:
    paths: dict[str, str] = {}

    def visit(item: Any) -> None:
        if isinstance(item, list):
            for list_item in item:
                visit(list_item)
            return
        if not isinstance(item, dict):
            return

        replacement = _public_file_label(item)
        for key, child in item.items():
            if key in LOCAL_PATH_KEYS and isinstance(child, str):
                paths[child] = replacement or Path(child).name
            visit(child)

    visit(value)
    return paths


def _public_file_label(value: dict[str, Any]) -> str | None:
    filename = value.get("filename")
    if filename:
        return str(filename)
    relative_path = value.get("relative_path")
    if relative_path:
        return str(relative_path)
    return None


def _replace_known_local_paths(value: str, known_paths: dict[str, str]) -> str:
    sanitized = value
    for path, replacement in sorted(
        known_paths.items(), key=lambda item: len(item[0]), reverse=True
    ):
        sanitized = sanitized.replace(path, replacement)
    return sanitized


def build_generated_file_metadata(
    *,
    workspace: Any,
    file_paths: Iterable[str | Path],
) -> dict[str, list[Any]]:
    file_refs: list[dict[str, Any]] = []
    artifacts: list[dict[str, str]] = []
    generated_files: list[str] = []

    for file_path in sorted({Path(path).resolve() for path in file_paths}):
        if not file_path.exists() or not file_path.is_file():
            continue
        if file_path.suffix.lower() not in GENERATED_ARTIFACT_EXTENSIONS:
            continue
        try:
            file_ref = build_workspace_file_ref(
                workspace=workspace, file_path=file_path
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to build FileRef for generated file %s: %s", file_path, exc
            )
            continue
        file_refs.append(file_ref)
        artifacts.append(build_inline_artifact(file_ref))
        generated_files.append(file_ref["filename"])

    return {
        "generated_files": generated_files,
        "file_refs": file_refs,
        "artifacts": artifacts,
    }


def scan_generated_artifact_files(root: str | Path) -> set[Path]:
    return set(snapshot_generated_artifact_files(root))


def snapshot_generated_artifact_files(root: str | Path) -> GeneratedArtifactSnapshot:
    root_path = Path(root)
    if not root_path.exists():
        return {}

    snapshot: GeneratedArtifactSnapshot = {}
    for file_path in root_path.rglob("*"):
        try:
            relative_path = file_path.relative_to(root_path)
            if (
                not file_path.is_file()
                or any(part.startswith(".") for part in relative_path.parts)
                or file_path.suffix.lower() not in GENERATED_ARTIFACT_EXTENSIONS
            ):
                continue
            file_stat = file_path.stat()
        except FileNotFoundError:
            continue
        snapshot[file_path] = (file_stat.st_mtime_ns, file_stat.st_size)

    return snapshot


def changed_generated_artifact_files(
    before: GeneratedArtifactSnapshot,
    after: GeneratedArtifactSnapshot,
) -> set[Path]:
    return {
        file_path
        for file_path, file_snapshot in after.items()
        if before.get(file_path) != file_snapshot
    }
