"""Helpers for deriving reusable cross-round execution context from persisted traces."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from ...core.file_ref import build_file_id_ref
from ...skills.utils import create_skill_manager
from ..models.task import TraceEvent


async def load_task_execution_recovery_state(
    db: Session,
    task_id: int,
    *,
    max_tool_events: int = 8,
) -> Dict[str, Any]:
    """Load reusable execution recovery state for a task."""
    return {
        "messages": load_task_execution_context_messages(
            db, task_id, max_tool_events=max_tool_events
        ),
        "skill_context": await load_task_recovered_skill_context(db, task_id),
    }


def load_task_execution_context_messages(
    db: Session,
    task_id: int,
    *,
    max_tool_events: int = 8,
) -> List[Dict[str, str]]:
    """Load reusable prior execution context for a task as planner-visible messages."""
    trace_events = (
        db.query(TraceEvent)
        .filter(
            TraceEvent.task_id == task_id,
            TraceEvent.build_id.is_(None),
            TraceEvent.event_type == "tool_execution_end",
        )
        .order_by(TraceEvent.timestamp.desc(), TraceEvent.id.desc())
        .limit(max_tool_events)
        .all()
    )

    tool_summaries: List[str] = []
    seen_summaries: set[str] = set()

    for trace_event in trace_events:
        data: Dict[str, Any] = (
            trace_event.data if isinstance(trace_event.data, dict) else {}
        )
        summary = summarize_tool_event(data)
        if not summary or summary in seen_summaries:
            continue
        seen_summaries.add(summary)
        tool_summaries.append(summary)

    failure_summary = load_latest_execution_failure_summary(db, task_id)
    if not tool_summaries and not failure_summary:
        return []

    tool_summaries.reverse()
    context_lines = []
    if failure_summary:
        context_lines.append(failure_summary)
    context_lines.extend(tool_summaries)
    content = (
        "=== Previous Execution Context ===\n"
        "Prior execution context for this task. Reuse when relevant; rerun only if needed.\n"
        + "\n".join(context_lines)
    )
    return [{"role": "system", "content": content}]


async def load_task_recovered_skill_context(db: Session, task_id: int) -> Optional[str]:
    """Load the latest selected skill context for a task, if any."""
    trace_event = (
        db.query(TraceEvent)
        .filter(
            TraceEvent.task_id == task_id,
            TraceEvent.build_id.is_(None),
            TraceEvent.event_type == "skill_select_end",
        )
        .order_by(TraceEvent.timestamp.desc(), TraceEvent.id.desc())
        .first()
    )
    if trace_event is None or not isinstance(trace_event.data, dict):
        return None

    selected = bool(trace_event.data.get("selected", False))
    skill_name = str(trace_event.data.get("skill_name") or "").strip()
    if not selected or not skill_name:
        return None

    return await _load_skill_context_by_name(skill_name)


_BINARY_MAGIC_PREFIXES = (
    "\x89PNG",
    "\xff\xd8\xff",
    "GIF87a",
    "GIF89a",
    "%PDF",
    "PK\x03\x04",
)
_DATA_URL_RE = re.compile(r"^data:[^;]+;base64,", re.IGNORECASE)


def summarize_tool_event(data: Dict[str, Any]) -> Optional[str]:
    """Summarize a persisted tool execution event into reusable context text."""
    tool_name = str(data.get("tool_name") or "").strip()
    if not tool_name:
        return None

    if not bool(data.get("success", True)):
        return None

    result = data.get("result")
    summary = _summarize_generic_result(result)
    if not summary:
        return None
    if "payload omitted" in summary:
        params_summary = _summarize_generic_result(
            data.get("tool_params"), max_length=120
        )
        if params_summary:
            summary = f"{summary} (tool_params: {params_summary})"
    return f"- Tool {tool_name} previously returned: {summary}"


def load_latest_execution_failure_summary(
    db: Session,
    task_id: int,
) -> Optional[str]:
    """Load a concise anchor for the latest failed execution, if any."""
    trace_event = (
        db.query(TraceEvent)
        .filter(
            TraceEvent.task_id == task_id,
            TraceEvent.build_id.is_(None),
            TraceEvent.event_type.in_(("dag_execute_end", "trace_error")),
        )
        .order_by(TraceEvent.timestamp.desc(), TraceEvent.id.desc())
        .first()
    )
    if trace_event is None or not isinstance(trace_event.data, dict):
        return None
    return summarize_execution_failure_event(trace_event.data)


def summarize_execution_failure_event(data: Dict[str, Any]) -> Optional[str]:
    result = data.get("result") if isinstance(data.get("result"), dict) else data
    if not isinstance(result, dict):
        return None

    success = result.get("success")
    status = str(result.get("status") or data.get("status") or "").strip()
    if success is not False and status not in {"failed", "error"}:
        return None

    error = str(result.get("error") or data.get("error_message") or "").strip()
    failure_reason = str(result.get("failure_reason") or "").strip()
    failed_step_id = str(result.get("failed_step_id") or "").strip()

    details: list[str] = []
    if failed_step_id:
        details.append(f"step={failed_step_id}")
    if failure_reason:
        details.append(f"reason={failure_reason}")
    if error:
        details.append(f"error={_compact_preview_string(error, max_string_length=180)}")

    if not details:
        return None
    return "- Previous execution failed: " + "; ".join(details)


def _summarize_generic_result(result: Any, max_length: int = 240) -> Optional[str]:
    if result is None:
        return None
    file_links = _collect_file_reference_links(result)
    compact_result = _compact_preview_value(result)
    if isinstance(compact_result, str):
        preview = compact_result.strip()
    else:
        try:
            preview = json.dumps(compact_result, ensure_ascii=False)
        except Exception:
            preview = str(compact_result).strip()

    file_summary = f"files: {', '.join(file_links)}" if file_links else ""
    if not preview:
        return file_summary or None
    if not file_summary:
        if len(preview) > max_length:
            return preview[: max_length - 3] + "..."
        return preview

    # FileRefs are durable execution state. Keep their canonical markdown links
    # intact and spend only the remaining preview budget on verbose provider
    # payloads (video URLs commonly appear before file_ref in tool results).
    separator = "; result: "
    remaining = max_length - len(file_summary) - len(separator)
    if remaining <= 3:
        return file_summary
    if len(preview) > remaining:
        preview = preview[: remaining - 3] + "..."
    return file_summary + separator + preview


_MARKDOWN_FILE_LINK_RE = re.compile(r"!?\[[^\]]*\]\((file:[^)\s]+)\)")


def _collect_file_reference_links(value: Any, *, limit: int = 5) -> List[str]:
    """Collect exact model-facing FileRef links from a nested tool result."""
    links: List[str] = []
    seen: set[str] = set()

    def add(link: str) -> None:
        normalized = link.strip()
        if not normalized or normalized in seen or len(links) >= limit:
            return
        seen.add(normalized)
        links.append(normalized)

    def visit(item: Any, depth: int = 0) -> None:
        if depth > 8 or len(links) >= limit:
            return
        if isinstance(item, dict):
            file_id = str(item.get("file_id") or "").strip()
            filename = str(item.get("filename") or item.get("name") or "").strip()
            markdown_link = item.get("markdown_link")
            if isinstance(markdown_link, str):
                matches = _MARKDOWN_FILE_LINK_RE.findall(markdown_link)
                if matches:
                    add(markdown_link)
            elif file_id and filename:
                try:
                    add(f"[{filename}]({build_file_id_ref(file_id)})")
                except ValueError:
                    pass

            for key, nested in item.items():
                if key in {"raw", "raw_response", "content", "video_url"}:
                    continue
                if isinstance(nested, (dict, list, tuple)):
                    visit(nested, depth + 1)
        elif isinstance(item, (list, tuple)):
            for nested in item:
                visit(nested, depth + 1)

    visit(value)
    return links


def _compact_preview_value(value: Any, *, max_string_length: int = 160) -> Any:
    if isinstance(value, str):
        return _compact_preview_string(value, max_string_length=max_string_length)
    if isinstance(value, list):
        return [
            _compact_preview_value(item, max_string_length=max_string_length)
            for item in value[:20]
        ]
    if isinstance(value, dict):
        compact: Dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            compact[key_text] = _compact_preview_value(
                item,
                max_string_length=max_string_length,
            )
        return compact
    return value


def _compact_preview_string(value: str, *, max_string_length: int) -> str:
    text = value.strip()
    if not text:
        return text
    if _looks_like_binary_or_encoded_payload(text):
        return f"[non-text payload omitted; {len(value)} characters]"
    if len(text) > max_string_length:
        return text[: max_string_length - 3] + "..."
    return text


def _looks_like_binary_or_encoded_payload(value: str) -> bool:
    if value.startswith(_BINARY_MAGIC_PREFIXES):
        return True
    if _DATA_URL_RE.match(value):
        return True

    sample = value[:4096]
    if not sample:
        return False

    control_chars = 0
    for char in sample:
        codepoint = ord(char)
        if char in "\n\r\t":
            continue
        if codepoint < 32 or 0x80 <= codepoint <= 0x9F:
            control_chars += 1

    return control_chars / max(len(sample), 1) > 0.02


async def _load_skill_context_by_name(skill_name: str) -> Optional[str]:
    skill_manager = create_skill_manager()
    skill = await skill_manager.get_skill(skill_name)
    if not skill:
        return None
    return _build_skill_context(skill)


def _build_skill_context(skill: Dict[str, Any]) -> str:
    content = str(skill.get("content", "")).strip()
    return f"## Available Skill: {skill['name']}\n\n{content}"
