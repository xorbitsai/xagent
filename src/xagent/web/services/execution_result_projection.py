"""Shared projection from agent execution results to chat-channel state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ...core.agent.execution_adapter import INTERRUPTED_USER_MESSAGE
from ..models.task import TaskStatus

EMPTY_CHANNEL_OUTPUT_FALLBACK = "Task completed, but no output was generated."


@dataclass(frozen=True)
class ChannelExecutionProjection:
    task_status: TaskStatus
    visible_text: str
    transcript_content: str
    message_type: str
    interactions: list[dict[str, Any]]


def project_execution_result_for_channel(
    result: dict[str, Any],
) -> ChannelExecutionProjection:
    """Project an execution result into the state chat channels should consume."""
    status = str(result.get("status") or "")
    chat_response = result.get("chat_response")
    chat_message = ""
    interactions: list[dict[str, Any]] = []

    if isinstance(chat_response, dict):
        chat_message = str(chat_response.get("message") or "")
        raw_interactions = chat_response.get("interactions")
        if isinstance(raw_interactions, list):
            interactions = [item for item in raw_interactions if isinstance(item, dict)]

    output = str(result.get("output") or "")
    base_text = chat_message or output
    transcript_content = base_text
    if status == "interrupted":
        # An interruption is control state, not an assistant answer. Show a
        # friendly status in every chat channel without adding it to the
        # conversation transcript used by later model turns.
        base_text = INTERRUPTED_USER_MESSAGE
        transcript_content = ""
        interactions = []
    elif not base_text.strip() and not interactions:
        base_text = EMPTY_CHANNEL_OUTPUT_FALLBACK
        transcript_content = base_text

    visible_text = _append_interactions(base_text, interactions)

    return ChannelExecutionProjection(
        task_status=_project_task_status(result, status),
        visible_text=visible_text,
        transcript_content=transcript_content,
        message_type="question"
        if status == "waiting_for_user" or interactions
        else "assistant_message",
        interactions=interactions,
    )


def _project_task_status(result: dict[str, Any], status: str) -> TaskStatus:
    if status == "waiting_for_user":
        return TaskStatus.WAITING_FOR_USER
    if status == "interrupted":
        return TaskStatus.PAUSED
    return TaskStatus.COMPLETED if result.get("success", False) else TaskStatus.FAILED


def _append_interactions(base_text: str, interactions: list[dict[str, Any]]) -> str:
    if not interactions:
        return base_text

    interaction_texts: list[str] = []
    for interaction in interactions:
        label = interaction.get("label") or interaction.get("field", "Input")
        options = interaction.get("options", [])
        if options:
            opts = []
            for opt in options:
                if isinstance(opt, dict):
                    opts.append(str(opt.get("label", opt.get("value", ""))))
                else:
                    opts.append(str(opt))
            interaction_texts.append(f"• {label}\n  Options: {', '.join(opts)}")
        else:
            interaction_texts.append(f"• {label}")

    return base_text + "\n\n" + "\n".join(interaction_texts)
