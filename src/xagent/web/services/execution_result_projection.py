"""Shared projection from agent execution results to chat-channel state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ...core.agent.execution_adapter import INTERRUPTED_USER_MESSAGE
from ...core.agent.result import normalize_tool_failure_code
from ..models.task import TaskStatus
from .assistant_history_safety import ASSISTANT_RESPONSE_MESSAGE_TYPE
from .client_error_messages import (
    CLIENT_SAFE_TASK_FAILURE,
    ClientErrorCode,
    client_error_message,
    mask_provider_secrets,
    model_error_client_projection,
    with_task_reference,
)

EMPTY_CHANNEL_OUTPUT_FALLBACK = "Task completed, but no output was generated."
_TOOL_FAILURE_CLIENT_CODES = {
    "excel_workbook_access_denied": ClientErrorCode.EXCEL_WORKBOOK_ACCESS_DENIED,
    "excel_site_not_found": ClientErrorCode.EXCEL_SITE_NOT_FOUND,
    "excel_workbook_locked": ClientErrorCode.EXCEL_WORKBOOK_LOCKED,
}


def client_safe_failure_message(
    result: Mapping[str, Any],
) -> tuple[str, ClientErrorCode] | None:
    """Map an allowlisted tool failure to a fixed client-facing message."""
    failure_code = normalize_tool_failure_code(result.get("failure_code"))
    if failure_code is None:
        return None
    client_code = _TOOL_FAILURE_CLIENT_CODES.get(failure_code)
    if client_code is None:
        return None
    return client_error_message(client_code), client_code


def interrupted_channel_result() -> dict[str, Any]:
    """The channel result of a run that rests PAUSED without a result of its
    own (lease recovery, a settlement that paused an interrupted run).

    Channels show it as an interruption the user can continue from. A fresh
    dict per call: it may be stored by reference in a command's result.
    """
    return {"success": True, "status": "interrupted"}


def completion_outcome_for_status(
    result: Mapping[str, Any] | None, status: TaskStatus
) -> str | None:
    """Keep semantic completion separate from failed or suspended execution."""
    if status != TaskStatus.COMPLETED or result is None:
        return None
    outcome = result.get("completion_outcome")
    return (
        outcome
        if isinstance(outcome, str) and outcome in {"completed", "partial", "blocked"}
        else None
    )


def execution_result_diagnostic_error(result: Mapping[str, Any]) -> str:
    """Owner-facing diagnostic text for a failed result: the dedicated key first, then ``error``.

    A result that carries a coded failure (``error_code`` is a string, today only
    the quota gate) reports the coded ``error`` text even when a model failure was
    also recorded: the gate describes why the turn ended.

    The dedicated key holds the provider exception text, which holders of the
    agent's API key read through the v1 task API, its event stream and A2A, so
    credential-shaped tokens in it are masked. The ``error`` fallback is returned
    as it is.
    """
    if type(result.get("error_code")) is str:
        return str(result.get("error") or "").strip()
    diagnostic = result.get("diagnostic_error")
    if type(diagnostic) is str and diagnostic.strip():
        return mask_provider_secrets(diagnostic.strip())
    return str(result.get("error") or "").strip()


@dataclass(frozen=True)
class FailedResultPresentation:
    """How one failed result is shown to its owner and on the task stream.

    The persisted transcript row is not part of it: a failed turn always stores
    the generic task-failure row, which the v2 context builder skips and the v1
    history reader replays as the generic sentence, so provider text never
    re-enters the model's context.
    """

    # -> ``tasks.error_message``; the caller applies its own empty fallback.
    diagnostic_error: str
    # -> chat bubble and ``task_completed.output``; ``None`` keeps the caller's text.
    visible_text: str | None
    # -> ``task_completed.error_code`` and ``task_completed.error_details``.
    error_code: str | None
    error_details: dict[str, Any] | None


def present_failed_result(result: Mapping[str, Any]) -> FailedResultPresentation:
    """Decide how one failed result is shown to its owner and on the task stream.

    A coded failure already on the result (the quota gate or a classified tool
    failure) takes precedence: its fields pass through and no model failure is
    projected. Otherwise a recorded model-provider failure is projected through
    ``model_error_client_projection``.
    """
    diagnostic_error = execution_result_diagnostic_error(result)
    existing_code = result.get("error_code")
    safe_failure = (
        None if existing_code is not None else client_safe_failure_message(result)
    )
    projection = (
        model_error_client_projection(result.get("model_error"))
        if existing_code is None and safe_failure is None
        else None
    )
    if projection is None:
        # A recorded model failure whose projection is withheld (an HTTP 400
        # record: ``model_error_client_projection`` returns ``None`` only for a
        # non-dict or an HTTP 400, and a malformed dict still projects to the
        # fixed fallback sentence) must not fall back to the caller's text: that
        # text is the adapter's ``output`` backfill, i.e. the raw ``error`` string
        # (plan generation puts the provider text there). Force the generic
        # sentence for every task-stream surface.
        withheld = type(result.get("model_error")) is dict and existing_code is None
        error_details = result.get("error_details")
        if safe_failure is not None:
            safe_message, safe_code = safe_failure
            return FailedResultPresentation(
                diagnostic_error=diagnostic_error,
                visible_text=safe_message,
                error_code=safe_code.value,
                error_details={"failure_code": result["failure_code"]},
            )
        return FailedResultPresentation(
            diagnostic_error=diagnostic_error,
            visible_text=CLIENT_SAFE_TASK_FAILURE if withheld else None,
            error_code=existing_code if type(existing_code) is str else None,
            error_details=error_details if type(error_details) is dict else None,
        )
    details = {key: projection[key] for key in ("kind", "status_code", "provider_code")}
    details["message"] = projection["error_message"]
    return FailedResultPresentation(
        diagnostic_error=diagnostic_error,
        visible_text=projection["error_message"],
        error_code=projection["error_code"],
        error_details=details,
    )


@dataclass(frozen=True)
class ChannelExecutionProjection:
    task_status: TaskStatus
    visible_text: str
    transcript_content: str
    message_type: str
    interactions: list[dict[str, Any]]
    diagnostic_error: str | None
    completion_outcome: str | None


def project_execution_result_for_channel(
    result: dict[str, Any],
    *,
    task_id: int | None = None,
) -> ChannelExecutionProjection:
    """Project an execution result into the state chat channels should consume.

    When ``task_id`` is given, the visible failure text ends with a reportable
    task reference; the transcript content never carries it. The default
    ``None`` keeps the visible failure text exactly ``CLIENT_SAFE_TASK_FAILURE``
    for callers that compare against it.
    """
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
    task_status = _project_task_status(result, status)
    diagnostic_error = None
    if task_status == TaskStatus.FAILED:
        diagnostic_error = (
            execution_result_diagnostic_error(result) or base_text.strip() or None
        )
    if status == "interrupted":
        # An interruption is control state, not an assistant answer. Show a
        # friendly status in every chat channel without adding it to the
        # conversation transcript used by later model turns.
        base_text = INTERRUPTED_USER_MESSAGE
        transcript_content = ""
        interactions = []
    elif task_status == TaskStatus.FAILED:
        transcript_content = CLIENT_SAFE_TASK_FAILURE
        safe_failure = (
            None
            if result.get("error_code") is not None
            else client_safe_failure_message(result)
        )
        base_text = with_task_reference(
            safe_failure[0] if safe_failure is not None else CLIENT_SAFE_TASK_FAILURE,
            task_id,
        )
        interactions = []
    elif not base_text.strip() and not interactions:
        base_text = EMPTY_CHANNEL_OUTPUT_FALLBACK
        transcript_content = base_text

    visible_text = _append_interactions(base_text, interactions)

    return ChannelExecutionProjection(
        task_status=task_status,
        visible_text=visible_text,
        transcript_content=transcript_content,
        message_type="question"
        if status == "waiting_for_user" or interactions
        else ASSISTANT_RESPONSE_MESSAGE_TYPE,
        interactions=interactions,
        diagnostic_error=diagnostic_error,
        completion_outcome=completion_outcome_for_status(result, task_status),
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
