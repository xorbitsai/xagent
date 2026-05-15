from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .result import extract_assistant_message
from .trace import (
    trace_ai_message,
    trace_error,
    trace_task_completion,
    trace_user_message,
)


@dataclass
class TraceEventCallback:
    """Bridge agent runner callbacks into the existing web trace stream."""

    async def on_run_start(
        self,
        *,
        runner: Any,
        context: Any,
        resume: bool = False,
        checkpoint: dict[str, Any] | None = None,
    ) -> None:
        tracer = getattr(runner, "tracer", None)
        if tracer is None or not callable(getattr(tracer, "trace_event", None)):
            return

        execution_id = str(getattr(context, "execution_id", "") or "")

        task = self._task_from_context(context)
        if task and not (resume or checkpoint):
            await trace_user_message(
                tracer,
                execution_id,
                task,
                {"context": self._context_payload(context)},
            )

    async def on_run_end(
        self, *, runner: Any, context: Any, result: dict[str, Any]
    ) -> None:
        tracer = getattr(runner, "tracer", None)
        if tracer is None or not callable(getattr(tracer, "trace_event", None)):
            return

        execution_id = str(
            result.get("execution_id") or getattr(context, "execution_id", "") or ""
        )
        status = str(result.get("status") or "")
        output = extract_assistant_message(result)
        data = {
            "execution_id": execution_id,
            "status": status or ("completed" if result.get("success") else "failed"),
            "pattern": result.get("pattern"),
            "context": self._context_payload(context),
        }

        if result.get("success"):
            if output:
                completion_result: dict[str, Any] = {"content": output}
                file_outputs = result.get("file_outputs")
                if file_outputs:
                    completion_result["file_outputs"] = file_outputs
                    completion_result["output"] = output
                await trace_ai_message(tracer, execution_id, output, data)
                await trace_task_completion(
                    tracer,
                    execution_id,
                    result=completion_result,
                    success=True,
                )
            return

        if status in {"interrupted", "waiting_for_user"}:
            # Paused/interrupted executions are resumable control states, not
            # completions. The web trace compatibility layer maps
            # TASK_END_GENERAL to task_completion, so do not emit it here.
            return

        await trace_error(
            tracer,
            execution_id,
            error_type="agent_error",
            error_message=str(result.get("error") or "agent execution failed"),
            data=data,
        )

    def _context_payload(self, context: Any) -> dict[str, Any] | None:
        to_dict = getattr(context, "to_dict", None)
        if callable(to_dict):
            payload = to_dict()
            return dict(payload) if isinstance(payload, dict) else None
        return None

    def _task_from_context(self, context: Any) -> str | None:
        metadata = getattr(context, "metadata", None)
        if isinstance(metadata, dict):
            task = metadata.get("task")
            if isinstance(task, str) and task:
                return task
        messages = getattr(context, "messages", [])
        for message in messages:
            if getattr(message, "role", None) == "user":
                content = getattr(message, "content", None)
                if isinstance(content, str) and content:
                    return content
        return None
