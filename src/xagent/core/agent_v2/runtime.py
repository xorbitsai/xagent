from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass, field
from typing import Any, Callable
from uuid import uuid4

from ..agent.trace import (
    TraceAction,
    TraceCategory,
    TraceEventType,
    TraceScope,
)


class LLMCallInterrupted(Exception):
    """Raised when an active LLM call is cancelled by an execution interrupt."""


@dataclass
class PatternRuntime:
    """Thin runtime services shared by execution patterns.

    Patterns own execution state; the runtime owns cross-cutting concerns such as
    checkpoint emission and interrupt requests.
    """

    tracer: Any | None = None
    execution_id: str | None = None
    interrupt_checker: Callable[[], bool] | Callable[[], Any] | None = None
    outbound_message_handler: Callable[[dict[str, Any]], Any] | None = None
    _interrupt_requested: bool = False
    interrupt_reason: str | None = None
    last_checkpoint: dict[str, Any] | None = None
    checkpoints: list[dict[str, Any]] = field(default_factory=list)
    outbound_messages: list[dict[str, Any]] = field(default_factory=list)
    spans: list[dict[str, Any]] = field(default_factory=list)
    finished_spans: list[dict[str, Any]] = field(default_factory=list)
    trace_runs: list[dict[str, Any]] = field(default_factory=list)
    active_react_step_id: str | None = None
    _active_llm_tasks: set[asyncio.Future[Any]] = field(
        default_factory=set,
        init=False,
        repr=False,
    )

    def request_interrupt(self, reason: str | None = None) -> None:
        self._interrupt_requested = True
        self.interrupt_reason = reason
        for task in list(self._active_llm_tasks):
            if not task.done():
                task.cancel()

    def clear_interrupt(self) -> None:
        self._interrupt_requested = False
        self.interrupt_reason = None

    async def should_interrupt(self) -> bool:
        if self._interrupt_requested:
            return True
        if self.interrupt_checker is None:
            return False

        result = self.interrupt_checker()
        if inspect.isawaitable(result):
            result = await result
        if result:
            self._interrupt_requested = True
            self.request_interrupt(self.interrupt_reason)
        return bool(result)

    async def run_llm_call(self, llm: Any, **kwargs: Any) -> Any:
        """Run an LLM call as a cancellable subtask owned by this runtime."""

        call = llm.chat(**kwargs)
        if not inspect.isawaitable(call):
            return call

        task: asyncio.Future[Any] = asyncio.ensure_future(call)
        self._active_llm_tasks.add(task)
        try:
            return await task
        except asyncio.CancelledError as exc:
            if self._interrupt_requested:
                raise LLMCallInterrupted(
                    self.interrupt_reason or "interrupted during LLM call"
                ) from exc
            raise
        finally:
            self._active_llm_tasks.discard(task)

    async def checkpoint(
        self,
        label: str,
        *,
        context: Any,
        pattern: Any,
        status: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = self._build_checkpoint_payload(
            label=label,
            context=context,
            pattern=pattern,
            status=status,
            metadata=metadata,
        )
        self.last_checkpoint = payload
        self.checkpoints.append(payload)
        await self._emit_checkpoint(payload)
        return payload

    async def send_message(
        self,
        *,
        message: str,
        message_type: str = "info",
        expect_response: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Emit an agent-to-user message through the runtime boundary."""

        payload = {
            "type": "agent_message",
            "execution_id": self.execution_id,
            "message": message,
            "message_type": message_type,
            "expect_response": expect_response,
            "metadata": metadata or {},
        }
        self.outbound_messages.append(payload)

        if self.outbound_message_handler is not None:
            await self._maybe_await(self.outbound_message_handler(payload))

        return payload

    async def on_pattern_start(self, *, context: Any, pattern: Any) -> None:
        if pattern.__class__.__name__ == "ReActPattern":
            self.active_react_step_id = f"react_{uuid4().hex[:8]}"
        await self._emit_pattern_trace(context=context, pattern=pattern, action="start")
        if self.tracer is None:
            return
        payload = {
            "name": self._pattern_trace_name(pattern),
            "execution_id": getattr(context, "execution_id", self.execution_id),
            "input": context.to_dict()
            if callable(getattr(context, "to_dict", None))
            else None,
            "metadata": {
                "execution_id": getattr(context, "execution_id", self.execution_id),
                "workspace_id": getattr(context, "workspace_id", None),
                "memory_session_id": getattr(context, "memory_session_id", None),
                "pattern": pattern.__class__.__name__,
            },
        }
        self.trace_runs.append(payload)
        start_trace = getattr(self.tracer, "start_trace", None)
        if callable(start_trace):
            await self._maybe_await(start_trace(**payload))

    async def on_pattern_end(
        self,
        *,
        context: Any,
        pattern: Any,
        result: dict[str, Any],
    ) -> None:
        await self._emit_pattern_trace(
            context=context,
            pattern=pattern,
            action="end",
            data={"result": result, "status": result.get("status")},
        )
        if pattern.__class__.__name__ == "ReActPattern":
            self.active_react_step_id = None
        if self.tracer is None:
            return
        finish_trace = getattr(self.tracer, "finish_trace", None)
        if callable(finish_trace):
            await self._maybe_await(
                finish_trace(
                    name=self._pattern_trace_name(pattern),
                    status="success",
                    output=result,
                    metadata={
                        "execution_id": getattr(
                            context, "execution_id", self.execution_id
                        ),
                        "pattern": pattern.__class__.__name__,
                    },
                )
            )

    async def on_pattern_error(
        self,
        *,
        context: Any,
        pattern: Any,
        error: Exception,
    ) -> None:
        await self._emit_trace_event(
            TraceEventType(TraceScope.TASK, TraceAction.ERROR, TraceCategory.GENERAL),
            task_id=self._task_id(context),
            step_id=self._step_id(context),
            data={
                "error_type": "agent_v2_pattern_error",
                "error_message": str(error),
                "pattern": pattern.__class__.__name__,
            },
        )
        if pattern.__class__.__name__ == "ReActPattern":
            self.active_react_step_id = None
        if self.tracer is None:
            return
        finish_trace = getattr(self.tracer, "finish_trace", None)
        if callable(finish_trace):
            await self._maybe_await(
                finish_trace(
                    name=self._pattern_trace_name(pattern),
                    status="error",
                    output={"error": str(error)},
                    metadata={
                        "execution_id": getattr(
                            context, "execution_id", self.execution_id
                        ),
                        "pattern": pattern.__class__.__name__,
                    },
                )
            )

    async def on_tool_start(self, *, tool_call: dict[str, Any]) -> None:
        await self._emit_trace_event(
            TraceEventType(TraceScope.ACTION, TraceAction.START, TraceCategory.TOOL),
            task_id=self._task_id_from_payload(tool_call),
            step_id=self._step_id_from_payload(tool_call),
            data={
                "tool_name": tool_call.get("name"),
                "tool_params": tool_call.get("args", {}),
                "tool_call_id": tool_call.get("id"),
            },
        )
        if self.tracer is None:
            return
        payload = {
            "name": f"tool.{tool_call['name']}",
            "execution_id": tool_call.get("id"),
            "metadata": {"arguments": tool_call.get("args", {})},
        }
        self.spans.append(payload)
        start_span = getattr(self.tracer, "start_span", None)
        if callable(start_span):
            await self._maybe_await(start_span(**payload))

    async def on_tool_end(self, *, tool_call: dict[str, Any], result: Any) -> None:
        success = self._tool_result_success(result)
        if not success:
            await self.on_tool_error(
                tool_call=tool_call, error=self._tool_result_error(result)
            )
            return

        await self._emit_trace_event(
            TraceEventType(TraceScope.ACTION, TraceAction.END, TraceCategory.TOOL),
            task_id=self._task_id_from_payload(tool_call),
            step_id=self._step_id_from_payload(tool_call),
            data={
                "tool_name": tool_call.get("name"),
                "tool_params": tool_call.get("args", {}),
                "tool_call_id": tool_call.get("id"),
                "result": result,
                "success": True,
            },
        )
        if self.tracer is None:
            return
        payload = {
            "name": f"tool.{tool_call['name']}",
            "status": "success",
            "output": result,
        }
        self.finished_spans.append(payload)
        finish_span = getattr(self.tracer, "finish_span", None)
        if callable(finish_span):
            await self._maybe_await(finish_span(**payload))

    async def on_tool_error(
        self,
        *,
        tool_call: dict[str, Any],
        error: Exception,
        result: Any | None = None,
    ) -> None:
        data = {
            "error_type": "agent_v2_tool_error",
            "error": str(error),
            "error_message": str(error),
            "tool_name": tool_call.get("name"),
            "tool_call_id": tool_call.get("id"),
        }
        if result is not None:
            data["result"] = result

        await self._emit_trace_event(
            TraceEventType(TraceScope.ACTION, TraceAction.ERROR, TraceCategory.TOOL),
            task_id=self._task_id_from_payload(tool_call),
            step_id=self._step_id_from_payload(tool_call),
            data=data,
        )
        if self.tracer is None:
            return
        payload = {
            "name": f"tool.{tool_call['name']}",
            "status": "error",
            "output": {"error": str(error)},
        }
        self.finished_spans.append(payload)
        finish_span = getattr(self.tracer, "finish_span", None)
        if callable(finish_span):
            await self._maybe_await(finish_span(**payload))

    def _tool_result_success(self, result: Any) -> bool:
        if isinstance(result, dict) and result.get("success") is False:
            return False
        return True

    def _tool_result_error(self, result: Any) -> Exception:
        if isinstance(result, dict):
            message = result.get("error") or result.get("message") or str(result)
        else:
            message = str(result)
        return RuntimeError(message)

    async def on_llm_start(
        self,
        *,
        context: Any,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        event_metadata = metadata or {}
        await self._emit_trace_event(
            TraceEventType(TraceScope.ACTION, TraceAction.START, TraceCategory.LLM),
            task_id=str(event_metadata.get("task_id") or self._task_id(context)),
            step_id=str(event_metadata.get("step_id") or self._step_id(context)),
            data={
                "context_messages_count": len(messages),
                "tools_count": len(tools or []),
                "context_preview": messages[-5:],
                **event_metadata,
            },
        )

    async def on_llm_end(
        self,
        *,
        context: Any,
        response: Any,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        event_metadata = metadata or {}
        usage = self._extract_token_usage(response)
        if usage is not None and callable(getattr(context, "record_llm_usage", None)):
            context.record_llm_usage(
                input_tokens=usage[0],
                output_tokens=usage[1],
                prompt_message_count=len(getattr(context, "messages", [])),
            )
        await self._emit_trace_event(
            TraceEventType(TraceScope.ACTION, TraceAction.END, TraceCategory.LLM),
            task_id=str(event_metadata.get("task_id") or self._task_id(context)),
            step_id=str(event_metadata.get("step_id") or self._step_id(context)),
            data={
                "response": self._short_response(response),
                "success": True,
                **(
                    {
                        "input_tokens": usage[0],
                        "output_tokens": usage[1],
                        "total_tokens": usage[0] + usage[1],
                    }
                    if usage is not None
                    else {}
                ),
                **event_metadata,
            },
        )

    def _extract_token_usage(self, response: Any) -> tuple[int, int] | None:
        usage = self._get_value(response, "usage")
        if usage is not None:
            input_tokens = self._first_int(
                usage, ("prompt_tokens", "input_tokens", "prompt_token_count")
            )
            output_tokens = self._first_int(
                usage,
                (
                    "completion_tokens",
                    "output_tokens",
                    "candidates_token_count",
                    "completion_token_count",
                ),
            )
            if input_tokens > 0 or output_tokens > 0:
                return input_tokens, output_tokens

        usage_metadata = self._get_value(response, "usage_metadata")
        if usage_metadata is not None:
            input_tokens = self._first_int(
                usage_metadata, ("prompt_token_count", "prompt_tokens", "input_tokens")
            )
            output_tokens = self._first_int(
                usage_metadata,
                ("candidates_token_count", "completion_tokens", "output_tokens"),
            )
            if input_tokens > 0 or output_tokens > 0:
                return input_tokens, output_tokens

        return None

    def _first_int(self, source: Any, keys: tuple[str, ...]) -> int:
        for key in keys:
            value = self._get_value(source, key)
            if isinstance(value, int):
                return value
            if isinstance(value, float):
                return int(value)
        return 0

    def _get_value(self, source: Any, key: str) -> Any:
        if isinstance(source, dict):
            return source.get(key)
        return getattr(source, key, None)

    async def on_llm_error(
        self,
        *,
        context: Any,
        error: Exception,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        event_metadata = metadata or {}
        await self._emit_trace_event(
            TraceEventType(TraceScope.ACTION, TraceAction.ERROR, TraceCategory.LLM),
            task_id=str(event_metadata.get("task_id") or self._task_id(context)),
            step_id=str(event_metadata.get("step_id") or self._step_id(context)),
            data={
                "error_type": "agent_v2_llm_error",
                "error": str(error),
                "error_message": str(error),
                "success": False,
                **event_metadata,
            },
        )

    async def compact_context_if_needed(
        self,
        *,
        context: Any,
        llm: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> Any:
        compact_if_needed = getattr(context, "compact_if_needed", None)
        if not callable(compact_if_needed):
            return None

        result = compact_if_needed(llm)
        if not getattr(result, "compacted", False):
            return result

        event_metadata = metadata or {}
        compact_data = {
            "agent_runtime": "v2",
            "compact_type": "execution_context",
            "strategy": getattr(result, "strategy", None),
            "original_count": getattr(result, "original_count", None),
            "final_count": getattr(result, "final_count", None),
            **(getattr(result, "metadata", None) or {}),
            **event_metadata,
        }
        await self._emit_trace_event(
            TraceEventType(TraceScope.ACTION, TraceAction.START, TraceCategory.COMPACT),
            task_id=str(event_metadata.get("task_id") or self._task_id(context)),
            step_id=str(event_metadata.get("step_id") or self._step_id(context)),
            data=compact_data,
        )
        await self._emit_trace_event(
            TraceEventType(TraceScope.ACTION, TraceAction.END, TraceCategory.COMPACT),
            task_id=str(event_metadata.get("task_id") or self._task_id(context)),
            step_id=str(event_metadata.get("step_id") or self._step_id(context)),
            data={**compact_data, "success": True},
        )
        return result

    async def on_dag_step_start(
        self,
        *,
        context: Any,
        step_id: str,
        data: dict[str, Any] | None = None,
    ) -> None:
        await self._emit_trace_event(
            TraceEventType(TraceScope.STEP, TraceAction.START, TraceCategory.DAG),
            task_id=self._task_id(context),
            step_id=step_id,
            data=data or {},
        )

    async def on_dag_step_end(
        self,
        *,
        context: Any,
        step_id: str,
        data: dict[str, Any] | None = None,
    ) -> None:
        await self._emit_trace_event(
            TraceEventType(TraceScope.STEP, TraceAction.END, TraceCategory.DAG),
            task_id=self._task_id(context),
            step_id=step_id,
            data=data or {},
        )

    async def on_dag_execution(
        self,
        *,
        context: Any,
        phase: str,
        data: dict[str, Any] | None = None,
    ) -> None:
        await self._emit_trace_event(
            TraceEventType(TraceScope.TASK, TraceAction.UPDATE, TraceCategory.DAG),
            task_id=self._task_id(context),
            data={"phase": phase, **(data or {})},
        )

    def _build_checkpoint_payload(
        self,
        *,
        label: str,
        context: Any,
        pattern: Any,
        status: str | None,
        metadata: dict[str, Any] | None,
    ) -> dict[str, Any]:
        execution_id = getattr(context, "execution_id", None) or self.execution_id
        pattern_state: dict[str, Any] | None = None
        get_state = getattr(pattern, "get_state", None)
        if callable(get_state):
            pattern_state = get_state()

        context_payload = (
            context.to_dict() if callable(getattr(context, "to_dict", None)) else None
        )
        execution_snapshot: dict[str, Any] | None = None
        get_execution_snapshot = getattr(pattern, "get_execution_snapshot", None)
        if callable(get_execution_snapshot):
            execution_snapshot = get_execution_snapshot(context)

        payload = {
            "type": "checkpoint",
            "label": label,
            "execution_id": execution_id,
            "pattern": pattern.__class__.__name__,
            "pattern_state": pattern_state,
            "context": context_payload,
            "status": status or getattr(pattern, "status", None),
            "metadata": metadata or {},
        }
        if execution_snapshot is not None:
            payload["execution_snapshot"] = execution_snapshot
        return payload

    async def _emit_checkpoint(self, payload: dict[str, Any]) -> None:
        if self.tracer is None:
            return

        checkpoint = getattr(self.tracer, "checkpoint", None)
        if callable(checkpoint):
            await self._maybe_await(checkpoint(**payload))
            return

        write_checkpoint = getattr(self.tracer, "write_checkpoint", None)
        if callable(write_checkpoint):
            await self._maybe_await(write_checkpoint(payload))
            return

        trace_event = getattr(self.tracer, "trace_event", None)
        if callable(trace_event):
            await self._maybe_await(
                trace_event(
                    self._checkpoint_trace_event_type(trace_event),
                    task_id=str(payload.get("execution_id") or self.execution_id),
                    data=payload,
                )
            )

    async def _maybe_await(self, result: Any) -> None:
        if inspect.isawaitable(result):
            await result

    async def _emit_pattern_trace(
        self,
        *,
        context: Any,
        pattern: Any,
        action: str,
        data: dict[str, Any] | None = None,
    ) -> None:
        category = self._pattern_category(pattern)
        if category is None:
            return
        event_action = TraceAction.START if action == "start" else TraceAction.END
        await self._emit_trace_event(
            TraceEventType(TraceScope.TASK, event_action, category),
            task_id=self._task_id(context),
            step_id=self._step_id(context),
            data={
                "agent_runtime": "v2",
                "pattern": pattern.__class__.__name__,
                **(data or {}),
            },
        )

    async def _emit_trace_event(
        self,
        event_type: Any,
        *,
        task_id: str | None = None,
        step_id: str | None = None,
        data: dict[str, Any] | None = None,
    ) -> None:
        if self.tracer is None:
            return
        trace_event = getattr(self.tracer, "trace_event", None)
        if not callable(trace_event):
            return
        try:
            await self._maybe_await(
                trace_event(
                    event_type,
                    task_id=task_id or self.execution_id,
                    step_id=step_id,
                    data=data or {},
                )
            )
        except Exception:
            # UI trace events are best-effort; checkpoint persistence remains strict.
            return

    def _pattern_category(self, pattern: Any) -> TraceCategory | None:
        name = pattern.__class__.__name__
        if name == "DAGPattern":
            return TraceCategory.DAG
        if name == "ReActPattern":
            return TraceCategory.REACT
        return None

    def _task_id(self, context: Any) -> str | None:
        return str(getattr(context, "execution_id", None) or self.execution_id or "")

    def _step_id(self, context: Any) -> str:
        metadata = getattr(context, "metadata", None)
        if isinstance(metadata, dict) and metadata.get("dag_step_id"):
            return str(metadata["dag_step_id"])
        if self.active_react_step_id:
            return self.active_react_step_id
        return self._task_id(context) or "root"

    def _task_id_from_payload(self, payload: dict[str, Any]) -> str | None:
        return str(payload.get("task_id") or self.execution_id or "")

    def _step_id_from_payload(self, payload: dict[str, Any]) -> str:
        return str(
            payload.get("step_id")
            or payload.get("dag_step_id")
            or self.active_react_step_id
            or payload.get("task_id")
            or self.execution_id
            or "root"
        )

    def _short_response(self, response: Any) -> Any:
        if isinstance(response, str):
            return response
        if isinstance(response, dict):
            return {
                key: response.get(key)
                for key in ("content", "answer", "output", "message", "tool_calls")
                if key in response
            }
        return str(response)

    def _pattern_trace_name(self, pattern: Any) -> str:
        del pattern
        return "agent.task"

    def _checkpoint_trace_event_type(self, trace_event: Any) -> Any:
        del trace_event
        # Runtime checkpoints are task-scoped progress events. Durable checkpoint
        # persistence uses TraceCheckpointStore, which emits system-scoped events.
        return TraceEventType(
            TraceScope.TASK,
            TraceAction.UPDATE,
            TraceCategory.GENERAL,
        )


def load_pattern_checkpoint(pattern: Any, checkpoint: dict[str, Any] | None) -> None:
    """Restore a pattern from a checkpoint when the payload matches it."""
    if not checkpoint:
        return
    if checkpoint.get("pattern") not in {None, pattern.__class__.__name__}:
        return

    load_state = getattr(pattern, "load_state", None)
    state = checkpoint.get("pattern_state")
    if callable(load_state) and isinstance(state, dict):
        load_state(state)
