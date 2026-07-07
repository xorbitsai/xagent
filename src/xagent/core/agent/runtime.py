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
    normalize_llm_trace_payload,
)
from ..model.chat.basic.base import BaseLLM
from ..model.chat.types import ChunkType
from .streaming import merge_streamed_tool_call_arguments


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
    last_final_answer_stream_message_id: str | None = None
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

    async def stream_final_answer(self, llm: Any, **kwargs: Any) -> Any:
        """Stream only the final user-facing answer through the outbound boundary."""

        if self.outbound_message_handler is None or not self._has_native_stream_chat(
            llm
        ):
            return await self.run_llm_call(llm, **kwargs)

        from .pattern.final_answer_stream import FinalAnswerStreamSession

        stream = FinalAnswerStreamSession(self)
        if await stream.start() is None:
            return await self.run_llm_call(llm, **kwargs)

        async def emit_text_delta(chunk: Any) -> None:
            delta = self._chunk_text_delta(chunk)
            if delta:
                await stream.emit_delta(delta)

        try:
            response = await self.run_streaming_llm_call(
                llm, on_chunk=emit_text_delta, **kwargs
            )
        except Exception as exc:
            await stream.fail(str(exc))
            raise
        content = self._response_content(response)

        await stream.finish(content)
        return response

    async def run_streaming_llm_call(
        self,
        llm: Any,
        *,
        on_chunk: Callable[[Any], Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Run a native streaming LLM call and reconstruct a chat-like response."""

        stream_chat = getattr(llm, "stream_chat", None)
        if not callable(stream_chat) or not self._has_native_stream_chat(llm):
            return await self.run_llm_call(llm, **kwargs)

        async def consume_stream() -> Any:
            content_parts: list[str] = []
            tool_call_chunks: dict[int, dict[str, Any]] = {}
            usage_payload: dict[str, Any] = {}
            provider_payload: dict[str, Any] = {}
            saw_payload_chunk = False
            async for chunk in stream_chat(**kwargs):
                await self._raise_if_interrupted("interrupted during LLM stream")
                self._raise_for_stream_error(chunk)
                text_delta = self._chunk_text_delta(chunk)
                if text_delta:
                    saw_payload_chunk = True
                    content_parts.append(text_delta)
                chunk_tool_calls = self._chunk_tool_calls(chunk)
                if chunk_tool_calls:
                    saw_payload_chunk = True
                    self._merge_tool_call_chunks(tool_call_chunks, chunk_tool_calls)
                chunk_usage = self._chunk_usage(chunk)
                if chunk_usage:
                    self._merge_usage(usage_payload, chunk_usage)
                self._merge_provider_payload(provider_payload, chunk)
                if on_chunk is not None:
                    await self._maybe_await(on_chunk(chunk))

            content = "".join(content_parts)
            tool_calls = [
                tool_call_chunks[index] for index in sorted(tool_call_chunks.keys())
            ]
            if tool_calls:
                response: dict[str, Any] = {
                    "content": content,
                    "tool_calls": tool_calls,
                }
                if usage_payload:
                    response["usage"] = usage_payload
                if provider_payload:
                    response.update(provider_payload)
                return response
            if not saw_payload_chunk:
                return await self.run_llm_call(llm, **kwargs)
            if usage_payload:
                response = {
                    "content": content,
                    "usage": usage_payload,
                }
                if provider_payload:
                    response.update(provider_payload)
                return response
            if provider_payload:
                return {"content": content, **provider_payload}
            return content

        task: asyncio.Future[Any] = asyncio.ensure_future(consume_stream())
        self._active_llm_tasks.add(task)
        try:
            return await task
        except asyncio.CancelledError as exc:
            if self._interrupt_requested:
                raise LLMCallInterrupted(
                    self.interrupt_reason or "interrupted during LLM stream"
                ) from exc
            raise
        finally:
            self._active_llm_tasks.discard(task)

    async def _raise_if_interrupted(self, message: str) -> None:
        if await self.should_interrupt():
            raise LLMCallInterrupted(self.interrupt_reason or message)

    def _raise_for_stream_error(self, chunk: Any) -> None:
        chunk_type = getattr(chunk, "type", None)
        is_error = callable(getattr(chunk, "is_error", None)) and chunk.is_error()
        if chunk_type == ChunkType.ERROR or is_error:
            raise RuntimeError(getattr(chunk, "content", "") or "LLM stream failed")

    def _chunk_text_delta(self, chunk: Any) -> str:
        chunk_type = getattr(chunk, "type", None)
        is_token = callable(getattr(chunk, "is_token", None)) and chunk.is_token()
        if chunk_type != ChunkType.TOKEN and not is_token:
            return ""
        return str(getattr(chunk, "delta", "") or getattr(chunk, "content", "") or "")

    def _chunk_tool_calls(self, chunk: Any) -> list[dict[str, Any]]:
        chunk_type = getattr(chunk, "type", None)
        is_tool_call = (
            callable(getattr(chunk, "is_tool_call", None)) and chunk.is_tool_call()
        )
        if chunk_type != ChunkType.TOOL_CALL and not is_tool_call:
            return []
        tool_calls = getattr(chunk, "tool_calls", None)
        return list(tool_calls or [])

    def _chunk_usage(self, chunk: Any) -> dict[str, Any]:
        chunk_type = getattr(chunk, "type", None)
        is_usage = callable(getattr(chunk, "is_usage", None)) and chunk.is_usage()
        if chunk_type != ChunkType.USAGE and not is_usage:
            return {}
        usage = getattr(chunk, "usage", None)
        if not usage:
            return {}
        if isinstance(usage, dict):
            return dict(usage)
        payload: dict[str, Any] = {}
        for key in (
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "input_tokens",
            "output_tokens",
            "prompt_token_count",
            "completion_token_count",
            "candidates_token_count",
        ):
            value = getattr(usage, key, None)
            if value is not None:
                payload[key] = value
        return payload

    def _merge_usage(
        self,
        current: dict[str, Any],
        incoming: dict[str, Any],
    ) -> None:
        for key, value in incoming.items():
            if isinstance(value, int):
                current[key] = value
            elif isinstance(value, float):
                current[key] = int(value)
            elif value is not None:
                current[key] = value

    def _merge_provider_payload(
        self,
        current: dict[str, Any],
        chunk: Any,
    ) -> None:
        raw = getattr(chunk, "raw", None)
        model_dump = getattr(raw, "model_dump", None)
        if callable(model_dump):
            raw = model_dump()
        if not isinstance(raw, dict):
            return
        for key in ("reasoning_content", "reasoning"):
            if key in raw and raw[key] is not None:
                current[key] = raw[key]
        provider_state = raw.get("_xagent_provider_state")
        if isinstance(provider_state, dict):
            current["_xagent_provider_state"] = provider_state

    def _merge_tool_call_chunks(
        self,
        accumulator: dict[int, dict[str, Any]],
        tool_calls: list[Any],
    ) -> None:
        for position, raw_tool_call in enumerate(tool_calls):
            tool_call = self._tool_call_to_dict(raw_tool_call)
            index_value = tool_call.get("index", position)
            index = index_value if isinstance(index_value, int) else position
            current = accumulator.setdefault(index, {})
            self._merge_tool_call_dict(current, tool_call)

    def _tool_call_to_dict(self, tool_call: Any) -> dict[str, Any]:
        if isinstance(tool_call, dict):
            return dict(tool_call)
        function_payload = getattr(tool_call, "function", None)
        payload: dict[str, Any] = {
            key: getattr(tool_call, key)
            for key in ("id", "index", "type")
            if getattr(tool_call, key, None) is not None
        }
        if function_payload is not None:
            payload["function"] = {
                key: getattr(function_payload, key)
                for key in ("name", "arguments")
                if getattr(function_payload, key, None) is not None
            }
        return payload

    def _merge_tool_call_dict(
        self,
        current: dict[str, Any],
        incoming: dict[str, Any],
    ) -> None:
        for key, value in incoming.items():
            if key == "function" and isinstance(value, dict):
                function_payload = current.setdefault("function", {})
                if isinstance(function_payload, dict):
                    self._merge_tool_call_function(function_payload, value)
                continue
            if value is not None:
                current[key] = value

    def _merge_tool_call_function(
        self,
        current: dict[str, Any],
        incoming: dict[str, Any],
    ) -> None:
        name = incoming.get("name")
        if name:
            current["name"] = name

        arguments = incoming.get("arguments")
        if not isinstance(arguments, str):
            if arguments is not None:
                current["arguments"] = arguments
            return

        existing = current.get("arguments")
        if not isinstance(existing, str) or not existing:
            current["arguments"] = arguments
        else:
            mode = incoming.get("arguments_mode") or incoming.get(
                "arguments_stream_mode"
            )
            current["arguments"] = merge_streamed_tool_call_arguments(
                existing,
                arguments,
                mode=str(mode) if isinstance(mode, str) else None,
            )

    def _response_content(self, response: Any) -> str:
        if isinstance(response, str):
            return response
        if isinstance(response, dict):
            value = (
                response.get("content")
                or response.get("answer")
                or response.get("output")
                or response.get("message")
                or ""
            )
            return str(value)
        return str(response)

    async def _emit_outbound(self, payload: dict[str, Any]) -> None:
        if self.outbound_message_handler is not None:
            await self._maybe_await(self.outbound_message_handler(payload))

    async def start_final_answer_stream(self) -> str | None:
        if self.outbound_message_handler is None:
            return None
        message_id = f"final_answer_{uuid4().hex}"
        self.last_final_answer_stream_message_id = None
        await self._emit_outbound(
            {
                "type": "final_answer_start",
                "message_id": message_id,
                "task_id": self.execution_id,
            }
        )
        return message_id

    async def emit_final_answer_delta(self, message_id: str, delta: str) -> None:
        if not delta:
            return
        await self._emit_outbound(
            {
                "type": "final_answer_delta",
                "message_id": message_id,
                "task_id": self.execution_id,
                "delta": delta,
            }
        )

    async def end_final_answer_stream(self, message_id: str, content: str) -> None:
        self.last_final_answer_stream_message_id = message_id
        await self._emit_outbound(
            {
                "type": "final_answer_end",
                "message_id": message_id,
                "task_id": self.execution_id,
                "content": content,
            }
        )

    async def fail_final_answer_stream(self, message_id: str, error: str) -> None:
        if self.last_final_answer_stream_message_id == message_id:
            self.last_final_answer_stream_message_id = None
        await self._emit_outbound(
            {
                "type": "final_answer_error",
                "message_id": message_id,
                "task_id": self.execution_id,
                "error": error,
            }
        )

    def _has_native_stream_chat(self, llm: Any) -> bool:
        stream_chat = getattr(type(llm), "stream_chat", None)
        return stream_chat is not None and stream_chat is not BaseLLM.stream_chat

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
        visible: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Emit an agent-to-user message through the runtime boundary."""

        outbound_metadata = dict(metadata or {})
        step_id = (
            outbound_metadata.get("step_id")
            or outbound_metadata.get("dag_step_id")
            or self.active_react_step_id
        )
        if step_id:
            outbound_metadata.setdefault("step_id", str(step_id))

        payload = {
            "type": "agent_message",
            "execution_id": self.execution_id,
            "message": message,
            "message_type": message_type,
            "expect_response": expect_response,
            "visible": visible,
            "metadata": outbound_metadata,
        }
        if step_id:
            payload["step_id"] = str(step_id)
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
                "error_type": "agent_pattern_error",
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
        data = {
            "tool_name": tool_call.get("name"),
            "tool_params": tool_call.get("args", {}),
            "tool_call_id": tool_call.get("id"),
        }
        assistant_content = tool_call.get("assistant_content")
        if isinstance(assistant_content, str) and assistant_content.strip():
            data["assistant_content"] = assistant_content.strip()

        await self._emit_trace_event(
            TraceEventType(TraceScope.ACTION, TraceAction.START, TraceCategory.TOOL),
            task_id=self._task_id_from_payload(tool_call),
            step_id=self._step_id_from_payload(tool_call),
            data=data,
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
            "error_type": "agent_tool_error",
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
        if isinstance(result, dict):
            if result.get("success") is False:
                return False
            status = result.get("status")
            if isinstance(status, str) and status.lower() == "error":
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
        context_data: dict[str, Any] = {
            "context_messages_count": len(messages),
            "tools_count": len(tools or []),
            "context_preview": messages[-5:],
            **event_metadata,
        }
        # Surface current context size vs. the compaction threshold so the UI
        # can show a context-usage gauge. Uses the same estimate that drives the
        # compaction decision, so the gauge fills exactly when compaction fires.
        estimate = getattr(context, "estimate_context_tokens", None)
        if callable(estimate):
            try:
                context_data["context_tokens"] = estimate()
            except Exception:  # noqa: BLE001 - gauge metadata must never break the call
                pass
        compact_config = getattr(context, "compact_config", None)
        threshold = getattr(compact_config, "threshold", None)
        if isinstance(threshold, int) and threshold > 0:
            context_data["context_threshold"] = threshold
        await self._emit_trace_event(
            TraceEventType(TraceScope.ACTION, TraceAction.START, TraceCategory.LLM),
            task_id=str(event_metadata.get("task_id") or self._task_id(context)),
            step_id=str(event_metadata.get("step_id") or self._step_id(context)),
            data=context_data,
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
                "error_type": "agent_llm_error",
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
        llm_compact_request_if_needed = getattr(
            context, "build_llm_compact_request_if_needed", None
        )
        compact_with_llm_response = getattr(context, "compact_with_llm_response", None)
        compact_if_needed = getattr(context, "compact_if_needed", None)
        result = None
        if (
            llm is not None
            and callable(getattr(llm, "chat", None))
            and callable(llm_compact_request_if_needed)
            and callable(compact_with_llm_response)
        ):
            request = llm_compact_request_if_needed()
            if request is not None:
                request_metadata = request.get("metadata") or {}
                llm_metadata = {**request_metadata, "purpose": "context_compaction"}
                try:
                    await self.on_llm_start(
                        context=context,
                        messages=request["messages"],
                        metadata=llm_metadata,
                    )
                    response = await self.run_llm_call(
                        llm,
                        messages=request["messages"],
                        max_tokens=request["max_tokens"],
                    )
                    await self.on_llm_end(
                        context=context,
                        response=response,
                        metadata=llm_metadata,
                    )
                    result = compact_with_llm_response(
                        response,
                        llm=llm,
                        original_tokens=request.get("original_tokens"),
                    )
                    for key, value in request_metadata.items():
                        result.metadata.setdefault(key, value)
                except LLMCallInterrupted:
                    raise
                except Exception as exc:  # noqa: BLE001
                    await self.on_llm_error(
                        context=context,
                        error=exc,
                        metadata=llm_metadata,
                    )
                    if not callable(compact_if_needed):
                        raise
                    result = compact_if_needed()
                    result.metadata["llm_compact_error"] = str(exc)
                    result.metadata["fallback_strategy"] = result.strategy
                    result.metadata.update(request_metadata)

        if result is None or not getattr(result, "compacted", False):
            if not callable(compact_if_needed):
                return result
            result = compact_if_needed(llm)

        if result is None:
            return None
        if inspect.isawaitable(result):
            result = await result
        if not getattr(result, "compacted", False):
            return result

        event_metadata = metadata or {}
        compact_data = {
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
        # Cap LLM event payloads at the tracer boundary. The normalizer
        # preserves reserved control / routing / metrics fields and only
        # truncates bulky content (messages, response, tool_calls, ...).
        # Non-LLM categories (TOOL / DAG / REACT / COMPACT / GENERAL)
        # pass through unchanged.
        if data and getattr(event_type, "category", None) == TraceCategory.LLM:
            data = normalize_llm_trace_payload(data)
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
