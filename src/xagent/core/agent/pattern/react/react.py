from __future__ import annotations

import asyncio
import inspect
import json
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from enum import Enum
from typing import Any, cast

from ...context.enrichment import (
    enrich_context_with_memory,
    enrich_context_with_skill,
    generate_and_store_react_memory,
    latest_user_text,
)
from ...language import final_answer_language_rule
from ...result import unwrap_final_answer_content
from ...runtime import LLMCallInterrupted, PatternRuntime
from ..base import AgentPattern, PatternResult
from ..final_answer_stream import ReActFinalAnswerStreamer


class ReActReasoningMode(str, Enum):
    """Reasoning strategy used by ReActPattern."""

    TOOL_CALLING = "tool_calling"
    REASONING_ACTION = "reasoning_action"


@dataclass
class ToolCallRecord:
    """Serializable ledger entry for a tool call."""

    tool_call_id: str
    tool_name: str
    args: dict[str, Any]
    args_hash: str
    status: str
    result: Any = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_call_id": self.tool_call_id,
            "tool_name": self.tool_name,
            "args": self.args,
            "args_hash": self.args_hash,
            "status": self.status,
            "result": self.result,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ToolCallRecord":
        return cls(
            tool_call_id=str(data["tool_call_id"]),
            tool_name=str(data["tool_name"]),
            args=dict(data.get("args") or {}),
            args_hash=str(data.get("args_hash", "")),
            status=str(data.get("status", "pending")),
            result=data.get("result"),
            error=data.get("error"),
        )


def _normalize_ask_user_interactions(interactions: Any) -> list[dict[str, Any]]:
    """Normalize common model variants into the frontend interaction contract."""

    if not isinstance(interactions, list):
        return []

    normalized: list[dict[str, Any]] = []
    for index, interaction in enumerate(interactions):
        if not isinstance(interaction, dict):
            continue

        item = dict(interaction)
        field = item.get("field") or item.get("id") or item.get("name")
        if not isinstance(field, str) or not field.strip():
            field = f"response_{index}"
        item["field"] = field.strip()

        if "options" not in item and isinstance(item.get("actions"), list):
            item["options"] = item["actions"]

        options = item.get("options")
        if isinstance(options, list):
            item["options"] = [
                {
                    key: value
                    for key, value in {
                        "label": option.get("label"),
                        "value": option.get("value"),
                        "description": option.get("description"),
                        "action_type": option.get("action_type"),
                    }.items()
                    if value is not None
                }
                for option in options
                if isinstance(option, dict)
                and isinstance(option.get("label"), str)
                and option.get("label")
                and isinstance(option.get("value"), str)
                and option.get("value")
            ]

        normalized.append(item)

    return normalized


class ReActPattern(AgentPattern):
    """Minimal ReAct loop for the execution runtime."""

    def __init__(
        self,
        llm: Any | None = None,
        *,
        # Intentionally high for interactive and long-running agent tasks; callers
        # can pass a lower value when they need stricter cost or latency bounds.
        max_iterations: int = 200,
        tool_choice: str | dict[str, Any] | None = "required",
        reasoning_mode: ReActReasoningMode | str = ReActReasoningMode.TOOL_CALLING,
        finalize_after_tool_result: bool = False,
    ) -> None:
        self.llm = llm
        self.max_iterations = max_iterations
        self.tool_choice = tool_choice
        self.reasoning_mode = ReActReasoningMode(reasoning_mode)
        self.finalize_after_tool_result = finalize_after_tool_result
        self.status = "idle"
        self.current_iteration = 0
        self.last_response: Any = None
        self.pending_tool_calls: list[dict[str, Any]] = []
        self.pending_tool_call_content: dict[str, str] = {}
        self.tool_ledger: dict[str, ToolCallRecord] = {}
        self.force_final_answer_next = False
        self.waiting_for_user_request: dict[str, Any] | None = None
        self.task_text: str | None = None
        self._memory_store: Any | None = None

    async def run(
        self,
        context: Any,
        tools: list[Any],
        llm: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        runtime = kwargs.get("runtime")
        if runtime is None:
            runtime = PatternRuntime(
                execution_id=getattr(context, "execution_id", None)
            )
        elif getattr(runtime, "execution_id", None) is None:
            setattr(runtime, "execution_id", getattr(context, "execution_id", None))

        active_llm = llm or self.llm
        compact_llm = kwargs.get("compact_llm")
        if active_llm is None:
            return PatternResult(
                success=False,
                error="ReActPattern requires an llm instance.",
            ).to_dict()

        await runtime.on_pattern_start(context=context, pattern=self)
        waiting_result = await self._resume_waiting_for_user_if_needed(
            context=context,
            runtime=runtime,
        )
        if waiting_result is not None:
            await runtime.on_pattern_end(
                context=context,
                pattern=self,
                result=waiting_result,
            )
            return waiting_result

        if self.reasoning_mode == ReActReasoningMode.REASONING_ACTION:
            self.status = "failed"
            result = PatternResult(
                success=False,
                error=(
                    "ReActPattern reasoning_action mode is reserved for a future "
                    "implementation; use tool_calling mode for this release."
                ),
                metadata={
                    "status": self.status,
                    "reasoning_mode": self.reasoning_mode.value,
                    "error_type": "not_implemented",
                },
            ).to_dict()
            await runtime.on_pattern_end(context=context, pattern=self, result=result)
            return result

        try:
            task_text = self._task_text(context)
            self._memory_store = kwargs.get("memory_store")
            await enrich_context_with_memory(
                context=context,
                query=task_text,
                category="react_memory",
                memory_store=self._memory_store,
                runtime=runtime,
                similarity_threshold=kwargs.get("memory_similarity_threshold"),
            )
            await enrich_context_with_skill(
                context=context,
                task=task_text,
                llm=active_llm,
                skill_manager=kwargs.get("skill_manager"),
                runtime=runtime,
                allowed_skills=kwargs.get("allowed_skills"),
            )
            result = await self._run_tool_calling_loop(
                context=context,
                tools=tools,
                llm=active_llm,
                compact_llm=compact_llm,
                runtime=runtime,
            )
        except LLMCallInterrupted:
            interrupted = await self._interrupt_if_requested(
                runtime=runtime,
                context=context,
                label="during_enrichment",
            )
            if interrupted is None:
                raise
            result = interrupted
        except Exception as exc:
            await runtime.on_pattern_error(context=context, pattern=self, error=exc)
            raise

        await runtime.on_pattern_end(context=context, pattern=self, result=result)
        return result

    async def _run_tool_calling_loop(
        self,
        *,
        context: Any,
        tools: list[Any],
        llm: Any,
        compact_llm: Any | None,
        runtime: PatternRuntime,
    ) -> dict[str, Any]:
        self.status = "thinking"
        base_tool_schemas = (
            []
            if self.tool_choice == "none"
            else self._tool_schemas_with_builtin_controls(tools)
        )

        for iteration in range(self.current_iteration, self.max_iterations):
            self.current_iteration = iteration
            if self.pending_tool_calls:
                self._ensure_pending_tool_call_envelope(context)
                pending_result = await self._execute_pending_tool_calls(
                    context=context,
                    tools=tools,
                    llm=llm,
                    runtime=runtime,
                )
                if pending_result is not None:
                    return pending_result
                self.current_iteration = iteration + 1
                self.status = "thinking"
                continue

            force_final_answer_now = self.force_final_answer_next or (
                self.finalize_after_tool_result
                and not self.pending_tool_calls
                and self._latest_tool_result_success(context)
            )
            tool_schemas = [] if force_final_answer_now else base_tool_schemas
            interrupted = await self._interrupt_if_requested(
                runtime=runtime,
                context=context,
                label="before_llm",
            )
            if interrupted is not None:
                return interrupted

            await runtime.compact_context_if_needed(
                context=context,
                llm=compact_llm,
                metadata={"iteration": iteration},
            )

            messages = self._messages_for_llm(
                context,
                has_tools=bool(tool_schemas),
                force_final_answer=force_final_answer_now,
                tool_names=self._schema_tool_names(tool_schemas),
            )
            await runtime.checkpoint("before_llm", context=context, pattern=self)
            await runtime.on_llm_start(
                context=context,
                messages=messages,
                tools=tool_schemas or None,
                metadata={"iteration": iteration},
            )
            answer_streamer: ReActFinalAnswerStreamer | None = None
            try:
                llm_kwargs = {
                    "messages": messages,
                    "tools": tool_schemas or None,
                    "tool_choice": self.tool_choice if tool_schemas else None,
                }
                if tool_schemas:
                    answer_streamer = ReActFinalAnswerStreamer(runtime)
                    response = await runtime.run_streaming_llm_call(
                        llm,
                        on_chunk=answer_streamer.handle_chunk,
                        **llm_kwargs,
                    )
                else:
                    response = await runtime.stream_final_answer(llm, **llm_kwargs)
            except LLMCallInterrupted:
                if answer_streamer is not None:
                    await answer_streamer.fail("interrupted during LLM stream")
                interrupted = await self._interrupt_if_requested(
                    runtime=runtime,
                    context=context,
                    label="during_llm",
                )
                if interrupted is not None:
                    return interrupted
                raise
            except Exception as exc:
                if answer_streamer is not None:
                    await answer_streamer.fail(str(exc))
                raise
            await runtime.on_llm_end(
                context=context,
                response=response,
                metadata={"iteration": iteration},
            )
            self.last_response = response
            normalized = self._normalize_llm_response(response)
            if force_final_answer_now and not normalized.get("tool_calls"):
                normalized["done"] = True

            assistant_content = normalized.get("content")
            if assistant_content is not None or normalized.get("tool_calls"):
                context.add_assistant_message(
                    assistant_content or "",
                    tool_calls=normalized.get("raw_tool_calls")
                    or normalized.get("tool_calls"),
                )

            tool_calls = normalized.get("tool_calls", [])
            if answer_streamer is not None:
                await self._finish_streamed_answer_if_final(
                    answer_streamer=answer_streamer,
                    assistant_content=assistant_content,
                    tool_calls=tool_calls,
                )
            if tool_calls:
                self._remember_tool_call_content(tool_calls, assistant_content)
                self.status = "acting"
                self.pending_tool_calls = list(tool_calls)
                await runtime.checkpoint("after_llm", context=context, pattern=self)
                pending_result = await self._execute_pending_tool_calls(
                    context=context,
                    tools=tools,
                    llm=llm,
                    runtime=runtime,
                )
                if pending_result is not None:
                    return pending_result
                self.current_iteration = iteration + 1
                self.status = "thinking"
                continue

            await runtime.checkpoint("after_llm", context=context, pattern=self)
            if normalized.get("done", True):
                return await self._finalize_success(
                    context=context,
                    llm=llm,
                    runtime=runtime,
                    response=assistant_content or normalized.get("raw"),
                )

        self.status = "max_iterations"
        await runtime.checkpoint("max_iterations", context=context, pattern=self)
        return PatternResult(
            success=False,
            error="ReActPattern reached max iterations without a final answer.",
            metadata={"iterations": self.max_iterations, "status": self.status},
        ).to_dict()

    async def _finish_streamed_answer_if_final(
        self,
        *,
        answer_streamer: ReActFinalAnswerStreamer,
        assistant_content: Any,
        tool_calls: list[dict[str, Any]],
    ) -> None:
        if not answer_streamer.started:
            return
        final_answer = self._final_answer_tool_content(tool_calls)
        if final_answer is not None and len(tool_calls) == 1:
            await answer_streamer.finish(final_answer)
            return
        if not tool_calls and assistant_content is not None:
            await answer_streamer.finish(str(assistant_content))

    def _final_answer_tool_content(
        self,
        tool_calls: list[dict[str, Any]],
    ) -> str | None:
        for tool_call in tool_calls:
            if tool_call.get("name") != "final_answer":
                continue
            args = tool_call.get("args")
            if isinstance(args, dict):
                return str(args.get("answer", ""))
        return None

    def _messages_for_llm(
        self,
        context: Any,
        *,
        has_tools: bool,
        force_final_answer: bool = False,
        tool_names: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        messages = list(context.get_messages_for_llm())
        if force_final_answer:
            instruction = (
                "You have already received the tool result needed for the current "
                "step. Do not call tools again. Produce the final answer for this "
                "step using the latest tool result. "
                f"{final_answer_language_rule()}"
            )
        elif has_tools:
            available_tools = ", ".join(tool_names or []) or "(none)"
            current_date = datetime.now(timezone.utc).date().isoformat()
            instruction = (
                "Use available tools when the user asks you to generate, compute, run, "
                "execute, inspect, read, write, or otherwise produce a concrete result "
                "that a tool can determine. After a successful tool call, base the "
                "final answer on the latest tool result instead of repeating the same "
                "tool work. When the current task is complete, call the final_answer "
                "tool exactly once instead of calling another work tool or returning "
                "plain assistant text. Do not write assistant text in the same "
                "response as a work tool call; call the tool directly. If a tool "
                "needs missing information from the user, call ask_user_question; do "
                "not ask the question as plain assistant text. If the latest user "
                "message explicitly asks you to call a named available tool, call "
                "that tool instead of paraphrasing the request. If a tool "
                "fails, retry with a corrected call when possible; "
                "otherwise explain the failure instead of presenting an unverified "
                "tutorial or example. Treat the latest user message as the controlling "
                "instruction for follow-up requests. If the user corrects a previous "
                "assumption, especially about dates or freshness, revise the answer "
                "instead of restating prior content. Do not introduce specific "
                "entities, incidents, dates, sources, or causal explanations "
                "that are not supported by the conversation, retrieved "
                "context, or tool results. If available context is insufficient, "
                "say so or use an appropriate tool to verify. "
                f"Current date (UTC): {current_date}. "
                "For recent, latest, current, or time-sensitive requests, use this "
                "date when forming search queries and judging source relevance. Only call "
                "tools that are present in the current tool schema for this LLM call; "
                "tool names mentioned in memory, previous tasks, plans, or error "
                "messages are unavailable unless they are included in the current "
                "schema. If a selected skill is already present in the system "
                "context, treat its main SKILL.md guidance as already read. Use "
                "skill documentation tools only when you need an additional "
                "referenced file, example, asset, or detail that is not already in "
                "the provided skill context."
                f"\n\nAvailable tool names for this LLM call are exactly: {available_tools}. "
                "Never call a tool name that is not in this list."
            )
        else:
            return messages
        completion_instruction = self._completion_evidence_instruction(context)
        if completion_instruction:
            instruction = f"{instruction}\n\n{completion_instruction}"
        if messages and messages[0].get("role") == "system":
            return [
                {
                    **messages[0],
                    "content": f"{messages[0].get('content', '')}\n\n{instruction}",
                },
                *messages[1:],
            ]
        return [{"role": "system", "content": instruction}, *messages]

    def _schema_tool_names(self, tool_schemas: list[dict[str, Any]]) -> list[str]:
        names: list[str] = []
        for schema in tool_schemas:
            function = schema.get("function")
            if isinstance(function, dict) and function.get("name"):
                names.append(str(function["name"]))
        return names

    def get_state(self) -> dict[str, Any]:
        """Return JSON-serializable ReAct state for checkpointing."""
        return {
            "reasoning_mode": self.reasoning_mode.value,
            "status": self.status,
            "current_iteration": self.current_iteration,
            "max_iterations": self.max_iterations,
            "finalize_after_tool_result": self.finalize_after_tool_result,
            "force_final_answer_next": self.force_final_answer_next,
            "waiting_for_user_request": self.waiting_for_user_request,
            "task_text": self.task_text,
            "last_response": self.last_response,
            "pending_tool_calls": self.pending_tool_calls,
            "pending_tool_call_content": self.pending_tool_call_content,
            "tool_ledger": {
                key: record.to_dict() for key, record in self.tool_ledger.items()
            },
        }

    def load_state(self, state: dict[str, Any]) -> None:
        """Restore ReAct state from a checkpoint payload."""
        self.reasoning_mode = ReActReasoningMode(
            state.get("reasoning_mode", ReActReasoningMode.TOOL_CALLING.value)
        )
        self.status = str(state.get("status", "idle"))
        self.current_iteration = int(state.get("current_iteration", 0))
        self.max_iterations = int(state.get("max_iterations", self.max_iterations))
        self.finalize_after_tool_result = bool(
            state.get("finalize_after_tool_result", self.finalize_after_tool_result)
        )
        self.force_final_answer_next = bool(state.get("force_final_answer_next", False))
        waiting_request = state.get("waiting_for_user_request")
        self.waiting_for_user_request = (
            dict(waiting_request) if isinstance(waiting_request, dict) else None
        )
        stored_task_text = state.get("task_text")
        self.task_text = str(stored_task_text) if stored_task_text else None
        self.last_response = state.get("last_response")
        self.pending_tool_calls = list(state.get("pending_tool_calls", []))
        self.pending_tool_call_content = dict(
            state.get("pending_tool_call_content", {})
        )
        self.tool_ledger = {
            key: ToolCallRecord.from_dict(value)
            for key, value in state.get("tool_ledger", {}).items()
        }

    async def _resume_waiting_for_user_if_needed(
        self,
        *,
        context: Any,
        runtime: PatternRuntime,
    ) -> dict[str, Any] | None:
        if self.status != "waiting_for_user" or not self.waiting_for_user_request:
            return None

        waiting_message_count = int(
            self.waiting_for_user_request.get("message_count", 0)
        )
        if len(getattr(context, "messages", [])) <= waiting_message_count:
            await runtime.checkpoint(
                "waiting_for_user",
                context=context,
                pattern=self,
                metadata={"waiting_for_user_request": self.waiting_for_user_request},
            )
            return {
                "success": False,
                "status": "waiting_for_user",
                "message": self.waiting_for_user_request.get("message", ""),
                "message_type": self.waiting_for_user_request.get(
                    "message_type", "question"
                ),
                "interactions": self.waiting_for_user_request.get("interactions"),
                "context": context,
            }

        self._mark_latest_user_message_as_waiting_response(
            context=context,
            after_message_count=waiting_message_count,
        )
        waiting_task = self.waiting_for_user_request.get("task_text")
        if waiting_task and self.task_text is None:
            self.task_text = str(waiting_task)
        self.waiting_for_user_request = None
        self.status = "thinking"
        return None

    def _task_text(self, context: Any) -> str:
        if self.task_text:
            return self.task_text
        self.task_text = latest_user_text(context)
        return self.task_text

    def _mark_latest_user_message_as_waiting_response(
        self,
        *,
        context: Any,
        after_message_count: int,
    ) -> None:
        messages = getattr(context, "messages", [])
        if not isinstance(messages, list):
            return

        for index in range(len(messages) - 1, after_message_count - 1, -1):
            message = messages[index]
            if getattr(message, "role", None) != "user":
                continue
            metadata = dict(getattr(message, "metadata", {}) or {})
            if metadata.get("response_to_waiting_for_user"):
                return
            waiting_request = self.waiting_for_user_request or {}
            metadata["response_to_waiting_for_user"] = {
                "tool_name": waiting_request.get("tool_name"),
                "tool_call_id": waiting_request.get("tool_call_id"),
                "question": waiting_request.get("message", ""),
                "message_type": waiting_request.get("message_type", "question"),
                "interactions": waiting_request.get("interactions"),
            }
            messages[index] = replace(message, metadata=metadata)
            return

    def _normalize_llm_response(self, response: Any) -> dict[str, Any]:
        if isinstance(response, str):
            content = unwrap_final_answer_content(response)
            return {
                "content": content,
                "tool_calls": [],
                "done": True,
                "raw": response,
            }

        if not isinstance(response, dict):
            text = unwrap_final_answer_content(str(response))
            return {"content": text, "tool_calls": [], "done": True, "raw": response}

        tool_calls = self._normalize_tool_calls(response.get("tool_calls", []))
        content_value: Any = response.get("content")
        if content_value is None:
            content_value = (
                response.get("answer")
                or response.get("output")
                or response.get("message")
            )
        if isinstance(content_value, str):
            content_value = unwrap_final_answer_content(content_value)

        done = response.get("done")
        if done is None:
            done = not tool_calls

        return {
            "content": content_value,
            "tool_calls": tool_calls,
            "raw_tool_calls": response.get("tool_calls", []),
            "done": bool(done),
            "raw": response,
        }

    def _normalize_tool_calls(self, tool_calls: list[Any]) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for index, tool_call in enumerate(tool_calls):
            if isinstance(tool_call, dict):
                function_payload = tool_call.get("function")
                if isinstance(function_payload, dict):
                    arguments = function_payload.get("arguments", {})
                    normalized.append(
                        {
                            "id": tool_call.get("id") or f"tool_call_{index}",
                            "name": function_payload.get("name"),
                            "args": self._coerce_arguments(arguments),
                        }
                    )
                    continue

                normalized.append(
                    {
                        "id": tool_call.get("id") or f"tool_call_{index}",
                        "name": tool_call.get("name"),
                        "args": self._coerce_arguments(
                            tool_call.get("args", tool_call.get("arguments", {}))
                        ),
                    }
                )
                continue

            function_payload = getattr(tool_call, "function", None)
            if function_payload is not None:
                normalized.append(
                    {
                        "id": getattr(tool_call, "id", None) or f"tool_call_{index}",
                        "name": getattr(function_payload, "name", None),
                        "args": self._coerce_arguments(
                            getattr(function_payload, "arguments", {})
                        ),
                    }
                )

        return [call for call in normalized if call.get("name")]

    def _remember_tool_call_content(
        self, tool_calls: list[dict[str, Any]], assistant_content: Any
    ) -> None:
        if not isinstance(assistant_content, str):
            return
        content = assistant_content.strip()
        if not content:
            return

        control_tool_names = self._control_tool_names()
        for tool_call in tool_calls:
            if tool_call.get("name") in control_tool_names:
                continue
            tool_call_id = str(tool_call.get("id") or "")
            if tool_call_id:
                self.pending_tool_call_content[tool_call_id] = content
            return

    def _coerce_arguments(self, arguments: Any) -> dict[str, Any]:
        if isinstance(arguments, dict):
            return arguments
        if isinstance(arguments, str):
            try:
                parsed = json.loads(arguments)
            except json.JSONDecodeError:
                return {"input": arguments}
            return parsed if isinstance(parsed, dict) else {"input": parsed}
        return {}

    def _build_tool_schema(self, tool: Any) -> dict[str, Any]:
        name = self._tool_name(tool)
        description = self._tool_description(tool)
        schema = self._tool_json_schema(tool)
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": schema,
            },
        }

    def _builtin_tool_schemas(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "final_answer",
                    "description": (
                        "Finish the current ReAct step and send the final answer to "
                        "the user. Use this once the latest tool results satisfy the "
                        "current user request. Do not call additional tools after "
                        f"this. {final_answer_language_rule()}"
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "answer": {
                                "type": "string",
                                "description": final_answer_language_rule(),
                            },
                        },
                        "required": ["answer"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "send_message",
                    "description": "Send a message to the user, optionally waiting for a response.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "message": {"type": "string"},
                            "message_type": {
                                "type": "string",
                                "enum": [
                                    "info",
                                    "question",
                                    "confirmation",
                                    "progress",
                                    "warning",
                                ],
                            },
                            "expect_response": {"type": "boolean"},
                        },
                        "required": ["message"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "ask_user_question",
                    "description": (
                        "Ask the user for structured input and pause execution until "
                        "the user responds. Use this only when execution cannot "
                        "continue without missing user-provided information, such "
                        "as a required file, URL, account, target object, permission, "
                        "or a choice between mutually exclusive actions with "
                        "different side effects. Do not use it to confirm execution "
                        "strategy, whether to search, whether to use memory, whether "
                        "to apply formatting preferences, or whether to proceed with "
                        "a sufficiently specified task; decide those yourself."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "message": {"type": "string"},
                            "interactions": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "type": {
                                            "type": "string",
                                            "enum": [
                                                "select_one",
                                                "select_multiple",
                                                "text_input",
                                                "file_upload",
                                                "confirm",
                                                "number_input",
                                                "action_cards",
                                            ],
                                        },
                                        "field": {"type": "string"},
                                        "label": {"type": "string"},
                                        "options": {
                                            "type": "array",
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "label": {"type": "string"},
                                                    "value": {"type": "string"},
                                                    "description": {"type": "string"},
                                                    "action_type": {
                                                        "type": "string",
                                                        "enum": [
                                                            "upload",
                                                            "input_url",
                                                            "none",
                                                        ],
                                                    },
                                                },
                                                "required": ["label", "value"],
                                            },
                                        },
                                        "placeholder": {"type": "string"},
                                        "multiline": {"type": "boolean"},
                                        "accept": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                        },
                                        "multiple": {"type": "boolean"},
                                    },
                                    "required": ["type", "field", "label"],
                                },
                            },
                        },
                        "required": ["message", "interactions"],
                    },
                },
            },
        ]

    def _tool_schemas_with_builtin_controls(
        self,
        tools: list[Any],
    ) -> list[dict[str, Any]]:
        control_tool_names = self._control_tool_names()
        external_tools = [
            self._build_tool_schema(tool)
            for tool in tools
            if self._tool_name(tool) not in control_tool_names
        ]
        return [*external_tools, *self._builtin_tool_schemas()]

    def _control_tool_names(self) -> set[str]:
        return {"final_answer", "send_message", "ask_user_question"}

    async def _handle_control_tool(
        self,
        tool_call: dict[str, Any],
        context: Any,
        llm: Any,
        runtime: PatternRuntime,
    ) -> dict[str, Any] | None:
        name = tool_call["name"]
        args = tool_call.get("args", {})

        if name == "final_answer":
            answer = str(args.get("answer", ""))
            self._record_tool_call(
                tool_call,
                status="completed",
                result={"answer": answer},
            )
            self.status = "completed"
            context.add_tool_result(
                tool_name=name,
                result={"answer": answer},
                tool_call_id=tool_call.get("id"),
            )
            if answer:
                context.add_assistant_message(answer)
            return await self._finalize_success(
                context=context,
                llm=llm,
                runtime=runtime,
                response=answer,
            )

        if name == "send_message":
            message = str(args.get("message", ""))
            expect_response = bool(args.get("expect_response", False))
            message_type = str(args.get("message_type", "info"))
            await runtime.send_message(
                message=message,
                message_type=message_type,
                expect_response=expect_response,
            )
            self._record_tool_call(
                tool_call,
                status="completed",
                result={"message": message, "expect_response": expect_response},
            )
            if expect_response:
                self.status = "waiting_for_user"
                context.add_tool_result(
                    tool_name=name,
                    result={
                        "status": "waiting_for_user",
                        "message": message,
                        "message_type": message_type,
                    },
                    tool_call_id=tool_call.get("id"),
                )
                self.waiting_for_user_request = {
                    "tool_call_id": tool_call.get("id"),
                    "tool_name": name,
                    "message": message,
                    "message_type": message_type,
                    "task_text": self.task_text,
                    "message_count": len(getattr(context, "messages", [])),
                }
                return {
                    "success": False,
                    "status": self.status,
                    "message": message,
                    "message_type": message_type,
                    "context": context,
                }
            context.add_tool_result(
                tool_name=name,
                result={"message": message, "status": "sent"},
                tool_call_id=tool_call.get("id"),
            )
            if message:
                context.add_assistant_message(message)
            return {
                "success": True,
                "status": "message_sent",
                "output": message,
                "response": message,
                "message": message,
            }

        if name == "ask_user_question":
            message = str(args.get("message", ""))
            interactions = _normalize_ask_user_interactions(
                args.get("interactions", [])
            )
            await runtime.send_message(
                message=message,
                message_type="question",
                expect_response=True,
                metadata={"interactions": interactions},
            )
            self._record_tool_call(
                tool_call,
                status="completed",
                result={
                    "message": message,
                    "expect_response": True,
                    "interactions": interactions,
                },
            )
            self.status = "waiting_for_user"
            context.add_tool_result(
                tool_name=name,
                result={
                    "status": "waiting_for_user",
                    "message": message,
                    "message_type": "question",
                    "interactions": interactions,
                },
                tool_call_id=tool_call.get("id"),
            )
            self.waiting_for_user_request = {
                "tool_call_id": tool_call.get("id"),
                "tool_name": name,
                "message": message,
                "message_type": "question",
                "interactions": interactions,
                "task_text": self.task_text,
                "message_count": len(getattr(context, "messages", [])),
            }
            return {
                "success": False,
                "status": self.status,
                "message": message,
                "message_type": "question",
                "interactions": interactions,
                "context": context,
            }

        return None

    async def _execute_pending_tool_calls(
        self,
        *,
        context: Any,
        tools: list[Any],
        llm: Any,
        runtime: PatternRuntime,
    ) -> dict[str, Any] | None:
        successful_tool_result = False
        while self.pending_tool_calls:
            interrupted = await self._interrupt_if_requested(
                runtime=runtime,
                context=context,
                label="before_tool",
            )
            if interrupted is not None:
                return interrupted

            tool_call = self.pending_tool_calls[0]
            control_result = await self._handle_control_tool(
                tool_call,
                context,
                llm,
                runtime,
            )
            if control_result is not None:
                self.pending_tool_calls = self.pending_tool_calls[1:]
                self._forget_tool_call_content(tool_call)
                await runtime.checkpoint(
                    str(control_result.get("status", "control_tool")),
                    context=context,
                    pattern=self,
                    metadata={"tool_call": tool_call},
                )
                if control_result.get("status") == "completed":
                    self.pending_tool_calls = []
                    return control_result
                if control_result.get("status") == "waiting_for_user":
                    return control_result
                continue

            await runtime.checkpoint(
                "before_tool",
                context=context,
                pattern=self,
                metadata={"tool_call": tool_call},
            )
            result = await self._execute_tool_safely(tool_call, tools, runtime)
            context.add_tool_result(
                tool_name=tool_call["name"],
                result=result,
                tool_call_id=tool_call.get("id"),
            )
            self.pending_tool_calls = self.pending_tool_calls[1:]
            self._forget_tool_call_content(tool_call)
            await runtime.checkpoint(
                "after_tool",
                context=context,
                pattern=self,
                metadata={"tool_call": tool_call},
            )
            if self._tool_result_success(result):
                successful_tool_result = True

        if self.finalize_after_tool_result and successful_tool_result:
            self.force_final_answer_next = True
        return None

    async def _finalize_success(
        self,
        *,
        context: Any,
        llm: Any,
        runtime: PatternRuntime,
        response: Any,
    ) -> dict[str, Any]:
        self.pending_tool_calls = []
        self.waiting_for_user_request = None
        self.force_final_answer_next = False
        self.status = "completed"
        await runtime.checkpoint("final", context=context, pattern=self)
        result = PatternResult(
            success=True,
            output=response,
            metadata={"response": response, "status": self.status},
        ).to_dict()
        await generate_and_store_react_memory(
            context=context,
            task=self._task_text(context),
            result=result,
            iterations=self.current_iteration + 1,
            llm=llm,
            memory_store=getattr(self, "_memory_store", None),
            runtime=runtime,
        )
        return result

    def _ensure_pending_tool_call_envelope(self, context: Any) -> None:
        if not self.pending_tool_calls:
            return
        messages = [
            message
            for message in getattr(context, "messages", [])
            if not getattr(message, "hidden", False)
        ]
        index = len(messages) - 1
        while index >= 0 and messages[index].role == "tool":
            index -= 1
        if index >= 0 and messages[index].role == "assistant":
            tool_calls = messages[index].tool_calls or []
            existing_ids = {
                str(tool_call.get("id"))
                for tool_call in tool_calls
                if isinstance(tool_call, dict) and tool_call.get("id")
            }
            pending_ids = {
                str(tool_call.get("id"))
                for tool_call in self.pending_tool_calls
                if tool_call.get("id")
            }
            if pending_ids and pending_ids.issubset(existing_ids):
                return
            if not pending_ids and len(tool_calls) >= len(self.pending_tool_calls):
                return

        context.add_assistant_message(
            "",
            tool_calls=[
                self._tool_call_for_context(tool_call)
                for tool_call in self.pending_tool_calls
            ],
        )

    def _tool_call_for_context(self, tool_call: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": tool_call.get("id"),
            "type": "function",
            "function": {
                "name": tool_call.get("name"),
                "arguments": json.dumps(tool_call.get("args", {}), default=str),
            },
        }

    def _tool_result_success(self, result: Any) -> bool:
        if not isinstance(result, dict):
            return True
        if result.get("success") is False:
            return False
        status = result.get("status")
        return not (isinstance(status, str) and status.lower() == "error")

    def _latest_tool_result_success(self, context: Any) -> bool:
        for message in reversed(getattr(context, "messages", [])):
            if getattr(message, "role", None) != "tool":
                continue
            metadata = getattr(message, "metadata", {}) or {}
            return self._tool_result_success(metadata.get("raw_result"))
        return False

    def _completion_evidence_instruction(self, context: Any) -> str:
        for message in reversed(getattr(context, "messages", [])):
            metadata = getattr(message, "metadata", {}) or {}
            evidence = metadata.get("dag_completion_evidence")
            if not isinstance(evidence, str):
                continue
            evidence = evidence.strip()
            if evidence:
                return (
                    "Step completion evidence: "
                    f"{evidence} "
                    "When the latest tool result or response satisfies this evidence "
                    "and the termination condition, call final_answer for this step. "
                    "Do not repeat the same work for a nicer variant unless the result "
                    "failed or the user explicitly requested a revision."
                )
        return ""

    async def _interrupt_if_requested(
        self,
        *,
        runtime: PatternRuntime,
        context: Any,
        label: str,
    ) -> dict[str, Any] | None:
        if not await runtime.should_interrupt():
            return None

        self.status = "interrupted"
        await runtime.checkpoint(
            "interrupted",
            context=context,
            pattern=self,
            metadata={"safe_point": label, "reason": runtime.interrupt_reason},
        )
        return PatternResult(
            success=False,
            error="ReActPattern interrupted.",
            metadata={
                "status": self.status,
                "interrupt_reason": runtime.interrupt_reason,
            },
        ).to_dict()

    async def _execute_tool_safely(
        self,
        tool_call: dict[str, Any],
        tools: list[Any],
        runtime: PatternRuntime,
    ) -> Any:
        tool_call = self._with_tool_call_content(tool_call)
        tool_call = self._with_runtime_step(tool_call, runtime)
        self._record_tool_call(tool_call, status="running")
        await runtime.on_tool_start(tool_call=tool_call)
        try:
            result = await self._execute_tool(tool_call, tools)
        except Exception as exc:  # noqa: BLE001
            error_result = {
                "success": False,
                "error": str(exc),
                "tool_name": tool_call["name"],
            }
            await runtime.on_tool_error(
                tool_call=tool_call, error=exc, result=error_result
            )
            self._record_tool_call(
                tool_call,
                status="failed",
                result=error_result,
                error=str(exc),
            )
            return error_result

        if not self._tool_result_success(result):
            error_message = str(result.get("error") or result.get("message") or result)
            await runtime.on_tool_error(
                tool_call=tool_call,
                error=RuntimeError(error_message),
                result=result,
            )
            self._record_tool_call(
                tool_call,
                status="failed",
                result=result,
                error=error_message,
            )
            return result

        self._record_tool_call(tool_call, status="completed", result=result)
        await runtime.on_tool_end(tool_call=tool_call, result=result)
        return result

    def _with_runtime_step(
        self, tool_call: dict[str, Any], runtime: PatternRuntime
    ) -> dict[str, Any]:
        if tool_call.get("step_id") or tool_call.get("dag_step_id"):
            return tool_call

        step_id = getattr(runtime, "active_react_step_id", None)
        if not step_id:
            return tool_call

        return {
            **tool_call,
            "step_id": str(step_id),
            "dag_step_id": str(step_id),
        }

    def _with_tool_call_content(self, tool_call: dict[str, Any]) -> dict[str, Any]:
        tool_call_id = str(tool_call.get("id") or "")
        content = self.pending_tool_call_content.get(tool_call_id)
        if not content:
            return tool_call
        return {
            **tool_call,
            "assistant_content": content,
        }

    def _forget_tool_call_content(self, tool_call: dict[str, Any]) -> None:
        tool_call_id = str(tool_call.get("id") or "")
        if tool_call_id:
            self.pending_tool_call_content.pop(tool_call_id, None)

    def _record_tool_call(
        self,
        tool_call: dict[str, Any],
        *,
        status: str,
        result: Any = None,
        error: str | None = None,
    ) -> None:
        tool_call_id = str(tool_call.get("id") or f"tool_call_{len(self.tool_ledger)}")
        args = dict(tool_call.get("args", {}))
        args_hash = self._args_hash(args)
        self.tool_ledger[tool_call_id] = ToolCallRecord(
            tool_call_id=tool_call_id,
            tool_name=str(tool_call["name"]),
            args=args,
            args_hash=args_hash,
            status=status,
            result=result,
            error=error,
        )

    def _args_hash(self, args: dict[str, Any]) -> str:
        try:
            return json.dumps(args, sort_keys=True, default=str)
        except TypeError:
            return str(args)

    def _tool_name(self, tool: Any) -> str:
        metadata = getattr(tool, "metadata", None)
        if metadata is not None and getattr(metadata, "name", None):
            return str(metadata.name)
        if getattr(tool, "name", None):
            return str(tool.name)
        if getattr(tool, "__name__", None):
            return str(tool.__name__)
        raise ValueError(f"Tool {tool!r} is missing a name.")

    def _tool_description(self, tool: Any) -> str:
        metadata = getattr(tool, "metadata", None)
        if metadata is not None and getattr(metadata, "description", None):
            return str(metadata.description)
        return (
            str(getattr(tool, "description", ""))
            or str(getattr(tool, "__doc__", "")).strip()
            or self._tool_name(tool)
        )

    def _tool_json_schema(self, tool: Any) -> dict[str, Any]:
        args_type = getattr(tool, "args_type", None)
        if callable(args_type):
            schema_type = args_type()
            if hasattr(schema_type, "model_json_schema"):
                return cast(dict[str, Any], schema_type.model_json_schema())
            if hasattr(schema_type, "schema"):
                return cast(dict[str, Any], schema_type.schema())
        for schema_attr in ("args_schema", "tool_call_schema"):
            schema_type = getattr(tool, schema_attr, None)
            if schema_type is None:
                continue
            if hasattr(schema_type, "model_json_schema"):
                return cast(dict[str, Any], schema_type.model_json_schema())
            if hasattr(schema_type, "schema"):
                return cast(dict[str, Any], schema_type.schema())
        args = getattr(tool, "args", None)
        if isinstance(args, dict) and args:
            return {"type": "object", "properties": args}
        if inspect.isfunction(tool):
            return self._signature_json_schema(tool)
        return {"type": "object", "properties": {}}

    def _signature_json_schema(self, fn: Any) -> dict[str, Any]:
        signature = inspect.signature(fn)
        properties: dict[str, Any] = {}
        required: list[str] = []
        for name, parameter in signature.parameters.items():
            if name in {"self", "cls"}:
                continue
            if parameter.kind in {
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            }:
                continue
            properties[name] = self._annotation_json_schema(parameter.annotation)
            if parameter.default is inspect.Parameter.empty:
                required.append(name)

        schema: dict[str, Any] = {"type": "object", "properties": properties}
        if required:
            schema["required"] = required
        return schema

    def _annotation_json_schema(self, annotation: Any) -> dict[str, Any]:
        if annotation is inspect.Parameter.empty:
            return {}
        if annotation is str or annotation == "str":
            return {"type": "string"}
        if annotation is int or annotation == "int":
            return {"type": "integer"}
        if annotation is float or annotation == "float":
            return {"type": "number"}
        if annotation is bool or annotation == "bool":
            return {"type": "boolean"}
        if annotation is dict or annotation == "dict":
            return {"type": "object"}
        if annotation is list or annotation == "list":
            return {"type": "array"}
        return {}

    async def _execute_tool(self, tool_call: dict[str, Any], tools: list[Any]) -> Any:
        tool = self._find_tool(tool_call["name"], tools)
        args = self._tool_args_for_execution(tool_call, tool)

        execute = getattr(tool, "execute", None)
        if callable(execute):
            return await self._invoke_callable(execute, **args)

        run_json_async = getattr(tool, "run_json_async", None)
        if callable(run_json_async):
            return await run_json_async(args)

        ainvoke = getattr(tool, "ainvoke", None)
        if callable(ainvoke):
            return await ainvoke(args)

        call = getattr(tool, "__call__", None)
        if callable(call):
            return await self._invoke_callable(call, **args)

        raise ValueError(
            f"Tool {tool_call['name']} does not expose a supported executor."
        )

    def _tool_args_for_execution(
        self, tool_call: dict[str, Any], tool: Any
    ) -> dict[str, Any]:
        args = dict(tool_call.get("args", {}))
        tool_name = self._tool_name(tool)
        if not tool_name.startswith("browser_"):
            return args

        step_id = tool_call.get("dag_step_id") or tool_call.get("step_id")
        if step_id and not args.get("session_id"):
            args.setdefault("_xagent_step_id", str(step_id))
        return args

    def _find_tool(self, name: str, tools: list[Any]) -> Any:
        for tool in tools:
            if self._tool_name(tool) == name:
                return tool
        raise ValueError(f"Tool not found: {name}")

    async def _invoke_callable(self, fn: Any, **kwargs: Any) -> Any:
        if inspect.iscoroutinefunction(fn):
            return await fn(**kwargs)
        result = await asyncio.to_thread(fn, **kwargs)
        if inspect.isawaitable(result):
            return await result
        return result
