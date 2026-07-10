"""ReAct execution pattern for the agent runtime.

In-turn tool concurrency (off by default; gated by ``tool_parallel_enabled``)
lets consecutive concurrency-safe tool calls in a single turn run as a bounded
concurrent batch instead of strictly serially. The implementation preserves
these invariants — referenced as I1–I6 in the code below — so that enabling the
flag never changes observable results, only latency:

- I1 (ordered backfill): tool results are written to the context in the model's
  original tool-call order, regardless of which tool finishes first.
- I2 (one result per call): every ``tool_call_id`` gets exactly one result,
  including failures.
- I3 (ledger order): ``tool_ledger`` insertion order matches input order after a
  batch, because the consecutive-count walks read it in reverse insertion order.
- I4 (control short-circuit): a control tool (final_answer / send_message /
  ask_user_question) owns its segment and ends the turn's tool execution; later
  tool calls in the same turn do not run.
- I5 (interrupt / resume): an interrupt is honored at a segment boundary with
  the remaining calls left pending; a crash mid-batch leaves the whole segment
  pending so resume re-runs it (safe because batched tools are read-only).
- I6 (trace pairing): tool trace spans pair START with END/ERROR by
  ``tool_call_id`` rather than tool name, so concurrent same-name calls do not
  cross-attribute (see ``tracing/langfuse/handler.py``).

When ``tool_parallel_enabled`` is False every segment is a single serial call,
making the loop byte-for-byte equivalent to the pre-concurrency behavior.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import inspect
import json
from dataclasses import dataclass, replace
from datetime import timezone
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
from ..base import AgentPattern, PatternResult, truncate_prompt_preview
from ..final_answer_stream import (
    ReActFinalAnswerStreamer,
)


class ReActReasoningMode(str, Enum):
    """Reasoning strategy used by ReActPattern."""

    TOOL_CALLING = "tool_calling"
    REASONING_ACTION = "reasoning_action"


REPEATED_TOOL_DECISION_REQUESTED_STATUS = "repeated_tool_decision_requested"
DEFAULT_REPEATED_TOOL_DECISION_CONSECUTIVE_TOOL_CALLS = 4
DEFAULT_REPEATED_TOOL_DECISION_CONSECUTIVE_WORK_TOOL_CALLS = 10
REACT_DECISION_TOOL_NAME = "react_decision"
REACT_DECISION_FINAL_ANSWER = "final_answer"
REACT_DECISION_TOOL_CALL = "tool_call"
UNGROUPED_TOOL_DECISION_CATEGORIES = frozenset({"basic", "other"})
REACT_RESPONSE_LANGUAGE_DESCRIPTION = (
    "Target natural language for user-facing prose in this ReAct response, "
    "for example English, Simplified Chinese, Traditional Chinese, or Spanish. "
    "For Chinese requests, choose Simplified Chinese or Traditional Chinese to "
    "match the request script; do not use generic Chinese. If the current user "
    "request explicitly asks to answer in another language, use that requested "
    "target language."
)


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
        repeated_tool_decision_after_consecutive_tool_calls: int | None = (
            DEFAULT_REPEATED_TOOL_DECISION_CONSECUTIVE_TOOL_CALLS
        ),
        repeated_tool_decision_after_consecutive_work_tool_calls: int | None = (
            DEFAULT_REPEATED_TOOL_DECISION_CONSECUTIVE_WORK_TOOL_CALLS
        ),
        tool_parallel_enabled: bool = False,
        tool_max_concurrency: int = 3,
    ) -> None:
        self.llm = llm
        self.max_iterations = max_iterations
        self.tool_choice = tool_choice
        self.reasoning_mode = ReActReasoningMode(reasoning_mode)
        self.finalize_after_tool_result = finalize_after_tool_result
        # In-turn tool concurrency (default off). When enabled, consecutive
        # concurrency-safe tool calls in a single turn run as a concurrent
        # batch bounded by ``tool_max_concurrency``.
        self.tool_parallel_enabled = tool_parallel_enabled
        self.tool_max_concurrency = max(1, int(tool_max_concurrency))
        self.repeated_tool_decision_after_consecutive_tool_calls = (
            repeated_tool_decision_after_consecutive_tool_calls
        )
        self.repeated_tool_decision_after_consecutive_work_tool_calls = (
            repeated_tool_decision_after_consecutive_work_tool_calls
        )
        self.status = "idle"
        self.current_iteration = 0
        self.last_response: Any = None
        self.pending_tool_calls: list[dict[str, Any]] = []
        self.pending_tool_call_content: dict[str, str] = {}
        self.tool_ledger: dict[str, ToolCallRecord] = {}
        self.force_final_answer_next = False
        self.repeated_tool_decision: dict[str, Any] | None = None
        self.waiting_for_user_request: dict[str, Any] | None = None
        self.task_text: str | None = None
        self._memory_store: Any | None = None
        self._tool_decision_groups_by_name: dict[str, str] = {}

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
        self._tool_decision_groups_by_name = self._tool_decision_groups_for_tools(tools)
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

            if self.repeated_tool_decision:
                decision_result = await self._run_repeated_tool_decision(
                    context=context,
                    llm=llm,
                    runtime=runtime,
                )
                if decision_result is not None:
                    return decision_result

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
            self.repeated_tool_decision = None
            self.last_response = response
            normalized = self._normalize_llm_response(response)
            if force_final_answer_now and not normalized.get("tool_calls"):
                normalized["done"] = True

            assistant_content = normalized.get("content")
            tool_calls = normalized.get("tool_calls", [])
            if assistant_content is not None or normalized.get("tool_calls"):
                metadata = (
                    self._provider_state_for_context(normalized) if tool_calls else {}
                )
                context.add_assistant_message(
                    assistant_content or "",
                    tool_calls=[
                        self._tool_call_for_context(tool_call)
                        for tool_call in tool_calls
                    ],
                    **({"metadata": metadata} if metadata else {}),
                )

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
            current_date = (
                context.created_at.astimezone(timezone.utc).date().isoformat()
            )
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

    def _tool_decision_groups_for_tools(self, tools: list[Any]) -> dict[str, str]:
        groups: dict[str, str] = {}
        for tool in tools:
            try:
                tool_name = self._tool_name(tool)
            except ValueError:
                continue
            groups[tool_name] = self._tool_decision_group(tool, tool_name)
        return groups

    def _tool_decision_group(self, tool: Any, tool_name: str) -> str:
        metadata = getattr(tool, "metadata", None)
        explicit_group = self._metadata_text(metadata, "decision_group") or (
            self._metadata_text(tool, "decision_group")
        )
        if explicit_group:
            return explicit_group

        category = getattr(metadata, "category", None) if metadata is not None else None
        category_value = getattr(category, "value", category)
        if category_value is not None:
            category_name = str(category_value).strip()
            if (
                category_name
                and category_name not in UNGROUPED_TOOL_DECISION_CATEGORIES
            ):
                return category_name
        return tool_name

    def _metadata_text(self, obj: Any, field_name: str) -> str:
        if obj is None:
            return ""
        value = getattr(obj, field_name, None)
        if not isinstance(value, str):
            return ""
        return value.strip()

    def _tool_decision_group_for_name(self, tool_name: str) -> str:
        return self._tool_decision_groups_by_name.get(tool_name, tool_name)

    def get_state(self) -> dict[str, Any]:
        """Return JSON-serializable ReAct state for checkpointing."""
        return {
            "reasoning_mode": self.reasoning_mode.value,
            "status": self.status,
            "current_iteration": self.current_iteration,
            "max_iterations": self.max_iterations,
            "finalize_after_tool_result": self.finalize_after_tool_result,
            "tool_parallel_enabled": self.tool_parallel_enabled,
            "tool_max_concurrency": self.tool_max_concurrency,
            "repeated_tool_decision_after_consecutive_tool_calls": (
                self.repeated_tool_decision_after_consecutive_tool_calls
            ),
            "repeated_tool_decision_after_consecutive_work_tool_calls": (
                self.repeated_tool_decision_after_consecutive_work_tool_calls
            ),
            "force_final_answer_next": self.force_final_answer_next,
            "repeated_tool_decision": self.repeated_tool_decision,
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
        if "tool_parallel_enabled" in state:
            self.tool_parallel_enabled = bool(state["tool_parallel_enabled"])
        if "tool_max_concurrency" in state:
            self.tool_max_concurrency = max(1, int(state["tool_max_concurrency"]))
        if "repeated_tool_decision_after_consecutive_tool_calls" in state:
            raw_threshold = state["repeated_tool_decision_after_consecutive_tool_calls"]
            self.repeated_tool_decision_after_consecutive_tool_calls = (
                int(raw_threshold) if raw_threshold is not None else None
            )
        elif "auto_reroute_after_consecutive_tool_calls" in state:
            raw_threshold = state["auto_reroute_after_consecutive_tool_calls"]
            self.repeated_tool_decision_after_consecutive_tool_calls = (
                int(raw_threshold) if raw_threshold is not None else None
            )

        if "repeated_tool_decision_after_consecutive_work_tool_calls" in state:
            raw_work_threshold = state[
                "repeated_tool_decision_after_consecutive_work_tool_calls"
            ]
            self.repeated_tool_decision_after_consecutive_work_tool_calls = (
                int(raw_work_threshold) if raw_work_threshold is not None else None
            )
        self.force_final_answer_next = bool(state.get("force_final_answer_next", False))
        repeated_tool_decision = state.get("repeated_tool_decision")
        self.repeated_tool_decision = (
            dict(repeated_tool_decision)
            if isinstance(repeated_tool_decision, dict)
            else None
        )
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

    def _provider_state_for_context(
        self, normalized_response: dict[str, Any]
    ) -> dict[str, Any]:
        raw = normalized_response.get("raw")
        marker_key = "_xagent_provider_state"
        if isinstance(raw, dict):
            raw_provider_state = raw.get(marker_key)
            if isinstance(raw_provider_state, dict):
                return {marker_key: raw_provider_state}
        if marker_key in normalized_response and isinstance(
            normalized_response[marker_key], dict
        ):
            return {marker_key: normalized_response[marker_key]}
        return {}

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
                        "this. Set response_language to the target output language "
                        "for this answer. "
                        f"{final_answer_language_rule()}"
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "response_language": {
                                "type": "string",
                                "description": REACT_RESPONSE_LANGUAGE_DESCRIPTION,
                            },
                            "answer": {
                                "type": "string",
                                "description": (
                                    "Complete user-facing answer. It must match "
                                    "response_language. "
                                    f"{final_answer_language_rule()}"
                                ),
                            },
                        },
                        "required": ["response_language", "answer"],
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
                            "visible": {"type": "boolean"},
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
            visible = bool(args.get("visible", True))
            await runtime.send_message(
                message=message,
                message_type=message_type,
                expect_response=expect_response,
                visible=visible,
            )
            self._record_tool_call(
                tool_call,
                status="completed",
                result={
                    "message": message,
                    "expect_response": expect_response,
                    "visible": visible,
                },
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
                visible=True,
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

    def _tool_is_concurrency_safe(self, name: str, tools: list[Any]) -> bool:
        """Whether ``name`` may run concurrently with other safe tools.

        Conservative: unknown tools and tools without the metadata flag are
        treated as not safe.
        """
        try:
            tool = self._find_tool(name, tools)
        except ValueError:
            return False
        metadata = getattr(tool, "metadata", None)
        return bool(getattr(metadata, "concurrency_safe", False))

    def _next_segment(
        self, pending: list[dict[str, Any]], tools: list[Any]
    ) -> tuple[list[dict[str, Any]], str]:
        """Slice the next consecutive segment off the front of ``pending``.

        Returns ``(segment, kind)`` where ``kind`` is one of:
        - ``"control"``: a single control tool (final_answer / send_message /
          ask_user_question), which always owns its segment;
        - ``"serial"``: a single tool executed on its own (a non-safe tool, any
          tool when the parallel flag is off, or a lone safe tool);
        - ``"concurrent"``: two or more consecutive concurrency-safe tools.
        """
        control_tool_names = self._control_tool_names()
        first = pending[0]
        if first["name"] in control_tool_names:
            return [first], "control"
        if not self.tool_parallel_enabled:
            return [first], "serial"
        if not self._tool_is_concurrency_safe(first["name"], tools):
            return [first], "serial"

        segment: list[dict[str, Any]] = []
        for tool_call in pending:
            name = tool_call["name"]
            if name in control_tool_names:
                break
            if not self._tool_is_concurrency_safe(name, tools):
                break
            segment.append(tool_call)
            # Cap a batch at the concurrency width. The loop re-checks the
            # interrupt before each segment, so a mid-turn interrupt is honored
            # after at most one wave instead of after every safe call the model
            # emitted. Tradeoff: a run longer than the width is split into
            # successive gather() barriers rather than one continuously
            # pipelined batch, so under skewed tool latencies a straggler in one
            # wave can briefly idle the next wave's slots. Acceptable for v1;
            # decouple batch size from the semaphore if profiling shows it costs.
            if len(segment) >= self.tool_max_concurrency:
                break
        # A lone safe tool degrades to serial: no gather/Semaphore overhead and
        # byte-for-byte identical behavior to the current serial path.
        if len(segment) == 1:
            return segment, "serial"
        return segment, "concurrent"

    def _backfill_result(
        self, tool_call: dict[str, Any], result: Any, context: Any
    ) -> None:
        """Record one tool result into the context and drop its cached content.

        Shared by the serial and concurrent paths so message-history ordering
        is produced the same way regardless of execution mode.
        """
        context.add_tool_result(
            tool_name=tool_call["name"],
            result=result,
            tool_call_id=tool_call.get("id"),
        )
        self._forget_tool_call_content(tool_call)

    async def _run_concurrent_batch(
        self,
        batch: list[dict[str, Any]],
        tools: list[Any],
        runtime: PatternRuntime,
        context: Any,
    ) -> list[Any]:
        """Run a segment of concurrency-safe tool calls and back-fill in order.

        Tools run under a Semaphore via ``asyncio.gather`` (results stay aligned
        to ``batch`` regardless of completion order, satisfying I1). Results are
        back-filled serially in the main coroutine after all tools finish, so
        message-history order is deterministic (I1) and every call gets exactly
        one result (I2). ``_execute_tool_safely`` already turns tool exceptions
        into error dicts, so any real exception captured by
        ``return_exceptions=True`` is an infra-callback/unexpected failure and is
        re-raised to halt the turn exactly like the serial path (I5).
        """
        semaphore = asyncio.Semaphore(self.tool_max_concurrency)

        async def _guarded(tool_call: dict[str, Any]) -> Any:
            async with semaphore:
                return await self._execute_tool_safely(tool_call, tools, runtime)

        raw_results = await asyncio.gather(
            *(_guarded(tool_call) for tool_call in batch),
            return_exceptions=True,
        )

        # Tool-level failures are already converted to error dicts inside
        # _execute_tool_safely, so anything coming back as a real exception is an
        # infra-callback failure (on_tool_start/on_tool_end) or an unexpected
        # bug. The serial path lets those propagate and halt the turn; re-raise
        # here so the concurrent path behaves identically instead of
        # mis-reporting infrastructure breakage to the model as a tool failure.
        # Re-raising before backfill leaves the whole (idempotent) segment
        # pending, so resume re-runs it cleanly (I5).
        for result in raw_results:
            if isinstance(result, BaseException):
                raise result

        for tool_call, result in zip(batch, raw_results):
            self._backfill_result(tool_call, result, context)
        self._reorder_ledger_for_batch(batch)
        return list(raw_results)

    def _reorder_ledger_for_batch(self, batch: list[dict[str, Any]]) -> None:
        """Reassert input order for this batch's ledger records (I3).

        Concurrent execution can interleave ``_record_tool_call`` writes, so the
        batch's records may land out of order in the insertion-ordered ledger.
        ``_consecutive_*_count`` walk the ledger in reverse insertion order, so
        we pop this batch's records and re-insert them at the tail in the
        original tool-call order. Records keep their latest (final) state; only
        their relative position is restored.
        """
        ids = [str(tool_call.get("id") or "") for tool_call in batch]
        records = {
            tool_id: self.tool_ledger.pop(tool_id)
            for tool_id in ids
            if tool_id in self.tool_ledger
        }
        for tool_id in ids:
            record = records.get(tool_id)
            if record is not None:
                self.tool_ledger[tool_id] = record

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

            segment, kind = self._next_segment(self.pending_tool_calls, tools)

            if kind == "control":
                tool_call = segment[0]
                control_result = await self._handle_control_tool(
                    tool_call,
                    context,
                    llm,
                    runtime,
                )
                self.pending_tool_calls = self.pending_tool_calls[1:]
                self._forget_tool_call_content(tool_call)
                await runtime.checkpoint(
                    str(control_result.get("status", "control_tool"))
                    if control_result is not None
                    else "control_tool",
                    context=context,
                    pattern=self,
                    metadata={"tool_call": tool_call},
                )
                if control_result is not None:
                    if control_result.get("status") == "completed":
                        self.pending_tool_calls = []
                        return control_result
                    if control_result.get("status") == "waiting_for_user":
                        return control_result
                continue

            if kind == "serial":
                tool_call = segment[0]
                await runtime.checkpoint(
                    "before_tool",
                    context=context,
                    pattern=self,
                    metadata={"tool_call": tool_call},
                )
                result = await self._execute_tool_safely(tool_call, tools, runtime)
                self._backfill_result(tool_call, result, context)
                self.pending_tool_calls = self.pending_tool_calls[1:]
                await runtime.checkpoint(
                    "after_tool",
                    context=context,
                    pattern=self,
                    metadata={"tool_call": tool_call},
                )
                results = [result]
            else:  # kind == "concurrent"
                await runtime.checkpoint(
                    "before_tool_batch",
                    context=context,
                    pattern=self,
                    metadata={"tool_calls": segment},
                )
                results = await self._run_concurrent_batch(
                    segment, tools, runtime, context
                )
                self.pending_tool_calls = self.pending_tool_calls[len(segment) :]
                await runtime.checkpoint(
                    "after_tool_batch",
                    context=context,
                    pattern=self,
                    metadata={"tool_calls": segment},
                )
                # In-flight tools are not cancellable, so an interrupt that
                # arrives during the batch is honored here, at the segment
                # boundary. The completed (read-only, concurrency-safe) results
                # are already recorded, which is correct for resume.
                interrupted = await self._interrupt_if_requested(
                    runtime=runtime,
                    context=context,
                    label="after_tool_batch",
                )
                if interrupted is not None:
                    return interrupted

            # Evaluate repeated-tool-decision once per segment, on its last call.
            requested_decision = await self._request_repeated_tool_decision_if_needed(
                tool_call=segment[-1],
                context=context,
                runtime=runtime,
            )
            if any(self._tool_result_success(result) for result in results):
                successful_tool_result = True
            if requested_decision:
                successful_tool_result = False

        if (
            self.finalize_after_tool_result
            and successful_tool_result
            and not self.repeated_tool_decision
        ):
            self.force_final_answer_next = True
        return None

    async def _request_repeated_tool_decision_if_needed(
        self,
        *,
        tool_call: dict[str, Any],
        context: Any,
        runtime: PatternRuntime,
    ) -> bool:
        if self.repeated_tool_decision is not None:
            return False

        metadata = self._repeated_tool_call_metadata(tool_call)
        if metadata is None:
            return False

        self.repeated_tool_decision = metadata
        await runtime.checkpoint(
            REPEATED_TOOL_DECISION_REQUESTED_STATUS,
            context=context,
            pattern=self,
            metadata=metadata,
        )
        return True

    async def _run_repeated_tool_decision(
        self,
        *,
        context: Any,
        llm: Any,
        runtime: PatternRuntime,
    ) -> dict[str, Any] | None:
        metadata = dict(self.repeated_tool_decision or {})
        if not metadata:
            return None

        messages = self._messages_for_repeated_tool_decision(context, metadata)
        decision_tools = [self._react_decision_tool_schema()]
        llm_metadata = {
            "phase": REPEATED_TOOL_DECISION_REQUESTED_STATUS,
            **metadata,
        }
        await runtime.on_llm_start(
            context=context,
            messages=messages,
            tools=decision_tools,
            metadata=llm_metadata,
        )
        try:
            response = await runtime.run_streaming_llm_call(
                llm,
                messages=messages,
                tools=decision_tools,
                tool_choice="required",
                thinking={"type": "disabled", "enable": False},
            )
        except LLMCallInterrupted:
            raise
        except Exception as exc:
            await runtime.on_llm_error(
                context=context,
                error=exc,
                metadata=llm_metadata,
            )
            raise

        await runtime.on_llm_end(
            context=context,
            response=response,
            metadata=llm_metadata,
        )
        self.last_response = response
        self.repeated_tool_decision = None

        decision = self._parse_react_decision(response)
        if decision is None:
            await runtime.checkpoint(
                "repeated_tool_decision_invalid",
                context=context,
                pattern=self,
                metadata={"response": response, **metadata},
            )
            return None

        if decision["action"] == REACT_DECISION_FINAL_ANSWER:
            self.force_final_answer_next = True
            await runtime.checkpoint(
                "repeated_tool_decision_final_requested",
                context=context,
                pattern=self,
                metadata={**metadata, "decision": decision},
            )
            context.add_system_message(
                "Repeated tool decision completion guidance:\n"
                "The repeated-tool decision selected final_answer, so the next "
                "normal ReAct step must produce the final user-facing answer from "
                "the accumulated conversation and tool results. Do not call more "
                "tools in that final step. Do not send a progress update or promise "
                "future work as the final answer; if the accumulated results are "
                "insufficient or show the task is incomplete, say that directly.",
                metadata={
                    "source": "repeated_tool_decision",
                    **metadata,
                },
            )
            return None

        await runtime.checkpoint(
            "repeated_tool_decision_continue",
            context=context,
            pattern=self,
            metadata={**metadata, "decision": decision},
        )
        missing_verification = decision.get("missing_verification", "").strip()
        if missing_verification:
            context.add_system_message(
                "Repeated tool decision continuation guidance:\n"
                "The previous repeated-tool decision chose to continue. The next "
                "work-tool call should retrieve or verify this specific missing "
                f"information: {missing_verification}",
                metadata={
                    "source": "repeated_tool_decision",
                    "missing_verification": missing_verification,
                },
            )
        return None

    def _repeated_tool_call_metadata(
        self,
        tool_call: dict[str, Any],
    ) -> dict[str, Any] | None:
        tool_name = str(tool_call.get("name") or "")
        if not tool_name or tool_name in self._control_tool_names():
            return None

        same_tool_threshold = self.repeated_tool_decision_after_consecutive_tool_calls
        if same_tool_threshold is not None and same_tool_threshold > 0:
            tool_group = self._tool_decision_group_for_name(tool_name)
            same_tool_count = self._consecutive_successful_tool_group_count(tool_group)
            if same_tool_count >= same_tool_threshold:
                return {
                    "trigger": "same_tool_successes",
                    "tool_name": tool_group,
                    "latest_tool_name": tool_name,
                    "consecutive_tool_calls": same_tool_count,
                    "threshold": same_tool_threshold,
                }

        work_tool_threshold = (
            self.repeated_tool_decision_after_consecutive_work_tool_calls
        )
        if work_tool_threshold is None or work_tool_threshold <= 0:
            return None

        work_tool_count = self._consecutive_work_tool_call_count()
        if work_tool_count < work_tool_threshold:
            return None
        return {
            "trigger": "work_tool_attempts",
            "tool_name": tool_name,
            "latest_tool_name": tool_name,
            "consecutive_tool_calls": work_tool_count,
            "threshold": work_tool_threshold,
        }

    def _messages_for_repeated_tool_decision(
        self,
        context: Any,
        metadata: dict[str, Any],
    ) -> list[dict[str, Any]]:
        messages = list(context.get_messages_for_llm())
        tool_name = str(metadata.get("tool_name") or "the tool")
        count = int(metadata.get("consecutive_tool_calls") or 0)
        trigger = metadata.get("trigger")
        if trigger == "work_tool_attempts":
            latest_tool_name = str(metadata.get("latest_tool_name") or tool_name)
            count_text = (
                f"{count} consecutive work-tool calls without a final answer"
                if count > 0
                else "repeated work-tool calls without a final answer"
            )
            call_context = (
                f"{count_text}; the latest work tool was {latest_tool_name}. "
                "Some attempts may have failed; count them as work already spent."
            )
        else:
            latest_tool_name = str(metadata.get("latest_tool_name") or tool_name)
            count_text = (
                f"{count} consecutive successful calls"
                if count > 0
                else "repeated successful calls"
            )
            if latest_tool_name != tool_name:
                call_context = (
                    f"{count_text} in the {tool_name} tool group; "
                    f"the latest tool was {latest_tool_name}."
                )
            else:
                call_context = f"{count_text} to {tool_name}."
        current_request = truncate_prompt_preview(
            latest_user_text(context) or "",
            limit=400,
        )
        request_anchor = (
            "Latest user request text:\n"
            f"{current_request or '(unavailable)'}\n\n"
            "Use this as the controlling request when deciding whether the "
            "accumulated tool results have completed the user's requested work."
        )
        prompt = (
            f"You must call {REACT_DECISION_TOOL_NAME} exactly once. Decide whether "
            "the current ReAct run should finish or make another work-tool call. "
            f"{request_anchor} "
            f"You have just made {call_context} action must be "
            f"{REACT_DECISION_FINAL_ANSWER} or {REACT_DECISION_TOOL_CALL}. Choose "
            f"{REACT_DECISION_FINAL_ANSWER} when the conversation and accumulated "
            "tool results are sufficient to answer the latest user request. A "
            "final answer means the latest user request is already completed; if "
            "the next user-facing answer would describe a future tool action or "
            "say work is still in progress, choose tool_call instead. Choose "
            f"{REACT_DECISION_TOOL_CALL} only when a specific missing fact, source, "
            "verification, or work step remains; the next normal ReAct turn will "
            "choose and call the actual work tool from the full available tool set. "
            "Do not call work tools in this decision. Do not put user-facing final "
            "answer text in this decision; the next normal ReAct step will produce "
            "the final answer if you choose final_answer. Treat the completed-call "
            "count in this instruction as authoritative; do not count the user's "
            "requested number as already completed. If the latest user request "
            "explicitly requires more completed work-tool calls or results than "
            f"the current context contains, choose {REACT_DECISION_TOOL_CALL}."
        )
        return [*messages, {"role": "user", "content": prompt}]

    def _react_decision_tool_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": REACT_DECISION_TOOL_NAME,
                "description": (
                    "Decide whether ReAct should finish in the next normal final "
                    "answer step or continue to another work-tool call."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": [
                                REACT_DECISION_FINAL_ANSWER,
                                REACT_DECISION_TOOL_CALL,
                            ],
                            "description": (
                                "final_answer when current context is sufficient; "
                                "tool_call when one more work-tool call is needed."
                            ),
                        },
                        "reason": {
                            "type": "string",
                            "description": "Brief reason for this decision.",
                        },
                        "missing_verification": {
                            "type": "string",
                            "description": (
                                "When action is tool_call, the specific missing "
                                "fact, source, verification, or work step that "
                                "requires another tool call."
                            ),
                        },
                    },
                    "required": ["action", "reason"],
                },
            },
        }

    def _parse_react_decision(self, response: Any) -> dict[str, str] | None:
        normalized = self._normalize_llm_response(response)
        for tool_call in normalized.get("tool_calls", []):
            if tool_call.get("name") != REACT_DECISION_TOOL_NAME:
                continue
            args = tool_call.get("args")
            if not isinstance(args, dict):
                return None
            action = str(args.get("action") or "").strip()
            if action not in {
                REACT_DECISION_FINAL_ANSWER,
                REACT_DECISION_TOOL_CALL,
            }:
                return None
            return {
                "action": action,
                "reason": str(args.get("reason") or ""),
                "missing_verification": str(args.get("missing_verification") or ""),
            }
        return None

    def _consecutive_successful_tool_group_count(self, tool_group: str) -> int:
        count = 0
        control_tool_names = self._control_tool_names()
        for record in reversed(list(self.tool_ledger.values())):
            if record.tool_name in control_tool_names:
                continue
            if self._tool_decision_group_for_name(record.tool_name) != tool_group:
                break
            if record.status != "completed" or not self._tool_result_success(
                record.result
            ):
                break
            count += 1
        return count

    def _consecutive_work_tool_call_count(self) -> int:
        count = 0
        control_tool_names = self._control_tool_names()
        for record in reversed(list(self.tool_ledger.values())):
            if record.tool_name in control_tool_names:
                continue
            if record.status not in {"completed", "failed"}:
                continue
            count += 1
        return count

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
                "arguments": json.dumps(
                    tool_call.get("args", {}),
                    ensure_ascii=False,
                    default=str,
                ),
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
        # Stamp a stable id on the *original* dict before the _with_* transforms
        # (which may return a copy). _record_tool_call only computes a fallback
        # key locally; without writing it back, the key drifts between the
        # running/completed writes as the ledger grows, and the still-id-less
        # dict that _backfill_result / _reorder_ledger_for_batch read desyncs
        # from the ledger (I2/I3). No await runs before the first record below,
        # so concurrent batch members get distinct fallback ids.
        if not tool_call.get("id"):
            tool_call["id"] = f"tool_call_{len(self.tool_ledger)}"
        tool_call = self._with_tool_call_content(tool_call)
        tool_call = self._with_runtime_step(tool_call, runtime)
        tool_call = self._with_trace_safe_tool_args(tool_call, tools)
        self._record_tool_call(tool_call, status="running")
        recorded_terminal = False
        try:
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
                recorded_terminal = True
                return error_result

            if not self._tool_result_success(result):
                error_message = str(
                    result.get("error") or result.get("message") or result
                )
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
                recorded_terminal = True
                return result

            self._record_tool_call(tool_call, status="completed", result=result)
            recorded_terminal = True
            await runtime.on_tool_end(tool_call=tool_call, result=result)
            return result
        finally:
            # An infra callback (on_tool_start) can raise before any terminal
            # record is written. Never leave the ledger stuck at "running": the
            # consecutive-count walks skip non-terminal records, which would
            # undercount repeated-tool-decision triggers. The exception still
            # propagates (serial path) or is captured by the batch gather.
            if not recorded_terminal:
                self._record_tool_call(
                    tool_call,
                    status="failed",
                    error="tool execution aborted before completion",
                )

    def _with_trace_safe_tool_args(
        self, tool_call: dict[str, Any], tools: list[Any]
    ) -> dict[str, Any]:
        try:
            tool = self._find_tool(tool_call["name"], tools)
        except Exception:  # noqa: BLE001
            return tool_call

        sanitizer = getattr(tool, "sanitize_tool_args_for_trace", None)
        if not callable(sanitizer):
            return tool_call

        raw_args = tool_call.get("args")
        if raw_args is None:
            args: dict[str, Any] = {}
            original_args: dict[str, Any] = {}
        elif isinstance(raw_args, dict):
            args = copy.deepcopy(raw_args)
            original_args = copy.deepcopy(raw_args)
        else:
            return tool_call
        sanitized = sanitizer(args)
        if not isinstance(sanitized, dict) or sanitized == original_args:
            return tool_call
        return {**tool_call, "args": sanitized}

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
        args = self._tool_call_args_dict(tool_call)
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
            canonical = json.dumps(args, sort_keys=True, default=str)
        except (TypeError, ValueError):
            canonical = str(args)
        # Digest instead of the raw JSON: the hash is persisted in every
        # ledger record, so large args would otherwise be stored twice.
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _tool_call_args_dict(
        self, tool_call: dict[str, Any], *, require_mapping: bool = False
    ) -> dict[str, Any]:
        raw_args = tool_call.get("args")
        if raw_args is None:
            return {}
        if isinstance(raw_args, dict):
            return dict(raw_args)
        if require_mapping:
            raise ValueError("Tool call args must be a JSON object.")
        return {}

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
        args = self._tool_call_args_dict(tool_call, require_mapping=True)
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
