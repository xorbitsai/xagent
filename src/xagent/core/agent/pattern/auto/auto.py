from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

from json_repair import loads as repair_json_loads

from ....model.chat.exceptions import LLMToolProtocolError
from ...context.enrichment import (
    MEMORY_CONTEXT_METADATA_KEY,
    RETRIEVED_MEMORIES_METADATA_KEY,
    SELECTED_SKILL_METADATA_KEY,
    SKILL_CONTEXT_METADATA_KEY,
    enrich_context_with_memory,
    latest_user_text,
)
from ...frame import ExecutionFrame, ExecutionSnapshot, ExecutionStatus
from ...language import (
    OUTPUT_LANGUAGE_METADATA_KEY,
    final_answer_language_rule,
    normalize_response_language_label,
    output_language_policy,
)
from ...runtime import (
    LLMCallInterrupted,
    PatternRuntime,
    prepare_llm_for_context,
    resolved_llm_metadata,
)
from ..base import (
    REQUIRED_TOOL_CALL_FAILURE_REASON,
    AgentPattern,
    PatternResult,
    RequiredToolCallError,
    append_user_message_preserving_turns,
    extract_required_tool_arguments,
    truncate_prompt_preview,
)
from ..dag import DAGPattern
from ..final_answer_stream import (
    FinalAnswerStreamSession,
    ToolCallStringFieldStreamer,
)
from ..react import ReActPattern

logger = logging.getLogger(__name__)


class AutoAction(str, Enum):
    """High-level action selected by AutoPattern."""

    FINAL_ANSWER = "final_answer"
    REACT = "react"
    PLAN_EXECUTE = "plan_execute"


@dataclass
class AutoDecision:
    """Serializable AutoPattern decision."""

    action: AutoAction
    reason: str = ""
    answer: str | None = None
    requires_current_or_external_facts: bool = False
    existing_context_sufficient: bool = True
    evidence_basis: str = ""
    missing_verification: str = ""
    response_language: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "action": self.action.value,
            "reason": self.reason,
            "requires_current_or_external_facts": (
                self.requires_current_or_external_facts
            ),
            "existing_context_sufficient": self.existing_context_sufficient,
            "evidence_basis": self.evidence_basis,
            "missing_verification": self.missing_verification,
        }
        if self.response_language:
            payload["response_language"] = self.response_language
        if self.answer is not None:
            payload["answer"] = self.answer
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AutoDecision":
        raw_action = payload.get("action") or payload.get("type")
        if raw_action == "dag":
            raw_action = AutoAction.PLAN_EXECUTE.value
        try:
            action = AutoAction(str(raw_action))
        except ValueError as exc:
            raise ValueError(f"Invalid AutoPattern action: {raw_action}") from exc
        return cls(
            action=action,
            reason=str(payload.get("reason", "")),
            answer=payload.get("answer"),
            requires_current_or_external_facts=_coerce_bool(
                payload.get("requires_current_or_external_facts"), default=False
            ),
            existing_context_sufficient=_coerce_bool(
                payload.get("existing_context_sufficient"), default=True
            ),
            evidence_basis=str(payload.get("evidence_basis", "")),
            missing_verification=str(payload.get("missing_verification", "")),
            response_language=normalize_response_language_label(
                str(payload.get("response_language", ""))
            ),
        )


@dataclass
class AutoDecisionResult:
    decision: AutoDecision
    final_answer_stream: FinalAnswerStreamSession | None = None


DECISION_TOOL_NAME = "select_execution_pattern"
MAX_DECISION_PARSE_ATTEMPTS = 2
AUTO_DECISION_REQUIRED_TOOL_MESSAGE = (
    "Auto routing failed because the model did not return the required "
    "decision tool call. Please retry."
)


class AutoDecisionArgumentsError(ValueError):
    """Raised when the routing tool call arguments cannot be parsed."""

    def __init__(self, message: str, *, arguments: str | None = None) -> None:
        super().__init__(message)
        self.arguments = arguments


def _coerce_bool(value: Any, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y"}:
            return True
        if normalized in {"false", "0", "no", "n"}:
            return False
    if value is None:
        return default
    return bool(value)


@dataclass
class _AutoChildRuntime:
    """Runtime adapter that captures child state before parent checkpoints."""

    parent: PatternRuntime
    auto_pattern: "AutoPattern"
    root_context: Any

    @property
    def execution_id(self) -> str | None:
        return self.parent.execution_id

    @execution_id.setter
    def execution_id(self, value: str | None) -> None:
        self.parent.execution_id = value

    @property
    def interrupt_reason(self) -> str | None:
        return self.parent.interrupt_reason

    @property
    def tracer(self) -> Any | None:
        return self.parent.tracer

    @property
    def active_react_step_id(self) -> str | None:
        return self.parent.active_react_step_id

    async def should_interrupt(self) -> bool:
        return await self.parent.should_interrupt()

    def clear_interrupt(self) -> None:
        self.parent.clear_interrupt()

    async def run_llm_call(self, llm: Any, **kwargs: Any) -> Any:
        return await self.parent.run_llm_call(llm, **kwargs)

    async def run_tool_call(self, invoke: Any) -> Any:
        return await self.parent.run_tool_call(invoke)

    async def stream_final_answer(self, llm: Any, **kwargs: Any) -> Any:
        return await self.parent.stream_final_answer(llm, **kwargs)

    async def start_final_answer_stream(self) -> str | None:
        return await self.parent.start_final_answer_stream()

    async def emit_final_answer_delta(self, message_id: str, delta: str) -> None:
        await self.parent.emit_final_answer_delta(message_id, delta)

    async def end_final_answer_stream(self, message_id: str, content: str) -> None:
        await self.parent.end_final_answer_stream(message_id, content)

    async def fail_final_answer_stream(self, message_id: str, error: str) -> None:
        await self.parent.fail_final_answer_stream(message_id, error)

    async def run_streaming_llm_call(
        self,
        llm: Any,
        *,
        on_chunk: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        return await self.parent.run_streaming_llm_call(
            llm,
            on_chunk=on_chunk,
            **kwargs,
        )

    async def send_message(
        self,
        *,
        message: str,
        message_type: str = "info",
        expect_response: bool = False,
        visible: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return await self.parent.send_message(
            message=message,
            message_type=message_type,
            expect_response=expect_response,
            visible=visible,
            metadata=metadata,
        )

    async def checkpoint(
        self,
        label: str,
        *,
        context: Any,
        pattern: Any,
        status: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.auto_pattern._capture_child_state(pattern)
        child_metadata = {
            "child_label": label,
            "child_pattern": pattern.__class__.__name__,
        }
        if metadata:
            child_metadata.update(metadata)
        return await self.parent.checkpoint(
            label=f"auto_child_{label}",
            context=self.root_context,
            pattern=self.auto_pattern,
            status=status,
            metadata=child_metadata,
        )

    async def on_tool_start(self, *, tool_call: dict[str, Any]) -> None:
        await self.parent.on_tool_start(tool_call=tool_call)

    async def on_tool_end(self, *, tool_call: dict[str, Any], result: Any) -> None:
        await self.parent.on_tool_end(tool_call=tool_call, result=result)

    async def on_tool_error(
        self,
        *,
        tool_call: dict[str, Any],
        error: Exception,
        result: Any | None = None,
    ) -> None:
        await self.parent.on_tool_error(tool_call=tool_call, error=error, result=result)

    async def on_tool_cancelled(
        self,
        *,
        tool_call: dict[str, Any],
        reason: str | None = None,
    ) -> None:
        await self.parent.on_tool_cancelled(tool_call=tool_call, reason=reason)

    async def on_llm_start(
        self,
        *,
        context: Any,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        await self.parent.on_llm_start(
            context=context,
            messages=messages,
            tools=tools,
            metadata=metadata,
        )

    async def on_llm_end(
        self,
        *,
        context: Any,
        response: Any,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        await self.parent.on_llm_end(
            context=context,
            response=response,
            metadata=metadata,
        )

    async def on_llm_error(
        self,
        *,
        context: Any,
        error: Exception,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        await self.parent.on_llm_error(
            context=context,
            error=error,
            metadata=metadata,
        )

    async def compact_context_if_needed(
        self,
        *,
        context: Any,
        llm: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> Any:
        return await self.parent.compact_context_if_needed(
            context=context,
            llm=llm,
            metadata=metadata,
        )

    async def on_dag_step_start(
        self,
        *,
        context: Any,
        step_id: str,
        data: dict[str, Any] | None = None,
    ) -> None:
        await self.parent.on_dag_step_start(
            context=context,
            step_id=step_id,
            data=data,
        )

    async def on_dag_step_end(
        self,
        *,
        context: Any,
        step_id: str,
        data: dict[str, Any] | None = None,
    ) -> None:
        await self.parent.on_dag_step_end(
            context=context,
            step_id=step_id,
            data=data,
        )

    async def on_dag_execution(
        self,
        *,
        context: Any,
        phase: str,
        data: dict[str, Any] | None = None,
    ) -> None:
        await self.parent.on_dag_execution(
            context=context,
            phase=phase,
            data=data,
        )

    async def on_pattern_start(self, *, context: Any, pattern: Any) -> None:
        await self.parent.on_pattern_start(context=context, pattern=pattern)

    async def on_pattern_end(
        self,
        *,
        context: Any,
        pattern: Any,
        result: dict[str, Any],
    ) -> None:
        await self.parent.on_pattern_end(
            context=context,
            pattern=pattern,
            result=result,
        )

    async def on_pattern_error(
        self,
        *,
        context: Any,
        pattern: Any,
        error: Exception,
    ) -> None:
        await self.parent.on_pattern_error(
            context=context, pattern=pattern, error=error
        )


class AutoPattern(AgentPattern):
    """Thin LLM-driven pattern that delegates to ReActPattern or DAGPattern."""

    def __init__(
        self,
        *,
        react_pattern: ReActPattern | None = None,
        dag_pattern: DAGPattern | None = None,
    ) -> None:
        self.react_pattern = react_pattern or ReActPattern()
        self.dag_pattern = dag_pattern
        self.status = "idle"
        self.decision: AutoDecision | None = None
        self.decision_user_messages: dict[str, Any] | None = None
        self.selected_pattern: str | None = None
        self.react_state: dict[str, Any] | None = None
        self.dag_state: dict[str, Any] | None = None
        self.last_result: dict[str, Any] | None = None

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

        await runtime.on_pattern_start(context=context, pattern=self)
        try:
            result = await self._run(
                context=context,
                tools=tools,
                llm=llm,
                runtime=runtime,
                skill_manager=kwargs.get("skill_manager"),
                allowed_skills=kwargs.get("allowed_skills"),
                memory_store=kwargs.get("memory_store"),
                memory_similarity_threshold=kwargs.get("memory_similarity_threshold"),
                compact_llm=kwargs.get("compact_llm"),
            )
        except RequiredToolCallError as exc:
            result = await self._fail(
                context=context,
                runtime=runtime,
                error=exc.user_message,
                failure_reason=exc.failure_reason,
                checkpoint_label="auto_decision_failed",
                extra_metadata=exc.to_metadata(),
            )
            await runtime.on_pattern_end(context=context, pattern=self, result=result)
            return result
        except Exception as exc:
            self.status = "failed"
            await runtime.on_pattern_error(context=context, pattern=self, error=exc)
            raise
        await runtime.on_pattern_end(context=context, pattern=self, result=result)
        return result

    async def _run(
        self,
        *,
        context: Any,
        tools: list[Any],
        llm: Any,
        runtime: PatternRuntime,
        memory_store: Any | None = None,
        memory_similarity_threshold: float | None = None,
        skill_manager: Any | None = None,
        allowed_skills: list[str] | None = None,
        compact_llm: Any | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        self._invalidate_stale_final_answer_decision(context)
        final_answer_stream: FinalAnswerStreamSession | None = None
        if self.decision is None:
            self.status = "deciding"
            task_text = latest_user_text(context)
            await enrich_context_with_memory(
                context=context,
                query=task_text,
                category="react_memory",
                memory_store=memory_store,
                runtime=runtime,
                similarity_threshold=memory_similarity_threshold,
            )
            interrupted = await self._interrupt_if_requested(
                runtime=runtime,
                context=context,
                label="auto_before_decision",
            )
            if interrupted is not None:
                return interrupted
            await runtime.checkpoint(
                "auto_before_decision", context=context, pattern=self
            )
            try:
                decision_result = await self._decide(
                    context=context,
                    tools=tools,
                    llm=llm,
                    compact_llm=compact_llm,
                    runtime=runtime,
                    memory_tools_available=memory_store is not None,
                )
                self.decision = decision_result.decision
                final_answer_stream = decision_result.final_answer_stream
            except LLMCallInterrupted:
                interrupted = await self._interrupt_if_requested(
                    runtime=runtime,
                    context=context,
                    label="auto_during_decision",
                )
                if interrupted is not None:
                    return interrupted
                raise
            self._normalize_decision()
            if self.decision is None:
                raise RuntimeError("AutoPattern decision was not set.")
            self._apply_response_language(context)
            self.decision_user_messages = self._user_message_signature(context)
            self.selected_pattern = self.decision.action.value
            logger.info(
                "AutoPattern selected %s for execution %s: %s",
                self.selected_pattern,
                getattr(context, "execution_id", None),
                self.decision.reason,
            )
            await runtime.checkpoint(
                "auto_after_decision", context=context, pattern=self
            )
            interrupted = await self._interrupt_if_requested(
                runtime=runtime,
                context=context,
                label="auto_after_decision",
            )
            if interrupted is not None:
                return interrupted

        if self.decision.action == AutoAction.FINAL_ANSWER:
            answer = self.decision.answer or ""
            if final_answer_stream is not None:
                await final_answer_stream.finish(answer)
            if answer:
                context.add_assistant_message(answer)
            self.status = "completed"
            result = PatternResult(
                success=True,
                output=answer,
                metadata={
                    "status": self.status,
                    "response": answer,
                    "auto_decision": self.decision.to_dict(),
                },
            ).to_dict()
            self.last_result = result
            await runtime.checkpoint("auto_final", context=context, pattern=self)
            return result

        child = self._selected_child()
        self._restore_child_state(child)
        self.status = "running"
        interrupted = await self._interrupt_if_requested(
            runtime=runtime,
            context=context,
            label="auto_before_child",
        )
        if interrupted is not None:
            return interrupted
        await runtime.checkpoint("auto_before_child", context=context, pattern=self)
        child_runtime = _AutoChildRuntime(
            parent=runtime,
            auto_pattern=self,
            root_context=context,
        )
        result = await child.run(
            context=context,
            tools=tools,
            llm=llm,
            runtime=child_runtime,
            memory_store=memory_store,
            memory_similarity_threshold=memory_similarity_threshold,
            skill_manager=skill_manager,
            allowed_skills=allowed_skills,
            compact_llm=compact_llm,
            **kwargs,
        )
        self._attach_decision_metadata(result)
        self._capture_child_state(child)
        self.status = str(
            result.get("status", "completed" if result.get("success") else "failed")
        )
        self.last_result = dict(result)
        await runtime.checkpoint("auto_after_child", context=context, pattern=self)
        return result

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
            "auto_interrupted",
            context=context,
            pattern=self,
            status=self.status,
            metadata={"safe_point": label, "reason": runtime.interrupt_reason},
        )
        result = PatternResult(
            success=False,
            error="AutoPattern interrupted.",
            metadata={
                "status": self.status,
                "interrupt_reason": runtime.interrupt_reason,
            },
        ).to_dict()
        self.last_result = result
        return result

    async def _fail(
        self,
        *,
        context: Any,
        runtime: PatternRuntime,
        error: str,
        failure_reason: str,
        checkpoint_label: str,
        extra_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.status = "failed"
        metadata: dict[str, Any] = {
            "status": self.status,
            "failure_reason": failure_reason,
        }
        if extra_metadata:
            metadata.update(extra_metadata)
        result = PatternResult(
            success=False,
            error=error,
            metadata=metadata,
        ).to_dict()
        self.last_result = result
        await runtime.checkpoint(
            checkpoint_label,
            context=context,
            pattern=self,
            status=self.status,
            metadata=metadata,
        )
        return result

    def _normalize_decision(self) -> None:
        if self.decision is None:
            return
        if (
            self.decision.action == AutoAction.FINAL_ANSWER
            and self.decision.requires_current_or_external_facts
            and not self.decision.existing_context_sufficient
        ):
            logger.warning(
                "AutoPattern selected final_answer for facts requiring external "
                "verification without sufficient context; falling back to react. "
                "reason=%s evidence_basis=%s missing_verification=%s",
                self.decision.reason,
                self.decision.evidence_basis,
                self.decision.missing_verification,
            )
            self.decision = AutoDecision(
                action=AutoAction.REACT,
                reason=(
                    "AutoPattern selected final_answer for a request requiring "
                    "current or external facts without sufficient supporting "
                    "context; falling back to react."
                ),
                requires_current_or_external_facts=True,
                existing_context_sufficient=False,
                evidence_basis=self.decision.evidence_basis,
                missing_verification=self.decision.missing_verification,
                response_language=self.decision.response_language,
            )
        if (
            self.decision.action == AutoAction.FINAL_ANSWER
            and not (self.decision.answer or "").strip()
        ):
            logger.warning(
                "AutoPattern selected final_answer without answer; falling back to "
                "react. reason=%s",
                self.decision.reason,
            )
            self.decision = AutoDecision(
                action=AutoAction.REACT,
                reason=(
                    "AutoPattern selected final_answer without a non-empty answer; "
                    "falling back to react."
                ),
                response_language=self.decision.response_language,
            )
        if self.decision.action == AutoAction.PLAN_EXECUTE and self.dag_pattern is None:
            raise ValueError(
                "AutoPattern selected plan_execute without a DAGPattern configured."
            )

    def _apply_response_language(self, context: Any) -> None:
        if self.decision is None:
            return
        response_language = normalize_response_language_label(
            self.decision.response_language
        )
        if not response_language:
            return
        metadata = self._context_metadata(context)
        if metadata is not None:
            # Auto is the current-turn language authority; replace any
            # request-scoped policy left by a previous turn.
            metadata[OUTPUT_LANGUAGE_METADATA_KEY] = response_language

    @staticmethod
    def _context_metadata(context: Any) -> dict[str, Any] | None:
        if isinstance(context, dict):
            metadata = context.get("metadata")
        else:
            metadata = getattr(context, "metadata", None)
        if isinstance(metadata, dict):
            return metadata
        return None

    def _clear_response_language(self, context: Any) -> None:
        metadata = self._context_metadata(context)
        if metadata is not None:
            metadata.pop(OUTPUT_LANGUAGE_METADATA_KEY, None)

    def _attach_decision_metadata(self, result: dict[str, Any]) -> None:
        if self.decision is None:
            return
        result.setdefault("auto_decision", self.decision.to_dict())
        metadata = result.get("metadata")
        if isinstance(metadata, dict):
            metadata.setdefault("auto_decision", self.decision.to_dict())

    async def _decide(
        self,
        *,
        context: Any,
        tools: list[Any],
        llm: Any,
        compact_llm: Any | None,
        runtime: PatternRuntime,
        memory_tools_available: bool = False,
    ) -> AutoDecisionResult:
        if llm is None:
            raise RuntimeError("AutoPattern requires an LLM with tool calling support.")

        # Re-derive the request-scoped language before routing so stale metadata
        # cannot bias the current decision prompt.
        self._clear_response_language(context)
        call_llm = await prepare_llm_for_context(
            llm=llm,
            messages=context.get_messages_for_llm(),
            context=context,
        )
        await runtime.compact_context_if_needed(
            context=context,
            llm=compact_llm,
            metadata={"phase": "auto_decision"},
        )

        base_messages = context.get_messages_for_llm()
        current_request = truncate_prompt_preview(
            latest_user_text(context) or "",
            limit=400,
        )
        decision_prompt = self._decision_prompt(
            tools,
            current_request=current_request,
            memory_tools_available=memory_tools_available,
        )
        decision_tools = [self._decision_tool_schema()]
        retry_feedback: str | None = None
        attempt = 0
        # This is a whole-loop budget, independent of the parse attempt budget.
        protocol_retries = 0
        while attempt < MAX_DECISION_PARSE_ATTEMPTS:
            messages = append_user_message_preserving_turns(
                base_messages,
                content=decision_prompt,
                section_title="Auto routing instruction",
            )
            if retry_feedback:
                messages = append_user_message_preserving_turns(
                    messages,
                    content=retry_feedback,
                    section_title="Auto routing retry feedback",
                )
            metadata: dict[str, Any] = {
                "phase": "auto_decision",
                **resolved_llm_metadata(call_llm),
            }
            total_retries = attempt + protocol_retries
            if total_retries:
                metadata["attempt"] = total_retries + 1
            await runtime.on_llm_start(
                context=context,
                messages=messages,
                tools=decision_tools,
                metadata=metadata,
            )
            answer_emitter = FinalAnswerStreamSession(
                runtime,
                enabled=True,
            )
            answer_streamer = ToolCallStringFieldStreamer(
                runtime=runtime,
                tool_name=DECISION_TOOL_NAME,
                field_name="answer",
                guard_field="action",
                guard_value=AutoAction.FINAL_ANSWER.value,
                emitter=answer_emitter,
            )
            try:
                response = await runtime.run_streaming_llm_call(
                    call_llm,
                    messages=messages,
                    tools=decision_tools,
                    tool_choice="required",
                    thinking={"type": "disabled", "enable": False},
                    on_chunk=answer_streamer.handle_chunk,
                )
            except LLMToolProtocolError as exc:
                await answer_streamer.fail(
                    f"invalid {exc.code} routing tool protocol, retrying"
                )
                await runtime.on_llm_error(
                    context=context,
                    error=exc,
                    metadata={
                        **metadata,
                        "phase": exc.code,
                        "protocol_code": exc.code,
                    },
                )
                if protocol_retries >= 1:
                    raise
                protocol_retries += 1
                retry_feedback = self._tool_protocol_retry_feedback(exc)
                logger.warning(
                    "AutoPattern routing received invalid %s tool protocol; "
                    "retrying decision. execution_id=%s attempt=%s",
                    exc.code,
                    getattr(context, "execution_id", None),
                    attempt + protocol_retries,
                )
                await runtime.checkpoint(
                    "auto_decision_retry",
                    context=context,
                    pattern=self,
                    metadata={
                        "attempt": attempt + protocol_retries,
                        "error": str(exc),
                        "protocol_code": exc.code,
                    },
                )
                continue
            except Exception as exc:
                await answer_streamer.fail(str(exc))
                await runtime.on_llm_error(
                    context=context, error=exc, metadata=metadata
                )
                raise
            await runtime.on_llm_end(
                context=context, response=response, metadata=metadata
            )
            try:
                decision = self._parse_decision(response, attempts=attempt + 1)
            except RequiredToolCallError:
                attempt += 1
                if attempt >= MAX_DECISION_PARSE_ATTEMPTS:
                    raise
                retry_feedback = self._required_tool_call_retry_feedback(
                    DECISION_TOOL_NAME
                )
                logger.warning(
                    "AutoPattern decision response omitted required %s tool call; "
                    "retrying decision. execution_id=%s attempt=%s",
                    DECISION_TOOL_NAME,
                    getattr(context, "execution_id", None),
                    attempt,
                )
                await runtime.checkpoint(
                    "auto_decision_retry",
                    context=context,
                    pattern=self,
                    metadata={
                        "attempt": attempt,
                        "failure_reason": REQUIRED_TOOL_CALL_FAILURE_REASON,
                        "required_tool_name": DECISION_TOOL_NAME,
                    },
                )
                continue
            except AutoDecisionArgumentsError as exc:
                attempt += 1
                if attempt >= MAX_DECISION_PARSE_ATTEMPTS:
                    raise
                retry_feedback = self._decision_retry_feedback(exc)
                logger.warning(
                    "AutoPattern decision arguments were invalid JSON; retrying "
                    "decision parse. execution_id=%s error=%s",
                    getattr(context, "execution_id", None),
                    exc,
                )
                await runtime.checkpoint(
                    "auto_decision_retry",
                    context=context,
                    pattern=self,
                    metadata={
                        "attempt": attempt,
                        "error": str(exc),
                    },
                )
                continue
            return AutoDecisionResult(
                decision=decision,
                final_answer_stream=(
                    answer_emitter
                    if decision.action == AutoAction.FINAL_ANSWER
                    else None
                ),
            )
        raise RuntimeError("AutoPattern decision retry loop exited unexpectedly.")

    def _required_tool_call_retry_feedback(self, tool_name: str) -> str:
        return (
            f"The previous response did not call the required {tool_name} tool. "
            f"Call {tool_name} exactly once with a complete routing decision. "
            "Do not answer in natural language."
        )

    def _decision_retry_feedback(self, error: AutoDecisionArgumentsError) -> str:
        argument_preview = self._truncate_retry_preview(error.arguments or "")
        return (
            f"The previous {DECISION_TOOL_NAME} tool call could not be used "
            f"because its arguments were invalid JSON: {error}. Call "
            f"{DECISION_TOOL_NAME} again exactly once. The tool arguments must be "
            "one complete valid JSON object. Do not truncate string values. If "
            "action is final_answer, include the complete non-empty answer field "
            "inside the same tool call."
            + (
                f"\nInvalid arguments preview:\n```json\n{argument_preview}\n```"
                if argument_preview
                else ""
            )
        )

    def _tool_protocol_retry_feedback(self, error: LLMToolProtocolError) -> str:
        if error.code == "malformed_tool_arguments":
            correction = (
                "The provider rejected the previous routing tool call because "
                "its arguments were malformed."
            )
        elif error.code == "unavailable_tool_call":
            correction = (
                "The provider rejected the previous response because it called "
                "a tool that is unavailable during Auto routing."
            )
        else:
            correction = "The provider rejected the previous routing tool call."
        return (
            f"{correction} Retry by calling exactly one currently available "
            f"routing tool: {DECISION_TOOL_NAME}. Use the exact tool name and "
            "one complete JSON object matching its schema. Do not answer in "
            "natural language."
        )

    def _truncate_retry_preview(self, value: str, *, limit: int = 1200) -> str:
        return truncate_prompt_preview(value, limit=limit)

    def _decision_prompt(
        self,
        tools: list[Any],
        *,
        current_request: str = "",
        memory_tools_available: bool = False,
    ) -> str:
        memory_rule = (
            "If the latest user message asks to remember, store, forget, or "
            "update personal information or preferences for future tasks, "
            "choose react so the memory tools can persist the change; never "
            "claim via final_answer that something was remembered, because "
            "final_answer cannot store anything. Likewise, if the user asks "
            "what they previously told you or what you remember about them "
            "and the retrieved context does not already contain the answer, "
            "choose react and look it up with the search_memory tool before "
            "concluding that nothing was recorded. "
            if memory_tools_available
            else ""
        )
        tool_count = len(tools)
        tool_names = self._execution_tool_names(tools)
        tool_capability_summary = (
            f"{tool_count} execution tools are available to the downstream "
            "execution pattern. Available execution tool names: "
            f"{', '.join(tool_names) or '(unavailable)'}."
            if tool_count
            else "No execution tools are available to the downstream execution pattern."
        )
        available_actions = ", ".join(self._available_auto_actions())
        language_anchor = (
            "Latest user request text, quoted for response_language selection:\n"
            f"{current_request or '(unavailable)'}\n\n"
            "Choose response_language from that latest user request, including "
            "any explicit language change requested inside it. Do not choose "
            "response_language from retrieved memories, source documents, "
            "tool results, or earlier turns. "
        )
        return (
            "Choose how the agent should handle the user request. "
            f"You must call the {DECISION_TOOL_NAME} tool exactly once. "
            f"action must be one of: {available_actions}. "
            f"{language_anchor}"
            "Use final_answer for simple conversational replies that need no tools; "
            "when action is final_answer, you must include a complete non-empty "
            "answer field in the same tool call. Put action before answer in the "
            "tool arguments. You must also classify whether "
            "the latest request requires current or external facts, and whether "
            "the existing context is sufficient evidence for those facts. "
            "Set response_language to the natural language that user-facing prose "
            "should use for this request, such as English, Simplified Chinese, "
            "Traditional Chinese, Spanish, or the language explicitly requested "
            "by the user. For Chinese requests, choose Simplified Chinese or "
            "Traditional Chinese to match the request script; do not use generic "
            "Chinese. This is a routing decision field only; do not translate or "
            "rewrite the request. "
            "If the latest user message explicitly asks to call or use an available "
            "tool, to pause for user input, or to wait for a user choice, choose "
            "react; do not choose final_answer merely to restate or paraphrase the "
            "requested tool action. "
            f"{memory_rule}"
            "A final answer means the latest user request is already complete and "
            "no tool or other work will happen after this routing decision. If the "
            "user-facing answer would describe a future tool action, ask the user "
            "to wait, or say work is still in progress, choose react instead. If "
            "satisfying the request requires an execution tool to create, modify, "
            "fetch, send, or otherwise produce a result that is not already present "
            "in the conversation, choose react even when the user did not explicitly "
            "mention a tool. "
            "A file reference, attachment link, filename, MIME type, or other "
            "attachment metadata does not mean the file contents have been "
            "inspected. If the answer depends on an uploaded image, document, "
            "audio file, or other attachment and no prior tool result explicitly "
            "contains the relevant inspection or parsed content, choose react and "
            "inspect it with the appropriate execution tool. Never claim to see, "
            "read, or hear attachment contents from reference metadata alone. "
            "Use final_answer only when the current conversation and available "
            "retrieved context already provide enough evidence for the answer. "
            "Available retrieved context includes "
            "prior tool results, knowledge base or RAG results, file contents, "
            "memory, and other trusted context already present in the conversation, "
            "but memory or skill instructions by themselves are not proof that a "
            "new public factual claim is supported. "
            "For requests about recent/latest/current public facts, news, security "
            "incidents, affected vendors, dates, vulnerabilities, versions, or "
            "source-backed claims, set requires_current_or_external_facts=true. "
            "If those facts are not explicitly supported by current conversation, "
            "prior tool results, files, or retrieved context, set "
            "existing_context_sufficient=false and choose react so the agent can "
            "verify using available tools or explicitly state that the context is "
            "insufficient; in that case, do not choose final_answer. "
            "Use react as the default tool-use mode for ordinary searches, "
            "single-target tool calls, narrow corrections, and short iterative work. "
            "For follow-up requests, interpret the latest user message in the full "
            "conversation context and route by the actual work required now. The "
            "latest/current user request is the task to answer; earlier messages "
            "are context only and must not be re-added as separate requirements "
            "unless the latest request explicitly asks to revisit them. Do not "
            "choose plan_execute merely because the request references, corrects, "
            "narrows, or extends a previous answer. "
            "Use plan_execute only when the current request requires explicit "
            "multi-step planning, dependency management, parallel subtask execution, "
            "multiple coordinated deliverables, or user-visible DAG execution. "
            f"{tool_capability_summary} Do not call those execution tools during "
            "this routing decision; only call the routing tool provided in this "
            "request."
        )

    @staticmethod
    def _execution_tool_names(tools: list[Any]) -> list[str]:
        names: list[str] = []
        seen: set[str] = set()
        for tool in tools:
            name: Any = None
            if isinstance(tool, dict):
                function = tool.get("function")
                if isinstance(function, dict):
                    name = function.get("name")
                if not name:
                    name = tool.get("name")
            else:
                metadata = getattr(tool, "metadata", None)
                if metadata is not None:
                    name = getattr(metadata, "name", None)
                if not name:
                    name = getattr(tool, "name", None)
                if not name:
                    name = getattr(tool, "__name__", None)

            normalized = str(name or "").strip()
            if normalized and normalized not in seen:
                names.append(normalized)
                seen.add(normalized)
        return names

    def _decision_tool_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": DECISION_TOOL_NAME,
                "description": (
                    "Select the execution pattern for this user request. If action "
                    "is final_answer, the answer argument is mandatory and must be "
                    "a complete non-empty final response to the user. "
                    f"{final_answer_language_rule()}"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": self._available_auto_actions(),
                            "description": "Execution action to use.",
                        },
                        "reason": {
                            "type": "string",
                            "description": "Brief reason for the selected action.",
                        },
                        "response_language": {
                            "type": "string",
                            "description": (
                                "Natural language to use for all user-facing prose "
                                "and persisted tool-argument prose for this request, "
                                "for example English, Simplified Chinese, Traditional "
                                "Chinese, or Spanish. For Chinese requests, choose "
                                "Simplified Chinese or Traditional Chinese to match "
                                "the request script; do not use generic Chinese. If "
                                "the current user request explicitly asks to answer in "
                                "another language, use that requested target language. "
                                f"{output_language_policy()}"
                            ),
                        },
                        "answer": {
                            "type": "string",
                            "description": (
                                "Required for every decision. When action is "
                                "final_answer, provide the complete non-empty final "
                                "response to the user. Use an empty string for react "
                                f"or plan_execute. {final_answer_language_rule()}"
                            ),
                        },
                        "requires_current_or_external_facts": {
                            "type": "boolean",
                            "description": (
                                "True when the latest request depends on current, "
                                "recent, public, source-backed, or otherwise "
                                "external facts rather than only conversation-local "
                                "reasoning."
                            ),
                        },
                        "existing_context_sufficient": {
                            "type": "boolean",
                            "description": (
                                "True only when the current conversation, prior tool "
                                "results, files, or retrieved context explicitly "
                                "support the facts needed to answer. Memory and skill "
                                "instructions are supporting context, not automatic "
                                "evidence for new public factual claims."
                            ),
                        },
                        "evidence_basis": {
                            "type": "string",
                            "description": (
                                "Briefly state what existing evidence supports the "
                                "answer, such as conversation context, prior tool "
                                "results, files, retrieved knowledge, or none."
                            ),
                        },
                        "missing_verification": {
                            "type": "string",
                            "description": (
                                "If existing_context_sufficient is false, state what "
                                "needs verification before a final answer is safe. "
                                "Use an empty string when no verification is missing."
                            ),
                        },
                    },
                    "required": [
                        "action",
                        "reason",
                        "response_language",
                        "answer",
                        "requires_current_or_external_facts",
                        "existing_context_sufficient",
                        "evidence_basis",
                        "missing_verification",
                    ],
                    "additionalProperties": False,
                },
            },
        }

    def _available_auto_actions(self) -> list[str]:
        actions = [AutoAction.FINAL_ANSWER.value, AutoAction.REACT.value]
        if self.dag_pattern is not None:
            actions.append(AutoAction.PLAN_EXECUTE.value)
        return actions

    def _parse_decision(self, response: Any, *, attempts: int = 1) -> AutoDecision:
        payload = self._extract_tool_arguments(
            response,
            DECISION_TOOL_NAME,
            attempts=attempts,
        )
        return AutoDecision.from_dict(payload)

    def _extract_tool_arguments(
        self,
        response: Any,
        tool_name: str,
        *,
        attempts: int = 1,
    ) -> dict[str, Any]:
        arguments = extract_required_tool_arguments(
            response,
            tool_name=tool_name,
            owner="AutoPattern decision",
            attempts=attempts,
            user_message=AUTO_DECISION_REQUIRED_TOOL_MESSAGE,
        )
        return self._coerce_arguments(arguments)

    def _coerce_arguments(self, arguments: Any) -> dict[str, Any]:
        if isinstance(arguments, dict):
            return arguments
        if not isinstance(arguments, str):
            raise TypeError("Tool call arguments must be an object or JSON string.")
        try:
            payload = json.loads(arguments)
        except json.JSONDecodeError as exc:
            payload = self._repair_json_arguments(
                arguments=arguments,
                original_error=exc,
            )
        if not isinstance(payload, dict):
            raise AutoDecisionArgumentsError(
                "Tool call arguments must decode to an object.",
                arguments=arguments,
            )
        return payload

    def _repair_json_arguments(
        self,
        *,
        arguments: str,
        original_error: json.JSONDecodeError,
    ) -> Any:
        if not self._is_structurally_complete_json_object(arguments):
            raise AutoDecisionArgumentsError(
                "Tool call arguments appear truncated and must be retried.",
                arguments=arguments,
            ) from original_error
        try:
            return repair_json_loads(arguments, logging=False)
        except Exception as exc:
            raise AutoDecisionArgumentsError(
                "Tool call arguments must be valid JSON.",
                arguments=arguments,
            ) from original_error or exc

    def _is_structurally_complete_json_object(self, arguments: str) -> bool:
        stripped = arguments.strip()
        if not stripped.startswith("{") or not stripped.endswith("}"):
            return False

        pairs = {"}": "{", "]": "["}
        stack: list[str] = []
        in_string = False
        escaped = False
        for char in stripped:
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue

            if char == '"':
                in_string = True
            elif char in "{[":
                stack.append(char)
            elif char in pairs:
                if not stack or stack.pop() != pairs[char]:
                    return False

        return not in_string and not escaped and not stack

    def _selected_child(self) -> AgentPattern:
        if self.decision is None:
            raise RuntimeError("AutoPattern has no decision.")
        if self.decision.action == AutoAction.REACT:
            return self.react_pattern
        if self.decision.action == AutoAction.PLAN_EXECUTE:
            if self.dag_pattern is None:
                raise RuntimeError(
                    "AutoPattern selected plan_execute without a DAGPattern."
                )
            return self.dag_pattern
        raise RuntimeError(f"AutoPattern action has no child: {self.decision.action}")

    def _capture_child_state(self, child: Any) -> None:
        get_state = getattr(child, "get_state", None)
        if not callable(get_state):
            return
        if child is self.react_pattern:
            self.react_state = get_state()
            return
        if child is self.dag_pattern:
            self.dag_state = get_state()

    def _restore_child_state(self, child: Any) -> None:
        state = None
        if child is self.react_pattern:
            state = self.react_state
        elif child is self.dag_pattern:
            state = self.dag_state
        if not state:
            return
        load_state = getattr(child, "load_state", None)
        if callable(load_state):
            load_state(state)

    def get_state(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "decision": self.decision.to_dict() if self.decision is not None else None,
            "decision_user_messages": self.decision_user_messages,
            "selected_pattern": self.selected_pattern,
            "react_state": self.react_state,
            "dag_state": self.dag_state,
            "last_result": self.last_result,
        }

    def load_state(self, state: dict[str, Any]) -> None:
        self.status = str(state.get("status", "idle"))
        decision = state.get("decision")
        self.decision = (
            AutoDecision.from_dict(decision) if isinstance(decision, dict) else None
        )
        decision_user_messages = state.get("decision_user_messages")
        self.decision_user_messages = (
            dict(decision_user_messages)
            if isinstance(decision_user_messages, dict)
            else None
        )
        self.selected_pattern = state.get("selected_pattern")
        self.react_state = state.get("react_state")
        self.dag_state = state.get("dag_state")
        self.last_result = state.get("last_result")

    def get_execution_snapshot(self, context: Any) -> dict[str, Any]:
        root_execution_id = getattr(context, "execution_id", "unknown")
        root_frame_id = f"{root_execution_id}:auto"
        child_frame_id = self._child_frame_id(root_execution_id)
        root_frame = ExecutionFrame(
            frame_id=root_frame_id,
            root_execution_id=root_execution_id,
            pattern_type="auto",
            status=self._execution_status(self.status),
            context=context.to_dict()
            if callable(getattr(context, "to_dict", None))
            else {},
            pattern_state=self.get_state(),
            children=[child_frame_id] if child_frame_id else [],
            active_child_id=child_frame_id,
            metadata={"selected_pattern": self.selected_pattern},
        )
        frames = {root_frame_id: root_frame}
        active_frame_ids = [root_frame_id]
        child_snapshot = self._child_execution_snapshot(context)
        if child_snapshot is not None:
            child_frames = {
                frame_id: ExecutionFrame.from_dict(frame)
                for frame_id, frame in child_snapshot.get("frames", {}).items()
                if isinstance(frame, dict)
            }
            child_root_id = self._child_snapshot_root_frame_id(
                child_snapshot, child_frames
            )
            if child_root_id is not None and child_root_id in child_frames:
                child_root = child_frames[child_root_id]
                child_root.parent_frame_id = root_frame_id
                root_frame.children = [child_root_id]
                root_frame.active_child_id = child_root_id
                frames.update(child_frames)
                child_active = [
                    frame_id
                    for frame_id in child_snapshot.get("active_frame_ids", [])
                    if frame_id in child_frames
                ]
                active_frame_ids.extend(child_active or [child_root_id])
        elif child_frame_id is not None:
            child_type, child_state = self._child_snapshot_state()
            frames[child_frame_id] = ExecutionFrame(
                frame_id=child_frame_id,
                parent_frame_id=root_frame_id,
                root_execution_id=root_execution_id,
                pattern_type=child_type,
                status=self._execution_status(self.status),
                context=context.to_dict()
                if callable(getattr(context, "to_dict", None))
                else {},
                pattern_state=child_state,
            )
            active_frame_ids.append(child_frame_id)
        return ExecutionSnapshot(
            root_execution_id=root_execution_id,
            status=self._execution_status(self.status),
            frames=frames,
            active_frame_ids=active_frame_ids,
            control_state={"selected_pattern": self.selected_pattern},
        ).to_dict()

    def _child_execution_snapshot(self, context: Any) -> dict[str, Any] | None:
        child = self._snapshot_child_pattern()
        get_snapshot = getattr(child, "get_execution_snapshot", None)
        if not callable(get_snapshot):
            return None
        try:
            snapshot = get_snapshot(context)
        except Exception:
            logger.exception("Failed to build AutoPattern child execution snapshot")
            return None
        return snapshot if isinstance(snapshot, dict) else None

    def _snapshot_child_pattern(self) -> Any | None:
        if self.selected_pattern == AutoAction.PLAN_EXECUTE.value:
            return self.dag_pattern
        # DAG exposes a rich execution snapshot for visualization. ReAct is
        # represented by the synthetic child frame created by AutoPattern.
        return None

    def _invalidate_stale_final_answer_decision(self, context: Any) -> None:
        if (
            self.decision is None
            or self.decision.action != AutoAction.FINAL_ANSWER
            or self.decision.answer is None
        ):
            return

        current_signature = self._user_message_signature(context)
        if self.decision_user_messages is None:
            stale = int(current_signature.get("count", 0)) > 1
        else:
            stale = current_signature != self.decision_user_messages

        if not stale:
            return

        logger.info(
            "AutoPattern invalidating cached final_answer decision after new user "
            "input. previous=%s current=%s",
            self.decision_user_messages,
            current_signature,
        )
        self.decision = None
        self.decision_user_messages = None
        self.selected_pattern = None
        self.last_result = None
        self.status = "idle"
        self._clear_request_scoped_enrichment(context)

    def _clear_request_scoped_enrichment(self, context: Any) -> None:
        metadata = getattr(context, "metadata", None)
        if not isinstance(metadata, dict):
            return
        for key in (
            RETRIEVED_MEMORIES_METADATA_KEY,
            MEMORY_CONTEXT_METADATA_KEY,
            SELECTED_SKILL_METADATA_KEY,
            SKILL_CONTEXT_METADATA_KEY,
        ):
            metadata.pop(key, None)

    def _user_message_signature(self, context: Any) -> dict[str, Any]:
        user_contents = [
            str(getattr(message, "content", ""))
            for message in getattr(context, "messages", [])
            if getattr(message, "role", None) == "user"
        ]
        return {
            "count": len(user_contents),
            "latest": user_contents[-1] if user_contents else "",
        }

    def _child_snapshot_root_frame_id(
        self,
        snapshot: dict[str, Any],
        frames: dict[str, ExecutionFrame],
    ) -> str | None:
        active_frame_ids = [
            frame_id
            for frame_id in snapshot.get("active_frame_ids", [])
            if isinstance(frame_id, str) and frame_id in frames
        ]
        if active_frame_ids:
            return active_frame_ids[0]
        for frame_id, frame in frames.items():
            if frame.parent_frame_id is None:
                return frame_id
        return None

    def _child_frame_id(self, root_execution_id: str) -> str | None:
        if self.selected_pattern in {
            AutoAction.REACT.value,
            AutoAction.PLAN_EXECUTE.value,
        }:
            return f"{root_execution_id}:auto:{self.selected_pattern}"
        return None

    def _child_snapshot_state(self) -> tuple[str, dict[str, Any]]:
        if self.selected_pattern == AutoAction.REACT.value:
            return "react", dict(self.react_state or {})
        return "dag", dict(self.dag_state or {})

    def _execution_status(self, status: str) -> ExecutionStatus:
        status_map = {
            "completed": ExecutionStatus.COMPLETED,
            "failed": ExecutionStatus.FAILED,
            "interrupted": ExecutionStatus.INTERRUPTED,
            "waiting_for_user": ExecutionStatus.WAITING_FOR_USER,
        }
        return status_map.get(status, ExecutionStatus.RUNNING)
