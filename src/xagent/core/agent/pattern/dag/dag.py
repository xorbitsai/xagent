from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any

from ....model.intent import goal_scope
from ...context.enrichment import (
    enrich_context_with_memory,
    latest_user_text,
)
from ...frame import ExecutionFrame, ExecutionSnapshot, ExecutionStatus
from ...language import (
    OUTPUT_LANGUAGE_METADATA_KEY,
    final_answer_language_rule,
    output_language_policy,
)
from ...result import unwrap_final_answer_content
from ...runtime import ExecutionInterrupted, LLMCallInterrupted, PatternRuntime
from ..base import AgentPattern, PatternResult, RequiredToolCallError
from ..final_answer_stream import FinalAnswerStreamSession, ToolCallStringFieldStreamer
from ..react import ReActPattern, ReActReasoningMode
from .plan_generator import (
    CallablePlanGenerator,
    ExecutionPlan,
    PlanGenerationRequest,
    PlanGenerator,
    PlanStep,
    PlanValidationError,
)

logger = logging.getLogger(__name__)

DAG_COMPLETION_TOOL_NAME = "assess_dag_completion"


@dataclass
class DAGCompletionAssessment:
    complete: bool
    answer: str = ""
    reason: str = ""
    missing_work: str = ""
    replan_instruction: str = ""


@dataclass
class _DAGStepRuntime:
    """Runtime adapter that checkpoints child ReAct state into the root DAG state."""

    parent: PatternRuntime
    dag_pattern: "DAGPattern"
    root_context: Any
    step_id: str

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
    def active_react_step_id(self) -> str:
        return self.step_id

    async def should_interrupt(self) -> bool:
        return await self.parent.should_interrupt()

    async def run_llm_call(self, llm: Any, **kwargs: Any) -> Any:
        return await self.parent.run_llm_call(llm, **kwargs)

    async def run_tool_call(self, invoke: Any) -> Any:
        return await self.parent.run_tool_call(invoke)

    async def run_streaming_llm_call(
        self,
        llm: Any,
        *,
        on_chunk: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        del on_chunk
        return await self.parent.run_streaming_llm_call(llm, **kwargs)

    async def stream_final_answer(self, llm: Any, **kwargs: Any) -> Any:
        return await self.parent.run_llm_call(llm, **kwargs)

    async def start_final_answer_stream(self) -> str | None:
        return None

    async def emit_final_answer_delta(self, message_id: str, delta: str) -> None:
        del message_id, delta

    async def end_final_answer_stream(self, message_id: str, content: str) -> None:
        del message_id, content

    async def fail_final_answer_stream(self, message_id: str, error: str) -> None:
        del message_id, error

    async def send_message(
        self,
        *,
        message: str,
        message_type: str = "info",
        expect_response: bool = False,
        visible: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        outbound_metadata = dict(metadata or {})
        outbound_metadata.setdefault("step_id", self.step_id)
        outbound_metadata.setdefault("dag_step_id", self.step_id)
        return await self.parent.send_message(
            message=message,
            message_type=message_type,
            expect_response=expect_response,
            visible=visible,
            metadata=outbound_metadata,
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
        self.dag_pattern._set_active_step_context(self.step_id, context.to_dict())
        get_state = getattr(pattern, "get_state", None)
        if callable(get_state):
            self.dag_pattern._set_active_step_pattern_state(
                self.step_id,
                get_state(),
            )
        step_metadata = {
            "active_step_id": self.step_id,
            "child_label": label,
        }
        if metadata:
            step_metadata.update(metadata)
        return await self.parent.checkpoint(
            label=f"dag_{label}",
            context=self.root_context,
            pattern=self.dag_pattern,
            status=status,
            metadata=step_metadata,
        )

    async def on_tool_start(self, *, tool_call: dict[str, Any]) -> None:
        await self.parent.on_tool_start(tool_call=self._with_step(tool_call))

    async def on_tool_end(self, *, tool_call: dict[str, Any], result: Any) -> None:
        await self.parent.on_tool_end(
            tool_call=self._with_step(tool_call), result=result
        )

    async def on_tool_error(
        self,
        *,
        tool_call: dict[str, Any],
        error: Exception,
        result: Any | None = None,
    ) -> None:
        await self.parent.on_tool_error(
            tool_call=self._with_step(tool_call), error=error, result=result
        )

    async def on_tool_cancelled(
        self,
        *,
        tool_call: dict[str, Any],
        reason: str | None = None,
    ) -> None:
        await self.parent.on_tool_cancelled(
            tool_call=self._with_step(tool_call), reason=reason
        )

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
            metadata={
                "task_id": self.root_context.execution_id,
                "step_id": self.step_id,
                "dag_step_id": self.step_id,
                **(metadata or {}),
            },
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
            metadata={
                "task_id": self.root_context.execution_id,
                "step_id": self.step_id,
                "dag_step_id": self.step_id,
                **(metadata or {}),
            },
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
            metadata={
                "task_id": self.root_context.execution_id,
                "step_id": self.step_id,
                "dag_step_id": self.step_id,
                **(metadata or {}),
            },
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
            metadata={
                "task_id": self.root_context.execution_id,
                "step_id": self.step_id,
                "dag_step_id": self.step_id,
                **(metadata or {}),
            },
        )

    def _with_step(self, tool_call: dict[str, Any]) -> dict[str, Any]:
        return {
            **tool_call,
            "task_id": self.root_context.execution_id,
            "step_id": self.step_id,
            "dag_step_id": self.step_id,
        }

    async def on_pattern_start(self, *, context: Any, pattern: Any) -> None:
        del context, pattern

    async def on_pattern_end(
        self,
        *,
        context: Any,
        pattern: Any,
        result: dict[str, Any],
    ) -> None:
        del context, pattern, result

    async def on_pattern_error(
        self,
        *,
        context: Any,
        pattern: Any,
        error: Exception,
    ) -> None:
        del context, pattern, error


@dataclass
class _RuntimeLLMProxy:
    runtime: PatternRuntime
    llm: Any

    async def chat(self, **kwargs: Any) -> Any:
        return await self.runtime.run_llm_call(self.llm, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.llm, name)


class DAGPattern(AgentPattern):
    """Minimal DAG execution pattern that reuses ReActPattern for each step."""

    def __init__(
        self,
        plan_generator: PlanGenerator | Any,
        *,
        react_max_iterations: int = 200,
        react_reasoning_mode: ReActReasoningMode
        | str = ReActReasoningMode.TOOL_CALLING,
        max_concurrency: int = 4,
        max_completion_replans: int = 3,
    ) -> None:
        self.plan_generator = (
            plan_generator
            if isinstance(plan_generator, PlanGenerator)
            else CallablePlanGenerator(plan_generator)
        )
        self.react_max_iterations = react_max_iterations
        self.react_reasoning_mode = ReActReasoningMode(react_reasoning_mode)
        self.max_concurrency = max(1, max_concurrency)
        self.max_completion_replans = max(0, max_completion_replans)
        self.status = "idle"
        self.plan: ExecutionPlan | None = None
        self.active_step_id: str | None = None
        self.active_step_pattern_state: dict[str, Any] | None = None
        self.active_step_context: dict[str, Any] | None = None
        self.active_step_ids: list[str] = []
        self.active_step_pattern_states: dict[str, dict[str, Any]] = {}
        self.active_step_contexts: dict[str, dict[str, Any]] = {}
        self.step_results: dict[str, Any] = {}
        self.planned_user_message_count = 0
        self.completion_feedback: str | None = None
        self.completion_replan_count = 0

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

        if llm is None:
            self.status = "failed"
            return PatternResult(
                success=False,
                error="DAGPattern requires an llm instance.",
                metadata={
                    "status": self.status,
                    "failure_reason": "missing_llm",
                },
            ).to_dict()

        await runtime.on_pattern_start(context=context, pattern=self)
        try:
            result = await self._run(
                context=context,
                tools=tools,
                llm=llm,
                compact_llm=kwargs.get("compact_llm"),
                runtime=runtime,
                memory_store=kwargs.get("memory_store"),
                memory_similarity_threshold=kwargs.get("memory_similarity_threshold"),
                skill_manager=kwargs.get("skill_manager"),
                allowed_skills=kwargs.get("allowed_skills"),
            )
        except RequiredToolCallError as exc:
            result = await self._fail(
                context=context,
                runtime=runtime,
                error=exc.user_message,
                failure_reason=exc.failure_reason,
                checkpoint_label="dag_plan_generation_failed",
                extra_metadata=exc.to_metadata(),
            )
            await runtime.on_pattern_end(context=context, pattern=self, result=result)
            return result
        except Exception as exc:
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
        compact_llm: Any | None = None,
        memory_store: Any | None = None,
        memory_similarity_threshold: float | None = None,
        skill_manager: Any | None = None,
        allowed_skills: list[str] | None = None,
    ) -> dict[str, Any]:
        if self.plan is None:
            self.status = "planning"
        try:
            if self.plan is None:
                task_text = latest_user_text(context)
                await enrich_context_with_memory(
                    context=context,
                    query=task_text,
                    category="dag_plan_execute_memory",
                    memory_store=memory_store,
                    runtime=runtime,
                    similarity_threshold=memory_similarity_threshold,
                )
                interrupted = await self._interrupt_if_requested(
                    runtime=runtime,
                    context=context,
                    label="dag_before_plan",
                )
                if interrupted is not None:
                    return interrupted
                await self._generate_plan(
                    context=context,
                    tools=tools,
                    llm=llm,
                    runtime=runtime,
                    replan=False,
                )
                interrupted = await self._interrupt_if_requested(
                    runtime=runtime,
                    context=context,
                    label="dag_after_plan",
                )
                if interrupted is not None:
                    return interrupted
            elif self._needs_replan(context):
                if not self._forward_user_response_to_waiting_step(context):
                    await self._generate_plan(
                        context=context,
                        tools=tools,
                        llm=llm,
                        runtime=runtime,
                        replan=True,
                    )
        except PlanValidationError as exc:
            return await self._fail(
                context=context,
                runtime=runtime,
                error=str(exc),
                failure_reason="invalid_plan",
                checkpoint_label="dag_plan_invalid",
            )
        except LLMCallInterrupted:
            interrupted = await self._interrupt_if_requested(
                runtime=runtime,
                context=context,
                label="dag_during_plan",
            )
            if interrupted is not None:
                return interrupted
            raise
        except RequiredToolCallError:
            raise
        except Exception as exc:  # noqa: BLE001
            return await self._fail(
                context=context,
                runtime=runtime,
                error=str(exc),
                failure_reason=(
                    "replan_generation_error"
                    if self.status == "replanning"
                    else "plan_generation_error"
                ),
                checkpoint_label="dag_plan_generation_failed",
            )

        while True:
            if self._needs_replan(context):
                if not self._forward_user_response_to_waiting_step(context):
                    try:
                        await self._generate_plan(
                            context=context,
                            tools=tools,
                            llm=llm,
                            runtime=runtime,
                            replan=True,
                        )
                    except PlanValidationError as exc:
                        return await self._fail(
                            context=context,
                            runtime=runtime,
                            error=str(exc),
                            failure_reason="invalid_plan",
                            checkpoint_label="dag_plan_invalid",
                        )
                    except LLMCallInterrupted:
                        interrupted = await self._interrupt_if_requested(
                            runtime=runtime,
                            context=context,
                            label="dag_during_replan",
                        )
                        if interrupted is not None:
                            return interrupted
                        raise
                    except RequiredToolCallError:
                        raise
                    except Exception as exc:  # noqa: BLE001
                        return await self._fail(
                            context=context,
                            runtime=runtime,
                            error=str(exc),
                            failure_reason="replan_generation_error",
                            checkpoint_label="dag_plan_generation_failed",
                        )

            interrupted = await self._interrupt_if_requested(
                runtime=runtime,
                context=context,
                label="dag_before_ready_steps",
            )
            if interrupted is not None:
                return interrupted

            ready_steps = self._ready_steps()
            if not ready_steps:
                if self.plan is None:
                    return await self._fail(
                        context=context,
                        runtime=runtime,
                        error="DAGPattern plan was not initialized.",
                        failure_reason="plan_not_initialized",
                        checkpoint_label="dag_plan_missing",
                    )
                if self._all_steps_completed():
                    completion_result = await self._handle_completed_plan(
                        context=context,
                        tools=tools,
                        llm=llm,
                        runtime=runtime,
                    )
                    if completion_result is not None:
                        return completion_result
                    continue
                failed_step = next(
                    (step for step in self.plan.steps if step.status == "failed"),
                    None,
                )
                if failed_step is not None:
                    return await self._fail(
                        context=context,
                        runtime=runtime,
                        error=failed_step.error or f"Step {failed_step.id} failed.",
                        failure_reason="step_failed",
                        checkpoint_label="dag_failed",
                        failed_step_id=failed_step.id,
                    )
                return await self._fail(
                    context=context,
                    runtime=runtime,
                    error="DAGPattern has no executable steps.",
                    failure_reason="no_executable_steps",
                    checkpoint_label="dag_no_executable_steps",
                )

            batch = ready_steps[: self.max_concurrency]
            step_result = await self._execute_ready_steps(
                steps=batch,
                root_context=context,
                tools=tools,
                llm=llm,
                compact_llm=compact_llm,
                runtime=runtime,
                memory_store=memory_store,
                memory_similarity_threshold=memory_similarity_threshold,
                skill_manager=skill_manager,
                allowed_skills=allowed_skills,
            )
            if step_result is not None:
                return step_result

    async def _execute_ready_steps(
        self,
        *,
        steps: list[PlanStep],
        root_context: Any,
        tools: list[Any],
        llm: Any,
        runtime: PatternRuntime,
        compact_llm: Any | None = None,
        memory_store: Any | None = None,
        memory_similarity_threshold: float | None = None,
        skill_manager: Any | None = None,
        allowed_skills: list[str] | None = None,
    ) -> dict[str, Any] | None:
        if not steps:
            return None
        if len(steps) == 1:
            step = steps[0]
            result = await self._execute_step(
                step=steps[0],
                root_context=root_context,
                tools=tools,
                llm=llm,
                compact_llm=compact_llm,
                runtime=runtime,
                memory_store=memory_store,
                memory_similarity_threshold=memory_similarity_threshold,
                skill_manager=skill_manager,
                allowed_skills=allowed_skills,
            )
            if result is not None:
                return result
            if step.status == "failed":
                return await self._fail(
                    context=root_context,
                    runtime=runtime,
                    error=step.error or f"Step {step.id} failed.",
                    failure_reason="step_failed",
                    checkpoint_label="dag_failed",
                    failed_step_id=step.id,
                )
            return None

        self.status = "running"
        for step in steps:
            self._mark_step_active(step.id)
            step.status = "running"
        await runtime.checkpoint(
            "dag_before_ready_batch",
            context=root_context,
            pattern=self,
            metadata={
                "active_step_ids": [step.id for step in steps],
                "max_concurrency": self.max_concurrency,
            },
        )

        tasks = {
            asyncio.create_task(
                self._execute_step(
                    step=step,
                    root_context=root_context,
                    tools=tools,
                    llm=llm,
                    compact_llm=compact_llm,
                    runtime=runtime,
                    memory_store=memory_store,
                    memory_similarity_threshold=memory_similarity_threshold,
                    skill_manager=skill_manager,
                    allowed_skills=allowed_skills,
                ),
                name=f"dag_step_{step.id}",
            ): step.id
            for step in steps
        }
        steps_by_id = {step.id: step for step in steps}
        pending = set(tasks)
        scheduled_step_ids = set(steps_by_id)

        def schedule_ready_steps() -> None:
            if self.plan is None:
                return
            available_slots = self.max_concurrency - len(pending)
            if available_slots <= 0:
                return
            for ready_step in self._pending_ready_steps():
                if ready_step.id in scheduled_step_ids:
                    continue
                self._mark_step_active(ready_step.id)
                ready_step.status = "running"
                task = asyncio.create_task(
                    self._execute_step(
                        step=ready_step,
                        root_context=root_context,
                        tools=tools,
                        llm=llm,
                        compact_llm=compact_llm,
                        runtime=runtime,
                        memory_store=memory_store,
                        memory_similarity_threshold=memory_similarity_threshold,
                        skill_manager=skill_manager,
                        allowed_skills=allowed_skills,
                    ),
                    name=f"dag_step_{ready_step.id}",
                )
                tasks[task] = ready_step.id
                steps_by_id[ready_step.id] = ready_step
                pending.add(task)
                scheduled_step_ids.add(ready_step.id)
                available_slots -= 1
                if available_slots <= 0:
                    break

        try:
            while pending:
                done, pending = await asyncio.wait(
                    pending,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in done:
                    result = task.result()
                    if result is not None:
                        await self._cancel_pending_steps(
                            pending,
                            step_ids_by_task=tasks,
                            steps_by_id=steps_by_id,
                        )
                        if pending:
                            await runtime.checkpoint(
                                "dag_after_cancelled_siblings",
                                context=root_context,
                                pattern=self,
                                status=self.status,
                                metadata={
                                    "active_step_ids": list(self.active_step_ids),
                                    "cancelled_step_ids": [
                                        step_id
                                        for task, step_id in tasks.items()
                                        if task in pending
                                    ],
                                },
                            )
                        return result
                    failed_step = next(
                        (
                            step
                            for step in steps_by_id.values()
                            if step.status == "failed"
                        ),
                        None,
                    )
                    if failed_step is not None:
                        await self._cancel_pending_steps(
                            pending,
                            step_ids_by_task=tasks,
                            steps_by_id=steps_by_id,
                        )
                        return await self._fail(
                            context=root_context,
                            runtime=runtime,
                            error=(
                                failed_step.error or f"Step {failed_step.id} failed."
                            ),
                            failure_reason="step_failed",
                            checkpoint_label="dag_failed",
                            failed_step_id=failed_step.id,
                        )
                    if self._needs_replan(root_context):
                        await self._cancel_pending_steps(
                            pending,
                            step_ids_by_task=tasks,
                            steps_by_id=steps_by_id,
                        )
                        return None
                    if self.status in {"interrupted", "waiting_for_user"}:
                        await self._cancel_pending_steps(
                            pending,
                            step_ids_by_task=tasks,
                            steps_by_id=steps_by_id,
                        )
                        return None
                schedule_ready_steps()
        except Exception:
            try:
                await self._cancel_pending_steps(
                    pending,
                    step_ids_by_task=tasks,
                    steps_by_id=steps_by_id,
                )
            except Exception:
                logger.exception("Failed to clean up cancelled DAG sibling steps.")
            raise
        return None

    async def _execute_step(
        self, *, step: PlanStep, **kwargs: Any
    ) -> dict[str, Any] | None:
        # While a step runs, its own objective is the active goal — so the
        # "auto" model routes that step on the step, not the whole-task request.
        with goal_scope(step.description or step.task):
            return await self._execute_step_impl(step=step, **kwargs)

    async def _execute_step_impl(
        self,
        *,
        step: PlanStep,
        root_context: Any,
        tools: list[Any],
        llm: Any,
        runtime: PatternRuntime,
        compact_llm: Any | None = None,
        memory_store: Any | None = None,
        memory_similarity_threshold: float | None = None,
        skill_manager: Any | None = None,
        allowed_skills: list[str] | None = None,
    ) -> dict[str, Any] | None:
        self.status = "running"
        self._mark_step_active(step.id)
        step.status = "running"

        active_context = self.active_step_contexts.get(step.id)
        output_language = self._output_language(root_context)
        if active_context is not None:
            child_context = type(root_context).from_dict(active_context)
            if output_language and not child_context.metadata.get(
                OUTPUT_LANGUAGE_METADATA_KEY
            ):
                child_context.metadata[OUTPUT_LANGUAGE_METADATA_KEY] = output_language
        else:
            child_context = root_context.create_child_context(
                metadata={
                    "dag_step_id": step.id,
                    "dag_step_name": step.task,
                    "dag_step_description": step.description or step.task,
                    "dag_dependencies": list(step.dependencies),
                    "dag_tool_names": list(step.tool_names),
                    OUTPUT_LANGUAGE_METADATA_KEY: output_language,
                },
            )
            if step.dependencies:
                child_context.add_user_message(
                    f"Dependency results: {self._dependency_summary(step)}",
                    metadata={
                        "kind": "dag_dependency_results",
                        "dag_step_id": step.id,
                    },
                )
            child_context.add_user_message(
                self._step_instruction(root_context=root_context, step=step),
                metadata={
                    "kind": "dag_step_instruction",
                    "dag_step_id": step.id,
                    "dag_completion_evidence": step.completion_evidence or "",
                },
            )

        react_pattern = ReActPattern(
            max_iterations=self.react_max_iterations,
            reasoning_mode=self.react_reasoning_mode,
            finalize_after_tool_result=False,
        )
        active_pattern_state = self.active_step_pattern_states.get(step.id)
        if active_pattern_state is not None:
            react_pattern.load_state(active_pattern_state)
            react_pattern.finalize_after_tool_result = False

        step_runtime = _DAGStepRuntime(
            parent=runtime,
            dag_pattern=self,
            root_context=root_context,
            step_id=step.id,
        )
        await runtime.checkpoint(
            "dag_before_step",
            context=root_context,
            pattern=self,
            metadata={"active_step_id": step.id},
        )
        await runtime.on_dag_step_start(
            context=root_context,
            step_id=step.id,
            data={
                "step_id": step.id,
                "step_name": step.task,
                "step_task": step.task,
                "description": step.description,
                "dependencies": list(step.dependencies),
                "tool_names": list(step.tool_names),
            },
        )
        try:
            step_tools = self._tools_for_step(tools, step.tool_names)
            result = await react_pattern.run(
                context=child_context,
                tools=step_tools,
                llm=llm,
                compact_llm=compact_llm,
                runtime=step_runtime,
                memory_store=memory_store,
                memory_similarity_threshold=memory_similarity_threshold,
                skill_manager=skill_manager,
                allowed_skills=allowed_skills,
            )
        except ExecutionInterrupted:
            raise
        except Exception as exc:
            step.status = "failed"
            step.error = str(exc)
            self._clear_active_step(step.id)
            await runtime.on_dag_step_end(
                context=root_context,
                step_id=step.id,
                data={
                    "step_id": step.id,
                    "step_name": step.task,
                    "description": step.description,
                    "dependencies": list(step.dependencies),
                    "tool_names": list(step.tool_names),
                    "status": "failed",
                    "error": step.error,
                },
            )
            return None

        status = result.get("status")
        if status == "interrupted":
            self.status = "interrupted"
            step.status = "interrupted"
            self._set_active_step_pattern_state(step.id, react_pattern.get_state())
            self._set_active_step_context(step.id, child_context.to_dict())
            await runtime.checkpoint(
                "dag_interrupted",
                context=root_context,
                pattern=self,
                metadata={"active_step_id": step.id},
            )
            if self._needs_replan(root_context):
                return None
            return {
                **result,
                "execution_id": root_context.execution_id,
                "context": root_context,
                "active_step_id": step.id,
            }
        if status == "waiting_for_user":
            self.status = "waiting_for_user"
            self._set_active_step_pattern_state(step.id, react_pattern.get_state())
            self._set_active_step_context(step.id, child_context.to_dict())
            await runtime.checkpoint(
                "dag_waiting_for_user",
                context=root_context,
                pattern=self,
                metadata={"active_step_id": step.id},
            )
            return {
                **result,
                "execution_id": root_context.execution_id,
                "context": root_context,
                "active_step_id": step.id,
            }
        if not result.get("success"):
            step.status = "failed"
            step.error = result.get("error", f"Step {step.id} failed.")
            await runtime.on_dag_step_end(
                context=root_context,
                step_id=step.id,
                data={
                    "step_id": step.id,
                    "step_name": step.task,
                    "description": step.description,
                    "dependencies": list(step.dependencies),
                    "tool_names": list(step.tool_names),
                    "status": "failed",
                    "error": step.error,
                },
            )
            self._clear_active_step(step.id)
            return None

        step.status = "completed"
        step.result = result.get("output", result.get("response", result))
        self.step_results[step.id] = step.result
        self._clear_active_step(step.id)
        await runtime.on_dag_step_end(
            context=root_context,
            step_id=step.id,
            data={
                "step_id": step.id,
                "step_name": step.task,
                "description": step.description,
                "dependencies": list(step.dependencies),
                "tool_names": list(step.tool_names),
                "status": "completed",
                "result": step.result,
            },
        )
        await runtime.checkpoint(
            "dag_after_step",
            context=root_context,
            pattern=self,
            metadata={"completed_step_id": step.id},
        )
        return None

    def get_state(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "plan": self.plan.to_dict() if self.plan is not None else None,
            "active_step_id": self.active_step_id,
            "active_step_pattern_state": self.active_step_pattern_state,
            "active_step_context": self.active_step_context,
            "active_step_ids": list(self.active_step_ids),
            "active_step_pattern_states": dict(self.active_step_pattern_states),
            "active_step_contexts": dict(self.active_step_contexts),
            "step_results": dict(self.step_results),
            "planned_user_message_count": self.planned_user_message_count,
            "max_concurrency": self.max_concurrency,
            "completion_feedback": self.completion_feedback,
            "completion_replan_count": self.completion_replan_count,
            "max_completion_replans": self.max_completion_replans,
        }

    def get_execution_snapshot(self, root_context: Any) -> dict[str, Any]:
        root_execution_id = getattr(root_context, "execution_id", "unknown")
        root_frame_id = f"{root_execution_id}:dag"
        active_child_ids = [
            self._child_frame_id(root_execution_id, step_id)
            for step_id in self.active_step_ids
        ]
        active_child_id = active_child_ids[0] if active_child_ids else None
        root_frame = ExecutionFrame(
            frame_id=root_frame_id,
            root_execution_id=root_execution_id,
            pattern_type="dag",
            status=self._execution_status(self.status),
            context=root_context.to_dict()
            if callable(getattr(root_context, "to_dict", None))
            else {},
            pattern_state=self.get_state(),
            children=active_child_ids,
            active_child_id=active_child_id,
            metadata={
                "active_step_id": self.active_step_id,
                "active_step_ids": list(self.active_step_ids),
                "plan_step_count": len(self.plan.steps) if self.plan else 0,
            },
        )
        frames = {root_frame_id: root_frame}
        active_frame_ids = [root_frame_id]
        for step_id, child_frame_id in zip(self.active_step_ids, active_child_ids):
            child_frame = ExecutionFrame(
                frame_id=child_frame_id,
                parent_frame_id=root_frame_id,
                root_execution_id=root_execution_id,
                pattern_type="react",
                status=self._execution_status(self.status),
                context=dict(self.active_step_contexts.get(step_id, {})),
                pattern_state=dict(self.active_step_pattern_states.get(step_id, {})),
                metadata={"dag_step_id": step_id},
            )
            frames[child_frame_id] = child_frame
            active_frame_ids.append(child_frame_id)

        return ExecutionSnapshot(
            root_execution_id=root_execution_id,
            status=self._execution_status(self.status),
            frames=frames,
            active_frame_ids=active_frame_ids,
            control_state={
                "planned_user_message_count": self.planned_user_message_count,
                "max_concurrency": self.max_concurrency,
            },
        ).to_dict()

    def load_state(self, state: dict[str, Any]) -> None:
        self.status = str(state.get("status", "idle"))
        plan_payload = state.get("plan")
        self.plan = (
            ExecutionPlan.from_dict(plan_payload)
            if isinstance(plan_payload, dict)
            else None
        )
        self.active_step_id = state.get("active_step_id")
        self.active_step_pattern_state = state.get("active_step_pattern_state")
        self.active_step_context = state.get("active_step_context")
        self.active_step_ids = list(state.get("active_step_ids", []))
        self.active_step_pattern_states = dict(
            state.get("active_step_pattern_states", {})
        )
        self.active_step_contexts = dict(state.get("active_step_contexts", {}))
        if self.active_step_id and self.active_step_id not in self.active_step_ids:
            self.active_step_ids.append(self.active_step_id)
        if self.active_step_id and self.active_step_pattern_state:
            self.active_step_pattern_states.setdefault(
                self.active_step_id,
                self.active_step_pattern_state,
            )
        if self.active_step_id and self.active_step_context:
            self.active_step_contexts.setdefault(
                self.active_step_id,
                self.active_step_context,
            )
        self._sync_legacy_active_step()
        self.step_results = dict(state.get("step_results", {}))
        self.planned_user_message_count = int(
            state.get("planned_user_message_count", 0)
        )
        self.max_concurrency = max(1, int(state.get("max_concurrency", 4)))
        feedback = state.get("completion_feedback")
        self.completion_feedback = str(feedback) if feedback else None
        self.completion_replan_count = int(state.get("completion_replan_count", 0))
        self.max_completion_replans = max(
            0,
            int(state.get("max_completion_replans", self.max_completion_replans)),
        )

    async def _fail(
        self,
        *,
        context: Any,
        runtime: PatternRuntime,
        error: str,
        failure_reason: str,
        checkpoint_label: str,
        failed_step_id: str | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.status = "failed"
        metadata: dict[str, Any] = {
            "status": self.status,
            "failure_reason": failure_reason,
        }
        if extra_metadata:
            metadata.update(extra_metadata)
        if failed_step_id is not None:
            metadata["failed_step_id"] = failed_step_id
        await runtime.checkpoint(
            checkpoint_label,
            context=context,
            pattern=self,
            status=self.status,
            metadata=metadata,
        )
        return PatternResult(
            success=False,
            error=error,
            metadata=metadata,
        ).to_dict()

    def _ready_steps(self) -> list[PlanStep]:
        if self.plan is None:
            return []
        ready: list[PlanStep] = []
        active_step_ids = set(self.active_step_ids)
        for step in self.plan.steps:
            if step.status in {"completed", "failed"}:
                continue
            if active_step_ids:
                if step.id in active_step_ids:
                    ready.append(step)
                continue
            if all(dep in self.step_results for dep in step.dependencies):
                ready.append(step)
        return ready

    def _pending_ready_steps(self) -> list[PlanStep]:
        if self.plan is None:
            return []
        return [
            step
            for step in self.plan.steps
            if step.status == "pending"
            and all(dep in self.step_results for dep in step.dependencies)
        ]

    def _all_steps_completed(self) -> bool:
        return self.plan is not None and all(
            step.status == "completed" for step in self.plan.steps
        )

    async def _handle_completed_plan(
        self,
        *,
        context: Any,
        tools: list[Any],
        llm: Any,
        runtime: PatternRuntime,
    ) -> dict[str, Any] | None:
        try:
            assessment = await self._assess_completion(
                context=context,
                llm=llm,
                runtime=runtime,
            )
        except LLMCallInterrupted:
            interrupted = await self._interrupt_if_requested(
                runtime=runtime,
                context=context,
                label="dag_during_completion_assessment",
            )
            if interrupted is not None:
                return interrupted
            raise
        except Exception as exc:  # noqa: BLE001
            return await self._fail(
                context=context,
                runtime=runtime,
                error=str(exc),
                failure_reason="completion_assessment_error",
                checkpoint_label="dag_completion_assessment_failed",
            )

        if assessment.complete:
            self.status = "completed"
            self.completion_feedback = None
            output = assessment.answer or self._final_output()
            await runtime.checkpoint("dag_completed", context=context, pattern=self)
            return PatternResult(
                success=True,
                output=output,
                metadata={
                    "status": self.status,
                    "step_results": self.step_results,
                    "completion_assessment": assessment.__dict__,
                },
            ).to_dict()

        if self.completion_replan_count >= self.max_completion_replans:
            return await self._fail(
                context=context,
                runtime=runtime,
                error=(
                    "DAG completion assessment requested additional work after "
                    f"{self.max_completion_replans} replans."
                ),
                failure_reason="completion_replan_limit_exceeded",
                checkpoint_label="dag_completion_replan_limit",
            )

        self.completion_replan_count += 1
        self.completion_feedback = (
            assessment.replan_instruction
            or assessment.missing_work
            or assessment.reason
        )
        await runtime.checkpoint(
            "dag_completion_incomplete",
            context=context,
            pattern=self,
            metadata={
                "completion_assessment": assessment.__dict__,
                "completion_replan_count": self.completion_replan_count,
            },
        )
        try:
            await self._generate_plan(
                context=context,
                tools=tools,
                llm=llm,
                runtime=runtime,
                replan=True,
            )
        except PlanValidationError as exc:
            return await self._fail(
                context=context,
                runtime=runtime,
                error=str(exc),
                failure_reason="invalid_plan",
                checkpoint_label="dag_plan_invalid",
            )
        except LLMCallInterrupted:
            interrupted = await self._interrupt_if_requested(
                runtime=runtime,
                context=context,
                label="dag_during_completion_replan",
            )
            if interrupted is not None:
                return interrupted
            raise
        except RequiredToolCallError:
            raise
        except Exception as exc:  # noqa: BLE001
            return await self._fail(
                context=context,
                runtime=runtime,
                error=str(exc),
                failure_reason="completion_replan_generation_error",
                checkpoint_label="dag_plan_generation_failed",
            )
        return None

    async def _assess_completion(
        self,
        *,
        context: Any,
        llm: Any,
        runtime: PatternRuntime,
    ) -> DAGCompletionAssessment:
        messages = self._completion_assessment_messages(context)
        tools = [self._completion_assessment_tool_schema()]
        await runtime.on_dag_execution(
            context=context,
            phase="completion_assessment",
            data={
                "completed_step_count": len(self.step_results),
                "plan_step_count": len(self.plan.steps) if self.plan else 0,
            },
        )
        await runtime.on_llm_start(
            context=context,
            messages=messages,
            tools=tools,
            metadata={"phase": "dag_completion_assessment"},
        )
        final_answer_stream = FinalAnswerStreamSession(runtime)
        streamer = ToolCallStringFieldStreamer(
            runtime=runtime,
            tool_name=DAG_COMPLETION_TOOL_NAME,
            field_name="answer",
            guard_field="status",
            guard_value="completed",
            emitter=final_answer_stream,
        )
        try:
            response = await runtime.run_streaming_llm_call(
                llm,
                messages=messages,
                tools=tools,
                tool_choice="required",
                thinking={"type": "disabled", "enable": False},
                on_chunk=streamer.handle_chunk,
            )
        except Exception as exc:
            await streamer.fail(str(exc))
            await runtime.on_llm_error(
                context=context,
                error=exc,
                metadata={"phase": "dag_completion_assessment"},
            )
            raise
        await runtime.on_llm_end(
            context=context,
            response=response,
            metadata={"phase": "dag_completion_assessment"},
        )
        assessment = self._parse_completion_assessment(response)
        if assessment.complete:
            await final_answer_stream.finish(assessment.answer)
        return assessment

    def _completion_assessment_messages(self, context: Any) -> list[dict[str, Any]]:
        latest_messages = [
            {"role": message.role, "content": message.content}
            for message in getattr(context, "messages", [])
            if getattr(message, "role", None) in {"user", "assistant", "tool"}
        ]
        authoritative_user_requests = [
            {"role": message.role, "content": message.content}
            for message in getattr(context, "messages", [])
            if getattr(message, "role", None) == "user"
        ]
        payload = {
            "output_language_policy": output_language_policy(
                getattr(context, "metadata", {}).get(OUTPUT_LANGUAGE_METADATA_KEY)
                if isinstance(getattr(context, "metadata", {}), dict)
                else None
            ),
            "authoritative_user_requests": authoritative_user_requests,
            "messages": latest_messages,
            "plan": self.plan.to_dict() if self.plan is not None else None,
            "step_results": self.step_results,
            "candidate_output": self._final_output(),
            "previous_completion_feedback": self.completion_feedback,
        }
        return [
            {
                "role": "system",
                "content": (
                    "Assess whether the completed DAG steps satisfy the user's "
                    "overall request. The authoritative_user_requests field is "
                    "the only source of required scope. The plan, step results, "
                    "briefs, inferred formats, and candidate output are evidence "
                    "of execution only; they cannot add deliverables, claims, "
                    "formats, or acceptance criteria that the user did not ask "
                    "for. Do not mark the goal incomplete solely because an "
                    "intermediate step proposed extra work. Call the assessment "
                    "tool exactly once. If "
                    "the goal is satisfied, choose status=completed and put the "
                    "final user-facing answer in answer. If anything material is "
                    "missing, choose status=incomplete, leave answer empty, and "
                    "state the missing work plus concise replan instructions. Put "
                    "status before answer in the tool arguments. "
                    f"{final_answer_language_rule(subject='output language policy')}"
                ),
            },
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]

    def _completion_assessment_tool_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": DAG_COMPLETION_TOOL_NAME,
                "description": (
                    "Decide whether DAG execution satisfies the overall user goal "
                    "and either provide the final answer or request more work."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "status": {
                            "type": "string",
                            "enum": ["completed", "incomplete"],
                        },
                        "reason": {"type": "string"},
                        "answer": {
                            "type": "string",
                            "description": (
                                "Final user-facing answer when status is completed; "
                                "empty when status is incomplete. "
                                f"{final_answer_language_rule(subject='output language policy')}"
                            ),
                        },
                        "missing_work": {
                            "type": "string",
                            "description": (
                                "What remains unresolved when status is incomplete."
                            ),
                        },
                        "replan_instruction": {
                            "type": "string",
                            "description": (
                                "Concise instruction for the next DAG replan when "
                                "status is incomplete."
                            ),
                        },
                    },
                    "required": [
                        "status",
                        "reason",
                        "answer",
                        "missing_work",
                        "replan_instruction",
                    ],
                    "additionalProperties": False,
                },
            },
        }

    def _parse_completion_assessment(self, response: Any) -> DAGCompletionAssessment:
        payload = self._extract_tool_arguments(response, DAG_COMPLETION_TOOL_NAME)
        status = str(payload.get("status", "")).strip().lower()
        if status not in {"completed", "incomplete"}:
            raise ValueError(f"Invalid DAG completion status: {status}")
        answer = str(payload.get("answer", ""))
        return DAGCompletionAssessment(
            complete=status == "completed",
            answer=answer,
            reason=str(payload.get("reason", "")),
            missing_work=str(payload.get("missing_work", "")),
            replan_instruction=str(payload.get("replan_instruction", "")),
        )

    def _extract_tool_arguments(self, response: Any, tool_name: str) -> dict[str, Any]:
        tool_calls = self._response_tool_calls(response)
        for tool_call in tool_calls:
            function_payload = self._function_payload(tool_call)
            if not function_payload or function_payload.get("name") != tool_name:
                continue
            return self._coerce_tool_arguments(function_payload.get("arguments", {}))
        raise ValueError(f"DAG completion requires a {tool_name} tool call response.")

    def _response_tool_calls(self, response: Any) -> list[Any]:
        if isinstance(response, dict):
            return list(response.get("tool_calls") or [])
        return list(getattr(response, "tool_calls", []) or [])

    def _function_payload(self, tool_call: Any) -> dict[str, Any] | None:
        if isinstance(tool_call, dict):
            function_payload = tool_call.get("function")
            return function_payload if isinstance(function_payload, dict) else None
        function_payload = getattr(tool_call, "function", None)
        if function_payload is None:
            return None
        return {
            "name": getattr(function_payload, "name", None),
            "arguments": getattr(function_payload, "arguments", {}),
        }

    def _coerce_tool_arguments(self, arguments: Any) -> dict[str, Any]:
        if isinstance(arguments, dict):
            return arguments
        if not isinstance(arguments, str):
            raise TypeError("Tool call arguments must be an object or JSON string.")
        payload = json.loads(arguments)
        if not isinstance(payload, dict):
            raise TypeError("Tool call arguments must decode to an object.")
        return payload

    def _final_output(self) -> Any:
        if self.plan is None:
            return self.step_results
        if len(self.plan.steps) == 1:
            return self.step_results.get(self.plan.steps[0].id)
        terminal_steps = self._terminal_steps()
        terminal_results = [
            self.step_results.get(step.id)
            for step in terminal_steps
            if step.id in self.step_results
        ]
        if len(terminal_results) == 1:
            return self._display_output(terminal_results[0])
        if terminal_results:
            return "\n\n".join(
                self._display_output(result) for result in terminal_results if result
            )
        return self.step_results

    def _display_output(self, result: Any) -> str:
        if isinstance(result, str):
            return unwrap_final_answer_content(result)
        return json.dumps(result, ensure_ascii=False, indent=2, default=str)

    def _terminal_steps(self) -> list[PlanStep]:
        if self.plan is None:
            return []
        dependency_ids = {
            dependency for step in self.plan.steps for dependency in step.dependencies
        }
        terminal_steps = [
            step
            for step in self.plan.steps
            if step.id not in dependency_ids and step.status == "completed"
        ]
        return terminal_steps or [
            step for step in self.plan.steps if step.status == "completed"
        ]

    def _dependency_summary(self, step: PlanStep) -> dict[str, Any]:
        return {dep: self.step_results.get(dep) for dep in step.dependencies}

    def _step_instruction(self, *, root_context: Any, step: PlanStep) -> str:
        language_policy = output_language_policy(self._output_language(root_context))
        dependency_note = (
            "Dependency results, if any, are provided immediately before this "
            "message. Use them as inputs for this step only."
            if step.dependencies
            else "This step has no dependencies."
        )
        suggested_tools = ", ".join(step.tool_names) if step.tool_names else "(none)"
        termination_condition = (
            step.termination_condition
            or "Return final_answer when the current step description is satisfied."
        )
        completion_evidence = step.completion_evidence or "(none)"
        return (
            "DAG STEP EXECUTION BOUNDARY\n"
            "The overall user goal is background context only. Do not execute it "
            "directly and do not use it to expand the current step's completion "
            "criteria.\n\n"
            "OUTPUT LANGUAGE POLICY\n"
            f"{language_policy}\n"
            "Use this policy only to preserve language for user-facing prose and "
            "persisted tool arguments. Do not use it to expand this step's "
            "completion criteria.\n\n"
            "CURRENT STEP - ONLY EXECUTABLE GOAL\n"
            f"Current DAG step id: {step.id}\n"
            f"Current DAG step title: {step.task}\n"
            f"Current DAG step description: {step.description or step.task}\n"
            f"Current DAG step dependencies: {list(step.dependencies)}\n"
            f"Suggested tools for this step: {suggested_tools}\n\n"
            "TERMINATION CONDITION - AUTHORITATIVE STOP RULE\n"
            f"{termination_condition}\n"
            f"Completion evidence: {completion_evidence}\n"
            "Treat this termination condition as authoritative for this step. "
            "Once it is satisfied, your next action must be final_answer for this "
            "step. Do not inspect, verify, revise, optimize, regenerate, or perform "
            "downstream work unless the termination condition explicitly requires "
            "that work.\n\n"
            f"{dependency_note}\n\n"
            "Execute only the current DAG step. The current step title and "
            "description plus the termination condition define the entire "
            "actionable goal for this ReAct run. "
            "Do not infer extra work from the overall user goal. Do not complete "
            "downstream, sibling, final synthesis, rendering, screenshots, visual "
            "inspection, export, or delivery work unless that work is explicitly "
            "part of this current step description. If the current step creates an "
            "artifact that a later step will render, inspect, export, or deliver, "
            "stop after creating that artifact and report it as this step's result. "
            "Treat the suggested tools as the primary tool scope for this step. "
            "Prefer those tools and avoid other tools unless this current step "
            "cannot be completed or recovered without them. If no suggested tools "
            "are listed, do not call tools unless the step clearly cannot be "
            "completed from the provided context and dependency results. When this "
            "step is done, return a final answer for this step only."
        )

    @staticmethod
    def _output_language(root_context: Any) -> str:
        if isinstance(root_context, dict):
            metadata = root_context.get("metadata", {})
        else:
            metadata = getattr(root_context, "metadata", {})
        if isinstance(metadata, dict):
            return str(metadata.get(OUTPUT_LANGUAGE_METADATA_KEY) or "").strip()
        return ""

    async def _generate_plan(
        self,
        *,
        context: Any,
        tools: list[Any],
        llm: Any,
        runtime: PatternRuntime,
        replan: bool,
    ) -> None:
        self.status = "replanning" if replan else "planning"
        if replan:
            self._clear_all_active_steps()
        await runtime.on_dag_execution(
            context=context,
            phase="replanning" if replan else "planning",
            data={
                "completed_step_count": len(self.step_results),
                "previous_step_count": len(self.plan.steps) if self.plan else 0,
            },
        )
        request = PlanGenerationRequest(
            context=context,
            execution_id=getattr(context, "execution_id", None),
            replan=replan,
            completed_step_results=dict(self.step_results),
            previous_plan=self.plan,
            available_tool_names=[self._tool_name(tool) for tool in tools],
            completion_feedback=self.completion_feedback,
        )
        self.plan = await self.plan_generator.generate_plan(
            request=request,
            llm=_RuntimeLLMProxy(runtime=runtime, llm=llm),
        )
        self.plan.validate()
        self._apply_completed_results_to_plan()
        self.planned_user_message_count = self._user_message_count(context)
        if replan:
            runtime.clear_interrupt()
        await runtime.checkpoint(
            "dag_replanned" if replan else "dag_plan_generated",
            context=context,
            pattern=self,
        )
        await runtime.on_dag_execution(
            context=context,
            phase="executing",
            data={
                "replan": replan,
                "plan_step_count": len(self.plan.steps) if self.plan else 0,
                "steps": [step.to_dict() for step in self.plan.steps]
                if self.plan
                else [],
            },
        )
        self.status = "running"

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
            "dag_interrupted",
            context=context,
            pattern=self,
            status=self.status,
            metadata={"safe_point": label, "reason": runtime.interrupt_reason},
        )
        return PatternResult(
            success=False,
            error="DAGPattern interrupted.",
            metadata={
                "status": self.status,
                "interrupt_reason": runtime.interrupt_reason,
            },
        ).to_dict()

    def _apply_completed_results_to_plan(self) -> None:
        if self.plan is None:
            return
        for step in self.plan.steps:
            if step.id in self.step_results:
                step.status = "completed"
                step.result = self.step_results[step.id]

    def _needs_replan(self, context: Any) -> bool:
        if self.status not in {"interrupted", "waiting_for_user", "replanning"}:
            return False
        return self._user_message_count(context) > self.planned_user_message_count

    def _forward_user_response_to_waiting_step(self, root_context: Any) -> bool:
        if self.status != "waiting_for_user":
            return False

        step_id = self._waiting_step_id()
        if step_id is None:
            return False

        root_user_messages = [
            message for message in root_context.messages if message.role == "user"
        ]
        if len(root_user_messages) <= self.planned_user_message_count:
            return False

        active_context = self.active_step_contexts.get(step_id)
        if not active_context:
            return False

        child_context = type(root_context).from_dict(active_context)
        for message in root_user_messages[self.planned_user_message_count :]:
            child_context.add_user_message(
                message.content,
                metadata={
                    **getattr(message, "metadata", {}),
                    "kind": "dag_waiting_user_response",
                    "forwarded_from_root": True,
                    "dag_step_id": step_id,
                },
            )

        self._set_active_step_context(step_id, child_context.to_dict())
        self.planned_user_message_count = len(root_user_messages)
        self.status = "running"
        return True

    def _waiting_step_id(self) -> str | None:
        for step_id in self.active_step_ids:
            state = self.active_step_pattern_states.get(step_id)
            if (
                isinstance(state, dict)
                and state.get("status") == "waiting_for_user"
                and state.get("waiting_for_user_request")
            ):
                return step_id
        return None

    def _user_message_count(self, context: Any) -> int:
        return sum(1 for message in context.messages if message.role == "user")

    def _tools_for_step(
        self,
        tools: list[Any],
        suggested_tool_names: list[str],
    ) -> list[Any]:
        suggested_order = [
            name.strip() for name in suggested_tool_names if name and name.strip()
        ]
        if not suggested_order:
            return tools

        rank = {name: index for index, name in enumerate(suggested_order)}
        ordered = sorted(
            enumerate(tools),
            key=lambda item: (
                rank.get(self._tool_name(item[1]), len(rank)),
                item[0],
            ),
        )
        return [tool for _, tool in ordered]

    def _tool_name(self, tool: Any) -> str:
        metadata = getattr(tool, "metadata", None)
        if metadata is not None and getattr(metadata, "name", None):
            return str(metadata.name)
        if getattr(tool, "name", None):
            return str(tool.name)
        return str(tool)

    async def _cancel_pending_steps(
        self,
        pending: set[asyncio.Task[Any]],
        *,
        step_ids_by_task: dict[asyncio.Task[Any], str] | None = None,
        steps_by_id: dict[str, PlanStep] | None = None,
    ) -> None:
        cancelled_step_ids: list[str] = []
        for task in pending:
            step_id = step_ids_by_task.get(task) if step_ids_by_task else None
            if step_id is not None:
                cancelled_step_ids.append(step_id)
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        for step_id in cancelled_step_ids:
            self._clear_active_step(step_id)
            step = steps_by_id.get(step_id) if steps_by_id else None
            if step is not None and step.status == "running":
                step.status = "pending"

    def _mark_step_active(self, step_id: str) -> None:
        if step_id not in self.active_step_ids:
            self.active_step_ids.append(step_id)
        self._sync_legacy_active_step()

    def _set_active_step_context(
        self,
        step_id: str,
        context: dict[str, Any],
    ) -> None:
        self.active_step_contexts[step_id] = context
        self._mark_step_active(step_id)
        self._sync_legacy_active_step()

    def _set_active_step_pattern_state(
        self,
        step_id: str,
        state: dict[str, Any],
    ) -> None:
        self.active_step_pattern_states[step_id] = state
        self._mark_step_active(step_id)
        self._sync_legacy_active_step()

    def _clear_active_step(self, step_id: str) -> None:
        self.active_step_ids = [
            active_step_id
            for active_step_id in self.active_step_ids
            if active_step_id != step_id
        ]
        self.active_step_contexts.pop(step_id, None)
        self.active_step_pattern_states.pop(step_id, None)
        self._sync_legacy_active_step()

    def _clear_all_active_steps(self) -> None:
        self.active_step_ids = []
        self.active_step_contexts = {}
        self.active_step_pattern_states = {}
        self._sync_legacy_active_step()

    def _sync_legacy_active_step(self) -> None:
        self.active_step_id = self.active_step_ids[0] if self.active_step_ids else None
        if self.active_step_id is None:
            self.active_step_context = None
            self.active_step_pattern_state = None
            return
        self.active_step_context = self.active_step_contexts.get(self.active_step_id)
        self.active_step_pattern_state = self.active_step_pattern_states.get(
            self.active_step_id
        )

    def _child_frame_id(self, root_execution_id: str, step_id: str | None) -> str:
        return f"{root_execution_id}:dag_step:{step_id or 'unknown'}"

    def _execution_status(self, status: str) -> ExecutionStatus:
        mapping = {
            "planning": ExecutionStatus.RUNNING,
            "running": ExecutionStatus.RUNNING,
            "replanning": ExecutionStatus.REPLANNING,
            "completed": ExecutionStatus.COMPLETED,
            "failed": ExecutionStatus.FAILED,
            "interrupted": ExecutionStatus.INTERRUPTED,
            "waiting_for_user": ExecutionStatus.WAITING_FOR_USER,
        }
        return mapping.get(status, ExecutionStatus.RUNNING)
