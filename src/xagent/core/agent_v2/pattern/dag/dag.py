from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any

from ...context.enrichment import (
    enrich_context_with_memory,
    enrich_context_with_skill,
    latest_user_text,
)
from ...frame import ExecutionFrame, ExecutionSnapshot, ExecutionStatus
from ...runtime import LLMCallInterrupted, PatternRuntime
from ..base import AgentPattern, PatternResult
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

    async def send_message(
        self,
        *,
        message: str,
        message_type: str = "info",
        expect_response: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return await self.parent.send_message(
            message=message,
            message_type=message_type,
            expect_response=expect_response,
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
    ) -> None:
        self.plan_generator = (
            plan_generator
            if isinstance(plan_generator, PlanGenerator)
            else CallablePlanGenerator(plan_generator)
        )
        self.react_max_iterations = react_max_iterations
        self.react_reasoning_mode = ReActReasoningMode(react_reasoning_mode)
        self.max_concurrency = max(1, max_concurrency)
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
                runtime=runtime,
                memory_store=kwargs.get("memory_store"),
                memory_similarity_threshold=kwargs.get("memory_similarity_threshold"),
                skill_manager=kwargs.get("skill_manager"),
                allowed_skills=kwargs.get("allowed_skills"),
            )
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
                try:
                    await enrich_context_with_skill(
                        context=context,
                        task=task_text,
                        llm=llm,
                        skill_manager=skill_manager,
                        runtime=runtime,
                        allowed_skills=allowed_skills,
                    )
                except LLMCallInterrupted:
                    interrupted = await self._interrupt_if_requested(
                        runtime=runtime,
                        context=context,
                        label="dag_during_enrichment",
                    )
                    if interrupted is not None:
                        return interrupted
                    raise
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
                    self.status = "completed"
                    await runtime.checkpoint(
                        "dag_completed", context=context, pattern=self
                    )
                    output = self._final_output()
                    return PatternResult(
                        success=True,
                        output=output,
                        metadata={
                            "status": self.status,
                            "step_results": self.step_results,
                        },
                    ).to_dict()
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
        memory_store: Any | None = None,
        memory_similarity_threshold: float | None = None,
        skill_manager: Any | None = None,
        allowed_skills: list[str] | None = None,
    ) -> dict[str, Any] | None:
        if not steps:
            return None
        if len(steps) == 1:
            return await self._execute_step(
                step=steps[0],
                root_context=root_context,
                tools=tools,
                llm=llm,
                runtime=runtime,
                memory_store=memory_store,
                memory_similarity_threshold=memory_similarity_threshold,
                skill_manager=skill_manager,
                allowed_skills=allowed_skills,
            )

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
                    runtime=runtime,
                    memory_store=memory_store,
                    memory_similarity_threshold=memory_similarity_threshold,
                    skill_manager=skill_manager,
                    allowed_skills=allowed_skills,
                )
            ): step.id
            for step in steps
        }
        steps_by_id = {step.id: step for step in steps}
        pending = set(tasks)
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
                        (step for step in steps if step.status == "failed"),
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
        self,
        *,
        step: PlanStep,
        root_context: Any,
        tools: list[Any],
        llm: Any,
        runtime: PatternRuntime,
        memory_store: Any | None = None,
        memory_similarity_threshold: float | None = None,
        skill_manager: Any | None = None,
        allowed_skills: list[str] | None = None,
    ) -> dict[str, Any] | None:
        self.status = "running"
        self._mark_step_active(step.id)
        step.status = "running"

        active_context = self.active_step_contexts.get(step.id)
        if active_context is not None:
            child_context = type(root_context).from_dict(active_context)
        else:
            child_context = root_context.create_child_context(
                metadata={
                    "dag_step_id": step.id,
                    "dag_step_name": step.task,
                    "dag_step_description": step.description or step.task,
                    "dag_dependencies": list(step.dependencies),
                    "dag_tool_names": list(step.tool_names),
                    "dag_original_goal": latest_user_text(root_context).strip(),
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
                },
            )

        react_pattern = ReActPattern(
            max_iterations=self.react_max_iterations,
            reasoning_mode=self.react_reasoning_mode,
        )
        active_pattern_state = self.active_step_pattern_states.get(step.id)
        if active_pattern_state is not None:
            react_pattern.load_state(active_pattern_state)

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
            result = await react_pattern.run(
                context=child_context,
                tools=tools,
                llm=llm,
                runtime=step_runtime,
                memory_store=memory_store,
                memory_similarity_threshold=memory_similarity_threshold,
                skill_manager=skill_manager,
                allowed_skills=allowed_skills,
            )
        except LLMCallInterrupted:
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

    async def _fail(
        self,
        *,
        context: Any,
        runtime: PatternRuntime,
        error: str,
        failure_reason: str,
        checkpoint_label: str,
        failed_step_id: str | None = None,
    ) -> dict[str, Any]:
        self.status = "failed"
        metadata: dict[str, Any] = {
            "status": self.status,
            "failure_reason": failure_reason,
        }
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

    def _all_steps_completed(self) -> bool:
        return self.plan is not None and all(
            step.status == "completed" for step in self.plan.steps
        )

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
            return result
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
        original_goal = latest_user_text(root_context).strip()
        dependency_note = (
            "Dependency results, if any, are provided immediately before this "
            "message. Use them as inputs for this step only."
            if step.dependencies
            else "This step has no dependencies."
        )
        suggested_tools = ", ".join(step.tool_names) if step.tool_names else "(none)"
        return (
            "DAG STEP EXECUTION BOUNDARY\n"
            f"Overall user goal: {original_goal or '(not provided)'}\n\n"
            f"Current DAG step id: {step.id}\n"
            f"Current DAG step title: {step.task}\n"
            f"Current DAG step description: {step.description or step.task}\n"
            f"Current DAG step dependencies: {list(step.dependencies)}\n"
            f"Suggested tools for this step: {suggested_tools}\n\n"
            f"{dependency_note}\n\n"
            "Execute only the current DAG step. Do not complete downstream, "
            "sibling, final synthesis, rendering, export, or delivery work unless "
            "that work is explicitly part of this current step description. If the "
            "overall user goal asks for more work than this step describes, leave "
            "that work for later DAG steps. Treat the suggested tools as the primary "
            "tool scope for this step. Prefer those tools and avoid other tools unless "
            "this current step cannot be completed or recovered without them. If no "
            "suggested tools are listed, do not call tools unless the step clearly "
            "cannot be completed from the provided context and dependency results. "
            "When this step is done, return a final answer for this step only."
        )

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
