from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from ...config import get_tool_max_concurrency, get_tool_parallel_enabled
from ..context_ref import ContextReference
from ..task_runtime import (
    PREFERRED_INPUT_MODALITIES_METADATA_KEY,
    normalize_input_modalities,
)
from .agent import Agent
from .attachments import build_image_context_references
from .pattern import AutoPattern, DAGPattern, LLMPlanGenerator, ReActPattern
from .registry import ExecutionRegistry
from .result import NO_OUTPUT_PLACEHOLDER
from .runner import AgentRunner
from .tracing import TraceEventCallback

logger = logging.getLogger(__name__)

INTERRUPTED_USER_MESSAGE = (
    "The previous run was stopped. You can send another message to continue."
)


@dataclass
class AgentExecutionConfig:
    name: str
    pattern: str
    llm: Any | None
    compact_llm: Any | None = None
    tools: list[Any] = field(default_factory=list)
    tracer: Any | None = None
    system_prompt: str | None = None
    workspace_base_dir: str = "workspace"
    allowed_external_dirs: list[str] | None = None
    scope_segments: tuple[str, ...] = ()
    current_task_id: str | None = None
    service_id: str | None = None
    registry: ExecutionRegistry | None = None
    dag_max_concurrency: int = 4
    react_max_iterations: int = 200
    tool_parallel_enabled: bool = field(default_factory=get_tool_parallel_enabled)
    tool_max_concurrency: int = field(default_factory=get_tool_max_concurrency)
    outbound_message_handler: Any | None = None
    # Polled per step by the pattern loop; returns a reason to interrupt the run
    # (e.g. a mid-run quota gate). None => never interrupt from a checker.
    interrupt_checker: Any | None = None
    conversation_history: list[dict[str, Any]] = field(default_factory=list)
    execution_context_messages: list[dict[str, Any]] = field(default_factory=list)
    recovered_skill_context: str | None = None
    memory_store: Any | None = None
    memory_similarity_threshold: float | None = None
    skill_manager: Any | None = None
    skill_scope_context: Any | None = None
    allowed_skills: list[str] | None = None
    skills_enabled: bool = True
    user_interaction_enabled: bool = True
    preferred_input_modalities: tuple[str, ...] = ()
    execution_metadata: dict[str, Any] = field(default_factory=dict)


class AgentExecutionAdapter:
    """Adapter that routes AgentService executions into agent."""

    def __init__(self, config: AgentExecutionConfig) -> None:
        self.config = config
        self.registry = config.registry or ExecutionRegistry()

    async def execute(
        self,
        *,
        task: str,
        context: dict[str, Any] | None = None,
        task_id: str | None = None,
    ) -> dict[str, Any]:
        if self.config.llm is None:
            error_msg = (
                f"Agent '{self.config.name}' has no LLM configured for agent execution."
            )
            logger.error(error_msg)
            return {
                "status": "error",
                "output": error_msg,
                "success": False,
                "error": error_msg,
                "metadata": {
                    "agent_name": self.config.name,
                    "execution_type": "agent_error",
                },
            }

        execution_id = str(
            task_id or self.config.current_task_id or self.config.service_id or ""
        )
        runner, execution_type = self._build_runner()
        handle = self.registry.start(
            runner,
            execution_id=execution_id,
            task=task,
            metadata=self._execution_metadata(
                execution_type=execution_type,
                request_context=context,
                include_request_context=True,
            ),
            workspace_id=self._workspace_id(execution_id),
            allowed_external_dirs=self.config.allowed_external_dirs,
            initial_messages=self._initial_messages(),
            task_context_refs=self._request_context_refs(context),
            interrupt_checker=self.config.interrupt_checker,
        )
        if handle.task is None:
            raise RuntimeError("Execution registry did not create a task.")
        result = await handle.task
        return self._normalize_result(
            result=result,
            execution_type=execution_type,
            execution_id=execution_id,
        )

    def start(
        self,
        *,
        task: str,
        context: dict[str, Any] | None = None,
        task_id: str | None = None,
    ) -> dict[str, Any]:
        if self.config.llm is None:
            raise ValueError(
                f"Agent '{self.config.name}' has no LLM configured for agent execution."
            )
        execution_id = str(
            task_id or self.config.current_task_id or self.config.service_id or ""
        )
        runner, execution_type = self._build_runner()
        handle = self.registry.start(
            runner,
            execution_id=execution_id,
            task=task,
            metadata=self._execution_metadata(
                execution_type=execution_type,
                request_context=context,
                include_request_context=True,
            ),
            workspace_id=self._workspace_id(execution_id),
            allowed_external_dirs=self.config.allowed_external_dirs,
            initial_messages=self._initial_messages(),
            task_context_refs=self._request_context_refs(context),
            interrupt_checker=self.config.interrupt_checker,
        )
        return handle.to_dict()

    def pause(self, execution_id: str, reason: str | None = None) -> bool:
        return self.registry.pause(execution_id, reason=reason)

    async def resume(self, execution_id: str, **kwargs: Any) -> dict[str, Any] | None:
        kwargs.setdefault("workspace_id", self._workspace_id(execution_id))
        # Carry the mid-run quota checker into the resumed run too, so a
        # paused-and-resumed continuation is gated like a fresh run.
        kwargs.setdefault("interrupt_checker", self.config.interrupt_checker)
        resume_metadata = dict(kwargs.get("metadata") or {})
        preferred_modalities = normalize_input_modalities(
            self.config.preferred_input_modalities
        )
        resume_metadata[PREFERRED_INPUT_MODALITIES_METADATA_KEY] = list(
            preferred_modalities
        )
        kwargs["metadata"] = resume_metadata
        handle = self.registry.get(execution_id)
        if handle is None:
            runner, execution_type = self._build_runner()
            self.registry.register(
                execution_id,
                runner,
                metadata=self._execution_metadata(execution_type=execution_type),
            )
        else:
            execution_type = str(
                handle.metadata.get("execution_type") or self._execution_type()
            )

        result = await self.registry.resume(execution_id, **kwargs)
        if result is None:
            return None
        return self._normalize_result(
            result=result,
            execution_type=execution_type,
            execution_id=execution_id,
        )

    async def post_user_message(
        self,
        execution_id: str,
        message: str | None = None,
        *,
        execution_message: str | None = None,
        display_message: str | None = None,
        files: list[dict[str, Any]] | None = None,
        turn_id: str | None = None,
        request_interrupt: bool = True,
        reason: str | None = None,
    ) -> bool:
        if self.registry.get(execution_id) is None:
            runner, execution_type = self._build_runner()
            self.registry.register(
                execution_id,
                runner,
                metadata=self._execution_metadata(execution_type=execution_type),
            )
        context = await self.registry.post_user_message(
            execution_id,
            message,
            execution_message=execution_message,
            display_message=display_message,
            files=files,
            turn_id=turn_id,
            request_interrupt=request_interrupt,
            reason=reason,
        )
        return context is not None

    def cancel(self, execution_id: str, reason: str | None = None) -> bool:
        return self.registry.cancel(execution_id, reason=reason)

    def get_status(self, execution_id: str) -> dict[str, Any] | None:
        return self.registry.get_status(execution_id)

    def list_statuses(self) -> list[dict[str, Any]]:
        return self.registry.list_statuses()

    def _workspace_id(self, execution_id: str) -> str:
        return str(
            self.config.service_id or self.config.current_task_id or execution_id
        )

    def _execution_metadata(
        self,
        *,
        execution_type: str,
        request_context: dict[str, Any] | None = None,
        include_request_context: bool = False,
    ) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            **self.config.execution_metadata,
            "execution_type": execution_type,
            "pattern": self.config.pattern,
        }
        if include_request_context:
            metadata["request_context"] = dict(request_context or {})
            metadata["selected_skill_context"] = self.config.recovered_skill_context
        preferred_modalities = normalize_input_modalities(
            self.config.preferred_input_modalities
        )
        if preferred_modalities:
            metadata[PREFERRED_INPUT_MODALITIES_METADATA_KEY] = list(
                preferred_modalities
            )
        return metadata

    def _build_runner(self) -> tuple[AgentRunner, str]:
        pattern, execution_type = self._build_pattern()
        skill_manager = (
            self.config.skill_manager if self.config.skills_enabled else None
        )
        if self.config.skills_enabled and skill_manager is None:
            from ...skills.utils import create_skill_manager

            skill_manager = create_skill_manager(
                context=self.config.skill_scope_context
            )
        agent = Agent(
            name=self.config.name,
            patterns=[pattern],
            tools=self.config.tools,
            llm=self.config.llm,
            compact_llm=self.config.compact_llm,
            system_prompt=self.config.system_prompt,
            metadata={
                **self.config.execution_metadata,
                "pattern": self.config.pattern,
            },
            memory_store=self.config.memory_store,
            memory_similarity_threshold=self.config.memory_similarity_threshold,
            skill_manager=skill_manager,
            allowed_skills=self.config.allowed_skills,
        )
        return (
            AgentRunner(
                agent=agent,
                tracer=self.config.tracer,
                callbacks=[TraceEventCallback()],
                workspace_base_dir=self.config.workspace_base_dir,
                scope_segments=self.config.scope_segments,
                outbound_message_handler=self.config.outbound_message_handler,
            ),
            execution_type,
        )

    def _build_pattern(self) -> tuple[Any, str]:
        if self.config.pattern == "dag_plan_execute":
            return (
                DAGPattern(
                    LLMPlanGenerator(),
                    max_concurrency=self.config.dag_max_concurrency,
                    user_interaction_enabled=self.config.user_interaction_enabled,
                ),
                "agent_dag",
            )
        if self.config.pattern == "auto":
            return (
                AutoPattern(
                    react_pattern=ReActPattern(
                        max_iterations=self.config.react_max_iterations,
                        tool_parallel_enabled=self.config.tool_parallel_enabled,
                        tool_max_concurrency=self.config.tool_max_concurrency,
                        user_interaction_enabled=(self.config.user_interaction_enabled),
                    ),
                    dag_pattern=DAGPattern(
                        LLMPlanGenerator(),
                        max_concurrency=self.config.dag_max_concurrency,
                        user_interaction_enabled=(self.config.user_interaction_enabled),
                    ),
                ),
                "agent_auto",
            )
        if self.config.pattern == "single_call":
            return (
                ReActPattern(
                    max_iterations=2,
                    finalize_after_tool_result=True,
                    tool_parallel_enabled=self.config.tool_parallel_enabled,
                    tool_max_concurrency=self.config.tool_max_concurrency,
                    user_interaction_enabled=self.config.user_interaction_enabled,
                ),
                "agent_single_call",
            )
        return (
            ReActPattern(
                max_iterations=self.config.react_max_iterations,
                tool_parallel_enabled=self.config.tool_parallel_enabled,
                tool_max_concurrency=self.config.tool_max_concurrency,
                user_interaction_enabled=self.config.user_interaction_enabled,
            ),
            "agent_react",
        )

    def _initial_messages(self) -> list[dict[str, Any]]:
        return [
            *self.config.execution_context_messages,
            *self.config.conversation_history,
        ]

    @staticmethod
    def _request_context_refs(
        context: dict[str, Any] | None,
    ) -> tuple[ContextReference, ...]:
        if not isinstance(context, dict):
            return ()
        references: list[ContextReference] = []
        seen: set[str] = set()
        for key in ("file_info", "files", "attachments"):
            for reference in build_image_context_references(context.get(key)):
                if reference.file_id in seen:
                    continue
                seen.add(reference.file_id)
                references.append(reference)
        return tuple(references)

    def _execution_type(self) -> str:
        if self.config.pattern == "dag_plan_execute":
            return "agent_dag"
        if self.config.pattern == "auto":
            return "agent_auto"
        if self.config.pattern == "single_call":
            return "agent_single_call"
        return "agent_react"

    def _normalize_result(
        self,
        *,
        result: dict[str, Any],
        execution_type: str,
        execution_id: str,
    ) -> dict[str, Any]:
        status = result.get(
            "status",
            "completed" if result.get("success") else "failed",
        )
        output: Any
        if status == "interrupted":
            # Keep the pattern-specific error in ``error``/``agent_result`` for
            # diagnostics, but never expose implementation names such as
            # ``ReActPattern`` as user-facing output.
            output = INTERRUPTED_USER_MESSAGE
        else:
            output = result.get("output", result.get("response", result.get("error")))
            if not output:
                # This backfill and the raw ``agent_result`` preserved below
                # must stay distinguishable: the delegated-child classifier
                # in ``agent_tool.py`` reads the pre-backfill answer from
                # ``agent_result`` specifically so a backfilled preamble
                # cannot be mistaken for a real final answer.
                output = self._latest_assistant_message(result.get("context"))
        if not output and result.get("success"):
            # A run that reports success with nothing to show is a bug in the
            # pattern that produced it, and the placeholder erases the evidence.
            # Log before substituting so there is something to debug from.
            #
            # Gated on ``success`` because an empty output is legitimate on every
            # non-success path: a ``waiting_for_user`` pause carries only its
            # message, so warning on emptiness alone would fire on the ordinary
            # first-turn clarification and drown the real signal.
            logger.warning(
                "Agent %r produced no output; substituting the placeholder. "
                "execution_type=%s pattern=%s status=%s success=%s task_id=%s",
                self.config.name,
                execution_type,
                self.config.pattern,
                status,
                result.get("success"),
                execution_id,
            )
        normalized = {
            "status": status,
            "output": output or NO_OUTPUT_PLACEHOLDER,
            "success": result.get("success", False),
            "error": result.get("error"),
            "metadata": {
                **self.config.execution_metadata,
                "agent_name": self.config.name,
                "execution_type": execution_type,
                "pattern": self.config.pattern,
                "task_id": execution_id,
            },
            "agent_result": result,
        }
        completion_outcome = result.get("completion_outcome")
        if completion_outcome in {"completed", "partial", "blocked"}:
            normalized["completion_outcome"] = completion_outcome
            normalized["metadata"]["completion_outcome"] = completion_outcome
        if status == "waiting_for_user":
            message = str(result.get("message") or output or "")
            interactions = result.get("interactions")
            normalized.update(
                {
                    "message": message,
                    "message_type": result.get("message_type", "question"),
                    "interactions": interactions,
                    "chat_response": {
                        "message": message,
                        "interactions": interactions
                        if isinstance(interactions, list)
                        else [],
                    },
                    # This top-level key is the supported contract for
                    # readers of the clarification draft. ``agent_result``
                    # above is a diagnostic snapshot of the raw pattern
                    # result, already read by ``agent_tool.py`` and
                    # ``websocket.py`` for other purposes -- it happens to
                    # carry the same draft too, but callers should not dig
                    # it out from there.
                    "clarification_draft": result.get("clarification_draft"),
                    # Empty list rather than ``None`` so a reader only ever
                    # needs one check (``if superseded:``) instead of also
                    # distinguishing "key absent" from "key present but
                    # empty".
                    "clarification_superseded_step_ids": (
                        result.get("clarification_superseded_step_ids") or []
                    ),
                }
            )
        if status == "interrupted":
            # A losing waiting step can still be superseded in a batch whose
            # winner is an interrupt rather than a question (the DAG ranks
            # an interrupt ahead of a waiting result within the same
            # wakeup), so this key must reach the top level here too --
            # otherwise a reader has no way to tell "no sibling was
            # superseded" apart from "this status never carries the key".
            # Same empty-list default as the waiting branch above, for the
            # same reason: one ``if superseded:`` check covers both.
            normalized["clarification_superseded_step_ids"] = (
                result.get("clarification_superseded_step_ids") or []
            )
        return normalized

    def _latest_assistant_message(self, context: Any) -> str | None:
        messages = getattr(context, "messages", None)
        if not isinstance(messages, list):
            return None
        for message in reversed(messages):
            if getattr(message, "role", None) != "assistant":
                continue
            content = getattr(message, "content", None)
            if isinstance(content, str) and content:
                return content
        return None
