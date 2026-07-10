from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from ...config import get_tool_max_concurrency, get_tool_parallel_enabled
from .agent import Agent
from .pattern import AutoPattern, DAGPattern, LLMPlanGenerator, ReActPattern
from .registry import ExecutionRegistry
from .runner import AgentRunner
from .tracing import TraceEventCallback

logger = logging.getLogger(__name__)


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
    tool_parallel_enabled: bool = field(default_factory=get_tool_parallel_enabled)
    tool_max_concurrency: int = field(default_factory=get_tool_max_concurrency)
    outbound_message_handler: Any | None = None
    conversation_history: list[dict[str, Any]] = field(default_factory=list)
    execution_context_messages: list[dict[str, Any]] = field(default_factory=list)
    recovered_skill_context: str | None = None
    memory_store: Any | None = None
    memory_similarity_threshold: float | None = None
    skill_manager: Any | None = None
    skill_scope_context: Any | None = None
    allowed_skills: list[str] | None = None


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
            metadata={
                "execution_type": execution_type,
                "pattern": self.config.pattern,
                "request_context": dict(context or {}),
                "selected_skill_context": self.config.recovered_skill_context,
            },
            workspace_id=self._workspace_id(execution_id),
            allowed_external_dirs=self.config.allowed_external_dirs,
            initial_messages=self._initial_messages(),
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
            metadata={
                "execution_type": execution_type,
                "pattern": self.config.pattern,
                "request_context": dict(context or {}),
                "selected_skill_context": self.config.recovered_skill_context,
            },
            workspace_id=self._workspace_id(execution_id),
            allowed_external_dirs=self.config.allowed_external_dirs,
            initial_messages=self._initial_messages(),
        )
        return handle.to_dict()

    def pause(self, execution_id: str, reason: str | None = None) -> bool:
        return self.registry.pause(execution_id, reason=reason)

    async def resume(self, execution_id: str, **kwargs: Any) -> dict[str, Any] | None:
        kwargs.setdefault("workspace_id", self._workspace_id(execution_id))
        handle = self.registry.get(execution_id)
        if handle is None:
            runner, execution_type = self._build_runner()
            self.registry.register(
                execution_id,
                runner,
                metadata={
                    "execution_type": execution_type,
                    "pattern": self.config.pattern,
                },
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
                metadata={
                    "execution_type": execution_type,
                    "pattern": self.config.pattern,
                },
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

    def _build_runner(self) -> tuple[AgentRunner, str]:
        pattern, execution_type = self._build_pattern()
        skill_manager = self.config.skill_manager
        if skill_manager is None:
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
            metadata={"pattern": self.config.pattern},
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
                ),
                "agent_dag",
            )
        if self.config.pattern == "auto":
            return (
                AutoPattern(
                    react_pattern=ReActPattern(
                        tool_parallel_enabled=self.config.tool_parallel_enabled,
                        tool_max_concurrency=self.config.tool_max_concurrency,
                    ),
                    dag_pattern=DAGPattern(
                        LLMPlanGenerator(),
                        max_concurrency=self.config.dag_max_concurrency,
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
                ),
                "agent_single_call",
            )
        return (
            ReActPattern(
                tool_parallel_enabled=self.config.tool_parallel_enabled,
                tool_max_concurrency=self.config.tool_max_concurrency,
            ),
            "agent_react",
        )

    def _initial_messages(self) -> list[dict[str, Any]]:
        return [
            *self.config.execution_context_messages,
            *self.config.conversation_history,
        ]

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
        output = result.get("output", result.get("response", result.get("error")))
        if not output:
            output = self._latest_assistant_message(result.get("context"))
        status = result.get(
            "status",
            "completed" if result.get("success") else "failed",
        )
        normalized = {
            "status": status,
            "output": output or "No output provided",
            "success": result.get("success", False),
            "error": result.get("error"),
            "metadata": {
                "agent_name": self.config.name,
                "execution_type": execution_type,
                "pattern": self.config.pattern,
                "task_id": execution_id,
            },
            "agent_result": result,
        }
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
                }
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
