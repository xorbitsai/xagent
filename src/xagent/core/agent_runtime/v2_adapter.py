from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from ..agent_v2 import Agent as V2Agent
from ..agent_v2 import AgentRunner as V2AgentRunner
from ..agent_v2 import AutoPattern as V2AutoPattern
from ..agent_v2 import DAGPattern as V2DAGPattern
from ..agent_v2 import (
    ExecutionRegistry,
    LLMPlanGenerator,
)
from ..agent_v2 import ReActPattern as V2ReActPattern
from ..agent_v2 import (
    TraceEventCallback,
)

logger = logging.getLogger(__name__)


@dataclass
class AgentV2ExecutionConfig:
    name: str
    pattern: str
    llm: Any | None
    tools: list[Any] = field(default_factory=list)
    tracer: Any | None = None
    system_prompt: str | None = None
    workspace_base_dir: str = "workspace"
    allowed_external_dirs: list[str] | None = None
    current_task_id: str | None = None
    service_id: str | None = None
    registry: ExecutionRegistry | None = None
    dag_max_concurrency: int = 4
    outbound_message_handler: Any | None = None
    conversation_history: list[dict[str, Any]] = field(default_factory=list)
    execution_context_messages: list[dict[str, Any]] = field(default_factory=list)
    recovered_skill_context: str | None = None
    memory_store: Any | None = None
    memory_similarity_threshold: float | None = None
    skill_manager: Any | None = None
    allowed_skills: list[str] | None = None


class AgentV2ExecutionAdapter:
    """Adapter that routes legacy AgentService executions into agent_v2."""

    def __init__(self, config: AgentV2ExecutionConfig) -> None:
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
                f"Agent '{self.config.name}' has no LLM configured for agent_v2 "
                "execution."
            )
            logger.error(error_msg)
            return {
                "status": "error",
                "output": error_msg,
                "success": False,
                "error": error_msg,
                "metadata": {
                    "agent_name": self.config.name,
                    "agent_runtime": "v2",
                    "execution_type": "agent_v2_error",
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
                "agent_runtime": "v2",
                "execution_type": execution_type,
                "legacy_pattern": self.config.pattern,
                "request_context": dict(context or {}),
                "selected_skill_context": self.config.recovered_skill_context,
            },
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
                f"Agent '{self.config.name}' has no LLM configured for agent_v2 "
                "execution."
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
                "agent_runtime": "v2",
                "execution_type": execution_type,
                "legacy_pattern": self.config.pattern,
                "request_context": dict(context or {}),
                "selected_skill_context": self.config.recovered_skill_context,
            },
            allowed_external_dirs=self.config.allowed_external_dirs,
            initial_messages=self._initial_messages(),
        )
        return handle.to_dict()

    def pause(self, execution_id: str, reason: str | None = None) -> bool:
        return self.registry.pause(execution_id, reason=reason)

    async def resume(self, execution_id: str, **kwargs: Any) -> dict[str, Any] | None:
        handle = self.registry.get(execution_id)
        if handle is None:
            runner, execution_type = self._build_runner()
            self.registry.register(
                execution_id,
                runner,
                metadata={
                    "agent_runtime": "v2",
                    "execution_type": execution_type,
                    "legacy_pattern": self.config.pattern,
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
        message: str,
        *,
        request_interrupt: bool = True,
        reason: str | None = None,
    ) -> bool:
        if self.registry.get(execution_id) is None:
            runner, execution_type = self._build_runner()
            self.registry.register(
                execution_id,
                runner,
                metadata={
                    "agent_runtime": "v2",
                    "execution_type": execution_type,
                    "legacy_pattern": self.config.pattern,
                },
            )
        context = await self.registry.post_user_message(
            execution_id,
            message,
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

    def _build_runner(self) -> tuple[V2AgentRunner, str]:
        v2_pattern, execution_type = self._build_pattern()
        skill_manager = self.config.skill_manager
        if skill_manager is None:
            from ...skills.utils import create_skill_manager

            skill_manager = create_skill_manager()
        v2_agent = V2Agent(
            name=self.config.name,
            patterns=[v2_pattern],
            tools=self.config.tools,
            llm=self.config.llm,
            system_prompt=self.config.system_prompt,
            metadata={"legacy_pattern": self.config.pattern},
            memory_store=self.config.memory_store,
            memory_similarity_threshold=self.config.memory_similarity_threshold,
            skill_manager=skill_manager,
            allowed_skills=self.config.allowed_skills,
        )
        return (
            V2AgentRunner(
                agent=v2_agent,
                tracer=self.config.tracer,
                callbacks=[TraceEventCallback()],
                workspace_base_dir=self.config.workspace_base_dir,
                outbound_message_handler=self.config.outbound_message_handler,
            ),
            execution_type,
        )

    def _build_pattern(self) -> tuple[Any, str]:
        if self.config.pattern == "dag_plan_execute":
            return (
                V2DAGPattern(
                    LLMPlanGenerator(),
                    max_concurrency=self.config.dag_max_concurrency,
                ),
                "agent_v2_dag",
            )
        if self.config.pattern == "auto":
            return (
                V2AutoPattern(
                    dag_pattern=V2DAGPattern(
                        LLMPlanGenerator(),
                        max_concurrency=self.config.dag_max_concurrency,
                    )
                ),
                "agent_v2_auto",
            )
        if self.config.pattern == "single_call":
            return (
                V2ReActPattern(max_iterations=1, tool_choice="none"),
                "agent_v2_single_call",
            )
        return V2ReActPattern(), "agent_v2_react"

    def _initial_messages(self) -> list[dict[str, Any]]:
        return [
            *self.config.execution_context_messages,
            *self.config.conversation_history,
        ]

    def _execution_type(self) -> str:
        if self.config.pattern == "dag_plan_execute":
            return "agent_v2_dag"
        if self.config.pattern == "auto":
            return "agent_v2_auto"
        if self.config.pattern == "single_call":
            return "agent_v2_single_call"
        return "agent_v2_react"

    def _normalize_result(
        self,
        *,
        result: dict[str, Any],
        execution_type: str,
        execution_id: str,
    ) -> dict[str, Any]:
        output = result.get("output", result.get("response", result.get("error")))
        status = result.get(
            "status",
            "completed" if result.get("success") else "failed",
        )
        return {
            "status": status,
            "output": output or "No output provided",
            "success": result.get("success", False),
            "error": result.get("error"),
            "metadata": {
                "agent_name": self.config.name,
                "agent_runtime": "v2",
                "execution_type": execution_type,
                "legacy_pattern": self.config.pattern,
                "task_id": execution_id,
            },
            "agent_v2_result": result,
        }
