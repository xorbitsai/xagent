"""Restricted execution path for code-defined built-in agents."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any, TypeAlias, cast

from ...model.chat.basic.base import BaseLLM
from ...tools.adapters.vibe import Tool
from ...tools.adapters.vibe.base import ToolCategory
from ..service import AgentService
from .registry import BUILTIN_AGENT_REGISTRY, BuiltinAgentRegistry
from .spec import BuiltinAgentRunContext, BuiltinAgentSpec

BuiltinModelResolverResult: TypeAlias = BaseLLM | Awaitable[BaseLLM | None] | None
BuiltinModelResolver: TypeAlias = Callable[
    [str, BuiltinAgentRunContext], BuiltinModelResolverResult
]
BuiltinServiceFactory: TypeAlias = Callable[..., AgentService]


class BuiltinAgentModelUnavailableError(RuntimeError):
    """Raised when no model is available for a built-in agent's model role."""


class BuiltinAgentCapabilityError(ValueError):
    """Raised when a built-in agent requests a forbidden capability."""


class BuiltinAgentExecutor:
    """Resolve and execute a registered built-in agent with least privilege."""

    def __init__(
        self,
        *,
        model_resolver: BuiltinModelResolver,
        registry: BuiltinAgentRegistry = BUILTIN_AGENT_REGISTRY,
        service_factory: BuiltinServiceFactory = AgentService,
    ) -> None:
        self._model_resolver = model_resolver
        self._registry = registry
        self._service_factory = service_factory

    async def execute(
        self,
        name: str,
        *,
        task: str,
        execution_id: str,
        request_context: Mapping[str, Any] | None = None,
        tracer: Any | None = None,
        workspace_base_dir: str | None = None,
    ) -> dict[str, Any]:
        """Execute one built-in agent run without persistence or default tools."""

        if not task.strip():
            raise ValueError("Built-in agent task must not be empty")

        spec = self._registry.require(name)
        run_context = BuiltinAgentRunContext(
            execution_id=execution_id,
            request_context=dict(request_context or {}),
            tracer=tracer,
            workspace_base_dir=workspace_base_dir,
        )
        model = await self._resolve_model(spec, run_context)
        tools = await self._build_tools(spec, run_context)
        self._validate_tools(spec, tools)

        metadata = {
            "agent_type": "builtin",
            "builtin_agent_name": spec.name,
            "builtin_agent_version": spec.version,
            "builtin_model_role": spec.model_role,
        }
        service_kwargs: dict[str, Any] = {
            "name": f"builtin:{spec.name}",
            "id": f"builtin:{spec.name}:{execution_id}",
            "task_id": execution_id,
            "pattern": spec.pattern,
            "llm": model,
            "tracer": tracer,
            "system_prompt": spec.system_prompt,
            "tools": tools,
            "tools_initialized": True,
            "tool_config": None,
            "enable_default_tools": False,
            "enable_workspace": spec.workspace_enabled,
            "memory_enabled": spec.memory_enabled,
            "skills_enabled": spec.skills_enabled,
            "user_interaction_enabled": False,
            "execution_metadata": metadata,
        }
        if workspace_base_dir is not None:
            service_kwargs["workspace_base_dir"] = workspace_base_dir

        service = self._service_factory(**service_kwargs)
        execution_context = dict(run_context.request_context)
        execution_context["builtin_agent"] = dict(metadata)
        result = await service.execute_task(
            task,
            context=execution_context,
            task_id=execution_id,
        )
        result_metadata = result.get("metadata")
        if not isinstance(result_metadata, dict):
            result_metadata = {}
            result["metadata"] = result_metadata
        result_metadata.update(metadata)
        result["builtin_artifacts"] = dict(run_context.artifacts)
        return result

    async def _resolve_model(
        self,
        spec: BuiltinAgentSpec,
        run_context: BuiltinAgentRunContext,
    ) -> BaseLLM:
        resolved = self._model_resolver(spec.model_role, run_context)
        if inspect.isawaitable(resolved):
            resolved = await resolved
        if resolved is None:
            raise BuiltinAgentModelUnavailableError(
                f"No model is available for built-in agent '{spec.name}' "
                f"with role '{spec.model_role}'"
            )
        return cast(BaseLLM, resolved)

    async def _build_tools(
        self,
        spec: BuiltinAgentSpec,
        run_context: BuiltinAgentRunContext,
    ) -> list[Tool]:
        if spec.build_tools is None:
            return []
        built = spec.build_tools(run_context)
        if inspect.isawaitable(built):
            built = await built
        if isinstance(built, (str, bytes)) or not isinstance(built, Sequence):
            raise TypeError(
                f"Built-in agent '{spec.name}' tool builder must return a sequence"
            )
        return list(built)

    def _validate_tools(self, spec: BuiltinAgentSpec, tools: list[Tool]) -> None:
        for tool in tools:
            metadata = getattr(tool, "metadata", None)
            category = getattr(metadata, "category", None)
            category_value = getattr(category, "value", category)
            if category_value == ToolCategory.AGENT.value:
                raise BuiltinAgentCapabilityError(
                    f"Built-in agent '{spec.name}' cannot use agent delegation tools"
                )
