"""Agent service facade for agent execution."""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, cast

from ...config import get_uploads_dir
from ..memory import MemoryStore
from ..memory.in_memory import InMemoryMemoryStore
from ..model.chat.basic.base import BaseLLM
from ..tools.adapters.vibe import Tool
from ..tools.adapters.vibe.config import (
    RequiredMCPUnavailableError,
    normalize_tool_allowlist,
)
from ..tools.adapters.vibe.connector_runtime import ConnectorRuntimeError
from ..workspace import TaskWorkspace, create_workspace
from .trace import Tracer
from .transcript import normalize_transcript_messages

logger = logging.getLogger(__name__)

_UNSET = object()


class AgentService:
    """Service facade that executes tasks through agent only."""

    def __init__(
        self,
        name: str,
        patterns: list[Any] | None = None,
        memory: MemoryStore | None = None,
        tools: list[Tool] | None = None,
        llm: BaseLLM | None = None,
        use_dag_pattern: bool | object = _UNSET,
        pattern: str | Any = "dag_plan_execute",
        tracer: Tracer | None = None,
        id: str | None = None,
        workspace: TaskWorkspace | None = None,
        workspace_base_dir: str | None = None,
        enable_workspace: bool = True,
        allowed_external_dirs: list[str] | None = None,
        scope_segments: tuple[str, ...] | None = None,
        task_id: str | None = None,
        fast_llm: BaseLLM | None = None,
        vision_llm: BaseLLM | None = None,
        compact_llm: BaseLLM | None = None,
        memory_similarity_threshold: float | None = None,
        memory_enabled: bool = True,
        tool_config: Any | None = None,
        agent_type: str = "standard",
        system_prompt: str | None = None,
        tools_initialized: bool | None = None,
        **agent_kwargs: Any,
    ) -> None:
        self.name = name
        self.memory = memory or InMemoryMemoryStore()
        # ``tools is not None`` (not ``bool(tools)``) — an explicitly empty
        # list is a legitimate "no tools allowed" result from the caller's
        # build path, distinct from "caller did not build tools yet".
        self.tools = list(tools) if tools is not None else []
        self.llm = llm
        self.fast_llm = fast_llm
        self.vision_llm = vision_llm
        self.compact_llm = compact_llm
        self.system_prompt = system_prompt
        self.memory_similarity_threshold = memory_similarity_threshold
        self.memory_enabled = memory_enabled
        self.tool_config = tool_config
        self.tracer = tracer or Tracer()
        # Default infer: if the caller passed ``tools`` (even ``[]``) the
        # constructed AgentService already has its final tool list, so
        # subsequent ``execute_task`` calls must not trigger a redundant
        # ``_ensure_tools_initialized`` rebuild against ``tool_config``.
        # Passing ``tools_initialized`` explicitly overrides the default
        # — only needed by tests / specialized callers that want to
        # retain the lazy-supplement path while also pre-supplying tools.
        if tools_initialized is None:
            self._tools_initialized = tools is not None
        else:
            self._tools_initialized = tools_initialized
        # Upstream's ``_ensure_tools_initialized`` gates the rebuild on
        # ``policy_signature == self._tool_policy_signature``. When the
        # caller pre-supplies tools (the PR1 fast path above) we also
        # need to capture the current signature, otherwise the very
        # first ``execute_task`` would see ``None != current`` and
        # rebuild anyway — re-introducing the duplicate factory call
        # PR1 was meant to eliminate. Re-reading the hook here is
        # cheap; the signature naturally invalidates when policy
        # changes later, restoring upstream's rebuild semantics.
        self._tool_policy_signature: Any | None = None
        if self._tools_initialized and self.tool_config:
            try:
                has_async_policy_refresh = callable(
                    getattr(self.tool_config, "refresh_runtime_policy", None)
                )
                self._tool_policy_signature = self._current_tool_policy_signature(
                    refresh_policy=not has_async_policy_refresh
                )
            except Exception as exc:
                logger.warning(
                    "Failed to capture initial tool policy signature for "
                    "AgentService '%s': %s; first execute may rebuild tools",
                    name,
                    exc,
                )
        self._is_paused = False
        self._pause_event = None
        self._current_runner = None
        self._execution_adapter: Any | None = None
        self._outbound_message_handler: Callable[[dict[str, Any]], Any] | None = None
        self._interrupt_checker: Callable[[], Any] | None = None
        self._conversation_history: list[dict[str, Any]] = []
        self._execution_context_messages: list[dict[str, str]] = []
        self._recovered_skill_context: str | None = None
        self.allowed_skills = self._get_allowed_skills_from_config(tool_config)
        self.skill_scope_context = self._get_skill_scope_context_from_config(
            tool_config
        )

        if use_dag_pattern is True:
            pattern = "dag_plan_execute"
        elif use_dag_pattern is not _UNSET and use_dag_pattern is not True:
            pattern = "react"
        self.pattern = str(pattern)
        self.use_dag_pattern = self.pattern == "dag_plan_execute"
        self.patterns = patterns or []

        if not id:
            raise ValueError("ID is required for AgentService")
        self.id = id
        self._current_task_id = str(task_id) if task_id else None
        self.workspace_base_dir = workspace_base_dir or str(get_uploads_dir())
        self.enable_workspace = enable_workspace
        self.allowed_external_dirs = allowed_external_dirs
        # ExecutionScope.workspace_segments this service was built under;
        # workspace_base_dir already includes them, but the workspace also
        # carries them explicitly for storage-key composition.
        self.scope_segments: tuple[str, ...] = tuple(scope_segments or ())
        self.workspace = workspace

        if (
            tool_config
            and hasattr(tool_config, "_workspace_config")
            and tool_config._workspace_config
            and not workspace
        ):
            from ..workspace import WorkspaceManager

            workspace_manager = WorkspaceManager()
            ws_config = tool_config._workspace_config
            self.workspace = workspace_manager.get_or_create_workspace(
                ws_config.get("base_dir", self.workspace_base_dir),
                ws_config.get("task_id", self.id),
                self.allowed_external_dirs,
                db_task_id=ws_config.get("db_task_id"),
                scope_segments=tuple(
                    ws_config.get("scope_segments") or self.scope_segments
                ),
            )
        elif self.enable_workspace:
            self._setup_workspace()

        if not self.tools and not self.tool_config:
            self.tool_config = self._create_default_tool_config()
            self.allowed_skills = self._get_allowed_skills_from_config(self.tool_config)

        # Compatibility shim for callers/tests that inspect service.agent.tools.
        self.agent = SimpleNamespace(
            name=self.name,
            tools=self.tools,
            patterns=self.patterns,
            llm=self.llm,
            memory_store=self.memory,
        )

        logger.info(
            "AgentService initialized for agent execution: name=%s, pattern=%s, "
            "llm=%s, compact_llm=%s",
            name,
            self.pattern,
            llm.model_name if llm else None,
            compact_llm.model_name if compact_llm else None,
        )

    def invalidate_tools(self) -> None:
        """Force the next execution to rebuild tools from ``tool_config``."""

        self._tools_initialized = False
        self._tool_policy_signature = None

    async def execute_task(
        self,
        task: str,
        context: dict[str, Any] | None = None,
        task_id: str | None = None,
    ) -> dict[str, Any]:
        has_files = bool(
            context and (context.get("uploaded_files") or context.get("file_info"))
        )
        if not task or not task.strip():
            if not has_files:
                raise ValueError("Task cannot be empty or whitespace-only")
        await self._ensure_tools_initialized()
        result = await self._execute_agent_task(task, context, task_id)
        file_outputs = self.get_output_files()
        if file_outputs:
            result["file_outputs"] = file_outputs
        return result

    async def pause_execution(self) -> bool:
        if self._is_paused:
            logger.warning("Agent '%s' is already paused", self.name)
            return True

        execution_id = self._current_task_id or self.id
        paused = self.pause_execution_by_id(
            str(execution_id), reason="paused by websocket"
        )
        if paused:
            self._is_paused = True
            logger.info(
                "Agent '%s' agent execution %s pause requested",
                self.name,
                execution_id,
            )
            return True
        logger.warning(
            "Agent '%s' could not find live agent execution %s to pause",
            self.name,
            execution_id,
        )
        return False

    async def resume_execution(self) -> None:
        if not self._is_paused:
            logger.warning("Agent '%s' is not paused", self.name)
            return
        self._is_paused = False

    def is_paused(self) -> bool:
        return self._is_paused

    def handle_websocket_input(self, user_input: str) -> bool:
        logger.info(
            "Synchronous websocket input ignored for agent service: %s", user_input
        )
        return False

    def set_outbound_message_handler(
        self,
        handler: Callable[[dict[str, Any]], Any] | None,
    ) -> None:
        self._outbound_message_handler = handler
        if self._execution_adapter is not None:
            self._execution_adapter.config.outbound_message_handler = handler

    def set_interrupt_checker(self, checker: Callable[[], Any] | None) -> None:
        """Register a per-step interrupt-checker (polled by the pattern loop at
        each LLM reply / tool call). Returning truthy — or a reason string —
        interrupts the run. Used to enforce mid-run quota; None disables it."""
        self._interrupt_checker = checker
        if self._execution_adapter is not None:
            self._execution_adapter.config.interrupt_checker = checker

    def set_allowed_skills(self, allowed_skills: list[str] | None) -> None:
        self.allowed_skills = allowed_skills
        if self._execution_adapter is not None:
            self._execution_adapter.config.allowed_skills = allowed_skills

    def supports_live_control(self) -> bool:
        return True

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
        if self._execution_adapter is None:
            self._execution_adapter = self._build_execution_adapter()
        return bool(
            await self._execution_adapter.post_user_message(
                execution_id,
                message,
                execution_message=execution_message,
                display_message=display_message,
                files=files,
                turn_id=turn_id,
                request_interrupt=request_interrupt,
                reason=reason,
            )
        )

    async def resume_execution_by_id(
        self,
        execution_id: str,
        **kwargs: Any,
    ) -> dict[str, Any] | None:
        self._is_paused = False
        await self._ensure_tools_initialized()
        if self._execution_adapter is None:
            self._execution_adapter = self._build_execution_adapter()
        else:
            self._execution_adapter.config.tools = self.tools
        return cast(
            dict[str, Any] | None,
            await self._execution_adapter.resume(execution_id, **kwargs),
        )

    def pause_execution_by_id(
        self, execution_id: str, reason: str | None = None
    ) -> bool:
        if self._execution_adapter is None:
            return False
        return bool(self._execution_adapter.pause(execution_id, reason=reason))

    def get_execution_status(self, execution_id: str) -> dict[str, Any] | None:
        if self._execution_adapter is None:
            return None
        return cast(
            dict[str, Any] | None, self._execution_adapter.get_status(execution_id)
        )

    def add_pattern(self, pattern: Any) -> None:
        self.patterns.append(pattern)
        self.agent.patterns = self.patterns

    def add_tool(self, tool: Tool) -> None:
        self.tools.append(tool)
        self.agent.tools = self.tools
        if self._execution_adapter is not None:
            self._execution_adapter.config.tools = self.tools

    def get_status(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "patterns_count": 1 if self.llm else 0,
            "tools_count": len(self.tools),
            "memory_type": self.memory.__class__.__name__,
            "ready": self.llm is not None,
            "execution_type": self._execution_type(),
            "llm_configured": self.llm is not None,
            "fast_llm_configured": self.fast_llm is not None,
            "vision_llm_configured": self.vision_llm is not None,
            "compact_llm_configured": self.compact_llm is not None,
            "dual_llm_enabled": self.fast_llm is not None,
            "compact_llm_enabled": self.compact_llm is not None,
        }

    def get_dag_pattern(self) -> Any | None:
        return None

    def set_conversation_history(self, messages: list[dict[str, Any]]) -> None:
        self._conversation_history = list(messages)
        if self._execution_adapter is not None:
            self._execution_adapter.config.conversation_history = (
                self._conversation_history
            )

    def set_execution_context_messages(self, messages: list[dict[str, Any]]) -> None:
        self._execution_context_messages = normalize_transcript_messages(messages)
        if self._execution_adapter is not None:
            self._execution_adapter.config.execution_context_messages = (
                self._execution_context_messages
            )

    def set_recovered_skill_context(self, skill_context: str | None) -> None:
        self._recovered_skill_context = skill_context
        if self._execution_adapter is not None:
            self._execution_adapter.config.recovered_skill_context = skill_context

    def get_task_info(self) -> dict[str, Any] | None:
        status = self.get_execution_status(self._current_task_id or self.id)
        if not status:
            return None
        metadata = status.get("metadata")
        return metadata if isinstance(metadata, dict) else None

    def skip_step(self, step_id: str) -> bool:
        return False

    def add_step_injection(
        self,
        step_id: str,
        pre_hook: Callable[[str, dict[str, Any]], str] | None = None,
        post_hook: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None,
    ) -> bool:
        return False

    def _setup_workspace(self) -> None:
        if not self.workspace:
            self.workspace = create_workspace(
                self.id,
                self.workspace_base_dir,
                self.allowed_external_dirs,
                scope_segments=self.scope_segments,
            )
        logger.info(
            "AgentService '%s' using workspace: %s",
            self.name,
            self.workspace.workspace_dir,
        )

    async def _execute_agent_task(
        self,
        task: str,
        context: dict[str, Any] | None = None,
        task_id: str | None = None,
    ) -> dict[str, Any]:
        if task_id:
            self._current_task_id = str(task_id)
        elif self._current_task_id is None:
            self._current_task_id = self.id

        if self._execution_adapter is None:
            self._execution_adapter = self._build_execution_adapter()
        else:
            self._execution_adapter.config.current_task_id = self._current_task_id
            self._execution_adapter.config.tools = self.tools
            self._execution_adapter.config.llm = self.llm
            self._execution_adapter.config.compact_llm = self.compact_llm
            self._execution_adapter.config.pattern = self.pattern
            self._execution_adapter.config.outbound_message_handler = (
                self._outbound_message_handler
            )
            self._execution_adapter.config.interrupt_checker = self._interrupt_checker
            self._execution_adapter.config.conversation_history = (
                self._conversation_history
            )
            self._execution_adapter.config.execution_context_messages = (
                self._execution_context_messages
            )
            self._execution_adapter.config.recovered_skill_context = (
                self._recovered_skill_context
            )
            self._execution_adapter.config.memory_store = (
                self.memory if self.memory_enabled else None
            )
            self._execution_adapter.config.allowed_skills = self.allowed_skills
            self._execution_adapter.config.skill_scope_context = (
                self.skill_scope_context
            )

        return cast(
            dict[str, Any],
            await self._execution_adapter.execute(
                task=task,
                context=context,
                task_id=task_id,
            ),
        )

    def _build_execution_adapter(self) -> Any:
        from .checkpoint import TraceCheckpointStore
        from .execution_adapter import AgentExecutionAdapter, AgentExecutionConfig

        checkpoint_reader_available = any(
            callable(getattr(handler, "load_latest_checkpoint", None))
            for handler in getattr(self.tracer, "handlers", [])
        )
        tracer = (
            TraceCheckpointStore(
                self.tracer,
                require_persisted=checkpoint_reader_available,
            )
            if self.tracer is not None
            else None
        )
        return AgentExecutionAdapter(
            AgentExecutionConfig(
                name=self.name,
                tools=self.tools,
                llm=self.llm,
                compact_llm=self.compact_llm,
                pattern=self.pattern,
                tracer=tracer,
                system_prompt=self.system_prompt,
                workspace_base_dir=self.workspace_base_dir,
                allowed_external_dirs=self.allowed_external_dirs,
                scope_segments=self.scope_segments,
                current_task_id=self._current_task_id,
                service_id=self.id,
                outbound_message_handler=self._outbound_message_handler,
                interrupt_checker=self._interrupt_checker,
                conversation_history=self._conversation_history,
                execution_context_messages=self._execution_context_messages,
                recovered_skill_context=self._recovered_skill_context,
                memory_store=self.memory if self.memory_enabled else None,
                memory_similarity_threshold=self.memory_similarity_threshold,
                skill_scope_context=self.skill_scope_context,
                allowed_skills=self.allowed_skills,
            )
        )

    def _get_allowed_skills_from_config(
        self, tool_config: Any | None
    ) -> list[str] | None:
        if tool_config and hasattr(tool_config, "get_allowed_skills"):
            return cast(list[str] | None, tool_config.get_allowed_skills())
        return None

    def _get_skill_scope_context_from_config(
        self, tool_config: Any | None
    ) -> Any | None:
        if tool_config and hasattr(tool_config, "get_skill_scope_context"):
            return tool_config.get_skill_scope_context()
        return None

    def get_workspace_files(self) -> dict[str, Any]:
        if self.workspace:
            return self.workspace.get_all_files()
        return {"error": "No workspace available", "files": []}

    def get_output_files(self) -> list[dict[str, Any]]:
        if self.workspace:
            return self.workspace.get_output_files()
        return []

    def add_file_to_workspace(
        self, file_path: str, target_subdir: str = "input"
    ) -> Path:
        if not self.workspace:
            raise ValueError("No workspace available")
        return self.workspace.copy_to_workspace(file_path, target_subdir)

    def cleanup_workspace(self) -> None:
        if self.workspace:
            workspace_path = str(self.workspace.workspace_dir)
            logger.info("Cleaning up workspace: %s", workspace_path)
            self.workspace.cleanup()
            self.workspace = None
            logger.info("Cleaned up workspace for AgentService '%s'", self.name)

    async def reconstruct_from_history(
        self,
        task_id: str,
        tracer_events: list[dict[str, Any]],
        plan_state: dict[str, Any] | None = None,
    ) -> None:
        self._current_task_id = str(task_id)
        logger.info("AgentService reconstruction prepared for task %s", task_id)

    def get_reconstruction_data(self) -> dict[str, Any]:
        return {
            "task_id": self._current_task_id,
            "agent_name": self.name,
            "patterns": 1 if self.llm else 0,
        }

    def _create_default_tool_config(self) -> Any:
        try:
            from ...core.tools.adapters.vibe.config import ToolConfig

            class DefaultToolConfig(ToolConfig):
                def __init__(self, workspace_config: dict[str, Any] | None = None):
                    config_dict: dict[str, Any] = {"workspace": workspace_config}
                    if workspace_config:
                        config_dict["task_id"] = workspace_config.get("task_id")
                    super().__init__(config_dict)

                def get_workspace_config(self) -> dict[str, Any] | None:
                    return self.workspace_config

                def get_file_tools_enabled(self) -> bool:
                    return True

                def get_basic_tools_enabled(self) -> bool:
                    return True

                def get_vision_model(self) -> Any | None:
                    return None

                def get_image_models(self) -> dict[str, Any]:
                    return {}

                async def get_mcp_server_configs(self) -> list[dict[str, Any]]:
                    return []

                def get_embedding_model(self) -> str | None:
                    return None

                def get_browser_tools_enabled(self) -> bool:
                    return True

                def get_task_id(self) -> str | None:
                    if self.workspace_config:
                        return self.workspace_config.get("task_id")
                    return None

                def get_user_id(self) -> int | None:
                    return None

                def get_db(self) -> Any:
                    return None

                def is_admin(self) -> bool:
                    return True

                def get_allowed_collections(self) -> list[Any] | None:
                    return None

                def get_allowed_skills(self) -> list[Any] | None:
                    return None

                def get_allowed_tools(self) -> list[Any] | None:
                    return None

                def get_enable_agent_tools(self) -> bool:
                    return False

            workspace_config = None
            if self.workspace:
                workspace_config = {
                    "base_dir": self.workspace.base_dir,
                    "task_id": self.workspace.id,
                    "allowed_external_dirs": [
                        str(d) for d in self.workspace.allowed_external_dirs
                    ],
                }
            return DefaultToolConfig(workspace_config)
        except Exception as exc:
            logger.warning("Failed to create default tool config: %s", exc)
            return None

    async def _ensure_tools_initialized(self) -> None:
        if self.tool_config:
            try:
                from ..tools.adapters.vibe.factory import ToolFactory

                if (
                    hasattr(self.tool_config, "_workspace_config")
                    and self.tool_config._workspace_config is not None
                ):
                    self.tool_config._workspace_config["task_id"] = self.id

                refresh_runtime_policy = getattr(
                    self.tool_config, "refresh_runtime_policy", None
                )
                if callable(refresh_runtime_policy):
                    await refresh_runtime_policy()
                    policy_signature = self._current_tool_policy_signature(
                        refresh_policy=False
                    )
                else:
                    policy_signature = self._current_tool_policy_signature()
                if (
                    self._tools_initialized
                    and policy_signature == self._tool_policy_signature
                ):
                    release_factory_runtime = getattr(
                        self.tool_config,
                        "release_prepared_factory_runtime",
                        None,
                    )
                    if callable(release_factory_runtime):
                        release_factory_runtime()
                    return

                # Rebuild the tool list so disabled tools disappear from reused agents.
                new_tools = await ToolFactory.create_all_tools(self.tool_config)
                self.tools = list(new_tools)

                if hasattr(self.tool_config, "get_allowed_tools"):
                    allowed_tools = self.tool_config.get_allowed_tools()
                    if allowed_tools is not None:
                        allowed_set = set(allowed_tools)
                        self.tools = [
                            tool
                            for tool in self.tools
                            if hasattr(tool, "name") and tool.name in allowed_set
                        ]

                self.agent.tools = self.tools
                if self._execution_adapter is not None:
                    self._execution_adapter.config.tools = self.tools
                self._tools_initialized = True
                self._tool_policy_signature = policy_signature
            except ConnectorRuntimeError:
                raise
            except RequiredMCPUnavailableError:
                raise
            except Exception as exc:
                logger.error("Failed to initialize tools from configuration: %s", exc)
                raise RuntimeError(
                    f"Tool initialization failed for AgentService '{self.name}': {exc}"
                ) from exc

    def _current_tool_policy_signature(
        self,
        *,
        refresh_policy: bool = True,
    ) -> tuple[Any, ...]:
        if not self.tool_config:
            return ()

        def _get_conf(attr: str, default: Any = None) -> Any:
            getter = getattr(self.tool_config, attr, None)
            return getter() if callable(getter) else default

        refresh_overrides = getattr(
            self.tool_config, "refresh_user_tool_overrides", None
        )
        if refresh_policy and callable(refresh_overrides):
            # Re-read the hook-backed policy before comparing signatures.
            refresh_overrides()

        refresh_allowlist = getattr(
            self.tool_config, "refresh_user_tool_allowlist", None
        )
        if refresh_policy and callable(refresh_allowlist):
            # The CA allowlist is resolved from the active execution scope, so
            # it can change between turns even when the config is reused.
            refresh_allowlist()

        overrides: Any = _get_conf("get_user_tool_overrides", {})
        if isinstance(overrides, dict):
            override_items = tuple(
                sorted(
                    (
                        str(name),
                        override.get("enabled") if isinstance(override, dict) else None,
                    )
                    for name, override in overrides.items()
                )
            )
        else:
            override_items = ()

        allowed_tools = _get_conf("get_allowed_tools")
        allowed_items = (
            None
            if allowed_tools is None
            else tuple(str(name) for name in allowed_tools)
        )

        tool_allowlist = normalize_tool_allowlist(_get_conf("get_user_tool_allowlist"))
        tool_allowlist_items = None if tool_allowlist is None else tuple(tool_allowlist)

        allowed_agent_ids = _get_conf("get_allowed_agent_ids")
        allowed_agent_items = (
            None
            if allowed_agent_ids is None
            else tuple(str(agent_id) for agent_id in allowed_agent_ids)
        )

        agent_tool_overrides = _get_conf("get_agent_tool_overrides", {})
        if isinstance(agent_tool_overrides, dict):
            agent_override_items = tuple(
                sorted(
                    (
                        str(agent_id),
                        tuple(
                            sorted(
                                (str(key), repr(value))
                                for key, value in override.items()
                            )
                        )
                        if isinstance(override, dict)
                        else repr(override),
                    )
                    for agent_id, override in agent_tool_overrides.items()
                )
            )
        else:
            agent_override_items = ()

        enable_global_agent_tools = _get_conf("get_enable_global_agent_tools", True)
        allow_cross_user_agent_ids = _get_conf("get_allow_cross_user_agent_ids", False)
        parent_task_id = _get_conf("get_parent_task_id")
        parent_tracer = _get_conf("get_parent_tracer")
        parent_tracer_identity = None if parent_tracer is None else id(parent_tracer)

        agent_call_stack = _get_conf("get_agent_call_stack", [])
        agent_call_stack_items = tuple(str(agent_id) for agent_id in agent_call_stack)
        mcp_failure_policy = _get_conf("get_mcp_failure_policy")
        mcp_failure_policy_value = getattr(mcp_failure_policy, "value", None)

        return (
            override_items,
            allowed_items,
            tool_allowlist_items,
            allowed_agent_items,
            agent_override_items,
            bool(enable_global_agent_tools),
            bool(allow_cross_user_agent_ids),
            None if parent_task_id is None else str(parent_task_id),
            parent_tracer_identity,
            agent_call_stack_items,
            mcp_failure_policy_value,
        )

    def _execution_type(self) -> str:
        if self.pattern == "dag_plan_execute":
            return "agent_dag"
        if self.pattern == "auto":
            return "agent_auto"
        if self.pattern == "single_call":
            return "agent_single_call"
        return "agent_react"
