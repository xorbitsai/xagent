"""Chat API route handlers"""

import asyncio
import logging
import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, cast

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import func, or_
from sqlalchemy.orm import Session

from ...config import (
    get_default_task_execution_mode,
    get_external_upload_dirs,
    get_uploads_dir,
)
from ...core.agent.service import AgentService
from ...core.execution_scope import (
    ExecutionScope,
    ScopeFingerprint,
    get_execution_scope,
    resolve_execution_scope,
    scope_fingerprint,
)
from ...core.memory.base import MemoryStore
from ...core.memory.in_memory import InMemoryMemoryStore
from ...core.model.chat.basic.base import BaseLLM
from ...core.model.chat.basic.deepseek import DeepSeekLLM
from ...core.model.chat.basic.openai import OpenAILLM
from ...core.model.chat.basic.zhipu import ZhipuLLM
from ...core.model.chat.token_context import aggregate_token_usage_by_model
from ...core.model.providers import is_placeholder_api_key
from ...core.tools.adapters.vibe.config import (
    MCPFailurePolicy,
    RequiredMCPUnavailableError,
)
from ...core.tools.adapters.vibe.selection_spec import should_load_mcp_server_configs
from ...core.workspace import scoped_user_root
from ..auth_dependencies import get_current_user
from ..dynamic_memory_store import get_memory_store
from ..models.agent import Agent, AgentStatus, is_workforce_generated_manager_agent
from ..models.chat_message import TaskChatMessage
from ..models.database import get_db, release_db_connection_if_clean
from ..models.model import Model as DBModel
from ..models.task import AgentType, Task, TaskStatus, TraceEvent
from ..models.user import User
from ..models.user_channel import UserChannel
from ..sandbox_keys import (
    USER_LIFECYCLE_TYPE,
    make_user_lifecycle_id,
    make_user_sandbox_key,
    parse_user_sandbox_key,
)
from ..schemas.chat import TaskCreateRequest, TaskCreateResponse
from ..services.agent_access import list_accessible_published_agents
from ..services.agent_team_scope import get_agent_team_scope, owned_agent_clause
from ..services.chat_history_service import (
    get_latest_waiting_question,
    load_task_transcript,
)
from ..services.connector_runtime import (
    bind_connector_runtime_selection_snapshot,
    prepare_connector_runtime_selection_snapshot,
)
from ..services.hot_path_cache import (
    cache_get,
    cache_set,
    cache_version_token,
    invalidate_task_cache,
    task_cache_ttl_seconds,
    web_task_detail_key,
    web_task_status_key,
)
from ..services.llm_utils import resolve_llms_from_names
from ..services.managed_file_ref import ensure_uploaded_file_local_path
from ..services.model_service import _get_visible_user_ids
from ..services.task_execution_context_service import (
    load_task_execution_recovery_state,
)
from ..services.task_lease_service import (
    acquire_task_lease,
    mark_task_paused_if_stale,
    run_task_lease_heartbeat,
    stop_task_lease_heartbeat,
)
from ..services.task_setup_snapshot import (
    TaskOwnerMismatchError,
    TaskSetupSnapshot,
    load_task_setup_snapshot_sync,
)
from ..services.trace_message_storage import decode_trace_events_data
from ..services.workforce_runtime import (
    WorkforceTaskRuntime,
    release_task_lease_with_workforce_sync,
    resolve_workforce_task_runtime,
    sync_workforce_run_status,
)
from ..tracing import create_task_tracer
from ..user_isolated_memory import UserContext
from ..utils.db_timezone import format_datetime_for_api, safe_timestamp_to_unix
from .public_trace_events import public_task_trace_filter

logger = logging.getLogger(__name__)

# Depth of the per-task recently-evicted scope-fingerprint memory used for
# resolver-flap detection; catches scope cycles up to this period.
_EVICTED_FINGERPRINT_MEMORY = 4

# Create router
chat_router = APIRouter(prefix="/api/chat", tags=["chat"])

_TERMINAL_CACHE_STATUSES = {TaskStatus.COMPLETED, TaskStatus.FAILED}


def _build_task_agent_config(
    request_agent_config: Optional[Dict[str, Any]],
    selected_file_ids: list[str],
) -> Optional[Dict[str, Any]]:
    """Build task agent_config with server-owned selected file ids."""
    task_agent_config: Dict[str, Any] = {}
    if isinstance(request_agent_config, dict):
        task_agent_config.update(request_agent_config)
        task_agent_config.pop("selected_file_ids", None)
    if selected_file_ids:
        task_agent_config["selected_file_ids"] = selected_file_ids
    return task_agent_config or None


def _is_published_agent(agent: Agent) -> bool:
    return getattr(agent.status, "value", agent.status) == AgentStatus.PUBLISHED.value


def _load_agent_for_task_create(
    db: Session,
    user: User,
    agent_id: int,
) -> Agent | None:
    agent = db.query(Agent).filter(Agent.id == agent_id).first()
    if agent is None:
        return None
    if is_workforce_generated_manager_agent(agent):
        return None
    # Team-scoped ownership: teammates may run a team-visible agent, and an
    # admins-only agent stays hidden from non-admins. Falls back to the
    # published-visibility path below when the caller does not own it.
    owned = (
        db.query(Agent)
        .filter(
            Agent.id == agent_id,
            owned_agent_clause(int(user.id), get_agent_team_scope(db, int(user.id))),
        )
        .first()
    )
    if owned is not None:
        return owned
    if not _is_published_agent(agent):
        return None
    visible_agent_ids = {
        int(item.id)
        for item in list_accessible_published_agents(
            db,
            user,
            purpose="agent_list",
        )
    }
    return agent if int(agent.id) in visible_agent_ids else None


def _load_agent_for_task_runtime(
    db: Session,
    task: Task,
    workforce_runtime: WorkforceTaskRuntime | None = None,
) -> Agent | None:
    task_agent_id = _int_id_or_none(getattr(task, "agent_id", None))
    if task_agent_id is None:
        return None

    agent = db.query(Agent).filter(Agent.id == task_agent_id).first()
    if agent is None:
        return None
    if is_workforce_generated_manager_agent(agent):
        if (
            workforce_runtime is not None
            and workforce_runtime.manager_agent_id == task_agent_id
        ):
            return agent
        return None
    # Team-scoped ownership keyed on the task owner: a teammate's task may run
    # a team-visible agent; admins-only stays hidden from non-admins.
    task_user_id = int(task.user_id)
    owned = (
        db.query(Agent)
        .filter(
            Agent.id == task_agent_id,
            owned_agent_clause(task_user_id, get_agent_team_scope(db, task_user_id)),
        )
        .first()
    )
    if owned is not None:
        return owned
    if (
        workforce_runtime is not None
        and workforce_runtime.manager_agent_id == task_agent_id
    ):
        return agent
    if not _is_published_agent(agent):
        return None

    user = db.query(User).filter(User.id == task.user_id).first()
    if user is None:
        return None
    visible_agent_ids = {
        int(item.id)
        for item in list_accessible_published_agents(
            db,
            user,
            purpose="agent_list",
        )
    }
    return agent if int(agent.id) in visible_agent_ids else None


def _get_task_activity_ids(db: Session, task_id: int) -> tuple[int, int]:
    max_trace_event_id = (
        db.query(func.max(TraceEvent.id))
        .filter(
            TraceEvent.task_id == task_id,
            public_task_trace_filter(TraceEvent),
        )
        .scalar()
        or 0
    )
    max_chat_message_id = (
        db.query(func.max(TaskChatMessage.id))
        .filter(TaskChatMessage.task_id == task_id)
        .scalar()
        or 0
    )
    return int(max_trace_event_id), int(max_chat_message_id)


@dataclass(frozen=True)
class AgentServiceMemoryPolicy:
    memory: MemoryStore
    memory_enabled: bool


def resolve_agent_service_memory_policy(
    *,
    task: Optional[Task] = None,
    agent_config: Optional[Mapping[str, Any]] = None,
) -> AgentServiceMemoryPolicy:
    """Resolve the memory store and enablement for an AgentService runtime."""
    config = agent_config
    if config is None:
        task_config = getattr(task, "agent_config", None)
        config = task_config if isinstance(task_config, Mapping) else {}

    if config.get("is_preview") is True:
        return AgentServiceMemoryPolicy(InMemoryMemoryStore(), False)

    if task is not None and task.agent_id:
        return AgentServiceMemoryPolicy(get_memory_store(), False)

    return AgentServiceMemoryPolicy(get_memory_store(), True)


def create_default_llm() -> Optional[BaseLLM]:
    """Create a default LLM instance based on environment configuration"""
    try:
        # For OpenAI: allow empty string API key (use is not None check)
        # For Zhipu: don't allow empty string API key (use truthy check)
        openai_api_key = os.getenv("OPENAI_API_KEY")
        zhipu_api_key = os.getenv("ZHIPU_API_KEY")
        deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")

        # Similarly for base_url: prefer OPENAI_BASE_URL if it exists (even if empty string)
        # Only fallback to ZHIPU_BASE_URL if OPENAI_BASE_URL is None
        openai_base_url = os.getenv("OPENAI_BASE_URL")
        zhipu_base_url = os.getenv("ZHIPU_BASE_URL")
        deepseek_base_url = os.getenv("DEEPSEEK_BASE_URL")

        # For model_name: prefer OPENAI_MODEL if it exists (even if empty string)
        # Only fallback to ZHIPU_MODEL_NAME if OPENAI_MODEL is None
        openai_model = os.getenv("OPENAI_MODEL")
        zhipu_model = os.getenv("ZHIPU_MODEL_NAME")
        deepseek_model = os.getenv("DEEPSEEK_MODEL_NAME")

        # Check if Zhipu
        zhipu_models = {
            "glm-4.7",
            "glm-4.7-flashx",
            "glm-4.6",
            "glm-4.5-air",
            "glm-4.5-airx",
            "glm-4-long",
            "glm-4-flashx-250414",
            "glm-4.7-flash",
            "glm-4-Flash-250414",
        }
        is_zhipu = (
            zhipu_base_url
            and any(
                domain in zhipu_base_url.lower()
                for domain in {"zhipu", "bigmodel.cn", "api.z.ai"}
            )
        ) or (
            zhipu_model
            and any(zhipu_model.lower().strip() in x.lower() for x in zhipu_models)
        )

        if is_zhipu:
            if zhipu_api_key:
                logger.info(f"Using Zhipu LLM with model: {zhipu_model}")
                # Use automatic thinking mode (None) by default
                thinking_mode_env = os.getenv("ZHIPU_THINKING_MODE", "auto").lower()
                thinking_mode = (
                    None if thinking_mode_env == "auto" else thinking_mode_env == "true"
                )
                return ZhipuLLM(
                    model_name=zhipu_model or "glm-4.7-flash",
                    api_key=zhipu_api_key,
                    base_url=zhipu_base_url,
                    thinking_mode=thinking_mode,
                )
            else:
                logger.error(
                    "Zhipu API key not found in environment variables. Set ZHIPU_API_KEY to enable Zhipu LLM functionality."
                )
                return None
        elif openai_api_key is not None and (
            openai_api_key == "" or not is_placeholder_api_key(openai_api_key)
        ):
            logger.info(f"Using OpenAI LLM with model: {openai_model}")
            return OpenAILLM(
                model_name=openai_model or "gpt-4o-mini",
                base_url=openai_base_url,
                api_key=openai_api_key,
            )
        elif deepseek_api_key and not is_placeholder_api_key(deepseek_api_key):
            logger.info(f"Using DeepSeek LLM with model: {deepseek_model}")
            return DeepSeekLLM(
                model_name=deepseek_model or "deepseek-v4-flash",
                base_url=deepseek_base_url,
                api_key=deepseek_api_key,
            )

        # No LLM available - AgentService will run without DAG pattern
        logger.error(
            "No API key found in environment variables. Set OPENAI_API_KEY, ZHIPU_API_KEY, or DEEPSEEK_API_KEY to enable LLM functionality."
        )
        return None

    except Exception as e:
        logger.error(f"Failed to create default LLM: {e}")
        return None


def _spec_wants_mcp(tool_selection_spec: Optional[Any]) -> bool:
    """Whether the caller's spec actually asked for MCP tools.

    Default agents (no spec, or ``_SpecAll``) should NOT trigger the
    MCP server DB query + per-server session init in
    ``WebToolConfig``. Only an explicit ``"mcp"`` plain entry or a
    ``"mcp:<server>"`` sub-category entry in the user's selection
    means MCP loading is wanted; anything else keeps the legacy
    no-MCP behaviour for cost reasons.

    Returns ``False`` for ``None`` (no spec / legacy caller),
    ``_SpecAll`` (no restriction means "build registered defaults",
    NOT "ALL including MCP"), and ``_SpecNone`` (zero tools). Only a
    ``_SpecByCategories`` whose normalized policy includes MCP (plain
    ``"mcp"`` category or a scoped ``mcp_servers``) wants MCP loading;
    this defers to the spec's own ``includes_mcp()`` dispatch.
    """
    return should_load_mcp_server_configs(tool_selection_spec)


def _build_tool_selection_spec_for_task(
    agent_config: Optional[dict],
    workforce_runtime: Optional[WorkforceTaskRuntime],
    *,
    task_id: int,
) -> Any:
    """Single SSOT normalizer for chat reconstruct + snapshot paths.

    Both ``_build_tools_for_task`` (reconstruct) and
    ``get_agent_for_task`` (snapshot) translate the same raw inputs
    (``agent_config['tool_categories']`` + optional workforce worker
    tool names) into a :class:`ToolSelectionSpec`. Centralising the
    call here keeps the two paths in lockstep and avoids the 30-line
    copy / paste that used to live in each.
    """
    from ...core.tools.adapters.vibe.selection_spec import ToolSelectionSpec

    tool_categories = agent_config.get("tool_categories") if agent_config else None
    spec = ToolSelectionSpec.from_raw(
        tool_categories=tool_categories,
        # Two orthogonal inputs for the workforce delegation case:
        #   - published_agent_ids declares the published-agent creator
        #     should run, scoped to the worker agents (dispatch);
        #   - name_allowlist narrows its output to the worker tool names
        #     (filter). The creator's own config.allowed_agent_ids does
        #     the per-agent DB filtering.
        published_agent_ids=(
            list(workforce_runtime.allowed_agent_ids) if workforce_runtime else None
        ),
        name_allowlist=(
            workforce_runtime.worker_tool_names if workforce_runtime else None
        ),
        extras_only_when_unconfigured=workforce_runtime is not None,
    )
    if spec.is_all():
        logger.info(
            f"Task {task_id} has no tool_categories restriction "
            "(legacy 'unconfigured' semantics) -- full default tool set will be built"
        )
    else:
        logger.info(
            f"Task {task_id} tool selection spec: "
            f"{type(spec).__name__} with categories={tool_categories}"
        )
    return spec


def _mcp_failure_policy_for_task_source(source: object) -> MCPFailurePolicy:
    """Map the persisted task source to its MCP setup contract."""
    if type(source) is str and source == "trigger":
        return MCPFailurePolicy.STRICT
    return MCPFailurePolicy.BEST_EFFORT


def _build_allowed_external_dirs(
    user_id: Optional[int],
    *,
    only_existing: bool = False,
    scope: Optional[ExecutionScope] = None,
) -> list[str]:
    """Build the allowed_external_dirs list for AgentService / tool
    workspace_config.

    Without this whitelist, file tools (read_file, read_csv_file,
    list_files, ...) restrict themselves to the per-task workspace dir
    and reject every uploaded file with "outside the allowed directory".

    The list always contains:
      - the user's upload directory ``<uploads>/user_<id>``
        (when ``only_existing`` is True, only if that directory exists)
      - any directories returned by ``get_external_upload_dirs()`` (used
        for shared knowledge bases configured at the deployment level)
    """
    dirs: list[str] = []
    if user_id is not None:
        # Default: the shared user-level upload dir, so already-uploaded
        # KB files stay reachable from every scope. With
        # ``isolate_external_dirs`` the entry becomes the scoped subtree,
        # keeping upload writes and the mount/enforcement allowlist
        # consistent per scope. Deployment-level external dirs
        # (XAGENT_EXTERNAL_UPLOAD_DIRS) are not user-root derived and stay
        # shared either way.
        segments = (
            scope.workspace_segments
            if scope is not None and scope.isolate_external_dirs
            else ()
        )
        user_upload_dir = scoped_user_root(get_uploads_dir(), user_id, segments)
        if not only_existing or user_upload_dir.exists():
            dirs.append(str(user_upload_dir))
    dirs.extend([str(d) for d in get_external_upload_dirs()])
    return dirs


def _build_workforce_system_prompt(
    base_system_prompt: Optional[str],
    workforce_runtime: Optional[WorkforceTaskRuntime],
) -> Optional[str]:
    prompts = []
    if workforce_runtime and workforce_runtime.manager_system_prompt:
        prompts.append(workforce_runtime.manager_system_prompt)
    if base_system_prompt:
        prompts.append(base_system_prompt)
    return "\n\n".join(prompts) if prompts else None


def _int_id_or_none(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


async def create_default_tools(
    db: Session,
    request: Any = None,
    user: Optional[User] = None,
    task_id: Optional[str] = None,
    workspace_owner_id: Optional[int] = None,
    allowed_collections: Optional[List[str]] = None,
    allowed_skills: Optional[List[str]] = None,
    excluded_agent_id: Optional[int] = None,
    vision_model: Optional[Any] = None,
    sandbox: Optional[Any] = None,
    llm: Optional[Any] = None,
    tool_selection_spec: Optional[Any] = None,
    allowed_agent_ids: Optional[List[int]] = None,
    agent_tool_overrides: Optional[Dict[int, Dict[str, Any]]] = None,
    enable_global_agent_tools: bool = True,
    allow_cross_user_agent_ids: bool = False,
    parent_task_id: Optional[str] = None,
    parent_tracer: Optional[Any] = None,
    agent_call_stack: Optional[List[int]] = None,
    scope: Optional[ExecutionScope] = None,
    connector_runtime_turn_id: Optional[str] = None,
    mcp_failure_policy: MCPFailurePolicy = MCPFailurePolicy.BEST_EFFORT,
    mcp_load_summary_tracer: Optional[Any] = None,
    mcp_load_summary_trace_task_id: Optional[str] = None,
) -> tuple[list[Any], Any]:
    """Create default tools and tool_config for AgentService using ToolFactory.

    ``selection_spec`` (a :class:`ToolSelectionSpec` or ``None``) is
    propagated into the ``WebToolConfig`` so :func:`ToolFactory.create_all_tools`
    can skip creators (and their internal DB / network I/O) for tool
    categories / MCP servers the agent does not need. ``None`` preserves
    the original "build everything" behavior for backward compat.
    """
    if not user:
        raise ValueError("User is required for tool creation")
    if not task_id:
        raise ValueError("Task ID is required for tool creation")

    # Create a WebToolConfig to properly initialize tools
    from ..tools.config import WebToolConfig

    owner_id = (
        int(workspace_owner_id) if workspace_owner_id is not None else int(user.id)
    )

    # Build allowed external directories so file tools can reach the task owner's
    # uploads (see _build_allowed_external_dirs docstring).
    allowed_external_dirs = _build_allowed_external_dirs(owner_id, scope=scope)
    scope_segments = scope.workspace_segments if scope is not None else ()

    tool_config = WebToolConfig(
        db=db,
        request=request,
        user=user,
        llm=llm,
        user_id=int(user.id),
        is_admin=bool(user.is_admin),
        workspace_config={
            "base_dir": str(
                scoped_user_root(get_uploads_dir(), owner_id, scope_segments)
            ),
            "task_id": task_id,
            "user_id": owner_id,
            "allowed_external_dirs": allowed_external_dirs,
            "scope_segments": scope_segments,
        },
        execution_scope=scope,
        # Only load MCP servers (a DB query + per-server session init)
        # when the caller actually picked MCP. Spec=None / _SpecAll
        # default agents shouldn't pay that cost; only explicit
        # ``mcp`` / ``mcp:<server>`` selection triggers MCP loading.
        # Derived from the spec rather than re-deriving from a raw
        # name list so the source of truth is in one place.
        include_mcp_tools=_spec_wants_mcp(tool_selection_spec),
        task_id=task_id,  # Pass task_id for browser session tracking
        browser_tools_enabled=True,  # Enable browser automation tools
        allowed_collections=allowed_collections,  # Agent Builder knowledge bases
        allowed_skills=allowed_skills,  # Agent Builder skills
        vision_model=vision_model,  # Pass task-specific vision model
        tool_selection_spec=tool_selection_spec,  # Preferred SSOT typed spec
        allowed_agent_ids=allowed_agent_ids,
        agent_tool_overrides=agent_tool_overrides,
        enable_global_agent_tools=enable_global_agent_tools,
        allow_cross_user_agent_ids=allow_cross_user_agent_ids,
        parent_task_id=parent_task_id,
        parent_tracer=parent_tracer,
        agent_call_stack=agent_call_stack,
        connector_runtime_turn_id=connector_runtime_turn_id,
        mcp_failure_policy=mcp_failure_policy,
        mcp_load_summary_tracer=mcp_load_summary_tracer,
        mcp_load_summary_trace_task_id=mcp_load_summary_trace_task_id,
    )

    # Store excluded_agent_id in tool_config for agent tool filtering
    if excluded_agent_id:
        tool_config._excluded_agent_id = excluded_agent_id

    # Use sandbox if available
    if sandbox:
        tool_config.set_sandbox(sandbox)

    from ...core.tools.adapters.vibe.factory import ToolFactory

    # Use ToolFactory to create proper xagent tools
    tools = await ToolFactory.create_all_tools(tool_config)

    logger.info(f"Created {len(tools)} default tools using ToolFactory")
    return tools, tool_config


async def update_task_title_from_agent(
    agent_service: AgentService, task_id: int, db: Session
) -> bool:
    """Update task title with generated task_name from agent service.

    This is a clean separation of concerns:
    - Core layer (AgentService) provides task info via get_task_info()
    - Web layer handles database updates

    Args:
        agent_service: The agent service that executed the task
        task_id: The task ID to update
        db: Database session

    Returns:
        True if title was updated, False otherwise
    """
    try:
        # Get task info from core layer (clean API)
        task_info = agent_service.get_task_info()

        if not task_info:
            logger.debug(f"No task info available for task {task_id}")
            return False

        task_name = task_info.get("task_name")
        if not task_name:
            logger.debug(f"No task_name in task info for task {task_id}")
            return False

        # Update database (web layer responsibility)
        from ..models.task import Task as TaskModel

        task_record = db.query(TaskModel).filter(TaskModel.id == task_id).first()
        if not task_record:
            logger.warning(f"No task record found for task_id={task_id}")
            return False

        # Only update if title is different
        if task_record.title != task_name:
            old_title = task_record.title
            task_record.title = task_name
            db.commit()
            logger.info(
                f"Updated task {task_id} title from '{old_title}' to '{task_name}'"
            )
            return True
        else:
            logger.debug(f"Task title already matches: '{task_record.title}'")
            return False

    except Exception as e:
        logger.error(
            f"Failed to update task title for task {task_id}: {e}", exc_info=True
        )
        return False


class AgentServiceManager:
    """Manage AgentService instances for different tasks"""

    def __init__(self, request: Optional[Any] = None) -> None:
        self._agents: Dict[int, AgentService] = {}
        # Building an AgentService performs multiple awaits (snapshot/tool/
        # sandbox setup). Two WebSocket connections can otherwise both observe
        # a cache miss, build different execution registries for the same task,
        # and leave pause/message control attached to the instance that loses
        # the final cache assignment.
        self._agent_build_locks: Dict[int, asyncio.Lock] = {}
        # Owner (runtime identity) each cached AgentService was built for. A
        # task_id-keyed cache must not silently hand back an instance built
        # under a different user (e.g. once built with the wrong identity).
        self._agent_owner_ids: Dict[int, Optional[int]] = {}
        self._agent_sandbox_keys: Dict[int, str] = {}
        # ExecutionScope fingerprint each cached AgentService was built
        # under (None sentinel = unscoped). Sandbox keys and workspace
        # paths are baked in at build time, so a task reassigned to a
        # different scope between turns must evict and rebuild instead of
        # silently executing in the old scope's namespace.
        self._agent_scope_fingerprints: Dict[int, Optional[ScopeFingerprint]] = {}
        # Recently evicted fingerprints per task (bounded): resolving back
        # to any of them points at a resolver cycling between scopes, which
        # silently rebuilds the agent every turn and defeats the cache. The
        # bounded memory catches cycles up to its depth (A->B->A and longer
        # A->B->C->A style periods), not just the immediate flap.
        self._agent_evicted_scope_fingerprints: Dict[
            int, deque[Optional[ScopeFingerprint]]
        ] = {}
        self._default_llm = create_default_llm()
        self.request = request

    @staticmethod
    def _parse_sandbox_key(sandbox_key: str) -> tuple[str, str]:
        """Split a recorded sandbox key into sandbox-manager lifecycle parts.

        Round-trips through the shared key helpers so the format lives in
        exactly one place; scoped keys (``user:{owner}:{suffix}``) map to
        the composite lifecycle id ``{owner}:{suffix}``.
        """
        owner_id, suffix = parse_user_sandbox_key(sandbox_key)
        return USER_LIFECYCLE_TYPE, make_user_lifecycle_id(owner_id, suffix)

    async def _acquire_sandbox_task(self, task_id: Optional[str]) -> Optional[str]:
        """Attach the task to its sandbox lifecycle for the execution.

        Returns the sandbox key on success, or None when the task runs
        without sandbox tracking (sandbox disabled, local-execution
        fallback, or no recorded sandbox for the task). The key is only
        ever read from the per-task map recorded at sandbox build time —
        never re-derived from the owner id, because a reconstructed
        owner-only key would silently miss a scope-suffixed sandbox and
        skip the ref-count attach.

        Raises:
            RuntimeError: The task's agent was built with a sandbox lease
                provider (recorded key) that has since been reclaimed by
                the idle sweep or capacity eviction. Running anyway would
                hit deleted containers with cryptic tool errors; failing
                clearly lets a retry rebuild the agent and transparently
                recreate the sandbox.
        """
        if task_id is None:
            return None
        try:
            task_key = int(task_id)
        except (TypeError, ValueError):
            return None

        sandbox_key = self._agent_sandbox_keys.get(task_key)
        if sandbox_key is None:
            return None

        from ..sandbox_manager import get_sandbox_manager

        sandbox_mgr = get_sandbox_manager()
        if sandbox_mgr is None:
            return None

        lifecycle_type, lifecycle_id = self._parse_sandbox_key(sandbox_key)
        if await sandbox_mgr.attach(lifecycle_type, lifecycle_id):
            return sandbox_key

        # Evict the stale cached agent so a retry rebuilds its tools
        # against a freshly created sandbox.
        self._agents.pop(task_key, None)
        self._agent_owner_ids.pop(task_key, None)
        self._agent_sandbox_keys.pop(task_key, None)
        self._agent_scope_fingerprints.pop(task_key, None)
        raise RuntimeError(
            f"The sandbox for task {task_key} was reclaimed before "
            "execution started (idle reclamation or capacity "
            "eviction). Please retry the task; the sandbox will be "
            "recreated automatically."
        )

    def _evict_agents_for_sandbox(self, sandbox_key: str) -> None:
        """Drop cached AgentService objects that were built with this sandbox.

        Only agents whose recorded key matches are evicted. The key is
        recorded at build time and never re-derived, so a cached agent
        without a recorded key was built for local execution and holds no
        reference to the released sandbox (the former owner-id supplement
        existed only for the removed owner-key attach fallback). Scoped and
        unscoped keys under the same owner are distinct sandboxes, so
        releasing one never evicts the other's agents.
        """
        task_ids = {
            task_key
            for task_key, agent_sandbox_key in self._agent_sandbox_keys.items()
            if agent_sandbox_key == sandbox_key
        }

        for task_key in task_ids:
            self._agents.pop(task_key, None)
            self._agent_owner_ids.pop(task_key, None)
            self._agent_sandbox_keys.pop(task_key, None)
            self._agent_scope_fingerprints.pop(task_key, None)
            logger.info(
                "Evicted cached AgentService for task %s after releasing sandbox %s",
                task_key,
                sandbox_key,
            )

    async def _get_or_create_task_sandbox(
        self,
        *,
        task_id: int,
        workspace_owner_id: int,
        workspace_config: Mapping[str, Any],
        scope: Optional[ExecutionScope] = None,
    ) -> Any | None:
        """Get the task's sandbox lease provider, or None for local execution.

        When ``scope`` carries a ``sandbox_key_suffix``, the lifecycle key
        becomes ``user:{owner}:{suffix}`` — a separate container family per
        scope under the same platform user. Unscoped execution keeps
        producing ``user:{owner}`` and reuses today's containers untouched.

        Capacity exhaustion and sandbox-service unavailability are distinct
        failure classes. For **unscoped** execution the historical behavior is
        kept: a ``SandboxCapacityError`` rejects the task by default (opt-in
        local fallback via XAGENT_SANDBOX_ALLOW_LOCAL_FALLBACK_ON_CAPACITY),
        and any other sandbox failure falls back to local execution.

        A scope that carries a ``sandbox_key_suffix`` isolates an untrusted /
        third-party workload in a per-scope container; running it outside that
        container would defeat the isolation the scope exists to provide. So
        when a suffix is present both fallbacks are disabled and the task fails
        closed regardless of configuration — neither capacity pressure (even
        with the opt-in flag on) nor a sandbox-service failure may downgrade
        such a task to local execution. A scope *without* a suffix shares the
        unscoped ``user:{owner}`` container and thus has no isolation to
        protect; it keeps the unscoped fallback behavior.

        Raises:
            SandboxCapacityError: The container cap is reached, nothing is
                evictable, and (for a task with no scope suffix) local fallback
                on capacity is not enabled, or the scope carries a suffix.
            Exception: A suffix-scoped execution hit a non-capacity sandbox
                failure; re-raised instead of falling back to local execution.
        """
        from ..sandbox_manager import SandboxCapacityError, get_sandbox_manager

        sandbox_mgr = get_sandbox_manager()
        if not sandbox_mgr:
            self._agent_sandbox_keys.pop(task_id, None)
            return None

        # The isolation boundary is the per-scope key suffix, not
        # scope-presence: a suffix-less scope resolves to the same
        # ``user:{owner}`` lifecycle key as unscoped execution, so it has no
        # container of its own to protect and must keep the unscoped fallback
        # behavior. Gating on the suffix (not ``scope is not None``) honors the
        # ExecutionScope contract that each field be consumed independently.
        suffix = scope.sandbox_key_suffix if scope is not None else None
        scoped = suffix is not None
        try:
            sandbox = await sandbox_mgr.get_or_create_lease_provider(
                USER_LIFECYCLE_TYPE,
                make_user_lifecycle_id(workspace_owner_id, suffix),
                workspace_config=workspace_config,
            )
        except SandboxCapacityError as e:
            self._agent_sandbox_keys.pop(task_id, None)
            from ...config import get_sandbox_allow_local_fallback_on_capacity

            if not scoped and get_sandbox_allow_local_fallback_on_capacity():
                logger.warning(
                    "Sandbox capacity reached for workspace owner %s; "
                    "falling back to local execution "
                    "(XAGENT_SANDBOX_ALLOW_LOCAL_FALLBACK_ON_CAPACITY): %s",
                    workspace_owner_id,
                    e,
                )
                return None
            logger.warning(
                "Sandbox capacity reached for workspace owner %s; "
                "rejecting task %s (scoped=%s): %s",
                workspace_owner_id,
                task_id,
                scoped,
                e,
            )
            raise
        except Exception as e:
            self._agent_sandbox_keys.pop(task_id, None)
            if scoped:
                logger.error(
                    "Sandbox creation failed for scoped task %s (workspace "
                    "owner %s); failing closed instead of running the scoped "
                    "workload locally: %s",
                    task_id,
                    workspace_owner_id,
                    e,
                )
                raise
            logger.warning(
                "Sandbox creation failed for workspace owner %s, "
                "falling back to local execution: %s",
                workspace_owner_id,
                e,
            )
            return None

        self._agent_sandbox_keys[task_id] = make_user_sandbox_key(
            workspace_owner_id, suffix
        )
        return sandbox

    async def _release_sandbox_task(self, sandbox_key: Optional[str]) -> None:
        if sandbox_key is None:
            return

        from ..sandbox_manager import get_sandbox_manager

        sandbox_mgr = get_sandbox_manager()
        if sandbox_mgr is None:
            return

        lifecycle_type, lifecycle_id = self._parse_sandbox_key(sandbox_key)
        try:
            await sandbox_mgr.release(
                lifecycle_type,
                lifecycle_id,
                on_last_release=lambda: self._evict_agents_for_sandbox(sandbox_key),
            )
        except Exception as exc:
            logger.warning(
                "Failed to release sandbox %s: %s",
                sandbox_key,
                exc,
            )

    def _get_task_llm_ids(self, task: Task, db: Session) -> List[Optional[str]]:
        """Return internal model_id identifiers for a task (never provider model_name)."""
        from ..services.llm_utils import CoreStorage, make_normalize_model_id

        core_storage = CoreStorage(db, DBModel)

        _normalize = make_normalize_model_id(core_storage)

        return [
            _normalize(
                getattr(task, "model_id", None), getattr(task, "model_name", None)
            ),
            _normalize(
                getattr(task, "small_fast_model_id", None),
                getattr(task, "small_fast_model_name", None),
            ),
            _normalize(
                getattr(task, "visual_model_id", None),
                getattr(task, "visual_model_name", None),
            ),
            _normalize(
                getattr(task, "compact_model_id", None),
                getattr(task, "compact_model_name", None),
            ),
        ]

    def set_task_llms(
        self, task_id: int, llm_ids: Optional[List[Optional[str]]], db: Session
    ) -> None:
        """Set LLM configuration for a specific task (configuration now stored in Task table)"""
        logger.info(f"set_task_llms called for task {task_id} with llm_ids: {llm_ids}")
        # Configuration is now stored in Task table, this method is kept for backward compatibility
        # If AgentService already exists, update its LLM configuration
        if task_id in self._agents:
            # This method doesn't have user context, use None for user_id
            default_llm, fast_llm, vision_llm, compact_llm = resolve_llms_from_names(
                llm_ids, db, None
            )
            agent = self._agents[task_id]
            agent.llm = default_llm
            agent.fast_llm = fast_llm
            agent.vision_llm = vision_llm
            agent.compact_llm = compact_llm
            logger.info(
                f"Updated LLM configuration for existing AgentService task {task_id}: default={default_llm.model_name if default_llm else None}, compact={compact_llm.model_name if compact_llm else None}"
            )

    def set_task_memory_similarity_threshold(
        self, task_id: int, threshold: Optional[float]
    ) -> None:
        """Set memory similarity threshold for a specific task's agent"""
        if task_id in self._agents:
            agent = self._agents[task_id]
            agent.memory_similarity_threshold = threshold
            logger.info(
                f"Set memory similarity threshold for task {task_id}: {threshold}"
            )
        else:
            logger.warning(
                f"Cannot set memory similarity threshold for non-existent task {task_id}"
            )

    def _load_persisted_conversation_history(self, task_id: int, db: Session) -> None:
        """Hydrate an agent's chat transcript from persisted task chat messages."""
        agent = self._agents.get(task_id)
        if agent is None:
            return

        conversation_history = load_task_transcript(db, task_id)
        if not conversation_history:
            return

        agent.set_conversation_history(conversation_history)
        logger.info(
            f"Loaded {len(conversation_history)} persisted chat messages for task {task_id}"
        )

    async def _load_persisted_execution_context(
        self, task_id: int, db: Session
    ) -> None:
        """Hydrate an agent with persisted reusable execution context."""
        agent = self._agents.get(task_id)
        if agent is None:
            return

        recovery_state = await load_task_execution_recovery_state(db, task_id)
        execution_context_messages = recovery_state.get("messages", [])
        if not execution_context_messages:
            execution_context_messages = []

        agent.set_execution_context_messages(execution_context_messages)
        skill_context = recovery_state.get("skill_context")
        agent.set_recovered_skill_context(skill_context)
        logger.info(
            f"Loaded {len(execution_context_messages)} persisted execution context messages for task {task_id}"
        )
        if skill_context:
            logger.info(f"Loaded recovered skill context for task {task_id}")

    # NOTE: The legacy ``_load_agent_builder_config`` instance method
    # used to live here; its body became a one-line delegate to
    # ``llm_utils.load_agent_builder_config`` after the runtime-config
    # refactor and no production caller remained (the snapshot loader
    # and ``_resolve_task_runtime_config`` both call the module-level
    # helper directly). Removed to avoid a zero-value wrapper that
    # only existed as a test-mock surface; tests now patch
    # ``llm_utils.load_agent_builder_config`` directly.

    @staticmethod
    def _pick_default_llm_with_warning(
        default_llm: Optional[BaseLLM],
        *,
        task_id: int,
        has_agent_builder_config: bool,
        agent_id: Optional[int],
        saved_model_ids: Optional[dict],
        user_id: Optional[int],
        saved_model_descriptors: Optional[dict] = None,
    ) -> BaseLLM:
        """Return the default LLM and log a context-rich WARNING.

        Used when no per-task / per-agent LLM could be resolved (e.g. the
        agent's saved model is unavailable or the caller has no access).

        ``saved_model_descriptors`` (when provided) carries human-readable
        ``model_id`` / ``model_name`` per slot, which is more useful in logs
        than the bare ``DBModel.id`` pks recorded in ``saved_model_ids``.
        """
        if default_llm is None:
            if has_agent_builder_config:
                saved_models_for_log = saved_model_descriptors or saved_model_ids or {}
                logger.error(
                    "Agent builder model unavailable and no global default LLM is configured. "
                    "task_id=%s agent_id=%s agent_saved_models=%s user_id=%s",
                    task_id,
                    agent_id,
                    saved_models_for_log,
                    user_id,
                )
                raise HTTPException(
                    status_code=500,
                    detail=(
                        "Agent model configuration is unavailable and no global "
                        "default model is configured."
                    ),
                )
            logger.error(
                "Task %s has no valid LLM configuration and no default LLM", task_id
            )
            raise HTTPException(
                status_code=500,
                detail="No valid LLM configuration is available for this task.",
            )

        fallback_model = (
            getattr(default_llm, "model_name", None) or type(default_llm).__name__
        )
        if has_agent_builder_config:
            saved_models_for_log = saved_model_descriptors or saved_model_ids or {}
            logger.warning(
                "Agent builder model unavailable, falling back to default LLM. "
                "task_id=%s agent_id=%s agent_saved_models=%s user_id=%s fallback_model=%s",
                task_id,
                agent_id,
                saved_models_for_log,
                user_id,
                fallback_model,
            )
        else:
            logger.warning(
                "Task %s has no valid LLM configuration, using default LLM %s",
                task_id,
                fallback_model,
            )
        return default_llm

    @staticmethod
    def _merge_agent_builder_llms(
        baseline_llms: tuple[
            Optional[BaseLLM],
            Optional[BaseLLM],
            Optional[BaseLLM],
            Optional[BaseLLM],
        ],
        agent_llms: tuple[
            Optional[BaseLLM],
            Optional[BaseLLM],
            Optional[BaseLLM],
            Optional[BaseLLM],
        ],
    ) -> tuple[
        Optional[BaseLLM],
        Optional[BaseLLM],
        Optional[BaseLLM],
        Optional[BaseLLM],
    ]:
        """Overlay agent LLMs without discarding already resolved task LLMs."""
        return cast(
            tuple[
                Optional[BaseLLM],
                Optional[BaseLLM],
                Optional[BaseLLM],
                Optional[BaseLLM],
            ],
            tuple(
                agent_llm or baseline_llm
                for baseline_llm, agent_llm in zip(baseline_llms, agent_llms)
            ),
        )

    def _resolve_task_runtime_config(
        self,
        *,
        task_id: int,
        task: Task,
        db: Session,
        user: Optional[User],
    ) -> dict[str, Any]:
        """Resolve task / agent-builder LLMs and execution pattern.

        Thin main-loop wrapper around
        ``llm_utils.resolve_task_runtime_config_core``. Adds the
        diagnostic logging and the
        ``_pick_default_llm_with_warning`` fallback that the worker-
        thread snapshot loader cannot run (the fallback raises
        ``HTTPException``, which would propagate badly out of a
        thread).

        Used by ``_reconstruct_agent_from_history`` on the
        main-loop reconstruct path. The normal-creation path
        (``get_agent_for_task``) consumes a ``TaskSetupSnapshot``
        which goes through the same core helper off-loop.
        """
        from ..services.llm_utils import resolve_task_runtime_config_core

        logger.info(
            "Task %s record: agent_type=%s, model_name=%s, compact_model_name=%s",
            task_id,
            task.agent_type,
            task.model_name,
            task.compact_model_name,
        )

        user_id_for_resolution: Optional[int] = (
            int(user.id)
            if user and user.id is not None
            else int(task.user_id)
            if task.user_id is not None
            else None
        )
        core = resolve_task_runtime_config_core(
            task, db, user_id=user_id_for_resolution
        )

        task_llm, task_fast_llm, task_vision_llm, task_compact_llm = core.llms

        logger.info(
            "Task %s execution_mode=%s -> pattern=%s",
            task_id,
            getattr(task, "execution_mode", None),
            core.task_pattern,
        )
        if core.agent_fields is not None:
            logger.info(
                "Task %s using Agent Builder config: %s",
                task_id,
                core.agent_fields.name,
            )
            if core.workforce is not None:
                # Workforce task keeps its own execution_mode rather
                # than inheriting from agent_config; surface that so
                # on-call doesn't confuse it with the legacy override.
                logger.info(
                    "Workforce task %s keeping task execution mode -> pattern=%s",
                    task_id,
                    core.task_pattern,
                )
            else:
                logger.info(
                    "Task %s using Agent Builder execution mode: %s -> pattern=%s",
                    task_id,
                    (core.agent_config or {}).get("execution_mode"),
                    core.task_pattern,
                )
        elif core.agent_config is not None:
            # Inline agent_config path (build-preview tasks routed
            # through normal task flow with config embedded in the row).
            logger.info(
                "Task %s using inline Agent Builder config: execution_mode=%s -> pattern=%s",
                task_id,
                core.agent_config.get("execution_mode"),
                core.task_pattern,
            )

        if not task_llm:
            task_llm = self._pick_default_llm_with_warning(
                self._default_llm,
                task_id=task_id,
                has_agent_builder_config=core.has_agent_builder_config,
                agent_id=getattr(task, "agent_id", None),
                saved_model_ids=(core.agent_config or {}).get("saved_model_ids"),
                saved_model_descriptors=(core.agent_config or {}).get(
                    "saved_model_descriptors"
                ),
                user_id=user_id_for_resolution,
            )

        logger.info(
            "Successfully loaded LLM configuration for task %s: compact_llm=%s",
            task_id,
            task_compact_llm.model_name if task_compact_llm else None,
        )
        return {
            "agent_config": core.agent_config,
            "task_llm": task_llm,
            "task_fast_llm": task_fast_llm,
            "task_vision_llm": task_vision_llm,
            "task_compact_llm": task_compact_llm,
            "task_pattern": core.task_pattern,
            "has_agent_builder_config": core.has_agent_builder_config,
        }

    def _load_task_inline_agent_config(self, task: Task) -> Optional[dict[str, Any]]:
        if not isinstance(task.agent_config, dict):
            return None

        inline_config = task.agent_config
        if not any(
            key in inline_config
            for key in ("instructions", "knowledge_bases", "skills", "tool_categories")
        ):
            return None

        return {
            "llms": (None, None, None, None),
            "execution_mode": getattr(task, "execution_mode", None) or "balanced",
            "instructions": inline_config.get("instructions"),
            "skills": inline_config.get("skills") or [],
            "knowledge_bases": inline_config.get("knowledge_bases") or [],
            "tool_categories": inline_config.get("tool_categories"),
            "memory_similarity_threshold": inline_config.get(
                "memory_similarity_threshold"
            ),
            "is_preview": inline_config.get("is_preview"),
            "preview_agent_id": inline_config.get("preview_agent_id"),
        }

    async def _build_tools_for_task(
        self,
        *,
        task_id: int,
        task: Task,
        db: Session,
        user: User,
        agent_config: Optional[dict],
        task_llm: Optional[BaseLLM],
        task_vision_llm: Optional[BaseLLM],
        parent_tracer: Optional[Any] = None,
        scope: Optional[ExecutionScope] = None,
    ) -> tuple[list[Any], Any]:
        """Build the tool set configured for a web task."""
        workforce_runtime = resolve_workforce_task_runtime(db, task)
        excluded_agent_id = None
        current_agent = _load_agent_for_task_runtime(db, task, workforce_runtime)
        if current_agent and _is_published_agent(current_agent):
            excluded_agent_id = int(current_agent.id)
            logger.info(
                f"Task {task_id} is associated with published agent "
                f"{current_agent.id} ({current_agent.name}), will exclude from "
                "agent tools"
            )
        elif agent_config and agent_config.get("preview_agent_id"):
            preview_user_id = int(task.user_id)
            current_agent = (
                db.query(Agent)
                .filter(
                    Agent.id == agent_config["preview_agent_id"],
                    owned_agent_clause(
                        preview_user_id, get_agent_team_scope(db, preview_user_id)
                    ),
                )
                .first()
            )
            if current_agent and current_agent.status == AgentStatus.PUBLISHED:
                excluded_agent_id = int(current_agent.id)
                logger.info(
                    f"Preview task {task_id} is for published agent "
                    f"{current_agent.id} ({current_agent.name}), will exclude from "
                    "agent tools"
                )

        workforce_runtime = resolve_workforce_task_runtime(db, task)
        tool_selection_spec = _build_tool_selection_spec_for_task(
            agent_config, workforce_runtime, task_id=task_id
        )
        workspace_owner_id = int(task.user_id)
        scope_segments = scope.workspace_segments if scope is not None else ()
        # The sandbox bind mount covers the scope's mount prefix (the full
        # workspace_segments when no prefix is declared); the deeper
        # workspace subtree lives inside it but is NOT isolated from a
        # co-mounted sibling — the rw mount and the code-execution tools
        # bypass scoped_user_root, so a mount prefix must only group scopes
        # of the same trust principal (see ExecutionScope.sandbox_mount_segments).
        mount_segments = scope.effective_mount_segments if scope is not None else ()
        sandbox_workspace_config = {
            "base_dir": str(
                scoped_user_root(get_uploads_dir(), workspace_owner_id, mount_segments)
            ),
            "task_id": f"web_task_{task_id}",
            "user_id": workspace_owner_id,
            "allowed_external_dirs": _build_allowed_external_dirs(
                workspace_owner_id, scope=scope
            ),
            "scope_segments": scope_segments,
        }

        # Sandbox startup is container/network work that can take seconds;
        # don't hold this session's read transaction (and its pool slot)
        # across it (issue #889).
        release_db_connection_if_clean(db)
        sandbox = await self._get_or_create_task_sandbox(
            task_id=task_id,
            workspace_owner_id=workspace_owner_id,
            workspace_config=sandbox_workspace_config,
            scope=scope,
        )

        return await create_default_tools(
            db,
            request=self.request,
            user=user,
            task_id=f"web_task_{task_id}",
            workspace_owner_id=int(task.user_id),
            scope=scope,
            allowed_collections=agent_config["knowledge_bases"]
            if agent_config
            else None,
            allowed_skills=agent_config["skills"] if agent_config else None,
            tool_selection_spec=tool_selection_spec,
            excluded_agent_id=excluded_agent_id,
            vision_model=task_vision_llm,
            sandbox=sandbox,
            llm=task_llm,
            allowed_agent_ids=workforce_runtime.allowed_agent_ids
            if workforce_runtime
            else None,
            agent_tool_overrides=workforce_runtime.agent_tool_overrides
            if workforce_runtime
            else None,
            enable_global_agent_tools=workforce_runtime.enable_global_agent_tools
            if workforce_runtime
            else True,
            allow_cross_user_agent_ids=workforce_runtime.allow_cross_user_agent_ids
            if workforce_runtime
            else False,
            parent_task_id=str(task_id) if workforce_runtime else None,
            parent_tracer=parent_tracer if workforce_runtime else None,
            agent_call_stack=workforce_runtime.agent_call_stack
            if workforce_runtime
            else None,
            connector_runtime_turn_id=None,
            mcp_failure_policy=_mcp_failure_policy_for_task_source(task.source),
            mcp_load_summary_tracer=parent_tracer,
            mcp_load_summary_trace_task_id=str(task_id),
        )

    async def get_agent_for_task(
        self,
        task_id: int,
        db: Optional[Session] = None,
        user: Optional[User] = None,
        task_setup_snapshot: Optional[TaskSetupSnapshot] = None,
        task_owner_user_id: Optional[int] = None,
        connector_runtime_turn_id: Optional[str] = None,
    ) -> AgentService:
        lock = self._agent_build_locks.get(task_id)
        if lock is None:
            lock = asyncio.Lock()
            self._agent_build_locks[task_id] = lock
        async with lock:
            return await self._get_agent_for_task_unlocked(
                task_id,
                db=db,
                user=user,
                task_setup_snapshot=task_setup_snapshot,
                task_owner_user_id=task_owner_user_id,
                connector_runtime_turn_id=connector_runtime_turn_id,
            )

    async def _get_agent_for_task_unlocked(
        self,
        task_id: int,
        db: Optional[Session] = None,
        user: Optional[User] = None,
        task_setup_snapshot: Optional[TaskSetupSnapshot] = None,
        task_owner_user_id: Optional[int] = None,
        connector_runtime_turn_id: Optional[str] = None,
    ) -> AgentService:
        """Get or create AgentService instance for specific task.

        ``task_setup_snapshot`` is an off-loop snapshot loaded by the
        upstream caller (``_schedule_bg._runner``). When provided, the
        in-method ``asyncio.to_thread(load_task_setup_snapshot_sync,
        ...)`` is skipped -- the snapshot is reused directly. WS
        callers and any caller that hasn't adopted the snapshot
        plumbing pass ``None`` and the Step-3 in-method thread call
        runs as before.

        ``task_owner_user_id`` is the task OWNER's id — the runtime identity
        the agent runs as (models, tools, OAuth, UserContext). It differs from
        the acting principal (``user``) when an admin operates on another
        user's task; callers that loaded/authorized the task should pass it.
        When omitted it falls back to the snapshot owner, then the task row's
        owner, then ``user.id``.
        """
        # Resolve the runtime identity (OWNER). Everything below — snapshot
        # load, model resolution, tool config, UserContext — runs as this
        # identity, never the acting principal. Precedence: explicit owner →
        # snapshot owner → the task row's owner (authoritative) → the acting
        # ``user`` as a last resort. Deriving the owner from the task row means
        # callers that pass only the acting ``user`` (e.g. WS resume / execute
        # / pause handlers, which authorize the task separately) still run as
        # the owner, not an admin acting on someone else's task.
        runtime_user_id: Optional[int] = task_owner_user_id
        if runtime_user_id is None and task_setup_snapshot is not None:
            runtime_user_id = int(task_setup_snapshot.task.user_id)
        if runtime_user_id is None and db is not None:
            try:
                owner_row = db.query(Task.user_id).filter(Task.id == task_id).first()
                if owner_row is not None and owner_row[0] is not None:
                    runtime_user_id = int(owner_row[0])
            except Exception:
                runtime_user_id = None
        if runtime_user_id is None and user is not None and user.id is not None:
            runtime_user_id = int(user.id)
        # The User object the runtime needs (tool overrides / tracer). Reuse the
        # passed ``user`` when it already IS the owner; otherwise load the owner.
        runtime_user: Optional[User] = user
        if (
            runtime_user_id is not None
            and db is not None
            and (user is None or user.id is None or int(user.id) != runtime_user_id)
        ):
            runtime_user = db.query(User).filter(User.id == runtime_user_id).first()

        # One turn resolves once: an activated turn (the orchestrated
        # execute/resume paths) already resolved this task's scope and
        # carries it in the contextvar, so prefer that over a fresh
        # loader/resolver round-trip. A None contextvar is ambiguous —
        # "explicitly unscoped turn" and "not inside an activated turn"
        # (pause/resume handlers, channels) look the same — so None falls
        # back to resolving per call, at the same place the owner is
        # resolved. Correctness is identical either way: the resolver is
        # idempotent per task and every activation site wraps operations
        # on the task it activated for.
        scope = get_execution_scope()
        if scope is None:
            scope = resolve_execution_scope(task_id)
        fingerprint = scope_fingerprint(scope)

        # Owner invariant: evict a cached AgentService built for a different
        # owner so we rebuild it under the correct runtime identity instead of
        # silently reusing the wrong one.
        if (
            task_id in self._agents
            and self._agent_owner_ids.get(task_id) != runtime_user_id
        ):
            logger.warning(
                "Evicting cached AgentService for task %s: built for owner %s, requested %s",
                task_id,
                self._agent_owner_ids.get(task_id),
                runtime_user_id,
            )
            # The evicted instance was built for the wrong owner; its workspace
            # lives under that owner's user-scoped path
            # (``user_{owner}/web_task_{task_id}``), a different directory from
            # the correct owner's. Clean it up so no wrong-owner workspace
            # residue is left behind -- this cannot touch the rebuilt owner's
            # workspace. Eviction itself must proceed even if cleanup fails.
            try:
                self._agents[task_id].cleanup_workspace()
            except Exception as e:
                logger.warning(
                    "Failed to clean up workspace while evicting wrong-owner "
                    "AgentService for task %s: %s",
                    task_id,
                    e,
                )
            del self._agents[task_id]
            self._agent_owner_ids.pop(task_id, None)
            self._agent_sandbox_keys.pop(task_id, None)
            self._agent_scope_fingerprints.pop(task_id, None)

        # Scope invariant: the cached instance baked its sandbox key (and,
        # later, workspace paths and memory dimensions) in at build time.
        # If the embedder reassigned the task to a different scope between
        # turns, reusing the instance would silently execute in the old
        # scope's namespace — evict and rebuild instead. The workspace is
        # NOT cleaned up here: same owner, and the old scope's data must
        # survive a scope reassignment.
        if (
            task_id in self._agents
            and self._agent_scope_fingerprints.get(task_id) != fingerprint
        ):
            evicted_fingerprint = self._agent_scope_fingerprints.get(task_id)
            recently_evicted = self._agent_evicted_scope_fingerprints.setdefault(
                task_id, deque(maxlen=_EVICTED_FINGERPRINT_MEMORY)
            )
            if fingerprint in recently_evicted:
                logger.warning(
                    "Execution scope for task %s cycled back to recently "
                    "evicted fingerprint %s (now evicting %s): probable "
                    "resolver bug — a resolver cycling between scopes "
                    "rebuilds the agent every turn and defeats the per-task "
                    "cache.",
                    task_id,
                    fingerprint,
                    evicted_fingerprint,
                )
            else:
                logger.warning(
                    "Evicting cached AgentService for task %s: built under "
                    "scope fingerprint %s, resolved %s",
                    task_id,
                    evicted_fingerprint,
                    fingerprint,
                )
            recently_evicted.append(evicted_fingerprint)
            del self._agents[task_id]
            self._agent_owner_ids.pop(task_id, None)
            self._agent_sandbox_keys.pop(task_id, None)
            self._agent_scope_fingerprints.pop(task_id, None)

        if task_id not in self._agents:
            # Check if task exists in database
            task_exists = False
            # ``task`` is widened to ``Task | _TaskFields | None`` because
            # the LLM-config block below rebinds it from an ORM ``Task``
            # to a frozen ``_TaskFields`` once the snapshot lands.
            # Downstream consumers only read primitive attributes
            # (``user_id``, ``agent_id``, ``agent_config``, ``status``)
            # which both types expose identically.
            task: Any = None
            if db is not None:
                try:
                    task = db.query(Task).filter(Task.id == task_id).first()
                    task_exists = task is not None
                except Exception as e:
                    logger.warning(
                        f"Failed to check task existence for task {task_id}: {e}"
                    )
                    task_exists = False
                    task = None

            if not task_exists:
                # Create new task record if it doesn't exist
                if db is not None and user is not None:
                    try:
                        new_task = Task(
                            user_id=user.id,  # Use actual user ID
                            title=f"Task {task_id}",
                            description="Auto-created task",
                            status=TaskStatus.PENDING,
                            connector_runtime_selected_refs=[],
                        )
                        db.add(new_task)
                        db.commit()
                        db.refresh(new_task)
                        logger.info(
                            f"Created new task record for task {task_id} with user_id={user.id}"
                        )
                    except Exception as e:
                        logger.error(
                            f"Failed to create task record for task {task_id}: {e}"
                        )
            else:
                should_reconstruct = task is not None and task.status in [
                    TaskStatus.RUNNING,
                    TaskStatus.PAUSED,
                    TaskStatus.WAITING_FOR_USER,
                ]
                # Brand-new SDK task pre-check: ``begin_turn`` flips the
                # task to RUNNING before this code runs, but a freshly
                # created task has no trace events or DAG plan to
                # recover -- ``_reconstruct_agent_from_history`` would
                # run two queries, find nothing, and log a misleading
                # "Failed to reconstruct" warning. Short-circuit the
                # wasted path here. ``PAUSED`` / ``WAITING_FOR_USER``
                # always have prior state by definition; only gate on
                # ``RUNNING``.
                if (
                    should_reconstruct
                    and task is not None
                    and task.status == TaskStatus.RUNNING
                    and db is not None
                    and not self._has_reconstructable_history(task_id, db)
                ):
                    logger.info(
                        f"Task {task_id} is RUNNING but has no reconstructable "
                        "history (no trace events, no DAG plan); skipping "
                        "reconstruct and going to normal creation."
                    )
                    should_reconstruct = False
                # Task exists in database, try to reconstruct from history only for active executions
                if db is not None and should_reconstruct:
                    try:
                        await self._reconstruct_agent_from_history(
                            task_id, db, scope=scope
                        )
                        self._load_persisted_conversation_history(task_id, db)
                        await self._load_persisted_execution_context(task_id, db)
                        self._agent_owner_ids[task_id] = runtime_user_id
                        self._agent_scope_fingerprints[task_id] = fingerprint
                        self._sync_connector_runtime_turn(
                            task_id, connector_runtime_turn_id
                        )
                        return self._agents[task_id]
                    except HTTPException:
                        raise
                    except RequiredMCPUnavailableError:
                        self._agents.pop(task_id, None)
                        self._agent_owner_ids.pop(task_id, None)
                        self._agent_sandbox_keys.pop(task_id, None)
                        self._agent_scope_fingerprints.pop(task_id, None)
                        raise
                    except Exception as e:
                        logger.warning(
                            f"Failed to reconstruct agent from history for task {task_id}: {e}"
                        )
                        # Clean up any partial reconstruction that might have occurred
                        if task_id in self._agents:
                            logger.info(
                                f"Cleaning up partially reconstructed agent for task {task_id}"
                            )
                            del self._agents[task_id]
                            self._agent_owner_ids.pop(task_id, None)
                            self._agent_sandbox_keys.pop(task_id, None)
                            self._agent_scope_fingerprints.pop(task_id, None)
                        # Continue with normal agent creation

            # Create tracer with all necessary handlers
            tracer = create_task_tracer(task_id, runtime_user)

            # Load the contiguous synchronous DB block (Task row,
            # per-task LLM resolution, optional Agent Builder lookup
            # with its 0-4 ``DBModel`` queries and 0-4 user-aware LLM
            # access checks) on a worker thread so the main event
            # loop stays responsive. Same set of reads the inline
            # code used to do; see ``load_task_setup_snapshot_sync``
            # for the strict no-ORM-leak invariant.
            logger.info(f"Loading LLM configuration for task {task_id} from database")
            agent_config: Optional[dict] = None
            task_pattern = "dag_plan_execute"
            use_dag = True  # Default to DAG pattern (for backward compatibility)
            excluded_agent_id: Optional[int] = None
            snapshot: Optional[TaskSetupSnapshot] = None
            try:
                if db is None:
                    raise ValueError("Database session is required")

                if task_setup_snapshot is not None:
                    # Caller already loaded the snapshot off-loop
                    # (typically ``_schedule_bg._runner``). Reuse it
                    # instead of re-spinning a worker thread. The loader's
                    # owner guard is bypassed on this branch, so re-assert it
                    # here: the snapshot's owner must match the runtime owner,
                    # or tools / model / UserContext would split from the
                    # workspace owner.
                    if (
                        runtime_user_id is not None
                        and int(task_setup_snapshot.task.user_id) != runtime_user_id
                    ):
                        raise TaskOwnerMismatchError(
                            task_id,
                            runtime_user_id,
                            int(task_setup_snapshot.task.user_id),
                        )
                    snapshot = task_setup_snapshot
                else:
                    snapshot = await asyncio.to_thread(
                        load_task_setup_snapshot_sync,
                        task_id,
                        runtime_user_id,
                    )

                if snapshot is not None:
                    task = snapshot.task
                    logger.info(
                        f"Task {task_id} record: agent_type={task.agent_type}, "
                        f"model_name={task.model_name}, "
                        f"compact_model_name={task.compact_model_name}"
                    )
                    task_pattern = snapshot.task_pattern
                    logger.info(
                        f"Task {task_id} execution_mode={task.execution_mode} "
                        f"-> pattern={task_pattern}"
                    )
                    task_llm = snapshot.task_llm
                    task_fast_llm = snapshot.task_fast_llm
                    task_vision_llm = snapshot.task_vision_llm
                    task_compact_llm = snapshot.task_compact_llm
                    agent_config = snapshot.agent_config
                    excluded_agent_id = snapshot.excluded_agent_id

                    if snapshot.agent is not None:
                        logger.info(
                            f"Task {task_id} using Agent Builder config: {snapshot.agent.name}"
                        )
                        if agent_config is not None:
                            logger.info(
                                f"Task {task_id} using Agent Builder execution "
                                f"mode: {agent_config.get('execution_mode')} "
                                f"-> pattern={task_pattern}"
                            )

                    if not task_llm:
                        # Two failure modes, two different policies:
                        #
                        # 1. Agent-builder agent whose configured models
                        #    can't be resolved → fail-fast via the
                        #    shared diagnostic helper. This is a real
                        #    configuration error (the agent row points
                        #    at models the runtime can't load); the
                        #    helper raises HTTPException(500) with
                        #    saved-model metadata in the log so
                        #    on-call can trace back to the agent row.
                        #
                        # 2. Plain task with no agent-builder layer and
                        #    no resolvable LLM (e.g. the deployment
                        #    runs with no LLM env keys at all, or the
                        #    task is tool-only and never calls an
                        #    LLM) → silent fallback to ``self._default_llm``
                        #    even if that itself is None. Some tasks
                        #    legitimately never invoke an LLM; we
                        #    cannot turn that case into a 500 without
                        #    breaking those callers.
                        if snapshot.agent is not None:
                            user_id_for_fallback: Optional[int] = (
                                runtime_user_id
                                if runtime_user_id is not None
                                else task.user_id
                            )
                            task_llm = self._pick_default_llm_with_warning(
                                self._default_llm,
                                task_id=task_id,
                                has_agent_builder_config=True,
                                agent_id=task.agent_id,
                                saved_model_ids=(agent_config or {}).get(
                                    "saved_model_ids"
                                ),
                                saved_model_descriptors=(agent_config or {}).get(
                                    "saved_model_descriptors"
                                ),
                                user_id=user_id_for_fallback,
                            )
                        else:
                            logger.warning(
                                f"Task {task_id} has no valid LLM configuration; "
                                "using default LLM (may be None for tool-only tasks)"
                            )
                            task_llm = self._default_llm

                    logger.info(
                        f"Successfully loaded LLM configuration for task {task_id}: "
                        f"compact_llm="
                        f"{task_compact_llm.model_name if task_compact_llm else None}"
                    )
                else:
                    # Task row vanished between the existence check and
                    # the snapshot read. Fall back to the original
                    # defaults so we still produce a usable AgentService.
                    logger.error(f"Task {task_id} not found in database!")
                    task_llm = self._default_llm
                    task_fast_llm = None
                    task_vision_llm = None
                    task_compact_llm = None
            except (HTTPException, TaskOwnerMismatchError):
                # Owner mismatch is an identity/authorization fault, not a
                # recoverable "LLM config failed to load" -- it must not fall
                # through to the default-LLM path, which would build the
                # runtime as the wrong user. Propagate it.
                raise
            except Exception as e:
                logger.error(
                    f"Failed to load LLM configuration from task {task_id} database: {e}"
                )
                task_llm = self._default_llm
                task_fast_llm = None
                task_vision_llm = None
                task_compact_llm = None
            llm_info = "database LLM configuration"

            try:
                # Runtime creation depends on the OWNER identity, not the
                # acting principal -- guard on the resolved runtime user.
                if runtime_user is None:
                    raise ValueError(
                        "Task owner / runtime user is required for agent creation"
                    )

                if not db:
                    raise ValueError(
                        "Database connection is required for agent creation"
                    )

                # ``excluded_agent_id`` for the legacy task-agent
                # (published-agent) case is pre-computed by the snapshot
                # loader (same SELECT as LLM resolution). Surface the log
                # line here so on-call still sees the exclusion in
                # production logs.
                if (
                    excluded_agent_id is not None
                    and snapshot is not None
                    and snapshot.agent is not None
                ):
                    logger.info(
                        f"Task {task_id} is associated with published agent "
                        f"{snapshot.agent.id} ({snapshot.agent.name}), "
                        "will exclude from agent tools"
                    )

                # Inline preview_agent_id case (#459 build-preview): the
                # snapshot path doesn't cover this because it resolves
                # excluded_agent_id only through ``task.agent_id``; the
                # preview agent is referenced inside the inline
                # ``agent_config`` dict on tasks with ``agent_id=None``.
                # Run this in addition to (not instead of) the snapshot
                # value above -- they're mutually exclusive by design
                # (either task has an agent_id OR has inline preview).
                if (
                    excluded_agent_id is None
                    and agent_config
                    and agent_config.get("preview_agent_id")
                ):
                    from ..models.agent import AgentStatus

                    if task is None:
                        raise ValueError(
                            f"Task {task_id} missing while resolving preview agent"
                        )
                    preview_user_id = int(task.user_id)
                    current_agent = (
                        db.query(Agent)
                        .filter(
                            Agent.id == agent_config["preview_agent_id"],
                            owned_agent_clause(
                                preview_user_id,
                                get_agent_team_scope(db, preview_user_id),
                            ),
                        )
                        .first()
                    )
                    if current_agent and current_agent.status == AgentStatus.PUBLISHED:
                        excluded_agent_id = int(current_agent.id)
                        logger.info(
                            f"Preview task {task_id} is for published agent {current_agent.id} ({current_agent.name}), will exclude from agent tools"
                        )

                workforce_runtime = (
                    resolve_workforce_task_runtime(db, task) if task else None
                )
                workspace_owner_id = (
                    int(task.user_id)
                    if task and task.user_id is not None
                    else int(runtime_user.id)
                )
                scope_segments = scope.workspace_segments if scope is not None else ()
                # Sandbox mount covers the scope's mount prefix (full
                # workspace_segments when no prefix is declared); the deeper
                # subtree is NOT isolated from a co-mounted sibling (rw mount,
                # code-execution tools bypass scoped_user_root), so a mount
                # prefix must only group scopes of the same trust principal
                # (see ExecutionScope.sandbox_mount_segments).
                mount_segments = (
                    scope.effective_mount_segments if scope is not None else ()
                )
                sandbox_workspace_config = {
                    "base_dir": str(
                        scoped_user_root(
                            get_uploads_dir(), workspace_owner_id, mount_segments
                        )
                    ),
                    "task_id": f"web_task_{task_id}",
                    "user_id": workspace_owner_id,
                    "allowed_external_dirs": _build_allowed_external_dirs(
                        workspace_owner_id, scope=scope
                    ),
                    "scope_segments": scope_segments,
                }

                # Sandbox startup is container/network work that can take
                # seconds; don't hold this session's read transaction (and
                # its pool slot) across it (issue #889).
                release_db_connection_if_clean(db)
                sandbox = await self._get_or_create_task_sandbox(
                    task_id=task_id,
                    workspace_owner_id=workspace_owner_id,
                    workspace_config=sandbox_workspace_config,
                    scope=scope,
                )

                tool_selection_spec = _build_tool_selection_spec_for_task(
                    agent_config, workforce_runtime, task_id=task_id
                )

                tools = await create_default_tools(
                    db,
                    request=self.request,
                    # Owner, not the acting principal: tool config (models,
                    # OAuth, KB/MCP/SQL visibility, is_admin scope) must be the
                    # task owner's. The is_admin tri-state in WebToolConfig keeps
                    # ``request``'s admin from widening this back.
                    user=runtime_user,
                    task_id=f"web_task_{task_id}",
                    workspace_owner_id=workspace_owner_id,
                    scope=scope,
                    allowed_collections=agent_config["knowledge_bases"]
                    if agent_config
                    else None,
                    allowed_skills=agent_config["skills"] if agent_config else None,
                    tool_selection_spec=tool_selection_spec,
                    excluded_agent_id=excluded_agent_id,
                    vision_model=task_vision_llm,  # Pass task-specific vision model
                    sandbox=sandbox,
                    llm=task_llm,  # Pass task-specific LLM
                    allowed_agent_ids=workforce_runtime.allowed_agent_ids
                    if workforce_runtime
                    else None,
                    agent_tool_overrides=workforce_runtime.agent_tool_overrides
                    if workforce_runtime
                    else None,
                    enable_global_agent_tools=workforce_runtime.enable_global_agent_tools
                    if workforce_runtime
                    else True,
                    allow_cross_user_agent_ids=workforce_runtime.allow_cross_user_agent_ids
                    if workforce_runtime
                    else False,
                    parent_task_id=str(task_id) if workforce_runtime else None,
                    parent_tracer=tracer if workforce_runtime else None,
                    agent_call_stack=workforce_runtime.agent_call_stack
                    if workforce_runtime
                    else None,
                    connector_runtime_turn_id=connector_runtime_turn_id,
                    mcp_failure_policy=_mcp_failure_policy_for_task_source(
                        task.source if task is not None else None
                    ),
                    mcp_load_summary_tracer=tracer,
                    mcp_load_summary_trace_task_id=str(task_id),
                )

                with UserContext(runtime_user_id):
                    # Unpack tools and tool_config from create_default_tools
                    tools_list, tool_config = tools

                    # Get system prompt from agent config (if available)
                    from .agents import enhance_system_prompt_with_kb

                    system_prompt = (
                        agent_config.get("instructions") if agent_config else None
                    )
                    kb_list = (
                        agent_config.get("knowledge_bases") if agent_config else None
                    )
                    system_prompt = enhance_system_prompt_with_kb(
                        system_prompt, kb_list
                    )
                    system_prompt = _build_workforce_system_prompt(
                        system_prompt, workforce_runtime
                    )

                    # Extract memory similarity threshold from agent config
                    memory_similarity_threshold = None
                    if agent_config and "memory_similarity_threshold" in agent_config:
                        memory_similarity_threshold = agent_config[
                            "memory_similarity_threshold"
                        ]
                    memory_policy = resolve_agent_service_memory_policy(
                        task=task,
                        agent_config=agent_config,
                    )

                    # Build allowed external directories for the task owner's uploads.
                    allowed_external_dirs = _build_allowed_external_dirs(
                        workspace_owner_id,
                        scope=scope,
                    )

                    # Create AgentService first (this creates the workspace)
                    self._agents[task_id] = AgentService(
                        name=f"web_chat_agent_task_{task_id}",
                        id=f"web_task_{task_id}",  # Use task ID only for workspace
                        llm=task_llm,
                        fast_llm=task_fast_llm,
                        vision_llm=task_vision_llm,
                        compact_llm=task_compact_llm,
                        tools=tools_list,
                        tool_config=tool_config,  # Pass tool_config for proper multi-tenancy
                        memory=memory_policy.memory,
                        pattern=task_pattern,  # Use pattern instead of use_dag_pattern
                        tracer=tracer,
                        enable_workspace=True,  # Enable workspace functionality
                        workspace_base_dir=str(
                            scoped_user_root(
                                get_uploads_dir(), workspace_owner_id, scope_segments
                            )
                        ),  # Use user- (and scope-) isolated base directory
                        allowed_external_dirs=allowed_external_dirs,  # Add allowed external directories
                        scope_segments=scope_segments,
                        task_id=str(task_id),  # Pass task_id for proper tracing
                        memory_similarity_threshold=memory_similarity_threshold,  # Set from task config
                        memory_enabled=memory_policy.memory_enabled,
                        system_prompt=system_prompt,  # Pass agent builder instructions
                    )

                    selected_file_ids: list[str] = []
                    if task and isinstance(task.agent_config, dict):
                        raw_selected_file_ids = task.agent_config.get(
                            "selected_file_ids"
                        )
                        if isinstance(raw_selected_file_ids, list):
                            selected_file_ids = [
                                str(item)
                                for item in raw_selected_file_ids
                                if isinstance(item, str) and item.strip()
                            ]

                    workspace = self._agents[task_id].workspace
                    if selected_file_ids and workspace is not None:
                        from ..models.uploaded_file import UploadedFile

                        for selected_file_id in selected_file_ids:
                            task_owner_id = (
                                int(task.user_id) if task else int(runtime_user.id)
                            )
                            uploaded_file = (
                                db.query(UploadedFile)
                                .filter(
                                    UploadedFile.file_id == selected_file_id,
                                    UploadedFile.user_id == task_owner_id,
                                    or_(
                                        UploadedFile.task_id == int(task_id),
                                        UploadedFile.task_id.is_(None),
                                    ),
                                )
                                .first()
                            )
                            if uploaded_file is None:
                                continue

                            source_path = ensure_uploaded_file_local_path(uploaded_file)
                            if not source_path.exists() or not source_path.is_file():
                                continue

                            if uploaded_file.task_id is None:
                                uploaded_file.task_id = int(task_id)
                                db.flush()

                            # Use the source file directly (user's upload directory) instead of copying
                            # This avoids duplicate files across the system.
                            # Resolve to an absolute path so Workspace.register_file
                            # doesn't try to interpret it as workspace-relative.
                            workspace.register_file(
                                str(source_path.resolve()), file_id=selected_file_id
                            )

                pattern_info = (
                    f"with DAG pattern and workspace using {llm_info}"
                    if use_dag
                    else "with workspace (no LLM configured)"
                )
                logger.info(
                    f"Created new AgentService for task {task_id} {pattern_info}"
                )

                if task_exists and db is not None:
                    self._load_persisted_conversation_history(task_id, db)
                    await self._load_persisted_execution_context(task_id, db)

            except Exception as e:
                logger.error(f"Failed to create AgentService for task {task_id}: {e}")
                # Re-raise the exception - no fallback logic allowed
                raise

        self._agent_owner_ids[task_id] = runtime_user_id
        self._agent_scope_fingerprints[task_id] = fingerprint
        self._sync_connector_runtime_turn(task_id, connector_runtime_turn_id)
        return self._agents[task_id]

    def _sync_connector_runtime_turn(
        self, task_id: int, connector_runtime_turn_id: Optional[str]
    ) -> None:
        if not connector_runtime_turn_id:
            logger.debug(
                "Skipping connector runtime turn sync for task %s: no turn id",
                task_id,
            )
            return

        agent = self._agents.get(task_id)
        if agent is None:
            logger.debug(
                "Skipping connector runtime turn sync for task %s turn %s: agent is not cached",
                task_id,
                connector_runtime_turn_id,
            )
            return

        tool_config = agent.tool_config
        if tool_config is None:
            logger.debug(
                "Skipping connector runtime turn sync for task %s turn %s: "
                "agent has no tool config",
                task_id,
                connector_runtime_turn_id,
            )
            return

        if tool_config.set_connector_runtime_turn_id(connector_runtime_turn_id):
            logger.info(
                "Refreshing connector runtime tools for task %s turn %s",
                task_id,
                connector_runtime_turn_id,
            )
            agent.invalidate_tools()
        else:
            logger.debug(
                "Connector runtime tools already use task %s turn %s",
                task_id,
                connector_runtime_turn_id,
            )

    def remove_agent(self, task_id: int, user_id: Optional[int] = None) -> None:
        """Remove AgentService instance for completed task"""
        if task_id in self._agents:
            # Log workspace path before cleanup
            workspace = self._agents[task_id].workspace
            if workspace is not None:
                workspace_path = str(workspace.workspace_dir)
            else:
                workspace_path = None
            if workspace_path:
                logger.info(
                    f"Deleting workspace path for task {task_id}: {workspace_path}"
                )

            # Clean up workspace before removing agent
            self._agents[task_id].cleanup_workspace()
            logger.info(f"Cleaned up workspace for task {task_id}")

            del self._agents[task_id]
            self._agent_owner_ids.pop(task_id, None)
            self._agent_sandbox_keys.pop(task_id, None)
            self._agent_scope_fingerprints.pop(task_id, None)
            self._agent_evicted_scope_fingerprints.pop(task_id, None)
            logger.info(f"Removed AgentService for task {task_id}")
        else:
            # If agent is not in memory, clean up workspace directory directly
            self._cleanup_workspace_directory(task_id, user_id)

        # Do not replace a lock held by an in-flight builder: a fresh lock would
        # let a second caller bypass single-flight and race the existing build.
        build_lock = self._agent_build_locks.get(task_id)
        if build_lock is not None and not build_lock.locked():
            self._agent_build_locks.pop(task_id, None)

        # LLM configuration is now stored in Task table, no need to clean up memory storage

    async def execute_task(
        self,
        agent_service: "AgentService",
        task: str,
        context: Optional[Dict[str, Any]] = None,
        task_id: Optional[str] = None,
        tracking_task_id: Optional[str] = None,
        db_session: Optional[Any] = None,
        *,
        manage_task_lease: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute task with automatic token tracking.

        This method wraps the agent's execute_task with token tracking if db_session is provided.

        Args:
            agent_service: The AgentService instance to use
            task: Task description
            context: Optional context data
            task_id: Optional task identifier passed to agent execution
            tracking_task_id: Optional task identifier used only for token tracking
            db_session: Optional database session for token tracking
            manage_task_lease: When False, skip acquire/release/heartbeat here.
                ``TaskTurnOrchestrator._schedule_bg`` already owns the lease
                lifecycle for REST/SDK turns; a nested acquire can return
                ``running_elsewhere`` or release the row before
                ``execute_task_background`` / ``finish_turn`` land the terminal
                snapshot, leaving tasks stuck RUNNING for SDK clients.

        Returns:
            Execution result dictionary
        """
        # Initialize tracker if db_session and task_id are provided
        tracker = None
        tracker_task_id = tracking_task_id or task_id
        lease = None
        lease_stop_event = None
        lease_heartbeat_task = None
        result: Dict[str, Any] | None = None
        sandbox_task_key = None
        # Reused below for the workforce-status sync so we don't re-query the row.
        gate_task = None

        # Quota gate: refuse to start a run when the team is out of monthly
        # quota. Fails open if the check itself errors, so quota infra problems
        # never block execution.
        if db_session and tracker_task_id:
            try:
                from ..services.quota_hooks import check_run_gate

                gate_task = (
                    db_session.query(Task)
                    .filter(Task.id == int(tracker_task_id))
                    .first()
                )
                if gate_task is None:
                    # Fail open (a delete-race can legitimately leave no row),
                    # but surface it rather than silently allowing the run.
                    logger.warning(
                        "Quota gate: task %s not found; allowing run", tracker_task_id
                    )
                gate_reason = check_run_gate(
                    db_session, getattr(gate_task, "user_id", None)
                )
                if gate_reason:
                    # The gate returns either a plain message or a structured
                    # mapping ({code, metric, limit, plan, message}). Surface the
                    # message as `output` (what every result consumer shows as
                    # the assistant reply) and forward the structured fields as
                    # error_code/error_details so the client can localise and
                    # branch (e.g. Free vs paid) without parsing the message.
                    if isinstance(gate_reason, Mapping):
                        reason_message = str(gate_reason.get("message") or "")
                        error_code = gate_reason.get("code")
                        error_details = dict(gate_reason)
                    else:
                        # Legacy/plain-string path: a hook that returns a bare
                        # message (the pre-structured shape) is assumed to be a
                        # quota refusal. The only shipped hook returns a Mapping,
                        # so this stays for back-compat with string-returning
                        # app layers.
                        reason_message = str(gate_reason)
                        error_code = "quota_exceeded"
                        error_details = None
                    return {
                        "success": False,
                        "status": "quota_exceeded",
                        "output": reason_message,
                        "error": reason_message,
                        "error_code": error_code,
                        "error_details": error_details,
                    }
            except Exception:
                logger.warning("Quota gate check failed open", exc_info=True)

        if manage_task_lease and db_session and tracker_task_id:
            from ..services.task_execution_controller import (
                task_execution_controller,
            )

            async with task_execution_controller.command(int(tracker_task_id)):
                lease = acquire_task_lease(db_session, int(tracker_task_id))
            if lease is None:
                return {
                    "success": False,
                    "status": "running_elsewhere",
                    "error": "Task is already running on another worker.",
                }
            lease_stop_event = asyncio.Event()
            lease_heartbeat_task = asyncio.create_task(
                run_task_lease_heartbeat(lease, lease_stop_event)
            )

        if db_session and tracker_task_id:
            try:
                # Reuse the row already fetched for the quota gate; only re-query
                # if the gate path didn't run or errored before assigning it.
                task_for_sync = gate_task
                if task_for_sync is None:
                    task_for_sync = (
                        db_session.query(Task)
                        .filter(Task.id == int(tracker_task_id))
                        .first()
                    )
                if task_for_sync is not None and sync_workforce_run_status(
                    db_session, task_for_sync, TaskStatus.RUNNING
                ):
                    db_session.commit()
            except Exception:
                logger.debug(
                    "Failed to sync workforce run status after lease acquisition",
                    exc_info=True,
                )
            try:
                from ..tracking.task_tracker import TaskTracker

                tracker = TaskTracker(
                    task_id=int(tracker_task_id),
                    db_session=db_session,
                )
                await tracker.start_tracking()
                # Enforce quota mid-run: the pattern loop polls this every step
                # (each LLM reply / tool call) and stops the run once its live
                # cost would push the team over quota, instead of only metering at
                # completion. Cleared in the finally so a reused agent_service
                # never keeps a finished run's tracker as its checker.
                agent_service.set_interrupt_checker(tracker.interrupt_reason_for_quota)
                logger.info(f"Started token tracking for task {tracker_task_id}")
            except Exception as e:
                logger.warning(
                    f"Failed to start token tracking for task {tracker_task_id}: {e}"
                )
                tracker = None

        try:
            # Inside the try so a reclaimed-sandbox raise still runs the
            # finally (heartbeat stop, lease release, tracker completion);
            # _release_sandbox_task(None) is a no-op when nothing attached.
            sandbox_task_key = await self._acquire_sandbox_task(tracker_task_id)

            logger.info(
                f"=== About to execute task: task_id={task_id}, has_db_session={db_session is not None} ==="
            )

            # All pre-run DB writes (lease, status sync) are committed by
            # now; release the session's pooled connection so it isn't
            # pinned in ``idle in transaction`` for the entire agent run
            # (issue #889). The tracker re-acquires per update and commits,
            # so the slot is only held during actual DB work from here on.
            release_db_connection_if_clean(db_session)

            # Execute the task
            result = await agent_service.execute_task(
                task=task, context=context, task_id=task_id
            )

            # If a mid-run quota gate stopped the run, surface the reason the way
            # the start gate does — a terminal quota_exceeded result carrying the
            # reason as output — instead of the pattern-interrupt path's silent
            # flip to PAUSED (which persists no message).
            if tracker is not None and isinstance(result, dict):
                quota_reason = getattr(tracker, "quota_interrupt_reason", None)
                if quota_reason:
                    result = {
                        **result,
                        "success": False,
                        "status": "quota_exceeded",
                        "output": quota_reason,
                        "error": quota_reason,
                        # A mid-run interrupt is always the quota checker, so the
                        # code is known even though the reason is a plain string.
                        # Match the start gate so the client pops the same dialog.
                        "error_code": "quota_exceeded",
                    }

            logger.info("=== Task executed successfully, updating title if needed ===")

            # Update task title with generated task_name (clean architecture: Core provides API, Web handles DB)
            if db_session and task_id and result and result.get("success"):
                await update_task_title_from_agent(
                    agent_service, int(task_id), db_session
                )

            return result
        finally:
            await stop_task_lease_heartbeat(lease_heartbeat_task, lease_stop_event)
            if manage_task_lease and db_session and lease:
                if result is None:
                    final_status = TaskStatus.FAILED
                else:
                    status = str(result.get("status") or "")
                    if status == "waiting_for_user":
                        final_status = TaskStatus.WAITING_FOR_USER
                    elif status == "interrupted":
                        final_status = TaskStatus.PAUSED
                    elif result.get("success", False):
                        final_status = TaskStatus.COMPLETED
                    else:
                        final_status = TaskStatus.FAILED
                release_task_lease_with_workforce_sync(
                    db_session, lease, status=final_status
                )
            # Complete tracking if it was started
            if tracker:
                # Drop the mid-run quota checker before metering so a reused
                # agent_service can't keep calling this finished run's tracker.
                agent_service.set_interrupt_checker(None)
                try:
                    await tracker.complete_tracking()
                    logger.info(f"Completed token tracking for task {tracker_task_id}")
                except Exception as e:
                    logger.error(
                        f"Failed to complete token tracking for task {tracker_task_id}: {e}"
                    )
            await self._release_sandbox_task(sandbox_task_key)

    def _cleanup_workspace_directory(
        self, task_id: int, user_id: Optional[int] = None
    ) -> None:
        """Clean up workspace directory for a task when agent is not in memory"""
        from ...core.workspace import TaskWorkspace

        # Try the scoped workspace first (when a resolver maps this task to
        # a scope), then the user-isolated workspace, then the legacy
        # uploads-root fallback.
        workspace_ids = []
        if user_id:
            # Contextvar-first for the same reason as get_agent_for_task:
            # cleanup inside an activated turn reuses the turn's resolution.
            scope = get_execution_scope()
            if scope is None:
                scope = resolve_execution_scope(task_id)
            segments = scope.workspace_segments if scope is not None else ()
            if segments:
                workspace_ids.append(
                    (
                        f"web_task_{task_id}",
                        str(scoped_user_root(get_uploads_dir(), user_id, segments)),
                    )
                )
            workspace_ids.append(
                (
                    f"web_task_{task_id}",
                    str(scoped_user_root(get_uploads_dir(), user_id)),
                )
            )
        workspace_ids.append((f"web_task_{task_id}", str(get_uploads_dir())))

        # Build allowed external directories (user's upload directory for knowledge base files).
        # Use only_existing=True here because cleanup runs against on-disk state.
        allowed_external_dirs = _build_allowed_external_dirs(
            user_id, only_existing=True
        )

        for workspace_id, base_dir in workspace_ids:
            workspace = TaskWorkspace(
                workspace_id, base_dir, allowed_external_dirs=allowed_external_dirs
            )
            workspace_path = str(workspace.workspace_dir)

            if workspace.workspace_dir.exists():
                logger.info(
                    f"Found existing workspace directory for task {task_id} (user {user_id}): {workspace_path}"
                )
                workspace.cleanup()
                logger.info(
                    f"Cleaned up workspace directory for task {task_id} (user {user_id}): {workspace_path}"
                )
                break
        else:
            logger.info(
                f"No workspace directory found for task {task_id} (user {user_id})"
            )

    def _has_reconstructable_history(self, task_id: int, db: Session) -> bool:
        """Cheap pre-check: does the task have prior state that
        ``_reconstruct_agent_from_history`` could actually recover?

        Reconstruct depends on either:

        * trace events from prior tool / LLM runs (``TraceEvent`` rows
          for the task with ``build_id IS NULL`` -- VIBE phase only,
          matches the same filter ``_reconstruct_agent_from_history``
          uses)
        * a ``DAGExecution.current_plan`` blob for the task

        A brand-new SDK task whose status was just flipped to RUNNING
        by ``begin_turn`` has neither -- ``_reconstruct_agent_from_history``
        would run the same two queries, return empty, log a warning,
        and fall through to normal creation. The check here saves the
        wasted-work cost plus the noisy ``Failed to reconstruct agent
        from history`` log line that misleads post-incident triage
        into thinking something actually failed.

        ``PAUSED`` / ``WAITING_FOR_USER`` tasks always have prior state
        by definition; the caller in ``get_agent_for_task`` should only
        gate on this check for ``RUNNING`` status.

        Two ``.first()`` queries: trace short-circuits the plan check
        when present (typical case for any agent that has run a step).
        """
        from ..models.task import DAGExecution, TraceEvent

        has_trace = (
            db.query(TraceEvent)
            .filter(
                TraceEvent.task_id == task_id,
                TraceEvent.build_id.is_(None),
            )
            .first()
            is not None
        )
        if has_trace:
            return True
        has_plan = (
            db.query(DAGExecution).filter(DAGExecution.task_id == task_id).first()
            is not None
        )
        return has_plan

    async def _reconstruct_agent_from_history(
        self,
        task_id: int,
        db: Session,
        scope: Optional[ExecutionScope] = None,
    ) -> None:
        """Reconstruct agent from historical data"""
        try:
            # Get task user information from database
            task = db.query(Task).filter(Task.id == task_id).first()
            user_id = task.user_id if task else None

            # Get tracer events from database
            tracer_events = []
            plan_state = None

            # Query trace events
            from ..models.task import DAGExecution, TraceEvent

            # Get tracer events (only VIBE phase, exclude BUILD phase)
            trace_events = (
                db.query(TraceEvent)
                .filter(
                    TraceEvent.task_id == task_id,
                    TraceEvent.build_id.is_(None),  # ← Only get VIBE events
                )
                .all()
            )
            decoded_event_data = decode_trace_events_data(
                db,
                task_id=task_id,
                data_items=[event.data for event in trace_events],
                strict=False,
            )
            for event, event_data in zip(trace_events, decoded_event_data):
                tracer_events.append(
                    {
                        "id": event.event_id,
                        "event_type": event.event_type,
                        "task_id": str(event.task_id),
                        "step_id": event.step_id,
                        "timestamp": event.timestamp.timestamp()
                        if event.timestamp
                        else None,
                        "data": event_data,
                        "parent_id": event.parent_event_id,
                    }
                )

            # Get DAG execution data
            dag_execution = (
                db.query(DAGExecution).filter(DAGExecution.task_id == task_id).first()
            )
            if dag_execution and dag_execution.current_plan:
                plan_state = (
                    dict(dag_execution.current_plan)
                    if dag_execution.current_plan
                    else None
                )

            if tracer_events or plan_state:
                # Create a minimal agent first
                tracer = create_task_tracer(
                    task_id,
                    user_id=int(user_id) if user_id is not None else None,
                )

                # Get LLM configuration from task database record
                try:
                    task = db.query(Task).filter(Task.id == task_id).first()
                    if task:
                        user = (
                            db.query(User).filter(User.id == task.user_id).first()
                            if task.user_id
                            else None
                        )
                        if user is None:
                            raise ValueError(
                                "User context is required for agent reconstruction"
                            )

                        runtime_config = self._resolve_task_runtime_config(
                            task_id=task_id,
                            task=task,
                            db=db,
                            user=user,
                        )
                        agent_config = runtime_config["agent_config"]
                        task_llm = runtime_config["task_llm"]
                        task_fast_llm = runtime_config["task_fast_llm"]
                        task_vision_llm = runtime_config["task_vision_llm"]
                        task_compact_llm = runtime_config["task_compact_llm"]
                        task_pattern = runtime_config["task_pattern"]

                        tools_list, tool_config = await self._build_tools_for_task(
                            task_id=task_id,
                            task=task,
                            db=db,
                            user=user,
                            agent_config=agent_config,
                            task_llm=task_llm,
                            task_vision_llm=task_vision_llm,
                            parent_tracer=tracer,
                            scope=scope,
                        )
                    else:
                        raise ValueError(
                            f"Task {task_id} not found in database during agent reconstruction"
                        )
                except Exception as e:
                    logger.error(
                        f"Failed to rebuild runtime configuration for task {task_id}: {e}"
                    )
                    raise

                # Build allowed external directories
                allowed_external_dirs = _build_allowed_external_dirs(
                    int(user_id) if user_id is not None else None,
                    scope=scope,
                )
                scope_segments = scope.workspace_segments if scope is not None else ()

                # Create agent with basic configuration
                if user_id is not None:
                    with UserContext(int(user_id)):
                        from .agents import enhance_system_prompt_with_kb

                        system_prompt = (
                            agent_config.get("instructions") if agent_config else None
                        )
                        kb_list = (
                            agent_config.get("knowledge_bases")
                            if agent_config
                            else None
                        )
                        system_prompt = enhance_system_prompt_with_kb(
                            system_prompt, kb_list
                        )
                        system_prompt = _build_workforce_system_prompt(
                            system_prompt,
                            resolve_workforce_task_runtime(db, task),
                        )
                        memory_similarity_threshold = None
                        if (
                            agent_config
                            and "memory_similarity_threshold" in agent_config
                        ):
                            memory_similarity_threshold = agent_config[
                                "memory_similarity_threshold"
                            ]
                        memory_policy = resolve_agent_service_memory_policy(
                            task=task,
                            agent_config=agent_config,
                        )
                        self._agents[task_id] = AgentService(
                            name=f"reconstructed_agent_task_{task_id}",
                            id=f"web_task_{task_id}",  # Use task ID only for workspace
                            llm=task_llm,
                            fast_llm=task_fast_llm,
                            vision_llm=task_vision_llm,
                            compact_llm=task_compact_llm,
                            tools=tools_list,
                            tool_config=tool_config,
                            memory=memory_policy.memory,
                            pattern=task_pattern,
                            tracer=tracer,
                            system_prompt=system_prompt,
                            enable_workspace=True,
                            workspace_base_dir=str(
                                scoped_user_root(
                                    get_uploads_dir(), int(user_id), scope_segments
                                )
                            ),  # Use user- (and scope-) isolated base directory
                            allowed_external_dirs=allowed_external_dirs,
                            scope_segments=scope_segments,
                            task_id=str(task_id),
                            memory_similarity_threshold=memory_similarity_threshold,
                            memory_enabled=memory_policy.memory_enabled,
                        )
                else:
                    raise ValueError(
                        "User context is required for agent reconstruction"
                    )

                await self._agents[task_id].reconstruct_from_history(
                    str(task_id), tracer_events, plan_state
                )
                self._load_persisted_conversation_history(task_id, db)
                await self._load_persisted_execution_context(task_id, db)

                logger.info(
                    f"Successfully reconstructed agent for task {task_id} from history"
                )
            else:
                logger.info(
                    f"No historical data found for task {task_id}, will create new agent"
                )
                # Don't create agent here, let the normal flow handle it
                # Raise an exception to indicate reconstruction is not possible
                raise ValueError(f"No historical data found for task {task_id}")

        except Exception as e:
            logger.error(
                f"Failed to reconstruct agent from history for task {task_id}: {e}"
            )
            raise

    def get_agent_workspace_files(self, task_id: int) -> Dict[str, Any]:
        """Get workspace files for a task"""
        if task_id not in self._agents:
            raise ValueError(f"No agent found for task {task_id}")

        return self._agents[task_id].get_workspace_files()

    def get_agent_output_files(self, task_id: int) -> List[Dict[str, Any]]:
        """Get output files for a task"""
        if task_id not in self._agents:
            raise ValueError(f"No agent found for task {task_id}")

        return self._agents[task_id].get_output_files()


# Global agent manager
# Global agent manager instance
_global_agent_manager = None


def get_agent_manager(request: Any = None) -> AgentServiceManager:
    """Get AgentServiceManager instance with request context."""
    global _global_agent_manager
    if _global_agent_manager is None:
        _global_agent_manager = AgentServiceManager(request=request)
    else:
        # Update request if provided
        if request is not None:
            _global_agent_manager.request = request
    return _global_agent_manager


def _build_unique_workspace_target(base_dir: Path, filename: str) -> Path:
    candidate = base_dir / filename
    if not candidate.exists():
        return candidate

    stem = candidate.stem
    suffix = candidate.suffix
    index = 1
    while True:
        next_candidate = base_dir / f"{stem}_{index}{suffix}"
        if not next_candidate.exists():
            return next_candidate
        index += 1


@chat_router.post("/task/create", response_model=TaskCreateResponse)
async def create_task(
    request: TaskCreateRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> TaskCreateResponse:
    """Create new chat task"""
    try:
        # Build task description with file information
        task_description = request.description or ""

        selected_file_ids: list[str] = []

        # Add file information to description if files are specified
        if request.files:
            from ..models.uploaded_file import UploadedFile

            file_info_list = []
            file_paths = []

            for file_id in request.files:
                uploaded_file = (
                    db.query(UploadedFile)
                    .filter(
                        UploadedFile.file_id == file_id,
                        UploadedFile.user_id == int(user.id),
                        UploadedFile.task_id.is_(None),
                    )
                    .first()
                )
                if uploaded_file is None:
                    file_info_list.append(f"File ID: {file_id} (File does not exist)")
                    continue

                selected_file_ids.append(str(file_id))

                file_path = ensure_uploaded_file_local_path(uploaded_file)
                file_paths.append(str(file_path))

                if file_path.exists():
                    file_info_list.append(
                        f"File: {uploaded_file.filename} (Path: {file_path})"
                    )
                else:
                    file_info_list.append(
                        f"File: {uploaded_file.filename} (File does not exist)"
                    )

            if file_info_list:
                if task_description:
                    task_description += "\n\nUploaded files:\n" + "\n".join(
                        file_info_list
                    )
                else:
                    task_description = "File processing task:\n" + "\n".join(
                        file_info_list
                    )

        # Set LLM configuration for this task first to get model info.
        # Prefer internal model identifiers (llm_ids).
        # If neither is provided but agent_id is, fetch from agent config.
        from ..models.user import UserDefaultModel, UserModel
        from ..services.llm_utils import CoreStorage

        core_storage = CoreStorage(db, DBModel)

        def _to_internal_model_id_if_accessible(
            model_ref: Optional[Any],
        ) -> Optional[str]:
            if model_ref is None:
                return None
            if isinstance(model_ref, str):
                model_ref = model_ref.strip()
                if not model_ref:
                    return None

            db_model = core_storage.get_db_model(model_ref)
            if not db_model:
                return None

            # Two-step access check: own → shared from visible users
            own_model = (
                db.query(UserModel)
                .filter(
                    UserModel.user_id == int(user.id),
                    UserModel.model_id == db_model.id,
                    UserModel.is_owner.is_(True),
                )
                .first()
            )
            if not own_model:
                visible_ids = _get_visible_user_ids(db, int(user.id))
                own_model = (
                    db.query(UserModel)
                    .filter(
                        UserModel.model_id == db_model.id,
                        UserModel.user_id.in_(visible_ids),
                        UserModel.is_shared.is_(True),
                    )
                    .first()
                )
            has_access = own_model is not None
            if not has_access:
                return None

            return str(db_model.model_id)

        def _normalize_llm_refs(llm_refs: List[Optional[Any]]) -> List[Optional[str]]:
            return [
                _to_internal_model_id_if_accessible(model_ref) for model_ref in llm_refs
            ]

        def _get_default_internal_model_ids() -> Dict[str, Optional[str]]:
            from ..models.model import Model as DBModel

            config_types = ["general", "small_fast", "visual", "compact"]
            defaults: Dict[str, Optional[str]] = {ct: None for ct in config_types}

            # User-specific defaults (Mode A: use DBModel JOIN).
            user_defaults = (
                db.query(UserDefaultModel)
                .join(DBModel, UserDefaultModel.model_id == DBModel.id)
                .filter(
                    UserDefaultModel.user_id == int(user.id),
                    DBModel.is_active,
                    UserDefaultModel.config_type.in_(config_types),
                )
                .all()
            )
            from ..services.model_service import _is_model_visible_to_user

            for row in user_defaults:
                if row.model:
                    if _is_model_visible_to_user(db, row.model.id, int(user.id)):
                        config_type = cast(str, row.config_type)
                        defaults[config_type] = str(row.model.model_id)

            # Fill missing defaults from visible users' shared defaults.
            if any(defaults[ct] is None for ct in config_types):
                visible_ids = _get_visible_user_ids(db, int(user.id))
                shared_defaults = (
                    db.query(UserDefaultModel)
                    .join(UserModel, UserDefaultModel.model_id == UserModel.model_id)
                    .filter(
                        UserDefaultModel.config_type.in_(config_types),
                        UserModel.is_shared.is_(True),
                        UserDefaultModel.user_id.in_(visible_ids),
                    )
                    .all()
                )
                for row in shared_defaults:
                    config_type = row.config_type  # type: ignore
                    if row.model and defaults.get(config_type) is None:
                        defaults[config_type] = str(row.model.model_id)

            return defaults

        selected_agent: Optional[Agent] = None
        if request.agent_id:
            selected_agent = _load_agent_for_task_create(
                db,
                user,
                int(request.agent_id),
            )
            if not selected_agent:
                raise HTTPException(
                    status_code=404,
                    detail="Agent not found or access denied",
                )

        llm_ids_to_use = request.llm_ids
        if selected_agent:
            if request.llm_ids:
                logger.warning(
                    f"Ignoring caller-supplied llm_ids {request.llm_ids} because agent_id {request.agent_id} is present."
                )
            llm_ids_to_use = None
            if selected_agent.models:
                # Fetch model configuration from agent
                agent_models = selected_agent.models
                # Agent Builder stores references that may be DB PKs; normalize to internal
                # model_id only if the current user has access.
                llm_ids_to_use = _normalize_llm_refs(
                    [
                        agent_models.get("general"),
                        agent_models.get("small_fast"),
                        agent_models.get("visual"),
                        agent_models.get("compact"),
                    ]
                )
                logger.info(
                    f"Using agent {request.agent_id} model configuration (llm_ids): {llm_ids_to_use}"
                )

        # Normalize any refs (pk/model_name/model_id) to internal model_id strings,
        # but only if the current user has access to the model.
        if llm_ids_to_use:
            llm_ids_to_use = _normalize_llm_refs(llm_ids_to_use)

        default_llm, fast_llm, vision_llm, compact_llm = resolve_llms_from_names(
            llm_ids_to_use, db, int(user.id)
        )

        # Extract provider model names from resolved LLM instances for database storage
        default_model_name = default_llm.model_name if default_llm else None
        fast_model_name = fast_llm.model_name if fast_llm else None
        visual_model_name = vision_llm.model_name if vision_llm else None
        compact_model_name = compact_llm.model_name if compact_llm else None

        # Persist both:
        # - *_model_id: internal stable identifier (preferred for selection)
        # - *_model_name: provider-facing model name (useful for display/audit)
        default_model_id: Optional[str] = None
        fast_model_id: Optional[str] = None
        visual_model_id: Optional[str] = None
        compact_model_id: Optional[str] = None

        if llm_ids_to_use and len(llm_ids_to_use) == 4:
            default_model_id = llm_ids_to_use[0]
            fast_model_id = llm_ids_to_use[1]
            visual_model_id = llm_ids_to_use[2]
            compact_model_id = llm_ids_to_use[3]

        if (
            default_model_id is None
            or fast_model_id is None
            or visual_model_id is None
            or compact_model_id is None
        ):
            default_ids = _get_default_internal_model_ids()
            default_model_id = default_model_id or default_ids.get("general")
            fast_model_id = fast_model_id or default_ids.get("small_fast")
            visual_model_id = visual_model_id or default_ids.get("visual")
            compact_model_id = compact_model_id or default_ids.get("compact")

        # Convert agent_type string to enum
        agent_type_enum = AgentType.STANDARD
        if request.agent_type:
            try:
                agent_type_enum = AgentType(request.agent_type)
            except ValueError:
                logger.warning(
                    f"Unknown agent_type '{request.agent_type}', using STANDARD"
                )
                agent_type_enum = AgentType.STANDARD

        # Convert examples to list of dicts if provided
        examples_data = None
        if request.examples:
            examples_data = [
                {"input": ex.input, "output": ex.output} for ex in request.examples
            ]

        task_agent_config = _build_task_agent_config(
            request.agent_config,
            selected_file_ids,
        )
        if request.is_preview:
            task_agent_config = task_agent_config or {}
            task_agent_config["is_preview"] = True

        task_execution_mode = request.execution_mode
        if not task_execution_mode:
            task_execution_mode = get_default_task_execution_mode(
                agent_id=request.agent_id,
            )

        # Create task with PENDING status and model configuration
        task_title = request.title if request.title else task_description
        if task_title and len(task_title) > 50:
            task_title = task_title[:50] + "..."

        task = Task(
            user_id=user.id,  # Use authenticated user ID
            title=task_title,
            description=task_description,
            status=TaskStatus.PENDING,
            model_id=default_model_id,
            small_fast_model_id=fast_model_id,
            visual_model_id=visual_model_id,
            compact_model_id=compact_model_id,
            model_name=default_model_name,
            small_fast_model_name=fast_model_name,
            visual_model_name=visual_model_name,
            compact_model_name=compact_model_name,
            agent_config=task_agent_config,
            execution_mode=task_execution_mode,
            process_description=request.process_description,
            examples=examples_data,
            agent_id=request.agent_id,  # Set agent_id if provided
            is_visible=False if request.is_preview else request.is_visible,
        )
        selected_refs = prepare_connector_runtime_selection_snapshot(
            db=db,
            agent=selected_agent,
            connector_user_id=int(user.id),
        )
        bind_connector_runtime_selection_snapshot(
            task=task, selected_refs=selected_refs
        )

        # Set agent_type using the property to avoid Column type issues
        task.agent_type_enum = agent_type_enum
        db.add(task)
        db.flush()

        # Set LLM configuration for this task in agent manager
        task_llm_ids_to_set = [
            default_model_id,
            fast_model_id,
            visual_model_id,
            compact_model_id,
        ]
        logger.info(
            f"Setting LLM configuration for task {task.id} with llm_ids: {task_llm_ids_to_set}"
        )
        get_agent_manager(request).set_task_llms(int(task.id), task_llm_ids_to_set, db)

        if selected_file_ids:
            from ..models.uploaded_file import UploadedFile

            (
                db.query(UploadedFile)
                .filter(
                    UploadedFile.file_id.in_(selected_file_ids),
                    UploadedFile.user_id == int(user.id),
                    UploadedFile.task_id.is_(None),
                )
                .update(
                    {UploadedFile.task_id: int(task.id)},
                    synchronize_session=False,
                )
            )

        db.commit()
        db.refresh(task)

        return TaskCreateResponse(
            task_id=task.id,
            title=task.title,
            status=task.status.value,
            created_at=format_datetime_for_api(task.created_at)
            if task.created_at
            else None,
            model_id=task.model_id,
            small_fast_model_id=task.small_fast_model_id,
            visual_model_id=task.visual_model_id,
            compact_model_id=task.compact_model_id,
            model_name=task.model_name,
            small_fast_model_name=task.small_fast_model_name,
            visual_model_name=task.visual_model_name,
            compact_model_name=task.compact_model_name,
            execution_mode=task.execution_mode,
            channel_id=task.channel_id,
            channel_name=task.channel_name,
            agent_id=task.agent_id,
            agent_name=task.agent.name if task.agent else None,
            agent_logo_url=task.agent.logo_url if task.agent else None,
            run_id=task.run_id,
            state_version=int(task.state_version or 0),
            control_state=str(task.control_state or "idle"),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Create task failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@chat_router.get("/tasks")
async def get_tasks(
    page: int = 1,
    per_page: int = 10,
    search: Optional[str] = None,
    agent_type: Optional[str] = None,
    exclude_agent_type: Optional[str] = None,
    execution_mode: Optional[str] = None,
    exclude_execution_mode: Optional[str] = None,
    include_hidden: bool = False,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """Get tasks list with pagination"""
    try:
        # Run synchronous database queries in thread pool to avoid blocking event loop
        def _get_tasks_sync() -> Dict[str, Any]:
            # Build base query - filter by current user, unless admin
            if user.is_admin:
                # Admin can see all tasks - include user relationship for admin
                from sqlalchemy.orm import joinedload

                query = db.query(Task).options(joinedload(Task.user))
            else:
                # Regular users can only see their own tasks
                query = db.query(Task).filter(Task.user_id == user.id)

            if not include_hidden:
                query = query.filter(Task.is_visible.is_(True))

            # Apply search filter if provided
            if search:
                query = query.filter(Task.title.ilike(f"%{search}%"))

            # Apply agent type filter if provided
            if agent_type:
                from ..models.task import AgentType

                try:
                    agent_type_enum = AgentType(agent_type)
                    if agent_type_enum.value == AgentType.STANDARD.value:
                        # For STANDARD agent type, include both 'standard' and NULL values
                        query = query.filter(
                            (Task.agent_type == agent_type_enum.value)
                            | (Task.agent_type.is_(None))
                        )
                    else:
                        # For other agent types, filter by exact value
                        query = query.filter(Task.agent_type == agent_type_enum.value)
                except ValueError:
                    # Invalid agent type, ignore filter
                    pass

            # Apply agent type exclusion filter if provided
            if exclude_agent_type:
                from ..models.task import AgentType

                try:
                    exclude_type_enum = AgentType(exclude_agent_type)
                    if exclude_type_enum.value == AgentType.STANDARD.value:
                        # Exclude STANDARD agent type (both 'standard' and NULL)
                        query = query.filter(
                            (Task.agent_type != exclude_type_enum.value)
                            & (Task.agent_type.isnot(None))
                        )
                    else:
                        # Exclude specific agent type
                        query = query.filter(Task.agent_type != exclude_type_enum.value)
                except ValueError:
                    # Invalid agent type, ignore filter
                    pass

            # Apply execution mode filter if provided
            if execution_mode:
                query = query.filter(Task.execution_mode == execution_mode)
            elif exclude_execution_mode:
                query = query.filter(Task.execution_mode != exclude_execution_mode)

            # Get total count
            total = query.count()

            # Apply pagination
            offset = (page - 1) * per_page
            query = (
                query.order_by(Task.created_at.desc()).offset(offset).limit(per_page)
            )
            tasks_query = query.all()

            # Batch fetch agents for tasks with agent_id
            agent_ids = {task.agent_id for task in tasks_query if task.agent_id}
            agents_map = {}
            if agent_ids:
                agents = db.query(Agent).filter(Agent.id.in_(agent_ids)).all()
                agents_map = {agent.id: agent for agent in agents}

            # Channel names are user-defined, so clients need the persisted type
            # to render a reliable platform indicator without guessing from text.
            channel_ids = {
                task.channel_id for task in tasks_query if task.channel_id is not None
            }
            channels_map = {}
            if channel_ids:
                channels = (
                    db.query(UserChannel.id, UserChannel.channel_type)
                    .filter(UserChannel.id.in_(channel_ids))
                    .all()
                )
                channels_map = {
                    channel_id: channel_type for channel_id, channel_type in channels
                }

            # Convert Task objects to dictionaries for JSON serialization
            tasks = []
            for task in tasks_query:
                try:
                    # Get the raw status value from the database
                    if hasattr(task, "status") and task.status is not None:
                        if hasattr(task.status, "value"):
                            status_value = task.status.value
                        else:
                            status_value = str(task.status)
                    else:
                        status_value = "unknown"

                    task_data = {
                        "task_id": task.id,
                        "title": task.title,
                        "status": status_value,
                        "run_id": task.run_id,
                        "state_version": int(task.state_version or 0),
                        "control_state": str(task.control_state or "idle"),
                        "created_at": format_datetime_for_api(task.created_at),
                        "updated_at": format_datetime_for_api(task.updated_at),
                        "model_id": task.model_id,
                        "small_fast_model_id": task.small_fast_model_id,
                        "visual_model_id": task.visual_model_id,
                        "compact_model_id": task.compact_model_id,
                        "model_name": task.model_name,
                        "small_fast_model_name": task.small_fast_model_name,
                        "visual_model_name": task.visual_model_name,
                        "execution_mode": task.execution_mode,
                        "input_tokens": task.input_tokens or 0,
                        "output_tokens": task.output_tokens or 0,
                        "total_tokens": task.total_tokens or 0,
                        "llm_calls": task.llm_calls or 0,
                        "agent_id": task.agent_id,
                        "channel_id": task.channel_id,
                        "channel_name": task.channel_name,
                        "channel_type": channels_map.get(task.channel_id),
                    }

                    if task.agent_id and task.agent_id in agents_map:
                        task_data["agent_logo_url"] = agents_map[task.agent_id].logo_url
                        task_data["agent_name"] = agents_map[task.agent_id].name

                    # Include user information for admin users
                    if user.is_admin:
                        task_data["user_id"] = task.user_id
                        task_data["username"] = (
                            task.user.username if task.user else "Unknown"
                        )

                    tasks.append(task_data)
                except Exception as e:
                    logger.warning(f"Error processing task {task.id}: {e}")
                    continue

            # Calculate pagination metadata
            total_pages = (total + per_page - 1) // per_page

            return {
                "tasks": tasks,
                "pagination": {
                    "page": page,
                    "per_page": per_page,
                    "total_count": total,
                    "total_pages": total_pages,
                    "has_next": page < total_pages,
                    "has_prev": page > 1,
                },
            }

        # Execute in thread pool to avoid blocking
        result = await asyncio.to_thread(_get_tasks_sync)

        return result
    except Exception as e:
        logger.error(f"Get tasks failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@chat_router.get("/task/{task_id}")
async def get_task(
    task_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get task details"""
    try:
        # Run synchronous database queries in thread pool to avoid blocking event loop
        def _get_task_sync() -> Dict[str, Any]:
            # Admin can see any task, regular users can only see their own
            if user.is_admin:
                task = db.query(Task).filter(Task.id == task_id).first()
            else:
                task = (
                    db.query(Task)
                    .filter(Task.id == task_id, Task.user_id == user.id)
                    .first()
                )
            if not task:
                raise HTTPException(status_code=404, detail="Task not found")

            mark_task_paused_if_stale(db, task)
            db.refresh(task)

            cache_key = web_task_detail_key(task_id)
            task_updated_at = cache_version_token(task.updated_at)

            # Get the raw status value safely
            if hasattr(task, "status") and task.status is not None:
                if hasattr(task.status, "value"):
                    status_value = task.status.value
                else:
                    status_value = str(task.status)
            else:
                status_value = "unknown"

            # Get DAG execution data
            dag_data = None
            from ..models.task import DAGExecution

            dag_execution = (
                db.query(DAGExecution).filter(DAGExecution.task_id == task_id).first()
            )
            dag_updated_at = (
                cache_version_token(dag_execution.updated_at) if dag_execution else None
            )
            activity_ids: tuple[int, int] | None = None
            cached = cache_get(cache_key)
            if (
                isinstance(cached, dict)
                and cached.get("updated_at") == task_updated_at
                and cached.get("dag_updated_at") == dag_updated_at
            ):
                if task.status in _TERMINAL_CACHE_STATUSES:
                    return cast(Dict[str, Any], cached["response"])
                activity_ids = _get_task_activity_ids(db, task_id)
                if cached.get("max_trace_event_id") == int(
                    activity_ids[0]
                ) and cached.get("max_chat_message_id") == int(activity_ids[1]):
                    return cast(Dict[str, Any], cached["response"])

            if task.status not in _TERMINAL_CACHE_STATUSES and activity_ids is None:
                activity_ids = _get_task_activity_ids(db, task_id)

            if dag_execution:
                dag_data = {
                    "phase": dag_execution.phase.value if dag_execution.phase else None,
                    "current_plan": dag_execution.current_plan,
                    "created_at": safe_timestamp_to_unix(dag_execution.created_at)
                    if dag_execution.created_at
                    else None,
                    "updated_at": safe_timestamp_to_unix(dag_execution.updated_at)
                    if dag_execution.updated_at
                    else None,
                }

            # If model_id columns are not populated (legacy rows), best-effort resolve them
            # from stored provider-facing model_name values.
            llm_ids = get_agent_manager()._get_task_llm_ids(task, db)
            model_id, small_fast_model_id, visual_model_id, compact_model_id = llm_ids
            waiting_question = None
            waiting_interactions = None
            if task.status == TaskStatus.WAITING_FOR_USER:
                waiting_question, waiting_interactions = get_latest_waiting_question(
                    db, task_id
                )

            # Fetch agent info if agent relationship is available
            agent_name = task.agent.name if task.agent else None
            agent_logo_url = task.agent.logo_url if task.agent else None

            model_usage = aggregate_token_usage_by_model(task.token_usage_details)
            response = {
                "task_id": task.id,
                "title": task.title,
                "description": task.description,
                "status": status_value,
                "run_id": task.run_id,
                "state_version": int(task.state_version or 0),
                "control_state": str(task.control_state or "idle"),
                "created_at": format_datetime_for_api(task.created_at),
                "updated_at": format_datetime_for_api(task.updated_at),
                "model_id": model_id,
                "small_fast_model_id": small_fast_model_id,
                "visual_model_id": visual_model_id,
                "compact_model_id": compact_model_id,
                "model_name": task.model_name,
                "small_fast_model_name": task.small_fast_model_name,
                "visual_model_name": task.visual_model_name,
                "compact_model_name": task.compact_model_name,
                "dag_data": dag_data,
                "input_tokens": task.input_tokens or 0,
                "output_tokens": task.output_tokens or 0,
                "total_tokens": task.total_tokens or 0,
                "llm_calls": task.llm_calls or 0,
                "cached_input_tokens": sum(
                    entry["cached_input_tokens"] for entry in model_usage
                ),
                "cache_write_input_tokens": sum(
                    entry["cache_write_input_tokens"] for entry in model_usage
                ),
                "model_usage": model_usage,
                "agent_id": task.agent_id,
                "agent_name": agent_name,
                "agent_logo_url": agent_logo_url,
                "channel_id": task.channel_id,
                "channel_name": task.channel_name,
                "waiting_question": waiting_question,
                "waiting_interactions": waiting_interactions,
            }
            cache_set(
                cache_key,
                {
                    "updated_at": task_updated_at,
                    "dag_updated_at": dag_updated_at,
                    "max_trace_event_id": (
                        activity_ids[0] if activity_ids is not None else None
                    ),
                    "max_chat_message_id": (
                        activity_ids[1] if activity_ids is not None else None
                    ),
                    "response": response,
                },
                ttl_seconds=task_cache_ttl_seconds(),
            )
            return response

        # Execute in thread pool to avoid blocking
        return await asyncio.to_thread(_get_task_sync)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get task failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@chat_router.get("/task/{task_id}/status")
async def get_task_status(
    task_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get task status"""
    try:
        # Run synchronous database queries in thread pool to avoid blocking event loop
        def _get_task_status_sync() -> Dict[str, Any]:
            # Admin can see any task, regular users can only see their own
            if user.is_admin:
                task = db.query(Task).filter(Task.id == task_id).first()
            else:
                task = (
                    db.query(Task)
                    .filter(Task.id == task_id, Task.user_id == user.id)
                    .first()
                )
            if not task:
                raise HTTPException(status_code=404, detail="Task not found")

            cache_key = web_task_status_key(task_id)
            task_updated_at = cache_version_token(task.updated_at)
            activity_ids: tuple[int, int] | None = None
            cached = cache_get(cache_key)
            if isinstance(cached, dict) and cached.get("updated_at") == task_updated_at:
                if task.status in _TERMINAL_CACHE_STATUSES:
                    return cast(Dict[str, Any], cached["response"])
                activity_ids = _get_task_activity_ids(db, task_id)
                if cached.get("max_trace_event_id") == int(
                    activity_ids[0]
                ) and cached.get("max_chat_message_id") == int(activity_ids[1]):
                    return cast(Dict[str, Any], cached["response"])

            if task.status not in _TERMINAL_CACHE_STATUSES and activity_ids is None:
                activity_ids = _get_task_activity_ids(db, task_id)

            # Get the raw status value safely
            if hasattr(task, "status") and task.status is not None:
                if hasattr(task.status, "value"):
                    status_value = task.status.value
                else:
                    status_value = str(task.status)
            else:
                status_value = "unknown"

            llm_ids = get_agent_manager()._get_task_llm_ids(task, db)
            model_id, small_fast_model_id, visual_model_id, compact_model_id = llm_ids
            waiting_question = None
            waiting_interactions = None
            if task.status == TaskStatus.WAITING_FOR_USER:
                waiting_question, waiting_interactions = get_latest_waiting_question(
                    db, task_id
                )

            # Fetch agent info if agent relationship is available
            agent_name = task.agent.name if task.agent else None
            agent_logo_url = task.agent.logo_url if task.agent else None

            response = {
                "task_id": task.id,
                "title": task.title,
                "status": status_value,
                "run_id": task.run_id,
                "state_version": int(task.state_version or 0),
                "control_state": str(task.control_state or "idle"),
                "created_at": format_datetime_for_api(task.created_at),
                "updated_at": format_datetime_for_api(task.updated_at),
                "model_id": model_id,
                "small_fast_model_id": small_fast_model_id,
                "visual_model_id": visual_model_id,
                "compact_model_id": compact_model_id,
                "model_name": task.model_name,
                "small_fast_model_name": task.small_fast_model_name,
                "visual_model_name": task.visual_model_name,
                "compact_model_name": task.compact_model_name,
                "input_tokens": task.input_tokens or 0,
                "output_tokens": task.output_tokens or 0,
                "total_tokens": task.total_tokens or 0,
                "llm_calls": task.llm_calls or 0,
                "agent_id": task.agent_id,
                "agent_name": agent_name,
                "agent_logo_url": agent_logo_url,
                "channel_id": task.channel_id,
                "channel_name": task.channel_name,
                "waiting_question": waiting_question,
                "waiting_interactions": waiting_interactions,
            }
            cache_set(
                cache_key,
                {
                    "updated_at": task_updated_at,
                    "max_trace_event_id": (
                        activity_ids[0] if activity_ids is not None else None
                    ),
                    "max_chat_message_id": (
                        activity_ids[1] if activity_ids is not None else None
                    ),
                    "response": response,
                },
                ttl_seconds=task_cache_ttl_seconds(),
            )
            return response

        # Execute in thread pool to avoid blocking
        return await asyncio.to_thread(_get_task_status_sync)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get task status failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@chat_router.put("/task/{task_id}")
async def update_task(
    task_id: int,
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> dict:
    """Update task details."""
    try:
        data = await request.json()
        title = data.get("title")

        if not title:
            raise HTTPException(status_code=400, detail="Title is required")

        # Verify task exists and belongs to user
        if user.is_admin:
            task = db.query(Task).filter(Task.id == task_id).first()
        else:
            task = (
                db.query(Task)
                .filter(Task.id == task_id, Task.user_id == user.id)
                .first()
            )

        if not task:
            raise HTTPException(status_code=404, detail="Task not found")

        task.title = title
        db.commit()
        invalidate_task_cache(task_id)

        return {"status": "success", "message": "Task updated successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update task {task_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@chat_router.delete("/task/{task_id}")
async def delete_task(
    task_id: int,
    request: Any = None,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """Delete a task and all related data"""
    try:
        # Run synchronous database queries in thread pool to avoid blocking event loop
        def _delete_task_sync() -> Task:
            # Get task - admin can delete any task, regular users can only delete their own
            if user.is_admin:
                task = db.query(Task).filter(Task.id == task_id).first()
            else:
                task = (
                    db.query(Task)
                    .filter(Task.id == task_id, Task.user_id == user.id)
                    .first()
                )
            if not task:
                raise HTTPException(status_code=404, detail="Task not found")

            # Delete related data in correct order to respect foreign key constraints
            logger.info(f"Deleting task {task_id} and all related data")

            # Delete task-owned rows that do not all have DB-level cascades.
            from ..models.task import (
                DAGExecution,
                TraceCheckpointBlob,
                TraceEvent,
                TraceMessageBlob,
            )

            db.query(TraceCheckpointBlob).filter(
                TraceCheckpointBlob.task_id == task_id
            ).delete(synchronize_session=False)
            db.query(TraceMessageBlob).filter(
                TraceMessageBlob.task_id == task_id
            ).delete(synchronize_session=False)
            db.query(TraceEvent).filter(TraceEvent.task_id == task_id).delete(
                synchronize_session=False
            )
            db.query(DAGExecution).filter(DAGExecution.task_id == task_id).delete(
                synchronize_session=False
            )

            # Note: tool_usages, agents, and agent_tools tables have been removed

            # Delete the task itself
            db.delete(task)
            db.commit()

            return task

        # Execute database operations in thread pool to avoid blocking
        task = await asyncio.to_thread(_delete_task_sync)
        invalidate_task_cache(task_id)

        # Remove agent from manager if it exists
        get_agent_manager(request).remove_agent(task_id, int(user.id))

        from .websocket import background_task_manager, manager

        connections = manager.detach_task_connections(task_id)

        async def _cleanup_runtime_state() -> None:
            await background_task_manager.cancel_task(task_id, timeout_seconds=0.05)
            for connection in list(connections):
                try:
                    await connection.close()
                except Exception as e:
                    logger.warning(f"Failed to close WebSocket connection: {e}")

        asyncio.create_task(_cleanup_runtime_state())

        logger.info(f"Task {task_id} deleted successfully")

        return {
            "success": True,
            "message": f"Task '{task.title}' deleted successfully",
            "task_id": task_id,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete task failed: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@chat_router.get("/workspace/{task_id}/files")
async def get_task_workspace_files(
    task_id: int,
    request: Any = None,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """Get all workspace files for a task"""
    try:
        # Run synchronous database queries in thread pool to avoid blocking event loop
        def _verify_task_sync() -> Task:
            # Verify task ownership - admin can access any task
            if user.is_admin:
                task = db.query(Task).filter(Task.id == task_id).first()
            else:
                task = (
                    db.query(Task)
                    .filter(Task.id == task_id, Task.user_id == user.id)
                    .first()
                )
            if not task:
                raise HTTPException(status_code=404, detail="Task not found")
            return task

        # Execute database operations in thread pool to avoid blocking
        await asyncio.to_thread(_verify_task_sync)

        workspace_files = get_agent_manager(request).get_agent_workspace_files(task_id)
        return {
            "success": True,
            "task_id": task_id,
            "workspace_files": workspace_files,
        }
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Get workspace files failed for task {task_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@chat_router.get("/workspace/{task_id}/output")
async def get_task_output_files(
    task_id: int,
    request: Any = None,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """Get output files for a task"""
    try:
        # Run synchronous database queries in thread pool to avoid blocking event loop
        def _verify_task_sync() -> Task:
            # Verify task ownership - admin can access any task
            if user.is_admin:
                task = db.query(Task).filter(Task.id == task_id).first()
            else:
                task = (
                    db.query(Task)
                    .filter(Task.id == task_id, Task.user_id == user.id)
                    .first()
                )
            if not task:
                raise HTTPException(status_code=404, detail="Task not found")
            return task

        # Execute database operations in thread pool to avoid blocking
        await asyncio.to_thread(_verify_task_sync)

        agent_service = get_agent_manager(request)
        output_files = agent_service.get_agent_output_files(task_id)
        return {
            "success": True,
            "task_id": task_id,
            "output_files": output_files,
            "file_count": len(output_files),
        }
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Get output files failed for task {task_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
