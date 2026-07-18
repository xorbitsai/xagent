"""
Agent Tool - Convert published agents into callable tools
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Optional, Type
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator

from .....config import get_agent_pattern_for_execution_mode, get_uploads_dir
from .....web.services.agent_store import AgentStore
from .....web.services.model_service import (
    _get_visible_user_ids,
    _is_model_visible_to_user,
)

# ``WebToolConfig`` is referenced as a module-level name in ``run_json_async``
# so the nested sub-agent's config is constructed via the module attribute
# (which also keeps it patchable in tests).
from .....web.tools.config import WebToolConfig
from ....tracing import create_agent_tracer
from ....utils.type_check import ensure_list
from ...core.document_search import find_missing_knowledge_bases
from .agent_tool_names import gen_agent_tool_name
from .base import AbstractBaseTool, ToolCategory, ToolVisibility

logger = logging.getLogger(__name__)
MAX_AGENT_NAME_LENGTH = 200


def _assignable_tool_categories() -> list[str]:
    return [cat.value for cat in ToolCategory if cat is not ToolCategory.OTHER]


class _DelegatedAgentDatabaseTraceHandler:
    """Persist child-agent traces without broadcasting them to the parent UI."""

    def __init__(
        self,
        *,
        task_id: int,
        build_id: str,
        metadata: Mapping[str, Any],
    ) -> None:
        from .....web.api.trace_handlers import DatabaseTraceHandler

        self.task_id = task_id
        self.build_id = build_id
        self.metadata = dict(metadata)
        self._handler = DatabaseTraceHandler(task_id, build_id=build_id)

    async def handle_event(self, event: Any) -> None:
        original_data = event.data
        base_data = original_data if isinstance(original_data, dict) else {}
        event.data = {**base_data, **self.metadata}
        try:
            await self._handler.handle_event(event)
        finally:
            event.data = original_data

    async def load_latest_checkpoint(
        self, execution_id: str
    ) -> Optional[dict[str, Any]]:
        return await self._handler.load_latest_checkpoint(execution_id)


def _normalize_agent_ids(agent_ids: Any) -> Optional[list[int]]:
    if agent_ids is None:
        return None

    if isinstance(agent_ids, (str, int)):
        values = [agent_ids]
    else:
        try:
            values = list(agent_ids)
        except TypeError:
            values = [agent_ids]

    normalized: list[int] = []
    seen: set[int] = set()
    for raw_agent_id in values:
        try:
            agent_id = int(raw_agent_id)
        except (TypeError, ValueError):
            continue
        if agent_id in seen:
            continue
        normalized.append(agent_id)
        seen.add(agent_id)
    return normalized


def _coerce_db_task_id(task_id: Any) -> Optional[int]:
    if task_id is None:
        return None

    if isinstance(task_id, bool):
        return None

    if isinstance(task_id, int):
        return task_id

    if not isinstance(task_id, str):
        return None

    normalized = task_id.strip()
    if normalized.isdecimal():
        return int(normalized)

    for prefix in ("web_task_", "task_"):
        if normalized.startswith(prefix):
            task_id_value = normalized.removeprefix(prefix)
            return int(task_id_value) if task_id_value.isdecimal() else None

    return None


def _apply_agent_visibility_filters(
    query: Any,
    agent_model: Any,
    *,
    user_id: int,
    allowed_agent_ids: Optional[list[int]],
    allow_cross_user_agent_ids: bool,
    db: Any,
) -> Any | None:
    from .....web.services.agent_team_scope import (
        get_agent_team_scope,
        owned_agent_clause,
    )

    normalized_allowed_agent_ids = _normalize_agent_ids(allowed_agent_ids)
    if normalized_allowed_agent_ids is not None:
        if not normalized_allowed_agent_ids:
            return None
        query = query.filter(agent_model.id.in_(normalized_allowed_agent_ids))

    # Cross-user execution is only valid for explicit allowlists. Global
    # discovery and direct execution stay scoped to what the user owns, which
    # under a team scope means team-visible agents (admins-only stay hidden).
    if not allow_cross_user_agent_ids or normalized_allowed_agent_ids is None:
        query = query.filter(
            owned_agent_clause(user_id, get_agent_team_scope(db, user_id))
        )

    query = _exclude_private_workforce_manager_agents(query, agent_model)

    return query


def _exclude_private_workforce_manager_agents(query: Any, agent_model: Any) -> Any:
    from .....web.models.agent import AgentOrigin

    return query.filter(
        agent_model.origin != AgentOrigin.WORKFORCE_GENERATED_MANAGER.value
    )


def _apply_owned_agent_tool_filters(
    query: Any, agent_model: Any, *, user_id: int, db: Any
) -> Any:
    from .....web.services.agent_team_scope import (
        get_agent_team_scope,
        owned_agent_clause,
    )

    query = query.filter(owned_agent_clause(user_id, get_agent_team_scope(db, user_id)))
    query = _exclude_private_workforce_manager_agents(query, agent_model)
    return query


def _normalize_agent_tool_overrides(
    overrides: Optional[Mapping[Any, Any]],
) -> dict[int, dict[str, Any]]:
    if not isinstance(overrides, Mapping):
        return {}

    normalized: dict[int, dict[str, Any]] = {}
    for raw_agent_id, raw_override in overrides.items():
        if not isinstance(raw_override, Mapping):
            continue
        try:
            agent_id = int(raw_agent_id)
        except (TypeError, ValueError):
            continue
        normalized[agent_id] = dict(raw_override)
    return normalized


def _truthy_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return bool(value)


def _string_override(overrides: Mapping[str, Any], *keys: str) -> Optional[str]:
    for key in keys:
        value = overrides.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


async def _missing_knowledge_bases_for_user(
    knowledge_bases: Optional[list[str]], db: Any, user_id: int
) -> list[str]:
    requested = ensure_list(knowledge_bases)
    if not requested:
        return []

    from .....web.models.user import User

    user = db.query(User).filter(User.id == user_id).first()
    return await find_missing_knowledge_bases(
        requested,
        user_id=user_id,
        is_admin=bool(user.is_admin) if user else False,
    )


class CreateAgentToolArgs(BaseModel):
    """Arguments for creating a new agent."""

    name: str = Field(description="Name of the agent to create")
    description: str = Field(
        description=(
            "IMPORTANT: Description of when to use this agent (e.g., 'Use this "
            "agent for data analysis tasks involving CSV files'). This helps users "
            "understand the agent's purpose and when to call it. Write this in the "
            "same natural language as the current output language policy unless the user "
            "explicitly asks for another language."
        )
    )
    instructions: str = Field(
        description=(
            "System instructions/prompt for the agent. Write persisted "
            "user-facing prose in the same natural language as the current output "
            "language policy unless the user explicitly asks for another language."
        )
    )
    tool_categories: Optional[list[str]] = Field(
        default=None,
        description="List of tool categories to allow (e.g., ['file', 'knowledge']). If None, all tools are available",
    )
    knowledge_bases: Optional[list[str]] = Field(
        default=None,
        description="List of knowledge base names or IDs to associate with this agent. (optional)",
    )
    skills: Optional[list[str]] = Field(
        default=None,
        description="List of skill names to allow. If None, all skills are available",
    )
    execution_mode: Optional[str] = Field(
        default="balanced",
        description="Execution mode for the agent: 'flash', 'balanced' (default), 'think', or 'auto'.",
    )

    @field_validator("tool_categories", "knowledge_bases", "skills", mode="before")
    @classmethod
    def parse_stringified_lists(cls, v: Any) -> Any:
        if isinstance(v, str):
            parsed = ensure_list(v)
            if parsed is not None:
                return parsed
        return v


class CreateAgentToolResult(BaseModel):
    """Result from creating a new agent."""

    agent_id: int = Field(description="The ID of the created agent")
    agent_name: str = Field(description="The name of the created agent")
    tool_name: str = Field(
        description="The tool name that can be used to call this agent"
    )
    markdown_link: str = Field(
        description="Markdown link to the agent (e.g., '[Agent Name](agent://123)')"
    )
    status: str = Field(description="Creation status")
    message: str = Field(description="Detailed message about the created agent")


class UpdateAgentToolArgs(BaseModel):
    """Arguments for updating an existing agent."""

    agent_id: int = Field(description="ID of the agent to update")
    name: Optional[str] = Field(
        default=None, description="New name for the agent (optional)"
    )
    description: Optional[str] = Field(
        default=None,
        description="New description of when to use this agent (optional)",
    )
    instructions: Optional[str] = Field(
        default=None,
        description="New system instructions/prompt for the agent (optional)",
    )
    tool_categories: Optional[list[str]] = Field(
        default=None,
        description="New list of tool categories to allow (optional). If None, existing value is kept",
    )
    knowledge_bases: Optional[list[str]] = Field(
        default=None,
        description="New list of knowledge base IDs or names to associate with this agent. (optional). If None, existing value is kept",
    )
    skills: Optional[list[str]] = Field(
        default=None,
        description="New list of skill names to allow (optional). If None, existing value is kept",
    )
    execution_mode: Optional[str] = Field(
        default=None,
        description="New execution mode for the agent: 'flash', 'balanced', 'think', or 'auto'.",
    )

    @field_validator("tool_categories", "knowledge_bases", "skills", mode="before")
    @classmethod
    def parse_stringified_lists(cls, v: Any) -> Any:
        if isinstance(v, str):
            parsed = ensure_list(v)
            if parsed is not None:
                return parsed
        return v


class UpdateAgentToolResult(BaseModel):
    """Result from updating an agent."""

    agent_id: int = Field(description="The ID of the updated agent")
    agent_name: str = Field(description="The name of the updated agent")
    tool_name: str = Field(
        description="The tool name that can be used to call this agent"
    )
    markdown_link: str = Field(
        description="Markdown link to the agent (e.g., '[Agent Name](agent://123)')"
    )
    status: str = Field(description="Update status")
    message: str = Field(description="Detailed message about the updated agent")


class ListAgentsToolArgs(BaseModel):
    """Arguments for listing agents."""

    status_filter: Optional[str] = Field(
        default=None,
        description="Filter by agent status: 'draft', 'published', or 'archived'. If None, shows all agents",
    )


class AgentInfo(BaseModel):
    """Information about a single agent."""

    agent_id: int = Field(description="Agent ID")
    name: str = Field(description="Agent name")
    description: str = Field(description="Agent description")
    status: str = Field(description="Agent status: draft, published, or archived")
    tool_name: str = Field(description="Tool name to call this agent")
    markdown_link: str = Field(description="Markdown link to the agent")
    execution_mode: str = Field(
        description="Execution mode: flash, balanced, think, or auto"
    )
    knowledge_bases: Optional[list[str]] = Field(
        default=None, description="Associated knowledge bases"
    )
    tool_categories: Optional[list[str]] = Field(
        default=None, description="Allowed tool categories"
    )
    skills: Optional[list[str]] = Field(default=None, description="Allowed skills")


class ListAgentsToolResult(BaseModel):
    """Result from listing agents."""

    agents: list[AgentInfo] = Field(description="List of agents")
    total_count: int = Field(description="Total number of agents")
    status: str = Field(description="List status")
    message: str = Field(description="Detailed message")


class AgentToolArgs(BaseModel):
    """Arguments for agent tool."""

    task: str = Field(description="The task to delegate to the agent")


class AgentToolResult(BaseModel):
    """Result from agent tool execution."""

    response: str = Field(description="The agent's response")
    file_outputs: Optional[list[dict[str, Any]]] = Field(
        default=None, description="Files generated by the delegated agent"
    )


class ListAvailableSkillsArgs(BaseModel):
    query: Optional[str] = Field(
        default=None, description="Optional search query to filter skills"
    )


class ListAvailableSkillsResult(BaseModel):
    skills: list[str] = Field(description="List of available skill names")


class ListAvailableSkillsTool(AbstractBaseTool):
    """Tool for listing available skills."""

    category: ToolCategory = ToolCategory.AGENT

    def __init__(
        self,
        db: Any = None,
        user_id: Optional[int] = None,
        task_id: Optional[str] = None,
        workspace_base_dir: Optional[str] = None,
    ):
        self._visibility = ToolVisibility.PUBLIC

    @property
    def name(self) -> str:
        return "list_available_skills"

    @property
    def description(self) -> str:
        return "List all available skills that can be assigned to an agent."

    def args_type(self) -> Type[BaseModel]:
        return ListAvailableSkillsArgs

    def return_type(self) -> Type[BaseModel]:
        return ListAvailableSkillsResult

    def run_json_sync(self, args: Mapping[str, Any]) -> Any:
        raise NotImplementedError("Only supports async execution.")

    async def run_json_async(self, args: Mapping[str, Any]) -> Any:
        import os

        skills_dir = os.path.join(
            os.path.dirname(__file__), "../../../../skills/builtin"
        )
        available_skills = []
        if os.path.exists(skills_dir):
            for skill_dir in os.listdir(skills_dir):
                skill_path = os.path.join(skills_dir, skill_dir)
                if os.path.isdir(skill_path):
                    available_skills.append(skill_dir)
        return ListAvailableSkillsResult(skills=available_skills).model_dump()


class ListToolCategoriesArgs(BaseModel):
    query: Optional[str] = Field(
        default=None, description="Optional search query to filter categories"
    )


class ListToolCategoriesResult(BaseModel):
    categories: list[str] = Field(description="List of available tool categories")


class ListToolCategoriesTool(AbstractBaseTool):
    """Tool for listing available tool categories."""

    category: ToolCategory = ToolCategory.AGENT

    def __init__(
        self,
        db: Any = None,
        user_id: Optional[int] = None,
        task_id: Optional[str] = None,
        workspace_base_dir: Optional[str] = None,
    ):
        self._visibility = ToolVisibility.PUBLIC

    @property
    def name(self) -> str:
        return "list_tool_categories"

    @property
    def description(self) -> str:
        return "List all available tool categories that can be assigned to an agent."

    def args_type(self) -> Type[BaseModel]:
        return ListToolCategoriesArgs

    def return_type(self) -> Type[BaseModel]:
        return ListToolCategoriesResult

    def run_json_sync(self, args: Mapping[str, Any]) -> Any:
        raise NotImplementedError("Only supports async execution.")

    async def run_json_async(self, args: Mapping[str, Any]) -> Any:
        return ListToolCategoriesResult(
            categories=_assignable_tool_categories()
        ).model_dump()


class CreateAgentTool(AbstractBaseTool):
    """
    Tool for creating a new draft agent during task execution.

    This allows agents to dynamically create new agents with specific capabilities
    by defining their name, instructions, and allowed tools/skills.
    """

    # Agent tools belong to the AGENT category
    category: ToolCategory = ToolCategory.AGENT

    def __init__(
        self,
        session_factory: Any,
        user_id: int,
        task_id: Optional[str] = None,
        workspace_base_dir: Optional[str] = None,
    ):
        """
        Initialize the create agent tool.

        Args:
            session_factory: Callable that returns a new SQLAlchemy session; a
                fresh session is opened per call and closed in ``finally``.
            user_id: User ID for ownership and model access
            task_id: Task ID for context
            workspace_base_dir: Base directory for workspace files
        """
        self._session_factory = session_factory
        self._user_id = user_id
        self._task_id = task_id
        if workspace_base_dir is None:
            workspace_base_dir = str(get_uploads_dir())
        self._workspace_base_dir = workspace_base_dir
        self._visibility = ToolVisibility.PUBLIC

    @property
    def name(self) -> str:
        """Tool name."""
        return "create_agent"

    @property
    def description(self) -> str:
        """Tool description."""
        # Get available skills (from builtin skills directory)
        import os

        skills_dir = os.path.join(
            os.path.dirname(__file__), "../../../../skills/builtin"
        )
        available_skills = []
        if os.path.exists(skills_dir):
            for skill_dir in os.listdir(skills_dir):
                skill_path = os.path.join(skills_dir, skill_dir)
                if os.path.isdir(skill_path):
                    available_skills.append(skill_dir)

        skills_list = ", ".join(available_skills) if available_skills else "none"
        categories_list = ", ".join(_assignable_tool_categories())

        return (
            "Create a new agent with specific capabilities during task execution. "
            "The agent will be created in DRAFT status and can be called immediately using the returned tool name.\n\n"
            "Parameters:\n"
            "- name: A short, descriptive name for the agent (e.g., 'researcher', 'data_analyzer')\n"
            "- description: IMPORTANT - Clear description of when to use this agent (e.g., 'Use this agent for data analysis tasks involving CSV files'). This helps users understand the agent's purpose. Write this in the same natural language as the current output language policy unless the user explicitly asks for another language.\n"
            f"- tool_categories (optional): Available categories: {categories_list}\n"
            f"  Example: ['file', 'knowledge', 'basic']\n"
            f"- knowledge_bases (optional): List of knowledge base names or IDs to link to this agent.\n"
            "  Only pass knowledge bases that already exist and are visible. "
            "If the requested knowledge base is missing, ask the user for a URL, "
            "file upload, or existing knowledge base choice before calling this tool.\n"
            f"- skills (optional): Available skills: {skills_list}\n"
            f"  Example: ['presentation-generator', 'evidence-based-rag']\n"
            "- instructions: System prompt/instructions defining the agent's behavior and expertise. Write persisted user-facing prose in the same natural language as the current output language policy unless the user explicitly asks for another language. Do not inherit another language from DAG step text, dependency results, tool results, source documents, retrieved memories, examples, or earlier turns.\n"
            "- execution_mode (optional): 'flash', 'balanced' (default), 'think', or 'auto'\n\n"
            "Returns:\n"
            "- agent_id: Database ID of the created agent\n"
            "- agent_name: Name of the agent\n"
            "- tool_name: Tool name that can be used to call this agent\n"
            "- markdown_link: Markdown link in format [Agent Name](agent://agent_id) - USE THIS FORMAT in your response\n"
            "- status: 'success' or 'error'\n"
            "- message: Detailed information about the created agent\n\n"
            "IMPORTANT: Always include the markdown_link in your response when creating an agent successfully. "
            "Use the format: [Agent Name](agent://agent_id). Use plain link syntax only — "
            "NEVER image syntax like ![name](agent://id); agent:// cannot render as an image."
        )

    @property
    def tags(self) -> list[str]:
        """Tool tags."""
        return ["agent", "create"]

    def args_type(self) -> Type[BaseModel]:
        """Argument type."""
        return CreateAgentToolArgs

    def return_type(self) -> Type[BaseModel]:
        """Return type."""
        return CreateAgentToolResult

    def run_json_sync(self, args: Mapping[str, Any]) -> Any:
        """Sync execution not supported."""
        raise NotImplementedError("CreateAgentTool only supports async execution.")

    def _build_name_candidate(self, base_name: str, suffix: str) -> str:
        clean_base = base_name.strip()
        if not suffix:
            return clean_base[:MAX_AGENT_NAME_LENGTH].rstrip()

        max_base_length = max(MAX_AGENT_NAME_LENGTH - len(suffix), 0)
        truncated_base = clean_base[:max_base_length].rstrip()
        if truncated_base:
            return f"{truncated_base}{suffix}"
        return suffix.strip()[:MAX_AGENT_NAME_LENGTH]

    def _resolve_available_agent_name(
        self, requested_name: str, db: Any
    ) -> tuple[str, Optional[str]]:
        from .....web.models.agent import Agent
        from .....web.services.agent_team_scope import (
            get_agent_team_scope,
            owned_agent_clause,
        )

        normalized_name = requested_name.strip()[:MAX_AGENT_NAME_LENGTH].rstrip()

        existing_names = {
            name
            for (name,) in db.query(Agent.name)
            .filter(
                owned_agent_clause(
                    self._user_id, get_agent_team_scope(db, self._user_id)
                )
            )
            .all()
        }

        if normalized_name not in existing_names:
            return normalized_name, None

        preferred_suffixes = [" Assistant", " V2", " Bot", " Workspace"]
        seen_candidates = {normalized_name}

        for suffix in preferred_suffixes:
            candidate = self._build_name_candidate(normalized_name, suffix)
            if candidate not in seen_candidates and candidate not in existing_names:
                return candidate, normalized_name
            seen_candidates.add(candidate)

        for index in range(2, 1000):
            candidate = self._build_name_candidate(normalized_name, f" {index}")
            if candidate not in seen_candidates and candidate not in existing_names:
                return candidate, normalized_name
            seen_candidates.add(candidate)

        fallback_candidate = self._build_name_candidate(
            normalized_name, f" {uuid4().hex[:8]}"
        )
        return fallback_candidate, normalized_name

    async def run_json_async(self, args: Mapping[str, Any]) -> Any:
        """Create a new agent with the given configuration."""
        from .....web.models.agent import AgentStatus
        from .....web.models.user import UserDefaultModel, UserModel
        from .db_session import tool_session_scope

        with tool_session_scope(self._session_factory) as db:
            try:
                agent_name = args.get("name", "").strip()
                agent_description = args.get("description", "").strip()
                instructions = args.get("instructions", "").strip()

                if not agent_name:
                    return CreateAgentToolResult(
                        agent_id=0,
                        agent_name="",
                        tool_name="",
                        markdown_link="",
                        status="error",
                        message="Error: Agent name is required",
                    ).model_dump()

                if not agent_description:
                    return CreateAgentToolResult(
                        agent_id=0,
                        agent_name="",
                        tool_name="",
                        markdown_link="",
                        status="error",
                        message="Error: Agent description is required. Please describe when to use this agent.",
                    ).model_dump()

                if not instructions:
                    return CreateAgentToolResult(
                        agent_id=0,
                        agent_name="",
                        tool_name="",
                        markdown_link="",
                        status="error",
                        message="Error: Agent instructions are required",
                    ).model_dump()

                requested_agent_name = agent_name
                agent_name, auto_renamed_from = self._resolve_available_agent_name(
                    requested_agent_name, db
                )

                # Get user's default model configuration
                from .....web.models.model import Model as DBModel

                user_defaults = (
                    db.query(UserDefaultModel)
                    .join(DBModel, UserDefaultModel.model_id == DBModel.id)
                    .filter(
                        UserDefaultModel.user_id == self._user_id,
                        DBModel.is_active,
                    )
                    .all()
                )

                # Prepare models configuration
                models_config = {}
                for default in user_defaults:
                    if default.config_type in [
                        "general",
                        "small_fast",
                        "visual",
                        "compact",
                    ]:
                        if default.model:
                            try:
                                if not _is_model_visible_to_user(
                                    db, default.model.id, self._user_id
                                ):
                                    continue
                            except Exception:
                                pass
                        models_config[default.config_type] = default.model_id

                missing_types = [
                    t
                    for t in ["general", "small_fast", "visual", "compact"]
                    if t not in models_config
                ]
                if missing_types:
                    # Fill missing with visible users' shared defaults
                    visible_ids = _get_visible_user_ids(db, self._user_id)
                    admin_defaults = (
                        db.query(UserDefaultModel)
                        .join(
                            UserModel, UserDefaultModel.model_id == UserModel.model_id
                        )
                        .filter(
                            UserDefaultModel.config_type.in_(missing_types),
                            UserModel.is_shared.is_(True),
                            UserDefaultModel.user_id.in_(visible_ids),
                        )
                        .all()
                    )
                    for admin_default in admin_defaults:
                        if admin_default.config_type not in models_config:
                            models_config[admin_default.config_type] = (
                                admin_default.model_id
                            )

                execution_mode = args.get("execution_mode", "balanced")
                if execution_mode not in ["flash", "balanced", "think", "auto"]:
                    execution_mode = "balanced"

                knowledge_bases = ensure_list(args.get("knowledge_bases"))
                missing_kbs = await _missing_knowledge_bases_for_user(
                    knowledge_bases, db, self._user_id
                )
                if missing_kbs:
                    return CreateAgentToolResult(
                        agent_id=0,
                        agent_name="",
                        tool_name="",
                        markdown_link="",
                        status="error",
                        message=(
                            "Error: Knowledge base(s) not found or not visible to this user: "
                            + ", ".join(missing_kbs)
                        ),
                    ).model_dump()

                agent = AgentStore(db).create_agent(
                    user_id=self._user_id,
                    name=agent_name,
                    description=agent_description,
                    instructions=instructions,
                    execution_mode=execution_mode,
                    models=models_config if models_config else None,
                    knowledge_bases=knowledge_bases,
                    skills=ensure_list(args.get("skills")),
                    tool_categories=ensure_list(args.get("tool_categories")),
                    status=AgentStatus.DRAFT,  # Create as DRAFT, not PUBLISHED
                    suggested_prompts=[],
                )

                # Generate the tool name and markdown link
                tool_name = gen_agent_tool_name(agent.id)
                markdown_link = f"[{agent_name}](agent://{agent.id})"

                rename_note = ""
                if auto_renamed_from:
                    rename_note = (
                        f"**Auto-renamed:** Requested name '{auto_renamed_from}' was already in use, "
                        f"so the agent was created as '{agent_name}'.\n\n"
                    )

                logger.info(
                    f"Created DRAFT agent '{agent_name}' (ID: {agent.id}) for user {self._user_id}"
                )

                return CreateAgentToolResult(
                    agent_id=agent.id,
                    agent_name=agent_name,
                    tool_name=tool_name,
                    markdown_link=markdown_link,
                    status="success",
                    message=(
                        f"✅ Agent created successfully\n\n"
                        f"{rename_note}"
                        f"**Agent Details:**\n"
                        f"- Agent ID: {agent.id}\n"
                        f"- Agent Name: {agent_name}\n"
                        f"- Tool Name: {tool_name}\n"
                        f"- Status: DRAFT (unpublished)\n\n"
                        f"**How to use this agent:**\n"
                        f"Include this link in your response: {markdown_link}\n"
                        f"Or use the tool: {tool_name}\n\n"
                        f"*The agent is ready to use and will be displayed as a clickable card.*"
                    ),
                ).model_dump()

            except Exception as e:
                error_msg = f"Error creating agent: {str(e)}"
                logger.error(error_msg, exc_info=True)
                return CreateAgentToolResult(
                    agent_id=0,
                    agent_name="",
                    tool_name="",
                    markdown_link="",
                    status="error",
                    message=error_msg,
                ).model_dump()


class UpdateAgentTool(AbstractBaseTool):
    """
    Tool for updating an existing agent during task execution.

    This allows agents to dynamically update agents with specific capabilities
    by modifying their name, description, instructions, and allowed tools/skills.
    """

    # Agent tools belong to the AGENT category
    category: ToolCategory = ToolCategory.AGENT

    def __init__(
        self,
        session_factory: Any,
        user_id: int,
        task_id: Optional[str] = None,
        workspace_base_dir: Optional[str] = None,
    ):
        """
        Initialize the update agent tool.

        Args:
            session_factory: Callable that returns a new SQLAlchemy session; a
                fresh session is opened per call and closed in ``finally``.
            user_id: User ID for ownership and access control
            task_id: Task ID for context
            workspace_base_dir: Base directory for workspace files
        """
        self._session_factory = session_factory
        self._user_id = user_id
        self._task_id = task_id
        if workspace_base_dir is None:
            workspace_base_dir = str(get_uploads_dir())
        self._workspace_base_dir = workspace_base_dir
        self._visibility = ToolVisibility.PUBLIC

    @property
    def name(self) -> str:
        """Tool name."""
        return "update_agent"

    @property
    def description(self) -> str:
        """Tool description."""
        # Get available skills (from builtin skills directory)
        import os

        skills_dir = os.path.join(
            os.path.dirname(__file__), "../../../../skills/builtin"
        )
        available_skills = []
        if os.path.exists(skills_dir):
            for skill_dir in os.listdir(skills_dir):
                skill_path = os.path.join(skills_dir, skill_dir)
                if os.path.isdir(skill_path):
                    available_skills.append(skill_dir)

        skills_list = ", ".join(available_skills) if available_skills else "none"
        categories_list = ", ".join(_assignable_tool_categories())

        return (
            "Update an existing agent with specific capabilities during task execution. "
            "DRAFT and PUBLISHED agents can both be updated; the agent keeps its current status.\n\n"
            "Parameters:\n"
            "- agent_id: The ID of the agent to update (required)\n"
            "- name (optional): New name for the agent\n"
            "- description (optional): New description of when to use this agent\n"
            f"- tool_categories (optional): Available categories: {categories_list}\n"
            f"  Example: ['file', 'knowledge', 'basic']\n"
            f"- knowledge_bases (optional): New list of knowledge base names or IDs to link to this agent.\n"
            "  Only pass knowledge bases that already exist and are visible. "
            "If the requested knowledge base is missing, ask the user for a URL, "
            "file upload, or existing knowledge base choice before calling this tool.\n"
            f"- skills (optional): Available skills: {skills_list}\n"
            f"  Example: ['presentation-generator', 'evidence-based-rag']\n"
            "- instructions (optional): New system prompt/instructions defining the agent's behavior\n"
            "- execution_mode (optional): 'flash', 'balanced', 'think', or 'auto'\n\n"
            "Returns:\n"
            "- agent_id: Database ID of the updated agent\n"
            "- agent_name: Name of the agent\n"
            "- tool_name: Tool name that can be used to call this agent\n"
            "- markdown_link: Markdown link in format [Agent Name](agent://agent_id)\n"
            "- status: 'success' or 'error'\n"
            "- message: Detailed information about the updated agent\n\n"
            "IMPORTANT: Updating a PUBLISHED agent does not unpublish it. "
            "It remains PUBLISHED with the updated configuration."
        )

    @property
    def tags(self) -> list[str]:
        """Tool tags."""
        return ["agent", "update"]

    def args_type(self) -> Type[BaseModel]:
        """Argument type."""
        return UpdateAgentToolArgs

    def return_type(self) -> Type[BaseModel]:
        """Return type."""
        return UpdateAgentToolResult

    def run_json_sync(self, args: Mapping[str, Any]) -> Any:
        """Sync execution not supported."""
        raise NotImplementedError("UpdateAgentTool only supports async execution.")

    async def run_json_async(self, args: Mapping[str, Any]) -> Any:
        """Update an existing agent with the given configuration."""
        from .....web.models.agent import Agent, AgentStatus
        from .db_session import tool_session_scope

        with tool_session_scope(self._session_factory) as db:
            try:
                agent_id = args.get("agent_id")

                if not agent_id:
                    return UpdateAgentToolResult(
                        agent_id=0,
                        agent_name="",
                        tool_name="",
                        markdown_link="",
                        status="error",
                        message="Error: Agent ID is required",
                    ).model_dump()

                # Find the agent
                query = db.query(Agent).filter(Agent.id == agent_id)
                query = _apply_owned_agent_tool_filters(
                    query,
                    Agent,
                    user_id=self._user_id,
                    db=db,
                )
                agent = query.first()

                if not agent:
                    return UpdateAgentToolResult(
                        agent_id=0,
                        agent_name="",
                        tool_name="",
                        markdown_link="",
                        status="error",
                        message=f"Error: Agent with ID {agent_id} not found",
                    ).model_dump()

                if agent.status == AgentStatus.ARCHIVED:
                    return UpdateAgentToolResult(
                        agent_id=agent_id,
                        agent_name=agent.name,
                        tool_name=gen_agent_tool_name(agent.id),
                        markdown_link=f"[{agent.name}](agent://{agent.id})",
                        status="error",
                        message=(
                            "Error: Archived agents cannot be updated. "
                            f"This agent is {agent.status.value.upper()}."
                        ),
                    ).model_dump()

                # Track changes
                changes = []
                updates: dict[str, Any] = {}

                # Update name if provided
                new_name = args.get("name", "").strip() if args.get("name") else None
                if new_name:
                    # Check for duplicate name (exclude current agent)
                    existing = _apply_owned_agent_tool_filters(
                        db.query(Agent).filter(
                            Agent.name == new_name,
                            Agent.id != agent_id,
                        ),
                        Agent,
                        user_id=self._user_id,
                        db=db,
                    ).first()
                    if existing:
                        return UpdateAgentToolResult(
                            agent_id=0,
                            agent_name="",
                            tool_name="",
                            markdown_link="",
                            status="error",
                            message=f"Error: Agent with name '{new_name}' already exists",
                        ).model_dump()
                    updates["name"] = new_name
                    changes.append(f"name → '{new_name}'")

                # Update description if provided
                new_description = (
                    args.get("description", "").strip()
                    if args.get("description")
                    else None
                )
                if new_description:
                    updates["description"] = new_description
                    changes.append("description updated")

                # Update instructions if provided
                new_instructions = (
                    args.get("instructions", "").strip()
                    if args.get("instructions")
                    else None
                )
                if new_instructions:
                    updates["instructions"] = new_instructions
                    changes.append("instructions updated")

                # Update tool_categories if provided
                new_tool_categories = ensure_list(args.get("tool_categories"))
                if new_tool_categories is not None:
                    updates["tool_categories"] = new_tool_categories
                    changes.append(f"tool_categories → {new_tool_categories}")

                # Update knowledge_bases if provided
                new_knowledge_bases = ensure_list(args.get("knowledge_bases"))
                if new_knowledge_bases is not None:
                    missing_kbs = await _missing_knowledge_bases_for_user(
                        new_knowledge_bases, db, self._user_id
                    )
                    if missing_kbs:
                        return UpdateAgentToolResult(
                            agent_id=0,
                            agent_name="",
                            tool_name="",
                            markdown_link="",
                            status="error",
                            message=(
                                "Error: Knowledge base(s) not found or not visible to this user: "
                                + ", ".join(missing_kbs)
                            ),
                        ).model_dump()
                    updates["knowledge_bases"] = new_knowledge_bases
                    changes.append(f"knowledge_bases → {new_knowledge_bases}")

                # Update skills if provided
                new_skills = ensure_list(args.get("skills"))
                if new_skills is not None:
                    updates["skills"] = new_skills
                    changes.append(f"skills → {new_skills}")

                # Update execution_mode if provided
                new_execution_mode = args.get("execution_mode")
                if new_execution_mode in ["flash", "balanced", "think", "auto"]:
                    updates["execution_mode"] = new_execution_mode
                    changes.append(f"execution_mode → {new_execution_mode}")

                # Check if there were any changes
                if not changes:
                    return UpdateAgentToolResult(
                        agent_id=agent_id,
                        agent_name=agent.name,
                        tool_name=gen_agent_tool_name(agent.id),
                        markdown_link=f"[{agent.name}](agent://{agent.id})",
                        status="success",
                        message=f"ℹ️ No updates were made to agent '{agent.name}' (ID: {agent_id}). "
                        f"Status: {agent.status.value.upper()}. "
                        f"All fields were the same or no values were provided.",
                    ).model_dump()

                agent = (
                    AgentStore(db).update_agent_fields(self._user_id, agent_id, updates)
                    or agent
                )

                # Generate the tool name and markdown link
                agent_name = str(agent.name)
                tool_name = gen_agent_tool_name(agent.id)
                markdown_link = f"[{agent_name}](agent://{agent.id})"

                logger.info(
                    f"Updated {agent.status.value.upper()} agent '{agent_name}' (ID: {agent.id}) for user {self._user_id}: {', '.join(changes)}"
                )

                return UpdateAgentToolResult(
                    agent_id=agent.id,
                    agent_name=agent_name,
                    tool_name=tool_name,
                    markdown_link=markdown_link,
                    status="success",
                    message=(
                        f"✅ Agent updated successfully\n\n"
                        f"**Agent Details:**\n"
                        f"- Agent ID: {agent.id}\n"
                        f"- Agent Name: {agent.name}\n"
                        f"- Tool Name: {tool_name}\n"
                        f"- Status: {agent.status.value.upper()}\n\n"
                        f"**Changes Applied:**\n"
                        + "\n".join(f"- {change}" for change in changes)
                        + f"\n\n**How to use this agent:**\n"
                        f"Include this link in your response: {markdown_link}\n"
                        f"Or use the tool: {tool_name}\n\n"
                        f"*The agent keeps its current publication status and will reflect the updated changes on next execution.*"
                    ),
                ).model_dump()

            except Exception as e:
                error_msg = f"Error updating agent: {str(e)}"
                logger.error(error_msg, exc_info=True)
                return UpdateAgentToolResult(
                    agent_id=0,
                    agent_name="",
                    tool_name="",
                    markdown_link="",
                    status="error",
                    message=error_msg,
                ).model_dump()


class ListAgentsTool(AbstractBaseTool):
    """
    Tool for listing all agents belonging to the current user.

    This allows users to see all their agents, their status, and get agent IDs
    for use with other agent management tools.
    """

    # Agent tools belong to the AGENT category
    category: ToolCategory = ToolCategory.AGENT

    def __init__(
        self,
        session_factory: Any,
        user_id: int,
        task_id: Optional[str] = None,
        workspace_base_dir: Optional[str] = None,
    ):
        """
        Initialize the list agents tool.

        Args:
            session_factory: Callable that returns a new database session per call
            user_id: User ID for filtering user's agents
            task_id: Task ID for context
            workspace_base_dir: Base directory for workspace files
        """
        self._session_factory = session_factory
        self._user_id = user_id
        self._task_id = task_id
        if workspace_base_dir is None:
            workspace_base_dir = str(get_uploads_dir())
        self._workspace_base_dir = workspace_base_dir
        self._visibility = ToolVisibility.PUBLIC

    @property
    def name(self) -> str:
        """Tool name."""
        return "list_agents"

    @property
    def description(self) -> str:
        """Tool description."""
        return (
            "List all agents belonging to the current user with their details.\n\n"
            "Parameters:\n"
            "- status_filter (optional): Filter by agent status: 'draft', 'published', or 'archived'. "
            "If None, shows all agents\n\n"
            "Returns:\n"
            "- agents: List of agent information including:\n"
            "  - agent_id: Database ID (use this for update_agent)\n"
            "  - name: Agent name\n"
            "  - description: When to use this agent\n"
            "  - status: draft, published, or archived\n"
            "  - tool_name: Tool name to call this agent\n"
            "  - markdown_link: Markdown link [Agent Name](agent://id)\n"
            "  - execution_mode: flash, balanced, think, or auto\n"
            "  - tool_categories: Allowed tool categories\n"
            "  - skills: Allowed skills\n"
            "- total_count: Total number of agents\n"
            "- status: 'success' or 'error'\n"
            "- message: Detailed information\n\n"
            "Use this tool to discover available agents and get agent IDs for updating agents."
        )

    @property
    def tags(self) -> list[str]:
        """Tool tags."""
        return ["agent", "list"]

    def args_type(self) -> Type[BaseModel]:
        """Argument type."""
        return ListAgentsToolArgs

    def return_type(self) -> Type[BaseModel]:
        """Return type."""
        return ListAgentsToolResult

    def run_json_sync(self, args: Mapping[str, Any]) -> Any:
        """Sync execution not supported."""
        raise NotImplementedError("ListAgentsTool only supports async execution.")

    async def run_json_async(self, args: Mapping[str, Any]) -> Any:
        """List all agents for the current user."""
        from .....web.models.agent import Agent
        from .db_session import tool_session_scope

        try:
            status_filter = args.get("status_filter", "").strip().lower()
            if status_filter and status_filter not in [
                "draft",
                "published",
                "archived",
            ]:
                return ListAgentsToolResult(
                    agents=[],
                    total_count=0,
                    status="error",
                    message=f"Error: Invalid status_filter '{status_filter}'. Must be 'draft', 'published', or 'archived'",
                ).model_dump()

            with tool_session_scope(self._session_factory) as db:
                # Build query
                query = _apply_owned_agent_tool_filters(
                    db.query(Agent),
                    Agent,
                    user_id=self._user_id,
                    db=db,
                )

                # Apply status filter if provided
                if status_filter:
                    query = query.filter(Agent.status == status_filter)

                # Order by status (draft first) then by name
                agents = query.order_by(Agent.status, Agent.name).all()

                # Build agent info list
                agent_infos = []
                for agent in agents:
                    tool_name = gen_agent_tool_name(agent.id)
                    markdown_link = f"[{agent.name}](agent://{agent.id})"

                    agent_info = AgentInfo(
                        agent_id=agent.id,
                        name=agent.name,
                        description=agent.description or "No description",
                        status=agent.status.value,
                        tool_name=tool_name,
                        markdown_link=markdown_link,
                        execution_mode=agent.execution_mode or "react",
                        knowledge_bases=agent.knowledge_bases,
                        tool_categories=agent.tool_categories,
                        skills=agent.skills if agent.skills else None,
                    )
                    agent_infos.append(agent_info.model_dump())

            total_count = len(agent_infos)
            filter_msg = (
                f" (filtered by status: {status_filter})" if status_filter else ""
            )

            logger.info(
                f"Listed {total_count} agents for user {self._user_id}{filter_msg}"
            )

            return ListAgentsToolResult(
                agents=agent_infos,
                total_count=total_count,
                status="success",
                message=(
                    f"✅ Found {total_count} agent(s){filter_msg}\n\n"
                    f"*DRAFT and PUBLISHED agents can be updated using update_agent. "
                    f"All agents can be called using their tool_name.*"
                ),
            ).model_dump()

        except Exception as e:
            error_msg = f"Error listing agents: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return ListAgentsToolResult(
                agents=[],
                total_count=0,
                status="error",
                message=error_msg,
            ).model_dump()


class AgentTool(AbstractBaseTool):
    """
    Tool that wraps a published agent for execution.

    This allows published agents to be called as tools from other agents.
    """

    # Agent tools belong to the AGENT category
    category: ToolCategory = ToolCategory.AGENT

    def __init__(
        self,
        agent_id: int,
        agent_name: str,
        agent_description: str,
        session_factory: Any,
        user_id: int,
        task_id: Optional[str] = None,
        workspace_base_dir: Optional[str] = None,
        tool_name: Optional[str] = None,
        tool_description: Optional[str] = None,
        extra_system_prompt: Optional[str] = None,
        parent_task_id: Optional[str] = None,
        parent_tracer: Optional[Any] = None,
        agent_call_stack: Optional[list[int]] = None,
        delegation_allowed_agent_ids: Optional[list[int]] = None,
        agent_tool_overrides: Optional[Mapping[Any, Any]] = None,
        enable_global_agent_tools: bool = True,
        delegation_allow_cross_user_agent_ids: bool = False,
        target_allowed_agent_ids: Optional[list[int]] = None,
        target_allow_cross_user_agent_ids: bool = False,
        runtime_metadata: Optional[dict[str, Any]] = None,
        execution_scope: Optional[Any] = None,
    ):
        """
        Initialize an agent tool.

        Args:
            agent_id: The database ID of the published agent
            agent_name: Name of the agent
            agent_description: Description of what this agent does
            session_factory: Session factory minting short-lived DB sessions
                for loading agent config and models
            user_id: User ID for model access
            task_id: Task ID for workspace isolation
            workspace_base_dir: Base directory for workspace files
            tool_name: Deprecated delegated tool name override. Agent tools always
                expose the canonical agent_<id> name.
            tool_description: Optional delegated tool description override
            extra_system_prompt: Optional system prompt appended during execution
            parent_task_id: Parent task ID for delegation metadata
            parent_tracer: Parent tracer that receives delegation summary events
            agent_call_stack: Active delegation stack for recursion prevention
            delegation_allowed_agent_ids: Agent IDs exposed to this delegated agent for nested agent calls
            agent_tool_overrides: Nested delegated agent tool overrides
            enable_global_agent_tools: Whether this delegated agent sees global agent tools
            delegation_allow_cross_user_agent_ids: Whether explicit nested agent IDs may cross users
            target_allowed_agent_ids: Agent IDs this tool may execute as its target
            target_allow_cross_user_agent_ids: Whether this tool may execute explicit cross-user target IDs
            runtime_metadata: Extra delegation metadata for tracing
        """
        self._agent_id = agent_id
        self._agent_name = agent_name
        self._agent_description = agent_description
        self._tool_name = tool_name
        self._tool_description = tool_description
        self._extra_system_prompt = extra_system_prompt
        self._session_factory = session_factory
        self._user_id = user_id
        self._task_id = task_id or f"agent_tool_{agent_id}"
        self._parent_task_id = parent_task_id or task_id
        self._parent_tracer = parent_tracer
        self._delegation_allowed_agent_ids = _normalize_agent_ids(
            delegation_allowed_agent_ids
        )
        self._agent_tool_overrides = _normalize_agent_tool_overrides(
            agent_tool_overrides
        )
        self._enable_global_agent_tools = bool(enable_global_agent_tools)
        self._delegation_allow_cross_user_agent_ids = bool(
            delegation_allow_cross_user_agent_ids
        )
        self._target_allowed_agent_ids = _normalize_agent_ids(target_allowed_agent_ids)
        self._target_allow_cross_user_agent_ids = bool(
            target_allow_cross_user_agent_ids
        )
        self._runtime_metadata = dict(runtime_metadata or {})
        self._agent_call_stack = _normalize_agent_ids(agent_call_stack) or []
        if agent_id not in self._agent_call_stack:
            self._agent_call_stack.append(agent_id)
        if workspace_base_dir is None:
            workspace_base_dir = str(get_uploads_dir())
        self._workspace_base_dir = workspace_base_dir
        # Snapshot the parent turn's ExecutionScope at construction (the
        # resolver already ran for the parent turn; a nested call has no
        # Task row to re-resolve from). An explicitly passed scope wins over
        # the ambient contextvar so builds outside an activated turn
        # (pause/resume paths) still capture the parent's scope.
        from .....core.execution_scope import get_execution_scope

        self._execution_scope = (
            execution_scope if execution_scope is not None else get_execution_scope()
        )
        self._visibility = ToolVisibility.PUBLIC

    @property
    def name(self) -> str:
        """Tool name."""
        return gen_agent_tool_name(self._agent_id)

    @property
    def description(self) -> str:
        """Tool description."""
        return self._tool_description or self._agent_description

    @property
    def tags(self) -> list[str]:
        """Tool tags."""
        return ["agent", "delegation"]

    def args_type(self) -> Type[BaseModel]:
        """Argument type."""
        return AgentToolArgs

    def return_type(self) -> Type[BaseModel]:
        """Return type."""
        return AgentToolResult

    def run_json_sync(self, args: Mapping[str, Any]) -> Any:
        """Sync execution not supported."""
        raise NotImplementedError("AgentTool only supports async execution.")

    def _build_delegation_trace_data(
        self,
        status: str,
        execution_task_id: Optional[str] = None,
        output: Optional[str] = None,
        error: Optional[str] = None,
        file_outputs: Optional[list[dict[str, Any]]] = None,
    ) -> dict[str, Any]:
        data: dict[str, Any] = {
            "event_type": f"workforce_delegation_{status}",
            "status": status,
            "agent_id": self._agent_id,
            "agent_name": self._agent_name,
            "tool_name": self.name,
        }
        data.update(self._runtime_metadata)
        if execution_task_id:
            data["worker_task_id"] = execution_task_id
        if output is not None:
            data["output"] = output[:2000]
            data["output_length"] = len(output)
        if error is not None:
            data["error"] = error
        if file_outputs:
            data["file_outputs"] = file_outputs
        return data

    async def _trace_delegation(
        self,
        status: str,
        execution_task_id: Optional[str] = None,
        output: Optional[str] = None,
        error: Optional[str] = None,
        file_outputs: Optional[list[dict[str, Any]]] = None,
    ) -> None:
        if (
            self._parent_tracer is None
            or self._parent_task_id is None
            or not self._runtime_metadata
        ):
            return

        trace_event = getattr(self._parent_tracer, "trace_event", None)
        if not callable(trace_event):
            return

        try:
            import inspect

            from .....core.agent.trace import (
                TraceAction,
                TraceCategory,
                TraceEventType,
                TraceScope,
            )

            if status not in {"start", "end", "error"}:
                return

            result = trace_event(
                TraceEventType(
                    TraceScope.TASK,
                    TraceAction.UPDATE,
                    TraceCategory.GENERAL,
                ),
                task_id=str(self._parent_task_id),
                data=self._build_delegation_trace_data(
                    status=status,
                    execution_task_id=execution_task_id,
                    output=output,
                    error=error,
                    file_outputs=file_outputs,
                ),
            )
            if inspect.isawaitable(result):
                await result
        except Exception:
            logger.debug("Failed to emit workforce delegation trace", exc_info=True)

    def _create_child_execution_tracer(
        self,
        *,
        execution_task_id: str,
        agent_name: str,
        parent_db_task_id: Optional[int],
    ) -> Any:
        """Create a child-owned tracer for delegated agent internals."""
        metadata = {
            "source": "xagent-agent-tool-child",
            "task_id": execution_task_id,
            "worker_task_id": execution_task_id,
            "parent_task_id": self._parent_task_id or self._task_id,
            "parent_db_task_id": parent_db_task_id,
            "agent_id": self._agent_id,
            "agent_name": agent_name,
            "agent_call_stack": self._agent_call_stack,
        }
        metadata.update(self._runtime_metadata)
        handlers: list[Any] = []
        if parent_db_task_id is not None:
            handlers.append(
                _DelegatedAgentDatabaseTraceHandler(
                    task_id=parent_db_task_id,
                    build_id=execution_task_id,
                    metadata=metadata,
                )
            )

        return create_agent_tracer(
            handlers=handlers,
            task_id=execution_task_id,
            user_id=self._user_id,
            trace_name=f"xagent-agent-tool-{self._agent_id}",
            session_id=str(self._parent_task_id or self._task_id),
            tags=["xagent", "agent-tool", "nested-agent"],
            metadata=metadata,
        )

    def _resolve_delegated_output_path(self, workspace: Any, raw_path: str) -> Path:
        raw = raw_path.strip()
        path = Path(raw)
        if path.is_absolute():
            return Path(workspace.resolve_path(raw))

        first_part = Path(raw).parts[0] if Path(raw).parts else ""
        default_dir = (
            "workspace" if first_part in {"input", "output", "temp"} else "output"
        )
        return Path(workspace.resolve_path(raw, default_dir=default_dir))

    def _parent_owned_file_outputs(
        self, file_outputs: Any, workspace: Any, db: Any
    ) -> Optional[list[dict[str, Any]]]:
        if not isinstance(file_outputs, list):
            return None

        parent_db_task_id = _coerce_db_task_id(self._parent_task_id)
        if parent_db_task_id is None:
            if self._runtime_metadata:
                logger.warning(
                    "Skipping delegated file outputs without a parent DB task id: %s",
                    self._parent_task_id,
                )
                return []
            return file_outputs

        from .....core.file_ref import build_file_ref
        from .....web.models.uploaded_file import UploadedFile

        normalized_outputs: list[dict[str, Any]] = []

        for item in file_outputs:
            item_file_id = ""
            item_filename = ""
            raw_paths: list[str] = []

            if isinstance(item, str):
                raw_paths = [item]
            elif isinstance(item, dict):
                if isinstance(item.get("file_id"), str):
                    item_file_id = str(item["file_id"]).strip()
                if isinstance(item.get("filename"), str):
                    item_filename = str(item["filename"]).strip()
                for key in ("file_path", "download_path", "relative_path", "path"):
                    value = item.get(key)
                    if isinstance(value, str) and value.strip():
                        raw_paths.append(value)
            else:
                continue

            file_record = None
            if item_file_id:
                file_record = (
                    db.query(UploadedFile)
                    .filter(
                        UploadedFile.file_id == item_file_id,
                        UploadedFile.user_id == self._user_id,
                        UploadedFile.task_id == parent_db_task_id,
                    )
                    .first()
                )

            if file_record is None and workspace is not None:
                for raw_path in raw_paths:
                    try:
                        resolved_path = self._resolve_delegated_output_path(
                            workspace, raw_path
                        )
                    except (FileNotFoundError, ValueError):
                        logger.debug(
                            "Failed to resolve delegated file output: %s",
                            raw_path,
                            exc_info=True,
                        )
                        continue

                    if not resolved_path.exists() or not resolved_path.is_file():
                        continue

                    try:
                        registered_file_id = workspace.register_file(
                            str(resolved_path),
                            db_session=db,
                        )
                    except (FileNotFoundError, ValueError):
                        logger.debug(
                            "Failed to register delegated file output: %s",
                            raw_path,
                            exc_info=True,
                        )
                        continue

                    file_record = (
                        db.query(UploadedFile)
                        .filter(
                            UploadedFile.file_id == registered_file_id,
                            UploadedFile.user_id == self._user_id,
                            UploadedFile.task_id == parent_db_task_id,
                        )
                        .first()
                    )
                    if file_record is not None:
                        break

            if file_record is None:
                logger.warning("Skipping unregistered delegated file output: %s", item)
                continue

            normalized_outputs.append(
                build_file_ref(
                    file_id=str(file_record.file_id),
                    filename=item_filename or str(file_record.filename),
                    mime_type=getattr(file_record, "mime_type", None),
                    size=getattr(file_record, "file_size", None),
                )
            )

        return normalized_outputs

    async def run_json_async(self, args: Mapping[str, Any]) -> Any:
        """Execute the agent with the given task."""
        from contextlib import nullcontext

        from .....core.execution_scope import ExecutionScopeContext
        from .....web.models.agent import Agent
        from .....web.user_isolated_memory import UserContext
        from .db_session import tool_session_scope

        execution_task_id: Optional[str] = None
        try:
            from .....core.agent.service import AgentService
            from .....web.services.llm_utils import UserAwareModelStorage

            # ---- Phase 1: load agent config + resolve models in a
            # short-lived session that is CLOSED before the sub-agent runs.
            # The sub-agent has its own ReAct/DAG loop that may issue
            # concurrent tool calls; holding the parent session across it is
            # the core hazard, so we capture everything Phase 2/3 needs into
            # plain locals while the session is open and never touch the
            # detached ORM instance afterwards.
            default_llm = None
            fast_llm = None
            vision_llm = None
            compact_llm = None
            agent_name = ""
            agent_instructions = None
            agent_knowledge_bases = None
            agent_skills = None
            agent_tool_categories = None
            agent_models = None
            agent_execution_mode = None

            with tool_session_scope(self._session_factory) as db:
                # Load agent from database - support both PUBLISHED and DRAFT
                agent = db.query(Agent).filter(
                    Agent.id == self._agent_id,
                    Agent.status.in_(["published", "draft"]),  # type: ignore[attr-defined]
                )
                agent = _apply_agent_visibility_filters(
                    agent,
                    Agent,
                    user_id=self._user_id,
                    allowed_agent_ids=self._target_allowed_agent_ids,
                    allow_cross_user_agent_ids=self._target_allow_cross_user_agent_ids,
                    db=db,
                )
                agent = agent.first() if agent is not None else None

                if not agent:
                    error_msg = f"Error: Agent {self._agent_id} not found"
                    await self._trace_delegation("error", error=error_msg)
                    return AgentToolResult(response=error_msg).model_dump(
                        exclude_none=True
                    )

                # Generate unique task ID for this execution
                execution_task_id = f"agent_{self._agent_id}_{uuid4().hex[:8]}"
                await self._trace_delegation(
                    "start", execution_task_id=execution_task_id
                )

                # Capture agent fields as plain locals before the session
                # closes (avoid detached-ORM access in Phase 2/3).
                agent_name = agent.name
                agent_instructions = agent.instructions
                agent_knowledge_bases = agent.knowledge_bases
                agent_skills = agent.skills
                agent_tool_categories = agent.tool_categories
                agent_models = agent.models
                agent_execution_mode = agent.execution_mode

                # Resolve models
                storage = UserAwareModelStorage(db)

                if agent_models:
                    from .agent_model_resolution import resolve_agent_model_llms

                    default_llm, fast_llm, vision_llm, compact_llm = (
                        resolve_agent_model_llms(
                            db, storage, agent_models, self._user_id
                        )
                    )
            # ---- Phase 1 session closed here. ----

            if not default_llm:
                error_msg = f"Error: No valid model configured for agent {agent_name}"
                await self._trace_delegation(
                    "error", execution_task_id=execution_task_id, error=error_msg
                )
                return AgentToolResult(response=error_msg).model_dump(exclude_none=True)

            # ---- Phase 2: build the child config with the FACTORY (no live
            # parent session) and run the sub-agent. ----

            # Create tool config with allowed collections, skills, and tools
            class MinimalRequest:
                def __init__(self, user_id: int):
                    self.user = type("obj", (), {"id": user_id})()

            # Delegated-agent tool selection goes through the shared
            # ``ToolSelectionSpec.from_raw`` normalizer.
            # ``None`` → _SpecAll (unconfigured legacy agent, all built-in
            #            tools) — but WITHOUT the user-level Custom API
            #            registry: a delegated agent may only call Custom
            #            APIs it explicitly selected via an ``mcp:<server>``
            #            connector, never inherit every API the *user* has
            #            configured (issue #798).
            # ``[]``  → _SpecNone (agent explicitly saved with zero tools).
            # Non-empty list → _SpecByCategories (scoped to declared tools).
            from .selection_spec import (
                ToolSelectionSpec,
                should_load_mcp_server_configs,
            )

            tool_selection_spec = ToolSelectionSpec.from_raw(
                tool_categories=agent_tool_categories,
                exclude_custom_api_when_unconfigured=True,
            )

            parent_db_task_id = _coerce_db_task_id(self._parent_task_id)
            _scope_segments = (
                self._execution_scope.workspace_segments
                if self._execution_scope is not None
                else ()
            )
            tool_config = WebToolConfig(
                db=None,
                db_factory=self._session_factory,
                request=MinimalRequest(self._user_id),
                user_id=self._user_id,
                allowed_collections=agent_knowledge_bases
                if agent_knowledge_bases is not None
                else None,
                allowed_skills=agent_skills,
                tool_selection_spec=tool_selection_spec,
                # Keep MCP config loading aligned with the direct chat path.
                # _SpecAll still admits MCP at the final filter layer for
                # compatibility, but it does not opt into MCP server init.
                include_mcp_tools=should_load_mcp_server_configs(tool_selection_spec),
                allowed_agent_ids=self._delegation_allowed_agent_ids,
                agent_tool_overrides=self._agent_tool_overrides,
                enable_global_agent_tools=self._enable_global_agent_tools,
                allow_cross_user_agent_ids=self._delegation_allow_cross_user_agent_ids,
                parent_task_id=self._parent_task_id,
                parent_tracer=self._parent_tracer,
                agent_call_stack=self._agent_call_stack,
                task_id=execution_task_id,
                workspace_config={
                    "base_dir": self._workspace_base_dir,
                    "task_id": execution_task_id,
                    "db_task_id": parent_db_task_id,
                    "scope_segments": _scope_segments,
                },
                execution_scope=self._execution_scope,
            )

            try:
                tracer = self._create_child_execution_tracer(
                    execution_task_id=execution_task_id,
                    agent_name=str(agent_name),
                    parent_db_task_id=parent_db_task_id,
                )

                # Create agent service. Memory stays disabled for sub-agent
                # runs: these are stateless one-shot executors, and any store
                # we hand them here would be discarded when the call returns —
                # the memory tools would report success while the note
                # silently evaporates (the pre-tool pipeline had the same
                # flaw, writing its end-of-run evaluation into a throwaway
                # InMemoryMemoryStore). Wire the parent's real store through
                # deliberately if sub-agent memory is ever wanted.
                agent_service = AgentService(
                    name=agent_name,
                    llm=default_llm,
                    fast_llm=fast_llm,
                    vision_llm=vision_llm,
                    compact_llm=compact_llm,
                    memory_enabled=False,
                    tool_config=tool_config,
                    pattern=get_agent_pattern_for_execution_mode(agent_execution_mode),
                    id=execution_task_id,
                    enable_workspace=True,
                    workspace_base_dir=self._workspace_base_dir,
                    scope_segments=_scope_segments,
                    task_id=execution_task_id,
                    tracer=tracer,
                )

                # Build execution context
                execution_context: dict[str, Any] = {}
                system_prompts = []
                if agent_instructions:
                    system_prompts.append(agent_instructions)
                if self._extra_system_prompt:
                    system_prompts.append(self._extra_system_prompt)
                if system_prompts:
                    execution_context["system_prompt"] = "\n\n".join(system_prompts)

                # Execute task. Re-activate the parent turn's scope snapshot
                # around the nested run (no re-resolution — the nested call
                # has no Task row), so the sub-agent's memory/workspace
                # operations land in the parent's scope. A None snapshot
                # leaves the ambient turn context untouched.
                scope_context = (
                    ExecutionScopeContext(self._execution_scope)
                    if self._execution_scope is not None
                    else nullcontext()
                )
                with UserContext(self._user_id), scope_context:
                    result = await agent_service.execute_task(
                        task=args["task"],
                        context=execution_context if execution_context else None,
                        task_id=execution_task_id,
                    )
            finally:
                tool_config.close()

            # ---- Phase 3: post-run file outputs in a fresh short-lived
            # session. Registering delegated outputs only flushes into the
            # session, so this owned session commits before closing to make
            # the parent-owned file records durable (the parent session used
            # to carry these to the request-level commit).
            output = result.get("output", "No response generated")
            with tool_session_scope(self._session_factory) as db:
                file_outputs = self._parent_owned_file_outputs(
                    result.get("file_outputs"), agent_service.workspace, db
                )
                db.commit()
            file_outputs = file_outputs if isinstance(file_outputs, list) else None
            logger.info(
                f"Agent tool {self.name} executed successfully, output length: {len(output)}"
            )
            await self._trace_delegation(
                "end",
                execution_task_id=execution_task_id,
                output=str(output),
                file_outputs=file_outputs,
            )
            return AgentToolResult(
                response=output,
                file_outputs=file_outputs,
            ).model_dump(exclude_none=True)

        except Exception as e:
            error_msg = f"Error executing agent {self._agent_id}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            await self._trace_delegation(
                "error", execution_task_id=execution_task_id, error=error_msg
            )
            return AgentToolResult(response=error_msg).model_dump(exclude_none=True)


def get_published_agents_tools(
    session_factory: Any,
    user_id: int,
    task_id: Optional[str] = None,
    workspace_base_dir: Optional[str] = None,
    excluded_agent_id: Optional[int] = None,
    include_draft: bool = False,
    draft_agent_ids_to_include: Optional[list[int]] = None,
    allowed_agent_ids: Optional[list[int]] = None,
    agent_tool_overrides: Optional[Mapping[Any, Any]] = None,
    enable_global_agent_tools: bool = True,
    allow_cross_user_agent_ids: bool = False,
    parent_task_id: Optional[str] = None,
    parent_tracer: Optional[Any] = None,
    agent_call_stack: Optional[list[int]] = None,
    execution_scope: Optional[Any] = None,
) -> list[AbstractBaseTool]:
    """
    Get tools for published (and optionally draft) agents.

    Args:
        session_factory: Session factory minting short-lived DB sessions
        user_id: User ID for model access
        task_id: Task ID for workspace isolation
        workspace_base_dir: Base directory for workspace files
        excluded_agent_id: Optional agent ID to exclude (to prevent self-calls)
        include_draft: Whether to include DRAFT agents (useful for dynamically created agents)
        draft_agent_ids_to_include: Specific DRAFT agent IDs to include (for agents created in current task)
        allowed_agent_ids: Explicit agent IDs that are allowed to be injected
        agent_tool_overrides: Per-agent tool metadata/runtime overrides
        enable_global_agent_tools: Whether to include globally visible published agents
        allow_cross_user_agent_ids: Whether explicit allowed IDs may cross users
        parent_task_id: Parent task ID for delegation metadata
        parent_tracer: Parent tracer for delegation summary events
        agent_call_stack: Active delegation stack for recursion prevention

    Returns:
        List of AgentTool instances
    """
    from .....config import get_uploads_dir
    from .....web.models.agent import Agent, AgentStatus
    from .db_session import tool_session_scope

    if workspace_base_dir is None:
        workspace_base_dir = str(get_uploads_dir())

    tools: list[AbstractBaseTool] = []
    normalized_overrides = _normalize_agent_tool_overrides(agent_tool_overrides)
    normalized_injected_agent_ids = _normalize_agent_ids(allowed_agent_ids)
    normalized_call_stack = _normalize_agent_ids(agent_call_stack) or []
    excluded_agent_ids = set(normalized_call_stack)

    if excluded_agent_id is not None:
        try:
            excluded_agent_ids.add(int(excluded_agent_id))
        except (TypeError, ValueError):
            pass

    if (
        normalized_injected_agent_ids is None
        and not enable_global_agent_tools
        and normalized_overrides
    ):
        normalized_injected_agent_ids = list(normalized_overrides.keys())

    if normalized_injected_agent_ids is not None:
        normalized_injected_agent_ids = [
            agent_id
            for agent_id in normalized_injected_agent_ids
            if agent_id not in excluded_agent_ids
        ]

    try:
        with tool_session_scope(session_factory) as db:
            if normalized_injected_agent_ids is not None:
                if not normalized_injected_agent_ids:
                    return []
                query = db.query(Agent).filter(
                    Agent.status.in_(["published"]),  # type: ignore[attr-defined]
                )
                query = _apply_agent_visibility_filters(
                    query,
                    Agent,
                    user_id=user_id,
                    allowed_agent_ids=normalized_injected_agent_ids,
                    allow_cross_user_agent_ids=allow_cross_user_agent_ids,
                    db=db,
                )
                if query is None:
                    return []
            elif not enable_global_agent_tools:
                return []
            elif include_draft:
                # Include both PUBLISHED and DRAFT agents
                query = db.query(Agent).filter(
                    Agent.status.in_(["published", "draft"]),  # type: ignore[attr-defined]
                )
                query = _apply_owned_agent_tool_filters(
                    query, Agent, user_id=user_id, db=db
                )
            else:
                # Only PUBLISHED agents
                query = db.query(Agent).filter(
                    Agent.status == "published",
                )
                query = _apply_owned_agent_tool_filters(
                    query, Agent, user_id=user_id, db=db
                )

            # Exclude the active delegation stack to prevent recursive self-calls.
            if excluded_agent_ids:
                query = query.filter(Agent.id.notin_(sorted(excluded_agent_ids)))

            agents = query.all()

            # If specific DRAFT agents should be included, add them
            if draft_agent_ids_to_include:
                normalized_draft_agent_ids = _normalize_agent_ids(
                    draft_agent_ids_to_include
                )
                normalized_draft_agent_ids = [
                    agent_id
                    for agent_id in (normalized_draft_agent_ids or [])
                    if agent_id not in excluded_agent_ids
                ]
                # Skip the query entirely when nothing survives the exclusion
                # filter, so we never issue an empty ``IN ()`` predicate.
                if normalized_draft_agent_ids:
                    draft_agents = _apply_owned_agent_tool_filters(
                        db.query(Agent).filter(
                            Agent.id.in_(normalized_draft_agent_ids),
                            Agent.status == "draft",
                        ),
                        Agent,
                        user_id=user_id,
                        db=db,
                    ).all()
                    # Merge without duplicates
                    existing_ids = {agent.id for agent in agents}
                    for draft_agent in draft_agents:
                        if draft_agent.id not in existing_ids:
                            agents.append(draft_agent)

            if normalized_injected_agent_ids is not None:
                agent_types = "selected PUBLISHED"
            else:
                agent_types = "PUBLISHED and DRAFT" if include_draft else "PUBLISHED"
            logger.info(
                f"Found {len(agents)} {agent_types} agents (excluded: {sorted(excluded_agent_ids)})"
            )

            for agent in agents:
                override = normalized_overrides.get(int(agent.id or 0), {})
                # Build description
                description = agent.description or f"Call {agent.name} agent"
                if agent.instructions:
                    # Add brief instructions to description
                    instructions_preview = agent.instructions[:200]
                    if len(agent.instructions) > 200:
                        instructions_preview += "..."
                    description += f". Instructions: {instructions_preview}"

                # Add status indicator for draft agents
                if agent.status == AgentStatus.DRAFT:
                    description = f"[DRAFT] {description}"

                tool_name = _string_override(override, "tool_name")
                tool_description = _string_override(
                    override, "description", "tool_description"
                )
                extra_system_prompt = _string_override(override, "extra_system_prompt")
                delegation_allowed_agent_ids = (
                    _normalize_agent_ids(override.get("allowed_agent_ids"))
                    if "allowed_agent_ids" in override
                    else None
                )
                delegation_agent_tool_overrides = _normalize_agent_tool_overrides(
                    override.get("agent_tool_overrides")
                )
                delegation_enable_global_agent_tools = _truthy_bool(
                    override.get("enable_global_agent_tools"), True
                )
                delegation_allow_cross_user_agent_ids = _truthy_bool(
                    override.get("allow_cross_user_agent_ids"), False
                )
                runtime_metadata = {
                    key: override[key]
                    for key in (
                        "workforce_run_id",
                        "workforce_id",
                        "workforce_name",
                        "worker_member_id",
                        "worker_alias",
                    )
                    if key in override
                }

                tool = AgentTool(
                    agent_id=agent.id,
                    agent_name=agent.name,
                    agent_description=tool_description or description,
                    session_factory=session_factory,
                    user_id=user_id,
                    task_id=task_id,
                    workspace_base_dir=workspace_base_dir,
                    tool_name=tool_name,
                    tool_description=tool_description,
                    extra_system_prompt=extra_system_prompt,
                    parent_task_id=parent_task_id,
                    parent_tracer=parent_tracer,
                    agent_call_stack=normalized_call_stack,
                    delegation_allowed_agent_ids=delegation_allowed_agent_ids,
                    agent_tool_overrides=delegation_agent_tool_overrides,
                    enable_global_agent_tools=delegation_enable_global_agent_tools,
                    delegation_allow_cross_user_agent_ids=delegation_allow_cross_user_agent_ids,
                    target_allowed_agent_ids=normalized_injected_agent_ids,
                    target_allow_cross_user_agent_ids=allow_cross_user_agent_ids,
                    runtime_metadata=runtime_metadata,
                    execution_scope=execution_scope,
                )
                tools.append(tool)
                logger.debug(f"Created agent tool: {tool.name}")

    except Exception as e:
        logger.error(f"Failed to load agents as tools: {e}", exc_info=True)

    return tools


# Register tool creator for auto-discovery
# Import at bottom to avoid circular import with factory
from .factory import register_tool  # noqa: E402

if TYPE_CHECKING:
    from xagent.web.tools.config import WebToolConfig


@register_tool(categories={"agent"}, selection_gate="published_agent")
async def create_agent_tools(config: "WebToolConfig") -> list[AbstractBaseTool]:
    """Create tools from published agents.

    Internal short-circuit on
    ``ToolSelectionSpec.includes_published_agent()`` skips the
    ``get_published_agents_tools`` DB enumeration when the spec sets
    ``published_agent_ids`` to an empty frozenset (explicit "no
    delegation"). Registry-level skip via ``categories={"agent"}``
    handles the case where the spec's category set doesn't include
    ``"agent"`` at all.
    """
    spec = (
        config.get_tool_selection_spec()
        if hasattr(config, "get_tool_selection_spec")
        else None
    )
    if spec is not None and not spec.includes_published_agent():
        return []
    if not config.get_enable_agent_tools():
        return []

    try:
        session_factory = config.get_session_factory()
        user_id = config.get_user_id()
        if not user_id:
            return []

        excluded_agent_id = config.get_excluded_agent_id() if config else None

        return get_published_agents_tools(
            session_factory=session_factory,
            user_id=user_id,
            task_id=config.get_task_id(),
            workspace_base_dir=None,  # Will use get_uploads_dir() default
            excluded_agent_id=excluded_agent_id,
            include_draft=False,  # Only PUBLISHED agents by default
            allowed_agent_ids=config.get_allowed_agent_ids(),
            agent_tool_overrides=config.get_agent_tool_overrides(),
            enable_global_agent_tools=config.get_enable_global_agent_tools(),
            allow_cross_user_agent_ids=config.get_allow_cross_user_agent_ids(),
            parent_task_id=config.get_parent_task_id(),
            parent_tracer=config.get_parent_tracer(),
            agent_call_stack=config.get_agent_call_stack(),
            execution_scope=config.get_execution_scope()
            if hasattr(config, "get_execution_scope")
            else None,
        )
    except Exception as e:
        logger.warning(f"Failed to create agent tools: {e}")
        return []


def _agent_management_tools_enabled(config: "WebToolConfig") -> bool:
    return config.get_enable_agent_tools() and config.get_enable_global_agent_tools()


@register_tool(categories={"agent"})
async def create_create_agent_tool(config: "WebToolConfig") -> list[AbstractBaseTool]:
    """Create the CreateAgentTool for dynamically creating agents."""
    if not _agent_management_tools_enabled(config):
        return []

    try:
        factory = config.get_session_factory()
        user_id = config.get_user_id()
        if not user_id:
            return []

        tool = CreateAgentTool(
            session_factory=factory,
            user_id=user_id,
            task_id=config.get_task_id(),
        )

        list_skills_tool = ListAvailableSkillsTool(
            user_id=user_id,
            task_id=config.get_task_id(),
        )

        list_categories_tool = ListToolCategoriesTool(
            user_id=user_id,
            task_id=config.get_task_id(),
        )

        logger.debug(
            f"Created CreateAgentTool and related list tools for user {user_id}"
        )
        return [tool, list_skills_tool, list_categories_tool]
    except Exception as e:
        logger.warning(f"Failed to create CreateAgentTool: {e}")
        return []


@register_tool(categories={"agent"})
async def create_update_agent_tool(config: "WebToolConfig") -> list[AbstractBaseTool]:
    """Create the UpdateAgentTool for dynamically updating agents."""
    if not _agent_management_tools_enabled(config):
        return []

    try:
        factory = config.get_session_factory()
        user_id = config.get_user_id()
        if not user_id:
            return []

        tool = UpdateAgentTool(
            session_factory=factory,
            user_id=user_id,
            task_id=config.get_task_id(),
            workspace_base_dir=None,  # Will use get_uploads_dir() default
        )
        logger.debug(f"Created UpdateAgentTool for user {user_id}")
        return [tool]
    except Exception as e:
        logger.warning(f"Failed to create UpdateAgentTool: {e}")
        return []


@register_tool(categories={"agent"})
async def create_list_agents_tool(config: "WebToolConfig") -> list[AbstractBaseTool]:
    """Create the ListAgentsTool for listing user's agents."""
    if not _agent_management_tools_enabled(config):
        return []

    try:
        factory = config.get_session_factory()
        user_id = config.get_user_id()
        if not user_id:
            return []

        tool = ListAgentsTool(
            session_factory=factory,
            user_id=user_id,
            task_id=config.get_task_id(),
            workspace_base_dir=None,  # Will use get_uploads_dir() default
        )
        logger.debug(f"Created ListAgentsTool for user {user_id}")
        return [tool]
    except Exception as e:
        logger.warning(f"Failed to create ListAgentsTool: {e}")
        return []
