"""
Agent Tool - Convert published agents into callable tools
"""

import logging
from typing import TYPE_CHECKING, Any, Mapping, Optional, Type

from pydantic import BaseModel, Field

from .base import AbstractBaseTool, ToolCategory, ToolVisibility

logger = logging.getLogger(__name__)


class CreateAgentToolArgs(BaseModel):
    """Arguments for creating a new agent."""

    name: str = Field(description="Name of the agent to create")
    description: str = Field(
        description="IMPORTANT: Description of when to use this agent (e.g., 'Use this agent for data analysis tasks involving CSV files'). This helps users understand the agent's purpose and when to call it."
    )
    instructions: str = Field(description="System instructions/prompt for the agent")
    tool_categories: Optional[list[str]] = Field(
        default=None,
        description="List of tool categories to allow (e.g., ['file', 'knowledge']). If None, all tools are available",
    )
    skills: Optional[list[str]] = Field(
        default=None,
        description="List of skill names to allow. If None, all skills are available",
    )


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


class AgentToolArgs(BaseModel):
    """Arguments for agent tool."""

    task: str = Field(description="The task to delegate to the agent")


class AgentToolResult(BaseModel):
    """Result from agent tool execution."""

    response: str = Field(description="The agent's response")


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
        db: Any,
        user_id: int,
        task_id: Optional[str] = None,
        workspace_base_dir: str = "uploads",
    ):
        """
        Initialize the create agent tool.

        Args:
            db: Database session for saving the agent
            user_id: User ID for ownership and model access
            task_id: Task ID for context
            workspace_base_dir: Base directory for workspace files
        """
        self._db = db
        self._user_id = user_id
        self._task_id = task_id
        self._workspace_base_dir = workspace_base_dir
        self._visibility = ToolVisibility.PUBLIC

    @property
    def name(self) -> str:
        """Tool name."""
        return "create_agent"

    @property
    def description(self) -> str:
        """Tool description."""
        # Get available tool categories
        from .base import ToolCategory

        available_categories = [cat.value for cat in ToolCategory]

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
        categories_list = ", ".join(available_categories)

        return (
            "Create a new agent with specific capabilities during task execution. "
            "The agent will be created in DRAFT status and can be called immediately using the returned tool name.\n\n"
            "Parameters:\n"
            "- name: A short, descriptive name for the agent (e.g., 'researcher', 'data_analyzer')\n"
            "- description: IMPORTANT - Clear description of when to use this agent (e.g., 'Use this agent for data analysis tasks involving CSV files'). This helps users understand the agent's purpose.\n"
            f"- tool_categories (optional): Available categories: {categories_list}\n"
            f"  Example: ['file', 'knowledge', 'basic']\n"
            f"- skills (optional): Available skills: {skills_list}\n"
            f"  Example: ['presentation-generator', 'poster-design']\n"
            "- instructions: System prompt/instructions defining the agent's behavior and expertise\n\n"
            "Returns:\n"
            "- agent_id: Database ID of the created agent\n"
            "- agent_name: Name of the agent\n"
            "- tool_name: Tool name that can be used to call this agent\n"
            "- markdown_link: Markdown link in format [Agent Name](agent://agent_id) - USE THIS FORMAT in your response\n"
            "- status: 'success' or 'error'\n"
            "- message: Detailed information about the created agent\n\n"
            "IMPORTANT: Always include the markdown_link in your response when creating an agent successfully. "
            "Use the format: [Agent Name](agent://agent_id)"
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

    async def run_json_async(self, args: Mapping[str, Any]) -> Any:
        """Create a new agent with the given configuration."""
        from .....web.models.agent import Agent, AgentStatus
        from .....web.services.llm_utils import UserAwareModelStorage

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

            # Check for duplicate name
            existing = (
                self._db.query(Agent)
                .filter(Agent.user_id == self._user_id, Agent.name == agent_name)
                .first()
            )
            if existing:
                return CreateAgentToolResult(
                    agent_id=0,
                    agent_name="",
                    tool_name="",
                    markdown_link="",
                    status="error",
                    message=f"Error: Agent with name '{agent_name}' already exists",
                ).model_dump()

            # Get user's default model configuration
            storage = UserAwareModelStorage(self._db)
            default_llm, fast_llm, vision_llm, compact_llm = (
                storage.get_configured_defaults(self._user_id)
            )

            # Prepare models configuration
            models_config = {}
            if default_llm:
                models_config["general"] = (
                    default_llm.model_id if hasattr(default_llm, "model_id") else None
                )
            if fast_llm:
                models_config["small_fast"] = (
                    fast_llm.model_id if hasattr(fast_llm, "model_id") else None
                )
            if vision_llm:
                models_config["visual"] = (
                    vision_llm.model_id if hasattr(vision_llm, "model_id") else None
                )
            if compact_llm:
                models_config["compact"] = (
                    compact_llm.model_id if hasattr(compact_llm, "model_id") else None
                )

            # Create the agent in DRAFT status
            agent = Agent(
                user_id=self._user_id,
                name=agent_name,
                description=agent_description,
                instructions=instructions,
                execution_mode="graph",
                models=models_config if models_config else None,
                knowledge_bases=None,  # No KB by default
                skills=args.get("skills"),
                tool_categories=args.get("tool_categories"),
                suggested_prompts=[],
                status=AgentStatus.DRAFT,  # Create as DRAFT, not PUBLISHED
            )

            self._db.add(agent)
            self._db.commit()
            self._db.refresh(agent)

            # Generate the tool name and markdown link
            tool_name = gen_agent_tool_name(agent_name)
            markdown_link = f"[{agent_name}](agent://{agent.id})"

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
        db: Any,
        user_id: int,
        task_id: Optional[str] = None,
        workspace_base_dir: str = "uploads",
    ):
        """
        Initialize an agent tool.

        Args:
            agent_id: The database ID of the published agent
            agent_name: Name of the agent
            agent_description: Description of what this agent does
            db: Database session for loading agent config and models
            user_id: User ID for model access
            task_id: Task ID for workspace isolation
            workspace_base_dir: Base directory for workspace files
        """
        self._agent_id = agent_id
        self._agent_name = agent_name
        self._agent_description = agent_description
        self._db = db
        self._user_id = user_id
        self._task_id = task_id or f"agent_tool_{agent_id}"
        self._workspace_base_dir = workspace_base_dir
        self._visibility = ToolVisibility.PUBLIC

    @property
    def name(self) -> str:
        """Tool name."""
        return f"call_agent_{self._agent_name.lower().replace(' ', '_')}"

    @property
    def description(self) -> str:
        """Tool description."""
        return self._agent_description

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

    async def run_json_async(self, args: Mapping[str, Any]) -> Any:
        """Execute the agent with the given task."""
        import uuid

        from .....web.models.agent import Agent
        from .....web.tools.config import WebToolConfig
        from .....web.user_isolated_memory import UserContext

        try:
            # Load agent from database - support both PUBLISHED and DRAFT
            agent = (
                self._db.query(Agent)
                .filter(
                    Agent.id == self._agent_id,
                    Agent.status.in_(["published", "draft"]),  # type: ignore[attr-defined]
                )
                .first()
            )

            if not agent:
                return AgentToolResult(
                    response=f"Error: Agent {self._agent_id} not found"
                ).model_dump()

            # Generate unique task ID for this execution
            execution_task_id = f"agent_{self._agent_id}_{uuid.uuid4().hex[:8]}"

            # Resolve models
            from .....core.agent.service import AgentService
            from .....core.memory.in_memory import InMemoryMemoryStore
            from .....web.services.llm_utils import UserAwareModelStorage

            storage = UserAwareModelStorage(self._db)
            default_llm = None
            fast_llm = None
            vision_llm = None
            compact_llm = None

            if agent.models:
                from .....web.models.model import Model as DBModel

                if agent.models.get("general"):
                    general_model = (
                        self._db.query(DBModel)
                        .filter(DBModel.id == agent.models["general"])
                        .first()
                    )
                    if general_model:
                        default_llm = storage.get_llm_by_name_with_access(
                            str(general_model.model_id), self._user_id
                        )

                if agent.models.get("small_fast"):
                    fast_model = (
                        self._db.query(DBModel)
                        .filter(DBModel.id == agent.models["small_fast"])
                        .first()
                    )
                    if fast_model:
                        fast_llm = storage.get_llm_by_name_with_access(
                            str(fast_model.model_id), self._user_id
                        )

                if agent.models.get("visual"):
                    visual_model = (
                        self._db.query(DBModel)
                        .filter(DBModel.id == agent.models["visual"])
                        .first()
                    )
                    if visual_model:
                        vision_llm = storage.get_llm_by_name_with_access(
                            str(visual_model.model_id), self._user_id
                        )

                if agent.models.get("compact"):
                    compact_model = (
                        self._db.query(DBModel)
                        .filter(DBModel.id == agent.models["compact"])
                        .first()
                    )
                    if compact_model:
                        compact_llm = storage.get_llm_by_name_with_access(
                            str(compact_model.model_id), self._user_id
                        )

            if not default_llm:
                return AgentToolResult(
                    response=f"Error: No valid model configured for agent {agent.name}"
                ).model_dump()

            # Create tool config with allowed collections, skills, and tools
            class MinimalRequest:
                def __init__(self, user_id: int):
                    self.user = type("obj", (), {"id": user_id})()

            allowed_tools = None
            if agent.tool_categories is not None:
                from .factory import ToolFactory

                temp_config = WebToolConfig(
                    db=self._db,
                    request=MinimalRequest(self._user_id),
                    user_id=self._user_id,
                    include_mcp_tools=False,
                    browser_tools_enabled=True,
                )
                all_tools = await ToolFactory.create_all_tools(temp_config)
                allowed_tools = []
                for tool in all_tools:
                    if hasattr(tool, "metadata") and hasattr(tool.metadata, "category"):
                        category = str(tool.metadata.category.value)
                        if category in agent.tool_categories:
                            tool_name = getattr(tool, "name", None)
                            if tool_name:
                                allowed_tools.append(tool_name)

            tool_config = WebToolConfig(
                db=self._db,
                request=MinimalRequest(self._user_id),
                user_id=self._user_id,
                allowed_collections=agent.knowledge_bases
                if agent.knowledge_bases is not None
                else None,
                allowed_skills=agent.skills if agent.skills is not None else None,
                allowed_tools=allowed_tools,
                task_id=execution_task_id,
                workspace_base_dir=self._workspace_base_dir,
            )

            # Create agent service
            memory = InMemoryMemoryStore()
            agent_service = AgentService(
                name=agent.name,
                llm=default_llm,
                fast_llm=fast_llm,
                vision_llm=vision_llm,
                compact_llm=compact_llm,
                memory=memory,
                tool_config=tool_config,
                use_dag_pattern=True,
                id=execution_task_id,
                enable_workspace=True,
                workspace_base_dir=self._workspace_base_dir,
                task_id=execution_task_id,
                tracer=None,
            )

            # Build execution context
            execution_context: dict[str, Any] = {}
            if agent.instructions:
                execution_context["system_prompt"] = agent.instructions

            # Execute task
            with UserContext(self._user_id):
                result = await agent_service.execute_task(
                    task=args["task"],
                    context=execution_context if execution_context else None,
                    task_id=execution_task_id,
                )

            output = result.get("output", "No response generated")
            logger.info(
                f"Agent tool {self.name} executed successfully, output length: {len(output)}"
            )
            return AgentToolResult(response=output).model_dump()

        except Exception as e:
            error_msg = f"Error executing agent {self._agent_id}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return AgentToolResult(response=error_msg).model_dump()


def gen_agent_tool_name(agent_name: str) -> str:
    """
    Generate the tool name for a published agent.

    This is a centralized function to ensure consistent naming across the codebase.
    Tool name format: call_agent_{agent_name_lower_with_underscores}

    Args:
        agent_name: The name of the agent

    Returns:
        The tool name that will be used for this agent
    """
    return f"call_agent_{agent_name.lower().replace(' ', '_')}"


def get_published_agents_tools(
    db: Any,
    user_id: int,
    task_id: Optional[str] = None,
    workspace_base_dir: str = "uploads",
    excluded_agent_id: Optional[int] = None,
    include_draft: bool = False,
    draft_agent_ids_to_include: Optional[list[int]] = None,
) -> list[AbstractBaseTool]:
    """
    Get tools for published (and optionally draft) agents.

    Args:
        db: Database session
        user_id: User ID for model access
        task_id: Task ID for workspace isolation
        workspace_base_dir: Base directory for workspace files
        excluded_agent_id: Optional agent ID to exclude (to prevent self-calls)
        include_draft: Whether to include DRAFT agents (useful for dynamically created agents)
        draft_agent_ids_to_include: Specific DRAFT agent IDs to include (for agents created in current task)

    Returns:
        List of AgentTool instances
    """
    from .....web.models.agent import Agent, AgentStatus

    tools: list[AbstractBaseTool] = []

    try:
        # Query agents - include both PUBLISHED and optionally DRAFT
        if include_draft:
            # Include both PUBLISHED and DRAFT agents
            query = db.query(Agent).filter(
                Agent.user_id == user_id,
                Agent.status.in_(["published", "draft"]),  # type: ignore[attr-defined]
            )
        else:
            # Only PUBLISHED agents
            query = db.query(Agent).filter(
                Agent.status == "published",
                Agent.user_id == user_id,
            )

        # Exclude the specified agent (to prevent self-calls)
        if excluded_agent_id is not None:
            query = query.filter(Agent.id != excluded_agent_id)

        agents = query.all()

        # If specific DRAFT agents should be included, add them
        if draft_agent_ids_to_include:
            draft_agents = (
                db.query(Agent)
                .filter(
                    Agent.id.in_(draft_agent_ids_to_include),
                    Agent.user_id == user_id,
                    Agent.status == "draft",
                )
                .all()
            )
            # Merge without duplicates
            existing_ids = {agent.id for agent in agents}
            for draft_agent in draft_agents:
                if draft_agent.id not in existing_ids:
                    agents.append(draft_agent)

        agent_types = "PUBLISHED and DRAFT" if include_draft else "PUBLISHED"
        logger.info(
            f"Found {len(agents)} {agent_types} agents (excluded: {excluded_agent_id})"
        )

        for agent in agents:
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

            tool = AgentTool(
                agent_id=agent.id,
                agent_name=agent.name,
                agent_description=description,
                db=db,
                user_id=user_id,
                task_id=task_id,
                workspace_base_dir=workspace_base_dir,
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


@register_tool
async def create_agent_tools(config: "WebToolConfig") -> list[AbstractBaseTool]:
    """Create tools from published agents."""
    if not config.get_enable_agent_tools():
        return []

    try:
        db = config.get_db()
        user_id = config.get_user_id()
        if not user_id:
            return []

        excluded_agent_id = config.get_excluded_agent_id() if config else None

        # Only include PUBLISHED agents by default
        # DRAFT agents are only available within the same task context after creation
        return get_published_agents_tools(
            db=db,
            user_id=user_id,
            task_id=config.get_task_id(),
            workspace_base_dir="uploads",
            excluded_agent_id=excluded_agent_id,
            include_draft=False,  # Only PUBLISHED agents
        )
    except Exception as e:
        logger.warning(f"Failed to create agent tools: {e}")
        return []


@register_tool
async def create_create_agent_tool(config: "WebToolConfig") -> list[AbstractBaseTool]:
    """Create the CreateAgentTool for dynamically creating agents."""
    if not config.get_enable_agent_tools():
        return []

    try:
        db = config.get_db()
        user_id = config.get_user_id()
        if not user_id:
            return []

        tool = CreateAgentTool(
            db=db,
            user_id=user_id,
            task_id=config.get_task_id(),
            workspace_base_dir="uploads",
        )
        logger.debug(f"Created CreateAgentTool for user {user_id}")
        return [tool]
    except Exception as e:
        logger.warning(f"Failed to create CreateAgentTool: {e}")
        return []
