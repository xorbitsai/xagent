"""Chat API route handlers"""

import asyncio
import logging
import os
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ...core.agent.service import AgentService
from ...core.agent.trace import Tracer
from ...core.model.chat.basic.base import BaseLLM
from ...core.model.chat.basic.openai import OpenAILLM
from ...core.model.chat.basic.zhipu import ZhipuLLM
from ..auth_dependencies import get_current_user
from ..config import ALLOWED_EXTERNAL_UPLOAD_DIRS, UPLOADS_DIR
from ..dynamic_memory_store import get_memory_store
from ..models.agent import Agent
from ..models.database import get_db
from ..models.model import Model as DBModel
from ..models.task import AgentType, Task, TaskStatus
from ..models.user import User
from ..schemas.chat import TaskCreateRequest, TaskCreateResponse
from ..services.llm_utils import resolve_llms_from_names
from ..user_isolated_memory import UserContext
from ..utils.db_timezone import format_datetime_for_api, safe_timestamp_to_unix
from .trace_handlers import DatabaseTraceHandler
from .ws_trace_handlers import WebSocketTraceHandler

logger = logging.getLogger(__name__)

# Create router
chat_router = APIRouter(prefix="/api/chat", tags=["chat"])


def create_default_llm() -> Optional[BaseLLM]:
    """Create a default LLM instance based on environment configuration"""
    try:
        # For OpenAI: allow empty string API key (use is not None check)
        # For Zhipu: don't allow empty string API key (use truthy check)
        openai_api_key = os.getenv("OPENAI_API_KEY")
        zhipu_api_key = os.getenv("ZHIPU_API_KEY")

        # Similarly for base_url: prefer OPENAI_BASE_URL if it exists (even if empty string)
        # Only fallback to ZHIPU_BASE_URL if OPENAI_BASE_URL is None
        openai_base_url = os.getenv("OPENAI_BASE_URL")
        zhipu_base_url = os.getenv("ZHIPU_BASE_URL")

        # For model_name: prefer OPENAI_MODEL if it exists (even if empty string)
        # Only fallback to ZHIPU_MODEL_NAME if OPENAI_MODEL is None
        openai_model = os.getenv("OPENAI_MODEL")
        zhipu_model = os.getenv("ZHIPU_MODEL_NAME")

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
        elif openai_api_key is not None:
            logger.info(f"Using OpenAI LLM with model: {openai_model}")
            return OpenAILLM(
                model_name=openai_model or "gpt-4o-mini",
                base_url=openai_base_url,
                api_key=openai_api_key,
            )

        # No LLM available - AgentService will run without DAG pattern
        logger.error(
            "No API key found in environment variables. Set OPENAI_API_KEY or ZHIPU_API_KEY to enable LLM functionality."
        )
        return None

    except Exception as e:
        logger.error(f"Failed to create default LLM: {e}")
        return None


async def create_default_tools(
    db: Session,
    request: Any = None,
    user: Optional[User] = None,
    task_id: Optional[str] = None,
    allowed_collections: Optional[List[str]] = None,
    allowed_skills: Optional[List[str]] = None,
    allowed_tools: Optional[List[str]] = None,
    excluded_agent_id: Optional[int] = None,
) -> tuple[list[Any], Any]:
    """Create default tools and tool_config for AgentService using ToolFactory"""
    if not user:
        raise ValueError("User is required for tool creation")
    if not task_id:
        raise ValueError("Task ID is required for tool creation")

    # Create a WebToolConfig to properly initialize tools
    from ..tools.config import WebToolConfig

    tool_config = WebToolConfig(
        db=db,
        request=request,
        user_id=int(user.id),
        is_admin=bool(user.is_admin),
        workspace_config={
            "base_dir": str(UPLOADS_DIR / f"user_{user.id}"),
            "task_id": task_id,
        },
        include_mcp_tools=False,  # Exclude MCP tools for default agent
        task_id=task_id,  # Pass task_id for browser session tracking
        browser_tools_enabled=True,  # Enable browser automation tools
        allowed_collections=allowed_collections,  # Agent Builder knowledge bases
        allowed_skills=allowed_skills,  # Agent Builder skills
        allowed_tools=allowed_tools,  # Agent Builder tool categories
    )

    # Store excluded_agent_id in tool_config for agent tool filtering
    if excluded_agent_id:
        tool_config._excluded_agent_id = excluded_agent_id

    from ...core.tools.adapters.vibe.factory import ToolFactory

    # Use ToolFactory to create proper xagent tools
    tools = await ToolFactory.create_all_tools(tool_config)  # type: ignore[arg-type]

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
        self._default_llm = create_default_llm()
        self.request = request

    def set_task_llms(
        self, task_id: int, llm_names: Optional[List[Optional[str]]], db: Session
    ) -> None:
        """Set LLM configuration for a specific task (configuration now stored in Task table)"""
        logger.info(
            f"set_task_llms called for task {task_id} with llm_names: {llm_names}"
        )
        # Configuration is now stored in Task table, this method is kept for backward compatibility
        # If AgentService already exists, update its LLM configuration
        if task_id in self._agents:
            # This method doesn't have user context, use None for user_id
            default_llm, fast_llm, vision_llm, compact_llm = resolve_llms_from_names(
                llm_names, db, None
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

    def _load_agent_builder_config(
        self, agent: Agent, db: Session, user_id: int
    ) -> dict:
        """Load all Agent Builder configuration.

        Returns dict with:
        - llms: (default_llm, fast_llm, vision_llm, compact_llm)
        - execution_mode: str
        - instructions: str (system prompt)
        - skills: List[str]
        - knowledge_bases: List[str]
        - tool_categories: List[str]
        """
        from ..services.llm_utils import UserAwareModelStorage

        storage = UserAwareModelStorage(db)

        default_llm = None
        fast_llm = None
        vision_llm = None
        compact_llm = None

        if agent.models:
            if agent.models.get("general"):
                general_model = (
                    db.query(DBModel)
                    .filter(DBModel.id == agent.models["general"])
                    .first()
                )
                if general_model:
                    default_llm = storage.get_llm_by_name_with_access(
                        str(general_model.model_id), user_id
                    )

            if agent.models.get("small_fast"):
                fast_model = (
                    db.query(DBModel)
                    .filter(DBModel.id == agent.models["small_fast"])
                    .first()
                )
                if fast_model:
                    fast_llm = storage.get_llm_by_name_with_access(
                        str(fast_model.model_id), user_id
                    )

            if agent.models.get("visual"):
                visual_model = (
                    db.query(DBModel)
                    .filter(DBModel.id == agent.models["visual"])
                    .first()
                )
                if visual_model:
                    vision_llm = storage.get_llm_by_name_with_access(
                        str(visual_model.model_id), user_id
                    )

            if agent.models.get("compact"):
                compact_model = (
                    db.query(DBModel)
                    .filter(DBModel.id == agent.models["compact"])
                    .first()
                )
                if compact_model:
                    compact_llm = storage.get_llm_by_name_with_access(
                        str(compact_model.model_id), user_id
                    )

        return {
            "llms": (default_llm, fast_llm, vision_llm, compact_llm),
            "execution_mode": agent.execution_mode,
            "instructions": agent.instructions,  # System prompt
            "skills": agent.skills or [],
            "knowledge_bases": agent.knowledge_bases or [],
            "tool_categories": agent.tool_categories or [],
        }

    async def get_agent_for_task(
        self, task_id: int, db: Optional[Session] = None, user: Optional[User] = None
    ) -> AgentService:
        """Get or create AgentService instance for specific task"""
        if task_id not in self._agents:
            # Check if task exists in database
            task_exists = False
            task = None
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
                # Task exists in database, try to reconstruct from history
                if db is not None:
                    try:
                        await self._reconstruct_agent_from_history(task_id, db)
                        return self._agents[task_id]
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
                        # Continue with normal agent creation

            # Create tracer with all necessary handlers
            tracer = Tracer()
            # Add console handler for logging
            from ...core.agent.trace import ConsoleTraceHandler

            tracer.add_handler(ConsoleTraceHandler())
            # Add database handler for persistence
            tracer.add_handler(DatabaseTraceHandler(task_id))
            # Add WebSocket handler for real-time updates
            tracer.add_handler(WebSocketTraceHandler(task_id))

            # Get LLM configuration from task database record
            logger.info(f"Loading LLM configuration for task {task_id} from database")
            agent_config = None  # Initialize agent_config to use later
            use_dag = True  # Default to DAG pattern
            try:
                if db is None:
                    raise ValueError("Database session is required")

                task = db.query(Task).filter(Task.id == task_id).first()
                if task:
                    # Log the actual task record for debugging
                    logger.info(
                        f"Task {task_id} record: agent_type={task.agent_type}, model_name={task.model_name}, compact_model_name={task.compact_model_name}"
                    )

                    # Check if this is a Text2SQL task
                    logger.info(
                        f"Task {task.id} agent_type: '{task.agent_type}', agent_type_enum: '{task.agent_type_enum}'"
                    )
                    if task.agent_type_enum == AgentType.TEXT2SQL:
                        logger.info(f"🎯 Creating Text2SQL agent for task {task.id}")
                        if user is not None:
                            return await self._create_text2sql_agent(
                                task, db, user, tracer
                            )
                        else:
                            raise ValueError(
                                "User context is required for Text2SQL agent creation"
                            )
                    else:
                        logger.info(
                            f"❌ Task {task.id} is not Text2SQL, using standard agent creation"
                        )

                    # Load configuration from task record - check for any configured models
                    llm_names: List[Optional[str]] = [
                        str(task.model_name) if task.model_name else None,
                        str(task.small_fast_model_name)
                        if task.small_fast_model_name
                        else None,
                        str(task.visual_model_name) if task.visual_model_name else None,
                        str(task.compact_model_name)
                        if task.compact_model_name
                        else None,
                    ]
                    logger.info(
                        f"Loading LLM configuration from task {task_id}: {llm_names}"
                    )
                    # Use user_id for model resolution if available
                    user_id_for_resolution: Optional[int] = (
                        int(user.id) if user and user.id is not None else None
                    )
                    task_llm, task_fast_llm, task_vision_llm, task_compact_llm = (
                        resolve_llms_from_names(llm_names, db, user_id_for_resolution)
                    )

                    # Override with Agent Builder configuration if task.agent_id exists
                    if task and task.agent_id and user:
                        agent = (
                            db.query(Agent).filter(Agent.id == task.agent_id).first()
                        )
                        if agent:
                            logger.info(
                                f"Task {task_id} using Agent Builder config: {agent.name}"
                            )
                            agent_config = self._load_agent_builder_config(
                                agent, db, int(user.id)
                            )
                            (
                                task_llm,
                                task_fast_llm,
                                task_vision_llm,
                                task_compact_llm,
                            ) = agent_config["llms"]
                            use_dag = agent_config["execution_mode"] == "graph"
                            logger.info(
                                f"Task {task_id} using execution mode: {agent.execution_mode}"
                            )

                    # If no models were resolved, use defaults
                    if not task_llm:
                        logger.warning(
                            f"Task {task_id} has no valid LLM configuration, using defaults"
                        )
                        task_llm = self._default_llm

                    logger.info(
                        f"Successfully loaded LLM configuration for task {task_id}: compact_llm={task_compact_llm.model_name if task_compact_llm else None}"
                    )
                else:
                    # Task record not found
                    logger.error(f"Task {task_id} not found in database!")
                    task_llm = self._default_llm
                    task_fast_llm = None
                    task_vision_llm = None
                    task_compact_llm = None
            except Exception as e:
                logger.error(
                    f"Failed to load LLM configuration from task {task_id} database: {e}"
                )
                # Fallback to defaults
                task_llm = self._default_llm
                task_fast_llm = None
                task_vision_llm = None
                task_compact_llm = None
            llm_info = "database LLM configuration"

            try:
                # Set user context for memory operations during agent creation
                if user is None:
                    raise ValueError("User context is required for agent creation")

                if not db:
                    raise ValueError(
                        "Database connection is required for agent creation"
                    )

                # Check if task has an associated published agent that should be excluded from agent tools
                excluded_agent_id = None
                if task and task.agent_id:
                    # Get the current agent to check if it's published
                    from ..models.agent import AgentStatus

                    current_agent = (
                        db.query(Agent).filter(Agent.id == task.agent_id).first()
                    )
                    if current_agent and current_agent.status == AgentStatus.PUBLISHED:
                        excluded_agent_id = int(current_agent.id)
                        logger.info(
                            f"Task {task_id} is associated with published agent {current_agent.id} ({current_agent.name}), will exclude from agent tools"
                        )

                # Filter tools by tool category using tool name to category mapping
                # Note: Tool names are stable, defined in code, no database storage needed
                allowed_tools = None
                if agent_config and agent_config.get("tool_categories"):
                    tool_categories = agent_config["tool_categories"]

                    # Tool name to category mapping (hardcoded from code)
                    TOOL_CATEGORY_MAP = {
                        "vision": ["understand_images"],
                        "image": ["generate_image", "edit_image"],
                        "knowledge": ["search_knowledge_base", "list_knowledge_bases"],
                        "file": [
                            "read_file",
                            "write_file",
                            "list_directory",
                            "file_search",
                        ],
                        "basic": ["calculator", "web_search", "zhipu_web_search"],
                        "browser": [
                            "browser_navigate",
                            "browser_click",
                            "browser_fill",
                            "browser_extract",
                        ],
                    }

                    allowed_tools = []
                    for category in tool_categories:
                        if category in TOOL_CATEGORY_MAP:
                            allowed_tools.extend(TOOL_CATEGORY_MAP[category])

                    logger.info(
                        f"🔧 Tool categories {tool_categories} mapped to {len(allowed_tools)} tools for task {task_id}"
                    )

                # Create tools using ToolFactory
                tools = await create_default_tools(
                    db,
                    request=self.request,
                    user=user,
                    task_id=f"web_task_{task_id}",
                    allowed_collections=agent_config["knowledge_bases"]
                    if agent_config
                    else None,
                    allowed_skills=agent_config["skills"] if agent_config else None,
                    allowed_tools=allowed_tools,
                    excluded_agent_id=excluded_agent_id,
                )

                with UserContext(int(user.id)):
                    # Extract Text2SQL configuration if this is a Text2SQL task
                    agent_kwargs = {}
                    if task and task.agent_type == "text2sql" and task.agent_config:
                        config: dict[str, Any] = (
                            task.agent_config
                            if isinstance(task.agent_config, dict)
                            else {}
                        )
                        logger.info(
                            f"🔧 Extracting Text2SQL config: {list(config.keys())}"
                        )
                        agent_kwargs.update(
                            {
                                "database_url": config.get("database_url"),
                                "database_name": config.get("database_name"),
                                "database_type": config.get("database_type"),
                                "schema_info": config.get("schema_info"),
                                "max_iterations": config.get("max_iterations", 3),
                                "read_only": config.get("read_only", True),
                                "available_tables": config.get("available_tables"),
                            }
                        )
                        logger.info(
                            f"✅ Text2SQL kwargs prepared: {list(agent_kwargs.keys())}"
                        )
                        logger.info(
                            f"🔗 Database URL: {config.get('database_url', 'NOT FOUND')}"
                        )

                    # Unpack tools and tool_config from create_default_tools
                    tools_list, tool_config = tools

                    # Get system prompt from agent config (if available)
                    system_prompt = (
                        agent_config.get("instructions") if agent_config else None
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
                        memory=get_memory_store(),  # Use dynamic memory store for auto-switching
                        use_dag_pattern=use_dag,
                        tracer=tracer,
                        agent_type=str(task.agent_type)
                        if task and task.agent_type
                        else "standard",
                        enable_workspace=True,  # Enable workspace functionality
                        workspace_base_dir=str(
                            UPLOADS_DIR / f"user_{user.id}"
                        ),  # Use user-isolated base directory
                        task_id=str(task_id),  # Pass task_id for proper tracing
                        memory_similarity_threshold=None,  # Will be set from task config
                        system_prompt=system_prompt,  # Pass agent builder instructions
                        **agent_kwargs,  # Pass Text2SQL-specific parameters
                    )

                pattern_info = (
                    f"with DAG pattern and workspace using {llm_info}"
                    if use_dag
                    else "with workspace (no LLM configured)"
                )
                logger.info(
                    f"Created new AgentService for task {task_id} {pattern_info}"
                )

            except Exception as e:
                logger.error(f"Failed to create AgentService for task {task_id}: {e}")
                # Re-raise the exception - no fallback logic allowed
                raise

        return self._agents[task_id]

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
            logger.info(f"Removed AgentService for task {task_id}")
        else:
            # If agent is not in memory, clean up workspace directory directly
            self._cleanup_workspace_directory(task_id, user_id)

        # LLM configuration is now stored in Task table, no need to clean up memory storage

    async def execute_task(
        self,
        agent_service: "AgentService",
        task: str,
        context: Optional[Dict[str, Any]] = None,
        task_id: Optional[str] = None,
        db_session: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Execute task with automatic token tracking.

        This method wraps the agent's execute_task with token tracking if db_session is provided.

        Args:
            agent_service: The AgentService instance to use
            task: Task description
            context: Optional context data
            task_id: Optional task identifier for tracking
            db_session: Optional database session for token tracking

        Returns:
            Execution result dictionary
        """
        # Initialize tracker if db_session and task_id are provided
        tracker = None
        if db_session and task_id:
            try:
                from ..tracking.task_tracker import TaskTracker

                tracker = TaskTracker(task_id=int(task_id), db_session=db_session)
                await tracker.start_tracking()
                logger.info(f"Started token tracking for task {task_id}")
            except Exception as e:
                logger.warning(
                    f"Failed to start token tracking for task {task_id}: {e}"
                )
                tracker = None

        try:
            logger.info(
                f"=== About to execute task: task_id={task_id}, has_db_session={db_session is not None} ==="
            )

            # Execute the task
            result = await agent_service.execute_task(
                task=task, context=context, task_id=task_id
            )

            logger.info("=== Task executed successfully, updating title if needed ===")

            # Update task title with generated task_name (clean architecture: Core provides API, Web handles DB)
            if db_session and task_id and result and result.get("success"):
                await update_task_title_from_agent(
                    agent_service, int(task_id), db_session
                )

            return result
        finally:
            # Complete tracking if it was started
            if tracker:
                try:
                    await tracker.complete_tracking()
                    logger.info(f"Completed token tracking for task {task_id}")
                except Exception as e:
                    logger.error(
                        f"Failed to complete token tracking for task {task_id}: {e}"
                    )

    def _cleanup_workspace_directory(
        self, task_id: int, user_id: Optional[int] = None
    ) -> None:
        """Clean up workspace directory for a task when agent is not in memory"""
        from ...core.workspace import TaskWorkspace

        # Try user-isolated workspace first, then fallback
        workspace_ids = []
        if user_id:
            workspace_ids.append(
                (f"web_task_{task_id}", str(UPLOADS_DIR / f"user_{user_id}"))
            )
        workspace_ids.append((f"web_task_{task_id}", str(UPLOADS_DIR)))

        # Build allowed external directories (user's upload directory for knowledge base files)
        allowed_external_dirs = []
        if user_id:
            user_upload_dir = UPLOADS_DIR / f"user_{user_id}"
            if user_upload_dir.exists():
                allowed_external_dirs.append(str(user_upload_dir))
                logger.info(
                    f"Added user upload directory to allowed external dirs: {user_upload_dir}"
                )

        # Add configured external upload directories (for knowledge base files from other projects)
        allowed_external_dirs.extend([str(d) for d in ALLOWED_EXTERNAL_UPLOAD_DIRS])
        if ALLOWED_EXTERNAL_UPLOAD_DIRS:
            logger.info(
                f"Added {len(ALLOWED_EXTERNAL_UPLOAD_DIRS)} external upload directories from config"
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

    async def _create_text2sql_agent(
        self, task: Task, db: Session, user: User, tracer: Tracer
    ) -> AgentService:
        """Create Text2SQL agent service"""

        try:
            # Extract Text2SQL configuration from task
            if not task.agent_config:
                raise ValueError(
                    "Text2SQL agent requires database configuration but agent_config is empty"
                )

            if not isinstance(task.agent_config, dict):
                raise ValueError(
                    f"Text2SQL agent_config must be a dictionary, got {type(task.agent_config)}"
                )

            config = task.agent_config

            # Validate required configuration
            if not config.get("database_url"):
                raise ValueError(
                    "Text2SQL agent configuration must include 'database_url'"
                )

            # Log configuration for debugging
            logger.info(
                f"Creating Text2SQL agent for task {task.id} with config: {config}"
            )

            # Get LLM configuration
            llm_names: List[Optional[str]] = [
                str(task.model_name) if task.model_name else None,
                str(task.small_fast_model_name) if task.small_fast_model_name else None,
                str(task.visual_model_name) if task.visual_model_name else None,
                str(task.compact_model_name) if task.compact_model_name else None,
            ]

            # Use user_id for model resolution if available
            user_id_for_resolution = int(user.id) if user else None
            task_llm, task_fast_llm, task_vision_llm, task_compact_llm = (
                resolve_llms_from_names(llm_names, db, user_id_for_resolution)
            )

            # Use default LLM if no specific LLM configured
            if not task_llm:
                logger.warning(
                    f"Text2SQL task {task.id} has no valid LLM configuration, using default"
                )
                task_llm = self._default_llm

            # Create workspace for the Text2SQL agent
            with UserContext(int(user.id)):
                # Extract database context information
                database_url = config["database_url"]  # Required field, no default
                database_name = config.get("database_name", "Unknown Database")
                database_type = self._infer_database_type(database_url)

                # Create AgentService with Text2SQL agent type

                agent_service = AgentService(
                    name=f"text2sql_task_{task.id}",
                    id=f"web_task_{task.id}",  # Required id parameter
                    agent_type="text2sql",
                    llm=task_llm,
                    fast_llm=task_fast_llm,
                    vision_llm=task_vision_llm,
                    compact_llm=task_compact_llm,
                    tracer=tracer,
                    enable_workspace=True,
                    task_id=str(task.id),
                    use_dag_pattern=False,  # Text2SQL agent provides its own patterns
                    # Pass Text2SQL-specific configuration
                    database_url=database_url,
                    database_name=database_name,
                    database_type=database_type,
                    schema_info=config.get("schema_info"),
                    max_iterations=config.get("max_iterations", 3),
                    read_only=config.get("read_only", True),
                    available_tables=config.get("available_tables"),
                    workspace_base_dir=str(UPLOADS_DIR / f"user_{user.id}"),
                )

                # Use Text2SQLAgent's execute method directly, not AgentService
                # This ensures our custom pattern and trace system is used
                if hasattr(agent_service.agent, "execute") and hasattr(
                    agent_service.agent, "_get_domain_name"
                ):
                    # Save original method

                    async def wrapped_execute(task, context=None, task_id=None):
                        return await self._execute_text2sql_agent_directly(
                            agent_service.agent,
                            task,
                            tracer,
                            task_id if task_id else str(task.id),
                        )

                    agent_service.execute_task = wrapped_execute

                # Store in manager
                self._agents[int(task.id)] = agent_service

                logger.info(
                    f"Successfully created Text2SQL agent for task {task.id} with database_url={config.get('database_url', 'sqlite:///xagent.db')}"
                )

                return agent_service

        except Exception as e:
            logger.error(f"Failed to create Text2SQL agent for task {task.id}: {e}")
            raise

    def _infer_database_type(self, database_url: str) -> str:
        """Infer database type from connection URL"""
        if database_url.startswith("mysql://") or database_url.startswith("mysql2://"):
            return "MySQL"
        elif database_url.startswith("postgresql://") or database_url.startswith(
            "postgres://"
        ):
            return "PostgreSQL"
        elif database_url.startswith("sqlite://"):
            return "SQLite"
        elif database_url.startswith("sqlserver://") or database_url.startswith(
            "mssql://"
        ):
            return "SQL Server"
        else:
            return "Unknown"

    async def _execute_text2sql_agent_directly(
        self, agent: Any, task: str, tracer: Any, task_id: str
    ) -> Dict[str, Any]:
        """
        Execute Text2SQL agent directly using its own execute method.

        This bypasses AgentService's standard execution flow and uses the
        Text2SQL agent's custom patterns and trace system.
        """
        try:
            # Call Text2SQLAgent's execute method directly
            result = await agent.execute(task=task, task_id=task_id)

            # Normalize return format to match AgentService format
            return {
                "status": "completed" if result.get("success") else "failed",
                "output": result.get(
                    "output", result.get("error", "No output provided")
                ),
                "success": result.get("success", False),
                "metadata": result.get(
                    "metadata",
                    {
                        "agent_name": getattr(agent, "name", "text2sql-agent"),
                        "patterns_used": len(agent.patterns),
                        "tools_available": len(agent.tools),
                        "execution_type": "text2sql_direct",
                    },
                ),
            }

        except Exception as e:
            logger.error(f"[TEXT2SQL WEB] Error executing Text2SQL agent directly: {e}")
            import traceback

            traceback.print_exc()

            return {
                "status": "failed",
                "output": f"Text2SQL execution error: {str(e)}",
                "success": False,
                "error": str(e),
                "metadata": {
                    "agent_name": getattr(agent, "name", "text2sql-agent"),
                    "execution_type": "text2sql_direct_error",
                },
            }

    async def _reconstruct_agent_from_history(self, task_id: int, db: Session) -> None:
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
            for event in trace_events:
                tracer_events.append(
                    {
                        "id": event.event_id,
                        "event_type": event.event_type,
                        "task_id": str(event.task_id),
                        "step_id": event.step_id,
                        "timestamp": event.timestamp.timestamp()
                        if event.timestamp
                        else None,
                        "data": event.data,
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
                tracer = Tracer()
                from ...core.agent.trace import ConsoleTraceHandler

                tracer.add_handler(ConsoleTraceHandler())
                tracer.add_handler(DatabaseTraceHandler(task_id))
                tracer.add_handler(WebSocketTraceHandler(task_id))

                # Get LLM configuration from task database record
                try:
                    task = db.query(Task).filter(Task.id == task_id).first()
                    if task:
                        # Load configuration from task record - check for any configured models
                        llm_names: List[Optional[str]] = [
                            str(task.model_name) if task.model_name else None,
                            str(task.small_fast_model_name)
                            if task.small_fast_model_name
                            else None,
                            str(task.visual_model_name)
                            if task.visual_model_name
                            else None,
                            str(task.compact_model_name)
                            if task.compact_model_name
                            else None,
                        ]
                        # Use user_id for model resolution if available
                        user_id_for_resolution = (
                            int(task.user_id) if task.user_id else None
                        )
                        task_llm, task_fast_llm, task_vision_llm, task_compact_llm = (
                            resolve_llms_from_names(
                                llm_names, db, user_id_for_resolution
                            )
                        )

                        # If no models were resolved, use defaults
                        if not task_llm:
                            task_llm = self._default_llm
                    else:
                        # Use defaults if no task record
                        task_llm = self._default_llm
                        task_fast_llm = None
                        task_vision_llm = None
                        task_compact_llm = None
                except Exception:
                    # Fallback to defaults
                    task_llm = self._default_llm
                    task_fast_llm = None
                    task_vision_llm = None
                    task_compact_llm = None
                use_dag = task_llm is not None

                # Create agent with basic configuration
                if user_id is not None:
                    with UserContext(int(user_id)):
                        self._agents[task_id] = AgentService(
                            name=f"reconstructed_agent_task_{task_id}",
                            id=f"web_task_{task_id}",  # Use task ID only for workspace
                            llm=task_llm,
                            fast_llm=task_fast_llm,
                            vision_llm=task_vision_llm,
                            compact_llm=task_compact_llm,
                            tools=[]
                            if db is None
                            else [],  # Tools loaded separately for reconstructed agents
                            memory=get_memory_store(),  # Use dynamic memory store for auto-switching
                            use_dag_pattern=use_dag,
                            tracer=tracer,
                            enable_workspace=True,
                            workspace_base_dir=str(
                                UPLOADS_DIR / f"user_{user_id}"
                            ),  # Use user-isolated base directory
                            task_id=str(task_id),
                            memory_similarity_threshold=None,  # Reconstructed agents use default
                        )
                else:
                    raise ValueError(
                        "User context is required for agent reconstruction"
                    )

                # Reconstruct from history synchronously
                import asyncio

                try:
                    # Try to get or create event loop
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # Create task in running loop
                        asyncio.create_task(
                            self._agents[task_id].reconstruct_from_history(
                                str(task_id), tracer_events, plan_state
                            )
                        )
                    else:
                        # Run in current loop
                        loop.run_until_complete(
                            self._agents[task_id].reconstruct_from_history(
                                str(task_id), tracer_events, plan_state
                            )
                        )
                except RuntimeError:
                    # No event loop, create a new one
                    asyncio.run(
                        self._agents[task_id].reconstruct_from_history(
                            str(task_id), tracer_events, plan_state
                        )
                    )

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

        # Add file information to description if files are specified
        if request.files:
            from ..config import UPLOADS_DIR

            file_info_list = []
            file_paths = []

            for filename in request.files:
                file_path = UPLOADS_DIR / filename
                file_paths.append(str(file_path))

                if file_path.exists():
                    file_info_list.append(f"File: {filename} (Path: {file_path})")
                else:
                    file_info_list.append(f"File: {filename} (File does not exist)")

            if file_info_list:
                if task_description:
                    task_description += "\n\nUploaded files:\n" + "\n".join(
                        file_info_list
                    )
                else:
                    task_description = "File processing task:\n" + "\n".join(
                        file_info_list
                    )

        # Set LLM configuration for this task first to get model info
        # If llm_names is not provided but agent_id is, fetch from agent config
        llm_names_to_use = request.llm_names
        if not llm_names_to_use and request.agent_id:
            # Fetch model configuration from agent
            from ..models.agent import Agent as AgentModel

            agent_db = (
                db.query(AgentModel).filter(AgentModel.id == request.agent_id).first()
            )
            if agent_db and agent_db.models:
                agent_models = agent_db.models
                llm_names_to_use = [
                    str(agent_models.get("general"))
                    if agent_models.get("general")
                    else None,
                    str(agent_models.get("small_fast"))
                    if agent_models.get("small_fast")
                    else None,
                    str(agent_models.get("visual"))
                    if agent_models.get("visual")
                    else None,
                    str(agent_models.get("compact"))
                    if agent_models.get("compact")
                    else None,
                ]
                logger.info(
                    f"Using agent {request.agent_id} model configuration: {llm_names_to_use}"
                )

        default_llm, fast_llm, vision_llm, compact_llm = resolve_llms_from_names(
            llm_names_to_use, db, int(user.id)
        )

        # Extract model names from resolved LLM instances for database storage
        default_model_name = default_llm.model_name if default_llm else None
        fast_model_name = fast_llm.model_name if fast_llm else None
        visual_model_name = vision_llm.model_name if vision_llm else None
        compact_model_name = compact_llm.model_name if compact_llm else None

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

        # Create task with PENDING status and model configuration
        task = Task(
            user_id=user.id,  # Use authenticated user ID
            title=request.title,
            description=task_description,
            status=TaskStatus.PENDING,
            model_name=default_model_name,
            small_fast_model_name=fast_model_name,
            visual_model_name=visual_model_name,
            compact_model_name=compact_model_name,
            agent_config=request.agent_config,
            vibe_mode=request.vibe_mode or "task",
            process_description=request.process_description,
            examples=examples_data,
            agent_id=request.agent_id,  # Set agent_id if provided
        )

        # Set agent_type using the property to avoid Column type issues
        task.agent_type_enum = agent_type_enum
        db.add(task)
        db.commit()
        db.refresh(task)

        # Set LLM configuration for this task in agent manager
        logger.info(
            f"Setting LLM configuration for task {task.id} with llm_names: {request.llm_names}"
        )
        get_agent_manager(request).set_task_llms(int(task.id), request.llm_names, db)

        # Set memory similarity threshold for this task if provided
        if request.memory_similarity_threshold is not None:
            get_agent_manager().set_task_memory_similarity_threshold(
                int(task.id), request.memory_similarity_threshold
            )

        return TaskCreateResponse(
            task_id=task.id,
            title=task.title,
            status=task.status.value,
            created_at=format_datetime_for_api(task.created_at)
            if task.created_at
            else None,
            model_name=task.model_name,
            small_fast_model_name=task.small_fast_model_name,
            visual_model_name=task.visual_model_name,
            vibe_mode=task.vibe_mode,
        )

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
    vibe_mode: Optional[str] = None,
    exclude_vibe_mode: Optional[str] = None,
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

            # Apply vibe mode filter if provided
            if vibe_mode:
                query = query.filter(Task.vibe_mode == vibe_mode)
            elif exclude_vibe_mode:
                query = query.filter(Task.vibe_mode != exclude_vibe_mode)

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
                        "created_at": format_datetime_for_api(task.created_at),
                        "updated_at": format_datetime_for_api(task.updated_at),
                        "model_name": task.model_name,
                        "small_fast_model_name": task.small_fast_model_name,
                        "visual_model_name": task.visual_model_name,
                        "vibe_mode": task.vibe_mode,
                        "input_tokens": task.input_tokens or 0,
                        "output_tokens": task.output_tokens or 0,
                        "total_tokens": task.total_tokens or 0,
                        "llm_calls": task.llm_calls or 0,
                        "agent_id": task.agent_id,
                    }

                    if task.agent_id and task.agent_id in agents_map:
                        task_data["agent_logo_url"] = agents_map[task.agent_id].logo_url

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

            return {
                "task_id": task.id,
                "title": task.title,
                "description": task.description,
                "status": status_value,
                "created_at": format_datetime_for_api(task.created_at),
                "updated_at": format_datetime_for_api(task.updated_at),
                "model_name": task.model_name,
                "small_fast_model_name": task.small_fast_model_name,
                "visual_model_name": task.visual_model_name,
                "dag_data": dag_data,
                "input_tokens": task.input_tokens or 0,
                "output_tokens": task.output_tokens or 0,
                "total_tokens": task.total_tokens or 0,
                "llm_calls": task.llm_calls or 0,
            }

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

            # Get the raw status value safely
            if hasattr(task, "status") and task.status is not None:
                if hasattr(task.status, "value"):
                    status_value = task.status.value
                else:
                    status_value = str(task.status)
            else:
                status_value = "unknown"

            return {
                "task_id": task.id,
                "title": task.title,
                "status": status_value,
                "created_at": format_datetime_for_api(task.created_at),
                "updated_at": format_datetime_for_api(task.updated_at),
                "model_name": task.model_name,
                "small_fast_model_name": task.small_fast_model_name,
                "visual_model_name": task.visual_model_name,
                "input_tokens": task.input_tokens or 0,
                "output_tokens": task.output_tokens or 0,
                "total_tokens": task.total_tokens or 0,
                "llm_calls": task.llm_calls or 0,
            }

        # Execute in thread pool to avoid blocking
        return await asyncio.to_thread(_get_task_status_sync)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get task status failed: {e}")
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

            # Delete DAG execution (if any)
            from ..models.task import DAGExecution

            dag_execution = (
                db.query(DAGExecution).filter(DAGExecution.task_id == task_id).first()
            )
            if dag_execution:
                db.delete(dag_execution)

            # Note: execution_logs table has been removed - replaced by trace_events

            # Delete trace events (only VIBE phase, BUILD sessions are separate)
            from ..models.task import TraceEvent

            db.query(TraceEvent).filter(
                TraceEvent.task_id == task_id,
                TraceEvent.build_id.is_(None),  # ← Only delete VIBE events
            ).delete()

            # Note: tool_usages, agents, and agent_tools tables have been removed

            # Delete the task itself
            db.delete(task)
            db.commit()

            return task

        # Execute database operations in thread pool to avoid blocking
        task = await asyncio.to_thread(_delete_task_sync)

        # Remove agent from manager if it exists
        get_agent_manager(request).remove_agent(task_id, int(user.id))

        from .websocket import background_task_manager, manager

        connections = manager.active_connections.pop(task_id, [])

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
