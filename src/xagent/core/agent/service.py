"""Agent service for executing tasks."""

import asyncio
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ..memory import MemoryStore
from ..memory.in_memory import InMemoryMemoryStore
from ..model.chat.basic.base import BaseLLM
from ..tools.adapters.vibe import Tool
from ..workspace import TaskWorkspace, create_workspace
from .exceptions import AgentException
from .pattern import AgentPattern
from .pattern.dag_plan_execute import DAGPlanExecutePattern
from .pattern.dag_plan_execute.models import ExecutionPhase
from .trace import Tracer

logger = logging.getLogger(__name__)


class AgentService:
    """Service for managing agent execution with proper configuration and error handling."""

    def __init__(
        self,
        name: str,
        patterns: Optional[List[AgentPattern]] = None,
        memory: Optional[MemoryStore] = None,
        tools: Optional[List[Tool]] = None,
        llm: Optional[BaseLLM] = None,
        use_dag_pattern: bool = True,
        tracer: Optional[Tracer] = None,
        id: Optional[str] = None,
        workspace: Optional[TaskWorkspace] = None,
        workspace_base_dir: str = "uploads",
        enable_workspace: bool = True,
        task_id: Optional[str] = None,
        fast_llm: Optional[BaseLLM] = None,
        vision_llm: Optional[BaseLLM] = None,
        compact_llm: Optional[BaseLLM] = None,
        memory_similarity_threshold: Optional[float] = None,
        tool_config: Optional[Any] = None,
        agent_type: str = "standard",
        system_prompt: Optional[str] = None,
        **agent_kwargs: Any,
    ) -> None:
        """Initialize AgentService with configurable components.

        Args:
            name: Agent name identifier
            patterns: List of agent patterns for task execution
            memory: Memory store for agent state
            tools: Available tools for agent execution (optional, can be combined with tool_config)
            llm: Language model for agent execution (required for DAG pattern)
            use_dag_pattern: Whether to use the DAG plan-execute pattern
            tracer: Tracer instance for event tracking
            id: Agent identifier for workspace management
            workspace: Pre-existing workspace to bind to
            workspace_base_dir: Base directory for workspace creation
            enable_workspace: Whether to enable workspace functionality
            fast_llm: Optional fast small model for easy tasks
            vision_llm: Optional vision model for image processing tasks
            compact_llm: Optional compact model for context compression tasks
            memory_similarity_threshold: Optional threshold for memory similarity search
            tool_config: Tool configuration object for dynamic tool loading (combined with tools parameter)
            agent_type: Type of agent to create (e.g., "standard", "text2sql")
            **agent_kwargs: Additional arguments for vertical agent creation
        """
        self.name = name
        self.memory = memory or InMemoryMemoryStore()
        self.tools = tools or []
        self.llm = llm
        self.fast_llm = fast_llm
        self.vision_llm = vision_llm
        self.compact_llm = compact_llm

        # Debug logging for LLM configuration
        logger = logging.getLogger(__name__)
        logger.info(
            f"AgentService initialized with llm={llm.model_name if llm else None}, compact_llm={compact_llm.model_name if compact_llm else None}"
        )
        self.memory_similarity_threshold = memory_similarity_threshold
        self.use_dag_pattern = use_dag_pattern
        self.tool_config = tool_config
        self.tracer = tracer or Tracer()  # Use provided tracer or create a new one

        # Lazy initialization flag for tools
        self._tools_initialized = False

        # Workspace management
        if not id:
            raise ValueError("ID is required for AgentService")
        self.id = id
        self.workspace_base_dir = workspace_base_dir
        self.enable_workspace = enable_workspace

        # Use workspace from tool_config if available and no explicit workspace was provided
        if (
            tool_config
            and hasattr(tool_config, "_workspace_config")
            and tool_config._workspace_config
            and not workspace
        ):
            # Use the same workspace that will be used for tools
            from ..workspace import WorkspaceManager

            workspace_manager = WorkspaceManager()
            ws_config = tool_config._workspace_config
            self.workspace = workspace_manager.get_or_create_workspace(
                ws_config.get("base_dir", self.workspace_base_dir),
                ws_config.get("task_id", self.id),
            )
        else:
            # Use provided workspace or create new one later in _setup_workspace()
            self.workspace = workspace  # type: ignore[assignment]

        # No workspace manager needed - direct workspace creation

        # Set up workspace if enabled
        if self.enable_workspace:
            self._setup_workspace()
            # Note: Tool setup moved to lazy initialization via _ensure_tools_initialized method

        # Auto-create default tool config if neither tools nor tool_config provided
        if not self.tools and not self.tool_config:
            self.tool_config = self._create_default_tool_config()

        # Set up patterns
        if patterns:
            self.patterns = patterns
        elif llm:
            if self.use_dag_pattern:
                # Get allowed_skills from tool_config if available
                allowed_skills = None
                if tool_config and hasattr(tool_config, "get_allowed_skills"):
                    allowed_skills = tool_config.get_allowed_skills()
                    if allowed_skills:
                        logger.info(f"Allowed skills configured: {allowed_skills}")

                # Create DAG pattern - automatically handles single/dual LLM configuration
                dag_pattern = DAGPlanExecutePattern(
                    llm=llm,
                    fast_llm=self.fast_llm,
                    compact_llm=self.compact_llm,
                    tracer=self.tracer,
                    workspace=self.workspace if self.enable_workspace else None,
                    task_id=task_id,
                    memory_store=self.memory,
                    allowed_skills=allowed_skills,
                )

                if self.fast_llm:
                    config_type = "dual LLM (fast)"
                else:
                    config_type = "single LLM"
                logger.info(
                    f"DAG pattern enabled for agent '{name}' with {config_type} configuration."
                )

                self.patterns = [dag_pattern]
                if not tools and not tool_config:
                    logger.info(
                        f"DAG pattern enabled for agent '{name}' with LLM but no tools. Will generate plans without tool execution."
                    )
            else:
                # Create simple ReAct pattern instead of DAG
                from .pattern.react import ReActPattern

                react_pattern = ReActPattern(
                    llm=llm, compact_llm=self.compact_llm or llm
                )
                self.patterns = [react_pattern]
                logger.info(f"ReAct pattern enabled for agent '{name}' with LLM")
        else:
            # No LLM available - cannot create functional patterns
            logger.warning(
                f"No LLM provided for agent '{name}'. Agent will not be able to execute tasks."
            )
            self.patterns = []

        logger.info(
            f"Initializing AgentService '{name}' with {len(self.patterns)} patterns"
        )

        # Pause/resume control variables
        self._is_paused = False
        self._pause_event: Optional[asyncio.Event] = None
        self._current_runner = None

        # Task continuation tracking
        self._current_task_id: Optional[str] = None
        # Set current task_id if provided
        if task_id:
            self._current_task_id = str(
                task_id
            )  # Always store as string for consistency

        # Create agent using factory
        from .vertical_agent_factory import VerticalAgentFactory

        # Prepare agent configuration
        agent_config = {
            "name": self.name,
            "llm": self.llm,
            "memory": self.memory,
            "tools": self.tools,
            "workspace": self.workspace,
            "tracer": self.tracer,
            "task_id": task_id,
            "patterns": self.patterns,  # Include patterns for standard agents
            "system_prompt": system_prompt,  # Pass system prompt from agent builder
            **agent_kwargs,  # Pass through any additional agent-specific arguments
        }

        # Create the agent using the factory
        logger.info(
            f"Creating agent with type: {agent_type}, config keys: {list(agent_config.keys())}"
        )
        self.agent = VerticalAgentFactory.create_agent(
            agent_type=agent_type, **agent_config
        )
        logger.info(f"Successfully created agent: {type(self.agent).__name__}")

        # Update any ReAct patterns in the agent with the tracer
        for pattern in self.agent.patterns:
            if hasattr(pattern, "tracer"):
                pattern.tracer = self.tracer

    async def execute_task(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None,
        task_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute a task using the agent.

        Args:
            task: Task description string
            context: Optional context data for task execution
            task_id: Optional task identifier for continuation

        Returns:
            Dictionary with execution result containing status, output, and metadata

        Raises:
            ValueError: If task is empty or invalid
            RuntimeError: If agent execution fails
            NotImplementedError: If task continuation is requested but not supported
        """
        if not task or not task.strip():
            raise ValueError("Task cannot be empty or whitespace-only")

        # Ensure tools are initialized before execution
        await self._ensure_tools_initialized()

        # Check if agent has any patterns to execute
        if not self.patterns:
            error_msg = f"Agent '{self.name}' has no execution patterns. Cannot execute tasks without patterns. This usually means LLM is not configured."
            logger.error(error_msg)
            return {
                "status": "error",
                "output": error_msg,
                "success": False,
                "error": error_msg,
                "metadata": {
                    "agent_name": self.name,
                    "patterns_used": 0,
                    "tools_available": len(self.tools),
                    "execution_type": "no_patterns_error",
                },
            }

        try:
            # Handle task continuation if task_id is provided
            if task_id:
                # Check if this is the same agent instance that was previously executing this task
                has_attr = hasattr(self, "_current_task_id")
                not_none = self._current_task_id is not None if has_attr else False
                str_equal = (
                    str(self._current_task_id) == str(task_id)
                    if has_attr and not_none
                    else False
                )

                if has_attr and not_none and str_equal:
                    # This agent is already associated with this task, continue execution
                    logger.info(f"Continuing execution for task {task_id}")

                    # Check if we have a DAG pattern that supports continuation
                    dag_pattern = self.get_dag_pattern()
                    if dag_pattern and hasattr(dag_pattern, "handle_continuation"):
                        # Check if DAG pattern has a current plan
                        if (
                            hasattr(dag_pattern, "current_plan")
                            and dag_pattern.current_plan
                        ):
                            # Use DAG pattern's continuation mechanism
                            result = await dag_pattern.handle_continuation(
                                task, context
                            )
                        else:
                            # DAG pattern exists but no current plan, fallback to normal execution
                            logger.info(
                                f"DAG pattern exists but no current plan for task {task_id}, using normal execution"
                            )
                            result = await self._execute_normal_task(
                                task, context, task_id
                            )
                    else:
                        # Fallback to normal execution for non-DAG patterns
                        logger.warning(
                            f"Task continuation requested but pattern doesn't support it: {self.patterns[0].__class__.__name__ if self.patterns else 'No patterns'}"
                        )
                        result = await self._execute_normal_task(task, context, task_id)
                else:
                    # Task continuation requested but this agent is not the original executor
                    raise NotImplementedError(
                        f"Task continuation for task_id {task_id} is not supported. Agent is not the original executor."
                    )
            else:
                # Normal task execution
                logger.debug("No task_id provided, executing normal task")
                result = await self._execute_normal_task(task, context, task_id)

            # Log detailed execution result
            success = result.get("success", False)
            if success:
                logger.info("Task execution completed successfully")
                if result.get("output"):
                    logger.info(f"Output: {result.get('output')}")
            else:
                logger.error("Task execution failed with status: False")
                if result.get("error"):
                    logger.error(f"Error: {result.get('error')}")
                if result.get("output"):
                    logger.error(f"Output: {result.get('output')}")
                # Log full result for debugging
                logger.error(f"Full result: {result}")

            # For DAG pattern, return more detailed result
            if self.use_dag_pattern and self.patterns:
                pattern = self.patterns[0]
                if isinstance(pattern, DAGPlanExecutePattern):
                    execution_status = pattern.get_execution_status()
                    chat_response = result.get("chat_response")
                    normalized_output = result.get("output")
                    if not normalized_output and isinstance(chat_response, dict):
                        normalized_output = chat_response.get(
                            "message", result.get("error", "No output provided")
                        )
                    if not normalized_output:
                        normalized_output = result.get("error", "No output provided")

                    # Normalize result format
                    return {
                        "status": "completed" if result.get("success") else "failed",
                        "output": normalized_output,
                        "success": result.get("success", False),
                        "chat_response": chat_response,
                        "dag_status": execution_status,
                        "metadata": {
                            "agent_name": self.name,
                            "patterns_used": len(self.patterns),
                            "tools_available": len(self.tools),
                            "execution_type": "dag_plan_execute",
                            "iterations": result.get("iterations", 0),
                            "phase": execution_status.get("phase", "unknown"),
                            "task_id": self._current_task_id,
                        },
                    }

            # Normalize result format for non-DAG patterns
            return {
                "status": "completed" if result.get("success") else "failed",
                "output": result.get(
                    "output", result.get("error", "No output provided")
                ),
                "success": result.get("success", False),
                "metadata": {
                    "agent_name": self.name,
                    "patterns_used": len(self.patterns),
                    "tools_available": len(self.tools),
                    "execution_type": "standard",
                },
            }

        except Exception as e:
            logger.error(f"Task execution failed: {str(e)}", exc_info=True)

            # Try to get more detailed error information if it's an AgentException
            detailed_error = str(e)
            error_details = {}

            if isinstance(e, AgentException):
                detailed_error = str(e)
                error_details = {
                    "exception_type": e.__class__.__name__,
                    "context": e.context,
                    "cause": str(e.cause) if e.cause else None,
                }

                # For DAGStepError, add step-specific information
                if hasattr(e, "step_id") and hasattr(e, "step_name"):
                    error_details.update(
                        {
                            "step_id": e.step_id,
                            "step_name": e.step_name,
                        }
                    )

                # For DAGExecutionError, add execution-specific information
                if hasattr(e, "failed_steps") and hasattr(e, "completed_steps"):
                    error_details.update(
                        {
                            "failed_steps_count": len(e.failed_steps),  # type: ignore
                            "completed_steps_count": int(e.completed_steps),  # type: ignore
                            "total_steps": str(getattr(e, "total_steps", 0)),
                            "primary_error": str(getattr(e, "primary_error", "")),
                            "failed_step_names": ", ".join(
                                [f"{s.id} ({s.name})" for s in e.failed_steps]
                            ),
                        }
                    )

            return {
                "status": "error",
                "output": f"Execution failed: {detailed_error}",
                "success": False,
                "error": detailed_error,
                "error_details": error_details,
                "metadata": {
                    "agent_name": self.name,
                    "patterns_used": len(self.patterns),
                    "tools_available": len(self.tools),
                    "execution_type": "error",
                },
            }

    async def pause_execution(self) -> None:
        """Pause the currently executing task."""
        if self._is_paused:
            logger.warning(f"Agent '{self.name}' is already paused")
            return

        # Set pause state
        self._is_paused = True

        # Create pause event (if waiting for pause confirmation is needed)
        self._pause_event = asyncio.Event()

        logger.info(f"Agent '{self.name}' execution paused")

        # Pause all patterns that support pause
        for pattern in self.patterns:
            if hasattr(pattern, "pause_execution"):
                try:
                    pattern.pause_execution()
                    logger.info(f"Paused pattern: {pattern.__class__.__name__}")
                except Exception as e:
                    logger.error(
                        f"Failed to pause pattern {pattern.__class__.__name__}: {e}"
                    )

    async def resume_execution(self) -> None:
        """Resume paused task execution."""
        if not self._is_paused:
            logger.warning(f"Agent '{self.name}' is not paused")
            return

        # Clear pause state
        self._is_paused = False

        # Set pause event if it exists to resume execution
        if self._pause_event:
            self._pause_event.set()
            self._pause_event = None

        logger.info(f"Agent '{self.name}' execution resumed")

        # Resume all patterns that support resume
        for pattern in self.patterns:
            if hasattr(pattern, "resume_execution"):
                try:
                    pattern.resume_execution()
                    logger.info(f"Resumed pattern: {pattern.__class__.__name__}")
                except Exception as e:
                    logger.error(
                        f"Failed to resume pattern {pattern.__class__.__name__}: {e}"
                    )

    def is_paused(self) -> bool:
        """Check if the task is in paused state."""
        return self._is_paused

    def handle_websocket_input(self, user_input: str) -> bool:
        """Handle new user input from WebSocket during execution."""
        logger.info(f"Processing WebSocket input: {user_input}")

        # Check if we have a DAG plan execute pattern
        dag_pattern = None
        for pattern in self.patterns:
            if hasattr(pattern, "set_new_user_input"):
                dag_pattern = pattern
                break

        if dag_pattern:
            # Pass the input to the DAG pattern
            dag_pattern.set_new_user_input(user_input)
            logger.info("Forwarded WebSocket input to DAG pattern")
            return True
        else:
            logger.warning("No DAG pattern found to handle WebSocket input")
            return False

    def add_pattern(self, pattern: AgentPattern) -> None:
        """Add a new pattern to the agent.

        Args:
            pattern: Agent pattern to add
        """
        self.patterns.append(pattern)
        self.agent.patterns.append(pattern)
        logger.info(
            f"Added pattern to agent '{self.name}': {pattern.__class__.__name__}"
        )

    def add_tool(self, tool: Tool) -> None:
        """Add a new tool to the agent.

        Args:
            tool: Tool to add
        """
        self.tools.append(tool)
        self.agent.tools.append(tool)
        logger.info(f"Added tool to agent '{self.name}': {tool.__class__.__name__}")

    def get_status(self) -> Dict[str, Any]:
        """Get current service status and configuration.

        Returns:
            Dictionary with service status information
        """
        status = {
            "name": self.name,
            "patterns_count": len(self.patterns),
            "tools_count": len(self.tools),
            "memory_type": self.memory.__class__.__name__,
            "ready": len(self.patterns) > 0,
            "execution_type": "dag_plan_execute"
            if self.use_dag_pattern
            else "standard",
            "llm_configured": self.llm is not None,
            "fast_llm_configured": self.fast_llm is not None,
            "vision_llm_configured": self.vision_llm is not None,
            "compact_llm_configured": self.compact_llm is not None,
            "dual_llm_enabled": self.fast_llm is not None,
            "compact_llm_enabled": self.compact_llm is not None,
        }

        # Add DAG-specific status if using DAG pattern
        if self.use_dag_pattern and self.patterns:
            dag_pattern = self.patterns[0]
            if isinstance(dag_pattern, DAGPlanExecutePattern):
                dag_status = dag_pattern.get_execution_status()
                status["dag_status"] = dag_status

        return status

    def get_dag_pattern(self) -> Optional[DAGPlanExecutePattern]:
        """Get the DAG pattern if it exists"""
        if self.use_dag_pattern and self.patterns:
            for pattern in self.patterns:
                if isinstance(pattern, DAGPlanExecutePattern):
                    return pattern
        return None

    def get_task_info(self) -> Optional[Dict[str, Any]]:
        """Get task information including generated task_name.

        Returns:
            Dictionary with task information including:
            - task_name: The generated task name for display
            - goal: The plan goal
            - steps_count: Number of steps in the plan
            Returns None if using non-DAG pattern or no plan exists.
        """
        dag_pattern = self.get_dag_pattern()
        if dag_pattern and hasattr(dag_pattern, "get_plan_info"):
            return dag_pattern.get_plan_info()
        return None

    def skip_step(self, step_id: str) -> bool:
        """Skip a step in DAG execution"""
        dag_pattern = self.get_dag_pattern()
        if dag_pattern:
            return dag_pattern.skip_step(step_id)
        return False

    def add_step_injection(
        self,
        step_id: str,
        pre_hook: Optional[Callable[[str, Dict[str, Any]], str]] = None,
        post_hook: Optional[Callable[[str, Dict[str, Any]], Dict[str, Any]]] = None,
    ) -> bool:
        """Add injection hooks to a DAG step"""
        dag_pattern = self.get_dag_pattern()
        if dag_pattern:
            return dag_pattern.add_step_injection(step_id, pre_hook, post_hook)
        return False

    # Workspace management methods
    def _setup_workspace(self) -> None:
        """Set up workspace for this agent service."""
        if not self.workspace:
            self.workspace = create_workspace(self.id, self.workspace_base_dir)

        logger.info(
            f"AgentService '{self.name}' using workspace: {self.workspace.workspace_dir}"
        )

    async def _execute_normal_task(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None,
        task_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute a normal task without continuation logic."""
        # Ensure tools are initialized before execution
        await self._ensure_tools_initialized()

        # Get agent runner and execute task
        runner = self.agent.get_runner()

        # Apply context if provided
        if context:
            for key, value in context.items():
                runner.context.state[key] = value

        # Store task_id for potential continuation
        if task_id:
            self._current_task_id = str(
                task_id
            )  # Always store as string for consistency
        else:
            # Generate a task_id for this execution to enable future continuation
            from uuid import uuid4

            self._current_task_id = f"task_{uuid4().hex[:8]}"

        # Update DAG pattern's task_id if it exists
        if self.use_dag_pattern and self.patterns:
            dag_pattern = self.patterns[0]
            if isinstance(dag_pattern, DAGPlanExecutePattern):
                dag_pattern.task_id = self._current_task_id

        # Call setup() on all tools that implement it
        for tool in self.tools:
            try:
                if hasattr(tool, "setup"):
                    await tool.setup(task_id=self._current_task_id)
            except Exception as e:
                logger.error(
                    f"Tool {tool.name if hasattr(tool, 'name') else 'unknown'} setup failed: {e}",
                    exc_info=True,
                )

        try:
            result = await runner.run(task)
        finally:
            # Call teardown() on all tools that implement it, even if execution fails
            for tool in self.tools:
                try:
                    if hasattr(tool, "teardown"):
                        await tool.teardown(task_id=self._current_task_id)
                except Exception as e:
                    logger.error(
                        f"Tool {tool.name if hasattr(tool, 'name') else 'unknown'} teardown failed: {e}",
                        exc_info=True,
                    )

        # Keep task_id for continuation even if execution fails
        # This allows retrying or continuing failed tasks
        return result

    def get_workspace_files(self) -> Dict[str, Any]:
        """Get all files in the workspace."""
        # First check own workspace
        if self.workspace:
            own_files = self.workspace.get_all_files()
            if "error" not in own_files and own_files.get("files"):
                return own_files

        # If no files in own workspace, check patterns' workspaces
        if self.patterns:
            for pattern in self.patterns:
                if hasattr(pattern, "workspace") and pattern.workspace:
                    try:
                        pattern_files = pattern.workspace.get_all_files()
                        if "error" not in pattern_files and pattern_files.get("files"):
                            return pattern_files  # type: ignore[no-any-return]
                    except Exception:
                        # Ignore errors checking pattern workspace
                        continue

        return {"error": "No workspace available", "files": []}

    def get_output_files(self) -> List[Dict[str, Any]]:
        """Get output files from the workspace."""
        # First check own workspace
        if self.workspace:
            own_files = self.workspace.get_output_files()
            if own_files:
                return own_files

        # If no files in own workspace, check patterns' workspaces (especially DAG pattern)
        if self.patterns:
            for pattern in self.patterns:
                if hasattr(pattern, "workspace") and pattern.workspace:
                    try:
                        pattern_files = pattern.workspace.get_output_files()
                        if pattern_files:
                            return pattern_files  # type: ignore[no-any-return]
                    except Exception:
                        # Ignore errors checking pattern workspace
                        continue

        return []

    def add_file_to_workspace(
        self, file_path: str, target_subdir: str = "input"
    ) -> Path:
        """Add a file to the workspace."""
        if not self.workspace:
            raise ValueError("No workspace available")

        return self.workspace.copy_to_workspace(file_path, target_subdir)

    def cleanup_workspace(self) -> None:
        """Clean up the workspace."""
        if self.workspace:
            workspace_path = str(self.workspace.workspace_dir)
            logger.info(f"Cleaning up workspace: {workspace_path}")
            self.workspace.cleanup()
            # Verify cleanup
            if not self.workspace.workspace_dir.exists():
                logger.info(f"Workspace successfully cleaned up: {workspace_path}")
            else:
                logger.warning(
                    f"Workspace still exists after cleanup: {workspace_path}"
                )
            self.workspace = None  # type: ignore[assignment]
            logger.info(f"Cleaned up workspace for AgentService '{self.name}'")

    async def reconstruct_from_history(
        self,
        task_id: str,
        tracer_events: List[Dict[str, Any]],
        plan_state: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Reconstruct agent state from historical execution records.

        Args:
            task_id: Task ID
            tracer_events: List of historical tracer events
            plan_state: Optional DAG plan state
        """
        logger.info(f"Reconstructing agent state for task {task_id}")

        # Set current task ID
        self._current_task_id = str(task_id)  # Always store as string for consistency

        # Reconstruct DAG pattern state if it exists
        dag_pattern = self.get_dag_pattern()
        if dag_pattern:
            if plan_state:
                await self._reconstruct_dag_pattern(
                    dag_pattern, plan_state, tracer_events
                )
            else:
                # Reset DAG pattern state if no plan_state
                logger.info(
                    f"No plan_state available for task {task_id}, resetting DAG pattern"
                )
                dag_pattern.current_plan = None
                dag_pattern.phase = ExecutionPhase.PLANNING
                if hasattr(dag_pattern, "step_execution_results"):
                    dag_pattern.step_execution_results = {}

        # Reconstruct execution context
        await self._reconstruct_context(tracer_events)

        logger.info(f"Agent state reconstruction completed for task {task_id}")

    async def _reconstruct_dag_pattern(
        self,
        dag_pattern: "DAGPlanExecutePattern",
        plan_state: Dict[str, Any],
        tracer_events: List[Dict[str, Any]],
    ) -> None:
        """Reconstruct DAG pattern state."""
        try:
            from .pattern.dag_plan_execute.models import (
                ExecutionPlan,
                PlanStep,
                StepStatus,
            )

            # Reconstruct ExecutionPlan
            steps_data = plan_state.get("steps", [])
            steps = []

            for step_data in steps_data:
                step = PlanStep(
                    id=step_data["id"],
                    name=step_data["name"],
                    description=step_data["description"],
                    tool_names=step_data.get("tool_names", []),
                    dependencies=step_data.get("dependencies", []),
                    status=StepStatus(step_data["status"]),
                    result=step_data.get("result"),
                    error=step_data.get("error"),
                    error_type=step_data.get("error_type"),
                    error_traceback=step_data.get("error_traceback"),
                    context=step_data.get("context", {}),
                    difficulty=step_data.get("difficulty", "hard"),
                )
                steps.append(step)

            # Create ExecutionPlan
            execution_plan = ExecutionPlan(
                id=plan_state["id"],
                goal=plan_state["goal"],
                iteration=plan_state.get("iteration", 1),
                steps=steps,
            )

            # Set DAG pattern's plan
            dag_pattern.current_plan = execution_plan

            # Reconstruct execution state from tracer events
            completed_steps = set()
            failed_steps = set()

            for event in tracer_events:
                if event.get("event_type", "").startswith("step_end_"):
                    step_id = event.get("step_id")
                    if step_id:
                        if event.get("data", {}).get("success", False):
                            completed_steps.add(step_id)
                        else:
                            failed_steps.add(step_id)

            # Set execution state
            from .utils import StepExecutionResult

            dag_pattern.step_execution_results = {
                step_id: StepExecutionResult(
                    step_id=step_id,
                    messages=[],
                    final_result={"status": "completed"}
                    if step_id in completed_steps
                    else {"status": "failed"},
                    agent_name="reconstructed_agent",
                )
                for step_id in completed_steps | failed_steps
            }

            logger.info(f"DAG pattern reconstructed with {len(steps)} steps")

        except Exception as e:
            logger.error(f"Failed to reconstruct DAG pattern: {e}")
            raise

    async def _reconstruct_context(self, tracer_events: List[Dict[str, Any]]) -> None:
        """Reconstruct execution context from tracer events."""
        try:
            # Reconstruct memory state
            for event in tracer_events:
                if event.get("event_type") == "task_end_general":
                    data = event.get("data", {})
                    if data.get("success"):
                        # Store successful results in memory
                        result = data.get("result", "")
                        if result:
                            from ..memory.core import MemoryNote

                            memory_note = MemoryNote(
                                content=result,
                                category="assistant_response",
                                metadata={"event_id": event.get("id")},
                            )
                            self.memory.add(memory_note)

            logger.info("Context reconstructed from tracer events")

        except Exception as e:
            logger.error(f"Failed to reconstruct context: {e}")
            raise

    def get_reconstruction_data(self) -> Dict[str, Any]:
        """Get current state data for reconstruction."""
        data: Dict[str, Any] = {
            "task_id": self._current_task_id,
            "agent_name": self.name,
            "patterns": len(self.patterns),
        }

        # Get DAG pattern state
        dag_pattern = self.get_dag_pattern()
        if (
            dag_pattern
            and hasattr(dag_pattern, "current_plan")
            and dag_pattern.current_plan
        ):
            data["plan_state"] = dag_pattern.current_plan.to_dict()
            data["execution_status"] = dag_pattern.get_execution_status()

        return data

    def _create_default_tool_config(self) -> Any:
        """Create default tool configuration for standalone usage."""
        try:
            from ...core.tools.adapters.vibe.config import ToolConfig

            # Create basic tool config without database dependency
            class DefaultToolConfig(ToolConfig):
                def __init__(self, workspace_config: Optional[Dict[str, Any]] = None):
                    self._workspace_config = workspace_config

                def get_workspace_config(self) -> Optional[Dict[str, Any]]:
                    return self._workspace_config

                def get_file_tools_enabled(self) -> bool:
                    return True

                def get_basic_tools_enabled(self) -> bool:
                    return True

                def get_vision_model(self) -> Optional[Any]:
                    return None

                def get_image_models(self) -> Dict[str, Any]:
                    return {}

                def get_mcp_server_configs(self) -> List[Dict[str, Any]]:
                    return []

                def get_embedding_model(self) -> Optional[str]:
                    return None

                def get_browser_tools_enabled(self) -> bool:
                    return True

                def get_task_id(self) -> Optional[str]:
                    if self._workspace_config:
                        return self._workspace_config.get("task_id")
                    return None

                def get_user_id(self) -> Optional[int]:
                    # Default to admin user (None) for standalone usage
                    return None

                def get_db(self) -> Any:
                    # No database for standalone usage
                    return None

                def is_admin(self) -> bool:
                    # Default to admin for standalone usage
                    return True

                def get_allowed_collections(self) -> Optional[List]:
                    return None

                def get_allowed_skills(self) -> Optional[List]:
                    return None

                def get_allowed_tools(self) -> Optional[List]:
                    return None

                def get_enable_agent_tools(self) -> bool:
                    return False

            workspace_config = None
            if self.workspace:
                workspace_config = {
                    "base_dir": self.workspace.base_dir,
                    "task_id": self.workspace.id,
                }

            return DefaultToolConfig(workspace_config)

        except Exception as e:
            logger.warning(f"Failed to create default tool config: {e}")
            return None

    async def _ensure_tools_initialized(self) -> None:
        """Ensure tools are initialized (lazy initialization)."""
        if self.tool_config and not self._tools_initialized:
            try:
                from ..tools.adapters.vibe.factory import ToolFactory

                # Update tool_config with current workspace to ensure tools use the same workspace
                if (
                    hasattr(self.tool_config, "_workspace_config")
                    and self.tool_config._workspace_config is not None
                ):
                    self.tool_config._workspace_config["task_id"] = self.id

                new_tools = await ToolFactory.create_all_tools(self.tool_config)

                # Merge with existing tools (prioritize explicitly passed tools)
                existing_tool_names = {
                    tool.name for tool in self.tools if hasattr(tool, "name")
                }
                for tool in new_tools:
                    if (
                        not hasattr(tool, "name")
                        or tool.name not in existing_tool_names
                    ):
                        self.tools.append(tool)

                # Filter tools by allowed_tools if specified
                if hasattr(self.tool_config, "get_allowed_tools"):
                    allowed_tools = self.tool_config.get_allowed_tools()
                    if allowed_tools is not None:
                        original_count = len(self.tools)
                        allowed_set = set(allowed_tools)
                        self.tools = [
                            tool
                            for tool in self.tools
                            if hasattr(tool, "name") and tool.name in allowed_set
                        ]
                        logger.info(
                            f"Filtered tools by allowed_tools: {original_count} -> {len(self.tools)} tools"
                        )

                # Sync tools to self.agent (for standard agents)
                if hasattr(self.agent, "tools") and isinstance(self.agent.tools, list):
                    self.agent.tools = self.tools

                logger.info(
                    f"Added {len(new_tools)} tools from configuration to AgentService '{self.name}', "
                    f"current tools count: {len(self.tools)}"
                )
                self._tools_initialized = True
            except Exception as e:
                logger.error(f"Failed to initialize tools from configuration: {e}")
                raise RuntimeError(
                    f"Tool initialization failed for AgentService '{self.name}': {e}"
                ) from e
