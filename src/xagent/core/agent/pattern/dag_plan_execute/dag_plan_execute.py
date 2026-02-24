"""
DAG-based Plan and Execute Pattern with dependencies and iterative goal checking.

This module implements an advanced plan-and-execute pattern that supports:
1. DAG (Directed Acyclic Graph) structured plans with step dependencies
2. Iterative goal checking and plan updates
3. Immutable plans with selective step execution (skip mechanism)
4. Pre/post step injection hooks for user prompt processing
5. Step execution visualization for web display
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set, cast
from uuid import uuid4

from ....memory import MemoryStore
from ....memory.in_memory import InMemoryMemoryStore
from ....model.chat.basic.base import BaseLLM
from ....tools.adapters.vibe import Tool
from ....workspace import TaskWorkspace
from ...context import AgentContext
from ...trace import (
    TraceCategory,
    Tracer,
    trace_dag_execution,
    trace_dag_plan_end,
    trace_dag_plan_start,
    trace_error,
    trace_memory_retrieve_end,
    trace_memory_retrieve_start,
    trace_memory_store_end,
    trace_memory_store_start,
    trace_task_completion,
    trace_task_start,
    trace_user_message,
)
from ..memory_utils import enhance_goal_with_memory

if TYPE_CHECKING:
    from ...agent import Agent
    # from ..react import ReActPattern  # Already in TYPE_CHECKING

from ...exceptions import LLMNotAvailableError, PatternExecutionError
from ...utils import ContextBuilder, StepExecutionResult
from ..base import AgentPattern

# Import the extracted modules
from .models import (
    ExecutionPhase,
    ExecutionPlan,
    PlanStep,
    StepInjection,
    StepStatus,
    UserInputMapper,
)
from .plan_executor import PlanExecutor
from .plan_generator import PlanGenerator
from .result_analyzer import ResultAnalyzer
from .step_agent_factory import StepAgentFactory

logger = logging.getLogger(__name__)


class DAGPlanExecutePattern(AgentPattern):
    """
    Enhanced DAG Plan-Execute pattern with dependency support and iterative refinement.

    Features:
    - DAG-structured plans with step dependencies
    - Each step is executed by a specialized ReAct agent
    - Intelligent context management and compaction
    - Iterative goal achievement checking
    - Real-time execution visualization
    """

    def __init__(
        self,
        llm: BaseLLM,
        max_iterations: int = 3,
        goal_check_enabled: bool = True,
        step_agent_factory: Optional[Callable[[str, List[Tool], str], "Agent"]] = None,
        context_compact_threshold: Optional[int] = None,
        tracer: Optional[Tracer] = None,
        task_id: Optional[str] = None,
        workspace: Optional[TaskWorkspace] = None,
        max_concurrency: int = 4,
        fast_llm: Optional[BaseLLM] = None,
        compact_llm: Optional[BaseLLM] = None,
        memory_store: Optional[MemoryStore] = None,
        skill_manager: Optional[Any] = None,
        allowed_skills: Optional[List[str]] = None,
    ):
        """Initialize the enhanced DAG Plan-Execute pattern.

        Args:
            llm: Language model for planning and goal checking
            max_iterations: Maximum number of planning iterations
            goal_check_enabled: Whether to perform goal checking after execution
            step_agent_factory: Factory function to create step agents
            context_compact_threshold: Token threshold for context compaction
            tracer: Tracer instance for event tracking
            workspace: Workspace for file output management
            max_concurrency: Maximum number of concurrent steps to execute
            fast_llm: Optional fast small model for easy tasks
            compact_llm: Optional LLM for context compaction, defaults to main LLM
            memory_store: Optional memory store for shared memory across all steps
            skill_manager: Optional skill manager for skill-based planning
            allowed_skills: Optional list of allowed skill names for filtering
        """
        self.llm = llm
        self.fast_llm = fast_llm
        self.compact_llm = (
            compact_llm or llm
        )  # Use main LLM if compact_llm not provided
        self.max_iterations = max_iterations
        self.goal_check_enabled = goal_check_enabled
        self.context_compact_threshold = context_compact_threshold
        self.tracer = tracer or Tracer()  # Use provided tracer or create a new one
        self.task_id = task_id  # Store the task_id for tracing
        # Workspace must be provided
        if not workspace:
            raise ValueError("Workspace is required for DAG Plan-Execute pattern")
        self.workspace = workspace
        self.max_concurrency = max_concurrency
        self.memory_store = memory_store
        self.allowed_skills = allowed_skills

        # skill_manager 是必需的，如果没有传入则创建默认的
        if skill_manager is None:
            from .....skills.utils import create_skill_manager

            skill_manager = create_skill_manager()

        self.skill_manager = skill_manager

        # Execution state
        self.current_plan: Optional[ExecutionPlan] = None
        self.skipped_steps: Set[str] = set()
        self.phase: ExecutionPhase = ExecutionPhase.PLANNING
        self._final_answer: Optional[str] = None
        self._context: Optional[AgentContext] = None
        self._skill_context: Optional[str] = (
            None  # Store skill context for execution phase
        )

        # Pause/resume control
        self._pause_event = asyncio.Event()
        self._pause_condition = asyncio.Condition()
        self._execution_interrupted = False
        self._pending_continuation: Optional[Dict[str, Any]] = (
            None  # 待处理的 continuation
        )
        self._pause_reason: Optional[str] = None
        self._pause_timestamp: Optional[datetime] = None

        # Enhanced components
        logger.info(
            f"DAGPlanExecutePattern initializing ContextBuilder with llm={self.llm.model_name if self.llm else None}, compact_llm={self.compact_llm.model_name if self.compact_llm else None}"
        )
        self.context_builder = ContextBuilder(
            self.llm, context_compact_threshold, compact_llm=self.compact_llm
        )
        self.step_agents: Dict[str, "Agent"] = {}
        self.step_execution_results: Dict[str, StepExecutionResult] = {}
        self.step_patterns: Dict[
            str, Any
        ] = {}  # Track ReAct patterns for pause control

        # User input mapping
        self.user_input_mapper = UserInputMapper()
        self._new_user_input: Optional[str] = None

        # Initialize StepAgentFactory first
        assert workspace is not None, "workspace must be provided"
        self.step_agent_factory = StepAgentFactory(
            llm=llm,
            tracer=tracer or Tracer(),
            workspace=workspace,  # Use the same workspace as the pattern
            default_factory=step_agent_factory,
            fast_llm=fast_llm,
            compact_llm=self.compact_llm,
            memory_store=memory_store,
        )

        # Initialize extracted components
        self.plan_generator = PlanGenerator(
            llm, fast_llm, skill_manager=skill_manager, allowed_skills=allowed_skills
        )
        assert workspace is not None, "workspace must be provided"
        self.plan_executor = PlanExecutor(
            llm=llm,
            tracer=tracer or Tracer(),
            workspace=workspace,  # Use the same workspace as the pattern
            memory_store=memory_store or InMemoryMemoryStore(),
            user_input_mapper=self.user_input_mapper,
            parent_pattern=self,
            context_compact_threshold=context_compact_threshold,
            max_concurrency=max_concurrency,
            compact_llm=self.compact_llm,
            step_agent_factory=self.step_agent_factory,
        )
        self.result_analyzer = ResultAnalyzer(llm, tracer or Tracer())

    async def run(
        self,
        task: str,
        memory: MemoryStore,
        tools: List[Tool],
        context: Optional[AgentContext] = None,
    ) -> Dict[str, Any]:
        """
        Execute the DAG plan-and-execute pattern.

        Args:
            task: Natural language task description (becomes the goal)
            memory: Memory store for state persistence
            tools: Available tools for execution
            context: Execution context

        Returns:
            Execution result with plan history and final status
        """
        # Store context and save original goal for use in all steps to preserve context
        self._context = context
        self._original_goal = task

        # 重置 continuation 和中断标志（确保每次执行都是干净的）
        self._continuation_requested = False
        self._execution_interrupted = False

        # For process mode, append examples to the task for LLM reference
        if context and hasattr(context, "state"):
            vibe_mode = context.state.get("vibe_mode")
            if vibe_mode == "process":
                examples = context.state.get("examples", [])
                if examples:
                    task += "\n\nTypical Example for Process Execution:\n"
                    task += "Use this example as the concrete input when executing the workflow:\n"
                    for i, ex in enumerate(examples, 1):
                        task += f"Example {i}:\n"
                        task += f"  Input: {ex.get('input', '')}\n"
                        task += f"  Output: {ex.get('output', '')}\n"
                    task += "\nIMPORTANT: When designing workflow steps that require specific input (e.g., search queries, API calls), use the INPUT from the example above as the concrete query string. Design the process to handle this specific case.\n"

        logger.info(f"Starting DAG Plan-Execute for task: {task[:100]}...")

        # Send user_message trace event
        if self.tracer:
            # Prepare trace data with file information
            trace_data = {
                "context": context.__dict__
                if hasattr(context, "__dict__")
                else context,
                "pattern": "DAG Plan-Execute",
            }

            # Add file information if present in context
            if context:
                # Handle both object with __dict__ and plain dict
                context_dict: Dict[str, Any]
                if hasattr(context, "__dict__"):
                    context_dict = context.__dict__
                else:
                    context_dict = cast(Dict[str, Any], context)

                if "file_info" in context_dict:
                    trace_data["files"] = context_dict["file_info"]
                if "uploaded_files" in context_dict:
                    trace_data["uploaded_files"] = context_dict["uploaded_files"]

            await trace_user_message(
                self.tracer,
                self.task_id or f"dag_plan_execute_{uuid4().hex[:8]}",
                task,  # task already includes examples for process mode
                trace_data,
            )

        # Validate LLM availability
        if not self.llm:
            raise LLMNotAvailableError(
                "No LLM configured for DAG Plan-Execute pattern",
                context={"pattern": "DAG Plan-Execute", "task": task[:100]},
            )

        tool_map: Dict[str, Any] = {}
        for tool in tools:
            # Try to get tool name from various sources
            if hasattr(tool, "name"):
                tool_name = tool.name
            elif hasattr(tool, "metadata") and hasattr(tool.metadata, "name"):
                tool_name = tool.metadata.name
            else:
                tool_name = str(id(tool))

            tool_map[tool_name] = tool

        # Save original tools for potential task continuation
        self._original_tools = tools
        execution_history: List[Dict[str, Any]] = []

        try:
            for iteration in range(1, self.max_iterations + 1):
                logger.info(f"Starting iteration {iteration}")

                # 优先检查是否有待处理的 continuation
                # 如果有，说明是 continuation 导致的 iteration，继续处理
                if self._pending_continuation:
                    logger.info("Starting new iteration for continuation processing")

                # 然后检查是否被中断（continuation 会在 planning 阶段处理，这里不检查）
                if self._execution_interrupted and not self._pending_continuation:
                    logger.info(
                        f"Execution interrupted at iteration {iteration}, stopping..."
                    )
                    break

                # Check for pause state at the beginning of each iteration
                if self._pause_event.is_set():
                    logger.info(
                        f"Execution paused at iteration {iteration}, waiting for resume..."
                    )
                    # Use Condition to properly wait for pause to be cleared
                    async with self._pause_condition:
                        await self._pause_condition.wait_for(
                            lambda: not self._pause_event.is_set()
                        )
                    logger.info(f"Execution resumed at iteration {iteration}")

                # Trace iteration start
                # execution_history contains previous iterations, so current iteration is len(execution_history) + 1
                global_iteration = len(execution_history) + 1
                logger.info(
                    f"DEBUG: Sending trace_task_start for iteration {iteration}, global_iteration: {global_iteration}, execution_history length: {len(execution_history)}"
                )
                await trace_task_start(
                    self.tracer,
                    self.task_id or f"iteration_{global_iteration}",
                    TraceCategory.DAG,
                    data={
                        "iteration": global_iteration,
                        "task_preview": task[:50],
                    },
                )

                # Phase 1: Planning (only in first iteration) or Plan Extension (for subsequent iterations)
                # Create execution history entry at the beginning of iteration
                iteration_data: Dict[str, Any] = {
                    "iteration": iteration,
                    "plan": None,  # Will be filled after plan generation
                    "results": [],
                    "timestamp": datetime.now().isoformat(),
                    "continuation": None,  # Will be filled if this is a continuation iteration
                }
                execution_history.append(iteration_data)

                # For subsequent iterations, preserve step execution results from previous iterations
                # This ensures that steps in later iterations can access results from earlier iterations
                if iteration > 1:
                    logger.info(
                        f"Preserving {len(self.step_execution_results)} step execution results from previous iterations for iteration {iteration}"
                    )
                    # Ensure plan executor has access to accumulated step results
                    self.plan_executor.step_execution_results.update(
                        self.step_execution_results
                    )

                if iteration == 1:
                    # First iteration: Generate initial plan
                    self.phase = ExecutionPhase.PLANNING

                    # Send dag_plan_start and dag_execution events to notify frontend
                    if hasattr(self, "tracer") and self.tracer and self.task_id:
                        await trace_dag_plan_start(
                            self.tracer,
                            self.task_id,
                            data={
                                "phase": "planning",
                                "iteration": iteration,
                            },
                        )
                        await trace_dag_execution(
                            self.tracer,
                            self.task_id,
                            "planning",
                            data={
                                "current_plan": {},
                                "created_at": datetime.now().isoformat(),
                            },
                        )

                    # Check for pause state before plan generation
                    if self._pause_event.is_set():
                        logger.info(
                            "Execution paused during planning phase, waiting for resume..."
                        )
                        # Use Condition to properly wait for pause to be cleared
                        async with self._pause_condition:
                            await self._pause_condition.wait_for(
                                lambda: not self._pause_event.is_set()
                            )
                        logger.info("Execution resumed during planning phase")

                    # Prepare parallel queries for memory and skill
                    enhanced_task = task
                    skill_context = None

                    # Create parallel tasks
                    memory_task = None
                    skill_task = None

                    if self.memory_store:
                        # Trace memory retrieval start
                        memory_task_id = f"dag_plan_memory_{int(time.time())}"
                        await trace_memory_retrieve_start(
                            self.tracer,
                            memory_task_id,
                            data={
                                "goal": task,
                                "memory_category": "dag_plan_execute_memory",
                                "iteration": iteration,
                            },
                        )

                        # Get current user context to pass to the thread
                        try:
                            from .....web.user_isolated_memory import current_user_id

                            user_id = current_user_id.get()
                        except ImportError:
                            # Fallback for non-web environment
                            user_id = None

                        memory_task = asyncio.to_thread(
                            self._lookup_relevant_memories_with_context,
                            self.memory_store,
                            task,
                            "dag_plan_execute_memory",
                            include_general=True,
                            user_id=user_id,
                        )

                    if self.skill_manager:
                        # Trace skill selection start
                        skill_task_id = f"dag_plan_skill_{int(time.time())}"
                        skill_task = self.skill_manager.select_skill(
                            task,
                            self.llm,
                            tracer=self.tracer,
                            task_id=skill_task_id,
                            allowed_skills=self.allowed_skills,
                        )

                    # Execute memory and skill queries in parallel
                    results = await asyncio.gather(
                        *filter(None, [memory_task, skill_task]),
                        return_exceptions=True,
                    )

                    # Process memory result
                    memory_result = (
                        results[0] if memory_task and len(results) > 0 else None
                    )
                    if memory_result and not isinstance(memory_result, Exception):
                        # Type narrowing: we know memory_result is the actual result, not Exception
                        memories: List[Dict[str, Any]] = memory_result  # type: ignore[assignment]
                        # Apply memory enhancement (task already includes examples for process mode)
                        enhanced_task = enhance_goal_with_memory(task, memories)

                        # Trace memory retrieval end
                        if self.memory_store:
                            await trace_memory_retrieve_end(
                                self.tracer,
                                memory_task_id,
                                data={
                                    "goal": task,
                                    "memories_found": len(memories),
                                    "memories_used": len(
                                        [
                                            m
                                            for m in memories
                                            if m.get("content", "").strip()
                                        ]
                                    ),
                                    "iteration": iteration,
                                },
                            )

                    # Process skill result
                    skill_result = (
                        results[1]
                        if memory_task and skill_task and len(results) > 1
                        else results[0]
                        if skill_task and len(results) > 0
                        else None
                    )
                    if skill_result and not isinstance(skill_result, Exception):
                        # Type narrowing: we know skill_result is the actual result, not Exception
                        skill: Dict[str, Any] = skill_result  # type: ignore[assignment]
                        if skill:
                            skill_context = self.plan_generator._build_skill_context(
                                skill
                            )
                            self._skill_context = (
                                skill_context  # Store for execution phase
                            )
                            logger.info(f"Using skill: {skill['name']}")
                        else:
                            logger.info("No relevant skill found")
                    elif skill_result and isinstance(skill_result, Exception):
                        logger.warning(f"Skill selection failed: {skill_result}")

                    plan = await self.plan_generator.generate_plan(
                        goal=enhanced_task,  # May include memory enhancement, examples already in task
                        tools=tools,
                        iteration=iteration,
                        history=execution_history,
                        tracer=self.tracer,
                        context=self._context,
                        skill_context=skill_context,  # Pass pre-fetched skill context
                    )

                    # Send trace event with generated plan to frontend
                    if hasattr(self, "tracer") and self.tracer and self.task_id:
                        await trace_dag_plan_end(
                            self.tracer,
                            self.task_id,
                            data={
                                "steps_count": len(plan.steps),
                                "plan_id": plan.id,
                                "plan_data": {
                                    "id": plan.id,
                                    "goal": plan.goal,
                                    "steps": [
                                        {
                                            "id": step.id,
                                            "name": step.name,
                                            "description": step.description,
                                            "tool_names": step.tool_names,
                                            "dependencies": step.dependencies,
                                            "status": step.status.value,
                                            "started_at": step.started_at.isoformat()
                                            if step.started_at
                                            else None,
                                            "completed_at": step.completed_at.isoformat()
                                            if step.completed_at
                                            else None,
                                            "conditional_branches": step.conditional_branches,
                                            "required_branch": step.required_branch,
                                            "is_conditional": step.is_conditional,
                                        }
                                        for step in plan.steps
                                    ],
                                },
                            },
                        )

                    # Store plan generation memory with insights (deferred to final insights generation)
                    # Planning insights will be generated together with execution insights at the end
                else:
                    # Subsequent iterations: Extend existing plan
                    if self.current_plan:
                        # Check for pause state before plan extension
                        if self._pause_event.is_set():
                            logger.info(
                                "Execution paused during plan extension phase, waiting for resume..."
                            )
                            # Use Condition to properly wait for pause to be cleared
                            async with self._pause_condition:
                                await self._pause_condition.wait_for(
                                    lambda: not self._pause_event.is_set()
                                )
                            logger.info("Execution resumed during plan extension phase")

                        additional_steps = await self.plan_generator.extend_plan(
                            goal=task,
                            tools=tools,
                            iteration=iteration,
                            history=execution_history,
                            current_plan=self.current_plan,
                            tracer=self.tracer,
                            user_input_context={"new_input": self._new_user_input}
                            if self._new_user_input
                            else None,
                            context=self._context,
                        )
                    else:
                        additional_steps = []

                    if additional_steps and self.current_plan:
                        plan = self.current_plan.extend_with_steps(additional_steps)
                        logger.info(
                            f"Extended plan with {len(additional_steps)} additional steps"
                        )

                        # Send trace event with updated plan data to frontend
                        if hasattr(self, "tracer") and self.tracer and self.task_id:
                            await trace_dag_plan_end(
                                self.tracer,
                                self.task_id,
                                data={
                                    "steps_count": len(plan.steps),
                                    "plan_id": plan.id,
                                    "plan_data": {
                                        "id": plan.id,
                                        "goal": plan.goal,
                                        "steps": [
                                            {
                                                "id": step.id,
                                                "name": step.name,
                                                "description": step.description,
                                                "tool_names": step.tool_names,
                                                "dependencies": step.dependencies,
                                                "status": step.status.value,
                                                "started_at": step.started_at.isoformat()
                                                if step.started_at
                                                else None,
                                                "completed_at": step.completed_at.isoformat()
                                                if step.completed_at
                                                else None,
                                                "conditional_branches": step.conditional_branches,
                                                "required_branch": step.required_branch,
                                                "is_conditional": step.is_conditional,
                                            }
                                            for step in plan.steps
                                        ],
                                    },
                                },
                            )
                    else:
                        plan = self.current_plan or ExecutionPlan(
                            id=str(uuid4()),
                            goal=task,
                            steps=[],
                            created_at=datetime.now(),
                        )
                        logger.info("No additional steps needed for this iteration")

                self.current_plan = plan
                # Update the plan in the history entry
                execution_history[-1]["plan"] = plan.to_dict()

                # 检查是否有待处理的 continuation（在 current_plan 设置后检查）
                if self._pending_continuation:
                    logger.info("Processing pending continuation")
                    continuation_data = self._pending_continuation
                    self._pending_continuation = None  # 清除标志
                    self._execution_interrupted = False  # 清除中断标志，允许继续执行

                    # Record continuation in execution history for goal checking
                    iteration_data["continuation"] = {
                        "user_input": continuation_data["additional_task"],
                        "context": continuation_data.get("context", {}),
                        "timestamp": datetime.now().isoformat(),
                    }

                    # 注意：trace_user_message 已经在 websocket handler 中立即发送
                    # 这里不再重复发送，避免消息重复显示

                    # 扩展 plan
                    additional_steps = await self.plan_generator.extend_plan(
                        goal=self.current_plan.goal,
                        tools=tools,  # 使用 tools 列表，而不是 tool_map
                        iteration=self.current_plan.iteration + 1,
                        history=execution_history,  # 传递实际的历史记录，会自动压缩
                        current_plan=self.current_plan,
                        tracer=self.tracer,
                        user_input_context={
                            "new_input": continuation_data["additional_task"]
                        },
                        context=self._context,
                    )

                    if additional_steps:
                        # 扩展当前 plan
                        new_plan = self.current_plan.extend_with_steps(additional_steps)
                        self.current_plan = new_plan
                        plan = new_plan  # 关键：更新局部变量 plan，否则会执行旧 plan
                        execution_history[-1]["plan"] = new_plan.to_dict()
                        logger.info(
                            f"Extended plan with {len(additional_steps)} steps via continuation"
                        )

                        # Send trace event with extended plan to frontend
                        if hasattr(self, "tracer") and self.tracer and self.task_id:
                            await trace_dag_plan_end(
                                self.tracer,
                                self.task_id,
                                data={
                                    "steps_count": len(new_plan.steps),
                                    "plan_id": new_plan.id,
                                    "plan_data": {
                                        "id": new_plan.id,
                                        "goal": new_plan.goal,
                                        "steps": [
                                            {
                                                "id": step.id,
                                                "name": step.name,
                                                "description": step.description,
                                                "tool_names": step.tool_names,
                                                "dependencies": step.dependencies,
                                                "status": step.status.value,
                                                "started_at": step.started_at.isoformat()
                                                if step.started_at
                                                else None,
                                                "completed_at": step.completed_at.isoformat()
                                                if step.completed_at
                                                else None,
                                                "conditional_branches": step.conditional_branches,
                                                "required_branch": step.required_branch,
                                                "is_conditional": step.is_conditional,
                                            }
                                            for step in new_plan.steps
                                        ],
                                    },
                                },
                            )
                    else:
                        logger.info(
                            "No additional steps generated via continuation, continuing with current plan"
                        )

                # Phase 2: Execution
                self.phase = ExecutionPhase.EXECUTING

                # Send dag_execution event to notify frontend
                if hasattr(self, "tracer") and self.tracer and self.task_id:
                    await trace_dag_execution(
                        self.tracer,
                        self.task_id,
                        "executing",
                        data={
                            "current_plan": self.current_plan.to_dict()
                            if self.current_plan
                            else {},
                            "created_at": datetime.now().isoformat(),
                        },
                    )

                # 检查是否有 continuation（在执行前检查，避免执行已废弃的 plan）
                if self._pending_continuation:
                    logger.info(
                        "Pending continuation detected before execution, starting new iteration"
                    )
                    # 不中断，直接进入下一个 iteration
                    # continuation 会在下一个 iteration 的 planning 阶段处理
                    continue

                # 检查是否被中断（在执行前检查）
                if self._execution_interrupted:
                    logger.info(
                        "Execution interrupted before execution phase, stopping..."
                    )
                    break

                # Check for pause state before execution phase
                if self._pause_event.is_set():
                    logger.info(
                        "Execution paused before execution phase, waiting for resume..."
                    )
                    # Use Condition to properly wait for pause to be cleared
                    async with self._pause_condition:
                        await self._pause_condition.wait_for(
                            lambda: not self._pause_event.is_set()
                        )
                    logger.info("Execution resumed before execution phase")

                execution_results = await self.plan_executor.execute_plan(
                    plan, tool_map, self._skill_context
                )

                # Send final dag_plan_end event with updated step statuses (including skipped steps)
                if hasattr(self, "tracer") and self.tracer and self.task_id:
                    await trace_dag_plan_end(
                        self.tracer,
                        self.task_id,
                        data={
                            "steps_count": len(plan.steps),
                            "plan_id": plan.id,
                            "plan_data": {
                                "id": plan.id,
                                "goal": plan.goal,
                                "steps": [
                                    {
                                        "id": step.id,
                                        "name": step.name,
                                        "description": step.description,
                                        "tool_names": step.tool_names,
                                        "dependencies": step.dependencies,
                                        "status": step.status.value,
                                        "started_at": step.started_at.isoformat()
                                        if step.started_at
                                        else None,
                                        "completed_at": step.completed_at.isoformat()
                                        if step.completed_at
                                        else None,
                                        "conditional_branches": step.conditional_branches,
                                        "required_branch": step.required_branch,
                                        "is_conditional": step.is_conditional,
                                    }
                                    for step in plan.steps
                                ],
                            },
                        },
                    )

                # Update results in the history entry
                execution_history[-1]["results"] = execution_results

                # Store execution results for context building
                for result in execution_results:
                    if result.get("status") == "completed":
                        step_id = result["step_id"]
                        self.step_execution_results[step_id] = StepExecutionResult(
                            step_id=step_id,
                            messages=[],  # This would be populated from actual execution
                            final_result=result["result"],
                            agent_name=f"step_agent_{result['step_name']}",
                        )

                # Execution insights will be generated together with other insights at the end

                # 在执行完成后立即检查 continuation
                if self._pending_continuation:
                    logger.info(
                        "Pending continuation detected after execution, starting new iteration"
                    )
                    # 不中断，直接进入下一个 iteration
                    # continuation 会在下一个 iteration 的 planning 阶段处理
                    continue

                # Phase 3: Goal Checking
                if self.goal_check_enabled:
                    self.phase = ExecutionPhase.CHECKING

                    # 检查是否被中断（在检查目标前检查）
                    if self._execution_interrupted:
                        logger.info(
                            "Execution interrupted before goal checking phase, stopping..."
                        )
                        break

                    # 检查是否有 continuation（在检查目标前检查）
                    if self._pending_continuation:
                        logger.info(
                            "Pending continuation detected before goal checking, starting new iteration"
                        )
                        # 不中断，直接进入下一个 iteration
                        # continuation 会在下一个 iteration 的 planning 阶段处理
                        continue

                    # Check for pause state before goal checking
                    if self._pause_event.is_set():
                        logger.info(
                            "Execution paused before goal checking phase, waiting for resume..."
                        )
                        # Use Condition to properly wait for pause to be cleared
                        async with self._pause_condition:
                            await self._pause_condition.wait_for(
                                lambda: not self._pause_event.is_set()
                            )
                        logger.info("Execution resumed before goal checking phase")

                    # Extract file outputs before goal checking
                    file_outputs = self._extract_file_outputs()

                    goal_check_result = (
                        await self.result_analyzer.check_goal_achievement(
                            goal=task,
                            history=execution_history,
                            file_outputs=file_outputs,
                        )
                    )

                    # Store memories using insights from goal check (synchronous, fast operation)
                    if self.memory_store and goal_check_result and plan:
                        memory_insights = goal_check_result.get("memory_insights", {})
                        await self._store_memory(
                            memory_insights=memory_insights,
                            task=task,
                            execution_results=execution_results,
                            plan_id=getattr(plan, "id", None),
                        )

                    # Check if goal was achieved and get final answer from the same LLM call
                    if goal_check_result.get("achieved", False):
                        logger.info(
                            "Goal achieved! Using final answer from goal check..."
                        )

                        # 检查是否有 continuation（在返回最终答案前检查）
                        if self._pending_continuation:
                            logger.info(
                                "Pending continuation detected before final answer, starting new iteration"
                            )
                            # 不中断，直接进入下一个 iteration
                            # continuation 会在下一个 iteration 的 planning 阶段处理
                            continue

                        # Check for pause state before returning final answer
                        if self._pause_event.is_set():
                            logger.info(
                                "Execution paused before final answer, waiting for resume..."
                            )
                            # Use Condition to properly wait for pause to be cleared
                            async with self._pause_condition:
                                await self._pause_condition.wait_for(
                                    lambda: not self._pause_event.is_set()
                                )
                            logger.info("Execution resumed before final answer")

                        # Store the final answer from goal check result
                        final_answer = goal_check_result.get("final_answer", "")
                        if final_answer:
                            self._final_answer = final_answer
                            logger.info(
                                f"Final answer ready (length: {len(final_answer)})"
                            )
                        else:
                            logger.warning(
                                "Final answer missing from goal check result"
                            )
                            self._final_answer = None

                        self.phase = ExecutionPhase.COMPLETED
                        break

                # Check if we should continue to next iteration
                if iteration >= self.max_iterations:
                    logger.info("Reached maximum iterations")
                    self.phase = ExecutionPhase.COMPLETED
                    break

            # Final result compilation
            final_result = self._compile_final_result(task, execution_history)

            # Check for agent-specific completion trace data in the final result
            if isinstance(final_result, dict) and "agent_trace_data" in final_result:
                pass

            # Prepare trace data
            result_str: str = json.dumps(final_result)

            # Trace overall completion
            await trace_task_completion(
                self.tracer,
                self.task_id or "dag_plan_execute",
                result=result_str,
                success=True,
            )

            logger.info("DAG Plan-Execute completed successfully")
            return final_result

        except Exception as e:
            self.phase = ExecutionPhase.FAILED
            logger.error(f"DAG Plan-Execute failed: {e}", exc_info=True)

            # Trace failure with detailed error information
            await trace_error(
                self.tracer,
                self.task_id or "dag_plan_execute",
                error_type="DAG_ERROR",
                error_message=str(e),
                data={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "phase": self.phase.value,
                    "iterations": len(execution_history),
                },
            )

            # Handle specific LLM-related errors gracefully for test compatibility
            from ...exceptions import DAGPlanGenerationError, LLMResponseError

            if isinstance(e, (LLMResponseError, DAGPlanGenerationError)):
                # Return graceful error result with detailed error information
                error_result = {
                    "success": False,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "goal": task,
                    "output": f"Pattern failed: {str(e)}",
                    "iterations": len(execution_history),
                    "phase": ExecutionPhase.FAILED.value,
                    "history": execution_history,
                }

                # Add detailed error information from failed steps if available
                if self.current_plan:
                    detailed_errors = []
                    for step in self.current_plan.steps:
                        if step.status == StepStatus.FAILED:
                            detailed_errors.append(
                                {
                                    "step_id": step.id,
                                    "step_name": step.name,
                                    "error": step.error,
                                    "error_type": step.error_type,
                                    "error_traceback": step.error_traceback,
                                }
                            )
                    if detailed_errors:
                        error_result["error_details"] = detailed_errors

                return error_result

            if isinstance(e, PatternExecutionError):
                raise

            # Wrap in pattern execution error with detailed context
            raise PatternExecutionError(
                "DAG Plan-Execute",
                f"Pattern failed: {str(e)}",
                context={
                    "task": task[:100],
                    "error_type": type(e).__name__,
                    "iterations": len(execution_history),
                },
                cause=e,
            )

    def pause_execution(self, reason: str = "User requested pause") -> None:
        """Pause the current execution."""
        logger.info(f"Pausing execution: {reason}")
        self._pause_reason = reason
        self._pause_timestamp = datetime.now()

        # Update phase to PAUSED
        self.phase = ExecutionPhase.PAUSED

        # Also pause the plan executor
        self.plan_executor.pause_execution()

        # Also pause all ReAct patterns
        for pattern in self.step_patterns.values():
            pattern.pause_execution()

        # Set the pause event
        self._pause_event.set()

    def resume_execution(self) -> None:
        """Resume paused execution."""
        if not self._pause_event.is_set():
            logger.warning("Execution is not paused, cannot resume")
            return

        logger.info("Resuming execution")
        self._pause_reason = None
        self._pause_timestamp = None

        # Update phase back to EXECUTING (or appropriate phase)
        # We can't know the exact previous phase, so default to EXECUTING
        self.phase = ExecutionPhase.EXECUTING

        # Clear the pause event to resume execution
        self._pause_event.clear()

        # Notify the condition to wake up waiting tasks
        async def _notify() -> None:
            async with self._pause_condition:
                self._pause_condition.notify_all()

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(_notify())
        except RuntimeError:
            pass

        # Also resume the plan executor
        self.plan_executor.resume_execution()

        # Also resume all ReAct patterns
        for pattern in self.step_patterns.values():
            pattern.resume_execution()

    def interrupt_execution(self, reason: str = "New user input received") -> None:
        """Interrupt current execution for plan modification."""
        logger.info(f"Interrupting execution: {reason}")
        self._execution_interrupted = True

        # If paused, resume to allow interruption to take effect
        if self._pause_event.is_set():
            self._pause_event.clear()

        # Also clear the plan executor's pause event
        if self.plan_executor._pause_event.is_set():
            self.plan_executor._pause_event.clear()

        # Also interrupt the plan executor
        self.plan_executor.interrupt_execution()

        # Also interrupt all ReAct patterns for immediate stop
        # AND clear their pause events so they can resume
        for pattern in self.step_patterns.values():
            pattern.interrupt_execution()
            if hasattr(pattern, "_pause_event") and pattern._pause_event.is_set():
                pattern._pause_event.clear()
                logger.info(
                    f"Resumed {type(pattern).__name__} from paused state for interruption"
                )

    def request_continuation(
        self, additional_task: str, context: Optional[Dict[str, Any]] = None
    ) -> None:
        """请求 continuation，由旧任务在适当时机自己处理"""
        logger.info(f"Continuation requested: {additional_task[:50]}...")

        self._pending_continuation = {
            "additional_task": additional_task,
            "context": context or {},
        }

        # 关键：不需要设置 _execution_interrupted，因为 continuation 是继续执行，不是中断执行
        # _execution_interrupted 会导致执行循环退出，而我们只是想要进入下一个迭代
        # self._execution_interrupted = True  # <-- 移除这行！

        # 如果暂停了，恢复执行以便处理 continuation
        if self._pause_event.is_set():
            logger.info("Resuming from paused state to process continuation")
            # Use resume_execution() to properly update phase and clear events
            self.resume_execution()

        # Note: We DON'T call interrupt_execution() on plan_executor or step_patterns
        # because continuation is NOT an interruption - it's a continuation of execution
        # The execution loop will check _pending_continuation and continue to next iteration

    def _compile_final_result(
        self, task: str, execution_history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Compile final result from execution history."""
        if not execution_history:
            return {
                "success": False,
                "error": "No execution iterations completed",
                "goal": task,
            }

        # Get the last iteration's results
        last_iteration = execution_history[-1]
        execution_results = last_iteration.get("results", [])

        # Count successful and failed steps
        successful_steps = [
            r for r in execution_results if r.get("status") in ["completed", "analyzed"]
        ]
        failed_steps = [r for r in execution_results if r.get("status") == "failed"]

        # Extract meaningful content from successful steps
        meaningful_content = []
        for step in successful_steps:
            step_result = step.get("result", {})
            if step_result and isinstance(step_result, dict):
                content = self._extract_meaningful_content(step_result)
                if content:
                    meaningful_content.append(
                        {
                            "step_name": step.get("step_name", "unknown"),
                            "content": content,
                        }
                    )

        # Build final result with detailed error information
        success = len(failed_steps) == 0

        # Extract detailed error information from failed steps
        detailed_errors = []
        if not success and self.current_plan:
            for step in self.current_plan.steps:
                if step.status == StepStatus.FAILED:
                    detailed_errors.append(
                        {
                            "step_id": step.id,
                            "step_name": step.name,
                            "error": step.error,
                            "error_type": step.error_type,
                            "error_traceback": step.error_traceback,
                        }
                    )

        # Determine the correct phase
        if self._pause_event.is_set():
            phase = ExecutionPhase.PAUSED.value
        else:
            phase = (
                ExecutionPhase.COMPLETED.value
                if success
                else ExecutionPhase.FAILED.value
            )

        result = {
            "success": success,
            "error": None if success else f"{len(failed_steps)} steps failed",
            "error_details": detailed_errors if not success else None,
            "goal": task,
            "output": getattr(self, "_final_answer", None)
            or self._generate_simple_summary(successful_steps, failed_steps),
            "iterations": len(execution_history),
            "phase": phase,
            "history": execution_history,
        }

        # Add file outputs if available
        file_outputs = self._extract_file_outputs()
        if file_outputs:
            result["file_outputs"] = file_outputs

        return result

    def _generate_simple_summary(
        self, successful_steps: List[Dict[str, Any]], failed_steps: List[Dict[str, Any]]
    ) -> str:
        """Generate a simple summary without LLM analysis."""
        if successful_steps and not failed_steps:
            return f"Task completed successfully with {len(successful_steps)} steps"
        elif successful_steps and failed_steps:
            return f"Partial success: {len(successful_steps)} steps completed, {len(failed_steps)} steps failed"
        else:
            return "Task failed with no successful steps"

    def _extract_file_outputs(self) -> List[Dict[str, str]]:
        """Get file outputs in array format for frontend consumption."""
        file_outputs = []

        # Check workspace for file outputs
        if self.workspace:
            try:
                workspace_files = self.workspace.get_output_files()
                if workspace_files:
                    for file_info in workspace_files:
                        filename = (
                            file_info.get("filename")
                            or file_info.get("file_path", "").split("/")[-1]
                        )
                        file_path = file_info.get("file_path", "")
                        relative_path = file_info.get("relative_path", "")

                        # Include workspace directory in download path
                        workspace_name = self.workspace.workspace_dir.name
                        full_relative_path = (
                            f"{workspace_name}/{relative_path}"
                            if relative_path
                            else f"{workspace_name}/{filename}"
                        )

                        file_outputs.append(
                            {
                                "filename": filename,
                                "file_path": file_path,
                                "relative_path": full_relative_path,
                                "download_path": full_relative_path,  # Just the path, not full URL
                            }
                        )
            except Exception as e:
                logger.warning(
                    f"Failed to get workspace output files for array format: {e}"
                )

        logger.info(f"File outputs array: {file_outputs}")
        return file_outputs

    def _compile_continuation_result(
        self,
        additional_task: str,
        execution_results: List[Dict[str, Any]],
        additional_steps: List[Any],
    ) -> Dict[str, Any]:
        """Compile result from task continuation execution."""
        # Count successful and failed steps
        successful_steps = [
            r for r in execution_results if r.get("status") in ["completed", "analyzed"]
        ]
        failed_steps = [r for r in execution_results if r.get("status") == "failed"]

        # Extract meaningful content from successful steps
        meaningful_content = []
        for step in successful_steps:
            step_result = step.get("result", {})
            if step_result and isinstance(step_result, dict):
                content = self._extract_meaningful_content(step_result)
                if content:
                    meaningful_content.append(
                        {
                            "step_name": step.get("step_name", "unknown"),
                            "content": content,
                        }
                    )

        # Generate summary
        if successful_steps and not failed_steps:
            summary = f"Task continuation completed successfully with {len(successful_steps)} additional steps"
        elif successful_steps and failed_steps:
            summary = f"Task continuation partially completed: {len(successful_steps)} steps completed, {len(failed_steps)} steps failed"
        else:
            summary = "Task continuation failed with no successful steps"

        # Combine all meaningful content
        combined_content = "\n\n".join([item["content"] for item in meaningful_content])

        result = {
            "success": len(failed_steps) == 0,
            "output": combined_content if combined_content else summary,
            "summary": summary,
            "additional_task": additional_task,
            "additional_steps": len(additional_steps),
            "successful_steps": len(successful_steps),
            "failed_steps": len(failed_steps),
            "execution_results": execution_results,
            "meaningful_content": meaningful_content,
        }

        # Add file outputs if available
        file_outputs = self._extract_file_outputs()
        if file_outputs:
            result["file_outputs"] = file_outputs

        return result

    def _extract_meaningful_content(self, result: Dict[str, Any]) -> str:
        """Intelligently extract the most meaningful content from step results."""
        if not isinstance(result, dict):
            return str(result).strip()

        # Try different keys in priority order
        content_sources = [
            # ReAct output results (highest priority)
            ("output", lambda x: str(x)),
            # Analysis results
            (
                "analysis_result",
                lambda x: self._extract_content_from_various_formats(x),
            ),
            (
                "direct_analysis",
                lambda x: self._extract_content_from_various_formats(x),
            ),
            # Tool execution results
            ("tool_result", lambda x: self._extract_content_from_various_formats(x)),
            ("execution_result", lambda x: str(x)),
            (
                "stdout",
                lambda x: str(x) if x and str(x).strip() else None,
            ),  # Only return if non-empty
            # General content fields
            ("content", lambda x: str(x)),
            ("text", lambda x: str(x)),
            ("answer", lambda x: str(x)),
            ("response", lambda x: str(x)),
            ("message", lambda x: str(x)),
            ("summary", lambda x: str(x)),
        ]

        for key, extractor in content_sources:
            if key in result and result[key]:
                try:
                    content = extractor(result[key])
                    if content and content.strip():
                        return content.strip()
                except Exception:
                    continue

        # If nothing found, try nested structures
        for key, value in result.items():
            if isinstance(value, dict) and "content" in value:
                try:
                    content = str(value["content"]).strip()
                    if content:
                        return content
                except Exception:
                    continue

        return ""

    def _extract_content_from_various_formats(self, data: Any) -> str:
        """Extract content from various nested formats."""
        if isinstance(data, str):
            return data
        elif isinstance(data, dict):
            # Try common content keys
            for key in ["content", "text", "output", "result", "answer"]:
                if key in data and data[key]:
                    return str(data[key])
            # Return the whole dict as string if no specific key found
            return str(data)
        else:
            return str(data)

    def get_detailed_error_info(self) -> Dict[str, Any]:
        """Get detailed error information for debugging and reporting."""
        if not self.current_plan or self.phase != ExecutionPhase.FAILED:
            return {"error": "No detailed error information available"}

        failed_steps = []
        for step in self.current_plan.steps:
            if step.status == StepStatus.FAILED:
                failed_steps.append(
                    {
                        "step_id": step.id,
                        "step_name": step.name,
                        "description": step.description,
                        "tool_names": step.tool_names,
                        "error": step.error,
                        "error_type": step.error_type,
                        "error_traceback": step.error_traceback,
                        "started_at": step.started_at.isoformat()
                        if step.started_at
                        else None,
                        "completed_at": step.completed_at.isoformat()
                        if step.completed_at
                        else None,
                        "dependencies": step.dependencies,
                    }
                )

        return {
            "phase": self.phase.value,
            "failed_steps_count": len(failed_steps),
            "total_steps_count": len(self.current_plan.steps),
            "failed_steps": failed_steps,
            "plan_id": self.current_plan.id,
            "goal": self.current_plan.goal,
        }

    def _get_tool_name(self, tool: Any) -> str:
        """Safely get tool name from various sources"""
        if hasattr(tool, "name"):
            name = tool.name
            return str(name) if name is not None else str(id(tool))
        elif hasattr(tool, "metadata") and hasattr(tool.metadata, "name"):
            name = tool.metadata.name
            return str(name) if name is not None else str(id(tool))
        else:
            return str(id(tool))

    def _get_tools_for_step(
        self, step: PlanStep, tool_map: Dict[str, Tool]
    ) -> List[Tool]:
        """Determine which tools this step should have access to"""

        if step.tool_names is None:
            # None means all available tools (legacy behavior)
            return list(tool_map.values())
        elif not step.tool_names:
            # Empty list means explicitly no tools - pure analysis or reasoning
            return []
        else:
            # Multiple tools - get all specified tools
            tools = []
            for tool_name in step.tool_names:
                if tool_name in tool_map:
                    tools.append(tool_map[tool_name])
            return tools

    def add_step_injection(
        self,
        step_id: str,
        pre_hook: Optional[Callable[[str, Dict[str, Any]], str]] = None,
        post_hook: Optional[Callable[[str, Dict[str, Any]], Dict[str, Any]]] = None,
    ) -> bool:
        """Add injection hooks to a DAG step"""
        if not self.current_plan:
            logger.warning("Cannot add injection hooks - no current plan")
            return False

        # Find the step by ID
        step = next((s for s in self.current_plan.steps if s.id == step_id), None)
        if not step:
            logger.warning(f"Step {step_id} not found in current plan")
            return False

        # Create or update injection hooks
        if step.injection is None:
            step.injection = StepInjection()

        if pre_hook:
            step.injection.pre_hook = pre_hook
        if post_hook:
            step.injection.post_hook = post_hook

        logger.info(f"Added injection hooks to step {step_id}")
        return True

    def get_execution_status(self) -> Dict[str, Any]:
        """Get current execution status for service compatibility"""
        return {
            "phase": self.phase.value if self.phase else None,
            "current_plan": self.current_plan.to_dict() if self.current_plan else None,
            "is_paused": self._pause_event.is_set(),
            "pause_reason": self._pause_reason,
            "execution_interrupted": self._execution_interrupted,
            "new_user_input": self._new_user_input,
        }

    def get_plan_info(self) -> Optional[Dict[str, Any]]:
        """Get current plan information including task_name.

        Returns:
            Dictionary with plan information including:
            - task_name: The generated task name for display
            - goal: The plan goal
            - steps_count: Number of steps in the plan
            - id: Plan ID
            Returns None if no plan exists.
        """
        if not self.current_plan:
            return None

        return {
            "task_name": self.current_plan.task_name,
            "goal": self.current_plan.goal,
            "steps_count": len(self.current_plan.steps),
            "id": self.current_plan.id,
            "iteration": self.current_plan.iteration,
        }

    def skip_step(self, step_id: str) -> bool:
        """Skip a specific step for service compatibility"""
        if self.current_plan:
            # Add to skipped steps in plan executor
            if hasattr(self.plan_executor, "skipped_steps"):
                self.plan_executor.skipped_steps.add(step_id)
                return True
        return False

    def reset_execution_state(self) -> None:
        """Reset the execution state to allow a fresh execution of the task.

        This method clears the current plan and execution-related flags while
        preserving the tracer and other configuration. This allows the same
        DAG pattern instance to be reused for a new task execution.
        """
        logger.info("Resetting DAG pattern execution state")

        # Clear the current plan
        self.current_plan = None

        # Reset execution flags
        self._execution_interrupted = False
        self._new_user_input = None
        self._pause_reason = None

        # Clear pause event if it exists
        if hasattr(self, "_pause_event") and self._pause_event:
            self._pause_event.clear()

        # Reset plan executor state if it has a reset method
        if hasattr(self.plan_executor, "reset"):
            self.plan_executor.reset()

        logger.info("DAG pattern execution state reset complete")

    async def handle_continuation(
        self, additional_task: str, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Handle task continuation by extending the current plan with additional steps.

        注意：新代码推荐使用 request_continuation() 方法，由旧任务在适当时机自己处理。
        这个方法保留用于兼容性，但在内部使用 request_continuation。
        """
        logger.info(f"handle_continuation called with: {additional_task[:50]}...")

        # 使用 request_continuation，由旧任务在适当时机处理
        self.request_continuation(additional_task, context)

        # 立即返回，不等待结果
        # 旧任务会在自己的主循环中处理 continuation
        return {
            "success": True,
            "output": "Continuation requested, will be processed by the running task",
            "continuation": True,
        }

    def _lookup_relevant_memories_with_context(
        self,
        memory_store: MemoryStore,
        query: str,
        category: Optional[str] = None,
        include_general: bool = True,
        limit: int = 5,
        similarity_threshold: Optional[float] = None,
        user_id: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Wrapper for lookup_relevant_memories that sets user context for thread execution.

        This method ensures that user context is properly set when calling
        lookup_relevant_memories from a different thread (e.g., via asyncio.to_thread).
        """
        # Set user context for this thread
        if user_id is not None:
            try:
                from .....web.user_isolated_memory import current_user_id

                context_token = current_user_id.set(user_id)
            except ImportError:
                # Fallback for non-web environment - proceed without user context
                from ..memory_utils import lookup_relevant_memories

                return lookup_relevant_memories(
                    memory_store,
                    query,
                    category,
                    include_general,
                    limit,
                    similarity_threshold,
                )

            try:
                # Call the original function with context set
                from ..memory_utils import lookup_relevant_memories

                return lookup_relevant_memories(
                    memory_store,
                    query,
                    category,
                    include_general,
                    limit,
                    similarity_threshold,
                )
            finally:
                # Clean up context
                current_user_id.reset(context_token)
        else:
            # No user ID provided, call function directly
            from ..memory_utils import lookup_relevant_memories

            return lookup_relevant_memories(
                memory_store,
                query,
                category,
                include_general,
                limit,
                similarity_threshold,
            )

    async def _store_memory(
        self,
        memory_insights: Dict[str, Any],
        task: str,
        execution_results: List[Dict[str, Any]],
        plan_id: Optional[str],
    ) -> None:
        """Store memory synchronously. Fast operation, doesn't block main execution significantly."""
        assert self.memory_store is not None, "memory_store must be set to store memory"
        try:
            from ..memory_utils import store_execution_result_memory

            should_store = memory_insights.get("should_store", False)

            if should_store:
                # Trace memory storage start
                if self.task_id:
                    await trace_memory_store_start(
                        self.tracer,
                        self.task_id,
                        data={
                            "task": task,
                            "memory_category": "execution_memory",
                            "plan_id": plan_id,
                            "classification": memory_insights.get("classification", {}),
                        },
                    )

                # Store memory (this may run in background)
                store_execution_result_memory(
                    self.memory_store,
                    execution_results,
                    task,
                    plan_id,
                    memory_insights.get("execution_insights", ""),
                    memory_insights.get("failure_analysis", ""),
                    memory_insights.get("classification", {}),
                )

                # Trace memory storage end
                reason = memory_insights.get("reason", "Unknown reason")
                if self.task_id:
                    await trace_memory_store_end(
                        self.tracer,
                        self.task_id,
                        data={
                            "storage_success": True,
                            "reason": reason,
                        },
                    )

                logger.info(
                    f"Stored valuable execution memory for task: {task[:100]}... Reason: {reason}"
                )
            else:
                # Trace memory storage decision (not storing)
                reason = memory_insights.get("reason", "Unknown reason")
                if self.task_id:
                    await trace_memory_store_end(
                        self.tracer,
                        self.task_id,
                        data={
                            "storage_success": False,
                            "reason": reason,
                            "decision": "not_worth_storing",
                        },
                    )

                logger.info(
                    f"Task not worth storing as memory: {task[:100]}... Reason: {reason}"
                )
        except Exception as e:
            logger.error(f"Background memory storage failed: {e}", exc_info=True)
