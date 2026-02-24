"""
Plan execution logic for DAG plan-execute pattern.
"""

import asyncio
import logging
import traceback
from collections import deque
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

if TYPE_CHECKING:
    from .dag_plan_execute import DAGPlanExecutePattern

from ....memory import MemoryStore
from ....memory.in_memory import InMemoryMemoryStore
from ....model.chat.basic.base import BaseLLM
from ....tools.adapters.vibe import Tool
from ....workspace import TaskWorkspace
from ...exceptions import DAGDeadlockError, DAGStepError
from ...trace import (
    TraceCategory,
    Tracer,
    trace_error,
    trace_step_end,
    trace_step_start,
    trace_task_end,
    trace_task_start,
)
from ...utils import ContextBuilder, StepExecutionResult
from .models import ExecutionPlan, PlanStep, StepStatus, UserInputMapper
from .step_agent_factory import StepAgentFactory

# Removed ReActPattern import to avoid circular import

logger = logging.getLogger(__name__)


class PlanExecutor:
    """Handles plan execution with dependency resolution and deadlock detection"""

    def __init__(
        self,
        llm: BaseLLM,
        tracer: Tracer,
        workspace: TaskWorkspace,
        memory_store: Optional[MemoryStore] = None,
        user_input_mapper: Optional[UserInputMapper] = None,
        parent_pattern: Optional["DAGPlanExecutePattern"] = None,
        context_compact_threshold: Optional[int] = None,
        max_concurrency: int = 4,
        step_agent_factory: Optional[StepAgentFactory] = None,
        compact_llm: Optional[BaseLLM] = None,
    ):
        self.llm = llm
        self.tracer = tracer
        self.workspace = workspace
        self.memory_store = memory_store or InMemoryMemoryStore()
        self.user_input_mapper = user_input_mapper or UserInputMapper()
        self.parent_pattern = parent_pattern
        self.max_concurrency = max_concurrency
        self.step_agent_factory = step_agent_factory
        self.compact_llm = (
            compact_llm or llm
        )  # Use main LLM if compact_llm not provided
        # Initialize context builder for dependency result management
        self.context_builder = ContextBuilder(
            llm, context_compact_threshold, compact_llm=self.compact_llm
        )
        # Store step execution results with message history
        self.step_execution_results: Dict[str, StepExecutionResult] = {}

        # Execution state
        self._pause_event = asyncio.Event()
        self._pause_condition = asyncio.Condition()
        self._execution_interrupted = False
        self.skipped_steps: Set[str] = set()
        self._semaphore = asyncio.Semaphore(max_concurrency)

    async def execute_plan(
        self,
        plan: ExecutionPlan,
        tool_map: Dict[str, Tool],
        skill_context: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Execute the plan using queue-driven concurrent execution

        Args:
            plan: Execution plan with steps
            tool_map: Tool name to tool mapping
            skill_context: Optional skill context to pass to step execution
        """
        logger.info(
            f"Executing plan {plan.id} with {len(plan.steps)} steps (max concurrency: {self.max_concurrency})"
        )

        # Reset interrupt flag at the start of execution
        self._execution_interrupted = False

        # Trace execution start
        trace_task_id = f"execute_{plan.id}"
        await trace_task_start(
            self.tracer,
            trace_task_id,
            TraceCategory.DAG,
            data={
                "plan_id": plan.id,
                "steps_count": len(plan.steps),
                "max_concurrency": self.max_concurrency,
                "iteration": plan.iteration,
            },
        )

        # Initialize queue with initial executable steps
        queue: deque = deque()
        completed_steps: Set[str] = set()
        execution_results: List[Dict[str, Any]] = []
        running_tasks: Set[str] = set()

        # Preserve existing step execution results for multi-iteration scenarios
        # This ensures that steps in later iterations can access results from earlier iterations
        logger.info(
            f"Starting execution with {len(self.step_execution_results)} existing step execution results"
        )

        # Get initial executable steps
        # Consider steps from previous iterations as completed if they have execution results
        completed_from_previous_iterations = set(self.step_execution_results.keys())
        total_completed = completed_steps.union(completed_from_previous_iterations)

        initial_executable = plan.get_executable_steps(
            total_completed, self.skipped_steps
        )
        for step in initial_executable:
            queue.append(step)

        logger.info(f"Initial executable steps: {[s.id for s in initial_executable]}")

        async def execute_step_with_completion(
            step: PlanStep,
        ) -> Optional[Dict[str, Any]]:
            """Execute a single step and handle completion"""
            step_id = step.id
            running_tasks.add(step_id)

            try:
                # Check for pause state before executing
                if self._pause_event.is_set():
                    logger.info(
                        f"Execution paused before step {step_id}, waiting for resume..."
                    )
                    await self._pause_event.wait()
                    logger.info(f"Execution resumed before step {step_id}")

                # Check if execution was interrupted
                if self._execution_interrupted:
                    logger.info(f"Execution interrupted for step {step_id}")
                    return None

                logger.info(
                    f"Executing step {step_id} (dependencies: {step.dependencies})"
                )

                async with self._semaphore:
                    result = await self._execute_step_with_react_agent(
                        step, tool_map, execution_results, skill_context
                    )

                # Handle successful completion
                step.status = StepStatus.COMPLETED
                step.result = result if isinstance(result, dict) else {"value": result}
                completed_steps.add(step_id)

                # Add to execution results
                execution_results.append(
                    {
                        "step_id": step_id,
                        "step_name": step.name,
                        "result": result,
                        "status": step.status.value,
                    }
                )

                logger.info(f"Step {step_id} completed successfully")

                # Check for new executable steps after this completion
                # Include steps from previous iterations in completed set
                completed_from_previous_iterations = set(
                    self.step_execution_results.keys()
                )
                total_completed = completed_steps.union(
                    completed_from_previous_iterations
                )

                new_executable = plan.get_executable_steps(
                    total_completed, self.skipped_steps
                )
                for new_step in new_executable:
                    # Check if step is not already in queue, running, or completed
                    if (
                        new_step.id not in [s.id for s in queue]
                        and new_step.id not in running_tasks
                        and new_step.id not in completed_steps
                        and new_step.id not in self.skipped_steps
                    ):
                        queue.append(new_step)
                        logger.info(f"Added new executable step {new_step.id} to queue")

                return result

            except InterruptedError:
                # Handle interruption for continuation
                logger.info(f"Step {step_id} interrupted for continuation")
                step.status = (
                    StepStatus.RUNNING
                )  # Leave as running, will be re-executed
                # Don't add to execution results or completed steps
                # Set the interrupt flag so the main execution loop knows to stop
                self._execution_interrupted = True
                return None

            except Exception as e:
                # Handle execution failure
                step.status = StepStatus.FAILED
                step.error = str(e)
                step.error_type = type(e).__name__
                step.error_traceback = traceback.format_exc()

                logger.error(f"Step {step_id} failed: {e}", exc_info=True)

                # Trace step failure
                await trace_error(
                    self.tracer,
                    f"step_{step_id}",
                    data={
                        "step_id": step_id,
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "step_name": step.name,
                    },
                )

                # Add failed step to execution results
                execution_results.append(
                    {
                        "step_id": step_id,
                        "step_name": step.name,
                        "result": {
                            "error": str(e),
                            "error_type": type(e).__name__,
                            "success": False,
                        },
                        "status": step.status.value,
                    }
                )

                # Even failed steps can unblock dependencies - but be more careful
                logger.info(
                    f"Checking for new executable steps after failure of {step_id}"
                )
                logger.info(f"Completed steps: {completed_steps}")
                logger.info(
                    f"Failed steps: {[s.id for s in plan.steps if s.status == StepStatus.FAILED]}"
                )

                # Include steps from previous iterations in completed set
                completed_from_previous_iterations = set(
                    self.step_execution_results.keys()
                )
                total_completed = completed_steps.union(
                    completed_from_previous_iterations
                )

                new_executable = plan.get_executable_steps(
                    total_completed, self.skipped_steps
                )
                logger.info(f"New executable steps: {[s.id for s in new_executable]}")

                for new_step in new_executable:
                    if (
                        new_step.id not in [s.id for s in queue]
                        and new_step.id not in running_tasks
                        and new_step.id not in completed_steps
                        and new_step.id not in self.skipped_steps
                    ):
                        # Double-check that this step's dependencies are actually met
                        dependencies_met = all(
                            dep in completed_steps
                            or any(
                                s.id == dep and s.status == StepStatus.FAILED
                                for s in plan.steps
                            )
                            for dep in new_step.dependencies
                        )

                        if dependencies_met:
                            queue.append(new_step)
                            logger.info(
                                f"Added new executable step {new_step.id} to queue (after failure)"
                            )
                        else:
                            logger.warning(
                                f"Step {new_step.id} dependencies not fully met, skipping"
                            )

                return None

            finally:
                running_tasks.remove(step_id)

        # Main execution loop with queue-driven concurrency
        tasks: List[asyncio.Task] = []

        while not plan.is_complete():
            # Check if execution was interrupted (check BEFORE pause to avoid issues)
            if self._execution_interrupted:
                logger.info(
                    "Execution interrupted for plan modification, stopping execution loop"
                )
                # Don't reset here, will be reset when execution is restarted
                break

            # Check for pause state
            if self._pause_event.is_set():
                logger.info(
                    f"Execution paused for plan {plan.id} (event is set, waiting...)"
                )
                # Use a Condition to properly wait for pause to be cleared
                # This avoids the busy loop problem with Event.wait()
                async with self._pause_condition:
                    await self._pause_condition.wait_for(
                        lambda: not self._pause_event.is_set()
                    )

                logger.info(f"Pause cleared, resuming execution for plan {plan.id}")

                # After resuming, check again if we were interrupted during the wait
                if self._execution_interrupted:
                    logger.info("Execution interrupted during pause wait, stopping")
                    break

            # Start new tasks if we have capacity and queue items
            while (
                len(tasks) < self.max_concurrency
                and queue
                and not self._pause_event.is_set()
                and not self._execution_interrupted
            ):
                step = queue.popleft()

                # Skip if already completed or running
                if step.id in completed_steps or step.id in running_tasks:
                    continue

                # Check if step should be skipped based on user input mapping
                input_id = self.user_input_mapper.get_input_id_by_step_id(step.id)
                if input_id:
                    connectivity = self._analyze_step_connectivity(
                        old_steps=plan.steps,
                        new_steps=[step],
                        completed_steps=completed_steps,
                    )

                    should_skip = self._should_skip_step(
                        step_id=step.id,
                        current_input_id=input_id,
                        new_input_id="current_input",
                        connectivity=connectivity,
                    )

                    if should_skip:
                        logger.info(
                            f"Skipping step {step.id} due to user input mapping"
                        )
                        step.status = StepStatus.SKIPPED
                        self.skipped_steps.add(step.id)

                        # Send trace event for skipped step
                        if hasattr(self, "tracer") and self.tracer:
                            trace_step_id = f"step_{step.id}"
                            await trace_step_end(
                                self.tracer,
                                trace_step_id,
                                step.id,
                                TraceCategory.DAG,
                                data={
                                    "step_id": step.id,
                                    "step_name": step.name,
                                    "status": StepStatus.SKIPPED.value,
                                    "skip_reason": "user_input_mapping",
                                },
                            )

                        continue

                # Create and start task
                task = asyncio.create_task(execute_step_with_completion(step))
                tasks.append(task)
                logger.info(f"Started task for step {step.id}")

            # Check for deadlock if no tasks are running and queue is empty but plan not complete
            if not tasks and not queue and not plan.is_complete():
                # Add a check to prevent infinite deadlock detection loops
                if not hasattr(self, "_deadlock_check_count"):
                    self._deadlock_check_count = 0
                self._deadlock_check_count += 1

                if self._deadlock_check_count > 3:
                    logger.error("Too many deadlock attempts, stopping execution")
                    break

                await self._check_deadlock(plan, completed_steps)
            else:
                # Reset deadlock check count when making progress
                if hasattr(self, "_deadlock_check_count"):
                    delattr(self, "_deadlock_check_count")

            # Wait for at least one task to complete
            if tasks:
                done, pending = await asyncio.wait(
                    tasks, return_when=asyncio.FIRST_COMPLETED
                )

                # Remove completed tasks
                tasks = list(pending)

                # Process completed tasks
                for task in done:
                    try:
                        await task  # Ensure any exceptions are handled
                    except InterruptedError:
                        logger.info(
                            "Task interrupted for continuation, stopping execution..."
                        )
                        self._execution_interrupted = True
                        # Break out of the for loop to handle continuation
                        break
                    except Exception as e:
                        logger.error(f"Task execution failed: {e}", exc_info=True)

                # Check if execution was interrupted during task processing
                if self._execution_interrupted:
                    logger.info("Execution interrupted, breaking main loop")
                    break
            else:
                # No tasks running, wait a bit before checking again
                await asyncio.sleep(0.1)

        # Cancel any remaining tasks
        for task in tasks:
            task.cancel()

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        # Mark steps that should be skipped due to conditional branches
        # Check all PENDING steps: if dependencies are met but step can't execute, mark as skipped
        for step in plan.steps:
            if step.status == StepStatus.PENDING and step.id not in self.skipped_steps:
                # Check if all dependencies are completed or skipped
                deps_met = all(
                    dep_id in completed_steps or dep_id in self.skipped_steps
                    for dep_id in step.dependencies
                )
                if deps_met:
                    # Dependencies are met, but step wasn't executed
                    # This means it was skipped due to conditional branch
                    if not step.can_execute(
                        completed_steps, self.skipped_steps, plan.active_branches
                    ):
                        logger.info(
                            f"Marking step {step.id} as skipped (conditional branch)"
                        )
                        step.status = StepStatus.SKIPPED
                        self.skipped_steps.add(step.id)

                        # Send trace event for skipped step
                        if hasattr(self, "tracer") and self.tracer:
                            trace_step_id = f"step_{step.id}"
                            await trace_step_end(
                                self.tracer,
                                trace_step_id,
                                step.id,
                                TraceCategory.DAG,
                                data={
                                    "step_id": step.id,
                                    "step_name": step.name,
                                    "status": StepStatus.SKIPPED.value,
                                    "skip_reason": "conditional_branch",
                                    "required_branch": step.required_branch,
                                },
                            )

        # Trace execution end
        await trace_task_end(
            self.tracer,
            trace_task_id,
            TraceCategory.DAG,
            data={
                "plan_id": plan.id,
                "completed_steps_count": len(completed_steps),
                "failed_steps_count": len(
                    [s for s in plan.steps if s.status == StepStatus.FAILED]
                ),
                "skipped_steps_count": len(
                    [s for s in plan.steps if s.status == StepStatus.SKIPPED]
                ),
                "iteration": plan.iteration,
            },
        )

        logger.info(f"Plan execution completed for {plan.id}")
        return execution_results

    def pause_execution(self) -> None:
        """Pause the current execution"""
        self._pause_event.set()
        logger.info("Execution paused")

    def resume_execution(self) -> None:
        """Resume paused execution"""
        self._pause_event.clear()
        logger.info("Execution resumed")

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

    def interrupt_execution(self) -> None:
        """Interrupt execution for plan modification"""
        self._execution_interrupted = True
        logger.info("Execution interrupted for plan modification")

    async def _execute_step_with_react_agent(
        self,
        step: PlanStep,
        tool_map: Dict[str, Tool],
        execution_results: Optional[List[Dict[str, Any]]] = None,
        skill_context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute a single step using ReAct agent

        Args:
            step: Plan step to execute
            tool_map: Tool name to tool mapping
            execution_results: Optional list of execution results
            skill_context: Optional skill context to pass to context builder
        """
        logger.info(f"Executing step {step.id}: {step.name}")

        # Trace step start with detailed context
        trace_step_id = f"step_{step.id}"
        step_start_data = {
            "step_id": step.id,
            "step_name": step.name,
            "tool_names": step.tool_names,
            "dependencies": step.dependencies,
            "description": step.description[:200] if step.description else "",
            "status": "starting",
            "start_time": datetime.now().isoformat(),
        }
        await trace_step_start(
            self.tracer,
            trace_step_id,
            step.id,
            TraceCategory.DAG,
            data=step_start_data,
        )

        step.status = StepStatus.RUNNING
        step.started_at = datetime.now()

        try:
            # Get tools for this step (handle steps with no tools)
            tool_names = step.get_available_tools()
            tools: List[Tool] = []

            if tool_names:
                for tool_name in tool_names:
                    tool = tool_map.get(tool_name)
                    if not tool:
                        raise DAGStepError(
                            step_id=step.id,
                            step_name=step.name,
                            message=f"Tool '{tool_name}' not found for step {step.id}",
                        )
                    tools.append(tool)

                logger.info(
                    f"Step {step.id} will use tools: {[t.metadata.name for t in tools]}"
                )

            # Use StepAgentFactory if available, otherwise fallback to direct ReAct pattern
            if self.step_agent_factory:
                # Create agent using factory based on step difficulty
                step_agent = self.step_agent_factory.create_step_agent(
                    step_name=step.name,
                    tools=tools,
                    difficulty=getattr(step, "difficulty", "hard"),
                )
                # Get the ReAct pattern from the agent
                react_pattern = step_agent.patterns[0] if step_agent.patterns else None
                # Type checking (ReActPattern is imported in TYPE_CHECKING block)
                if not react_pattern or not hasattr(react_pattern, "set_step_context"):
                    raise DAGStepError(
                        step_id=step.id,
                        step_name=step.name,
                        message="Failed to create ReAct pattern for step",
                    )
                # Set step context for proper tracing correlation
                react_pattern.set_step_context(step_id=step.id, step_name=step.name)
                # Register the ReAct pattern with the parent DAG pattern for pause control
                if self.parent_pattern and hasattr(
                    self.parent_pattern, "step_patterns"
                ):
                    self.parent_pattern.step_patterns[step.id] = react_pattern
            else:
                # Fallback to direct ReAct pattern creation
                from ..react import ReActPattern

                react_pattern = ReActPattern(
                    llm=self.llm,
                    tracer=self.tracer,
                    compact_llm=self.compact_llm,
                )
                # Set step context for proper tracing correlation
                react_pattern.set_step_context(step_id=step.id, step_name=step.name)
                # Register the ReAct pattern with the parent DAG pattern for pause control
                if self.parent_pattern and hasattr(
                    self.parent_pattern, "step_patterns"
                ):
                    self.parent_pattern.step_patterns[step.id] = react_pattern

            # Build context using ContextBuilder with original goal and skill context
            original_goal = (
                getattr(self.parent_pattern, "_original_goal", None)
                if self.parent_pattern
                else None
            )
            context_messages = await self.context_builder.build_context_for_step(
                step_name=step.name,
                step_description=step.description,
                dependencies=step.dependencies,
                dependency_results=self.step_execution_results,
                task_id=step.id,
                original_goal=original_goal,
                skill_context=skill_context,
            )

            # Add the current step task, with tool info and original goal context
            tool_names = step.get_available_tools()

            # Get original goal for context
            original_goal = (
                getattr(self.parent_pattern, "_original_goal", None)
                if self.parent_pattern
                else None
            )
            goal_reminder = (
                f"\nOVERALL GOAL: {original_goal}\n" if original_goal else ""
            )

            # Special handling for conditional nodes
            if step.is_conditional:
                valid_branches = list(step.conditional_branches.keys())
                task_message = (
                    f"{goal_reminder}"
                    f"Execute: {step.name} (Conditional Node)\n"
                    f"Description: {step.description}\n\n"
                    f"IMPORTANT: You must choose ONE of the following branches:\n"
                    f"{', '.join(valid_branches)}\n\n"
                    f"In your final JSON response, set the 'answer' field to ONLY contain the branch name "
                    f"(e.g., '{valid_branches[0]}' or '{valid_branches[1]}').\n\n"
                    f"Example:\n"
                    f'{{\n  "type": "final_answer",\n  "reasoning": "Based on the analysis, the answer was found",\n  "answer": "{valid_branches[0]}",\n  "success": true,\n  "error": null\n}}\n'
                )
            elif tool_names:
                task_message = (
                    f"{goal_reminder}"
                    f"Execute: {step.name}\n"
                    f"Description: {step.description}\n"
                    f"Available tools: {', '.join(tool_names)}\n"
                    f"You may use any of these tools as needed to complete the task."
                )
                if original_goal:
                    task_message += "\nRemember: This step contributes to achieving the overall goal above."
            else:
                task_message = f"{goal_reminder}Execute: {step.name}\nDescription: {step.description}"
                if original_goal:
                    task_message += "\nRemember: This step contributes to achieving the overall goal above."
            context_messages.append({"role": "user", "content": task_message})

            # Execute the step with enhanced messages
            result = await react_pattern.run_with_context(  # type: ignore[attr-defined]
                messages=context_messages,
                tools=tools,
            )

            # Ensure result is properly typed
            if not isinstance(result, dict):
                result = {"output": str(result), "success": True}

            step.completed_at = datetime.now()

            # Store step execution result with complete message history for ContextBuilder
            execution_history = result.get("execution_history", context_messages)

            step_execution_result = StepExecutionResult(
                step_id=step.id,
                messages=execution_history,  # Complete conversation history
                final_result=result,
                agent_name="ReAct",
                compact_available=True,
            )
            self.step_execution_results[step.id] = step_execution_result

            # Trace step completion with detailed execution information
            step_trace_data = {
                "step_id": step.id,
                "step_name": step.name,
                "execution_time": (step.completed_at - step.started_at).total_seconds(),
                "result": result,
                # Add execution details for better trace visibility
                "tool_names": step.tool_names,
                "status": StepStatus.COMPLETED.value,
                "start_time": step.started_at.isoformat() if step.started_at else None,
                "end_time": step.completed_at.isoformat()
                if step.completed_at
                else None,
            }

            # Extract meaningful execution details from result if available
            if isinstance(result, dict):
                # Include tool execution results
                if "tool_name" in result:
                    step_trace_data["executed_tool"] = result["tool_name"]
                if "tool_args" in result:
                    step_trace_data["tool_parameters"] = result["tool_args"]
                if "iterations" in result:
                    step_trace_data["react_iterations"] = result["iterations"]
                # Include success status
                if "success" in result:
                    step_trace_data["success"] = result["success"]

            # Check for agent-specific trace data in the result (added by format_query_result tools)
            # This avoids circular dependencies by letting tools add data directly to results
            if isinstance(result, dict) and "agent_trace_data" in result:
                agent_trace_data = result["agent_trace_data"]
                if agent_trace_data:
                    step_trace_data["agent_data"] = agent_trace_data
            # Also check nested result structures
            elif (
                isinstance(result, dict)
                and "result" in result
                and isinstance(result["result"], dict)
            ):
                nested_result = result["result"]
                if "agent_trace_data" in nested_result:
                    agent_trace_data = nested_result["agent_trace_data"]
                    if agent_trace_data:
                        step_trace_data["agent_data"] = agent_trace_data

            await trace_step_end(
                self.tracer,
                trace_step_id,
                step.id,
                TraceCategory.DAG,
                data=step_trace_data,
            )

            # Handle conditional nodes: extract branch from final answer
            if step.is_conditional:
                from .models import extract_branch_key_from_final_answer

                # Get final answer from result
                final_answer = None
                if isinstance(result, dict):
                    final_answer = result.get("final_answer") or result.get(
                        "output", ""
                    )

                if final_answer:
                    valid_branches = list(step.conditional_branches.keys())
                    branch_key = extract_branch_key_from_final_answer(
                        str(final_answer), valid_branches
                    )

                    if branch_key:
                        # Get the plan from parent pattern and set active branch
                        if (
                            self.parent_pattern
                            and hasattr(self.parent_pattern, "current_plan")
                            and self.parent_pattern.current_plan is not None
                        ):
                            plan = self.parent_pattern.current_plan
                            plan.set_active_branch(step.id, branch_key)
                            logger.info(
                                f"Conditional node {step.id} selected branch: {branch_key} -> {step.conditional_branches[branch_key]}"
                            )
                            step_trace_data["selected_branch"] = branch_key
                            step_trace_data["next_step"] = step.conditional_branches[
                                branch_key
                            ]
                    else:
                        # Branch key extraction failed - this is an error
                        error_msg = (
                            f"Conditional node {step.id} failed to return a valid branch key. "
                            f"Valid branches: {valid_branches}. "
                            f"Final answer: {str(final_answer)[:200]}"
                        )
                        logger.error(error_msg)

                        # Mark step as failed
                        step.status = StepStatus.FAILED
                        step.error = "Invalid branch key"
                        step.error_type = "ConditionalBranchError"

                        # Trace the failure
                        step_trace_data["branch_extraction_failed"] = True
                        step_trace_data["valid_branches"] = valid_branches
                        step_trace_data["final_answer_preview"] = str(final_answer)[
                            :500
                        ]

                        await trace_step_end(
                            self.tracer,
                            trace_step_id,
                            step.id,
                            TraceCategory.DAG,
                            data=step_trace_data,
                        )

                        # Raise error so ReAct can retry
                        raise DAGStepError(
                            step_id=step.id,
                            step_name=step.name,
                            message=error_msg,
                        )

            step.status = StepStatus.COMPLETED

            logger.info(
                f"Step {step.id} completed in {(step.completed_at - step.started_at).total_seconds():.2f}s"
            )
            return result

        except Exception as e:
            step.completed_at = datetime.now()
            step.status = StepStatus.FAILED
            step.error = str(e)
            step.error_type = type(e).__name__
            step.error_traceback = traceback.format_exc()

            # Trace step failure with detailed error information
            error_trace_data = {
                "step_id": step.id,
                "step_name": step.name,
                "error": str(e),
                "error_type": type(e).__name__,
                "execution_time": (step.completed_at - step.started_at).total_seconds(),
                "tool_names": step.tool_names,
                "status": StepStatus.FAILED.value,
                "start_time": step.started_at.isoformat() if step.started_at else None,
                "end_time": step.completed_at.isoformat()
                if step.completed_at
                else None,
                "error_traceback": step.error_traceback,
            }
            await trace_error(
                self.tracer,
                trace_step_id,
                data=error_trace_data,
            )

            logger.error(
                f"Step {step.id} failed after {(step.completed_at - step.started_at).total_seconds():.2f}s: {e}",
                exc_info=True,
            )
            raise

    def _detect_circular_dependencies(
        self, steps: List[PlanStep], blocked_deps: Dict[str, List[str]]
    ) -> List[List[str]]:
        """Detect circular dependencies using DFS"""
        # Build adjacency list for the dependency graph
        graph: Dict[str, List[str]] = {}
        for step in steps:
            graph[step.id] = []
            for dep in step.dependencies:
                if dep in blocked_deps.get(step.id, []):
                    graph[step.id].append(dep)

        # Use DFS to detect cycles
        visited = set()
        rec_stack = set()
        cycles = []

        def dfs(node: str, path: List[str]) -> None:
            if node in rec_stack:
                # Found a cycle
                cycle_start = path.index(node)
                cycle = path[cycle_start:]
                cycles.append(cycle)
                return

            if node in visited:
                return

            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in graph.get(node, []):
                if (
                    neighbor in graph
                ):  # Only consider nodes that are in our current graph
                    dfs(neighbor, path.copy())

            rec_stack.remove(node)
            path.pop()

        for node in graph:
            if node not in visited:
                dfs(node, [])

        return cycles

    def _analyze_step_connectivity(
        self,
        old_steps: List[PlanStep],
        new_steps: List[PlanStep],
        completed_steps: Set[str],
    ) -> Dict[str, Any]:
        """Analyze connectivity between old and new steps"""
        # This is a simplified implementation - the full logic would analyze
        # which steps are connected and how they affect dependency resolution
        return {
            "old_steps_count": len(old_steps),
            "new_steps_count": len(new_steps),
            "completed_steps_count": len(completed_steps),
            "is_connected": True,  # Simplified
        }

    async def _check_deadlock(
        self, plan: ExecutionPlan, completed_steps: Set[str]
    ) -> None:
        """Check for deadlock situation"""
        pending_steps = [s for s in plan.steps if s.status == StepStatus.PENDING]

        if not pending_steps:
            return

        # Analyze the deadlock situation
        pending_step_ids = [s.id for s in pending_steps]
        blocked_deps = {}

        for step in pending_steps:
            missing_deps = [
                dep for dep in step.dependencies if dep not in completed_steps
            ]
            blocked_deps[step.id] = missing_deps

        # Detect true circular dependencies using DFS
        circular_deps = self._detect_circular_dependencies(pending_steps, blocked_deps)

        # Enhanced logging for debugging
        logger.error("DAG deadlock detected!")
        logger.error(f"Pending steps: {pending_step_ids}")
        logger.error(f"Completed steps: {list(completed_steps)}")
        logger.error(f"Blocked dependencies: {blocked_deps}")
        if circular_deps:
            logger.error(f"True circular dependencies: {circular_deps}")
        else:
            logger.warning(
                "No true circular dependencies found - may be a temporary blocking situation"
            )

        # Check if any of the blocking dependencies are actually failed steps
        failed_steps = [s for s in plan.steps if s.status == StepStatus.FAILED]
        failed_step_ids = {s.id for s in failed_steps}

        can_continue = False
        steps_to_force = []

        for step in pending_steps:
            # If all missing dependencies are from failed steps, we can continue
            missing_deps = blocked_deps[step.id]
            if all(
                dep in failed_step_ids or dep in completed_steps for dep in missing_deps
            ):
                steps_to_force.append(step)
                can_continue = True

        if can_continue and steps_to_force:
            # Force execution of steps whose dependencies are only failed steps
            logger.warning(
                f"Forcing execution of steps with failed dependencies: {[s.id for s in steps_to_force]}"
            )

            # Mark failed dependencies as "completed" for the purpose of dependency resolution
            for step in steps_to_force:
                for dep in step.dependencies:
                    if dep in failed_step_ids:
                        completed_steps.add(dep)
                        logger.warning(
                            f"Marking failed step {dep} as completed to unblock {step.id}"
                        )

            return  # Continue execution
        else:
            # No steps can be forced - this is a true deadlock
            logger.error("No steps can be forced to continue execution")
            if circular_deps:
                logger.error(f"True circular dependencies detected: {circular_deps}")
            else:
                logger.error("No circular dependencies, but execution cannot continue")

        # Check if we have true circular dependencies
        if circular_deps:
            # True deadlock due to circular dependencies
            raise DAGDeadlockError(
                pending_steps=pending_step_ids,
                blocked_dependencies=blocked_deps,
                context={
                    "plan_id": plan.id,
                    "completed_steps": list(completed_steps),
                    "failed_steps": list(failed_step_ids),
                    "circular_dependencies": circular_deps,
                },
            )
        else:
            # No circular dependencies found - this might be a temporary situation
            # Check if there are any steps that could become executable
            potentially_executable = []
            for step in pending_steps:
                missing_deps = blocked_deps[step.id]
                # Check if missing dependencies are running
                running_steps = [
                    s for s in plan.steps if s.status == StepStatus.RUNNING
                ]
                running_step_ids = {s.id for s in running_steps}

                if any(dep in running_step_ids for dep in missing_deps):
                    potentially_executable.append(step.id)

            if potentially_executable:
                logger.info(
                    f"Steps {potentially_executable} may become executable when running dependencies complete"
                )
                # Wait a bit for running steps to complete
                await asyncio.sleep(1.0)
                return
            else:
                # No running dependencies - this is likely a real deadlock without cycles
                logger.error(
                    "No executable steps and no running dependencies. This appears to be a deadlock."
                )
                raise DAGDeadlockError(
                    pending_steps=pending_step_ids,
                    blocked_dependencies=blocked_deps,
                    context={
                        "plan_id": plan.id,
                        "completed_steps": list(completed_steps),
                        "failed_steps": list(failed_step_ids),
                        "circular_dependencies": [],
                        "note": "No circular dependencies found, but no progress possible",
                    },
                )

    def _should_skip_step(
        self,
        step_id: str,
        current_input_id: str,
        new_input_id: str,
        connectivity: Dict[str, Any],
    ) -> bool:
        """Determine if a step should be skipped based on user input mapping"""
        # Simplified implementation - the full logic would check if the step
        # is connected to the current user input context
        return False
