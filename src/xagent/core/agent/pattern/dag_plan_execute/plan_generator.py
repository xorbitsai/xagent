"""
Plan generation logic for DAG plan-execute pattern.
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from ....model.chat.basic.base import BaseLLM
from ....model.chat.token_context import add_token_usage
from ....tools.adapters.vibe import Tool
from ...context import AgentContext
from ...exceptions import DAGPlanGenerationError, LLMResponseError
from ...trace import trace_dag_plan_end, trace_dag_plan_start
from ...utils.compact import CompactConfig, CompactUtils
from ...utils.llm_utils import clean_messages, extract_json_from_markdown
from .models import (
    ChatResponse,
    ExecutionPlan,
    Interaction,
    InteractionType,
    PlanGeneratorResult,
    PlanStep,
)
from .schemas import ClassificationResponse

logger = logging.getLogger(__name__)


class PlanGenerator:
    """Handles plan generation and extension logic"""

    # Common planning strategy guidelines
    _PLANNING_GUIDELINES = (
        "PLANNING STRATEGY GUIDELINES:\n"
        "- For tasks requiring MANY ITERATIONS (like browser automation, coding, complex problem-solving):\n"
        "  * Use fewer steps but provide each step with as MANY relevant tools as possible\n"
        "  * These tasks often require trial-and-error, multiple approaches, and dynamic adaptation\n"
        "  * Having comprehensive tools allows the step to iterate and refine without being blocked\n"
        "  * Examples: browser tools (click, type, scroll, screenshot, extract, wait), coding tools (write, execute, debug, test)\n"
        "- For DATA ANALYSIS tasks: Create logical phases (data collection, processing, analysis, reporting)\n"
        "- For SIMPLE TASKS: Prefer single steps with comprehensive tool sets over many fine-grained steps\n"
        "- Only break down tasks when they truly require dependency chains or different skill sets\n"
        "- PRINCIPLE: More tools in a step = greater flexibility and ability to handle iterative, complex work\n\n"
    )

    _PLANNING_EXAMPLE = (
        "Example:\n"
        "{\n"
        '  "plan": {\n'
        '    "steps": [\n'
        '      {"id": "step1", "name": "Research", "description": "Gather information", "tool_names": ["web_search"], "dependencies": [], "difficulty": "hard"},\n'
        '      {"id": "step2", "name": "Analyze Data", "description": "Analyze collected information", "tool_names": ["execute_python_code"], "dependencies": ["step1"], "difficulty": "hard"},\n'
        '      {"id": "step3", "name": "Organize Results", "description": "Summarize and format findings", "tool_names": [], "dependencies": ["step1", "step2"], "difficulty": "easy"}\n'
        "    ]\n"
        "  }\n"
        "}\n\n"
    )

    def __init__(
        self,
        llm: BaseLLM,
        fast_llm: Optional[BaseLLM] = None,
        skill_manager: Optional[Any] = None,
        allowed_skills: Optional[List[str]] = None,
    ):
        self.llm = llm
        self.fast_llm = fast_llm
        self.skill_manager = skill_manager
        self.allowed_skills = allowed_skills

    def _validate_timeout(self, timeout: Any) -> Optional[int]:
        """Validate timeout value from LLM response"""
        if timeout is None:
            return None
        try:
            return int(timeout)
        except (ValueError, TypeError):
            logger.warning(f"Invalid timeout value from LLM: {timeout}")
            return None

    async def _generate_plan_with_flow(
        self,
        tracer: Any,
        goal: str,
        iteration: int,
        tools: List[Tool],
        history: List[Dict[str, Any]],
        prompt_builder_func: Any,
        is_extension: bool = False,
        current_plan: Optional[ExecutionPlan] = None,
        user_input_context: Optional[Dict[str, Any]] = None,
        context: Optional[AgentContext] = None,
        send_trace_event: bool = True,
        skill_context: Optional[str] = None,
    ) -> tuple[Optional[ExecutionPlan], List[PlanStep]]:
        """
        Execute the complete plan generation flow: start -> build prompt -> LLM -> parse -> validate -> end

        Returns:
            tuple: (execution_plan, additional_steps)
                For generation: execution_plan contains all steps, additional_steps is empty
                For extension: execution_plan is None, additional_steps contains new steps
        """
        # Send plan start event
        trace_task_id = (
            f"plan_extension_{uuid4()}" if is_extension else f"plan_{uuid4()}"
        )

        event_data = {
            "goal": goal,
            "iteration": iteration,
            "tools_count": len(tools),
            "history_length": len(history),
        }

        if is_extension:
            event_data.update(
                {
                    "extension": True,
                    "current_plan_id": current_plan.id if current_plan else None,
                    "current_steps_count": len(current_plan.steps)
                    if current_plan
                    else 0,
                    "user_input_context": user_input_context,
                }
            )

        if send_trace_event:
            await trace_dag_plan_start(tracer, trace_task_id, data=event_data)

        try:
            # skill_context is now pre-fetched and passed in from caller
            # This allows parallel execution with memory lookup
            pass

            # Build prompt using provided function
            if is_extension:
                prompt = prompt_builder_func(
                    goal,
                    iteration,
                    history,
                    current_plan,
                    user_input_context,
                    tools,
                    context,
                )
            else:
                prompt = prompt_builder_func(
                    goal, iteration, history, tools, context, skill_context
                )

            # Call LLM
            logger.info(
                f"Calling LLM for {'plan extension' if is_extension else 'planning'}, prompt: {prompt}"
            )
            response = await self._call_llm_with_retry(messages=prompt)

            content = response["content"] if isinstance(response, dict) else response
            logger.info(f"LLM response received, length: {len(str(content))}")

            # Parse response with retry mechanism
            error_context = None
            max_validation_retries = 2

            for validation_attempt in range(max_validation_retries + 1):
                try:
                    # For first attempt, use normal parsing. For retries, parse with error context.
                    if validation_attempt == 0:
                        parsed_data = await self._parse_plan_response_with_retry(
                            content, prompt, error_context=None
                        )
                    else:
                        parsed_data = await self._parse_plan_response_with_retry(
                            content, prompt, error_context=error_context
                        )

                    steps_data = parsed_data.get("steps", [])
                    task_name = parsed_data.get("task_name")

                    if not steps_data:
                        if is_extension:
                            logger.info(
                                "No additional steps generated for plan extension - LLM returned empty plan"
                            )
                            return None, []
                        else:
                            raise LLMResponseError(
                                "Failed to parse plan response from LLM",
                                response=content,
                                expected_format="JSON object with plan field containing steps array",
                                context={"iteration": iteration, "goal": goal[:100]},
                            )

                    # Create PlanStep objects and validate for first attempt
                    if is_extension:
                        # For extension, we need to handle existing step IDs
                        existing_step_ids = (
                            {step.id for step in current_plan.steps}
                            if current_plan
                            else set()
                        )
                        additional_steps = []

                        for step_data in steps_data:
                            step_id = step_data.get("id", str(uuid4()))
                            # Ensure unique step ID
                            while step_id in existing_step_ids:
                                step_id = str(uuid4())
                            existing_step_ids.add(step_id)

                            step = PlanStep(
                                id=step_id,
                                name=step_data["name"],
                                description=step_data["description"],
                                tool_names=step_data.get("tool_names", []),
                                dependencies=step_data.get("dependencies", []),
                                difficulty=step_data.get("difficulty", "hard"),
                                conditional_branches=step_data.get(
                                    "conditional_branches", {}
                                ),
                                required_branch=step_data.get("required_branch"),
                            )
                            additional_steps.append(step)

                        # Validate dependencies (both existing and new steps)
                        all_step_ids = existing_step_ids.copy()
                        for step in additional_steps:
                            all_step_ids.add(step.id)

                        for step in additional_steps:
                            invalid_deps = [
                                dep
                                for dep in step.dependencies
                                if dep not in all_step_ids
                            ]
                            if invalid_deps:
                                logger.warning(
                                    f"New step {step.id} has invalid dependencies: {invalid_deps}"
                                )
                                step.dependencies = [
                                    dep
                                    for dep in step.dependencies
                                    if dep in all_step_ids
                                ]

                        # Validate tool references for the new steps
                        self._validate_steps_tools(additional_steps, tools)

                        # Send trace event for extension
                        plan_data = {
                            "id": current_plan.id if current_plan else "unknown",
                            "goal": goal,
                            "task_name": task_name,  # Include task_name in trace event
                            "steps": [
                                {
                                    "id": step.id,
                                    "name": step.name,
                                    "description": step.description,
                                    "tool_names": step.tool_names,
                                    "dependencies": step.dependencies,
                                    "status": step.status.value
                                    if hasattr(step.status, "value")
                                    else str(step.status),
                                    "conditional_branches": step.conditional_branches,
                                    "required_branch": step.required_branch,
                                    "is_conditional": step.is_conditional,
                                }
                                for step in additional_steps
                            ],
                        }

                        if send_trace_event:
                            await trace_dag_plan_end(
                                tracer,
                                trace_task_id,
                                data={
                                    "steps_count": len(additional_steps),
                                    "plan_id": current_plan.id
                                    if current_plan
                                    else "unknown",
                                    "plan_data": plan_data,
                                    "extension": True,
                                },
                            )

                        return None, additional_steps
                    else:
                        # For generation, create all steps
                        steps = []
                        step_ids = set()

                        for step_data in steps_data:
                            step_id = step_data.get("id", str(uuid4()))
                            step_ids.add(step_id)

                            step = PlanStep(
                                id=step_id,
                                name=step_data["name"],
                                description=step_data["description"],
                                tool_names=step_data.get("tool_names", []),
                                dependencies=step_data.get("dependencies", []),
                                difficulty=step_data.get("difficulty", "hard"),
                                conditional_branches=step_data.get(
                                    "conditional_branches", {}
                                ),
                                required_branch=step_data.get("required_branch"),
                            )
                            steps.append(step)

                        # Validate dependencies
                        for step in steps:
                            invalid_deps = [
                                dep for dep in step.dependencies if dep not in step_ids
                            ]
                            if invalid_deps:
                                logger.warning(
                                    f"Step {step.id} has invalid dependencies: {invalid_deps}"
                                )
                                step.dependencies = [
                                    dep for dep in step.dependencies if dep in step_ids
                                ]

                        # Create execution plan
                        plan = ExecutionPlan(
                            id=str(uuid4()),
                            goal=goal,
                            steps=steps,
                            iteration=iteration,
                            task_name=task_name,  # Add task_name for display
                        )

                        # Validate the generated plan - this can raise DAGPlanGenerationError
                        self._validate_plan(plan, tools)

                        # Prepare plan data for trace event
                        plan_data = {
                            "id": plan.id,
                            "goal": goal,
                            "task_name": task_name,  # Include task_name in trace event
                            "steps": [
                                {
                                    "id": step.id,
                                    "name": step.name,
                                    "description": step.description,
                                    "tool_names": step.tool_names,
                                    "dependencies": step.dependencies,
                                    "status": step.status.value
                                    if hasattr(step.status, "value")
                                    else str(step.status),
                                    "conditional_branches": step.conditional_branches,
                                    "required_branch": step.required_branch,
                                    "is_conditional": step.is_conditional,
                                }
                                for step in steps
                            ],
                        }

                        # Send trace event for generation
                        if send_trace_event:
                            await trace_dag_plan_end(
                                tracer,
                                trace_task_id,
                                data={
                                    "steps_count": len(steps),
                                    "plan_id": plan.id,
                                    "plan_data": plan_data,
                                },
                            )

                        return plan, []

                except DAGPlanGenerationError as e:
                    if validation_attempt < max_validation_retries:
                        logger.warning(
                            f"Plan validation failed (attempt {validation_attempt + 1}/{max_validation_retries + 1}): {e}. Retrying with error context..."
                        )

                        # Build error context for retry
                        available_tools = set()
                        available_tools_list = []
                        for tool in tools:
                            tool_name = None
                            if hasattr(tool, "metadata") and hasattr(
                                tool.metadata, "name"
                            ):
                                tool_name = tool.metadata.name
                            elif hasattr(tool, "name"):
                                tool_name = tool.name

                            if tool_name and tool_name not in available_tools:
                                available_tools.add(tool_name)
                                available_tools_list.append(tool_name)

                        error_context = {
                            "error_type": "DAGPlanGenerationError",
                            "error_message": str(e),
                            "missing_tools": e.context.get("missing_tools", [])
                            if e.context
                            else [],
                            "available_tools": available_tools_list,
                        }

                        # Need to regenerate content with error context, so call LLM again
                        # Add error context to the messages for retry
                        missing_tools = (
                            e.context.get("missing_tools", []) if e.context else []
                        )
                        retry_messages = clean_messages(prompt) + [
                            {
                                "role": "user",
                                "content": f"\n\nPREVIOUS ERROR INFORMATION:\n"
                                f"Error Type: DAGPlanGenerationError\n"
                                f"Error Message: {str(e)}\n"
                                f"Missing Tools: {', '.join(missing_tools)}\n\n"
                                f"Available Tools:\n"
                                f"{', '.join(available_tools_list)}\n\n"
                                f"COMMON TOOL NAMING MISTAKES TO AVOID:\n"
                                f"- Use 'write_file' NOT 'write__file' (single underscore, not double)\n"
                                f"- Use 'read_file' NOT 'read__file'\n"
                                f"- Do NOT add any prefixes or suffixes to tool names\n"
                                f"- Do NOT invent tools that are not in the Available Tools list\n\n"
                                f"Please correct your plan by:\n"
                                f"1. Removing references to non-existent tools: {', '.join(missing_tools)}\n"
                                f"2. Using only tools from the Available Tools list above\n"
                                f"3. If you need functionality not covered by available tools, break the task into simpler steps that can use available tools\n\n"
                                f"Retry your plan generation with correct tool names.",
                            }
                        ]

                        # Call LLM again with error context
                        response = await self._call_llm_with_retry(
                            messages=retry_messages
                        )
                        content = (
                            response["content"]
                            if isinstance(response, dict)
                            else response
                        )

                        # Continue to next iteration to retry parsing and validation
                        continue
                    else:
                        # Max retries reached, re-raise the exception
                        logger.error(
                            f"Plan validation failed after {max_validation_retries + 1} attempts: {e}"
                        )
                        raise

        except Exception as e:
            if isinstance(e, (LLMResponseError, DAGPlanGenerationError)):
                raise
            else:
                logger.error(
                    f"Error in {'plan extension' if is_extension else 'plan generation'}: {e}"
                )
                if is_extension:
                    raise DAGPlanGenerationError(
                        f"Plan extension failed: {str(e)}",
                        goal=goal,
                        iteration=iteration,
                        llm_response=str(response) if "response" in locals() else None,
                        context={"goal": goal[:100], "tools_count": len(tools)},
                        cause=e,
                    )
                else:
                    raise DAGPlanGenerationError(
                        f"Plan generation failed: {str(e)}",
                        goal=goal,
                        iteration=iteration,
                        llm_response=str(response) if "response" in locals() else None,
                        context={"goal": goal[:100], "tools_count": len(tools)},
                        cause=e,
                    )

        # This should never be reached due to proper exception handling and early returns
        # but satisfies mypy's control flow analysis
        raise DAGPlanGenerationError(
            "Unexpected execution path in plan generation",
            goal=goal,
            iteration=iteration,
            context={"goal": goal[:100], "tools_count": len(tools)},
        )

    async def should_chat_directly(
        self,
        goal: str,
        tools: List[Tool],
        iteration: int,
        history: List[Dict[str, Any]],
        tracer: Any,
        context: Optional[AgentContext] = None,
    ) -> PlanGeneratorResult:
        """
        Quick check if the goal can be answered directly without planning.

        Returns:
            PlanGeneratorResult with type="chat" and chat_response if should chat
            PlanGeneratorResult with type="plan" (no plan object) if should generate plan

        This is called BEFORE skill selection to quickly determine the execution path.
        """
        logger.info(f"[should_chat_directly] Checking goal: {goal[:100]}")
        logger.info(
            f"[should_chat_directly] Context: history_count={len(history)}, tools_count={len(tools)}, iteration={iteration}"
        )
        if context:
            logger.info(
                f"[should_chat_directly] Agent context: {type(context).__name__}"
            )

        # Build classification prompt (no skill context needed for quick check)
        messages = self._build_classification_prompt(
            goal, history, tools, context, skill_context=None
        )
        logger.info(
            f"[should_chat_directly] Built prompt with {len(messages)} messages"
        )

        # Call LLM to analyze
        try:
            # Prepare output_config with JSON schema from Pydantic model
            # This ensures the required fields are present and provides type safety
            # Note: ClaudeLLM will automatically fix the schema for API compatibility
            output_config = {
                "format": {
                    "type": "json_schema",
                    "schema": ClassificationResponse.model_json_schema(),
                }
            }

            response = await self._call_llm_with_retry(
                messages=messages, output_config=output_config
            )

            content = response["content"] if isinstance(response, dict) else response
            usage = response.get("usage") if isinstance(response, dict) else None

            # Record token usage
            if usage:
                logger.info(
                    f"[should_chat_directly] LLM usage - "
                    f"prompt_tokens: {usage.get('prompt_tokens', 0)}, "
                    f"completion_tokens: {usage.get('completion_tokens', 0)}, "
                    f"total_tokens: {usage.get('total_tokens', 0)}"
                )

            logger.info(f"LLM analysis response received, length: {len(str(content))}")

            # Log raw response for debugging parsing issues
            logger.debug(f"Raw LLM response: {str(content)[:500]}")

            # Try to parse as JSON
            try:
                json_str = extract_json_from_markdown(content)
                parsed = json.loads(json_str)

                if not isinstance(parsed, dict):
                    raise ValueError("Response is not a dictionary")

                response_type = parsed.get("type", "")

                if response_type == "chat":
                    # Parse chat response
                    chat_data = parsed.get("chat", {})
                    message = chat_data.get("message", "")

                    interactions = []
                    for interaction_data in chat_data.get("interactions", []):
                        interaction_type = InteractionType(
                            interaction_data.get("type", "text_input")
                        )
                        interaction = Interaction(
                            type=interaction_type,
                            field=interaction_data.get("field"),
                            label=interaction_data.get("label"),
                            options=interaction_data.get("options"),
                            placeholder=interaction_data.get("placeholder"),
                            multiline=interaction_data.get("multiline"),
                            min=interaction_data.get("min"),
                            max=interaction_data.get("max"),
                            default=interaction_data.get("default"),
                            accept=interaction_data.get("accept"),
                            multiple=interaction_data.get("multiple"),
                        )
                        interactions.append(interaction)

                    chat_response = ChatResponse(
                        message=message,
                        interactions=interactions if interactions else None,
                        timeout=self._validate_timeout(chat_data.get("timeout")),
                    )

                    return PlanGeneratorResult(type="chat", chat_response=chat_response)

                elif response_type == "plan":
                    # Need to generate plan
                    return PlanGeneratorResult(type="plan", plan=None)

                else:
                    # Unknown type, assume plan generation is needed
                    logger.info(
                        f"Unknown response type '{response_type}', treating as plan"
                    )
                    return PlanGeneratorResult(type="plan", plan=None)

            except (json.JSONDecodeError, ValueError, KeyError) as e:
                logger.warning(f"Failed to parse LLM response as structured JSON: {e}")
                # On parse error, assume plan is needed
                return PlanGeneratorResult(type="plan", plan=None)

        except Exception as e:
            logger.error(f"Error during analysis: {e}")
            # On error, assume plan is needed
            return PlanGeneratorResult(type="plan", plan=None)

    def _build_classification_prompt(
        self,
        goal: str,
        history: List[Dict[str, Any]],
        tools: List[Tool],
        context: Optional[AgentContext] = None,
        skill_context: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """Build prompt for classifying user input (chat vs plan)"""
        logger.debug(
            f"[_build_classification_prompt] Building prompt with {len(history)} history items, {len(tools)} tools"
        )

        # Build tools context
        tools_context = ""
        if tools:
            tools_context = "\n\nAVAILABLE TOOLS:\n"
            for tool in tools:
                if tool.metadata:
                    tool_name = tool.metadata.name
                    tool_description = (
                        tool.metadata.description or f"Execute {tool_name}"
                    )
                else:
                    tool_name = getattr(tool, "name", "unknown_tool")
                    tool_description = getattr(
                        tool, "description", f"Execute {tool_name}"
                    )
                tools_context += f"- {tool_name}: {tool_description}\n"

        # Build system prompt
        system_prompt = """You are an intelligent task assistant. Analyze the user's input and decide:

1. **Direct Answer (type: "chat")** - If the user asks a simple question that you can answer directly without executing any tasks
2. **Need Clarification (type: "chat")** - If you need more information to help the user effectively
3. **Need Execution (type: "plan")** - If the user's request requires multi-step execution with tools

## Response Format

### For Chat (direct answer or clarification):
```json
{
  "type": "chat",
  "chat": {
    "message": "Your response to the user",
    "timeout": 60,
    "interactions": [
      {
        "type": "select_one|select_multiple|text_input|file_upload|confirm|number_input",
        "field": "field_name",
        "label": "Display label",
        "options": [{"value": "A", "label": "Option A"}],
        "placeholder": "...",
        "multiline": false,
        "min": 1,
        "max": 100,
        "default": true,
        "accept": [".csv", ".xlsx"],
        "multiple": false
      }
    ]
  }
}
```

### For Plan (execution required - just indicate this, don't generate the plan):
```json
{
  "type": "plan"
}
```

## Interaction Types
- **select_one**: Single choice from options
- **select_multiple**: Multiple choices from options
- **text_input**: Single-line text input
- **file_upload**: File upload with type restrictions
- **confirm**: Yes/No confirmation
- **number_input**: Numeric input with min/max

## Important Guidelines
- Use the SAME LANGUAGE as the user's goal for all text
- Only use "plan" type when multi-step tool execution is clearly needed
- For simple questions, clarifications, or information gathering, use "chat" type
- When returning type="plan", do NOT include plan details - just the type indicator
- interactions is optional - omit if no user input is needed
- timeout: Use 60 seconds ONLY if the task can proceed with default assumptions when the user does nothing (auto-continuation).
- **STRICT RULE**: If the task CANNOT proceed without specific user input or file upload (e.g., mandatory missing information), you MUST NOT include a "timeout".
- If you previously asked for input with a timeout, and the user replied "Continue" (auto-continued) but you still lack the mandatory info, ask again WITHOUT a timeout.
"""

        if tools_context:
            system_prompt += f"\n{tools_context}\n"

        # Build messages list with history
        messages = [{"role": "system", "content": system_prompt}]

        # Add conversation history as separate messages
        if history:
            messages.extend(history)

        # Add current goal as the final user message
        messages.append(
            {
                "role": "user",
                "content": f"User input: {goal}\n\nAnalyze and respond with appropriate JSON.",
            }
        )

        return messages

    async def generate_plan(
        self,
        goal: str,
        tools: List[Tool],
        iteration: int,
        history: List[Dict[str, Any]],
        tracer: Any,
        context: Optional[AgentContext] = None,
        skill_context: Optional[str] = None,
    ) -> ExecutionPlan:
        """
        Generate a DAG-structured execution plan

        Raises:
            DAGPlanGenerationError: When plan generation fails
            LLMResponseError: When LLM response is invalid
        """
        logger.info(f"Generating plan for iteration {iteration}")

        # Use the unified plan generation flow
        # Note: We don't send trace events here, letting the caller (dag_plan_execute.py) manage them
        plan, _ = await self._generate_plan_with_flow(
            tracer=tracer,
            goal=goal,
            iteration=iteration,
            tools=tools,
            history=history,
            prompt_builder_func=self._build_planning_prompt,
            is_extension=False,
            context=context,
            send_trace_event=False,  # Don't send trace events, caller will manage
            skill_context=skill_context,  # Pass pre-fetched skill context
        )

        # For plan generation (is_extension=False), plan should never be None
        assert plan is not None, "Plan generation should never return None"

        # Validate the generated plan
        self._validate_plan(plan, tools)

        return plan

    async def extend_plan(
        self,
        goal: str,
        tools: List[Tool],
        iteration: int,
        history: List[Dict[str, Any]],
        current_plan: ExecutionPlan,
        tracer: Any,
        user_input_context: Optional[Dict[str, Any]] = None,
        context: Optional[AgentContext] = None,
    ) -> List[PlanStep]:
        """Extend existing plan with additional steps (maintains immutability)"""
        logger.info(f"Extending plan for iteration {iteration}")

        # Use the unified plan generation flow
        _, additional_steps = await self._generate_plan_with_flow(
            tracer=tracer,
            goal=goal,
            iteration=iteration,
            tools=tools,
            history=history,
            prompt_builder_func=self._build_plan_extension_prompt,
            is_extension=True,
            current_plan=current_plan,
            user_input_context=user_input_context,
            context=context,
        )

        return additional_steps

    def _build_plan_extension_prompt(
        self,
        goal: str,
        iteration: int,
        history: List[Dict[str, Any]],
        current_plan: ExecutionPlan,
        user_input_context: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Tool]] = None,
        context: Optional[AgentContext] = None,
    ) -> List[Dict[str, str]]:
        """Build prompt for extending existing plan with additional steps"""

        history_context = ""
        if history:
            # Convert history to messages format for compacting
            history_messages = [
                {"role": "user", "content": json.dumps(iteration_data, default=str)}
                for iteration_data in history
            ]

            # Use CompactUtils.truncate_messages with default config
            max_messages = CompactConfig().fallback_truncate_count
            history_messages = CompactUtils.truncate_messages(
                history_messages, max_messages=max_messages
            )

            # Convert back to JSON string
            history_str = "\n".join([msg["content"] for msg in history_messages])
            history_context = f"\nPrevious execution history:\n{history_str}\n"

        # Get current plan summary
        current_steps_summary = []
        for step in current_plan.steps:
            status_emoji = (
                "⏳"
                if step.status.value == "pending"
                else "✅"
                if step.status.value == "completed"
                else "❌"
            )
            desc_preview = step.description[:100] + (
                "..." if len(step.description) > 100 else ""
            )
            current_steps_summary.append(
                f"{status_emoji} {step.id}: {step.name}\n   Description: {desc_preview}\n   Status: {step.status.value}"
            )

        current_plan_context = "\nCurrent plan steps:\n" + "\n\n".join(
            current_steps_summary
        )

        # Check if custom system prompt is provided in context
        custom_prompt = ""
        if (
            context
            and hasattr(context, "state")
            and "system_prompt" in context.state
            and context.state["system_prompt"]
        ):
            custom_prompt = f"\n\n{context.state['system_prompt']}\n\n"

        # Build tools context
        tools_context = ""
        if tools:
            tools_context = "\n\nAVAILABLE TOOLS:\n"
            for tool in tools:
                if tool.metadata:
                    tool_name = tool.metadata.name
                    tool_description = (
                        tool.metadata.description or f"Execute {tool_name}"
                    )
                else:
                    tool_name = getattr(tool, "name", "unknown_tool")
                    tool_description = getattr(
                        tool, "description", f"Execute {tool_name}"
                    )
                tools_context += f"- {tool_name}: {tool_description}\n"
        else:
            tools_context = "\n\nNote: No tools are currently available. Please generate additional conceptual steps that would help achieve the goal. Use hypothetical tool names that would be appropriate for each step."

        system_prompt = custom_prompt + (
            "You are an AI planning assistant that extends existing execution plans.\n"
            "The current plan has already been created and some steps may have been executed.\n"
            "Your task is to add ADDITIONAL steps that are needed to achieve the goal.\n"
            "You must NOT modify or recreate existing steps - only add new ones.\n"
            "New steps can depend on existing steps or other new steps.\n\n"
            "IMPORTANT: When a NEW USER REQUEST is provided:\n"
            "- Review all PENDING steps (marked with ⏳) and their descriptions carefully\n"
            "- Check if those pending steps' descriptions still align with the new user requirements\n"
            "- If pending steps will produce results that conflict with the new request, add new steps to modify/adjust those results\n"
            "- Remember: Pending steps cannot be changed, so you must add compensating steps if needed\n\n"
            f"You have access to the following tools that you can use in your plan steps:{tools_context}\n\n"
            f"{self._PLANNING_GUIDELINES}"
            "DIFFICULTY ASSESSMENT: For each additional step, you must assess its difficulty level:\n"
            "- 'easy': Simple tasks that can be done quickly (basic search, simple calculations, straightforward analysis)\n"
            "- 'hard': Complex tasks requiring deep thinking, creative problem-solving, or extensive processing\n\n"
            "Always return valid JSON format for the new steps only."
            f"{self._PLANNING_EXAMPLE}"
        )

        logger.debug(f"PLAN EXTENSION SYSTEM PROMPT:\n{system_prompt}")

        user_prompt = (
            f"Goal: {goal}\n"
            f"Iteration: {iteration}\n"
            f"Current Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"{current_plan_context}\n"
            f"{history_context}"
        )

        # Add new user input if provided
        if user_input_context and "new_input" in user_input_context:
            new_input = user_input_context["new_input"]
            user_prompt += f"\n{'=' * 80}\n"
            user_prompt += f"NEW USER REQUEST: {new_input}\n"
            user_prompt += f"{'=' * 80}\n"
            user_prompt += (
                "The user has provided additional requirements or modifications.\n"
            )
            user_prompt += "Review the PENDING steps above - if their descriptions conflict with this request, add new steps to address it.\n"

            # Add file information if available
            if "uploaded_files" in user_input_context:
                uploaded_files = user_input_context["uploaded_files"]
                file_info = user_input_context.get("file_info", [])
                user_prompt += f"\nUPLOADED FILES: {len(uploaded_files)} files available for processing:\n"
                for f in file_info:
                    file_name = f.get("name", "unknown")
                    file_size = f.get("size", 0)
                    file_type = f.get("type", "unknown")
                    user_prompt += f"- {file_name} ({file_size} bytes, {file_type})\n"
                user_prompt += "These files have been uploaded and are available in the workspace.\n"
                user_prompt += (
                    "You should consider these files when planning additional steps.\n"
                )
        else:
            user_prompt += "Based on the execution results and current plan, what ADDITIONAL steps are needed to achieve the goal?\n"

        user_prompt += (
            "Create additional steps as a JSON object with a 'plan' field containing a 'steps' array. Each step must have:\n"
            "- id: unique identifier (string, different from existing step IDs)\n"
            "- name: step name (string)\n"
            "- description: what this step does (string)\n"
            "- tool_names: list of tools available for this step (array of strings, can be empty)\n"
            "- dependencies: list of step IDs this step depends on (array of strings)\n"
            "- difficulty: 'easy' or 'hard' (string)\n"
            "CRITICAL DEPENDENCY RULES:\n"
            "- You can ONLY depend on step IDs that are listed in the 'Current plan steps' above\n"
            "- You CANNOT depend on step IDs that don't exist or that you are creating in this response\n"
            "- If you need multiple new steps that depend on each other, create them without dependencies first\n"
            "- Invalid dependencies will be automatically removed and may cause execution issues\n\n"
            "Format:\n"
            "{\n"
            '  "plan": {\n'
            '    "goal": "repeat the original goal here",\n'
            '    "steps": [\n'
            "      // step objects here\n"
            "    ]\n"
            "  }\n"
            "}\n\n"
            'If no additional steps are needed, return an object with empty steps array: {"plan": {"goal": "goal here", "steps": []}}\n'
            "Return only the JSON object, no additional text."
        )

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _build_skill_context(self, skill: Dict[str, Any]) -> str:
        """Build skill context string from skill info"""
        # Use the complete SKILL.md content directly
        content = skill.get("content", "")

        return f"## 🧰 Available Skill: {skill['name']}\n\n{content}"

    def _build_planning_prompt(
        self,
        goal: str,
        iteration: int,
        history: List[Dict[str, Any]],
        tools: List[Tool],
        context: Optional[AgentContext] = None,
        skill_context: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """Build prompt for plan generation"""

        # Check if custom system prompt is provided in context
        custom_prompt = ""
        use_custom_role = False
        if (
            context
            and hasattr(context, "state")
            and "system_prompt" in context.state
            and context.state["system_prompt"]
        ):
            # User's instruction replaces the default role "You are an AI planning assistant..."
            # But keeps the planning capability description
            custom_prompt = context.state["system_prompt"]
            use_custom_role = True

        # Build tools context
        tools_context = ""
        if tools:
            tools_context = "\n\nAVAILABLE TOOLS:\n"
            for tool in tools:
                if tool.metadata:
                    tool_name = tool.metadata.name
                    tool_description = (
                        tool.metadata.description or f"Execute {tool_name}"
                    )
                else:
                    tool_name = getattr(tool, "name", "unknown_tool")
                    tool_description = getattr(
                        tool, "description", f"Execute {tool_name}"
                    )
                tools_context += f"- {tool_name}: {tool_description}\n"
        else:
            tools_context = "\n\nNote: No tools are currently available. Please generate a conceptual execution plan that breaks down the task into logical steps. Use hypothetical tool names that would be appropriate for each step (e.g., 'data_analyzer', 'report_generator', etc.). Each step should be executable conceptually even without actual tools.\n"

        # Build system prompt with role or custom instruction
        if use_custom_role:
            # Use user's instruction as role, then add planning capability
            system_prompt = (
                custom_prompt
                + "\n\n"
                + (
                    "Use planning capabilities to break down tasks into steps.\n"
                    "Create plans as DAGs (Directed Acyclic Graphs) where steps can have dependencies.\n"
                    "Each step should specify which previous steps it depends on.\n"
                    "Steps with no dependencies can run in parallel.\n"
                    f"You have access to the following tools that you can use in your plan steps:{tools_context}\n"
                )
            )
        else:
            # Use default role
            system_prompt = custom_prompt + (
                "You are an AI planning assistant that creates detailed execution plans.\n"
                "Create plans as DAGs (Directed Acyclic Graphs) where steps can have dependencies.\n"
                "Each step should specify which previous steps it depends on.\n"
                "Steps with no dependencies can run in parallel.\n"
                f"You have access to the following tools that you can use in your plan steps:{tools_context}\n"
            )

        # Add skill context if available
        if skill_context:
            system_prompt += (
                "\n" + skill_context + "\n\n"
                "IMPORTANT: A skill is available above that provides domain knowledge and templates. "
                "Use this skill's knowledge and templates to improve the quality and relevance of your plan.\n"
            )

        system_prompt += (
            "IMPORTANT: Not every step needs to use a tool. Some steps can be pure analysis or organization tasks.\n"
            "- Use tools for: web searches, calculations, code execution, data processing\n"
            "- For pure analysis tasks (summarizing, organizing, explaining, formatting results): set tool_name to null or empty string\n\n"
            "CONDITIONAL BRANCHING:\n"
            "- Some steps can be CONDITIONAL NODES that branch execution based on runtime conditions\n"
            "- Use 'conditional_branches' field to define a step as a conditional node\n"
            '- Format: {"branch_key": "next_step_id"}\n'
            "- Example: A step that checks if human assistance is needed might have:\n"
            '  conditional_branches: {"human": "human_response_step", "kb": "knowledge_base_step"}\n'
            "- Conditional nodes MUST return a branch key (like 'human' or 'kb') as their final answer\n"
            "- Steps that depend on a conditional node should use 'required_branch' to specify which branch they belong to\n"
            '- Example: {"id": "human_response", "dependencies": ["check_human"], "required_branch": "human"}\n'
            "- Only steps on the selected branch will execute; others will be automatically skipped\n\n"
            f"{self._PLANNING_GUIDELINES}"
            "DIFFICULTY ASSESSMENT: For each step, you must assess its difficulty level:\n"
            "- 'easy': Simple tasks that can be done quickly (basic search, simple calculations, straightforward analysis)\n"
            "- 'hard': Complex tasks requiring deep thinking, creative problem-solving, or extensive processing\n\n"
            "LANGUAGE REQUIREMENT:\n"
            "- You MUST use the SAME LANGUAGE as the user's goal for: task_name, step names, and step descriptions\n"
            "- Examples:\n"
            "  * Chinese goal → Chinese task_name and steps (e.g., Data Analysis Report)\n"
            "  * English goal → English task_name and steps (e.g., Data Analysis Report)\n"
            "  * Japanese goal → Japanese task_name and steps (e.g., データ分析レポート)\n\n"
            "Always return valid JSON format."
        )

        logger.debug(f"PLAN GENERATION SYSTEM PROMPT:\n{system_prompt}")

        logger.info(
            f"[generate_plan] Goal: {goal[:100]}, history_count={len(history)}, iteration={iteration}"
        )

        # Build messages list with history
        messages = [{"role": "system", "content": system_prompt}]

        # Add conversation history as separate messages
        if history:
            messages.extend(history)
            # Log history preview
            first_msg = history[0]
            last_msg = history[-1]
            logger.info(
                f"[generate_plan] History: {len(history)} messages, first={first_msg.get('role')}, last={last_msg.get('role')}"
            )

        # Add current goal as the final user message
        messages.append(
            {
                "role": "user",
                "content": f"Goal: {goal}\nIteration: {iteration}\nCurrent Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\nCreate a step-by-step execution plan as a JSON object.",
            }
        )

        # Add skill reminder if available
        if skill_context:
            messages[-1]["content"] += (
                "\n"
                "A skill is available that can help with this task. "
                "Consider its knowledge and templates when creating the plan.\n\n"
            )

        # Add plan generation instructions
        messages[-1]["content"] += (
            f"**IMPORTANT**: The plan MUST include a 'task_name' field with a concise, descriptive title (3-10 words).\n\n"
            f"Required fields:\n"
            f"- task_name: A concise title for this task (REQUIRED, 3-10 words, meaningful for display in task lists, **MUST USE THE SAME LANGUAGE AS THE GOAL**)\n"
            f"- steps: array of execution steps\n\n"
            f"Each step must have:\n"
            f"- id: unique identifier (string)\n"
            f"- name: step name (string)\n"
            f"- description: what this step does (string)\n"
            f"- tool_names: list of tools available for this step (array of strings, can be empty)\n"
            f"- dependencies: list of step IDs this step depends on (array of strings)\n"
            f"- difficulty: 'easy' or 'hard' (string)\n"
            f"- requires_vision: true if step requires visual capabilities, false otherwise (boolean)\n"
            f"- conditional_branches: OPTIONAL dictionary mapping branch keys to next step IDs (object)\n"
            f"- required_branch: OPTIONAL branch key that this step requires (string)\n\n"
            f"Example with conditional branching:\n"
            f"{{\n"
            f'  "plan": {{\n'
            f'    "task_name": "Customer Support Query Resolution",\n'
            f'    "goal": "Customer support query handling",\n'
            f'    "steps": [\n'
            f'      {{"id": "check_intent", "name": "Check Intent", "description": "Determine if human assistance is explicitly requested", "tool_names": [], "dependencies": [], "difficulty": "easy", "conditional_branches": {{"human": "human_step", "kb": "kb_step"}}}},\n'
            f'      {{"id": "human_step", "name": "Human Response", "description": "Connect to human agent", "tool_names": [], "dependencies": ["check_intent"], "difficulty": "easy", "required_branch": "human"}},\n'
            f'      {{"id": "kb_step", "name": "KB Search", "description": "Search knowledge base", "tool_names": ["knowledge_search"], "dependencies": ["check_intent"], "difficulty": "hard", "required_branch": "kb"}}\n'
            f"    ]\n"
            f"  }}\n"
            f"}}\n\n"
            f"Example without conditional branching:\n"
            f"{{\n"
            f'  "plan": {{\n'
            f'    "task_name": "Data Analysis and Report",\n'
            f'    "goal": "{goal}",\n'
            f'    "steps": [\n'
            f'      {{"id": "step1", "name": "Research", "description": "Gather information", "tool_names": ["web_search", "zhipu_web_search"], "dependencies": [], "difficulty": "hard"}},\n'
            f'      {{"id": "step2", "name": "Analyze Data", "description": "Analyze collected information", "tool_names": ["execute_python_code"], "dependencies": ["step1"], "difficulty": "hard"}},\n'
            f'      {{"id": "step3", "name": "Organize Results", "description": "Summarize and format findings", "tool_names": [], "dependencies": ["step1", "step2"], "difficulty": "easy"}}\n'
            f"    ]\n"
            f"  }}\n"
            f"}}\n\n"
            f"Return only the JSON object, no additional text.\n"
            f"**IMPORTANT LANGUAGE REQUIREMENT**: Use the SAME LANGUAGE as the goal for:\n"
            f"- task_name (title)\n"
            f"- step names\n"
            f"- step descriptions\n\n"
            f"For example:\n"
            f"- If goal is in Chinese (中文), task_name must be in Chinese\n"
            f"- If goal is in English, task_name must be in English\n"
            f"- If goal is in Japanese (日本語), task_name must be in Japanese\n"
            f"- etc.\n"
        )

        return messages

    def _build_tools_schema(self, tools: List[Tool]) -> List[Dict[str, Any]]:
        """Build tools schema for LLM function calling"""
        tools_schema = []

        for tool in tools:
            try:
                # Get tool arguments schema from the tool's args_type
                args_type = tool.args_type()

                # Convert Pydantic model to JSON schema
                if hasattr(args_type, "model_json_schema"):
                    args_schema = args_type.model_json_schema()
                else:
                    # Fallback for simpler cases
                    args_schema = {"type": "object", "properties": {}, "required": []}

                # Get tool name and description safely
                if tool.metadata:
                    tool_name = tool.metadata.name
                    tool_description = (
                        tool.metadata.description or f"Execute {tool_name}"
                    )
                else:
                    tool_name = getattr(tool, "name", "unknown_tool")
                    tool_description = getattr(
                        tool, "description", f"Execute {tool_name}"
                    )

                tool_schema = {
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "description": tool_description,
                        "parameters": args_schema,
                    },
                }
                tools_schema.append(tool_schema)

            except Exception as e:
                # Get tool name safely for error message
                if tool.metadata:
                    tool_name = tool.metadata.name
                    tool_description = (
                        tool.metadata.description or f"Execute {tool_name}"
                    )
                else:
                    tool_name = getattr(tool, "name", "unknown_tool")
                    tool_description = getattr(
                        tool, "description", f"Execute {tool_name}"
                    )

                logger.warning(f"Failed to build schema for tool {tool_name}: {e}")
                # Fallback schema
                tool_schema = {
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "description": tool_description,
                        "parameters": {
                            "type": "object",
                            "properties": {},
                            "required": [],
                        },
                    },
                }
                tools_schema.append(tool_schema)

        return tools_schema

    def _parse_plan_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response into plan steps"""
        try:
            # Try to extract JSON from markdown code blocks first
            response = extract_json_from_markdown(response)

            # Try to extract JSON from response
            response = response.strip()

            logger.info(f"response: {response}")
            # Find JSON object in response
            start_idx = response.find("{")
            end_idx = response.rfind("}") + 1

            if start_idx < 0 or end_idx <= start_idx:
                raise LLMResponseError(
                    "Could not find valid JSON object in response",
                    response=response,
                    expected_format="JSON object with plan field containing steps array",
                    context={"response_preview": response[:200]},
                )

            json_str = response[start_idx:end_idx]
            data = json.loads(json_str)

            # Handle both formats: {"plan": {"steps": [...]}} and {"steps": [...]}
            plan_data = None

            if isinstance(data, dict):
                if "plan" in data:
                    # Standard format: {"plan": {"steps": [...]}}
                    plan_data = data["plan"]
                elif "steps" in data:
                    # Alternative format: {"steps": [...]}
                    plan_data = data
                    logger.info(
                        "Using alternative JSON format with 'steps' at root level"
                    )

            if (
                plan_data is None
                or not isinstance(plan_data, dict)
                or "steps" not in plan_data
            ):
                raise LLMResponseError(
                    "Response missing valid 'steps' field. Expected format: {'plan': {'steps': [...]}} or {'steps': [...]}",
                    response=response,
                    expected_format="JSON object with steps array (either in 'plan' field or at root)",
                    context={"parsed_data": str(data)[:200]},
                )

            steps = plan_data["steps"]
            if not isinstance(steps, list):
                raise LLMResponseError(
                    "Steps field is not an array",
                    response=response,
                    expected_format="JSON object with plan field containing steps array",
                    context={"steps_type": type(steps).__name__},
                )

            # Validate step structure
            validated_steps = []
            for i, step in enumerate(steps):
                if not isinstance(step, dict):
                    logger.warning(f"Step {i} is not a dictionary")
                    continue

                # Ensure required fields
                required_fields = ["name", "description", "tool_names"]
                if not all(field in step for field in required_fields):
                    logger.warning(f"Step {i} missing required fields: {step}")
                    continue

                # Set default ID if not provided
                if "id" not in step:
                    step["id"] = f"step_{i + 1}"

                # Set default dependencies if not provided
                if "dependencies" not in step:
                    step["dependencies"] = []

                validated_steps.append(step)

            # Extract task_name if present
            task_name = plan_data.get("task_name")

            # Log if task_name is missing
            if not task_name:
                logger.warning(
                    f"LLM did not generate task_name in plan. Plan data keys: {list(plan_data.keys())}"
                )
            else:
                logger.info(f"LLM generated task_name: {task_name}")

            return {"steps": validated_steps, "task_name": task_name}

        except json.JSONDecodeError as e:
            raise LLMResponseError(
                "Failed to parse JSON response from LLM",
                response=response,
                expected_format="JSON object with plan field containing steps array",
                context={"parse_error": str(e), "response_preview": response[:200]},
                cause=e,
            )
        except Exception as e:
            import traceback

            error_traceback = traceback.format_exc()
            logger.error(
                f"Failed to parse plan response: {str(e)}\ncontent: \n{response}\n{error_traceback}"
            )
            raise LLMResponseError(
                "Failed to parse plan response due to unexpected error",
                response=response,
                expected_format="JSON object with plan field containing steps array",
                context={"error": str(e), "traceback": error_traceback},
                cause=e,
            )

    async def _call_llm_with_retry(
        self, messages: List[Dict[str, str]], **kwargs: Any
    ) -> Any:
        """Call LLM with retry mechanism: JSON mode first, then fallback to normal mode."""
        try:
            # Clean messages before sending to LLM
            cleaned_messages = clean_messages(messages)

            # Use streaming API to collect complete response
            full_content = ""
            usage = {}
            tool_calls = []

            # Check if output_config is provided (for structured outputs with JSON schema)
            # If not, fall back to response_format for simple JSON mode
            llm_params = {}
            if "output_config" in kwargs:
                # Use output_config for structured outputs with JSON schema
                llm_params["output_config"] = kwargs.pop("output_config")
            else:
                # Fall back to simple JSON object mode
                llm_params["response_format"] = {"type": "json_object"}

            async for chunk in self.llm.stream_chat(
                messages=cleaned_messages,
                **llm_params,
                **kwargs,
            ):
                if chunk.is_token():
                    full_content += chunk.delta
                elif chunk.is_tool_call():
                    tool_calls = chunk.tool_calls
                elif chunk.is_usage():
                    usage = chunk.usage
                elif chunk.is_error():
                    raise RuntimeError(f"LLM stream error: {chunk.content}")

            # Record token usage
            if usage:
                logger.info(
                    f"Plan generation LLM usage - prompt_tokens: {usage.get('prompt_tokens', 0)}, "
                    f"completion_tokens: {usage.get('completion_tokens', 0)}, "
                    f"total_tokens: {usage.get('total_tokens', 0)}"
                )

                # Add token usage to tracker
                add_token_usage(
                    input_tokens=usage.get("prompt_tokens", 0),
                    output_tokens=usage.get("completion_tokens", 0),
                    model=self.llm.model_name,
                    call_type="plan_generation",
                )

            # Return format (compatible with original chat())
            if tool_calls:
                return {
                    "content": full_content,
                    "tool_calls": tool_calls,
                    "usage": usage,
                }
            return {"content": full_content, "usage": usage}

        except Exception as e:
            logger.warning(f"JSON mode call failed, retrying with normal mode: {e}")

            # Fallback: Use streaming API to retry (without JSON mode)
            try:
                full_content = ""
                usage = {}

                async for chunk in self.llm.stream_chat(
                    messages=cleaned_messages,
                    **kwargs,
                ):
                    if chunk.is_token():
                        full_content += chunk.delta
                    elif chunk.is_usage():
                        usage = chunk.usage
                    elif chunk.is_error():
                        raise RuntimeError(f"LLM stream error: {chunk.content}")

                # Record token usage for fallback
                if usage:
                    add_token_usage(
                        input_tokens=usage.get("prompt_tokens", 0),
                        output_tokens=usage.get("completion_tokens", 0),
                        model=self.llm.model_name,
                        call_type="plan_generation_fallback",
                    )

                return {"content": full_content, "usage": usage}
            except Exception as e2:
                logger.error(f"Normal mode call also failed: {e2}")
                raise

    async def _parse_plan_response_with_retry(
        self,
        content: str,
        messages: List[Dict[str, str]],
        error_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Parse plan response with retry mechanism."""
        # First attempt to parse
        try:
            parsed_data = self._parse_plan_response(content)
            if parsed_data.get("steps"):
                return parsed_data
        except LLMResponseError as e:
            logger.warning(f"First parsing attempt failed: {e}")
            # Continue to retry with error context

        # If parsing failed, retry with non-JSON mode and error information
        logger.warning(
            "JSON parsing failed, retrying LLM call without JSON mode with format clarification and error context"
        )

        # Retry LLM call without JSON mode with clearer format instructions and error context
        cleaned_messages = clean_messages(messages)

        # Build error context for retry
        error_message = ""
        if error_context:
            error_message = "\n\nPREVIOUS ERROR INFORMATION:\n"
            error_message += (
                f"Error Type: {error_context.get('error_type', 'Unknown')}\n"
            )
            error_message += f"Error Message: {error_context.get('error_message', 'Unknown error')}\n"

            # Add specific guidance based on error type
            if error_context.get("error_type") in [
                "DAGStepError",
                "DAGPlanGenerationError",
            ]:
                missing_tools = error_context.get("missing_tools", [])
                if missing_tools:
                    error_message += f"Missing Tools: {', '.join(missing_tools)}\n"
                    error_message += "IMPORTANT: Only use tools from the AVAILABLE TOOLS list provided above.\n"
                    error_message += "Do NOT invent or assume tool names that don't exist in the available tools.\n"

            error_message += (
                "\nPlease correct your plan based on this error information and retry."
            )

        content = (
            f'Please ensure your response is a valid JSON object with the exact format:\n{{\n  "plan": {{\n    "steps": [...]\n  }}\n}}\n\nThe \'steps\' array should contain the execution steps. Do not include any other fields at the top level.{error_message}'
            f"{self._PLANNING_EXAMPLE}"
        )

        # Add format clarification message with error context
        retry_messages = cleaned_messages + [
            {
                "role": "user",
                "content": content,
            }
        ]

        new_response = await self.llm.chat(messages=retry_messages)
        new_content = (
            new_response.get("content", "")
            if isinstance(new_response, dict)
            else str(new_response)
        )

        # Parse the new response
        try:
            parsed_data = self._parse_plan_response(new_content)
            if parsed_data.get("steps"):
                logger.info("Second parsing attempt with error context succeeded")
                return parsed_data
        except LLMResponseError as e:
            logger.warning(f"Second parsing attempt with error context failed: {e}")

        logger.warning("Both parsing attempts failed, returning empty result")
        return {"steps": [], "task_name": None}

    def _validate_steps_tools(self, steps: List[PlanStep], tools: List[Tool]) -> None:
        """
        Validate tool references for a list of steps

        Args:
            steps: List of PlanStep objects to validate
            tools: Available tools

        Raises:
            DAGPlanGenerationError: If any step references non-existent tools
        """
        # Build tool name map for validation
        available_tools = set()
        for tool in tools:
            if hasattr(tool, "metadata") and hasattr(tool.metadata, "name"):
                available_tools.add(tool.metadata.name)
            elif hasattr(tool, "name"):
                available_tools.add(tool.name)

        # Check for missing tools
        missing_tools = []
        for step in steps:
            if step.tool_names:
                for tool_name in step.tool_names:
                    if tool_name not in available_tools:
                        missing_tools.append(f"{step.id}:{tool_name}")
                        logger.warning(
                            f"Step {step.id} references missing tool: {tool_name}"
                        )

        if missing_tools:
            logger.error(f"Step validation failed: missing tools {missing_tools}")
            raise DAGPlanGenerationError(
                "Generated plan references non-existent tools",
                goal="",
                iteration=1,
                llm_response=None,
                context={
                    "missing_tools": missing_tools,
                    "available_tools": list(available_tools),
                },
            )

    def _validate_plan(self, plan: ExecutionPlan, tools: List[Tool]) -> None:
        """
        Validate the generated plan for tool existence

        Args:
            plan: The generated execution plan
            tools: Available tools

        Raises:
            DAGPlanGenerationError: If validation fails
        """
        logger.info(f"Validating plan {plan.id} with {len(plan.steps)} steps")

        # Build tool name map for validation
        available_tools = set()
        for tool in tools:
            if hasattr(tool, "metadata") and hasattr(tool.metadata, "name"):
                available_tools.add(tool.metadata.name)
            elif hasattr(tool, "name"):
                available_tools.add(tool.name)

        logger.debug(f"Available tools for validation: {available_tools}")

        # Check for missing tools
        missing_tools = []
        for step in plan.steps:
            if step.tool_names:
                for tool_name in step.tool_names:
                    if tool_name not in available_tools:
                        missing_tools.append(f"{step.id}:{tool_name}")
                        logger.warning(
                            f"Step {step.id} references missing tool: {tool_name}"
                        )

        if missing_tools:
            logger.error(f"Plan validation failed: missing tools {missing_tools}")
            raise DAGPlanGenerationError(
                "Generated plan references non-existent tools",
                goal=plan.goal,
                iteration=getattr(plan, "iteration", 1),
                llm_response=None,
                context={
                    "missing_tools": missing_tools,
                    "available_tools": list(available_tools),
                },
            )

        # Validate conditional branches
        available_step_ids = {step.id for step in plan.steps}
        invalid_branches = []
        for step in plan.steps:
            if step.conditional_branches:
                for branch_key, target_step_id in step.conditional_branches.items():
                    if target_step_id not in available_step_ids:
                        invalid_branches.append(
                            f"{step.id}.{branch_key} -> {target_step_id}"
                        )
                        logger.warning(
                            f"Step {step.id} branch '{branch_key}' points to non-existent step: {target_step_id}"
                        )

        if invalid_branches:
            logger.error(f"Plan validation failed: invalid branches {invalid_branches}")
            raise DAGPlanGenerationError(
                "Generated plan has conditional branches pointing to non-existent steps",
                goal=plan.goal,
                iteration=getattr(plan, "iteration", 1),
                llm_response=None,
                context={
                    "invalid_branches": invalid_branches,
                    "available_step_ids": list(available_step_ids),
                },
            )

        logger.info(f"Plan {plan.id} validation passed successfully")
