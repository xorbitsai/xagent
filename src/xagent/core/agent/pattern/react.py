"""ReAct (Reasoning and Acting) Pattern Implementation

This module implements a simplified ReAct pattern where the LLM must output
structured actions, eliminating parsing complexity and improving stability.
"""

__all__ = ["ReActPattern", "ReActStepType"]

import asyncio
import json
import logging
import time
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from json_repair import loads as repair_loads
from pydantic import BaseModel, Field, ValidationError

from ...memory import MemoryStore
from ...model.chat.basic.base import BaseLLM
from ...tools.adapters.vibe import Tool
from ..context import AgentContext
from ..exceptions import (
    LLMNotAvailableError,
    MaxIterationsError,
    PatternExecutionError,
    ToolNotFoundError,
)
from ..trace import (
    TraceCategory,
    Tracer,
    trace_action_end,
    trace_action_start,
    trace_ai_message,
    trace_error,
    trace_llm_call_start,
    trace_memory_generate_end,
    trace_memory_generate_start,
    trace_memory_retrieve_end,
    trace_memory_retrieve_start,
    trace_memory_store_end,
    trace_memory_store_start,
    trace_task_completion,
    trace_task_end,
    trace_task_start,
    trace_tool_execution_start,
    trace_user_message,
)
from ..transcript import normalize_transcript_messages
from ..utils.compact import CompactConfig, CompactUtils
from ..utils.llm_utils import clean_messages
from .base import AgentPattern, notify_condition
from .memory_utils import enhance_goal_with_memory, store_react_task_memory

logger = logging.getLogger(__name__)

CONTEXT_KEY_FILE_INFO = "file_info"
CONTEXT_KEY_UPLOADED_FILES = "uploaded_files"


class ReActStepType(Enum):
    """Types of steps in ReAct execution"""

    THOUGHT = "thought"
    ACTION = "action"
    OBSERVATION = "observation"
    FINAL_ANSWER = "final_answer"


class Action(BaseModel):
    """Structured action output from LLM."""

    type: str = Field(description="Type of action: 'tool_call' or 'final_answer'")
    reasoning: str = Field(description="Reasoning for this action")

    # For tool calls
    tool_name: Optional[str] = Field(None, description="Name of the tool to call")
    tool_args: Optional[Dict[str, Any]] = Field(
        None, description="Arguments for the tool"
    )

    # For final answer
    answer: Optional[str] = Field(
        None, description="Final answer when type is 'final_answer'"
    )
    success: Optional[bool] = Field(
        True, description="Whether the final answer represents success (default: True)"
    )
    error: Optional[str] = Field(None, description="Error message if success is False")

    # Allow additional fields for flexibility
    code: Optional[str] = Field(None, description="Code content for programming tasks")

    class Config:
        extra = "allow"  # Allow extra fields for flexibility


class ToolRegistry:
    """Registry for managing available tools."""

    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool."""
        self._tools[tool.metadata.name] = tool

    def register_all(self, tools: List[Tool]) -> None:
        """Register multiple tools."""
        for tool in tools:
            self.register(tool)

    def get(self, tool_name: str) -> Tool:
        """Get a tool by name."""
        if tool_name not in self._tools:
            raise ToolNotFoundError(f"Tool '{tool_name}' not found")
        return self._tools[tool_name]

    def list_tools(self) -> List[str]:
        """List all registered tool names."""
        return list(self._tools.keys())

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Get schemas for all registered tools."""
        schemas = []
        for tool in self._tools.values():
            schema = {
                "type": "function",
                "function": {
                    "name": tool.metadata.name,
                    "description": tool.metadata.description,
                    "parameters": self._get_tool_parameters(tool),
                },
            }
            schemas.append(schema)
        return schemas

    def _get_tool_parameters(self, tool: Tool) -> Dict[str, Any]:
        """Extract tool parameters schema."""
        try:
            args_type = tool.args_type()
            if args_type:
                return args_type.model_json_schema()
        except Exception as e:
            logger.warning(f"Failed to get schema for tool {tool.metadata.name}: {e}")

        return {
            "type": "object",
            "properties": {},
            "required": [],
        }


class ReActPattern(AgentPattern):
    """
    Action-based ReAct Pattern

    This pattern forces the LLM to output structured actions, eliminating
    parsing complexity and improving stability.

    Execution flow:
    1. LLM outputs structured Action (tool_call or final_answer)
    2. If tool_call: execute tool and return observation
    3. If final_answer: return the answer
    4. Repeat until final answer or max iterations
    """

    def __init__(
        self,
        llm: BaseLLM,
        max_iterations: int = 200,
        tracer: Optional[Tracer] = None,
        compact_threshold: Optional[int] = None,
        enable_auto_compact: bool = True,
        compact_llm: Optional[BaseLLM] = None,
        memory_store: Optional[MemoryStore] = None,
        is_sub_agent: bool = False,
    ):
        """
        Initialize ReAct pattern.

        Args:
            llm: Language model for reasoning
            max_iterations: Maximum number of action cycles
            tracer: Tracer instance for event tracking
            compact_threshold: Token threshold for triggering context compaction (default: 12000)
            enable_auto_compact: Whether to enable automatic context compaction (default: True)
            compact_llm: Optional LLM for context compaction, defaults to main LLM
            is_sub_agent: Whether this pattern is running as a sub-agent (e.g., in a DAG step)
        """
        self.llm = llm
        self.is_sub_agent = is_sub_agent
        self.compact_llm = (
            compact_llm or llm
        )  # Use main LLM if compact_llm not provided
        self.max_iterations = max_iterations
        self.tracer = tracer or Tracer()
        self.tool_registry = ToolRegistry()
        self.memory_store = memory_store
        self._last_response: Any = None
        self._last_messages: List[Dict[str, str]] = []
        self._current_step_id: Optional[str] = None
        self._current_step_name: Optional[str] = None
        self._current_action_id: Optional[str] = None
        self._pause_event = asyncio.Event()
        self._pause_condition = asyncio.Condition()
        self._interrupt_event = (
            asyncio.Event()
        )  # For immediate interruption (continuation)
        self._context: Optional[AgentContext] = None
        self._conversation_history: List[Dict[str, str]] = []
        self._execution_context_messages: List[Dict[str, str]] = []

        # Context compaction configuration
        self.compact_config = CompactConfig(
            enabled=enable_auto_compact,
            threshold=compact_threshold or CompactConfig().threshold,
        )
        self._compact_stats = {"total_compacts": 0, "tokens_saved": 0}

    def _estimate_message_tokens(self, messages: List[Dict[str, str]]) -> int:
        """
        Estimate token count for messages using simple character-based approximation.

        Args:
            messages: List of messages to estimate

        Returns:
            Estimated token count
        """
        return CompactUtils.estimate_tokens(messages)

    def set_conversation_history(self, messages: List[Dict[str, Any]]) -> None:
        """Replace the persisted top-level conversation transcript for a new run."""
        self._conversation_history = normalize_transcript_messages(messages)

    def set_execution_context_messages(self, messages: List[Dict[str, Any]]) -> None:
        """Load persisted execution-state context for a new run."""
        self._execution_context_messages = normalize_transcript_messages(messages)

    async def _compact_react_context(
        self, messages: List[Dict[str, str]], iteration: int
    ) -> List[Dict[str, str]]:
        """
        Compact ReAct conversation context when it exceeds token threshold.

        Args:
            messages: Current conversation messages
            iteration: Current iteration number for logging

        Returns:
            Compact messages list
        """
        if not self.compact_config.enabled:
            return messages

        original_tokens = CompactUtils.estimate_tokens(messages)
        if original_tokens <= self.compact_config.threshold:
            return messages

        logger.info(
            f"ReAct context ({original_tokens} tokens) exceeds threshold ({self.compact_config.threshold}) at iteration {iteration}, compacting with {self.compact_llm.model_name}..."
        )

        try:
            # Format messages for compaction
            conversation_text = CompactUtils.format_messages_for_compact(messages)

            # Build ReAct-specific compaction prompt
            compact_prompt = [
                {
                    "role": "system",
                    "content": "You are compacting a ReAct (Reasoning and Acting) conversation history. "
                    "Preserve: 1) Recent tool calls and their results, 2) Current task and goal, "
                    "3) Important reasoning and decisions from the last few iterations, "
                    "4) Key observations and findings. Remove: 1) Redundant system messages, "
                    "2) Old reasoning that's no longer relevant, 3) Failed attempts that aren't needed for context. "
                    "CRITICAL: You must return the response in the exact same format: \n"
                    "USER: message content\n"
                    "ASSISTANT: message content\n"
                    "SYSTEM: message content\n\n"
                    "Each message must start with the role followed by a colon and space.",
                },
                {
                    "role": "user",
                    "content": f"Current iteration: {iteration}\n"
                    f"Original tokens: {original_tokens}\n"
                    f"Target threshold: {self.compact_config.threshold}\n\n"
                    f"ReAct conversation to compact:\n{conversation_text}\n\n"
                    f"IMPORTANT: Return the compacted conversation in the exact same format shown above. "
                    f"Each line must start with USER:, ASSISTANT:, or SYSTEM: followed by the message content.",
                },
            ]

            # Clean messages before sending to LLM
            cleaned_compact_prompt = clean_messages(compact_prompt)

            # Get compacted response
            response = await self.compact_llm.chat(messages=cleaned_compact_prompt)
            content = (
                response
                if isinstance(response, str)
                else response.get("content", str(response))
            )

            # Parse back to messages format
            compacted_messages = CompactUtils.parse_compact_response(content)

            # Validate compact result
            if not compacted_messages:
                logger.warning(
                    "Compact resulted in empty messages, using smart truncation fallback"
                )
                # Smart truncation: preserve first system message and last few messages
                return self._smart_truncate_react_messages(
                    messages, target_tokens=self.compact_config.threshold
                )

            # Ensure system message is preserved
            if compacted_messages and compacted_messages[0].get("role") != "system":
                original_system = next(
                    (msg for msg in messages if msg.get("role") == "system"), None
                )
                if original_system:
                    compacted_messages.insert(0, original_system)

            final_tokens = CompactUtils.estimate_tokens(compacted_messages)
            tokens_saved = original_tokens - final_tokens

            logger.info(
                f"Successfully compacted ReAct context: {len(messages)} -> {len(compacted_messages)} messages, "
                f"{original_tokens} -> {final_tokens} tokens ({tokens_saved} saved)"
            )

            # Update compact stats
            self._compact_stats["total_compacts"] += 1
            self._compact_stats["tokens_saved"] += tokens_saved

            return compacted_messages

        except Exception as e:
            logger.error(f"ReAct compact failed: {e}, using fallback truncation")
            # Fallback: truncate to last N messages
            truncated_messages = CompactUtils.truncate_messages(
                messages, self.compact_config.fallback_truncate_count
            )
            final_tokens = CompactUtils.estimate_tokens(truncated_messages)
            tokens_saved = original_tokens - final_tokens

            # Still update stats for fallback compaction
            self._compact_stats["total_compacts"] += 1
            self._compact_stats["tokens_saved"] += tokens_saved

            return truncated_messages

    def _smart_truncate_react_messages(
        self, messages: List[Dict[str, str]], target_tokens: int
    ) -> List[Dict[str, str]]:
        """Smart truncate ReAct messages to preserve important content while reducing tokens."""
        if not messages:
            return messages

        # Always keep the first system message if it exists
        system_messages = [msg for msg in messages if msg.get("role") == "system"]
        other_messages = [msg for msg in messages if msg.get("role") != "system"]

        # Start with system messages
        result = system_messages.copy()

        # Add messages from the end (most recent) until we reach target token count
        current_tokens = CompactUtils.estimate_tokens(result)
        for msg in reversed(other_messages):
            msg_tokens = CompactUtils.estimate_tokens([msg])
            if current_tokens + msg_tokens > target_tokens:
                break
            result.insert(len(system_messages), msg)  # Insert after system messages
            current_tokens += msg_tokens

        logger.info(
            f"ReAct smart truncation: {len(messages)} -> {len(result)} messages, "
            f"{CompactUtils.estimate_tokens(messages)} -> {current_tokens} tokens"
        )

        return result

    def _format_messages_for_compact(self, messages: List[Dict[str, str]]) -> str:
        """Format messages as text for compaction."""
        formatted_lines = []
        for msg in messages:
            role = msg.get("role", "unknown").upper()
            content = msg.get("content", "")
            formatted_lines.append(f"{role}: {content}")
        return "\n".join(formatted_lines)

    def _parse_compact_response(self, response: str) -> List[Dict[str, str]]:
        """Parse compacted response back to messages format."""
        messages = []
        lines = response.strip().split("\n")

        current_role = None
        current_content = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check if this is a role line
            if line.startswith(("USER:", "ASSISTANT:", "SYSTEM:")):
                # Save previous message
                if current_role and current_content:
                    messages.append(
                        {
                            "role": current_role.lower(),
                            "content": "\n".join(current_content),
                        }
                    )

                # Start new message
                parts = line.split(":", 1)
                current_role = parts[0].lower()
                current_content = [parts[1].strip()] if len(parts) > 1 else []
            else:
                # Continue current message
                if current_role:
                    current_content.append(line)

        # Save final message
        if current_role and current_content:
            messages.append(
                {"role": current_role, "content": "\n".join(current_content)}
            )

        return messages

    def _fallback_truncate_messages(
        self, messages: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """Fallback truncation method when compaction fails."""
        if len(messages) <= 10:
            return messages

        # Always preserve system message and recent messages
        system_msg = next(
            (msg for msg in messages if msg.get("role") == "system"), None
        )
        recent_messages = messages[-8:]  # Keep last 8 messages

        result = []
        if system_msg:
            result.append(system_msg)
        result.extend(recent_messages)

        logger.info(
            f"Used fallback truncation: {len(messages)} -> {len(result)} messages"
        )
        return result

    async def _check_and_compact_context(
        self, messages: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """
        Check context length and compact if necessary before LLM call.

        Args:
            messages: Current conversation messages

        Returns:
            Messages list, possibly compacted
        """
        estimated_tokens = self._estimate_message_tokens(messages)
        if estimated_tokens > self.compact_config.threshold:
            # Get current iteration from messages (count assistant messages as iterations)
            iteration = len([msg for msg in messages if msg.get("role") == "assistant"])

            logger.debug(
                f"Context length check: {estimated_tokens} tokens > {self.compact_config.threshold} threshold, "
                f"triggering compaction at iteration {iteration}"
            )

            return await self._compact_react_context(messages, iteration)

        return messages

    def get_compact_stats(self) -> Dict[str, Any]:
        """
        Get context compaction statistics.

        Returns:
            Dictionary with compaction statistics
        """
        return {
            **self._compact_stats,
            "enabled": self.compact_config.enabled,
            "threshold": self.compact_config.threshold,
            "current_message_count": len(self._last_messages),
            "current_estimated_tokens": self._estimate_message_tokens(
                self._last_messages
            ),
        }

    def set_step_context(
        self, step_id: Optional[str] = None, step_name: Optional[str] = None
    ) -> None:
        """
        Set the current step context for tracing purposes.

        Args:
            step_id: Current step ID
            step_name: Current step name
        """
        self._current_step_id = step_id
        self._current_step_name = step_name

    def pause_execution(self) -> None:
        """Pause the current execution"""
        self._pause_event.set()
        logger.info("ReAct execution paused")

    def resume_execution(self) -> None:
        """Resume paused execution"""
        self._pause_event.clear()
        notify_condition(self._pause_condition)
        logger.info("ReAct execution resumed")

    def interrupt_execution(self) -> None:
        """Interrupt execution immediately for continuation/plan modification"""
        self._interrupt_event.set()
        logger.info("ReAct execution interrupted")

    def _populate_trace_data(
        self,
        trace_data: Dict[str, Any],
        target_key: str,
        context: Any,
        context_dict: Dict[str, Any],
        source_key: str,
    ) -> None:
        """
        Helper to populate trace data from context or context state.

        Args:
            trace_data: The trace data dictionary to populate
            target_key: The key to set in trace_data
            context: The context object
            context_dict: Dictionary representation of context
            source_key: The key to look for in context
        """
        if source_key in context_dict:
            trace_data[target_key] = context_dict[source_key]
        elif (
            hasattr(context, "state")
            and isinstance(context.state, dict)
            and source_key in context.state
        ):
            trace_data[target_key] = context.state[source_key]

    async def run(
        self,
        task: str,
        memory: MemoryStore,
        tools: List[Tool],
        context: Optional[AgentContext] = None,
    ) -> Dict[str, Any]:
        """
        Execute the action-based ReAct pattern.

        Args:
            task: The task to accomplish
            memory: Memory store for persistence
            tools: Available tools
            context: Execution context

        Returns:
            Execution result with success status and output

        Raises:
            LLMNotAvailableError: When no LLM is configured
            MaxIterationsError: When max iterations reached
            PatternExecutionError: When execution fails
        """
        logger.info(f"Starting Action-based ReAct execution for task: {task[:100]}...")

        # Store context for use in _build_system_prompt
        self._context = context

        # Trace task start
        task_id = f"react_{context.task_id if context else uuid4()}"

        # Emit user message trace if this is the main agent
        if not self.is_sub_agent:
            # Use original task_id for user message to ensure it matches the task
            user_msg_task_id = (
                str(context.task_id)
                if context and hasattr(context, "task_id") and context.task_id
                else task_id
            )

            trace_data: Dict[str, Any] = {}
            if context:
                context_dict = {}
                if hasattr(context, "to_dict"):
                    context_dict = context.to_dict()
                elif hasattr(context, "dict"):
                    context_dict = context.dict()
                elif isinstance(context, dict):
                    context_dict = context

                # Check for file info in context
                self._populate_trace_data(
                    trace_data,
                    "files",
                    context,
                    context_dict,
                    CONTEXT_KEY_FILE_INFO,
                )
                self._populate_trace_data(
                    trace_data,
                    CONTEXT_KEY_UPLOADED_FILES,
                    context,
                    context_dict,
                    CONTEXT_KEY_UPLOADED_FILES,
                )

            await trace_user_message(
                self.tracer,
                user_msg_task_id,
                task,
                trace_data,
            )

        # For standalone React execution, create a virtual step context
        if not hasattr(self, "_current_step_id") or not self._current_step_id:
            step_id = f"{task_id}_main"
            self.set_step_context(step_id=step_id, step_name="main")
        else:
            step_id = self._current_step_id

        await trace_task_start(
            self.tracer,
            task_id,
            TraceCategory.REACT,
            data={
                "pattern": "ReAct",
                "task": task[:100],
                "max_iterations": self.max_iterations,
                "tools": [tool.metadata.name for tool in tools],
                "step_id": step_id,
                "step_name": getattr(self, "_current_step_name", "main"),
            },
        )

        if not self.llm:
            await trace_error(
                self.tracer,
                task_id,
                step_id,
                error_type="ConfigurationError",
                error_message="No LLM configured for Action-based ReAct pattern",
                data={"pattern": "ReAct", "task": task[:100]},
            )
            raise LLMNotAvailableError(
                "No LLM configured for Action-based ReAct pattern",
                context={"pattern": "ReAct", "task": task[:100]},
            )

        # Register tools
        self.tool_registry.register_all(tools)

        # Enhance task with memory if available
        enhanced_task = task
        if self.memory_store:
            # Trace memory retrieval start
            task_id = f"react_memory_{int(time.time())}"
            await trace_memory_retrieve_start(
                self.tracer,
                task_id,
                data={
                    "task": task,
                    "memory_category": "react_memory",
                },
            )

            # Get current user context to pass to the thread
            try:
                from ....web.user_isolated_memory import current_user_id

                user_id = current_user_id.get()
            except ImportError:
                # Fallback for non-web environment
                user_id = None

            memories = await asyncio.to_thread(
                self._lookup_relevant_memories_with_context,
                self.memory_store,
                task,
                "react_memory",
                include_general=True,
                user_id=user_id,
            )
            enhanced_task = enhance_goal_with_memory(task, memories)

            # Trace memory retrieval end
            await trace_memory_retrieve_end(
                self.tracer,
                task_id,
                data={
                    "task": task,
                    "memories_found": len(memories),
                    "memories_used": len(
                        [m for m in memories if m.get("content", "").strip()]
                    ),
                },
            )

        # Build initial messages
        messages = [
            {"role": "system", "content": self._build_system_prompt()},
        ]
        if self._execution_context_messages:
            messages.extend(self._execution_context_messages)
        if self._conversation_history:
            messages.extend(self._conversation_history)
        messages.append({"role": "user", "content": enhanced_task})

        # Use the shared execution loop
        return await self._execute_react_loop(
            messages=messages,
            task_id=task_id,
            step_id=step_id,
            max_iterations=self.max_iterations,
            task_description=task,
        )

    async def _execute_react_loop(
        self,
        messages: List[Dict[str, str]],
        task_id: str,
        step_id: str,
        max_iterations: int,
        task_description: str = "task",
    ) -> Dict[str, Any]:
        """
        Core ReAct execution loop with proper error handling and retry logic.

        This method contains the shared execution logic used by both run() and run_with_context().
        """
        # Store initial messages
        self._last_messages = messages.copy()

        # Execute action loop
        for iteration in range(max_iterations):
            logger.info(f"Action ReAct iteration {iteration + 1}/{max_iterations}")

            # Check for pause state before each iteration
            if self._pause_event.is_set():
                logger.info(
                    f"ReAct execution paused at iteration {iteration + 1}, waiting for resume..."
                )
                async with self._pause_condition:
                    await self._pause_condition.wait_for(
                        lambda: not self._pause_event.is_set()
                    )
                logger.info(f"ReAct execution resumed at iteration {iteration + 1}")

            # Check for interrupt state (for continuation/plan modification)
            if self._interrupt_event.is_set():
                logger.info(
                    f"ReAct execution interrupted at iteration {iteration + 1}, stopping..."
                )
                self._interrupt_event.clear()  # Clear for next use
                raise InterruptedError("Execution interrupted for plan modification")

            try:
                # Set action ID for tracing (needed for tool execution correlation)
                action_id = f"{task_id}_action_{iteration + 1}"
                self._current_action_id = action_id

                # Trace action start
                await trace_action_start(
                    self.tracer,
                    task_id,
                    step_id,
                    TraceCategory.REACT,
                    data={
                        "iteration": iteration + 1,
                        "task_id": task_id,
                        "step_id": step_id,
                        "step_name": getattr(self, "_current_step_name", "main"),
                    },
                )

                # Get structured action from LLM (first call: determine action type)
                action = await self._get_action_from_llm(messages)

                # If action is tool_call, make a second LLM call to get actual tool invocation
                # Only do this for main ReAct mode (not DAG mode which uses native calling from start)
                if action.type == "tool_call" and step_id == "main":
                    # Emit reasoning trace before second call
                    if action.reasoning:
                        await trace_ai_message(
                            self.tracer,
                            task_id,
                            message=action.reasoning,
                            data={
                                "content": action.reasoning,
                                "message_type": "reasoning",
                                "action_type": action.type,
                                "step_id": step_id,
                                "step_name": getattr(
                                    self, "_current_step_name", "main"
                                ),
                            },
                        )

                    # Second call: Invoke tool using native tool calling
                    action = await self._invoke_tool_via_native_call(messages)

                # Execute the action
                result = await self._execute_action(action, messages, task_id, step_id)

                # Trace action end
                await trace_action_end(
                    self.tracer,
                    task_id,
                    step_id,
                    TraceCategory.REACT,
                    data={
                        "action_type": action.type,
                        "tool_name": action.tool_name
                        if action.type == "tool_call"
                        else None,
                        "result_type": result["type"],
                        "step_id": step_id,
                        "step_name": getattr(self, "_current_step_name", "main"),
                        "reasoning": action.reasoning,
                    },
                )

                # Check if this is the final answer
                if result["type"] == "final_answer":
                    logger.info(f"Action ReAct completed in {iteration + 1} iterations")
                    logger.debug(
                        f"Final answer content: {result.get('content', 'NO_CONTENT')[:200]}"
                    )
                    logger.debug(f"Is sub-agent: {self.is_sub_agent}")

                    # Get the success status from the result
                    # This may be False if LLM indicated task failure
                    success_status = result.get("success", True)

                    # Generate comprehensive insights and store memories
                    await self._generate_and_store_react_memories(
                        f"{task_description} (iteration {iteration + 1})",
                        result["content"],
                        iteration + 1,
                        messages,
                    )

                    # Only send task completion events if NOT a sub-agent
                    # Sub-agents (DAG steps) should not trigger task-level completion
                    if not self.is_sub_agent:
                        logger.debug(
                            f"Tracing AI message with content length: {len(result.get('content', ''))}"
                        )
                        # Trace AI message with the final result
                        await trace_ai_message(
                            self.tracer,
                            task_id,
                            message=result["content"],
                            data={"content": result["content"]},
                        )

                        logger.debug("Tracing task completion")
                        # Trace task completion
                        await trace_task_completion(
                            self.tracer,
                            task_id,
                            result=result["content"],
                            success=success_status,
                        )

                        logger.debug("Tracing task end")
                        # Trace task end (REACT specific)
                        await trace_task_end(
                            self.tracer,
                            task_id,
                            TraceCategory.REACT,
                            data={
                                "result": result["content"],
                                "success": success_status,
                            },
                        )

                    return {
                        "success": success_status,
                        "output": result["content"],
                        "iterations": iteration + 1,
                        "execution_history": messages,
                        "pattern": "react",
                    }

                # Add observation to conversation for tool results
                observation_content = f"Tool result from {result.get('tool_name', 'unknown')}:\n{result['content']}\n\nBased on this result, if you have enough information to answer the user's question, provide your final answer. Otherwise, call another tool."
                messages.append({"role": "user", "content": observation_content})

                # Update stored messages
                self._last_messages = messages.copy()

            except Exception as e:
                # Generate insights and store memories even for failures
                try:
                    await self._generate_and_store_react_memories(
                        task_description,
                        f"Failed after {iteration + 1} iterations: {str(e)}",
                        iteration + 1,
                        messages,
                    )
                except Exception as mem_error:
                    logger.error(f"Failed to store failure memories: {mem_error}")

                # Trace error
                await trace_error(
                    self.tracer,
                    task_id,
                    step_id,
                    error_type="PatternExecutionError",
                    error_message=f"Iteration {iteration + 1} failed: {str(e)}",
                    data={
                        "task": task_description[:100],
                        "messages_count": len(messages),
                        "iteration": iteration + 1,
                        "step_id": step_id,
                        "step_name": getattr(self, "_current_step_name", "main"),
                        "action_id": action_id,
                    },
                )

                # Retryable errors - continue to next iteration
                logger.warning(
                    f"Iteration {iteration + 1} failed with retryable error: {str(e)}"
                )
                continue  # Continue to next iteration instead of raising

        # Max iterations reached - generate insights and store memories before raising
        try:
            await self._generate_and_store_react_memories(
                task_description,
                f"Max iterations ({max_iterations}) reached. Last message: {messages[-1]['content'][:200] if messages else 'No messages'}",
                max_iterations,
                messages,
            )
        except Exception as mem_error:
            logger.error(f"Failed to store max iterations memories: {mem_error}")

        # Max iterations reached - trace before raising
        await trace_error(
            self.tracer,
            task_id,
            step_id,
            error_type="MaxIterationsError",
            error_message=f"Max iterations ({max_iterations}) reached",
            data={
                "task": task_description[:100],
                "final_messages_count": len(messages),
                "total_iterations": len(
                    [m for m in messages if m.get("role") == "assistant"]
                ),
                "step_id": step_id,
                "step_name": getattr(self, "_current_step_name", "main"),
            },
        )

        raise MaxIterationsError(
            pattern_name="ReAct",
            max_iterations=max_iterations,
            final_state=f"Last message: {messages[-1]['content'][:100] if messages else 'No messages'}",
            context={
                "task": task_description[:100],
                "final_messages_count": len(messages),
            },
        )

    async def run_with_context(
        self,
        messages: List[Dict[str, str]],
        tools: List[Tool],
        max_iterations: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Run ReAct with pre-built message context.

        This is used when the agent is called as a sub-agent with existing context.
        """
        max_iter = max_iterations or self.max_iterations

        # Use existing step context - must be set externally
        if not hasattr(self, "_current_step_id") or not self._current_step_id:
            raise PatternExecutionError(
                pattern_name="ReAct",
                message="run_with_context called without step context. Call set_step_context() first.",
                context={"method": "run_with_context"},
            )

        step_id = self._current_step_id
        task_id = getattr(self, "_current_task_id", f"react_context_{uuid4()}")
        self._current_task_id = task_id

        # Register tools
        self.tool_registry.register_all(tools)

        # Enhance task with memory if available (extract from user message)
        enhanced_messages = messages
        if self.memory_store and messages:
            # Find the user message with the task
            user_message = None
            for msg in messages:
                if msg.get("role") == "user":
                    user_message = msg["content"]
                    break

            if user_message:
                # Trace memory retrieval start
                memory_task_id = f"{task_id}_memory_retrieval"
                await trace_memory_retrieve_start(
                    self.tracer,
                    memory_task_id,
                    data={
                        "task": user_message,
                        "memory_category": "react_memory",
                    },
                    step_id=step_id,
                )

                # Get current user context to pass to the thread
                try:
                    from ....web.user_isolated_memory import current_user_id

                    user_id = current_user_id.get()
                except ImportError:
                    # Fallback for non-web environment
                    user_id = None

                memories = await asyncio.to_thread(
                    self._lookup_relevant_memories_with_context,
                    self.memory_store,
                    user_message,
                    "react_memory",
                    include_general=True,
                    user_id=user_id,
                )

                # Trace memory retrieval end
                await trace_memory_retrieve_end(
                    self.tracer,
                    memory_task_id,
                    data={
                        "task": user_message,
                        "memories_found": len(memories),
                        "memories_used": len(
                            [m for m in memories if m.get("content", "").strip()]
                        ),
                    },
                    step_id=step_id,
                )

                # Add memory context to the user message
                if memories:
                    memory_context = "\n\nRelevant Memories:\n" + "\n".join(
                        [
                            f"- {m.get('content', '')}"
                            for m in memories
                            if m.get("content", "").strip()
                        ]
                    )
                    enhanced_messages = []
                    for msg in messages:
                        if msg.get("role") == "user":
                            enhanced_messages.append(
                                {
                                    "role": "user",
                                    "content": msg["content"] + memory_context,
                                }
                            )
                        else:
                            enhanced_messages.append(msg)

        # Make a copy to avoid modifying the original
        working_messages = enhanced_messages.copy()

        # Ensure the first message is our system prompt with Action format requirements
        if working_messages and working_messages[0].get("role") == "system":
            # Merge existing system prompt with our Action requirements
            existing_system_prompt = working_messages[0]["content"]
            enhanced_system_prompt = self._build_enhanced_system_prompt(
                existing_system_prompt
            )
            working_messages[0]["content"] = enhanced_system_prompt
        else:
            # If no system prompt, add ours
            working_messages.insert(
                0, {"role": "system", "content": self._build_system_prompt()}
            )

        # Store initial messages
        self._last_messages = working_messages.copy()

        try:
            # Use the shared execution loop
            return await self._execute_react_loop(
                messages=working_messages,
                task_id=task_id,
                step_id=step_id,
                max_iterations=max_iter,
                task_description="Context-based task",
            )

        except Exception as e:
            # Generate insights and store memories even for failures
            try:
                await self._generate_and_store_react_memories(
                    "Context-based task (failed)",
                    f"Task failed: {str(e)}",
                    max_iter,
                    working_messages,
                )
            except Exception as mem_error:
                logger.error(
                    f"Failed to store failure memories in run_with_context: {mem_error}"
                )

            # For critical failures, log full context before re-raising
            if isinstance(e, (LLMNotAvailableError, ToolNotFoundError)):
                self._log_full_context(
                    working_messages,
                    getattr(self, "_last_response", None),
                    f"Critical error in run_with_context: {type(e).__name__}",
                )
                raise

            # For other exceptions, log context and return error result
            self._log_full_context(
                working_messages,
                getattr(self, "_last_response", None),
                f"Unexpected error in run_with_context: {type(e).__name__}",
            )
            return {
                "success": False,
                "error": f"ReAct execution failed: {str(e)}",
                "output": f"Execution error: {str(e)}",
                "execution_history": working_messages,
                "pattern": "react",
                "type": "error",
            }

    def _build_system_prompt(self) -> str:
        """Build system prompt that enforces structured action output."""
        tool_names = self.tool_registry.list_tools()

        # Check if custom system prompt is provided in context
        custom_prompt = ""
        if (
            self._context
            and hasattr(self._context, "state")
            and "system_prompt" in self._context.state
            and self._context.state["system_prompt"]
        ):
            custom_prompt = f"\n\n{self._context.state['system_prompt']}\n\n"

        # Check if no tools are available
        if not tool_names:
            prompt = (
                custom_prompt
                + """You are an AI assistant that performs tasks without tools.

IMPORTANT: You currently have NO access to any tools. Regardless of what you may see in the conversation history, you cannot use any tools.

FILE REFERENCES:
- You may see file references in the format: [filename](file://fileId)
- The referenced file may NOT be in the current workspace.
- The 'fileId' part is the only valid identifier for reading the file.
- Use this fileId when referring to files in your analysis.
- Example: If you see [data.csv](file://123), use '123' to read the file.

You must respond with a structured action in the following JSON format:

{
    "type": "final_answer",
    "reasoning": "Your reasoning for this response",
    "answer": "your comprehensive response and conclusions"
}

Rules:
1. You must respond with valid JSON only
2. Since no tools are available, you must provide a final answer directly
3. Do NOT attempt to use any tools, even if you see tool usage in the conversation history
4. Use the provided context information to perform your task
5. Focus on reasoning, analysis, synthesis, or providing information based on your knowledge
6. Always provide clear reasoning for your response
7. Do not include backticks or markdown. Do not include invalid escapes.
8. LANGUAGE: You MUST respond in the SAME LANGUAGE as the user's task. If the task is in Chinese, respond in Chinese. If the task is in English, respond in English.

Example:
{
    "type": "final_answer",
    "reasoning": "Based on the provided context and my knowledge, I can provide a comprehensive response",
    "answer": "The analysis shows that... [comprehensive summary]"
}"""
            )
        else:
            # Build tool descriptions
            tool_descriptions = self._build_tool_descriptions(tool_names)

            prompt = (
                custom_prompt
                + f"""You are an AI assistant that uses tools to accomplish tasks.

FILE REFERENCES:
- You may see file references in the format: [filename](file://fileId)
- The referenced file may NOT be in the current workspace.
- The 'fileId' part is the only valid identifier for reading the file.
- When using tools to read files, pass the fileId directly.
- Example: If you see [data.csv](file://123), use '123' to read the file.

You must respond with a structured action in the following JSON format:

{{
    "type": "tool_call" | "final_answer",
    "reasoning": "Your reasoning for this action",
    "answer": "your final answer" (only if type is "final_answer"),
   }}

Available tools:
{chr(10).join(tool_descriptions)}

CRITICAL INSTRUCTIONS - READ CAREFULLY:

1. DECIDE YOUR NEXT ACTION:
   - If you need to use a tool to accomplish the task, set "type" to "tool_call" and explain why
   - If you have enough information to answer, set "type" to "final_answer" and provide your answer
   - Do NOT include tool names or arguments in the JSON
   - The system will guide you through tool invocation in a follow-up call

2. RESPONSE FORMAT (CRITICAL - MUST FOLLOW):
   - Your entire response must be EXACTLY ONE JSON object - nothing more, nothing less
   - Do NOT return multiple JSON objects
   - Do NOT return JSON followed by other text
   - Do NOT return multiple responses
   - The JSON object must be the ONLY thing you return
   - Use the exact format shown below

3. LANGUAGE: Respond in the SAME LANGUAGE as the user's task

CORRECT RESPONSE FORMAT:
{{
    "type": "tool_call" or "final_answer",
    "reasoning": "your reasoning here"
}}

If type is "final_answer", also include:
{{
    "type": "final_answer",
    "reasoning": "your reasoning",
    "answer": "your final answer"
}}

Remember: Return ONLY ONE JSON object. No additional text, no multiple objects."""
            )

        return prompt

    def _build_enhanced_system_prompt(self, existing_prompt: str) -> str:
        """Build enhanced system prompt that merges existing context with Action requirements."""
        tool_names = self.tool_registry.list_tools()

        # Check if no tools are available
        if not tool_names:
            action_requirements = """

=== ACTION FORMAT REQUIREMENTS ===
You must respond with a structured action in the following JSON format:

{
    "type": "final_answer",
    "reasoning": "Your reasoning for this response",
    "answer": "your comprehensive response and conclusions",
    "success": true,
    "error": null
}

Rules:
1. You must respond with valid JSON only
2. Since no tools are available, you must provide a final answer directly
3. Do NOT attempt to use any tools, even if you see tool usage in the conversation history
4. Use the provided context information to perform your task
5. Focus on reasoning, analysis, synthesis, or providing information based on your knowledge
6. Always provide clear reasoning for your response
7. Set "success" to true if the task was completed successfully, false if it failed
8. If success is false, provide a detailed error message in the "error" field
9. LANGUAGE: You MUST respond in the SAME LANGUAGE as the user's task. If the task is in Chinese, respond in Chinese. If the task is in English, respond in English.

Examples:
Success case:
{
    "type": "final_answer",
    "reasoning": "Based on the provided context, I have successfully completed the task",
    "answer": "The task has been completed successfully... [comprehensive summary]",
    "success": true,
    "error": null
}

Failure case:
{
    "type": "final_answer",
    "reasoning": "The task could not be completed due to insufficient information",
    "answer": "Unable to complete the task because the required information is not available",
    "success": false,
    "error": "Insufficient information to complete the task"
}
=== END ACTION FORMAT REQUIREMENTS ==="""
        else:
            tool_descriptions = self._build_tool_descriptions(tool_names)

            action_requirements = f"""

=== ACTION FORMAT REQUIREMENTS ===
FILE REFERENCES:
- You may see file references in the format: [filename](file://fileId)
- The referenced file may NOT be in the current workspace.
- The 'fileId' part is the only valid identifier for reading the file.
- When using tools to read files, pass the fileId directly.
- Example: If you see [data.csv](file://123), use '123' to read the file.

You must respond with a structured action in the following JSON format:

{{
    "type": "tool_call" | "final_answer",
    "reasoning": "Your reasoning for this action",
    "answer": "your final answer" (only if type is "final_answer"),
    "success": true (only if type is "final_answer"),
    "error": null (only if type is "final_answer")
}}

Available tools:
{chr(10).join(tool_descriptions)}

You have access to the above tools. Use them when needed to complete the task.
When you decide to call a tool, ONLY set the action type to "tool_call" and explain why.
Do NOT include tool names or arguments in the JSON; the system will automatically invoke
the appropriate tool through the native function calling API based on your decision.
After using tools, provide a clear summary of the results in the SAME LANGUAGE as the user's task.
"""

        return existing_prompt + action_requirements

    def _build_tool_descriptions(self, tool_names: List[str]) -> List[str]:
        """
        Build formatted tool descriptions with name and description.

        Args:
            tool_names: List of tool names to describe

        Returns:
            List of formatted tool description strings
        """
        tool_descriptions = []
        for tool_name in tool_names:
            try:
                tool = self.tool_registry.get(tool_name)
                desc = tool.metadata.description or ""
                tool_descriptions.append(f"- {tool_name}: {desc}")
            except Exception:
                tool_descriptions.append(f"- {tool_name}")
        return tool_descriptions

    def _build_tool_schema(self, tool: Tool) -> Dict[str, Any]:
        """Build OpenAI-style function schema for a tool."""
        # Get the tool's argument schema
        args_type = tool.args_type()

        # Build properties from the Pydantic model
        properties = {}
        required = []

        if hasattr(args_type, "model_fields"):
            for field_name, field_info in args_type.model_fields.items():
                # Get field type
                field_type = field_info.annotation
                if field_type is ... or field_type is None:
                    # Skip field if type is undefined
                    continue

                # Convert Python type to JSON schema type
                if field_type is str:
                    json_type = "string"
                elif field_type is int:
                    json_type = "integer"
                elif field_type is float:
                    json_type = "number"
                elif field_type is bool:
                    json_type = "boolean"
                elif hasattr(field_type, "__origin__"):  # Handle List, Dict, etc.
                    origin = getattr(field_type, "__origin__", None)
                    if origin is list:
                        json_type = "array"
                    elif origin is dict:
                        json_type = "object"
                    else:
                        json_type = "string"
                else:
                    json_type = "string"

                # Build property
                property_schema = {
                    "type": json_type,
                    "description": field_info.description or f"Parameter: {field_name}",
                }

                # Add default value if present and not undefined
                if (
                    field_info.default is not ...
                    and field_info.default is not None
                    and field_info.default != ""
                ):
                    property_schema["default"] = field_info.default

                properties[field_name] = property_schema

                # Check if required
                if field_info.is_required():
                    required.append(field_name)

        return {
            "type": "function",
            "function": {
                "name": tool.metadata.name,
                "description": tool.metadata.description
                or f"Tool: {tool.metadata.name}",
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

    async def _get_action_from_llm(self, messages: List[Dict[str, str]]) -> Action:
        """Get structured action from LLM."""
        # Check context length and compact if necessary
        final_messages = await self._check_and_compact_context(messages)

        # Prepare chat parameters
        chat_kwargs: Dict[str, Any] = {
            "messages": final_messages,
        }

        # Get tool schemas
        tool_schemas = self.tool_registry.get_tool_schemas()

        # Determine if we're in DAG mode (step_id != "main")
        # In DAG mode, use traditional single-phase tool calling for backward compatibility
        # In main ReAct mode, use two-phase tool calling
        is_dag_mode = (
            hasattr(self, "_current_step_id")
            and self._current_step_id
            and self._current_step_id != "main"
        )

        if not is_dag_mode:
            # First call: Request JSON output format for action type decision
            chat_kwargs["response_format"] = {"type": "json_object"}
        else:
            # DAG mode: Use native tool calling from the start
            if tool_schemas:
                chat_kwargs["tools"] = tool_schemas
                chat_kwargs["tool_choice"] = "auto"

        # Disable thinking mode if supported
        if (
            hasattr(self.llm, "supports_thinking_mode")
            and self.llm.supports_thinking_mode
        ):
            chat_kwargs["thinking"] = {"type": "disabled"}

        # Trace LLM call start (use current step context if available)
        if hasattr(self, "_current_step_id") and self._current_step_id:
            step_id = self._current_step_id
            task_id = getattr(self, "_current_task_id", f"react_{uuid4()}")
        else:
            # For standalone execution, create virtual context
            task_id = f"react_llm_{uuid4()}"
            step_id = f"{task_id}_step"

        await trace_llm_call_start(
            self.tracer,
            task_id,
            step_id,
            data={
                "action": "LLM call started",
                "model_name": getattr(self.llm, "model_name", type(self.llm).__name__),
                "task_type": "LLM call",
                "attempt": 1,
                "messages_count": len(messages),
                "messages": messages,
                "has_tools": bool(tool_schemas),
                "tools_count": len(tool_schemas),
                "tools": tool_schemas if tool_schemas else [],
                "tool_choice": chat_kwargs.get("tool_choice", "auto"),
                "thinking_mode": chat_kwargs.get("thinking", "not set"),
                "step_name": getattr(self, "_current_step_name", "main"),
                "step_id": step_id,
            },
        )

        response: Any = None
        llm_trace_sent = False

        async def log_llm_completion(
            response_payload: Any,
            is_tool_call_flag: bool,
            reasoning_value: Optional[str],
        ) -> None:
            nonlocal llm_trace_sent
            if llm_trace_sent:
                return
            await trace_action_end(
                self.tracer,
                task_id,
                step_id,
                TraceCategory.LLM,
                data={
                    "action": "LLM call completed",
                    "model_name": getattr(
                        self.llm, "model_name", type(self.llm).__name__
                    ),
                    "task_type": "LLM call",
                    "attempt": 1,
                    "response_type": type(response_payload).__name__,
                    "is_tool_call": is_tool_call_flag,
                    "response": response_payload,
                    "chat_kwargs": chat_kwargs,
                    "usage": usage,
                    "step_id": step_id,
                    "step_name": getattr(self, "_current_step_name", "main"),
                    "reasoning": reasoning_value,
                },
            )
            llm_trace_sent = True

        try:
            # Clean messages before sending to LLM
            cleaned_messages = clean_messages(messages)
            chat_kwargs["messages"] = cleaned_messages

            # Get LLM response using streaming API
            full_content = ""
            usage = {}
            tool_calls_from_stream = []

            async for chunk in self.llm.stream_chat(**chat_kwargs):
                if chunk.is_token():
                    full_content += chunk.delta
                elif chunk.is_tool_call():
                    tool_calls_from_stream = chunk.tool_calls
                elif chunk.is_usage():
                    usage = chunk.usage
                elif chunk.is_error():
                    raise RuntimeError(f"LLM stream error: {chunk.content}")

            # Record token usage
            if usage:
                logger.info(
                    f"LLM call usage - prompt_tokens: {usage.get('prompt_tokens', 0)}, "
                    f"completion_tokens: {usage.get('completion_tokens', 0)}, "
                    f"total_tokens: {usage.get('total_tokens', 0)}"
                )

            # Construct response object (maintaining compatibility with original chat() format)
            reasoning_text = full_content.strip()
            if reasoning_text:
                extracted_reasoning: Optional[str] = None
                try:
                    parsed_reasoning = repair_loads(reasoning_text, logging=False)
                    if isinstance(parsed_reasoning, dict):
                        extracted_reasoning = parsed_reasoning.get("reasoning")
                        if not extracted_reasoning:
                            # Some models put the explanation under "content"
                            content_value = parsed_reasoning.get("content")
                            if isinstance(content_value, str):
                                extracted_reasoning = content_value
                    elif isinstance(parsed_reasoning, list):
                        # Look for first dict item with reasoning/content fields
                        for item in parsed_reasoning:
                            if isinstance(item, dict):
                                extracted_reasoning = item.get("reasoning") or item.get(
                                    "content"
                                )
                                if extracted_reasoning:
                                    break
                except Exception:
                    extracted_reasoning = None

                if extracted_reasoning:
                    reasoning_text = extracted_reasoning.strip()

            if tool_calls_from_stream:
                response = {
                    "type": "tool_call",
                    "tool_calls": tool_calls_from_stream,
                    "raw": {"usage": usage} if usage else {},
                }
                if reasoning_text:
                    # Preserve assistant text so reasoning is not lost in traces
                    response["reasoning"] = reasoning_text
                    response["content"] = reasoning_text
            else:
                response = full_content

        except Exception as e:
            # Trace LLM call error
            try:
                await log_llm_completion(
                    response,
                    isinstance(response, dict) and response.get("type") == "tool_call",
                    None,
                )
            except Exception:
                pass
            await trace_error(
                self.tracer,
                task_id,
                step_id,
                error_type="LLMCallError",
                error_message=f"LLM call failed: {str(e)}",
                data={
                    "llm_type": type(self.llm).__name__,
                    "chat_kwargs": chat_kwargs,
                    "step_name": getattr(self, "_current_step_name", "main"),
                },
            )
            raise

        # Handle None response
        if response is None:
            await log_llm_completion(response, False, None)
            raise PatternExecutionError(
                pattern_name="ReAct",
                message="LLM returned None response",
                context={"chat_kwargs": chat_kwargs},
            )

        # Handle empty response - should trigger retry
        if isinstance(response, str) and not response.strip():
            await log_llm_completion(response, False, None)
            raise PatternExecutionError(
                pattern_name="ReAct",
                message="LLM returned empty response",
                context={"chat_kwargs": chat_kwargs},
            )

        # Handle native tool calls
        action: Optional[Action] = None
        if isinstance(response, dict) and response.get("type") == "tool_call":
            action = self._convert_native_tool_call_to_action(response)
            await log_llm_completion(response, True, action.reasoning)
            return action

        # Handle dict response with type="final_answer" (from native tool calling)
        if isinstance(response, dict) and response.get("type") == "final_answer":
            action = self._try_parse_action_from_dict(response, str(response))
            if action:
                await log_llm_completion(response, False, action.reasoning)
                return action

        # Handle string response - try to parse as JSON first
        if isinstance(response, str):
            # Try to parse as JSON first
            try:
                parsed = json.loads(response.strip())
                if isinstance(parsed, dict):
                    action = self._try_parse_action_from_dict(parsed, response.strip())
                    if action:
                        await log_llm_completion(
                            parsed, action.type == "tool_call", action.reasoning
                        )
                        return action
                    else:
                        logger.warning(
                            f"First call: Parsed JSON but unknown type: {parsed.get('type')}"
                        )
            except json.JSONDecodeError:
                # JSON parsing failed - might be multiple JSON objects
                # Try to use json_repair to handle multiple JSON objects
                try:
                    repaired = repair_loads(response.strip(), logging=False)

                    if isinstance(repaired, tuple):
                        action_data, repair_log = repaired
                    else:
                        action_data = repaired

                    # Handle when json_repair returns a list (multiple JSON objects)
                    # gpt-5.4 in streaming mode often returns multiple JSONs even with response_format='json_object'
                    # We take the first one as the intended action
                    if isinstance(action_data, list):
                        # Log all items for debugging
                        for i, item in enumerate(action_data):
                            if isinstance(item, dict):
                                logger.warning(
                                    f"First call: JSON object {i}: type={item.get('type', 'UNKNOWN')}, keys={list(item.keys())}"
                                )
                            else:
                                logger.warning(
                                    f"First call: JSON object {i}: {type(item).__name__}"
                                )

                        # Take the first JSON object
                        if action_data and isinstance(action_data[0], dict):
                            action_data = action_data[0]
                            logger.info(
                                f"First call: Selected first JSON object from multiple (type: {action_data.get('type', 'UNKNOWN')})"
                            )
                        else:
                            # First item is not a dict, raise error
                            raise PatternExecutionError(
                                pattern_name="ReAct",
                                message=f"LLM returned multiple JSON objects but the first one is not a valid dict (count: {len(action_data)})",
                                context={
                                    "json_object_count": len(action_data),
                                    "first_object_type": type(action_data[0]).__name__
                                    if action_data
                                    else "none",
                                    "response_preview": response[:500]
                                    if response
                                    else None,
                                },
                            )

                    if isinstance(action_data, dict):
                        action = self._try_parse_action_from_dict(
                            action_data, response.strip()
                        )
                        if action:
                            await log_llm_completion(
                                action_data,
                                action.type == "tool_call",
                                action.reasoning,
                            )
                            return action
                except Exception as repair_error:
                    # Re-raise PatternExecutionError as it's not a repair failure
                    if isinstance(repair_error, PatternExecutionError):
                        raise
                    # Not valid JSON, treat as direct text response
                    pass
            except AttributeError:
                pass

            # Fallback: treat unparsable string as direct text response
            action = Action(
                type="final_answer",
                reasoning="LLM provided direct response",
                answer=response.strip(),
                success=True,
                error=None,
            )
            await log_llm_completion(response, False, action.reasoning)

            return action

        # Parse JSON response (for when response_format="json_object" is enforced)
        try:
            content = self._extract_content(response)
            repaired = repair_loads(content, logging=True)

            if isinstance(repaired, tuple):
                action_data, repair_log = repaired
            else:
                action_data = repaired

            normalized_action_data = self._normalize_action_data(action_data)
            action = Action.model_validate(normalized_action_data)
            await log_llm_completion(
                normalized_action_data, action.type == "tool_call", action.reasoning
            )

            if action.type == "tool_call":
                if tool_schemas:
                    # JSON responses must not attempt to specify tool details.
                    # Require the model to trigger native function calling.
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "SYSTEM REMINDER: Tool calls must be executed via the native "
                                "function calling interface. Respond again, trigger the tool "
                                'directly, and only set "type" to "tool_call" in your JSON.'
                            ),
                        }
                    )
                    raise PatternExecutionError(
                        pattern_name="ReAct",
                        message=(
                            "Tool call requested via JSON without a native tool call. "
                            "Tools must be invoked through the function calling API."
                        ),
                        context={"response": response},
                    )
                else:
                    # No tools available but model attempted to call one.
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "SYSTEM REMINDER: No tools are available for this task. "
                                "Provide a final answer JSON instead of requesting a tool."
                            ),
                        }
                    )
                    raise PatternExecutionError(
                        pattern_name="ReAct",
                        message="Tool call requested when no tools are available.",
                        context={"response": response},
                    )

            return action
        except json.JSONDecodeError as e:
            logging.info(f"invalid json response: {content}")
            # JSON parsing failed - raise error for retry
            await log_llm_completion(response, False, None)
            raise PatternExecutionError(
                pattern_name="ReAct",
                message=f"JSON response is invalid: {str(e)}",
                context={
                    "response": response,
                    "json_error": str(e),
                    "content_length": len(content),
                    "content_preview": content[:200] + "..."
                    if len(content) > 200
                    else content,
                },
            )
        except ValidationError as e:
            await log_llm_completion(response, False, None)
            raise PatternExecutionError(
                pattern_name="ReAct",
                message=f"Invalid action format: {str(e)}",
                context={"response": response},
            )

    async def _invoke_tool_via_native_call(
        self, messages: List[Dict[str, str]]
    ) -> Action:
        """
        Second LLM call to invoke tool using native tool calling API.

        This method is called when the first LLM call returns action.type == "tool_call".
        It makes a second call with tool_schemas, prompting the LLM to use native
        function calling to select and invoke the appropriate tool.
        """
        tool_schemas = self.tool_registry.get_tool_schemas()

        if not tool_schemas:
            raise PatternExecutionError(
                pattern_name="ReAct",
                message="No tools available for tool calling",
                context={"available_tools": self.tool_registry.list_tools()},
            )

        # Add a system prompt to guide LLM to use native tool calling
        messages_with_prompt = messages + [
            {
                "role": "user",
                "content": (
                    "IMPORTANT INSTRUCTION FOR THIS STEP:\n"
                    "You have access to the following tools. Use the NATIVE FUNCTION CALLING interface "
                    "to invoke the appropriate tool now.\n\n"
                    "DO NOT respond with JSON format. DO NOT return a structured action JSON.\n"
                    "Instead, use the native function calling API to directly invoke the tool.\n\n"
                    "The system will handle the tool execution and return the result to you."
                ),
            }
        ]

        # Prepare chat parameters for native tool calling
        chat_kwargs: Dict[str, Any] = {
            "messages": messages_with_prompt,
            "tools": tool_schemas,
            "tool_choice": "auto",
        }

        # Disable thinking mode if supported
        if (
            hasattr(self.llm, "supports_thinking_mode")
            and self.llm.supports_thinking_mode
        ):
            chat_kwargs["thinking"] = {"type": "disabled"}

        try:
            # Clean messages before sending to LLM
            cleaned_messages = clean_messages(messages_with_prompt)
            chat_kwargs["messages"] = cleaned_messages

            # Get LLM response using streaming API
            full_content = ""
            usage = {}
            tool_calls_from_stream = []

            async for chunk in self.llm.stream_chat(**chat_kwargs):
                if chunk.is_token():
                    full_content += chunk.delta
                elif chunk.is_tool_call():
                    tool_calls_from_stream = chunk.tool_calls
                elif chunk.is_usage():
                    usage = chunk.usage
                elif chunk.is_error():
                    logger.error(f"Second call: Got error chunk: {chunk.content}")
                    raise RuntimeError(f"LLM stream error: {chunk.content}")

            # Construct response object
            if tool_calls_from_stream:
                response = {
                    "type": "tool_call",
                    "tool_calls": tool_calls_from_stream,
                    "reasoning": full_content.strip() if full_content else "",
                    "raw": {"usage": usage} if usage else {},
                }
            else:
                # LLM didn't use native tool calling
                logger.error("Second call: LLM did not use native tool calling!")
                logger.error(f"Second call: full_content = {full_content[:500]}")
                logger.error(f"Second call: chat_kwargs = {chat_kwargs}")
                raise PatternExecutionError(
                    pattern_name="ReAct",
                    message="LLM did not invoke native tool calling in second call",
                    context={"chat_kwargs": chat_kwargs, "full_content": full_content},
                )

            # Convert native tool call to Action
            return self._convert_native_tool_call_to_action(response)

        except Exception as e:
            logger.error(f"Tool invocation via native call failed: {str(e)}")
            raise PatternExecutionError(
                pattern_name="ReAct",
                message=f"Failed to invoke tool via native calling: {str(e)}",
                context={"error": str(e), "chat_kwargs": chat_kwargs},
            )

    def _try_parse_action_from_dict(
        self, data: dict, answer_fallback: str = ""
    ) -> Optional[Action]:
        """Try to create an Action from a parsed dict.

        Attempts strict validation first (model_validate), then falls back
        to manual field extraction. Returns None when the type field is
        neither 'tool_call' nor 'final_answer'.

        Args:
            data: Parsed dict with at least a "type" key.
            answer_fallback: Default answer text used when the "answer"
                key is absent in *data*.
        """
        action_type = data.get("type")
        if action_type not in ("tool_call", "final_answer"):
            return None

        try:
            return Action.model_validate(data)
        except Exception as e:
            logger.warning(f"Failed to validate {action_type} dict: {e}")

            if action_type == "tool_call":
                return Action(
                    type="tool_call",
                    reasoning=data.get("reasoning", "LLM wants to call a tool"),
                    tool_name=data.get("tool_name"),
                    tool_args=data.get("tool_args"),
                )
            else:
                return Action(
                    type="final_answer",
                    reasoning=data.get("reasoning", "LLM provided final answer"),
                    answer=data.get("answer", answer_fallback),
                    success=data.get("success", True),
                    error=data.get("error"),
                )

    def _normalize_action_data(self, action_data: Any) -> Any:
        """
        Normalize LLM action output to ensure a dict is passed to Action validation.

        Some LLMs may return a list of fragments or actions. We:
        - Use the first non-empty element
        - If it's a dict, return it directly
        - If it's a string, attempt to repair/parse JSON inside it
        - Otherwise, fall back to a final_answer action with the stringified value
        """
        if isinstance(action_data, dict):
            return self._ensure_action_type(action_data)

        if not isinstance(action_data, list):
            return action_data

        if not action_data:
            raise PatternExecutionError(
                pattern_name="ReAct", message="Action list is empty"
            )

        primary = next(
            (item for item in action_data if item not in (None, "", {})),
            action_data[0],
        )

        if isinstance(primary, dict):
            return self._ensure_action_type(primary)

        if isinstance(primary, str):
            try:
                repaired_inner = repair_loads(primary, logging=False)
                if isinstance(repaired_inner, dict):
                    return repaired_inner
            except Exception:
                pass

            return {
                "type": "final_answer",
                "reasoning": "Fallback: converted list response to final answer",
                "answer": primary,
            }

        return {
            "type": "final_answer",
            "reasoning": "Fallback: converted list response to final answer",
            "answer": str(primary),
        }

    def _ensure_action_type(self, action_data: Dict[str, Any]) -> Dict[str, Any]:
        """Infer missing or incorrect action 'type' from common fields."""
        action_type = action_data.get("type")
        # LLM may return incorrect action type
        if action_type in ("tool_call", "final_answer"):
            return action_data

        inferred: Optional[str] = None
        if action_data.get("tool_name") or action_data.get("tool_args"):
            inferred = "tool_call"
        elif (
            action_data.get("answer") is not None
            or action_data.get("success") is not None
            or action_data.get("error") is not None
        ):
            inferred = "final_answer"

        if inferred:
            action_data = dict(action_data)
            action_data["type"] = inferred
        return action_data

    def _convert_native_tool_call_to_action(self, response: Dict[str, Any]) -> Action:
        """Convert native tool call to Action format."""
        tool_calls = response.get("tool_calls", [])
        if not tool_calls:
            raise PatternExecutionError(
                pattern_name="ReAct", message="Native tool call missing tool_calls"
            )

        tool_call = tool_calls[0]
        function_info = tool_call.get("function", {})

        # Extract tool name and remove any 'functions.' prefix
        tool_name = function_info.get("name", "")
        if tool_name.startswith("functions."):
            tool_name = tool_name[len("functions.") :]

        reasoning = response.get("reasoning") or response.get("content") or ""

        return Action(
            type="tool_call",
            reasoning=reasoning,
            tool_name=tool_name,
            tool_args=json.loads(function_info.get("arguments", "{}")),
        )

    async def _execute_action(
        self,
        action: Action,
        messages: List[Dict[str, str]],
        task_id: Optional[str] = None,
        step_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute the structured action."""
        if action.type == "tool_call":
            if not action.tool_name:
                raise PatternExecutionError(
                    pattern_name="ReAct", message="Tool call missing tool_name"
                )

            # For native tool calling, add a minimal assistant message
            # Don't include the full tool call JSON as it confuses the LLM
            # Just add a placeholder to maintain conversation flow
            messages.append(
                {
                    "role": "assistant",
                    "content": f"[Calling tool: {action.tool_name}]",
                }
            )

            # Execute tool
            tool_args = action.tool_args or {}
            tool_execution_id = f"tool_{action.tool_name}_{uuid4().hex[:8]}"
            is_sandboxed = False

            try:
                tool = self.tool_registry.get(action.tool_name)

                # Check if tool runs in sandbox (duck-type detection)
                is_sandboxed = getattr(tool, "is_sandboxed", False)

                # Trace tool execution start
                if task_id is not None and step_id is not None:
                    await trace_tool_execution_start(
                        self.tracer,
                        task_id,
                        step_id,
                        action.tool_name,
                        data={
                            "tool_args": tool_args,
                            "tool_execution_id": tool_execution_id,
                            "step_id": step_id,
                            "step_name": getattr(self, "_current_step_name", "main"),
                            "sandboxed": is_sandboxed,
                        },
                    )

                result = await tool.run_json_async(tool_args)

                # Trace tool execution end
                if task_id is not None and step_id is not None:
                    await trace_action_end(
                        self.tracer,
                        task_id,
                        step_id,
                        TraceCategory.TOOL,
                        data={
                            "tool_name": action.tool_name,
                            "tool_args": tool_args,
                            "tool_execution_id": tool_execution_id,
                            "result": result,
                            "success": True,
                            "step_id": step_id,
                            "step_name": getattr(self, "_current_step_name", "main"),
                            "sandboxed": is_sandboxed,
                        },
                    )

                return {
                    "type": "observation",
                    "content": f"Tool '{action.tool_name}' executed successfully. Result: {result}",
                    "tool_name": action.tool_name,
                    "tool_args": tool_args,
                    "result": result,
                    "tool_execution_id": tool_execution_id,
                }

            except Exception as e:
                # Trace tool execution error
                if task_id is not None and step_id is not None:
                    await trace_error(
                        self.tracer,
                        task_id,
                        step_id,
                        error_type="ToolExecutionError",
                        error_message=f"Tool execution failed: {str(e)}",
                        data={
                            "tool_name": action.tool_name,
                            "tool_args": tool_args,
                            "tool_execution_id": tool_execution_id,
                            "step_id": step_id,
                            "step_name": getattr(self, "_current_step_name", "main"),
                            "sandboxed": is_sandboxed,
                        },
                    )

                return {
                    "type": "observation",
                    "content": f"Tool '{action.tool_name}' execution failed: {str(e)}",
                    "tool_name": action.tool_name,
                    "tool_args": tool_args,
                    "error": str(e),
                    "tool_execution_id": tool_execution_id,
                }

        elif action.type == "final_answer":
            logger.debug(
                f"_execute_action: Processing final_answer with answer: {str(action.answer)[:100] if action.answer else 'NO_ANSWER'}"
            )
            if not action.answer:
                raise PatternExecutionError(
                    pattern_name="ReAct", message="Final answer missing answer"
                )

            # Check if this is a failed final answer
            if action.success is False:
                error_message = (
                    action.error or "Task failed as indicated by the AI assistant"
                )
                logger.warning(f"Final answer indicates failure: {error_message}")

                # Add action to conversation history
                messages.append(
                    {
                        "role": "assistant",
                        "content": json.dumps(action.model_dump(), indent=2),
                    }
                )

                # Return failure result directly - LLM has decided the task cannot be completed
                # This is consistent with LangChain's AgentFinish design: once LLM returns a final answer,
                # the agent loop stops regardless of success/failure status
                return {
                    "type": "final_answer",
                    "content": action.answer,
                    "reasoning": action.reasoning,
                    "success": False,
                    "error": error_message,
                }

            # Add action to conversation history
            messages.append(
                {
                    "role": "assistant",
                    "content": json.dumps(action.model_dump(), indent=2),
                }
            )

            result = {
                "type": "final_answer",
                "content": action.answer,
                "reasoning": action.reasoning,
                "success": action.success if action.success is not None else True,
                "error": action.error,
            }
            logger.debug(
                f"_execute_action: Returning final_answer result with content length: {len(result.get('content', ''))}"
            )
            return result

        else:
            raise PatternExecutionError(
                pattern_name="ReAct", message=f"Unknown action type: {action.type}"
            )

    def _extract_content(self, response: Any) -> str:
        """Extract content from LLM response."""
        if response is None:
            return ""
        elif isinstance(response, str):
            # Check if this is a JSON string (e.g., "{\"type\": ...}")
            # Try to parse it to extract the actual content
            try:
                parsed = repair_loads(response, logging=False)
                if isinstance(parsed, dict):
                    # Successfully parsed JSON string, return stringified version
                    return json.dumps(parsed, ensure_ascii=False)
            except Exception:
                # Not a JSON string, return as-is
                pass
            return response
        elif isinstance(response, list):
            # Some providers return list fragments; join them into a single string
            try:
                return "".join(
                    part if isinstance(part, str) else json.dumps(part)
                    for part in response
                    if part is not None
                )
            except Exception:
                return str(response)
        elif isinstance(response, dict):
            if "content" in response:
                content = response["content"]
                return str(content) if content is not None else ""
            elif "message" in response:
                message = response["message"]
                return str(message) if message is not None else ""
            else:
                return str(response)
        else:
            return str(response)

    async def _generate_and_store_react_memories(
        self, task: str, result: str, iterations: int, messages: List[Dict[str, str]]
    ) -> None:
        """
        Generate comprehensive insights for ReAct task execution and store memories.

        Args:
            task: The original task
            result: The final result/output
            iterations: Number of iterations taken
            messages: Full conversation history
        """

        try:
            # Trace memory generation start
            import time

            task_id = (
                getattr(self, "task_id", None)
                or (
                    getattr(self.parent_pattern, "task_id", None)
                    if hasattr(self, "parent_pattern") and self.parent_pattern
                    else None
                )
                or f"react_memory_{int(time.time())}"
            )
            await trace_memory_generate_start(
                self.tracer,
                task_id,
                data={
                    "task": task,
                    "iterations": iterations,
                    "result_length": len(result),
                    "messages_count": len(messages),
                    "step_id": getattr(self, "_current_step_id", None),
                },
            )

            # Generate all insights with a single LLM call
            insights = await self._generate_react_insights(
                task, result, iterations, messages
            )

            # Trace memory generation end
            await trace_memory_generate_end(
                self.tracer,
                task_id,
                data={
                    "insights_generated": insights is not None,
                    "should_store": insights.get("should_store", False)
                    if insights
                    else False,
                    "reason": insights.get("reason", "") if insights else "",
                    "step_id": getattr(self, "_current_step_id", None),
                },
            )

            if insights:
                # Check if LLM recommends storing this memory
                should_store = insights.get("should_store", False)
                reason = insights.get("reason", "Unknown reason")

                if should_store and self.memory_store:
                    # Trace memory storage start
                    await trace_memory_store_start(
                        self.tracer,
                        task_id,
                        data={
                            "task": task,
                            "memory_category": "react_memory",
                            "classification": insights.get("classification", {}),
                            "step_id": getattr(self, "_current_step_id", None),
                        },
                    )

                    # Store the main task memory only if valuable and memory_store is available
                    await asyncio.to_thread(
                        store_react_task_memory,
                        memory_store=self.memory_store,
                        task=task,
                        result={
                            "success": True,
                            "output": result,
                            "iterations": iterations,
                            "history": messages[-10:]
                            if len(messages) > 10
                            else messages,  # Last 10 messages for context
                        },
                        tool_usage_insights=insights.get("tool_usage_insights", ""),
                        reasoning_strategy=insights.get("reasoning_strategy", ""),
                        classification={
                            # Map new simplified format to expected fields
                            "user_preferences": insights.get("user_preferences", ""),
                            "core_insight": insights.get("core_insight", ""),
                            "failure_patterns": insights.get("failure_patterns", ""),
                            "success_patterns": insights.get("success_patterns", ""),
                        },
                    )

                    # Trace memory storage end
                    await trace_memory_store_end(
                        self.tracer,
                        task_id,
                        data={
                            "storage_success": True,
                            "reason": reason,
                            "step_id": getattr(self, "_current_step_id", None),
                        },
                    )

                    logger.info(
                        f"Stored valuable ReAct memory for task: {task[:100]}... Reason: {reason}"
                    )
                elif should_store and not self.memory_store:
                    # Trace memory storage decision (would store but no memory_store)
                    await trace_memory_store_end(
                        self.tracer,
                        task_id,
                        data={
                            "storage_success": False,
                            "reason": f"Would store but no memory_store available: {reason}",
                            "decision": "no_memory_store",
                            "step_id": getattr(self, "_current_step_id", None),
                        },
                    )

                    logger.info(
                        f"Would store valuable ReAct memory but no memory_store: {task[:100]}... Reason: {reason}"
                    )
                else:
                    # Trace memory storage decision (not storing)
                    await trace_memory_store_end(
                        self.tracer,
                        task_id,
                        data={
                            "storage_success": False,
                            "reason": reason,
                            "decision": "not_worth_storing",
                            "step_id": getattr(self, "_current_step_id", None),
                        },
                    )

                    logger.info(
                        f"ReAct task not worth storing as memory: {task[:100]}... Reason: {reason}"
                    )

        except Exception as e:
            logger.error(f"Failed to generate and store ReAct memories: {e}")

    async def _generate_react_insights(
        self, task: str, result: str, iterations: int, messages: List[Dict[str, str]]
    ) -> Optional[Dict[str, Any]]:
        """
        Generate comprehensive insights for ReAct task execution by reusing existing messages context.

        Args:
            task: The original task
            result: The final result/output
            iterations: Number of iterations taken
            messages: Full conversation history

        Returns:
            Dictionary with comprehensive insights or None if generation fails
        """
        try:
            # Analyze tool usage from conversation
            tool_calls = []
            for msg in messages:
                if msg.get("role") == "assistant":
                    try:
                        content = msg.get("content", "")
                        if content.startswith("{") and content.endswith("}"):
                            action_data = json.loads(content)
                            if action_data.get("type") == "tool_call":
                                tool_name = action_data.get("tool_name")
                                if tool_name:
                                    tool_calls.append(tool_name)
                    except (json.JSONDecodeError, AttributeError):
                        pass

            # Count unique tools used
            unique_tools = list(set(tool_calls))
            tool_usage_count = len(tool_calls)

            # Build analysis summary for strict memory evaluation
            analysis_summary = f"""MEMORY STORAGE EVALUATION:

TASK: {task}
ITERATIONS: {iterations}
TOOL CALLS: {tool_usage_count} (unique: {len(unique_tools)}): {", ".join(unique_tools) if unique_tools else "None"}

RESULT PREVIEW:
{result[:300]}{"..." if len(result) > 300 else ""}

CRITICAL STORAGE DECISION:
Evaluate if this execution contains UNIQUE, NON-OBVIOUS insights that would be valuable for FUTURE tasks.

STORAGE CRITERIA (ALL must be met):
1. **NO ROUTINE EXECUTIONS** - Standard tool usage patterns should NOT be stored
2. **UNIQUE FAILURES** - Non-obvious failures with important lessons learned
3. **INNOVATIVE STRATEGIES** - Novel approaches not commonly known
4. **DOMAIN EXPERTISE** - Specialized knowledge hard to obtain otherwise
5. **USER PREFERENCES** - Clear user behavioral patterns or preferences
6. **REUSABLE PATTERNS** - Abstract patterns applicable to many scenarios

REJECT STORAGE IF:
- Routine task completion following standard patterns
- Generic "effective tool usage" descriptions
- Common problem-solving approaches
- Obvious strategies that don't provide new insights
- General information easily obtainable elsewhere

Provide JSON response:
{{
    "should_store": true/false,
    "reason": "Specific explanation of unique value or rejection reason",
    "core_insight": "Single sentence capturing the essential learning if stored",
    "user_preferences": "Observable user preferences (e.g., language preference, detail level, response format)",
    "failure_patterns": "Non-obvious failure modes and solutions discovered",
    "success_patterns": "Key strategies that led to success beyond standard approaches"
}}

STORAGE THRESHOLD: Be extremely conservative. Only store truly exceptional insights. When in doubt, set should_store to false."""

            # Reuse existing messages context by appending the analysis request
            # This preserves the full conversation context and provides better continuity
            working_messages = messages.copy()  # Don't modify original messages
            working_messages.append({"role": "user", "content": analysis_summary})

            # Clean messages before sending to LLM
            cleaned_working_messages = clean_messages(working_messages)

            # Get comprehensive insights from LLM using the existing context
            response = await self.llm.chat(messages=cleaned_working_messages)
            if isinstance(response, dict):
                response_text = response.get("content", str(response))
            else:
                response_text = str(response)

            # Parse JSON response
            insights_data = json.loads(response_text)
            assert isinstance(insights_data, dict), "Insights data must be a dictionary"

            logger.info(
                f"Generated comprehensive ReAct insights for task: {task[:50]}..."
            )
            return insights_data

        except Exception as e:
            logger.error(f"Failed to generate ReAct insights: {e}")
            return None

    def _log_full_context(
        self, messages: List[Dict[str, str]], response: Any, error_type: str
    ) -> None:
        """Log full context for debugging."""
        logger.error(f"=== {error_type} ===")
        logger.error(f"Messages count: {len(messages)}")
        for i, msg in enumerate(messages[-3:]):  # Show last 3 messages
            logger.error(
                f"Message {i}: {msg.get('role', 'unknown')} - {msg.get('content', '')[:200]}"
            )
        logger.error(f"LLM response: {response}")
        logger.error("=== END CONTEXT ===")

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
                from ....web.user_isolated_memory import current_user_id

                context_token = current_user_id.set(user_id)
            except ImportError:
                # Fallback for non-web environment - proceed without user context
                from .memory_utils import lookup_relevant_memories

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
                from .memory_utils import lookup_relevant_memories

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
            from .memory_utils import lookup_relevant_memories

            return lookup_relevant_memories(
                memory_store,
                query,
                category,
                include_general,
                limit,
                similarity_threshold,
            )
