"""Single-Call Tool Pattern Implementation

This module implements a simple pattern that executes a single tool call
based on the task content and returns the result directly.

This pattern is useful for simple, one-shot tool invocations without
complex reasoning or multi-step execution.
"""

__all__ = ["SingleCallPattern"]

import json
import logging
from typing import Any, Dict, List, Optional
from uuid import uuid4

from ...memory import MemoryStore
from ...model.chat.basic.base import BaseLLM
from ...tools.adapters.vibe import Tool
from ..context import AgentContext
from ..exceptions import (
    LLMNotAvailableError,
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
    trace_task_completion,
    trace_task_end,
    trace_task_start,
    trace_tool_execution_start,
    trace_user_message,
)
from ..transcript import normalize_transcript_messages
from ..utils.llm_utils import clean_messages
from .base import AgentPattern, ToolRegistry

logger = logging.getLogger(__name__)


class SingleCallPattern(AgentPattern):
    """
    Single-Call Tool Pattern

    This pattern executes a single tool call based on the task content
    and returns the result directly. It uses the same trace format as
    ReAct for compatibility with the frontend.

    Execution flow:
    1. Parse task to identify tool and arguments
    2. Execute the tool
    3. Return the result
    """

    def __init__(
        self,
        llm: Optional[BaseLLM] = None,
        tracer: Optional[Tracer] = None,
        memory_store: Optional[MemoryStore] = None,
    ):
        """
        Initialize SingleCall pattern.

        Args:
            llm: Language model for tool selection (optional, can use direct tool calls)
            tracer: Tracer instance for event tracking
            memory_store: Memory store for persistence (optional)
        """
        self.llm = llm
        self.tracer = tracer or Tracer()
        self.memory_store = memory_store
        self.tool_registry = ToolRegistry()
        self._context: Optional[AgentContext] = None
        self._conversation_history: List[Dict[str, str]] = []
        self._execution_context_messages: List[Dict[str, str]] = []

    def set_conversation_history(self, messages: List[Dict[str, Any]]) -> None:
        """Replace the persisted top-level conversation transcript for a new run."""
        self._conversation_history = normalize_transcript_messages(messages)

    def set_execution_context_messages(self, messages: List[Dict[str, Any]]) -> None:
        """Load persisted execution-state context for a new run."""
        self._execution_context_messages = normalize_transcript_messages(messages)

    async def run(
        self,
        task: str,
        memory: MemoryStore,
        tools: List[Tool],
        context: Optional[AgentContext] = None,
    ) -> Dict[str, Any]:
        """
        Execute the single-call pattern.

        Uses native LLM tool calling to let the model decide whether to:
        1. Call a tool (if needed)
        2. Provide a final answer directly

        Args:
            task: The task to accomplish
            memory: Memory store for persistence
            tools: Available tools
            context: Execution context

        Returns:
            Execution result with success status and output

        Raises:
            LLMNotAvailableError: When LLM is not available
            PatternExecutionError: When execution fails
        """
        logger.info(f"Starting SingleCall execution for task: {task[:100]}...")

        # LLM is required for SingleCall pattern
        if not self.llm:
            return {
                "success": False,
                "error": "LLM is required for SingleCall pattern",
                "output": "Execution error: LLM is required for SingleCall pattern",
                "pattern": "single_call",
            }

        # Store context
        self._context = context

        # Create task ID for tracing
        task_id = f"single_call_{context.task_id if context else uuid4()}"

        # Create a virtual step ID for ReAct-compatible tracing
        step_id = f"{task_id}_main"

        # Emit user message trace
        if context:
            await trace_user_message(
                self.tracer,
                task_id,
                task,
                {},  # No additional trace data for single call
            )

        # Trace task start (using REACT category for frontend compatibility)
        await trace_task_start(
            self.tracer,
            task_id,
            TraceCategory.REACT,  # Use REACT for frontend compatibility
            data={
                "pattern": "SingleCall",
                "task": task[:100],
                "tools": [tool.metadata.name for tool in tools],
                "step_id": step_id,
                "step_name": "main",
            },
        )

        # Register tools
        self.tool_registry.register_all(tools)

        try:
            # Execute single LLM call with native tool calling
            result = await self._execute_single_llm_call(task, task_id, step_id)

            # Trace AI message with the result
            await trace_ai_message(
                self.tracer,
                task_id,
                message=result["output"],
                data={"content": result["output"]},
            )

            # Trace task completion
            await trace_task_completion(
                self.tracer,
                task_id,
                result=result["output"],
                success=result["success"],
            )

            # Trace task end (REACT category for frontend compatibility)
            await trace_task_end(
                self.tracer,
                task_id,
                TraceCategory.REACT,  # Use REACT for frontend compatibility
                data={
                    "result": result["output"],
                    "success": result["success"],
                },
            )

            return result

        except Exception as e:
            # Trace error
            await trace_error(
                self.tracer,
                task_id,
                step_id,
                error_type=type(e).__name__,
                error_message=str(e),
                data={
                    "task": task[:100],
                    "pattern": "SingleCall",
                },
            )

            # Re-raise critical exceptions
            if isinstance(e, (ToolNotFoundError, LLMNotAvailableError)):
                raise

            # Return error result for other exceptions
            return {
                "success": False,
                "error": str(e),
                "output": f"Execution error: {str(e)}",
                "pattern": "single_call",
            }

    async def _execute_single_llm_call(
        self, task: str, task_id: str, step_id: str
    ) -> Dict[str, Any]:
        """
        Execute a single LLM call with native tool calling.

        The LLM will decide whether to:
        1. Call a tool (if needed for the task)
        2. Provide a final answer directly

        Args:
            task: The task description
            task_id: Task ID for tracing
            step_id: Step ID for tracing

        Returns:
            Execution result dictionary
        """
        # Get tool schemas for native function calling
        tool_schemas = self.tool_registry.get_tool_schemas()

        # Build messages with conversation history and execution context
        messages: List[Dict[str, str]] = []

        # Add execution context messages first (may contain system prompt with KB info)
        if self._execution_context_messages:
            messages.extend(self._execution_context_messages)
        else:
            # Fallback system prompt if no execution context
            messages.append(
                {"role": "system", "content": "You are a helpful assistant."}
            )

        # Add conversation history if available
        if self._conversation_history:
            messages.extend(self._conversation_history)

        # Add current task
        messages.append({"role": "user", "content": task})

        # Prepare chat kwargs with tools
        chat_kwargs: Dict[str, Any] = {
            "messages": messages,
        }

        # Add tools if available
        if tool_schemas:
            chat_kwargs["tools"] = tool_schemas
            chat_kwargs["tool_choice"] = "auto"

        # Trace LLM call start
        await trace_llm_call_start(
            self.tracer,
            task_id,
            step_id,
            data={
                "action": "LLM call started",
                "model_name": getattr(self.llm, "model_name", type(self.llm).__name__),
                "task_type": "SingleCall execution",
                "attempt": 1,
                "messages_count": len(messages),
                "has_tools": bool(tool_schemas),
                "tools_count": len(tool_schemas),
                "tool_choice": chat_kwargs.get("tool_choice", "auto"),
                "step_name": "main",
                "step_id": step_id,
            },
        )

        try:
            # Clean messages
            cleaned_messages = clean_messages(messages)
            chat_kwargs["messages"] = cleaned_messages

            # Get LLM response
            if not self.llm:
                raise PatternExecutionError(
                    self.__class__.__name__, "SingleCall pattern requires an LLM"
                )
            response = await self.llm.chat(**chat_kwargs)

            logger.info(f"SingleCall LLM response type: {type(response)}")

            # Check if LLM made a tool call
            if isinstance(response, dict) and response.get("type") == "tool_call":
                tool_calls = response.get("tool_calls", [])
                if tool_calls:
                    tool_call = tool_calls[0]
                    function_info = tool_call.get("function", {})
                    tool_name = function_info.get("name", "")
                    arguments_str = function_info.get("arguments", "{}")
                    try:
                        tool_args = json.loads(arguments_str)
                    except json.JSONDecodeError as e:
                        raise PatternExecutionError(
                            self.__class__.__name__,
                            f"Failed to parse tool arguments as JSON: {e}",
                        ) from e

                    logger.info(
                        f"SingleCall executing tool: {tool_name} with args: {tool_args}"
                    )

                    # Trace action start
                    await trace_action_start(
                        self.tracer,
                        task_id,
                        step_id,
                        TraceCategory.REACT,
                        data={
                            "iteration": 1,
                            "task_id": task_id,
                            "step_id": step_id,
                            "step_name": "main",
                        },
                    )

                    # Execute the tool
                    tool_result = await self._execute_tool(
                        tool_name, tool_args, task_id, step_id
                    )

                    # Trace action end
                    await trace_action_end(
                        self.tracer,
                        task_id,
                        step_id,
                        TraceCategory.REACT,
                        data={
                            "action_type": "tool_call",
                            "tool_name": tool_name,
                            "result_type": "observation",
                            "step_id": step_id,
                            "step_name": "main",
                            "reasoning": f"Tool call: {tool_name}",
                        },
                    )

                    # Check if tool execution failed
                    if not tool_result.get("success", True):
                        # Return error result directly
                        return {
                            "success": False,
                            "output": tool_result.get(
                                "content", "Tool execution failed"
                            ),
                            "error": tool_result.get("error", "Tool execution failed"),
                            "tool_name": tool_name,
                            "tool_args": tool_args,
                            "tool_result": tool_result,
                            "pattern": "single_call",
                        }

                    # Tool execution succeeded - generate final answer
                    logger.info("SingleCall generating final answer from tool result")
                    final_answer = await self._generate_final_answer_from_tool_result(
                        task, tool_name, tool_result, task_id, step_id
                    )

                    # Return final answer with tool info
                    return {
                        "success": True,
                        "output": final_answer,
                        "tool_name": tool_name,
                        "tool_args": tool_args,
                        "tool_result": tool_result,
                        "pattern": "single_call",
                    }

            # Extract text response (final answer)
            if isinstance(response, dict):
                response_text = response.get("content", str(response))
            else:
                response_text = str(response)

            logger.info(f"SingleCall got final answer: {response_text[:200]}")

            # Return final answer
            return {
                "success": True,
                "output": response_text,
                "pattern": "single_call",
                "is_final_answer": True,
            }

        except Exception as e:
            logger.error(f"SingleCall execution failed: {e}")
            # Trace error
            await trace_error(
                self.tracer,
                task_id,
                step_id,
                error_type="SingleCallError",
                error_message=f"SingleCall execution failed: {str(e)}",
                data={
                    "llm_type": type(self.llm).__name__,
                    "step_name": "main",
                },
            )
            raise PatternExecutionError(
                pattern_name="SingleCall",
                message=f"SingleCall execution failed: {str(e)}",
                context={"task": task[:100]},
            )

    async def _execute_tool(
        self, tool_name: str, tool_args: Dict[str, Any], task_id: str, step_id: str
    ) -> Dict[str, Any]:
        """
        Execute a tool with the given arguments.

        Args:
            tool_name: Name of the tool to execute
            tool_args: Arguments for the tool
            task_id: Task ID for tracing
            step_id: Step ID for tracing

        Returns:
            Tool execution result
        """
        try:
            tool = self.tool_registry.get(tool_name)

            # Create tool execution ID
            tool_execution_id = f"tool_{tool_name}_{uuid4().hex[:8]}"

            # Check if tool runs in sandbox
            is_sandboxed = getattr(tool, "is_sandboxed", False)

            # Trace tool execution start
            await trace_tool_execution_start(
                self.tracer,
                task_id,
                step_id,
                tool_name,
                data={
                    "tool_args": tool_args,
                    "tool_execution_id": tool_execution_id,
                    "step_id": step_id,
                    "step_name": "main",
                    "sandboxed": is_sandboxed,
                },
            )

            # Execute tool
            result = await tool.run_json_async(tool_args)

            # Trace tool execution end
            await trace_action_end(
                self.tracer,
                task_id,
                step_id,
                TraceCategory.TOOL,
                data={
                    "tool_name": tool_name,
                    "tool_args": tool_args,
                    "tool_execution_id": tool_execution_id,
                    "result": result,
                    "success": True,
                    "step_id": step_id,
                    "step_name": "main",
                    "sandboxed": is_sandboxed,
                },
            )

            return {
                "success": True,
                "content": str(result),
                "result": result,
                "tool_execution_id": tool_execution_id,
            }

        except Exception as e:
            # Trace tool execution error
            await trace_error(
                self.tracer,
                task_id,
                step_id,
                error_type="ToolExecutionError",
                error_message=f"Tool execution failed: {str(e)}",
                data={
                    "tool_name": tool_name,
                    "tool_args": tool_args,
                    "step_id": step_id,
                    "step_name": "main",
                },
            )

            return {
                "success": False,
                "content": f"Tool execution failed: {str(e)}",
                "error": str(e),
                "tool_name": tool_name,
                "tool_args": tool_args,
            }

    async def _generate_final_answer_from_tool_result(
        self,
        task: str,
        tool_name: str,
        tool_result: Dict[str, Any],
        task_id: str,
        step_id: str,
    ) -> str:
        """
        Generate final answer based on tool execution result.

        Args:
            task: Original task
            tool_name: Name of the tool that was executed
            tool_result: Result from tool execution
            task_id: Task ID for tracing
            step_id: Step ID for tracing

        Returns:
            Final answer string
        """
        # Build messages for final answer generation
        # Include conversation history and execution context
        messages: List[Dict[str, str]] = []

        # Add system prompt for final answer generation
        system_prompt = """You are an AI assistant. Based on the tool execution result below, provide a helpful and accurate answer to the user's question.

IMPORTANT:
- Do NOT call any tools. Just provide a direct answer based on the tool result.
- Use the information from the tool execution to answer the user's question.
"""

        messages.append({"role": "system", "content": system_prompt})

        # Add execution context messages if available
        if self._execution_context_messages:
            messages.extend(self._execution_context_messages)

        # Add conversation history if available
        if self._conversation_history:
            messages.extend(self._conversation_history)

        # Add original task
        messages.append({"role": "user", "content": task})

        # Add tool call and result as assistant message
        tool_content = tool_result.get("content", str(tool_result.get("result", "")))
        messages.append(
            {
                "role": "assistant",
                "content": f"Used tool: {tool_name}\nTool result: {tool_content}",
            }
        )

        # Prepare chat kwargs
        chat_kwargs: Dict[str, Any] = {
            "messages": messages,
        }

        # Trace LLM call for final answer
        await trace_llm_call_start(
            self.tracer,
            task_id,
            step_id,
            data={
                "action": "LLM call for final answer",
                "model_name": getattr(self.llm, "model_name", type(self.llm).__name__),
                "task_type": "Final answer generation",
                "attempt": 2,
                "messages_count": len(messages),
                "messages": messages,
                "step_name": "main",
                "step_id": step_id,
            },
        )

        try:
            # Clean messages
            cleaned_messages = clean_messages(messages)
            chat_kwargs["messages"] = cleaned_messages

            # Get LLM response for final answer
            if self.llm is None:
                raise LLMNotAvailableError(
                    "LLM is required for final answer generation"
                )
            response = await self.llm.chat(**chat_kwargs)

            # Extract response content
            if isinstance(response, dict):
                final_answer_str: str = response.get("content", str(response))
            else:
                final_answer_str = str(response)

            logger.info(f"SingleCall generated final answer: {final_answer_str[:200]}")
            return final_answer_str

        except Exception as e:
            logger.error(f"SingleCall failed to generate final answer: {e}")
            # Fallback: return tool result directly
            return str(tool_content)
