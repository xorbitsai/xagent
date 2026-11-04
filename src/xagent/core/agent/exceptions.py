"""
Agent execution exceptions hierarchy.

This module defines specific exceptions for different failure scenarios
in agent execution, replacing generic string error handling.
"""

from typing import Any, Dict, List, Optional


class AgentException(Exception):
    """Base exception for all agent-related errors."""

    def __init__(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        super().__init__(message)
        self.context = context or {}
        self.cause = cause

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for serialization."""
        return {
            "type": self.__class__.__name__,
            "message": str(self),
            "context": self.context,
            "cause": str(self.cause) if self.cause else None,
        }


class AgentConfigurationError(AgentException):
    """Raised when agent is misconfigured."""

    pass


class LLMError(AgentException):
    """Base class for LLM-related errors."""

    pass


class LLMNotAvailableError(LLMError):
    """Raised when LLM is required but not configured."""

    pass


class LLMResponseError(LLMError):
    """Raised when LLM returns invalid or unparsable response."""

    def __init__(
        self,
        message: str,
        response: Any = None,
        expected_format: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        super().__init__(message, context=context, cause=cause)
        self.response = response
        self.expected_format = expected_format
        if response is not None:
            self.context["response"] = str(response)[:500]  # Truncate for safety
        if expected_format:
            self.context["expected_format"] = expected_format


class ToolError(AgentException):
    """Base class for tool execution errors."""

    pass


class ToolNotFoundError(ToolError):
    """Raised when a required tool is not available."""

    def __init__(
        self, tool_name: str, available_tools: Optional[List[str]] = None, **kwargs: Any
    ) -> None:
        message = f"Tool '{tool_name}' not found"
        if available_tools:
            message += f". Available tools: {', '.join(available_tools)}"
        super().__init__(message, **kwargs)
        self.tool_name = tool_name
        self.available_tools = available_tools or []
        self.context.update(
            {"tool_name": tool_name, "available_tools": self.available_tools}
        )


class ToolExecutionError(ToolError):
    """Raised when tool execution fails."""

    def __init__(
        self,
        tool_name: str,
        message: str,
        tool_args: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(f"Tool '{tool_name}' execution failed: {message}", **kwargs)
        self.tool_name = tool_name
        self.tool_args = tool_args or {}
        self.context.update({"tool_name": tool_name, "tool_args": self.tool_args})


class PatternError(AgentException):
    """Base class for pattern execution errors."""

    pass


class PatternExecutionError(PatternError):
    """Raised when a pattern fails to execute."""

    def __init__(
        self,
        pattern_name: str,
        message: str,
        iteration: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(f"Pattern '{pattern_name}' failed: {message}", **kwargs)
        self.pattern_name = pattern_name
        self.iteration = iteration
        self.context.update({"pattern_name": pattern_name, "iteration": iteration})


class MaxIterationsError(PatternError):
    """Raised when pattern reaches maximum iterations without completion."""

    def __init__(
        self,
        pattern_name: str,
        max_iterations: int,
        final_state: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        message = (
            f"Pattern '{pattern_name}' reached maximum iterations ({max_iterations})"
        )
        if final_state:
            message += f". Final state: {final_state}"

        # Add execution summary to message if available
        if "execution_summary" in kwargs.get("context", {}):
            execution_summary = kwargs["context"]["execution_summary"]
            message += "\n\nExecution summary:\n" + "\n".join(execution_summary)

        # Add iteration count if available
        if "total_iterations" in kwargs.get("context", {}):
            total_iterations = kwargs["context"]["total_iterations"]
            message += f"\n\nTotal iterations completed: {total_iterations}"

        super().__init__(message, **kwargs)
        self.pattern_name = pattern_name
        self.max_iterations = max_iterations
        self.final_state = final_state
        self.context.update(
            {
                "pattern_name": pattern_name,
                "max_iterations": max_iterations,
                "final_state": final_state,
            }
        )


class DAGError(AgentException):
    """Base class for DAG execution errors."""

    pass


class DAGPlanGenerationError(DAGError):
    """Raised when DAG plan generation fails."""

    def __init__(
        self,
        message: str,
        iteration: Optional[int] = None,
        llm_response: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        # Extract context and cause from kwargs, filter out other parameters
        context = kwargs.pop("context", None)
        cause = kwargs.pop("cause", None)

        # Any remaining kwargs can be added to context
        if kwargs:
            context = context or {}
            context.update(kwargs)

        super().__init__(
            f"DAG plan generation failed: {message}", context=context, cause=cause
        )
        self.iteration = iteration
        self.llm_response = llm_response
        if iteration is not None:
            self.context["iteration"] = iteration
        if llm_response:
            self.context["llm_response"] = llm_response[:500]  # Truncate


class DAGStepError(DAGError):
    """Raised when a DAG step fails."""

    def __init__(
        self,
        step_id: str,
        step_name: str,
        message: str,
        dependencies: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        # Build a more detailed error message that includes original exception info
        detailed_message = f"DAG step '{step_id}' ({step_name}) failed: {message}"

        # If there's a cause exception, include its type in the message
        if "cause" in kwargs and kwargs["cause"]:
            cause = kwargs["cause"]
            detailed_message += (
                f" (caused by: {cause.__class__.__name__}: {str(cause)})"
            )

        super().__init__(detailed_message, **kwargs)
        self.step_id = step_id
        self.step_name = step_name
        self.dependencies = dependencies or []
        self.context.update(
            {
                "step_id": step_id,
                "step_name": step_name,
                "dependencies": self.dependencies,
            }
        )

        # Add original exception details to context if available
        if "cause" in kwargs and kwargs["cause"]:
            cause = kwargs["cause"]
            self.context.update(
                {
                    "original_exception_type": cause.__class__.__name__,
                    "original_exception_message": str(cause),
                }
            )


class DAGDependencyError(DAGError):
    """Raised when DAG dependency validation fails."""

    def __init__(
        self, step_id: str, invalid_dependencies: List[str], **kwargs: Any
    ) -> None:
        message = f"Step '{step_id}' has invalid dependencies: {', '.join(invalid_dependencies)}"
        super().__init__(message, **kwargs)
        self.step_id = step_id
        self.invalid_dependencies = invalid_dependencies
        self.context.update(
            {"step_id": step_id, "invalid_dependencies": invalid_dependencies}
        )


class DAGDeadlockError(DAGError):
    """Raised when DAG execution is stuck due to dependency deadlock."""

    def __init__(
        self,
        pending_steps: List[str],
        blocked_dependencies: Dict[str, List[str]],
        **kwargs: Any,
    ) -> None:
        message = f"DAG execution deadlock detected. Pending steps: {', '.join(pending_steps)}"
        if blocked_dependencies:
            blocked_deps_str = "; ".join(
                f"{step}: {', '.join(deps)}"
                for step, deps in blocked_dependencies.items()
            )
            message += f". Blocked dependencies: {blocked_deps_str}"

        super().__init__(message, **kwargs)
        self.pending_steps = pending_steps
        self.blocked_dependencies = blocked_dependencies
        self.context.update(
            {
                "pending_steps": pending_steps,
                "blocked_dependencies": blocked_dependencies,
            }
        )


class DAGExecutionError(DAGError):
    """Raised when DAG execution fails due to step failures."""

    def __init__(
        self,
        message: str,
        failed_steps: List[Any],
        completed_steps: int,
        total_steps: int,
        primary_error: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.failed_steps = failed_steps
        self.completed_steps = completed_steps
        self.total_steps = total_steps
        self.primary_error = primary_error
        self.context.update(
            {
                "failed_steps_count": len(failed_steps),
                "completed_steps": completed_steps,
                "total_steps": total_steps,
                "primary_error": primary_error,
                "failed_step_names": [f"{s.id} ({s.name})" for s in failed_steps],
            }
        )


class ReActError(PatternError):
    """Base class for ReAct pattern errors."""

    pass


class ReActParsingError(ReActError):
    """Raised when ReAct cannot parse LLM response."""

    def __init__(
        self,
        message: str,
        response: Any = None,
        iteration: Optional[int] = None,
        expected_format: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        super().__init__(
            f"ReAct parsing failed: {message}", context=context, cause=cause
        )
        self.response = response
        self.iteration = iteration
        self.expected_format = expected_format
        if response is not None:
            self.context["response"] = str(response)[:500]
        if iteration is not None:
            self.context["iteration"] = iteration
        if expected_format:
            self.context["expected_format"] = expected_format


class ContextError(AgentException):
    """Base class for context management errors."""

    pass


class ContextCompactionError(ContextError):
    """Raised when context compaction fails."""

    def __init__(
        self,
        message: str,
        original_size: Optional[int] = None,
        target_step: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(f"Context compaction failed: {message}", **kwargs)
        self.original_size = original_size
        self.target_step = target_step
        self.context.update(
            {"original_size": original_size, "target_step": target_step}
        )


class AgentToolError(ToolError):
    """Raised when AgentTool (nested agent) execution fails."""

    def __init__(
        self,
        agent_name: str,
        message: str,
        sub_agent_error: Optional[Exception] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(f"Sub-agent '{agent_name}' failed: {message}", **kwargs)
        self.agent_name = agent_name
        self.sub_agent_error = sub_agent_error
        self.context.update(
            {
                "agent_name": agent_name,
                "sub_agent_error": str(sub_agent_error) if sub_agent_error else None,
            }
        )


def create_execution_error(
    error_type: str,
    message: str,
    context: Optional[Dict[str, Any]] = None,
    cause: Optional[Exception] = None,
) -> AgentException:
    """
    Factory function to create appropriate exception types.

    Useful for converting from legacy string-based error handling.
    """
    error_map = {
        "pattern_execution_error": PatternExecutionError,
        "max_iterations": MaxIterationsError,
        "dag_plan_generation": DAGPlanGenerationError,
        "dag_step_error": DAGStepError,
        "dag_dependency_error": DAGDependencyError,
        "react_parsing_error": ReActParsingError,
        "context_compaction_error": ContextCompactionError,
        "agent_tool_error": AgentToolError,
    }

    # Use error_map to get appropriate exception type
    exception_cls = error_map.get(error_type, AgentException)
    return exception_cls(message, context=context, cause=cause)  # type: ignore
