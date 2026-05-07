import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from ...memory import MemoryStore
from ...tools.adapters.vibe import Tool
from ..context import AgentContext

logger = logging.getLogger(__name__)


def notify_condition(condition: asyncio.Condition) -> None:
    """Schedule a notify_all on an asyncio.Condition from sync code.

    Used by pause/resume/interrupt methods that are synchronous but need
    to wake up coroutines blocked on a Condition.
    """

    async def _notify() -> None:
        async with condition:
            condition.notify_all()

    try:
        loop = asyncio.get_running_loop()
        loop.create_task(_notify())
    except RuntimeError:
        pass


class AgentPattern(ABC):
    """
    Abstract interface for agent execution patterns (e.g., React, Plan, Reflect).
    Each pattern must implement the 'run' method.
    """

    @abstractmethod
    async def run(
        self,
        task: str,
        memory: MemoryStore,
        tools: list[Tool],
        context: Optional[AgentContext] = None,
    ) -> dict[str, Any]:
        """
        Execute the pattern with given task, memory, tools, and context.

        Returns:
            dict with at least a 'success' boolean field.
        """


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

    model_config = ConfigDict(extra="allow")  # Allow extra fields for flexibility

    @classmethod
    def get_decision_schema(cls) -> Dict[str, Any]:
        """
        Get manually crafted JSON Schema for first-phase decision.

        This is a simple, provider-agnostic schema that works across
        different LLM providers (OpenAI, Gemini, etc.).

        Returns:
            OpenAI-compatible JSON Schema dict
        """
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "action_decision",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "type": {
                            "type": "string",
                            "enum": ["tool_call", "final_answer"],
                        },
                        "reasoning": {"type": "string"},
                        "answer": {"type": "string"},
                        "success": {"type": "boolean"},
                        "error": {"type": ["string", "null"]},
                    },
                    "required": ["type", "reasoning"],
                },
            },
        }


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
            from ..exceptions import ToolNotFoundError

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
                schema = args_type.model_json_schema()
                return self._clean_json_schema(schema)
        except Exception as e:
            logger.warning(f"Failed to get schema for tool {tool.metadata.name}: {e}")

        return {
            "type": "object",
            "properties": {},
            "required": [],
        }

    def _clean_json_schema(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Clean up Pydantic json schema for LLMs (e.g. remove anyOf with null)."""
        if not isinstance(schema, dict):
            return schema

        cleaned = {}
        for k, v in schema.items():
            if k == "anyOf" and isinstance(v, list):
                # Check if this is an Optional type (anyOf with null)
                non_null_types = [
                    t for t in v if isinstance(t, dict) and t.get("type") != "null"
                ]
                if len(non_null_types) == 1:
                    # Merge the non-null type directly
                    cleaned.update(self._clean_json_schema(non_null_types[0]))
                    continue

            if isinstance(v, dict):
                cleaned[k] = self._clean_json_schema(v)
            elif isinstance(v, list):
                cleaned[k] = [
                    self._clean_json_schema(item) if isinstance(item, dict) else item
                    for item in v
                ]
            else:
                cleaned[k] = v

        return cleaned
