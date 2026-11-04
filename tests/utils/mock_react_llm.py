"""
Mock LLM implementation specifically for ReAct pattern testing.

This module provides a MockReactLLM class that simulates LLM responses
for ReAct pattern testing. It generates structured action responses
that match the ReAct pattern's expected format.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from xagent.core.model.chat.basic.base import BaseLLM

logger = logging.getLogger(__name__)


class MockReactLLM(BaseLLM):
    """
    Mock implementation of BaseLLM for testing ReAct patterns.

    Provides structured action responses that match the ReAct pattern's
    expected JSON format for tool calls and final answers.
    """

    def __init__(self) -> None:
        """Initialize the mock ReAct LLM."""
        self.call_count = 0
        self._model_name = "mock_react_llm"

    @property
    def abilities(self) -> List[str]:
        return ["chat"]

    @property
    def model_name(self) -> str:
        """Get the model name/identifier."""
        return self._model_name

    @property
    def supports_thinking_mode(self) -> bool:
        """Mock LLM doesn't support thinking mode"""
        return False

    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: List[Dict[str, Any]] | None = None,
        tool_choice: str | Dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> str | Dict[str, Any]:
        """
        Mock chat completion that generates appropriate responses for different patterns.

        Detects the type of request and returns appropriate responses:
        - For ReAct pattern: structured actions
        - For DAG pattern: execution plans
        - Default: simple responses
        """
        self.call_count += 1
        logger.info(
            f"MockReactLLM chat call #{self.call_count} with {len(tools) if tools else 0} tools"
        )

        # Extract the user message content
        user_message = ""
        for message in messages:
            if message.get("role") == "user":
                user_message = message.get("content", "")
                break

        # Check if this is a system prompt for ReAct
        system_prompt = ""
        for message in messages:
            if message.get("role") == "system":
                system_prompt = message.get("content", "")
                break

        # Determine response type based on message content
        if (
            "execution plan" in user_message.lower()
            or "json array" in user_message.lower()
            or "DAG" in system_prompt
            or "additional steps" in user_message.lower()
            or "extend existing plan" in system_prompt.lower()
        ):
            # DAG plan generation response
            return self._generate_mock_plan(user_message, tools)
        elif "goal" in user_message.lower() and "achieved" in user_message.lower():
            # Goal achievement check response for DAG
            return self._generate_mock_goal_check(user_message)
        elif "final_answer" in system_prompt or "analysis" in user_message.lower():
            # Analysis/final answer response for ReAct
            return self._generate_final_answer_response(user_message)
        elif tools and tool_choice:
            # Tool call response for ReAct
            return self._generate_tool_call_response(user_message, tools)
        else:
            # Default to simple response
            return "I'm a mock LLM. I can help with various tasks."

    def _generate_final_answer_response(self, user_message: str) -> str:
        """Generate a structured final answer response."""
        logger.info("Generating mock final answer response")

        # Extract task from message
        task = user_message.strip()
        if len(task) > 100:
            task = task[:100] + "..."

        response = {
            "type": "final_answer",
            "reasoning": f"I have analyzed the task '{task}' and can provide a comprehensive answer",
            "answer": f"Based on my analysis, the task '{task}' has been completed successfully. Here is the result: Mock analysis completed for demonstration purposes.",
        }

        return json.dumps(response)

    def _generate_tool_call_response(
        self, user_message: str, tools: List[Dict[str, Any]]
    ) -> str:
        """Generate a structured tool call response."""
        logger.info("Generating mock tool call response")

        # Get available tool names
        available_tools = []
        for tool in tools:
            if tool.get("type") == "function" and "function" in tool:
                available_tools.append(tool["function"]["name"])

        # If no tools provided, use default tool names
        if not available_tools:
            available_tools = ["calculator", "search", "analyze"]

        # Choose a tool based on the task
        task = user_message.lower()
        tool_name = available_tools[0]  # Default to first tool

        if "calculate" in task or "math" in task:
            tool_name = (
                "calculator" if "calculator" in available_tools else available_tools[0]
            )
            tool_args = {"expression": "2 + 2"}
        elif "search" in task or "find" in task:
            tool_name = "search" if "search" in available_tools else available_tools[0]
            tool_args = {"query": "test query"}
        else:
            # Default tool call
            tool_args = {"input": "test input"}

        response = {
            "type": "tool_call",
            "reasoning": f"I need to use the {tool_name} tool to help complete this task",
            "tool_name": tool_name,
            "tool_args": tool_args,
        }

        return json.dumps(response)

    def _generate_mock_plan(
        self, user_message: str, tools: List[Dict[str, Any]] | None = None
    ) -> str:
        """Generate a mock execution plan for DAG pattern."""
        logger.info("Generating mock execution plan for DAG pattern")

        # Extract goal from the message
        goal_start = user_message.find("Goal: ")
        goal_end = user_message.find("\n", goal_start) if goal_start != -1 else -1
        goal = (
            "analyze data"
            if goal_start == -1
            else user_message[goal_start + 6 : goal_end].strip()
        )

        # Extract available tool names from tools parameter
        available_tools = []
        if tools:
            for tool in tools:
                if tool.get("type") == "function" and "function" in tool:
                    available_tools.append(tool["function"]["name"])

        # If no tools provided, use simple tool names that work with analysis
        if not available_tools:
            available_tools = [
                "calculator",
            ]

        # Generate a simple DAG plan with analysis steps (no tool requirements)
        plan_steps = [
            {
                "id": "step_1",
                "name": "Task Analysis",
                "description": "Analyze the task requirements",
                "tool_names": [],
                "dependencies": [],
                "difficulty": "hard",
            },
            {
                "id": "step_2",
                "name": "Execution",
                "description": "Execute the main task",
                "tool_names": [],
                "dependencies": ["step_1"],
                "difficulty": "hard",
            },
            {
                "id": "step_3",
                "name": "Result Analysis",
                "description": "Analyze and summarize results",
                "tool_names": [],
                "dependencies": ["step_2"],
                "difficulty": "easy",
            },
        ]

        # Return plan in new dictionary format
        plan_data = {"plan": {"goal": goal, "steps": plan_steps}}

        return json.dumps(plan_data, indent=2)

    def get_call_count(self) -> int:
        """Get the number of chat calls made to this mock LLM."""
        return self.call_count

    def _generate_mock_goal_check(self, user_message: str) -> str:
        """Generate a mock goal achievement check response."""
        logger.info("Generating mock goal achievement check")

        # For testing purposes, always return achieved=True
        return json.dumps(
            {
                "achieved": True,
                "reason": "Task execution completed and meets the specified goal criteria",
            }
        )

    def reset_call_count(self) -> None:
        """Reset the call counter."""
        self.call_count = 0
