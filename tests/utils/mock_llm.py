"""
Mock LLM implementation for testing purposes.

This module provides a MockLLM class that simulates LLM responses for DAG pattern testing.
It generates realistic plan responses and goal achievement checks without requiring a real LLM service.
"""

from __future__ import annotations

import json
import logging
import random
from typing import Any, Dict, List

from xagent.core.model.chat.basic.base import BaseLLM

logger = logging.getLogger(__name__)


class MockLLM(BaseLLM):
    """
    Mock implementation of BaseLLM for testing DAG plan-execute patterns.

    Provides realistic responses for plan generation and goal checking without
    requiring actual LLM API calls.
    """

    def __init__(self) -> None:
        """Initialize the mock LLM."""
        self.call_count = 0
        self._abilities = ["chat", "tool_calling"]
        self._model_name = "mock_llm"

    @property
    def supports_thinking_mode(self) -> bool:
        """Mock LLM doesn't support thinking mode"""
        return False

    @property
    def abilities(self) -> List[str]:
        """Get the list of abilities supported by this Mock LLM implementation."""
        return self._abilities

    @property
    def model_name(self) -> str:
        """Get the model name/identifier."""
        return self._model_name

    def get_call_count(self) -> int:
        """Get the current call count."""
        return self.call_count

    def reset_call_count(self) -> None:
        """Reset the call count to zero."""
        self.call_count = 0

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
        Mock chat completion that generates realistic responses for DAG planning.

        Detects the type of request (planning vs goal checking) and returns
        appropriate mock responses. Now supports tools parameter.
        """
        self.call_count += 1
        logger.info(
            f"MockLLM chat call #{self.call_count} with {len(tools) if tools else 0} tools"
        )

        # Extract the user message content
        user_message = ""
        for message in messages:
            if message.get("role") == "user":
                user_message = message.get("content", "")
                break

        # Determine response type based on message content
        if (
            "execution plan" in user_message.lower()
            or "json array" in user_message.lower()
            or "plan" in user_message.lower()
        ):
            return self._generate_mock_plan(user_message, tools)
        elif "goal" in user_message.lower() and "achieved" in user_message.lower():
            return self._generate_mock_goal_check(user_message)
        else:
            return "I'm a mock LLM. I can help with plan generation and goal checking."

    def _generate_mock_plan(
        self, user_message: str, tools: List[Dict[str, Any]] | None = None
    ) -> str:
        """Generate a mock execution plan based on the user message and available tools."""
        logger.info("Generating mock execution plan")

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

        # If no tools provided, use default tool names
        if not available_tools:
            available_tools = [
                "fetch_data",
                "validate_data",
                "analyze_stats",
                "create_charts",
                "generate_report",
            ]

        # Generate a realistic DAG plan based on common task patterns and available tools
        if "analyze" in goal.lower() or "data" in goal.lower():
            steps = [
                {
                    "id": "step_1",
                    "name": "Data Collection",
                    "description": "Gather required data from sources",
                    "tool_names": [
                        available_tools[0] if len(available_tools) > 0 else "fetch_data"
                    ],
                    "dependencies": [],
                    "difficulty": "hard",
                },
                {
                    "id": "step_2",
                    "name": "Data Validation",
                    "description": "Validate data quality and completeness",
                    "tool_names": [
                        available_tools[1]
                        if len(available_tools) > 1
                        else "validate_data"
                    ],
                    "dependencies": ["step_1"],
                    "difficulty": "hard",
                },
                {
                    "id": "step_3",
                    "name": "Statistical Analysis",
                    "description": "Perform statistical analysis on the data",
                    "tool_names": [
                        available_tools[2]
                        if len(available_tools) > 2
                        else "analyze_stats"
                    ],
                    "dependencies": ["step_2"],
                    "difficulty": "hard",
                },
                {
                    "id": "step_4",
                    "name": "Generate Visualizations",
                    "description": "Create charts and graphs",
                    "tool_names": [
                        available_tools[3]
                        if len(available_tools) > 3
                        else "create_charts"
                    ],
                    "dependencies": ["step_2"],
                    "difficulty": "easy",
                },
                {
                    "id": "step_5",
                    "name": "Generate Report",
                    "description": "Compile analysis results into a report",
                    "tool_names": [
                        available_tools[4]
                        if len(available_tools) > 4
                        else "generate_report"
                    ],
                    "dependencies": ["step_3", "step_4"],
                    "difficulty": "easy",
                },
            ]
        elif "search" in goal.lower() or "research" in goal.lower():
            steps = [
                {
                    "id": "step_1",
                    "name": "Search Query Preparation",
                    "description": "Prepare and optimize search queries",
                    "tool_names": [
                        available_tools[0]
                        if len(available_tools) > 0
                        else "prepare_query"
                    ],
                    "dependencies": [],
                    "difficulty": "easy",
                },
                {
                    "id": "step_2",
                    "name": "Web Search",
                    "description": "Perform web search for information",
                    "tool_names": [
                        available_tools[1] if len(available_tools) > 1 else "web_search"
                    ],
                    "dependencies": ["step_1"],
                    "difficulty": "hard",
                },
                {
                    "id": "step_3",
                    "name": "Information Analysis",
                    "description": "Analyze and organize search results",
                    "tool_names": [
                        available_tools[2]
                        if len(available_tools) > 2
                        else "analyze_info"
                    ],
                    "dependencies": ["step_2"],
                    "difficulty": "hard",
                },
            ]
        else:
            # Default simple plan
            steps = [
                {
                    "id": "step_1",
                    "name": "Task Analysis",
                    "description": "Analyze the task requirements",
                    "tool_names": [
                        available_tools[0]
                        if len(available_tools) > 0
                        else "analyze_task"
                    ],
                    "dependencies": [],
                    "difficulty": "easy",
                },
                {
                    "id": "step_2",
                    "name": "Task Execution",
                    "description": "Execute the main task",
                    "tool_names": [
                        available_tools[1]
                        if len(available_tools) > 1
                        else "execute_task"
                    ],
                    "dependencies": ["step_1"],
                    "difficulty": "hard",
                },
            ]

        # Return the new dictionary format
        plan_data = {"plan": {"goal": goal, "steps": steps}}

        return json.dumps(plan_data, indent=2)

    def _generate_mock_goal_check(self, user_message: str) -> str:
        """Generate a mock goal achievement check response."""
        logger.info("Generating mock goal achievement check")
        logger.info(f"Goal check message: {user_message[:200]}...")

        # Simulate goal achievement based on execution history
        # In a real scenario, this would analyze the actual results

        # Check for actual step failure (not just the word "failed" in stats)
        if "failed: " in user_message.lower() and not (
            "failed: 0" in user_message.lower() or "failed: []" in user_message.lower()
        ):
            return json.dumps(
                {
                    "achieved": False,
                    "reason": "Some execution steps failed, preventing goal completion",
                }
            )

        # If multiple iterations mentioned, be optimistic after iteration 1
        if "iteration" in user_message.lower():
            iteration_count = user_message.lower().count("iteration")
            # For demo purposes, let's be more optimistic after iteration 1
            if iteration_count >= 1:
                achieved = True
            else:
                # Higher chance of achievement in later iterations
                achievement_probability = min(0.3 + (iteration_count * 0.3), 0.9)
                achieved = random.random() < achievement_probability

            if achieved:
                return json.dumps(
                    {
                        "achieved": True,
                        "reason": "All required steps completed successfully and goal criteria met",
                    }
                )
            else:
                return json.dumps(
                    {
                        "achieved": False,
                        "reason": "Goal not yet achieved, additional iterations may be needed",
                    }
                )

        # Default: check for completion keywords
        if any(
            keyword in user_message.lower()
            for keyword in ["completed", "finished", "done", "success"]
        ):
            return json.dumps(
                {
                    "achieved": True,
                    "reason": "All execution steps completed successfully",
                }
            )
        else:
            return json.dumps(
                {
                    "achieved": False,
                    "reason": "Goal not yet achieved, more work needed",
                }
            )
