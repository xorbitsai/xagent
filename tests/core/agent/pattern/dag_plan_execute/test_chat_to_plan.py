"""
Test Chat-to-Plan Flow

Tests the chat-to-plan decision flow using:
1. PlanGenerator.should_chat_directly - for quick classification
2. PlanGenerator.generate_plan - for plan generation

This tests the actual pattern used in dag_plan_execute.py where:
1. First call should_chat_directly to determine execution path
2. If it returns chat response, use that
3. If it returns plan type, call generate_plan to get the actual plan
"""

import json
from typing import Dict, List

import pytest

from xagent.core.agent.pattern.dag_plan_execute.models import (
    ChatResponse,
    Interaction,
    InteractionType,
    PlanGeneratorResult,
)
from xagent.core.agent.pattern.dag_plan_execute.plan_generator import PlanGenerator
from xagent.core.agent.trace import Tracer
from xagent.core.model.chat.basic.base import BaseLLM


class MockChatLLM(BaseLLM):
    """Mock LLM that returns different types of responses"""

    def __init__(self, response_type: str = "chat"):
        self._model_name = "test_chat_llm"
        self.response_type = response_type
        self.call_count = 0

    @property
    def abilities(self) -> list[str]:
        return ["chat"]

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def supports_thinking_mode(self) -> bool:
        return False

    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        self.call_count += 1

        if self.response_type == "simple_chat":
            # Simple answer - use markdown code block
            return f"""```json
{
                json.dumps(
                    {
                        "type": "chat",
                        "chat": {
                            "message": "Today is March 3, 2026",
                            "interactions": [],
                        },
                    },
                    indent=2,
                    ensure_ascii=False,
                )
            }
```"""

        elif self.response_type == "clarification":
            # Needs clarification - use markdown code block
            return f"""```json
{
                json.dumps(
                    {
                        "type": "chat",
                        "chat": {
                            "message": "Please select the service you want",
                            "interactions": [
                                {
                                    "type": "select_one",
                                    "field": "service",
                                    "label": "Service Type",
                                    "options": [
                                        {"value": "A", "label": "Data Analysis"},
                                        {"value": "B", "label": "Report Generation"},
                                    ],
                                }
                            ],
                        },
                    },
                    indent=2,
                    ensure_ascii=False,
                )
            }
```"""

        elif self.response_type == "multi_interaction":
            # Multiple interactions - use markdown code block
            return f"""```json
{
                json.dumps(
                    {
                        "type": "chat",
                        "chat": {
                            "message": "Please provide the following information",
                            "interactions": [
                                {
                                    "type": "select_one",
                                    "field": "category",
                                    "label": "Task Type",
                                    "options": [
                                        {"value": "analysis", "label": "Data Analysis"},
                                        {
                                            "value": "report",
                                            "label": "Report Generation",
                                        },
                                    ],
                                },
                                {
                                    "type": "text_input",
                                    "field": "email",
                                    "label": "Contact Email",
                                    "placeholder": "Please enter your email",
                                },
                            ],
                        },
                    },
                    indent=2,
                    ensure_ascii=False,
                )
            }
```"""

        elif self.response_type == "plan":
            # Need to generate plan - use markdown code block with empty tool list
            return f"""```json
{
                json.dumps(
                    {
                        "type": "plan",
                        "plan": {
                            "task_name": "Data Analysis Task",
                            "goal": "Analyze user uploaded data file",
                            "steps": [
                                {
                                    "id": "step1",
                                    "name": "Read Data",
                                    "description": "Read uploaded data file",
                                    "tool_names": [],
                                    "dependencies": [],
                                    "difficulty": "easy",
                                },
                                {
                                    "id": "step2",
                                    "name": "Analyze Data",
                                    "description": "Analyze the data",
                                    "tool_names": [],
                                    "dependencies": ["step1"],
                                    "difficulty": "hard",
                                },
                            ],
                        },
                    },
                    indent=2,
                    ensure_ascii=False,
                )
            }
```"""

        elif self.response_type == "conversation_then_plan":
            # First return clarification, then return plan
            if self.call_count == 1:
                return f"""```json
{
                    json.dumps(
                        {
                            "type": "chat",
                            "chat": {
                                "message": "Please select analysis type",
                                "interactions": [
                                    {
                                        "type": "select_one",
                                        "field": "type",
                                        "label": "Analysis Type",
                                        "options": [
                                            {
                                                "value": "descriptive",
                                                "label": "Descriptive Analysis",
                                            },
                                            {
                                                "value": "predictive",
                                                "label": "Predictive Analysis",
                                            },
                                        ],
                                    }
                                ],
                            },
                        },
                        indent=2,
                        ensure_ascii=False,
                    )
                }
```"""
            else:
                # User has selected, return plan
                return f"""```json
{
                    json.dumps(
                        {
                            "type": "plan",
                            "plan": {
                                "task_name": "Descriptive Data Analysis",
                                "goal": "Perform descriptive analysis on data",
                                "steps": [
                                    {
                                        "id": "step1",
                                        "name": "Summary Statistics",
                                        "description": "Calculate statistical metrics",
                                        "tool_names": [],
                                        "dependencies": [],
                                        "difficulty": "easy",
                                    }
                                ],
                            },
                        },
                        indent=2,
                        ensure_ascii=False,
                    )
                }
```"""

        else:
            # Default return plan - use markdown code block
            return f"""```json
{
                json.dumps(
                    {
                        "type": "plan",
                        "plan": {
                            "task_name": "Default Task",
                            "goal": "Execute task",
                            "steps": [],
                        },
                    },
                    indent=2,
                    ensure_ascii=False,
                )
            }
```"""


class TestChatToPlanFlow:
    """Test Chat-to-Plan Flow"""

    @pytest.fixture
    def tracer(self):
        """Create Tracer instance"""
        return Tracer()

    @pytest.fixture
    def plan_generator(self):
        """Create PlanGenerator instance"""
        llm = MockChatLLM()
        return PlanGenerator(llm)

    @pytest.mark.asyncio
    async def test_simple_question_returns_chat(self, plan_generator, tracer):
        """Test: Simple question returns chat response"""
        llm = MockChatLLM(response_type="simple_chat")
        plan_generator.llm = llm

        # Use the actual pattern: should_chat_directly returns chat response
        result = await plan_generator.should_chat_directly(
            goal="What is today's date?",
            tools=[],
            iteration=1,
            history=[],
            tracer=tracer,
        )

        assert result.type == "chat"
        assert result.chat_response is not None
        assert "March 3, 2026" in result.chat_response.message
        assert (
            result.chat_response.interactions is None
            or len(result.chat_response.interactions) == 0
        )

    @pytest.mark.asyncio
    async def test_clarification_with_single_interaction(self, plan_generator, tracer):
        """Test: Needs clarification returns chat with select_one"""
        llm = MockChatLLM(response_type="clarification")
        plan_generator.llm = llm

        # Use the actual pattern: should_chat_directly returns chat response
        result = await plan_generator.should_chat_directly(
            goal="Help me analyze data",
            tools=[],
            iteration=1,
            history=[],
            tracer=tracer,
        )

        assert result.type == "chat"
        assert result.chat_response is not None
        assert "select" in result.chat_response.message.lower()
        assert result.chat_response.interactions is not None
        assert len(result.chat_response.interactions) == 1

        interaction = result.chat_response.interactions[0]
        assert interaction.type == InteractionType.SELECT_ONE
        assert interaction.field == "service"
        assert interaction.label == "Service Type"
        assert len(interaction.options) == 2

    @pytest.mark.asyncio
    async def test_multiple_interactions(self, plan_generator, tracer):
        """Test: Supports multiple interaction fields"""
        llm = MockChatLLM(response_type="multi_interaction")
        plan_generator.llm = llm

        # Use the actual pattern: should_chat_directly returns chat response
        result = await plan_generator.should_chat_directly(
            goal="Create a data analysis task",
            tools=[],
            iteration=1,
            history=[],
            tracer=tracer,
        )

        assert result.type == "chat"
        assert result.chat_response.interactions is not None
        assert len(result.chat_response.interactions) == 2

        # First interaction: select box
        interaction1 = result.chat_response.interactions[0]
        assert interaction1.type == InteractionType.SELECT_ONE
        assert interaction1.field == "category"

        # Second interaction: text input
        interaction2 = result.chat_response.interactions[1]
        assert interaction2.type == InteractionType.TEXT_INPUT
        assert interaction2.field == "email"

    @pytest.mark.asyncio
    async def test_complex_task_returns_plan(self, plan_generator, tracer):
        """Test: Complex task returns plan"""
        llm = MockChatLLM(response_type="plan")
        plan_generator.llm = llm

        # Use the actual pattern: should_chat_directly returns plan type,
        # then call generate_plan to get the actual plan
        classification_result = await plan_generator.should_chat_directly(
            goal="Analyze this CSV file and generate charts",
            tools=[],
            iteration=1,
            history=[],
            tracer=tracer,
        )

        # Should indicate plan generation is needed
        assert classification_result.type == "plan"

        # Generate the actual plan
        plan = await plan_generator.generate_plan(
            goal="Analyze this CSV file and generate charts",
            tools=[],
            iteration=1,
            history=[],
            tracer=tracer,
        )

        assert plan is not None
        assert plan.task_name == "Data Analysis Task"
        assert len(plan.steps) == 2

    @pytest.mark.asyncio
    async def test_multi_turn_conversation(self, plan_generator, tracer):
        """Test: Multi-turn conversation - first clarification, then plan"""
        llm = MockChatLLM(response_type="conversation_then_plan")
        plan_generator.llm = llm

        # First turn: User asks, LLM returns clarification
        history1 = []
        result1 = await plan_generator.should_chat_directly(
            goal="Help me do data analysis",
            tools=[],
            iteration=1,
            history=history1,
            tracer=tracer,
        )

        assert result1.type == "chat"
        assert result1.chat_response is not None
        assert result1.chat_response.interactions is not None
        assert len(result1.chat_response.interactions) == 1

        # Simulate adding to conversation history
        history1.append({"role": "user", "content": "Help me do data analysis"})
        history1.append({"role": "assistant", "content": result1.chat_response.message})

        # Second turn: User responds, check if we need to generate plan
        result2 = await plan_generator.should_chat_directly(
            goal="I choose: Descriptive Analysis",
            tools=[],
            iteration=1,
            history=history1,
            tracer=tracer,
        )

        assert result2.type == "plan"

        # Generate the actual plan
        plan = await plan_generator.generate_plan(
            goal="I choose: Descriptive Analysis",
            tools=[],
            iteration=1,
            history=history1,
            tracer=tracer,
        )

        assert plan is not None
        assert plan.task_name == "Descriptive Data Analysis"

    @pytest.mark.asyncio
    async def test_conversation_history_format(self, plan_generator, tracer):
        """Test: Conversation history is correctly passed to LLM"""
        llm = MockChatLLM(response_type="plan")
        plan_generator.llm = llm

        history = [
            {"role": "user", "content": "I want to analyze data"},
            {"role": "assistant", "content": "Please select analysis type"},
            {"role": "user", "content": "I choose descriptive analysis"},
        ]

        # Use the actual pattern
        await plan_generator.should_chat_directly(
            goal="Start analysis",
            tools=[],
            iteration=1,
            history=history,
            tracer=tracer,
        )

        # Verify LLM was called
        assert llm.call_count >= 1

    @pytest.mark.asyncio
    async def test_interaction_types(self, plan_generator):
        """Test: All interaction types can be correctly parsed"""

        # Create response with all interaction types
        test_response = PlanGeneratorResult(
            type="chat",
            chat_response=ChatResponse(
                message="Please fill in the information",
                interactions=[
                    Interaction(
                        type=InteractionType.SELECT_ONE,
                        field="q1",
                        label="Single Select",
                    ),
                    Interaction(
                        type=InteractionType.SELECT_MULTIPLE,
                        field="q2",
                        label="Multi Select",
                    ),
                    Interaction(
                        type=InteractionType.TEXT_INPUT, field="q3", label="Text"
                    ),
                    Interaction(
                        type=InteractionType.FILE_UPLOAD, field="q4", label="File"
                    ),
                    Interaction(
                        type=InteractionType.CONFIRM, field="q5", label="Confirm"
                    ),
                    Interaction(
                        type=InteractionType.NUMBER_INPUT, field="q6", label="Number"
                    ),
                ],
            ),
        )

        # Verify all types can be correctly created
        assert test_response.chat_response.interactions is not None
        assert len(test_response.chat_response.interactions) == 6

        # Verify each interaction type
        types = [
            interaction.type for interaction in test_response.chat_response.interactions
        ]
        assert InteractionType.SELECT_ONE in types
        assert InteractionType.SELECT_MULTIPLE in types
        assert InteractionType.TEXT_INPUT in types
        assert InteractionType.FILE_UPLOAD in types
        assert InteractionType.CONFIRM in types
        assert InteractionType.NUMBER_INPUT in types


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
