"""Test nested agent architecture using AgentTool"""

import json
from typing import Any, List

import pytest

from xagent.core.agent.agent import Agent
from xagent.core.agent.pattern.dag_plan_execute import DAGPlanExecutePattern
from xagent.core.agent.pattern.react import ReActPattern
from xagent.core.agent.tools.agent_tool import AgentTool, CompactMode
from xagent.core.memory.base import MemoryResponse, MemoryStore
from xagent.core.memory.in_memory import InMemoryMemoryStore
from xagent.core.model.chat.basic.base import BaseLLM
from xagent.core.model.chat.types import ChunkType, StreamChunk
from xagent.core.tools.adapters.vibe import Tool, ToolMetadata
from xagent.core.workspace import TaskWorkspace


class MockLLM(BaseLLM):
    def __init__(self, responses=None):
        self.responses = responses or []
        self.call_count = 0
        self._model_name = "mock_llm"

    @property
    def abilities(self) -> List[str]:
        return ["chat", "tool_calling"]

    @property
    def model_name(self) -> str:
        """Get the model name/identifier."""
        return self._model_name

    @property
    def supports_thinking_mode(self) -> bool:
        """Mock LLM doesn't support thinking mode"""
        return False

    async def chat(self, messages: list[dict[str, str]], **kwargs) -> str:
        if self.call_count < len(self.responses):
            response = self.responses[self.call_count]
            self.call_count += 1
            return response

        # Default response
        return '{"type": "final_answer", "content": "Task completed by sub-agent", "answer": "Task completed by sub-agent", "reasoning": "The task has been completed"}'

    async def stream_chat(self, messages: list[dict[str, str]], **kwargs):
        """Stream chat implementation for testing native tool calling."""
        if self.call_count >= len(self.responses):
            # Default response
            response_json = {
                "type": "final_answer",
                "answer": "Task completed by sub-agent",
                "reasoning": "The task has been completed",
            }
        else:
            # Parse the response
            response_text = self.responses[self.call_count]
            self.call_count += 1
            try:
                response_json = json.loads(response_text)
            except json.JSONDecodeError:
                # If not JSON, treat as final answer
                response_json = {
                    "type": "final_answer",
                    "reasoning": "Response received",
                    "answer": response_text,
                }

        # Check if this is a tool call
        if response_json.get("type") == "tool_call":
            # Return native tool call format
            tool_name = response_json.get("tool_name", "")
            tool_args = response_json.get("tool_args", {})

            yield StreamChunk(
                type=ChunkType.TOOL_CALL,
                content="",
                delta="",
                tool_calls=[
                    {
                        "function": {
                            "name": tool_name,
                            "arguments": json.dumps(tool_args),
                        }
                    }
                ],
            )
            yield StreamChunk(type=ChunkType.END, finish_reason="tool_calls")
        else:
            # Return text content (final_answer)
            answer = response_json.get(
                "answer", response_json.get("content", "Task completed")
            )
            yield StreamChunk(type=ChunkType.TOKEN, content=answer, delta=answer)
            yield StreamChunk(type=ChunkType.END, finish_reason="stop")


class MockCalculatorTool(Tool):
    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(name="calculator", description="Simple calculator")

    def args_type(self):
        return dict

    def return_type(self):
        return dict

    def state_type(self):
        return None

    def is_async(self):
        return True

    async def run_json_async(self, args: dict[str, Any]) -> Any:
        expression = args.get("expression", "2+2")
        try:
            result = eval(expression)  # Only for testing
            return {"result": result, "expression": expression}
        except Exception as e:
            return {"error": str(e), "expression": expression}

    def run_json_sync(self, args: dict[str, Any]) -> Any:
        return {"result": 4}

    async def save_state_json(self):
        return {}

    async def load_state_json(self, state: dict[str, Any]):
        pass

    def return_value_as_string(self, value: Any) -> str:
        return str(value)


class DummyMemoryStore(MemoryStore):
    def add(self, note):
        return MemoryResponse(success=True)

    def get(self, note_id: str):
        return MemoryResponse(success=True)

    def update(self, note):
        return MemoryResponse(success=True)

    def delete(self, note_id: str):
        return MemoryResponse(success=True)

    def search(self, query: str, k: int = 5, filters=None):
        return []

    def clear(self):
        pass


@pytest.mark.asyncio
async def test_agent_tool_basic():
    """Test basic AgentTool functionality"""
    # Create a specialized ReAct agent for calculations
    calc_responses = [
        '{"type": "tool_call", "reasoning": "I need to calculate 5 times 3", "tool_name": "calculator", "tool_args": {"expression": "5*3"}}',
        '{"type": "final_answer", "answer": "15", "reasoning": "The calculation is complete"}',
    ]

    calc_llm = MockLLM(calc_responses)
    calc_pattern = ReActPattern(calc_llm, max_iterations=3)
    calc_memory = InMemoryMemoryStore()
    calc_tools = [MockCalculatorTool()]

    calc_agent = Agent(
        name="CalculatorAgent",
        patterns=[calc_pattern],
        memory=calc_memory,
        tools=calc_tools,
    )

    # Create AgentTool wrapper
    agent_tool = AgentTool(calc_agent, compact_mode=CompactMode.COMPACT)

    # Test the tool
    result = await agent_tool.run_json_async(
        {"task": "Calculate 5 times 3", "context": {"source": "parent_agent"}}
    )

    assert result["success"] is True
    assert result["output"] == "15"
    assert result["agent_name"] == "CalculatorAgent"
    assert result["compact_mode"] == "compact"


@pytest.mark.asyncio
async def test_nested_agents_in_dag(tmp_path):
    """Test nested agents within a DAG execution"""

    # Create a specialized math agent
    math_responses = [
        '{"type": "final_answer", "content": "Math calculation completed: 42", "answer": "42", "reasoning": "The math calculation is complete"}'
    ]
    math_llm = MockLLM(math_responses)
    math_pattern = ReActPattern(math_llm, max_iterations=2)
    math_agent = Agent(
        name="MathAgent",
        patterns=[math_pattern],
        memory=InMemoryMemoryStore(),
        tools=[MockCalculatorTool()],
    )

    # Create agent tool
    math_agent_tool = AgentTool(math_agent, CompactMode.COMPACT)

    # Create a main DAG agent that uses the math agent as a tool
    class MockDAGLLM(BaseLLM):
        def __init__(self):
            self.call_count = 0
            self._model_name = "mock_dag_llm"

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

        async def chat(self, messages: list[dict[str, str]], **kwargs) -> str:
            # Handle JSON mode - return dict instead of string
            if kwargs.get("response_format") == {"type": "json_object"}:
                # Mock LLM response for task analysis in JSON mode
                if any(
                    "task execution analyzer" in msg.get("content", "").lower()
                    for msg in messages
                ):
                    return '{"success": true, "direct_answer": "Math calculation completed successfully with result 42", "file_outputs": [], "confidence": "high", "reasoning": "The math agent completed the calculation successfully."}'
                # Mock LLM response for Chinese task analysis in JSON mode
                elif any("任务分析助手" in msg.get("content", "") for msg in messages):
                    return '{"success": true, "direct_answer": "Math calculation completed successfully with result 42", "file_outputs": [], "confidence": "high", "reasoning": "The math agent completed the calculation successfully."}'

            content = " ".join(msg.get("content", "") for msg in messages).lower()

            if "execution plan" in content or "step-by-step" in content:
                # Return plan in new dictionary format
                return """{
                    "plan": {
                        "goal": "Solve a complex mathematical problem",
                        "steps": [{
                            "id": "step1",
                            "name": "math_calculation",
                            "description": "Perform mathematical calculation using specialist agent",
                            "tool_names": ["agent_MathAgent"],
                            "dependencies": [],
                            "difficulty": "hard"
                        }]
                    }
                }"""
            elif "goal" in content and "achieved" in content:
                # Goal achievement check
                return '{"achieved": true, "reason": "Math agent completed the calculation"}'
            else:
                # ReAct response for step execution
                return (
                    '{"type": "final_answer", "content": "Step completed successfully"}'
                )

    dag_llm = MockDAGLLM()
    dag_pattern = DAGPlanExecutePattern(
        llm=dag_llm,
        max_iterations=1,
        goal_check_enabled=True,
        workspace=TaskWorkspace(id="test_workspace", base_dir=str(tmp_path)),
    )
    dag_memory = InMemoryMemoryStore()
    dag_tools = [math_agent_tool]

    dag_agent = Agent(
        name="MainAgent", patterns=[dag_pattern], memory=dag_memory, tools=dag_tools
    )

    # Execute the nested agent scenario
    runner = dag_agent.get_runner()
    result = await runner.run("Solve a complex mathematical problem")

    assert result["success"] is True
    assert "output" in result


@pytest.mark.asyncio
async def test_agent_tool_full_mode():
    """Test AgentTool with full mode (non-compact)"""
    responses = [
        '{"type": "final_answer", "content": "Full execution completed", "answer": "Full execution completed", "reasoning": "The task has been completed successfully"}'
    ]

    llm = MockLLM(responses)
    pattern = ReActPattern(llm, max_iterations=2)
    agent = Agent(
        name="FullAgent",
        patterns=[pattern],
        memory=InMemoryMemoryStore(),
        tools=[MockCalculatorTool()],
    )

    # Test full mode
    agent_tool = AgentTool(agent, CompactMode.FULL)

    result = await agent_tool.run_json_async({"task": "Test full mode execution"})

    assert result["success"] is True
    assert result["compact_mode"] == "full"
    assert result["agent_name"] == "FullAgent"
    # Full mode should include execution history
    assert "execution_history" in result


@pytest.mark.asyncio
async def test_agent_tool_query_details():
    """Test querying agent details after execution"""
    responses = [
        '{"type": "final_answer", "content": "Detailed task completed", "answer": "Detailed task completed", "reasoning": "The detailed task has been completed"}'
    ]

    llm = MockLLM(responses)
    pattern = ReActPattern(llm, max_iterations=2)
    agent = Agent(
        name="QueryableAgent",
        patterns=[pattern],
        memory=InMemoryMemoryStore(),
        tools=[MockCalculatorTool()],
    )

    agent_tool = AgentTool(agent)

    # Execute first
    await agent_tool.run_json_async({"task": "Complete a task for querying"})

    # Query details
    details = await agent_tool.query_agent_details("What was the result?")
    assert isinstance(details, str)


@pytest.mark.asyncio
async def test_agent_tool_error_handling():
    """Test AgentTool error handling"""
    # Create an agent with no patterns (invalid configuration)
    agent = Agent(
        name="EmptyAgent",
        patterns=[],  # No patterns - should cause error
        memory=InMemoryMemoryStore(),
        tools=[],
    )

    agent_tool = AgentTool(agent)

    with pytest.raises(Exception):  # Should raise AgentConfigurationError
        await agent_tool.run_json_async({"task": "This should fail"})


@pytest.mark.asyncio
async def test_three_level_nesting(tmp_path):
    """Test three levels of nested agents"""

    # Level 1: Basic calculator agent
    calc_responses = [
        '{"type": "final_answer", "content": "Basic calc: 10", "answer": "10", "reasoning": "The basic calculation is complete"}'
    ]
    calc_llm = MockLLM(calc_responses)
    calc_agent = Agent(
        name="Calculator",
        patterns=[ReActPattern(calc_llm, max_iterations=1)],
        memory=InMemoryMemoryStore(),
        tools=[MockCalculatorTool()],
    )

    # Level 2: Math specialist that uses calculator
    math_responses = [
        '{"type": "final_answer", "content": "Advanced math: 100", "answer": "100", "reasoning": "The advanced math calculation is complete"}'
    ]
    math_llm = MockLLM(math_responses)
    math_agent = Agent(
        name="MathSpecialist",
        patterns=[ReActPattern(math_llm, max_iterations=1)],
        memory=InMemoryMemoryStore(),
        tools=[AgentTool(calc_agent)],
    )

    # Level 3: Problem solver that uses math specialist
    class MockSolverLLM(BaseLLM):
        def __init__(self):
            self._model_name = "mock_solver_llm"

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

        async def chat(self, messages: list[dict[str, str]], **kwargs) -> str:
            # Handle JSON mode - return dict instead of string
            if kwargs.get("response_format") == {"type": "json_object"}:
                # Mock LLM response for task analysis in JSON mode
                if any(
                    "task execution analyzer" in msg.get("content", "").lower()
                    for msg in messages
                ):
                    return '{"success": true, "direct_answer": "Problem solved successfully using nested agents", "file_outputs": [], "confidence": "high", "reasoning": "The problem was solved through three levels of nested agents."}'
                # Mock LLM response for Chinese task analysis in JSON mode
                elif any("任务分析助手" in msg.get("content", "") for msg in messages):
                    return '{"success": true, "direct_answer": "Problem solved successfully using nested agents", "file_outputs": [], "confidence": "high", "reasoning": "The problem was solved through three levels of nested agents."}'

            content = " ".join(msg.get("content", "") for msg in messages).lower()

            if "execution plan" in content or "step-by-step" in content:
                return """{
                    "plan": {
                        "goal": "Solve a complex multi-step problem",
                        "steps": [{
                            "id": "step1",
                            "name": "solve_problem",
                            "description": "Use math specialist to solve the problem",
                            "tool_names": ["agent_MathSpecialist"],
                            "dependencies": [],
                            "difficulty": "hard"
                        }]
                    }
                }"""
            elif "goal" in content and "achieved" in content:
                return (
                    '{"achieved": true, "reason": "Problem solved using nested agents"}'
                )
            else:
                return '{"type": "final_answer", "content": "Problem solved"}'

    solver_llm = MockSolverLLM()
    solver_agent = Agent(
        name="ProblemSolver",
        patterns=[
            DAGPlanExecutePattern(
                solver_llm,
                max_iterations=1,
                workspace=TaskWorkspace(id="test_workspace", base_dir=str(tmp_path)),
            )
        ],
        memory=InMemoryMemoryStore(),
        tools=[AgentTool(math_agent)],
    )

    # Execute three-level nesting
    runner = solver_agent.get_runner()
    result = await runner.run("Solve a complex multi-step problem")

    assert result["success"] is True


if __name__ == "__main__":
    pytest.main([__file__])
