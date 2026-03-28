import json
from typing import Any

import pytest

from xagent.core.agent.context import AgentContext
from xagent.core.agent.pattern.react import ReActPattern
from xagent.core.memory.base import MemoryResponse, MemoryStore
from xagent.core.model.chat.basic.base import BaseLLM
from xagent.core.model.chat.types import ChunkType, StreamChunk
from xagent.core.tools.adapters.vibe import Tool, ToolMetadata


class MockReActLLM(BaseLLM):
    def __init__(self, responses=None):
        self.responses = responses or []
        self.call_count = 0
        self._abilities = ["chat", "tool_calling"]
        self._model_name = "mock_react_llm"

    @property
    def supports_thinking_mode(self) -> bool:
        """Mock LLM doesn't support thinking mode"""
        return False

    @property
    def abilities(self) -> list[str]:
        """Get the list of abilities supported by this Mock LLM implementation."""
        return self._abilities

    @property
    def model_name(self) -> str:
        """Get the model name/identifier."""
        return self._model_name

    async def chat(self, messages: list[dict[str, str]], **kwargs) -> str:
        if self.call_count < len(self.responses):
            response = self.responses[self.call_count]
            self.call_count += 1
            return response

        # Default final answer matching new Action schema
        return '{"type": "final_answer", "reasoning": "Task completed successfully", "answer": "Task completed successfully", "success": true, "error": null}'

    async def stream_chat(self, messages: list[dict[str, str]], **kwargs):
        """Stream chat implementation for testing native tool calling."""
        if self.call_count >= len(self.responses):
            # Default response
            response_json = {
                "type": "final_answer",
                "reasoning": "Task completed successfully",
                "answer": "Task completed successfully",
                "success": True,
                "error": None,
            }
        else:
            # Parse the response
            response_text = self.responses[self.call_count]
            self.call_count += 1
            try:
                response_json = json.loads(response_text)
            except json.JSONDecodeError:
                # If not JSON, return as-is (for invalid JSON tests)
                yield StreamChunk(
                    type=ChunkType.TOKEN, content=response_text, delta=response_text
                )
                yield StreamChunk(type=ChunkType.END, finish_reason="stop")
                return

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
            # Return full JSON as text (for final_answer with all fields)
            full_json = json.dumps(response_json, ensure_ascii=False)
            yield StreamChunk(type=ChunkType.TOKEN, content=full_json, delta=full_json)
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
        expression = args.get("expression", "")
        try:
            # Simple evaluation for testing
            result = eval(expression)  # Note: Only for testing, never use in production
            return {"result": result, "expression": expression}
        except Exception as e:
            return {"error": str(e), "expression": expression}

    def run_json_sync(self, args: dict[str, Any]) -> Any:
        return {"result": 42}

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

    def get_stats(self):
        return {}

    def list_all(self, limit: int = 100, offset: int = 0):
        return []


@pytest.mark.asyncio
async def test_react_basic_execution():
    """Test basic ReAct pattern execution"""
    responses = [
        # First call: return action type
        '{"type": "tool_call", "reasoning": "I need to calculate 2+2"}',
        # Second call: return native tool call
        '{"type": "tool_call", "reasoning": "Calling calculator", "tool_name": "calculator", "tool_args": {"expression": "2+2"}}',
        # Third call: return final answer
        '{"type": "final_answer", "reasoning": "The calculation is complete", "answer": "The result is 4", "success": true, "error": null}',
    ]

    llm = MockReActLLM(responses)
    memory = DummyMemoryStore()
    tools = [MockCalculatorTool()]
    pattern = ReActPattern(llm, max_iterations=5)

    result = await pattern.run(
        task="Calculate 2+2",
        memory=memory,
        tools=tools,
        context=AgentContext(),
    )

    assert result["success"] is True
    assert result["output"] == "The result is 4"
    assert result["iterations"] == 2  # Still 2 iterations (tool_call + final_answer)
    assert "execution_history" in result


@pytest.mark.asyncio
async def test_react_with_context():
    """Test ReAct pattern with pre-built context"""
    responses = [
        '{"type": "final_answer", "reasoning": "Based on the context, the answer is 42", "answer": "Based on the context, the answer is 42", "success": true, "error": null}'
    ]

    llm = MockReActLLM(responses)
    tools = [MockCalculatorTool()]
    pattern = ReActPattern(llm, max_iterations=3)

    # Set step context as required
    pattern.set_step_context(step_id="test_step_1", step_name="test_step")

    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "What is the ultimate answer?"},
        {"role": "assistant", "content": "Let me think about this..."},
    ]

    result = await pattern.run_with_context(
        messages=messages,
        tools=tools,
    )

    assert result["success"] is True
    assert result["output"] == "Based on the context, the answer is 42"
    assert "execution_history" in result


@pytest.mark.asyncio
async def test_react_tool_execution():
    """Test ReAct pattern with tool execution"""
    responses = [
        # First call: return action type
        '{"type": "tool_call", "reasoning": "I need to calculate something"}',
        # Second call: return native tool call
        '{"type": "tool_call", "reasoning": "Calling calculator", "tool_name": "calculator", "tool_args": {"expression": "10*5"}}',
        # Third call: return final answer
        '{"type": "final_answer", "reasoning": "The calculation is complete", "answer": "The calculation result is 50", "success": true, "error": null}',
    ]

    llm = MockReActLLM(responses)
    memory = DummyMemoryStore()
    tools = [MockCalculatorTool()]
    pattern = ReActPattern(llm, max_iterations=5)

    result = await pattern.run(
        task="Calculate 10 times 5",
        memory=memory,
        tools=tools,
        context=AgentContext(),
    )

    assert result["success"] is True
    assert result["output"] == "The calculation result is 50"
    assert result["pattern"] == "react"


@pytest.mark.asyncio
async def test_react_max_iterations():
    """Test ReAct pattern hitting max iterations"""
    # Return tool calls that never lead to final answer
    # Each iteration now requires 2 calls: action type + native tool call
    responses = [
        # Iteration 1, call 1: action type
        '{"type": "tool_call", "reasoning": "I need to calculate something"}',
        # Iteration 1, call 2: native tool call
        '{"type": "tool_call", "reasoning": "Calling calculator", "tool_name": "calculator", "tool_args": {"expression": "1+1"}}',
        # Iteration 2, call 1: action type
        '{"type": "tool_call", "reasoning": "I need to calculate more"}',
        # Iteration 2, call 2: native tool call
        '{"type": "tool_call", "reasoning": "Still calculating", "tool_name": "calculator", "tool_args": {"expression": "2+2"}}',
    ]

    llm = MockReActLLM(responses)
    memory = DummyMemoryStore()
    tools = [MockCalculatorTool()]
    pattern = ReActPattern(llm, max_iterations=2)  # Low max to trigger limit

    with pytest.raises(Exception):  # Should raise MaxIterationsError
        await pattern.run(
            task="Keep calculating without final answer",
            memory=memory,
            tools=tools,
            context=AgentContext(),
        )


@pytest.mark.asyncio
async def test_react_invalid_json():
    """Test ReAct pattern with invalid JSON response"""
    # In native tool calling mode, text responses are treated as final_answer
    responses = [
        "invalid json response",  # Treated as direct text response -> final_answer
    ]

    llm = MockReActLLM(responses)
    memory = DummyMemoryStore()
    tools = [MockCalculatorTool()]
    pattern = ReActPattern(llm, max_iterations=3)

    result = await pattern.run(
        task="Test invalid response",
        memory=memory,
        tools=tools,
        context=AgentContext(),
    )

    # Should complete with the text response as final answer
    assert result["success"] is True
    assert result["output"] == "invalid json response"


@pytest.mark.asyncio
async def test_react_tool_not_found():
    """Test ReAct pattern with non-existent tool"""
    responses = [
        # First call: action type
        '{"type": "tool_call", "reasoning": "Trying to use non-existent tool"}',
        # Second call: native tool call with non-existent tool
        '{"type": "tool_call", "reasoning": "Calling nonexistent", "tool_name": "nonexistent", "tool_args": {}}',
        # Third call: final answer after tool failure
        '{"type": "final_answer", "reasoning": "Could not complete task due to missing tool", "answer": "Could not complete task due to missing tool"}',
    ]

    llm = MockReActLLM(responses)
    memory = DummyMemoryStore()
    tools = [MockCalculatorTool()]
    pattern = ReActPattern(llm, max_iterations=3)

    # With backward compatibility, missing tool is handled gracefully
    result = await pattern.run(
        task="Use non-existent tool",
        memory=memory,
        tools=tools,
        context=AgentContext(),
    )

    # Should complete successfully with error message in output
    assert result["success"] is True
    assert "Could not complete task due to missing tool" in result["output"]


@pytest.mark.asyncio
async def test_react_none_response():
    """Test ReAct pattern with None response from LLM - should retry and eventually complete"""
    responses = [None, None, None]  # LLM returns None multiple times

    llm = MockReActLLM(responses)
    memory = DummyMemoryStore()
    tools = [MockCalculatorTool()]
    pattern = ReActPattern(llm, max_iterations=3)

    # With new retry logic, None triggers retries
    # After exhausting responses, default final answer is used
    result = await pattern.run(
        task="Test None response",
        memory=memory,
        tools=tools,
        context=AgentContext(),
    )

    # Should complete successfully after retries using default final answer
    assert result["success"] is True
    assert result["output"] == "Task completed successfully"


@pytest.mark.asyncio
async def test_react_self_reflection():
    """Test ReAct pattern with self-reflection on failure"""
    responses = [
        # Iteration 1, call 1: action type
        '{"type": "tool_call", "reasoning": "Trying calculator"}',
        # Iteration 1, call 2: native tool call (will fail)
        '{"type": "tool_call", "reasoning": "Calling calculator with invalid expr", "tool_name": "calculator", "tool_args": {"expression": "invalid expression"}}',
        # Iteration 2, call 1: action type
        '{"type": "tool_call", "reasoning": "The previous action failed, let me try a different approach"}',
        # Iteration 2, call 2: native tool call
        '{"type": "tool_call", "reasoning": "Calling calculator with valid expr", "tool_name": "calculator", "tool_args": {"expression": "2+2"}}',
        # Final answer
        '{"type": "final_answer", "reasoning": "After retrying, the answer is 4", "answer": "After retrying, the answer is 4"}',
    ]

    llm = MockReActLLM(responses)
    memory = DummyMemoryStore()
    tools = [MockCalculatorTool()]
    pattern = ReActPattern(llm, max_iterations=5)

    result = await pattern.run(
        task="Calculate something with reflection",
        memory=memory,
        tools=tools,
        context=AgentContext(),
    )

    assert result["success"] is True
    assert result["output"] == "After retrying, the answer is 4"


@pytest.mark.asyncio
async def test_react_analysis_step():
    """Test ReAct pattern with analysis step (no tools)"""
    responses = [
        '{"type": "final_answer", "reasoning": "Based on the provided context, I can analyze this task", "answer": "This is an analysis result that synthesizes the information without using any tools."}',
    ]

    llm = MockReActLLM(responses)
    memory = DummyMemoryStore()
    tools = []  # No tools - this is an analysis step
    pattern = ReActPattern(llm, max_iterations=3)

    result = await pattern.run(
        task="Analyze the following information and provide a summary",
        memory=memory,
        tools=tools,
        context=AgentContext(),
    )

    # Should complete successfully with direct analysis
    assert result["success"] is True
    assert (
        result["output"]
        == "This is an analysis result that synthesizes the information without using any tools."
    )
    assert result["iterations"] == 1  # Should complete in one iteration for analysis


@pytest.mark.asyncio
async def test_react_analysis_step_with_context():
    """Test ReAct pattern with analysis step using context builder"""
    responses = [
        '{"type": "final_answer", "reasoning": "Based on the context from previous steps, I can provide a comprehensive analysis", "answer": "The analysis shows that the previous calculations were successful and the results are consistent."}',
    ]

    llm = MockReActLLM(responses)
    tools = []  # No tools - this is an analysis step
    pattern = ReActPattern(llm, max_iterations=3)

    # Set step context as required
    pattern.set_step_context(step_id="test_analysis_step", step_name="analysis_step")

    # Test with context messages (like from DAG plan execute)
    context_messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that analyzes results.",
        },
        {"role": "user", "content": "Previous step results: Calculation result was 42"},
        {
            "role": "user",
            "content": "Analyze the following information and provide a summary",
        },
    ]

    result = await pattern.run_with_context(
        messages=context_messages,
        tools=tools,
        max_iterations=3,
    )

    # Should complete successfully with contextual analysis
    assert result["success"] is True
    assert (
        "The analysis shows that the previous calculations were successful"
        in result["output"]
    )
    assert result["iterations"] == 1  # Should complete in one iteration for analysis


@pytest.mark.asyncio
async def test_react_failure_detection():
    """Test ReAct pattern failure detection when final_answer indicates failure"""
    responses = [
        '{"type": "final_answer", "reasoning": "The task cannot be completed", "answer": "I was unable to complete the task", "success": false, "error": "Required tool not available"}',
    ]

    llm = MockReActLLM(responses)
    memory = DummyMemoryStore()
    tools = []  # No tools available
    pattern = ReActPattern(llm, max_iterations=3)

    # With new behavior: success: false in final_answer returns failure directly
    # No retry occurs, LLM's assessment is respected
    result = await pattern.run(
        task="Complete a task that will fail",
        memory=memory,
        tools=tools,
        context=AgentContext(),
    )

    # Should return the failure result as provided by LLM
    assert result["success"] is False
    assert "unable to complete the task" in result["output"].lower()


@pytest.mark.asyncio
async def test_react_failure_detection_with_context():
    """Test ReAct pattern with success:false returns failure directly"""
    # LLM returns success:false, which should be returned as-is
    # No retry occurs, LLM's assessment is respected
    responses = [
        '{"type": "final_answer", "reasoning": "Cannot proceed due to missing dependencies", "answer": "TASK FAILED: Missing required data from previous steps", "success": false, "error": "Missing required data from previous steps"}',
    ]

    llm = MockReActLLM(responses)
    tools = []  # No tools available
    pattern = ReActPattern(llm, max_iterations=3)

    # Set step context as required
    pattern.set_step_context(step_id="test_failure_step", step_name="failure_step")

    # Test with context messages
    context_messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that analyzes results.",
        },
        {"role": "user", "content": "Complete a task that will fail"},
    ]

    # With new behavior: failure is returned directly, no retry
    result = await pattern.run_with_context(
        messages=context_messages,
        tools=tools,
        max_iterations=3,
    )

    # Should return the failure result as provided by LLM
    assert result["success"] is False
    assert "TASK FAILED" in result["output"]
    assert "Missing required data from previous steps" in result["output"]


@pytest.mark.asyncio
async def test_react_truncated_json():
    """Test ReAct pattern with truncated JSON response in native tool calling mode"""
    # Truncated JSON (missing closing brace)
    # json_repair can repair this by adding the missing brace, so it becomes valid tool_call
    # This triggers a second call which returns the default final_answer
    truncated_json = '{"type": "tool_call", "reasoning": "I need to calculate", "tool_name": "calculator", "tool_args": {"expression": "2+2"'

    responses = [truncated_json]

    llm = MockReActLLM(responses)
    memory = DummyMemoryStore()
    tools = [MockCalculatorTool()]
    pattern = ReActPattern(llm, max_iterations=3)

    result = await pattern.run(
        task="Test truncated JSON",
        memory=memory,
        tools=tools,
        context=AgentContext(),
    )

    # json_repair repairs the truncated JSON to valid tool_call
    # which triggers second call that returns default final_answer
    assert result["success"] is True
    assert result["output"] == "Task completed successfully"
    assert "execution_history" in result


@pytest.mark.asyncio
async def test_react_multiple_json_objects():
    """Test ReAct pattern with multiple JSON objects - should select first one"""

    # Multiple JSON objects concatenated - gpt-5.4 behavior in streaming mode
    # First JSON is tool_call, second is something else (code), third is final_answer
    multiple_jsons = '{"type": "tool_call", "reasoning": "Need to calculate"}{"code":"import random","capture_output":true}{"type": "final_answer", "reasoning": "Done", "answer": "Result"}'

    # Override stream_chat to return multiple JSONs
    class AlwaysMultipleJSONsLLM(MockReActLLM):
        async def stream_chat(self, messages, **kwargs):
            # Always return multiple JSONs
            yield StreamChunk(
                type=ChunkType.TOKEN, content=multiple_jsons, delta=multiple_jsons
            )
            yield StreamChunk(type=ChunkType.END, finish_reason="stop")

    llm = AlwaysMultipleJSONsLLM([])
    memory = DummyMemoryStore()
    tools = [MockCalculatorTool()]
    pattern = ReActPattern(
        llm, max_iterations=1
    )  # Only 1 iteration to avoid retry loop

    # Should fail because first JSON (tool_call) doesn't have tool_name/tool_args
    # This will cause second call to fail with "LLM did not invoke native tool calling"
    # After max_iterations is reached, it should raise MaxIterationsError
    from xagent.core.agent.exceptions import MaxIterationsError

    with pytest.raises(MaxIterationsError):
        await pattern.run(
            task="Test multiple JSONs",
            memory=memory,
            tools=tools,
            context=AgentContext(),
        )


@pytest.mark.asyncio
async def test_react_successful_final_answer():
    """Test ReAct pattern successful final_answer with explicit success: true"""
    responses = [
        '{"type": "final_answer", "reasoning": "Task completed successfully", "answer": "The task has been completed successfully", "success": true, "error": null}',
    ]

    llm = MockReActLLM(responses)
    memory = DummyMemoryStore()
    tools = []
    pattern = ReActPattern(llm, max_iterations=3)

    result = await pattern.run(
        task="Complete a simple task",
        memory=memory,
        tools=tools,
        context=AgentContext(),
    )

    # Should complete successfully
    assert result["success"] is True
    assert result["output"] == "The task has been completed successfully"
    assert result["iterations"] == 1


if __name__ == "__main__":
    pytest.main([__file__])
