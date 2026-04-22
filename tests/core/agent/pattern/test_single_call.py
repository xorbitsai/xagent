"""Tests for SingleCall pattern"""

import ast
import json
import operator
from typing import Any

import pytest
from pydantic import BaseModel

from xagent.core.agent.context import AgentContext
from xagent.core.agent.pattern.single_call import SingleCallPattern
from xagent.core.memory.base import MemoryResponse, MemoryStore
from xagent.core.model.chat.basic.base import BaseLLM
from xagent.core.model.chat.types import ChunkType, StreamChunk
from xagent.core.tools.adapters.vibe import Tool, ToolMetadata

# Safe calculator for testing (replaces eval())
_SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}


def _safe_calculate(expression: str) -> float:
    """Safely evaluate a simple arithmetic expression."""
    try:
        node = ast.parse(expression, mode="eval")

        def _eval(node):
            if isinstance(node, ast.Constant):
                return node.value
            elif isinstance(node, ast.BinOp):
                left = _eval(node.left)
                right = _eval(node.right)
                op_type = type(node.op)
                if op_type in _SAFE_OPERATORS:
                    return _SAFE_OPERATORS[op_type](left, right)
                else:
                    raise ValueError(f"Unsupported operator: {op_type}")
            else:
                raise ValueError(f"Unsupported expression: {type(node)}")

        return _eval(node.body)
    except (ValueError, SyntaxError):
        raise ValueError("Invalid expression")


class MockSingleCallLLM(BaseLLM):
    """Mock LLM for testing SingleCall pattern"""

    def __init__(self, response=None, final_answer=None):
        # Default to native tool call format
        self.response = response or {
            "type": "tool_call",
            "tool_calls": [
                {
                    "function": {
                        "name": "test_tool",
                        "arguments": json.dumps({"arg1": "value1"}),
                    }
                }
            ],
        }
        # Final answer to return on second call (after tool execution)
        self.final_answer = (
            final_answer
            or "The tool was executed successfully and the result is: test result"
        )
        self.call_count = 0
        self._abilities = ["chat", "tool_calling"]
        self._model_name = "mock_single_call_llm"

    @property
    def abilities(self) -> list[str]:
        return self._abilities

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def supports_thinking_mode(self) -> bool:
        """Mock LLM doesn't support thinking mode"""
        return False

    async def chat(self, messages: list[dict[str, str]], **kwargs) -> Any:
        self.call_count += 1

        # First call: return tool call
        # Second call: return final answer
        if self.call_count == 1:
            return self.response
        else:
            # Return final answer as text
            return {"type": "text", "content": self.final_answer}

    async def stream_chat(self, messages: list[dict[str, str]], **kwargs):
        """Stream chat implementation"""
        if (
            self.call_count == 0
            and isinstance(self.response, dict)
            and "tool_calls" in self.response
        ):
            # First call: Native tool call format
            yield StreamChunk(
                type=ChunkType.TOOL_CALL, tool_calls=self.response["tool_calls"]
            )
        else:
            # Second call or text format
            content = self.final_answer
            yield StreamChunk(type=ChunkType.TOKEN, content=content, delta=content)
        yield StreamChunk(type=ChunkType.END)


class MockTestTool(Tool):
    """Mock tool for testing"""

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(name="test_tool", description="A test tool")

    def args_type(self):
        class TestArgs(BaseModel):
            arg1: str
            arg2: str = "default"

        return TestArgs

    async def run_json_async(self, args: dict[str, Any]) -> Any:
        return {"result": f"Executed with args: {args}"}


class MockCalculatorTool(Tool):
    """Mock calculator tool for testing"""

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(name="calculator", description="Calculator tool")

    def args_type(self):
        class CalculatorArgs(BaseModel):
            expression: str

        return CalculatorArgs

    async def run_json_async(self, args: dict[str, Any]) -> Any:
        expression = args.get("expression", "")
        try:
            result = _safe_calculate(expression)
            return {"result": result}
        except Exception:
            return {"error": "Invalid expression"}


@pytest.fixture
def mock_llm():
    """Create a mock LLM"""
    return MockSingleCallLLM()


@pytest.fixture
def mock_tools():
    """Create mock tools"""
    return [MockTestTool(), MockCalculatorTool()]


@pytest.fixture
def mock_memory():
    """Create a mock memory store"""

    class MockMemoryStore(MemoryStore):
        async def add(self, *args, **kwargs):
            return MemoryResponse(success=True, id="test_id")

        async def search(self, *args, **kwargs):
            return []

        async def get(self, *args, **kwargs):
            return None

        async def delete(self, *args, **kwargs):
            pass

        async def clear(self, *args, **kwargs):
            pass

        async def update(self, *args, **kwargs):
            pass

        async def get_stats(self, *args, **kwargs):
            return {}

        async def list_all(self, *args, **kwargs):
            return []

        async def add_memory(self, *args, **kwargs):
            return MemoryResponse(success=True, id="test_id")

        async def search_memory(self, *args, **kwargs):
            return []

    return MockMemoryStore()


@pytest.fixture
def mock_context():
    """Create a mock agent context"""
    return AgentContext(task_id="test_task_id")


@pytest.mark.asyncio
async def test_single_call_basic_execution(
    mock_llm, mock_tools, mock_memory, mock_context
):
    """Test basic SingleCall pattern execution with native tool calling"""
    pattern = SingleCallPattern(llm=mock_llm)

    result = await pattern.run(
        task="Use test_tool with arg1=value1",
        memory=mock_memory,
        tools=mock_tools,
        context=mock_context,
    )

    assert result["success"] is True
    assert result["pattern"] == "single_call"
    assert "output" in result
    assert "tool_name" in result
    assert result["tool_name"] == "test_tool"
    assert result["tool_args"]["arg1"] == "value1"


@pytest.mark.asyncio
async def test_single_call_final_answer(mock_memory, mock_context):
    """Test SingleCall pattern when LLM returns final answer directly"""
    mock_llm = MockSingleCallLLM(
        response={"type": "text", "content": "The answer is 42"}
    )

    pattern = SingleCallPattern(llm=mock_llm)

    result = await pattern.run(
        task="What is the meaning of life?",
        memory=mock_memory,
        tools=[],
        context=mock_context,
    )

    assert result["success"] is True
    assert result["pattern"] == "single_call"
    assert result["output"] == "The answer is 42"
    assert result.get("is_final_answer") is True


@pytest.mark.asyncio
async def test_single_call_tool_not_found(mock_memory, mock_context):
    """Test SingleCall pattern with non-existent tool"""
    mock_llm = MockSingleCallLLM(
        response={
            "type": "tool_call",
            "tool_calls": [
                {"function": {"name": "non_existent_tool", "arguments": json.dumps({})}}
            ],
        },
        final_answer="Error: Tool not found",
    )

    pattern = SingleCallPattern(llm=mock_llm)

    result = await pattern.run(
        task="Use non_existent_tool",
        memory=mock_memory,
        tools=[],
        context=mock_context,
    )

    assert result["success"] is False
    # Check that error information is present
    assert "error" in result


@pytest.mark.asyncio
async def test_single_call_requires_llm(mock_tools, mock_memory, mock_context):
    """Test that SingleCall pattern raises error without LLM"""
    pattern = SingleCallPattern(llm=None)

    result = await pattern.run(
        task="Use calculator to compute 5 * 5",
        memory=mock_memory,
        tools=mock_tools,
        context=mock_context,
    )

    assert result["success"] is False
    assert "error" in result


@pytest.mark.asyncio
async def test_single_call_conversation_history(mock_memory, mock_context):
    """Test SingleCall pattern with conversation history"""
    mock_llm = MockSingleCallLLM(
        response={"type": "text", "content": "I said hello before"}
    )

    pattern = SingleCallPattern(llm=mock_llm)

    # Set conversation history
    history = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]
    pattern.set_conversation_history(history)

    result = await pattern.run(
        task="What did I say?",
        memory=mock_memory,
        tools=[],
        context=mock_context,
    )

    assert result["success"] is True
    # Check that LLM was called with history
    assert mock_llm.call_count == 1


@pytest.mark.asyncio
async def test_single_call_execution_context(mock_memory, mock_context):
    """Test SingleCall pattern with execution context messages"""
    mock_llm = MockSingleCallLLM(
        response={"type": "text", "content": "Got the context"}
    )

    pattern = SingleCallPattern(llm=mock_llm)

    # Set execution context
    context_msgs = [{"role": "system", "content": "You are a helpful assistant"}]
    pattern.set_execution_context_messages(context_msgs)

    result = await pattern.run(
        task="Hello",
        memory=mock_memory,
        tools=[],
        context=mock_context,
    )

    assert result["success"] is True
    assert mock_llm.call_count == 1


@pytest.mark.asyncio
async def test_single_call_with_calculator(mock_tools, mock_memory, mock_context):
    """Test SingleCall pattern with calculator tool"""
    mock_llm = MockSingleCallLLM(
        response={
            "type": "tool_call",
            "tool_calls": [
                {
                    "function": {
                        "name": "calculator",
                        "arguments": json.dumps({"expression": "2 + 2"}),
                    }
                }
            ],
        },
        final_answer="The result of 2 + 2 is 4",
    )

    pattern = SingleCallPattern(llm=mock_llm)

    result = await pattern.run(
        task="Calculate 2 + 2",
        memory=mock_memory,
        tools=mock_tools,
        context=mock_context,
    )

    assert result["success"] is True
    assert result["tool_name"] == "calculator"
    assert "4" in result["output"]


@pytest.mark.asyncio
async def test_single_call_trace_compatibility(
    mock_llm, mock_tools, mock_memory, mock_context
):
    """Test that SingleCall pattern uses REACT trace category for frontend compatibility"""
    from xagent.core.agent.trace import TraceEvent, TraceHandler, Tracer

    # Create a custom handler to collect events
    collected_events = []

    class CollectingTraceHandler(TraceHandler):
        async def handle_event(self, event: TraceEvent) -> None:
            collected_events.append(event)

    tracer = Tracer()
    tracer.add_handler(CollectingTraceHandler())
    pattern = SingleCallPattern(llm=mock_llm, tracer=tracer)

    result = await pattern.run(
        task="Use test_tool",
        memory=mock_memory,
        tools=mock_tools,
        context=mock_context,
    )

    # Check that traces were created
    assert len(collected_events) > 0, "No trace events were collected"

    # Check for events with SingleCall pattern data
    single_call_events = [
        e
        for e in collected_events
        if hasattr(e, "data") and e.data.get("pattern") == "SingleCall"
    ]
    assert len(single_call_events) > 0, "No SingleCall pattern events found"

    assert result["success"] is True
