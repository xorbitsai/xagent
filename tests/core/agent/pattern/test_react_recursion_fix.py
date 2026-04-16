"""
Tests for RecursionError handling in ReAct pattern.

These tests verify that JSON parsing errors (including RecursionError) are handled gracefully
without causing infinite loops, and that LLM can see errors and attempt recovery.
"""

import json
from unittest.mock import patch

import pytest

from xagent.core.agent.context import AgentContext
from xagent.core.agent.exceptions import PatternExecutionError
from xagent.core.agent.pattern.react import ReActPattern
from xagent.core.memory.base import MemoryResponse, MemoryStore
from xagent.core.model.chat.basic.base import BaseLLM


class MockReActLLM(BaseLLM):
    """Mock LLM for testing RecursionError scenarios"""

    def __init__(self, responses=None, stream_chunks=None):
        self.responses = responses or []
        self.stream_chunks = stream_chunks or []
        self.call_count = 0
        self._abilities = ["chat", "tool_calling"]
        self._model_name = "mock_react_llm"
        self.messages_history = []

    @property
    def supports_thinking_mode(self) -> bool:
        return False

    @property
    def abilities(self) -> list[str]:
        return self._abilities

    @property
    def model_name(self) -> str:
        return self._model_name

    async def chat(self, messages: list[dict[str, str]], **kwargs):
        self.messages_history.append(messages)
        if self.call_count < len(self.responses):
            response = self.responses[self.call_count]
            self.call_count += 1
            # If response is a string, return it as-is (will be processed by _extract_content)
            return response
        self.call_count += 1
        return '{"type": "final_answer", "reasoning": "Done", "answer": "Done", "success": true}'

    async def stream_chat(self, messages: list[dict[str, str]], **kwargs):
        self.messages_history.append(messages)
        from xagent.core.model.chat.types import ChunkType, StreamChunk

        # Get the full response for this call
        if self.call_count < len(self.stream_chunks):
            chunks_data = self.stream_chunks[self.call_count]
        else:
            # Use responses array as fallback
            response = (
                self.responses[self.call_count - len(self.stream_chunks)]
                if self.call_count - len(self.stream_chunks) < len(self.responses)
                else '{"type": "final_answer", "answer": "Done"}'
            )
            chunks_data = [{"delta": response, "type": "token"}]

        # Increment call_count for both paths
        self.call_count += 1

        # Yield chunks
        accumulated_content = ""
        for chunk_data in chunks_data:
            if isinstance(chunk_data, dict):
                delta = chunk_data.get("delta", "")
                accumulated_content += delta
                yield StreamChunk(
                    type=ChunkType.TOKEN,
                    delta=delta,
                    content=accumulated_content,
                )
            elif isinstance(chunk_data, str):
                accumulated_content += chunk_data
                yield StreamChunk(
                    type=ChunkType.TOKEN,
                    delta=chunk_data,
                    content=accumulated_content,
                )
            else:
                yield chunk_data


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


# ============================================================================
# STRATEGY 1: Simulate RecursionError - Verify PatternExecutionError Handling
# ============================================================================


@pytest.mark.asyncio
async def test_recursion_error_converted_to_observation():
    """Test that RecursionError is converted to observation so LLM can see and retry."""

    # Mock LLM: first gets error, then returns correct answer
    llm = MockReActLLM(
        responses=[
            '{"type": "final_answer", "reasoning": "Done", "answer": "Done", "success": true}'
        ],
        stream_chunks=[
            # First call - will trigger RecursionError
            [{"delta": '{"type": "final_answer", "answer": "test"}'}],
            # Second call - after seeing error, returns valid JSON
            [
                {
                    "delta": '{"type": "final_answer", "reasoning": "Done", "answer": "Done", "success": true}'
                }
            ],
        ],
    )

    # Mock repair_loads: first time throws RecursionError, second time works
    repair_call_count = [0]

    def side_effect_repair(content, logging=True):
        repair_call_count[0] += 1
        if repair_call_count[0] == 1:
            # First call triggers RecursionError
            raise RecursionError("maximum recursion depth exceeded")
        # Subsequent calls work normally
        import json

        return json.loads(content)

    with patch("xagent.core.agent.pattern.react.repair_loads") as mock_repair:
        mock_repair.side_effect = side_effect_repair

        pattern = ReActPattern(llm=llm, max_iterations=10)
        memory = DummyMemoryStore()
        tools = []

        # Execute task - should handle RecursionError gracefully
        result = await pattern.run(
            task="Test task",
            memory=memory,
            tools=tools,
            context=AgentContext(),
        )

        # Verify:
        # 1. Doesn't crash with RecursionError
        assert result is not None
        # 2. repair_loads was called at least once (error occurred)
        assert repair_call_count[0] >= 1
        # 3. Task completes successfully (LLM saw error and recovered)
        assert result.get("success") is True
        # 4. Error was handled through observation (not infinite loop)
        assert result.get("iterations", 0) <= 10


# ============================================================================
# STRATEGY 2: 497+ Level Deeply Nested JSON
# ============================================================================


@pytest.mark.asyncio
async def test_deeply_nested_json_handling():
    """Test that deeply nested JSON (497+ levels) is handled gracefully.

    This test verifies that when LLM consistently returns malformed JSON
    (deeply nested that triggers RecursionError), the agent properly handles
    it by:
    1. Converting the error to an observation
    2. Allowing LLM to see the error and retry
    3. Eventually hitting max_iterations if LLM never corrects itself

    However, since the MockReActLLM has a fallback mechanism that returns
    a valid final_answer when stream_chunks are exhausted, we test a
    different scenario: LLM returns bad JSON once, then corrects itself.
    """

    # Create 497 levels of nested JSON (triggers RecursionError threshold)
    deep_json = "{"
    for i in range(497):
        deep_json += '{"a":'
    deep_json += '"x"' + "}" * 497
    deep_json += "}"

    # LLM: first returns deeply nested JSON, then valid final_answer
    llm = MockReActLLM(
        responses=[],
        stream_chunks=[
            # First call - deeply nested JSON that triggers RecursionError
            [{"delta": deep_json}],
            # Second call - LLM sees error and returns valid JSON
            [
                {
                    "delta": '{"type": "final_answer", "reasoning": "Fixed after seeing error", "answer": "Success!", "success": true}'
                }
            ],
        ],
    )

    pattern = ReActPattern(llm=llm, max_iterations=5)
    memory = DummyMemoryStore()
    tools = []

    # Execute task - should succeed after LLM corrects the error
    result = await pattern.run(
        task="Test task",
        memory=memory,
        tools=tools,
        context=AgentContext(),
    )

    # Verify: task completes successfully after LLM sees error and corrects
    assert result is not None
    assert result.get("success") is True


# ============================================================================
# STRATEGY 3: Error Handling Consistency
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error_name,error_exception",
    [
        ("RecursionError", RecursionError("max recursion")),
        ("JSONDecodeError", json.JSONDecodeError("test", "doc", 0)),
    ],
)
async def test_json_error_handling_consistency(error_name, error_exception):
    """Test that JSON parsing errors are caught and converted to observations."""

    # LLM: first gets error observation, then returns valid answer
    llm = MockReActLLM(
        responses=[
            '{"type": "final_answer", "reasoning": "Fixed after seeing error", "answer": "Success!", "success": true}'
        ],
        stream_chunks=[
            # First call - will trigger parsing error that gets converted to observation
            [{"delta": "invalid json {{{"}],
            # Second call - after seeing error observation, returns valid JSON
            [
                {
                    "delta": '{"type": "final_answer", "reasoning": "Fixed after seeing error", "answer": "Success!", "success": true}'
                }
            ],
        ],
    )

    call_count = [0]
    messages_history = []

    original_stream_chat = llm.stream_chat

    async def tracking_stream_chat(messages, **kwargs):
        call_count[0] += 1
        messages_history.append(messages.copy())
        async for chunk in original_stream_chat(messages, **kwargs):
            yield chunk

    llm.stream_chat = tracking_stream_chat

    repair_call_count = [0]

    def side_effect_repair(content, logging=True):
        repair_call_count[0] += 1
        # First few calls fail (may be called from multiple locations in new main)
        if repair_call_count[0] <= 2:
            raise error_exception
        return json.loads(content)

    with patch("xagent.core.agent.pattern.react.repair_loads") as mock_repair:
        mock_repair.side_effect = side_effect_repair

        pattern = ReActPattern(llm=llm, max_iterations=5)
        result = await pattern.run(
            task="Test task",
            memory=DummyMemoryStore(),
            tools=[],
            context=AgentContext(),
        )

        # Verify the fix works correctly:
        # 1. repair_loads was called (error was detected)
        assert repair_call_count[0] >= 1, (
            f"Expected repair_loads to be called at least once, got {repair_call_count[0]}"
        )
        # 2. LLM was called multiple times (second call saw the error observation)
        assert call_count[0] >= 2, (
            f"Expected LLM to be called at least twice, got {call_count[0]}"
        )
        # 3. Task completes successfully (LLM saw error and recovered)
        assert result.get("success") is True, (
            f"Expected success=True, got {result.get('success')}"
        )
        # 4. Error observation was added to messages
        error_observation_found = any(
            "Failed to parse your response" in msg.get("content", "")
            for msg in messages_history[-1]
            if msg.get("role") == "user"
        )
        assert error_observation_found, "Error observation should be in messages"


# ============================================================================
# STRATEGY 4: PatternExecutionError Converted to Observation
# ============================================================================


@pytest.mark.asyncio
async def test_pattern_execution_error_converted_to_observation():
    """Test that PatternExecutionError is converted to observation so LLM can see the error."""

    # Track messages to verify error was added
    messages_history = []

    # LLM: first gets error (which becomes observation), then succeeds
    llm = MockReActLLM(
        responses=[
            '{"type": "final_answer", "reasoning": "Fixed", "answer": "Fixed!", "success": true}'
        ],
        stream_chunks=[
            # First call - will trigger PatternExecutionError
            [{"delta": "invalid json {{{"}],
            # Second call - after seeing error in observation, returns valid JSON
            [
                {
                    "delta": '{"type": "final_answer", "reasoning": "Fixed", "answer": "Fixed!", "success": true}'
                }
            ],
        ],
    )

    # Track calls to verify error was seen - track stream_chat instead of chat
    call_count = [0]

    original_stream_chat = llm.stream_chat

    async def tracking_stream_chat(messages, **kwargs):
        call_count[0] += 1
        messages_history.append(messages.copy())
        async for chunk in original_stream_chat(messages, **kwargs):
            yield chunk

    llm.stream_chat = tracking_stream_chat

    # Mock repair_loads: first fails with JSONDecodeError, second succeeds
    repair_call_count = [0]

    def side_effect_repair(content, logging=True):
        repair_call_count[0] += 1
        # First two calls fail (may be called from multiple locations in new main)
        if repair_call_count[0] <= 2:
            raise json.JSONDecodeError("Expecting value", "{{{", 0)
        # Third call succeeds (second LLM call with valid JSON)
        return json.loads(content)

    with patch("xagent.core.agent.pattern.react.repair_loads") as mock_repair:
        mock_repair.side_effect = side_effect_repair

        pattern = ReActPattern(llm=llm, max_iterations=5)
        result = await pattern.run(
            task="Test task",
            memory=DummyMemoryStore(),
            tools=[],
            context=AgentContext(),
        )

        # Verify:
        # 1. repair_loads was called (error occurred)
        assert repair_call_count[0] >= 1
        # 2. LLM was called multiple times (second call saw the error observation)
        assert call_count[0] >= 2
        # 3. Task completes successfully (LLM saw error and corrected)
        assert result.get("success") is True
        # 4. Second call's messages include the error observation
        second_call_messages = messages_history[1] if len(messages_history) > 1 else []
        error_observation_found = any(
            "Failed to parse your response" in msg.get("content", "")
            or (
                "Observation:" in msg.get("content", "")
                and "JSON" in msg.get("content", "")
            )
            for msg in second_call_messages
        )
        assert error_observation_found, (
            "Error observation should be in second call's messages"
        )


# ============================================================================
# STRATEGY 5: LLM Can See Error
# ============================================================================


@pytest.mark.asyncio
async def test_llm_can_see_json_parsing_error():
    """Test that when JSON parsing fails, LLM sees the error in next iteration."""

    # LLM first returns bad JSON, then correct answer
    llm = MockReActLLM(
        responses=[
            '{"type": "final_answer", "reasoning": "Fixed", "answer": "Fixed!", "success": true}'
        ],
        stream_chunks=[
            # First: error - will cause JSON parsing to fail
            [{"delta": "invalid json {{{"}],
            # After seeing error, valid response
            [
                {
                    "delta": '{"type": "final_answer", "reasoning": "Fixed", "answer": "Fixed!", "success": true}'
                }
            ],
        ],
    )

    # Mock repair_loads: first fails, second succeeds
    repair_call_count = [0]

    def side_effect_repair(content, logging=True):
        repair_call_count[0] += 1
        if repair_call_count[0] == 1:
            raise json.JSONDecodeError("test", "doc", 0)
        return json.loads(content)

    with patch("xagent.core.agent.pattern.react.repair_loads") as mock_repair:
        mock_repair.side_effect = side_effect_repair

        pattern = ReActPattern(llm=llm, max_iterations=5)

        # Should succeed after second call
        result = await pattern.run(
            task="Test task",
            memory=DummyMemoryStore(),
            tools=[],
            context=AgentContext(),
        )

        # Verify: repair_loads was called (error was caught and handled)
        assert repair_call_count[0] >= 1
        # Task completes successfully
        assert result.get("success") is True


# ============================================================================
# STRATEGY 6: Native Tool Calling - RecursionError in tool arguments
# ============================================================================


def test_convert_native_tool_call_recursion_error():
    """Test that RecursionError in _convert_native_tool_call_to_action is handled.

    This test verifies the fix for the review comment about native tool calling
    compatibility. The _convert_native_tool_call_to_action method parses tool
    arguments using json.loads, which may raise RecursionError for deeply nested JSON.
    The fix ensures this is caught and converted to PatternExecutionError.

    This is a unit test of the specific method, not an integration test.
    """

    from xagent.core.agent.pattern.react import ReActPattern

    # Create a minimal ReActPattern instance (just for the method)
    pattern = ReActPattern(llm=MockReActLLM())

    # Create a tool call response with valid JSON structure
    response = {
        "type": "tool_call",
        "tool_calls": [
            {
                "function": {
                    "name": "calculator",
                    "arguments": '{"expression": "2+2"}',  # Valid JSON
                }
            }
        ],
        "reasoning": "I need to calculate",
    }

    # Mock json.loads to raise RecursionError
    with patch("json.loads") as mock_json_loads:
        mock_json_loads.side_effect = RecursionError("maximum recursion depth exceeded")

        # Should raise PatternExecutionError, not RecursionError
        with pytest.raises(PatternExecutionError) as exc_info:
            pattern._convert_native_tool_call_to_action(response)

        # Verify the error is PatternExecutionError
        assert exc_info.value.pattern_name == "ReAct"
        assert "Failed to parse tool arguments JSON" in str(exc_info.value)
        assert "RecursionError" in exc_info.value.context.get("error_type", "")


def test_convert_native_tool_call_json_decode_error():
    """Test that JSONDecodeError in _convert_native_tool_call_to_action is handled."""

    from xagent.core.agent.pattern.react import ReActPattern

    pattern = ReActPattern(llm=MockReActLLM())

    response = {
        "type": "tool_call",
        "tool_calls": [
            {
                "function": {
                    "name": "calculator",
                    "arguments": "invalid json {{{",  # Invalid JSON
                }
            }
        ],
        "reasoning": "I need to calculate",
    }

    # Should raise PatternExecutionError, not JSONDecodeError
    with pytest.raises(PatternExecutionError) as exc_info:
        pattern._convert_native_tool_call_to_action(response)

    # Verify the error is PatternExecutionError
    assert exc_info.value.pattern_name == "ReAct"
    assert "Failed to parse tool arguments JSON" in str(exc_info.value)
    assert "JSONDecodeError" in exc_info.value.context.get("error_type", "")


# ============================================================================
# STRATEGY 7: Path 2 - Non-string response with RecursionError in repair_loads
# ============================================================================


@pytest.mark.asyncio
async def test_path2_recursion_error_in_repair_loads():
    """Test that RecursionError in repair_loads for non-string responses is handled.

    This addresses the deep code review finding that Path 2 (line 1876)
    was missing RecursionError handling. When LLM returns a non-string
    response (like a dict or list) and repair_loads encounters deeply
    nested JSON, it should raise PatternExecutionError, not propagate
    RecursionError to the generic exception handler.

    The test simulates a dict response that requires repair_loads processing,
    and repair_loads throws RecursionError.
    """

    # Track calls
    repair_call_count = [0]

    # Create an LLM that returns a dict response (non-string)
    # When LLM returns a dict (not a string), it goes through Path 2
    llm = MockReActLLM(
        responses=[
            # First response is a dict (non-string) that triggers Path 2
            # This dict will be processed by _extract_content -> repair_loads
            {"type": "final_answer", "answer": "test"},
        ],
        stream_chunks=[
            # Empty stream_chunks - forces use of responses array
        ],
    )

    # Mock repair_loads to raise RecursionError on first call
    def side_effect_repair(content, logging=True):
        repair_call_count[0] += 1
        if repair_call_count[0] == 1:
            # First call triggers RecursionError (simulating deeply nested JSON)
            raise RecursionError("maximum recursion depth exceeded")
        # Subsequent calls work normally
        import json

        return json.loads(content)

    with patch("xagent.core.agent.pattern.react.repair_loads") as mock_repair:
        mock_repair.side_effect = side_effect_repair

        pattern = ReActPattern(llm=llm, max_iterations=5)
        # The pattern.run should handle the RecursionError gracefully
        # by converting it to PatternExecutionError -> observation
        result = await pattern.run(
            task="Test task",
            memory=DummyMemoryStore(),
            tools=[],
            context=AgentContext(),
        )

    # Verify:
    # 1. repair_loads was called (error was triggered)
    assert repair_call_count[0] >= 1, (
        f"Expected repair_loads to be called, got {repair_call_count[0]}"
    )
    # 2. Task completes - either successfully or hitting max_iterations
    # The key is that it doesn't crash with RecursionError
    assert result is not None, "Result should not be None"
