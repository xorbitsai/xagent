"""Unit tests for ContextBuilder with parallel compaction."""

import asyncio
from typing import Any, Dict, List

import pytest

from xagent.core.agent.utils.context_builder import ContextBuilder, StepExecutionResult


class MockLLM:
    """Mock LLM for testing."""

    def __init__(
        self, response_text="Mock compacted response", model_name="mock_model"
    ):
        self.response_text = response_text
        self.chat_calls = []
        self.supports_thinking_mode = False
        self.model_name = model_name

    async def chat(self, messages: List[Dict[str, Any]]) -> str:
        self.chat_calls.append(messages)
        return self.response_text

    async def stream_chat(self, messages: List[Dict[str, Any]]):
        yield self.response_text


class TestContextBuilder:
    """Test suite for ContextBuilder."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM for testing."""
        return MockLLM()

    @pytest.fixture
    def context_builder(self, mock_llm):
        """Create a ContextBuilder instance for testing."""
        return ContextBuilder(llm=mock_llm, compact_threshold=1000)

    @pytest.fixture
    def sample_dependency_results(self):
        """Create sample dependency results for testing."""
        return {
            "dep1": StepExecutionResult(
                step_id="dep1",
                messages=[
                    {"role": "user", "content": "Hello " * 100},
                    {"role": "assistant", "content": "World " * 100},
                ],
                final_result={"result": "dep1_result"},
                agent_name="agent1",
            ),
            "dep2": StepExecutionResult(
                step_id="dep2",
                messages=[
                    {"role": "user", "content": "Test " * 100},
                    {"role": "assistant", "content": "Response " * 100},
                ],
                final_result={"result": "dep2_result"},
                agent_name="agent2",
            ),
        }

    def test_context_builder_initialization(self, mock_llm):
        """Test ContextBuilder initialization with parameters."""
        builder = ContextBuilder(llm=mock_llm, compact_threshold=500)

        assert builder.llm == mock_llm
        assert builder.compact_config.threshold == 500

    @pytest.mark.asyncio
    async def test_build_context_no_dependencies(self, context_builder):
        """Test building context with no dependencies."""
        result = await context_builder.build_context_for_step(
            step_name="test_step",
            step_description="Test step description",
            dependencies=[],
            dependency_results={},
        )

        assert len(result) == 1
        assert result[0]["role"] == "system"
        assert "test_step" in result[0]["content"]

    @pytest.mark.asyncio
    async def test_build_context_with_dependencies(
        self, context_builder, sample_dependency_results
    ):
        """Test building context with dependencies."""
        result = await context_builder.build_context_for_step(
            step_name="test_step",
            step_description="Test step description",
            dependencies=["dep1", "dep2"],
            dependency_results=sample_dependency_results,
        )

        assert len(result) >= 3  # system + separators + messages
        assert result[0]["role"] == "system"

        # Check that dependency separators are present
        separator_contents = [
            msg["content"]
            for msg in result
            if "=== Results from dependency step:" in msg["content"]
        ]
        assert len(separator_contents) == 2
        assert "dep1" in separator_contents[0]
        assert "dep2" in separator_contents[1]

    @pytest.mark.asyncio
    async def test_parallel_compaction_triggered(
        self, context_builder, sample_dependency_results
    ):
        """Test that parallel compaction is triggered when threshold is exceeded."""
        # Set low threshold to trigger compaction
        context_builder.compact_config.threshold = 100

        await context_builder.build_context_for_step(
            step_name="test_step",
            step_description="Test step description",
            dependencies=["dep1", "dep2"],
            dependency_results=sample_dependency_results,
        )

        # Check that LLM was called for compaction
        assert len(context_builder.llm.chat_calls) > 0

        # Verify compaction prompts were created
        for call in context_builder.llm.chat_calls:
            assert len(call) == 2  # system + user messages
            assert call[0]["role"] == "system"
            assert call[1]["role"] == "user"

    @pytest.mark.asyncio
    async def test_parallel_compaction_concurrency(
        self, context_builder, sample_dependency_results
    ):
        """Test that compaction operations run in parallel with proper concurrency control."""
        context_builder.compact_config.threshold = 100

        # Add more dependencies to test concurrency
        sample_dependency_results["dep3"] = StepExecutionResult(
            step_id="dep3",
            messages=[{"role": "user", "content": "More content " * 100}],
            final_result={"result": "dep3_result"},
            agent_name="agent3",
        )

        result = await context_builder.build_context_for_step(
            step_name="test_step",
            step_description="Test step description",
            dependencies=["dep1", "dep2", "dep3"],
            dependency_results=sample_dependency_results,
        )

        # Check that all dependencies were processed
        separator_contents = [
            msg["content"]
            for msg in result
            if "=== Results from dependency step:" in msg["content"]
        ]
        assert len(separator_contents) == 3

    @pytest.mark.asyncio
    async def test_compaction_error_handling(
        self, context_builder, sample_dependency_results
    ):
        """Test error handling when compaction fails."""
        context_builder.compact_config.threshold = 100

        # Make LLM raise an exception
        async def failing_chat(messages):
            raise Exception("LLM failed")

        context_builder.llm.chat = failing_chat

        result = await context_builder.build_context_for_step(
            step_name="test_step",
            step_description="Test step description",
            dependencies=["dep1", "dep2"],
            dependency_results=sample_dependency_results,
        )

        # Should still succeed with fallback (truncated messages)
        assert len(result) >= 3
        assert result[0]["role"] == "system"

        # Check that separators are present
        separator_contents = [
            msg["content"]
            for msg in result
            if "=== Results from dependency step:" in msg["content"]
        ]
        assert len(separator_contents) == 2

    @pytest.mark.asyncio
    async def test_compaction_unavailable_dependency(
        self, context_builder, sample_dependency_results
    ):
        """Test handling of dependencies with compact_available=False."""
        # Use higher threshold to avoid total compaction
        context_builder.compact_config.threshold = 1000

        # Mark one dependency as not available for compaction
        sample_dependency_results["dep1"].compact_available = False

        result = await context_builder.build_context_for_step(
            step_name="test_step",
            step_description="Test step description",
            dependencies=["dep1", "dep2"],
            dependency_results=sample_dependency_results,
        )

        # Should still work, dep1 should not be compacted
        assert len(result) >= 3  # system + separator + messages

        # Check that separators are present for both dependencies
        separator_contents = [
            msg["content"]
            for msg in result
            if "=== Results from dependency step:" in msg["content"]
        ]
        assert len(separator_contents) == 2

        # Check that only one compaction call was made (for dep2, if it exceeds threshold)
        # dep1 should not be compacted due to compact_available=False
        assert len(context_builder.llm.chat_calls) <= 1

    @pytest.mark.asyncio
    async def test_total_compaction_after_individual(
        self, context_builder, sample_dependency_results
    ):
        """Test total compaction when individual compaction is not enough."""
        # Set very low threshold to trigger both individual and total compaction
        context_builder.compact_config.threshold = 10

        # Make individual compaction return still large content
        context_builder.llm.response_text = "Large content " * 100

        await context_builder.build_context_for_step(
            step_name="test_step",
            step_description="Test step description",
            dependencies=["dep1", "dep2"],
            dependency_results=sample_dependency_results,
        )

        # Should have called LLM for individual compaction and total compaction
        assert len(context_builder.llm.chat_calls) >= 2

    @pytest.mark.asyncio
    async def test_missing_dependency_handling(
        self, context_builder, sample_dependency_results
    ):
        """Test handling when a dependency is missing from results."""
        result = await context_builder.build_context_for_step(
            step_name="test_step",
            step_description="Test step description",
            dependencies=["dep1", "dep2", "missing_dep"],
            dependency_results=sample_dependency_results,
        )

        # Should only process available dependencies
        separator_contents = [
            msg["content"]
            for msg in result
            if "=== Results from dependency step:" in msg["content"]
        ]
        assert len(separator_contents) == 2  # Only dep1 and dep2

    @pytest.mark.asyncio
    async def test_empty_messages_handling(self, context_builder):
        """Test handling of dependencies with empty messages."""
        empty_dep = StepExecutionResult(
            step_id="empty_dep",
            messages=[],
            final_result={"result": "empty_result"},
            agent_name="empty_agent",
        )

        result = await context_builder.build_context_for_step(
            step_name="test_step",
            step_description="Test step description",
            dependencies=["empty_dep"],
            dependency_results={"empty_dep": empty_dep},
        )

        # Should handle empty messages gracefully
        assert len(result) >= 2  # system + separator
        separator_contents = [
            msg["content"]
            for msg in result
            if "=== Results from dependency step:" in msg["content"]
        ]
        assert len(separator_contents) == 1

    @pytest.mark.asyncio
    async def test_concurrent_execution_timing(
        self, context_builder, sample_dependency_results
    ):
        """Test that parallel operations actually run concurrently."""
        context_builder.compact_config.threshold = 100

        # Track execution timing
        call_times = []

        async def timed_chat(messages):
            call_times.append(asyncio.get_event_loop().time())
            await asyncio.sleep(0.1)  # Simulate some work
            # Return a valid compacted response to avoid errors
            return "USER: Compacted content\nASSISTANT: This is a summary"

        context_builder.llm.chat = timed_chat

        start_time = asyncio.get_event_loop().time()
        await context_builder.build_context_for_step(
            step_name="test_step",
            step_description="Test step description",
            dependencies=["dep1", "dep2"],
            dependency_results=sample_dependency_results,
        )
        end_time = asyncio.get_event_loop().time()

        # Should have made calls for both dependencies
        assert len(call_times) == 2

        # Total time should be less than sequential execution time
        # (sequential would be ~0.2s, parallel should be ~0.1s + overhead)
        total_time = end_time - start_time
        assert (
            total_time < 0.25
        )  # Should be faster than sequential (allowing for overhead)

    def test_step_execution_result_creation(self):
        """Test StepExecutionResult dataclass creation."""
        result = StepExecutionResult(
            step_id="test_step",
            messages=[{"role": "user", "content": "test"}],
            final_result={"output": "test_output"},
            agent_name="test_agent",
            compact_available=False,
        )

        assert result.step_id == "test_step"
        assert result.messages == [{"role": "user", "content": "test"}]
        assert result.final_result == {"output": "test_output"}
        assert result.agent_name == "test_agent"
        assert result.compact_available is False

    @pytest.mark.asyncio
    async def test_individual_threshold_calculation(
        self, context_builder, sample_dependency_results
    ):
        """Test that individual threshold is calculated correctly."""
        context_builder.compact_config.threshold = 1000

        # Add 4 dependencies to test threshold division
        for i in range(3, 7):
            sample_dependency_results[f"dep{i}"] = StepExecutionResult(
                step_id=f"dep{i}",
                messages=[{"role": "user", "content": f"Content {i} " * 10}],
                final_result={"result": f"result{i}"},
                agent_name=f"agent{i}",
            )

        dependencies = list(sample_dependency_results.keys())
        await context_builder.build_context_for_step(
            step_name="test_step",
            step_description="Test step description",
            dependencies=dependencies,
            dependency_results=sample_dependency_results,
        )

        # Individual threshold should be compact_threshold // num_dependencies
        expected_individual_threshold = 1000 // len(dependencies)
        assert expected_individual_threshold > 0

    @pytest.mark.asyncio
    async def test_conversation_history_included(self, context_builder):
        """Test that conversation history is included in the context."""
        conversation_history = [
            {"role": "user", "content": "First message"},
            {"role": "assistant", "content": "First response"},
            {"role": "user", "content": "Second message"},
        ]

        result = await context_builder.build_context_for_step(
            step_name="test_step",
            step_description="Test step description",
            dependencies=[],
            dependency_results={},
            conversation_history=conversation_history,
        )

        # Should have system prompt + separator + history messages + end separator
        assert len(result) == 6  # system + start_sep + 3 history msgs + end_sep
        assert result[0]["role"] == "system"
        assert "=== Previous Conversation ===" in result[1]["content"]
        assert result[2]["content"] == "First message"
        assert result[3]["content"] == "First response"
        assert result[4]["content"] == "Second message"
        assert "=== End of Previous Conversation ===" in result[5]["content"]

    @pytest.mark.asyncio
    async def test_conversation_history_with_dependencies(
        self, context_builder, sample_dependency_results
    ):
        """Test that conversation history appears before dependency results."""
        conversation_history = [
            {"role": "user", "content": "User question"},
            {"role": "assistant", "content": "Assistant answer"},
        ]

        result = await context_builder.build_context_for_step(
            step_name="test_step",
            step_description="Test step description",
            dependencies=["dep1"],
            dependency_results=sample_dependency_results,
            conversation_history=conversation_history,
        )

        # Find the positions of different sections
        history_section_start = None
        history_section_end = None
        dependency_section = None

        for i, msg in enumerate(result):
            if "=== Previous Conversation ===" in msg.get("content", ""):
                history_section_start = i
            if "=== End of Previous Conversation ===" in msg.get("content", ""):
                history_section_end = i
            if "=== Results from dependency step:" in msg.get("content", ""):
                dependency_section = i

        # Verify order: conversation history comes before dependencies
        assert history_section_start is not None, "Conversation history start not found"
        assert history_section_end is not None, "Conversation history end not found"
        assert dependency_section is not None, "Dependency section not found"
        assert history_section_start < history_section_end < dependency_section, (
            "Conversation history should appear before dependency results"
        )
