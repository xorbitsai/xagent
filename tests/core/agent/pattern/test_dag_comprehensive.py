"""
Comprehensive tests for DAG plan-execute pattern covering workspace integration,
tool sharing, concurrency, and error scenarios.
"""

import asyncio
import json
from unittest.mock import Mock

import pytest

from xagent.core.agent.pattern.dag_plan_execute import DAGPlanExecutePattern
from xagent.core.model.chat.basic.openai import OpenAILLM
from xagent.core.model.chat.types import ChunkType, StreamChunk
from xagent.core.tools.adapters.vibe.workspace_file_tool import (
    create_workspace_file_tools,
)
from xagent.core.workspace import TaskWorkspace


def create_mock_stream_chat(mock_llm):
    """Create a mock stream_chat function that properly handles tool calls."""

    async def mock_stream_chat(**kwargs):
        # Get response from chat mock
        content = await mock_llm.chat(**kwargs)

        # Try to parse as JSON to determine response type
        try:
            response_data = json.loads(content)
            if response_data.get("type") == "tool_call":
                # Return native tool call format
                tool_name = response_data.get("tool_name", "")
                tool_args = response_data.get("tool_args", {})
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
                return
        except (json.JSONDecodeError, AttributeError):
            pass

        # Yield as text stream chunk (for final_answer)
        yield StreamChunk(
            type=ChunkType.TOKEN,
            content=content,
            delta=content,
        )
        yield StreamChunk(type=ChunkType.END, finish_reason="stop")

    return mock_stream_chat


class TestDAGComprehensive:
    """Comprehensive tests for DAG plan-execute pattern."""

    def test_step_agent_tool_sharing(self, tmp_path):
        """Test that step agents share the same tool instances with same workspace."""
        # Create workspace
        workspace = TaskWorkspace("test_task", str(tmp_path))

        # Create workspace tools
        workspace_tools = create_workspace_file_tools(workspace)

        # Create mock LLM
        mock_llm = Mock(spec=OpenAILLM)

        # Create DAG pattern
        dag_pattern = DAGPlanExecutePattern(llm=mock_llm, workspace=workspace)

        # Create tool map like DAG does
        tool_map = {tool.metadata.name: tool for tool in workspace_tools}

        # Simulate creating two step agents (like step3 and step4 from the error)
        step3_tools = dag_pattern._get_tools_for_step(
            Mock(id="step3", tool_names=None), tool_map
        )
        step4_tools = dag_pattern._get_tools_for_step(
            Mock(id="step4", tool_names=None), tool_map
        )

        # Both step agents should get the same tools
        assert len(step3_tools) == len(step4_tools)

        # Find write and read tools in both sets
        step3_write = None
        step3_read = None
        step4_write = None
        step4_read = None

        for tool in step3_tools:
            if tool.metadata.name == "write_file":
                step3_write = tool
            elif tool.metadata.name == "read_file":
                step3_read = tool

        for tool in step4_tools:
            if tool.metadata.name == "write_file":
                step4_write = tool
            elif tool.metadata.name == "read_file":
                step4_read = tool

        # Both steps should have access to both tools
        assert step3_write is not None
        assert step3_read is not None
        assert step4_write is not None
        assert step4_read is not None

        # The tools should be the same instances (same workspace)
        assert step3_write is step4_write, "Write tools should be the same instance"
        assert step3_read is step4_read, "Read tools should be the same instance"

        # Test that both tools operate on the same workspace
        # Step3 writes a file
        test_content = "Test content"
        write_result = step3_write.func("test.txt", test_content)
        assert isinstance(write_result, dict)
        assert write_result.get("success") is True
        assert isinstance(write_result.get("file_id"), str)

        # Verify file exists
        output_file = workspace.output_dir / "test.txt"
        assert output_file.exists()

        # Step4 reads the file
        read_content = step4_read.func("test.txt")
        assert read_content == test_content

    def test_tool_workspace_reference_consistency(self, tmp_path):
        """Test that all tools reference the same workspace instance."""
        workspace = TaskWorkspace("test_task", str(tmp_path))
        workspace_tools = create_workspace_file_tools(workspace)

        # Check that all tools share the same workspace instance
        workspace_refs = set()

        for tool in workspace_tools:
            if hasattr(tool.func, "__self__"):
                tool_instance = tool.func.__self__
                if hasattr(tool_instance, "workspace"):
                    workspace_refs.add(id(tool_instance.workspace))

        # All tools should reference the same workspace instance
        assert len(workspace_refs) == 1, (
            f"All tools should share same workspace, found {len(workspace_refs)} different references"
        )

    def test_path_resolution_consistency(self, tmp_path):
        """Test that write and read operations resolve paths consistently."""
        workspace = TaskWorkspace("test_task", str(tmp_path))
        workspace_tools = create_workspace_file_tools(workspace)

        # Find write and read tools
        write_tool = None
        read_tool = None

        for tool in workspace_tools:
            if tool.metadata.name == "write_file":
                write_tool = tool
            elif tool.metadata.name == "read_file":
                read_tool = tool

        assert write_tool is not None
        assert read_tool is not None

        # Test path resolution
        test_filename = "gradient_hello.html"

        # Both tools should resolve to the same path
        write_tool_instance = write_tool.func.__self__
        read_tool_instance = read_tool.func.__self__

        write_resolved = write_tool_instance._resolve_path(test_filename, "output")
        read_resolved = read_tool_instance._resolve_path(test_filename, "output")

        assert write_resolved == read_resolved, (
            "Write and read should resolve to same path"
        )
        assert write_resolved == workspace.output_dir / test_filename

    async def test_write_then_read_in_dag_context(self, tmp_path):
        """Test the exact scenario from the error: write then read in DAG context."""
        # Create workspace
        workspace = TaskWorkspace("test_task", str(tmp_path))

        # Create workspace tools
        workspace_tools = create_workspace_file_tools(workspace)

        # Create tool map like the DAG pattern does
        tool_map = {tool.metadata.name: tool for tool in workspace_tools}

        # Get the write and read tools
        write_tool = tool_map["write_file"]
        read_tool = tool_map["read_file"]

        # Test the exact content from the error
        html_content = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-s"""

        # Write the file (like step 3 in the DAG)
        write_result = write_tool.func("gradient_hello.html", html_content)
        assert isinstance(write_result, dict)
        assert write_result.get("success") is True
        assert isinstance(write_result.get("file_id"), str)

        # Verify file exists immediately after write
        output_file = workspace.output_dir / "gradient_hello.html"
        assert output_file.exists(), f"File should exist at {output_file}"
        assert output_file.read_text() == html_content

        # Try to read the file (like step 4 in the DAG)
        # This should NOT fail with "文件不存在"
        read_content = read_tool.func("gradient_hello.html")
        assert read_content == html_content

        # Also test with a small delay to check for timing issues
        await asyncio.sleep(0.1)
        read_content_delayed = read_tool.func("gradient_hello.html")
        assert read_content_delayed == html_content

    def test_multiple_step_agents_same_tools(self, tmp_path):
        """Test that multiple step agents using the same tools work correctly."""
        workspace = TaskWorkspace("test_task", str(tmp_path))
        workspace_tools = create_workspace_file_tools(workspace)
        tool_map = {tool.metadata.name: tool for tool in workspace_tools}

        # Simulate two different step agents using the same tools
        write_tool = tool_map["write_file"]
        read_tool = tool_map["read_file"]

        # First agent writes
        content = "Test content"
        write_result = write_tool.func("test.txt", content)
        assert isinstance(write_result, dict)
        assert write_result.get("success") is True
        assert isinstance(write_result.get("file_id"), str)

        # Second agent reads
        read_content = read_tool.func("test.txt")
        assert read_content == content

        # Verify file is in the right location
        assert (workspace.output_dir / "test.txt").exists()

    async def test_concurrent_write_read_same_workspace(self, tmp_path):
        """Test concurrent write and read operations on the same workspace."""
        workspace = TaskWorkspace("test_task", str(tmp_path))
        workspace_tools = create_workspace_file_tools(workspace)

        # Get tools
        write_tool = None
        read_tool = None

        for tool in workspace_tools:
            if tool.metadata.name == "write_file":
                write_tool = tool
            elif tool.metadata.name == "read_file":
                read_tool = tool

        assert write_tool is not None
        assert read_tool is not None

        # Test concurrent access
        test_content = "Concurrent test content"
        test_filename = "concurrent_test.txt"

        async def write_task():
            # Small delay to simulate concurrent execution
            await asyncio.sleep(0.01)
            result = write_tool.func(test_filename, test_content)
            return result

        async def read_task():
            # Small delay to simulate concurrent execution
            await asyncio.sleep(0.02)
            try:
                content = read_tool.func(test_filename)
                return content
            except FileNotFoundError:
                return "FILE_NOT_FOUND"

        # Run tasks concurrently
        write_result, read_result = await asyncio.gather(write_task(), read_task())

        # Write should succeed
        assert isinstance(write_result, dict)
        assert write_result.get("success") is True
        assert isinstance(write_result.get("file_id"), str)

        # Read might fail if there's a race condition
        # But eventually the file should be readable
        if read_result == "FILE_NOT_FOUND":
            # Retry read after a short delay
            await asyncio.sleep(0.1)
            read_result_retry = read_tool.func(test_filename)
            assert read_result_retry == test_content
        else:
            assert read_result == test_content

    async def test_sequential_write_read_with_delay(self, tmp_path):
        """Test sequential write and read with various delays to identify timing issues."""
        workspace = TaskWorkspace("test_task", str(tmp_path))
        workspace_tools = create_workspace_file_tools(workspace)

        write_tool = None
        read_tool = None

        for tool in workspace_tools:
            if tool.metadata.name == "write_file":
                write_tool = tool
            elif tool.metadata.name == "read_file":
                read_tool = tool

        assert write_tool is not None
        assert read_tool is not None

        test_content = "Timing test content"
        test_filename = "timing_test.txt"

        # Test various delays between write and read
        for delay in [0, 0.001, 0.01, 0.1]:
            # Write file
            write_result = write_tool.func(test_filename, test_content)
            assert isinstance(write_result, dict)
            assert write_result.get("success") is True
            assert isinstance(write_result.get("file_id"), str)

            # Add delay
            if delay > 0:
                await asyncio.sleep(delay)

            # Read file
            try:
                read_content = read_tool.func(test_filename)
                assert read_content == test_content, f"Failed with delay {delay}"
            except FileNotFoundError as e:
                pytest.fail(f"FileNotFoundError with delay {delay}: {e}")

            # Clean up for next iteration
            (workspace.output_dir / test_filename).unlink(missing_ok=True)

    def test_file_not_found_error_message(self, tmp_path):
        """Test that the error message matches what we see in the logs."""
        workspace = TaskWorkspace("test_task", str(tmp_path))
        workspace_tools = create_workspace_file_tools(workspace)

        read_tool = None
        for tool in workspace_tools:
            if tool.metadata.name == "read_file":
                read_tool = tool

        assert read_tool is not None

        # Try to read a non-existent file
        with pytest.raises(FileNotFoundError) as exc_info:
            read_tool.func("gradient_hello.html")

        # Check that the error message matches what we see in the logs
        error_msg = str(exc_info.value)
        assert (
            "File 'gradient_hello.html' not found in workspace directories" in error_msg
        )

    async def test_sequential_execution_with_dependencies(self):
        """Test that steps with dependencies execute sequentially, not concurrently."""
        import asyncio
        import time
        from unittest.mock import MagicMock, Mock

        from xagent.core.agent.pattern.dag_plan_execute.models import (
            ExecutionPlan,
            PlanStep,
        )
        from xagent.core.agent.pattern.dag_plan_execute.plan_executor import (
            PlanExecutor,
        )
        from xagent.core.model.chat.basic.base import BaseLLM
        from xagent.core.workspace import TaskWorkspace

        # Create a simple mock tool that doesn't require LLM interaction
        class SimpleMockTool:
            def __init__(self, name, execution_time=0.1):
                self._metadata = Mock()
                self._metadata.name = name
                self.execution_time = execution_time
                self.execution_times = []

            @property
            def metadata(self):
                return self._metadata

            def args_type(self):
                from pydantic import BaseModel

                class Args(BaseModel):
                    pass

                return Args

            def return_type(self):
                from pydantic import BaseModel, Field

                class Result(BaseModel):
                    result: str = Field(..., description="Execution result")
                    success: bool = Field(
                        ..., description="Whether execution succeeded"
                    )

                return Result

            def state_type(self):
                return None

            def is_async(self):
                return True

            def return_value_as_string(self, value):
                return str(value)

            async def run_json_async(self, args):
                start_time = time.time()
                await asyncio.sleep(self.execution_time)
                end_time = time.time()
                self.execution_times.append((start_time, end_time))
                return {"result": f"Result from {self.metadata.name}", "success": True}

            def run_json_sync(self, args):
                return {"result": f"Result from {self.metadata.name}", "success": True}

            async def save_state_json(self):
                return {}

            async def load_state_json(self, state):
                pass

        # Create tools with different execution times
        tool_a = SimpleMockTool("tool_A", execution_time=0.2)
        tool_b = SimpleMockTool("tool_B", execution_time=0.3)
        tool_c = SimpleMockTool("tool_C", execution_time=0.4)

        tool_map = {
            "tool_A": tool_a,
            "tool_B": tool_b,
            "tool_C": tool_c,
        }

        # Create execution plan with A -> B -> C dependencies
        steps = [
            PlanStep(
                id="step_A",
                name="Step A",
                description="Execute step A",
                tool_names=["tool_A"],
                dependencies=[],
            ),
            PlanStep(
                id="step_B",
                name="Step B",
                description="Execute step B",
                tool_names=["tool_B"],
                dependencies=["step_A"],
            ),
            PlanStep(
                id="step_C",
                name="Step C",
                description="Execute step C",
                tool_names=["tool_C"],
                dependencies=["step_B"],
            ),
        ]

        plan = ExecutionPlan(
            id="test_plan_sequential",
            goal="Test sequential execution A -> B -> C",
            steps=steps,
        )

        # Create plan executor with proper mocks
        mock_llm = Mock(spec=BaseLLM)

        # Add required properties to the mock
        mock_llm.abilities = ["chat"]
        mock_llm.supports_thinking_mode = False

        # Track which tools have been called to prevent infinite loops
        _executed_tools = set()

        # Mock the LLM to return tool call responses that trigger tool execution
        async def mock_chat(messages, **kwargs):
            # Return a proper ReAct response that triggers tool execution
            # Extract the tool name from messages
            tool_name = None
            for msg in messages:
                content = msg.get("content", "")
                if "step A" in content or "Execute step A" in content:
                    tool_name = "tool_A"
                    break
                elif "step B" in content or "Execute step B" in content:
                    tool_name = "tool_B"
                    break
                elif "step C" in content or "Execute step C" in content:
                    tool_name = "tool_C"
                    break

            if tool_name:
                # Check if this tool was already executed — if so, return final answer
                if tool_name in _executed_tools:
                    return '{"type": "final_answer", "content": "Task completed successfully", "answer": "Task completed successfully", "reasoning": "The task has been completed"}'
                _executed_tools.add(tool_name)
                # Return a proper ReAct tool_call response without extra content field
                return (
                    '{"type": "tool_call", "reasoning": "I need to execute the tool for this step", "tool_name": "'
                    + tool_name
                    + '", "tool_args": {}}'
                )

            # Fallback to final answer response
            return '{"type": "final_answer", "content": "Task completed successfully", "answer": "Task completed successfully", "reasoning": "The task has been completed"}'

        mock_llm.chat = mock_chat

        # Mock stream_chat to work with the ReAct pattern
        async def mock_stream_chat(**kwargs):
            import json

            from xagent.core.model.chat.types import ChunkType, StreamChunk

            # Get response from chat mock
            content = await mock_llm.chat(**kwargs)

            # Try to parse as JSON to determine response type
            try:
                response_data = json.loads(content)
                if response_data.get("type") == "tool_call":
                    # Return native tool call format
                    tool_name = response_data.get("tool_name", "")
                    tool_args = response_data.get("tool_args", {})
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
                    return
            except (json.JSONDecodeError, AttributeError):
                pass

            # Yield as text stream chunk (for final_answer)
            yield StreamChunk(
                type=ChunkType.TOKEN,
                content=content,
                delta=content,
            )
            yield StreamChunk(type=ChunkType.END, finish_reason="stop")

        mock_llm.stream_chat = mock_stream_chat

        # Create async mock for tracer
        async def async_trace_event(*args, **kwargs):
            return "trace_id"

        tracer = MagicMock()
        tracer.trace_event = async_trace_event

        workspace = TaskWorkspace("test_task")
        plan_executor = PlanExecutor(
            llm=mock_llm,
            tracer=tracer,
            workspace=workspace,
        )

        # Execute the plan
        results = await plan_executor.execute_plan(plan, tool_map)

        # Verify execution order was sequential
        assert len(tool_a.execution_times) == 1, "Tool A should execute once"
        assert len(tool_b.execution_times) == 1, "Tool B should execute once"
        assert len(tool_c.execution_times) == 1, "Tool C should execute once"

        a_start, a_end = tool_a.execution_times[0]
        b_start, b_end = tool_b.execution_times[0]
        c_start, c_end = tool_c.execution_times[0]

        # Verify sequential execution: A must finish before B starts, B before C
        assert a_end <= b_start, (
            f"A should finish before B starts. A end: {a_end}, B start: {b_start}"
        )
        assert b_end <= c_start, (
            f"B should finish before C starts. B end: {b_end}, C start: {c_start}"
        )

        # Verify no concurrent execution between dependent steps
        assert not (a_start < b_start < a_end), (
            "A and B should not execute concurrently"
        )
        assert not (b_start < c_start < b_end), (
            "B and C should not execute concurrently"
        )

        # Verify results
        assert len(results) == 3, "Should have 3 execution results"
        assert all("step_id" in result for result in results), (
            "Each result should have step_id"
        )

    async def test_complex_dependency_scenarios(self):
        """Test various complex dependency scenarios to ensure proper execution order."""
        import asyncio
        import time
        from unittest.mock import MagicMock, Mock

        # Create mock tools with execution timing that implement full Tool interface
        from xagent.core.agent.pattern.dag_plan_execute.models import (
            ExecutionPlan,
            PlanStep,
        )
        from xagent.core.agent.pattern.dag_plan_execute.plan_executor import (
            PlanExecutor,
        )
        from xagent.core.model.chat.basic.base import BaseLLM
        from xagent.core.tools.adapters.vibe.function import FunctionTool
        from xagent.core.workspace import TaskWorkspace

        # Create async functions for the tools
        async def create_tool_function(name, execution_time):
            execution_times = []

            async def tool_func():
                start_time = time.time()
                await asyncio.sleep(execution_time)
                end_time = time.time()
                execution_times.append((start_time, end_time))
                return {"result": f"Result from {name}", "success": True}

            # Create the tool with proper interface
            tool = FunctionTool(
                tool_func,
                name=name,
                description=f"Mock tool {name}",
            )

            # Add execution_times as an attribute
            tool.execution_times = execution_times
            return tool

        # Test 1: Diamond-shaped dependencies (A -> B, A -> C, B -> D, C -> D)
        async def test_diamond_dependencies():
            tools = {}
            for name in ["A", "B", "C", "D"]:
                tools[name] = await create_tool_function(
                    f"tool_{name}", execution_time=0.1
                )

            tool_map = {f"tool_{name}": tool for name, tool in tools.items()}

            steps = [
                PlanStep(
                    id="step_A",
                    name="Step A",
                    description="Execute step A",
                    tool_names=["tool_A"],
                    dependencies=[],
                ),
                PlanStep(
                    id="step_B",
                    name="Step B",
                    description="Execute step B",
                    tool_names=["tool_B"],
                    dependencies=["step_A"],
                ),
                PlanStep(
                    id="step_C",
                    name="Step C",
                    description="Execute step C",
                    tool_names=["tool_C"],
                    dependencies=["step_A"],
                ),
                PlanStep(
                    id="step_D",
                    name="Step D",
                    description="Execute step D",
                    tool_names=["tool_D"],
                    dependencies=["step_B", "step_C"],
                ),
            ]

            plan = ExecutionPlan(
                id="test_diamond",
                goal="Test diamond dependencies",
                steps=steps,
            )

            mock_llm = Mock(spec=BaseLLM)

            # Add required properties to the mock
            mock_llm.abilities = ["chat"]
            mock_llm.supports_thinking_mode = False

            # Mock the LLM to return tool call responses that trigger tool execution
            async def mock_chat(messages, **kwargs):
                # Return a tool call response that directly executes the tool
                # Look at the most recent user message to find the step description
                tool_name = None

                # Check the most recent user message for step descriptions
                for message in reversed(messages):
                    if message.get("role") == "user":
                        content = message.get("content", "")
                        # Match exact step descriptions with correct tool names
                        if (
                            "Execute step A" in content
                            and "A1" not in content
                            and "A2" not in content
                        ):
                            tool_name = "tool_A"
                        elif (
                            "Execute step B" in content
                            and "B1" not in content
                            and "B2" not in content
                        ):
                            tool_name = "tool_B"
                        elif "Execute step C" in content:
                            tool_name = "tool_C"
                        elif "Execute step D" in content:
                            tool_name = "tool_D"
                        elif "Execute step A1" in content:
                            tool_name = "tool_A1"
                        elif "Execute step A2" in content:
                            tool_name = "tool_A2"
                        elif "Execute step B1" in content:
                            tool_name = "tool_B1"
                        elif "Execute step B2" in content:
                            tool_name = "tool_B2"
                        elif "Execute step E" in content:
                            tool_name = "tool_E"
                        elif "Execute step F" in content:
                            tool_name = "tool_F"
                        break

                if tool_name:
                    # Return a proper ReAct tool_call response without extra content field
                    return (
                        '{"type": "tool_call", "reasoning": "I need to execute the tool for this step", "tool_name": "'
                        + tool_name
                        + '", "tool_args": {}}'
                    )

                # Fallback to final answer response
                return '{"type": "final_answer", "content": "Task completed successfully", "answer": "Task completed successfully", "reasoning": "The task has been completed"}'

            mock_llm.chat = mock_chat

            # Mock stream_chat using the helper function
            mock_llm.stream_chat = create_mock_stream_chat(mock_llm)

            # Create async mock for tracer
            async def async_trace_event(*args, **kwargs):
                return "trace_id"

            tracer = MagicMock()
            tracer.trace_event = async_trace_event

            workspace = TaskWorkspace("test_task")
            plan_executor = PlanExecutor(
                llm=mock_llm,
                tracer=tracer,
                workspace=workspace,
            )

            results = await plan_executor.execute_plan(plan, tool_map)

            # Verify execution order
            a_start, a_end = tools["A"].execution_times[0]
            b_start, b_end = tools["B"].execution_times[0]
            c_start, c_end = tools["C"].execution_times[0]
            d_start, d_end = tools["D"].execution_times[0]

            # A must finish before B and C start
            assert a_end <= b_start, "A should finish before B starts"
            assert a_end <= c_start, "A should finish before C starts"

            # B and C can execute in parallel (no dependency between them)
            # But D must wait for both B and C to finish
            assert b_end <= d_start, "B should finish before D starts"
            assert c_end <= d_start, "C should finish before D starts"

            assert len(results) == 4, "Should have 4 execution results"

        # Test 2: Multiple independent chains that converge
        async def test_converging_chains():
            tools = {}
            for name in ["A1", "A2", "B1", "B2", "C"]:
                tools[name] = await create_tool_function(
                    f"tool_{name}", execution_time=0.1
                )

            tool_map = {f"tool_{name}": tool for name, tool in tools.items()}

            steps = [
                PlanStep(
                    id="step_A1",
                    name="Step A1",
                    description="Execute step A1",
                    tool_names=["tool_A1"],
                    dependencies=[],
                ),
                PlanStep(
                    id="step_A2",
                    name="Step A2",
                    description="Execute step A2",
                    tool_names=["tool_A2"],
                    dependencies=["step_A1"],
                ),
                PlanStep(
                    id="step_B1",
                    name="Step B1",
                    description="Execute step B1",
                    tool_names=["tool_B1"],
                    dependencies=[],
                ),
                PlanStep(
                    id="step_B2",
                    name="Step B2",
                    description="Execute step B2",
                    tool_names=["tool_B2"],
                    dependencies=["step_B1"],
                ),
                PlanStep(
                    id="step_C",
                    name="Step C",
                    description="Execute step C",
                    tool_names=["tool_C"],
                    dependencies=["step_A2", "step_B2"],
                ),
            ]

            plan = ExecutionPlan(
                id="test_converging",
                goal="Test converging chains",
                steps=steps,
            )

            mock_llm = Mock(spec=BaseLLM)

            # Mock the LLM to return tool call responses that trigger tool execution
            async def mock_chat(messages, **kwargs):
                # Return a tool call response that directly executes the tool
                # Look at all messages to find the step description
                tool_name = None

                # Check all messages for step descriptions
                for message in reversed(messages):
                    if message.get("role") == "user":
                        content = message.get("content", "")
                        # More specific matches first to avoid incorrect matching
                        if "Execute step A1" in content:
                            tool_name = "tool_A1"
                        elif "Execute step A2" in content:
                            tool_name = "tool_A2"
                        elif "Execute step B1" in content:
                            tool_name = "tool_B1"
                        elif "Execute step B2" in content:
                            tool_name = "tool_B2"
                        elif "Execute step C" in content:
                            tool_name = "tool_C"
                        elif "Execute step A" in content:
                            tool_name = "tool_A"
                        elif "Execute step B" in content:
                            tool_name = "tool_B"
                        elif "Execute step D" in content:
                            tool_name = "tool_D"
                        elif "Execute step E" in content:
                            tool_name = "tool_E"
                        elif "Execute step F" in content:
                            tool_name = "tool_F"
                        break

                if tool_name:
                    # Return a proper ReAct tool_call response without extra content field
                    return (
                        '{"type": "tool_call", "reasoning": "I need to execute the tool for this step", "tool_name": "'
                        + tool_name
                        + '", "tool_args": {}}'
                    )

                # Fallback to final answer response
                return '{"type": "final_answer", "content": "Task completed successfully", "answer": "Task completed successfully", "reasoning": "The task has been completed"}'

            mock_llm.chat = mock_chat

            # Mock stream_chat using the helper function
            mock_llm.stream_chat = create_mock_stream_chat(mock_llm)

            # Create async mock for tracer
            async def async_trace_event(*args, **kwargs):
                return "trace_id"

            tracer = MagicMock()
            tracer.trace_event = async_trace_event

            workspace = TaskWorkspace("test_task")
            plan_executor = PlanExecutor(
                llm=mock_llm,
                tracer=tracer,
                workspace=workspace,
            )

            results = await plan_executor.execute_plan(plan, tool_map)

            # Verify execution order
            a1_start, a1_end = tools["A1"].execution_times[0]
            a2_start, a2_end = tools["A2"].execution_times[0]
            b1_start, b1_end = tools["B1"].execution_times[0]
            b2_start, b2_end = tools["B2"].execution_times[0]
            c_start, c_end = tools["C"].execution_times[0]

            # Chain A: A1 -> A2
            assert a1_end <= a2_start, "A1 should finish before A2 starts"

            # Chain B: B1 -> B2
            assert b1_end <= b2_start, "B1 should finish before B2 starts"

            # Chains A and B can execute in parallel (no dependencies between them)
            # But C must wait for both A2 and B2 to finish
            assert a2_end <= c_start, "A2 should finish before C starts"
            assert b2_end <= c_start, "B2 should finish before C starts"

            assert len(results) == 5, "Should have 5 execution results"

        # Test 3: Deep nested dependencies
        async def test_deep_nested_dependencies():
            tools = {}
            for name in ["A", "B", "C", "D", "E", "F"]:
                tools[name] = await create_tool_function(
                    f"tool_{name}", execution_time=0.05
                )

            tool_map = {f"tool_{name}": tool for name, tool in tools.items()}

            steps = [
                PlanStep(
                    id="step_A",
                    name="Step A",
                    description="Execute step A",
                    tool_names=["tool_A"],
                    dependencies=[],
                ),
                PlanStep(
                    id="step_B",
                    name="Step B",
                    description="Execute step B",
                    tool_names=["tool_B"],
                    dependencies=["step_A"],
                ),
                PlanStep(
                    id="step_C",
                    name="Step C",
                    description="Execute step C",
                    tool_names=["tool_C"],
                    dependencies=["step_B"],
                ),
                PlanStep(
                    id="step_D",
                    name="Step D",
                    description="Execute step D",
                    tool_names=["tool_D"],
                    dependencies=["step_C"],
                ),
                PlanStep(
                    id="step_E",
                    name="Step E",
                    description="Execute step E",
                    tool_names=["tool_E"],
                    dependencies=["step_D"],
                ),
                PlanStep(
                    id="step_F",
                    name="Step F",
                    description="Execute step F",
                    tool_names=["tool_F"],
                    dependencies=["step_E"],
                ),
            ]

            plan = ExecutionPlan(
                id="test_deep_nested",
                goal="Test deep nested dependencies",
                steps=steps,
            )

            mock_llm = Mock(spec=BaseLLM)

            # Mock the LLM to return tool call responses that trigger tool execution
            async def mock_chat(messages, **kwargs):
                # Return a tool call response that directly executes the tool
                # Look at all messages to find the step description
                tool_name = None

                # Check all messages for step descriptions
                for message in reversed(messages):
                    if message.get("role") == "user":
                        content = message.get("content", "")
                        if "Execute step A" in content:
                            tool_name = "tool_A"
                        elif "Execute step B" in content:
                            tool_name = "tool_B"
                        elif "Execute step C" in content:
                            tool_name = "tool_C"
                        elif "Execute step D" in content:
                            tool_name = "tool_D"
                        elif "Execute step A1" in content:
                            tool_name = "tool_A1"
                        elif "Execute step A2" in content:
                            tool_name = "tool_A2"
                        elif "Execute step B1" in content:
                            tool_name = "tool_B1"
                        elif "Execute step B2" in content:
                            tool_name = "tool_B2"
                        elif "Execute step E" in content:
                            tool_name = "tool_E"
                        elif "Execute step F" in content:
                            tool_name = "tool_F"
                        break

                if tool_name:
                    # Return a proper ReAct tool_call response without extra content field
                    return (
                        '{"type": "tool_call", "reasoning": "I need to execute the tool for this step", "tool_name": "'
                        + tool_name
                        + '", "tool_args": {}}'
                    )

                # Fallback to final answer response
                return '{"type": "final_answer", "content": "Task completed successfully", "answer": "Task completed successfully", "reasoning": "The task has been completed"}'

            mock_llm.chat = mock_chat

            # Mock stream_chat using the helper function
            mock_llm.stream_chat = create_mock_stream_chat(mock_llm)

            # Create async mock for tracer
            async def async_trace_event(*args, **kwargs):
                return "trace_id"

            tracer = MagicMock()
            tracer.trace_event = async_trace_event

            workspace = TaskWorkspace("test_task")
            plan_executor = PlanExecutor(
                llm=mock_llm,
                tracer=tracer,
                workspace=workspace,
            )

            results = await plan_executor.execute_plan(plan, tool_map)

            # Verify strict sequential execution
            execution_times = []
            for name in ["A", "B", "C", "D", "E", "F"]:
                start, end = tools[name].execution_times[0]
                execution_times.append((name, start, end))

            # Check that each step finishes before the next one starts
            for i in range(len(execution_times) - 1):
                current_name, current_start, current_end = execution_times[i]
                next_name, next_start, next_end = execution_times[i + 1]

                assert current_end <= next_start, (
                    f"{current_name} should finish before {next_name} starts. "
                    f"{current_name} end: {current_end}, {next_name} start: {next_start}"
                )

            assert len(results) == 6, "Should have 6 execution results"

        # Run all complex dependency tests
        await test_diamond_dependencies()
        await test_converging_chains()
        await test_deep_nested_dependencies()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
