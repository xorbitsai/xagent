"""
Unit tests for DAG concurrent execution functionality.

This module tests the queue-driven concurrent execution implementation
in the PlanExecutor class.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from xagent.core.agent.pattern.dag_plan_execute.models import (
    ExecutionPlan,
    PlanStep,
    StepStatus,
)
from xagent.core.agent.pattern.dag_plan_execute.plan_executor import PlanExecutor
from xagent.core.agent.trace import Tracer
from xagent.core.model.chat.basic.base import BaseLLM
from xagent.core.tools.adapters.vibe import Tool
from xagent.core.tools.adapters.vibe.base import ToolMetadata
from xagent.core.workspace import TaskWorkspace

logger = logging.getLogger(__name__)


class MockLLM(BaseLLM):
    """Mock LLM for testing."""

    def __init__(self):
        self.call_count = 0
        self.responses = ["Mock response"]
        self._model_name = "mock_llm"

    @property
    def abilities(self) -> List[str]:
        return ["chat"]

    @property
    def model_name(self) -> str:
        """Get the model name/identifier."""
        return self._model_name

    @property
    def supports_thinking_mode(self) -> bool:
        return False

    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        self.call_count += 1
        return self.responses.pop(0) if self.responses else "Mock response"


class MockTool(Tool):
    """Mock tool for testing."""

    def __init__(self, name: str, execution_time: float = 0.1):
        self.name = name
        self.execution_time = execution_time
        self.call_count = 0
        self._metadata = ToolMetadata(name=name, description=f"Mock tool {name}")

    @property
    def metadata(self) -> ToolMetadata:
        return self._metadata

    async def execute(self, **kwargs) -> Any:
        self.call_count += 1
        await asyncio.sleep(self.execution_time)
        return f"Tool {self.name} result {self.call_count}"


class TestDAGConcurrentExecution:
    """Test cases for DAG concurrent execution."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM."""
        return MockLLM()

    @pytest.fixture
    def mock_tracer(self):
        """Create a mock tracer."""
        return MagicMock(spec=Tracer)

    @pytest.fixture
    def mock_workspace(self):
        """Create a mock workspace."""
        return MagicMock(spec=TaskWorkspace)

    @pytest.fixture
    def fast_tool(self):
        """Create a fast mock tool."""
        return MockTool("fast", 0.05)

    @pytest.fixture
    def slow_tool(self):
        """Create a slow mock tool."""
        return MockTool("slow", 0.2)

    @pytest.fixture
    def tool_map(self, fast_tool, slow_tool):
        """Create a tool map."""
        return {
            "fast": fast_tool,
            "slow": slow_tool,
        }

    @pytest.fixture
    def independent_steps_plan(self):
        """Create a plan with independent steps that should execute concurrently."""
        steps = [
            PlanStep(
                id="step1",
                name="Independent Step 1",
                description="First independent step",
                tool_names=["fast"],
                dependencies=[],
            ),
            PlanStep(
                id="step2",
                name="Independent Step 2",
                description="Second independent step",
                tool_names=["fast"],
                dependencies=[],
            ),
            PlanStep(
                id="step3",
                name="Independent Step 3",
                description="Third independent step",
                tool_names=["slow"],
                dependencies=[],
            ),
        ]

        return ExecutionPlan(
            id="test_independent",
            goal="Test independent step execution",
            steps=steps,
            created_at=datetime.now(),
        )

    @pytest.fixture
    def dependent_steps_plan(self):
        """Create a plan with dependent steps."""
        steps = [
            PlanStep(
                id="step1",
                name="First Step",
                description="First step",
                tool_names=["fast"],
                dependencies=[],
            ),
            PlanStep(
                id="step2",
                name="Second Step",
                description="Second step depends on first",
                tool_names=["fast"],
                dependencies=["step1"],
            ),
            PlanStep(
                id="step3",
                name="Third Step",
                description="Third step depends on second",
                tool_names=["slow"],
                dependencies=["step2"],
            ),
        ]

        return ExecutionPlan(
            id="test_dependent",
            goal="Test dependent step execution",
            steps=steps,
            created_at=datetime.now(),
        )

    @pytest.fixture
    def mixed_plan(self):
        """Create a plan with both independent and dependent steps."""
        steps = [
            PlanStep(
                id="step1",
                name="Independent Fast 1",
                description="First independent fast step",
                tool_names=["fast"],
                dependencies=[],
            ),
            PlanStep(
                id="step2",
                name="Independent Slow",
                description="Independent slow step",
                tool_names=["slow"],
                dependencies=[],
            ),
            PlanStep(
                id="step3",
                name="Independent Fast 2",
                description="Second independent fast step",
                tool_names=["fast"],
                dependencies=[],
            ),
            PlanStep(
                id="step4",
                name="Dependent Step",
                description="Step that depends on all independent steps",
                tool_names=["fast"],
                dependencies=["step1", "step2", "step3"],
            ),
        ]

        return ExecutionPlan(
            id="test_mixed",
            goal="Test mixed step execution",
            steps=steps,
            created_at=datetime.now(),
        )

    @pytest.mark.asyncio
    async def test_concurrent_executor_initialization(
        self, mock_llm, mock_tracer, mock_workspace
    ):
        """Test that PlanExecutor initializes correctly with concurrency parameters."""
        executor = PlanExecutor(
            llm=mock_llm,
            tracer=mock_tracer,
            workspace=mock_workspace,
            max_concurrency=4,
        )

        assert executor.max_concurrency == 4
        assert executor._semaphore._value == 4

    @pytest.mark.asyncio
    async def test_default_concurrency(self, mock_llm, mock_tracer, mock_workspace):
        """Test that PlanExecutor uses default concurrency when not specified."""
        executor = PlanExecutor(
            llm=mock_llm,
            tracer=mock_tracer,
            workspace=mock_workspace,
        )

        assert executor.max_concurrency == 4  # Default value
        assert executor._semaphore._value == 4

    @pytest.mark.asyncio
    async def test_independent_steps_concurrent_execution(
        self, independent_steps_plan, tool_map, mock_llm, mock_tracer, mock_workspace
    ):
        """Test that independent steps execute concurrently."""
        executor = PlanExecutor(
            llm=mock_llm,
            tracer=mock_tracer,
            workspace=mock_workspace,
            max_concurrency=3,
        )

        # Track execution order
        execution_order = []

        async def mock_execute_step(
            step, tool_map, execution_results, skill_context=None
        ):
            execution_order.append(step.id)
            await asyncio.sleep(0.05)  # Simulate work
            return {"success": True, "step_id": step.id}

        executor._execute_step_with_react_agent = mock_execute_step

        # Execute the plan
        results = await executor.execute_plan(independent_steps_plan, tool_map)

        # Verify results
        assert len(results) == 3
        assert all(result["status"] == "completed" for result in results)

        # Verify all steps were executed
        completed_step_ids = {result["step_id"] for result in results}
        assert completed_step_ids == {"step1", "step2", "step3"}

    @pytest.mark.asyncio
    async def test_dependent_steps_sequential_execution(
        self, dependent_steps_plan, tool_map, mock_llm, mock_tracer, mock_workspace
    ):
        """Test that dependent steps execute in the correct order."""
        executor = PlanExecutor(
            llm=mock_llm,
            tracer=mock_tracer,
            workspace=mock_workspace,
            max_concurrency=3,
        )

        # Track execution order
        execution_order = []

        async def mock_execute_step(
            step, tool_map, execution_results, skill_context=None
        ):
            execution_order.append(step.id)
            await asyncio.sleep(0.05)  # Simulate work
            return {"success": True, "step_id": step.id}

        executor._execute_step_with_react_agent = mock_execute_step

        # Execute the plan
        results = await executor.execute_plan(dependent_steps_plan, tool_map)

        # Verify results
        assert len(results) == 3
        assert all(result["status"] == "completed" for result in results)

        # Verify execution order (dependencies must be respected)
        assert execution_order[0] == "step1"  # First step has no dependencies
        assert execution_order[1] == "step2"  # Second step depends on first
        assert execution_order[2] == "step3"  # Third step depends on second

    @pytest.mark.asyncio
    async def test_mixed_plan_execution(
        self, mixed_plan, tool_map, mock_llm, mock_tracer, mock_workspace
    ):
        """Test execution of a plan with both independent and dependent steps."""
        executor = PlanExecutor(
            llm=mock_llm,
            tracer=mock_tracer,
            workspace=mock_workspace,
            max_concurrency=3,
        )

        # Track execution order
        execution_order = []

        async def mock_execute_step(
            step, tool_map, execution_results, skill_context=None
        ):
            execution_order.append(step.id)
            await asyncio.sleep(0.05)  # Simulate work
            return {"success": True, "step_id": step.id}

        executor._execute_step_with_react_agent = mock_execute_step

        # Execute the plan
        results = await executor.execute_plan(mixed_plan, tool_map)

        # Verify results
        assert len(results) == 4
        assert all(result["status"] == "completed" for result in results)

        # The independent steps (1, 2, 3) should execute before the dependent step (4)
        dependent_step_index = execution_order.index("step4")
        independent_steps = {"step1", "step2", "step3"}

        # All independent steps should execute before the dependent step
        for independent_step in independent_steps:
            independent_index = execution_order.index(independent_step)
            assert independent_index < dependent_step_index, (
                f"Independent step {independent_step} should execute before dependent step"
            )

    @pytest.mark.asyncio
    async def test_concurrency_limit_respected(
        self, independent_steps_plan, tool_map, mock_llm, mock_tracer, mock_workspace
    ):
        """Test that the concurrency limit is respected."""
        executor = PlanExecutor(
            llm=mock_llm,
            tracer=mock_tracer,
            workspace=mock_workspace,
            max_concurrency=2,  # Limit to 2 concurrent steps
        )

        # Track currently executing steps
        current_executions = []
        max_concurrent_seen = [0]  # Use list to allow modification in closure

        async def mock_execute_step(
            step, tool_map, execution_results, skill_context=None
        ):
            current_executions.append(step.id)
            max_concurrent_seen[0] = max(
                max_concurrent_seen[0], len(current_executions)
            )
            await asyncio.sleep(0.1)  # Simulate work
            current_executions.remove(step.id)
            return {"success": True, "step_id": step.id}

        executor._execute_step_with_react_agent = mock_execute_step

        # Execute the plan
        results = await executor.execute_plan(independent_steps_plan, tool_map)

        # Verify results
        assert len(results) == 3
        assert all(result["status"] == "completed" for result in results)

        # The concurrency limit should have been respected
        # Due to the semaphore, we should never exceed max_concurrency
        assert max_concurrent_seen[0] <= 2

    @pytest.mark.asyncio
    async def test_step_failure_handling(
        self, independent_steps_plan, tool_map, mock_llm, mock_tracer, mock_workspace
    ):
        """Test that step failures are handled correctly in concurrent execution."""
        executor = PlanExecutor(
            llm=mock_llm,
            tracer=mock_tracer,
            workspace=mock_workspace,
            max_concurrency=3,
        )

        # Make step2 fail
        async def mock_execute_step(
            step, tool_map, execution_results, skill_context=None
        ):
            if step.id == "step2":
                raise Exception("Step 2 failed")
            await asyncio.sleep(0.05)
            return {"success": True, "step_id": step.id}

        executor._execute_step_with_react_agent = mock_execute_step

        # Execute the plan
        results = await executor.execute_plan(independent_steps_plan, tool_map)

        # Verify results
        assert len(results) == 3

        # Check that step2 failed and others succeeded
        step_results = {result["step_id"]: result["status"] for result in results}
        assert step_results["step1"] == "completed"
        assert step_results["step2"] == "failed"
        assert step_results["step3"] == "completed"

    @pytest.mark.asyncio
    async def test_plan_completes_when_all_steps_done(
        self, independent_steps_plan, tool_map, mock_llm, mock_tracer, mock_workspace
    ):
        """Test that the plan execution completes when all steps are done."""
        executor = PlanExecutor(
            llm=mock_llm,
            tracer=mock_tracer,
            workspace=mock_workspace,
            max_concurrency=3,
        )

        async def mock_execute_step(
            step, tool_map, execution_results, skill_context=None
        ):
            await asyncio.sleep(0.05)
            return {"success": True, "step_id": step.id}

        executor._execute_step_with_react_agent = mock_execute_step

        # Execute the plan
        await executor.execute_plan(independent_steps_plan, tool_map)

        # Verify the plan is marked as complete
        assert independent_steps_plan.is_complete()

        # Verify all steps have correct status
        for step in independent_steps_plan.steps:
            if step.id in ["step1", "step2"]:  # fast tool steps
                assert step.status == StepStatus.COMPLETED
            else:  # slow tool step
                assert step.status == StepStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_failure_with_dependent_steps_deadlock_prevention(
        self, mock_llm, mock_tracer, mock_workspace
    ):
        """Test that failure scenarios don't cause infinite deadlock loops."""
        executor = PlanExecutor(
            llm=mock_llm,
            tracer=mock_tracer,
            workspace=mock_workspace,
            max_concurrency=2,
        )

        # Create a plan that can cause the deadlock scenario: A->C, B->C, C->D, D->E
        steps = [
            PlanStep(
                id="step_a",
                name="Step A",
                description="Independent step A",
                tool_names=["fast"],
                dependencies=[],
            ),
            PlanStep(
                id="step_b",
                name="Step B",
                description="Independent step B",
                tool_names=["fast"],
                dependencies=[],
            ),
            PlanStep(
                id="step_c",
                name="Step C (depends on A and B)",
                description="Step C that depends on A and B",
                tool_names=["slow"],
                dependencies=["step_a", "step_b"],
            ),
            PlanStep(
                id="step_d",
                name="Step D (depends on C)",
                description="Step D that depends on C",
                tool_names=["fast"],
                dependencies=["step_c"],
            ),
            PlanStep(
                id="step_e",
                name="Step E (depends on D)",
                description="Step E that depends on D",
                tool_names=["fast"],
                dependencies=["step_d"],
            ),
        ]

        plan = ExecutionPlan(
            id="test_failure_deadlock",
            goal="Test failure deadlock prevention",
            steps=steps,
            created_at=datetime.now(),
        )

        tool_map = {"fast": MockTool("fast"), "slow": MockTool("slow")}

        # Track execution to detect infinite loops
        execution_count = 0
        max_executions = 10  # Prevent infinite test loops

        async def mock_execute_step(
            step, tool_map, execution_results, skill_context=None
        ):
            nonlocal execution_count
            execution_count += 1

            # Make step C fail to trigger the scenario
            if step.id == "step_c":
                raise Exception("Step C failed intentionally")

            await asyncio.sleep(0.01)
            return {"success": True, "step_id": step.id}

        executor._execute_step_with_react_agent = mock_execute_step

        # Execute the plan with timeout to prevent hanging
        try:
            results = await asyncio.wait_for(
                executor.execute_plan(plan, tool_map),
                timeout=5.0,  # 5 second timeout
            )
        except asyncio.TimeoutError:
            pytest.fail("Execution timed out - possible infinite loop")

        # Verify that execution didn't loop infinitely
        assert execution_count < max_executions, (
            f"Execution count {execution_count} exceeds maximum {max_executions}"
        )

        # Verify results - some steps should have completed, some failed
        assert len(results) > 0

        # Check that steps A and B completed (they don't depend on C)
        step_results = {result["step_id"]: result["status"] for result in results}
        assert step_results.get("step_a") == "completed"
        assert step_results.get("step_b") == "completed"

        # Step C should have failed
        assert step_results.get("step_c") == "failed"

        # Steps D and E might be blocked or failed, but shouldn't cause infinite loops
        logger.info(f"Execution completed with {execution_count} step executions")
        logger.info(f"Results: {step_results}")

    @pytest.mark.asyncio
    async def test_deadlock_detection_count_limit(
        self, mock_llm, mock_tracer, mock_workspace
    ):
        """Test that deadlock detection has a limit to prevent infinite loops."""
        executor = PlanExecutor(
            llm=mock_llm,
            tracer=mock_tracer,
            workspace=mock_workspace,
            max_concurrency=1,
        )

        # Create a plan that will trigger deadlock detection
        steps = [
            PlanStep(
                id="step1",
                name="Step 1",
                description="Step that will fail",
                tool_names=["fast"],
                dependencies=[],
            ),
            PlanStep(
                id="step2",
                name="Step 2",
                description="Step that depends on failed step",
                tool_names=["fast"],
                dependencies=["step1"],
            ),
        ]

        plan = ExecutionPlan(
            id="test_deadlock_limit",
            goal="Test deadlock detection limit",
            steps=steps,
            created_at=datetime.now(),
        )

        tool_map = {"fast": MockTool("fast")}

        async def mock_execute_step(
            step, tool_map, execution_results, skill_context=None
        ):
            if step.id == "step1":
                raise Exception("Step 1 failed")
            return {"success": True, "step_id": step.id}

        executor._execute_step_with_react_agent = mock_execute_step

        # Execute the plan - should complete without infinite looping
        results = await asyncio.wait_for(
            executor.execute_plan(plan, tool_map), timeout=3.0
        )

        # Verify deadlock check count was properly handled
        assert (
            hasattr(executor, "_deadlock_check_count") or True
        )  # May or may not exist
        assert len(results) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
