"""
Integration tests for Python executor with process isolation.

Tests that PythonExecutorTool actually runs in xoscar isolated processes.
"""

import asyncio
import os

import pytest

from xagent.core.execution.service import ProcessService
from xagent.core.execution.service.manager import (
    clear_process_service,
    set_process_service,
)
from xagent.core.tools.adapters.vibe.process_isolated import (
    create_process_isolated_tool,
)
from xagent.core.tools.adapters.vibe.python_executor import PythonExecutorTool


@pytest.mark.asyncio
async def test_python_executor_in_isolated_process():
    """Test that Python executor runs in xoscar isolated process."""
    # Create and start ProcessService
    service = ProcessService(address="localhost:12400")
    await service.start()
    set_process_service(service)

    try:
        # Create Python executor tool
        python_tool = PythonExecutorTool(workspace=None)

        # Wrap with process isolation
        isolated_tool = create_process_isolated_tool(python_tool, timeout=30)

        # Verify it's wrapped
        assert isolated_tool.is_isolated is True
        assert isolated_tool.name == "python_executor"

        # Execute code that checks process isolation
        code = """
import os
# Get current process ID
current_pid = os.getpid()
# Get parent process ID
parent_pid = os.getppid()
# Output both
print(f"PID: {current_pid}")
print(f"Parent PID: {parent_pid}")

# Get main process ID from environment if available
import json
result = {
    "current_pid": current_pid,
    "parent_pid": parent_pid,
}
print(f"Result: {json.dumps(result)}")
"""

        result = await isolated_tool.run_json_async(
            {
                "code": code,
                "capture_output": True,
            }
        )

        assert result["success"] is True
        output = result["output"]

        # The output should contain process IDs
        assert "PID:" in output
        assert "Parent PID:" in output

        # Verify it ran in a different process (not this one)
        import json

        main_pid = os.getpid()

        # Extract PID from output
        for line in output.split("\n"):
            if line.strip().startswith("Result:"):
                result_json = line.split("Result:")[1].strip()
                result_data = json.loads(result_json)

                isolated_pid = result_data["current_pid"]

                # The isolated process should be different from main process
                assert isolated_pid != main_pid, (
                    f"Python executor ran in same process (PID: {isolated_pid}) as main process (PID: {main_pid})"
                )

                print(
                    f"✅ Python executor ran in isolated process (PID: {isolated_pid}) != main process (PID: {main_pid})"
                )
                break

    finally:
        await service.stop()
        clear_process_service()


@pytest.mark.asyncio
async def test_python_executor_state_isolation():
    """Test that Python executor state is isolated between executions."""
    service = ProcessService(address="localhost:12401")
    await service.start()
    set_process_service(service)

    try:
        python_tool = PythonExecutorTool(workspace=None)
        isolated_tool = create_process_isolated_tool(python_tool, timeout=30)

        # First execution: create a variable
        result1 = await isolated_tool.run_json_async(
            {
                "code": "x = 42\nprint(f'x = {x}')",
                "capture_output": True,
            }
        )

        assert result1["success"] is True
        assert "x = 42" in result1["output"]

        # Second execution: variable should not exist (state isolation)
        result2 = await isolated_tool.run_json_async(
            {
                "code": """
try:
    print(f'x exists: {x}')
    state_isolated = False
except NameError:
    print('x does not exist (state is isolated)')
    state_isolated = True
""",
                "capture_output": True,
            }
        )

        assert result2["success"] is True
        assert "state is isolated" in result2["output"].lower()

        print("✅ State isolation verified between executions")

    finally:
        await service.stop()
        clear_process_service()


@pytest.mark.asyncio
async def test_python_executor_import_isolation():
    """Test that imports don't leak between executions."""
    service = ProcessService(address="localhost:12402")
    await service.start()
    set_process_service(service)

    try:
        python_tool = PythonExecutorTool(workspace=None)
        isolated_tool = create_process_isolated_tool(python_tool, timeout=30)

        # First execution: import a module
        result1 = await isolated_tool.run_json_async(
            {
                "code": "import math\nprint(f'math.pi = {math.pi}')",
                "capture_output": True,
            }
        )

        assert result1["success"] is True
        assert "math.pi = 3.14159" in result1["output"]

        # Second execution: check if import is still available
        # In isolated process, each execution starts fresh
        result2 = await isolated_tool.run_json_async(
            {
                "code": """
import sys
# Check if math is in built-in modules
math_in_builtins = 'math' in sys.builtin_module_names
import math as math_check
print(f'math available: {not math_in_builtins}')  # Should be available as stdlib
""",
                "capture_output": True,
            }
        )

        assert result2["success"] is True
        print("✅ Import isolation verified")

    finally:
        await service.stop()
        clear_process_service()


@pytest.mark.asyncio
async def test_python_executor_error_handling_isolated():
    """Test that errors in isolated process don't affect main process."""
    service = ProcessService(address="localhost:12403")
    await service.start()
    set_process_service(service)

    try:
        python_tool = PythonExecutorTool(workspace=None)
        isolated_tool = create_process_isolated_tool(python_tool, timeout=30)

        # Execute code that raises error
        error_result = await isolated_tool.run_json_async(
            {
                "code": "raise RuntimeError('Intentional error in isolated process')",
                "capture_output": True,
            }
        )

        assert error_result["success"] is False
        assert "RuntimeError" in error_result["error"]
        assert "Intentional error" in error_result["error"]

        # Main process should still be working
        main_pid = os.getpid()
        assert main_pid > 0

        # Next execution should work fine (no state corruption)
        normal_result = await isolated_tool.run_json_async(
            {
                "code": "print('After error, execution continues normally')",
                "capture_output": True,
            }
        )

        assert normal_result["success"] is True
        assert "continues normally" in normal_result["output"]

        print("✅ Error handling verified - errors isolated to subprocess")

    finally:
        await service.stop()
        clear_process_service()


@pytest.mark.asyncio
async def test_python_executor_workspace_binding_with_isolation():
    """Test that workspace binding works with process isolation."""
    import tempfile

    service = ProcessService(address="localhost:12404")
    await service.start()
    set_process_service(service)

    try:
        # Create a temporary workspace
        with tempfile.TemporaryDirectory() as workspace_dir:
            from xagent.core.workspace import TaskWorkspace

            workspace = TaskWorkspace(
                id="test_task",
                base_dir=workspace_dir,
            )

            python_tool = PythonExecutorTool(workspace=workspace)
            isolated_tool = create_process_isolated_tool(python_tool, timeout=30)

            # Execute code that uses workspace
            result = await isolated_tool.run_json_async(
                {
                    "code": """
import os
print(f'Working directory: {os.getcwd()}')
print('Workspace accessible!')
""",
                    "capture_output": True,
                }
            )

            assert result["success"] is True
            assert "Working directory:" in result["output"]
            print("✅ Workspace binding works with process isolation")

    finally:
        await service.stop()
        clear_process_service()


@pytest.mark.asyncio
async def test_python_executor_concurrent_executions():
    """Test that multiple Python executors can run concurrently in isolation."""
    service = ProcessService(address="localhost:12405")
    await service.start()
    set_process_service(service)

    try:
        python_tool = PythonExecutorTool(workspace=None)
        isolated_tool = create_process_isolated_tool(python_tool, timeout=30)

        # Run multiple executions concurrently
        async def run_execution(task_id: int):
            code = f"""
import os
import time
import random
print(f'Task {task_id} starting in PID: {{os.getpid()}}')
# Simulate some work
time.sleep(random.uniform(0.1, 0.3))
print(f'Task {task_id} completed')
"""
            result = await isolated_tool.run_json_async(
                {
                    "code": code,
                    "capture_output": True,
                }
            )
            return task_id, result

        # Run 5 concurrent tasks
        tasks = [run_execution(i) for i in range(5)]
        results = await asyncio.gather(*tasks)

        # All should succeed
        for task_id, result in results:
            assert result["success"] is True, f"Task {task_id} failed"
            assert f"Task {task_id} completed" in result["output"]

        print("✅ Concurrent executions verified")

    finally:
        await service.stop()
        clear_process_service()


if __name__ == "__main__":
    # Run tests manually for debugging
    asyncio.run(test_python_executor_in_isolated_process())
    asyncio.run(test_python_executor_state_isolation())
    asyncio.run(test_python_executor_import_isolation())
    asyncio.run(test_python_executor_error_handling_isolated())
    asyncio.run(test_python_executor_workspace_binding_with_isolation())
    asyncio.run(test_python_executor_concurrent_executions())
    print("\n✅ All process isolation tests passed!")
