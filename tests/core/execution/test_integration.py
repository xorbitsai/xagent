"""
Integration tests for process isolation with xagent tools.

Tests that the ProcessService integrates correctly with existing tools.
"""

import asyncio

import pytest

from xagent.core.execution.service import ProcessService
from xagent.core.execution.service.manager import (
    clear_process_service,
    set_process_service,
)


def _free_local_port() -> int:
    import socket

    with socket.socket() as sock:
        sock.bind(("localhost", 0))
        return int(sock.getsockname()[1])


@pytest.mark.asyncio
async def test_process_service_manager():
    """Test ProcessService global manager."""
    service = ProcessService(n_workers=2, address="localhost:12352")

    # Set as global service
    set_process_service(service)

    # Verify we can retrieve it
    from xagent.core.execution.service.manager import get_process_service

    retrieved_service = get_process_service()
    assert retrieved_service is service

    # Cleanup
    clear_process_service()
    assert get_process_service() is None


@pytest.mark.asyncio
async def test_execution_result_serialization():
    """Test ExecutionResult serialization."""
    from xagent.core.execution.service import ExecutionResult

    # Create result
    result = ExecutionResult(
        success=True,
        output="Test output",
        error="",
        return_code=0,
        metadata={"key": "value"},
        execution_time=1.5,
        memory_used_mb=100.0,
    )

    # Convert to dict
    result_dict = result.to_dict()
    assert result_dict["success"] is True
    assert result_dict["output"] == "Test output"
    assert result_dict["execution_time"] == 1.5

    # Convert back from dict
    restored_result = ExecutionResult.from_dict(result_dict)
    assert restored_result.success == result.success
    assert restored_result.output == result.output
    assert restored_result.execution_time == result.execution_time


@pytest.mark.asyncio
async def test_isolation_type_enum():
    """Test IsolationType enum."""
    from xagent.core.execution.service import IsolationType

    assert IsolationType.PROCESS.value == "process"
    assert IsolationType.SANDBOX.value == "sandbox"


@pytest.mark.asyncio
async def test_service_status_enum():
    """Test ServiceStatus enum."""
    from xagent.core.execution.service import ServiceStatus

    assert ServiceStatus.STARTING.value == "starting"
    assert ServiceStatus.RUNNING.value == "running"
    assert ServiceStatus.STOPPING.value == "stopping"
    assert ServiceStatus.STOPPED.value == "stopped"
    assert ServiceStatus.ERROR.value == "error"


@pytest.mark.asyncio
async def test_service_info():
    """Test ServiceInfo dataclass."""
    from xagent.core.execution.service import ServiceInfo, ServiceStatus

    info = ServiceInfo(
        name="test_service",
        status=ServiceStatus.RUNNING,
        resource_info={"workers": 4},
        metrics={"executions": 100},
    )

    # Convert to dict
    info_dict = info.to_dict()
    assert info_dict["name"] == "test_service"
    assert info_dict["status"] == "running"
    assert info_dict["resource_info"]["workers"] == 4
    assert info_dict["metrics"]["executions"] == 100


@pytest.mark.asyncio
async def test_process_service_not_started_error():
    """Test error when calling execute before starting service."""
    service = ProcessService(n_workers=2, address="localhost:12353")

    # Don't start the service
    with pytest.raises(RuntimeError, match="ProcessService not started"):
        await service.execute_python(code="print('test')")


@pytest.mark.asyncio
async def test_python_execution_with_workspace():
    """Test Python execution with workspace directory."""
    import tempfile

    service = ProcessService(n_workers=2, address="localhost:12354")

    await service.start()

    try:
        # Create a temporary workspace
        with tempfile.TemporaryDirectory() as workspace:
            # Test that workspace is accessible
            result = await service.execute_python(
                code="import os\nprint(f'CWD: {os.getcwd()}')\nprint('Files:', os.listdir('.'))",
                workspace=workspace,
                timeout=10,
            )

            assert result.success is True
            # The output should contain the workspace path
            assert workspace in result.output or "CWD:" in result.output

    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_tool_factory_process_isolated_python_executor_runs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
):
    """ToolFactory-created execute_python_code should run through process isolation."""
    import os

    import xagent.web.sandbox_manager as sandbox_manager_module
    from xagent.core.tools.adapters.vibe.basic_tools import create_basic_tools
    from xagent.core.tools.adapters.vibe.config import ToolConfig
    from xagent.core.tools.adapters.vibe.factory import ToolFactory, ToolRegistry

    clear_process_service()
    sandbox_manager_module._sandbox_manager = None
    sandbox_manager_module._sandbox_manager_initialized = True
    monkeypatch.setenv("XAGENT_PROCESS_ISOLATION_ENABLED", "true")

    async def _create_registered_tools(config):
        return await create_basic_tools(config)

    monkeypatch.setattr(
        ToolRegistry,
        "create_registered_tools",
        _create_registered_tools,
    )

    service = ProcessService(n_workers=1, address=f"localhost:{_free_local_port()}")
    await service.start()
    set_process_service(service)

    try:
        tools = await ToolFactory.create_all_tools(
            ToolConfig(
                {
                    "workspace": {
                        "task_id": "process_isolated_python_executor",
                        "base_dir": str(tmp_path),
                    },
                    "allowed_tools": ["execute_python_code"],
                    "basic_tools_enabled": True,
                    "file_tools_enabled": False,
                    "browser_tools_enabled": False,
                }
            )
        )
        assert len(tools) == 1

        tool = tools[0]
        assert tool.name == "execute_python_code"
        assert tool.is_isolated is True

        parent_pid = os.getpid()
        result = await asyncio.wait_for(
            tool.run_json_async(
                {
                    "code": "import os\nprint(os.getpid())\nprint('process-ok')",
                }
            ),
            timeout=30,
        )

        assert result["success"] is True
        assert "process-ok" in result["output"]
        child_pid = int(result["output"].splitlines()[0])
        assert child_pid != parent_pid
    finally:
        await service.stop()
        clear_process_service()
