"""
Test ProcessService functionality.

This test requires xoscar to be installed.
"""

import asyncio

import pytest

from xagent.core.execution.service import ProcessService


@pytest.mark.asyncio
async def test_process_service_lifecycle():
    """Test ProcessService start and stop."""
    service = ProcessService(n_workers=2, address="localhost:12346")

    # Test start
    await service.start()
    assert service.status.value == "running"

    # Test health check
    is_healthy = await service.health_check()
    assert is_healthy is True

    # Test get_info
    info = service.get_info()
    assert info.name == "process"
    assert info.status.value == "running"
    assert info.resource_info["n_workers"] == 2

    # Test stop
    await service.stop()
    assert service.status.value == "stopped"


@pytest.mark.asyncio
async def test_python_execution():
    """Test Python code execution in isolated process."""
    service = ProcessService(n_workers=2, address="localhost:12347")

    await service.start()

    try:
        # Test simple Python execution
        result = await service.execute_python(
            code="print('Hello from isolated process!')\nresult = 2 + 2\nprint(f'2 + 2 = {result}')",
            timeout=10,
        )

        assert result.success is True
        assert "Hello from isolated process!" in result.output
        assert "2 + 2 = 4" in result.output
        assert result.return_code == 0

    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_python_execution_with_error():
    """Test Python code execution with syntax error."""
    service = ProcessService(n_workers=2, address="localhost:12348")

    await service.start()

    try:
        # Test Python code with error
        result = await service.execute_python(
            code="print('Before error')\nraise ValueError('Test error')",
            timeout=10,
        )

        assert result.success is False
        assert "ValueError: Test error" in result.error
        assert result.return_code == 1

    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_python_execution_timeout():
    """Test Python code execution with timeout."""
    service = ProcessService(n_workers=2, address="localhost:12349")

    await service.start()

    try:
        # Test Python code that exceeds timeout
        result = await service.execute_python(
            code="import time\ntime.sleep(10)\nprint('This should not print')",
            timeout=2,
        )

        assert result.success is False
        assert "timed out" in result.error.lower()
        assert service.get_info().resource_info["active_actors"] == 0

    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_sub_pool_startup_timeout(monkeypatch):
    """Test ProcessService does not hang forever when sub-pool startup stalls."""
    from xagent.core.execution.service import ServiceStatus

    class HangingPool:
        async def append_sub_pool(self, **kwargs):
            await asyncio.sleep(10)

        async def remove_sub_pool(self, *args, **kwargs):
            raise AssertionError("remove_sub_pool should not run without an address")

    service = ProcessService(n_workers=1, address="localhost:12356")
    service._status = ServiceStatus.RUNNING
    service._pool = HangingPool()

    result = await service.execute_python("print('never runs')", timeout=0.1)

    assert result.success is False
    assert "timed out during sub-pool startup" in result.error.lower()
    assert service.get_info().resource_info["active_actors"] == 0


@pytest.mark.asyncio
async def test_command_execution():
    """Test shell command execution in isolated process."""
    service = ProcessService(n_workers=2, address="localhost:12350")

    await service.start()

    try:
        # Test simple command
        result = await service.execute_command(
            command="echo 'Hello from command!'",
            timeout=10,
        )

        assert result.success is True
        assert "Hello from command!" in result.output
        assert result.return_code == 0

    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_command_execution_preserves_shell_semantics(tmp_path):
    """Test process command execution preserves existing shell semantics."""
    service = ProcessService(n_workers=2, address="localhost:12355")

    await service.start()

    try:
        pipe_result = await service.execute_command(
            command="echo hello | wc -c",
            workspace=str(tmp_path),
            timeout=10,
        )
        assert pipe_result.success is True
        assert pipe_result.output.strip() == "6"

        redirect_result = await service.execute_command(
            command="echo hi > out.txt",
            workspace=str(tmp_path),
            timeout=10,
        )
        assert redirect_result.success is True
        assert (tmp_path / "out.txt").read_text().strip() == "hi"

        chain_result = await service.execute_command(
            command=f"cd {tmp_path} && pwd",
            timeout=10,
        )
        assert chain_result.success is True
        assert chain_result.output.strip() == str(tmp_path)

    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_javascript_execution():
    """Test JavaScript code execution in isolated process."""
    service = ProcessService(n_workers=2, address="localhost:12351")

    await service.start()

    try:
        # Test simple JavaScript execution
        result = await service.execute_javascript(
            code="console.log('Hello from JavaScript!');\nconst result = 2 + 2;\nconsole.log(`2 + 2 = ${result}`);",
            timeout=10,
        )

        assert result.success is True
        assert "Hello from JavaScript!" in result.output
        assert "2 + 2 = 4" in result.output
        assert result.return_code == 0

    finally:
        await service.stop()
