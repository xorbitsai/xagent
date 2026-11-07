"""
Test generic sandboxed tool wrapper

Test tool execution functionality in sandbox
"""

import asyncio

import pytest

try:
    import boxlite  # noqa: F401
except ImportError:
    pytest.skip(
        "boxlite not installed, skipping sandbox integration tests",
        allow_module_level=True,
    )

from xagent.core.tools.adapters.vibe.command_executor import CommandExecutorToolForBasic
from xagent.core.tools.adapters.vibe.javascript_executor import (
    JavaScriptExecutorToolForBasic,
)
from xagent.core.tools.adapters.vibe.python_executor import PythonExecutorToolForBasic
from xagent.core.tools.adapters.vibe.sandboxed_tool.sandboxed_tool_wrapper import (
    create_sandboxed_tool,
)
from xagent.sandbox import DEFAULT_SANDBOX_IMAGE
from xagent.sandbox.base import SandboxConfig
from xagent.sandbox.boxlite_sandbox import (
    BoxliteSandboxService,
    MemBoxliteStore,
    SandboxTemplate,
)


@pytest.fixture(scope="module")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


def _check_boxlite_available() -> bool:
    """Check if boxlite is available"""
    try:
        try:
            boxlite.Boxlite.default()
            print("\n✓ Boxlite initialized successfully")
            return True
        except BaseException as e:
            error_msg = f"✗ Boxlite initialization failed: {type(e).__name__}: {e}"
            print(f"\n{error_msg}")
            return False
    except ImportError as e:
        error_msg = f"✗ Boxlite import failed: {type(e).__name__}: {e}"
        print(f"\n{error_msg}")
        return False


requires_boxlite = pytest.mark.skipif(
    not _check_boxlite_available(), reason="Requires boxlite runtime"
)


async def _create_sandbox(service: BoxliteSandboxService, name: str):
    """Helper function: create sandbox instance"""
    template = SandboxTemplate()
    template.type = "image"
    template.image = DEFAULT_SANDBOX_IMAGE
    config = SandboxConfig(
        cpus=1,
        memory=1024,
        volumes=[],
    )
    sandbox = await service.get_or_create(name, template=template, config=config)
    return sandbox


@requires_boxlite
class TestSandboxedToolWrapper:
    """Test sandboxed tool wrapper"""

    @pytest.mark.asyncio(loop_scope="module")
    async def test_python_executor_tool_sandboxed(self):
        """Test Python executor tool running in sandbox"""
        print("\n=== Test sandboxed Python executor tool ===")

        service = BoxliteSandboxService(MemBoxliteStore())
        sandbox_name = "test_python_executor_sandboxed"

        try:
            # Cleanup
            try:
                await service.delete(sandbox_name)
            except Exception:
                pass

            # Create sandbox
            sandbox = await _create_sandbox(service, sandbox_name)

            # Create sandboxed tool
            python_executor = PythonExecutorToolForBasic(None)
            sandboxed_executor = await create_sandboxed_tool(
                tool=python_executor,
                sandbox=sandbox,
            )

            # Test simple code execution
            result = await sandboxed_executor.run_json_async(
                {
                    "code": "print('Hello from sandbox')",
                    "capture_output": True,
                }
            )
            print(f"Execution result: {result}")
            assert result["success"] is True
            assert "Hello from sandbox" in result["output"]

            # Test calculation
            result = await sandboxed_executor.run_json_async(
                {
                    "code": "result = 10 * 5\nprint(f'Result: {result}')",
                    "capture_output": True,
                }
            )
            print(f"Calculation result: {result}")
            assert result["success"] is True
            assert "50" in result["output"]

            print("✅ Python executor tool sandbox test passed")

        finally:
            try:
                await service.delete(sandbox_name)
            except Exception:
                pass

    @pytest.mark.asyncio(loop_scope="module")
    async def test_multiple_tools_same_sandbox(self):
        """Test multiple tools sharing the same sandbox"""
        print("\n=== Test multiple tools sharing sandbox ===")

        service = BoxliteSandboxService(MemBoxliteStore())
        sandbox_name = "test_shared_sandbox"

        try:
            # Cleanup
            try:
                await service.delete(sandbox_name)
            except Exception:
                pass

            # Create sandbox
            sandbox = await _create_sandbox(service, sandbox_name)

            # Create first sandboxed tool
            python_executor = PythonExecutorToolForBasic(None)
            sandboxed_py1 = await create_sandboxed_tool(
                tool=python_executor,
                sandbox=sandbox,
            )

            # Create second tool, reusing the same sandbox
            sandboxed_py2 = await create_sandboxed_tool(
                tool=python_executor,
                sandbox=sandbox,
            )

            # Test first tool
            py_result1 = await sandboxed_py1.run_json_async(
                {
                    "code": "print('Tool 1 works!')",
                    "capture_output": True,
                }
            )
            print(f"Tool 1 result: {py_result1}")
            assert py_result1["success"] is True
            assert "Tool 1 works!" in py_result1["output"]

            # Test second tool
            py_result2 = await sandboxed_py2.run_json_async(
                {
                    "code": "print('Tool 2 works!')",
                    "capture_output": True,
                }
            )
            print(f"Tool 2 result: {py_result2}")
            assert py_result2["success"] is True
            assert "Tool 2 works!" in py_result2["output"]

            print("✅ Shared sandbox test passed")

        finally:
            try:
                await service.delete(sandbox_name)
            except Exception:
                pass

    @pytest.mark.asyncio(loop_scope="module")
    async def test_tool_with_error_handling(self):
        """Test tool error handling"""
        print("\n=== Test tool error handling ===")

        service = BoxliteSandboxService(MemBoxliteStore())
        sandbox_name = "test_error_handling"

        try:
            # Cleanup
            try:
                await service.delete(sandbox_name)
            except Exception:
                pass

            # Create sandbox
            sandbox = await _create_sandbox(service, sandbox_name)

            # Create sandboxed tool
            python_executor = PythonExecutorToolForBasic(None)
            sandboxed_executor = await create_sandboxed_tool(
                tool=python_executor,
                sandbox=sandbox,
            )

            # Test syntax error
            result = await sandboxed_executor.run_json_async(
                {
                    "code": "print('missing quote)",
                    "capture_output": True,
                }
            )
            print(f"Error handling result: {result}")
            # Python executor should return error information
            assert result["success"] is False
            assert len(result["error"]) > 0

            print("✅ Error handling test passed")

        finally:
            try:
                await service.delete(sandbox_name)
            except Exception:
                pass

    @pytest.mark.asyncio(loop_scope="module")
    async def test_tool_metadata_preservation(self):
        """Test if tool metadata is correctly preserved"""
        print("\n=== Test tool metadata preservation ===")

        service = BoxliteSandboxService(MemBoxliteStore())
        sandbox_name = "test_metadata"

        try:
            # Cleanup
            try:
                await service.delete(sandbox_name)
            except Exception:
                pass

            # Create sandbox
            sandbox = await _create_sandbox(service, sandbox_name)

            # Create sandboxed tool
            python_executor = PythonExecutorToolForBasic(None)
            sandboxed_executor = await create_sandboxed_tool(
                tool=python_executor,
                sandbox=sandbox,
            )

            # Check metadata - should be identical to original tool
            assert sandboxed_executor.name == python_executor.name
            assert sandboxed_executor.description == python_executor.description
            assert sandboxed_executor.tags == python_executor.tags
            assert (
                sandboxed_executor.metadata.category
                == python_executor.metadata.category
            )
            assert (
                sandboxed_executor.metadata.visibility
                == python_executor.metadata.visibility
            )

            # Check argument and return types
            assert sandboxed_executor.args_type() == python_executor.args_type()
            assert sandboxed_executor.return_type() == python_executor.return_type()

            print("✅ Metadata preservation test passed")

        finally:
            try:
                await service.delete(sandbox_name)
            except Exception:
                pass


@requires_boxlite
class TestTools:
    @pytest.mark.asyncio(loop_scope="module")
    async def test_execution_runtime_has_no_platform_source(self):
        service = BoxliteSandboxService(MemBoxliteStore())
        sandbox_name = "test_isolated_tool_runtime"
        try:
            sandbox = await _create_sandbox(service, sandbox_name)
            for path in ("/app/src", "/app/tests"):
                assert (await sandbox.exec("test", "-e", path)).exit_code != 0
            tool = await create_sandboxed_tool(
                PythonExecutorToolForBasic(None), sandbox
            )
            result = await tool.run_json_async(
                {
                    "code": "import importlib.util; print(importlib.util.find_spec('xagent'))"
                }
            )
            assert result["success"], result
            assert "None" in result["output"]
            command_tool = await create_sandboxed_tool(
                CommandExecutorToolForBasic(None), sandbox
            )
            command_result = await command_tool.run_json_async(
                {
                    "command": "test ! -e /app/src && test ! -e /app/tests && printf isolated-shell"
                }
            )
            assert command_result["success"], command_result
            assert "isolated-shell" in command_result["output"]
            js_tool = await create_sandboxed_tool(
                JavaScriptExecutorToolForBasic(None), sandbox
            )
            js_result = await js_tool.run_json_async(
                {
                    "code": "const fs = require('fs'); if (fs.existsSync('/app/src') || "
                    "fs.existsSync('/app/tests')) throw new Error('source visible'); console.log(6 * 7);"
                }
            )
            assert js_result["success"], js_result
            assert "42" in js_result["output"]
        finally:
            await service.delete(sandbox_name)


@requires_boxlite
class TestSandboxVsLocal:
    """Compare sandbox vs local execution results"""

    @pytest.mark.asyncio(loop_scope="module")
    async def test_compare_sandbox_vs_local(self):
        """
        Compare sandbox and local execution for the same inputs.
        Both should return identical results.
        """
        print("\n=== Compare sandbox vs local execution ===")

        service = BoxliteSandboxService(MemBoxliteStore())
        sandbox_name = "test_compare_sandbox_local"

        test_cases = [
            {
                "name": "import + expression (no print)",
                "args": {"code": "import math\nmath.factorial(50)"},
            },
            {
                "name": "print output",
                "args": {"code": "print('hello world')", "capture_output": True},
            },
            {
                "name": "calculation with print",
                "args": {
                    "code": "x = 2 + 3\nprint(f'result={x}')",
                    "capture_output": True,
                },
            },
            {
                "name": "syntax error",
                "args": {"code": "print('missing quote)", "capture_output": True},
            },
            {
                "name": "variable assignment only",
                "args": {"code": "x = 42\ny = 'hello'"},
            },
        ]

        try:
            # Cleanup
            try:
                await service.delete(sandbox_name)
            except Exception:
                pass

            # Create sandbox
            sandbox = await _create_sandbox(service, sandbox_name)

            # Create tools
            python_tool = PythonExecutorToolForBasic(None)
            sandboxed_tool = await create_sandboxed_tool(
                tool=python_tool,
                sandbox=sandbox,
            )

            for tc in test_cases:
                print(f"\n--- {tc['name']} ---")
                print(f"  Input: {tc['args']}")

                # Local execution
                local_result = python_tool.run_json_sync(tc["args"])

                # Sandbox execution
                sandbox_result = await sandboxed_tool.run_json_async(tc["args"])

                print(f"  Local:   {local_result}")
                print(f"  Sandbox: {sandbox_result}")

                # Compare
                assert local_result["success"] == sandbox_result["success"], (
                    f"[{tc['name']}] success mismatch: "
                    f"local={local_result['success']}, sandbox={sandbox_result['success']}"
                )

                if local_result["success"]:
                    # Normalize environment-specific paths for comparison
                    # e.g. module repr contains different paths on macOS vs Linux
                    import re

                    def _normalize(s: str) -> str:
                        # Replace module file paths: <module 'x' from '/path/to/x.so'>
                        s = re.sub(r"from '[^']*'", "from '<path>'", s)
                        # Normalize built-in modules: <module 'x' (built-in)>
                        s = re.sub(r"\(built-in\)", "from '<path>'", s)
                        return s

                    local_norm = _normalize(local_result["output"])
                    sandbox_norm = _normalize(sandbox_result["output"])
                    assert local_norm == sandbox_norm, (
                        f"[{tc['name']}] output mismatch:\n"
                        f"  local:   {local_result['output']!r}\n"
                        f"  sandbox: {sandbox_result['output']!r}"
                    )
                else:
                    # For errors, both should have non-empty error
                    assert len(local_result["error"]) > 0, (
                        f"[{tc['name']}] local error empty"
                    )
                    assert len(sandbox_result["error"]) > 0, (
                        f"[{tc['name']}] sandbox error empty"
                    )

                print("  ✅ Match")

            print("\n✅ All sandbox vs local comparisons passed")

        finally:
            try:
                await service.delete(sandbox_name)
            except Exception:
                pass
