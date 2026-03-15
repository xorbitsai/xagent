"""
Test generic sandboxed tool wrapper

Test tool execution functionality in sandbox
"""

import asyncio

import pytest

from src.xagent.core.tools.adapters.vibe.javascript_executor import (
    get_javascript_executor_tool,
)
from src.xagent.core.tools.adapters.vibe.python_executor import get_python_executor_tool
from src.xagent.core.tools.adapters.vibe.sandboxed_tool.sandboxed_tool_wrapper import (
    create_sandboxed_tool,
    upload_code_to_sandbox,
)
from src.xagent.sandbox.base import SandboxConfig
from src.xagent.sandbox.boxlite_sandbox import (
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
        import boxlite

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
    template.image = "python:slim"
    config = SandboxConfig(
        cpus=1,
        memory=1024,
    )
    sandbox = await service.get_or_create(name, template=template, config=config)
    await upload_code_to_sandbox(sandbox, contain_tests=True)
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
            python_executor = get_python_executor_tool(None)
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
            python_executor = get_python_executor_tool(None)
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
            python_executor = get_python_executor_tool(None)
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
            python_executor = get_python_executor_tool(None)
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

    @pytest.mark.asyncio(loop_scope="module")
    async def test_upload_code_to_sandbox(self):
        """Test upload code to sandbox"""
        print("\n=== Test upload code to sandbox ===")

        service = BoxliteSandboxService(MemBoxliteStore())
        sandbox_name = "test_upload_code_to_sandbox"

        try:
            # Cleanup
            try:
                await service.delete(sandbox_name)
            except Exception:
                pass

            # Create sandbox
            sandbox = await _create_sandbox(service, sandbox_name)

            # Check if version file exists
            version_check = await sandbox.exec("cat", "/app/.xagent_version")
            assert version_check.exit_code == 0, "Version file should exist"
            version1 = version_check.stdout.strip()
            print(f"Version hash: {version1}")

            # Create a test file in sandbox (to verify second upload is skipped)
            print("\nCreating test marker file...")
            await sandbox.write_file(
                content="# This is a test marker file",
                remote_path="/app/src/test_marker.txt",
                overwrite=True,
            )

            # Second upload
            await upload_code_to_sandbox(sandbox, contain_tests=True)

            # Check if version is the same
            version_check2 = await sandbox.exec("cat", "/app/.xagent_version")
            assert version_check2.exit_code == 0
            version2 = version_check2.stdout.strip()
            assert version1 == version2, "Version should be the same"

            # Test marker file should still exist (proves upload was skipped)
            check2 = await sandbox.exec("test", "-f", "/app/src/test_marker.txt")
            assert check2.exit_code == 0, "Test marker file should still exist"

            # Force upload
            await upload_code_to_sandbox(sandbox, force_upload=True, contain_tests=True)
            check3 = await sandbox.exec("test", "-f", "/app/src/test_marker.txt")
            assert check3.exit_code == 1, "Test marker file should not exist"

            print("✓ Version check functionality works correctly")

        finally:
            try:
                await service.delete(sandbox_name)
            except Exception:
                pass


@requires_boxlite
class TestTools:
    """Test tool execution in sandbox"""

    @pytest.mark.asyncio(loop_scope="module")
    async def test_python_executor_in_sandbox(self):
        """
        Launch a sandbox, upload code + tests via create_sandboxed_tool,
        then run tests/core/tools/test_python_executor.py inside the sandbox.
        """
        print("\n=== Test python_executor test suite in sandbox ===")

        service = BoxliteSandboxService(MemBoxliteStore())
        sandbox_name = "test_py_executor_suite"

        try:
            # Cleanup
            try:
                await service.delete(sandbox_name)
            except Exception:
                pass

            # Create sandbox
            sandbox = await _create_sandbox(service, sandbox_name)

            # Create sandboxed tool with contain_tests=True to upload tests
            sandboxed_tool = await create_sandboxed_tool(
                tool=get_python_executor_tool(None),
                sandbox=sandbox,
            )

            # Get sandbox instance for direct exec
            sb = await sandboxed_tool.get_sandbox_for_test()

            # Verify tests were uploaded
            check = await sb.exec(
                "test", "-f", "/app/tests/core/tools/test_python_executor.py"
            )
            assert check.exit_code == 0, (
                "test_python_executor.py should exist in sandbox"
            )

            # Install pytest in sandbox
            install_result = await sb.exec("pip", "install", "pytest", "pytest-asyncio")
            assert install_result.exit_code == 0, (
                f"Failed to install pytest: {install_result.stderr}"
            )

            # Run test_python_executor.py in sandbox
            test_result = await sb.exec(
                "python",
                "-m",
                "pytest",
                "/app/tests/core/tools/test_python_executor.py",
                "-v",
                "--tb=short",
                env={"PYTHONPATH": "/app/src"},
            )

            print(f"\n--- pytest stdout ---\n{test_result.stdout}")
            if test_result.stderr:
                print(f"\n--- pytest stderr ---\n{test_result.stderr}")
            print(f"\nExit code: {test_result.exit_code}")

            assert test_result.exit_code == 0, (
                f"pytest failed with exit code {test_result.exit_code}\n"
                f"stdout:\n{test_result.stdout}\n"
                f"stderr:\n{test_result.stderr}"
            )

            print("✅ Python executor test suite passed in sandbox")

        finally:
            try:
                await service.delete(sandbox_name)
            except Exception:
                pass

    @pytest.mark.skip(reason="Requires custom image with Python and Node.js")
    @pytest.mark.asyncio(loop_scope="module")
    async def test_javascript_executor_in_sandbox(self):
        """
        Launch a sandbox, upload code + tests via create_sandboxed_tool,
        then run tests/core/tools/test_javascript_executor.py inside the sandbox.
        """
        print("\n=== Test javascript_executor test suite in sandbox ===")

        service = BoxliteSandboxService(MemBoxliteStore())
        sandbox_name = "test_js_executor_suite"

        try:
            # Cleanup
            try:
                await service.delete(sandbox_name)
            except Exception:
                pass

            # Create sandbox
            sandbox = await _create_sandbox(service, sandbox_name)

            # Create sandboxed tool with contain_tests=True to upload tests
            sandboxed_tool = await create_sandboxed_tool(
                tool=get_javascript_executor_tool(None),
                sandbox=sandbox,
            )

            # Get sandbox instance for direct exec
            sb = await sandboxed_tool.get_sandbox_for_test()

            # Verify tests were uploaded
            check = await sb.exec(
                "test", "-f", "/app/tests/core/tools/test_javascript_executor.py"
            )
            assert check.exit_code == 0, (
                "test_javascript_executor.py should exist in sandbox"
            )

            # Install pytest and Node.js dependencies in sandbox
            install_result = await sb.exec("pip", "install", "pytest")
            assert install_result.exit_code == 0, (
                f"Failed to install pytest: {install_result.stderr}"
            )

            # Run test_javascript_executor.py in sandbox
            # Skip TestJavaScriptExecutorTool which requires langchain_core
            test_result = await sb.exec(
                "python",
                "-m",
                "pytest",
                "/app/tests/core/tools/test_javascript_executor.py",
                "-v",
                "--tb=short",
                "-k",
                "not TestJavaScriptExecutorTool",
                env={"PYTHONPATH": "/app/src"},
            )

            print(f"\n--- pytest stdout ---\n{test_result.stdout}")
            if test_result.stderr:
                print(f"\n--- pytest stderr ---\n{test_result.stderr}")
            print(f"\nExit code: {test_result.exit_code}")

            assert test_result.exit_code == 0, (
                f"pytest failed with exit code {test_result.exit_code}\n"
                f"stdout:\n{test_result.stdout}\n"
                f"stderr:\n{test_result.stderr}"
            )

            print("✅ JavaScript executor test suite passed in sandbox")

        finally:
            try:
                await service.delete(sandbox_name)
            except Exception:
                pass


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
            python_tool = get_python_executor_tool(None)
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
                        return re.sub(r"from '[^']*'", "from '<path>'", s)

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
