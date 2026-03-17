"""
Tests for CommandExecutor tool
"""

import os
import sys

import pytest

from xagent.core.tools.adapters.vibe.command_executor import (
    CommandExecutorArgs,
    CommandExecutorResult,
    CommandExecutorTool,
)
from xagent.core.tools.core.command_executor import (
    CommandExecutorCore,
    execute_command,
    execute_script,
)


@pytest.fixture
def command_executor():
    """Create CommandExecutorTool instance for testing"""
    return CommandExecutorTool()


@pytest.fixture
def temp_dir(tmp_path):
    """Create a temporary directory for file operations"""
    return str(tmp_path)


class TestCommandExecutorTool:
    """Test cases for CommandExecutorTool"""

    def test_tool_properties(self, command_executor):
        """Test basic tool properties"""
        assert command_executor.name == "command_executor"
        assert "shell" in command_executor.tags or "command" in command_executor.tags
        assert command_executor.args_type() == CommandExecutorArgs
        assert command_executor.return_type() == CommandExecutorResult

    def test_simple_echo_command(self, command_executor):
        """Test simple echo command"""
        result = command_executor.run_json_sync({"command": "echo Hello World"})

        assert result["success"] is True
        assert "Hello World" in result["output"]
        assert result["error"] == ""
        assert result["return_code"] == 0

    def test_command_with_pipe(self, command_executor):
        """Test command with pipe operation"""
        result = command_executor.run_json_sync(
            {"command": 'echo "apple\\nbanana\\ncherry" | grep banana'}
        )

        assert result["success"] is True
        assert "banana" in result["output"]
        assert result["return_code"] == 0

    def test_list_directory(self, command_executor):
        """Test listing directory contents"""
        result = command_executor.run_json_sync({"command": "ls -la /tmp"})

        assert result["success"] is True
        assert len(result["output"]) > 0
        assert result["return_code"] == 0

    def test_command_with_timeout(self, command_executor):
        """Test command execution with timeout"""
        # Sleep command that should complete within timeout
        result = command_executor.run_json_sync({"command": "sleep 0.1", "timeout": 5})

        assert result["success"] is True
        assert result["return_code"] == 0

    def test_command_timeout_exceeded(self, command_executor):
        """Test command that exceeds timeout"""
        # Sleep longer than timeout
        result = command_executor.run_json_sync({"command": "sleep 5", "timeout": 1})

        assert result["success"] is False
        assert "timed out" in result["error"].lower()
        assert result["return_code"] == -999  # TIMEOUT_EXIT_CODE

    def test_invalid_command(self, command_executor):
        """Test handling of invalid command"""
        result = command_executor.run_json_sync({"command": "nonexistentcommand12345"})

        assert result["success"] is False
        assert (
            "not found" in result["error"].lower()
            or "command not found" in result["error"].lower()
        )
        assert result["return_code"] != 0

    def test_command_with_stderr(self, command_executor):
        """Test that stderr is captured"""
        result = command_executor.run_json_sync({"command": 'echo "error message" >&2'})

        # Command succeeds but stderr is captured
        assert result["success"] is True
        assert "error message" in result["error"]

    def test_command_failure_nonzero_exit(self, command_executor):
        """Test command that fails with non-zero exit code"""
        result = command_executor.run_json_sync(
            {"command": "ls /nonexistent_directory_12345"}
        )

        assert result["success"] is False
        assert result["return_code"] != 0

    def test_command_with_redirection(self, command_executor, temp_dir):
        """Test command with output redirection"""
        output_file = os.path.join(temp_dir, "output.txt")
        result = command_executor.run_json_sync(
            {"command": f'echo "test content" > {output_file}'}
        )

        assert result["success"] is True
        assert os.path.exists(output_file)
        with open(output_file) as f:
            assert "test content" in f.read()

    def test_command_chain(self, command_executor):
        """Test chaining multiple commands with &&"""
        result = command_executor.run_json_sync(
            {"command": 'echo "first" && echo "second"'}
        )

        assert result["success"] is True
        assert "first" in result["output"]
        assert "second" in result["output"]

    def test_command_with_quotes(self, command_executor):
        """Test command with quoted arguments"""
        result = command_executor.run_json_sync({"command": 'echo "hello world"'})

        assert result["success"] is True
        assert "hello world" in result["output"]

    def test_grep_command(self, command_executor):
        """Test grep command for text search"""
        result = command_executor.run_json_sync(
            {"command": 'echo -e "apple\\nbanana\\ncherry" | grep banana'}
        )

        assert result["success"] is True
        assert "banana" in result["output"]

    def test_wc_command(self, command_executor):
        """Test wc command for word count"""
        result = command_executor.run_json_sync(
            {"command": 'echo "test content here" | wc -w'}
        )

        assert result["success"] is True
        assert len(result["output"].strip()) > 0

    def test_cat_command(self, command_executor, temp_dir):
        """Test cat command to read file"""
        test_file = os.path.join(temp_dir, "test.txt")
        with open(test_file, "w") as f:
            f.write("test content")

        result = command_executor.run_json_sync({"command": f"cat {test_file}"})

        assert result["success"] is True
        assert "test content" in result["output"]

    def test_head_command(self, command_executor):
        """Test head command for limiting output"""
        result = command_executor.run_json_sync({"command": "seq 1 100 | head -5"})

        assert result["success"] is True
        assert "1" in result["output"]
        assert "5" in result["output"]

    def test_tail_command(self, command_executor):
        """Test tail command for showing end of file"""
        result = command_executor.run_json_sync({"command": "seq 1 10 | tail -3"})

        assert result["success"] is True
        assert "8" in result["output"]
        assert "10" in result["output"]

    @pytest.mark.asyncio
    async def test_async_execution_same_as_sync(self, command_executor):
        """Test that async execution produces same results as sync"""
        command = "echo test"

        sync_result = command_executor.run_json_sync({"command": command})
        async_result = await command_executor.run_json_async({"command": command})

        assert sync_result == async_result

    def test_args_validation(self):
        """Test CommandExecutorArgs validation"""
        # Valid args with defaults
        args = CommandExecutorArgs(command="ls")
        assert args.command == "ls"
        assert args.timeout is None  # default

        # Custom args
        args = CommandExecutorArgs(command="sleep 1", timeout=5)
        assert args.command == "sleep 1"
        assert args.timeout == 5

    def test_result_model(self):
        """Test CommandExecutorResult model"""
        # Success result
        result = CommandExecutorResult(
            success=True, output="test output", error="", return_code=0
        )
        assert result.success is True
        assert result.output == "test output"
        assert result.error == ""
        assert result.return_code == 0

        # Error result
        result = CommandExecutorResult(
            success=False, output="", error="Some error", return_code=1
        )
        assert result.success is False
        assert result.output == ""
        assert result.error == "Some error"
        assert result.return_code == 1


class TestCommandExecutorCore:
    """Test cases for CommandExecutorCore"""

    def test_basic_execution(self):
        """Test basic command execution"""
        executor = CommandExecutorCore()
        result = executor.execute_command("echo test")

        assert result["success"] is True
        assert "test" in result["output"]
        assert result["return_code"] == 0

    def test_working_directory_change(self, tmp_path):
        """Test execution in specific working directory"""
        test_dir = str(tmp_path)
        executor = CommandExecutorCore(working_directory=test_dir)

        result = executor.execute_command("pwd")

        assert result["success"] is True
        assert test_dir in result["output"]

    def test_custom_timeout(self):
        """Test custom timeout setting"""
        executor = CommandExecutorCore()

        # Should complete within default timeout
        result = executor.execute_command("sleep 0.1")

        assert result["success"] is True

        # Test with custom timeout parameter
        result = executor.execute_command("sleep 0.1", timeout=5)
        assert result["success"] is True

    def test_shell_parameter(self):
        """Test shell parameter"""
        executor = CommandExecutorCore()

        # With shell=True (default)
        result = executor.execute_command("echo test", shell=True)
        assert result["success"] is True

        # With shell=False
        result = executor.execute_command(["echo", "test"], shell=False)
        assert result["success"] is True


class TestConvenienceFunctions:
    """Test convenience functions"""

    def test_execute_command_function(self):
        """Test execute_command convenience function"""
        result = execute_command("echo convenience test")

        assert result["success"] is True
        assert "convenience test" in result["output"]

    def test_execute_command_with_working_directory(self, tmp_path):
        """Test execute_command with working directory"""
        test_dir = str(tmp_path)
        result = execute_command("pwd", working_directory=test_dir)

        assert result["success"] is True
        assert test_dir in result["output"]

    def test_execute_command_with_timeout(self):
        """Test execute_command with timeout"""
        result = execute_command("sleep 0.1", timeout=5)

        assert result["success"] is True

    def test_execute_script_function(self):
        """Test execute_script convenience function"""
        script = """
echo "Script line 1"
echo "Script line 2"
"""
        result = execute_script(script, interpreter="bash")

        assert result["success"] is True
        assert "Script line 1" in result["output"]
        assert "Script line 2" in result["output"]

    def test_execute_script_with_working_directory(self, tmp_path):
        """Test execute_script with working directory"""
        test_dir = str(tmp_path)
        script = "pwd"
        result = execute_script(script, interpreter="bash", working_directory=test_dir)

        assert result["success"] is True
        assert test_dir in result["output"]

    def test_execute_script_with_timeout(self):
        """Test execute_script with timeout"""
        script = "#!/bin/bash\nsleep 0.1"
        result = execute_script(script, interpreter="bash", timeout=5)

        assert result["success"] is True


class TestEdgeCases:
    """Test edge cases and error conditions"""

    def test_empty_command(self, command_executor):
        """Test handling of empty command"""
        result = command_executor.run_json_sync({"command": ""})

        # Empty command actually succeeds in shell (returns exit code 0)
        # but produces no output
        assert result["success"] is True
        assert result["output"] == ""
        assert result["return_code"] == 0

    def test_very_long_command(self, command_executor):
        """Test handling of very long command"""
        long_command = "echo " + "x" * 10000
        result = command_executor.run_json_sync({"command": long_command})

        # Should handle long commands
        assert result["success"] is True

    def test_command_with_special_characters(self, command_executor):
        """Test command with special characters"""
        result = command_executor.run_json_sync({"command": 'echo "test@#$%^&*()"'})
        assert result["success"] is True
        assert "test@#$%^&*()" in result["output"]

    def test_command_with_newlines(self, command_executor):
        """Test command with embedded newlines"""
        result = command_executor.run_json_sync(
            {"command": 'echo "line1\\nline2\\nline3"'}
        )

        assert result["success"] is True
        assert "line1" in result["output"]
        assert "line2" in result["output"]
        assert "line3" in result["output"]

    def test_zero_timeout(self, command_executor):
        """Test command with zero timeout"""
        # Zero timeout should now raise ValueError
        with pytest.raises(ValueError, match="timeout must be positive"):
            command_executor.run_json_sync({"command": "echo test", "timeout": 0})

    def test_negative_timeout(self, command_executor):
        """Test command with negative timeout"""
        # Negative timeout should now raise ValueError
        with pytest.raises(ValueError, match="timeout must be positive"):
            command_executor.run_json_sync({"command": "echo test", "timeout": -1})


class TestPlatformSpecific:
    """Platform-specific tests"""

    @pytest.mark.skipif(sys.platform != "darwin", reason="macOS only")
    def test_macos_specific_command(self, command_executor):
        """Test macOS-specific command"""
        result = command_executor.run_json_sync({"command": "sw_vers"})

        assert result["success"] is True
        assert "macOS" in result["output"] or "Product" in result["output"]

    @pytest.mark.skipif(sys.platform != "linux", reason="Linux only")
    def test_linux_specific_command(self, command_executor):
        """Test Linux-specific command"""
        result = command_executor.run_json_sync({"command": "uname -a"})

        assert result["success"] is True
        assert "Linux" in result["output"]

    def test_uname_command(self, command_executor):
        """Test uname command (works on most Unix-like systems)"""
        result = command_executor.run_json_sync({"command": "uname"})

        assert result["success"] is True
        assert len(result["output"].strip()) > 0


class TestExecuteScriptFunction:
    """Test cases for the execute_script convenience function"""

    def test_execute_script_function(self):
        """Test execute_script convenience function"""
        script = "#!/bin/bash\necho 'script output'"
        result = execute_script(script, interpreter="bash")

        assert result["success"] is True
        assert "script output" in result["output"]

    def test_execute_script_with_python(self):
        """Test execute_script with Python interpreter"""
        script = "print('python script output')"
        result = execute_script(script, interpreter="python")

        assert result["success"] is True
        assert "python script output" in result["output"]

    def test_execute_script_with_timeout(self):
        """Test execute_script with timeout"""
        script = "#!/bin/bash\nsleep 0.1"
        result = execute_script(script, interpreter="bash", timeout=5)

        assert result["success"] is True


class TestConcurrentExecution:
    """Test cases for concurrent command execution"""

    def test_concurrent_execution(self):
        """Test that concurrent executions don't interfere"""
        import threading

        results = []

        def run_cmd(work_dir, thread_id):
            try:
                executor = CommandExecutorCore(working_directory=work_dir)
                result = executor.execute_command("pwd")
                results.append((thread_id, result["output"].strip(), result["success"]))
            except Exception as e:
                results.append((thread_id, str(e), False))

        threads = [
            threading.Thread(target=run_cmd, args=("/tmp", 1)),
            threading.Thread(target=run_cmd, args=("/home", 2)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Both threads should complete successfully
        assert len(results) == 2
        for tid, output, success in results:
            assert success is True
            assert len(output) > 0


class TestTimeoutValidation:
    """Test cases for timeout validation"""

    def test_negative_timeout_raises_error(self):
        """Test that negative timeout raises ValueError"""
        executor = CommandExecutorCore()

        with pytest.raises(ValueError, match="timeout must be positive"):
            executor.execute_command("echo test", timeout=-1)

    def test_zero_timeout_raises_error(self):
        """Test that zero timeout raises ValueError"""
        executor = CommandExecutorCore()

        with pytest.raises(ValueError, match="timeout must be positive"):
            executor.execute_command("echo test", timeout=0)


class TestWorkingDirectoryValidation:
    """Test cases for working directory validation"""

    def test_nonexistent_working_directory(self):
        """Test that nonexistent working directory raises FileNotFoundError"""
        executor = CommandExecutorCore(working_directory="/nonexistent/path/xyz")

        with pytest.raises(FileNotFoundError, match="does not exist"):
            executor.execute_command("echo test")

    def test_file_as_working_directory(self, tmp_path):
        """Test that using a file (not directory) as working directory raises error"""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test")

        executor = CommandExecutorCore(working_directory=str(test_file))

        with pytest.raises(NotADirectoryError, match="not a directory"):
            executor.execute_command("echo test")


class TestOutputSizeLimit:
    """Test cases for output size limiting"""

    def test_large_output_truncation(self, command_executor):
        """Test that very large output is truncated"""
        # Generate a command that produces lots of output (more than 10MB)
        # Use Python to generate large output
        result = command_executor.run_json_sync(
            {"command": "python -c \"print('x' * 11_000_000)\""}
        )

        assert result["success"] is True
        # Output should be truncated
        assert "[OUTPUT TRUNCATED]" in result["output"]
        # Output should be truncated to MAX_OUTPUT_SIZE + suffix
        assert len(result["output"]) <= 10 * 1024 * 1024 + 100
