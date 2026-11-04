"""
Tests for CommandExecutor tool
"""

import os
import shlex
import shutil
import sys
from pathlib import Path
from typing import Sequence
from unittest.mock import Mock

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
from xagent.core.tools.core.command_path_guard import WorkspaceCommandPathGuard
from xagent.core.tools.core.command_policy import (
    CommandPathViolation,
    CommandPolicyGuard,
    CommandPolicyViolation,
    resolve_trusted_executable,
)
from xagent.core.workspace import TaskWorkspace


@pytest.fixture
def command_executor():
    """Create CommandExecutorTool instance for testing"""
    return CommandExecutorTool()


@pytest.fixture
def temp_dir(tmp_path):
    """Create a temporary directory for file operations"""
    return str(tmp_path)


@pytest.fixture
def mock_run(monkeypatch):
    run = Mock()
    target = "xagent.core.tools.core.command_executor.subprocess.run"
    monkeypatch.setattr(target, run)
    return run


class TestCommandExecutorTool:
    """Test cases for CommandExecutorTool"""

    def test_tool_properties(self, command_executor):
        """Test basic tool properties"""
        assert command_executor.name == "command_executor"
        assert "shell" in command_executor.tags or "command" in command_executor.tags
        assert command_executor.args_type() == CommandExecutorArgs
        assert command_executor.return_type() == CommandExecutorResult

    def test_description_includes_workspace_cwd_and_search_scope(self, tmp_path):
        """Test that shell guidance exposes cwd and discourages broad searches."""
        workspace = Mock()
        workspace.resolve_path.return_value = tmp_path
        tool = CommandExecutorTool(workspace=workspace)

        description = tool.description

        assert f"current working directory: {tmp_path}" in description
        assert "Use concrete paths" in description
        assert "Only search for files when no usable path was provided" in description
        assert "Do not run broad recursive searches from `/`" in description

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

    def test_list_directory(self, command_executor, temp_dir):
        """Test listing directory contents"""
        result = command_executor.run_json_sync(
            {"command": f"ls -la {shlex.quote(temp_dir)}"}
        )

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


class TestCommandPolicyFoundation:
    @staticmethod
    def _workspace_guard(tmp_path):
        workspace = TaskWorkspace("task", str(tmp_path / "workspace"))
        guard = WorkspaceCommandPathGuard(workspace)
        return workspace, guard

    @pytest.mark.parametrize("working_directory", [None, ""])
    def test_executor_adopts_guard_working_directory(
        self,
        tmp_path,
        mock_run,
        working_directory,
    ):
        workspace, guard = self._workspace_guard(tmp_path)
        canonical_cwd = workspace.resolve_path("").resolve()
        mock_run.return_value = Mock(stdout="", stderr="", returncode=0)

        executor = CommandExecutorCore(
            working_directory=working_directory,
            path_guard=guard,
        )
        result = executor.execute_command(["true"], shell=False)

        assert executor.working_directory == str(canonical_cwd)
        assert mock_run.call_args.kwargs["cwd"] == canonical_cwd
        assert result["success"] is True

    @pytest.mark.parametrize("spelling", ["exact", "equivalent", "symlink"])
    def test_executor_accepts_canonically_equivalent_guard_cwd(
        self,
        tmp_path,
        mock_run,
        spelling,
    ):
        workspace, guard = self._workspace_guard(tmp_path)
        canonical_cwd = workspace.resolve_path("").resolve()
        if spelling == "exact":
            caller_cwd = str(canonical_cwd)
        elif spelling == "equivalent":
            caller_cwd = f"{canonical_cwd}/."
        else:
            alias = tmp_path / "workspace-alias"
            alias.symlink_to(canonical_cwd, target_is_directory=True)
            caller_cwd = str(alias)
        mock_run.return_value = Mock(stdout="", stderr="", returncode=0)

        executor = CommandExecutorCore(
            working_directory=caller_cwd,
            path_guard=guard,
        )
        executor.execute_command(["true"], shell=False)

        assert executor.working_directory == str(canonical_cwd)
        assert mock_run.call_args.kwargs["cwd"] == canonical_cwd

    def test_executor_keeps_guard_cwd_after_caller_symlink_retarget(
        self,
        tmp_path,
        mock_run,
    ):
        workspace, guard = self._workspace_guard(tmp_path)
        canonical_cwd = workspace.resolve_path("").resolve()
        other_cwd = tmp_path / "other"
        other_cwd.mkdir()
        alias = tmp_path / "workspace-alias"
        alias.symlink_to(canonical_cwd, target_is_directory=True)
        executor = CommandExecutorCore(
            working_directory=str(alias),
            path_guard=guard,
        )
        alias.unlink()
        alias.symlink_to(other_cwd, target_is_directory=True)
        mock_run.return_value = Mock(stdout="", stderr="", returncode=0)

        executor.execute_command(["true"], shell=False)

        assert mock_run.call_args.kwargs["cwd"] == canonical_cwd

    def test_executor_rejects_guard_cwd_mismatch_before_validation_or_spawn(
        self,
        tmp_path,
        mock_run,
    ):
        class CwdBoundGuard:
            def __init__(self, execution_cwd):
                self.execution_cwd = execution_cwd
                self.validate = Mock()
                self.validate_argv = Mock()

        guard_cwd = tmp_path / "guard"
        guard_cwd.mkdir()
        caller_cwd = tmp_path / "caller"
        caller_cwd.mkdir()
        guard = CwdBoundGuard(guard_cwd.resolve())

        with pytest.raises(
            ValueError,
            match="working_directory does not match command policy execution cwd",
        ):
            CommandExecutorCore(
                working_directory=str(caller_cwd),
                path_guard=guard,
            )

        guard.validate.assert_not_called()
        guard.validate_argv.assert_not_called()
        mock_run.assert_not_called()

    def test_executor_preserves_legacy_cwd_for_guard_without_binding(
        self,
        tmp_path,
        mock_run,
    ):
        guard = Mock(spec=CommandPolicyGuard)
        caller_cwd = f"{tmp_path}/."
        mock_run.return_value = Mock(stdout="", stderr="", returncode=0)

        executor = CommandExecutorCore(
            working_directory=caller_cwd,
            path_guard=guard,
        )
        executor.execute_command(["true"], shell=False)

        assert executor.working_directory == caller_cwd
        assert mock_run.call_args.kwargs["cwd"] == caller_cwd

    def test_executor_preserves_legacy_cwd_for_policy_with_no_requirement(
        self,
        tmp_path,
        mock_run,
    ):
        class OptionalCwdGuard:
            execution_cwd = None

            def validate(self, command):
                return None

            def validate_argv(self, argv):
                return None

        caller_cwd = f"{tmp_path}/."
        mock_run.return_value = Mock(stdout="", stderr="", returncode=0)

        executor = CommandExecutorCore(
            working_directory=caller_cwd,
            path_guard=OptionalCwdGuard(),
        )
        executor.execute_command(["true"], shell=False)

        assert executor.working_directory == caller_cwd
        assert mock_run.call_args.kwargs["cwd"] == caller_cwd

    def test_executor_working_directory_is_read_only_with_bound_policy(
        self,
        tmp_path,
        mock_run,
    ):
        workspace, guard = self._workspace_guard(tmp_path)
        canonical_cwd = workspace.resolve_path("").resolve()
        other_cwd = tmp_path / "other"
        other_cwd.mkdir()
        executor = CommandExecutorCore(path_guard=guard)
        mock_run.return_value = Mock(stdout="", stderr="", returncode=0)

        with pytest.raises(AttributeError):
            executor.working_directory = str(other_cwd)
        executor.execute_command(["true"], shell=False)

        assert mock_run.call_args.kwargs["cwd"] == canonical_cwd

    @pytest.mark.parametrize("replacement", ["other_guard", "none"])
    def test_executor_path_guard_is_read_only_and_keeps_original_binding(
        self,
        tmp_path,
        mock_run,
        replacement,
    ):
        class CwdBoundGuard:
            def __init__(self, execution_cwd):
                self.execution_cwd = execution_cwd
                self.validate = Mock()
                self.validate_argv = Mock()

        original_cwd = tmp_path / "original"
        original_cwd.mkdir()
        other_cwd = tmp_path / "other"
        other_cwd.mkdir()
        original_guard = CwdBoundGuard(original_cwd.resolve())
        other_guard = CwdBoundGuard(other_cwd.resolve())
        executor = CommandExecutorCore(path_guard=original_guard)
        mock_run.return_value = Mock(stdout="", stderr="", returncode=0)

        with pytest.raises(AttributeError):
            executor.path_guard = other_guard if replacement == "other_guard" else None
        result = executor.execute_command(["true"], shell=False)

        original_guard.validate_argv.assert_called_once_with(["true"])
        other_guard.validate_argv.assert_not_called()
        assert mock_run.call_args.kwargs["cwd"] == original_cwd.resolve()
        assert result["success"] is True

    def test_trusted_executable_resolution_rejects_relative_command_paths(self):
        bash = shutil.which("bash")
        assert bash is not None
        relative_bash = os.path.relpath(bash, Path.cwd())

        with pytest.raises(CommandPolicyViolation):
            resolve_trusted_executable(relative_bash)

    @pytest.mark.parametrize(
        ("command", "shell"),
        [
            ("cat sibling.txt", True),
            (["cat", "sibling.txt"], False),
        ],
    )
    def test_injected_policy_rejects_before_process_spawn(
        self,
        mock_run,
        command,
        shell,
    ):
        class RejectingGuard:
            def validate(self, command: str) -> None:
                raise CommandPathViolation(
                    access="read",
                    path=Path("/private/sibling/secret.txt"),
                )

            def validate_argv(self, argv: Sequence[str]) -> None:
                raise CommandPathViolation(
                    access="read",
                    path=Path("/private/sibling/secret.txt"),
                )

        guard: CommandPolicyGuard = RejectingGuard()

        result = CommandExecutorCore(path_guard=guard).execute_command(
            command,
            shell=shell,
        )

        assert result["success"] is False
        assert result["return_code"] == 126
        assert result["error"] == (
            "Command rejected by workspace path policy: "
            "path is outside allowed read paths"
        )
        assert "/private/sibling" not in result["error"]
        mock_run.assert_not_called()

    def test_generic_policy_rejection_redacts_internal_reason(
        self,
        mock_run,
        caplog,
    ):
        guard = Mock(spec=CommandPolicyGuard)
        guard.validate.side_effect = CommandPolicyViolation(
            "parser rejected /private/sibling/secret.txt"
        )

        result = CommandExecutorCore(path_guard=guard).execute_command("true")

        assert result["success"] is False
        assert result["return_code"] == 126
        assert result["error"].endswith("command denied by policy")
        assert "/private/sibling" not in result["error"]
        assert "/private/sibling/secret.txt" in caplog.text
        mock_run.assert_not_called()

    def test_unexpected_policy_failure_is_fail_closed(self, mock_run):
        guard = Mock(spec=CommandPolicyGuard)
        guard.validate.side_effect = RuntimeError("internal parser detail")

        result = CommandExecutorCore(path_guard=guard).execute_command("true")

        assert result["success"] is False
        assert result["return_code"] == 126
        assert result["error"].endswith("command validation failed")
        assert "internal parser detail" not in result["error"]
        mock_run.assert_not_called()

    def test_injected_policy_rejects_execute_script_without_side_effects(
        self,
        tmp_path,
        mock_run,
        monkeypatch,
    ):
        guard = Mock(spec=CommandPolicyGuard)
        create_tempfile = Mock()
        marker = tmp_path / "fake-bash-ran"
        fake_bin = tmp_path / "bin"
        fake_bin.mkdir()
        fake_bash = fake_bin / "bash"
        fake_bash.write_text(
            f"#!/bin/sh\ntouch {shlex.quote(str(marker))}\n",
            encoding="utf-8",
        )
        fake_bash.chmod(0o755)
        monkeypatch.setenv(
            "PATH",
            f"{fake_bin}{os.pathsep}{os.environ.get('PATH', '')}",
        )
        monkeypatch.setattr(
            "xagent.core.tools.core.command_executor.tempfile.NamedTemporaryFile",
            create_tempfile,
        )

        result = CommandExecutorCore(path_guard=guard).execute_script("echo allowed")

        guard.validate.assert_not_called()
        guard.validate_argv.assert_not_called()
        mock_run.assert_not_called()
        assert result["success"] is False
        assert result["return_code"] == 126
        assert result["error"] == (
            "Command rejected by workspace path policy: command denied by policy"
        )
        create_tempfile.assert_not_called()
        assert not marker.exists()

    def test_unguarded_execute_script_preserves_real_tempfile_semantics(self):
        result = CommandExecutorCore().execute_script(
            'printf \'%s\\n%s\\n\' "$0" "${BASH_SOURCE[0]}"',
        )

        script_path, bash_source = result["output"].splitlines()
        assert result["success"] is True
        assert script_path == bash_source
        assert Path(script_path).is_absolute()
        assert not Path(script_path).exists()

    @pytest.mark.parametrize("interpreter", ["bash", "python", "bash -O extglob", "'"])
    def test_injected_policy_rejects_execute_script_for_all_interpreters(
        self,
        mock_run,
        interpreter,
    ):
        guard = Mock(spec=CommandPolicyGuard)

        result = CommandExecutorCore(path_guard=guard).execute_script(
            "echo unsafe",
            interpreter=interpreter,
        )

        assert result["success"] is False
        assert result["return_code"] == 126
        guard.validate.assert_not_called()
        guard.validate_argv.assert_not_called()
        mock_run.assert_not_called()

    def test_injected_policy_allows_shell_command_execution(self, mock_run):
        guard = Mock(spec=CommandPolicyGuard)
        mock_run.return_value = Mock(stdout="allowed", stderr="", returncode=0)
        bash = shutil.which("bash")
        assert bash is not None
        trusted_bash = str(Path(bash).resolve())

        result = CommandExecutorCore(path_guard=guard).execute_command("printf allowed")

        guard.validate.assert_called_once_with("printf allowed")
        guard.validate_argv.assert_not_called()
        mock_run.assert_called_once()
        assert mock_run.call_args.args[0] == "printf allowed"
        assert mock_run.call_args.kwargs["shell"] is True
        assert mock_run.call_args.kwargs["executable"] == trusted_bash
        assert result["success"] is True

    def test_injected_policy_rejects_path_shadowed_bash_before_spawn(
        self,
        tmp_path,
        mock_run,
        monkeypatch,
    ):
        guard = Mock(spec=CommandPolicyGuard)
        marker = tmp_path / "fake-bash-ran"
        fake_bin = tmp_path / "bin"
        fake_bin.mkdir()
        fake_bash = fake_bin / "bash"
        fake_bash.write_text(
            f"#!/bin/sh\ntouch {shlex.quote(str(marker))}\n",
            encoding="utf-8",
        )
        fake_bash.chmod(0o755)
        monkeypatch.setenv(
            "PATH",
            f"{fake_bin}{os.pathsep}{os.environ.get('PATH', '')}",
        )

        result = CommandExecutorCore(path_guard=guard).execute_command("printf allowed")

        guard.validate.assert_called_once_with("printf allowed")
        assert result["success"] is False
        assert result["return_code"] == 126
        mock_run.assert_not_called()
        assert not marker.exists()

    def test_injected_policy_uses_absolute_bash_identity_for_argv_execution(
        self,
        mock_run,
    ):
        guard = Mock(spec=CommandPolicyGuard)
        mock_run.return_value = Mock(stdout="allowed", stderr="", returncode=0)
        command = ["bash", "-c", "printf allowed"]
        bash = shutil.which("bash")
        assert bash is not None
        trusted_bash = str(Path(bash).resolve())

        result = CommandExecutorCore(path_guard=guard).execute_command(
            command,
            shell=False,
        )

        guard.validate_argv.assert_called_once_with(command)
        assert mock_run.call_args.args[0] == [
            trusted_bash,
            "-c",
            "printf allowed",
        ]
        assert result["success"] is True

    @pytest.mark.parametrize(
        "command",
        [
            ["env", "bash", "-c", "printf unsafe"],
            ["timeout", "1", "bash", "-c", "printf unsafe"],
        ],
    )
    def test_workspace_policy_rejects_wrapped_bash_argv_before_spawn(
        self,
        tmp_path,
        mock_run,
        monkeypatch,
        command,
    ):
        workspace, guard = self._workspace_guard(tmp_path)
        marker = tmp_path / "fake-bash-ran"
        fake_bin = workspace.output_dir / "bin"
        fake_bin.mkdir()
        fake_bash = fake_bin / "bash"
        fake_bash.write_text(
            f"touch {shlex.quote(str(marker))}\n",
            encoding="utf-8",
        )
        fake_bash.chmod(0o755)
        monkeypatch.setenv(
            "PATH",
            f"{fake_bin}{os.pathsep}{os.environ.get('PATH', '')}",
        )

        result = CommandExecutorCore(path_guard=guard).execute_command(
            command,
            shell=False,
        )

        assert result["success"] is False
        assert result["return_code"] == 126
        mock_run.assert_not_called()
        assert not marker.exists()

    def test_workspace_policy_uses_absolute_bash_identity_for_direct_argv(
        self,
        tmp_path,
        mock_run,
    ):
        _, guard = self._workspace_guard(tmp_path)
        command = ["bash", "-c", "printf allowed"]
        bash = shutil.which("bash")
        assert bash is not None
        trusted_bash = str(Path(bash).resolve())
        mock_run.return_value = Mock(stdout="allowed", stderr="", returncode=0)

        result = CommandExecutorCore(path_guard=guard).execute_command(
            command,
            shell=False,
        )

        assert mock_run.call_args.args[0] == [
            trusted_bash,
            "-c",
            "printf allowed",
        ]
        assert result["success"] is True

    def test_workspace_policy_preserves_non_bash_wrapper_argv(
        self,
        tmp_path,
        mock_run,
    ):
        _, guard = self._workspace_guard(tmp_path)
        command = ["env", "printf", "allowed"]
        mock_run.return_value = Mock(stdout="allowed", stderr="", returncode=0)

        result = CommandExecutorCore(path_guard=guard).execute_command(
            command,
            shell=False,
        )

        assert mock_run.call_args.args[0] == command
        assert result["success"] is True

    def test_injected_policy_allows_argv_command_execution(self, mock_run):
        guard = Mock(spec=CommandPolicyGuard)
        mock_run.return_value = Mock(stdout="allowed", stderr="", returncode=0)
        command = ["printf", "allowed"]

        result = CommandExecutorCore(path_guard=guard).execute_command(
            command,
            shell=False,
        )

        guard.validate.assert_not_called()
        guard.validate_argv.assert_called_once_with(command)
        mock_run.assert_called_once()
        assert mock_run.call_args.args[0] == command
        assert mock_run.call_args.kwargs["shell"] is False
        assert result["success"] is True


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
