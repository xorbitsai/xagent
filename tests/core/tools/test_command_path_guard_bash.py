"""Bash-semantics regressions for the scoped command path guard."""

import os
import shlex
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from xagent.core.tools.core import command_path_guard as command_path_guard_module
from xagent.core.tools.core.command_executor import CommandExecutorCore
from xagent.core.tools.core.command_path_guard import WorkspaceCommandPathGuard
from xagent.core.tools.core.command_policy import (
    CommandPathViolation,
    CommandPolicyViolation,
    resolve_trusted_executable,
)
from xagent.core.workspace import TaskWorkspace


@pytest.fixture
def scoped_command_workspace(tmp_path):
    """Workspace plus same-mount sibling and read-only external roots."""
    alice_base = tmp_path / "clients" / "1" / "end_users" / "7"
    external = tmp_path / "external" / "7"
    external.mkdir(parents=True, exist_ok=True)
    workspace = TaskWorkspace(
        "task",
        str(alice_base),
        allowed_external_dirs=[str(external)],
    )
    external_file = external / "reference.txt"
    external_file.write_text("external reference", encoding="utf-8")

    sibling = tmp_path / "clients" / "1" / "end_users" / "8"
    sibling.mkdir(parents=True, exist_ok=True)
    sibling_file = sibling / "secret.txt"
    sibling_file.write_text("sibling secret", encoding="utf-8")

    return workspace, external_file, sibling_file


def _guarded_executor(workspace):
    return CommandExecutorCore(
        str(workspace.resolve_path("")),
        path_guard=WorkspaceCommandPathGuard(workspace),
    )


class _GuardedCommandTool:
    def __init__(self, workspace):
        self._executor = _guarded_executor(workspace)

    def run_json_sync(self, args):
        return self._executor.execute_command(
            args["command"],
            timeout=args.get("timeout"),
        )


def _guarded_tool(workspace):
    return _GuardedCommandTool(workspace)


def _write_comment_script_bytes(path, size):
    prefix = b"#"
    path.write_bytes(prefix + (b"x" * (size - len(prefix))))


@pytest.fixture(scope="module")
def trusted_bash_executable():
    try:
        return resolve_trusted_executable("bash")
    except CommandPolicyViolation:
        pytest.skip("trusted Bash is unavailable")


class TestScopedCommandPathGuardBash:
    """Bash parsing and shell-state policy checks."""

    @pytest.mark.parametrize(
        ("command_name", "access"),
        [("cat", "read"), ("rm", "write")],
    )
    def test_minimal_file_command_set_uses_workspace_authorization(
        self,
        scoped_command_workspace,
        command_name,
        access,
    ):
        workspace, _, sibling_file = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPathViolation) as exc_info:
            guard.validate(f"{command_name} {shlex.quote(str(sibling_file))}")

        assert exc_info.value.access == access

    def test_unknown_commands_keep_the_explicit_fail_open_contract(
        self,
        scoped_command_workspace,
    ):
        workspace, _, sibling_file = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate(f"python {shlex.quote(str(sibling_file))}")

    def test_recognized_name_workspace_script_is_inspected_before_dispatch(
        self,
        scoped_command_workspace,
        monkeypatch,
    ):
        workspace, _, sibling_file = scoped_command_workspace
        script = workspace.output_dir / "cat"
        script.write_text(
            f"#!/usr/bin/env bash\ncat {shlex.quote(str(sibling_file))}\n",
            encoding="utf-8",
        )
        script.chmod(0o755)
        monkeypatch.setenv(
            "PATH",
            f"{workspace.output_dir}{os.pathsep}{os.environ.get('PATH', '')}",
        )
        tool = _guarded_tool(workspace)

        result = tool.run_json_sync({"command": str(script)})

        assert result["return_code"] == 126
        assert "sibling secret" not in result["output"]

    def test_bare_recognized_name_uses_validated_executable_identity(
        self,
        scoped_command_workspace,
        monkeypatch,
    ):
        workspace, _, sibling_file = scoped_command_workspace
        script = workspace.output_dir / "cat"
        script.write_text(
            f"#!/usr/bin/env bash\n/bin/cat {shlex.quote(str(sibling_file))}\n",
            encoding="utf-8",
        )
        script.chmod(0o755)
        monkeypatch.setenv(
            "PATH",
            f"{workspace.output_dir}{os.pathsep}{os.environ.get('PATH', '')}",
        )
        tool = _guarded_tool(workspace)

        result = tool.run_json_sync({"command": "cat"})

        assert result["return_code"] == 126
        assert "sibling secret" not in result["output"]

    @pytest.mark.parametrize(
        "command",
        [
            "PATH=. cat own.txt",
            "env PATH=. cat own.txt",
            "export PATH=.; cat own.txt",
            "hash -p ./cat cat; cat own.txt",
        ],
    )
    def test_rejects_runtime_command_identity_changes(
        self,
        scoped_command_workspace,
        command,
    ):
        workspace, _, _ = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPolicyViolation):
            guard.validate(command)

    def test_rejects_alias_definition_before_shell_execution(
        self,
        scoped_command_workspace,
    ):
        workspace, _, sibling_file = scoped_command_workspace
        tool = _guarded_tool(workspace)

        result = tool.run_json_sync(
            {"command": (f"alias leak='cat {shlex.quote(str(sibling_file))}'\nleak")}
        )

        assert result["return_code"] == 126
        assert "command denied by policy" in result["error"]
        assert "sibling secret" not in result["output"]

    def test_allows_literal_quoted_here_document(
        self,
        scoped_command_workspace,
    ):
        workspace, _, _ = scoped_command_workspace
        tool = _guarded_tool(workspace)

        result = tool.run_json_sync(
            {"command": "cat <<'EOF'\n$HOME `printf not-executed`\nEOF\n"}
        )

        assert result["success"] is True
        assert result["output"] == "$HOME `printf not-executed`\n"

    @pytest.mark.parametrize(
        "command",
        [
            "cat <<'EOF'\r\nbody\r\nEOF\r\nprintf AFTER\r\n",
            "cat <<'EOF'\nbody\r\nEOF\r\nprintf AFTER\n",
        ],
    )
    def test_crlf_heredoc_syntax_fails_closed(
        self,
        scoped_command_workspace,
        command,
    ):
        workspace, _, _ = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(
            CommandPolicyViolation,
            match="cannot safely parse shell command",
        ):
            guard.validate(command)

    def test_quoted_heredoc_normalization_does_not_mask_later_commands(
        self,
        scoped_command_workspace,
    ):
        workspace, _, sibling_file = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPathViolation):
            guard.validate(
                f"printf %s \"<<'EOF'\"\ncat {shlex.quote(str(sibling_file))}\nEOF\n"
            )

    def test_nested_quoted_heredoc_body_is_masked_once(self):
        source = "cat <<'OUTER'\nreport <<'INNER'\nOUTER\nINNER\nprintf live\n"

        normalized = command_path_guard_module._normalize_policy_shell_source(source)

        assert len(normalized) == len(source)
        assert normalized.splitlines()[2] == "OUTER"
        assert normalized.endswith("printf live\n")

    def test_multiple_quoted_heredocs_are_consumed_in_declaration_order(self):
        source = (
            "cat <<'FIRST' <<-'SECOND'\n"
            "first <<'NESTED'\n"
            "FIRST\n"
            "\tsecond time\n"
            "\tSECOND\n"
            "printf live\n"
        )

        normalized = command_path_guard_module._normalize_policy_shell_source(source)

        assert len(normalized) == len(source)
        assert normalized.splitlines()[2] == "FIRST"
        assert normalized.splitlines()[4] == "\tSECOND"
        assert normalized.endswith("printf live\n")

    def test_time_keyword_normalization_ignores_multiline_quoted_literal(self):
        source = 'printf "%s" "literal\ntime cat own.txt"\ntime cat own.txt\n'

        normalized = command_path_guard_module._normalize_policy_shell_source(source)

        assert normalized == (
            'printf "%s" "literal\ntime cat own.txt"\n__t_ cat own.txt\n'
        )

    def test_allows_time_keyword_with_validated_nested_command(
        self,
        scoped_command_workspace,
    ):
        workspace, _, _ = scoped_command_workspace
        own_file = workspace.output_dir / "own.txt"
        own_file.write_text("own", encoding="utf-8")
        tool = _guarded_tool(workspace)

        result = tool.run_json_sync({"command": "time cat own.txt"})

        assert result["success"] is True
        assert result["output"] == "own"

    def test_time_keyword_propagates_builtin_directory_state(
        self,
        scoped_command_workspace,
    ):
        workspace, external_file, _ = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPathViolation) as exc_info:
            guard.validate(
                f"time cd {shlex.quote(str(external_file.parent))} "
                "&& printf leaked > leaked.txt"
            )

        assert exc_info.value.access == "write"

    @pytest.mark.parametrize("shell_name", ["sh", "dash", "zsh"])
    def test_rejects_shell_dialects_not_owned_by_policy_parser(
        self,
        scoped_command_workspace,
        shell_name,
    ):
        workspace, _, _ = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPolicyViolation, match="shell dialect"):
            guard.validate(f"{shell_name} -c 'cat own.txt'")

    def test_guarded_shell_execution_uses_bash_dialect(
        self,
        scoped_command_workspace,
        monkeypatch,
    ):
        workspace, _, _ = scoped_command_workspace
        captured = {}

        def fake_run(command, **kwargs):
            captured.update(kwargs)
            return type(
                "Completed",
                (),
                {"returncode": 0, "stdout": "", "stderr": ""},
            )()

        monkeypatch.setattr(
            "xagent.core.tools.core.command_executor.subprocess.run",
            fake_run,
        )
        executor = _guarded_executor(workspace)

        result = executor.execute_command("printf ok")

        assert result["success"] is True
        assert os.path.basename(captured["executable"]) == "bash"

    def test_unsupported_operator_error_is_actionable(
        self,
        scoped_command_workspace,
    ):
        workspace, _, _ = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(
            CommandPolicyViolation,
            match="unsupported shell operator.*separate command calls",
        ):
            guard.validate("printf one & printf two")

    @pytest.mark.parametrize(
        "command_template",
        [
            "printf changed >> {path}",
            "cat missing 2> {path}",
        ],
    )
    def test_append_and_stderr_redirects_require_write_authorization(
        self,
        scoped_command_workspace,
        command_template,
    ):
        workspace, external_file, _ = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPathViolation) as exc_info:
            guard.validate(
                command_template.format(path=shlex.quote(str(external_file)))
            )

        assert exc_info.value.access == "write"

    def test_possible_directory_state_growth_is_bounded(
        self,
        scoped_command_workspace,
    ):
        workspace, _, _ = scoped_command_workspace
        for index in range(17):
            (workspace.output_dir / f"d{index}").mkdir()
        guard = WorkspaceCommandPathGuard(workspace)
        command = " || ".join(f"cd d{index}" for index in range(17))

        with pytest.raises(CommandPolicyViolation, match="too many possible"):
            guard.validate(command)

    @pytest.mark.parametrize(
        "command_template",
        [
            "cat {{{path},missing}}",
            "{{cat,printf}} {path}",
            "rm -f {{{path},missing}}",
            "printf changed > {{{path},own.txt}}",
            "bash -c 'cat {{{path},missing}}'",
        ],
    )
    def test_rejects_brace_expansion_before_read_write_or_nested_execution(
        self, scoped_command_workspace, command_template
    ):
        workspace, _, sibling_file = scoped_command_workspace
        tool = _guarded_tool(workspace)
        command = command_template.format(path=str(sibling_file))

        result = tool.run_json_sync({"command": command})

        assert result["success"] is False
        assert result["return_code"] == 126
        assert sibling_file.read_text(encoding="utf-8") == "sibling secret"

    @pytest.mark.parametrize("operand", ["{a,b}", "{1..3}", "{a..z..2}"])
    def test_rejects_unmodeled_brace_expansion_forms(
        self, scoped_command_workspace, operand
    ):
        workspace, _, _ = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPolicyViolation, match="dynamic path operand"):
            guard.validate(f"cat {operand}")

    @pytest.mark.parametrize("operand", ["'{a,b}'", r"\{1..3\}"])
    def test_allows_quoted_or_escaped_literal_braces(
        self, scoped_command_workspace, operand
    ):
        workspace, _, _ = scoped_command_workspace
        literal_name = operand.replace("'", "").replace("\\", "")
        (workspace.output_dir / literal_name).write_text("literal", encoding="utf-8")
        tool = _guarded_tool(workspace)

        result = tool.run_json_sync({"command": f"cat {operand}"})

        assert result["success"] is True
        assert result["output"] == "literal"

    @pytest.mark.parametrize("operand", ["*", "file?.txt", "[ab].txt"])
    def test_rejects_unmodeled_glob_expansion_forms(
        self, scoped_command_workspace, operand
    ):
        workspace, _, _ = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPolicyViolation, match="dynamic path operand"):
            guard.validate(f"cat {operand}")

    @pytest.mark.parametrize(
        ("operand", "literal_name"),
        [
            ("'*'", "*"),
            ('"file?.txt"', "file?.txt"),
            (r"\[ab\].txt", "[ab].txt"),
        ],
    )
    def test_allows_quoted_or_escaped_literal_globs(
        self, scoped_command_workspace, operand, literal_name
    ):
        workspace, _, _ = scoped_command_workspace
        (workspace.output_dir / literal_name).write_text("literal", encoding="utf-8")
        tool = _guarded_tool(workspace)

        result = tool.run_json_sync({"command": f"cat {operand}"})

        assert result["success"] is True
        assert result["output"] == "literal"

    def test_rejects_active_glob_before_execution(self, scoped_command_workspace):
        workspace, _, _ = scoped_command_workspace
        tool = _guarded_tool(workspace)

        result = tool.run_json_sync({"command": "cat *.txt"})

        assert result["success"] is False
        assert result["return_code"] == 126

    @pytest.mark.parametrize(
        "command_template",
        [
            "exec cat {path}",
            "env cat {path}",
            "env -i NAME=value cat {path}",
            "timeout --signal=TERM 5 cat {path}",
            "nohup cat {path}",
            "nice -n 5 cat {path}",
            "stdbuf -oL cat {path}",
            "command -p cat {path}",
            "setsid -f cat {path}",
            "ionice -c 2 -n 7 cat {path}",
            "/usr/bin/time -p cat {path}",
        ],
    )
    def test_rejects_supported_wrapper_commands_accessing_sibling(
        self, scoped_command_workspace, command_template
    ):
        workspace, _, sibling_file = scoped_command_workspace
        tool = _guarded_tool(workspace)

        result = tool.run_json_sync(
            {"command": command_template.format(path=shlex.quote(str(sibling_file)))}
        )

        assert result["success"] is False
        assert result["return_code"] == 126
        assert "sibling secret" not in result["output"]

    @pytest.mark.parametrize(
        "command",
        [
            "exec cat own.txt",
            "env cat own.txt",
            "env -i NAME=value cat own.txt",
            "timeout --signal=TERM 5 cat own.txt",
            "nohup cat own.txt",
            "nice -n 5 cat own.txt",
            "stdbuf -oL cat own.txt",
            "command -p cat own.txt",
            "setsid -f cat own.txt",
            "ionice -c 2 -n 7 cat own.txt",
            "/usr/bin/time -p cat own.txt",
        ],
    )
    def test_supported_wrapper_commands_preserve_nested_command_classification(
        self, scoped_command_workspace, command
    ):
        workspace, _, _ = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate(command)

    @pytest.mark.parametrize(
        "command",
        [
            "sudo -n cat own.txt",
            "env sudo -n cat own.txt",
            "command sudo -n cat own.txt",
        ],
    )
    def test_rejects_sudo_privilege_wrapper(
        self,
        scoped_command_workspace,
        command,
    ):
        workspace, _, _ = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(
            CommandPolicyViolation,
            match="cannot safely inspect privilege escalation via sudo",
        ):
            guard.validate(command)

    def test_rejects_trusted_absolute_sudo_path_when_available(
        self,
        scoped_command_workspace,
    ):
        sudo = shutil.which("sudo")
        if sudo is None:
            pytest.skip("sudo is unavailable")
        workspace, _, _ = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(
            CommandPolicyViolation,
            match="cannot safely inspect privilege escalation via sudo",
        ):
            guard.validate(f"{shlex.quote(sudo)} -n cat own.txt")

    def test_rejects_path_shadowed_sudo_before_script_inspection(
        self,
        scoped_command_workspace,
        monkeypatch,
    ):
        workspace, _, _ = scoped_command_workspace
        fake_sudo = workspace.output_dir / "sudo"
        fake_sudo.write_text("#!/usr/bin/env bash\n:\n", encoding="utf-8")
        fake_sudo.chmod(0o755)
        monkeypatch.setenv(
            "PATH",
            f"{workspace.output_dir}{os.pathsep}{os.environ.get('PATH', '')}",
        )
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(
            CommandPolicyViolation,
            match="cannot safely inspect privilege escalation via sudo",
        ):
            guard.validate("sudo -n cat own.txt")

    @pytest.mark.parametrize(
        "command",
        [
            "chroot / cat own.txt",
            "sudo --chroot=/ cat own.txt",
            "sudo --shell",
            "/usr/bin/time --output=timing.txt cat own.txt",
        ],
    )
    def test_rejects_wrapper_modes_that_change_unmodeled_execution_context(
        self, scoped_command_workspace, command
    ):
        workspace, _, _ = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPolicyViolation):
            guard.validate(command)

    @pytest.mark.parametrize(
        ("variable", "value", "command_template"),
        [
            ("COMMAND", "cat", 'env "$COMMAND" own.txt'),
            ("ENV_ARGS", "UNUSED cat", "env -u $ENV_ARGS {path}"),
            ("TIMEOUT_ARGS", "1 cat", "timeout $TIMEOUT_ARGS {path}"),
        ],
    )
    def test_rejects_dynamic_wrapper_argv_shape(
        self,
        scoped_command_workspace,
        monkeypatch,
        variable,
        value,
        command_template,
    ):
        workspace, _, sibling_file = scoped_command_workspace
        monkeypatch.setenv(variable, value)
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPolicyViolation):
            guard.validate(command_template.format(path=shlex.quote(str(sibling_file))))

    def test_wrapper_nesting_depth_is_bounded(self, scoped_command_workspace):
        workspace, _, _ = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(
            CommandPolicyViolation, match="wrapper nesting depth exceeded"
        ):
            guard.validate(f"{'env ' * 33}cat own.txt")

    def test_wrapper_nesting_depth_survives_nested_dispatch(
        self, scoped_command_workspace
    ):
        workspace, _, _ = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(
            CommandPolicyViolation, match="wrapper nesting depth exceeded"
        ):
            guard.validate(f"{'command xargs ' * 33}cat own.txt")

    def test_env_chdir_applies_only_to_nested_command(self, scoped_command_workspace):
        workspace, external_file, _ = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate(
            f"env --chdir={shlex.quote(str(external_file.parent))} "
            f"cat {shlex.quote(external_file.name)}"
        )

        with pytest.raises(CommandPathViolation) as exc_info:
            guard.validate(
                f"env -C {shlex.quote(str(external_file.parent))} "
                f"rm -f {shlex.quote(external_file.name)}"
            )
        assert exc_info.value.access == "write"

    @pytest.mark.parametrize("directory_command", ["cd", "pushd"])
    def test_command_wrapper_propagates_shell_builtin_directory_state(
        self, scoped_command_workspace, directory_command
    ):
        workspace, external_file, _ = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPathViolation) as exc_info:
            guard.validate(
                f"command {directory_command} "
                f"{shlex.quote(str(external_file.parent))} "
                "&& printf leak > leak.txt"
            )
        assert exc_info.value.access == "write"

    @pytest.mark.parametrize("redirect", ["2>&1", "1>&2", "3<&0"])
    def test_descriptor_duplication_is_not_treated_as_path(
        self, scoped_command_workspace, redirect
    ):
        workspace, external_file, _ = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate(
            f"cd {shlex.quote(str(external_file.parent))} && printf ok {redirect}"
        )

    @pytest.mark.parametrize(
        "command_template",
        [
            "bash --rcfile {path} -i",
            "bash --init-file {path} -i -c exit",
            "bash --rcfile {path} -i -c exit",
        ],
    )
    def test_rejects_bash_file_options_outside_workspace(
        self, scoped_command_workspace, command_template
    ):
        workspace, _, sibling_file = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPathViolation):
            guard.validate(command_template.format(path=shlex.quote(str(sibling_file))))

    @pytest.mark.parametrize("option", ["--rcfile", "--init-file"])
    def test_bash_file_option_consumes_dash_prefixed_argument(
        self,
        scoped_command_workspace,
        option,
    ):
        workspace, _, _ = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPathViolation):
            guard.validate(f"bash {option} -c 'printf safe'")

    @pytest.mark.parametrize("option", ["--rcfile", "--init-file"])
    def test_rejects_attached_bash_long_file_option(
        self,
        scoped_command_workspace,
        option,
    ):
        workspace, _, _ = scoped_command_workspace
        init_file = workspace.output_dir / "safe.rc"
        init_file.write_text(":\n", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(
            CommandPolicyViolation,
            match="cannot safely inspect bash option",
        ):
            guard.validate(f"bash {option}={shlex.quote(str(init_file))} -c exit")

    @pytest.mark.parametrize("option", ["-o", "+o", "-O", "+O"])
    def test_bash_named_option_consumes_required_value(
        self,
        scoped_command_workspace,
        option,
    ):
        workspace, _, _ = scoped_command_workspace
        option_value = "posix" if option in {"-o", "+o"} else "extglob"
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate(f"bash {option} {option_value} -c 'printf safe'")

    @pytest.mark.parametrize(
        ("case", "expected_markers"),
        [
            ("minus-o", {"command"}),
            ("plus-o", {"command"}),
            ("minus-O", {"command"}),
            ("plus-O", {"command"}),
            ("short-cluster", {"command"}),
            ("stdin", {"stdin"}),
            ("file-option", {"init", "command"}),
            ("terminator", {"script"}),
            ("command-text", {"command"}),
            ("bare-minus-o", {"listing", "stdin"}),
            ("bare-plus-o", {"listing", "stdin"}),
            ("bare-minus-O", {"listing", "stdin"}),
            ("bare-plus-O", {"listing", "stdin"}),
            ("bare-minus-xo", {"listing", "stdin"}),
            ("bare-plus-xO", {"listing", "stdin"}),
        ],
    )
    def test_trusted_bash_invocation_outcome_matrix(
        self,
        scoped_command_workspace,
        trusted_bash_executable,
        case,
        expected_markers,
    ):
        workspace, _, _ = scoped_command_workspace
        markers = {
            "command": "__COMMAND_MARKER__",
            "init": "__INIT_MARKER__",
            "script": "__SCRIPT_MARKER__",
            "stdin": "__STDIN_MARKER__",
        }
        script = workspace.output_dir / "-policy-script"
        script.write_text(
            f"printf %s {markers['script']}\n",
            encoding="utf-8",
        )
        init_file = workspace.output_dir / "policy.rc"
        init_file.write_text(
            f"printf %s {markers['init']}\n",
            encoding="utf-8",
        )
        named_options = {
            "minus-o": ["-o", "posix"],
            "plus-o": ["+o", "posix"],
            "minus-O": ["-O", "extglob"],
            "plus-O": ["+O", "extglob"],
        }
        if case in named_options:
            arguments = [
                *named_options[case],
                "-c",
                f"printf %s {markers['command']}",
            ]
            stdin = ""
        elif case == "short-cluster":
            arguments = ["-xc", f"printf %s {markers['command']}"]
            stdin = ""
        elif case == "stdin":
            arguments = ["-s"]
            stdin = f"printf %s {markers['stdin']}"
        elif case == "file-option":
            arguments = [
                "--rcfile",
                str(init_file),
                "-i",
                "-c",
                f"printf %s {markers['command']}",
            ]
            stdin = ""
        elif case == "terminator":
            arguments = ["--", script.name]
            stdin = ""
        elif case == "command-text":
            arguments = ["-c", f"printf %s {markers['command']}"]
            stdin = ""
        else:
            arguments = [
                {
                    "bare-minus-o": "-o",
                    "bare-plus-o": "+o",
                    "bare-minus-O": "-O",
                    "bare-plus-O": "+O",
                    "bare-minus-xo": "-xo",
                    "bare-plus-xO": "+xO",
                }[case]
            ]
            stdin = f"printf %s {markers['stdin']}"

        environment = os.environ.copy()
        for name in command_path_guard_module._IMPLICIT_SHELL_ENVIRONMENT:
            environment.pop(name, None)
        completed = subprocess.run(
            [str(trusted_bash_executable), *arguments],
            cwd=workspace.output_dir,
            env=environment,
            input=stdin,
            capture_output=True,
            text=True,
            timeout=5,
        )

        assert completed.returncode == 0
        for outcome, marker in markers.items():
            if outcome in expected_markers:
                assert marker in completed.stdout
            else:
                assert marker not in completed.stdout
        if "listing" in expected_markers:
            assert completed.stdout.index(markers["stdin"]) > 0

        guard = WorkspaceCommandPathGuard(workspace)
        if expected_markers == {"stdin"} or "listing" in expected_markers:
            with pytest.raises(
                CommandPolicyViolation,
                match="without command text or a script",
            ):
                guard.validate_argv(["bash", *arguments])
        else:
            guard.validate_argv(["bash", *arguments])

    @pytest.mark.parametrize("option", ["-o", "+o", "-O", "+O", "-xo", "+xO"])
    def test_bare_bash_named_option_listing_remains_stdin_fed_and_rejected(
        self,
        scoped_command_workspace,
        option,
    ):
        workspace, _, _ = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(
            CommandPolicyViolation,
            match="without command text or a script",
        ):
            guard.validate(f"bash {option}")

    def test_guarded_executor_runs_named_bash_option_value_once(
        self,
        scoped_command_workspace,
    ):
        workspace, _, _ = scoped_command_workspace
        executor = _guarded_executor(workspace)

        result = executor.execute_command(
            ["bash", "-o", "posix", "-c", "printf %s __COMMAND_MARKER__"],
            shell=False,
        )

        assert result["return_code"] == 0
        assert result["output"] == "__COMMAND_MARKER__"

    @pytest.mark.parametrize("option", ["-o", "+o", "-O", "+O"])
    def test_guarded_executor_rejects_bare_bash_named_option_as_stdin_fed(
        self,
        scoped_command_workspace,
        caplog,
        option,
    ):
        workspace, _, _ = scoped_command_workspace
        executor = _guarded_executor(workspace)

        result = executor.execute_command(["bash", option], shell=False)

        assert result["return_code"] == 126
        assert result["error"].endswith("command denied by policy")
        assert "without command text or a script" in caplog.text
        assert "missing bash argument" not in caplog.text

    @pytest.mark.parametrize(
        "command",
        [
            "bash --rcfile",
            "bash --init-file",
            "bash -c",
        ],
    )
    def test_rejects_missing_bash_option_argument(
        self,
        scoped_command_workspace,
        command,
    ):
        workspace, _, _ = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPolicyViolation, match="missing bash argument"):
            guard.validate(command)

    @pytest.mark.parametrize(
        "command",
        [
            "bash $BASH_OPTIONS -c 'printf safe'",
            "bash -o $BASH_OPTION -c 'printf safe'",
            "bash --norc $BASH_OPTIONS -c 'printf safe'",
        ],
    )
    def test_rejects_dynamic_bash_option_region(
        self,
        scoped_command_workspace,
        command,
    ):
        workspace, _, _ = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(
            CommandPolicyViolation,
            match="cannot safely inspect dynamic bash option region",
        ):
            guard.validate(command)

    @pytest.mark.parametrize("cluster", ["-xc", "-cx", "-sc", "-cs"])
    def test_bash_short_clusters_consume_command_text(
        self,
        scoped_command_workspace,
        cluster,
    ):
        workspace, _, sibling_file = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate(
            f"bash {cluster} 'printf %s \"$1\"' ignored "
            f"{shlex.quote(str(sibling_file))}"
        )

    @pytest.mark.parametrize("command", ["bash", "bash -s", "bash --", "bash -"])
    def test_rejects_stdin_fed_or_missing_bash_input(
        self,
        scoped_command_workspace,
        command,
    ):
        workspace, _, _ = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(
            CommandPolicyViolation, match="without command text or a script"
        ):
            guard.validate(command)

    def test_dynamic_bash_command_positionals_are_not_option_region(
        self,
        scoped_command_workspace,
    ):
        workspace, _, _ = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate("bash -c 'printf safe' ignored \"$DYNAMIC_POSITIONAL\"")

    def test_shell_c_positional_arguments_are_not_treated_as_file_paths(
        self, scoped_command_workspace
    ):
        workspace, _, sibling_file = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate(
            f"bash -c 'printf %s \"$1\"' ignored {shlex.quote(str(sibling_file))}"
        )

    @pytest.mark.parametrize(
        "command_template",
        [
            'cat "$TARGET"',
            'cat "$(printf %s {path})"',
            "cat `printf %s {path}`",
        ],
    )
    def test_rejects_unresolved_expansion_in_supported_path_operand(
        self, scoped_command_workspace, command_template
    ):
        workspace, _, sibling_file = scoped_command_workspace
        tool = _guarded_tool(workspace)
        command = command_template.format(path=shlex.quote(str(sibling_file)))
        if "$TARGET" in command:
            command = f"TARGET={shlex.quote(str(sibling_file))}; {command}"

        result = tool.run_json_sync({"command": command})

        assert result["success"] is False
        assert result["return_code"] == 126
        assert "sibling secret" not in result["output"]

    def test_rejects_unresolved_expansion_in_redirect_path(
        self, scoped_command_workspace
    ):
        workspace, _, sibling_file = scoped_command_workspace
        tool = _guarded_tool(workspace)

        result = tool.run_json_sync(
            {
                "command": (
                    f"TARGET={shlex.quote(str(sibling_file))}; "
                    'printf changed > "$TARGET"'
                )
            }
        )

        assert result["success"] is False
        assert result["return_code"] == 126
        assert sibling_file.read_text(encoding="utf-8") == "sibling secret"

    def test_allows_unresolved_expansion_in_non_path_operand(
        self, scoped_command_workspace, monkeypatch
    ):
        workspace, _, _ = scoped_command_workspace
        own_file = workspace.output_dir / "own.txt"
        own_file.write_text("needle", encoding="utf-8")
        monkeypatch.setenv("PATTERN", "needle")
        tool = _guarded_tool(workspace)

        result = tool.run_json_sync(
            {"command": ('bash -c \'printf "%s\\n" "$1"\' _ "$PATTERN"')}
        )

        assert result["success"] is True
        assert result["output"] == "needle\n"

    def test_tilde_path_is_resolved_as_a_static_path(
        self, scoped_command_workspace, monkeypatch
    ):
        workspace, _, _ = scoped_command_workspace
        monkeypatch.setenv("HOME", str(workspace.output_dir))
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate("cat ~/own.txt")

    @pytest.mark.parametrize(
        "command",
        [
            "echo '",
            "for ((i=0;i<1;i++)); do cat sibling.txt; done",
            "coproc cat sibling.txt",
            "select x in a b; do cat sibling.txt; done",
            "cat $'sibling.txt'",
        ],
    )
    def test_unparsed_top_level_shell_input_fails_closed(
        self, scoped_command_workspace, command
    ):
        workspace, _, _ = scoped_command_workspace
        tool = _guarded_tool(workspace)

        result = tool.run_json_sync({"command": command})

        assert result["success"] is False
        assert result["return_code"] == 126

    def test_unparsed_nested_shell_input_fails_closed(self, scoped_command_workspace):
        workspace, _, _ = scoped_command_workspace
        tool = _guarded_tool(workspace)

        result = tool.run_json_sync({"command": 'sh -c "echo \'"'})

        assert result["success"] is False
        assert result["return_code"] == 126

    @pytest.mark.parametrize(
        "command_template",
        [
            "eval 'cat {path}'",
            "trap 'cat {path}' EXIT",
            "builtin eval 'cat {path}'",
            "command eval 'cat {path}'",
        ],
    )
    def test_rejects_shell_text_reentry_builtins(
        self, scoped_command_workspace, command_template
    ):
        workspace, _, sibling_file = scoped_command_workspace
        tool = _guarded_tool(workspace)

        result = tool.run_json_sync(
            {"command": command_template.format(path=shlex.quote(str(sibling_file)))}
        )

        assert result["success"] is False
        assert result["return_code"] == 126
        assert "sibling secret" not in result["output"]

    @pytest.mark.parametrize(
        "command_template",
        [
            "BASH_ENV=hook.sh bash -c true",
            "env BASH_ENV=hook.sh bash -c true",
            "export BASH_ENV=hook.sh; bash -c true",
        ],
    )
    def test_rejects_implicit_shell_initialization_files(
        self, scoped_command_workspace, command_template
    ):
        workspace, _, sibling_file = scoped_command_workspace
        (workspace.output_dir / "hook.sh").write_text(
            f"cat {shlex.quote(str(sibling_file))}\n",
            encoding="utf-8",
        )
        tool = _guarded_tool(workspace)

        result = tool.run_json_sync({"command": command_template})

        assert result["success"] is False
        assert result["return_code"] == 126
        assert "sibling secret" not in result["output"]

    def test_rejects_ambient_shell_initialization_file(
        self,
        scoped_command_workspace,
        monkeypatch,
    ):
        workspace, _, sibling_file = scoped_command_workspace
        hook = workspace.output_dir / "hook.sh"
        hook.write_text(
            f"cat {shlex.quote(str(sibling_file))}\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("BASH_ENV", str(hook))
        tool = _guarded_tool(workspace)

        result = tool.run_json_sync({"command": "printf safe"})

        assert result["success"] is False
        assert result["return_code"] == 126
        assert "sibling secret" not in result["output"]

    def test_rejects_ambient_cdpath_that_changes_directory_resolution(
        self,
        scoped_command_workspace,
        monkeypatch,
    ):
        workspace, _, sibling_file = scoped_command_workspace
        monkeypatch.setenv("CDPATH", str(sibling_file.parent.parent))
        tool = _guarded_tool(workspace)

        result = tool.run_json_sync(
            {"command": f"cd {shlex.quote(sibling_file.parent.name)}; cat secret.txt"}
        )

        assert result["return_code"] == 126
        assert "sibling secret" not in result["output"]

    @pytest.mark.parametrize(
        "command_template",
        [
            "cat <<EOF\n$(cat {path})\nEOF",
            "cat <<EOF\n`cat {path}`\nEOF",
            'cat <<<"$(cat {path})"',
        ],
    )
    def test_rejects_shell_execution_from_here_input(
        self, scoped_command_workspace, command_template
    ):
        workspace, _, sibling_file = scoped_command_workspace
        tool = _guarded_tool(workspace)

        result = tool.run_json_sync(
            {"command": command_template.format(path=shlex.quote(str(sibling_file)))}
        )

        assert result["success"] is False
        assert result["return_code"] == 126
        assert "sibling secret" not in result["output"]

    def test_allows_literal_here_document(self, scoped_command_workspace):
        workspace, _, _ = scoped_command_workspace
        tool = _guarded_tool(workspace)

        result = tool.run_json_sync({"command": "cat <<EOF\nliteral\nEOF"})

        assert result["success"] is True
        assert result["output"] == "literal\n"

    @pytest.mark.parametrize("command", ["", "   ", "\n", "# comment"])
    def test_guard_allows_noop_shell_input(self, scoped_command_workspace, command):
        workspace, _, _ = scoped_command_workspace
        tool = _guarded_tool(workspace)

        result = tool.run_json_sync({"command": command})

        assert result["success"] is True
        assert result["return_code"] == 0
        assert result["output"] == ""

    def test_guard_rejects_oversized_shell_input(self, scoped_command_workspace):
        workspace, _, _ = scoped_command_workspace
        tool = _guarded_tool(workspace)

        result = tool.run_json_sync({"command": "printf " + "x" * (64 * 1024)})

        assert result["success"] is False
        assert result["return_code"] == 126
        assert "command denied by policy" in result["error"]

    def test_guard_rejects_null_byte_input(self, scoped_command_workspace):
        workspace, _, _ = scoped_command_workspace
        tool = _guarded_tool(workspace)

        result = tool.run_json_sync({"command": "printf 'before\x00after'"})

        assert result["success"] is False
        assert result["return_code"] == 126
        assert "command denied by policy" in result["error"]

    @pytest.mark.parametrize(
        "command_template",
        [
            "cat <(cat {path})",
            "cat >(rm -f {path})",
        ],
    )
    def test_process_substitution_uses_nested_command_policy(
        self,
        scoped_command_workspace,
        command_template,
    ):
        workspace, _, sibling_file = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPathViolation):
            guard.validate(command_template.format(path=shlex.quote(str(sibling_file))))

    @pytest.mark.parametrize("absolute", [False, True])
    def test_rejects_direct_shell_script_with_out_of_scope_access(
        self, scoped_command_workspace, absolute
    ):
        workspace, _, sibling_file = scoped_command_workspace
        script = workspace.output_dir / "deploy.sh"
        script.write_text(
            f"#!/bin/sh\ncat {shlex.quote(str(sibling_file))}\n",
            encoding="utf-8",
        )
        script.chmod(0o755)
        command = str(script) if absolute else "./deploy.sh"
        tool = _guarded_tool(workspace)

        result = tool.run_json_sync({"command": shlex.quote(command)})

        assert result["success"] is False
        assert result["return_code"] == 126
        assert "sibling secret" not in result["output"]

    def test_rejects_direct_shell_script_in_literal_argv(
        self,
        scoped_command_workspace,
    ):
        workspace, _, sibling_file = scoped_command_workspace
        script = workspace.output_dir / "deploy.sh"
        script.write_text(
            f"#!/bin/sh\ncat {shlex.quote(str(sibling_file))}\n",
            encoding="utf-8",
        )
        script.chmod(0o755)
        executor = _guarded_executor(workspace)

        result = executor.execute_command(["./deploy.sh"], shell=False)

        assert result["return_code"] == 126
        assert "sibling secret" not in result["output"]

    @pytest.mark.parametrize("shebang", ["#!", "#!   "])
    def test_rejects_direct_script_without_shebang_interpreter(
        self,
        scoped_command_workspace,
        shebang,
    ):
        workspace, _, _ = scoped_command_workspace
        script = workspace.output_dir / "malformed-shebang"
        script.write_text(f"{shebang}\nprintf safe\n", encoding="utf-8")
        script.chmod(0o755)
        executor = _guarded_executor(workspace)

        result = executor.execute_command(["./malformed-shebang"], shell=False)

        assert result["return_code"] == 126
        assert "command denied by policy" in result["error"]
        assert "command validation failed" not in result["error"]

    @pytest.mark.parametrize("missing_flag", ["O_NONBLOCK", "O_NOFOLLOW"])
    def test_direct_script_inspection_requires_secure_open_flags(
        self,
        scoped_command_workspace,
        monkeypatch,
        missing_flag,
    ):
        workspace, _, _ = scoped_command_workspace
        script = workspace.output_dir / "direct-script"
        script.write_text("#!/usr/bin/env bash\nprintf safe\n", encoding="utf-8")
        script.chmod(0o755)
        monkeypatch.delattr(os, missing_flag)
        executor = _guarded_executor(workspace)

        result = executor.execute_command(["./direct-script"], shell=False)

        assert result["return_code"] == 126
        assert "command denied by policy" in result["error"]
        assert "command validation failed" not in result["error"]

    @pytest.mark.parametrize(
        "command",
        [
            "sh",
            "sh -s",
            "printf 'cat own.txt' | sh",
            "sh < run.sh",
        ],
    )
    def test_rejects_shell_input_that_cannot_be_inspected(
        self, scoped_command_workspace, command
    ):
        workspace, _, _ = scoped_command_workspace
        tool = _guarded_tool(workspace)

        result = tool.run_json_sync({"command": command})

        assert result["success"] is False
        assert result["return_code"] == 126

    @pytest.mark.parametrize(
        "invocation", ["bash {script}", "sh {script}", ". {script}", "source {script}"]
    )
    def test_rejects_shell_script_that_accesses_sibling(
        self, scoped_command_workspace, invocation
    ):
        workspace, _, sibling_file = scoped_command_workspace
        script = workspace.output_dir / "read-sibling.sh"
        script.write_text(
            f"cat {shlex.quote(str(sibling_file))}\n",
            encoding="utf-8",
        )
        tool = _guarded_tool(workspace)

        result = tool.run_json_sync(
            {"command": invocation.format(script=shlex.quote(str(script)))}
        )

        assert result["success"] is False
        assert result["return_code"] == 126
        assert "sibling secret" not in result["output"]

    def test_rejects_dynamically_created_shell_script_before_partial_execution(
        self, scoped_command_workspace
    ):
        workspace, _, sibling_file = scoped_command_workspace
        script = f"cat {shlex.quote(str(sibling_file))}"
        tool = _guarded_tool(workspace)

        result = tool.run_json_sync(
            {
                "command": (
                    f"printf '%s\\n' {shlex.quote(script)} > run.sh && bash run.sh"
                )
            }
        )

        assert result["success"] is False
        assert result["return_code"] == 126
        assert not (workspace.output_dir / "run.sh").exists()

    def test_shell_script_arguments_are_not_treated_as_paths(
        self, scoped_command_workspace
    ):
        workspace, _, sibling_file = scoped_command_workspace
        script = workspace.output_dir / "print-arg.sh"
        script.write_text("printf '%s\\n' \"$1\"\n", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate(
            f"bash {shlex.quote(str(script))} {shlex.quote(str(sibling_file))}"
        )

    @pytest.mark.parametrize("script_name", ["-c", "--rcfile"])
    def test_shell_option_terminator_treats_dash_prefixed_name_as_script(
        self,
        scoped_command_workspace,
        script_name,
    ):
        workspace, _, sibling_file = scoped_command_workspace
        script = workspace.output_dir / script_name
        script.write_text("printf 'safe script\\n'\n", encoding="utf-8")
        tool = _guarded_tool(workspace)

        result = tool.run_json_sync(
            {
                "command": (
                    f"bash -- {shlex.quote(script_name)} "
                    f"{shlex.quote(str(sibling_file))}"
                )
            }
        )

        assert result["return_code"] == 0
        assert result["output"] == "safe script\n"

    def test_rejects_unsafe_dash_prefixed_script_after_option_terminator(
        self, scoped_command_workspace
    ):
        workspace, _, sibling_file = scoped_command_workspace
        script = workspace.output_dir / "-c"
        script.write_text(
            f"cat {shlex.quote(str(sibling_file))}\n",
            encoding="utf-8",
        )
        tool = _guarded_tool(workspace)

        result = tool.run_json_sync({"command": "bash -- -c safe-argument"})

        assert result["return_code"] == 126
        assert "sibling secret" not in result["output"]

    def test_recursive_shell_script_inspection_is_bounded(
        self, scoped_command_workspace
    ):
        workspace, _, _ = scoped_command_workspace
        script = workspace.output_dir / "loop.sh"
        script.write_text("source loop.sh\n", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPolicyViolation, match="depth exceeded"):
            guard.validate("bash loop.sh")

    def test_rejects_unsafe_bash_initialization_file(self, scoped_command_workspace):
        workspace, _, sibling_file = scoped_command_workspace
        script = workspace.output_dir / "unsafe.rc"
        script.write_text(
            f"cat {shlex.quote(str(sibling_file))}\n",
            encoding="utf-8",
        )
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPathViolation):
            guard.validate(f"bash --rcfile {shlex.quote(str(script))} -i -c exit")

    def test_directory_stack_keeps_relative_paths_bound_to_real_cwd(
        self, scoped_command_workspace
    ):
        workspace, _, _ = scoped_command_workspace
        subdirectory = workspace.output_dir / "sub"
        subdirectory.mkdir()
        forbidden = workspace.base_dir / "forbidden" / "secret.txt"
        forbidden.parent.mkdir()
        forbidden.write_text("outside workspace", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPathViolation):
            guard.validate("pushd sub && popd && cat ../../forbidden/secret.txt")

        guard.validate("pushd sub && popd && cat own.txt")

    def test_cd_dash_uses_tracked_previous_directory(self, scoped_command_workspace):
        workspace, _, _ = scoped_command_workspace
        (workspace.output_dir / "sub").mkdir()
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate("cd sub && cd - && cat own.txt")

    def test_rejects_background_directory_state(self, scoped_command_workspace):
        workspace, _, _ = scoped_command_workspace
        (workspace.output_dir / "sub").mkdir()
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPolicyViolation):
            guard.validate("cd sub & cat own.txt")

    @pytest.mark.parametrize("operator", [";", "||"])
    def test_validates_each_possible_directory_state_across_shell_operator(
        self,
        scoped_command_workspace,
        operator,
    ):
        workspace, _, _ = scoped_command_workspace
        (workspace.output_dir / "sub").mkdir()
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate(f"cd sub {operator} cat own.txt")

        with pytest.raises(CommandPathViolation):
            guard.validate(f"cd sub {operator} cat ../../forbidden/secret.txt")

    @pytest.mark.parametrize(
        "command",
        [
            "cd sub; cat own.txt",
            "cd sub && cat own.txt; echo done",
            "cd sub && cat own.txt || echo fail",
        ],
    )
    def test_allows_common_directory_chains(
        self,
        scoped_command_workspace,
        command,
    ):
        workspace, _, _ = scoped_command_workspace
        (workspace.output_dir / "sub").mkdir()
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate(command)

    def test_tracks_conditional_directory_state_at_unconditional_join(
        self, scoped_command_workspace
    ):
        workspace, _, _ = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate("cd missing && true; rm -f ../outside.txt")
        guard.validate("cd sub && printf reached-only-after-success")

    def test_source_state_propagates_but_child_shell_state_does_not(
        self, scoped_command_workspace
    ):
        workspace, _, _ = scoped_command_workspace
        (workspace.output_dir / "sub").mkdir()
        script = workspace.output_dir / "change-directory.sh"
        script.write_text("cd sub\n", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate(
            f". {shlex.quote(str(script))} && cat ../../forbidden/secret.txt"
        )
        with pytest.raises(CommandPathViolation):
            guard.validate(
                f"bash {shlex.quote(str(script))} && cat ../../forbidden/secret.txt"
            )

    @pytest.mark.parametrize(
        "command_template",
        [
            'COMMAND=rm; printf "%s\\n" own.txt | xargs "$COMMAND"',
        ],
    )
    def test_rejects_dynamic_nested_command_names(
        self, scoped_command_workspace, command_template
    ):
        workspace, _, _ = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPolicyViolation):
            guard.validate(command_template)

    @pytest.mark.parametrize("effect_command", ["rm -f safe.sh", "python -c pass"])
    def test_effect_before_script_rejects_but_script_before_effect_is_allowed(
        self,
        scoped_command_workspace,
        effect_command,
    ):
        workspace, _, _ = scoped_command_workspace
        script = workspace.output_dir / "safe.sh"
        script.write_text(":\n", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(
            CommandPolicyViolation,
            match="cannot inspect a script affected by an earlier command",
        ):
            guard.validate(f"{effect_command}; bash safe.sh")

        guard.validate(f"bash safe.sh; {effect_command}")

    @pytest.mark.parametrize("effect_command", ["rm -f safe.sh", "python -c pass"])
    @pytest.mark.parametrize("script_first", [False, True])
    def test_pipeline_rejects_concurrent_script_effects_in_both_directions(
        self,
        scoped_command_workspace,
        effect_command,
        script_first,
    ):
        workspace, _, _ = scoped_command_workspace
        script = workspace.output_dir / "safe.sh"
        script.write_text(":\n", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)
        members = ["bash safe.sh", effect_command]
        if not script_first:
            members.reverse()

        with pytest.raises(
            CommandPolicyViolation,
            match="cannot inspect a script affected by a concurrent command",
        ):
            guard.validate(" | ".join(members))

    def test_effect_ledger_resets_for_fresh_top_level_validation(
        self,
        scoped_command_workspace,
    ):
        workspace, _, _ = scoped_command_workspace
        script = workspace.output_dir / "safe.sh"
        script.write_text(":\n", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate("python -c pass")
        guard.validate("bash safe.sh")

    def test_parse_attempt_budget_is_shared_across_nested_scripts(
        self,
        scoped_command_workspace,
        monkeypatch,
    ):
        workspace, _, _ = scoped_command_workspace
        (workspace.output_dir / "outer.sh").write_text(
            "bash inner.sh\n",
            encoding="utf-8",
        )
        (workspace.output_dir / "inner.sh").write_text(":\n", encoding="utf-8")
        monkeypatch.setattr(
            command_path_guard_module,
            "_MAX_POLICY_PARSE_ATTEMPTS",
            2,
            raising=False,
        )
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(
            CommandPolicyViolation,
            match="command policy parse attempt budget exceeded",
        ):
            guard.validate("bash outer.sh")

    def test_node_state_evaluation_budget_is_deterministic(
        self,
        scoped_command_workspace,
        monkeypatch,
    ):
        workspace, _, _ = scoped_command_workspace
        monkeypatch.setattr(
            command_path_guard_module,
            "_MAX_POLICY_NODE_STATE_EVALUATIONS",
            1,
            raising=False,
        )
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(
            CommandPolicyViolation,
            match="command policy node-state evaluation budget exceeded",
        ):
            guard.validate("true; true")

    def test_argv_token_budget_accepts_boundary_and_rejects_exhaustion(
        self,
        scoped_command_workspace,
    ):
        workspace, _, _ = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)
        limit = 8192

        guard.validate_argv(["unknown", *(["value"] * (limit - 1))])
        with pytest.raises(
            CommandPolicyViolation,
            match="command policy argv token budget exceeded",
        ):
            guard.validate_argv(["unknown", *(["value"] * limit)])

    def test_nested_public_reentry_shares_session_and_exception_resets_it(
        self,
        scoped_command_workspace,
        monkeypatch,
    ):
        workspace, _, _ = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)
        original = guard._validate_command_values
        reentered = False

        def reenter(*args, **kwargs):
            nonlocal reentered
            if not reentered:
                reentered = True
                guard.validate_argv(["nested", "value"])
            return original(*args, **kwargs)

        monkeypatch.setattr(command_path_guard_module, "_MAX_POLICY_ARGV_TOKENS", 3)
        monkeypatch.setattr(guard, "_validate_command_values", reenter)

        with pytest.raises(
            CommandPolicyViolation,
            match="command policy argv token budget exceeded",
        ):
            guard.validate_argv(["outer", "value"])

        monkeypatch.setattr(guard, "_validate_command_values", original)
        guard.validate_argv(["fresh", "value", "control"])

    def test_validation_sessions_are_isolated_across_concurrent_contexts(
        self,
        scoped_command_workspace,
        monkeypatch,
    ):
        workspace, _, _ = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)
        monkeypatch.setattr(command_path_guard_module, "_MAX_POLICY_ARGV_TOKENS", 1)

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(guard.validate_argv, (["one"], ["two"])))

        assert results == [None, None]

    def test_cumulative_source_character_budget_is_shared(
        self,
        scoped_command_workspace,
        monkeypatch,
    ):
        workspace, _, _ = scoped_command_workspace
        command = "bash -c ':'"
        monkeypatch.setattr(
            command_path_guard_module,
            "_MAX_POLICY_SOURCE_CHARS",
            len(command),
        )
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(
            CommandPolicyViolation,
            match="command policy source character budget exceeded",
        ):
            guard.validate(command)

    def test_cumulative_script_byte_budget_is_shared(
        self,
        scoped_command_workspace,
        monkeypatch,
    ):
        workspace, _, _ = scoped_command_workspace
        script_source = "# comment\n"
        (workspace.output_dir / "one.sh").write_text(script_source, encoding="utf-8")
        (workspace.output_dir / "two.sh").write_text(script_source, encoding="utf-8")
        monkeypatch.setattr(
            command_path_guard_module,
            "_MAX_POLICY_SCRIPT_BYTES",
            len(script_source.encode("utf-8")) * 2 - 1,
        )
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(
            CommandPolicyViolation,
            match="command policy script byte budget exceeded",
        ):
            guard.validate("bash one.sh; bash two.sh")

    @pytest.mark.parametrize("invocation", ["bash {script}", "source {script}"])
    @pytest.mark.parametrize("offset", [-1, 0, 1])
    def test_policy_script_byte_limit_boundary(
        self,
        scoped_command_workspace,
        invocation,
        offset,
    ):
        workspace, _, _ = scoped_command_workspace
        limit = command_path_guard_module._MAX_INSPECTED_SCRIPT_BYTES
        script = workspace.output_dir / f"boundary-{offset}.sh"
        _write_comment_script_bytes(script, limit + offset)
        guard = WorkspaceCommandPathGuard(workspace)
        command = invocation.format(script=shlex.quote(str(script)))

        if offset <= 0:
            guard.validate(command)
        else:
            with pytest.raises(
                CommandPolicyViolation,
                match=rf"shell policy script exceeds the {limit}-byte inspection limit",
            ):
                guard.validate(command)

    @pytest.mark.parametrize("offset", [0, 1])
    def test_policy_script_byte_limit_counts_multibyte_utf8(
        self,
        scoped_command_workspace,
        offset,
    ):
        workspace, _, _ = scoped_command_workspace
        limit = command_path_guard_module._MAX_INSPECTED_SCRIPT_BYTES
        script = workspace.output_dir / f"multibyte-{offset}.sh"
        payload = b"#" + ("é".encode("utf-8") * ((limit - 1) // 2))
        payload += b"x" * (limit + offset - len(payload))
        script.write_bytes(payload)
        guard = WorkspaceCommandPathGuard(workspace)

        if offset == 0:
            guard.validate(f"bash {shlex.quote(str(script))}")
        else:
            with pytest.raises(
                CommandPolicyViolation,
                match=rf"shell policy script exceeds the {limit}-byte inspection limit",
            ):
                guard.validate(f"bash {shlex.quote(str(script))}")

    def test_script_size_rejection_keeps_executor_generic_error(
        self,
        scoped_command_workspace,
    ):
        workspace, _, _ = scoped_command_workspace
        limit = command_path_guard_module._MAX_INSPECTED_SCRIPT_BYTES
        script = workspace.output_dir / "oversized.sh"
        _write_comment_script_bytes(script, limit + 1)
        tool = _guarded_tool(workspace)

        result = tool.run_json_sync({"command": f"bash {shlex.quote(str(script))}"})

        assert result["return_code"] == 126
        assert result["error"].endswith("command denied by policy")
        assert "inspection limit" not in result["error"]

    @pytest.mark.parametrize(
        "command",
        [
            "env python -c pass; bash safe.sh",
            "(python -c pass); bash safe.sh",
            "source effect.sh; bash safe.sh",
            "bash -c 'python -c pass'; bash safe.sh",
        ],
    )
    def test_unknown_effect_propagates_through_nested_shell_regions(
        self,
        scoped_command_workspace,
        command,
    ):
        workspace, _, _ = scoped_command_workspace
        (workspace.output_dir / "safe.sh").write_text("# safe\n", encoding="utf-8")
        (workspace.output_dir / "effect.sh").write_text(
            "python -c pass\n",
            encoding="utf-8",
        )
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(
            CommandPolicyViolation,
            match="cannot inspect a script affected by an earlier command",
        ):
            guard.validate(command)

    @pytest.mark.parametrize("command", ["env true", "xargs true"])
    def test_nested_argv_dispatch_charges_shared_token_budget(
        self,
        scoped_command_workspace,
        monkeypatch,
        command,
    ):
        workspace, _, _ = scoped_command_workspace
        monkeypatch.setattr(command_path_guard_module, "_MAX_POLICY_ARGV_TOKENS", 2)
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(
            CommandPolicyViolation,
            match="command policy argv token budget exceeded",
        ):
            guard.validate(command)

    def test_deterministic_budgets_allow_ordinary_nested_validation(
        self,
        scoped_command_workspace,
        monkeypatch,
    ):
        workspace, _, _ = scoped_command_workspace
        monkeypatch.setattr(command_path_guard_module, "_MAX_POLICY_PARSE_ATTEMPTS", 2)
        monkeypatch.setattr(command_path_guard_module, "_MAX_POLICY_SOURCE_CHARS", 32)
        monkeypatch.setattr(
            command_path_guard_module,
            "_MAX_POLICY_NODE_STATE_EVALUATIONS",
            2,
        )
        monkeypatch.setattr(command_path_guard_module, "_MAX_POLICY_ARGV_TOKENS", 4)
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate("bash -c ':'")

    def test_comment_only_input_consumes_parse_attempt_budget(
        self,
        scoped_command_workspace,
        monkeypatch,
    ):
        workspace, _, _ = scoped_command_workspace
        monkeypatch.setattr(command_path_guard_module, "_MAX_POLICY_PARSE_ATTEMPTS", 0)
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(
            CommandPolicyViolation,
            match="command policy parse attempt budget exceeded",
        ):
            guard.validate("# comment only\n")

    def test_new_effect_rejection_keeps_executor_exit_126_contract(
        self,
        scoped_command_workspace,
    ):
        workspace, _, _ = scoped_command_workspace
        (workspace.output_dir / "safe.sh").write_text("# safe\n", encoding="utf-8")
        tool = _guarded_tool(workspace)

        result = tool.run_json_sync(
            {"command": "python -c pass; bash safe.sh"},
        )

        assert result["success"] is False
        assert result["return_code"] == 126
        assert "command denied by policy" in result["error"]


class TestCdSymlinkTraversal:
    """`cd`/`pushd` logical-vs-physical divergence must not escape the workspace.

    Bash's ``cd`` (without ``-P``) is logical: it collapses ``..`` textually
    against the pre-symlink path. The guard resolves physically. When a target
    crosses a symlink and carries enough ``..`` to round-trip past its real
    depth, the two disagree and a fully-classified ``cd`` + ``cat``/``rm``
    sequence could reach a sibling tenant's files. The guard fails closed on any
    directory change that traverses a symlink.
    """

    def test_cd_through_symlink_with_dotdot_is_rejected(self, scoped_command_workspace):
        workspace, _, _ = scoped_command_workspace
        ws = Path(workspace.resolve_path(""))
        (ws / "a" / "b" / "c" / "d" / "e" / "f").mkdir(parents=True)
        (ws / "s").symlink_to(ws / "a" / "b" / "c" / "d" / "e" / "f")
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPolicyViolation, match="traverses a symlink"):
            guard.validate("cd s/../../../../../..")

    def test_pushd_through_symlink_is_rejected(self, scoped_command_workspace):
        workspace, _, _ = scoped_command_workspace
        ws = Path(workspace.resolve_path(""))
        (ws / "deep" / "nested").mkdir(parents=True)
        (ws / "link").symlink_to(ws / "deep" / "nested")
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPolicyViolation, match="traverses a symlink"):
            guard.validate("pushd link/../..")

    def test_symlink_free_cd_still_resolves(self, scoped_command_workspace):
        workspace, _, _ = scoped_command_workspace
        ws = Path(workspace.resolve_path(""))
        (ws / "sub" / "inner").mkdir(parents=True)
        guard = WorkspaceCommandPathGuard(workspace)

        # Real directories with a textual ``..`` cross no symlink and stay in
        # the workspace, so navigation and a subsequent read are allowed.
        guard.validate("cd sub/inner/.. && cat placeholder.txt")

    def test_guarded_executor_blocks_cd_symlink_escape(self, scoped_command_workspace):
        workspace, _, sibling_file = scoped_command_workspace
        ws = Path(workspace.resolve_path(""))
        (ws / "a" / "b" / "c" / "d" / "e" / "f").mkdir(parents=True)
        (ws / "s").symlink_to(ws / "a" / "b" / "c" / "d" / "e" / "f")
        tool = _guarded_tool(workspace)
        rel_secret = os.path.relpath(sibling_file, ws)

        result = tool.run_json_sync(
            {"command": f"cd s/../../../../../.. && cat {rel_secret}"}
        )

        assert result["success"] is False
        assert result["return_code"] == 126
        assert "sibling secret" not in (result.get("output") or "")
        assert sibling_file.read_text(encoding="utf-8") == "sibling secret"

    def test_guarded_executor_blocks_cd_symlink_escape_rm_variant(
        self, scoped_command_workspace
    ):
        workspace, _, sibling_file = scoped_command_workspace
        ws = Path(workspace.resolve_path(""))
        (ws / "a" / "b" / "c" / "d" / "e" / "f").mkdir(parents=True)
        (ws / "s").symlink_to(ws / "a" / "b" / "c" / "d" / "e" / "f")
        tool = _guarded_tool(workspace)
        rel_secret = os.path.relpath(sibling_file, ws)

        result = tool.run_json_sync(
            {"command": f"cd s/../../../../../.. && rm {rel_secret}"}
        )

        assert result["success"] is False
        assert result["return_code"] == 126
        assert sibling_file.exists()

    def test_quoted_brace_literal_is_checked_as_a_file_path(
        self, scoped_command_workspace
    ):
        # ``{}`` is no longer a cwd sentinel: a quoted literal is authorized as
        # the file itself and still resolves inside the workspace.
        workspace, _, _ = scoped_command_workspace
        ws = Path(workspace.resolve_path(""))
        (ws / "{}").write_text("brace", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate("cat '{}'")


class TestNoEffectCommandClassification:
    """Commands with no filesystem write effect must not poison script inspection.

    The unknown-effect flag stays session-wide (an *unknown* command has an
    unknowable target set), but commands that cannot write the filesystem, and
    path-scoped writers whose writes are recorded per-path, no longer trip it.
    """

    @pytest.mark.parametrize(
        "setup_command",
        [
            "echo hi",
            "printf hi",
            "pwd",
            "true",
            "test -f safe.sh",
            "export FOO=bar",
            "declare BAR=baz",
        ],
    )
    def test_no_effect_command_before_script_is_allowed(
        self, scoped_command_workspace, setup_command
    ):
        workspace, _, _ = scoped_command_workspace
        (workspace.output_dir / "safe.sh").write_text(":\n", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate(f"{setup_command} && bash safe.sh")

    def test_mkdir_before_script_is_allowed_and_scopes_write(
        self, scoped_command_workspace
    ):
        workspace, _, _ = scoped_command_workspace
        (workspace.output_dir / "safe.sh").write_text(":\n", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        # mkdir writes only its operand (recorded per-path), so a later,
        # unrelated script inspection is not blocked.
        guard.validate("mkdir out && bash safe.sh")

    def test_mkdir_outside_workspace_is_rejected(self, scoped_command_workspace):
        workspace, _, sibling_file = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPathViolation):
            guard.validate(f"mkdir {shlex.quote(str(sibling_file.parent / 'x'))}")

    @pytest.mark.parametrize(
        "mkdir_command",
        [
            "mkdir -m ../../../../../../etc newdir",
            "mkdir --mode ../../../../../../etc newdir",
            "mkdir -m0755 newdir",
            "mkdir --mode=0755 newdir",
        ],
    )
    def test_mkdir_mode_value_is_not_treated_as_a_path(
        self, scoped_command_workspace, mkdir_command
    ):
        # The `-m`/`--mode` value is consumed as the mode, not validated as a
        # path operand, so an out-of-workspace-looking mode token does not cause
        # a spurious rejection. Only the real directory operand is checked.
        workspace, _, _ = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate(mkdir_command)

    def test_mkdir_real_directory_outside_workspace_still_rejected_with_mode(
        self, scoped_command_workspace
    ):
        # Skipping the mode value must not skip the real path operand.
        workspace, _, sibling_file = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        target = shlex.quote(str(sibling_file.parent / "x"))
        with pytest.raises(CommandPathViolation):
            guard.validate(f"mkdir -m 0755 {target}")

    def test_redirect_write_from_no_effect_command_still_blocks_inspection(
        self, scoped_command_workspace
    ):
        # A no-effect command with a redirect still registers the redirect's
        # write, so inspecting that script afterwards is rejected. Declassifying
        # the command does not exempt its redirections.
        workspace, _, _ = scoped_command_workspace
        (workspace.output_dir / "safe.sh").write_text(":\n", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(
            CommandPolicyViolation,
            match="cannot inspect a script affected by an earlier command",
        ):
            guard.validate("echo hi > safe.sh; bash safe.sh")

    def test_unknown_command_still_poisons_script_inspection(
        self, scoped_command_workspace
    ):
        # Genuinely unknown commands keep the session-wide fail-closed contract.
        workspace, _, _ = scoped_command_workspace
        (workspace.output_dir / "safe.sh").write_text(":\n", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(
            CommandPolicyViolation,
            match="cannot inspect a script affected by an earlier command",
        ):
            guard.validate("python -c pass && bash safe.sh")
