"""Best-effort workspace path guard for shell commands.

This module deliberately provides a cooperative guard, not an operating-system
security boundary. For supported commands, it rejects out-of-scope paths and
path-bearing arguments that cannot be resolved statically. Unknown commands
remain unchanged.
"""

from __future__ import annotations

import errno
import logging
import os
import shutil
import stat
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import (
    Any,
    Iterable,
    Iterator,
    Sequence,
    SupportsIndex,
    cast,
)

try:
    import bashlex
except ModuleNotFoundError as _bashlex_import_error:  # pragma: no cover
    # ``bashlex`` ships in the optional ``command-path-guard`` extra, not the
    # base dependencies. Import lazily so importing this module (and collecting
    # its tests) never fails on a plain install; construction fails closed with
    # an actionable hint when the parser is genuinely required.
    bashlex = None
    _BASHLEX_IMPORT_ERROR: ModuleNotFoundError | None = _bashlex_import_error
else:
    _BASHLEX_IMPORT_ERROR = None

from ...workspace import TaskWorkspace
from .command_policy import (
    CommandPathViolation,
    CommandPolicyViolation,
    PathAccess,
    resolve_trusted_executable,
)

logger = logging.getLogger(__name__)

_MAX_COMMAND_POLICY_INPUT_CHARS = 64 * 1024
# Bounded budget for reading script/config files off disk before inspection.
# Numerically equal to the character input cap but kept in its own byte domain
# (compared against ``os.fstat().st_size`` and ``len(raw_bytes)``) so the two
# limits can move independently instead of silently coupling chars to bytes.
_MAX_INSPECTED_SCRIPT_BYTES = 64 * 1024
_MAX_INSPECTED_SCRIPT_DEPTH = 8
_MAX_COMMAND_WRAPPER_DEPTH = 32
_MAX_POSSIBLE_SHELL_STATES = 16
_MAX_POLICY_PARSE_ATTEMPTS = 64
_MAX_POLICY_SOURCE_CHARS = 1024 * 1024
_MAX_POLICY_SCRIPT_BYTES = 1024 * 1024
_MAX_POLICY_NODE_STATE_EVALUATIONS = 4096
_MAX_POLICY_ARGV_TOKENS = 8192


@dataclass
class _EffectScope:
    prior_written_paths: frozenset[Path] = frozenset()
    prior_unknown_effect: bool = False
    written_paths: set[Path] = field(default_factory=set)
    unknown_effect: bool = False
    inspected_scripts: set[Path] = field(default_factory=set)

    @property
    def visible_written_paths(self) -> frozenset[Path]:
        return self.prior_written_paths | self.written_paths

    @property
    def has_visible_unknown_effect(self) -> bool:
        return self.prior_unknown_effect or self.unknown_effect

    def inspect_script(self, script_path: Path) -> None:
        if self.has_visible_unknown_effect or script_path in self.visible_written_paths:
            raise CommandPolicyViolation(
                "cannot inspect a script affected by an earlier command"
            )
        self.inspected_scripts.add(script_path)

    def merge(self, child: _EffectScope) -> None:
        self.written_paths.update(child.written_paths)
        self.unknown_effect = self.unknown_effect or child.unknown_effect
        self.inspected_scripts.update(child.inspected_scripts)


@dataclass
class _ValidationSession:
    parse_attempts: int = 0
    source_chars: int = 0
    script_bytes: int = 0
    node_state_evaluations: int = 0
    argv_tokens: int = 0
    wrapped_bash_dispatch: bool = False
    effects: _EffectScope = field(default_factory=_EffectScope)

    def charge_parse(self, source_chars: int) -> None:
        self.parse_attempts += 1
        if self.parse_attempts > _MAX_POLICY_PARSE_ATTEMPTS:
            raise CommandPolicyViolation("command policy parse attempt budget exceeded")
        self.source_chars += source_chars
        if self.source_chars > _MAX_POLICY_SOURCE_CHARS:
            raise CommandPolicyViolation(
                "command policy source character budget exceeded"
            )

    def charge_script_bytes(self, script_bytes: int) -> None:
        self.script_bytes += script_bytes
        if self.script_bytes > _MAX_POLICY_SCRIPT_BYTES:
            raise CommandPolicyViolation("command policy script byte budget exceeded")

    def charge_node_state_evaluation(self) -> None:
        self.node_state_evaluations += 1
        if self.node_state_evaluations > _MAX_POLICY_NODE_STATE_EVALUATIONS:
            raise CommandPolicyViolation(
                "command policy node-state evaluation budget exceeded"
            )

    def charge_argv_tokens(self, argv_tokens: int) -> None:
        self.argv_tokens += argv_tokens
        if self.argv_tokens > _MAX_POLICY_ARGV_TOKENS:
            raise CommandPolicyViolation("command policy argv token budget exceeded")


_VALIDATION_SESSION: ContextVar[_ValidationSession | None] = ContextVar(
    "command_validation_session",
    default=None,
)


@contextmanager
def _validation_session_scope() -> Iterator[_ValidationSession]:
    session = _VALIDATION_SESSION.get()
    token = None
    if session is None:
        session = _ValidationSession()
        token = _VALIDATION_SESSION.set(session)
    try:
        yield session
    finally:
        if token is not None:
            _VALIDATION_SESSION.reset(token)


def _active_validation_session() -> _ValidationSession:
    session = _VALIDATION_SESSION.get()
    if session is None:
        raise RuntimeError("command validation requires an active session")
    return session


_READ_COMMANDS = {"cat"}
# Path-scoped writers: every operand is validated as a workspace write and
# recorded per-path, so they never poison the session-wide unknown-effect flag.
_WRITE_COMMANDS = {"rm", "mkdir"}
_SHELL_COMMANDS = {"bash"}
_UNSUPPORTED_SHELL_COMMANDS = {"dash", "sh", "zsh"}
_UNSUPPORTED_PRIVILEGE_COMMANDS = {"sudo"}
# Commands with no filesystem write effect of their own. Redirections are
# validated separately, so these can never mutate a later-inspected script and
# must not poison the session-wide unknown-effect flag. Keep this conservative:
# only add a command here once it is known to be unable to write the filesystem.
_NO_FILESYSTEM_EFFECT_COMMANDS = {
    "echo",
    "printf",
    "pwd",
    "true",
    "false",
    ":",
    "test",
    "[",
    "export",
    "declare",
    "typeset",
}
# Options that consume a following token as their value, per classified command,
# so the value is not misread as a path operand (e.g. `mkdir -m 0755 dir`).
_COMMAND_VALUE_OPTIONS = {
    "mkdir": frozenset({"-m", "--mode"}),
}
_BASH_FILE_OPTIONS = {"--init-file", "--rcfile"}
_BASH_LONG_FLAG_OPTIONS = {
    "--debug",
    "--debugger",
    "--dump-po-strings",
    "--dump-strings",
    "--help",
    "--login",
    "--noediting",
    "--noprofile",
    "--norc",
    "--posix",
    "--pretty-print",
    "--protected",
    "--restricted",
    "--verbose",
    "--version",
    "--wordexp",
}
_BASH_SHORT_FLAG_OPTIONS = frozenset("abefhiklmnprstuvxBCDHP")
_IMPLICIT_SHELL_ENVIRONMENT = {"BASH_ENV", "CDPATH", "ENV", "ZDOTDIR"}
_REJECTED_SHELL_ASSIGNMENTS = {
    *_IMPLICIT_SHELL_ENVIRONMENT,
    "BASHOPTS",
    "PATH",
    "SHELLOPTS",
}
_SAFE_DEVICE_PATHS = {
    Path("/dev/null"),
    Path("/dev/stdin"),
    Path("/dev/stdout"),
    Path("/dev/stderr"),
}
_POLICY_TIME_WRAPPER = "__t_"


def _secure_script_open_flags() -> int:
    """Return mandatory flags for race-resistant, non-blocking script reads."""
    nonblocking = getattr(os, "O_NONBLOCK", None)
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if not isinstance(nonblocking, int) or not isinstance(nofollow, int):
        raise CommandPolicyViolation(
            "secure direct-script inspection is unavailable on this platform"
        )
    return os.O_RDONLY | nonblocking | nofollow


@dataclass(frozen=True)
class _HeredocDeclaration:
    delimiter: str
    strip_tabs: bool
    quoted: bool


def _is_time_keyword_position(source: str, position: int) -> bool:
    if not source.startswith("time", position):
        return False
    after = position + len("time")
    if after >= len(source) or source[after] not in " \t":
        return False
    cursor = position - 1
    while cursor >= 0 and source[cursor] in " \t":
        cursor -= 1
    if cursor < 0 or source[cursor] in ";\n":
        return True
    return source[max(0, cursor - 1) : cursor + 1] in {"&&", "||"}


def _parse_heredoc_delimiter(
    source: str,
    start: int,
    normalized: list[str],
) -> tuple[_HeredocDeclaration | None, int]:
    cursor = start
    while cursor < len(source) and source[cursor] in " \t":
        cursor += 1
    if cursor >= len(source) or source[cursor] in "\r\n;|&()<>":
        return None, cursor

    token_start = cursor
    delimiter: list[str] = []
    quote: str | None = None
    quoted = False
    while cursor < len(source):
        character = source[cursor]
        if quote is None and character in " \t\r\n;|&()<>":
            break
        if quote is None and character in {"'", '"'}:
            quote = character
            quoted = True
            normalized[cursor] = " "
        elif quote == character:
            quote = None
            normalized[cursor] = " "
        elif quote is None and character == "\\":
            quoted = True
            normalized[cursor] = " "
            cursor += 1
            if cursor >= len(source):
                return None, cursor
            delimiter.append(source[cursor])
        else:
            delimiter.append(character)
        cursor += 1

    if quote is not None or not delimiter:
        return None, cursor
    delimiter_value = "".join(delimiter)
    if quoted:
        normalized[token_start:cursor] = delimiter_value.ljust(cursor - token_start)
    return (
        _HeredocDeclaration(
            delimiter=delimiter_value,
            strip_tabs=False,
            quoted=quoted,
        ),
        cursor,
    )


def _consume_heredoc_bodies(
    source: str,
    start: int,
    declarations: Sequence[_HeredocDeclaration],
    normalized: list[str],
) -> int:
    cursor = start
    for declaration in declarations:
        while cursor <= len(source):
            line_end = source.find("\n", cursor)
            if line_end < 0:
                line_end = len(source)
            line = source[cursor:line_end]
            comparable = line.lstrip("\t") if declaration.strip_tabs else line
            if comparable == declaration.delimiter:
                cursor = line_end + (line_end < len(source))
                break
            if declaration.quoted:
                for index in range(cursor, line_end):
                    normalized[index] = "x"
            if line_end == len(source):
                raise CommandPolicyViolation("cannot safely parse shell command")
            cursor = line_end + 1
        else:
            raise CommandPolicyViolation("cannot safely parse shell command")
    return cursor


def _has_active_unmodeled_expansion(raw_word: str) -> bool:
    """Return whether Bash may expand syntax omitted from the bashlex AST."""
    brace_has_expander: list[bool] = []
    bracket_start: int | None = None
    in_single_quote = False
    in_double_quote = False
    escaped = False
    index = 0

    while index < len(raw_word):
        character = raw_word[index]
        if escaped:
            escaped = False
            index += 1
            continue
        if character == "\\" and not in_single_quote:
            escaped = True
            index += 1
            continue
        if character == "'" and not in_double_quote:
            in_single_quote = not in_single_quote
            index += 1
            continue
        if character == '"' and not in_single_quote:
            in_double_quote = not in_double_quote
            index += 1
            continue
        if in_single_quote or in_double_quote:
            index += 1
            continue

        if character in {"*", "?"}:
            return True
        if character == "[" and bracket_start is None:
            bracket_start = index
        elif character == "]" and bracket_start is not None:
            if index > bracket_start + 1:
                return True
        elif character == "{":
            brace_has_expander.append(False)
        elif character == "," and brace_has_expander:
            brace_has_expander[-1] = True
        elif (
            character == "."
            and index + 1 < len(raw_word)
            and raw_word[index + 1] == "."
            and brace_has_expander
        ):
            brace_has_expander[-1] = True
            index += 1
        elif character == "}" and brace_has_expander:
            if brace_has_expander.pop():
                return True
        index += 1

    return False


def _mark_unmodeled_expansions(nodes: Sequence[Any], source: str) -> None:
    """Attach source-aware expansion metadata missing from the Bash AST."""
    pending = list(nodes)
    while pending:
        node = pending.pop()
        if getattr(node, "kind", None) == "word":
            start, end = cast(tuple[int, int], node.pos)
            node.xagent_has_unmodeled_expansion = _has_active_unmodeled_expansion(
                source[start:end]
            )
        for value in vars(node).values():
            if getattr(value, "kind", None) is not None:
                pending.append(value)
            elif isinstance(value, (list, tuple)):
                pending.extend(
                    item for item in value if getattr(item, "kind", None) is not None
                )


def _normalize_policy_shell_source(command: str) -> str:
    """Normalize narrowly supported Bash syntax that bashlex cannot parse.

    The normalized text is used only for policy parsing; the executor receives
    the original command. Replacements preserve source length so bashlex node
    positions still describe the original command. Quoted here-document bodies
    are literal by Bash definition, so masking them cannot hide executable
    expansion or nested commands.
    """
    normalized = list(command)
    declarations: list[_HeredocDeclaration] = []
    in_single_quote = False
    in_double_quote = False
    escaped = False
    in_comment = False
    cursor = 0

    while cursor < len(command):
        character = command[cursor]
        if in_comment:
            if character == "\n":
                in_comment = False
            else:
                cursor += 1
                continue
        if escaped:
            escaped = False
            cursor += 1
            continue
        if character == "\\" and not in_single_quote:
            escaped = True
            cursor += 1
            continue
        if character == "'" and not in_double_quote:
            in_single_quote = not in_single_quote
            cursor += 1
            continue
        if character == '"' and not in_single_quote:
            in_double_quote = not in_double_quote
            cursor += 1
            continue
        if in_single_quote or in_double_quote:
            cursor += 1
            continue

        if character == "#" and (cursor == 0 or command[cursor - 1] in " \t\r\n;|&()"):
            in_comment = True
            cursor += 1
            continue
        if _is_time_keyword_position(command, cursor):
            normalized[cursor : cursor + len("time")] = _POLICY_TIME_WRAPPER
            cursor += len("time")
            continue
        if (
            command.startswith("<<", cursor)
            and not command.startswith("<<<", cursor)
            and (cursor == 0 or command[cursor - 1] != "<")
        ):
            strip_tabs = command.startswith("<<-", cursor)
            delimiter_start = cursor + (3 if strip_tabs else 2)
            declaration, delimiter_end = _parse_heredoc_delimiter(
                command,
                delimiter_start,
                normalized,
            )
            if declaration is None:
                raise CommandPolicyViolation("cannot safely parse shell command")
            declarations.append(replace(declaration, strip_tabs=strip_tabs))
            cursor = delimiter_end
            continue
        if character == "\n" and declarations:
            cursor = _consume_heredoc_bodies(
                command,
                cursor + 1,
                declarations,
                normalized,
            )
            declarations.clear()
            continue
        cursor += 1

    if declarations or in_single_quote or in_double_quote or escaped:
        raise CommandPolicyViolation("cannot safely parse shell command")
    return "".join(normalized)


class _CommandValue(str):
    """A shell word plus whether bashlex proved it has no runtime expansion."""

    is_static: bool

    def __new__(cls, value: str, *, is_static: bool = True) -> _CommandValue:
        instance = super().__new__(cls, value)
        instance.is_static = is_static
        return instance

    def __getitem__(self, key: SupportsIndex | slice) -> str:
        value = super().__getitem__(key)
        if isinstance(key, slice):
            return _CommandValue(value, is_static=self.is_static)
        return value

    def split(
        self,
        sep: str | None = None,
        maxsplit: SupportsIndex = -1,
    ) -> list[str]:
        # Runtime values retain the marker while the public return type keeps
        # this str subclass substitutable for normal parser code.
        return [
            _CommandValue(value, is_static=self.is_static)
            for value in super().split(sep, maxsplit)
        ]


@dataclass(frozen=True)
class _ShellState:
    cwd: Path
    directory_stack: tuple[Path, ...] = ()
    previous_cwd: Path | None = None
    script_depth: int = 0
    wrapper_depth: int = 0


@dataclass(frozen=True)
class _CommandWrapperGrammar:
    flag_options: frozenset[str] = frozenset()
    value_options: frozenset[str] = frozenset()
    attached_short_value_options: frozenset[str] = frozenset()
    consumes_leading_scalar: bool = False
    allows_assignments: bool = False
    cwd_options: frozenset[str] = frozenset()
    terminal_options: frozenset[str] = frozenset()
    rejected_options: frozenset[str] = frozenset()
    propagates_state: bool = False


@dataclass(frozen=True)
class _BashInvocation:
    initialization_files: tuple[str, ...]
    command_text: str | None = None
    script: str | None = None
    reads_stdin: bool = False
    lists_options: bool = False


_COMMAND_WRAPPER_GRAMMARS = {
    _POLICY_TIME_WRAPPER: _CommandWrapperGrammar(
        flag_options=frozenset({"-p"}),
        propagates_state=True,
    ),
    "builtin": _CommandWrapperGrammar(
        propagates_state=True,
    ),
    "command": _CommandWrapperGrammar(
        flag_options=frozenset({"-p"}),
        terminal_options=frozenset({"-v", "-V"}),
        propagates_state=True,
    ),
    "env": _CommandWrapperGrammar(
        flag_options=frozenset({"-", "-0", "-i", "--ignore-environment", "--null"}),
        value_options=frozenset({"-a", "--argv0", "-C", "--chdir", "-u", "--unset"}),
        attached_short_value_options=frozenset({"-a", "-C", "-u"}),
        allows_assignments=True,
        cwd_options=frozenset({"-C", "--chdir"}),
        terminal_options=frozenset({"--help", "--version"}),
        rejected_options=frozenset({"-S", "--split-string"}),
    ),
    "exec": _CommandWrapperGrammar(
        flag_options=frozenset({"-c", "-l"}),
        value_options=frozenset({"-a"}),
        attached_short_value_options=frozenset({"-a"}),
    ),
    "ionice": _CommandWrapperGrammar(
        flag_options=frozenset({"-t", "--ignore"}),
        value_options=frozenset(
            {
                "-c",
                "--class",
                "-n",
                "--classdata",
                "-P",
                "--pgid",
                "-p",
                "--pid",
                "-u",
                "--uid",
            }
        ),
        attached_short_value_options=frozenset({"-c", "-n", "-P", "-p", "-u"}),
        terminal_options=frozenset(
            {
                "-P",
                "--pgid",
                "-p",
                "--pid",
                "-u",
                "--uid",
                "-h",
                "--help",
                "-V",
                "--version",
            }
        ),
    ),
    "nice": _CommandWrapperGrammar(
        value_options=frozenset({"-n", "--adjustment"}),
        attached_short_value_options=frozenset({"-n"}),
        terminal_options=frozenset({"--help", "--version"}),
    ),
    "nohup": _CommandWrapperGrammar(
        terminal_options=frozenset({"--help", "--version"}),
    ),
    "setsid": _CommandWrapperGrammar(
        flag_options=frozenset({"-c", "--ctty", "-f", "--fork", "-w", "--wait"}),
        terminal_options=frozenset({"-h", "--help", "-V", "--version"}),
    ),
    "stdbuf": _CommandWrapperGrammar(
        value_options=frozenset({"-e", "--error", "-i", "--input", "-o", "--output"}),
        attached_short_value_options=frozenset({"-e", "-i", "-o"}),
        terminal_options=frozenset({"--help", "--version"}),
    ),
    "time": _CommandWrapperGrammar(
        flag_options=frozenset(
            {"-a", "--append", "-p", "--portability", "-v", "--verbose"}
        ),
        value_options=frozenset({"-f", "--format"}),
        attached_short_value_options=frozenset({"-f"}),
        terminal_options=frozenset({"--help", "-V", "--version"}),
        rejected_options=frozenset({"-o", "--output"}),
    ),
    "timeout": _CommandWrapperGrammar(
        flag_options=frozenset(
            {"-f", "--foreground", "--preserve-status", "-v", "--verbose"}
        ),
        value_options=frozenset({"-k", "--kill-after", "-s", "--signal"}),
        attached_short_value_options=frozenset({"-k", "-s"}),
        consumes_leading_scalar=True,
        terminal_options=frozenset({"--help", "--version"}),
    ),
}

_CLASSIFIED_EXECUTABLE_COMMANDS = {
    *_READ_COMMANDS,
    *_WRITE_COMMANDS,
    *_SHELL_COMMANDS,
    *_UNSUPPORTED_SHELL_COMMANDS,
    *_UNSUPPORTED_PRIVILEGE_COMMANDS,
    "chroot",
    "xargs",
    *(
        name
        for name in _COMMAND_WRAPPER_GRAMMARS
        if name not in {"builtin", "command", "exec", _POLICY_TIME_WRAPPER}
    ),
}


class WorkspaceCommandPathGuard:
    """Validate Bash language boundaries against one task workspace."""

    def __init__(self, workspace: TaskWorkspace) -> None:
        if bashlex is None:
            raise ModuleNotFoundError(
                "WorkspaceCommandPathGuard requires the 'bashlex' parser. Install "
                "the optional extra: pip install 'xagent[command-path-guard]'."
            ) from _BASHLEX_IMPORT_ERROR
        self._workspace = workspace
        self._initial_cwd = workspace.resolve_path("").resolve()

    @property
    def execution_cwd(self) -> Path:
        """Return the canonical directory whose paths this guard validates."""
        return self._initial_cwd

    @staticmethod
    def _parse_shell(command: str) -> list[Any]:
        if len(command) > _MAX_COMMAND_POLICY_INPUT_CHARS:
            raise CommandPolicyViolation("command is too large to inspect")
        if "\x00" in command:
            raise CommandPolicyViolation("command contains a null byte")
        _active_validation_session().charge_parse(len(command))
        if not any(
            line.strip() and not line.lstrip().startswith("#")
            for line in command.splitlines()
        ):
            return []
        policy_source = _normalize_policy_shell_source(command)
        if bashlex is None:  # pragma: no cover - guarded at construction
            raise CommandPolicyViolation("shell command parser is unavailable")
        try:
            nodes = cast(list[Any], bashlex.parse(policy_source))
        except Exception as exc:
            # bashlex exposes several parser-internal exception types. Normalize
            # all of them at this boundary so malformed input never reaches the
            # executor through a parser-version-specific failure mode.
            #
            # Invariant: input bashlex cannot model (``$(( ))``, ``[[ ]]``,
            # ``case``, ``select``, ``coproc``, array assignments, ...) MUST stay
            # fail-closed here. Never relax this into a fail-open path to reduce
            # false positives on unparsable input.
            logger.debug("Command path guard rejected unparsed shell input")
            raise CommandPolicyViolation("cannot safely parse shell command") from exc
        _mark_unmodeled_expansions(nodes, policy_source)
        return nodes

    def validate(self, command: str) -> None:
        """Reject unsupported syntax and unsafe paths in ``command``."""
        with _validation_session_scope():
            active_environment = self._active_implicit_shell_environment()
            if active_environment is not None:
                raise CommandPolicyViolation(
                    "cannot safely inspect implicit shell initialization via "
                    f"{active_environment}"
                )
            nodes = self._parse_shell(command)
            states: tuple[_ShellState, ...] = (_ShellState(cwd=self._initial_cwd),)
            for node in nodes:
                states = self._validate_node_states(node, states)

    def _validate_node_states(
        self,
        node: Any,
        states: Sequence[_ShellState],
    ) -> tuple[_ShellState, ...]:
        if getattr(node, "kind", None) == "list":
            return self._validate_list_states(node, states)
        return self._dedupe_shell_states(
            self._validate_node(node, state) for state in states
        )

    def _validate_list_states(
        self,
        node: Any,
        states: Sequence[_ShellState],
    ) -> tuple[_ShellState, ...]:
        active = self._dedupe_shell_states(states)
        deferred: tuple[_ShellState, ...] = ()
        successful = active
        failed = active

        for part in node.parts:
            if getattr(part, "kind", None) in {"operator", "pipe"}:
                operator = getattr(part, "op", None)
                if operator == "&&":
                    deferred = self._dedupe_shell_states((*deferred, *failed))
                    active = successful
                elif operator == "||":
                    deferred = self._dedupe_shell_states((*deferred, *successful))
                    active = failed
                elif operator == ";":
                    active = self._dedupe_shell_states(
                        (*deferred, *successful, *failed)
                    )
                    deferred = ()
                else:
                    raise CommandPolicyViolation(
                        f"unsupported shell operator {operator!r}; split the "
                        "operation into separate command calls"
                    )
                continue

            before = active
            successful = self._validate_node_states(part, before)
            # A state-changing shell builtin can fail without changing the
            # process cwd. Keep that failure state until the shell operator
            # determines whether the next command runs.
            failed = before

        return self._dedupe_shell_states((*deferred, *successful, *failed))

    @staticmethod
    def _dedupe_shell_states(
        states: Iterable[_ShellState],
    ) -> tuple[_ShellState, ...]:
        unique = tuple(dict.fromkeys(states))
        if len(unique) > _MAX_POSSIBLE_SHELL_STATES:
            raise CommandPolicyViolation("too many possible shell directory states")
        return unique

    def validate_argv(self, argv: Sequence[str]) -> None:
        """Reject out-of-policy paths in literal, pre-tokenized arguments.

        No shell interprets this form, so shell expansion syntax remains literal.
        """
        self._validate_and_prepare_argv(argv, prepare_execution=False)

    def prepare_argv_for_execution(self, argv: Sequence[str]) -> list[str]:
        """Validate argv and bind direct Bash to its trusted executable identity."""
        return self._validate_and_prepare_argv(argv, prepare_execution=True)

    def _validate_and_prepare_argv(
        self,
        argv: Sequence[str],
        *,
        prepare_execution: bool,
    ) -> list[str]:
        with _validation_session_scope() as session:
            session.charge_argv_tokens(len(argv))
            execution_argv = list(argv)
            if not argv:
                return execution_argv
            if (
                prepare_execution
                and os.path.basename(execution_argv[0]) in _SHELL_COMMANDS
            ):
                execution_argv[0] = os.fspath(
                    resolve_trusted_executable(execution_argv[0])
                )
            self._validate_command_values(
                execution_argv[0],
                execution_argv[1:],
                _ShellState(cwd=self._initial_cwd),
                charge_argv=False,
            )
            if prepare_execution and session.wrapped_bash_dispatch:
                raise CommandPolicyViolation(
                    "cannot safely execute Bash through a command wrapper"
                )
            return execution_argv

    def _validate_node(self, node: Any, state: _ShellState) -> _ShellState:
        _active_validation_session().charge_node_state_evaluation()
        kind = getattr(node, "kind", None)
        if kind == "list":
            states = self._validate_list_states(node, (state,))
            if len(states) != 1:
                raise CommandPolicyViolation(
                    "cannot propagate multiple shell directory states"
                )
            return states[0]

        if kind == "pipeline":
            self._validate_pipeline(node, state)
            return state

        if kind == "compound":
            current = state
            reserved_words = [
                getattr(part, "word", None)
                for part in getattr(node, "list", ())
                if getattr(part, "kind", None) == "reservedword"
            ]
            for part in getattr(node, "list", ()):
                current = self._validate_node(part, current)
            for redirect in getattr(node, "redirects", ()):
                self._validate_redirect(redirect, current)
            return state if reserved_words[:1] == ["("] else current

        if kind == "command":
            return self._validate_command(node, state)

        self._validate_nested_nodes(node, state)
        return state

    def _validate_pipeline(self, node: Any, state: _ShellState) -> None:
        session = _active_validation_session()
        parent_scope = session.effects
        member_scopes: list[_EffectScope] = []
        try:
            for part in node.parts:
                if getattr(part, "kind", None) == "pipe":
                    continue
                member_scope = _EffectScope(
                    prior_written_paths=parent_scope.visible_written_paths,
                    prior_unknown_effect=parent_scope.has_visible_unknown_effect,
                )
                session.effects = member_scope
                self._validate_node(part, state)
                member_scopes.append(member_scope)
        finally:
            session.effects = parent_scope

        unknown_member_count = sum(scope.unknown_effect for scope in member_scopes)
        write_owner_counts: dict[Path, int] = {}
        for member_scope in member_scopes:
            for path in member_scope.written_paths:
                write_owner_counts[path] = write_owner_counts.get(path, 0) + 1

        for member_scope in member_scopes:
            concurrent_unknown = unknown_member_count > int(member_scope.unknown_effect)
            concurrent_write = any(
                write_owner_counts.get(script_path, 0)
                > int(script_path in member_scope.written_paths)
                for script_path in member_scope.inspected_scripts
            )
            if member_scope.inspected_scripts and (
                concurrent_unknown or concurrent_write
            ):
                raise CommandPolicyViolation(
                    "cannot inspect a script affected by a concurrent command"
                )

        for member_scope in member_scopes:
            parent_scope.merge(member_scope)

    def _validate_command(self, node: Any, state: _ShellState) -> _ShellState:
        words: list[Any] = []
        for part in node.parts:
            kind = getattr(part, "kind", None)
            if kind == "redirect":
                self._validate_redirect(part, state)
            elif kind == "word":
                words.append(part)
                self._validate_nested_nodes(part, state)
            elif kind == "assignment":
                self._reject_implicit_shell_environment((str(part.word),))
                self._validate_nested_nodes(part, state)
            else:
                self._validate_nested_nodes(part, state)

        if not words:
            return state

        command_word = self._command_value(words[0])
        if not command_word.is_static:
            raise CommandPolicyViolation("cannot resolve dynamic command name")
        args = self._command_values(words[1:])

        return self._validate_command_values(command_word, args, state)

    def _validate_command_values(
        self,
        command_word: str,
        args: Sequence[str],
        state: _ShellState,
        *,
        charge_argv: bool = True,
    ) -> _ShellState:
        if charge_argv:
            _active_validation_session().charge_argv_tokens(1 + len(args))
        command_name = os.path.basename(command_word)
        if command_name in _UNSUPPORTED_PRIVILEGE_COMMANDS:
            raise CommandPolicyViolation(
                f"cannot safely inspect privilege escalation via {command_name}"
            )
        if self._is_direct_command_path(command_word):
            if not self._is_trusted_system_command(command_word):
                self._inspect_direct_shell_script(command_word, state)
                return state
        elif command_name in _CLASSIFIED_EXECUTABLE_COMMANDS:
            if not self._is_trusted_system_command(command_word):
                discovered = shutil.which(command_name)
                if discovered is not None:
                    self._inspect_direct_shell_script(discovered, state)
                    return state

        if command_name in {"cd", "pushd", "popd"}:
            return self._change_directory(command_name, args, state)

        if command_name in {"alias", "eval", "hash", "trap"}:
            raise CommandPolicyViolation(
                f"cannot safely inspect shell text executed by {command_name}"
            )
        if command_name in {"declare", "export", "typeset"}:
            self._reject_implicit_shell_environment(args)
        if command_name in _READ_COMMANDS:
            self._check_operands(args, state.cwd, "read")
        elif command_name in _WRITE_COMMANDS:
            self._check_operands(
                args,
                state.cwd,
                "write",
                value_options=_COMMAND_VALUE_OPTIONS.get(command_name, frozenset()),
            )
        elif command_name in _SHELL_COMMANDS:
            self._check_nested_shell(command_name, args, state)
        elif command_name in _UNSUPPORTED_SHELL_COMMANDS:
            raise CommandPolicyViolation(
                f"shell dialect {command_name} does not match the Bash policy parser"
            )
        elif command_name in {".", "source"}:
            return self._check_sourced_shell(args, state)
        elif command_name == "xargs":
            self._check_xargs(args, state)
        elif command_name == "chroot":
            raise CommandPolicyViolation(
                "cannot safely inspect chroot filesystem remapping"
            )
        elif command_name in _COMMAND_WRAPPER_GRAMMARS:
            return self._check_command_wrapper(command_name, args, state)
        elif command_name in _NO_FILESYSTEM_EFFECT_COMMANDS:
            # No filesystem write effect of its own; any redirection is validated
            # separately, so this cannot have mutated a later-inspected script.
            pass
        else:
            _active_validation_session().effects.unknown_effect = True
        return state

    def _validate_redirect(self, node: Any, state: _ShellState) -> None:
        redirect_type = getattr(node, "type", "")
        if redirect_type in {"<<", "<<-"}:
            heredoc = getattr(node, "heredoc", None)
            body = str(getattr(heredoc, "value", ""))
            if "$" in body or "`" in body:
                raise CommandPolicyViolation(
                    "cannot safely inspect expanding here-document"
                )
            return
        output = getattr(node, "output", None)
        if redirect_type == "<<<":
            if getattr(output, "kind", None) is not None:
                self._validate_nested_nodes(output, state)
            return
        # bashlex emits descriptor-duplication targets such as 2>&1 as integers.
        if output is None or getattr(output, "kind", None) != "word":
            return
        raw_path = self._command_value(output)
        access: PathAccess = "read" if redirect_type == "<" else "write"
        self._check_path(raw_path, state.cwd, access)

    @staticmethod
    def _active_implicit_shell_environment() -> str | None:
        return next(
            (name for name in _IMPLICIT_SHELL_ENVIRONMENT if os.environ.get(name)),
            None,
        )

    @staticmethod
    def _policy_home_directory() -> str:
        home = os.environ.get("HOME")
        if not home:
            raise CommandPolicyViolation(
                "cannot resolve bare cd without HOME in the execution environment"
            )
        return home

    @staticmethod
    def _reject_symlink_traversing_chdir(target_word: str, cwd: Path) -> None:
        """Reject a directory change whose target path crosses a symlink.

        The guard tracks a *physically* resolved cwd, but Bash's ``cd``/``pushd``
        (without ``-P``) are *logical*: they collapse ``..`` textually against
        the pre-symlink path instead of following symlinks. When a target both
        crosses a symlink and carries ``..`` segments, the guard's resolved cwd
        diverges from where Bash actually lands, so a later relative operand can
        resolve inside the workspace while Bash reads/writes a sibling tenant's
        files. Physical and logical resolution agree whenever no symlink is
        traversed, so rejecting symlink-crossing directory changes keeps the
        tracked cwd equal to Bash's logical cwd. Fail closed rather than emulate
        logical ``cd``. ``cd -``/``popd``/stack rotation restore an
        already-validated, symlink-free cwd and do not reach this check.
        """
        if isinstance(target_word, _CommandValue) and not target_word.is_static:
            # Dynamic targets are rejected by the resolver below; do not probe
            # an unexpanded literal against the filesystem.
            return
        target = Path(target_word).expanduser()
        if target.is_absolute():
            probe = Path(target.anchor)
            parts = target.parts[1:]
        else:
            probe = cwd
            parts = target.parts
        for part in parts:
            if part == ".":
                continue
            if part == "..":
                probe = probe.parent
                continue
            probe = probe / part
            try:
                crosses_symlink = probe.is_symlink()
            except OSError as exc:
                raise CommandPolicyViolation(
                    "cannot inspect directory change target"
                ) from exc
            if crosses_symlink:
                raise CommandPolicyViolation(
                    "cannot safely resolve a directory change that traverses a "
                    "symlink; Bash resolves 'cd' logically while the guard "
                    "resolves physically"
                )

    def _change_directory(
        self,
        command_name: str,
        values: Sequence[str],
        state: _ShellState,
    ) -> _ShellState:
        operands = self._operands(values)
        if command_name == "cd":
            if len(operands) > 1:
                raise CommandPolicyViolation("cannot safely resolve cd arguments")
            target_word = operands[0] if operands else self._policy_home_directory()
            if target_word == "-":
                if state.previous_cwd is None:
                    raise CommandPolicyViolation(
                        "cannot resolve cd - without a previous directory"
                    )
                target = state.previous_cwd
            else:
                self._reject_symlink_traversing_chdir(target_word, state.cwd)
                target = self._check_path(target_word, state.cwd, "read")
            return replace(state, cwd=target, previous_cwd=state.cwd)

        if any(value.startswith("-") and value != "--" for value in values):
            raise CommandPolicyViolation(
                f"cannot safely resolve {command_name} options"
            )

        if command_name == "pushd":
            if len(operands) > 1:
                raise CommandPolicyViolation("cannot safely resolve pushd arguments")
            if operands:
                if operands[0] == "-":
                    raise CommandPolicyViolation("cannot safely resolve pushd target")
                self._reject_symlink_traversing_chdir(operands[0], state.cwd)
                target = self._check_path(operands[0], state.cwd, "read")
                stack = (state.cwd, *state.directory_stack)
            else:
                if not state.directory_stack:
                    raise CommandPolicyViolation(
                        "cannot resolve pushd without a directory stack"
                    )
                target = state.directory_stack[0]
                stack = (state.cwd, *state.directory_stack[1:])
            return replace(
                state,
                cwd=target,
                directory_stack=stack,
                previous_cwd=state.cwd,
            )

        if operands:
            raise CommandPolicyViolation("cannot safely resolve popd arguments")
        if not state.directory_stack:
            raise CommandPolicyViolation(
                "cannot resolve popd without a directory stack"
            )
        return replace(
            state,
            cwd=state.directory_stack[0],
            directory_stack=state.directory_stack[1:],
            previous_cwd=state.cwd,
        )

    def _check_command_wrapper(
        self,
        command_name: str,
        values: Sequence[str],
        state: _ShellState,
    ) -> _ShellState:
        if state.wrapper_depth >= _MAX_COMMAND_WRAPPER_DEPTH:
            raise CommandPolicyViolation("wrapper nesting depth exceeded")
        grammar = _COMMAND_WRAPPER_GRAMMARS[command_name]
        self._reject_dynamic_values(f"{command_name} arguments", values)
        index = 0
        options_done = False
        terminal_mode = False
        nested_state = replace(state, wrapper_depth=state.wrapper_depth + 1)

        while index < len(values):
            value = values[index]
            if not options_done and value == "--":
                options_done = True
                index += 1
                continue
            if options_done or not value.startswith("-"):
                break
            if value in grammar.rejected_options or any(
                value.startswith(f"{option}=")
                for option in grammar.rejected_options
                if option.startswith("--")
            ):
                raise CommandPolicyViolation(
                    f"cannot safely inspect {command_name} option {value}"
                )
            option, argument, consumed = self._parse_wrapper_option(
                command_name,
                values,
                index,
                grammar,
            )
            if option in grammar.cwd_options and argument is not None:
                target = self._check_path(argument, nested_state.cwd, "read")
                nested_state = replace(nested_state, cwd=target)
            terminal_mode = terminal_mode or option in grammar.terminal_options
            index += consumed

        if grammar.consumes_leading_scalar:
            if index >= len(values):
                return state
            index += 1

        if grammar.allows_assignments:
            assignments_start = index
            while index < len(values) and "=" in values[index]:
                index += 1
            self._reject_implicit_shell_environment(values[assignments_start:index])

        if terminal_mode or index >= len(values):
            return state

        command_word = values[index]
        if isinstance(command_word, _CommandValue) and not command_word.is_static:
            raise CommandPolicyViolation("cannot resolve dynamic command name")
        if os.path.basename(command_word) in _SHELL_COMMANDS:
            _active_validation_session().wrapped_bash_dispatch = True
        # Wrappers spawn a child command: validate its paths against the adjusted
        # child cwd, but never propagate child shell directory state to the parent.
        validated_state = self._validate_command_values(
            command_word, values[index + 1 :], nested_state
        )
        if not grammar.propagates_state:
            return state
        return replace(validated_state, wrapper_depth=state.wrapper_depth)

    def _parse_wrapper_option(
        self,
        command_name: str,
        values: Sequence[str],
        index: int,
        grammar: _CommandWrapperGrammar,
    ) -> tuple[str, str | None, int]:
        value = values[index]
        if value in grammar.flag_options or (
            value in grammar.terminal_options and value not in grammar.value_options
        ):
            return value, None, 1
        if value in grammar.value_options:
            if index + 1 >= len(values):
                raise CommandPolicyViolation(
                    f"missing {command_name} argument for {value}"
                )
            return value, values[index + 1], 2

        matching_long = next(
            (
                option
                for option in grammar.value_options
                if option.startswith("--") and value.startswith(f"{option}=")
            ),
            None,
        )
        if matching_long is not None:
            return (
                matching_long,
                self._derived_value(value, value.split("=", 1)[1]),
                1,
            )

        matching_short = next(
            (
                option
                for option in grammar.attached_short_value_options
                if value.startswith(option) and len(value) > len(option)
            ),
            None,
        )
        if matching_short is not None:
            return (
                matching_short,
                self._derived_value(value, value[len(matching_short) :]),
                1,
            )

        raise CommandPolicyViolation(
            f"cannot safely inspect {command_name} option {value}"
        )

    def _check_operands(
        self,
        values: Sequence[str],
        cwd: Path,
        access: PathAccess,
        *,
        value_options: frozenset[str] = frozenset(),
    ) -> None:
        for raw_path in self._operands(values, value_options=value_options):
            self._check_path(raw_path, cwd, access)

    def _read_policy_script(self, raw_path: str, cwd: Path) -> str:
        script_path = self._check_path(raw_path, cwd, "read")
        session = _active_validation_session()
        session.effects.inspect_script(script_path)

        descriptor: int | None = None
        try:
            descriptor = os.open(script_path, _secure_script_open_flags())
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                raise CommandPathViolation(access="read", path=script_path)
            if metadata.st_size > _MAX_INSPECTED_SCRIPT_BYTES:
                raise CommandPolicyViolation(
                    "shell policy script exceeds the "
                    f"{_MAX_INSPECTED_SCRIPT_BYTES}-byte inspection limit"
                )
            session.charge_script_bytes(metadata.st_size)
            with os.fdopen(descriptor, "rb") as script_file:
                descriptor = None
                raw_script = script_file.read(_MAX_INSPECTED_SCRIPT_BYTES + 1)
            if len(raw_script) > _MAX_INSPECTED_SCRIPT_BYTES:
                raise CommandPolicyViolation(
                    "shell policy script exceeds the "
                    f"{_MAX_INSPECTED_SCRIPT_BYTES}-byte inspection limit"
                )
            script = raw_script.decode("utf-8")
        except OSError as exc:
            # A missing script may be created earlier in the same shell string;
            # skipping inspection would allow its embedded file I/O unchecked.
            if exc.errno == errno.ELOOP:
                raise CommandPolicyViolation(
                    f"{script_path} symbolic link changed during inspection"
                ) from exc
            raise CommandPathViolation(access="read", path=script_path) from exc
        except UnicodeError as exc:
            raise CommandPolicyViolation(
                f"{script_path} is not a UTF-8 policy script"
            ) from exc
        finally:
            if descriptor is not None:
                os.close(descriptor)
        return script

    def _inspect_shell_script(
        self,
        raw_path: str,
        state: _ShellState,
        *,
        propagate_state: bool,
    ) -> _ShellState:
        if state.script_depth >= _MAX_INSPECTED_SCRIPT_DEPTH:
            raise CommandPolicyViolation("shell script inspection depth exceeded")
        script = self._read_policy_script(raw_path, state.cwd)
        nested_state = replace(state, script_depth=state.script_depth + 1)
        validated_states = self._validate_shell_states(script, nested_state)
        if not propagate_state:
            return state
        if len(validated_states) != 1:
            raise CommandPolicyViolation(
                "cannot propagate multiple shell directory states from source"
            )
        validated_state = validated_states[0]
        return replace(validated_state, script_depth=state.script_depth)

    @staticmethod
    def _is_direct_command_path(command_word: str) -> bool:
        return "/" in command_word

    @staticmethod
    def _is_trusted_system_command(command_word: str) -> bool:
        try:
            resolve_trusted_executable(command_word)
            return True
        except CommandPolicyViolation:
            return False

    def _inspect_direct_shell_script(
        self,
        raw_path: str,
        state: _ShellState,
    ) -> None:
        if state.script_depth >= _MAX_INSPECTED_SCRIPT_DEPTH:
            raise CommandPolicyViolation("shell script inspection depth exceeded")
        script_path = self._check_workspace_path(raw_path, state.cwd, "read")
        script = self._read_policy_script(str(script_path), state.cwd)

        first_line = script.splitlines()[0] if script.splitlines() else ""
        if first_line.startswith("#!"):
            interpreter_parts = first_line[2:].strip().split()
            interpreter = (
                os.path.basename(interpreter_parts[0]) if interpreter_parts else ""
            )
            if interpreter == "env":
                interpreter = (
                    os.path.basename(interpreter_parts[1])
                    if len(interpreter_parts) > 1
                    else ""
                )
            if interpreter not in _SHELL_COMMANDS:
                raise CommandPolicyViolation(
                    "direct scripts must use the Bash policy shell dialect"
                )

        nested_state = replace(state, script_depth=state.script_depth + 1)
        self._validate_shell_states(script, nested_state)

    @staticmethod
    def _reject_implicit_shell_environment(values: Sequence[str]) -> None:
        for value in values:
            name, _, _ = str(value).partition("=")
            if name in _REJECTED_SHELL_ASSIGNMENTS:
                raise CommandPolicyViolation(
                    f"cannot safely inspect shell execution environment via {name}"
                )

    def _check_nested_shell(
        self,
        command_name: str,
        literals: Sequence[str],
        state: _ShellState,
    ) -> None:
        active_environment = self._active_implicit_shell_environment()
        if active_environment is not None:
            raise CommandPolicyViolation(
                "cannot safely inspect implicit shell initialization via "
                f"{active_environment}"
            )
        invocation = self._parse_bash_invocation(literals)
        for initialization_file in invocation.initialization_files:
            self._inspect_shell_script(
                initialization_file,
                state,
                propagate_state=False,
            )
        if invocation.command_text is not None:
            self._validate_shell_states(invocation.command_text, state)
            return
        if invocation.reads_stdin:
            listing_context = (
                " after listing options" if invocation.lists_options else ""
            )
            raise CommandPolicyViolation(
                f"cannot inspect {command_name} input without command text or a script"
                f"{listing_context}"
            )
        if invocation.script is None:
            raise CommandPolicyViolation(
                f"cannot inspect {command_name} input without command text or a script"
            )
        self._inspect_shell_script(invocation.script, state, propagate_state=False)

    @staticmethod
    def _parse_bash_invocation(
        values: Sequence[str],
    ) -> _BashInvocation:
        initialization_files: list[str] = []
        index = 0
        while index < len(values):
            value = values[index]
            if isinstance(value, _CommandValue) and not value.is_static:
                raise CommandPolicyViolation(
                    "cannot safely inspect dynamic bash option region"
                )
            if value == "--":
                index += 1
                return _BashInvocation(
                    initialization_files=tuple(initialization_files),
                    script=values[index] if index < len(values) else None,
                    reads_stdin=index >= len(values),
                )
            if value in _BASH_FILE_OPTIONS:
                index += 1
                if index >= len(values):
                    raise CommandPolicyViolation(f"missing bash argument for {value}")
                argument = values[index]
                if isinstance(argument, _CommandValue) and not argument.is_static:
                    raise CommandPolicyViolation(
                        "cannot safely inspect dynamic bash option region"
                    )
                initialization_files.append(argument)
                index += 1
                continue
            if any(value.startswith(f"{option}=") for option in _BASH_FILE_OPTIONS):
                raise CommandPolicyViolation(
                    f"cannot safely inspect bash option {value}"
                )
            if value in _BASH_LONG_FLAG_OPTIONS:
                index += 1
                continue
            if value.startswith("--"):
                raise CommandPolicyViolation(
                    f"cannot safely inspect bash option {value}"
                )
            if value == "-":
                return _BashInvocation(
                    initialization_files=tuple(initialization_files),
                    reads_stdin=True,
                )
            if value.startswith(("-", "+")) and len(value) > 1:
                named_option_count = 0
                has_command_text = False
                stdin_mode = False
                for option in value[1:]:
                    if option in {"o", "O"}:
                        named_option_count += 1
                    elif option == "c":
                        has_command_text = True
                    elif option == "s":
                        stdin_mode = True
                    elif option not in _BASH_SHORT_FLAG_OPTIONS:
                        raise CommandPolicyViolation(
                            f"cannot safely inspect bash option {value}"
                        )

                index += 1
                for _ in range(named_option_count):
                    if index >= len(values):
                        return _BashInvocation(
                            initialization_files=tuple(initialization_files),
                            reads_stdin=True,
                            lists_options=True,
                        )
                    argument = values[index]
                    if isinstance(argument, _CommandValue) and not argument.is_static:
                        raise CommandPolicyViolation(
                            "cannot safely inspect dynamic bash option region"
                        )
                    index += 1
                if has_command_text:
                    if index >= len(values):
                        raise CommandPolicyViolation(
                            f"missing bash argument for {value}"
                        )
                    command_text = values[index]
                    if (
                        isinstance(command_text, _CommandValue)
                        and not command_text.is_static
                    ):
                        raise CommandPolicyViolation(
                            "cannot safely inspect dynamic bash option region"
                        )
                    return _BashInvocation(
                        initialization_files=tuple(initialization_files),
                        command_text=command_text,
                    )
                if stdin_mode:
                    return _BashInvocation(
                        initialization_files=tuple(initialization_files),
                        reads_stdin=True,
                    )
                continue
            return _BashInvocation(
                initialization_files=tuple(initialization_files),
                script=value,
            )
        return _BashInvocation(
            initialization_files=tuple(initialization_files),
            reads_stdin=True,
        )

    def _check_sourced_shell(
        self,
        values: Sequence[str],
        state: _ShellState,
    ) -> _ShellState:
        operands = self._operands(values)
        if not operands:
            raise CommandPolicyViolation("cannot inspect source without a script")
        return self._inspect_shell_script(
            operands[0],
            state,
            propagate_state=True,
        )

    def _check_xargs(self, values: Sequence[str], state: _ShellState) -> None:
        command_start = len(values)
        index = 0
        options_with_value = {
            "-a",
            "--arg-file",
            "-d",
            "--delimiter",
            "-E",
            "--eof",
            "-I",
            "--replace",
            "-L",
            "--max-lines",
            "-n",
            "--max-args",
            "-P",
            "--max-procs",
            "-s",
            "--max-chars",
        }
        attached_prefixes = ("-a", "-d", "-E", "-I", "-L", "-n", "-P", "-s")
        while index < len(values):
            value = values[index]
            if value == "--":
                command_start = index + 1
                break
            if value in options_with_value:
                index += 2
                continue
            if value.startswith("--") and "=" in value:
                index += 1
                continue
            if value.startswith(attached_prefixes) and len(value) > 2:
                index += 1
                continue
            if value.startswith("-") and value != "-":
                index += 1
                continue
            command_start = index
            break

        if command_start < len(values):
            # Validate the fixed command first so wrapper limits and any static
            # path violations keep their precise diagnostics. Runtime stdin
            # arguments remain uninspectable regardless of that result.
            self._validate_nested_command_words(values[command_start:], state)
        raise CommandPolicyViolation(
            "cannot safely inspect runtime file arguments produced by xargs"
        )

    def _validate_nested_command_words(
        self,
        words: Sequence[str],
        state: _ShellState,
    ) -> None:
        if not words:
            return
        if isinstance(words[0], _CommandValue) and not words[0].is_static:
            raise CommandPolicyViolation("cannot resolve dynamic command name")
        self._validate_command_values(words[0], words[1:], state)

    def _validate_shell_states(
        self,
        command: str,
        state: _ShellState,
    ) -> tuple[_ShellState, ...]:
        if isinstance(command, _CommandValue) and not command.is_static:
            raise CommandPolicyViolation("cannot resolve dynamic shell command text")
        nodes = self._parse_shell(command)
        current: tuple[_ShellState, ...] = (state,)
        for node in nodes:
            current = self._validate_node_states(node, current)
        return current

    def _validate_nested_nodes(self, node: Any, state: _ShellState) -> None:
        for attr_name in ("parts", "command", "list"):
            child = getattr(node, attr_name, None)
            if isinstance(child, list):
                for item in child:
                    if getattr(item, "kind", None) is not None:
                        self._validate_node(item, state)
            elif getattr(child, "kind", None) is not None:
                self._validate_node(child, state)

    @staticmethod
    def _command_value(word: Any) -> _CommandValue:
        parts = getattr(word, "parts", ())
        return _CommandValue(
            str(word.word),
            # Tilde expansion is deterministic at the policy boundary and is
            # resolved by Path.expanduser(); parser parts and source-marked
            # expansions that cannot be resolved here remain dynamic.
            is_static=not getattr(word, "xagent_has_unmodeled_expansion", False)
            and (
                not parts
                or all(getattr(part, "kind", None) == "tilde" for part in parts)
            ),
        )

    def _command_values(self, words: Sequence[Any]) -> list[_CommandValue]:
        return [self._command_value(word) for word in words]

    @staticmethod
    def _derived_value(source: str, value: str) -> _CommandValue:
        return _CommandValue(
            value,
            is_static=(not isinstance(source, _CommandValue) or source.is_static),
        )

    @staticmethod
    def _reject_dynamic_values(context: str, values: Sequence[str]) -> None:
        if any(
            isinstance(value, _CommandValue) and not value.is_static for value in values
        ):
            raise CommandPolicyViolation(f"cannot resolve dynamic {context}")

    @staticmethod
    def _operands(
        values: Sequence[str],
        *,
        value_options: frozenset[str] = frozenset(),
    ) -> list[str]:
        operands: list[str] = []
        options_done = False
        skip_next_value = False
        for value in values:
            if skip_next_value:
                # Separated value of a preceding value-consuming option.
                skip_next_value = False
                continue
            if not options_done and value == "--":
                options_done = True
                continue
            if not options_done and value.startswith("-") and value != "-":
                # An option that consumes a following token must not let that
                # value be misread as a path operand. Attached forms (`-m0755`,
                # `--mode=0755`) are already dropped whole; only a separated
                # value needs to be skipped explicitly.
                if value in value_options:
                    skip_next_value = True
                continue
            operands.append(value)
        return operands

    def _check_path(self, raw_path: str, cwd: Path, access: PathAccess) -> Path:
        return self._resolve_policy_path(
            raw_path,
            cwd,
            access,
            include_external_dirs=access == "read",
        )

    def _check_workspace_path(
        self,
        raw_path: str,
        cwd: Path,
        access: PathAccess,
    ) -> Path:
        return self._resolve_policy_path(
            raw_path,
            cwd,
            access,
            include_external_dirs=False,
        )

    def _resolve_policy_path(
        self,
        raw_path: str,
        cwd: Path,
        access: PathAccess,
        *,
        include_external_dirs: bool,
    ) -> Path:
        if isinstance(raw_path, _CommandValue) and not raw_path.is_static:
            raise CommandPolicyViolation(
                "cannot resolve dynamic path operand; enumerate concrete paths "
                "instead of using active globs or shell expansions"
            )

        if raw_path in {"", "-"}:
            return cwd

        candidate = Path(raw_path).expanduser()
        if not candidate.is_absolute():
            candidate = cwd / candidate

        if candidate in _SAFE_DEVICE_PATHS:
            return candidate

        try:
            resolved = self._workspace.resolve_authorized_path(
                candidate,
                base_dir=cwd,
                include_external_dirs=include_external_dirs,
            )
        except ValueError as exc:
            raise CommandPathViolation(access=access, path=candidate) from exc
        if access == "write":
            _active_validation_session().effects.written_paths.add(resolved)
        return resolved
