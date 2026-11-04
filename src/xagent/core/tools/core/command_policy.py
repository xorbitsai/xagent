"""Contracts shared by command policy implementations and execution."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Literal, Protocol, Sequence, runtime_checkable

PathAccess = Literal["read", "write"]

_TRUSTED_EXECUTABLE_ROOTS = tuple(
    path.resolve()
    for path in (
        Path("/bin"),
        Path("/usr/bin"),
        Path("/usr/local/bin"),
        Path("/opt/homebrew/bin"),
    )
)


class CommandPolicyViolation(ValueError):
    """A command cannot be authorized under the active cooperative policy."""


class CommandPathViolation(CommandPolicyViolation):
    """A command path falls outside its allowed roots."""

    def __init__(self, *, access: PathAccess, path: Path) -> None:
        self.access = access
        self.path = path
        super().__init__(f"path is outside allowed {access} paths")


class CommandPolicyGuard(Protocol):
    """Validate command representations before process creation."""

    def validate(self, command: str) -> None: ...

    def validate_argv(self, argv: Sequence[str]) -> None: ...


@runtime_checkable
class CwdBoundCommandPolicy(Protocol):
    """Optionally bind command execution to one canonical absolute directory."""

    @property
    def execution_cwd(self) -> Path | None: ...


@runtime_checkable
class CommandArgvExecutionPolicy(Protocol):
    """Validate argv once and return the exact vector safe to execute."""

    def prepare_argv_for_execution(self, argv: Sequence[str]) -> list[str]: ...


def resolve_trusted_executable(executable: str) -> Path:
    """Resolve an executable to one canonical identity under a system root."""
    if "/" in executable and not Path(executable).is_absolute():
        raise CommandPolicyViolation(
            f"restricted command execution requires a trusted {executable} executable"
        )
    discovered = shutil.which(executable)
    if discovered is None:
        raise CommandPolicyViolation(
            f"restricted command execution requires a trusted {executable} executable"
        )
    try:
        resolved = Path(discovered).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise CommandPolicyViolation(
            f"cannot resolve trusted executable identity for {executable}"
        ) from exc
    if not resolved.is_file() or not any(
        resolved.is_relative_to(root) for root in _TRUSTED_EXECUTABLE_ROOTS
    ):
        raise CommandPolicyViolation(
            f"restricted command execution requires a trusted {executable} executable"
        )
    return resolved
