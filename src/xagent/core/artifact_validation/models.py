"""Transport-neutral contracts for bounded, format-specific artifact checks."""

from dataclasses import dataclass
from typing import Any, Callable, Literal

ValidationStatus = Literal["valid", "invalid", "unchecked"]


class InvalidArtifact(Exception):
    """A known format error; the message must be safe for user/model output."""


class UncheckedArtifact(Exception):
    """The check cannot establish validity (dependency, budget, encryption)."""


@dataclass(frozen=True)
class ValidationLimits:
    max_bytes: int = 32 * 1024 * 1024
    max_expanded_bytes: int = 64 * 1024 * 1024
    max_entries: int = 2048
    max_units: int = 200_000
    max_pixels: int = 25_000_000


@dataclass(frozen=True)
class ArtifactContent:
    """An immutable snapshot. Checks must not read paths or change files."""

    filename: str
    data: bytes
    limits: ValidationLimits


@dataclass(frozen=True)
class ArtifactCheck:
    name: str
    extensions: frozenset[str]
    run: Callable[[ArtifactContent], None]


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: ValidationStatus
    message: str


@dataclass(frozen=True)
class ValidationReport:
    status: ValidationStatus
    checks: tuple[CheckResult, ...]
    sha256: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "sha256": self.sha256,
            "checks": [
                {"name": c.name, "status": c.status, "message": c.message}
                for c in self.checks
            ],
        }


def unchecked(message: str, *, sha256: str | None = None) -> ValidationReport:
    return ValidationReport(
        "unchecked", (CheckResult("availability", "unchecked", message),), sha256
    )
