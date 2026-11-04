"""Checks compose by extension; delivery callers do not know about formats."""

import hashlib
import logging
from pathlib import Path

from .models import (
    ArtifactCheck,
    ArtifactContent,
    CheckResult,
    InvalidArtifact,
    UncheckedArtifact,
    ValidationReport,
    unchecked,
)

logger = logging.getLogger(__name__)


class ArtifactCheckRegistry:
    def __init__(self) -> None:
        self._checks: dict[str, ArtifactCheck] = {}

    def register(self, check: ArtifactCheck) -> None:
        if not check.name or check.name in self._checks:
            raise ValueError(f"Duplicate or empty artifact check name: {check.name}")
        if not check.extensions or any(
            not ext.startswith(".") or ext != ext.lower() for ext in check.extensions
        ):
            raise ValueError("Check extensions must be nonempty lowercase suffixes")
        self._checks[check.name] = check

    def supports(self, filename: str) -> bool:
        suffix = Path(filename).suffix.lower()
        return any(suffix in check.extensions for check in self._checks.values())

    @property
    def extensions(self) -> frozenset[str]:
        return frozenset(
            ext for check in self._checks.values() for ext in check.extensions
        )

    def validate(self, content: ArtifactContent) -> ValidationReport:
        digest = hashlib.sha256(content.data).hexdigest()
        if len(content.data) > content.limits.max_bytes:
            return unchecked("File exceeds the validation byte budget.", sha256=digest)
        suffix = Path(content.filename).suffix.lower()
        checks = [c for c in self._checks.values() if suffix in c.extensions]
        if not checks:
            return unchecked(
                "No validator is installed for this format.", sha256=digest
            )
        results = []
        for check in checks:
            try:
                check.run(content)
            except InvalidArtifact as exc:
                results.append(CheckResult(check.name, "invalid", str(exc)))
            except UncheckedArtifact as exc:
                results.append(CheckResult(check.name, "unchecked", str(exc)))
            except ImportError:
                results.append(
                    CheckResult(
                        check.name, "unchecked", "Parser dependency is unavailable."
                    )
                )
            except Exception:
                # A checker bug is not evidence of a corrupt user file. Never
                # expose parser exceptions (which can contain paths/content).
                logger.exception("Artifact check %s failed unexpectedly", check.name)
                results.append(
                    CheckResult(
                        check.name, "unchecked", "Validator could not complete."
                    )
                )
            else:
                results.append(CheckResult(check.name, "valid", "Format check passed."))
            if results[-1].status != "valid":
                # Structural/budget preflights run before expensive decoders.
                break
        status = results[-1].status
        return ValidationReport(status, tuple(results), digest)
