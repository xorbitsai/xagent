"""Regenerate ``legacy_redaction_golden.json`` from the pre-branch sources.

This is a manual maintenance tool, not a test helper: nothing imports or
invokes it during a test run. The fixture records, for every sample shape the
logging branch must keep masking, what the old redactors produced at
``ad253647`` -- the last commit on ``main`` before the branch. The tests in
``tests/web/test_request_log_redaction.py`` compare against those recorded
outputs to prove the new implementation is never weaker.

Usage, from the repository root and with the project virtualenv active::

    .venv/bin/python tests/web/fixtures/generate_legacy_redaction_golden.py
    .venv/bin/python tests/web/fixtures/generate_legacy_redaction_golden.py --base ad253647
    .venv/bin/python tests/web/fixtures/generate_legacy_redaction_golden.py --check

The sample set lives in the fixture itself. To add a sample, append an
``input`` with any ``legacy_output`` placeholder, run the script, and it
recomputes every output from the old source. Never edit ``legacy_output`` by
hand. ``--check`` recomputes the outputs and exits non-zero when the fixture
would change, which is useful before committing a hand-edited input.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

BASE_COMMIT = "ad253647"
FIXTURE_PATH = Path(__file__).with_name("legacy_redaction_golden.json")
REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE_FILES = {
    "security": "src/xagent/core/utils/security.py",
    "request_log_redaction": "src/xagent/web/request_log_redaction.py",
}


@dataclass(frozen=True)
class GoldenFunction:
    """One old function and the section of the fixture it fills in."""

    section: str
    module: str
    attribute: str


GOLDEN_FUNCTIONS = (
    GoldenFunction("redact_sensitive_text", "security", "redact_sensitive_text"),
    GoldenFunction(
        "redact_url_credentials_for_logging",
        "security",
        "redact_url_credentials_for_logging",
    ),
    GoldenFunction(
        "redact_request_target", "request_log_redaction", "redact_request_target"
    ),
)


def _load_source(base: str, relative_path: str, directory: Path) -> ModuleType:
    """Extract one file at ``base`` and load it as an isolated module."""
    completed = subprocess.run(
        ["git", "show", f"{base}:{relative_path}"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    path = directory / f"{Path(relative_path).stem}_{base}.py"
    path.write_text(completed.stdout, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(
        f"_legacy_{Path(relative_path).stem}_{base}", path
    )
    if spec is None or spec.loader is None:  # pragma: no cover - importlib detail
        raise SystemExit(f"cannot load {relative_path} at {base}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _resolve_callables(
    modules: dict[str, ModuleType],
) -> dict[str, Callable[[str], str]]:
    return {
        spec.section: getattr(modules[spec.module], spec.attribute)
        for spec in GOLDEN_FUNCTIONS
    }


def regenerate(base: str, *, check: bool) -> int:
    if not REPO_ROOT.joinpath(".git").exists():
        raise SystemExit(f"{REPO_ROOT} is not a git checkout")
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory(prefix="legacy-redaction-") as directory:
        modules = {
            name: _load_source(base, relative_path, Path(directory))
            for name, relative_path in SOURCE_FILES.items()
        }
        callables = _resolve_callables(modules)
        for spec in GOLDEN_FUNCTIONS:
            if spec.section not in fixture:
                raise SystemExit(f"fixture has no {spec.section!r} section")
            for record in fixture[spec.section]:
                record["legacy_output"] = callables[spec.section](record["input"])

    rendered = json.dumps(fixture, indent=2, ensure_ascii=False) + "\n"
    current = FIXTURE_PATH.read_text(encoding="utf-8")
    for spec in GOLDEN_FUNCTIONS:
        print(f"{spec.section}: {len(fixture[spec.section])} records")
    if check:
        if rendered != current:
            print(
                f"{FIXTURE_PATH} is stale; rerun without --check to rewrite it",
                file=sys.stderr,
            )
            return 1
        print(f"{FIXTURE_PATH} is up to date")
        return 0
    if rendered != current:
        FIXTURE_PATH.write_text(rendered, encoding="utf-8")
        print(f"wrote {FIXTURE_PATH}")
    else:
        print(f"{FIXTURE_PATH} already matches {base}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--base",
        default=BASE_COMMIT,
        help=f"commit the legacy redactors are read from (default: {BASE_COMMIT})",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the recorded outputs without rewriting the fixture",
    )
    args = parser.parse_args(argv)
    return regenerate(args.base, check=args.check)


if __name__ == "__main__":
    raise SystemExit(main())
