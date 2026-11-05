"""Contract tests for .github/scripts/format_release_notes.py.

The fixtures are the raw GitHub generated notes for v0.8.4...v0.8.5 and the
body that xprobebot published for v0.8.5, so the release workflow keeps
producing the same output as the bot it replaces.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / ".github" / "scripts" / "format_release_notes.py"
FIXTURES = Path(__file__).parent / "resources" / "release_notes"

_spec = importlib.util.spec_from_file_location("format_release_notes", SCRIPT)
assert _spec and _spec.loader
fmt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fmt)


def _read(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def test_matches_published_v0_8_5_body() -> None:
    out = fmt.refactor_release_notes(
        _read("v0.8.5_generated.md"), "xagent", "2026-11-01"
    )
    assert out == _read("v0.8.5_published.md")


def test_cli_reads_stdin_and_file() -> None:
    expected = _read("v0.8.5_published.md")
    args = [
        sys.executable,
        str(SCRIPT),
        "--repo-name",
        "xagent",
        "--date",
        "2026-11-01",
    ]
    from_file = subprocess.run(
        [*args, str(FIXTURES / "v0.8.5_generated.md")],
        capture_output=True,
        text=True,
        check=True,
    )
    from_stdin = subprocess.run(
        args,
        input=_read("v0.8.5_generated.md"),
        capture_output=True,
        text=True,
        check=True,
    )
    assert from_file.stdout == expected
    assert from_stdin.stdout == expected


RAW = """## What's Changed
* BUG: upper bug by @a in https://x/pull/1
* fix(scope): lower fix by @a in https://x/pull/2
* feature: long feature by @a in https://x/pull/3
* perf: speed by @a in https://x/pull/4
* test: not a label by @a in https://x/pull/5
* chore(ci): chore by @a in https://x/pull/6
* no label here by @a in https://x/pull/7
* doc: docs by @a in https://x/pull/8
* tst: tests by @a in https://x/pull/9
* enh: enhance by @a in https://x/pull/10
* BLD: build by @a in https://x/pull/11
* typ: types by @a in https://x/pull/12
* cln: cleanup by @a in https://x/pull/13
* ref(core): refactor by @a in https://x/pull/14

**Full Changelog**: https://github.com/o/r/compare/v1.0.0...v1.0.1
"""


def test_grouping_and_order() -> None:
    out = fmt.refactor_release_notes(RAW, "r", "2026-01-02")
    assert out.splitlines() == [
        "# What's new in 1.0.1 (2026-01-02)",
        "These are the changes in r v1.0.1.",
        "## New features",
        "* feature: long feature by @a in https://x/pull/3",
        "## Enhancements",
        "* enh: enhance by @a in https://x/pull/10",
        "* BLD: build by @a in https://x/pull/11",
        "* perf: speed by @a in https://x/pull/4",
        "* typ: types by @a in https://x/pull/12",
        "* cln: cleanup by @a in https://x/pull/13",
        "* ref(core): refactor by @a in https://x/pull/14",
        "## Bug fixes",
        "* BUG: upper bug by @a in https://x/pull/1",
        "* fix(scope): lower fix by @a in https://x/pull/2",
        "## Tests",
        "* tst: tests by @a in https://x/pull/9",
        "## Documentation",
        "* doc: docs by @a in https://x/pull/8",
        "## Others",
        "* test: not a label by @a in https://x/pull/5",
        "* no label here by @a in https://x/pull/7",
        "* chore(ci): chore by @a in https://x/pull/6",
        "",
        "**Full Changelog**: https://github.com/o/r/compare/v1.0.0...v1.0.1",
    ]
