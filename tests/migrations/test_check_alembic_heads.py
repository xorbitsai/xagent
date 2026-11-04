#!/usr/bin/env python
"""Regression coverage for scripts/check_alembic_heads.sh.

That script is the repository's guard against a broken Alembic revision graph,
and it runs on every PR through the pre-commit job. Until these tests existed,
nothing proved it could actually fail -- and a check that cannot fail is worse
than no check, because it reads as protection that is not there.

Alembic enforces none of these invariants by itself:

* a branched graph is legal, and ``get_heads()`` returns every head with no
  warning and no error;
* a duplicate revision ID only produces a ``UserWarning``;
* only an unresolvable ``down_revision`` raises, and it raises a bare
  ``KeyError``.

Each test below builds a throwaway Alembic environment with one defect injected
and asserts the script rejects it, plus a healthy control that must pass so a
failure is attributable to the defect rather than to the fixture harness.

Usage:
    pytest tests/migrations/test_check_alembic_heads.py
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

project_root = Path(__file__).parent.parent.parent
SCRIPT = project_root / "scripts" / "check_alembic_heads.sh"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="requires bash")


def _write_migration(
    versions_dir: Path, filename: str, revision: str, down_revision: str | None
) -> None:
    down = "None" if down_revision is None else f'"{down_revision}"'
    versions_dir.joinpath(filename).write_text(
        textwrap.dedent(
            f'''\
            """fixture migration"""

            revision = "{revision}"
            down_revision = {down}
            branch_labels = None
            depends_on = None


            def upgrade() -> None:
                pass


            def downgrade() -> None:
                pass
            '''
        ),
        encoding="utf-8",
    )


def _alembic_env(tmp_path: Path, migrations: list[tuple[str, str, str | None]]) -> Path:
    """Build a minimal Alembic environment and return its root directory."""
    versions = tmp_path / "mig" / "versions"
    versions.mkdir(parents=True)
    tmp_path.joinpath("alembic.ini").write_text(
        "[alembic]\nscript_location = mig\n", encoding="utf-8"
    )
    for filename, revision, down_revision in migrations:
        _write_migration(versions, filename, revision, down_revision)
    return tmp_path


def _run_script(cwd: Path, home: Path) -> subprocess.CompletedProcess[str]:
    """Run the script with cwd as the Alembic root.

    The parent environment is inherited so that bash and the coreutils the
    script shells out to stay findable wherever they live -- a pinned PATH
    breaks on any host that does not keep them under /usr/bin.

    ALEMBIC_CHECK_PYTHON pins the invocation to this interpreter, which also
    makes the inherited PATH harmless: the script's default `uv run alembic`
    would try to resolve a uv project, and a fixture directory deliberately is
    not one. It carries a bare path, never a command line, so it needs no shell
    quoting even when sys.executable contains spaces. HOME is redirected so no
    user-level config leaks into the run.
    """
    env = os.environ.copy()
    env["ALEMBIC_CHECK_PYTHON"] = sys.executable
    env["HOME"] = str(home)
    return subprocess.run(
        ["bash", str(SCRIPT)],
        cwd=cwd,
        capture_output=True,
        text=True,
        env=env,
    )


def _run_check(env_root: Path) -> subprocess.CompletedProcess[str]:
    return _run_script(cwd=env_root, home=env_root)


def _report(result: subprocess.CompletedProcess[str]) -> str:
    """Render a script run for an assertion message."""
    return (
        f"exit={result.returncode}\n"
        f"--- stdout ---\n{result.stdout}"
        f"--- stderr ---\n{result.stderr}"
    )


def test_accepts_single_head(tmp_path: Path) -> None:
    """Positive control: without this, a failing test below proves nothing."""
    env = _alembic_env(tmp_path, [("a.py", "aaa", None), ("b.py", "bbb", "aaa")])
    result = _run_check(env)
    assert result.returncode == 0, _report(result)
    assert "single head confirmed" in result.stdout, _report(result)


def test_rejects_multiple_heads(tmp_path: Path) -> None:
    env = _alembic_env(
        tmp_path,
        [("a.py", "aaa", None), ("b.py", "bbb", "aaa"), ("c.py", "ccc", "aaa")],
    )
    result = _run_check(env)
    assert result.returncode == 1, _report(result)
    assert "expected exactly 1 head, found 2" in result.stderr, _report(result)
    assert "merge heads" in result.stderr, _report(result)


def test_rejects_duplicate_revision_ids(tmp_path: Path) -> None:
    """A duplicate ID must be named as such.

    Alembic only warns here, and the pre-fix script mis-reported it as two
    heads, sending the reader looking for a branch that does not exist.
    """
    env = _alembic_env(
        tmp_path,
        [("a.py", "aaa", None), ("d1.py", "bbb", "aaa"), ("d2.py", "bbb", "aaa")],
    )
    result = _run_check(env)
    assert result.returncode == 1, _report(result)
    assert "duplicate revision ID" in result.stderr, _report(result)


def test_rejects_unresolvable_down_revision(tmp_path: Path) -> None:
    """An orphaned down_revision must be named as such.

    Alembic raises a bare KeyError; the pre-fix script mis-reported it as
    "found 0 heads".
    """
    env = _alembic_env(
        tmp_path, [("a.py", "aaa", None), ("orphan.py", "bbb", "does_not_exist")]
    )
    result = _run_check(env)
    assert result.returncode == 1, _report(result)
    assert "could not read the revision graph" in result.stderr, _report(result)
    assert "not on disk" in result.stderr, _report(result)


def test_accepts_this_repository(tmp_path: Path) -> None:
    """The script must pass against the real migrations directory.

    Also a sanity check on the fixture-based tests above: if the script were
    broken outright, this would fail too.
    """
    result = _run_script(cwd=project_root, home=tmp_path)
    assert result.returncode == 0, _report(result)
    assert "single head confirmed" in result.stdout, _report(result)
