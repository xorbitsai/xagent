"""Zero-production-caller gate for ``resolve_publishable_clarification``.

This resolver ships with no production writer: nothing in the finalize
paths that settle a task run against its lease calls it yet. A static test
asserts exactly that -- no production code path calls it -- so the module
cannot end up half-wired, called from one finalizer but not the other two.

The gate's expected call count today is zero. It becomes exactly three
(one call in each of the two finalizers that settle a running task, one in
the finalizer that settles a resumed one) only once all three of those
finalizers call it, the task-side protocol marker exists, and the read
surface that consumes a published row is switched on together -- not one
finalizer at a time. This mirrors the existing zero-to-nonzero gate this
suite already carries for the interaction-staging primitive; see
``test_interaction_staging_production_gate.py`` for the precedent.

AST, not substring grep: a source-text scan would also match this test
file's own docstrings and this module's own definition of the gated name,
forcing every future mention of ``resolve_publishable_clarification`` in
prose to dodge the scanner.

Known blind spots, not fixed here because closing them is out of scope for
a static AST scan of one package tree:

(a) Dynamic access: ``importlib.import_module(...)`` plus ``getattr(...)``
    matches none of the node shapes below.
(b) A gated name reached through an alias chain the AST walk does not
    resolve -- re-exporting it under a different name from a third module
    and importing *that* name elsewhere.
(c) The primitive-module exclusion is stem-keyed, not path-keyed: any
    other file anywhere under this package that happened to also be named
    ``task_clarification_draft.py`` would be excluded from the scan right
    alongside the real one, on filename alone.
(d) The scan root is ``xagent.__path__`` (``src/xagent``), not the
    repository root -- a caller under the top-level ``scripts/`` directory
    is invisible to this gate entirely.
"""

from __future__ import annotations

import ast
import textwrap
from pathlib import Path

import xagent
from xagent.web.services import (  # noqa: F401 -- negative control import
    task_clarification_draft as resolver_module,
)

GATED_NAMES = frozenset({"resolve_publishable_clarification"})
PRIMITIVE_MODULE = "task_clarification_draft"


def _production_uses(source: str) -> set[str]:
    """Gated names this module imports or calls."""
    found: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[-1] == PRIMITIVE_MODULE:
                found.update(a.name for a in node.names if a.name in GATED_NAMES)
                if any(a.name == "*" for a in node.names):
                    found.update(GATED_NAMES)
            found.update(a.name for a in node.names if a.name == PRIMITIVE_MODULE)
        elif isinstance(node, ast.Import):
            found.update(
                a.name.split(".")[-1]
                for a in node.names
                if a.name.split(".")[-1] == PRIMITIVE_MODULE
            )
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in GATED_NAMES:
                found.add(func.id)
            elif isinstance(func, ast.Attribute) and func.attr in GATED_NAMES:
                found.add(func.attr)
    return found


def _production_modules() -> list[Path]:
    root = Path(next(iter(xagent.__path__)))
    modules = [path for path in root.rglob("*.py") if path.stem != PRIMITIVE_MODULE]
    assert modules, "production scan set is empty"
    return modules


# --------------------------------------------------------------------------
# The gate itself
# --------------------------------------------------------------------------


def test_no_production_module_calls_the_resolver() -> None:
    offenders: dict[str, set[str]] = {}
    for path in _production_modules():
        try:
            source = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise RuntimeError(
                f"{path}: not valid UTF-8, cannot be AST-scanned by this gate"
            ) from exc
        uses = _production_uses(source)
        if uses:
            offenders[str(path)] = uses
    assert offenders == {}, offenders


# --------------------------------------------------------------------------
# Positive controls -- three import/call shapes, each individually detected
# --------------------------------------------------------------------------


def test_detects_from_import_and_direct_call() -> None:
    source = textwrap.dedent(
        """
        from xagent.web.services.task_clarification_draft import resolve_publishable_clarification

        def finalize(result, task, lease, anchor, now):
            return resolve_publishable_clarification(
                result, task=task, lease=lease, anchor=anchor, now=now
            )
        """
    )
    assert _production_uses(source) == {"resolve_publishable_clarification"}


def test_detects_module_import_and_attribute_call() -> None:
    source = textwrap.dedent(
        """
        from xagent.web.services import task_clarification_draft

        def finalize(**kwargs):
            return task_clarification_draft.resolve_publishable_clarification(**kwargs)
        """
    )
    uses = _production_uses(source)
    assert "task_clarification_draft" in uses
    assert "resolve_publishable_clarification" in uses


def test_detects_bare_module_import() -> None:
    source = "import xagent.web.services.task_clarification_draft\n"
    assert _production_uses(source) == {"task_clarification_draft"}


def test_detects_star_import() -> None:
    source = "from xagent.web.services.task_clarification_draft import *\n"
    assert _production_uses(source) == GATED_NAMES


# --------------------------------------------------------------------------
# Negative controls -- test consumers and the module's own definition
# --------------------------------------------------------------------------


def test_test_module_consumers_are_not_flagged() -> None:
    scanned = {str(path) for path in _production_modules()}
    this_file = str(Path(__file__).resolve())
    assert this_file not in scanned
    assert all("/tests/" not in path for path in scanned)


def test_resolver_module_itself_is_not_flagged() -> None:
    modules = _production_modules()
    assert all(path.stem != PRIMITIVE_MODULE for path in modules)
