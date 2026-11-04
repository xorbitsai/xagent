"""Call-site gate for the clarification resolver and its payload codec.

``resolve_publishable_clarification`` is called only in the three
finalizers that settle a waiting run, and ``build_clarification_payload``
and ``parse_clarification_payload`` are not called outside their own
module. The gate compares each name's production call sites, as
``(file name, enclosing function)``, with ``EXPECTED_CALL_SITES`` by
equality, so another function in the same module calling one of these
names is red too. It looks at calls only: importing a name without calling
it is not counted.

AST, not substring grep: a source-text scan would also match this test
file's own docstrings and the defining module's own text, forcing every
future mention of these names in prose to dodge the scanner.

Known blind spots, not fixed here because closing them is out of scope for
a static AST scan of one package tree:

(a) Dynamic access: ``importlib.import_module(...)`` plus ``getattr(...)``
    matches none of the node shapes below.
(b) A name reached through an alias chain the AST walk does not resolve --
    re-exporting it under a different name from a third module and calling
    *that* name elsewhere.
(c) The defining-module exclusion is file-name-keyed, not path-keyed: any
    other file anywhere under this package that happened to also be named
    ``task_clarification_draft.py`` would be excluded from the scan right
    alongside the real one, on file name alone.
(d) The scan root is ``xagent.__path__`` (``src/xagent``), not the
    repository root -- a caller under the top-level ``scripts/`` directory
    is invisible to this gate entirely.
"""

from __future__ import annotations

import ast
import textwrap
from pathlib import Path

import pytest

import xagent

# Each name's production call sites, as (file name, enclosing function).
EXPECTED_CALL_SITES: dict[str, set[tuple[str, str | None]]] = {
    "resolve_publishable_clarification": {
        ("task_execution.py", "_finalize_task_execution_result_isolated"),
        ("task_execution.py", "_finalize_resumed_task"),
        ("managed_task_lease.py", "finalize_managed_task_lease_result"),
    },
    "build_clarification_payload": set(),
    "parse_clarification_payload": set(),
}
DEFINING_FILE = "task_clarification_draft.py"


def _enclosing_function_name(tree: ast.Module, lineno: int) -> str | None:
    best = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start = node.lineno
            end = max(
                (getattr(n, "lineno", start) for n in ast.walk(node)), default=start
            )
            if start <= lineno <= end:
                if best is None or start > best.lineno:
                    best = node
    return best.name if best is not None else None


def _call_sites(source: str, file_name: str) -> list[tuple[str, str, str | None]]:
    """``(name, file name, enclosing function)`` of every call of a watched
    name in ``source``, one entry per call."""
    tree = ast.parse(source)
    sites: list[tuple[str, str, str | None]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name):
            name = func.id
        elif isinstance(func, ast.Attribute):
            name = func.attr
        else:
            continue
        if name in EXPECTED_CALL_SITES:
            sites.append((name, file_name, _enclosing_function_name(tree, node.lineno)))
    return sites


def _production_modules() -> list[Path]:
    root = Path(next(iter(xagent.__path__)))
    modules = [path for path in root.rglob("*.py") if path.name != DEFINING_FILE]
    assert modules, "production scan set is empty"
    return modules


def _production_call_sites() -> dict[str, list[tuple[str, str | None]]]:
    found: dict[str, list[tuple[str, str | None]]] = {
        name: [] for name in EXPECTED_CALL_SITES
    }
    for path in _production_modules():
        try:
            source = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise RuntimeError(
                f"{path}: not valid UTF-8, cannot be AST-scanned by this gate"
            ) from exc
        for name, file_name, function in _call_sites(source, path.name):
            found[name].append((file_name, function))
    return found


# --------------------------------------------------------------------------
# The gate itself
# --------------------------------------------------------------------------


def test_clarification_names_are_called_only_at_the_expected_sites() -> None:
    found = _production_call_sites()
    # Sorted lists, so a second call in one expected function is red too.
    assert {name: sorted(sites) for name, sites in found.items()} == {
        name: sorted(sites) for name, sites in EXPECTED_CALL_SITES.items()
    }


# --------------------------------------------------------------------------
# Positive controls -- each call shape is recorded with its enclosing function
# --------------------------------------------------------------------------


def test_records_a_direct_call_with_its_enclosing_function() -> None:
    source = textwrap.dedent(
        """
        from xagent.web.services.task_clarification_draft import resolve_publishable_clarification

        def finalize(result, task, lease, anchor, now):
            return resolve_publishable_clarification(
                result, task=task, lease=lease, anchor=anchor, now=now
            )
        """
    )
    assert _call_sites(source, "x.py") == [
        ("resolve_publishable_clarification", "x.py", "finalize")
    ]


def test_records_an_attribute_call_with_its_enclosing_function() -> None:
    source = textwrap.dedent(
        """
        from xagent.web.services import task_clarification_draft

        def finalize(**kwargs):
            return task_clarification_draft.resolve_publishable_clarification(**kwargs)
        """
    )
    assert _call_sites(source, "x.py") == [
        ("resolve_publishable_clarification", "x.py", "finalize")
    ]


def test_records_a_module_level_call_without_an_enclosing_function() -> None:
    source = "resolve_publishable_clarification({})\n"
    assert _call_sites(source, "x.py") == [
        ("resolve_publishable_clarification", "x.py", None)
    ]


@pytest.mark.parametrize(
    "name", ["build_clarification_payload", "parse_clarification_payload"]
)
def test_records_a_call_of_each_payload_codec_name(name: str) -> None:
    source = f"def handler(value):\n    return {name}(value)\n"
    assert _call_sites(source, "x.py") == [(name, "x.py", "handler")]


def test_an_import_without_a_call_is_not_recorded() -> None:
    source = (
        "from xagent.web.services.task_clarification_draft import "
        "resolve_publishable_clarification\n"
    )
    assert _call_sites(source, "x.py") == []


# --------------------------------------------------------------------------
# Negative controls -- test consumers and the defining module
# --------------------------------------------------------------------------


def test_test_module_consumers_are_not_scanned() -> None:
    scanned = {str(path) for path in _production_modules()}
    this_file = str(Path(__file__).resolve())
    assert this_file not in scanned
    assert [path for path in scanned if "/tests/" in path] == []


def test_the_defining_module_is_not_scanned() -> None:
    modules = _production_modules()
    assert [path for path in modules if path.name == DEFINING_FILE] == []
