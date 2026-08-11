"""Zero-production-caller gate for ``stage_interaction_request`` and
``interaction_handoff``.

Because the delivery series intentionally ships the table with no production
writer and no production reader, a static test asserts exactly that: no
production code path calls the interaction staging primitive or the handoff
context manager. The gate exists so the table cannot be half-wired -- a task
that reports "waiting" with no answerable question, or an answerable question
the readers cannot see, is worse than a table nothing uses yet.

The gate is removable only by the change that wires all three finalizers, adds
the Task-side protocol marker, and switches the read surface together. Removing
it piecemeal recreates the divergence it was written to prevent.

Being part of the change that wires the first production caller does not, on
its own, grant permission to remove this gate: the condition above -- all
three finalizers, the protocol marker, and the read surface, together, in
one change -- is what actually retires it, not mere membership in that
change.

AST, not substring grep: a source-text scan would also match this test file's
own docstrings and this module's own definitions of the two gated names,
forcing every future mention of ``stage_interaction_request`` or
``interaction_handoff`` in prose to dodge the scanner. See
``test_trace_event_staging.py::test_trace_event_staging_module_sends_no_notifications``
for the precedent this follows -- the trace staging work moved that check from
substring to AST for the identical reason.

Known blind spots, not fixed here because closing them is out of scope for
a static AST scan of one package tree:

(a) Dynamic access: ``importlib.import_module(...)`` plus ``getattr(...)``
    matches none of the node shapes above. The gate pins the ordinary
    import-and-call surface, not every conceivable route.
(b) A gated name reached through an alias chain the AST walk does not
    resolve -- e.g. re-exporting ``stage_interaction_request`` under a
    different name from a third module and importing *that* name
    elsewhere. ``_production_uses`` matches names as written, not values
    traced through re-assignment.
(c) The primitive-module exclusion in ``_production_modules`` is
    stem-keyed (``path.stem != PRIMITIVE_MODULE``), not path-keyed: any
    other file anywhere under this package that happened to also be named
    ``task_interaction_staging.py`` would be excluded from the scan right
    alongside the real one, on filename alone.
(d) The scan root is ``xagent.__path__`` (``src/xagent``), not the
    repository root -- a caller under the top-level ``scripts/`` directory
    (outside that tree) is invisible to this gate entirely, positive
    controls and all.
"""

from __future__ import annotations

import ast
import textwrap
from pathlib import Path

import xagent
from xagent.web.services import (  # noqa: F401 -- negative control import
    task_interaction_staging as staging,
)

GATED_NAMES = frozenset({"stage_interaction_request", "interaction_handoff"})
PRIMITIVE_MODULE = "task_interaction_staging"


def _production_uses(source: str) -> set[str]:
    """Gated names this module imports or calls."""
    found: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom):
            # Two independent forms, not one gated by the other:
            # ``from ...task_interaction_staging import stage_interaction_request``
            # (module path ends with the primitive's module name) versus
            # ``from ...services import task_interaction_staging`` (the
            # module name is imported, from its *parent* package -- gating
            # this on the same module-path check would never fire, since
            # the parent package's own path does not end with
            # "task_interaction_staging").
            if (node.module or "").split(".")[-1] == PRIMITIVE_MODULE:
                found.update(a.name for a in node.names if a.name in GATED_NAMES)
                # ``from ...task_interaction_staging import *`` names neither
                # gated name explicitly -- the loop above finds nothing --
                # but it binds both of them into the importing module's
                # namespace all the same. Scoped to a module-path match
                # (not every star import in the codebase, which says
                # nothing about this primitive at all): both gated names
                # count as found whenever this specific module is the one
                # being star-imported.
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
    """Every source file under the installed xagent package, excluding the
    primitive's own definition module. Walked from the package's own
    __path__, not a hardcoded ``src/`` path -- a hardcoded path would scan an
    empty directory once this package is installed as a wheel, passing this
    gate for the wrong reason (nothing to scan) instead of the right one
    (nothing found). ``xagent.__file__`` itself is ``None`` -- the top-level
    package is a namespace package with no ``__init__.py`` -- so this walks
    ``__path__`` instead, which every package (namespace or regular) has."""

    root = Path(next(iter(xagent.__path__)))
    modules = [path for path in root.rglob("*.py") if path.stem != PRIMITIVE_MODULE]
    # A scan set that came back empty would make test_no_production_module_
    # imports_or_calls_the_gated_names pass vacuously -- offenders would stay
    # {} not because nothing was found, but because nothing was scanned.
    # xagent.__path__ resolving to the wrong root (a packaging change, a
    # broken install) is exactly the failure mode this catches before it
    # can masquerade as "gate still clean".
    assert modules, "production scan set is empty"
    return modules


# --------------------------------------------------------------------------
# T-GATE-1: the gate itself
# --------------------------------------------------------------------------


def test_no_production_module_imports_or_calls_the_gated_names() -> None:
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
# T-GATE-2: positive controls -- three import/call shapes, each must be
# individually detected
# --------------------------------------------------------------------------


def test_detects_from_import_and_direct_call() -> None:
    source = textwrap.dedent(
        """
        from xagent.web.services.task_interaction_staging import stage_interaction_request

        def handler(db, **kwargs):
            return stage_interaction_request(db, **kwargs)
        """
    )
    assert _production_uses(source) == {"stage_interaction_request"}


def test_detects_context_manager_call_by_name() -> None:
    source = textwrap.dedent(
        """
        from xagent.web.services.task_interaction_staging import interaction_handoff

        def handler(db, lease, task, anchor, now):
            with interaction_handoff(db, lease, task=task, anchor=anchor, now=now) as h:
                pass
        """
    )
    assert _production_uses(source) == {"interaction_handoff"}


def test_detects_module_import_and_attribute_call() -> None:
    source = textwrap.dedent(
        """
        from xagent.web.services import task_interaction_staging

        def handler(db, **kwargs):
            return task_interaction_staging.stage_interaction_request(db, **kwargs)
        """
    )
    uses = _production_uses(source)
    assert "task_interaction_staging" in uses
    assert "stage_interaction_request" in uses


def test_detects_bare_module_import() -> None:
    source = "import xagent.web.services.task_interaction_staging\n"
    assert _production_uses(source) == {"task_interaction_staging"}


def test_detects_star_import() -> None:
    """``from ...task_interaction_staging import *`` names neither gated
    name in its own ``ast.ImportFrom.names`` list -- a plain per-alias scan
    finds nothing there and would let this shape through as a false
    negative, even though it binds both names into the importing module's
    namespace. No call site is needed to expose the gap: the import alone
    must report both gated names found."""

    source = "from xagent.web.services.task_interaction_staging import *\n"
    assert _production_uses(source) == GATED_NAMES


# --------------------------------------------------------------------------
# T-GATE-3: negative controls -- test consumers and the module's own
# definitions must not be flagged
# --------------------------------------------------------------------------


def test_test_module_consumers_are_not_flagged() -> None:
    """Test consumers are allowed: this file, and every other test module
    that imports the two gated names (``test_interaction_staging.py``, this
    file's own negative-control import above), live under ``tests/``, which
    ``_production_modules`` never walks -- it starts from ``xagent.__path__``
    (``src/xagent``), not the repository root. The scanner (``_production_uses``)
    is a generic AST function that would legitimately find those imports if
    pointed at a test file directly (already exercised by T-GATE-2); what
    makes test consumption safe is the *scan set*, not the scanner being
    unable to see it. This test pins that boundary directly, on the real
    scan set, rather than re-deriving it from a synthetic source string."""

    scanned = {str(path) for path in _production_modules()}
    this_file = str(Path(__file__).resolve())
    assert this_file not in scanned
    assert all("/tests/" not in path for path in scanned)


def test_primitive_module_itself_is_not_flagged() -> None:
    """The primitive module's own body -- definitions, its one internal call
    from ``_InteractionHandoff.stage()`` to ``stage_interaction_request``,
    and its docstrings mentioning both names in prose -- must not trip the
    gate. ``_production_modules()`` excludes it by filename; this test pins
    that exclusion rather than re-deriving it."""

    from xagent.web.services import task_interaction_staging as module

    modules = _production_modules()
    assert all(path.stem != "task_interaction_staging" for path in modules)
    # Also confirm the module's own source, scanned directly, DOES contain a
    # gated name (stage_interaction_request, called by name from
    # _InteractionHandoff.stage()) -- proving the filename exclusion is
    # doing real work, not vacuously passing because the file has nothing
    # for the scanner to find. interaction_handoff itself is never called
    # from within its own module (only entered via a caller's `with`), so
    # it does not appear here the same way -- that asymmetry is expected,
    # not a gap in the scanner.
    own_source = Path(module.__file__).read_text(encoding="utf-8")
    assert "stage_interaction_request" in _production_uses(own_source)


def test_staging_module_defines_no_second_vocabulary_copy() -> None:
    """The staging module must consume the model's column-domain constants,
    never redeclare them. The merge-order rule for the origin vocabulary is
    that the model owns the one literal list (INTERACTION_ORIGIN_VOCABULARY,
    INTERACTION_PROTOCOL_VERSION); every other surface derives from it. This
    guard fails if task_interaction_staging.py grows its own literal set,
    tuple, or list re-enumerating vocabulary members, its own integer
    protocol-version constant, or an inline origin-normalization function --
    each of which would be a drift-capable second copy.
    """
    import xagent.web.services.task_interaction_staging as staging_module

    source = Path(staging_module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)

    from xagent.web.models.task_interaction import INTERACTION_ORIGIN_VOCABULARY

    for node in ast.walk(tree):
        # No literal collection may re-enumerate origin vocabulary members.
        if isinstance(node, (ast.Set, ast.Tuple, ast.List)):
            literals = {
                el.value
                for el in node.elts
                if isinstance(el, ast.Constant) and isinstance(el.value, str)
            }
            overlap = literals & INTERACTION_ORIGIN_VOCABULARY
            assert not overlap, (
                f"literal collection re-enumerates origin vocabulary members "
                f"{sorted(overlap)}; derive from the model's "
                f"INTERACTION_ORIGIN_VOCABULARY instead"
            )
        # No inline origin-normalization logic.
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            name = node.name.lower()
            assert not ("normalize" in name and "origin" in name), (
                f"{node.name} looks like inline origin normalization; the "
                "model module owns normalize_interaction_origin"
            )
    # _PROTOCOL_VERSION and _ORIGIN_VOCABULARY must be Name-aliases of the
    # model's constants, not fresh literals.
    aliases = {
        target.id: node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
        and target.id in {"_PROTOCOL_VERSION", "_ORIGIN_VOCABULARY"}
    }
    assert set(aliases) == {"_PROTOCOL_VERSION", "_ORIGIN_VOCABULARY"}
    for name, value in aliases.items():
        assert isinstance(value, ast.Name), (
            f"{name} must alias the model's constant by name, got "
            f"{ast.dump(value)[:80]}"
        )
    assert aliases["_ORIGIN_VOCABULARY"].id == "INTERACTION_ORIGIN_VOCABULARY"
    assert aliases["_PROTOCOL_VERSION"].id == "INTERACTION_PROTOCOL_VERSION"
