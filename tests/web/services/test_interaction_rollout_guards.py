"""Static AST guards enforcing the hard invariant: the interaction rollout
mode can decide whether a NEW native row gets produced, and nothing else.
It can never affect how an already-published row is read, answered, or
closed. Covers the import/reference guards on read-side consumers, the
counted-call-site guards on the module's own public symbols, and the
guard machinery's own self-check.

Everything here parses real source files with the standard library ``ast``
module under whatever interpreter runs pytest. This suite must run under
Python >= 3.11: the codebase uses ``match`` statements elsewhere, which an
older stdlib ``ast`` cannot parse, and a parse failure in this suite must
fail loudly (see test_tw4 below), never silently skip a file.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

import xagent.web.api.a2a as a2a_module
import xagent.web.api.chat as chat_module
import xagent.web.api.websocket as websocket_module
import xagent.web.models.task_interaction as task_interaction_module
import xagent.web.services.chat_history_service as chat_history_service_module

pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 11),
    reason="AST guards require Python >= 3.11 to parse this codebase's syntax",
)

# ---------------------------------------------------------------------------
# Import guard: the module half (three modules, zero import expected)
# ---------------------------------------------------------------------------

_IMPORT_GUARD_MODULES = {
    "chat.py": chat_module,
    "chat_history_service.py": chat_history_service_module,
    "a2a.py": a2a_module,
}

# websocket.py is explicitly, by name, exempted from the module-half check:
# it hosts both a read-side consumer (_load_historical_stream_snapshot_sync)
# and two finalizers that will eventually need to import interaction_rollout
# once the gate is wired into task finalization. A guard that would be
# deleted the day that wiring lands is not a guard worth having at the
# module level -- see the function-half check below, which covers this
# module with an assertion that survives that wiring.
_WEBSOCKET_IMPORT_EXEMPTION = frozenset({"websocket.py"})


def _module_source_path(module) -> Path:
    return Path(module.__file__)


def _imports_interaction_rollout(tree: ast.Module) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any("interaction_rollout" in alias.name for alias in node.names):
                return True
        elif isinstance(node, ast.ImportFrom):
            if node.module and "interaction_rollout" in node.module:
                return True
            if any("interaction_rollout" in alias.name for alias in node.names):
                return True
    return False


def _parse(path: Path) -> ast.Module:
    source = path.read_text()
    try:
        return ast.parse(source, filename=str(path))
    except SyntaxError as exc:  # pragma: no cover - guard self-check below
        raise AssertionError(f"failed to parse {path}: {exc}") from exc


@pytest.mark.parametrize("label", sorted(_IMPORT_GUARD_MODULES))
def test_tw1a_import_guard_modules_do_not_import_interaction_rollout(label):
    module = _IMPORT_GUARD_MODULES[label]
    tree = _parse(_module_source_path(module))
    assert not _imports_interaction_rollout(tree), (
        f"{label} must not import interaction_rollout: read-side consumers "
        "and close-side statements must stay unaware of the rollout mode "
        "(see the module docstring's hard invariant)"
    )


def test_tw1c_websocket_module_half_exemption_is_named_and_exactly_one_module():
    assert _WEBSOCKET_IMPORT_EXEMPTION == {"websocket.py"}


# ---------------------------------------------------------------------------
# Reference guard: the function half (websocket.py's replacement for
# module-level zero-import: 5 closure seeds, own-scope-only banned-name scan)
# ---------------------------------------------------------------------------

_BANNED_NAMES = frozenset(
    {
        "interaction_rollout",
        "evaluate_native_publication",
        "get_interaction_rollout_policy",
        # Included even though the production call-count guard below would
        # also catch a read-side consumer calling this: a fail-closed guard
        # should never depend on a second guard to notice the same
        # violation.
        "validate_interaction_rollout_at_startup",
        "InteractionRolloutPolicy",
        # The module-private accessor and its backing singleton: a
        # read-side consumer reaching for either one bypasses the public
        # accessor (and its own fail-fast check) entirely, so both must be
        # banned directly rather than relying on the public name alone to
        # catch it.
        "_require_policy",
        "_policy",
    }
)

# The 5 closure seeds are not 3: get_task and get_task_status's own bodies
# (the code outside their nested _get_task_sync / _get_task_status_sync
# definitions) are themselves part of the read path -- they are the
# *callers* of the nested sync functions, not reachable *from* them, so a
# scan that only checked the 3 innermost functions would miss a banned name
# referenced directly in the outer function body.
_CLOSURE_SEEDS = [
    ("chat.py", "get_task"),
    ("chat.py", "get_task._get_task_sync"),
    ("chat.py", "get_task_status"),
    ("chat.py", "get_task_status._get_task_status_sync"),
    ("websocket.py", "_load_historical_stream_snapshot_sync"),
]

_SOURCE_BY_LABEL = {
    "chat.py": chat_module,
    "websocket.py": websocket_module,
}


def _direct_scope_functions(scope) -> list:
    """Function/async-function defs belonging to ``scope``'s own scope.

    Walks through control-flow wrappers (``try``/``if``/``with``/loops --
    none of which introduce a new Python scope) but does not descend into a
    further-nested function or class body. ``_get_task_sync`` sits inside a
    ``try:`` block directly in ``get_task``'s body, so a check that only
    looked at immediate AST children (``ast.iter_child_nodes(scope)``, no
    deeper) would silently never find it -- exactly the failure mode this
    helper (and the ast.walk-based approach in general) exists to avoid.
    """
    results = []

    def _walk(node):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                results.append(child)
                # A new scope -- do not look inside it for *this* scope's
                # functions; it gets scanned separately under its own name.
            elif isinstance(child, ast.ClassDef):
                pass  # a new scope; irrelevant to this codebase's shape here
            else:
                _walk(child)

    _walk(scope)
    return results


def _find_qualified_function(tree: ast.Module, qualname: str):
    """Resolve a dotted qualname (``outer.inner``) to its FunctionDef node.

    Not a Module.body-only lookup: ``_get_task_sync`` and
    ``_get_task_status_sync`` are nested one level deep inside a ``try:``
    block, and a shallow walk would silently never find them.
    """
    parts = qualname.split(".")
    scope = tree
    found = None
    for index, name in enumerate(parts):
        candidates = _direct_scope_functions(scope)
        found = next((c for c in candidates if c.name == name), None)
        if found is None:
            raise AssertionError(
                f"function {qualname!r} not found (stuck resolving "
                f"part {index} = {name!r})"
            )
        scope = found
    return found


def _own_scope_identifiers(func_node) -> set[str]:
    """Every Name id / Attribute attr referenced in this function's own
    body -- not descending into further-nested function definitions, which
    get scanned separately under their own qualified name. Without this
    exclusion, an outer function would be flagged for merely *containing*
    a banned reference inside an inner definition, turning a real guard
    into noise on the first unrelated nested helper."""
    identifiers: set[str] = set()

    class _Visitor(ast.NodeVisitor):
        def visit_FunctionDef(self, node):  # noqa: N802 - ast visitor API
            if node is func_node:
                self.generic_visit(node)
            # else: a deeper nested def -- belongs to its own scope, skip.

        def visit_AsyncFunctionDef(self, node):  # noqa: N802
            self.visit_FunctionDef(node)

        def visit_Name(self, node):  # noqa: N802
            identifiers.add(node.id)

        def visit_Attribute(self, node):  # noqa: N802
            identifiers.add(node.attr)
            self.generic_visit(node)

    _Visitor().visit(func_node)
    return identifiers


def _scan_closure_for_banned_names() -> dict[str, set[str]]:
    """For each of the 5 closure seeds, the banned names (if any) found in
    its own scope. Real production files, always re-parsed from disk."""
    hits: dict[str, set[str]] = {}
    trees = {
        label: _parse(_module_source_path(module))
        for label, module in _SOURCE_BY_LABEL.items()
    }
    for label, qualname in _CLOSURE_SEEDS:
        func_node = _find_qualified_function(trees[label], qualname)
        found = _own_scope_identifiers(func_node) & _BANNED_NAMES
        if found:
            hits[f"{label}::{qualname}"] = found
    return hits


def test_tw1b_closure_seeds_reference_no_banned_names():
    # _get_task_sync / _get_task_status_sync sit one level inside a `try:`
    # block in get_task / get_task_status, not as direct ast.Module.body
    # children -- a scanner that only checked Module.body would silently
    # never find either one, turning this into an always-green false guard.
    # _find_qualified_function (used above to resolve the 5 closure seeds)
    # walks through control-flow wrappers for exactly this reason.
    hits = _scan_closure_for_banned_names()
    assert hits == {}, f"read-side consumer(s) reference the rollout module: {hits}"


# ---------------------------------------------------------------------------
# The guard machinery's own self-check -- a parse failure must fail loudly
# ---------------------------------------------------------------------------


def test_tw4_syntax_error_in_a_scanned_file_fails_loudly(tmp_path):
    broken = tmp_path / "broken.py"
    broken.write_text("def f(:\n    pass\n")
    with pytest.raises(AssertionError):
        _parse(broken)


# ---------------------------------------------------------------------------
# Counted-call-site guards: production symbols (src/, excluding tests/)
# ---------------------------------------------------------------------------


def _src_root() -> Path:
    # xagent.web.api.chat is .../src/xagent/web/api/chat.py
    return _module_source_path(chat_module).parents[2]


def _iter_src_files():
    root = _src_root()
    for path in sorted(root.rglob("*.py")):
        yield path


def _count_calls(symbol: str) -> list[tuple[Path, int]]:
    hits: list[tuple[Path, int]] = []
    for path in _iter_src_files():
        tree = _parse(path)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Name) and func.id == symbol:
                hits.append((path, node.lineno))
            elif isinstance(func, ast.Attribute) and func.attr == symbol:
                hits.append((path, node.lineno))
    return hits


def test_tw2a_evaluate_native_publication_has_zero_production_callers():
    hits = _count_calls("evaluate_native_publication")
    assert hits == [], f"unexpected production caller(s): {hits}"


_GET_POLICY_SITE_WHITELIST = {
    ("app.py", "readiness_check"),
    ("admin_interaction_rollout.py", "get_interaction_rollout_diagnostics"),
}


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


def test_tw2c_get_interaction_rollout_policy_has_at_most_two_whitelisted_sites():
    # The bound is <=2, not ==2, on purpose: a count of 1 (e.g. if the admin
    # diagnostics endpoint moved into its own later change) must stay valid
    # too -- only a site outside the whitelist, or more than 2 sites total,
    # may fail this guard.
    hits = _count_calls("get_interaction_rollout_policy")
    seen = set()
    for path, lineno in hits:
        tree = _parse(path)
        func_name = _enclosing_function_name(tree, lineno)
        seen.add((path.name, func_name))

    assert len(hits) <= 2, f"more than 2 call sites: {hits}"
    for site in seen:
        assert site in _GET_POLICY_SITE_WHITELIST, f"unlisted call site: {site}"


def test_tw2d_validate_interaction_rollout_at_startup_has_exactly_one_caller():
    hits = _count_calls("validate_interaction_rollout_at_startup")
    assert len(hits) == 1, f"expected exactly 1 production caller, got: {hits}"
    path, _ = hits[0]
    assert path.name == "app.py"


# ---------------------------------------------------------------------------
# Task.source construction literals and post-construction mutation
# ---------------------------------------------------------------------------


def _source_literal_call_sites() -> list[tuple[Path, int, str]]:
    """Every ``source=`` / ``task_source=`` keyword whose value is a string
    literal, scoped to ``src/xagent/web`` (the audited scope -- scanning by
    keyword name rather than by callee, since Task.source is written
    through several different call targets, not just ``Task(...)``; a
    callee-restricted scan would silently miss most of them). The
    migrations directory is excluded even though it is part of that
    audited scope: historical backfills are allowed to write retired
    literals current code no longer produces, and a scan that is not
    kwarg-name-scoped elsewhere in the tree would pick up unrelated APIs
    that happen to share the same keyword name (e.g. a "source" kwarg on
    an entirely unrelated personal-database helper)."""
    hits: list[tuple[Path, int, str]] = []
    root = _src_root() / "web"
    for path in sorted(root.rglob("*.py")):
        if "migrations" in path.parts:
            continue
        tree = _parse(path)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            for kw in node.keywords:
                if kw.arg in ("source", "task_source") and isinstance(
                    kw.value, ast.Constant
                ):
                    if isinstance(kw.value.value, str):
                        hits.append((path, node.lineno, kw.value.value))
    return hits


def test_tw3a_source_literals_are_a_subset_of_task_source_literals():
    hits = _source_literal_call_sites()
    literals = {value for _, _, value in hits}
    assert literals <= task_interaction_module.TASK_SOURCE_LITERALS, (
        f"literal(s) outside TASK_SOURCE_LITERALS: "
        f"{literals - task_interaction_module.TASK_SOURCE_LITERALS}"
    )


# Named, exact -- not a wildcard rule. Each is a bot/handler setting its
# *own* instance attribute (self.channel_id), unrelated to any Task ORM
# row; audited as the complete set at the pinned baseline (see the task
# book's fact audit). Paths are relative to src/xagent/web.
_TASK_ATTRIBUTE_MUTATION_EXEMPT_SITES = frozenset(
    {
        ("channels/feishu/bot.py", "channel_id"),
        ("channels/slack/bot.py", "channel_id"),
        ("channels/slack/trace_handler.py", "channel_id"),
        ("channels/telegram/bot.py", "channel_id"),
    }
)


def _attribute_assignments(
    attr_names: tuple[str, ...],
) -> list[tuple[Path, int, str, bool]]:
    """Every assignment to ``x.<attr> = ...`` / ``x.<attr> += ...`` in
    src/xagent/web where ``<attr>`` is one of ``attr_names``, tagged with
    whether the receiver ``x`` is ``self``. Receiver is not filtered out at
    collection time -- the self/non-self split is the test's job, so a
    drift in which sites are self-only is visible instead of silently
    never collected."""
    hits: list[tuple[Path, int, str, bool]] = []
    web_root = _src_root() / "web"
    for path in sorted(web_root.rglob("*.py")):
        tree = _parse(path)
        for node in ast.walk(tree):
            targets = []
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, ast.AugAssign):
                targets = [node.target]
            for target in targets:
                if isinstance(target, ast.Attribute) and target.attr in attr_names:
                    is_self = (
                        isinstance(target.value, ast.Name) and target.value.id == "self"
                    )
                    hits.append((path, node.lineno, target.attr, is_self))
    return hits


def test_tw3b_no_post_construction_mutation_outside_the_named_self_exemption():
    hits = _attribute_assignments(("source", "channel_id"))
    web_root = _src_root() / "web"

    non_self = [
        (path, lineno, attr) for path, lineno, attr, is_self in hits if not is_self
    ]
    assert non_self == [], f"non-self post-construction mutation(s) found: {non_self}"

    self_sites = {
        (str(path.relative_to(web_root)), attr)
        for path, _lineno, attr, is_self in hits
        if is_self
    }
    assert self_sites == _TASK_ATTRIBUTE_MUTATION_EXEMPT_SITES, (
        f"the self-attribute exemption set drifted from the audited 4 "
        f"sites: found {self_sites}"
    )
