"""Regression test guarding against the class of bug fixed in PR #2449.

20260826_seed_deputy_mcp_app.py's downgrade() used to compare "description"
against _deputy_app_row()["description"] (the *current* text) as one of
several columns proving a public_mcp_apps row is still "this migration's"
seeded row and safe to delete. 20260916_update_deputy_description.py's
downgrade() reverts that column to the *original* pre-backfill text before
this migration's downgrade() ever runs (later migrations unwind first), so
the equality check never matched and the row -- along with the
oauth_providers row underneath it -- was silently orphaned instead of
removed on a full downgrade. The fix accepts either the original or current
text as "uncustomized" (see _ORIGINAL_DEPUTY_DESCRIPTION).

employment-hero/salesforce/myob's seed migrations have the identical naive
guard today but no paired description-update migration yet, so the bug is
latent there, not live. This test statically scans every migration in the
versions directory (not just ones matching a *_seed_*_mcp_app.py naming
convention -- see _candidate_migrations) and fails the moment a migration
using the vulnerable guard shape is paired with a description-update
migration for the same app_id, so the fix (or an equivalent one) has to land
alongside it, instead of being rediscovered by a future PR review.

Known scope boundary: this is a source-pattern scanner, not a full dataflow
analyzer. It resolves compare_columns/APP_ID from literals written inline at
the call site; it does not trace a value assigned to an intermediate
variable elsewhere in the file, and it only recognizes `_row_matches_seeded_
shape` called by its bare name (not via an attribute access or an aliased
import) -- acceptable here because every migration in this repo is
deliberately self-contained (no cross-migration imports, per e.g.
_ORIGINAL_DEPUTY_DESCRIPTION's own comment), so there is nothing to alias or
import in the first place.
"""

import ast
from pathlib import Path

VERSIONS_DIR = Path(__file__).parent.parent.parent / "src/xagent/migrations/versions"


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _module_app_id(tree: ast.Module) -> str | None:
    # Only the module's top-level statements, not ast.walk(tree) -- APP_ID is
    # a module-level constant by convention, and walking the whole tree would
    # false-positive on a same-named local inside some other function.
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if (
                any(isinstance(t, ast.Name) and t.id == "APP_ID" for t in node.targets)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            ):
                return node.value.value
        elif isinstance(node, ast.AnnAssign):
            # Covers a future `APP_ID: str = "..."` style, matching how
            # `revision`/`down_revision` are already annotated in these
            # migration files.
            if (
                isinstance(node.target, ast.Name)
                and node.target.id == "APP_ID"
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            ):
                return node.value.value
    return None


def _downgrade_function(tree: ast.Module) -> ast.FunctionDef | None:
    # Top-level only, for the same reason as _module_app_id: downgrade() is
    # always a module-level function in these migrations.
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "downgrade":
            return node
    return None


def _literal_container_elements(node: ast.AST | None) -> list[ast.expr] | None:
    """Return the elements of a Set/List/Tuple literal, or of a
    frozenset(...)/set(...) call wrapping one of those -- the shapes
    actually used for a fixed collection of column names or acceptable
    values in this codebase -- or None if `node` isn't one of those."""
    if isinstance(node, (ast.Set, ast.List, ast.Tuple)):
        return list(node.elts)
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in ("frozenset", "set")
        and len(node.args) == 1
    ):
        return _literal_container_elements(node.args[0])
    return None


def _calls_row_matches_seeded_shape(downgrade: ast.FunctionDef) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(downgrade)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_row_matches_seeded_shape"
        )
    ]


def _compare_columns_arg(call: ast.Call) -> ast.AST | None:
    # compare_columns is the function's 3rd parameter -- target it
    # specifically (3rd positional arg, or the keyword) rather than treating
    # every positional arg as a candidate, so a coincidental literal
    # elsewhere in the call can't false-positive, and a call that passes it
    # by keyword isn't silently missed (false negative).
    if len(call.args) >= 3:
        return call.args[2]
    for keyword in call.keywords:
        if keyword.arg == "compare_columns":
            return keyword.value
    return None


def _has_naive_description_guard(downgrade: ast.FunctionDef) -> bool:
    """True if downgrade() calls `_row_matches_seeded_shape(...)` with a
    compare_columns literal that includes "description" -- the shape of the
    guard that silently no-ops once a sibling migration has reverted that
    column to a different (but still "uncustomized") value."""
    for call in _calls_row_matches_seeded_shape(downgrade):
        elements = _literal_container_elements(_compare_columns_arg(call))
        if elements is None:
            continue
        if any(
            isinstance(element, ast.Constant) and element.value == "description"
            for element in elements
        ):
            return True
    return False


def _is_description_subscript(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Subscript)
        and isinstance(node.slice, ast.Constant)
        and node.slice.value == "description"
    )


def _has_description_membership_check(downgrade: ast.FunctionDef) -> bool:
    """True if downgrade() compares a `something["description"]` value for
    membership in a container of values (`row["description"] in (a, b)`) --
    the shape of 20260826_seed_deputy_mcp_app.py's separate, non-naive
    description check. This is what has to replace "description" in
    compare_columns for removing it to be a real fix, rather than just
    trading the orphaning bug for silently discarding a description-only
    customization instead (an admin who PATCHed only that field would no
    longer be protected by anything)."""
    for node in ast.walk(downgrade):
        if isinstance(node, ast.Compare) and any(
            isinstance(op, (ast.In, ast.NotIn)) for op in node.ops
        ):
            operands = [node.left, *node.comparators]
            if any(_is_description_subscript(operand) for operand in operands):
                return True
    return False


def _is_vulnerable(seed_path: Path, description_update_app_ids: set[str]) -> bool:
    tree = _parse(seed_path)
    app_id = _module_app_id(tree)
    downgrade = _downgrade_function(tree)
    if app_id is None or downgrade is None:
        return False
    if app_id not in description_update_app_ids:
        return False
    if not _calls_row_matches_seeded_shape(downgrade):
        # Never used the structural-match guard to begin with (e.g. an
        # unconditional delete, like 20260817_seed_github_mcp_app.py) --
        # nothing to regress: it never offered description-customization
        # protection in the first place.
        return False
    if _has_naive_description_guard(downgrade):
        return True
    # "description" isn't in compare_columns, but downgrade() does gate
    # deletion on other columns matching -- if nothing replaced description
    # there, an admin who customized only that field loses the protection
    # compare_columns used to give it, and downgrade silently deletes their
    # row instead of preserving it. Deliberately not gated on "does the
    # module define an _ORIGINAL_... constant" instead: that alone doesn't
    # prove anything actually consumes it -- only that a same-named
    # constant exists somewhere in the file.
    return not _has_description_membership_check(downgrade)


def _all_migration_files() -> list[Path]:
    return sorted(VERSIONS_DIR.glob("*.py"))


def _candidate_migrations() -> list[Path]:
    """Every migration whose downgrade() calls _row_matches_seeded_shape at
    all -- the function whose naive "description" comparison caused the
    orphaning bug. Found by inspecting every migration file directly rather
    than matching a *_seed_*_mcp_app.py filename convention: that glob
    would silently miss a differently-named or multi-app seed migration
    (e.g. 20260526_seed_builtin_microsoft_graph_mcp_apps.py's plural
    filename) that adopts the same pattern later."""
    candidates = []
    for path in _all_migration_files():
        downgrade = _downgrade_function(_parse(path))
        if downgrade is not None and _calls_row_matches_seeded_shape(downgrade):
            candidates.append(path)
    return candidates


def _description_update_app_ids() -> set[str]:
    app_ids = set()
    for path in VERSIONS_DIR.glob("*_update_*_description.py"):
        app_id = _module_app_id(_parse(path))
        if app_id:
            app_ids.add(app_id)
    return app_ids


def test_seed_migrations_with_paired_description_migration_guard_description_safely():
    description_update_app_ids = _description_update_app_ids()
    # Sanity check the scan actually found the known description-backfill
    # migrations, so a future rename of the naming convention would fail
    # loudly here instead of this test silently checking nothing.
    assert "deputy" in description_update_app_ids
    assert "github" in description_update_app_ids

    candidates = _candidate_migrations()
    assert candidates, "expected at least the deputy migration to be a candidate"

    # A candidate whose APP_ID this scanner can't resolve (e.g. a multi-app
    # migration keyed by a list of ids instead of a single top-level
    # APP_ID) must fail loudly for manual review, not be silently treated
    # as "unanalyzable == safe" by _is_vulnerable's own early return.
    unresolved = [
        path.name for path in candidates if _module_app_id(_parse(path)) is None
    ]
    assert not unresolved, (
        "These migrations call _row_matches_seeded_shape but this scanner "
        "could not resolve a single top-level APP_ID for them -- likely a "
        "multi-app migration. Extend _module_app_id or special-case them "
        f"explicitly instead of leaving them unanalyzed: {unresolved}"
    )

    vulnerable = [
        path.name
        for path in candidates
        if _is_vulnerable(path, description_update_app_ids)
    ]
    assert not vulnerable, (
        "These migrations use the _row_matches_seeded_shape guard in a way "
        "that either compares 'description' for exact equality against the "
        "current text, or drops it from compare_columns with nothing "
        "replacing the protection it gave -- and are paired with a sibling "
        "*_update_*_description.py migration that reverts that column on "
        "its own downgrade. The same combination silently orphaned "
        "public_mcp_apps/oauth_providers rows for deputy (PR #2449). Apply "
        "the accept-original-or-current-text fix from "
        "20260826_seed_deputy_mcp_app.py's _ORIGINAL_DEPUTY_DESCRIPTION "
        f"pattern to: {vulnerable}"
    )


def test_is_vulnerable_flags_the_historical_deputy_bug_shape(tmp_path):
    """Self-test for the scanner above: reproduce the pre-fix shape (a seed
    migration whose downgrade() still compares "description" for exact
    equality) in isolation and confirm the detector actually flags it once a
    matching description-update migration exists -- otherwise the assertion
    above could be passing for the wrong reason (e.g. a typo in the AST
    walk)."""
    seed_path = tmp_path / "20260101_seed_widget_mcp_app.py"
    seed_path.write_text(
        """
APP_ID = "widget"


def _widget_app_row():
    return {"description": "current text"}


def downgrade():
    app_row = None
    if app_row is not None and _row_matches_seeded_shape(
        app_row,
        _widget_app_row(),
        {"name", "description", "icon"},
    ):
        pass
"""
    )

    assert _is_vulnerable(seed_path, {"widget"})
    # Without a sibling description-update migration for this app_id, the
    # same naive guard is not (yet) a live bug.
    assert not _is_vulnerable(seed_path, set())


def test_is_vulnerable_accepts_the_deputy_fix_shape(tmp_path):
    """Companion self-test: a seed migration shaped like the real applied
    fix -- "description" removed from compare_columns *and* replaced by a
    separate membership check against _ORIGINAL_WIDGET_DESCRIPTION -- must
    not be flagged, even when paired with a sibling description-update
    migration."""
    seed_path = tmp_path / "20260101_seed_widget_mcp_app.py"
    seed_path.write_text(
        """
APP_ID = "widget"

_ORIGINAL_WIDGET_DESCRIPTION = "old text"


def _widget_app_row():
    return {"description": "current text"}


def downgrade():
    app_row = None
    description_is_uncustomized = app_row is not None and app_row._mapping[
        "description"
    ] in (
        _widget_app_row()["description"],
        _ORIGINAL_WIDGET_DESCRIPTION,
    )
    if (
        app_row is not None
        and description_is_uncustomized
        and _row_matches_seeded_shape(
            app_row,
            _widget_app_row(),
            {"name", "icon"},
        )
    ):
        pass
"""
    )

    assert not _is_vulnerable(seed_path, {"widget"})


def test_is_vulnerable_flags_an_original_constant_that_was_never_wired_up(tmp_path):
    """Regression test for a false negative caught in review: defining an
    "_ORIGINAL_..." constant alone proves nothing if "description" is still
    left in the compare_columns set -- the equality check it feeds is just
    as naive as if the constant didn't exist. _is_vulnerable must key off
    the actual guard shape, not off whether some "_ORIGINAL_"-prefixed name
    merely exists somewhere in the file."""
    seed_path = tmp_path / "20260101_seed_widget_mcp_app.py"
    seed_path.write_text(
        """
APP_ID = "widget"

_ORIGINAL_WIDGET_DESCRIPTION = "old text"


def _widget_app_row():
    return {"description": "current text"}


def downgrade():
    app_row = None
    if app_row is not None and _row_matches_seeded_shape(
        app_row,
        _widget_app_row(),
        {"name", "description", "icon"},
    ):
        pass
"""
    )

    assert _is_vulnerable(seed_path, {"widget"})


def test_is_vulnerable_flags_description_dropped_without_replacement(tmp_path):
    """Regression test for a gap caught in review: removing "description"
    from compare_columns is only half the fix. Without a replacement check
    protecting description-only customizations (the membership-check shape
    exercised above), the migration trades the orphaning bug for a
    different one -- silently deleting a row an admin customized only by
    its description, instead of preserving it."""
    seed_path = tmp_path / "20260101_seed_widget_mcp_app.py"
    seed_path.write_text(
        """
APP_ID = "widget"

_ORIGINAL_WIDGET_DESCRIPTION = "old text"


def _widget_app_row():
    return {"description": "current text"}


def downgrade():
    app_row = None
    if app_row is not None and _row_matches_seeded_shape(
        app_row,
        _widget_app_row(),
        {"name", "icon"},
    ):
        pass
"""
    )

    assert _is_vulnerable(seed_path, {"widget"})


def test_is_vulnerable_ignores_migrations_without_the_structural_guard(tmp_path):
    """A migration whose downgrade() never calls _row_matches_seeded_shape
    (an unconditional delete, like 20260817_seed_github_mcp_app.py) never
    offered description-customization protection to begin with, so pairing
    it with a description-update migration isn't a regression -- there is
    no protection for description to have been dropped from."""
    seed_path = tmp_path / "20260101_seed_widget_mcp_app.py"
    seed_path.write_text(
        """
APP_ID = "widget"


def downgrade():
    pass
"""
    )

    assert not _is_vulnerable(seed_path, {"widget"})


def test_is_vulnerable_recognizes_list_and_frozenset_compare_columns(tmp_path):
    """compare_columns is always written as a `{...}` set literal in this
    codebase today, but the guard logic doesn't depend on that -- confirm a
    list literal or a frozenset(...)/set(...) call wrapping one is
    recognized too, so a stylistic variation doesn't silently evade
    detection."""
    seed_path = tmp_path / "20260101_seed_widget_mcp_app.py"
    seed_path.write_text(
        """
APP_ID = "widget"


def _widget_app_row():
    return {"description": "current text"}


def downgrade():
    app_row = None
    if app_row is not None and _row_matches_seeded_shape(
        app_row,
        _widget_app_row(),
        frozenset(["name", "description", "icon"]),
    ):
        pass
"""
    )

    assert _is_vulnerable(seed_path, {"widget"})


def test_real_deputy_seed_migration_is_not_vulnerable():
    """Positive real-world check that the applied fix satisfies the
    scanner, using the actual file rather than a fabricated stand-in."""
    path = VERSIONS_DIR / "20260826_seed_deputy_mcp_app.py"
    assert path.exists()
    assert not _is_vulnerable(path, {"deputy"})


def test_real_github_seed_migration_is_not_vulnerable_despite_unconditional_delete():
    """github's downgrade() unconditionally deletes the app row (no
    _row_matches_seeded_shape call at all) and is already paired with
    20260914_update_github_description.py -- confirms the "never used the
    structural guard" exemption in _is_vulnerable doesn't accidentally
    apply to (or miss) a real file it wasn't modeled on."""
    path = VERSIONS_DIR / "20260817_seed_github_mcp_app.py"
    assert path.exists()
    assert not _is_vulnerable(path, {"github"})


def test_real_seed_migrations_are_not_vulnerable_today():
    """employment-hero/salesforce/myob have the naive guard today but no
    sibling description migration, so pairing them with a fabricated one
    here (not their real app_id membership) proves the scanner *would*
    catch it without asserting anything about the migrations' current
    real-world shape. Asserting "these real files are vulnerable today"
    instead would make this test fail the moment someone properly fixes
    them -- the opposite of what a regression guard should do; that
    coverage already lives in the synthetic-fixture tests above."""
    for filename in [
        "20260826_seed_employment_hero_mcp_app.py",
        "20260818_seed_salesforce_mcp_app.py",
        "20260903_seed_myob_mcp_app.py",
    ]:
        path = VERSIONS_DIR / filename
        assert path.exists()
        assert not _is_vulnerable(path, set())


def test_multi_app_migrations_are_included_in_the_scan_but_not_flagged():
    """Regression test for the coverage gap caught in review: a
    `*_seed_*_mcp_app.py` glob would silently miss a differently-named or
    multi-app seed migration -- like this plural-filename migration, or the
    very first table-creation migration -- if either later adopted the
    structural guard. Confirm both are visible to _all_migration_files
    today (so a future change to them would actually be scanned) and are
    correctly excluded from _candidate_migrations for the right reason:
    neither calls _row_matches_seeded_shape yet, not because the scanner
    can't see them."""
    all_names = {path.name for path in _all_migration_files()}
    assert "20260526_seed_builtin_microsoft_graph_mcp_apps.py" in all_names
    assert "f1427c3a7261_add_oauthprovider_and_publicmcpapp_.py" in all_names

    candidate_names = {path.name for path in _candidate_migrations()}
    assert "20260526_seed_builtin_microsoft_graph_mcp_apps.py" not in candidate_names
    assert "f1427c3a7261_add_oauthprovider_and_publicmcpapp_.py" not in candidate_names
