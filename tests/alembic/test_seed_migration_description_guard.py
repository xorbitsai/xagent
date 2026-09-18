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
latent there, not live. This test statically scans every seed migration and
every description-update migration in the versions directory and fails the
moment someone adds a description-update migration for an app whose seed
migration still has the naive guard -- so the fix (or an equivalent one) has
to land alongside it, instead of being rediscovered by a future PR review.
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


def _has_naive_description_guard(downgrade: ast.FunctionDef) -> bool:
    """True if downgrade() calls `_row_matches_seeded_shape(...)` with a
    compare_columns set literal that includes "description" -- the shape of
    the guard that silently no-ops once a sibling migration has reverted
    that column to a different (but still "uncustomized") value."""
    for node in ast.walk(downgrade):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_row_matches_seeded_shape"
        ):
            # compare_columns is the function's 3rd parameter -- target it
            # specifically (3rd positional arg, or the keyword) rather than
            # scanning every positional arg, so a coincidental set literal
            # elsewhere in the call can't false-positive, and a call that
            # passes it by keyword isn't silently missed (false negative).
            compare_columns_node = None
            if len(node.args) >= 3:
                compare_columns_node = node.args[2]
            else:
                for keyword in node.keywords:
                    if keyword.arg == "compare_columns":
                        compare_columns_node = keyword.value
                        break
            if isinstance(compare_columns_node, ast.Set):
                for element in compare_columns_node.elts:
                    if (
                        isinstance(element, ast.Constant)
                        and element.value == "description"
                    ):
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
    # Deliberately not also gated on "does the module define an _ORIGINAL_
    # constant": that alone doesn't prove the naive equality check was
    # actually replaced, only that a same-named constant exists somewhere in
    # the file. A migration is only actually protected once "description" is
    # gone from the compare_columns set _has_naive_description_guard checks
    # below -- exactly what the real fix (20260826_seed_deputy_mcp_app.py)
    # does.
    return _has_naive_description_guard(downgrade)


def _seed_migrations() -> list[Path]:
    return sorted(VERSIONS_DIR.glob("*_seed_*_mcp_app.py"))


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

    vulnerable = [
        path.name
        for path in _seed_migrations()
        if _is_vulnerable(path, description_update_app_ids)
    ]

    assert not vulnerable, (
        "These seed migrations compare 'description' for exact equality "
        "against the current text in downgrade()'s shape guard, but also "
        "have a sibling *_update_*_description.py migration that reverts "
        "the column on its own downgrade -- the same combination that "
        "silently orphaned public_mcp_apps/oauth_providers rows for "
        "deputy (PR #2449). Apply the accept-original-or-current-text fix "
        "from 20260826_seed_deputy_mcp_app.py's _ORIGINAL_DEPUTY_DESCRIPTION "
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
    """Companion self-test: a seed migration that removes "description" from
    the compare_columns set entirely (the applied fix's actual shape --
    _ORIGINAL_DEPUTY_DESCRIPTION is checked separately, outside
    _row_matches_seeded_shape) must not be flagged, even when paired with a
    sibling description-update migration."""
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

    assert not _is_vulnerable(seed_path, {"widget"})


def test_is_vulnerable_flags_an_original_constant_that_was_never_wired_up(tmp_path):
    """Regression test for a false negative caught in review: defining an
    "_ORIGINAL_..." constant alone proves nothing if "description" is still
    left in the compare_columns set -- the equality check it feeds is just
    as naive as if the constant didn't exist. _is_vulnerable must key
    entirely off compare_columns, not off whether some "_ORIGINAL_"-prefixed
    name merely exists somewhere in the file."""
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


def test_real_deputy_seed_migration_is_not_vulnerable():
    """Positive real-world check that the applied fix satisfies the
    scanner, using the actual file rather than a fabricated stand-in."""
    path = VERSIONS_DIR / "20260826_seed_deputy_mcp_app.py"
    assert path.exists()
    assert not _is_vulnerable(path, {"deputy"})


def test_real_seed_migrations_are_only_latently_vulnerable_today():
    """employment-hero/salesforce/myob have the naive guard today but no
    sibling description migration -- purely preventative. Confirm the
    scanner would actually catch it, using the real file content, the
    moment one is added -- this is what makes the main test above a
    tripwire instead of a no-op."""
    for app_id, filename in [
        ("employment-hero", "20260826_seed_employment_hero_mcp_app.py"),
        ("salesforce", "20260818_seed_salesforce_mcp_app.py"),
        ("myob", "20260903_seed_myob_mcp_app.py"),
    ]:
        path = VERSIONS_DIR / filename
        assert path.exists()
        assert not _is_vulnerable(path, set())
        assert _is_vulnerable(path, {app_id})
