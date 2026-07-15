"""Relative-path containment tests for the workspace path resolvers.

Absolute-path resolution already re-checks that the target stays inside the
workspace. Relative paths were resolved against a base directory and returned
without a second check, so a ``../``-laden relative path could escape the
workspace once ``Path.resolve()`` collapsed the ``..`` segments. These tests
pin the containment re-check on every relative branch, across both
independent implementations (``TaskWorkspace`` and ``WorkspaceFileOperations``,
which do not delegate to one another).
"""

import pytest

from xagent.core.tools.core.workspace_file_tool import WorkspaceFileOperations
from xagent.core.workspace import TaskWorkspace

# Relative inputs that resolve outside the workspace and must be rejected.
# ``plain`` climbs out directly; ``prefixed`` starts under a legit "output/"
# prefix before climbing out — both share the same expected outcome.
TRAVERSAL_PATHS = [
    pytest.param("../../other/secret.txt", id="plain"),
    pytest.param("output/../../../escape.txt", id="prefixed"),
]


@pytest.fixture
def workspace(tmp_path):
    return TaskWorkspace("task7", str(tmp_path))


@pytest.fixture
def ops(workspace):
    return WorkspaceFileOperations(workspace)


# --------------------------------------------------------------------------
# SITE 1 — TaskWorkspace.resolve_path / resolve_path_with_search
# --------------------------------------------------------------------------


@pytest.mark.parametrize("rel_path", TRAVERSAL_PATHS)
def test_resolve_path_rejects_relative_traversal(workspace, rel_path):
    with pytest.raises(ValueError):
        workspace.resolve_path(rel_path)


@pytest.mark.parametrize("default_dir", ["input", "output", "temp", "other"])
def test_resolve_path_rejects_traversal_for_every_default_dir(workspace, default_dir):
    # The input/temp branches and the ``else`` fallback (resolving against
    # workspace_dir) share the output branch's logic; pin that each still
    # rejects an out-of-workspace traversal, not just default_dir="output".
    with pytest.raises(ValueError):
        workspace.resolve_path("../../other/secret.txt", default_dir=default_dir)


def test_resolve_path_allows_legitimate_relative_path(workspace):
    resolved = workspace.resolve_path("report.txt")

    assert resolved.is_relative_to(workspace.workspace_dir.resolve())
    assert resolved == (workspace.output_dir / "report.txt").resolve()


def test_resolve_path_with_search_rejects_existing_sibling_via_traversal(
    workspace, tmp_path
):
    workspace.input_dir.mkdir(parents=True, exist_ok=True)

    # A real file outside the workspace that a traversal would otherwise reach:
    # input_dir/../../other/secret.txt == tmp_path/other/secret.txt.
    sibling = tmp_path / "other" / "secret.txt"
    sibling.parent.mkdir(parents=True, exist_ok=True)
    sibling.write_text("sibling end user's secret", encoding="utf-8")

    with pytest.raises(ValueError):
        workspace.resolve_path_with_search("../../other/secret.txt")


def test_resolve_path_with_search_rejects_nonexistent_traversal_without_oracle(
    workspace,
):
    # Containment is checked before existence, so a traversal to a path that
    # does NOT exist raises ValueError (the same as an existing target) rather
    # than FileNotFoundError. Otherwise the exception type would leak whether a
    # file exists outside the workspace.
    workspace.input_dir.mkdir(parents=True, exist_ok=True)

    with pytest.raises(ValueError):
        workspace.resolve_path_with_search("../../other/does_not_exist.txt")


def test_resolve_path_with_search_finds_legitimate_file(workspace):
    target = workspace.output_dir / "data.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("a,b\n1,2\n", encoding="utf-8")

    resolved = workspace.resolve_path_with_search("data.csv")

    assert resolved == target.resolve()


def test_resolve_path_with_search_preserves_legacy_file_path_ref(workspace):
    target = workspace.output_dir / "data.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("a,b\n1,2\n", encoding="utf-8")

    resolved = workspace.resolve_path_with_search("file:output/data.csv")

    assert resolved == target.resolve()


@pytest.mark.parametrize(
    "file_ref",
    [
        "file:file-id",
        "file://file-id",
        "file://file%20id",
    ],
)
def test_resolve_path_with_search_resolves_internal_file_refs(
    workspace, monkeypatch, file_ref
):
    target = workspace.input_dir / "photo.jpg"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"photo")
    expected_id = "file id" if "%20" in file_ref else "file-id"
    monkeypatch.setattr(
        workspace,
        "resolve_file_id",
        lambda file_id: target if file_id == expected_id else None,
    )

    assert workspace.resolve_path_with_search(file_ref) == target


def test_resolve_path_with_search_does_not_treat_local_file_uri_as_file_id(
    workspace, monkeypatch
):
    resolved_ids: list[str] = []
    monkeypatch.setattr(
        workspace,
        "resolve_file_id",
        lambda file_id: resolved_ids.append(file_id) or None,
    )

    with pytest.raises(FileNotFoundError):
        workspace.resolve_path_with_search("file:///tmp/photo.jpg")

    assert resolved_ids == []


@pytest.mark.parametrize("file_ref", ["file:file-id", "file://file-id"])
def test_resolve_file_id_accepts_internal_file_refs(workspace, file_ref):
    target = workspace.input_dir / "photo.jpg"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"photo")
    workspace._file_id_to_path["file-id"] = target

    assert workspace.resolve_file_id(file_ref) == target


def test_resolve_path_with_search_rejects_symlink_escape_in_fuzzy_match(
    workspace, tmp_path
):
    # A symlink inside a search dir that resolves outside the workspace must
    # not be returned by the fuzzy-match branch, even when its stem fuzzy-
    # matches the request. Under the pre-fix code the out-of-tree target was
    # returned unchecked; now containment rejects it and, with no other match,
    # the search reports the file as not found.
    workspace.output_dir.mkdir(parents=True, exist_ok=True)
    secret = tmp_path / "other" / "secret.txt"
    secret.parent.mkdir(parents=True, exist_ok=True)
    secret.write_text("out-of-tree secret", encoding="utf-8")

    link = workspace.output_dir / "report.txt"
    link.symlink_to(secret)

    with pytest.raises(FileNotFoundError):
        workspace.resolve_path_with_search("reportt.txt")


def test_resolve_path_with_search_skips_symlink_escape_but_finds_legit_match(
    workspace, tmp_path
):
    # The containment gate skips a rogue candidate and keeps searching, so a
    # legitimate fuzzy match in a later directory is still returned.
    workspace.output_dir.mkdir(parents=True, exist_ok=True)
    workspace.temp_dir.mkdir(parents=True, exist_ok=True)

    secret = tmp_path / "other" / "secret.txt"
    secret.parent.mkdir(parents=True, exist_ok=True)
    secret.write_text("out-of-tree secret", encoding="utf-8")
    # Rogue escaping symlink in output/ (searched before temp/).
    (workspace.output_dir / "report.txt").symlink_to(secret)
    # Legitimate in-workspace file that also fuzzy-matches the request.
    legit = workspace.temp_dir / "report.txt"
    legit.write_text("in-workspace report", encoding="utf-8")

    resolved = workspace.resolve_path_with_search("reportt.txt")

    assert resolved == legit.resolve()
    assert resolved.is_relative_to(workspace.workspace_dir.resolve())


def test_resolve_path_with_search_skips_exact_match_symlink_escape_for_legit_later(
    workspace, tmp_path
):
    # An escaping symlink named exactly like the request in an EARLIER search
    # dir (input/) must not abort the search: a legitimate same-named file in a
    # LATER dir (output/) stays reachable via exact match. Only a lexical
    # ``..`` traversal hard-aborts; a symlinked leaf is skipped.
    workspace.input_dir.mkdir(parents=True, exist_ok=True)
    workspace.output_dir.mkdir(parents=True, exist_ok=True)

    secret = tmp_path / "other" / "notes.txt"
    secret.parent.mkdir(parents=True, exist_ok=True)
    secret.write_text("out-of-tree secret", encoding="utf-8")
    (workspace.input_dir / "notes.txt").symlink_to(secret)

    legit = workspace.output_dir / "notes.txt"
    legit.write_text("in-workspace notes", encoding="utf-8")

    resolved = workspace.resolve_path_with_search("notes.txt")

    assert resolved == legit.resolve()
    assert resolved.is_relative_to(workspace.workspace_dir.resolve())


def test_resolve_path_still_rejects_absolute_escape(workspace, tmp_path):
    # Regression: the absolute-path branch keeps rejecting out-of-workspace paths.
    outside = tmp_path / "outside.txt"
    outside.write_text("x", encoding="utf-8")

    with pytest.raises(ValueError):
        workspace.resolve_path(str(outside))


def test_resolve_path_accepts_traversal_into_allowed_external_dir(tmp_path):
    # The other half of #824's expected behavior: a relative path that climbs
    # out of the workspace but lands inside a legitimately-allowed external dir
    # is ACCEPTED, not rejected.
    external = tmp_path / "external"
    external.mkdir(parents=True, exist_ok=True)
    target = external / "shared.txt"
    target.write_text("shared", encoding="utf-8")

    workspace = TaskWorkspace(
        "task7", str(tmp_path), allowed_external_dirs=[str(external)]
    )
    # output_dir/../../external/shared.txt == tmp_path/external/shared.txt.
    resolved = workspace.resolve_path("../../external/shared.txt", default_dir="output")

    assert resolved == target.resolve()


# --------------------------------------------------------------------------
# SITE 2 — WorkspaceFileOperations._resolve_path (separate implementation;
# it ignores allowed_external_dirs and confines strictly to workspace_dir)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("rel_path", TRAVERSAL_PATHS)
def test_ops_resolve_path_rejects_relative_traversal(ops, rel_path):
    with pytest.raises(ValueError):
        ops._resolve_path(rel_path, "output")


def test_ops_resolve_path_allows_legitimate_relative(workspace, ops):
    resolved = ops._resolve_path("report.txt", "output")

    assert resolved.is_relative_to(workspace.workspace_dir.resolve())
    assert resolved == (workspace.output_dir / "report.txt").resolve()


def test_write_file_refuses_relative_traversal_end_to_end(ops, tmp_path):
    with pytest.raises(ValueError):
        ops.write_file("../../other/pwned.txt", "malicious content")

    # Nothing was written outside the workspace.
    assert not (tmp_path / "other" / "pwned.txt").exists()


def test_ops_resolve_path_ignores_allowed_external_dirs(tmp_path):
    # WorkspaceFileOperations confines strictly to workspace_dir and, unlike
    # TaskWorkspace, does NOT honor allowed_external_dirs. Prove the documented
    # difference: the same path is accepted by TaskWorkspace but rejected here.
    external = tmp_path / "external"
    external.mkdir(parents=True, exist_ok=True)
    target = external / "shared.txt"
    target.write_text("shared", encoding="utf-8")

    workspace = TaskWorkspace(
        "task7", str(tmp_path), allowed_external_dirs=[str(external)]
    )
    assert workspace.resolve_path(str(target)) == target.resolve()

    ops = WorkspaceFileOperations(workspace)
    with pytest.raises(ValueError):
        ops._resolve_path(str(target), "output")
