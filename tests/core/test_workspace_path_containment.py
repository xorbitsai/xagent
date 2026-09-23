"""Relative-path containment tests for the workspace path resolvers.

Absolute-path resolution already re-checks that the target stays inside the
workspace. Relative paths were resolved against a base directory and returned
without a second check, so a ``../``-laden relative path could escape the
workspace once ``Path.resolve()`` collapsed the ``..`` segments. These tests
pin the containment re-check on every relative branch, across both
independent implementations (``TaskWorkspace`` and ``WorkspaceFileOperations``,
which do not delegate to one another).
"""

import os
from pathlib import Path

import pytest

from xagent.core.tools.core.workspace_file_tool import WorkspaceFileOperations
from xagent.core.workspace import SPILL_DIR_NAME, TaskWorkspace

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


def test_resolve_authorized_path_uses_explicit_base(workspace, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    target = workspace.output_dir / "report.txt"

    resolved = workspace.resolve_authorized_path(
        "report.txt",
        base_dir=workspace.output_dir,
    )

    assert resolved == target.resolve()


def test_resolve_authorized_path_rejects_relative_base(workspace):
    with pytest.raises(ValueError, match="base_dir must be absolute"):
        workspace.resolve_authorized_path(
            "report.txt",
            base_dir="relative/output",
        )


def test_resolve_authorized_path_can_exclude_external_roots(tmp_path):
    external = tmp_path / "external"
    external.mkdir()
    workspace = TaskWorkspace(
        "task7",
        str(tmp_path / "workspace"),
        allowed_external_dirs=[str(external)],
    )

    assert (
        workspace.resolve_authorized_path(
            external / "reference.txt",
            base_dir=workspace.output_dir,
            include_external_dirs=True,
        )
        == (external / "reference.txt").resolve()
    )
    with pytest.raises(ValueError):
        workspace.resolve_authorized_path(
            external / "reference.txt",
            base_dir=workspace.output_dir,
            include_external_dirs=False,
        )


def test_resolve_authorized_path_rejects_symlink_escape(workspace, tmp_path):
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    link = workspace.output_dir / "outside-link"
    link.symlink_to(outside)

    with pytest.raises(ValueError):
        workspace.resolve_authorized_path(
            link,
            base_dir=workspace.output_dir,
            include_external_dirs=False,
        )


@pytest.mark.parametrize("error_type", [OSError, RuntimeError])
def test_resolve_authorized_path_normalizes_resolution_failures(
    workspace, monkeypatch, error_type
):
    candidate = workspace.output_dir / "unresolvable"
    original_resolve = Path.resolve

    def fail_candidate(path, *args, **kwargs):
        if path == candidate:
            raise error_type("resolution failed")
        return original_resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", fail_candidate)

    with pytest.raises(ValueError, match="Failed to resolve path"):
        workspace.resolve_authorized_path(
            candidate,
            base_dir=workspace.output_dir,
        )


def test_resolve_authorized_path_workspace_match_ignores_broken_external_root(
    tmp_path,
):
    external = tmp_path / "external"
    external.mkdir()
    workspace = TaskWorkspace(
        "task7",
        str(tmp_path / "workspace"),
        allowed_external_dirs=[str(external)],
    )
    external.rmdir()
    external.symlink_to(external.name)
    target = workspace.output_dir / "report.txt"

    assert (
        workspace.resolve_authorized_path(
            target,
            base_dir=workspace.output_dir,
        )
        == target.resolve()
    )


def test_resolve_authorized_path_broken_external_root_stays_fail_closed(tmp_path):
    external = tmp_path / "external"
    external.mkdir()
    workspace = TaskWorkspace(
        "task8",
        str(tmp_path / "workspace"),
        allowed_external_dirs=[str(external)],
    )
    external.rmdir()
    external.symlink_to(external.name)

    with pytest.raises(ValueError, match="Failed to resolve path"):
        workspace.resolve_authorized_path(
            tmp_path / "outside" / "report.txt",
            base_dir=workspace.output_dir,
        )


def test_resolve_authorized_path_does_not_expand_user_home(workspace, monkeypatch):
    monkeypatch.setenv("HOME", str(workspace.base_dir / "other-home"))

    resolved = workspace.resolve_authorized_path(
        "~/report.txt",
        base_dir=workspace.output_dir,
    )

    assert resolved == (workspace.output_dir / "~" / "report.txt").resolve()


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


# --------------------------------------------------------------------------
# SITE 3 — the engine-owned output subtree is refused by every write path
# --------------------------------------------------------------------------

ENGINE_SPELLINGS = [
    pytest.param(f"{SPILL_DIR_NAME}/x.json", id="no-output-prefix"),
    pytest.param(f"output/{SPILL_DIR_NAME}/x.json", id="output-prefix"),
    pytest.param(f"./output/{SPILL_DIR_NAME}/x.json", id="dot-slash"),
    pytest.param(f"output/sub/../{SPILL_DIR_NAME}/x.json", id="dotdot"),
    pytest.param(f"output/{SPILL_DIR_NAME}/sub/x.json", id="nested"),
    pytest.param(f"output/{SPILL_DIR_NAME}", id="directory-itself"),
]


@pytest.fixture
def engine_file(workspace):
    spill_dir = workspace.output_dir / SPILL_DIR_NAME
    spill_dir.mkdir(parents=True)
    target = spill_dir / "x.json"
    target.write_text('["engine"]', encoding="utf-8")
    return target


@pytest.mark.parametrize("spelling", ENGINE_SPELLINGS)
def test_write_file_refuses_the_engine_subtree(ops, workspace, spelling):
    with pytest.raises(ValueError, match="engine-owned"):
        ops.write_file(spelling, "planted")
    # Nothing was created, not even the directory.
    assert not (workspace.output_dir / SPILL_DIR_NAME).exists()


@pytest.mark.parametrize("spelling", ENGINE_SPELLINGS)
def test_create_directory_refuses_the_engine_subtree(ops, workspace, spelling):
    with pytest.raises(ValueError, match="engine-owned"):
        ops.create_directory(spelling)
    assert not (workspace.output_dir / SPILL_DIR_NAME).exists()


def test_write_json_file_refuses_the_engine_subtree(ops, workspace):
    with pytest.raises(ValueError, match="engine-owned"):
        ops.write_json_file(f"output/{SPILL_DIR_NAME}/x.json", {"planted": True})
    assert not (workspace.output_dir / SPILL_DIR_NAME).exists()


def test_write_csv_file_refuses_the_engine_subtree(ops, workspace):
    with pytest.raises(ValueError, match="engine-owned"):
        ops.write_csv_file(f"output/{SPILL_DIR_NAME}/x.csv", [{"a": "1"}])
    assert not (workspace.output_dir / SPILL_DIR_NAME).exists()


def test_delete_file_refuses_an_existing_engine_file(ops, engine_file):
    with pytest.raises(ValueError, match="engine-owned"):
        ops.delete_file(f"output/{SPILL_DIR_NAME}/x.json")
    assert engine_file.exists()


def test_append_file_refuses_an_existing_engine_file(ops, engine_file):
    with pytest.raises(ValueError, match="engine-owned"):
        ops.append_file(f"{SPILL_DIR_NAME}/x.json", "planted")
    assert engine_file.read_text(encoding="utf-8") == '["engine"]'


def test_edit_file_refuses_an_existing_engine_file(ops, engine_file):
    with pytest.raises(ValueError, match="engine-owned"):
        ops.edit_file(
            f"{SPILL_DIR_NAME}/x.json",
            [{"operation_type": "replace", "line_number": 1, "content": "planted"}],
        )
    assert engine_file.read_text(encoding="utf-8") == '["engine"]'


def test_find_and_replace_refuses_an_existing_engine_file(ops, engine_file):
    with pytest.raises(ValueError, match="engine-owned"):
        ops.find_and_replace(f"{SPILL_DIR_NAME}/x.json", "engine", "planted")
    assert engine_file.read_text(encoding="utf-8") == '["engine"]'


@pytest.fixture
def html_source(workspace):
    """A real source file: prepare_html_asset resolves its source first."""
    source = workspace.input_dir / "logo.png"
    source.write_bytes(b"png")
    return source


def test_prepare_html_asset_refuses_the_engine_subtree_as_html_target(
    ops, workspace, html_source
):
    with pytest.raises(ValueError, match="engine-owned"):
        ops.prepare_html_asset("logo.png", f"{SPILL_DIR_NAME}/index.html")
    assert not (workspace.output_dir / SPILL_DIR_NAME).exists()


def test_prepare_html_asset_refuses_the_engine_subtree_as_assets_dir(
    ops, workspace, html_source
):
    with pytest.raises(ValueError, match="engine-owned"):
        ops.prepare_html_asset("logo.png", "index.html", assets_subdir=SPILL_DIR_NAME)
    assert not (workspace.output_dir / SPILL_DIR_NAME).exists()


def test_prepare_html_asset_names_the_resolved_asset_directory_it_refuses(
    ops, html_source
):
    """The asset directory is the HTML path's parent joined with
    assets_subdir, so the refusal names that directory as resolved, in its
    workspace-relative form, rather than the assets_subdir argument alone."""
    with pytest.raises(ValueError, match=f"Path 'output/{SPILL_DIR_NAME}' is"):
        ops.prepare_html_asset("logo.png", "index.html", assets_subdir=SPILL_DIR_NAME)


def test_prepare_html_asset_names_the_real_target_when_the_html_parent_is_swapped(
    ops, workspace, html_source, monkeypatch
):
    """When the HTML path's parent becomes a symlink into the engine
    directory after the HTML path was resolved, the refusal is reached
    through that side, and only the resolved form names the real target."""
    reserved = workspace.output_dir / SPILL_DIR_NAME
    reserved.mkdir()
    resolve_html_output_path = ops._resolve_html_output_path

    def resolve_then_swap_the_parent(html_path: str) -> Path:
        resolved = resolve_html_output_path(html_path)
        os.symlink(reserved, workspace.output_dir / "report")
        return resolved

    monkeypatch.setattr(ops, "_resolve_html_output_path", resolve_then_swap_the_parent)
    with pytest.raises(
        ValueError, match=f"Path 'output/{SPILL_DIR_NAME}/assets' is"
    ) as refused:
        ops.prepare_html_asset("logo.png", "report/index.html", assets_subdir="assets")
    assert "'assets' is" not in str(refused.value)
    assert not (reserved / "assets").exists()


def test_reads_are_unaffected(ops, engine_file):
    assert ops.read_file(f"{SPILL_DIR_NAME}/x.json") == '["engine"]'
    assert ops.read_file(f"output/{SPILL_DIR_NAME}/x.json") == '["engine"]'
    assert ops.file_exists(f"{SPILL_DIR_NAME}/x.json") is True


def test_a_near_miss_directory_is_still_writable(ops, workspace):
    result = ops.write_file(f"{SPILL_DIR_NAME}-mine/x.json", "mine")
    assert result["success"] is True
    assert (workspace.output_dir / f"{SPILL_DIR_NAME}-mine" / "x.json").exists()


EXISTING_FILE_WRITES = [
    pytest.param(lambda ops, p: ops.append_file(p, "planted"), id="append_file"),
    pytest.param(
        lambda ops, p: ops.edit_file(
            p, [{"operation_type": "replace", "line_number": 1, "content": "planted"}]
        ),
        id="edit_file",
    ),
    pytest.param(
        lambda ops, p: ops.find_and_replace(p, "engine", "planted"),
        id="find_and_replace",
    ),
    pytest.param(lambda ops, p: ops.delete_file(p), id="delete_file"),
]


@pytest.mark.parametrize("write", EXISTING_FILE_WRITES)
@pytest.mark.parametrize("directory_exists", [True, False], ids=["dir", "no-dir"])
def test_a_missing_engine_file_is_refused_before_it_is_reported_missing(
    ops, workspace, write, directory_exists
):
    """The refusal does not depend on existence, so it says the same thing
    as write_file about the same target and never reports not-found for a
    name inside the engine's subtree."""
    if directory_exists:
        (workspace.output_dir / SPILL_DIR_NAME).mkdir(parents=True)
    with pytest.raises(ValueError, match="engine-owned"):
        write(ops, f"{SPILL_DIR_NAME}/nope.json")
    assert not (workspace.output_dir / SPILL_DIR_NAME / "nope.json").exists()
    assert (workspace.output_dir / SPILL_DIR_NAME).exists() is directory_exists


@pytest.mark.parametrize("write", EXISTING_FILE_WRITES)
def test_a_missing_ordinary_file_still_reports_not_found(ops, workspace, write):
    """Outside the engine's subtree the search resolver's answer is unchanged."""
    workspace.output_dir.mkdir(parents=True, exist_ok=True)
    with pytest.raises(FileNotFoundError):
        write(ops, f"{SPILL_DIR_NAME}-mine/nope.json")
    assert not (workspace.output_dir / f"{SPILL_DIR_NAME}-mine").exists()


@pytest.mark.parametrize("write", EXISTING_FILE_WRITES)
def test_a_symlink_loop_target_still_reports_not_found(ops, workspace, write):
    """A name the search resolver reports missing but that cannot be resolved
    as a write target either keeps the not-found answer. On interpreters
    where ``Path.resolve`` raises RuntimeError on a loop, that error is the
    write resolver's, and it must not replace the search resolver's."""
    workspace.output_dir.mkdir(parents=True, exist_ok=True)
    loop = workspace.output_dir / "loopy"
    try:
        os.symlink(loop, loop)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not available on this platform/user")
    with pytest.raises(FileNotFoundError):
        write(ops, "output/loopy")


@pytest.mark.parametrize("write", EXISTING_FILE_WRITES)
@pytest.mark.parametrize(
    ("target", "written_path"),
    [
        pytest.param(
            f"{SPILL_DIR_NAME}/nope.json",
            f"{SPILL_DIR_NAME}/nope.json",
            id="engine-owned",
        ),
        pytest.param("output/nope.txt", "nope.txt", id="ordinary"),
    ],
)
def test_a_policy_refusal_still_reports_not_found(
    ops, workspace, monkeypatch, write, target, written_path
):
    """A name the search resolver reports missing but that the write
    resolver cannot even check -- because its own authority check
    (``requires_exact_file_operation_scope``) refuses first -- keeps the
    not-found answer too, whether or not the name would otherwise fall
    inside the engine's subtree. This pins the ``_resolve_existing_write_path``
    docstring's "or policy refusal" clause: without it, deleting ``ValueError``
    from the inner ``except (ValueError, OSError, RuntimeError)`` left every
    other test in this module green."""
    workspace.output_dir.mkdir(parents=True, exist_ok=True)

    def deny() -> bool:
        raise ValueError("policy backend down")

    monkeypatch.setattr(workspace, "requires_exact_file_operation_scope", deny)

    with pytest.raises(FileNotFoundError):
        write(ops, target)
    assert not (workspace.output_dir / written_path).exists()


@pytest.mark.parametrize("write", EXISTING_FILE_WRITES)
def test_a_containment_refusal_still_reports_not_found(tmp_path, write):
    """A relative target that climbs into a directory the search resolver
    honors via ``allowed_external_dirs`` -- but that ``_resolve_path`` does
    not, per ``test_ops_resolve_path_ignores_allowed_external_dirs`` above --
    still keeps the search resolver's not-found answer instead of the write
    resolver's containment ValueError, when the name does not exist there."""
    external = tmp_path / "external"
    external.mkdir(parents=True, exist_ok=True)
    workspace = TaskWorkspace(
        "task7", str(tmp_path), allowed_external_dirs=[str(external)]
    )
    workspace.output_dir.mkdir(parents=True, exist_ok=True)
    ops = WorkspaceFileOperations(workspace)

    with pytest.raises(FileNotFoundError):
        write(ops, "../../external/nope.txt")
    assert not (external / "nope.txt").exists()
