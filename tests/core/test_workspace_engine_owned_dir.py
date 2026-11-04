"""The engine-owned tool-results directory is invisible to every listing."""

import errno
import os
import threading
from pathlib import Path

import pytest

from xagent.core.tools.core.workspace_file_tool import WorkspaceFileOperations
from xagent.core.workspace import SPILL_DIR_NAME, MockWorkspace, TaskWorkspace


@pytest.fixture
def workspace(tmp_path):
    return TaskWorkspace("task_w", str(tmp_path))


@pytest.fixture
def spilled(workspace):
    """One engine file in the spill directory and one ordinary output file."""
    spill_dir = workspace.output_dir / SPILL_DIR_NAME
    spill_dir.mkdir(parents=True)
    engine_file = spill_dir / "acme-stored-result.json"
    engine_file.write_text("[]", encoding="utf-8")
    user_file = workspace.output_dir / "report.txt"
    user_file.write_text("report", encoding="utf-8")
    return engine_file, user_file


def test_get_all_files_omits_the_engine_directory(workspace, spilled):
    engine_file, user_file = spilled
    paths = {entry["file_path"] for entry in workspace.get_all_files()["output"]}
    assert str(user_file) in paths
    assert str(engine_file) not in paths


def test_get_output_files_omits_the_engine_directory(workspace, spilled):
    engine_file, user_file = spilled
    paths = {entry["file_path"] for entry in workspace.get_output_files()}
    assert str(user_file) in paths
    assert str(engine_file) not in paths


def test_get_output_files_non_recursive_omits_a_look_alike_file(workspace):
    """The non-recursive branch only scans the top of output/, where the
    engine directory can only be met as a regular file that happens to be
    named exactly like it, not as the directory itself."""
    look_alike = workspace.output_dir / SPILL_DIR_NAME
    workspace.output_dir.mkdir(parents=True, exist_ok=True)
    look_alike.write_text("not the engine directory", encoding="utf-8")
    sibling = workspace.output_dir / "ok.txt"
    sibling.write_text("ok", encoding="utf-8")

    paths = {
        entry["file_path"]
        for entry in workspace.get_output_files(include_subdirs=False)
    }
    assert str(sibling) in paths
    assert str(look_alike) not in paths


def test_scan_all_files_omits_the_engine_directory(workspace, spilled):
    engine_file, user_file = spilled
    scanned = workspace._scan_all_files()
    assert user_file in scanned
    assert engine_file not in scanned


# The output root is addressed by its absolute path. A bare top-level segment
# such as "output" does not resolve to that directory itself: the resolver
# behind the named-directory branch only reads a leading "output"/"input"/
# "temp" segment when the string contains a slash, so a lone "output" is
# instead treated as a name to look up inside the resolver's own default
# output directory, and raises FileNotFoundError because no such nested
# directory exists.
@pytest.mark.parametrize("show_hidden", [False, True])
def test_named_directory_listing_of_output_root_omits_the_engine_directory(
    workspace, spilled, show_hidden
):
    engine_file, user_file = spilled
    ops = WorkspaceFileOperations(workspace)
    listing = ops.list_files(
        str(workspace.output_dir), show_hidden=show_hidden, recursive=True
    )
    paths = {entry["path"] for entry in listing["files"]}
    assert str(user_file) in paths
    assert str(engine_file) not in paths
    # The directory entry itself is gone too, which is what stops the descent.
    assert str(workspace.output_dir / SPILL_DIR_NAME) not in paths


@pytest.mark.parametrize("show_hidden", [False, True])
def test_named_directory_listing_of_the_engine_directory_itself_is_empty(
    workspace, spilled, show_hidden
):
    """Addressing the engine directory by its own path returns nothing."""
    engine_file, _ = spilled
    ops = WorkspaceFileOperations(workspace)
    listing = ops.list_files(
        f"output/{SPILL_DIR_NAME}", show_hidden=show_hidden, recursive=True
    )
    assert listing["files"] == []
    assert engine_file.exists()


SYMLINK_LOOP_CASES = [
    pytest.param("direct", "symlink_loop", False, True, id="direct-show-hidden-false"),
    pytest.param("direct", "symlink_loop", True, True, id="direct-show-hidden-true"),
    pytest.param("nested", "symlink_loop", False, True, id="nested-show-hidden-false"),
    pytest.param("nested", "symlink_loop", True, True, id="nested-show-hidden-true"),
    pytest.param(
        "direct", ".symlink_loop", True, True, id="direct-dotted-show-hidden-true"
    ),
    pytest.param(
        "direct", ".symlink_loop", False, False, id="direct-dotted-show-hidden-false"
    ),
]


@pytest.mark.parametrize(
    "location, entry_name, show_hidden, reaches_stat", SYMLINK_LOOP_CASES
)
def test_named_directory_listing_reports_a_symlink_loop_as_a_filesystem_error(
    workspace, location, entry_name, show_hidden, reaches_stat
):
    # The ownership check resolves each entry, and a symlink loop cannot be
    # resolved. An entry that reaches the check this way still reaches
    # item.stat() below, so the listing fails with the filesystem's own
    # ELOOP error rather than one raised by the check. A dotted entry with
    # show_hidden off is skipped by the hidden-name rule before the check
    # ever runs, so the listing succeeds and simply omits it.
    parent = (
        workspace.output_dir if location == "direct" else workspace.output_dir / "sub"
    )
    parent.mkdir(parents=True, exist_ok=True)
    loop_path = parent / entry_name
    try:
        os.symlink(loop_path, loop_path)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not available on this platform/user")

    ops = WorkspaceFileOperations(workspace)
    if reaches_stat:
        with pytest.raises(OSError) as raised:
            ops.list_files(
                str(workspace.output_dir), show_hidden=show_hidden, recursive=True
            )
        assert raised.value.errno == errno.ELOOP
    else:
        listing = ops.list_files(
            str(workspace.output_dir), show_hidden=show_hidden, recursive=True
        )
        assert listing["files"] == []


def test_spill_temp_files_are_hidden_too(workspace):
    """The writer's .tmp name is not a dotfile; the directory rule is what hides it."""
    spill_dir = workspace.output_dir / SPILL_DIR_NAME
    spill_dir.mkdir(parents=True)
    leftover = spill_dir / "acme-stored-result.json.4242.partial.tmp"
    leftover.write_text("partial", encoding="utf-8")
    assert leftover not in workspace._scan_all_files()


def test_ordinary_output_subdirectory_is_still_listed(workspace):
    """The rule is a directory rule, not a name-prefix rule."""
    near_miss = workspace.output_dir / f"{SPILL_DIR_NAME}-mine"
    near_miss.mkdir(parents=True)
    mine = near_miss / "x.json"
    mine.write_text("{}", encoding="utf-8")
    assert mine in workspace._scan_all_files()
    assert str(mine) in {e["file_path"] for e in workspace.get_output_files()}


def test_auto_registration_ignores_the_engine_directory(workspace, mocker):
    """A file the engine drops in its own directory never becomes a file record."""
    registered: list[str] = []
    mocker.patch.object(
        TaskWorkspace,
        "register_file",
        lambda self, path, db_session=None: registered.append(str(path)) or "fid",
    )
    spill_dir = workspace.output_dir / SPILL_DIR_NAME
    spill_dir.mkdir(parents=True)
    with workspace.auto_register_files():
        (spill_dir / "acme-stored-result.json").write_text("[]", encoding="utf-8")
        (workspace.output_dir / "report.txt").write_text("report", encoding="utf-8")

    assert registered == [str(workspace.output_dir / "report.txt")]


def test_ownership_holds_when_the_output_dir_itself_is_a_symlink(tmp_path):
    """The check resolves the root too, so a symlinked output/ still matches.

    ``output_dir`` is built by appending names to a resolved ``base_dir``; no
    construction step resolves it, so only the comparison site can see through
    a symlink standing where ``output/`` does. Drop ``.resolve()`` from the
    parent of the reserved segment and both the listing and the write refusal
    go silently permissive.
    """

    workspace = TaskWorkspace("task_symlinked_output", str(tmp_path / "base"))
    elsewhere = workspace.workspace_dir / "real-output"
    (elsewhere / SPILL_DIR_NAME).mkdir(parents=True)
    workspace.output_dir.rmdir()
    try:
        workspace.output_dir.symlink_to(elsewhere, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not available on this platform/user")

    engine_file = workspace.output_dir / SPILL_DIR_NAME / "acme-stored-result.json"
    engine_file.write_text("[]", encoding="utf-8")
    user_file = workspace.output_dir / "report.txt"
    user_file.write_text("report", encoding="utf-8")

    assert workspace.is_engine_owned_path(engine_file) is True
    listed = {entry["file_path"] for entry in workspace.get_output_files()}
    assert str(user_file) in listed
    assert str(engine_file) not in listed
    with pytest.raises(ValueError, match="engine-owned"):
        WorkspaceFileOperations(workspace).write_file(
            str(workspace.output_dir / SPILL_DIR_NAME / "mine.txt"), "x"
        )


def _symlink(target, link):
    try:
        os.symlink(target, link)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not available on this platform/user")


def test_a_symlink_standing_in_for_the_reserved_name_does_not_move_the_directory(
    workspace,
):
    """The reserved directory is a name under output/, not its symlink target.

    The model's code execution tools run with output/ as their working
    directory, so a direct writer can put a symlink where the reserved name
    would go. Following it would hand that writer the choice of which
    directory is protected: every file under the target would drop out of the
    deliverables and become unwritable.
    """
    reports = workspace.output_dir / "reports"
    reports.mkdir(parents=True)
    report = reports / "quarterly.pdf"
    report.write_text("report", encoding="utf-8")
    _symlink("reports", workspace.output_dir / SPILL_DIR_NAME)

    assert workspace.is_engine_owned_path(report) is False
    assert str(report) in {e["file_path"] for e in workspace.get_output_files()}
    assert str(report) in {e["file_path"] for e in workspace.get_all_files()["output"]}
    assert report in workspace._scan_all_files()
    WorkspaceFileOperations(workspace).write_file(
        f"output/reports/{report.name}", "rewritten"
    )
    assert report.read_text(encoding="utf-8") == "rewritten"


def test_a_loop_at_the_reserved_name_does_not_fail_unrelated_callers(workspace):
    """A loop where the reserved directory would be is one entry's problem.

    Resolving the reserved name itself put the failure on the right-hand side
    of the comparison, where it had nothing to do with the path being asked
    about, so every caller failed about every path.
    """
    workspace.output_dir.mkdir(parents=True, exist_ok=True)
    ordinary = workspace.output_dir / "plain.txt"
    ordinary.write_text("plain", encoding="utf-8")
    loop = workspace.output_dir / SPILL_DIR_NAME
    _symlink(loop, loop)

    assert workspace.is_engine_owned_path(ordinary) is False
    assert str(ordinary) in {e["file_path"] for e in workspace.get_output_files()}
    assert str(ordinary) in {
        e["file_path"] for e in workspace.get_all_files()["output"]
    }
    assert ordinary in workspace._scan_all_files()
    WorkspaceFileOperations(workspace).write_file("output/plain.txt", "rewritten")
    assert ordinary.read_text(encoding="utf-8") == "rewritten"


def test_a_loop_at_the_reserved_temp_name_does_not_fail_the_listings(workspace):
    """The temp reserved root is compared the same way, for the same reason."""
    workspace.temp_dir.mkdir(parents=True, exist_ok=True)
    workspace.output_dir.mkdir(parents=True, exist_ok=True)
    ordinary = workspace.output_dir / "plain.txt"
    ordinary.write_text("plain", encoding="utf-8")
    loop = workspace.internal_temp_dir
    _symlink(loop, loop)

    all_files = workspace.get_all_files()
    assert str(ordinary) in {e["file_path"] for e in workspace.get_output_files()}
    assert str(ordinary) in {
        e["file_path"] for e in workspace.get_output_files(include_subdirs=False)
    }
    assert str(ordinary) in {e["file_path"] for e in all_files["output"]}
    assert all_files["temp"] == []
    assert ordinary in workspace._scan_all_files()


def test_an_aliased_reserved_temp_name_still_hides_its_target(workspace):
    """Unlike the engine-owned output subtree, this name's target stays hidden.

    The engine creates this directory itself, so the only way its name
    points elsewhere is an alias placed by something else. The name is
    reserved, and so is the directory inside temp/ that the name resolves
    to, so scratch data does not surface as a new user file just because it
    physically lives one directory over.
    """
    workspace.temp_dir.mkdir(parents=True, exist_ok=True)
    workspace.output_dir.mkdir(parents=True, exist_ok=True)
    alias_target = workspace.temp_dir / "drafts"
    alias_target.mkdir(parents=True)
    note = alias_target / "note.txt"
    note.write_text("note", encoding="utf-8")
    _symlink(alias_target, workspace.internal_temp_dir)

    all_files = workspace.get_all_files()
    assert all_files["temp"] == []
    assert note not in workspace._scan_all_files()


ESCAPING_ALIAS_TARGETS = [
    pytest.param("..", id="workspace-root"),
    pytest.param("../output", id="output-dir"),
]


@pytest.mark.parametrize("relative_target", ESCAPING_ALIAS_TARGETS)
def test_an_alias_escaping_temp_does_not_hide_the_rest_of_the_workspace(
    workspace, relative_target
):
    """A directory this name points at is reserved only when it is a direct child of temp/.

    Nothing the engine writes ever lives outside temp/, so an alias that
    escapes it names no scratch data to protect; honouring it anyway would
    let a symlink placed under temp/ blank out listings anywhere else in
    the workspace.
    """
    workspace.temp_dir.mkdir(parents=True, exist_ok=True)
    workspace.output_dir.mkdir(parents=True, exist_ok=True)
    plain = workspace.output_dir / "plain.txt"
    plain.write_text("plain", encoding="utf-8")
    _symlink(relative_target, workspace.internal_temp_dir)

    assert str(plain) in {e["file_path"] for e in workspace.get_output_files()}
    assert str(plain) in {
        e["file_path"] for e in workspace.get_output_files(include_subdirs=False)
    }
    assert str(plain) in {e["file_path"] for e in workspace.get_all_files()["output"]}


NON_CHILD_ALIAS_TARGETS = [
    pytest.param(".", id="temp-itself"),
    pytest.param("absolute-temp", id="temp-itself-absolute"),
    pytest.param("a/b", id="two-levels-deep"),
]


@pytest.mark.parametrize("relative_target", NON_CHILD_ALIAS_TARGETS)
def test_an_alias_that_is_not_a_direct_child_of_temp_hides_nothing(
    workspace, relative_target
):
    """The alias branch honours a direct child of temp/ and nothing else.

    A link to temp/ itself would put every temp/ file "inside" the alias
    and empty the temp listing. A link two levels down names a directory
    the engine's own scratch writer refuses to use (it requires the reserved
    root to be a direct child of temp/), so there is no scratch data there
    to keep hidden either.
    """
    workspace.temp_dir.mkdir(parents=True, exist_ok=True)
    workspace.output_dir.mkdir(parents=True, exist_ok=True)
    user_file = workspace.temp_dir / "user_scratch.txt"
    user_file.write_text("mine", encoding="utf-8")
    deep = workspace.temp_dir / "a" / "b"
    deep.mkdir(parents=True)
    deep_file = deep / "inside_deep.txt"
    deep_file.write_text("deep", encoding="utf-8")
    if relative_target == "absolute-temp":
        relative_target = str(workspace.temp_dir.resolve())
    _symlink(relative_target, workspace.internal_temp_dir)

    temp_listing = {e["file_path"] for e in workspace.get_all_files()["temp"]}
    assert str(user_file) in temp_listing
    assert str(deep_file) in temp_listing
    scanned = set(workspace._scan_all_files())
    assert user_file in scanned
    assert deep_file in scanned


def test_an_alias_outside_the_workspace_does_not_hide_output(workspace, tmp_path):
    workspace.temp_dir.mkdir(parents=True, exist_ok=True)
    workspace.output_dir.mkdir(parents=True, exist_ok=True)
    plain = workspace.output_dir / "plain.txt"
    plain.write_text("plain", encoding="utf-8")
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    _symlink(outside, workspace.internal_temp_dir)

    assert str(plain) in {e["file_path"] for e in workspace.get_output_files()}
    assert str(plain) in {
        e["file_path"] for e in workspace.get_output_files(include_subdirs=False)
    }
    assert str(plain) in {e["file_path"] for e in workspace.get_all_files()["output"]}


CASE_SPELLINGS = [
    SPILL_DIR_NAME.upper(),
    SPILL_DIR_NAME.capitalize(),
    SPILL_DIR_NAME.title(),
]


@pytest.mark.parametrize("spelling", CASE_SPELLINGS)
def test_a_case_variant_of_the_reserved_name_is_reserved(workspace, spilled, spelling):
    """One rule on every operating system, so one expectation in one test.

    On a case-insensitive file system this spelling is the engine's own
    directory, and a segment-by-segment comparison would let the write reach
    the engine's bytes. On a case-sensitive file system it is a different
    directory that the rule reserves anyway. Either way the answer, and this
    assertion, are the same.
    """
    engine_file, _ = spilled
    assert spelling != SPILL_DIR_NAME
    target = workspace.output_dir / spelling / engine_file.name
    assert workspace.is_engine_owned_path(target) is True
    with pytest.raises(ValueError, match="engine-owned"):
        WorkspaceFileOperations(workspace).write_file(
            f"output/{spelling}/{engine_file.name}", "rewritten"
        )
    assert engine_file.read_text(encoding="utf-8") == "[]"


OUTPUT_SEGMENT_SPELLINGS = ["OUTPUT", "Output"]


def _respell_output_segment(
    workspace: TaskWorkspace, spelling: str, *rest: str
) -> Path:
    """Build an absolute path with the output/ segment itself respelled.

    Every entry a listing returns is an absolute path (get_output_files and
    friends), so a direct writer already holds one; respelling the segment
    the workspace itself calls "output" is a shape that writer can produce
    without any other knowledge of the workspace layout.
    """
    respelled_output = workspace.output_dir.resolve().with_name(spelling)
    return respelled_output.joinpath(*rest)


@pytest.mark.parametrize("spelling", OUTPUT_SEGMENT_SPELLINGS)
def test_a_case_variant_of_the_output_segment_still_reserves_the_directory(
    workspace, spilled, spelling
):
    """The whole path down to the reserved name is compared case folded,
    not only the reserved name's own segment."""
    engine_file, _ = spilled
    target = _respell_output_segment(
        workspace, spelling, SPILL_DIR_NAME, engine_file.name
    )
    assert workspace.is_engine_owned_path(target) is True
    with pytest.raises(ValueError, match="engine-owned"):
        WorkspaceFileOperations(workspace).write_file(str(target), "rewritten")
    assert engine_file.read_text(encoding="utf-8") == "[]"


@pytest.mark.parametrize("spelling", OUTPUT_SEGMENT_SPELLINGS)
def test_a_case_variant_of_the_output_segment_refuses_delete_too(
    workspace, spilled, spelling
):
    engine_file, _ = spilled
    target = _respell_output_segment(
        workspace, spelling, SPILL_DIR_NAME, engine_file.name
    )
    with pytest.raises(ValueError, match="engine-owned"):
        WorkspaceFileOperations(workspace).delete_file(str(target))
    assert engine_file.exists()
    assert engine_file.read_text(encoding="utf-8") == "[]"


@pytest.mark.parametrize("spelling", OUTPUT_SEGMENT_SPELLINGS)
def test_a_case_variant_of_the_output_segment_is_refused_before_the_directory_exists(
    workspace, spelling
):
    workspace.output_dir.mkdir(parents=True, exist_ok=True)
    target = _respell_output_segment(workspace, spelling, SPILL_DIR_NAME, "x.json")
    with pytest.raises(ValueError, match="engine-owned"):
        WorkspaceFileOperations(workspace).write_file(str(target), "planted")
    assert not (workspace.output_dir / SPILL_DIR_NAME).exists()


def test_output_dir_itself_is_not_engine_owned(workspace):
    """The reserved subtree sits inside output/; output/ is not the subtree."""
    assert workspace.is_engine_owned_path(workspace.output_dir) is False


def test_a_plain_file_directly_in_output_is_not_engine_owned(workspace):
    workspace.output_dir.mkdir(parents=True, exist_ok=True)
    plain = workspace.output_dir / "report.txt"
    plain.write_text("report", encoding="utf-8")
    assert workspace.is_engine_owned_path(plain) is False


def test_the_reserved_name_only_matters_directly_under_output(workspace):
    """The rule reserves a name one level under output/, not the name anywhere."""
    workspace.temp_dir.mkdir(parents=True, exist_ok=True)
    workspace.input_dir.mkdir(parents=True, exist_ok=True)
    under_temp = workspace.temp_dir / SPILL_DIR_NAME / "x.json"
    under_input = workspace.input_dir / SPILL_DIR_NAME / "x.json"

    assert workspace.is_engine_owned_path(under_temp) is False
    assert workspace.is_engine_owned_path(under_input) is False


def test_a_same_named_directory_nested_deeper_than_the_first_level_is_ordinary(
    workspace,
):
    """The rule matches only the first segment under output/, not any depth."""
    nested = workspace.output_dir / "sub" / SPILL_DIR_NAME
    nested.mkdir(parents=True)
    deep_file = nested / "x.json"
    deep_file.write_text("{}", encoding="utf-8")

    assert workspace.is_engine_owned_path(deep_file) is False
    assert deep_file in workspace._scan_all_files()
    assert str(deep_file) in {e["file_path"] for e in workspace.get_output_files()}
    WorkspaceFileOperations(workspace).write_file(
        f"output/sub/{SPILL_DIR_NAME}/rewritten.json", "mine"
    )
    assert (nested / "rewritten.json").read_text(encoding="utf-8") == "mine"


def test_a_case_variant_nested_deeper_than_the_first_level_is_also_ordinary(workspace):
    """A case variant of the reserved name only matters at the first segment."""
    nested = workspace.output_dir / "sub" / SPILL_DIR_NAME.upper()
    nested.mkdir(parents=True)
    deep_file = nested / "x.json"
    deep_file.write_text("{}", encoding="utf-8")

    assert workspace.is_engine_owned_path(deep_file) is False
    assert deep_file in workspace._scan_all_files()


NEAR_MISS_DIRECTORY_NAMES = [f"{SPILL_DIR_NAME}-backup", f"a{SPILL_DIR_NAME}"]


@pytest.mark.parametrize("name", NEAR_MISS_DIRECTORY_NAMES)
def test_a_directory_whose_name_only_resembles_the_reserved_name_is_ordinary(
    workspace, name
):
    near_miss = workspace.output_dir / name
    near_miss.mkdir(parents=True)
    sibling = near_miss / "x.json"
    sibling.write_text("{}", encoding="utf-8")

    assert workspace.is_engine_owned_path(sibling) is False
    assert sibling in workspace._scan_all_files()
    assert str(sibling) in {e["file_path"] for e in workspace.get_output_files()}


def test_a_file_whose_name_only_resembles_the_reserved_name_is_ordinary(workspace):
    workspace.output_dir.mkdir(parents=True, exist_ok=True)
    look_alike = workspace.output_dir / f"{SPILL_DIR_NAME}.txt"
    look_alike.write_text("not the engine directory", encoding="utf-8")

    assert workspace.is_engine_owned_path(look_alike) is False
    assert look_alike in workspace._scan_all_files()


# --------------------------------------------------------------------------
# The write-side resolver on the workspace itself
# --------------------------------------------------------------------------

WRITE_PATH_SPELLINGS = [
    pytest.param(f"{SPILL_DIR_NAME}/x.json", id="relative"),
    pytest.param(f"./{SPILL_DIR_NAME}/x.json", id="dot-slash"),
    pytest.param(f"sub/../{SPILL_DIR_NAME}/x.json", id="dotdot"),
    pytest.param(f"{SPILL_DIR_NAME}/sub/x.json", id="nested"),
    pytest.param(SPILL_DIR_NAME, id="directory-itself"),
    pytest.param(f"{SPILL_DIR_NAME.upper()}/x.json", id="case-variant"),
    pytest.param("absolute", id="absolute"),
]


@pytest.mark.parametrize("spelling", WRITE_PATH_SPELLINGS)
def test_resolve_write_path_refuses_the_engine_subtree(workspace, spelling):
    """Every spelling resolve_path accepts for the subtree is refused for writing."""
    if spelling == "absolute":
        spelling = str(workspace.output_dir / SPILL_DIR_NAME / "x.json")
    assert workspace.resolve_path(spelling, default_dir="output")  # resolvable
    with pytest.raises(ValueError, match="engine-owned"):
        workspace.resolve_write_path(spelling, default_dir="output")
    assert not (workspace.output_dir / SPILL_DIR_NAME).exists()


def test_resolve_write_path_returns_what_resolve_path_returns_elsewhere(workspace):
    for spelling, default_dir in [
        ("report.txt", "output"),
        (f"{SPILL_DIR_NAME}-mine/x.json", "output"),
        ("notes.txt", "temp"),
        (str(workspace.output_dir / "sub" / "report.txt"), "output"),
    ]:
        assert workspace.resolve_write_path(
            spelling, default_dir=default_dir
        ) == workspace.resolve_path(spelling, default_dir=default_dir)


MOCK_WRITE_SPELLINGS = [
    pytest.param("report.txt", "output", id="output-file"),
    pytest.param("notes.txt", "temp", id="temp-file"),
    pytest.param(f"{SPILL_DIR_NAME}/x.json", "output", id="engine-spelling"),
]


@pytest.mark.parametrize("spelling, default_dir", MOCK_WRITE_SPELLINGS)
def test_the_mock_workspace_keeps_the_write_entry_point(spelling, default_dir):
    """Tools built for a listing get a MockWorkspace; the write entry point
    the converted tools call must exist on it and answer as resolve_path
    does. The mock never writes to disk, so nothing is refused there."""
    mock = MockWorkspace()
    assert mock.resolve_write_path(spelling, default_dir) == mock.resolve_path(
        spelling, default_dir
    )


@pytest.mark.parametrize("spelling, default_dir", MOCK_WRITE_SPELLINGS)
def test_the_mock_workspace_owns_nothing_and_refuses_nothing(spelling, default_dir):
    """The predicate and the refusal the file tool calls exist on the mock
    too, and answer the same way for every spelling, the engine's included:
    nothing is engine-owned where nothing is written."""
    mock = MockWorkspace()
    target = mock.resolve_path(spelling, default_dir)
    assert mock.is_engine_owned_path(target) is False
    assert mock.refuse_engine_owned_write(target, spelling) == target


def test_the_file_tool_refusal_is_the_workspace_refusal(workspace, monkeypatch):
    """One owner for the decision: the file tool calls through, it does not
    re-implement the check. Replacing the workspace's refusal changes what
    the file tool raises."""

    def sentinel(self, resolved_path, requested):
        raise ValueError(f"SENTINEL for {requested}")

    monkeypatch.setattr(TaskWorkspace, "refuse_engine_owned_write", sentinel)
    ops = WorkspaceFileOperations(workspace)
    with pytest.raises(ValueError, match="SENTINEL for output/plain.txt"):
        ops.write_file("output/plain.txt", "x")
    assert not (workspace.output_dir / "plain.txt").exists()


# --------------------------------------------------------------------------
# A reserved name that is already taken is announced once, at construction
# --------------------------------------------------------------------------

TAKEN_RESERVED_NAME_SHAPES = [
    pytest.param("directory-with-a-file", id="directory-with-a-file"),
    pytest.param("empty-directory", id="empty-directory"),
    pytest.param("plain-file", id="plain-file"),
    pytest.param("dangling-symlink", id="dangling-symlink"),
]


@pytest.mark.parametrize("shape", TAKEN_RESERVED_NAME_SHAPES)
def test_a_taken_reserved_name_is_announced_when_the_workspace_is_built(
    tmp_path, caplog, shape
):
    """Files already under the reserved name drop out of every listing and
    become unwritable with no error on those paths, so the one place that
    runs once per workspace object says so."""
    reserved = tmp_path / "task_w" / "output" / SPILL_DIR_NAME
    reserved.parent.mkdir(parents=True)
    if shape == "directory-with-a-file":
        reserved.mkdir()
        (reserved / "Q3-report.txt").write_text("mine", encoding="utf-8")
    elif shape == "empty-directory":
        reserved.mkdir()
    elif shape == "plain-file":
        reserved.write_text("mine", encoding="utf-8")
    else:
        _symlink("does-not-exist", reserved)

    with caplog.at_level("WARNING", logger="xagent.core.workspace"):
        TaskWorkspace("task_w", str(tmp_path))

    warnings = [r for r in caplog.records if "reserved" in r.getMessage()]
    assert len(warnings) == 1
    assert warnings[0].levelname == "WARNING"
    assert "task_w" in warnings[0].getMessage()
    assert str(reserved) in warnings[0].getMessage()


def test_an_absent_reserved_name_is_not_announced(tmp_path, caplog):
    with caplog.at_level("WARNING", logger="xagent.core.workspace"):
        workspace = TaskWorkspace("task_w", str(tmp_path))
    assert not (workspace.output_dir / SPILL_DIR_NAME).exists()
    assert [r for r in caplog.records if "reserved" in r.getMessage()] == []


def _reserved_warnings(caplog):
    return [r for r in caplog.records if "reserved" in r.getMessage()]


def test_a_taken_reserved_name_is_announced_once_per_process_per_workspace(
    tmp_path, caplog
):
    """Every tool family builds its own workspace object for the same task,
    so the announcement is keyed on the reserved path, not on the object:
    the same workspace built again says nothing, another workspace with the
    same name taken is announced on its own."""
    for name in ("task_a", "task_b"):
        (tmp_path / name / "output" / SPILL_DIR_NAME).mkdir(parents=True)

    with caplog.at_level("WARNING", logger="xagent.core.workspace"):
        TaskWorkspace("task_a", str(tmp_path))
        TaskWorkspace("task_a", str(tmp_path))
        TaskWorkspace("task_a", str(tmp_path))
    assert len(_reserved_warnings(caplog)) == 1
    assert "task_a" in _reserved_warnings(caplog)[0].getMessage()

    with caplog.at_level("WARNING", logger="xagent.core.workspace"):
        TaskWorkspace("task_b", str(tmp_path))
        TaskWorkspace("task_b", str(tmp_path))
    messages = [r.getMessage() for r in _reserved_warnings(caplog)]
    assert len(messages) == 2
    assert "task_b" in messages[1]


def test_concurrent_builds_of_one_workspace_announce_the_taken_name_once(
    tmp_path, caplog
):
    """The set of announced names is shared by every thread that builds a
    workspace, so concurrent builds of the same workspace race for one slot."""
    (tmp_path / "task_c" / "output" / SPILL_DIR_NAME).mkdir(parents=True)
    starter = threading.Barrier(8)

    def build() -> None:
        starter.wait()
        TaskWorkspace("task_c", str(tmp_path))

    threads = [threading.Thread(target=build) for _ in range(8)]
    with caplog.at_level("WARNING", logger="xagent.core.workspace"):
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    assert len(_reserved_warnings(caplog)) == 1


def test_cleanup_forgets_the_announcement_so_a_fresh_occupant_is_announced_again(
    tmp_path, caplog
):
    """cleanup() removes the whole workspace directory, so a brand new
    occupant that later shows up at the same path is a new fact, not a
    repeat of the one that was cleaned away."""
    (tmp_path / "task_d" / "output" / SPILL_DIR_NAME).mkdir(parents=True)

    with caplog.at_level("WARNING", logger="xagent.core.workspace"):
        workspace = TaskWorkspace("task_d", str(tmp_path))
    assert len(_reserved_warnings(caplog)) == 1

    workspace.cleanup()
    (tmp_path / "task_d" / "output" / SPILL_DIR_NAME).mkdir(parents=True)

    with caplog.at_level("WARNING", logger="xagent.core.workspace"):
        TaskWorkspace("task_d", str(tmp_path))
    assert len(_reserved_warnings(caplog)) == 2


# --------------------------------------------------------------------------
# The two listing predicates have different scopes, on purpose
# --------------------------------------------------------------------------


@pytest.mark.parametrize("show_hidden", [False, True])
def test_named_temp_listing_shows_the_internal_scratch_root_only_when_asked(
    workspace, show_hidden
):
    """The named-directory listing hides only the engine tool-results
    directory; the internal scratch root under temp/ is governed there by
    the ordinary hidden-name rule, so show_hidden surfaces it. The
    workspace-wide listings never do."""
    workspace.internal_temp_dir.mkdir(parents=True)
    scratch = workspace.internal_temp_dir / "frame.bin"
    scratch.write_bytes(b"x")
    note = workspace.temp_dir / "note.txt"
    note.write_text("note", encoding="utf-8")

    listing = WorkspaceFileOperations(workspace).list_files(
        str(workspace.temp_dir), show_hidden=show_hidden, recursive=True
    )
    named = {entry["path"] for entry in listing["files"]}
    assert str(note) in named
    assert (str(scratch) in named) is show_hidden

    assert {e["file_path"] for e in workspace.get_all_files()["temp"]} == {str(note)}
    assert scratch not in workspace._scan_all_files()
    assert note in workspace._scan_all_files()


# --------------------------------------------------------------------------
# What the model is told before it tries
# --------------------------------------------------------------------------

RESERVED_NOTE = (
    f"output/{SPILL_DIR_NAME}/ is reserved for the engine and refuses writes."
)

WRITE_SIDE_FILE_TOOLS = {
    "write_file",
    "prepare_html_asset",
    "append_file",
    "delete_file",
    "create_directory",
    "write_json_file",
    "write_csv_file",
    "edit_file",
    "find_and_replace",
}


def test_every_write_side_file_tool_description_names_the_reserved_directory(
    workspace,
):
    """The refusal is one turn late as a teacher; every tool that can reach
    it says so up front, and no read-side tool does."""
    from xagent.core.tools.adapters.vibe.workspace_file_tool import (
        create_workspace_file_tools,
    )

    tools = {tool.name: tool for tool in create_workspace_file_tools(workspace)}
    assert WRITE_SIDE_FILE_TOOLS <= set(tools)
    for name, tool in tools.items():
        assert (RESERVED_NOTE in tool.description) is (name in WRITE_SIDE_FILE_TOOLS), (
            name
        )


def test_the_sql_export_description_names_the_reserved_directory(workspace):
    from xagent.core.tools.adapters.vibe.sql_tool import SqlQueryTool

    tools = {tool.name: tool for tool in SqlQueryTool(workspace).get_tools()}
    assert RESERVED_NOTE in tools["execute_sql_query"].description
    assert RESERVED_NOTE not in tools["get_database_type"].description
