"""Slice 3 of #757: scope-aware workspace paths in the web layer.

Covers ``_build_allowed_external_dirs`` scoping (shared by default,
scope-local under ``isolate_external_dirs``) and the websocket
output-path task-scope check tolerating scope segments between the user
root and the task dir.
"""

import pytest

from xagent.core.execution_scope import (
    ExecutionScope,
    set_execution_scope_resolver,
)
from xagent.web.api.chat import _build_allowed_external_dirs
from xagent.web.api.websocket import (
    _output_path_in_current_task_scope,
    _scope_segments_for_task,
)
from xagent.web.services.workspace_binding import _build_external_allowlist


@pytest.fixture(autouse=True)
def _clear_resolver():
    set_execution_scope_resolver(None)
    yield
    set_execution_scope_resolver(None)


class TestAllowedExternalDirs:
    def test_unscoped_is_byte_identical(self, monkeypatch, tmp_path):
        monkeypatch.setenv("XAGENT_UPLOADS_DIR", str(tmp_path))
        dirs = _build_allowed_external_dirs(7)
        assert str(tmp_path / "user_7") in dirs

    def test_default_sharing_ignores_scope(self, monkeypatch, tmp_path):
        """isolate_external_dirs=False: every scope of the user still gets
        the shared user-level upload dir (already-uploaded KB files must not
        silently disappear under a new scope)."""
        monkeypatch.setenv("XAGENT_UPLOADS_DIR", str(tmp_path))
        scope = ExecutionScope(workspace_segments=("tenant-a",))
        assert _build_allowed_external_dirs(7, scope=scope) == (
            _build_allowed_external_dirs(7)
        )

    def test_isolation_builds_scope_local_dir(self, monkeypatch, tmp_path):
        monkeypatch.setenv("XAGENT_UPLOADS_DIR", str(tmp_path))
        scope = ExecutionScope(
            workspace_segments=("tenant-a",), isolate_external_dirs=True
        )
        dirs = _build_allowed_external_dirs(7, scope=scope)
        assert str(tmp_path / "user_7" / "tenant-a") in dirs
        assert str(tmp_path / "user_7") not in dirs

    def test_isolation_flag_with_no_segments_stays_user_level(
        self, monkeypatch, tmp_path
    ):
        """Fields are independent: the flag with no segments isolates to the
        (segment-less) scoped root, which IS the user root."""
        monkeypatch.setenv("XAGENT_UPLOADS_DIR", str(tmp_path))
        scope = ExecutionScope(isolate_external_dirs=True)
        dirs = _build_allowed_external_dirs(7, scope=scope)
        assert str(tmp_path / "user_7") in dirs


class TestAllowedExternalDirsMatchesWorkspaceBinding:
    """Two independent implementations of the same computation exist today:
    ``chat._build_allowed_external_dirs`` (consumed by ``WebToolConfig``) and
    ``workspace_binding._build_external_allowlist`` (the sandbox mount
    intent's folding input) -- see the ``workspace_binding`` module
    docstring for why they are not yet collapsed onto one. Pin them
    equivalent across scope shapes so a change to one that silently
    diverges from the other is caught here instead of only showing up as a
    runtime access-policy/mount mismatch.
    """

    @pytest.mark.parametrize(
        "scope",
        [
            None,
            ExecutionScope(workspace_segments=("tenant-a",)),
            ExecutionScope(
                workspace_segments=("tenant-a",), isolate_external_dirs=True
            ),
            ExecutionScope(isolate_external_dirs=True),
            ExecutionScope(
                sandbox_key_suffix="ca-1",
                workspace_segments=("ca1", "actor7"),
                sandbox_mount_segments=("ca1",),
                isolate_external_dirs=True,
            ),
        ],
    )
    def test_equivalent_across_scope_shapes(self, monkeypatch, tmp_path, scope):
        monkeypatch.setenv("XAGENT_UPLOADS_DIR", str(tmp_path))
        # get_external_upload_dirs() only includes dirs that exist on disk.
        external_dir = tmp_path.parent / f"{tmp_path.name}-shared-kb"
        external_dir.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("XAGENT_EXTERNAL_UPLOAD_DIRS", str(external_dir))

        assert tuple(_build_allowed_external_dirs(7, scope=scope)) == tuple(
            candidate.path for candidate in _build_external_allowlist(7, scope)
        )

    def test_provenance_marks_only_the_user_upload_dir_as_scope_derived(
        self, monkeypatch, tmp_path
    ):
        """The folding input additionally carries each entry's provenance,
        which is what decides whether an unfoldable candidate may become its
        own bind. Only the scope-narrowed user upload dir is workspace
        derived; deployment external dirs are operator-declared mounts."""
        monkeypatch.setenv("XAGENT_UPLOADS_DIR", str(tmp_path))
        external_dir = tmp_path.parent / f"{tmp_path.name}-shared-kb"
        external_dir.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("XAGENT_EXTERNAL_UPLOAD_DIRS", str(external_dir))

        scope = ExecutionScope(
            workspace_segments=("tenant-a",), isolate_external_dirs=True
        )
        candidates = _build_external_allowlist(7, scope)

        assert [(c.path, c.origin) for c in candidates] == [
            (str(tmp_path / "user_7" / "tenant-a"), "scope"),
            (str(external_dir), "deployment"),
        ]


class TestScopeSegmentsForTask:
    def test_none_task_id_is_unscoped(self):
        """The legacy-preview backfill can infer an owner but no task id;
        no task identity means unscoped — never a resolver query for the
        literal string "None"."""
        seen = []
        set_execution_scope_resolver(lambda task_id: seen.append(task_id))
        assert _scope_segments_for_task(None) == ()
        assert seen == []

    def test_scoped_task_yields_its_segments(self):
        set_execution_scope_resolver(
            lambda task_id: ExecutionScope(workspace_segments=("tenant-a",))
        )
        assert _scope_segments_for_task(42) == ("tenant-a",)


class TestOutputPathTaskScopeCheck:
    def test_unscoped_layouts_unchanged(self):
        assert _output_path_in_current_task_scope(
            "user_1/web_task_5/output/a.txt", 5, 1
        )
        assert _output_path_in_current_task_scope("web_task_5/output/a.txt", 5, 1)
        assert not _output_path_in_current_task_scope(
            "user_1/web_task_6/output/a.txt", 5, 1
        )
        assert not _output_path_in_current_task_scope(
            "user_2/web_task_5/output/a.txt", 5, 1
        )

    def test_scoped_layout_accepted(self):
        assert _output_path_in_current_task_scope(
            "user_1/tenant-a/web_task_5/output/a.txt", 5, 1
        )
        assert _output_path_in_current_task_scope(
            "user_1/tenant-a/proj/web_task_5/output/a.txt", 5, 1
        )

    def test_scoped_layout_of_other_task_rejected(self):
        assert not _output_path_in_current_task_scope(
            "user_1/tenant-a/web_task_6/output/a.txt", 5, 1
        )
        assert not _output_path_in_current_task_scope(
            "user_1/tenant-a/web_task_5/input/a.txt", 5, 1
        )

    def test_segment_named_like_the_task_dir_does_not_shadow_it(self):
        """The segment charset allows a scope segment literally named like
        the task dir; the scan must not stop at it and reject the real task
        dir further down (Gemini round-2 finding on #789)."""
        assert _output_path_in_current_task_scope(
            "user_1/web_task_5/web_task_5/output/a.txt", 5, 1
        )
        # ...but a lookalike segment alone still is not an output path.
        assert not _output_path_in_current_task_scope(
            "user_1/web_task_5/other-segment/input/a.txt", 5, 1
        )
