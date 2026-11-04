"""Slice 3 of #757: scope-aware workspace paths and storage keys (core layer).

Covers the ``scoped_user_root`` entry point, the storage-key builders'
scope-segment insertion (and that #759 prefix-scope enforcement admits the
scoped keys), the workspace's scoped canonical roots, and the allowed-path
enforcement confining scoped executions.
"""

import os
import stat
from pathlib import Path

import pytest

from xagent.core.execution_scope import (
    InvalidScopeComponentError,
)
from xagent.core.file_storage import (
    StorageKeyScopeError,
    get_unscoped_file_storage,
    get_user_file_storage,
)
from xagent.core.file_storage.keys import (
    build_task_output_storage_key,
    build_upload_storage_key,
)
from xagent.core.file_storage.storage import normalize_storage_key
from xagent.core.workspace import TaskWorkspace, scoped_user_root

SEGMENTS = ("tenant-a", "proj-1")


class TestScopedUserRoot:
    def test_unscoped_is_byte_identical_to_legacy_layout(self, tmp_path):
        assert scoped_user_root(tmp_path, 7) == tmp_path / "user_7"

    def test_segments_inserted_after_user_root(self, tmp_path):
        assert (
            scoped_user_root(tmp_path, 7, SEGMENTS)
            == tmp_path / "user_7" / "tenant-a" / "proj-1"
        )

    def test_none_base_uses_uploads_dir(self, monkeypatch, tmp_path):
        monkeypatch.setenv("XAGENT_UPLOADS_DIR", str(tmp_path / "uploads"))
        assert scoped_user_root(None, 7) == tmp_path / "uploads" / "user_7"

    def test_invalid_segment_rejects_instead_of_falling_back(self, tmp_path):
        """Falling back to the unscoped path would silently merge namespaces."""
        with pytest.raises(InvalidScopeComponentError):
            scoped_user_root(tmp_path, 7, ("ok", "../escape"))
        with pytest.raises(InvalidScopeComponentError):
            scoped_user_root(tmp_path, 7, ("",))


class TestScopedStorageKeys:
    def test_unscoped_keys_are_byte_identical(self):
        assert (
            build_upload_storage_key(7, "fid", "a.txt") == "users/7/uploads/fid/a.txt"
        )
        assert (
            build_task_output_storage_key(7, 5, "fid", "output/a.txt")
            == "users/7/tasks/5/outputs/fid/output/a.txt"
        )

    def test_scope_segments_inserted_after_user_root(self):
        assert (
            build_upload_storage_key(7, "fid", "a.txt", scope_segments=SEGMENTS)
            == "users/7/tenant-a/proj-1/uploads/fid/a.txt"
        )
        assert (
            build_task_output_storage_key(
                7, 5, "fid", "output/a.txt", scope_segments=SEGMENTS
            )
            == "users/7/tenant-a/proj-1/tasks/5/outputs/fid/output/a.txt"
        )

    def test_invalid_segment_rejects(self):
        with pytest.raises(InvalidScopeComponentError):
            build_task_output_storage_key(
                7, 5, "fid", "output/a.txt", scope_segments=("a/b",)
            )

    def test_scoped_keys_pass_strict_normalization(self):
        key = build_task_output_storage_key(
            7, 5, "fid", "output/a.txt", scope_segments=SEGMENTS
        )
        assert normalize_storage_key(key, strict=True) == key

    def test_two_scopes_produce_disjoint_storage_prefixes(self):
        key_a = build_task_output_storage_key(
            7, 5, "fid", "o.txt", scope_segments=("tenant-a",)
        )
        key_b = build_task_output_storage_key(
            7, 5, "fid", "o.txt", scope_segments=("tenant-b",)
        )
        unscoped = build_task_output_storage_key(7, 5, "fid", "o.txt")
        assert len({key_a, key_b, unscoped}) == 3


class TestScopedKeysUnderPrefixEnforcement:
    """The scoped prefix extends #759's user-bound prefix, so per-user
    prefix-scope enforcement admits scoped keys and still rejects other
    users' keys."""

    @pytest.fixture
    def local_storage(self, monkeypatch, tmp_path):
        monkeypatch.setenv("XAGENT_FILE_STORAGE_URI", (tmp_path / "objects").as_uri())
        monkeypatch.setenv(
            "XAGENT_FILE_MATERIALIZE_DIR", str(tmp_path / "materialized")
        )
        get_unscoped_file_storage.cache_clear()
        yield get_unscoped_file_storage()
        get_unscoped_file_storage.cache_clear()

    def test_scoped_key_round_trips_through_user_storage(self, local_storage, tmp_path):
        storage = get_user_file_storage(7)
        source = tmp_path / "source.txt"
        source.write_text("scoped", encoding="utf-8")
        key = build_task_output_storage_key(
            7, 5, "fid", "output/source.txt", scope_segments=SEGMENTS
        )

        stored = storage.put_file(source, key)

        assert stored.key == key
        assert storage.exists(key)

    def test_other_users_scoped_key_is_rejected(self, local_storage, tmp_path):
        storage = get_user_file_storage(7)
        source = tmp_path / "source.txt"
        source.write_text("scoped", encoding="utf-8")
        foreign = build_task_output_storage_key(
            8, 5, "fid", "output/source.txt", scope_segments=SEGMENTS
        )

        with pytest.raises(StorageKeyScopeError):
            storage.put_file(source, foreign)


class TestScopedWorkspace:
    def _scoped_workspace(self, tmp_path, segments=SEGMENTS) -> TaskWorkspace:
        base = scoped_user_root(tmp_path, 7, segments)
        return TaskWorkspace(
            "web_task_5", str(base), scope_segments=segments, db_task_id=5
        )

    def test_workspace_dir_lands_in_scoped_subtree(self, tmp_path):
        workspace = self._scoped_workspace(tmp_path)
        assert (
            workspace.workspace_dir
            == (tmp_path / "user_7" / "tenant-a" / "proj-1" / "web_task_5").resolve()
        )

    def test_task_directories_inherit_sandbox_group_write(self, tmp_path, monkeypatch):
        base = tmp_path / "user_7"
        base.mkdir()
        monkeypatch.setattr("xagent.core.workspace.get_uploads_dir", lambda: tmp_path)
        real_stat = Path.stat

        def sandbox_root_stat(path: Path, *args, **kwargs):
            result = real_stat(path, *args, **kwargs)
            if path == base:
                values = list(result)
                values[0] |= stat.S_ISGID
                values[5] = 1010
                return os.stat_result(values)
            return result

        assigned_gids: list[int] = []
        requested_modes: list[int] = []
        real_fchmod = os.fchmod

        def record_fchmod(fd: int, mode: int) -> None:
            requested_modes.append(mode)
            real_fchmod(fd, mode)

        monkeypatch.setattr(Path, "stat", sandbox_root_stat)
        monkeypatch.setattr(
            "xagent.core.workspace.os.fchown",
            lambda _fd, _uid, gid: assigned_gids.append(gid),
        )
        monkeypatch.setattr("xagent.core.workspace.os.fchmod", record_fchmod)

        workspace = TaskWorkspace("web_task_5", str(base))

        assert stat.S_IMODE(real_stat(workspace.workspace_dir).st_mode) & 0o777 == 0o750
        for directory in (
            workspace.input_dir,
            workspace.output_dir,
            workspace.temp_dir,
        ):
            assert stat.S_IMODE(real_stat(directory).st_mode) & 0o777 == 0o770
        assert (
            stat.S_IMODE(real_stat(workspace.internal_temp_dir).st_mode) & 0o777
            == 0o750
        )
        # workspace + input/output/temp + the private runtime temp directory.
        assert assigned_gids == [1010] * 5
        assert requested_modes == [0o2750, *([0o2770] * 3), 0o2750]

    def test_scoped_parent_chain_and_existing_contents_are_repaired(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr("xagent.core.workspace.get_uploads_dir", lambda: tmp_path)
        prepared_root = tmp_path / "user_7"
        base = prepared_root / "tenant-a" / "proj-1"
        output = base / "web_task_5" / "output"
        output.mkdir(parents=True)
        (base / "web_task_5" / "input").mkdir()
        (base / "web_task_5" / "temp").mkdir()
        legacy = output / "legacy.txt"
        legacy.write_text("legacy", encoding="utf-8")
        legacy.chmod(0o600)
        (base / "web_task_5" / ".xagent-permissions-v2").write_text("")

        prepared_inode = prepared_root.stat().st_ino
        real_fstat = os.fstat

        def sandbox_anchor_fstat(fd: int):
            result = real_fstat(fd)
            if result.st_ino == prepared_inode:
                values = list(result)
                values[0] |= stat.S_ISGID
                values[5] = 1010
                return os.stat_result(values)
            return result

        assigned_gids: list[int] = []
        requested_modes: list[int] = []
        real_fchmod = os.fchmod

        def record_fchmod(fd: int, mode: int) -> None:
            requested_modes.append(mode)
            real_fchmod(fd, mode)

        monkeypatch.setattr("xagent.core.workspace.os.fstat", sandbox_anchor_fstat)
        monkeypatch.setattr(
            "xagent.core.workspace.os.fchown",
            lambda _fd, _uid, gid: assigned_gids.append(gid),
        )
        monkeypatch.setattr(
            "xagent.core.workspace.os.fchmod",
            record_fchmod,
        )

        TaskWorkspace("web_task_5", str(base), scope_segments=SEGMENTS, db_task_id=5)

        assert (
            stat.S_IMODE((prepared_root / "tenant-a").stat().st_mode) & 0o777 == 0o755
        )
        assert stat.S_IMODE(base.stat().st_mode) & 0o777 == 0o755
        assert stat.S_IMODE(legacy.stat().st_mode) == 0o660
        assert assigned_gids
        assert requested_modes.count(0o2750) >= 2

    def test_two_scopes_produce_disjoint_workspace_dirs(self, tmp_path):
        ws_a = self._scoped_workspace(tmp_path, ("tenant-a",))
        ws_b = self._scoped_workspace(tmp_path, ("tenant-b",))
        unscoped = TaskWorkspace("web_task_5", str(scoped_user_root(tmp_path, 7)))
        assert (
            len(
                {
                    ws_a.workspace_dir,
                    ws_b.workspace_dir,
                    unscoped.workspace_dir,
                }
            )
            == 3
        )

    def test_unscoped_workspace_layout_is_byte_identical(self, tmp_path):
        workspace = TaskWorkspace("web_task_5", str(tmp_path / "user_7"))
        assert workspace.workspace_dir == (tmp_path / "user_7" / "web_task_5").resolve()
        assert workspace.scope_segments == ()

    def test_canonical_task_root_respects_scope_segments(self, tmp_path):
        """``_user_workspace_base_dir`` must recognize the scoped base and
        not append a duplicate ``user_{id}`` (or lose the segments)."""
        workspace = self._scoped_workspace(tmp_path)
        assert workspace._user_workspace_base_dir(7) == workspace.base_dir

        unscoped = TaskWorkspace("web_task_5", str(tmp_path / "user_7"))
        assert unscoped._user_workspace_base_dir(7) == unscoped.base_dir

        raw_base = TaskWorkspace(
            "web_task_5", str(tmp_path), scope_segments=("tenant-a",)
        )
        assert (
            raw_base._user_workspace_base_dir(7)
            == raw_base.base_dir / "user_7" / "tenant-a"
        )

    def test_invalid_scope_segments_reject_at_construction(self, tmp_path):
        with pytest.raises(InvalidScopeComponentError):
            TaskWorkspace("web_task_5", str(tmp_path), scope_segments=("a:b",))


class TestScopedAllowedPathEnforcement:
    def test_scoped_task_cannot_escape_its_subtree(self, tmp_path):
        """With scope-local external dirs (isolate_external_dirs), absolute
        paths in the user root outside the scope's segments are rejected."""
        scoped_base = scoped_user_root(tmp_path, 7, ("tenant-a",))
        other_scope_file = scoped_user_root(tmp_path, 7, ("tenant-b",)) / "secret.txt"
        other_scope_file.parent.mkdir(parents=True)
        other_scope_file.write_text("other scope")
        user_level_file = tmp_path / "user_7" / "shared.txt"
        user_level_file.write_text("user level")

        workspace = TaskWorkspace(
            "web_task_5",
            str(scoped_base),
            allowed_external_dirs=[str(scoped_base)],
            scope_segments=("tenant-a",),
        )

        inside = workspace.workspace_dir / "output" / "mine.txt"
        assert workspace.resolve_path(str(inside)) == inside.resolve()

        with pytest.raises(ValueError):
            workspace.resolve_path(str(other_scope_file))
        with pytest.raises(ValueError):
            workspace.resolve_path(str(user_level_file))

    def test_default_sharing_keeps_user_root_reachable(self, tmp_path):
        """isolate_external_dirs=False (default): the shared user-level
        upload dir stays in the allowlist, so already-uploaded KB files do
        not silently disappear under a new scope."""
        scoped_base = scoped_user_root(tmp_path, 7, ("tenant-a",))
        user_root = scoped_user_root(tmp_path, 7)
        shared_file = user_root / "kb.txt"
        shared_file.parent.mkdir(parents=True, exist_ok=True)
        shared_file.write_text("shared")

        workspace = TaskWorkspace(
            "web_task_5",
            str(scoped_base),
            allowed_external_dirs=[str(user_root)],
            scope_segments=("tenant-a",),
        )

        assert workspace.resolve_path(str(shared_file)) == shared_file.resolve()
