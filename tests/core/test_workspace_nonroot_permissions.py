import errno
import os
import stat
import threading
from pathlib import Path

import pytest

import xagent.core.workspace as workspace_module
from xagent.core.workspace import TaskWorkspace, WorkspaceFileRegistration


@pytest.fixture
def workspace(tmp_path):
    return TaskWorkspace("task7", str(tmp_path))


def test_permission_migration_repairs_nested_legacy_files(
    workspace, tmp_path, monkeypatch
):
    nested = workspace.output_dir / "legacy" / "nested"
    nested.mkdir(parents=True)
    nested.chmod(0o700)
    report = nested / "report.txt"
    report.write_text("legacy", encoding="utf-8")
    report.chmod(0o4600)
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    outside.chmod(0o600)
    (workspace.output_dir / "outside-link").symlink_to(outside)
    requested_modes: list[int] = []
    real_fchmod = os.fchmod

    def record_fchmod(fd: int, mode: int) -> None:
        requested_modes.append(mode)
        real_fchmod(fd, mode)

    monkeypatch.setattr(workspace_module.os, "fchown", lambda *_args: None)
    monkeypatch.setattr(workspace_module.os, "fchmod", record_fchmod)

    flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
    workspace_fd = os.open(workspace.workspace_dir, flags)
    try:
        workspace._migrate_existing_sandbox_permissions(workspace_fd, flags)
    finally:
        os.close(workspace_fd)

    assert stat.S_IMODE(nested.stat().st_mode) & 0o777 == 0o770
    assert stat.S_IMODE(report.stat().st_mode) == 0o660
    assert 0o2770 in requested_modes
    assert stat.S_IMODE(outside.stat().st_mode) == 0o600
    assert (workspace.workspace_dir / ".xagent-permissions-v2").is_file()


@pytest.mark.skipif(
    not hasattr(os, "O_PATH") or os.geteuid() == 0,
    reason="Linux O_PATH and an unprivileged user are required",
)
def test_prepare_parent_traverses_search_only_component(tmp_path, monkeypatch):
    uploads_root = tmp_path / "uploads"
    search_only = uploads_root / "search-only"
    workspace_parent = search_only / "actor"
    workspace_parent.mkdir(parents=True)
    search_only.chmod(0o111)
    monkeypatch.setattr(workspace_module, "get_uploads_dir", lambda: uploads_root)
    subject = object.__new__(TaskWorkspace)
    subject.workspace_dir = workspace_parent / "task"

    try:
        assert subject._prepare_sandbox_workspace_parent() is False
    finally:
        search_only.chmod(0o755)


@pytest.mark.skipif(
    not hasattr(os, "O_PATH") or os.geteuid() == 0,
    reason="Linux O_PATH and an unprivileged user are required",
)
def test_prepare_parent_falls_back_when_search_only_component_cannot_reopen(
    tmp_path, monkeypatch
):
    uploads_root = tmp_path / "uploads"
    search_only = uploads_root / "search-only"
    workspace_parent = search_only / "actor"
    workspace_parent.mkdir(parents=True)
    uploads_inode = uploads_root.stat().st_ino
    search_only.chmod(0o111)
    real_fstat = os.fstat

    def prepared_uploads_fstat(fd: int):
        result = real_fstat(fd)
        if result.st_ino == uploads_inode:
            values = list(result)
            values[5] = 1010
            values[0] |= stat.S_ISGID
            return os.stat_result(values)
        return result

    monkeypatch.setattr(workspace_module, "get_uploads_dir", lambda: uploads_root)
    monkeypatch.setattr(workspace_module.os, "fstat", prepared_uploads_fstat)
    subject = object.__new__(TaskWorkspace)
    subject.workspace_dir = workspace_parent / "task"

    try:
        assert subject._prepare_sandbox_workspace_parent() is False
    finally:
        search_only.chmod(0o755)


@pytest.mark.skipif(os.geteuid() != 0, reason="requires real chown privileges")
def test_permission_migration_assigns_real_sandbox_gid(tmp_path):
    workspace = TaskWorkspace("privileged", str(tmp_path))
    legacy = workspace.output_dir / "legacy.txt"
    legacy.write_text("legacy", encoding="utf-8")
    os.chown(legacy, 0, 0)
    legacy.chmod(0o600)
    marker = workspace.workspace_dir / ".xagent-permissions-v2"
    marker.unlink(missing_ok=True)
    flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)

    workspace_fd = os.open(workspace.workspace_dir, flags)
    try:
        workspace._migrate_existing_sandbox_permissions(workspace_fd, flags, force=True)
    finally:
        os.close(workspace_fd)

    legacy_stat = legacy.stat()
    assert legacy_stat.st_gid == 1010
    assert stat.S_IMODE(legacy_stat.st_mode) == 0o660


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO not available")
def test_permission_migration_marker_fifo_never_blocks(workspace, monkeypatch):
    marker = workspace.workspace_dir / ".xagent-permissions-v2"
    os.mkfifo(marker)
    monkeypatch.setattr(workspace_module.os, "fchown", lambda *_args: None)
    monkeypatch.setattr(workspace_module.os, "fchmod", lambda *_args: None)
    outcome: list[BaseException | None] = []
    flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)

    def migrate() -> None:
        workspace_fd = os.open(workspace.workspace_dir, flags)
        try:
            workspace._migrate_existing_sandbox_permissions(workspace_fd, flags)
        except BaseException as exc:
            outcome.append(exc)
        else:
            outcome.append(None)
        finally:
            os.close(workspace_fd)

    worker = threading.Thread(target=migrate, daemon=True)
    worker.start()
    worker.join(timeout=2)
    try:
        assert not worker.is_alive(), "opening a marker FIFO must not block"
    finally:
        if worker.is_alive():
            writer_fd = os.open(marker, os.O_WRONLY | os.O_NONBLOCK)
            os.close(writer_fd)
            worker.join(timeout=2)
    assert outcome == [None]
    assert marker.is_file()


def test_permission_migration_marker_directory_does_not_break_workspace(
    workspace, monkeypatch
):
    marker = workspace.workspace_dir / ".xagent-permissions-v2"
    marker.mkdir()
    monkeypatch.setattr(workspace_module.os, "fchown", lambda *_args: None)
    flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)

    workspace_fd = os.open(workspace.workspace_dir, flags)
    try:
        workspace._migrate_existing_sandbox_permissions(workspace_fd, flags)
    finally:
        os.close(workspace_fd)

    assert marker.is_dir()


def test_permission_migration_treats_vanished_entry_as_complete(workspace, monkeypatch):
    vanishing = workspace.output_dir / "vanishing.txt"
    vanishing.write_text("temporary", encoding="utf-8")
    real_stat = os.stat

    def remove_before_stat(path, *args, dir_fd=None, **kwargs):
        if path == vanishing.name and dir_fd is not None:
            os.unlink(path, dir_fd=dir_fd)
            raise FileNotFoundError(errno.ENOENT, "entry vanished", path)
        return real_stat(path, *args, dir_fd=dir_fd, **kwargs)

    monkeypatch.setattr(workspace_module.os, "stat", remove_before_stat)
    flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
    output_fd = os.open(workspace.output_dir, flags)
    try:
        assert workspace._migrate_sandbox_tree(output_fd)
    finally:
        os.close(output_fd)


def test_failed_registration_validation_does_not_mutate_permissions(
    workspace, monkeypatch
):
    output = workspace.output_dir / "invalid.txt"
    output.write_text("invalid", encoding="utf-8")
    grants: list[WorkspaceFileRegistration] = []
    monkeypatch.setattr(
        workspace,
        "_grant_sandbox_access_best_effort",
        grants.append,
    )
    monkeypatch.setattr(
        workspace,
        "_load_file_registration_plans",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("invalid owner")),
    )

    with pytest.raises(ValueError, match="invalid owner"):
        workspace._register_files_locked(((str(output), None),), db_session=None)

    assert grants == []


def test_grant_sandbox_access_is_best_effort(workspace, monkeypatch):
    output = workspace.output_dir / "host-created.txt"
    output.write_text("host", encoding="utf-8")
    monkeypatch.setattr(
        workspace,
        "_grant_registered_path_sandbox_access",
        lambda _registration: (_ for _ in ()).throw(PermissionError("denied")),
    )

    workspace.grant_sandbox_access(output)


def test_prepared_directory_outside_upload_root_uses_legacy_layout(
    tmp_path, monkeypatch
):
    configured_uploads = tmp_path / "configured-uploads"
    configured_uploads.mkdir()
    external_base = tmp_path / "external"
    external_base.mkdir()
    real_stat = Path.stat

    def accidentally_prepared_stat(path: Path, *args, **kwargs):
        result = real_stat(path, *args, **kwargs)
        if path == external_base:
            values = list(result)
            values[5] = 1010
            values[0] |= stat.S_ISGID
            return os.stat_result(values)
        return result

    chowns: list[tuple[int, int]] = []
    monkeypatch.setattr(workspace_module, "get_uploads_dir", lambda: configured_uploads)
    monkeypatch.setattr(Path, "stat", accidentally_prepared_stat)
    monkeypatch.setattr(
        workspace_module.os,
        "fchown",
        lambda _fd, uid, gid: chowns.append((uid, gid)),
    )

    workspace = TaskWorkspace("external-task", str(external_base))

    assert chowns == []
    assert stat.S_IMODE(real_stat(workspace.workspace_dir).st_mode) == 0o755


def test_registered_output_preserves_creating_uid(workspace, monkeypatch):
    output = workspace.output_dir / "sandbox.txt"
    output.write_text("result", encoding="utf-8")
    registration = WorkspaceFileRegistration(
        path=output.resolve(),
        relative_path="output/sandbox.txt",
        category="output",
        is_workspace_file=True,
    )
    real_stat = Path.stat

    def prepared_root_stat(path: Path, *args, **kwargs):
        result = real_stat(path, *args, **kwargs)
        if path == workspace.workspace_dir:
            values = list(result)
            values[5] = 1010
            values[0] |= stat.S_ISGID
            return os.stat_result(values)
        return result

    chowns: list[tuple[int, int]] = []
    monkeypatch.setattr(Path, "stat", prepared_root_stat)
    monkeypatch.setattr(
        workspace_module.os,
        "fchown",
        lambda _fd, uid, gid: chowns.append((uid, gid)),
    )
    monkeypatch.setattr(workspace_module.os, "fchmod", lambda *_args: None)

    workspace._grant_registered_path_sandbox_access(registration)

    assert chowns[-1] == (-1, 1010)


def test_bound_durable_input_is_made_readable_by_sandbox(workspace, monkeypatch):
    durable_input = workspace.input_dir / "restored.txt"
    durable_input.write_text("restored", encoding="utf-8")
    durable_input.chmod(0o600)
    registration = WorkspaceFileRegistration(
        path=durable_input.resolve(),
        relative_path="input/restored.txt",
        category="input",
        is_workspace_file=True,
    )
    real_stat = Path.stat

    def prepared_root_stat(path: Path, *args, **kwargs):
        result = real_stat(path, *args, **kwargs)
        if path == workspace.workspace_dir:
            values = list(result)
            values[5] = 1010
            values[0] |= stat.S_ISGID
            return os.stat_result(values)
        return result

    chowns: list[tuple[int, int]] = []
    monkeypatch.setattr(Path, "stat", prepared_root_stat)
    monkeypatch.setattr(
        workspace_module.os,
        "fchown",
        lambda _fd, uid, gid: chowns.append((uid, gid)),
    )

    workspace.bind_already_durable_file(registration, file_id="restored-file")

    assert stat.S_IMODE(durable_input.stat().st_mode) == 0o660
    assert chowns[-1] == (-1, 1010)


def test_permission_marker_is_hidden_from_workspace_listing(workspace):
    marker = workspace.workspace_dir / ".xagent-permissions-v2"
    marker.write_text("", encoding="utf-8")

    listed = workspace.get_all_files()

    assert all(
        item["filename"] != marker.name
        for entries in listed.values()
        for item in entries
    )


def test_permission_migration_retries_after_one_entry_fails(workspace, monkeypatch):
    blocked = workspace.output_dir / "blocked.txt"
    blocked.write_text("blocked", encoding="utf-8")
    healthy = workspace.output_dir / "healthy.txt"
    healthy.write_text("healthy", encoding="utf-8")
    blocked_inode = blocked.stat().st_ino
    real_fchmod = os.fchmod

    def selective_fchmod(fd: int, mode: int) -> None:
        if os.fstat(fd).st_ino == blocked_inode:
            raise PermissionError("legacy inode is restricted")
        real_fchmod(fd, mode)

    monkeypatch.setattr(workspace_module.os, "fchown", lambda *_args: None)
    monkeypatch.setattr(workspace_module.os, "fchmod", selective_fchmod)
    flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
    workspace_fd = os.open(workspace.workspace_dir, flags)
    try:
        workspace._migrate_existing_sandbox_permissions(workspace_fd, flags)
    finally:
        os.close(workspace_fd)

    assert stat.S_IMODE(healthy.stat().st_mode) == 0o660
    assert not (workspace.workspace_dir / ".xagent-permissions-v2").exists()

    monkeypatch.setattr(workspace_module.os, "fchmod", real_fchmod)
    workspace_fd = os.open(workspace.workspace_dir, flags)
    try:
        workspace._migrate_existing_sandbox_permissions(workspace_fd, flags)
    finally:
        os.close(workspace_fd)

    assert stat.S_IMODE(blocked.stat().st_mode) == 0o660
    assert (workspace.workspace_dir / ".xagent-permissions-v2").is_file()


def test_workspace_recovers_replaced_internal_temp_entry(tmp_path, monkeypatch):
    base_dir = tmp_path / "prepared"
    base_dir.mkdir()
    real_stat = Path.stat

    def prepared_parent_stat(path: Path, *args, **kwargs):
        result = real_stat(path, *args, **kwargs)
        if path == base_dir:
            values = list(result)
            values[5] = 1010
            values[0] |= stat.S_ISGID
            return os.stat_result(values)
        return result

    monkeypatch.setattr(Path, "stat", prepared_parent_stat)
    monkeypatch.setattr(workspace_module, "get_uploads_dir", lambda: tmp_path)
    monkeypatch.setattr(workspace_module.os, "fchown", lambda *_args: None)

    workspace = TaskWorkspace("task-internal", str(base_dir))
    workspace.internal_temp_dir.rmdir()
    workspace.internal_temp_dir.write_text("sandbox replacement", encoding="utf-8")

    rebuilt = TaskWorkspace(
        "task-internal", str(base_dir), migrate_sandbox_permissions=False
    )

    assert rebuilt.internal_temp_dir.is_dir()


def test_workspace_retries_internal_temp_directory_enoent(tmp_path, monkeypatch):
    base_dir = tmp_path / "prepared-enoent"
    base_dir.mkdir()
    real_stat = Path.stat
    real_open = os.open
    attempts = 0

    def prepared_parent_stat(path: Path, *args, **kwargs):
        result = real_stat(path, *args, **kwargs)
        if path == base_dir:
            values = list(result)
            values[5] = 1010
            values[0] |= stat.S_ISGID
            return os.stat_result(values)
        return result

    def disappear_once(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal attempts
        if path == ".xagent-internal" and dir_fd is not None and attempts == 0:
            attempts += 1
            raise FileNotFoundError(errno.ENOENT, "entry disappeared", path)
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(Path, "stat", prepared_parent_stat)
    monkeypatch.setattr(workspace_module, "get_uploads_dir", lambda: tmp_path)
    monkeypatch.setattr(workspace_module.os, "fchown", lambda *_args: None)
    monkeypatch.setattr(workspace_module.os, "open", disappear_once)

    rebuilt = TaskWorkspace(
        "task-internal-enoent", str(base_dir), migrate_sandbox_permissions=False
    )

    assert attempts == 1
    assert rebuilt.internal_temp_dir.is_dir()


def test_workspace_tolerates_repeated_internal_temp_enoent(tmp_path, monkeypatch):
    base_dir = tmp_path / "prepared-racing-enoent"
    base_dir.mkdir()
    real_stat = Path.stat
    real_open = os.open
    attempts = 0

    def prepared_parent_stat(path: Path, *args, **kwargs):
        result = real_stat(path, *args, **kwargs)
        if path == base_dir:
            values = list(result)
            values[5] = 1010
            values[0] |= stat.S_ISGID
            return os.stat_result(values)
        return result

    def keep_disappearing(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal attempts
        if path == ".xagent-internal" and dir_fd is not None:
            attempts += 1
            raise FileNotFoundError(errno.ENOENT, "entry disappeared", path)
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(Path, "stat", prepared_parent_stat)
    monkeypatch.setattr(workspace_module, "get_uploads_dir", lambda: tmp_path)
    monkeypatch.setattr(workspace_module.os, "fchown", lambda *_args: None)
    monkeypatch.setattr(workspace_module.os, "open", keep_disappearing)

    workspace = TaskWorkspace(
        "task-internal-racing", str(base_dir), migrate_sandbox_permissions=False
    )

    assert attempts == 3
    assert workspace.workspace_dir.is_dir()


def test_workspace_migration_is_attempted_once_per_process(tmp_path, monkeypatch):
    base_dir = tmp_path / "prepared-once"
    base_dir.mkdir()
    real_stat = Path.stat

    def prepared_parent_stat(path: Path, *args, **kwargs):
        result = real_stat(path, *args, **kwargs)
        if path == base_dir:
            values = list(result)
            values[5] = 1010
            values[0] |= stat.S_ISGID
            return os.stat_result(values)
        return result

    attempts: list[str] = []
    monkeypatch.setattr(Path, "stat", prepared_parent_stat)
    monkeypatch.setattr(workspace_module, "get_uploads_dir", lambda: tmp_path)
    monkeypatch.setattr(workspace_module.os, "fchown", lambda *_args: None)
    monkeypatch.setattr(
        TaskWorkspace,
        "_migrate_existing_sandbox_permissions",
        lambda self, *_args, **_kwargs: attempts.append(str(self.workspace_dir)),
    )

    TaskWorkspace("task-once", str(base_dir))
    TaskWorkspace("task-once", str(base_dir))

    assert attempts == [str(base_dir / "task-once")]


def test_permission_migration_does_not_retain_pending_directory_fds(
    workspace, monkeypatch
):
    for index in range(20):
        (workspace.output_dir / f"directory-{index}").mkdir()

    real_open = os.open
    real_dup = os.dup
    real_close = os.close
    owned_fds: set[int] = set()
    peak_owned = 0

    flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
    output_fd = real_open(workspace.output_dir, flags)

    def tracked_open(path, open_flags, mode=0o777, *, dir_fd=None):
        nonlocal peak_owned
        opened = real_open(path, open_flags, mode, dir_fd=dir_fd)
        owned_fds.add(opened)
        peak_owned = max(peak_owned, len(owned_fds))
        if peak_owned > 2:
            raise OSError(errno.EMFILE, "too many retained directory fds")
        return opened

    def bounded_dup(fd: int) -> int:
        nonlocal peak_owned
        duplicated = real_dup(fd)
        owned_fds.add(duplicated)
        peak_owned = max(peak_owned, len(owned_fds))
        if peak_owned > 2:
            raise OSError(errno.EMFILE, "too many retained directory fds")
        return duplicated

    def tracked_close(fd: int) -> None:
        owned_fds.discard(fd)
        real_close(fd)

    monkeypatch.setattr(workspace_module.os, "open", tracked_open)
    monkeypatch.setattr(workspace_module.os, "dup", bounded_dup)
    monkeypatch.setattr(workspace_module.os, "close", tracked_close)
    monkeypatch.setattr(workspace_module.os, "fchown", lambda *_args: None)
    try:
        assert workspace._migrate_sandbox_tree(output_fd)
    finally:
        real_close(output_fd)

    assert peak_owned <= 2
    assert not owned_fds


def test_permission_migration_skips_hardlinked_files(workspace, monkeypatch):
    source = workspace.output_dir / "source.txt"
    source.write_text("shared", encoding="utf-8")
    linked = workspace.output_dir / "linked.txt"
    os.link(source, linked)
    source.chmod(0o600)
    monkeypatch.setattr(workspace_module.os, "fchown", lambda *_args: None)

    flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
    workspace_fd = os.open(workspace.workspace_dir, flags)
    try:
        workspace._migrate_existing_sandbox_permissions(workspace_fd, flags)
    finally:
        os.close(workspace_fd)

    assert stat.S_IMODE(source.stat().st_mode) == 0o600
