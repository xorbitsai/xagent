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


def test_registered_output_is_transferred_to_backend_uid(workspace, monkeypatch):
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

    assert chowns[-1] == (os.geteuid(), 1010)


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
