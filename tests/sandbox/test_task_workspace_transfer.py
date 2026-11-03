from __future__ import annotations

import hashlib
import json
import os
import pickle
import shutil
from pathlib import Path

import pytest

from xagent.sandbox.base import ExecResult, SandboxContractError
from xagent.sandbox.task_workspace_transfer import (
    _MANIFEST_SCRIPT,
    _RESET_SCRIPT,
    ExplicitInput,
    TaskWorkspaceTransfer,
    build_call_plan,
    map_guest_result,
    parse_guest_manifest,
    scan_host_tree,
)


class _LocalSandbox:
    def __init__(self, logical_root: Path, local_root: Path) -> None:
        self.logical_root = logical_root
        self.local_root = local_root

    def _local(self, remote: str | Path) -> Path:
        remote_path = Path(remote)
        if remote_path.is_relative_to(self.logical_root):
            return self.local_root / remote_path.relative_to(self.logical_root)
        return self.local_root.parent / "inputs" / remote_path.name

    async def exec(self, command: str, *args: str, **kwargs) -> ExecResult:
        del kwargs
        if command == "python" and args[:4] == ("-I", "-S", "-c", _RESET_SCRIPT):
            root = self._local(args[4])
            if root.exists():
                for current, directories, _files in os.walk(root):
                    Path(current).chmod(0o700)
                    for name in directories:
                        (Path(current) / name).chmod(0o700)
            shutil.rmtree(root, ignore_errors=True)
            root.mkdir(parents=True)
            return ExecResult(exit_code=0, stdout="", stderr="")
        if command == "python" and args[:4] == (
            "-I",
            "-S",
            "-c",
            _MANIFEST_SCRIPT,
        ):
            state = scan_host_tree(self._local(args[4]))
            entries = [
                {"path": str(path), "type": "dir", "mode": mode}
                for path, mode in state.directories
            ]
            entries.extend(
                {
                    "path": str(item.relative),
                    "type": "file",
                    "mode": item.mode,
                    "size": item.size,
                    "sha256": item.sha256,
                }
                for item in state.files
            )
            return ExecResult(
                exit_code=0,
                stdout=json.dumps({"entries": entries, "total": state.total_bytes}),
                stderr="",
            )
        if command == "mkdir":
            self._local(args[-1]).mkdir(parents=True, exist_ok=True)
        elif command == "chmod":
            self._local(args[-1]).chmod(int(args[0], 8))
        elif command == "rm":
            target = self._local(args[-1])
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink(missing_ok=True)
        else:
            raise AssertionError((command, args))
        return ExecResult(exit_code=0, stdout="", stderr="")

    async def upload_file(
        self, local_path: str, remote_path: str, overwrite: bool = False
    ) -> None:
        target = self._local(remote_path)
        if target.exists() and not overwrite:
            raise FileExistsError(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local_path, target)

    async def download_file(
        self,
        remote_path: str,
        local_path: str,
        overwrite: bool = False,
        *,
        max_bytes: int | None = None,
    ) -> None:
        target = Path(local_path)
        if target.exists() and not overwrite:
            raise FileExistsError(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        source = self._local(remote_path)
        if max_bytes is not None and source.stat().st_size > max_bytes:
            raise SandboxContractError("too large")
        shutil.copyfile(source, target)

    async def download_file_bounded(
        self,
        remote_path: str,
        local_path: str,
        *,
        max_bytes: int,
    ) -> None:
        await self.download_file(
            remote_path, local_path, overwrite=False, max_bytes=max_bytes
        )


def test_scan_rejects_symlinks(tmp_path: Path) -> None:
    root = tmp_path / "task"
    root.mkdir()
    (root / "target").write_text("safe")
    (root / "link").symlink_to(root / "target")
    with pytest.raises(SandboxContractError, match="unsupported"):
        scan_host_tree(root)

    linked_root = tmp_path / "linked-root"
    linked_root.symlink_to(root, target_is_directory=True)
    with pytest.raises(SandboxContractError, match="Unsafe"):
        scan_host_tree(linked_root)


def test_manifest_rejects_unsafe_and_duplicate_paths() -> None:
    unsafe = json.dumps(
        {
            "entries": [
                {
                    "path": "../escape",
                    "type": "file",
                    "mode": 0o600,
                    "size": 0,
                    "sha256": "0" * 64,
                }
            ],
            "total": 0,
        }
    )
    with pytest.raises(SandboxContractError, match="unsafe"):
        parse_guest_manifest(unsafe)


@pytest.mark.asyncio
async def test_task_round_trip_publishes_only_changed_files(tmp_path: Path) -> None:
    host = tmp_path / "owner" / "web_task_7"
    host.mkdir(parents=True)
    (host / "input").mkdir()
    (host / "output").mkdir()
    (host / "input" / "source.txt").write_text("source")
    guest_path = Path("/home/sandbox/.xagent-workspaces/task-7")
    guest = tmp_path / "guest"
    sandbox = _LocalSandbox(guest_path, guest)
    transfer = TaskWorkspaceTransfer(host, guest_path)

    await transfer.checkout(sandbox)  # type: ignore[arg-type]
    (tmp_path / "owner" / "unrelated.txt").write_text("independent")
    (guest / "output" / "result.txt").write_text("result")
    await transfer.check_in(sandbox)  # type: ignore[arg-type]

    assert (host / "output" / "result.txt").read_text() == "result"
    assert (tmp_path / "owner" / "unrelated.txt").read_text() == "independent"


@pytest.mark.asyncio
async def test_nested_existing_upload_keeps_trusted_serving_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    host = tmp_path / "owner" / "web_task_served"
    preview = host / "uploads" / "preview"
    preview.mkdir(parents=True)
    served = preview / "document.html"
    served.write_text("before")
    (host / "uploads").chmod(0o750)
    preview.chmod(0o750)
    served.chmod(0o640)
    guest_path = Path("/home/sandbox/.xagent-workspaces/task-served")
    guest = tmp_path / "guest"
    sandbox = _LocalSandbox(guest_path, guest)
    transfer = TaskWorkspaceTransfer(host, guest_path)
    await transfer.checkout(sandbox)  # type: ignore[arg-type]
    (guest / "uploads" / "preview" / "document.html").write_text("after")

    real_stat = Path.stat

    def trusted_root_stat(
        path: Path, *, follow_symlinks: bool = True
    ) -> os.stat_result:
        metadata = real_stat(path, follow_symlinks=follow_symlinks)
        if path != host:
            return metadata
        values = list(metadata)
        values[5] = 1010
        return os.stat_result(values)

    file_chowns: list[tuple[Path, int, int, bool]] = []
    directory_chowns: list[tuple[int, int, int]] = []
    monkeypatch.setattr(Path, "stat", trusted_root_stat)
    monkeypatch.setattr(
        os,
        "chown",
        lambda path, uid, gid, *, follow_symlinks=True: file_chowns.append(
            (Path(path), uid, gid, follow_symlinks)
        ),
    )
    monkeypatch.setattr(
        os,
        "fchown",
        lambda descriptor, uid, gid: directory_chowns.append((descriptor, uid, gid)),
    )

    await transfer.check_in(sandbox)  # type: ignore[arg-type]

    assert served.read_text() == "after"
    assert any(
        path.name == "document.html" and gid == 1010 and follow is False
        for path, _uid, gid, follow in file_chowns
    )
    assert directory_chowns
    assert all(gid == 1010 for _descriptor, _uid, gid in directory_chowns)


@pytest.mark.asyncio
async def test_same_file_host_change_fails_closed(tmp_path: Path) -> None:
    host = tmp_path / "web_task_8"
    host.mkdir()
    target = host / "output.txt"
    target.write_text("before")
    guest_path = Path("/home/sandbox/.xagent-workspaces/task-8")
    guest = tmp_path / "guest"
    sandbox = _LocalSandbox(guest_path, guest)
    transfer = TaskWorkspaceTransfer(host, guest_path)

    await transfer.checkout(sandbox)  # type: ignore[arg-type]
    target.write_text("host")
    (guest / "output.txt").write_text("guest")
    with pytest.raises(SandboxContractError, match="Host changed"):
        await transfer.check_in(sandbox)  # type: ignore[arg-type]
    assert target.read_text() == "host"


@pytest.mark.asyncio
async def test_readonly_guest_modes_do_not_wedge_next_checkout(tmp_path: Path) -> None:
    host = tmp_path / "web_task_9"
    protected = host / "output" / "nested"
    protected.mkdir(parents=True)
    (protected / "value.txt").write_text("value")
    guest_path = Path("/home/sandbox/.xagent-workspaces/task-9")
    guest = tmp_path / "guest"
    sandbox = _LocalSandbox(guest_path, guest)
    transfer = TaskWorkspaceTransfer(host, guest_path)

    await transfer.checkout(sandbox)  # type: ignore[arg-type]
    (guest / "output" / "nested").chmod(0o555)
    await transfer.cleanup(sandbox, ())  # type: ignore[arg-type]
    await transfer.checkout(sandbox)  # type: ignore[arg-type]
    assert (guest / "output" / "nested" / "value.txt").read_text() == "value"


def test_call_plan_copies_only_explicit_external_files(tmp_path: Path) -> None:
    from xagent.core.workspace import TaskWorkspace

    base = tmp_path / "owner"
    external = tmp_path / "external"
    external.mkdir()
    chosen = external / "chosen.txt"
    chosen.write_text("chosen")
    (external / "not-chosen.txt").write_text("not chosen")
    workspace = TaskWorkspace(
        "web_task_10",
        str(base),
        allowed_external_dirs=[str(external)],
        migrate_sandbox_permissions=False,
    )
    guest_root = Path("/home/sandbox/.xagent-workspaces/task-10")

    plan = build_call_plan(
        workspace,
        {"source": str(chosen), "target": str(workspace.output_dir / "out.txt")},
        host_root=workspace.workspace_dir,
        guest_root=guest_root,
    )

    assert len(plan.inputs) == 1
    assert plan.inputs[0].host_path == chosen.resolve()
    assert str(guest_root) in str(plan.args["target"])
    assert str(chosen) not in str(plan.args["source"])


def test_call_plan_handles_command_strings_and_maps_results(tmp_path: Path) -> None:
    from xagent.core.workspace import TaskWorkspace

    base = tmp_path / "owner"
    external = tmp_path / "external"
    external.mkdir()
    chosen = external / "chosen file.txt"
    chosen.write_text("chosen")
    workspace = TaskWorkspace(
        "web_task_13",
        str(base),
        allowed_external_dirs=[str(external)],
        migrate_sandbox_permissions=False,
    )
    guest_root = Path("/home/sandbox/.xagent-workspaces/task-13")

    plan = build_call_plan(
        workspace,
        {"command": f'cp "{chosen}" "{workspace.output_dir / "copy.txt"}"'},
        host_root=workspace.workspace_dir,
        guest_root=guest_root,
    )

    assert len(plan.inputs) == 1
    assert str(chosen) not in str(plan.args["command"])
    assert str(guest_root / "output" / "copy.txt") in str(plan.args["command"])
    assert map_guest_result(
        {"source": str(plan.inputs[0].guest_path)},
        guest_root,
        workspace.workspace_dir,
        plan.inputs,
    ) == {"source": str(chosen)}

    with pytest.raises(SandboxContractError, match="escapes"):
        build_call_plan(
            workspace,
            {"target": str(workspace.workspace_dir / ".." / "other" / "out.txt")},
            host_root=workspace.workspace_dir,
            guest_root=guest_root,
        )


def test_pickled_workspace_projects_only_inside_tool_runner(
    tmp_path: Path, monkeypatch
) -> None:
    import xagent.core.workspace as workspace_module

    TaskWorkspace = workspace_module.TaskWorkspace

    workspace = TaskWorkspace(
        "web_task_12",
        str(tmp_path / "owner"),
        migrate_sandbox_permissions=False,
    )
    payload = pickle.dumps(workspace)
    guest = Path("/home/sandbox/.xagent-workspaces/task-12")
    input_dir = Path("/home/sandbox/.xagent-inputs/token")
    monkeypatch.setenv("XAGENT_SANDBOX_TOOL_RUNNER", "1")
    # The production marker is intentionally captured before imports so
    # agent-authored code cannot enable runner privileges at runtime. This
    # in-process test therefore replaces the captured predicate explicitly.
    monkeypatch.setattr(workspace_module, "in_sandbox_tool_runner", lambda: True)
    monkeypatch.setenv(
        "XAGENT_SANDBOX_HOST_WORKSPACE_ROOT", str(workspace.workspace_dir)
    )
    monkeypatch.setenv("XAGENT_SANDBOX_GUEST_WORKSPACE_ROOT", str(guest))
    monkeypatch.setenv("XAGENT_SANDBOX_GUEST_INPUT_DIRS", json.dumps([str(input_dir)]))

    projected = pickle.loads(payload)

    assert projected.workspace_dir == guest
    assert projected.output_dir == guest / "output"
    assert projected.allowed_external_dirs == [input_dir]


@pytest.mark.asyncio
async def test_explicit_input_is_not_published(tmp_path: Path) -> None:
    host = tmp_path / "web_task_11"
    host.mkdir()
    source = tmp_path / "shared.txt"
    source.write_text("host")
    guest_path = Path("/home/sandbox/.xagent-workspaces/task-11")
    sandbox = _LocalSandbox(guest_path, tmp_path / "guest")
    token = hashlib.sha256(str(source).encode()).hexdigest()[:24]
    staged = Path("/home/sandbox/.xagent-inputs") / token / "shared.txt"
    transfer = TaskWorkspaceTransfer(host, guest_path)
    item = ExplicitInput(
        source,
        staged,
        source.stat().st_size,
        hashlib.sha256(b"host").hexdigest(),
    )

    await transfer.checkout(sandbox, (item,))  # type: ignore[arg-type]
    sandbox._local(staged).write_text("guest")
    await transfer.check_in(sandbox)  # type: ignore[arg-type]
    assert source.read_text() == "host"
