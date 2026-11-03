from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path

import pytest

import xagent.sandbox.workspace_transfer as workspace_transfer_module
from xagent.sandbox.base import ExecResult, SandboxContractError
from xagent.sandbox.workspace_transfer import (
    _MANIFEST_SCRIPT,
    _RESET_SCRIPT,
    WorkspaceTransfer,
    WorkspaceTransferRoot,
    _parse_remote_manifest,
    _scan_local_tree,
    finish_workspace_transfer,
)


class _LocalTransferSandbox:
    def __init__(self, logical_root: Path, guest_root: Path) -> None:
        self.logical_root = logical_root
        self.guest_root = guest_root

    def _guest(self, remote: str) -> Path:
        path = Path(remote)
        return self.guest_root / path.relative_to(self.logical_root)

    async def prepare_workspace_roots(self, paths: list[str]) -> None:
        for path in paths:
            self._guest(path).mkdir(parents=True, exist_ok=True)

    async def exec(
        self,
        command: str,
        *args: str,
        env: dict[str, str] | None = None,
        max_output_bytes: int | None = None,
    ) -> ExecResult:
        del env, max_output_bytes
        if command == "python" and args[:4] == ("-I", "-S", "-c", _RESET_SCRIPT):
            for root in json.loads(args[4]):
                guest = self._guest(root)
                shutil.rmtree(guest, ignore_errors=True)
                guest.mkdir(parents=True)
            return ExecResult(exit_code=0, stdout="", stderr="")
        if command == "python" and args[:4] == (
            "-I",
            "-S",
            "-c",
            _MANIFEST_SCRIPT,
        ):
            snapshot = _scan_local_tree(self._guest(args[4]))
            entries = [
                {"path": str(path), "type": "dir", "mode": mode}
                for path, mode in snapshot.directories
            ]
            entries.extend(
                {
                    "path": str(item.relative),
                    "type": "file",
                    "mode": item.mode,
                    "size": item.size,
                }
                for item in snapshot.files
            )
            return ExecResult(
                exit_code=0,
                stdout=json.dumps({"entries": entries, "total": snapshot.total_bytes}),
                stderr="",
            )
        if command == "mkdir":
            self._guest(args[-1]).mkdir(parents=True, exist_ok=True)
            return ExecResult(exit_code=0, stdout="", stderr="")
        if command == "chmod":
            self._guest(args[-1]).chmod(int(args[0], 8))
            return ExecResult(exit_code=0, stdout="", stderr="")
        raise AssertionError((command, args))

    async def upload_file(
        self, local_path: str, remote_path: str, overwrite: bool = False
    ) -> None:
        target = self._guest(remote_path)
        assert overwrite is False and not target.exists()
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
        assert overwrite is False and not target.exists()
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self._guest(remote_path), target)
        if max_bytes is not None and target.stat().st_size > max_bytes:
            target.unlink()
            raise SandboxContractError("Sandbox download exceeds its byte limit")


def test_local_scan_rejects_links_and_special_entries(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "target").write_text("x")
    (root / "link").symlink_to(root / "target")

    with pytest.raises(SandboxContractError, match="unsupported"):
        _scan_local_tree(root)


@pytest.mark.parametrize("unsafe", ["", ".", "..", "../escape", "/absolute"])
def test_remote_manifest_rejects_unsafe_paths(unsafe: str) -> None:
    payload = json.dumps(
        {
            "entries": [{"path": unsafe, "type": "file", "mode": 0o600, "size": 1}],
            "total": 1,
        }
    )
    with pytest.raises(SandboxContractError, match="unsafe path"):
        _parse_remote_manifest(payload)


@pytest.mark.asyncio
async def test_transfer_round_trip_replaces_writable_tree_atomically(
    tmp_path: Path,
) -> None:
    host = tmp_path / "host" / "workspace"
    host.mkdir(parents=True)
    (host / "input.txt").write_text("input")
    (host / "removed.txt").write_text("old")
    guest = tmp_path / "guest"
    sandbox = _LocalTransferSandbox(host, guest)
    transfer = WorkspaceTransfer([WorkspaceTransferRoot(host, writable=True)])

    await transfer.checkout(sandbox)  # type: ignore[arg-type]
    assert (guest / "input.txt").read_text() == "input"
    (guest / "removed.txt").unlink()
    (guest / "output").mkdir()
    (guest / "output" / "result.txt").write_text("result")

    await transfer.check_in(sandbox)  # type: ignore[arg-type]

    assert not (host / "removed.txt").exists()
    assert (host / "output" / "result.txt").read_text() == "result"
    assert not list(host.parent.glob(".workspace.sandbox-transfer-*"))


@pytest.mark.asyncio
async def test_check_in_rejects_and_preserves_concurrent_backend_write(
    tmp_path: Path,
) -> None:
    host = tmp_path / "host" / "workspace"
    host.mkdir(parents=True)
    (host / "input.txt").write_text("input")
    guest = tmp_path / "guest"
    sandbox = _LocalTransferSandbox(host, guest)
    transfer = WorkspaceTransfer([WorkspaceTransferRoot(host, writable=True)])

    await transfer.checkout(sandbox)  # type: ignore[arg-type]
    (host / "uploaded-after-checkout.txt").write_text("independent")
    (guest / "output.txt").write_text("guest")

    with pytest.raises(SandboxContractError, match="changed after checkout"):
        await transfer.check_in(sandbox)  # type: ignore[arg-type]

    assert (host / "uploaded-after-checkout.txt").read_text() == "independent"
    assert not (host / "output.txt").exists()


@pytest.mark.asyncio
async def test_check_in_rejects_open_host_writer_before_old_tree_is_removed(
    tmp_path: Path,
) -> None:
    host = tmp_path / "host" / "workspace"
    host.mkdir(parents=True)
    target = host / "shared.txt"
    target.write_text("before")
    guest = tmp_path / "guest"
    sandbox = _LocalTransferSandbox(host, guest)
    transfer = WorkspaceTransfer([WorkspaceTransferRoot(host, writable=True)])

    await transfer.checkout(sandbox)  # type: ignore[arg-type]
    (guest / "guest-output.txt").write_text("guest")
    with target.open("a") as writer:
        with pytest.raises(SandboxContractError, match="open writer"):
            await transfer.check_in(sandbox)  # type: ignore[arg-type]
        writer.write("-host")

    assert target.read_text() == "before-host"
    assert not (host / "guest-output.txt").exists()


@pytest.mark.asyncio
async def test_checkout_does_not_adopt_write_after_snapshot(tmp_path: Path) -> None:
    host = tmp_path / "host" / "workspace"
    host.mkdir(parents=True)
    (host / "input.txt").write_text("input")
    guest = tmp_path / "guest"

    class _WriteAfterUploadSandbox(_LocalTransferSandbox):
        async def upload_file(self, *args, **kwargs) -> None:
            await super().upload_file(*args, **kwargs)
            (host / "late-upload.txt").write_text("independent")

    transfer = WorkspaceTransfer([WorkspaceTransferRoot(host, writable=True)])

    with pytest.raises(SandboxContractError, match="changed during checkout"):
        await transfer.checkout(  # type: ignore[arg-type]
            _WriteAfterUploadSandbox(host, guest)
        )

    assert (host / "late-upload.txt").read_text() == "independent"


@pytest.mark.asyncio
async def test_internal_helper_timeout_cancels_owned_execution(
    tmp_path: Path, monkeypatch
) -> None:
    host = tmp_path / "host" / "workspace"
    host.mkdir(parents=True)
    guest = tmp_path / "guest"
    cancelled = asyncio.Event()

    class _HangingHelperSandbox(_LocalTransferSandbox):
        async def exec(self, command: str, *args: str, **kwargs) -> ExecResult:
            if command == "python" and args[:3] == ("-I", "-S", "-c"):
                try:
                    await asyncio.Event().wait()
                finally:
                    cancelled.set()
            return await super().exec(command, *args, **kwargs)

    monkeypatch.setattr(workspace_transfer_module, "_HELPER_TIMEOUT_SECONDS", 0.01)
    transfer = WorkspaceTransfer([WorkspaceTransferRoot(host, writable=True)])

    with pytest.raises(SandboxContractError, match="helper timed out"):
        await transfer.checkout(  # type: ignore[arg-type]
            _HangingHelperSandbox(host, guest)
        )

    assert cancelled.is_set()


@pytest.mark.asyncio
async def test_check_in_preserves_published_root_permissions(tmp_path: Path) -> None:
    host = tmp_path / "host" / "workspace"
    host.mkdir(parents=True)
    host.chmod(0o751)
    guest = tmp_path / "guest"
    sandbox = _LocalTransferSandbox(host, guest)
    transfer = WorkspaceTransfer([WorkspaceTransferRoot(host, writable=True)])

    await transfer.checkout(sandbox)  # type: ignore[arg-type]
    (guest / "output.txt").write_text("guest")
    await transfer.check_in(sandbox)  # type: ignore[arg-type]

    assert host.stat().st_mode & 0o777 == 0o751


@pytest.mark.asyncio
async def test_readonly_ancestor_supplies_inputs_but_only_child_is_published(
    tmp_path: Path,
) -> None:
    ancestor = tmp_path / "host" / "user"
    writable = ancestor / "project"
    writable.mkdir(parents=True)
    (ancestor / "sibling-upload.txt").write_text("sibling")
    (writable / "input.txt").write_text("input")
    guest = tmp_path / "guest"
    sandbox = _LocalTransferSandbox(ancestor, guest)
    transfer = WorkspaceTransfer(
        [
            WorkspaceTransferRoot(ancestor, writable=False),
            WorkspaceTransferRoot(writable, writable=True),
        ]
    )

    await transfer.checkout(sandbox)  # type: ignore[arg-type]
    assert (guest / "sibling-upload.txt").read_text() == "sibling"
    (guest / "sibling-upload.txt").write_text("must-not-publish")
    (guest / "project" / "output.txt").write_text("output")
    await transfer.check_in(sandbox)  # type: ignore[arg-type]

    assert (ancestor / "sibling-upload.txt").read_text() == "sibling"
    assert (writable / "output.txt").read_text() == "output"


@pytest.mark.asyncio
async def test_check_in_bounds_file_growth_after_manifest(tmp_path: Path) -> None:
    host = tmp_path / "host" / "workspace"
    host.mkdir(parents=True)
    guest = tmp_path / "guest"

    class _GrowingSandbox(_LocalTransferSandbox):
        async def download_file(self, *args, **kwargs) -> None:
            self._guest(args[0]).write_bytes(b"grown")
            await super().download_file(*args, **kwargs)

    sandbox = _GrowingSandbox(host, guest)
    transfer = WorkspaceTransfer([WorkspaceTransferRoot(host, writable=True)])
    await transfer.checkout(sandbox)  # type: ignore[arg-type]
    (guest / "output.txt").write_bytes(b"x")

    with pytest.raises(SandboxContractError, match="byte limit"):
        await transfer.check_in(sandbox)  # type: ignore[arg-type]

    assert not (host / "output.txt").exists()


@pytest.mark.asyncio
async def test_read_only_transfer_root_is_not_published(tmp_path: Path) -> None:
    host = tmp_path / "host" / "external"
    host.mkdir(parents=True)
    (host / "source.txt").write_text("source")
    guest = tmp_path / "guest"
    sandbox = _LocalTransferSandbox(host, guest)
    transfer = WorkspaceTransfer([WorkspaceTransferRoot(host, writable=False)])

    await transfer.checkout(sandbox)  # type: ignore[arg-type]
    (guest / "source.txt").write_text("changed")
    await transfer.check_in(sandbox)  # type: ignore[arg-type]

    assert (host / "source.txt").read_text() == "source"


@pytest.mark.asyncio
async def test_checkout_recovers_interrupted_replace(tmp_path: Path) -> None:
    host = tmp_path / "host" / "workspace"
    host.mkdir(parents=True)
    (host / "kept.txt").write_text("kept")
    backup = host.parent / ".workspace.sandbox-transfer-backup"
    host.rename(backup)
    guest = tmp_path / "guest"
    transfer = WorkspaceTransfer([WorkspaceTransferRoot(host, writable=True)])

    await transfer.checkout(  # type: ignore[arg-type]
        _LocalTransferSandbox(host, guest)
    )

    assert not backup.exists()
    assert (host / "kept.txt").read_text() == "kept"
    assert (guest / "kept.txt").read_text() == "kept"


@pytest.mark.asyncio
async def test_cancellation_waits_for_owned_transfer_to_settle() -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    settled = asyncio.Event()

    async def operation() -> None:
        started.set()
        await release.wait()
        settled.set()

    waiter = asyncio.create_task(finish_workspace_transfer(operation()))
    await started.wait()
    waiter.cancel()
    await asyncio.sleep(0)
    assert not waiter.done()

    release.set()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    assert settled.is_set()
