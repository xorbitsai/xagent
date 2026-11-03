"""Serialized workspace transfer for sandboxes without writable host binds.

This module is deliberately backend-agnostic and uses only the narrow
``Sandbox`` file/exec contract.  It is enabled by ``SandboxManager`` only for
the opt-in transfer mode; the historical bind mode never imports it on its
hot path.
"""

from __future__ import annotations

import asyncio
import ctypes
import json
import os
import shutil
import stat
import sys
import tempfile
from collections.abc import Awaitable
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, TypeVar

from .base import ExecResult, Sandbox, SandboxContractError

_MAX_FILES = 20_000
# The Docker transfer workspace lives on a 256 MiB sandbox-local tmpfs. Keep
# one published tree below that storage ceiling so accepted 100 MiB uploads
# fit while metadata, read-only inputs, and generated output retain headroom.
_MAX_TOTAL_BYTES = 200 * 1024 * 1024
_MAX_MANIFEST_BYTES = 16 * 1024 * 1024
_HELPER_TIMEOUT_SECONDS = 30.0
_T = TypeVar("_T")
_HostTreeRevision = tuple[tuple[str, int, int, int, int], ...]

_RESET_SCRIPT = r"""
import json, os, shutil, stat, sys
for raw in json.loads(sys.argv[1]):
    root = os.path.abspath(raw)
    if root != raw or root == "/":
        raise SystemExit("invalid workspace root")
    try:
        mode = os.lstat(root).st_mode
    except FileNotFoundError:
        mode = None
    if mode is not None:
        if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
            if stat.S_ISDIR(mode):
                shutil.rmtree(root)
            else:
                os.unlink(root)
        else:
            for name in os.listdir(root):
                path = os.path.join(root, name)
                child_mode = os.lstat(path).st_mode
                if stat.S_ISDIR(child_mode) and not stat.S_ISLNK(child_mode):
                    shutil.rmtree(path)
                else:
                    os.unlink(path)
    os.makedirs(root, mode=0o750, exist_ok=True)
"""

_MANIFEST_SCRIPT = r"""
import json, os, stat, sys
root = os.path.abspath(sys.argv[1])
if root != sys.argv[1] or root == "/":
    raise SystemExit("invalid workspace root")
entries = []
total = 0
if os.path.lexists(root):
    root_mode = os.lstat(root).st_mode
    if stat.S_ISLNK(root_mode) or not stat.S_ISDIR(root_mode):
        raise SystemExit("workspace root is not a directory")
    for current, dirs, files in os.walk(root, topdown=True, followlinks=False):
        dirs.sort()
        files.sort()
        for name in list(dirs):
            path = os.path.join(current, name)
            mode = os.lstat(path).st_mode
            if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
                raise SystemExit("workspace contains unsupported entry")
            rel = os.path.relpath(path, root)
            entries.append({"path": rel, "type": "dir", "mode": stat.S_IMODE(mode)})
        for name in files:
            path = os.path.join(current, name)
            info = os.lstat(path)
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
                raise SystemExit("workspace contains unsupported entry")
            total += info.st_size
            rel = os.path.relpath(path, root)
            entries.append({"path": rel, "type": "file", "mode": stat.S_IMODE(info.st_mode), "size": info.st_size})
print(json.dumps({"entries": entries, "total": total}, separators=(",", ":")))
"""


@dataclass(frozen=True)
class WorkspaceTransferRoot:
    """One identical backend/guest path and whether guest changes persist."""

    path: Path
    writable: bool


@dataclass(frozen=True)
class _FileEntry:
    relative: Path
    mode: int
    size: int


@dataclass(frozen=True)
class _TreeSnapshot:
    directories: tuple[tuple[Path, int], ...]
    files: tuple[_FileEntry, ...]
    total_bytes: int


def _safe_relative(raw: str) -> Path:
    value = Path(raw)
    if not raw or value.is_absolute() or raw in {".", ".."} or ".." in value.parts:
        raise SandboxContractError("Workspace manifest contains an unsafe path")
    return value


def _enforce_limits(file_count: int, total_bytes: int) -> None:
    if file_count > _MAX_FILES:
        raise SandboxContractError(
            f"Workspace transfer exceeds the {_MAX_FILES} regular-file limit"
        )
    if total_bytes > _MAX_TOTAL_BYTES:
        raise SandboxContractError(
            f"Workspace transfer exceeds the {_MAX_TOTAL_BYTES}-byte limit"
        )


def _scan_local_tree(root: Path) -> _TreeSnapshot:
    if root == Path("/") or not root.is_absolute():
        raise SandboxContractError(f"Unsafe workspace transfer root: {root}")
    if root.is_symlink() or not root.is_dir():
        raise SandboxContractError(
            f"Workspace transfer root is not a directory: {root}"
        )

    directories: list[tuple[Path, int]] = []
    files: list[_FileEntry] = []
    total = 0
    for current, dir_names, file_names in os.walk(root, followlinks=False):
        dir_names.sort()
        file_names.sort()
        current_path = Path(current)
        for name in list(dir_names):
            path = current_path / name
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
                raise SandboxContractError(
                    f"Workspace contains an unsupported directory entry: {path}"
                )
            directories.append((path.relative_to(root), stat.S_IMODE(info.st_mode)))
        for name in file_names:
            path = current_path / name
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
                raise SandboxContractError(
                    f"Workspace contains an unsupported file entry: {path}"
                )
            total += info.st_size
            files.append(
                _FileEntry(
                    path.relative_to(root), stat.S_IMODE(info.st_mode), info.st_size
                )
            )
            _enforce_limits(len(files), total)
    return _TreeSnapshot(tuple(directories), tuple(files), total)


def _host_tree_revision(root: Path) -> _HostTreeRevision:
    """Return a metadata revision used to reject concurrent backend writes."""
    snapshot = _scan_local_tree(root)
    revision: list[tuple[str, int, int, int, int]] = []
    for relative, mode in snapshot.directories:
        info = (root / relative).lstat()
        revision.append(
            (str(relative), stat.S_IFDIR | mode, 0, info.st_mtime_ns, info.st_ctime_ns)
        )
    for entry in snapshot.files:
        info = (root / entry.relative).lstat()
        revision.append(
            (
                str(entry.relative),
                stat.S_IFREG | entry.mode,
                entry.size,
                info.st_mtime_ns,
                info.st_ctime_ns,
            )
        )
    return tuple(revision)


def _process_open_files_under(root: Path) -> tuple[Path, ...]:
    """Return backend descriptors that still reference ``root``.

    This runs only after the live root has been atomically exchanged with the
    staged tree, so no new descriptor can enter the old tree between this
    check and publication. Linux exposes
    same-uid backend workers through ``/proc``; ``/dev/fd`` is the
    current-process fallback for local development platforms.
    """
    proc = Path("/proc")
    process_directories = list(proc.glob("[0-9]*")) if proc.is_dir() else []
    opened: list[Path] = []
    for process_directory in process_directories:
        # A process can open a relative path after the exchange without holding
        # a file descriptor if its cwd is still inside the retired tree.
        try:
            cwd_target = Path(os.readlink(process_directory / "cwd"))
        except OSError:
            pass
        else:
            if cwd_target == root or cwd_target.is_relative_to(root):
                opened.append(cwd_target)

        fd_directory = process_directory / "fd"
        try:
            descriptors = tuple(fd_directory.iterdir())
        except OSError:
            continue
        for descriptor in descriptors:
            try:
                target = Path(os.readlink(descriptor))
            except OSError:
                continue
            if target.is_absolute() and (target == root or target.is_relative_to(root)):
                opened.append(target)

        # A writable mmap remains usable after its originating fd is closed.
        # Reject every mapping into the retired tree (including read-only maps;
        # the conservative false-positive is preferable to losing a late write).
        try:
            mappings = (process_directory / "maps").read_text().splitlines()
        except OSError:
            continue
        for mapping in mappings:
            fields = mapping.split(maxsplit=5)
            if len(fields) < 6 or not fields[5].startswith("/"):
                continue
            target = Path(fields[5].removesuffix(" (deleted)"))
            if target == root or target.is_relative_to(root):
                opened.append(target)

    if not proc.is_dir():
        # Darwin's /dev/fd entries are not readable symlinks. F_GETPATH gives
        # the post-rename path of each current-process descriptor instead.
        try:
            import fcntl

            try:
                cwd_target = Path.cwd()
            except OSError:
                pass
            else:
                if cwd_target == root or cwd_target.is_relative_to(root):
                    opened.append(cwd_target)

            for raw_fd in os.listdir("/dev/fd"):
                try:
                    resolved = fcntl.fcntl(int(raw_fd), 50, b"\0" * 1024)
                    target = Path(resolved.split(b"\0", 1)[0].decode())
                except (OSError, ValueError, UnicodeDecodeError):
                    continue
                if target == root or target.is_relative_to(root):
                    opened.append(target)
        except (ImportError, OSError):
            pass
    return tuple(opened)


def _exchange_paths(left: Path, right: Path) -> None:
    """Atomically exchange two same-filesystem directory entries.

    A pair of ordinary renames leaves a gap where another backend process can
    recreate ``left``.  Linux and Darwin both expose an atomic exchange
    primitive; transfer mode fails closed on platforms/filesystems that do not
    support one rather than risking a lost concurrent write.
    """
    library = ctypes.CDLL(None, use_errno=True)
    left_raw = os.fsencode(left)
    right_raw = os.fsencode(right)
    if sys.platform.startswith("linux"):
        try:
            renameat2 = library.renameat2
        except AttributeError as exc:
            raise SandboxContractError(
                "Atomic workspace publication is unavailable on this host"
            ) from exc
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        result = renameat2(-100, left_raw, -100, right_raw, 2)
    elif sys.platform == "darwin":
        try:
            renamex_np = library.renamex_np
        except AttributeError as exc:
            raise SandboxContractError(
                "Atomic workspace publication is unavailable on this host"
            ) from exc
        renamex_np.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        renamex_np.restype = ctypes.c_int
        result = renamex_np(left_raw, right_raw, 2)
    else:
        raise SandboxContractError(
            "Atomic workspace publication is unavailable on this host"
        )
    if result != 0:
        error_number = ctypes.get_errno()
        raise SandboxContractError(
            f"Atomic workspace publication failed: {os.strerror(error_number)}"
        )


async def _run_helper(
    sandbox: Sandbox,
    script: str,
    *args: str,
    max_output_bytes: int | None = None,
) -> ExecResult:
    """Run an isolated internal helper with a bounded, killable lifetime."""
    try:
        return await asyncio.wait_for(
            sandbox.exec(
                "python",
                "-I",
                "-S",
                "-c",
                script,
                *args,
                max_output_bytes=max_output_bytes,
            ),
            timeout=_HELPER_TIMEOUT_SECONDS,
        )
    except TimeoutError as exc:
        raise SandboxContractError("Sandbox workspace helper timed out") from exc


def _recover_interrupted_replace(destination: Path) -> None:
    """Restore a backup left by the pre-atomic-exchange implementation."""
    backup = destination.parent / f".{destination.name}.sandbox-transfer-backup"
    if not backup.exists() and not backup.is_symlink():
        return
    backup_info = backup.lstat()
    if stat.S_ISLNK(backup_info.st_mode) or not stat.S_ISDIR(backup_info.st_mode):
        raise SandboxContractError(f"Unsafe workspace transfer backup: {backup}")
    if destination.exists() or destination.is_symlink():
        raise SandboxContractError(
            f"Workspace transfer needs manual recovery; destination and backup exist: {destination}"
        )
    os.replace(backup, destination)


def recover_workspace_root(destination: Path) -> None:
    """Recover one interrupted publish before any caller recreates its root."""
    _recover_interrupted_replace(destination)


def _parse_remote_manifest(raw: str) -> _TreeSnapshot:
    if len(raw.encode("utf-8")) > _MAX_MANIFEST_BYTES:
        raise SandboxContractError("Sandbox workspace manifest is too large")
    try:
        payload = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as exc:
        raise SandboxContractError(
            "Sandbox returned an invalid workspace manifest"
        ) from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("entries"), list):
        raise SandboxContractError("Sandbox returned an invalid workspace manifest")

    directories: list[tuple[Path, int]] = []
    files: list[_FileEntry] = []
    total = 0
    seen: set[Path] = set()
    for item in payload["entries"]:
        if not isinstance(item, dict):
            raise SandboxContractError("Sandbox returned an invalid workspace entry")
        relative = _safe_relative(str(item.get("path", "")))
        if relative in seen:
            raise SandboxContractError("Sandbox workspace manifest contains duplicates")
        seen.add(relative)
        mode = item.get("mode")
        if not isinstance(mode, int) or mode < 0 or mode > 0o7777:
            raise SandboxContractError(
                "Sandbox workspace manifest contains an invalid mode"
            )
        safe_mode = mode & 0o755
        if item.get("type") == "dir":
            directories.append((relative, safe_mode))
        elif item.get("type") == "file":
            size = item.get("size")
            if not isinstance(size, int) or size < 0:
                raise SandboxContractError(
                    "Sandbox workspace manifest contains an invalid size"
                )
            total += size
            files.append(_FileEntry(relative, safe_mode, size))
            _enforce_limits(len(files), total)
        else:
            raise SandboxContractError(
                "Sandbox workspace manifest contains an unsupported type"
            )
    if payload.get("total") != total:
        raise SandboxContractError(
            "Sandbox workspace manifest byte total is inconsistent"
        )
    return _TreeSnapshot(tuple(directories), tuple(files), total)


def _dedupe_roots(
    roots: Sequence[WorkspaceTransferRoot],
) -> tuple[WorkspaceTransferRoot, ...]:
    ordered = sorted(roots, key=lambda item: (len(item.path.parts), str(item.path)))
    result: list[WorkspaceTransferRoot] = []
    for item in ordered:
        if not item.path.is_absolute() or item.path == Path("/"):
            raise SandboxContractError(f"Unsafe workspace transfer root: {item.path}")
        covering = next(
            (root for root in result if item.path.is_relative_to(root.path)), None
        )
        if covering is not None:
            if item.writable and not covering.writable:
                # The ancestor supplies authorized read-only inputs while the
                # nested task root is the only tree that may be published.
                result.append(item)
                continue
            if not item.writable and covering.writable:
                raise SandboxContractError(
                    "Read-only workspace root is nested under a writable transfer root"
                )
            continue
        result.append(item)
    return tuple(result)


class WorkspaceTransfer:
    """Copy workspace trees at an exclusive sandbox lease boundary."""

    def __init__(self, roots: Sequence[WorkspaceTransferRoot]) -> None:
        self._roots = _dedupe_roots(roots)
        self._checkout_roots = tuple(
            item
            for item in self._roots
            if not any(
                item.path.is_relative_to(other.path) and item.path != other.path
                for other in self._roots
            )
        )
        self._host_revisions: dict[Path, _HostTreeRevision] = {}
        self.lock = asyncio.Lock()

    @property
    def roots(self) -> tuple[WorkspaceTransferRoot, ...]:
        """Normalized roots used as part of provider cache identity."""
        return self._roots

    async def checkout(self, sandbox: Sandbox) -> None:
        """Replace guest trees with a verified snapshot of backend files."""
        snapshots = []
        for item in self._checkout_roots:
            await asyncio.to_thread(_recover_interrupted_replace, item.path)

        # Capture the writable baseline before reading any source tree.  Do not
        # re-read it after checkout: a backend write in that small gap would
        # otherwise be treated as part of the baseline even though it was never
        # copied into the guest and could then be removed at check-in.
        writable_revisions = {
            item.path: await asyncio.to_thread(_host_tree_revision, item.path)
            for item in self._roots
            if item.writable
        }
        checkout_revisions: dict[Path, _HostTreeRevision] = {}
        for item in self._checkout_roots:
            before = await asyncio.to_thread(_host_tree_revision, item.path)
            snapshots.append(await asyncio.to_thread(_scan_local_tree, item.path))
            checkout_revisions[item.path] = before
        _enforce_limits(
            sum(len(snapshot.files) for snapshot in snapshots),
            sum(snapshot.total_bytes for snapshot in snapshots),
        )
        await sandbox.prepare_workspace_roots(
            [str(item.path) for item in self._checkout_roots]
        )
        reset = await _run_helper(
            sandbox,
            _RESET_SCRIPT,
            json.dumps([str(item.path) for item in self._checkout_roots]),
        )
        if reset.exit_code != 0:
            raise SandboxContractError(
                f"Failed to reset sandbox workspace: {reset.stderr or reset.stdout}"
            )
        for item, snapshot in zip(self._checkout_roots, snapshots):
            for relative, mode in snapshot.directories:
                result = await sandbox.exec("mkdir", "-p", str(item.path / relative))
                if result.exit_code != 0:
                    raise SandboxContractError(
                        f"Failed to create sandbox workspace directory: {result.stderr}"
                    )
                await sandbox.exec(
                    "chmod", f"{mode & 0o755:o}", str(item.path / relative)
                )
            for entry in snapshot.files:
                await sandbox.upload_file(
                    str(item.path / entry.relative),
                    str(item.path / entry.relative),
                    overwrite=False,
                )
            after = await asyncio.to_thread(_host_tree_revision, item.path)
            if after != checkout_revisions[item.path]:
                raise SandboxContractError(
                    f"Backend workspace changed during checkout: {item.path}"
                )
        for path, revision in writable_revisions.items():
            if await asyncio.to_thread(_host_tree_revision, path) != revision:
                raise SandboxContractError(
                    f"Backend workspace changed during checkout: {path}"
                )
        self._host_revisions = writable_revisions

    async def check_in(self, sandbox: Sandbox) -> None:
        """Atomically publish writable guest trees back to backend storage."""
        remaining_bytes = _MAX_TOTAL_BYTES
        total_files = 0
        for item in self._roots:
            if not item.writable:
                continue
            result = await _run_helper(
                sandbox,
                _MANIFEST_SCRIPT,
                str(item.path),
                max_output_bytes=_MAX_MANIFEST_BYTES,
            )
            if result.exit_code != 0 or result.truncated:
                raise SandboxContractError(
                    f"Failed to inspect sandbox workspace: {result.stderr or result.stdout}"
                )
            snapshot = _parse_remote_manifest(result.stdout)
            total_files += len(snapshot.files)
            _enforce_limits(
                total_files,
                _MAX_TOTAL_BYTES - remaining_bytes + snapshot.total_bytes,
            )
            remaining_bytes -= snapshot.total_bytes
            staging = Path(
                tempfile.mkdtemp(
                    prefix=f".{item.path.name}.sandbox-transfer-", dir=item.path.parent
                )
            )
            published = False
            try:
                for relative, mode in snapshot.directories:
                    target = staging / relative
                    target.mkdir(parents=True, exist_ok=True)
                    target.chmod(mode & 0o755)
                for entry in snapshot.files:
                    target = staging / entry.relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    await sandbox.download_file(
                        str(item.path / entry.relative),
                        str(target),
                        overwrite=False,
                        max_bytes=entry.size,
                    )
                    if target.stat().st_size != entry.size:
                        raise SandboxContractError(
                            "Sandbox workspace file changed during transfer"
                        )
                    target.chmod(entry.mode & 0o755)
                expected = self._host_revisions.get(item.path)
                if expected is None:
                    raise SandboxContractError(
                        f"Workspace check-in has no checkout revision: {item.path}"
                    )
                await asyncio.to_thread(
                    self._replace_tree, item.path, staging, expected
                )
                published = True
            finally:
                if not published:
                    shutil.rmtree(staging, ignore_errors=True)

    @staticmethod
    def _replace_tree(
        destination: Path, staging: Path, expected: _HostTreeRevision
    ) -> None:
        destination_info = destination.lstat()
        if stat.S_ISLNK(destination_info.st_mode) or not stat.S_ISDIR(
            destination_info.st_mode
        ):
            raise SandboxContractError(
                f"Workspace destination changed during transfer: {destination}"
            )
        if _host_tree_revision(destination) != expected:
            raise SandboxContractError(
                f"Backend workspace changed after checkout: {destination}"
            )
        os.chmod(staging, stat.S_IMODE(destination_info.st_mode))
        if (staging.stat().st_uid, staging.stat().st_gid) != (
            destination_info.st_uid,
            destination_info.st_gid,
        ):
            os.chown(staging, destination_info.st_uid, destination_info.st_gid)
        _exchange_paths(destination, staging)
        try:
            open_files = _process_open_files_under(staging)
            if open_files:
                raise SandboxContractError(
                    "Backend workspace has an open writer during publish: "
                    f"{open_files[0]}"
                )
            if _host_tree_revision(staging) != expected:
                raise SandboxContractError(
                    f"Backend workspace changed during publish: {destination}"
                )
        except BaseException:
            _exchange_paths(destination, staging)
            raise
        shutil.rmtree(staging)


async def finish_workspace_transfer(operation: Awaitable[_T]) -> _T:
    """Drain one owned transfer before propagating caller cancellation.

    Docker archive operations use worker threads which keep running after an
    await is cancelled.  Releasing the workspace lock before those operations
    settle would let the next lease race a half-finished checkout/check-in.
    """
    task = asyncio.ensure_future(operation)
    cancellation: asyncio.CancelledError | None = None
    while True:
        try:
            result = await asyncio.shield(task)
        except asyncio.CancelledError as error:
            current = asyncio.current_task()
            if current is None or current.cancelling() <= 0:
                raise
            if cancellation is None:
                cancellation = error
            if not task.done():
                continue
            try:
                result = task.result()
            except BaseException as transfer_error:
                raise cancellation from transfer_error
        except BaseException as transfer_error:
            if cancellation is not None:
                raise cancellation from transfer_error
            raise
        if cancellation is not None:
            raise cancellation
        return result
