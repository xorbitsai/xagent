"""Task-scoped Docker workspace transfer without writable host bind mounts.

Only one ``TaskWorkspace.workspace_dir`` is synchronized at a time.  The
guest uses a fixed, sandbox-owned path and publication touches only files the
tool actually changed, so unrelated activity elsewhere in the owner's upload
tree cannot discard a successful tool result.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import shlex
import shutil
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence, TypeVar

from .base import ExecResult, Sandbox, SandboxContractError

MAX_TASK_FILES = 20_000
MAX_TASK_BYTES = 200 * 1024 * 1024
MAX_MANIFEST_BYTES = 16 * 1024 * 1024
HELPER_TIMEOUT_SECONDS = 30.0
_T = TypeVar("_T")

_RESET_SCRIPT = r"""
import os, shutil, stat, sys
root = os.path.abspath(sys.argv[1])
if root != sys.argv[1] or not root.startswith('/home/sandbox/.xagent-workspaces/'):
    raise SystemExit('invalid task workspace root')
if os.path.lexists(root):
    mode = os.lstat(root).st_mode
    if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
        raise SystemExit('task workspace root is not a directory')
    def fail(error):
        raise error
    for current, dirs, _files in os.walk(root, topdown=True, followlinks=False, onerror=fail):
        os.chmod(current, 0o700)
        for name in dirs:
            path = os.path.join(current, name)
            mode = os.lstat(path).st_mode
            if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
                raise SystemExit('unsupported task workspace entry')
            os.chmod(path, 0o700)
    shutil.rmtree(root)
os.makedirs(root, mode=0o700)
"""

_MANIFEST_SCRIPT = r"""
import hashlib, json, os, stat, sys
root = os.path.abspath(sys.argv[1])
if root != sys.argv[1] or not root.startswith('/home/sandbox/.xagent-workspaces/'):
    raise SystemExit('invalid task workspace root')
entries = []
total = 0
file_count = 0
def fail(error):
    raise error
for current, dirs, files in os.walk(root, topdown=True, followlinks=False, onerror=fail):
    dirs.sort(); files.sort()
    for name in dirs:
        path = os.path.join(current, name)
        info = os.lstat(path)
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise SystemExit('unsupported task workspace entry')
        entries.append({'path': os.path.relpath(path, root), 'type': 'dir', 'mode': stat.S_IMODE(info.st_mode)})
    for name in files:
        path = os.path.join(current, name)
        info = os.lstat(path)
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise SystemExit('unsupported task workspace entry')
        digest = hashlib.sha256()
        with open(path, 'rb') as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
        total += info.st_size
        file_count += 1
        entries.append({'path': os.path.relpath(path, root), 'type': 'file', 'mode': stat.S_IMODE(info.st_mode), 'size': info.st_size, 'sha256': digest.hexdigest()})
        if file_count > 20000 or total > 209715200:
            raise SystemExit('task workspace exceeds transfer limits')
print(json.dumps({'entries': entries, 'total': total}, separators=(',', ':')))
"""


@dataclass(frozen=True)
class FileState:
    relative: Path
    mode: int
    size: int
    sha256: str


@dataclass(frozen=True)
class TreeState:
    directories: tuple[tuple[Path, int], ...]
    files: tuple[FileState, ...]
    total_bytes: int

    @property
    def files_by_path(self) -> dict[Path, FileState]:
        return {item.relative: item for item in self.files}


@dataclass(frozen=True)
class ExplicitInput:
    host_path: Path
    guest_path: Path
    size: int
    sha256: str


@dataclass(frozen=True)
class WorkspaceCallPlan:
    args: Mapping[str, Any]
    inputs: tuple[ExplicitInput, ...]
    env: dict[str, str]


def _safe_relative(raw: object) -> Path:
    if not isinstance(raw, str):
        raise SandboxContractError("Task workspace manifest path is not a string")
    value = Path(raw)
    if not raw or value.is_absolute() or raw in {".", ".."} or ".." in value.parts:
        raise SandboxContractError("Task workspace manifest contains an unsafe path")
    return value


def _enforce_limits(file_count: int, total_bytes: int) -> None:
    if file_count > MAX_TASK_FILES:
        raise SandboxContractError(
            f"Task workspace exceeds the {MAX_TASK_FILES}-file transfer limit"
        )
    if total_bytes > MAX_TASK_BYTES:
        raise SandboxContractError(
            f"Task workspace exceeds the {MAX_TASK_BYTES}-byte transfer limit"
        )


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _digest_fd(file_fd: int) -> str:
    digest = hashlib.sha256()
    with os.fdopen(os.dup(file_fd), "rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def scan_host_tree(root: Path) -> TreeState:
    """Capture one task tree through no-follow directory descriptors."""
    root = root.absolute()
    if not root.is_absolute() or root == Path("/"):
        raise SandboxContractError(f"Unsafe task workspace root: {root}")
    root.mkdir(parents=True, exist_ok=True)
    directories: list[tuple[Path, int]] = []
    files: list[FileState] = []
    total = 0
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
    file_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        root_fd = os.open(root, directory_flags)
    except OSError as exc:
        raise SandboxContractError(f"Unsafe task workspace root: {root}") from exc
    pending: list[tuple[int, Path]] = [(root_fd, Path("."))]
    try:
        while pending:
            current_fd, current_relative = pending.pop()
            try:
                for name in sorted(os.listdir(current_fd), reverse=True):
                    relative = (
                        Path(name)
                        if current_relative == Path(".")
                        else current_relative / name
                    )
                    info = os.stat(name, dir_fd=current_fd, follow_symlinks=False)
                    if stat.S_ISDIR(info.st_mode):
                        try:
                            child_fd = os.open(name, directory_flags, dir_fd=current_fd)
                        except OSError as exc:
                            raise SandboxContractError(
                                f"Task workspace directory changed during scan: {relative}"
                            ) from exc
                        directories.append((relative, stat.S_IMODE(info.st_mode)))
                        pending.append((child_fd, relative))
                        continue
                    if not stat.S_ISREG(info.st_mode):
                        raise SandboxContractError(
                            f"Task workspace contains an unsupported entry: {relative}"
                        )
                    try:
                        file_fd = os.open(name, file_flags, dir_fd=current_fd)
                    except OSError as exc:
                        raise SandboxContractError(
                            f"Task workspace file changed during scan: {relative}"
                        ) from exc
                    try:
                        opened_info = os.fstat(file_fd)
                        if not stat.S_ISREG(opened_info.st_mode):
                            raise SandboxContractError(
                                f"Task workspace contains an unsupported entry: {relative}"
                            )
                        total += opened_info.st_size
                        files.append(
                            FileState(
                                relative=relative,
                                mode=stat.S_IMODE(opened_info.st_mode),
                                size=opened_info.st_size,
                                sha256=_digest_fd(file_fd),
                            )
                        )
                        _enforce_limits(len(files), total)
                    finally:
                        os.close(file_fd)
            finally:
                os.close(current_fd)
    finally:
        for directory_fd, _relative in pending:
            os.close(directory_fd)
    directories.sort(key=lambda item: (len(item[0].parts), str(item[0])))
    files.sort(key=lambda item: str(item.relative))
    return TreeState(tuple(directories), tuple(files), total)


def _open_host_parent(root: Path, relative: Path) -> int:
    """Open/create a publication parent without following guest-controlled links."""
    flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
    current_fd = os.open(root, flags)
    try:
        for component in relative.parent.parts:
            try:
                os.mkdir(component, mode=0o750, dir_fd=current_fd)
            except FileExistsError:
                pass
            next_fd = os.open(component, flags, dir_fd=current_fd)
            os.close(current_fd)
            current_fd = next_fd
        return current_fd
    except BaseException:
        os.close(current_fd)
        raise


def _file_state_at(parent_fd: int, name: str, relative: Path) -> FileState | None:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        file_fd = os.open(name, flags, dir_fd=parent_fd)
    except FileNotFoundError:
        return None
    try:
        info = os.fstat(file_fd)
        if not stat.S_ISREG(info.st_mode):
            raise SandboxContractError(f"Host publication target is unsafe: {relative}")
        digest = hashlib.sha256()
        with os.fdopen(os.dup(file_fd), "rb") as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
        return FileState(
            relative=relative,
            mode=stat.S_IMODE(info.st_mode),
            size=info.st_size,
            sha256=digest.hexdigest(),
        )
    finally:
        os.close(file_fd)


def parse_guest_manifest(raw: str) -> TreeState:
    if len(raw.encode("utf-8")) > MAX_MANIFEST_BYTES:
        raise SandboxContractError("Task workspace manifest is too large")
    try:
        payload = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as exc:
        raise SandboxContractError("Sandbox returned an invalid manifest") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("entries"), list):
        raise SandboxContractError("Sandbox returned an invalid manifest")

    directories: list[tuple[Path, int]] = []
    files: list[FileState] = []
    seen: set[Path] = set()
    total = 0
    for item in payload["entries"]:
        if not isinstance(item, dict):
            raise SandboxContractError("Sandbox returned an invalid manifest entry")
        relative = _safe_relative(item.get("path"))
        if relative in seen:
            raise SandboxContractError("Task workspace manifest contains duplicates")
        seen.add(relative)
        mode = item.get("mode")
        if not isinstance(mode, int) or not 0 <= mode <= 0o7777:
            raise SandboxContractError("Task workspace manifest mode is invalid")
        safe_mode = mode & 0o755
        if item.get("type") == "dir":
            directories.append((relative, safe_mode))
            continue
        size = item.get("size")
        digest = item.get("sha256")
        if (
            item.get("type") != "file"
            or not isinstance(size, int)
            or size < 0
            or not isinstance(digest, str)
            or len(digest) != 64
        ):
            raise SandboxContractError("Task workspace manifest file is invalid")
        try:
            bytes.fromhex(digest)
        except ValueError as exc:
            raise SandboxContractError(
                "Task workspace manifest digest is invalid"
            ) from exc
        total += size
        files.append(FileState(relative, safe_mode, size, digest))
        _enforce_limits(len(files), total)
    if payload.get("total") != total:
        raise SandboxContractError("Task workspace manifest total is inconsistent")
    return TreeState(tuple(directories), tuple(files), total)


async def _helper(
    sandbox: Sandbox, script: str, root: Path, *, cap: int | None = None
) -> ExecResult:
    try:
        return await asyncio.wait_for(
            sandbox.exec(
                "python", "-I", "-S", "-c", script, str(root), max_output_bytes=cap
            ),
            timeout=HELPER_TIMEOUT_SECONDS,
        )
    except TimeoutError as exc:
        raise SandboxContractError("Task workspace helper timed out") from exc


class TaskWorkspaceTransfer:
    """Synchronize one host task root to one fixed guest root per tool lease."""

    def __init__(self, host_root: Path, guest_root: Path) -> None:
        self.host_root = host_root.resolve()
        self.guest_root = guest_root
        self.lock = asyncio.Lock()
        self._baseline: TreeState | None = None

    async def checkout(
        self, sandbox: Sandbox, inputs: Sequence[ExplicitInput] = ()
    ) -> None:
        baseline = await asyncio.to_thread(scan_host_tree, self.host_root)
        reset = await _helper(sandbox, _RESET_SCRIPT, self.guest_root)
        if reset.exit_code != 0:
            raise SandboxContractError(
                f"Failed to reset task workspace: {reset.stderr or reset.stdout}"
            )
        for relative, _mode in baseline.directories:
            result = await sandbox.exec("mkdir", "-p", str(self.guest_root / relative))
            if result.exit_code != 0:
                raise SandboxContractError("Failed to create task workspace directory")
        for item in baseline.files:
            await sandbox.upload_file(
                str(self.host_root / item.relative),
                str(self.guest_root / item.relative),
                overwrite=False,
            )
        for relative, mode in reversed(baseline.directories):
            chmod = await sandbox.exec(
                "chmod", f"{mode & 0o755:o}", str(self.guest_root / relative)
            )
            if chmod.exit_code != 0:
                raise SandboxContractError("Failed to set task workspace permissions")
        for explicit_input in inputs:
            before = await asyncio.to_thread(
                _capture_explicit_input, explicit_input.host_path
            )
            if before != explicit_input:
                raise SandboxContractError(
                    "External sandbox input changed before upload"
                )
            await sandbox.upload_file(
                str(explicit_input.host_path),
                str(explicit_input.guest_path),
                overwrite=True,
            )
            after = await asyncio.to_thread(
                _capture_explicit_input, explicit_input.host_path
            )
            if after != explicit_input:
                raise SandboxContractError(
                    "External sandbox input changed during upload"
                )
        if await asyncio.to_thread(scan_host_tree, self.host_root) != baseline:
            raise SandboxContractError("Task workspace changed during checkout")
        self._baseline = baseline

    async def check_in(self, sandbox: Sandbox) -> None:
        baseline = self._baseline
        if baseline is None:
            raise SandboxContractError("Task workspace check-in has no checkout")
        result = await _helper(
            sandbox, _MANIFEST_SCRIPT, self.guest_root, cap=MAX_MANIFEST_BYTES
        )
        if result.exit_code != 0 or result.truncated:
            raise SandboxContractError(
                f"Failed to inspect task workspace: {result.stderr or result.stdout}"
            )
        guest = parse_guest_manifest(result.stdout)
        old = baseline.files_by_path
        new = guest.files_by_path
        changed = [path for path, state in new.items() if old.get(path) != state]
        removed = [path for path in old if path not in new]
        if not changed and not removed:
            return

        staging = Path(
            tempfile.mkdtemp(
                prefix=f".{self.host_root.name}.xagent-publish-",
                dir=self.host_root.parent,
            )
        )
        try:
            for relative in changed:
                item = new[relative]
                target = staging / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                await sandbox.download_file_bounded(
                    str(self.guest_root / relative),
                    str(target),
                    max_bytes=item.size,
                )
                if target.stat().st_size != item.size or _digest(target) != item.sha256:
                    raise SandboxContractError(
                        "Task workspace file changed while it was downloaded"
                    )

            for relative in (*changed, *removed):
                parent_fd = await asyncio.to_thread(
                    _open_host_parent, self.host_root, relative
                )
                try:
                    current = await asyncio.to_thread(
                        _file_state_at, parent_fd, relative.name, relative
                    )
                    if current != old.get(relative):
                        raise SandboxContractError(
                            f"Host changed a file being published: {relative}"
                        )
                finally:
                    os.close(parent_fd)

            for relative in changed:
                staged = staging / relative
                staged.chmod(new[relative].mode & 0o755)
                parent_fd = await asyncio.to_thread(
                    _open_host_parent, self.host_root, relative
                )
                try:
                    current = await asyncio.to_thread(
                        _file_state_at, parent_fd, relative.name, relative
                    )
                    if current != old.get(relative):
                        raise SandboxContractError(
                            f"Host changed a file being published: {relative}"
                        )
                    if current is None:
                        try:
                            os.link(
                                staged,
                                relative.name,
                                dst_dir_fd=parent_fd,
                                follow_symlinks=False,
                            )
                        except FileExistsError:
                            raise SandboxContractError(
                                f"Host created a file being published: {relative}"
                            ) from None
                        staged.unlink()
                    else:
                        os.replace(staged, relative.name, dst_dir_fd=parent_fd)
                finally:
                    os.close(parent_fd)
            for relative in removed:
                parent_fd = await asyncio.to_thread(
                    _open_host_parent, self.host_root, relative
                )
                try:
                    current = await asyncio.to_thread(
                        _file_state_at, parent_fd, relative.name, relative
                    )
                    if current != old[relative]:
                        raise SandboxContractError(
                            f"Host changed a file being removed: {relative}"
                        )
                    try:
                        os.unlink(relative.name, dir_fd=parent_fd)
                    except FileNotFoundError:
                        raise SandboxContractError(
                            f"Host changed a file being removed: {relative}"
                        ) from None
                finally:
                    os.close(parent_fd)
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    async def cleanup(self, sandbox: Sandbox, inputs: Sequence[ExplicitInput]) -> None:
        await _helper(sandbox, _RESET_SCRIPT, self.guest_root)
        for directory in sorted({item.guest_path.parent for item in inputs}):
            result = await sandbox.exec("rm", "-rf", str(directory))
            if result.exit_code != 0:
                raise SandboxContractError("Failed to clean sandbox input staging")
        self._baseline = None


def _capture_explicit_input(path: Path) -> ExplicitInput:
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise SandboxContractError(
            "External sandbox inputs must be explicit regular files"
        )
    token = hashlib.sha256(str(path).encode()).hexdigest()[:24]
    guest = Path("/home/sandbox/.xagent-inputs") / token / path.name
    return ExplicitInput(path, guest, info.st_size, _digest(path))


def _rewrite_value(value: Any, replacements: Sequence[tuple[str, str]]) -> Any:
    if isinstance(value, str):
        for source, target in sorted(replacements, key=lambda item: -len(item[0])):
            # Every source is a complete, separately authorized path. Do not
            # treat it as a directory prefix: that would turn an unchecked
            # `/allowed/../other` string into a guest traversal.
            pattern = re.compile(
                rf"(?<![A-Za-z0-9_./-]){re.escape(source)}"
                rf"(?=$|[\s'\";:,)=\]}}])"
            )
            value = pattern.sub(lambda _match: target, value)
        return value
    if isinstance(value, dict):
        return {key: _rewrite_value(item, replacements) for key, item in value.items()}
    if isinstance(value, list):
        return [_rewrite_value(item, replacements) for item in value]
    if isinstance(value, tuple):
        return tuple(_rewrite_value(item, replacements) for item in value)
    return value


def _candidate_paths(value: str) -> set[str]:
    """Extract complete absolute path arguments without interpreting code."""
    candidates = {value} if os.path.isabs(value) else set()
    try:
        tokens = shlex.split(value)
    except ValueError:
        tokens = []
    for token in tokens:
        candidate = token.strip("()[]{}<>,;:")
        if os.path.isabs(candidate):
            candidates.add(candidate)
    return candidates


def build_call_plan(
    workspace: Any,
    args: Mapping[str, Any],
    *,
    host_root: Path,
    guest_root: Path,
) -> WorkspaceCallPlan:
    """Map task paths and exact, authorized external file arguments."""
    replacements: list[tuple[str, str]] = []
    inputs: list[ExplicitInput] = []
    seen: set[Path] = set()
    total_input_bytes = 0

    def collect_path(raw_path: str) -> None:
        nonlocal total_input_bytes
        candidate = Path(raw_path)
        try:
            resolved = workspace.resolve_authorized_path(
                candidate,
                base_dir=workspace.workspace_dir,
                include_external_dirs=True,
            )
        except (OSError, ValueError) as exc:
            lexical_root = str(host_root)
            if raw_path == lexical_root or raw_path.startswith(lexical_root + os.sep):
                raise SandboxContractError(
                    "Task workspace argument escapes its authorized root"
                ) from exc
            return
        if resolved.is_relative_to(host_root):
            guest_path = guest_root / resolved.relative_to(host_root)
            replacements.append((raw_path, str(guest_path)))
            return
        if resolved in seen:
            return
        item = _capture_explicit_input(resolved)
        seen.add(resolved)
        total_input_bytes += item.size
        _enforce_limits(len(seen), total_input_bytes)
        inputs.append(item)
        replacements.append((str(resolved), str(item.guest_path)))

    def collect(value: Any) -> None:
        if isinstance(value, dict):
            for item in value.values():
                collect(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                collect(item)
        elif isinstance(value, str):
            for candidate in _candidate_paths(value):
                collect_path(candidate)

    collect(args)
    env = {
        "XAGENT_SANDBOX_HOST_WORKSPACE_ROOT": str(host_root),
        "XAGENT_SANDBOX_GUEST_WORKSPACE_ROOT": str(guest_root),
        "XAGENT_SANDBOX_GUEST_INPUT_DIRS": json.dumps(
            sorted({str(item.guest_path.parent) for item in inputs})
        ),
    }
    return WorkspaceCallPlan(
        args=_rewrite_value(dict(args), replacements),
        inputs=tuple(inputs),
        env=env,
    )


def map_guest_result(
    result: Any,
    guest_root: Path,
    host_root: Path,
    inputs: Sequence[ExplicitInput] = (),
) -> Any:
    replacements: list[tuple[str, str]] = []
    input_paths = {item.guest_path: item.host_path for item in inputs}

    def collect(value: Any) -> None:
        if isinstance(value, dict):
            for item in value.values():
                collect(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                collect(item)
        elif isinstance(value, str):
            for raw_path in _candidate_paths(value):
                candidate = Path(raw_path)
                normalized = Path(os.path.normpath(candidate))
                if normalized.is_relative_to(guest_root):
                    replacements.append(
                        (
                            raw_path,
                            str(host_root / normalized.relative_to(guest_root)),
                        )
                    )
                elif normalized in input_paths:
                    replacements.append((raw_path, str(input_paths[normalized])))

    collect(result)
    return _rewrite_value(result, replacements)


async def finish_transfer(awaitable: Any) -> Any:
    """Drain archive worker threads before a task cancellation propagates."""
    task = asyncio.ensure_future(awaitable)
    cancellation: asyncio.CancelledError | None = None
    while True:
        try:
            value = await asyncio.shield(task)
        except asyncio.CancelledError as exc:
            if cancellation is None:
                cancellation = exc
            if not task.done():
                continue
            try:
                value = task.result()
            except BaseException as transfer_error:
                raise cancellation from transfer_error
        except BaseException as transfer_error:
            if cancellation is not None:
                raise cancellation from transfer_error
            raise
        if cancellation is not None:
            raise cancellation
        return value
