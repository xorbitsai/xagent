"""
Boxlite sandbox implementation.
"""

from __future__ import annotations

import abc
import asyncio
import logging
import os
import tempfile
import uuid
from typing import Optional

import boxlite  # type: ignore[import-not-found]
from boxlite import SimpleBox  # type: ignore[unused-ignore]

from .base import (
    CodeType,
    ExecResult,
    Sandbox,
    SandboxConfig,
    SandboxInfo,
    SandboxService,
    SandboxSnapshot,
    SandboxTemplate,
)

logger = logging.getLogger(__name__)


class BoxliteStore(abc.ABC):
    """
    Store for persisting Boxlite data.
    """

    @abc.abstractmethod
    def get_info(self, name: str) -> Optional[SandboxInfo]:
        """Get sandbox info."""

    @abc.abstractmethod
    def add_info(self, name: str, info: SandboxInfo) -> None:
        """Add sandbox info."""

    @abc.abstractmethod
    def update_info_state(self, name: str, state: str) -> None:
        """Update sandbox info state."""

    @abc.abstractmethod
    def delete_info(self, name: str) -> None:
        """Delete sandbox info."""


class MemBoxliteStore(BoxliteStore):
    """
    In-memory implementation of BoxliteStore.
    """

    def __init__(self) -> None:
        self._metadata: dict[str, SandboxInfo] = {}

    def get_info(self, name: str) -> Optional[SandboxInfo]:
        return self._metadata.get(name)

    def add_info(self, name: str, info: SandboxInfo) -> None:
        self._metadata[name] = info

    def update_info_state(self, name: str, state: str) -> None:
        if name in self._metadata:
            self._metadata[name].state = state

    def delete_info(self, name: str) -> None:
        if name in self._metadata:
            del self._metadata[name]


def _get_state(raw_info: boxlite.BoxInfo) -> str:  # type: ignore[no-any-unimported]
    """Map boxlite BoxInfo.state to 'running' / 'stopped' / 'unknown'."""
    state = raw_info.state.status.lower()
    if "running" in state:
        return "running"
    if any(key in state for key in ("stopped", "paused", "exited", "created")):
        return "stopped"
    return "unknown"


def _get_info_from_box_info(raw_info: boxlite.BoxInfo) -> SandboxInfo:  # type: ignore[no-any-unimported]
    tpl = SandboxTemplate(
        type="image",
        image=raw_info.image,
    )
    cfg = SandboxConfig(
        cpus=raw_info.cpus,
        memory=raw_info.memory_mib,
        env=None,  # Cannot retrieve
        volumes=None,  # Cannot retrieve
        network_isolated=None,  # Cannot retrieve
        ports=None,  # Cannot retrieve
    )
    info = SandboxInfo(
        name=raw_info.name,
        state=_get_state(raw_info),
        template=tpl,
        config=cfg,
        created_at=raw_info.created_at,
    )
    return info


class BoxliteSandbox(Sandbox):
    """
    Boxlite implementation.
    """

    def __init__(  # type: ignore[no-any-unimported]
        self,
        sandbox_name: str,
        box: SimpleBox,
        info: SandboxInfo,
        store: BoxliteStore,
    ) -> None:
        self._box = box
        self._name = sandbox_name
        self._info = info
        self._store = store

    # --- Properties ---

    @property
    def name(self) -> str:
        """Sandbox name (unique identifier)."""
        return self._name

    # --- Lifecycle ---

    async def stop(self) -> None:
        """Stop the sandbox, preserving its state."""
        await self._box.stop()
        self._store.update_info_state(self._name, "stopped")

    async def info(self) -> SandboxInfo:
        """Get sandbox status information."""
        # Update state in real-time
        # box.info() will throw an error when the box has been stopped, so we need to use the box._runtime.get_info(name) method instead.
        box_info = await self._box._runtime.get_info(self._name)
        self._info.state = _get_state(box_info)

        return self._info

    # --- Execution ---

    async def exec(
        self,
        command: str,
        *args: str,
        env: Optional[dict[str, str]] = None,
    ) -> ExecResult:
        res = await self._box.exec(command, *args, env=env)

        # Filter out seccomp warnings (known issue on macOS, usually appears on first line)
        stderr = res.stderr
        if stderr and "seccomp not available" in stderr:
            lines = stderr.split("\n", 1)  # Split only the first line
            if len(lines) > 0 and "seccomp not available" in lines[0]:
                # Remove first line, keep the rest
                stderr = lines[1] if len(lines) > 1 else ""

        return ExecResult(
            exit_code=res.exit_code,
            stdout=res.stdout,
            stderr=stderr,
            error_message=res.error_message,
        )

    async def run_code(
        self,
        code: str,
        code_type: CodeType = "python",
        env: Optional[dict[str, str]] = None,
    ) -> ExecResult:
        """
        Execute code snippet.
        """
        if code_type == "python":
            return await self.exec("python", "-c", code, env=env)
        elif code_type == "javascript":
            return await self.exec("node", "-e", code, env=env)

    # --- File Operations ---

    async def upload_file(
        self, local_path: str, remote_path: str, overwrite: bool = False
    ) -> None:
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"Local file not found: {local_path}")

        if not overwrite:
            check = await self.exec("test", "-e", remote_path)
            if check.exit_code == 0:
                raise FileExistsError(f"Remote file already exists: {remote_path}")

        # First copy to temp directory (if copying directly to mount directory, it won't be readable on host)
        # Note: Cannot use /tmp as it's a tmpfs mount, copy_in will fail
        # Use /var/tmp or other non-tmpfs directory

        temp_filename = f"_upload_{uuid.uuid4().hex}"
        temp_remote = f"/var/tmp/{temp_filename}"

        # Ensure /var/tmp exists
        await self.exec("mkdir", "-p", "/var/tmp")

        # copy_in to temporary location
        await self._box.copy_in(local_path, temp_remote, overwrite=overwrite)

        # Verify temporary file exists
        check = await self.exec("test", "-f", temp_remote)
        if check.exit_code != 0:
            raise RuntimeError(
                f"Failed to copy file to temporary location: {temp_remote}"
            )

        # Create target directory
        remote_dir = os.path.dirname(remote_path)
        if remote_dir:
            await self.exec("mkdir", "-p", remote_dir)

        # Use mv command to move to target location (supports volume mounts)
        result = await self.exec("mv", temp_remote, remote_path)

        if result.exit_code != 0:
            # Clean up temporary file
            await self.exec("rm", "-f", temp_remote)
            raise RuntimeError(f"Failed to move file to {remote_path}: {result.stderr}")

    async def download_file(
        self, remote_path: str, local_path: str, overwrite: bool = False
    ) -> None:
        check = await self.exec("test", "-e", remote_path)
        if check.exit_code != 0:
            raise FileNotFoundError(f"Remote file not found: {remote_path}")

        if not overwrite and os.path.exists(local_path):
            raise FileExistsError(f"Local file already exists: {local_path}")

        # First copy to temp directory (avoid volume mount issues)

        temp_filename = f"_download_{uuid.uuid4().hex}"
        temp_remote = f"/var/tmp/{temp_filename}"

        # Ensure /var/tmp exists
        await self.exec("mkdir", "-p", "/var/tmp")

        # Use cp command to copy to temporary location (supports volume mounts)
        result = await self.exec("cp", remote_path, temp_remote)
        if result.exit_code != 0:
            raise RuntimeError(
                f"Failed to copy file to temporary location: {result.stderr}"
            )

        # Create local directory
        local_dir = os.path.dirname(local_path)
        if local_dir:
            os.makedirs(local_dir, exist_ok=True)

        # copy_out from temporary location to local
        try:
            await self._box.copy_out(temp_remote, local_path, overwrite=overwrite)
        finally:
            # Clean up temporary file
            await self.exec("rm", "-f", temp_remote)

    async def write_file(
        self, content: str, remote_path: str, overwrite: bool = False
    ) -> None:
        # Write to local temp file
        fd, tmp = tempfile.mkstemp(suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(content)

            # Use upload_file to upload (supports volume mounts)
            await self.upload_file(tmp, remote_path, overwrite=overwrite)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    async def read_file(self, remote_path: str) -> str:
        # Use download_file to read (supports volume mounts)
        fd, tmp = tempfile.mkstemp(suffix=".tmp")
        os.close(fd)
        try:
            await self.download_file(remote_path, tmp, overwrite=True)
            with open(tmp, "r", encoding="utf-8") as f:
                return f.read()
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)


async def _create_or_reuse_box(  # type: ignore[no-any-unimported]
    name: str,
    template: SandboxTemplate,
    config: SandboxConfig,
    runtime: boxlite.Boxlite,
) -> SimpleBox:
    """Create a new Box."""
    # Build SimpleBox parameters
    kwargs: dict = {
        "image": template.image,
        "cpus": config.cpus,
        "memory_mib": config.memory,
        "disk_size_gb": 10,  # Increased to accommodate packages + workspace files
        "auto_remove": False,  # Don't auto-remove, preserve state
        "runtime": runtime,
        "name": name,
        "reuse_existing": True,  # Allow reusing existing box
        "working_dir": config.working_dir,
    }

    if config.env:
        kwargs["env"] = list(config.env.items())

    if config.volumes:
        kwargs["volumes"] = [
            (h, g, mode.lower() == "ro") for h, g, mode in config.volumes
        ]

    if config.ports:
        kwargs["ports"] = config.ports

    if config.network_isolated:
        sec = boxlite.SecurityOptions()
        sec.network_enabled = False
        kwargs["advanced"] = sec

    # Create SimpleBox
    box = SimpleBox(**kwargs)
    await box.start()
    return box


class BoxliteSandboxService(SandboxService):
    """
    Boxlite implementation.
    """

    def __init__(self, store: BoxliteStore, home_dir: Optional[str] = None) -> None:
        """
        Initialize BoxliteSandboxService。

        Args:
            store: Storage for persisting sandbox information
            home_dir: Boxlite's home directory, used to store data such as mirroring and VMs.
                    If None, use the default directory (usually ~/.boxlite)
        """
        if home_dir:
            self._runtime = boxlite.Boxlite(boxlite.Options(home_dir=home_dir))
        else:
            self._runtime = boxlite.Boxlite.default()
        self._store = store
        # Lock for protecting concurrent creation, one lock per name
        self._locks: dict[str, asyncio.Lock] = {}
        # Lock for protecting the _locks dict itself
        self._locks_lock = asyncio.Lock()

    async def get_or_create(
        self,
        name: str,
        template: Optional[SandboxTemplate] = None,
        config: Optional[SandboxConfig] = None,
    ) -> BoxliteSandbox:
        # Snapshot creation not supported
        if template is not None and template.type == "snapshot":
            raise NotImplementedError("Unsupported")

        # Get or create lock for this name
        async with self._locks_lock:
            if name not in self._locks:
                self._locks[name] = asyncio.Lock()
            lock = self._locks[name]

        # Use lock to protect entire get_or_create process
        async with lock:
            # Check if box already exists
            raw_box = await self._runtime.get(name)
            if raw_box:
                # Box exists, get or restore info
                info = self._store.get_info(name)
                if not info:
                    # DB data lost, read from boxlite
                    info = _get_info_from_box_info(raw_box.info())
            else:
                # Box doesn't exist, create new one
                tpl = template or SandboxTemplate(type="image", image="python:slim")
                cfg = config or SandboxConfig()
                info = SandboxInfo(name=name, state="running", template=tpl, config=cfg)

            # Create or reuse box
            box = await _create_or_reuse_box(
                name, info.template, info.config, self._runtime
            )

            # Update created_at if this is a new box
            if box.created:
                info.created_at = box.info().created_at
                self._store.add_info(name, info)

            # Update state
            self._store.update_info_state(name, "running")
            return BoxliteSandbox(
                sandbox_name=name, box=box, info=info, store=self._store
            )

    async def list_sandboxes(self) -> list[SandboxInfo]:
        # Use boxlite as source of truth
        raw_list: list[boxlite.BoxInfo] = await self._runtime.list_info()  # type: ignore[no-any-unimported]
        result: list[SandboxInfo] = []
        for raw_info in raw_list:
            box_name = raw_info.name
            info = self._store.get_info(box_name)
            if info:
                result.append(info)
                info.state = _get_state(raw_info)  # Refresh state
            else:
                # DB data lost, read from boxlite
                info = _get_info_from_box_info(raw_info)
                result.append(info)
        return result

    async def delete(self, name: str) -> None:
        box = await self._runtime.get(name)
        if not box:
            self._store.delete_info(name)
            # clear lock
            async with self._locks_lock:
                self._locks.pop(name, None)
            return

        try:
            await self._runtime.remove(name, force=True)
            self._store.delete_info(name)
            # clear lock
            async with self._locks_lock:
                self._locks.pop(name, None)
        except Exception as e:
            raise RuntimeError(f"delete {name!r} error: {e}") from e

    async def supports_snapshots(self) -> bool:
        return False

    async def create_snapshot(self, name: str, snapshot_id: str) -> SandboxSnapshot:
        raise NotImplementedError("Unsupported")

    async def list_snapshots(self) -> list[SandboxSnapshot]:
        raise NotImplementedError("Unsupported")

    async def delete_snapshot(self, snapshot_id: str) -> None:
        raise NotImplementedError("Unsupported")
