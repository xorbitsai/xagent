"""
Abstract interface for Sandbox Service.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Literal, Optional

TemplateType = Literal["image", "snapshot"]
"""Supported template types."""

CodeType = Literal["python", "javascript"]
"""Supported code execution types."""


@dataclass
class SandBoxTemplate:
    """
    Template for creating a sandbox.
    """

    _type: Optional[TemplateType] = "image"
    """Template type."""

    image: Optional[str] = None
    """Container image, required when _type=image."""

    snapshot_id: Optional[str] = None
    """Snapshot ID, required when _type=snapshot."""

@dataclass
class SandboxConfig:
    """
    Configuration parameters for creating a sandbox.
    """

    cpus: Optional[int] = 1
    """CPU core limit."""

    memory: Optional[int] = 512
    """Memory limit in MB."""

    env: Optional[dict[str, str]] = None
    """Environment variables to inject."""

    volumes: Optional[list[tuple[str, str, str]]] = None
    """Volume mounts as (host_path, guest_path, mode).
    Mode: 'ro' (read-only) or 'rw' (read-write)."""

    network_isolated: Optional[bool] = False
    """Network isolation. True blocks external network access."""

    ports: Optional[list[tuple[int, int]]] = None
    """Port mappings as [(host_port, guest_port)]."""

@dataclass
class SandboxInfo:
    """Sandbox status information."""

    name: str
    """Sandbox name."""

    state: str
    """Sandbox state:
    - 'running': Running
    - 'stopped': Stopped
    - 'unknown': Unknown
    """

    template: SandBoxTemplate
    """Template used to create this sandbox."""

    config: SandboxConfig
    """Configuration used to create this sandbox."""

    created_at: Optional[str] = None
    """Creation time in ISO 8601 format."""

@dataclass
class SandboxSnapshot:
    """Sandbox snapshot information."""

    snapshot_id: str
    """Snapshot ID."""

    metadata: dict = field(default_factory=dict)
    """Snapshot metadata."""

    created_at: Optional[str] = None
    """Creation time in ISO 8601 format."""

@dataclass
class ExecResult:
    """Execution result of a command or code."""

    exit_code: int
    """Exit code. 0 indicates success, non-zero indicates failure."""

    stdout: str
    """Standard output."""

    stderr: str
    """Standard error output."""

    @property
    def success(self) -> bool:
        return self.exit_code == 0

class Sandbox(abc.ABC):
    """
    Abstract interface for a sandbox instance.

    Supports two usage patterns:

        # Manual stop
        try:
            result = await sandbox.exec("echo hello")
        finally:
            await sandbox.stop()

        # Auto-stop with async context manager
        async with sandbox:
            result = await sandbox.exec("echo hello")
    """

    async def __aenter__(self) -> "Sandbox":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.stop()

    # --- Properties ---

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Sandbox name (unique identifier)."""

    # --- Lifecycle ---

    @abc.abstractmethod
    async def stop(self) -> None:
        """Stop the sandbox, preserving its state."""

    @abc.abstractmethod
    async def info(self) -> SandboxInfo:
        """Get sandbox status information."""

    # --- Execution ---

    @abc.abstractmethod
    async def exec(
        self,
        command: str,
        *args: str,
        env: Optional[dict[str, str]] = None,
    ) -> ExecResult:
        """Execute a shell command in the sandbox.

        Args:
            command: Shell command to execute.
            args: Command arguments.
            env: Additional environment variables (merged with existing).

        Returns:
            ExecResult: Execution result with exit code, stdout, and stderr.
        """

    @abc.abstractmethod
    async def run_code(
        self,
        code: str,
        code_type: CodeType = "python",
        env: Optional[dict[str, str]] = None,
    ) -> ExecResult:
        """Execute code in the sandbox.

        Args:
            code: Code string to execute.
            code_type: Code type.
            env: Additional environment variables (merged with existing).

        Returns:
            ExecResult: Execution result with exit code, stdout, and stderr.
        """

    # --- File Operations ---

    @abc.abstractmethod
    async def upload_file(self, local_path: str, remote_path: str, overwrite: bool = False) -> None:
        """Upload a local file to the sandbox.

        Args:
            local_path: Local file path.
            remote_path: Target path in sandbox (including filename).
            overwrite: Whether to overwrite if target exists. Default False.

        Raises:
            FileNotFoundError: Local file not found.
            FileExistsError: Target exists and overwrite=False.
        """

    @abc.abstractmethod
    async def download_file(self, remote_path: str, local_path: str, overwrite: bool = False) -> None:
        """Download a file from the sandbox.

        Args:
            remote_path: Source path in sandbox.
            local_path: Local target path (including filename).
            overwrite: Whether to overwrite if local file exists. Default False.

        Raises:
            FileNotFoundError: Source file not found in sandbox.
            FileExistsError: Local file exists and overwrite=False.
        """

    @abc.abstractmethod
    async def write_file(self, content: str, remote_path: str, overwrite: bool = False) -> None:
        """Write string content directly to a sandbox file.

        Args:
            content: Text content to write.
            remote_path: Target path in sandbox (including filename).
            overwrite: Whether to overwrite if target exists. Default False.

        Raises:
            FileExistsError: Target exists and overwrite=False.
        """

    @abc.abstractmethod
    async def read_file(self, remote_path: str) -> str:
        """Read file content from the sandbox.

        Args:
            remote_path: File path in sandbox.

        Raises:
            FileNotFoundError: File not found in sandbox.
        """

class SandboxService(abc.ABC):
    """
    Abstract interface for sandbox lifecycle management.

    Typical usage:

        service = BoxliteService()

        # Get or create sandbox
        async with await service.get_or_create("my-box") as sandbox:
            result = await sandbox.exec("python train.py")
            print(sandbox.name)  # "my-box"

        # List all sandboxes
        boxes = await service.list()
        print(boxes)

        # Delete sandbox
        await service.delete("my-box")

        # Create snapshot
        await service.create_snapshot("my-box", "my-box-v1.0")

        # Create from snapshot
        await service.get_or_create("my-box", template=SandBoxTemplate(_type="snapshot", snapshot_id="my-box-v1.0"))
    """

    @abc.abstractmethod
    async def get_or_create(
        self,
        name: str,
        template: Optional[SandBoxTemplate] = None,
        config: Optional[SandboxConfig] = None,
    ) -> Sandbox:
        """Get or create a sandbox, handling resume automatically.

        Behavior:
        - Exists and running → return directly
        - Exists and stopped → resume and return
        - Does not exist → create and return

        Args:
            name: Sandbox name (unique identifier).
            template: Template for creation only. Ignored for existing sandboxes.
            config: Configuration for creation only. Ignored for existing sandboxes.

        Returns:
            Sandbox: Operational sandbox instance.
        """

    @abc.abstractmethod
    async def list(self) -> list[SandboxInfo]:
        """List all sandboxes (both running and stopped).

        Returns:
            list[SandboxInfo]: List of sandbox status information.
        """

    @abc.abstractmethod
    async def delete(self, name: str) -> None:
        """Permanently delete a sandbox and release all resources.

        Args:
            name: Sandbox name to delete.
        """

    @abc.abstractmethod
    async def create_snapshot(self, name: str, snapshot_id: str) -> SandboxSnapshot:
        """Create a sandbox snapshot.

        Args:
            name: Sandbox name.
            snapshot_id: Unique snapshot identifier.
        """

    @abc.abstractmethod
    async def list_snapshot(self) -> list[SandboxSnapshot]:
        """List all sandbox snapshots.

        Returns:
            list[SandboxSnapshot]: List of snapshot information.
        """

    @abc.abstractmethod
    async def delete_snapshot(self, snapshot_id: str) -> None:
        """Permanently delete a sandbox snapshot.

        Args:
            snapshot_id: Unique snapshot identifier.
        """
