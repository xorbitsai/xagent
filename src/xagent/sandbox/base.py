"""
Abstract interface for Sandbox Service.
"""

from __future__ import annotations

import abc
from typing import Literal, Optional

from pydantic import BaseModel, Field

TemplateType = Literal["image", "snapshot"]
"""Supported template types."""

CodeType = Literal["python", "javascript"]
"""Supported code execution types."""


class SandboxNotFoundError(Exception):
    """Raised when a requested sandbox resource no longer exists."""


class SandboxTemplate(BaseModel):
    """
    Template for creating a sandbox.

    `type="image"` creates a sandbox from a container image.

    `type="snapshot"` creates a sandbox from a previously committed filesystem
    snapshot. A snapshot is only a creation template for the new sandbox's
    initial filesystem contents; runtime configuration such as working
    directory, environment variables, volume mounts, network isolation, and
    port mappings still comes from `SandboxConfig` on the current
    `get_or_create()` call.
    """

    type: Optional[TemplateType] = Field(default="image", description="Template type")

    image: Optional[str] = Field(
        default=None, description="Container image, required when type=image"
    )

    snapshot_id: Optional[str] = Field(
        default=None, description="Snapshot ID, required when type=snapshot"
    )


class SandboxConfig(BaseModel):
    """
    Configuration parameters for creating a sandbox.
    """

    working_dir: Optional[str] = Field(default="/home", description="Working dir")

    cpus: Optional[int] = Field(default=1, ge=1, description="CPU core limit")

    memory: Optional[int] = Field(default=512, ge=128, description="Memory limit in MB")

    env: Optional[dict[str, str]] = Field(
        default=None, description="Environment variables to inject"
    )

    volumes: Optional[list[tuple[str, str, str]]] = Field(
        default=None,
        description="Volume mounts as (host_path, guest_path, mode). Mode: 'ro' (read-only) or 'rw' (read-write)",
    )

    network_isolated: Optional[bool] = Field(
        default=False,
        description="Network isolation. True blocks external network access",
    )

    ports: Optional[list[tuple[int, int]]] = Field(
        default=None, description="Port mappings as [(host_port, guest_port)]"
    )


class SandboxInfo(BaseModel):
    """Sandbox status information."""

    name: str = Field(description="Sandbox name")

    state: str = Field(description="Sandbox state: 'running', 'stopped', or 'unknown'")

    template: SandboxTemplate = Field(
        description="Template used to create this sandbox"
    )

    config: SandboxConfig = Field(
        description="Configuration used to create this sandbox"
    )

    created_at: Optional[str] = Field(
        default=None, description="Creation time in ISO 8601 format"
    )


class SandboxSnapshot(BaseModel):
    """Sandbox snapshot information."""

    snapshot_id: str = Field(description="Snapshot ID")

    metadata: dict = Field(default_factory=dict, description="Snapshot metadata")

    created_at: Optional[str] = Field(
        default=None, description="Creation time in ISO 8601 format"
    )


class ExecResult(BaseModel):
    """Execution result of a command or code."""

    exit_code: int = Field(
        description="Exit code. 0 indicates success, non-zero indicates failure"
    )

    stdout: str = Field(description="Standard output")

    stderr: str = Field(description="Standard error output")

    truncated: bool = Field(
        default=False,
        description="True if output was capped at max_output_bytes and cut short",
    )

    error_message: Optional[str] = Field(default=None, description="Error message")

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

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
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
        max_output_bytes: Optional[int] = None,
    ) -> ExecResult:
        """Execute a shell command in the sandbox.

        Args:
            command: Shell command to execute.
            args: Command arguments.
            env: Additional environment variables (merged with existing).
            max_output_bytes: When set, cap each of stdout/stderr at this many
                bytes, reading no more once the cap is hit (so a flood of output
                can't grow the host client unbounded). Sets ``truncated`` on the
                result. ``None`` (default) keeps the unbounded one-shot read.

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
    async def upload_file(
        self, local_path: str, remote_path: str, overwrite: bool = False
    ) -> None:
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
    async def download_file(
        self, remote_path: str, local_path: str, overwrite: bool = False
    ) -> None:
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
    async def write_file(
        self, content: str, remote_path: str, overwrite: bool = False
    ) -> None:
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
        boxes = await service.list_sandboxes()
        print(boxes)

        # Delete sandbox
        await service.delete("my-box")

        # Create snapshot
        await service.create_snapshot("my-box", "my-box-v1.0")

        # Create from snapshot
        await service.get_or_create("my-box", template=SandboxTemplate(_type="snapshot", snapshot_id="my-box-v1.0"))
    """

    @abc.abstractmethod
    async def get_or_create(
        self,
        name: str,
        template: Optional[SandboxTemplate] = None,
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
    async def list_sandboxes(self) -> list[SandboxInfo]:
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
    async def supports_snapshots(self) -> bool:
        """Check if this sandbox service supports snapshot operations.

        Returns:
            bool: True if snapshots are supported, False otherwise.
        """

    @abc.abstractmethod
    async def create_snapshot(self, name: str, snapshot_id: str) -> SandboxSnapshot:
        """Create a sandbox snapshot.

        Args:
            name: Sandbox name.
            snapshot_id: Unique snapshot identifier.
        """

    @abc.abstractmethod
    async def list_snapshots(self) -> list[SandboxSnapshot]:
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
