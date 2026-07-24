"""Core SSH MCP seams. Concrete adapters implement these in later phases."""

from __future__ import annotations

from contextlib import AbstractAsyncContextManager
from typing import Any, Protocol, runtime_checkable

from .types import (
    BoundTargetInfo,
    MaterializedSshPaths,
    ResolvedSshTarget,
    SensitiveSshCredential,
    SshExecutionContext,
    SshSecretHandle,
)


@runtime_checkable
class SandboxLike(Protocol):
    """Structural type for the sandbox methods the SSH runner and sandbox
    materializer use. Lets them narrow ``object`` without importing a concrete
    sandbox, so they can drop ``# type: ignore`` on these calls. ``exec``
    returns Any (its result exposes ``exit_code``/``stdout``/``stderr``)."""

    async def exec(
        self,
        command: str,
        *args: str,
        env: dict[str, str] | None = None,
        max_output_bytes: int | None = None,
    ) -> Any: ...

    async def write_file(
        self, content: str, remote_path: str, overwrite: bool = False
    ) -> None: ...

    async def upload_file(
        self, local_path: str, remote_path: str, overwrite: bool = False
    ) -> None: ...

    async def download_file(
        self, remote_path: str, local_path: str, overwrite: bool = False
    ) -> None: ...


@runtime_checkable
class SshTargetProvider(Protocol):
    """Resolves an agent binding alias to a concrete, authorized target."""

    async def resolve(
        self, context: SshExecutionContext, target_alias: str
    ) -> ResolvedSshTarget: ...

    async def list_bound_targets(
        self, context: SshExecutionContext
    ) -> list[BoundTargetInfo]: ...


@runtime_checkable
class SshSecretStore(Protocol):
    """Exchanges a secret handle for decrypted credential material."""

    async def read_version(
        self, secret_handle: SshSecretHandle
    ) -> SensitiveSshCredential: ...


@runtime_checkable
class SshAuditSink(Protocol):
    """Records one append-only audit event per SSH operation. The executor calls
    it after each operation resolves — on success and on failure. Auditing is
    best-effort: implementations must not raise into the caller (the executor
    swallows sink errors so a logging failure never fails the operation)."""

    async def record(
        self,
        *,
        context: SshExecutionContext,
        operation: str,
        status: str,
        target: ResolvedSshTarget | None = None,
        error_code: str | None = None,
    ) -> None: ...


@runtime_checkable
class SandboxSecretMaterializer(Protocol):
    """Materializes one credential + known_hosts into a sandbox for one call.

    Returns an async context manager yielding the written paths; the
    implementation is responsible for strict permissions and for cleaning up
    on normal exit, exception, timeout, and cancellation.

    ``sandbox`` is typed as ``object`` here to keep the domain layer decoupled
    from any concrete sandbox implementation; real adapters narrow it.
    """

    def materialize_ssh(
        self,
        sandbox: object,
        credential: SensitiveSshCredential,
        known_hosts: str,
    ) -> AbstractAsyncContextManager[MaterializedSshPaths]: ...
