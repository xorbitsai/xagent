"""Core SSH MCP seams. Concrete adapters implement these in later phases."""

from __future__ import annotations

from contextlib import AbstractAsyncContextManager
from typing import Protocol, runtime_checkable

from .types import (
    BoundTargetInfo,
    MaterializedSshPaths,
    ResolvedSshTarget,
    SensitiveSshCredential,
    SshExecutionContext,
    SshSecretHandle,
)


@runtime_checkable
class SshTargetProvider(Protocol):
    """Resolves an agent binding alias to a concrete, authorized target."""

    async def resolve(
        self, context: SshExecutionContext, target_alias: str
    ) -> ResolvedSshTarget: ...

    async def list_bound_targets(self, context: SshExecutionContext) -> list[BoundTargetInfo]: ...


@runtime_checkable
class SshSecretStore(Protocol):
    """Exchanges a secret handle for decrypted credential material."""

    async def read_version(self, secret_handle: SshSecretHandle) -> SensitiveSshCredential: ...


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
