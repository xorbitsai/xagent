"""In-memory reference adapters and a local test SSH server for SSH MCP tests."""

from __future__ import annotations

import os
import tempfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass

import asyncssh

from xagent.core.ssh.errors import SshError, SshErrorCode
from xagent.core.ssh.types import (
    BoundTargetInfo,
    MaterializedSshPaths,
    ResolvedSshTarget,
    SensitiveSshCredential,
    SshExecutionContext,
    SshSecretHandle,
)


class InMemorySshTargetProvider:
    """Resolves aliases from a preloaded (agent_id, alias) -> target map."""

    def __init__(self, targets: dict[tuple[int, str], ResolvedSshTarget]) -> None:
        self._targets = targets

    async def resolve(self, context: SshExecutionContext, target_alias: str) -> ResolvedSshTarget:
        key = (context.agent_id, target_alias)
        target = self._targets.get(key)
        if target is None:
            raise SshError(SshErrorCode.TARGET_NOT_FOUND, "target alias is not bound")
        return target

    async def list_bound_targets(self, context: SshExecutionContext) -> list[BoundTargetInfo]:
        return [
            BoundTargetInfo(
                alias=alias,
                display_name=None,
                capabilities=target.capabilities,
            )
            for (agent_id, alias), target in self._targets.items()
            if agent_id == context.agent_id
        ]


class InMemorySshSecretStore:
    """Returns credentials from a preloaded version_id -> credential map."""

    def __init__(self, versions: dict[str, SensitiveSshCredential]) -> None:
        self._versions = versions

    async def read_version(self, secret_handle: SshSecretHandle) -> SensitiveSshCredential:
        credential = self._versions.get(secret_handle.version_id)
        if credential is None:
            raise SshError(SshErrorCode.SECRET_UNAVAILABLE, "credential version unavailable")
        return credential


class LocalTmpSecretMaterializer:
    """Materializes secrets to a local private temp dir (test/reference impl).

    Ignores the sandbox argument: Phase 0 has no real sandbox. It exists to
    lock down the security-critical behavior — 0700 dir, 0600 files, cleanup
    on every exit path — before real sandbox adapters land in Phase 3.
    """

    @asynccontextmanager
    async def materialize_ssh(
        self,
        sandbox: object,
        credential: SensitiveSshCredential,
        known_hosts: str,
    ) -> AsyncIterator[MaterializedSshPaths]:
        directory = tempfile.mkdtemp(prefix="xagent-ssh-")
        os.chmod(directory, 0o700)
        key_path = os.path.join(directory, "id_key")
        known_hosts_path = os.path.join(directory, "known_hosts")
        try:
            _write_private(key_path, credential.private_key)
            _write_private(known_hosts_path, known_hosts.encode("utf-8"))
            yield MaterializedSshPaths(private_key_path=key_path, known_hosts_path=known_hosts_path)
        finally:
            for path in (key_path, known_hosts_path):
                _best_effort_shred(path)
            if os.path.isdir(directory):
                os.rmdir(directory)


def _write_private(path: str, data: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        os.write(fd, data)
    finally:
        os.close(fd)


def _best_effort_shred(path: str) -> None:
    if not os.path.exists(path):
        return
    try:
        size = os.path.getsize(path)
        fd = os.open(path, os.O_WRONLY)
        try:
            os.write(fd, b"\x00" * size)
        finally:
            os.close(fd)
    except OSError:
        pass
    with suppress(OSError):
        os.unlink(path)


@dataclass
class RunningSshServer:
    """A local test SSH server bound to loopback."""

    host: str
    port: int
    server: asyncssh.SSHAcceptor
    host_public_key: str
    client_private_key: str

    async def close(self) -> None:
        self.server.close()
        await self.server.wait_closed()


class _EchoServer(asyncssh.SSHServer):
    def connection_made(self, conn: asyncssh.SSHServerConnection) -> None:
        self._conn = conn

    def begin_auth(self, username: str) -> bool:
        # Require public-key auth (return True = auth required).
        return True


async def _handle_session(process: asyncssh.SSHServerProcess) -> None:
    # Echo the requested command back; enough to prove connectivity.
    command = process.command or ""
    process.stdout.write(f"ran: {command}")
    process.exit(0)


async def start_test_ssh_server() -> RunningSshServer:
    """Start an asyncssh server on 127.0.0.1 that accepts one client key."""
    host_key = asyncssh.generate_private_key("ssh-ed25519")
    client_key = asyncssh.generate_private_key("ssh-ed25519")
    authorized = client_key.export_public_key().decode()

    server = await asyncssh.create_server(
        _EchoServer,
        host="127.0.0.1",
        port=0,
        server_host_keys=[host_key],
        authorized_client_keys=asyncssh.import_authorized_keys(authorized),
        process_factory=_handle_session,
    )
    port = server.get_addresses()[0][1]
    return RunningSshServer(
        host="127.0.0.1",
        port=port,
        server=server,
        host_public_key=host_key.export_public_key().decode(),
        client_private_key=client_key.export_private_key().decode(),
    )
