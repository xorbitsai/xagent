"""SSH runner that executes the OpenSSH binary inside a leased sandbox (§15.2).

The in-process ``AsyncsshRunner`` connects from the backend; this runner instead
runs ``ssh``/``sftp`` inside an agent-inaccessible sandbox where the materializer
put the key. It applies the strict, non-interactive OpenSSH configuration the
design mandates (§14) via argv (never a local shell), pins the egress-authorized
IP while verifying the host key under the hostname (``HostKeyAlias``), and clamps
runtime with the ``timeout`` coreutil.

SFTP transfers stage through the sandbox: the sandbox has no task workspace, so
upload copies the host file into a private sandbox dir first (and download copies
the fetched file back out), cleaning the staging dir on every exit path.
"""

from __future__ import annotations

import os
import secrets
import shlex
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress

from .errors import SshError, SshErrorCode
from .runner import SshRunResult

_TIMEOUT_EXIT = 124  # `timeout` coreutil signals expiry
_SSH_TRANSPORT_EXIT = 255  # ssh/sftp connection/auth/host-key failure
_TRANSFER_TIMEOUT_SECONDS = 300
# Not /dev/shm: Docker's archive API can't extract into a tmpfs mount (see the
# materializer). A normal sandbox-private dir, cleaned per transfer.
_SECRET_ROOT = "/tmp"  # noqa: S108 — sandbox-private, cleaned per transfer


def _common_ssh_opts(
    *, hostname: str, port: int, private_key_path: str, known_hosts_path: str
) -> list[str]:
    """The strict, non-interactive OpenSSH options shared by ssh/sftp (§14).
    ``-o Port=`` is used uniformly so ssh (-p) and sftp/scp (-P) never diverge."""
    return [
        "-F",
        "/dev/null",  # ignore any user ssh config
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "IdentityAgent=none",
        "-o",
        "GlobalKnownHostsFile=/dev/null",
        "-o",
        "UpdateHostKeys=no",
        "-o",
        "PasswordAuthentication=no",
        "-o",
        "KbdInteractiveAuthentication=no",
        "-o",
        "ForwardAgent=no",
        "-o",
        "ClearAllForwardings=yes",
        "-o",
        "PermitLocalCommand=no",
        "-o",
        f"UserKnownHostsFile={known_hosts_path}",
        # Verify the host key under the hostname while connecting to the pinned IP.
        "-o",
        f"HostKeyAlias={hostname}",
        "-o",
        f"Port={port}",
        "-i",
        private_key_path,
    ]


def _ssh_argv(
    *,
    hostname: str,
    connect_ip: str,
    port: int,
    username: str,
    private_key_path: str,
    known_hosts_path: str,
    command: str | None = None,
) -> list[str]:
    """Full ``ssh`` argv connecting to the vetted IP. ``command`` (if given) is
    the trailing arg the remote shell interprets."""
    argv = [
        "ssh",
        *_common_ssh_opts(
            hostname=hostname,
            port=port,
            private_key_path=private_key_path,
            known_hosts_path=known_hosts_path,
        ),
        f"{username}@{connect_ip}",
    ]
    if command is not None:
        argv.append(command)
    return argv


def _raise_for_transport(exit_code: int, stderr: str) -> None:
    """Map exceptional ssh/sftp exits to stable, non-leaking SshErrors. A normal
    non-zero remote exit is NOT exceptional and is handled by the caller."""
    if exit_code == _TIMEOUT_EXIT:
        raise SshError(SshErrorCode.COMMAND_TIMEOUT, "operation timed out")
    if exit_code == _SSH_TRANSPORT_EXIT:
        if "Host key verification failed" in stderr:
            raise SshError(
                SshErrorCode.HOST_KEY_MISMATCH, "host key verification failed"
            )
        # Refused / unreachable / auth failure: one stable code, no raw detail.
        raise SshError(SshErrorCode.CONNECTION_TIMEOUT, "ssh connection failed")


class SandboxSshRunner:
    """Runs ssh/sftp inside the leased sandbox handed to each call.

    ``secret_root`` is where SFTP staging dirs are created (see the
    materializer); overridable for tests."""

    def __init__(self, *, secret_root: str = _SECRET_ROOT) -> None:
        self._secret_root = secret_root

    async def execute(
        self,
        *,
        sandbox: object,
        hostname: str,
        connect_ip: str,
        port: int,
        username: str,
        private_key_path: str,
        known_hosts_path: str,
        command: str,
        timeout_seconds: int,
        egress_config: object = None,
    ) -> SshRunResult:
        argv = _ssh_argv(
            hostname=hostname,
            connect_ip=connect_ip,
            port=port,
            username=username,
            private_key_path=private_key_path,
            known_hosts_path=known_hosts_path,
            command=command,
        )
        result = await sandbox.exec("timeout", str(timeout_seconds), *argv)  # type: ignore[attr-defined]
        _raise_for_transport(result.exit_code, result.stderr or "")
        return SshRunResult(
            exit_code=result.exit_code,
            stdout=result.stdout or "",
            stderr=result.stderr or "",
            truncated=False,
        )

    async def upload(
        self,
        *,
        sandbox: object,
        hostname: str,
        connect_ip: str,
        port: int,
        username: str,
        private_key_path: str,
        known_hosts_path: str,
        local_path: str,
        remote_path: str,
        overwrite: bool,
        egress_config: object = None,
    ) -> None:
        async with self._staging(sandbox) as stage:
            staged = f"{stage}/payload"
            # Host task-workspace file → sandbox (the sandbox has no workspace).
            await sandbox.upload_file(local_path, staged, overwrite=True)  # type: ignore[attr-defined]
            if not overwrite and await self._remote_exists(
                sandbox,
                hostname=hostname,
                connect_ip=connect_ip,
                port=port,
                username=username,
                private_key_path=private_key_path,
                known_hosts_path=known_hosts_path,
                remote_path=remote_path,
            ):
                raise SshError(
                    SshErrorCode.OPERATION_NOT_ALLOWED,
                    "remote destination already exists",
                )
            batch = f"put {shlex.quote(staged)} {shlex.quote(remote_path)}\n"
            await self._run_sftp(
                sandbox,
                batch=batch,
                stage=stage,
                hostname=hostname,
                connect_ip=connect_ip,
                port=port,
                username=username,
                private_key_path=private_key_path,
                known_hosts_path=known_hosts_path,
            )

    async def download(
        self,
        *,
        sandbox: object,
        hostname: str,
        connect_ip: str,
        port: int,
        username: str,
        private_key_path: str,
        known_hosts_path: str,
        remote_path: str,
        local_path: str,
        overwrite: bool,
        egress_config: object = None,
    ) -> None:
        if not overwrite and os.path.exists(local_path):
            raise SshError(
                SshErrorCode.OPERATION_NOT_ALLOWED, "local destination already exists"
            )
        async with self._staging(sandbox) as stage:
            staged = f"{stage}/payload"
            batch = f"get {shlex.quote(remote_path)} {shlex.quote(staged)}\n"
            await self._run_sftp(
                sandbox,
                batch=batch,
                stage=stage,
                hostname=hostname,
                connect_ip=connect_ip,
                port=port,
                username=username,
                private_key_path=private_key_path,
                known_hosts_path=known_hosts_path,
            )
            # Sandbox → host task workspace.
            await sandbox.download_file(staged, local_path, overwrite=overwrite)  # type: ignore[attr-defined]

    # --- internals ---------------------------------------------------------

    @asynccontextmanager
    async def _staging(self, sandbox: object) -> AsyncIterator[str]:
        """A private 0700 dir for one transfer, removed on every exit."""
        stage = f"{self._secret_root}/xagent-xfer-{secrets.token_hex(16)}"
        result = await sandbox.exec("mkdir", "-p", "-m", "700", stage)  # type: ignore[attr-defined]
        if getattr(result, "exit_code", 0) != 0:
            raise SshError(
                SshErrorCode.SANDBOX_UNAVAILABLE, "could not create staging directory"
            )
        try:
            yield stage
        finally:
            with suppress(Exception):
                await sandbox.exec("rm", "-rf", stage)  # type: ignore[attr-defined]

    async def _remote_exists(
        self,
        sandbox: object,
        *,
        hostname: str,
        connect_ip: str,
        port: int,
        username: str,
        private_key_path: str,
        known_hosts_path: str,
        remote_path: str,
    ) -> bool:
        argv = _ssh_argv(
            hostname=hostname,
            connect_ip=connect_ip,
            port=port,
            username=username,
            private_key_path=private_key_path,
            known_hosts_path=known_hosts_path,
            command=f"test -e {shlex.quote(remote_path)}",
        )
        result = await sandbox.exec("timeout", str(_TRANSFER_TIMEOUT_SECONDS), *argv)  # type: ignore[attr-defined]
        _raise_for_transport(result.exit_code, result.stderr or "")
        return bool(result.exit_code == 0)

    async def _run_sftp(
        self,
        sandbox: object,
        *,
        batch: str,
        stage: str,
        hostname: str,
        connect_ip: str,
        port: int,
        username: str,
        private_key_path: str,
        known_hosts_path: str,
    ) -> None:
        batch_path = f"{stage}/batch"
        await sandbox.write_file(content=batch, remote_path=batch_path, overwrite=True)  # type: ignore[attr-defined]
        argv = [
            "sftp",
            "-b",
            batch_path,
            *_common_ssh_opts(
                hostname=hostname,
                port=port,
                private_key_path=private_key_path,
                known_hosts_path=known_hosts_path,
            ),
            f"{username}@{connect_ip}",
        ]
        result = await sandbox.exec("timeout", str(_TRANSFER_TIMEOUT_SECONDS), *argv)  # type: ignore[attr-defined]
        _raise_for_transport(result.exit_code, result.stderr or "")
        if result.exit_code != 0:
            raise SshError(SshErrorCode.OPERATION_NOT_ALLOWED, "sftp transfer failed")
