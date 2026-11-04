"""asyncssh-backed SSH runner (Phase 3).

Establishes a connection with the strict security configuration required by the
design (§14): host-key verification against a pinned known_hosts file,
public-key-only auth, no ssh-agent, no forwarding, and no reading of the user's
ssh config. Supports command execution and SFTP upload/download. This is one
implementation of the runner seam; a sandbox ssh-binary runner (design §15.2)
can implement the same shape later.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from .egress import EgressPolicyConfig, check_ip
from .errors import SshError, SshErrorCode

if TYPE_CHECKING:
    import asyncssh


@dataclass(frozen=True)
class SshRunResult:
    """Outcome of a remote command. Carries no secret material."""

    exit_code: int
    stdout: str
    stderr: str
    truncated: bool


@runtime_checkable
class SshRunner(Protocol):
    """Runs a command against a materialized key + known_hosts. The asyncssh
    implementation runs in-process; a sandbox ssh-binary runner runs inside a
    leased sandbox (design §15.2).

    ``sandbox`` is the leased sandbox for this call (None for the in-process
    runner). ``connect_ip`` is the egress-authorized address the executor
    resolved; the sandbox runner connects to it (with the hostname as a
    HostKeyAlias) to pin the vetted IP. The in-process runner ignores it and
    re-checks the peer itself."""

    async def execute(
        self,
        *,
        sandbox: object | None,
        hostname: str,
        connect_ip: str,
        port: int,
        username: str,
        private_key_path: str,
        known_hosts_path: str,
        command: str,
        timeout_seconds: int,
        egress_config: EgressPolicyConfig,
        max_output_bytes: int,
    ) -> SshRunResult: ...

    async def upload(
        self,
        *,
        sandbox: object | None,
        hostname: str,
        connect_ip: str,
        port: int,
        username: str,
        private_key_path: str,
        known_hosts_path: str,
        local_path: str,
        remote_path: str,
        overwrite: bool,
        egress_config: EgressPolicyConfig,
        timeout_seconds: int,
    ) -> None: ...

    async def download(
        self,
        *,
        sandbox: object | None,
        hostname: str,
        connect_ip: str,
        port: int,
        username: str,
        private_key_path: str,
        known_hosts_path: str,
        remote_path: str,
        local_path: str,
        overwrite: bool,
        egress_config: EgressPolicyConfig,
        timeout_seconds: int,
    ) -> None: ...


class AsyncsshRunner:
    """Runs commands and SFTP transfers over SSH with strict, non-interactive
    security settings, in-process (self-hosted, no sandbox subsystem).

    Accepts ``sandbox`` and ``connect_ip`` for seam parity but ignores them: it
    connects by hostname and re-checks the actual peer IP (``_authorize_peer``)
    as its own DNS-rebinding backstop."""

    @asynccontextmanager
    async def _connect(
        self,
        *,
        hostname: str,
        port: int,
        username: str,
        private_key_path: str,
        known_hosts_path: str,
        egress_config: EgressPolicyConfig,
        timeout_seconds: int,
    ) -> AsyncIterator[asyncssh.SSHClientConnection]:
        # Imported lazily so merely loading this module (and thus the SSH MCP
        # tools) never hard-requires asyncssh: agents that don't execute SSH,
        # and environments without the dep, still load every other tool.
        import asyncssh

        try:
            async with asyncssh.connect(
                hostname,
                port=port,
                username=username,
                # Pinned host key file → strict verification; a mismatch or an
                # unknown host raises before anything runs.
                known_hosts=known_hosts_path,
                client_keys=[private_key_path],
                # No ssh-agent, no ssh config, no forwarding, public-key only.
                agent_path=None,
                config=None,
                agent_forwarding=False,
                x509_trusted_certs=None,
                x509_trusted_cert_paths=None,
                preferred_auth=["publickey"],
                # Bound the TCP-connect and auth handshake; without these a slow
                # or malicious target stalls the worker indefinitely, since the
                # per-op wait_for below only wraps the post-connect work (M1).
                connect_timeout=timeout_seconds,
                login_timeout=timeout_seconds,
            ) as conn:
                # DNS-rebinding defense: re-check the IP actually connected to,
                # and refuse before doing anything if the policy denies it.
                _authorize_peer(conn, egress_config)
                yield conn
        except asyncssh.HostKeyNotVerifiable as exc:
            # Message is deliberately generic — no host key material.
            raise SshError(
                SshErrorCode.HOST_KEY_MISMATCH,
                "host key verification failed",
                cause=exc,
            ) from exc
        except (asyncssh.Error, OSError, TimeoutError) as exc:
            # PermissionDenied, ConnectionLost (asyncssh raises this on
            # connect_timeout/login_timeout expiry too), connection-refused
            # (OSError), etc. all otherwise escape raw — skipping the executor's
            # failure audit and the tool wrapper. Map them to one stable code so
            # both runners fail identically. TimeoutError is kept only as a
            # defensive backstop. (SshError from _authorize_peer is none of
            # these, so it propagates.)
            raise SshError(
                SshErrorCode.CONNECTION_FAILED,
                "ssh connection failed",
                cause=exc,
            ) from exc

    async def execute(
        self,
        *,
        sandbox: object | None = None,
        hostname: str,
        connect_ip: str | None = None,
        port: int,
        username: str,
        private_key_path: str,
        known_hosts_path: str,
        command: str,
        timeout_seconds: int,
        egress_config: EgressPolicyConfig,
        max_output_bytes: int,
    ) -> SshRunResult:
        async with self._connect(
            hostname=hostname,
            port=port,
            username=username,
            private_key_path=private_key_path,
            known_hosts_path=known_hosts_path,
            egress_config=egress_config,
            timeout_seconds=timeout_seconds,
        ) as conn:
            try:
                out, err, truncated, exit_status = await asyncio.wait_for(
                    _run_capped(conn, command, max_output_bytes),
                    timeout=timeout_seconds,
                )
            except TimeoutError as exc:
                raise SshError(
                    SshErrorCode.COMMAND_TIMEOUT,
                    "command timed out",
                    cause=exc,
                ) from exc

        return SshRunResult(
            exit_code=exit_status if exit_status is not None else -1,
            stdout=out.decode("utf-8", "replace"),
            stderr=err.decode("utf-8", "replace"),
            truncated=truncated,
        )

    async def upload(
        self,
        *,
        sandbox: object | None = None,
        hostname: str,
        connect_ip: str | None = None,
        port: int,
        username: str,
        private_key_path: str,
        known_hosts_path: str,
        local_path: str,
        remote_path: str,
        overwrite: bool,
        egress_config: EgressPolicyConfig,
        timeout_seconds: int,
    ) -> None:
        async with (
            self._connect(
                hostname=hostname,
                port=port,
                username=username,
                private_key_path=private_key_path,
                known_hosts_path=known_hosts_path,
                egress_config=egress_config,
                timeout_seconds=timeout_seconds,
            ) as conn,
            conn.start_sftp_client() as sftp,
        ):
            # Bound the whole transfer under one shared budget — a stalled peer
            # must not hang the worker, and the exists-check + put must not each
            # get a full timeout (m3). The connect handshake is separately
            # bounded in _connect (M1).
            async def _put() -> None:
                if not overwrite and await sftp.exists(remote_path):
                    raise SshError(
                        SshErrorCode.OPERATION_NOT_ALLOWED,
                        "remote destination already exists",
                    )
                await sftp.put(local_path, remote_path)

            try:
                await asyncio.wait_for(_put(), timeout_seconds)
            except TimeoutError as exc:
                raise SshError(
                    SshErrorCode.COMMAND_TIMEOUT, "transfer timed out", cause=exc
                ) from exc

    async def download(
        self,
        *,
        sandbox: object | None = None,
        hostname: str,
        connect_ip: str | None = None,
        port: int,
        username: str,
        private_key_path: str,
        known_hosts_path: str,
        remote_path: str,
        local_path: str,
        overwrite: bool,
        egress_config: EgressPolicyConfig,
        timeout_seconds: int,
    ) -> None:
        if not overwrite and os.path.exists(local_path):
            raise SshError(
                SshErrorCode.OPERATION_NOT_ALLOWED, "local destination already exists"
            )
        async with (
            self._connect(
                hostname=hostname,
                port=port,
                username=username,
                private_key_path=private_key_path,
                known_hosts_path=known_hosts_path,
                egress_config=egress_config,
                timeout_seconds=timeout_seconds,
            ) as conn,
            conn.start_sftp_client() as sftp,
        ):
            try:
                await asyncio.wait_for(
                    sftp.get(remote_path, local_path), timeout_seconds
                )
            except TimeoutError as exc:
                raise SshError(
                    SshErrorCode.COMMAND_TIMEOUT, "transfer timed out", cause=exc
                ) from exc


async def _run_capped(
    conn: asyncssh.SSHClientConnection, command: str, cap: int
) -> tuple[bytes, bytes, bool, int | None]:
    """Run ``command`` and read stdout/stderr concurrently, stopping each at
    ``cap`` bytes so a flood of remote output can't grow the host process
    unbounded (#3). asyncssh has no native byte limit and ``conn.run`` buffers
    the whole stream, so we drive a process and drain it ourselves. Returns
    (stdout, stderr, truncated, exit_status)."""
    async with conn.create_process(command, encoding=None) as proc:
        (out, out_truncated), (err, err_truncated) = await asyncio.gather(
            _drain(proc.stdout, cap), _drain(proc.stderr, cap)
        )
        truncated = out_truncated or err_truncated
        # Only wait for the exit status when we read to EOF; if we stopped early
        # the remote may keep writing forever, so don't block on it. This means
        # exit_status is None (→ exit_code -1 in execute()) whenever truncated
        # is True — the exit code is meaningless in that case, so callers must
        # read it together with the truncated flag (m2). The process is killed
        # by create_process's context exit either way.
        if not truncated:
            await proc.wait()
        return out, err, truncated, proc.exit_status


async def _drain(reader: asyncssh.SSHReader[bytes], cap: int) -> tuple[bytes, bool]:
    """Read up to ``cap`` bytes from ``reader``; report whether more remained."""
    buf = bytearray()
    while len(buf) < cap:
        chunk = await reader.read(cap - len(buf))
        if not chunk:
            return bytes(buf), False
        buf.extend(chunk)
    # Cap hit: peek one more byte to flag truncation, then drop the rest.
    extra = await reader.read(1)
    return bytes(buf), bool(extra)


def _authorize_peer(
    conn: asyncssh.SSHClientConnection, config: EgressPolicyConfig
) -> None:
    """Re-check the connected peer IP against the egress policy. Raises
    EGRESS_DENIED if the actual peer is not permitted (closes on context exit)."""
    peername = conn.get_extra_info("peername")
    peer_ip = peername[0] if peername else ""
    decision = check_ip(peer_ip, config)
    if not decision.allowed:
        raise SshError(
            SshErrorCode.EGRESS_DENIED, "connection peer denied by egress policy"
        )
