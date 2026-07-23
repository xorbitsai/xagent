"""asyncssh-backed SSH runner (Phase 3).

Establishes a connection with the strict security configuration required by the
design (§14): host-key verification against a pinned known_hosts file,
public-key-only auth, no ssh-agent, no forwarding, and no reading of the user's
ssh config. Command execution only for now; SFTP transfer arrives in a later
part. This is one implementation of the runner seam; a sandbox ssh-binary
runner (design §15.2) can implement the same shape later.
"""

from __future__ import annotations

import asyncio
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
    implementation runs in-process; a sandbox ssh-binary runner can implement
    the same shape (design §15.2)."""

    async def execute(
        self,
        *,
        hostname: str,
        port: int,
        username: str,
        private_key_path: str,
        known_hosts_path: str,
        command: str,
        timeout_seconds: int,
        egress_config: EgressPolicyConfig,
    ) -> SshRunResult: ...


class AsyncsshRunner:
    """Runs commands over SSH with strict, non-interactive security settings."""

    async def execute(
        self,
        *,
        hostname: str,
        port: int,
        username: str,
        private_key_path: str,
        known_hosts_path: str,
        command: str,
        timeout_seconds: int,
        egress_config: EgressPolicyConfig,
    ) -> SshRunResult:
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
                # unknown host raises before any command runs.
                known_hosts=known_hosts_path,
                client_keys=[private_key_path],
                # No ssh-agent, no ssh config, no forwarding, public-key only.
                agent_path=None,
                config=None,
                agent_forwarding=False,
                x509_trusted_certs=None,
                x509_trusted_cert_paths=None,
                preferred_auth=["publickey"],
            ) as conn:
                # DNS-rebinding defense: re-check the IP actually connected to,
                # and refuse before running anything if the policy denies it.
                _authorize_peer(conn, egress_config)
                result = await asyncio.wait_for(
                    conn.run(command, check=False), timeout=timeout_seconds
                )
        except asyncssh.HostKeyNotVerifiable as exc:
            # Message is deliberately generic — no host key material.
            raise SshError(
                SshErrorCode.HOST_KEY_MISMATCH,
                "host key verification failed",
                cause=exc,
            ) from exc
        except TimeoutError as exc:
            raise SshError(
                SshErrorCode.COMMAND_TIMEOUT,
                "command timed out",
                cause=exc,
            ) from exc

        return SshRunResult(
            exit_code=result.exit_status if result.exit_status is not None else -1,
            stdout=_as_text(result.stdout),
            stderr=_as_text(result.stderr),
            truncated=False,
        )


def _authorize_peer(conn: asyncssh.SSHClientConnection, config: EgressPolicyConfig) -> None:
    """Re-check the connected peer IP against the egress policy. Raises
    EGRESS_DENIED if the actual peer is not permitted (closes on context exit)."""
    peername = conn.get_extra_info("peername")
    peer_ip = peername[0] if peername else ""
    decision = check_ip(peer_ip, config)
    if not decision.allowed:
        raise SshError(SshErrorCode.EGRESS_DENIED, "connection peer denied by egress policy")


def _as_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)
