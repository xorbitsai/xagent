"""Integration tests for the asyncssh SSH runner (Phase 3).

Exercises the real security config against a local asyncssh server: strict
host-key verification via a known_hosts file, public-key-only auth, and command
execution with captured stdout/stderr/exit code.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pytest

from tests.core.ssh.helpers import (
    LocalTmpSecretMaterializer,
    RunningSshServer,
    start_test_ssh_server,
)
from xagent.core.ssh import SshError, SshErrorCode
from xagent.core.ssh.egress import EgressPolicyConfig
from xagent.core.ssh.runner import AsyncsshRunner
from xagent.core.ssh.types import SensitiveSshCredential

pytestmark = pytest.mark.integration

# The test SSH server binds loopback, which the default egress policy denies
# (loopback is also "private"); tests that need a successful connection
# allowlist it explicitly — the allowlist wins over every deny rule.
_ALLOW_LOOPBACK = EgressPolicyConfig(allow_cidrs=("127.0.0.0/8",))


def _known_hosts_line(host: str, port: int, host_public_key: str) -> str:
    algo, blob = host_public_key.split()[:2]
    token = host if port == 22 else f"[{host}]:{port}"
    return f"{token} {algo} {blob}\n"


@asynccontextmanager
async def _materialized(
    server: RunningSshServer, *, host_public_key: str | None = None
) -> AsyncIterator[tuple[str, str]]:
    """Materialize the client key + a known_hosts pinning the given host key
    (defaults to the server's real key). Yields (private_key_path, known_hosts_path)."""
    known_hosts = _known_hosts_line(
        server.host, server.port, host_public_key or server.host_public_key
    )
    cred = SensitiveSshCredential(
        private_key=server.client_private_key.encode("utf-8"),
        public_key="",
        key_algorithm="ssh-ed25519",
    )
    async with LocalTmpSecretMaterializer().materialize_ssh(
        None, cred, known_hosts
    ) as paths:
        yield paths.private_key_path, paths.known_hosts_path


async def test_execute_runs_command_over_ssh() -> None:
    server = await start_test_ssh_server()
    try:
        async with _materialized(server) as (key_path, known_hosts_path):
            result = await AsyncsshRunner().execute(
                hostname=server.host,
                port=server.port,
                username="deploy",
                private_key_path=key_path,
                known_hosts_path=known_hosts_path,
                command="uptime",
                timeout_seconds=10,
                egress_config=_ALLOW_LOOPBACK,
                max_output_bytes=1 << 20,
            )
        assert result.exit_code == 0
        assert result.stdout == "ran: uptime"
        assert result.truncated is False
    finally:
        await server.close()


async def test_execute_denied_peer_ip_rejected() -> None:
    # Host key is valid, but the connected peer (loopback) is denied by the
    # egress policy — the runner must reject after connect, before running.
    server = await start_test_ssh_server()
    try:
        async with _materialized(server) as (key_path, known_hosts_path):
            with pytest.raises(SshError) as exc:
                await AsyncsshRunner().execute(
                    hostname=server.host,
                    port=server.port,
                    username="deploy",
                    private_key_path=key_path,
                    known_hosts_path=known_hosts_path,
                    command="uptime",
                    timeout_seconds=10,
                    egress_config=EgressPolicyConfig(),
                    max_output_bytes=1 << 20,
                )
        assert exc.value.code == SshErrorCode.EGRESS_DENIED
    finally:
        await server.close()


async def test_execute_times_out() -> None:
    server = await start_test_ssh_server()
    try:
        async with _materialized(server) as (key_path, known_hosts_path):
            with pytest.raises(SshError) as exc:
                await AsyncsshRunner().execute(
                    hostname=server.host,
                    port=server.port,
                    username="deploy",
                    private_key_path=key_path,
                    known_hosts_path=known_hosts_path,
                    command="sleep 10",
                    timeout_seconds=1,
                    egress_config=_ALLOW_LOOPBACK,
                    max_output_bytes=1 << 20,
                )
        assert exc.value.code == SshErrorCode.COMMAND_TIMEOUT
    finally:
        await server.close()


async def test_connect_passes_timeout_kwargs(monkeypatch) -> None:
    # The connect/handshake phase must be time-bounded, not just the post-connect
    # work, so connect_timeout/login_timeout must reach asyncssh.connect (M1).
    import asyncssh

    captured: dict[str, object] = {}
    real_connect = asyncssh.connect

    def spy_connect(*args, **kwargs):
        captured.update(kwargs)
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(asyncssh, "connect", spy_connect)
    server = await start_test_ssh_server()
    try:
        async with _materialized(server) as (key_path, known_hosts_path):
            await AsyncsshRunner().execute(
                hostname=server.host,
                port=server.port,
                username="deploy",
                private_key_path=key_path,
                known_hosts_path=known_hosts_path,
                command="uptime",
                timeout_seconds=7,
                egress_config=_ALLOW_LOOPBACK,
                max_output_bytes=1 << 20,
            )
        assert captured.get("connect_timeout") == 7
        assert captured.get("login_timeout") == 7
    finally:
        await server.close()


async def test_connect_failure_maps_to_connection_failed(monkeypatch) -> None:
    # A stalled/failed handshake (here a connect timeout) must map to the stable
    # CONNECTION_FAILED code, not escape raw (M1).
    import asyncssh

    def boom(*args, **kwargs):
        raise TimeoutError("connect timed out")

    monkeypatch.setattr(asyncssh, "connect", boom)
    server = await start_test_ssh_server()
    try:
        async with _materialized(server) as (key_path, known_hosts_path):
            with pytest.raises(SshError) as exc:
                await AsyncsshRunner().execute(
                    hostname=server.host,
                    port=server.port,
                    username="deploy",
                    private_key_path=key_path,
                    known_hosts_path=known_hosts_path,
                    command="uptime",
                    timeout_seconds=1,
                    egress_config=_ALLOW_LOOPBACK,
                    max_output_bytes=1 << 20,
                )
        assert exc.value.code == SshErrorCode.CONNECTION_FAILED
    finally:
        await server.close()


async def test_upload_transfers_local_file_to_remote(tmp_path) -> None:
    server = await start_test_ssh_server()
    local = tmp_path / "local.txt"
    local.write_text("payload-up")
    remote = tmp_path / "remote.txt"
    try:
        async with _materialized(server) as (key_path, known_hosts_path):
            await AsyncsshRunner().upload(
                hostname=server.host,
                port=server.port,
                username="deploy",
                private_key_path=key_path,
                known_hosts_path=known_hosts_path,
                local_path=str(local),
                remote_path=str(remote),
                overwrite=False,
                egress_config=_ALLOW_LOOPBACK,
                timeout_seconds=30,
            )
        assert remote.read_text() == "payload-up"
    finally:
        await server.close()


async def test_download_transfers_remote_file_to_local(tmp_path) -> None:
    server = await start_test_ssh_server()
    remote = tmp_path / "remote.txt"
    remote.write_text("payload-down")
    local = tmp_path / "local.txt"
    try:
        async with _materialized(server) as (key_path, known_hosts_path):
            await AsyncsshRunner().download(
                hostname=server.host,
                port=server.port,
                username="deploy",
                private_key_path=key_path,
                known_hosts_path=known_hosts_path,
                remote_path=str(remote),
                local_path=str(local),
                overwrite=False,
                egress_config=_ALLOW_LOOPBACK,
                timeout_seconds=30,
            )
        assert local.read_text() == "payload-down"
    finally:
        await server.close()


async def test_upload_refuses_existing_remote_without_overwrite(tmp_path) -> None:
    server = await start_test_ssh_server()
    local = tmp_path / "local.txt"
    local.write_text("new")
    remote = tmp_path / "remote.txt"
    remote.write_text("existing")
    try:
        async with _materialized(server) as (key_path, known_hosts_path):
            with pytest.raises(SshError) as exc:
                await AsyncsshRunner().upload(
                    hostname=server.host,
                    port=server.port,
                    username="deploy",
                    private_key_path=key_path,
                    known_hosts_path=known_hosts_path,
                    local_path=str(local),
                    remote_path=str(remote),
                    overwrite=False,
                    egress_config=_ALLOW_LOOPBACK,
                    timeout_seconds=30,
                )
        assert exc.value.code == SshErrorCode.OPERATION_NOT_ALLOWED
        assert remote.read_text() == "existing"  # unchanged
    finally:
        await server.close()


async def test_execute_wrong_host_key_fails_before_auth() -> None:
    server = await start_test_ssh_server()
    # Pin a different (bogus) host key so verification must fail.
    import asyncssh

    bogus = asyncssh.generate_private_key("ssh-ed25519").export_public_key().decode()
    try:
        async with _materialized(server, host_public_key=bogus) as (
            key_path,
            known_hosts_path,
        ):
            with pytest.raises(SshError) as exc:
                await AsyncsshRunner().execute(
                    hostname=server.host,
                    port=server.port,
                    username="deploy",
                    private_key_path=key_path,
                    known_hosts_path=known_hosts_path,
                    command="uptime",
                    timeout_seconds=10,
                    egress_config=_ALLOW_LOOPBACK,
                    max_output_bytes=1 << 20,
                )
        assert exc.value.code == SshErrorCode.HOST_KEY_MISMATCH
    finally:
        await server.close()


async def test_execute_caps_large_output() -> None:
    # Output is capped as it is read, not buffered whole then trimmed (#3). The
    # test server echoes "ran: uptime" (11 bytes); a 5-byte cap cuts it short.
    server = await start_test_ssh_server()
    try:
        async with _materialized(server) as (key_path, known_hosts_path):
            result = await AsyncsshRunner().execute(
                hostname=server.host,
                port=server.port,
                username="deploy",
                private_key_path=key_path,
                known_hosts_path=known_hosts_path,
                command="uptime",
                timeout_seconds=10,
                egress_config=_ALLOW_LOOPBACK,
                max_output_bytes=5,
            )
        assert result.truncated is True
        assert len(result.stdout.encode("utf-8")) == 5
    finally:
        await server.close()


async def test_transfer_times_out(tmp_path) -> None:
    # A stalled in-process transfer must be bounded like execute(), not hang the
    # worker forever (N1). A zero timeout forces the wait_for to expire.
    server = await start_test_ssh_server()
    local = tmp_path / "local.txt"
    local.write_text("x")
    remote = tmp_path / "remote.txt"
    try:
        async with _materialized(server) as (key_path, known_hosts_path):
            with pytest.raises(SshError) as exc:
                await AsyncsshRunner().upload(
                    hostname=server.host,
                    port=server.port,
                    username="deploy",
                    private_key_path=key_path,
                    known_hosts_path=known_hosts_path,
                    local_path=str(local),
                    remote_path=str(remote),
                    overwrite=True,
                    egress_config=_ALLOW_LOOPBACK,
                    timeout_seconds=0,
                )
        assert exc.value.code == SshErrorCode.COMMAND_TIMEOUT
    finally:
        await server.close()
