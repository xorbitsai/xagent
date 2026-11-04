"""End-to-end sandbox SSH path (§21.3 / §15.2) against a real local sshd.

A local-exec sandbox double runs the *real* ``ssh``/``sftp`` binaries (the only
thing that differs from a container is where the process runs), so this exercises
the whole chain the way production does: SandboxTmpfsSecretMaterializer writes the
key + known_hosts, SandboxSshRunner builds the strict argv and pins the vetted IP
with HostKeyAlias, and the executor leases/cleans up. Requires ``ssh``/``sftp`` on
PATH; the ``timeout`` wrapper is stripped by the double (absent on macOS,
its expiry mapping is unit-tested separately).
"""

from __future__ import annotations

import asyncio
import os
import shutil
from contextlib import asynccontextmanager

import pytest

from tests.core.ssh.helpers import (
    InMemorySshSecretStore,
    InMemorySshTargetProvider,
    RunningSshServer,
    start_test_ssh_server,
)
from xagent.core.ssh import SshError, SshErrorCode
from xagent.core.ssh.egress import EgressPolicyConfig
from xagent.core.ssh.executor import SshExecutor
from xagent.core.ssh.sandbox_materializer import SandboxTmpfsSecretMaterializer
from xagent.core.ssh.sandbox_runner import SandboxSshRunner
from xagent.core.ssh.types import (
    ActorRef,
    PrincipalRef,
    ResolvedSshTarget,
    SensitiveSshCredential,
    SshExecutionContext,
    SshSecretHandle,
)
from xagent.sandbox.base import ExecResult

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        shutil.which("ssh") is None or shutil.which("sftp") is None,
        reason="requires OpenSSH ssh/sftp binaries",
    ),
]

_ALLOW_LOOPBACK = EgressPolicyConfig(allow_cidrs=("127.0.0.0/8",))


class _LocalExecSandbox:
    """Sandbox double: runs argv as host subprocesses and does file ops on the
    host fs. Strips a leading ``timeout <n>`` wrapper (macOS lacks `timeout`).
    Records exec argv so tests can assert no secret ever reaches the command line."""

    def __init__(self) -> None:
        self.execs: list[tuple[str, ...]] = []

    async def exec(
        self, command: str, *args: str, env=None, max_output_bytes=None
    ) -> ExecResult:
        self.execs.append((command, *args))
        argv = [command, *args]
        if argv and argv[0] == "timeout":
            argv = argv[2:]
        proc = await asyncio.create_subprocess_exec(
            *argv, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        out, err = await proc.communicate()
        return ExecResult(
            exit_code=proc.returncode or 0,
            stdout=out.decode("utf-8", "replace"),
            stderr=err.decode("utf-8", "replace"),
        )

    async def write_file(
        self, content: str, remote_path: str, overwrite: bool = False
    ) -> None:
        os.makedirs(os.path.dirname(remote_path), exist_ok=True)
        with open(remote_path, "w") as fh:
            fh.write(content)

    async def upload_file(
        self, local_path: str, remote_path: str, overwrite: bool = False
    ) -> None:
        os.makedirs(os.path.dirname(remote_path), exist_ok=True)
        shutil.copyfile(local_path, remote_path)

    async def download_file(
        self, remote_path: str, local_path: str, overwrite: bool = False
    ) -> None:
        if not overwrite and os.path.exists(local_path):
            raise FileExistsError(local_path)
        shutil.copyfile(remote_path, local_path)


def _ctx() -> SshExecutionContext:
    return SshExecutionContext(
        actor=ActorRef(actor_type="user", actor_id="u"),
        execution_principal=PrincipalRef(principal_type="user", principal_id="u"),
        agent_id=1,
        task_id=1,
        turn_id=None,
        request_id="r",
    )


def _known_hosts(server: RunningSshServer, *, host_key: str | None = None) -> str:
    # Bare-hostname key: the sandbox runner verifies under HostKeyAlias=<hostname>,
    # which OpenSSH looks up by the bare alias (no [host]:port bracket).
    algo, blob = (host_key or server.host_public_key).split()[:2]
    return f"{server.host} {algo} {blob}\n"


def _target(
    server: RunningSshServer, *, capabilities, known_hosts=None
) -> ResolvedSshTarget:
    return ResolvedSshTarget(
        target_public_id="t",
        hostname=server.host,
        port=server.port,
        username="deploy",
        remote_root=None,
        capabilities=capabilities,
        approval_policy="not_required",
        secret_handle=SshSecretHandle(credential_id="c", version_id="v"),
        known_hosts=known_hosts if known_hosts is not None else _known_hosts(server),
        credential_public_id="c",
        credential_version_id="v",
        host_key_fingerprint="SHA256:x",
    )


def _executor(
    server: RunningSshServer, target: ResolvedSshTarget, sandbox, tmp_path
) -> SshExecutor:
    @asynccontextmanager
    async def _lease():
        yield sandbox

    return SshExecutor(
        provider=InMemorySshTargetProvider({(1, "prod"): target}),
        secret_store=InMemorySshSecretStore(
            {
                "v": SensitiveSshCredential(
                    server.client_private_key.encode(), "", "ssh-ed25519"
                )
            }
        ),
        materializer=SandboxTmpfsSecretMaterializer(secret_root=str(tmp_path)),
        runner=SandboxSshRunner(secret_root=str(tmp_path)),
        egress_config=_ALLOW_LOOPBACK,
        sandbox_lease=_lease,
    )


async def test_execute_end_to_end_over_real_ssh(tmp_path) -> None:
    server = await start_test_ssh_server()
    sandbox = _LocalExecSandbox()
    try:
        outcome = await _executor(
            server,
            _target(server, capabilities=frozenset({"execute"})),
            sandbox,
            tmp_path,
        ).execute(_ctx(), target_alias="prod", command="uptime", timeout_seconds=10)
        assert outcome.exit_code == 0
        assert "ran: uptime" in outcome.stdout
        # The private key never appeared on any command line.
        joined = " ".join(part for call in sandbox.execs for part in call)
        assert "BEGIN OPENSSH PRIVATE KEY" not in joined
        # tmpfs secret dir was removed on exit — no key left behind on disk.
        assert not any(p.name == "id_key" for p in tmp_path.rglob("id_key"))
    finally:
        await server.close()


async def test_wrong_host_key_fails_before_command(tmp_path) -> None:
    server = await start_test_ssh_server()
    other = await start_test_ssh_server()  # borrow a different, non-matching host key
    sandbox = _LocalExecSandbox()
    try:
        target = _target(
            server,
            capabilities=frozenset({"execute"}),
            known_hosts=_known_hosts(server, host_key=other.host_public_key),
        )
        with pytest.raises(SshError) as exc:
            await _executor(server, target, sandbox, tmp_path).execute(
                _ctx(), target_alias="prod", command="uptime", timeout_seconds=10
            )
        assert exc.value.code == SshErrorCode.HOST_KEY_MISMATCH
    finally:
        await server.close()
        await other.close()


async def test_upload_then_download_roundtrip(tmp_path) -> None:
    server = await start_test_ssh_server()
    sandbox = _LocalExecSandbox()
    src = tmp_path / "src.txt"
    src.write_text("payload-via-sandbox")
    remote = tmp_path / "remote.txt"
    fetched = tmp_path / "fetched.txt"
    try:
        ex = _executor(
            server,
            _target(server, capabilities=frozenset({"upload", "download"})),
            sandbox,
            tmp_path,
        )
        # overwrite=True skips the remote-existence probe: the echo test server
        # returns exit 0 for any exec (incl. `test -e`), so the probe would
        # false-positive. The probe's own logic is covered by the unit tests.
        await ex.upload(
            _ctx(),
            target_alias="prod",
            local_path=str(src),
            remote_path=str(remote),
            overwrite=True,
        )
        assert remote.read_text() == "payload-via-sandbox"
        await ex.download(
            _ctx(),
            target_alias="prod",
            remote_path=str(remote),
            local_path=str(fetched),
        )
        assert fetched.read_text() == "payload-via-sandbox"
    finally:
        await server.close()
