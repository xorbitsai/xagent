"""Executor sandbox-lease boundary + connect_ip threading (Phase 3 §15.2).

Unit-level: no real SSH server. A fake lease, recording materializer and
recording runner assert that the executor leases one sandbox spanning
materialize+run, hands that same sandbox to both, threads a resolved+authorized
IP as ``connect_ip``, and always exits the lease (success and failure).
"""

from __future__ import annotations

from contextlib import asynccontextmanager

import pytest

from tests.core.ssh.helpers import InMemorySshSecretStore, InMemorySshTargetProvider
from xagent.core.ssh import SshError, SshErrorCode
from xagent.core.ssh.egress import EgressPolicyConfig
from xagent.core.ssh.executor import SshExecutor
from xagent.core.ssh.runner import SshRunResult
from xagent.core.ssh.types import (
    ActorRef,
    MaterializedSshPaths,
    PrincipalRef,
    ResolvedSshTarget,
    SensitiveSshCredential,
    SshExecutionContext,
    SshSecretHandle,
)

_ALLOW = EgressPolicyConfig(allow_cidrs=("203.0.113.0/24",))


async def _resolver(hostname: str, port: int) -> list[str]:
    return ["203.0.113.7"]


def _ctx() -> SshExecutionContext:
    return SshExecutionContext(
        actor=ActorRef(actor_type="user", actor_id="u"),
        execution_principal=PrincipalRef(principal_type="user", principal_id="u"),
        agent_id=1,
        task_id=42,
        turn_id=None,
        request_id="r",
    )


def _target(
    capabilities: frozenset[str] = frozenset({"execute", "upload", "download"}),
) -> ResolvedSshTarget:
    return ResolvedSshTarget(
        target_public_id="t",
        hostname="host.example",
        port=2222,
        username="deploy",
        remote_root=None,
        capabilities=capabilities,
        approval_policy="not_required",
        secret_handle=SshSecretHandle(credential_id="c", version_id="v"),
        known_hosts="host.example ssh-ed25519 AAAA\n",
        credential_public_id="c",
        credential_version_id="v",
        host_key_fingerprint="SHA256:x",
    )


class _FakeSandbox:
    pass


class _RecordingLease:
    """Zero-arg callable → async CM yielding a fixed sandbox; counts enter/exit."""

    def __init__(self, sandbox: object) -> None:
        self.sandbox = sandbox
        self.entered = 0
        self.exited = 0

    def __call__(self):
        return self._cm()

    @asynccontextmanager
    async def _cm(self):
        self.entered += 1
        try:
            yield self.sandbox
        finally:
            self.exited += 1


class _RecordingMaterializer:
    def __init__(self) -> None:
        self.sandbox: object = "unset"

    @asynccontextmanager
    async def materialize_ssh(self, sandbox, credential, known_hosts):
        self.sandbox = sandbox
        yield MaterializedSshPaths(private_key_path="/k", known_hosts_path="/kh")


class _RecordingRunner:
    def __init__(self, *, boom: bool = False) -> None:
        self.sandbox: object = "unset"
        self.connect_ip: str | None = None
        self._boom = boom

    async def execute(self, **kwargs) -> SshRunResult:
        self.sandbox = kwargs.get("sandbox", "missing")
        self.connect_ip = kwargs.get("connect_ip")
        if self._boom:
            raise SshError(SshErrorCode.COMMAND_TIMEOUT, "boom")
        return SshRunResult(exit_code=0, stdout="ok", stderr="", truncated=False)

    async def upload(self, **kwargs) -> None:
        self.sandbox = kwargs.get("sandbox", "missing")
        self.connect_ip = kwargs.get("connect_ip")

    async def download(self, **kwargs) -> None:
        self.sandbox = kwargs.get("sandbox", "missing")
        self.connect_ip = kwargs.get("connect_ip")


def _executor(*, runner, materializer, lease, target=None) -> SshExecutor:
    target = target or _target()
    return SshExecutor(
        provider=InMemorySshTargetProvider({(1, "prod"): target}),
        secret_store=InMemorySshSecretStore(
            {"v": SensitiveSshCredential(b"KEY", "", "ssh-ed25519")}
        ),
        materializer=materializer,
        runner=runner,
        egress_config=_ALLOW,
        resolver=_resolver,
        sandbox_lease=lease,
    )


async def test_execute_leases_sandbox_and_threads_it_to_both() -> None:
    sandbox = _FakeSandbox()
    lease = _RecordingLease(sandbox)
    mat = _RecordingMaterializer()
    runner = _RecordingRunner()
    await _executor(runner=runner, materializer=mat, lease=lease).execute(
        _ctx(), target_alias="prod", command="uptime", timeout_seconds=10
    )
    assert mat.sandbox is sandbox
    assert runner.sandbox is sandbox
    assert runner.connect_ip == "203.0.113.7"
    assert lease.entered == 1
    assert lease.exited == 1


async def test_lease_exited_on_runner_failure() -> None:
    lease = _RecordingLease(_FakeSandbox())
    await_raises = _executor(
        runner=_RecordingRunner(boom=True),
        materializer=_RecordingMaterializer(),
        lease=lease,
    )
    with pytest.raises(SshError):
        await await_raises.execute(
            _ctx(), target_alias="prod", command="uptime", timeout_seconds=10
        )
    assert lease.entered == 1
    assert lease.exited == 1


async def test_no_lease_uses_none_sandbox() -> None:
    """Self-hosted path: no sandbox_lease → sandbox is None (host materializer)."""
    mat = _RecordingMaterializer()
    runner = _RecordingRunner()
    ex = SshExecutor(
        provider=InMemorySshTargetProvider({(1, "prod"): _target()}),
        secret_store=InMemorySshSecretStore(
            {"v": SensitiveSshCredential(b"KEY", "", "ssh-ed25519")}
        ),
        materializer=mat,
        runner=runner,
        egress_config=_ALLOW,
        resolver=_resolver,
    )
    await ex.execute(_ctx(), target_alias="prod", command="uptime", timeout_seconds=10)
    assert mat.sandbox is None
    assert runner.sandbox is None


async def test_upload_threads_sandbox_and_connect_ip() -> None:
    sandbox = _FakeSandbox()
    lease = _RecordingLease(sandbox)
    runner = _RecordingRunner()
    await _executor(
        runner=runner, materializer=_RecordingMaterializer(), lease=lease
    ).upload(_ctx(), target_alias="prod", local_path="/tmp/x", remote_path="/tmp/y")
    assert runner.sandbox is sandbox
    assert runner.connect_ip == "203.0.113.7"
    assert lease.exited == 1
