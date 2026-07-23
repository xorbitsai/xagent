"""Tests for SshExecutor orchestration + limits (Phase 3, §11.2 / §19.4).

Happy path is a real end-to-end run (in-memory provider/store + reference
materializer + asyncssh runner + local test server). Denials and limit
behavior use targeted inputs / a recording runner.
"""

from __future__ import annotations

import pytest

from tests.core.ssh.helpers import (
    InMemorySshSecretStore,
    InMemorySshTargetProvider,
    LocalTmpSecretMaterializer,
    RunningSshServer,
    start_test_ssh_server,
)
from xagent.core.ssh import SshError, SshErrorCode
from xagent.core.ssh.egress import EgressPolicyConfig
from xagent.core.ssh.executor import SshExecutor
from xagent.core.ssh.runner import AsyncsshRunner, SshRunResult
from xagent.core.ssh.types import (
    ActorRef,
    PrincipalRef,
    ResolvedSshTarget,
    SensitiveSshCredential,
    SshExecutionContext,
    SshSecretHandle,
)

pytestmark = pytest.mark.integration

_ALLOW_LOOPBACK = EgressPolicyConfig(allow_cidrs=("127.0.0.0/8",))


def _ctx(agent_id: int = 1) -> SshExecutionContext:
    return SshExecutionContext(
        actor=ActorRef(actor_type="user", actor_id="u"),
        execution_principal=PrincipalRef(principal_type="user", principal_id="u"),
        agent_id=agent_id,
        task_id=None,
        turn_id=None,
        request_id="r",
    )


def _known_hosts(server: RunningSshServer) -> str:
    algo, blob = server.host_public_key.split()[:2]
    return f"[{server.host}]:{server.port} {algo} {blob}\n"


def _target(
    server: RunningSshServer, *, capabilities: frozenset[str] = frozenset({"execute"})
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
        known_hosts=_known_hosts(server),
        credential_public_id="c",
        credential_version_id="v",
        host_key_fingerprint="SHA256:x",
    )


def _executor(server: RunningSshServer, target: ResolvedSshTarget, **kwargs) -> SshExecutor:
    return SshExecutor(
        provider=InMemorySshTargetProvider({(1, "prod"): target}),
        secret_store=InMemorySshSecretStore(
            {"v": SensitiveSshCredential(server.client_private_key.encode(), "", "ssh-ed25519")}
        ),
        materializer=LocalTmpSecretMaterializer(),
        runner=AsyncsshRunner(),
        egress_config=_ALLOW_LOOPBACK,
        **kwargs,
    )


async def test_execute_happy_path() -> None:
    server = await start_test_ssh_server()
    try:
        outcome = await _executor(server, _target(server)).execute(
            _ctx(), target_alias="prod", command="uptime", timeout_seconds=10
        )
        assert outcome.exit_code == 0
        assert outcome.stdout == "ran: uptime"
        assert outcome.truncated is False
        assert outcome.duration_ms >= 0
    finally:
        await server.close()


async def test_execute_capability_denied() -> None:
    server = await start_test_ssh_server()
    try:
        ex = _executor(server, _target(server, capabilities=frozenset({"download"})))
        with pytest.raises(SshError) as exc:
            await ex.execute(_ctx(), target_alias="prod", command="uptime", timeout_seconds=10)
        assert exc.value.code == SshErrorCode.OPERATION_NOT_ALLOWED
    finally:
        await server.close()


async def test_execute_egress_preflight_denies_private_host() -> None:
    server = await start_test_ssh_server()
    try:
        target = _target(server)
        # A private hostname must be rejected by the pre-flight resolve+check
        # before any secret is read or connection attempted.
        private = ResolvedSshTarget(**{**target.__dict__, "hostname": "10.0.0.5"})
        ex = SshExecutor(
            provider=InMemorySshTargetProvider({(1, "prod"): private}),
            secret_store=InMemorySshSecretStore(
                {"v": SensitiveSshCredential(b"unused", "", "ssh-ed25519")}
            ),
            materializer=LocalTmpSecretMaterializer(),
            runner=AsyncsshRunner(),
            egress_config=EgressPolicyConfig(),
        )
        with pytest.raises(SshError) as exc:
            await ex.execute(_ctx(), target_alias="prod", command="uptime", timeout_seconds=10)
        assert exc.value.code == SshErrorCode.EGRESS_DENIED
    finally:
        await server.close()


async def test_execute_output_capped() -> None:
    server = await start_test_ssh_server()
    try:
        outcome = await _executor(server, _target(server), max_output_bytes=5).execute(
            _ctx(), target_alias="prod", command="uptime", timeout_seconds=10
        )
        # "ran: uptime" is 11 bytes; capped to 5 with a truncation marker.
        assert outcome.truncated is True
        assert len(outcome.stdout.encode("utf-8")) <= 5
    finally:
        await server.close()


class _RecordingRunner:
    def __init__(self) -> None:
        self.timeout_seconds: int | None = None

    async def execute(self, **kwargs) -> SshRunResult:
        self.timeout_seconds = kwargs["timeout_seconds"]
        return SshRunResult(exit_code=0, stdout="x", stderr="", truncated=False)


async def test_execute_timeout_clamped_to_max() -> None:
    server = await start_test_ssh_server()
    try:
        runner = _RecordingRunner()
        ex = SshExecutor(
            provider=InMemorySshTargetProvider({(1, "prod"): _target(server)}),
            secret_store=InMemorySshSecretStore(
                {"v": SensitiveSshCredential(server.client_private_key.encode(), "", "ssh-ed25519")}
            ),
            materializer=LocalTmpSecretMaterializer(),
            runner=runner,
            egress_config=_ALLOW_LOOPBACK,
            max_timeout_seconds=5,
        )
        await ex.execute(_ctx(), target_alias="prod", command="uptime", timeout_seconds=100)
        assert runner.timeout_seconds == 5
    finally:
        await server.close()
