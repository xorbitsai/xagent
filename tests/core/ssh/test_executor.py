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


def _executor(
    server: RunningSshServer, target: ResolvedSshTarget, **kwargs
) -> SshExecutor:
    return SshExecutor(
        provider=InMemorySshTargetProvider({(1, "prod"): target}),
        secret_store=InMemorySshSecretStore(
            {
                "v": SensitiveSshCredential(
                    server.client_private_key.encode(), "", "ssh-ed25519"
                )
            }
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
            await ex.execute(
                _ctx(), target_alias="prod", command="uptime", timeout_seconds=10
            )
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
            await ex.execute(
                _ctx(), target_alias="prod", command="uptime", timeout_seconds=10
            )
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


async def test_upload_happy_path(tmp_path) -> None:
    server = await start_test_ssh_server()
    local = tmp_path / "local.txt"
    local.write_text("via-executor")
    remote = tmp_path / "remote.txt"
    try:
        ex = _executor(server, _target(server, capabilities=frozenset({"upload"})))
        await ex.upload(
            _ctx(), target_alias="prod", local_path=str(local), remote_path=str(remote)
        )
        assert remote.read_text() == "via-executor"
    finally:
        await server.close()


async def test_download_happy_path(tmp_path) -> None:
    server = await start_test_ssh_server()
    remote = tmp_path / "remote.txt"
    remote.write_text("fetched")
    local = tmp_path / "local.txt"
    try:
        ex = _executor(server, _target(server, capabilities=frozenset({"download"})))
        await ex.download(
            _ctx(), target_alias="prod", remote_path=str(remote), local_path=str(local)
        )
        assert local.read_text() == "fetched"
    finally:
        await server.close()


async def test_upload_capability_denied(tmp_path) -> None:
    server = await start_test_ssh_server()
    try:
        ex = _executor(server, _target(server, capabilities=frozenset({"execute"})))
        with pytest.raises(SshError) as exc:
            await ex.upload(
                _ctx(),
                target_alias="prod",
                local_path=str(tmp_path / "x"),
                remote_path="/tmp/x",
            )
        assert exc.value.code == SshErrorCode.OPERATION_NOT_ALLOWED
    finally:
        await server.close()


async def test_upload_remote_root_escape_denied(tmp_path) -> None:
    server = await start_test_ssh_server()
    local = tmp_path / "local.txt"
    local.write_text("x")
    try:
        target = _target(server, capabilities=frozenset({"upload"}))
        confined = ResolvedSshTarget(**{**target.__dict__, "remote_root": "/srv/app"})
        ex = _executor(server, confined)
        with pytest.raises(SshError) as exc:
            await ex.upload(
                _ctx(),
                target_alias="prod",
                local_path=str(local),
                remote_path="/srv/app/../../etc/passwd",
            )
        assert exc.value.code == SshErrorCode.OPERATION_NOT_ALLOWED
    finally:
        await server.close()


class _RecordingAuditSink:
    def __init__(self, *, boom: bool = False) -> None:
        self.events: list[dict] = []
        self._boom = boom

    async def record(
        self, *, context, operation, status, target=None, error_code=None
    ) -> None:
        if self._boom:
            raise RuntimeError("audit backend down")
        self.events.append(
            {
                "operation": operation,
                "status": status,
                "error_code": error_code,
                "target_public_id": target.target_public_id if target else None,
            }
        )


async def test_execute_emits_success_audit() -> None:
    server = await start_test_ssh_server()
    sink = _RecordingAuditSink()
    try:
        await _executor(server, _target(server), audit_sink=sink).execute(
            _ctx(), target_alias="prod", command="uptime", timeout_seconds=10
        )
        assert sink.events == [
            {
                "operation": "ssh_execute",
                "status": "success",
                "error_code": None,
                "target_public_id": "t",
            }
        ]
    finally:
        await server.close()


async def test_execute_emits_failure_audit_with_error_code() -> None:
    server = await start_test_ssh_server()
    sink = _RecordingAuditSink()
    try:
        ex = _executor(
            server,
            _target(server, capabilities=frozenset({"download"})),
            audit_sink=sink,
        )
        with pytest.raises(SshError):
            await ex.execute(
                _ctx(), target_alias="prod", command="uptime", timeout_seconds=10
            )
        assert len(sink.events) == 1
        assert sink.events[0]["status"] == "failure"
        assert sink.events[0]["error_code"] == "ssh_operation_not_allowed"
    finally:
        await server.close()


async def test_audit_sink_failure_never_breaks_execution() -> None:
    server = await start_test_ssh_server()
    try:
        # A sink that raises must not fail the operation — auditing is best-effort.
        outcome = await _executor(
            server, _target(server), audit_sink=_RecordingAuditSink(boom=True)
        ).execute(_ctx(), target_alias="prod", command="uptime", timeout_seconds=10)
        assert outcome.exit_code == 0
    finally:
        await server.close()


async def test_upload_emits_audit(tmp_path) -> None:
    server = await start_test_ssh_server()
    local = tmp_path / "local.txt"
    local.write_text("x")
    remote = tmp_path / "remote.txt"
    sink = _RecordingAuditSink()
    try:
        await _executor(
            server, _target(server, capabilities=frozenset({"upload"})), audit_sink=sink
        ).upload(
            _ctx(), target_alias="prod", local_path=str(local), remote_path=str(remote)
        )
        assert sink.events[0]["operation"] == "ssh_upload"
        assert sink.events[0]["status"] == "success"
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
                {
                    "v": SensitiveSshCredential(
                        server.client_private_key.encode(), "", "ssh-ed25519"
                    )
                }
            ),
            materializer=LocalTmpSecretMaterializer(),
            runner=runner,
            egress_config=_ALLOW_LOOPBACK,
            max_timeout_seconds=5,
        )
        await ex.execute(
            _ctx(), target_alias="prod", command="uptime", timeout_seconds=100
        )
        assert runner.timeout_seconds == 5
    finally:
        await server.close()


def test_cap_outputs_reserves_stderr_slice() -> None:
    # A large stdout must not zero out stderr entirely (N3) — diagnostics on a
    # failing command matter more than the tail of stdout.
    from xagent.core.ssh.executor import _cap_outputs

    budget = 1 << 20
    out, err, truncated = _cap_outputs("o" * (2 * budget), "e" * 1000, budget)
    assert truncated is True
    assert len(err.encode("utf-8")) == 1000  # stderr preserved, not dropped
    assert len(out.encode("utf-8")) <= budget - 1000


def test_constrain_remote_path_rejects_injection_chars() -> None:
    # A quote or newline in remote_path could break out of the sandbox runner's
    # sftp batch file and inject a second transfer command (#1). The shared
    # transfer prologue must reject them for both runners.
    from xagent.core.ssh.executor import _constrain_remote_path

    for bad in (
        '/root/x"',
        "/root/a\nget /etc/shadow /tmp/exfil",
        "/root/b\r",
        "/root/\x00c",
    ):
        with pytest.raises(SshError) as exc:
            _constrain_remote_path(bad, None)
        assert exc.value.code == SshErrorCode.OPERATION_NOT_ALLOWED


def test_constrain_remote_path_allows_leading_double_slash() -> None:
    # normpath keeps a leading "//" (POSIX), which used to fail the root-prefix
    # check and wrongly reject a valid in-root path (m4). It must be accepted.
    from xagent.core.ssh.executor import _constrain_remote_path

    _constrain_remote_path("//root/x", "/root")  # must not raise


def test_transfer_timeout_clamped_to_max() -> None:
    # SFTP transfers used a fixed 300s budget that ignored the deployment's
    # max_timeout_seconds clamp (m3); a tighter max must also tighten transfers.
    ex = SshExecutor(
        provider=InMemorySshTargetProvider({}),
        secret_store=InMemorySshSecretStore({}),
        materializer=LocalTmpSecretMaterializer(),
        runner=AsyncsshRunner(),
        egress_config=_ALLOW_LOOPBACK,
        max_timeout_seconds=10,
    )
    assert ex._transfer_timeout_seconds == 10


class _BoomRunner:
    """A runner whose execute raises a non-SshError, unexpected exception."""

    async def execute(self, **kwargs) -> SshRunResult:
        raise RuntimeError("unexpected runner failure")


async def test_execute_audits_unexpected_non_ssh_error() -> None:
    # A non-SshError (or a CancelledError) used to slip past `except SshError`
    # and leave no audit trail; the failure audit must fire for any error (#6).
    server = await start_test_ssh_server()
    sink = _RecordingAuditSink()
    try:
        ex = SshExecutor(
            provider=InMemorySshTargetProvider({(1, "prod"): _target(server)}),
            secret_store=InMemorySshSecretStore(
                {
                    "v": SensitiveSshCredential(
                        server.client_private_key.encode(), "", "ssh-ed25519"
                    )
                }
            ),
            materializer=LocalTmpSecretMaterializer(),
            runner=_BoomRunner(),
            egress_config=_ALLOW_LOOPBACK,
            audit_sink=sink,
        )
        with pytest.raises(RuntimeError):
            await ex.execute(
                _ctx(), target_alias="prod", command="uptime", timeout_seconds=10
            )
        assert len(sink.events) == 1
        assert sink.events[0]["status"] == "failure"
        assert sink.events[0]["error_code"] is None
    finally:
        await server.close()
