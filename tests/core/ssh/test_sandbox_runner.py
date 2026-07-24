"""SandboxSshRunner: runs the OpenSSH binary inside a leased sandbox (§14/§15.2).

Unit-level with a fake sandbox — no real container/sshd (that is a Chunk 5
integration test). Covers the argv security contract, IP pinning, and the
exit-code → result/error mapping.
"""

from __future__ import annotations

import pytest

from xagent.core.ssh.egress import EgressPolicyConfig
from xagent.core.ssh.errors import SshError, SshErrorCode
from xagent.core.ssh.sandbox_runner import SandboxSshRunner, _ssh_argv
from xagent.sandbox.base import ExecResult

_EGRESS = EgressPolicyConfig(allow_cidrs=("0.0.0.0/0",))


class _FakeSandbox:
    """Records exec argv; returns a queued ExecResult per call."""

    def __init__(self, results: list[ExecResult]) -> None:
        self.execs: list[tuple[str, ...]] = []
        self._results = results

    async def exec(
        self, command: str, *args: str, env=None, max_output_bytes=None
    ) -> ExecResult:
        self.execs.append((command, *args))
        return self._results.pop(0)


def _run_kwargs(**over):
    base = {
        "sandbox": None,
        "hostname": "host.example",
        "connect_ip": "203.0.113.7",
        "port": 2222,
        "username": "deploy",
        "private_key_path": "/dev/shm/x/id_key",
        "known_hosts_path": "/dev/shm/x/known_hosts",
        "command": "uptime",
        "timeout_seconds": 30,
        "egress_config": _EGRESS,
    }
    base.update(over)
    return base


# ---- argv contract ---------------------------------------------------------


def test_ssh_argv_strict_security_opts() -> None:
    argv = _ssh_argv(
        hostname="host.example",
        connect_ip="203.0.113.7",
        port=2222,
        username="deploy",
        private_key_path="/k",
        known_hosts_path="/kh",
    )
    joined = " ".join(argv)
    assert argv[0] == "ssh"
    for opt in (
        "BatchMode=yes",
        "StrictHostKeyChecking=yes",
        "IdentitiesOnly=yes",
        "GlobalKnownHostsFile=/dev/null",
        "PasswordAuthentication=no",
        "KbdInteractiveAuthentication=no",
        "ForwardAgent=no",
        "UserKnownHostsFile=/kh",
        "HostKeyAlias=host.example",
        "Port=2222",
    ):
        assert opt in joined, opt
    assert "-F" in argv and "/dev/null" in argv  # no ssh config
    assert "-i" in argv and "/k" in argv


def test_ssh_argv_connects_to_ip_with_hostname_alias() -> None:
    """DNS-rebinding pin: connect to the vetted IP, verify host key under the
    hostname (HostKeyAlias), and pass the command as the trailing arg."""
    argv = _ssh_argv(
        hostname="host.example",
        connect_ip="203.0.113.7",
        port=22,
        username="deploy",
        private_key_path="/k",
        known_hosts_path="/kh",
        command="uptime",
    )
    assert "HostKeyAlias=host.example" in " ".join(argv)
    assert "deploy@203.0.113.7" in argv
    assert "host.example" not in [
        a for a in argv if a == "host.example"
    ]  # not a target
    assert argv[-1] == "uptime"


# ---- exit-code mapping -----------------------------------------------------


async def test_execute_success_returns_result() -> None:
    sandbox = _FakeSandbox([ExecResult(exit_code=0, stdout="ran: uptime", stderr="")])
    result = await SandboxSshRunner().execute(**_run_kwargs(sandbox=sandbox))
    assert result.exit_code == 0
    assert result.stdout == "ran: uptime"
    assert result.truncated is False


async def test_execute_nonzero_remote_exit_is_result_not_error() -> None:
    """A failing remote command is a normal result, not a transport error."""
    sandbox = _FakeSandbox([ExecResult(exit_code=3, stdout="", stderr="boom")])
    result = await SandboxSshRunner().execute(**_run_kwargs(sandbox=sandbox))
    assert result.exit_code == 3


async def test_execute_timeout_maps_to_command_timeout() -> None:
    sandbox = _FakeSandbox([ExecResult(exit_code=124, stdout="", stderr="")])
    with pytest.raises(SshError) as exc:
        await SandboxSshRunner().execute(**_run_kwargs(sandbox=sandbox))
    assert exc.value.code == SshErrorCode.COMMAND_TIMEOUT


async def test_execute_host_key_mismatch_maps() -> None:
    sandbox = _FakeSandbox(
        [ExecResult(exit_code=255, stdout="", stderr="Host key verification failed.")]
    )
    with pytest.raises(SshError) as exc:
        await SandboxSshRunner().execute(**_run_kwargs(sandbox=sandbox))
    assert exc.value.code == SshErrorCode.HOST_KEY_MISMATCH


async def test_execute_generic_transport_error_no_leak() -> None:
    sandbox = _FakeSandbox(
        [
            ExecResult(
                exit_code=255,
                stdout="",
                stderr="ssh: connect to host ... port 22: refused",
            )
        ]
    )
    with pytest.raises(SshError) as exc:
        await SandboxSshRunner().execute(**_run_kwargs(sandbox=sandbox))
    # A stable, generic code — the raw stderr is not the user-facing message.
    # Exit 255 that isn't a host-key failure means connection/auth failed, not a
    # timeout (that's exit 124), so it maps to CONNECTION_FAILED.
    assert exc.value.code == SshErrorCode.CONNECTION_FAILED
    assert "refused" not in str(exc.value)


async def test_execute_propagates_truncated_from_sandbox() -> None:
    # The sandbox exec caps output and flags it; the runner surfaces that (#3).
    sandbox = _FakeSandbox(
        [ExecResult(exit_code=0, stdout="x", stderr="", truncated=True)]
    )
    result = await SandboxSshRunner().execute(**_run_kwargs(sandbox=sandbox))
    assert result.truncated is True


async def test_execute_wraps_in_timeout_and_runs_ssh() -> None:
    sandbox = _FakeSandbox([ExecResult(exit_code=0, stdout="", stderr="")])
    await SandboxSshRunner().execute(**_run_kwargs(sandbox=sandbox, timeout_seconds=45))
    call = sandbox.execs[0]
    assert call[0] == "timeout"
    assert "45" in call[:3]
    assert "ssh" in call


# ---- SFTP staging ----------------------------------------------------------


class _XferSandbox:
    """Fake sandbox for SFTP tests: exec returns 0 by default (override per
    command prefix), and records file-transfer + write_file calls."""

    def __init__(self, *, remote_exists: bool = False, sftp_exit: int = 0) -> None:
        self.execs: list[tuple[str, ...]] = []
        self.uploads: list[tuple[str, str]] = []
        self.downloads: list[tuple[str, str]] = []
        self.writes: list[tuple[str, str]] = []
        self._remote_exists = remote_exists
        self._sftp_exit = sftp_exit

    async def exec(
        self, command: str, *args: str, env=None, max_output_bytes=None
    ) -> ExecResult:
        self.execs.append((command, *args))
        argv = (command, *args)
        if "ssh" in argv and any("test -e" in a for a in argv):
            return ExecResult(
                exit_code=0 if self._remote_exists else 1, stdout="", stderr=""
            )
        if "sftp" in argv:
            return ExecResult(exit_code=self._sftp_exit, stdout="", stderr="")
        return ExecResult(exit_code=0, stdout="", stderr="")

    async def write_file(
        self, content: str, remote_path: str, overwrite: bool = False
    ) -> None:
        self.writes.append((remote_path, content))

    async def upload_file(
        self, local_path: str, remote_path: str, overwrite: bool = False
    ) -> None:
        self.uploads.append((local_path, remote_path))

    async def download_file(
        self, remote_path: str, local_path: str, overwrite: bool = False
    ) -> None:
        self.downloads.append((remote_path, local_path))


def _xfer_kwargs(**over):
    base = {
        "sandbox": None,
        "hostname": "host.example",
        "connect_ip": "203.0.113.7",
        "port": 22,
        "username": "deploy",
        "private_key_path": "/dev/shm/x/id_key",
        "known_hosts_path": "/dev/shm/x/known_hosts",
        "overwrite": False,
        "egress_config": _EGRESS,
    }
    base.update(over)
    return base


async def test_upload_stages_host_file_then_sftp_put() -> None:
    sandbox = _XferSandbox(remote_exists=False)
    await SandboxSshRunner().upload(
        **_xfer_kwargs(
            sandbox=sandbox, local_path="/host/ws/a.txt", remote_path="/srv/a.txt"
        )
    )
    # Host file was staged into the sandbox first.
    assert sandbox.uploads and sandbox.uploads[0][0] == "/host/ws/a.txt"
    # The sftp batch performs a put of the staged file.
    batch = "".join(c for _, c in sandbox.writes)
    assert batch.startswith("put ")
    assert "/srv/a.txt" in batch


async def test_upload_overwrite_false_rejects_existing_remote() -> None:
    sandbox = _XferSandbox(remote_exists=True)
    with pytest.raises(SshError) as exc:
        await SandboxSshRunner().upload(
            **_xfer_kwargs(
                sandbox=sandbox, local_path="/host/a", remote_path="/srv/a.txt"
            )
        )
    assert exc.value.code == SshErrorCode.OPERATION_NOT_ALLOWED
    # No sftp put was attempted.
    assert not any("sftp" in c for c in sandbox.execs)


async def test_download_sftp_get_then_copies_to_host(tmp_path) -> None:
    sandbox = _XferSandbox()
    local = tmp_path / "out.txt"
    await SandboxSshRunner().download(
        **_xfer_kwargs(sandbox=sandbox, remote_path="/srv/a.txt", local_path=str(local))
    )
    batch = "".join(c for _, c in sandbox.writes)
    assert batch.startswith("get ")
    # Fetched file was copied back out to the host workspace path.
    assert sandbox.downloads and sandbox.downloads[0][1] == str(local)


async def test_download_overwrite_false_rejects_existing_local(tmp_path) -> None:
    local = tmp_path / "out.txt"
    local.write_text("existing")
    sandbox = _XferSandbox()
    with pytest.raises(SshError) as exc:
        await SandboxSshRunner().download(
            **_xfer_kwargs(
                sandbox=sandbox, remote_path="/srv/a.txt", local_path=str(local)
            )
        )
    assert exc.value.code == SshErrorCode.OPERATION_NOT_ALLOWED


async def test_sftp_failure_maps_to_error() -> None:
    sandbox = _XferSandbox(sftp_exit=1)
    with pytest.raises(SshError):
        await SandboxSshRunner().upload(
            **_xfer_kwargs(
                sandbox=sandbox, local_path="/host/a", remote_path="/srv/a.txt"
            )
        )


async def test_staging_removed_after_transfer() -> None:
    sandbox = _XferSandbox()
    await SandboxSshRunner().upload(
        **_xfer_kwargs(sandbox=sandbox, local_path="/host/a", remote_path="/srv/a.txt")
    )
    assert any(c[0] == "rm" for c in sandbox.execs)


async def test_transfer_threads_timeout_into_exec() -> None:
    # The transfer's ssh/sftp invocations are clamped with `timeout <n>` (N1).
    sandbox = _XferSandbox()
    await SandboxSshRunner().upload(
        **_xfer_kwargs(
            sandbox=sandbox,
            local_path="/host/a",
            remote_path="/srv/a.txt",
            timeout_seconds=99,
        )
    )
    assert any(c[0] == "timeout" and c[1] == "99" for c in sandbox.execs)
