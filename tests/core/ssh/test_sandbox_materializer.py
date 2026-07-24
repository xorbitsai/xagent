"""SandboxTmpfsSecretMaterializer: writes one credential into a leased sandbox's
tmpfs for a single call (design §12.3 / §15.2).

A fake sandbox records exec argv and write_file payloads so we can assert the
key material travels only through write_file (never argv/logs), the dir/files
get 0700/0600, and the dir is removed on every exit path.
"""

from __future__ import annotations

import pytest

from xagent.core.ssh.sandbox_materializer import SandboxTmpfsSecretMaterializer
from xagent.core.ssh.types import SensitiveSshCredential
from xagent.sandbox.base import ExecResult

_KEY = "-----BEGIN OPENSSH PRIVATE KEY-----\nsecret-bytes\n-----END OPENSSH PRIVATE KEY-----\n"
_KNOWN_HOSTS = "host.example ssh-ed25519 AAAABBBB\n"


class _FakeSandbox:
    def __init__(self, *, fail_mkdir: bool = False) -> None:
        self.execs: list[tuple[str, ...]] = []
        self.writes: list[tuple[str, str]] = []
        self._fail_mkdir = fail_mkdir

    async def exec(
        self, command: str, *args: str, env=None, max_output_bytes=None
    ) -> ExecResult:
        self.execs.append((command, *args))
        if self._fail_mkdir and command == "mkdir":
            return ExecResult(exit_code=1, stdout="", stderr="cannot create")
        return ExecResult(exit_code=0, stdout="", stderr="")

    async def write_file(
        self, content: str, remote_path: str, overwrite: bool = False
    ) -> None:
        self.writes.append((remote_path, content))


def _cred() -> SensitiveSshCredential:
    return SensitiveSshCredential(_KEY.encode(), "", "ssh-ed25519")


async def test_materializes_key_and_known_hosts_into_private_dir() -> None:
    sandbox = _FakeSandbox()
    mat = SandboxTmpfsSecretMaterializer()
    async with mat.materialize_ssh(sandbox, _cred(), _KNOWN_HOSTS) as paths:
        # Paths live under a random /tmp dir (Docker archive API can't target
        # /dev/shm).
        assert paths.private_key_path.startswith("/tmp/")
        assert paths.known_hosts_path.startswith("/tmp/")
        # Key + known_hosts content went through write_file, not argv.
        written = dict(sandbox.writes)
        assert written[paths.private_key_path] == _KEY
        assert written[paths.known_hosts_path] == _KNOWN_HOSTS


async def test_secret_never_in_exec_argv() -> None:
    sandbox = _FakeSandbox()
    async with SandboxTmpfsSecretMaterializer().materialize_ssh(
        sandbox, _cred(), _KNOWN_HOSTS
    ):
        pass
    joined = " ".join(part for call in sandbox.execs for part in call)
    assert "secret-bytes" not in joined
    assert "BEGIN OPENSSH PRIVATE KEY" not in joined


async def test_dir_created_0700_and_files_chmod_0600() -> None:
    sandbox = _FakeSandbox()
    async with SandboxTmpfsSecretMaterializer().materialize_ssh(
        sandbox, _cred(), _KNOWN_HOSTS
    ) as paths:
        pass
    # mkdir with an explicit 0700 mode for the private dir.
    mkdir = [c for c in sandbox.execs if c[0] == "mkdir"]
    assert mkdir and "700" in mkdir[0]
    # chmod 600 covering both secret files.
    chmods = [c for c in sandbox.execs if c[0] == "chmod" and "600" in c]
    chmod_targets = {t for c in chmods for t in c[2:]}
    assert paths.private_key_path in chmod_targets
    assert paths.known_hosts_path in chmod_targets


async def test_dir_removed_on_success() -> None:
    sandbox = _FakeSandbox()
    async with SandboxTmpfsSecretMaterializer().materialize_ssh(
        sandbox, _cred(), _KNOWN_HOSTS
    ) as paths:
        directory = paths.private_key_path.rsplit("/", 1)[0]
    rms = [c for c in sandbox.execs if c[0] == "rm"]
    assert any(directory in c for c in rms)


async def test_dir_removed_on_exception() -> None:
    sandbox = _FakeSandbox()
    directory = None
    with pytest.raises(RuntimeError):
        async with SandboxTmpfsSecretMaterializer().materialize_ssh(
            sandbox, _cred(), _KNOWN_HOSTS
        ) as paths:
            directory = paths.private_key_path.rsplit("/", 1)[0]
            raise RuntimeError("boom")
    rms = [c for c in sandbox.execs if c[0] == "rm"]
    assert any(directory in c for c in rms)


async def test_mkdir_failure_raises_before_yield() -> None:
    sandbox = _FakeSandbox(fail_mkdir=True)
    with pytest.raises(Exception):  # noqa: B017,PT011
        async with SandboxTmpfsSecretMaterializer().materialize_ssh(
            sandbox, _cred(), _KNOWN_HOSTS
        ):
            pass
    # No secret was written if the private dir could not be created.
    assert sandbox.writes == []
