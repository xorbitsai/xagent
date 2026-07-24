import dataclasses

import pytest

from xagent.core.ssh.types import (
    ActorRef,
    MaterializedSshPaths,
    PrincipalRef,
    ResolvedSshTarget,
    SensitiveSshCredential,
    SshExecutionContext,
    SshSecretHandle,
)


def _make_resolved() -> ResolvedSshTarget:
    return ResolvedSshTarget(
        target_public_id="t-1",
        hostname="example.com",
        port=22,
        username="deploy",
        remote_root="/srv",
        capabilities=frozenset({"execute", "upload"}),
        approval_policy="always",
        secret_handle=SshSecretHandle(credential_id="c-1", version_id="v-1"),
        known_hosts="example.com ssh-ed25519 AAAA...\n",
        credential_public_id="c-1",
        credential_version_id="v-1",
        host_key_fingerprint="SHA256:abc",
    )


def test_value_objects_are_frozen() -> None:
    resolved = _make_resolved()
    with pytest.raises(dataclasses.FrozenInstanceError):
        resolved.hostname = "evil.com"  # type: ignore[misc]


def test_execution_context_holds_actor_and_principal() -> None:
    ctx = SshExecutionContext(
        actor=ActorRef(actor_type="user", actor_id="u-1"),
        execution_principal=PrincipalRef(principal_type="team", principal_id="team-9"),
        agent_id=42,
        task_id=None,
        turn_id=None,
        request_id="req-1",
    )
    assert ctx.execution_principal.principal_type == "team"
    assert ctx.agent_id == 42


def test_platform_principal_allows_none_id() -> None:
    principal = PrincipalRef(principal_type="platform", principal_id=None)
    assert principal.principal_id is None


def test_sensitive_credential_repr_does_not_leak_private_key() -> None:
    cred = SensitiveSshCredential(
        private_key=b"-----BEGIN OPENSSH PRIVATE KEY-----\nSECRETMATERIAL\n",
        public_key="ssh-ed25519 AAAA...",
        key_algorithm="ssh-ed25519",
    )
    assert b"SECRETMATERIAL" not in repr(cred).encode()
    assert "SECRETMATERIAL" not in str(cred)
    assert "redacted" in repr(cred).lower()
    assert cred.public_key == "ssh-ed25519 AAAA..."


def test_materialized_paths_carry_both_files() -> None:
    paths = MaterializedSshPaths(
        private_key_path="/tmp/x/id", known_hosts_path="/tmp/x/known_hosts"
    )
    assert paths.private_key_path.endswith("id")
    assert paths.known_hosts_path.endswith("known_hosts")
