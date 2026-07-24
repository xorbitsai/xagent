import os
import stat

import pytest

from tests.core.ssh.helpers import (
    InMemorySshSecretStore,
    InMemorySshTargetProvider,
    LocalTmpSecretMaterializer,
)
from xagent.core.ssh.errors import SshError, SshErrorCode
from xagent.core.ssh.interfaces import (
    SandboxSecretMaterializer,
    SshSecretStore,
    SshTargetProvider,
)
from xagent.core.ssh.types import (
    ActorRef,
    PrincipalRef,
    ResolvedSshTarget,
    SensitiveSshCredential,
    SshExecutionContext,
    SshSecretHandle,
)


def _ctx(agent_id: int = 1) -> SshExecutionContext:
    return SshExecutionContext(
        actor=ActorRef(actor_type="user", actor_id="u-1"),
        execution_principal=PrincipalRef(principal_type="user", principal_id="u-1"),
        agent_id=agent_id,
        task_id=None,
        turn_id=None,
        request_id="req-1",
    )


def _target() -> ResolvedSshTarget:
    return ResolvedSshTarget(
        target_public_id="t-1",
        hostname="example.com",
        port=22,
        username="deploy",
        remote_root=None,
        capabilities=frozenset({"execute"}),
        approval_policy="always",
        secret_handle=SshSecretHandle(credential_id="c-1", version_id="v-1"),
        known_hosts="example.com ssh-ed25519 AAAA...\n",
        credential_public_id="c-1",
        credential_version_id="v-1",
        host_key_fingerprint="SHA256:abc",
    )


def _credential() -> SensitiveSshCredential:
    return SensitiveSshCredential(
        private_key=b"PRIVATEKEYBYTES",
        public_key="ssh-ed25519 AAAA...",
        key_algorithm="ssh-ed25519",
    )


def test_adapters_satisfy_protocols() -> None:
    assert isinstance(InMemorySshTargetProvider({}), SshTargetProvider)
    assert isinstance(InMemorySshSecretStore({}), SshSecretStore)
    assert isinstance(LocalTmpSecretMaterializer(), SandboxSecretMaterializer)


async def test_resolve_read_materialize_round_trip() -> None:
    provider = InMemorySshTargetProvider({(1, "prod"): _target()})
    store = InMemorySshSecretStore({"v-1": _credential()})
    materializer = LocalTmpSecretMaterializer()

    resolved = await provider.resolve(_ctx(), "prod")
    credential = await store.read_version(resolved.secret_handle)

    async with materializer.materialize_ssh(
        object(), credential, resolved.known_hosts
    ) as paths:
        with open(paths.private_key_path, "rb") as handle:
            assert handle.read() == b"PRIVATEKEYBYTES"
        key_mode = stat.S_IMODE(os.stat(paths.private_key_path).st_mode)
        dir_mode = stat.S_IMODE(
            os.stat(os.path.dirname(paths.private_key_path)).st_mode
        )
        assert key_mode == 0o600
        assert dir_mode == 0o700
        saved_dir = os.path.dirname(paths.private_key_path)

    # cleanup happened on normal exit
    assert not os.path.exists(saved_dir)


async def test_materializer_cleans_up_on_exception() -> None:
    materializer = LocalTmpSecretMaterializer()
    saved_dir = ""
    with pytest.raises(RuntimeError):
        async with materializer.materialize_ssh(
            object(), _credential(), "known\n"
        ) as paths:
            saved_dir = os.path.dirname(paths.private_key_path)
            raise RuntimeError("boom")
    assert saved_dir
    assert not os.path.exists(saved_dir)


async def test_unknown_alias_raises_target_not_found() -> None:
    provider = InMemorySshTargetProvider({})
    with pytest.raises(SshError) as excinfo:
        await provider.resolve(_ctx(), "nope")
    assert excinfo.value.code is SshErrorCode.TARGET_NOT_FOUND


async def test_unknown_version_raises_secret_unavailable() -> None:
    store = InMemorySshSecretStore({})
    with pytest.raises(SshError) as excinfo:
        await store.read_version(SshSecretHandle(credential_id="c", version_id="x"))
    assert excinfo.value.code is SshErrorCode.SECRET_UNAVAILABLE


async def test_list_bound_targets_filters_by_agent() -> None:
    from xagent.core.ssh import BoundTargetInfo  # noqa: F401

    provider = InMemorySshTargetProvider(
        {(1, "prod"): _target(), (2, "other"): _target()}
    )
    infos = await provider.list_bound_targets(_ctx(agent_id=1))
    assert [i.alias for i in infos] == ["prod"]
    assert "execute" in infos[0].capabilities
