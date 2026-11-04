import pytest

from xagent.web.services.ssh_runtime import (
    get_ssh_target_provider,
    set_ssh_target_provider_hook,
)


class _FakeProvider:
    """Minimal object satisfying the @runtime_checkable SshTargetProvider."""

    async def resolve(self, context, target_alias):  # pragma: no cover - stub
        raise NotImplementedError

    async def list_bound_targets(self, context):  # pragma: no cover - stub
        return []


def test_provider_hook_install_and_clear() -> None:
    assert get_ssh_target_provider(object()) is None
    provider = _FakeProvider()
    set_ssh_target_provider_hook(lambda session_factory: provider)
    try:
        assert get_ssh_target_provider(object()) is provider
    finally:
        set_ssh_target_provider_hook(None)
    assert get_ssh_target_provider(object()) is None


def test_provider_hook_rejects_object_missing_interface() -> None:
    # A mis-wired factory must fail loudly at the seam, not deep in a tool call.
    set_ssh_target_provider_hook(lambda session_factory: object())  # type: ignore[arg-type,return-value]
    try:
        with pytest.raises(TypeError):
            get_ssh_target_provider(object())
    finally:
        set_ssh_target_provider_hook(None)
