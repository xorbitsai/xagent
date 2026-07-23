from xagent.web.services.ssh_runtime import (
    get_ssh_target_provider,
    set_ssh_target_provider_hook,
)


def test_provider_hook_install_and_clear() -> None:
    assert get_ssh_target_provider(object()) is None
    sentinel = object()
    set_ssh_target_provider_hook(lambda session_factory: sentinel)  # type: ignore[arg-type,return-value]
    try:
        assert get_ssh_target_provider(object()) is sentinel
    finally:
        set_ssh_target_provider_hook(None)
    assert get_ssh_target_provider(object()) is None
