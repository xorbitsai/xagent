from pathlib import Path

import pytest

from xagent.core.computer.session import (
    BrowserRuntimeKind,
    ComputerSessionBinding,
    validate_browser_profile_id,
)


def test_ephemeral_binding_uses_logical_session() -> None:
    binding = ComputerSessionBinding()

    assert binding.manager_session_id("task-1:step") == "task-1:step"
    assert binding.persistent_profile_dir() is None
    assert binding.manager_owner_id() is None


def test_persistent_binding_is_scoped_by_user_and_profile(tmp_path: Path) -> None:
    binding = ComputerSessionBinding.from_values(
        runtime_kind=BrowserRuntimeKind.PERSISTENT_PLAYWRIGHT,
        owner_task_id="task-1",
        user_id=42,
        profile_id="work",
        profile_root=tmp_path,
    )

    assert binding.manager_session_id("ignored") == "computer-profile:user_42:work"
    assert binding.persistent_profile_dir() == tmp_path / "user_42" / "work"
    assert binding.manager_owner_id() == "task-1"


@pytest.mark.parametrize("profile_id", ["../escape", ".hidden", "bad/name", ""])
def test_profile_id_rejects_unsafe_path_components(profile_id: str) -> None:
    with pytest.raises(ValueError, match="profile ID"):
        validate_browser_profile_id(profile_id)


def test_persistent_binding_requires_authenticated_user(tmp_path: Path) -> None:
    binding = ComputerSessionBinding.from_values(
        runtime_kind="persistent_playwright",
        owner_task_id="task-1",
        user_id=None,
        profile_id="default",
        profile_root=tmp_path,
    )

    with pytest.raises(ValueError, match="authenticated user_id"):
        binding.persistent_profile_dir()


def test_persistent_binding_requires_task_owner(tmp_path: Path) -> None:
    binding = ComputerSessionBinding.from_values(
        runtime_kind="persistent_playwright",
        owner_task_id=None,
        user_id=1,
        profile_id="default",
        profile_root=tmp_path,
    )

    with pytest.raises(ValueError, match="owning task"):
        binding.manager_owner_id()


def test_extension_binding_requires_authenticated_user_and_owner() -> None:
    binding = ComputerSessionBinding.from_values(
        runtime_kind="extension_relay",
        owner_task_id="task-1",
        user_id=4,
        profile_id="default",
        profile_root=None,
    )

    assert binding.is_extension_relay is True
    assert binding.is_user_controlled is True
    assert binding.require_user_id() == 4
    assert binding.require_owner_task_id() == "task-1"
