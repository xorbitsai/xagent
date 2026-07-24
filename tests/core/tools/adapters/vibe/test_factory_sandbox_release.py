"""Regression tests for ToolFactory release boundaries.

Issue #889 requires the config's DB connection to be released again before
sandbox workspace setup because override/allowlist reads may reopen it.
"""

import pytest

from xagent.core.tools.adapters.vibe.factory import ToolFactory, ToolRegistry


class _FakeSandbox:
    pass


class _FakeConfig:
    def __init__(self, calls):
        self._calls = calls

    def get_tool_selection_spec(self):
        return None

    def get_allowed_tools(self):
        return None

    def get_user_tool_overrides(self):
        self._calls.append("load_overrides")
        return {}

    def get_user_tool_allowlist(self):
        self._calls.append("load_allowlist")
        return None

    def release_db_connection(self):
        self._calls.append("release_db")

    def get_sandbox(self):
        return _FakeSandbox()

    def get_workspace_config(self):
        # ``_mock_`` selects MockWorkspace: no on-disk directories.
        return {"task_id": "_mock_", "base_dir": "/tmp"}

    def get_max_output_length(self):
        return 10000

    def get_max_field_count(self):
        return 100

    def get_max_recursion_depth(self):
        return 5


class _FailingPrepareConfig:
    def __init__(self, calls: list[str]) -> None:
        self._calls = calls

    async def prepare_factory_runtime(self) -> None:
        self._calls.append("prepare")
        raise RuntimeError("prepare failed")

    def release_prepared_factory_runtime(self) -> None:
        self._calls.append("release")


class _FailingPrepareAndReleaseConfig(_FailingPrepareConfig):
    def release_prepared_factory_runtime(self) -> None:
        super().release_prepared_factory_runtime()
        raise ValueError("release failed")


class _FailingReleaseConfig:
    def __init__(self, calls: list[str]) -> None:
        self._calls = calls

    def release_prepared_factory_runtime(self) -> None:
        self._calls.append("release")
        raise ValueError("release failed")


@pytest.mark.asyncio
async def test_release_prepared_runtime_when_prepare_fails():
    calls: list[str] = []

    with pytest.raises(RuntimeError, match="prepare failed"):
        await ToolFactory.create_all_tools(_FailingPrepareConfig(calls))

    assert calls == ["prepare", "release"]


@pytest.mark.asyncio
async def test_prepare_error_wins_when_release_also_fails(caplog):
    calls: list[str] = []

    with pytest.raises(RuntimeError, match="prepare failed"):
        await ToolFactory.create_all_tools(_FailingPrepareAndReleaseConfig(calls))

    assert calls == ["prepare", "release"]
    assert "Failed to release prepared tool-factory runtime" in caplog.text


@pytest.mark.asyncio
async def test_release_error_propagates_without_primary_error(monkeypatch):
    calls: list[str] = []

    async def build_tools(config, apply_user_override_filter=True):
        calls.append("build")
        return []

    monkeypatch.setattr(ToolFactory, "_create_all_tools_prepared", build_tools)

    with pytest.raises(ValueError, match="release failed"):
        await ToolFactory.create_all_tools(_FailingReleaseConfig(calls))

    assert calls == ["build", "release"]


@pytest.mark.asyncio
async def test_release_db_before_sandbox_workspace_setup(monkeypatch):
    calls: list[str] = []

    async def fake_create_registered_tools(config):
        return []

    monkeypatch.setattr(
        ToolRegistry,
        "create_registered_tools",
        staticmethod(fake_create_registered_tools),
    )

    from xagent.core.tools.adapters.vibe.sandboxed_tool import (
        sandboxed_tool_wrapper,
    )

    async def fake_create_workspace_in_sandbox(sandbox, workspace):
        calls.append("sandbox_exec")

    monkeypatch.setattr(
        sandboxed_tool_wrapper,
        "create_workspace_in_sandbox",
        fake_create_workspace_in_sandbox,
    )

    await ToolFactory.create_all_tools(_FakeConfig(calls))

    assert "sandbox_exec" in calls
    assert "release_db" in calls
    # The DB release happens after the last config DB reads (overrides /
    # allowlist) and before the sandbox workspace exec.
    assert calls.index("release_db") > calls.index("load_overrides")
    assert calls.index("release_db") > calls.index("load_allowlist")
    assert calls.index("release_db") < calls.index("sandbox_exec")
