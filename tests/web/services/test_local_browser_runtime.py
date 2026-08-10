from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from xagent.core.computer.native_browser import (
    LOCAL_BROWSER_TASK_EXTENSION,
)
from xagent.core.task_runtime import (
    TaskRuntimeClientError,
    TaskRuntimeContext,
    TaskRuntimeContribution,
    merge_task_runtime_contributions,
)
from xagent.core.tools.adapters.vibe.browser_tools import (
    _has_local_browser_runtime,
    create_browser_tools,
)
from xagent.web.models.task import Task
from xagent.web.models.user import User
from xagent.web.services.local_browser_runtime import (
    LocalBrowserTaskRuntimeProvider,
    register_local_browser_runtime,
    unregister_local_browser_runtime,
)
from xagent.web.services.task_runtime import (
    agent_config_with_task_extension_bindings,
    register_task_extension,
    registered_task_extensions,
    unregister_task_extension,
)


class FakeSession:
    def __init__(self, *, task: Any, user: Any) -> None:
        self.task = task
        self.user = user
        self.model: Any = None
        self.closed = False
        self.committed = False

    def query(self, model: Any) -> "FakeSession":
        self.model = model
        return self

    def filter(self, *_args: Any) -> "FakeSession":
        return self

    def first(self) -> Any:
        if self.model is Task:
            return self.task
        if self.model is User:
            return self.user
        raise AssertionError(f"unexpected query model: {self.model}")

    def close(self) -> None:
        self.closed = True

    def commit(self) -> None:
        self.committed = True


def make_context(
    *,
    bound: bool,
    admin: bool,
    workspace: Any = object(),
    extension: str = LOCAL_BROWSER_TASK_EXTENSION,
):
    agent_config = (
        agent_config_with_task_extension_bindings({}, [extension]) if bound else {}
    )
    sessions: list[FakeSession] = []
    task = SimpleNamespace(id=7, user_id=3, agent_config=agent_config)
    user = SimpleNamespace(id=3, is_admin=admin)

    def session_factory() -> FakeSession:
        session = FakeSession(task=task, user=user)
        sessions.append(session)
        return session

    return (
        TaskRuntimeContext(
            task_id=7,
            user_id=3,
            source="internal",
            session_factory=session_factory,
            workspace=workspace,
        ),
        sessions,
    )


def test_local_browser_registration_is_explicit_and_lifespan_scoped() -> None:
    unregister_task_extension(LOCAL_BROWSER_TASK_EXTENSION)
    try:
        register_local_browser_runtime()
        register_local_browser_runtime()
        assert registered_task_extensions().count(LOCAL_BROWSER_TASK_EXTENSION) == 1

        unregister_local_browser_runtime()
        assert LOCAL_BROWSER_TASK_EXTENSION not in registered_task_extensions()
    finally:
        unregister_task_extension(LOCAL_BROWSER_TASK_EXTENSION)


def test_local_browser_registration_rejects_a_foreign_provider_collision() -> None:
    unregister_task_extension(LOCAL_BROWSER_TASK_EXTENSION)
    try:
        register_task_extension(
            LOCAL_BROWSER_TASK_EXTENSION,
            LocalBrowserTaskRuntimeProvider(),
        )

        with pytest.raises(ValueError, match="already registered"):
            register_local_browser_runtime()
    finally:
        unregister_task_extension(LOCAL_BROWSER_TASK_EXTENSION)


def test_local_browser_create_requires_enablement_admin_and_valid_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = LocalBrowserTaskRuntimeProvider()
    context, sessions = make_context(bound=True, admin=True)
    monkeypatch.delenv("XAGENT_NATIVE_BROWSER_ENABLED", raising=False)

    with pytest.raises(TaskRuntimeClientError, match="disabled") as disabled:
        provider.on_task_created(context, {})
    assert disabled.value.status_code == 403

    monkeypatch.setenv("XAGENT_NATIVE_BROWSER_ENABLED", "true")
    non_admin, _ = make_context(bound=True, admin=False)
    with pytest.raises(TaskRuntimeClientError, match="administrators"):
        provider.on_task_created(non_admin, {})

    with pytest.raises(TaskRuntimeClientError, match="explicitly selected"):
        provider.on_task_created(context, {})

    provider.on_task_created(
        context,
        {
            "pid": 100,
            "window_id": 20,
            "application": "Google Chrome",
            "title": "Inbox",
        },
    )
    assert sessions[-1].committed is True

    with pytest.raises(TaskRuntimeClientError, match="perception_mode"):
        provider.on_task_created(
            context,
            {
                "pid": 100,
                "window_id": 20,
                "application": "Google Chrome",
                "perception_mode": "hidden_dom_fallback",
            },
        )

    with pytest.raises(TaskRuntimeClientError, match="only accepts Google Chrome"):
        provider.on_task_created(
            context,
            {
                "pid": 100,
                "window_id": 20,
                "application": "Music",
            },
        )

    monkeypatch.setenv("XAGENT_NATIVE_BROWSER_APP_NAME", "Terminal")
    with pytest.raises(TaskRuntimeClientError, match="supported browser"):
        provider.on_task_created(
            context,
            {
                "pid": 100,
                "window_id": 20,
                "application": "Terminal",
            },
        )


@pytest.mark.parametrize(
    ("configuration", "message"),
    [
        ({"window_id": 20, "application": "Google Chrome"}, "requires pid"),
        (
            {"pid": None, "window_id": 20, "application": "Google Chrome"},
            "requires pid and window_id",
        ),
        (
            {"pid": 100, "application": "Google Chrome"},
            "requires pid and window_id",
        ),
        (
            {"pid": True, "window_id": 20, "application": "Google Chrome"},
            "must be integers",
        ),
        (
            {"pid": "100", "window_id": 20, "application": "Google Chrome"},
            "must be integers",
        ),
        (
            {"pid": 100, "window_id": 20.0, "application": "Google Chrome"},
            "must be integers",
        ),
    ],
)
def test_local_browser_rejects_missing_or_non_integer_window_identity(
    monkeypatch: pytest.MonkeyPatch,
    configuration: dict[str, Any],
    message: str,
) -> None:
    monkeypatch.setenv("XAGENT_NATIVE_BROWSER_ENABLED", "true")
    provider = LocalBrowserTaskRuntimeProvider()
    context, _ = make_context(bound=True, admin=True)

    with pytest.raises(TaskRuntimeClientError, match=message):
        provider.on_task_created(context, configuration)


def test_local_browser_contributes_standard_computer_tool_only_when_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("XAGENT_NATIVE_BROWSER_ENABLED", "true")
    provider = LocalBrowserTaskRuntimeProvider()
    unbound, _ = make_context(bound=False, admin=True)
    assert provider.build_runtime(unbound) is None

    bound, sessions = make_context(bound=True, admin=True)
    provider.on_task_created(
        bound,
        {
            "pid": 100,
            "window_id": 20,
            "application": "Google Chrome",
        },
    )
    contribution = provider.build_runtime(bound)

    assert isinstance(contribution, TaskRuntimeContribution)
    assert [tool.name for tool in contribution.tools] == ["computer"]
    assert contribution.preferred_input_modalities == ("image",)
    assert "not a browser extension or remote relay" in (contribution.environment or "")
    assert sessions[-1].closed is True


@pytest.mark.asyncio
async def test_bound_local_browser_fails_closed_after_admin_demotion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("XAGENT_NATIVE_BROWSER_ENABLED", "true")
    provider = LocalBrowserTaskRuntimeProvider()
    context, sessions = make_context(bound=True, admin=True)
    provider.on_task_created(
        context,
        {
            "pid": 100,
            "window_id": 20,
            "application": "Google Chrome",
        },
    )
    sessions[-1].user.is_admin = False

    contribution = provider.build_runtime(context)

    assert isinstance(contribution, TaskRuntimeContribution)
    assert [tool.name for tool in contribution.tools] == ["computer"]
    assert (
        await create_browser_tools(
            SimpleNamespace(
                get_browser_tools_enabled=lambda: True,
                get_task_runtime_contribution=lambda: merge_task_runtime_contributions(
                    {LOCAL_BROWSER_TASK_EXTENSION: contribution}
                ),
            )
        )
        == []
    )
    result = await contribution.tools[0].run_json_async({})
    assert result["success"] is False
    assert "authorization was revoked" in result["error"]
    assert provider.public_metadata(context) == {
        "kind": "local_browser",
        "enabled": False,
        "reason": "authorization_revoked",
        "perception_mode": "auto",
        "control_transport": "native_accessibility",
    }


@pytest.mark.asyncio
async def test_local_browser_rechecks_admin_before_environment_creation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("XAGENT_NATIVE_BROWSER_ENABLED", "true")
    provider = LocalBrowserTaskRuntimeProvider()
    context, sessions = make_context(bound=True, admin=True)
    provider.on_task_created(
        context,
        {
            "pid": 100,
            "window_id": 20,
            "application": "Google Chrome",
        },
    )
    contribution = provider.build_runtime(context)
    assert isinstance(contribution, TaskRuntimeContribution)

    sessions[-1].user.is_admin = False
    result = await contribution.tools[0].run_json_async({})

    assert result["success"] is False
    assert "no longer an Xagent administrator" in result["error"]


@pytest.mark.asyncio
async def test_local_browser_binding_suppresses_colliding_playwright_family() -> None:
    contribution = merge_task_runtime_contributions(
        {
            LOCAL_BROWSER_TASK_EXTENSION: TaskRuntimeContribution(
                tools=(SimpleNamespace(name="computer"),)
            )
        }
    )
    config = SimpleNamespace(
        get_browser_tools_enabled=lambda: True,
        get_task_runtime_contribution=lambda: contribution,
    )

    assert await create_browser_tools(config) == []


def test_unbound_local_browser_provider_does_not_suppress_playwright() -> None:
    contribution = merge_task_runtime_contributions(
        {LOCAL_BROWSER_TASK_EXTENSION: None}
    )

    assert _has_local_browser_runtime(contribution) is False


def test_local_browser_public_metadata_is_bound_task_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("XAGENT_NATIVE_BROWSER_ENABLED", "true")
    provider = LocalBrowserTaskRuntimeProvider()
    unbound, _ = make_context(bound=False, admin=True)
    bound, _ = make_context(bound=True, admin=True)

    assert provider.public_metadata(unbound) is None
    assert provider.public_metadata(bound) == {
        "kind": "local_browser",
        "enabled": True,
        "perception_mode": "auto",
        "control_transport": "native_accessibility",
    }
