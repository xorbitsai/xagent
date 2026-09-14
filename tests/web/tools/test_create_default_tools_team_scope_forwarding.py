"""``create_default_tools`` forwards ``agent_creator_user_id`` and
``declared_knowledge_bases`` into the ``WebToolConfig`` it builds, the same
way it already forwards ``connector_team_id``.

Scope: this covers the *definition* -- the signature at
``create_default_tools(...)`` and its body's construction of
``WebToolConfig(...)`` -- by calling the function directly with explicit
values, the way a call site would. It does NOT cover whether either of the
two real call sites in ``chat.py`` (``_build_tools_for_task`` and the
snapshot branch inside ``_get_agent_for_task_unlocked``) actually derives
and passes those values in the first place; that requires the surrounding
session/task/workforce machinery those methods are embedded in, which is
out of reach of a fast unit test. ``test_chat_team_scope_wiring_static.py``
(source-level guards) and ``test_chat_team_scope_wiring_behavioral.py``
(driving both call sites with a fake ``TaskSetupSnapshot`` and a spy on
``create_default_tools``) close that gap instead.

The arguments below are chosen to isolate forwarding, not to describe a
production call: both real call sites derive ``allowed_collections`` and
``declared_knowledge_bases`` from the same stored agent field, so the two
are equal by construction there. Passing a declaration while leaving
``allowed_collections`` unset keeps a dropped keyword from being masked by
a value that happens to arrive through the other one.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from xagent.core.tools.adapters.vibe.factory import ToolFactory


@pytest.mark.asyncio
async def test_create_default_tools_forwards_creator_and_declaration_to_web_tool_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xagent.web.services.agent_service_manager import create_default_tools

    captured: dict[str, Any] = {}

    class _FakeToolConfig:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

        def set_task_runtime_contribution(self, contribution: Any) -> None:
            pass

    async def create_tools(config: Any) -> list:
        return []

    monkeypatch.setattr("xagent.web.tools.config.WebToolConfig", _FakeToolConfig)
    monkeypatch.setattr(
        "xagent.web.models.database.get_session_local", lambda: object()
    )
    monkeypatch.setattr(ToolFactory, "create_all_tools", create_tools)

    await create_default_tools(
        None,
        user=SimpleNamespace(id=7, is_admin=False),
        task_id="web_task_11",
        connector_team_id=42,
        agent_creator_user_id=99,
        declared_knowledge_bases=["kb1", "kb2"],
    )

    assert captured.get("connector_team_id") == 42
    assert captured.get("agent_creator_user_id") == 99
    assert captured.get("declared_knowledge_bases") == ["kb1", "kb2"]


@pytest.mark.asyncio
async def test_create_default_tools_leaves_creator_and_declaration_unset_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Negative control: the common no-agent case must not accidentally
    populate either new value."""
    from xagent.web.services.agent_service_manager import create_default_tools

    captured: dict[str, Any] = {}

    class _FakeToolConfig:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

        def set_task_runtime_contribution(self, contribution: Any) -> None:
            pass

    async def create_tools(config: Any) -> list:
        return []

    monkeypatch.setattr("xagent.web.tools.config.WebToolConfig", _FakeToolConfig)
    monkeypatch.setattr(
        "xagent.web.models.database.get_session_local", lambda: object()
    )
    monkeypatch.setattr(ToolFactory, "create_all_tools", create_tools)

    await create_default_tools(
        None,
        user=SimpleNamespace(id=7, is_admin=False),
        task_id="web_task_11",
    )

    assert captured.get("agent_creator_user_id") is None
    assert captured.get("declared_knowledge_bases") is None
