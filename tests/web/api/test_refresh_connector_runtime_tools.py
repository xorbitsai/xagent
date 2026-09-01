"""``AgentServiceManager.refresh_connector_runtime_tools`` busts a cached
agent's connector runtime/MCP config cache without rebinding
``_connector_runtime_turn_id`` - see the method's own docstring in
``chat.py`` for why a websocket resume must not pass a fabricated turn id
through ``get_agent_for_task`` instead.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from xagent.web.api.chat import AgentServiceManager


def test_refresh_connector_runtime_tools_invalidates_cache_and_tools():
    manager = AgentServiceManager()
    agent = MagicMock()
    manager._agents[42] = agent

    manager.refresh_connector_runtime_tools(42)

    agent.tool_config.invalidate_connector_runtime_cache.assert_called_once_with()
    agent.invalidate_tools.assert_called_once_with()
    # Only the cache is busted - the turn id itself must never be touched
    # here, unlike ``set_connector_runtime_turn_id``.
    agent.tool_config.set_connector_runtime_turn_id.assert_not_called()


def test_refresh_connector_runtime_tools_is_noop_for_uncached_task():
    manager = AgentServiceManager()

    # No agent cached for this task id - must not raise.
    manager.refresh_connector_runtime_tools(999)


def test_refresh_connector_runtime_tools_is_noop_when_tool_config_lacks_support():
    manager = AgentServiceManager()
    agent = MagicMock()
    agent.tool_config = object()
    manager._agents[42] = agent

    manager.refresh_connector_runtime_tools(42)

    agent.invalidate_tools.assert_not_called()


def test_refresh_connector_runtime_tools_is_noop_when_tool_config_is_none():
    manager = AgentServiceManager()
    agent = MagicMock()
    agent.tool_config = None
    manager._agents[42] = agent

    manager.refresh_connector_runtime_tools(42)

    agent.invalidate_tools.assert_not_called()
