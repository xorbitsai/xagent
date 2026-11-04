from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from xagent.core.agent.trace import SYSTEM_INFO
from xagent.core.tools.adapters.vibe.config import (
    MCPToolLoadSummary,
    MCPUnavailableSummary,
)
from xagent.web.tools.config import WebToolConfig


@pytest.mark.asyncio
async def test_web_tool_config_emits_safe_audit_only_mcp_summary() -> None:
    tracer = AsyncMock()
    workforce_parent_tracer = object()
    config = WebToolConfig(
        db=None,
        request=None,
        user_id=7,
        parent_tracer=workforce_parent_tracer,
        mcp_load_summary_tracer=tracer,
        mcp_load_summary_trace_task_id="42",
    )
    summary = MCPToolLoadSummary(
        requested_servers=("Gmail", "Slack", "Notion"),
        loaded_servers=("Gmail",),
        failures=(
            MCPUnavailableSummary("Slack", "initialize"),
            MCPUnavailableSummary.from_values(
                "Notion",
                "initialize",
                cause_type="HTTPStatusError",
                status_code=403,
            ),
        ),
        successful_tool_count=3,
    )

    await config.emit_mcp_load_summary(summary)

    tracer.trace_event.assert_awaited_once_with(
        SYSTEM_INFO,
        task_id="42",
        data={
            "__audit_only__": True,
            "event_type": "mcp_load_summary",
            "requested_servers": ["Gmail", "Slack", "Notion"],
            "loaded_servers": ["Gmail"],
            "failures": [
                {
                    "server_name": "Slack",
                    "reason": "initialize",
                    "error_type": None,
                    "status_code": None,
                },
                {
                    "server_name": "Notion",
                    "reason": "initialize",
                    "error_type": "HTTPStatusError",
                    "status_code": 403,
                },
            ],
            "successful_tool_count": 3,
        },
        require_persisted=False,
    )
    assert config.get_parent_tracer() is workforce_parent_tracer


@pytest.mark.asyncio
async def test_web_tool_config_summary_observer_defaults_to_noop() -> None:
    config = WebToolConfig(db=None, request=None, user_id=7)

    await config.emit_mcp_load_summary(MCPToolLoadSummary())
