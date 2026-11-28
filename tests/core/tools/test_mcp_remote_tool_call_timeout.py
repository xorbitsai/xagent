"""Tests for the remote MCP tool call timeout config (issue #3251).

A remote MCP tool call can hang a turn past the SSE read timeout if the
call blocks on session init, an OAuth token refresh, or a reconnect.
This tests that the config function returns the expected defaults and
respects the environment variable.
"""

from xagent.config import get_mcp_remote_tool_call_timeout_seconds


class TestRemoteToolCallTimeoutConfig:
    """Test the config function for remote MCP tool call timeout."""

    def test_default_value(self, monkeypatch):
        """Default timeout is 300 seconds (matches SSE read timeout)."""
        monkeypatch.delenv("XAGENT_MCP_REMOTE_TOOL_CALL_TIMEOUT_SECONDS", raising=False)
        assert get_mcp_remote_tool_call_timeout_seconds() == 300

    def test_env_override(self, monkeypatch):
        """Environment variable overrides the default."""
        monkeypatch.setenv("XAGENT_MCP_REMOTE_TOOL_CALL_TIMEOUT_SECONDS", "120")
        assert get_mcp_remote_tool_call_timeout_seconds() == 120

    def test_zero_disables(self, monkeypatch):
        """Setting 0 disables the timeout."""
        monkeypatch.setenv("XAGENT_MCP_REMOTE_TOOL_CALL_TIMEOUT_SECONDS", "0")
        assert get_mcp_remote_tool_call_timeout_seconds() == 0
