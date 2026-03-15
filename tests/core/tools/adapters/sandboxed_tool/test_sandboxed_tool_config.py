"""
Unit tests for sandboxed tool configuration module
"""

from unittest.mock import mock_open, patch

from src.xagent.core.tools.adapters.vibe.sandboxed_tool.sandboxed_tool_config import (
    SandboxedToolConfig,
    SandboxedToolConfigManager,
    get_sandbox_tool_config,
    is_sandbox_enabled,
)

# Mock configuration data
MOCK_CONFIG_YAML = """
# Test configuration
execute_python_code:
  sandbox_enabled: true
  tool_class: "xagent.core.tools.adapters.vibe.python_executor:PythonExecutorTool"
  packages: []
  env_vars: []

web_search:
  sandbox_enabled: true
  tool_class: "xagent.core.tools.adapters.vibe.web_search:WebSearchTool"
  packages:
    - httpx
    - httpx2
  env_vars:
    - GOOGLE_API_KEY
    - GOOGLE_CSE_ID
"""


class TestSandboxedToolConfig:
    """Test SandboxedToolConfig class"""

    def test_default_config(self):
        """Test default configuration"""
        config = SandboxedToolConfig()
        assert config.sandbox_enabled is False
        assert config.tool_class is None
        assert config.packages == []
        assert config.env_vars == []

    def test_custom_config(self):
        """Test custom configuration"""
        config = SandboxedToolConfig(
            sandbox_enabled=True,
            tool_class="test.module:TestClass",
            packages=["httpx"],
            env_vars=["API_KEY"],
        )
        assert config.sandbox_enabled is True
        assert config.tool_class == "test.module:TestClass"
        assert config.packages == ["httpx"]
        assert config.env_vars == ["API_KEY"]


class TestConfigLoading:
    """Test configuration loading"""

    def setup_method(self):
        """Reset SandboxedToolConfigManager to ensure mock success"""
        SandboxedToolConfigManager._instance = None
        if hasattr(SandboxedToolConfigManager, "_config"):
            SandboxedToolConfigManager._config = None
        import src.xagent.core.tools.adapters.vibe.sandboxed_tool.sandboxed_tool_config as config_module

        if hasattr(config_module, "_config_manager"):
            config_module._config_manager._config = None

    @patch("builtins.open", new_callable=mock_open, read_data=MOCK_CONFIG_YAML)
    @patch("pathlib.Path.exists", return_value=True)
    def test_get_sandboxed_tool_config(self, mock_exists, mock_file):
        """Test getting sandbox-enabled tool configuration"""
        config = get_sandbox_tool_config("execute_python_code")
        assert config.sandbox_enabled is True
        assert (
            config.tool_class
            == "xagent.core.tools.adapters.vibe.python_executor:PythonExecutorTool"
        )
        assert isinstance(config.packages, list)
        assert isinstance(config.env_vars, list)

    @patch("builtins.open", new_callable=mock_open, read_data=MOCK_CONFIG_YAML)
    @patch("pathlib.Path.exists", return_value=True)
    def test_get_unknown_tool_config(self, mock_exists, mock_file):
        """Test getting unknown tool configuration (returns default values)"""
        config = get_sandbox_tool_config("unknown_tool")
        assert config.sandbox_enabled is False
        assert config.packages == []
        assert config.env_vars == []
        assert config.tool_class is None

    @patch("builtins.open", new_callable=mock_open, read_data=MOCK_CONFIG_YAML)
    @patch("pathlib.Path.exists", return_value=True)
    def test_is_sandbox_enabled(self, mock_exists, mock_file):
        """Test checking if tool has sandbox enabled"""
        assert is_sandbox_enabled("execute_python_code") is True
        assert is_sandbox_enabled("web_search") is True
        assert is_sandbox_enabled("unknown_tool") is False

    @patch("builtins.open", new_callable=mock_open, read_data=MOCK_CONFIG_YAML)
    @patch("pathlib.Path.exists", return_value=True)
    def test_tool_with_dependencies(self, mock_exists, mock_file):
        """Test tool configuration with dependencies"""
        config = get_sandbox_tool_config("web_search")
        assert config.sandbox_enabled is True
        assert "httpx" in config.packages
        assert "httpx2" in config.packages
        assert "GOOGLE_API_KEY" in config.env_vars
        assert "GOOGLE_CSE_ID" in config.env_vars
