"""
Sandboxed tool configuration management module

Load all tool configurations from config file 'sandboxed_tool_config.yml'
"""

import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


class SandboxedToolConfig:
    """Sandboxed tool configuration"""

    def __init__(
        self,
        sandbox_enabled: bool = False,
        packages: list[str] | None = None,
        env_vars: list[str] | None = None,
        tool_class: str | None = None,
    ):
        """
        Initialize sandboxed tool configuration

        Args:
            sandbox_enabled: Whether to enable sandbox
            packages: Dependency package list
            env_vars: Environment variable list
            tool_class: module.path:ClassName         # Reconstruct tool object in sandbox
                tool = Class(); result = tool.run_json_sync(args)
        """
        self.sandbox_enabled = sandbox_enabled
        self.packages = packages or []
        self.env_vars = env_vars or []
        self.tool_class = tool_class

    def __repr__(self) -> str:
        return (
            f"SandboxedToolConfig(sandbox_enabled={self.sandbox_enabled}, "
            f"packages={self.packages}, env_vars={self.env_vars}, "
            f"tool_class={self.tool_class})"
        )


class SandboxedToolConfigManager:
    """Sandboxed tool configuration manager"""

    _instance: "SandboxedToolConfigManager | None" = None
    _config: dict[str, SandboxedToolConfig] | None = None

    def __new__(cls) -> "SandboxedToolConfigManager":
        """Singleton pattern"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize configuration manager"""
        if self._config is None:
            self._load_config()

    def _load_config(self) -> None:
        """Load all tool configurations from config file"""
        config_file = Path(__file__).parent / "sandboxed_tool_config.yml"

        if not config_file.exists():
            logger.warning(f"Sandboxed tool config not found: {config_file}")
            self._config = {}
            return

        try:
            with open(config_file, "r", encoding="utf-8") as f:
                raw_config = yaml.safe_load(f)

            if not raw_config:
                logger.warning("Empty sandboxed tool config")
                self._config = {}
                return

            # Parse configuration
            self._config = {}
            for tool_name, tool_config in raw_config.items():
                if tool_config is None:
                    self._config[tool_name] = SandboxedToolConfig()
                    continue

                if not isinstance(tool_config, dict):
                    logger.warning(
                        f"Invalid config format for {tool_name}: {tool_config}"
                    )
                    self._config[tool_name] = SandboxedToolConfig()
                    continue

                self._config[tool_name] = SandboxedToolConfig(
                    sandbox_enabled=tool_config.get("sandbox_enabled", False),
                    packages=tool_config.get("packages", []),
                    env_vars=tool_config.get("env_vars", []),
                    tool_class=tool_config.get("tool_class"),
                )

            logger.debug(f"Loaded config for {len(self._config)} tools")

        except Exception as e:
            logger.error(f"Error loading sandboxed tool config: {e}")
            self._config = {}

    def get_config(self, tool_name: str) -> SandboxedToolConfig:
        """
        Get sandbox configuration for a tool

        Args:
            tool_name: Tool name (tool.name)

        Returns:
            Sandboxed tool configuration
        """
        if self._config is None:
            self._load_config()

        assert self._config is not None
        return self._config.get(tool_name, SandboxedToolConfig())

    def is_sandbox_enabled(self, tool_name: str) -> bool:
        """
        Check if sandbox is enabled for a tool

        Args:
            tool_name: Tool name (tool.name)

        Returns:
            Whether sandbox is enabled
        """
        config = self.get_config(tool_name)
        return config.sandbox_enabled

    def reload(self) -> None:
        """Reload configuration file"""
        self._config = None
        self._load_config()


# Global configuration manager instance
_config_manager = SandboxedToolConfigManager()


def get_sandbox_tool_config(tool_name: str) -> SandboxedToolConfig:
    """
    Get sandbox configuration for a tool

    Args:
        tool_name: Tool name (tool.name)

    Returns:
        Sandboxed tool configuration
    """
    return _config_manager.get_config(tool_name)


def is_sandbox_enabled(tool_name: str) -> bool:
    """
    Check if sandbox is enabled for a tool

    Args:
        tool_name: Tool name (tool.name)

    Returns:
        Whether sandbox is enabled
    """
    return _config_manager.is_sandbox_enabled(tool_name)


def reload_config() -> None:
    """Reload configuration file"""
    _config_manager.reload()
