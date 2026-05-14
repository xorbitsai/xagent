"""
Tool Configuration Management

Provides abstract and concrete configuration classes for tool creation.
This allows different contexts (web, standalone) to provide configuration
to the ToolFactory in a unified way.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from ..... import config as _root_config


class BaseToolConfig(ABC):
    """Abstract base class for tool configuration."""

    @abstractmethod
    def get_workspace_config(self) -> Optional[Dict[str, Any]]:
        """Get workspace configuration."""
        pass

    @abstractmethod
    def get_vision_model(self) -> Optional[Any]:
        """Get vision model."""
        pass

    @abstractmethod
    def get_image_models(self) -> Dict[str, Any]:
        """Get image models."""
        pass

    @abstractmethod
    def get_asr_models(self) -> Dict[str, Any]:
        """Get ASR (speech-to-text) models."""
        pass

    @abstractmethod
    def get_tts_models(self) -> Dict[str, Any]:
        """Get TTS (text-to-speech) models."""
        pass

    @abstractmethod
    async def get_mcp_server_configs(self) -> List[Dict[str, Any]]:
        """Get MCP server configurations."""
        pass

    @abstractmethod
    def get_file_tools_enabled(self) -> bool:
        """Whether to include file tools."""
        pass

    @abstractmethod
    def get_basic_tools_enabled(self) -> bool:
        """Whether to include basic tools."""
        pass

    @abstractmethod
    def get_embedding_model(self) -> Optional[str]:
        """Get embedding model ID."""
        pass

    @abstractmethod
    def get_browser_tools_enabled(self) -> bool:
        """Whether to include browser automation tools."""
        pass

    @abstractmethod
    def get_task_id(self) -> Optional[str]:
        """Get task ID for session tracking."""
        pass

    @abstractmethod
    def get_allowed_collections(self) -> Optional[List[str]]:
        """Get allowed knowledge base collections. None means all collections are allowed."""
        pass

    @abstractmethod
    def get_allowed_skills(self) -> Optional[List[str]]:
        """Get allowed skill names. None means all skills are allowed."""
        pass

    @abstractmethod
    def get_user_id(self) -> Optional[int]:
        """Get current user ID for multi-tenancy."""
        pass

    @abstractmethod
    def is_admin(self) -> bool:
        """Whether current user is admin."""
        pass

    @abstractmethod
    def get_enable_agent_tools(self) -> bool:
        """Whether to include published agents as tools."""
        pass

    @abstractmethod
    def get_image_generate_model(self) -> Optional[Any]:
        """Get default image generation model."""
        pass

    @abstractmethod
    def get_custom_api_configs(self) -> List[Dict[str, Any]]:
        """Get custom API configurations."""
        pass

    @abstractmethod
    def get_image_edit_model(self) -> Optional[Any]:
        """Get default image editing model."""
        pass

    @abstractmethod
    def get_sandbox(self) -> Optional[Any]:
        """Get sandbox instance for sandboxed executors. Returns None if not available."""
        pass

    def get_tool_credential(self, tool_name: str, field_name: str) -> Optional[str]:
        return None

    def get_sql_connections(self) -> Dict[str, str]:
        return {}

    @abstractmethod
    def get_db(self) -> Optional[Any]:
        """Get database session. Returns None for standalone usage."""
        pass

    @abstractmethod
    def get_asr_model(self) -> Optional[Any]:
        """Get default ASR (speech-to-text) model."""
        pass

    @abstractmethod
    def get_tts_model(self) -> Optional[Any]:
        """Get default TTS (text-to-speech) model."""
        pass

    @abstractmethod
    def get_llm(self) -> Optional[Any]:
        """Get default LLM for general tasks."""
        pass

    def get_max_output_length(self) -> int:
        """Get maximum output length in characters.

        Reads from XAGENT_TOOL_MAX_OUTPUT_LENGTH env var if set.
        See :mod:`xagent.config` for details.
        """
        return _root_config.get_tool_max_output_length()

    def get_max_field_count(self) -> int:
        """Get maximum number of fields/items in dict/list for output filtering.

        Reads from XAGENT_TOOL_MAX_FIELD_COUNT env var if set.
        See :mod:`xagent.config` for details.
        """
        return _root_config.get_tool_max_field_count()

    def get_max_recursion_depth(self) -> int:
        """Get maximum recursion depth for output filtering.

        Reads from XAGENT_TOOL_MAX_RECURSION_DEPTH env var if set.
        See :mod:`xagent.config` for details.
        """
        return _root_config.get_tool_max_recursion_depth()


class ToolConfig(BaseToolConfig):
    """Tool configuration that uses provided config dict for standalone usage."""

    def __init__(self, config_dict: Dict[str, Any]):
        # Extract configurations from dict
        workspace_config = config_dict.get("workspace")
        config_dict.get("vision_model")  # Unused in base config
        config_dict.get("image_models", [])  # Unused in base config
        config_dict.get("asr_models", [])  # Unused in base config
        config_dict.get("tts_models", [])  # Unused in base config
        mcp_server_configs = config_dict.get("mcp_servers", [])
        file_tools_enabled = config_dict.get("file_tools_enabled", True)
        basic_tools_enabled = config_dict.get("basic_tools_enabled", True)
        embedding_model = config_dict.get("embedding_model")
        browser_tools_enabled = config_dict.get("browser_tools_enabled", True)
        task_id = config_dict.get("task_id")
        allowed_collections = config_dict.get("allowed_collections")
        allowed_skills = config_dict.get("allowed_skills")
        allowed_tools = config_dict.get("allowed_tools")
        user_id = config_dict.get("user_id")
        is_admin = config_dict.get("is_admin", False)
        tool_credentials = config_dict.get("tool_credentials", {})

        # Output limit configuration (uses environment variable as default)
        # Store custom values if provided, otherwise use None to fall back to base class defaults
        self._custom_max_output_length: int | None = None
        try:
            self._custom_max_output_length = int(
                config_dict.get("max_output_length")  # type: ignore[arg-type]
            )
        except (TypeError, ValueError):
            pass
        self._custom_max_field_count: int | None = None
        try:
            self._custom_max_field_count = int(
                config_dict.get("max_field_count")  # type: ignore[arg-type]
            )
        except (TypeError, ValueError):
            pass
        self._custom_max_recursion_depth: int | None = None
        try:
            self._custom_max_recursion_depth = int(
                config_dict.get("max_recursion_depth")  # type: ignore[arg-type]
            )
        except (TypeError, ValueError):
            pass

        self.workspace_config: Optional[Dict[str, Any]] = workspace_config
        self.vision_model: Optional[Any] = (
            None  # Standalone usage typically doesn't have web context
        )
        self.image_models: Dict[
            str, Any
        ] = {}  # Standalone usage typically doesn't have web context
        self.asr_models: Dict[
            str, Any
        ] = {}  # Standalone usage typically doesn't have web context
        self.tts_models: Dict[
            str, Any
        ] = {}  # Standalone usage typically doesn't have web context
        self.mcp_server_configs: List[Dict[str, Any]] = mcp_server_configs
        self.file_tools_enabled: bool = bool(file_tools_enabled)
        self.basic_tools_enabled: bool = bool(basic_tools_enabled)
        self.embedding_model: Optional[str] = embedding_model
        self.browser_tools_enabled: bool = bool(browser_tools_enabled)
        self.task_id: Optional[str] = task_id
        self.allowed_collections: Optional[List[str]] = allowed_collections
        self.allowed_skills: Optional[List[str]] = allowed_skills
        self.allowed_tools: Optional[List[str]] = allowed_tools
        self.user_id: Optional[int] = user_id
        self.is_admin_value: bool = bool(is_admin)
        self.tool_credentials: Dict[str, Dict[str, str]] = tool_credentials

    def get_workspace_config(self) -> Optional[Dict[str, Any]]:
        return self.workspace_config

    def get_vision_model(self) -> Optional[Any]:
        return self.vision_model

    def get_image_models(self) -> Dict[str, Any]:
        return self.image_models

    def get_asr_models(self) -> Dict[str, Any]:
        return self.asr_models

    def get_tts_models(self) -> Dict[str, Any]:
        return self.tts_models

    async def get_mcp_server_configs(self) -> List[Dict[str, Any]]:
        return self.mcp_server_configs

    def get_file_tools_enabled(self) -> bool:
        return self.file_tools_enabled

    def get_basic_tools_enabled(self) -> bool:
        return self.basic_tools_enabled

    def get_embedding_model(self) -> Optional[str]:
        return self.embedding_model

    def get_browser_tools_enabled(self) -> bool:
        return self.browser_tools_enabled

    def get_task_id(self) -> Optional[str]:
        return self.task_id

    def get_allowed_collections(self) -> Optional[List[str]]:
        return self.allowed_collections

    def get_allowed_skills(self) -> Optional[List[str]]:
        return self.allowed_skills

    def get_user_id(self) -> Optional[int]:
        return self.user_id

    def is_admin(self) -> bool:
        return self.is_admin_value

    def get_enable_agent_tools(self) -> bool:
        return True

    def get_image_generate_model(self) -> Optional[Any]:
        return None  # Standalone config doesn't have web context

    def get_custom_api_configs(self) -> List[Dict[str, Any]]:
        return []  # Standalone config doesn't have web context for custom APIs by default

    def get_image_edit_model(self) -> Optional[Any]:
        return None  # Standalone config doesn't have web context

    def get_asr_model(self) -> Optional[Any]:
        return None  # Standalone config doesn't have web context

    def get_tts_model(self) -> Optional[Any]:
        return None  # Standalone config doesn't have web context

    def get_llm(self) -> Optional[Any]:
        return None  # Standalone config doesn't have web context

    def get_allowed_tools(self) -> Optional[List[str]]:
        return self.allowed_tools

    def get_sandbox(self) -> Optional[Any]:
        return None  # Standalone config doesn't have sandbox

    def get_tool_credential(self, tool_name: str, field_name: str) -> Optional[str]:
        tool_data = self.tool_credentials.get(tool_name)
        if not isinstance(tool_data, dict):
            return None
        value = tool_data.get(field_name)
        return value if isinstance(value, str) and value else None

    def get_sql_connections(self) -> Dict[str, str]:
        return {}

    def get_max_output_length(self) -> int:
        if self._custom_max_output_length is not None:
            return self._custom_max_output_length
        return super().get_max_output_length()

    def get_max_field_count(self) -> int:
        if self._custom_max_field_count is not None:
            return self._custom_max_field_count
        return super().get_max_field_count()

    def get_max_recursion_depth(self) -> int:
        if self._custom_max_recursion_depth is not None:
            return self._custom_max_recursion_depth
        return super().get_max_recursion_depth()

    def get_db(self) -> Optional[Any]:
        """ToolConfig (standalone) does not have database access."""
        return None
