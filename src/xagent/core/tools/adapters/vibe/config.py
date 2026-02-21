"""
Tool Configuration Management

Provides abstract and concrete configuration classes for tool creation.
This allows different contexts (web, standalone) to provide configuration
to the ToolFactory in a unified way.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


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
    def get_mcp_server_configs(self) -> List[Dict[str, Any]]:
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
    def get_image_edit_model(self) -> Optional[Any]:
        """Get default image editing model."""
        pass


class ToolConfig(BaseToolConfig):
    """Tool configuration that uses provided config dict for standalone usage."""

    def __init__(self, config_dict: Dict[str, Any]):
        # Extract configurations from dict
        workspace_config = config_dict.get("workspace")
        config_dict.get("vision_model")  # Unused in base config
        config_dict.get("image_models", [])  # Unused in base config
        mcp_server_configs = config_dict.get("mcp_servers", [])
        file_tools_enabled = config_dict.get("file_tools_enabled", True)
        basic_tools_enabled = config_dict.get("basic_tools_enabled", True)
        embedding_model = config_dict.get("embedding_model")
        browser_tools_enabled = config_dict.get("browser_tools_enabled", True)
        task_id = config_dict.get("task_id")
        allowed_collections = config_dict.get("allowed_collections")
        allowed_skills = config_dict.get("allowed_skills")
        user_id = config_dict.get("user_id")
        is_admin = config_dict.get("is_admin", False)

        self.workspace_config: Optional[Dict[str, Any]] = workspace_config
        self.vision_model: Optional[Any] = (
            None  # Standalone usage typically doesn't have web context
        )
        self.image_models: Dict[
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
        self.user_id: Optional[int] = user_id
        self.is_admin_value: bool = bool(is_admin)

    def get_workspace_config(self) -> Optional[Dict[str, Any]]:
        return self.workspace_config

    def get_vision_model(self) -> Optional[Any]:
        return self.vision_model

    def get_image_models(self) -> Dict[str, Any]:
        return self.image_models

    def get_mcp_server_configs(self) -> List[Dict[str, Any]]:
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

    def get_image_edit_model(self) -> Optional[Any]:
        return None  # Standalone config doesn't have web context
