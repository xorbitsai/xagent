"""
Web-specific tool configuration for xagent

Provides web-specific configuration classes that load from database
and other web-specific sources.
"""

import logging
from typing import Any, Dict, List, Optional

from ...core.tools.adapters.vibe.config import BaseToolConfig


class WebToolConfig(BaseToolConfig):
    """Web-specific tool configuration that loads from database."""

    def __init__(
        self,
        db: Any,
        request: Any,
        user_id: Optional[int] = None,
        is_admin: bool = False,
        workspace_config: Optional[Dict[str, Any]] = None,
        vision_model: Optional[Any] = None,
        include_mcp_tools: bool = True,
        task_id: Optional[str] = None,
        workspace_base_dir: str = "uploads",
        browser_tools_enabled: bool = True,
        allowed_collections: Optional[List[str]] = None,
        allowed_skills: Optional[List[str]] = None,
        allowed_tools: Optional[List[str]] = None,
    ):
        self.db = db
        self.request = request
        self._user_id = user_id or self._get_user_id_from_request(request)
        self._is_admin_value = is_admin or self._get_is_admin_from_request(request)
        # Initialize workspace_config with base_dir and task_id if provided
        if workspace_config is None:
            workspace_config = {}
        if task_id:
            workspace_config["task_id"] = task_id
        # Ensure base_dir is in workspace_config (required by ToolFactory._create_workspace)
        if "base_dir" not in workspace_config:
            workspace_config["base_dir"] = workspace_base_dir
        self._workspace_config = workspace_config
        self._explicit_vision_model = vision_model
        self._include_mcp_tools = include_mcp_tools
        self._task_id = task_id
        self._browser_tools_enabled = browser_tools_enabled
        self._allowed_collections = allowed_collections
        self._allowed_skills = allowed_skills
        self._allowed_tools = allowed_tools
        self._excluded_agent_id: Optional[int] = None

        # Cache for loaded configurations
        self._cached_vision_config: Optional[Any] = None
        self._cached_image_configs: Optional[Dict[str, Any]] = None
        self._cached_image_generate_model: Optional[Any] = None
        self._cached_image_edit_model: Optional[Any] = None
        self._cached_mcp_configs: Optional[List[Dict[str, Any]]] = None
        self._cached_embedding_model: Optional[str] = None

    def _get_user_id_from_request(self, request: Any) -> int:
        """Extract user ID from request using JWT authentication."""
        try:
            from ..auth_dependencies import get_user_from_websocket_token

            # Check if this is a FastAPI request with proper authentication
            if hasattr(request, "headers") and hasattr(request, "query_params"):
                # Try to extract user from Authorization header
                auth_header = request.headers.get("authorization")
                if auth_header:
                    user = get_user_from_websocket_token(auth_header, self.db)
                    if user:
                        return int(user.id)

            # If request has a user attribute directly, use it
            if hasattr(request, "user") and request.user:
                return int(request.user.id)

            # If no authentication, this should raise an exception
            raise ValueError("Authentication required")

        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Failed to get user ID from request: {e}")
            # Fallback to default user ID for backward compatibility
            # In production, this should raise an exception instead
            return 1

    def _get_is_admin_from_request(self, request: Any) -> bool:
        """Extract is_admin flag from request."""
        try:
            # If request has a user attribute directly, check is_admin
            if hasattr(request, "user") and request.user:
                return bool(request.user.is_admin)

            return False

        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to get is_admin from request: {e}")
            return False

    def get_workspace_config(self) -> Optional[Dict[str, Any]]:
        """Get workspace configuration."""
        return self._workspace_config

    def get_file_tools_enabled(self) -> bool:
        """Whether to include file tools."""
        return True

    def get_basic_tools_enabled(self) -> bool:
        """Whether to include basic tools."""
        return True

    def get_vision_model(self) -> Optional[Any]:
        """Get vision model, prioritizing explicitly provided model over database."""
        if hasattr(self, "_explicit_vision_model") and self._explicit_vision_model:
            return self._explicit_vision_model

        if self._cached_vision_config is None:
            self._cached_vision_config = self._load_vision_model()
        return self._cached_vision_config

    def get_image_models(self) -> Dict[str, Any]:
        """Load image models from database."""
        if self._cached_image_configs is None:
            self._cached_image_configs = self._load_image_models()
        return self._cached_image_configs

    def get_image_generate_model(self) -> Optional[Any]:
        """Get default image generation model from database."""
        if self._cached_image_generate_model is None:
            self._cached_image_generate_model = self._load_image_generate_model()
        return self._cached_image_generate_model

    def get_image_edit_model(self) -> Optional[Any]:
        """Get default image editing model from database."""
        if self._cached_image_edit_model is None:
            self._cached_image_edit_model = self._load_image_edit_model()
        return self._cached_image_edit_model

    def get_mcp_server_configs(self) -> List[Dict[str, Any]]:
        """Load MCP server configurations from database."""
        if self._cached_mcp_configs is None:
            self._cached_mcp_configs = self._load_mcp_server_configs()
        return self._cached_mcp_configs

    def get_embedding_model(self) -> Optional[str]:
        """Load default embedding model ID from database."""
        if self._cached_embedding_model is None:
            self._cached_embedding_model = self._load_embedding_model()
        return self._cached_embedding_model

    def get_browser_tools_enabled(self) -> bool:
        """Whether to include browser automation tools."""
        return self._browser_tools_enabled

    def get_task_id(self) -> Optional[str]:
        """Get task ID for session tracking."""
        return self._task_id

    def get_allowed_collections(self) -> Optional[List[str]]:
        """Get allowed knowledge base collections. None means all collections are allowed."""
        return self._allowed_collections

    def get_allowed_skills(self) -> Optional[List[str]]:
        """Get allowed skill names. None means all skills are allowed."""
        return self._allowed_skills

    def get_allowed_tools(self) -> Optional[List[str]]:
        """Get allowed tool names. None means all tools are allowed."""
        return self._allowed_tools

    def get_excluded_agent_id(self) -> Optional[int]:
        """Get agent ID to exclude from agent tools (to prevent self-calls)."""
        return getattr(self, "_excluded_agent_id", None)

    def get_user_id(self) -> Optional[int]:
        """Get current user ID for multi-tenancy."""
        return self._user_id

    def get_db(self) -> Any:
        """Get database session."""
        return self.db

    def is_admin(self) -> bool:
        """Whether current user is admin."""
        return self._is_admin_value

    def get_enable_agent_tools(self) -> bool:
        """Whether to include published agents as tools."""
        return True

    def _load_embedding_model(self) -> Optional[str]:
        """Load embedding model ID from database via model service."""
        from ...web.services.model_service import get_default_embedding_model

        return get_default_embedding_model(self._user_id)

    def _load_vision_model(self) -> Optional[Any]:
        """Load vision model from database via model service."""
        try:
            from ...web.services.model_service import get_default_vision_model

            return get_default_vision_model(self._user_id)

        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to load vision model: {e}")
            return None

    def _load_image_models(self) -> Dict[str, Any]:
        """Load image models from database via model service."""
        try:
            from ...web.services.model_service import get_image_models

            return get_image_models(self.db, self._user_id)

        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to load image models: {e}")

            return {}

    def _load_image_generate_model(self) -> Optional[Any]:
        """Load default image generation model from database via model service."""
        try:
            from ...web.services.model_service import get_default_image_generate_model

            return get_default_image_generate_model(self._user_id)

        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to load default image generation model: {e}")
            return None

    def _load_image_edit_model(self) -> Optional[Any]:
        """Load default image editing model from database via model service."""
        try:
            from ...web.services.model_service import get_default_image_edit_model

            return get_default_image_edit_model(self._user_id)

        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to load default image editing model: {e}")
            return None

    def _load_mcp_server_configs(self) -> List[Dict[str, Any]]:
        """Load MCP server configurations from database with user context."""
        logger = logging.getLogger(__name__)
        configs = []

        try:
            from ...web.models.mcp import MCPServer, UserMCPServer

            # Query active MCP servers for this user
            servers = (
                self.db.query(MCPServer)
                .join(UserMCPServer, MCPServer.id == UserMCPServer.mcpserver_id)
                .filter(UserMCPServer.user_id == self._user_id, UserMCPServer.is_active)
                .all()
            )

            logger.info(
                f"Found {len(servers)} active MCP servers for user {self._user_id}"
            )

            for server in servers:
                # Build config dict from server model
                config = {
                    "name": server.name,
                    "transport": server.transport,
                    "description": server.description,
                }

                # Add transport-specific configuration
                transport_config = {}

                if server.transport == "stdio":
                    if server.command:
                        transport_config["command"] = server.command
                    if server.args:
                        transport_config["args"] = server.args
                    if server.env:
                        transport_config["env"] = server.env
                    if server.cwd:
                        transport_config["cwd"] = server.cwd

                elif server.transport in ["sse", "websocket", "streamable_http"]:
                    if server.url:
                        transport_config["url"] = server.url
                    if server.headers:
                        transport_config["headers"] = server.headers

                # Add Docker-specific config if managed internally
                if server.managed == "internal":
                    if server.docker_url:
                        transport_config["docker_url"] = server.docker_url
                    if server.docker_image:
                        transport_config["docker_image"] = server.docker_image
                    if server.docker_environment:
                        transport_config["docker_environment"] = (
                            server.docker_environment
                        )
                    if server.docker_working_dir:
                        transport_config["docker_working_dir"] = (
                            server.docker_working_dir
                        )
                    if server.volumes:
                        transport_config["volumes"] = server.volumes
                    if server.bind_ports:
                        transport_config["bind_ports"] = server.bind_ports
                    if server.restart_policy:
                        transport_config["restart_policy"] = server.restart_policy
                    if server.auto_start is not None:
                        transport_config["auto_start"] = server.auto_start

                config["config"] = transport_config

                # Add user context for MCP tool isolation
                config["user_id"] = str(self._user_id)
                config["allow_users"] = [str(self._user_id)]  # Only allow current user

                configs.append(config)
                logger.debug(
                    f"Loaded MCP server config: {server.name} ({server.transport})"
                )

        except Exception as e:
            logger.warning(f"Failed to load MCP server configs: {e}", exc_info=True)

        logger.info(f"Loaded {len(configs)} MCP server configurations")
        return configs
