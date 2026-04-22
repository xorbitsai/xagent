"""
Web-specific tool configuration for xagent

Provides web-specific configuration classes that load from database
and other web-specific sources.
"""

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import httpx

from ...config import get_uploads_dir
from ...core.tools.adapters.vibe.config import BaseToolConfig
from ..services.tool_credentials import get_sql_connection_map, resolve_tool_credential

logger = logging.getLogger(__name__)


async def refresh_oauth_token_if_needed(
    db: Any, oauth_account: Any, provider_name: str
) -> bool:
    """Check if token is expired (or close to expiring) and refresh if needed."""
    if not oauth_account.expires_at:
        return True  # Assume valid if no expiration is set

    # Check if expired (or expiring within 5 minutes)
    now = datetime.now(timezone.utc)

    # Handle timezone naive vs aware
    expires_at = oauth_account.expires_at
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)

    if expires_at > now + timedelta(minutes=5):
        return True  # Token is still valid

    if not oauth_account.refresh_token:
        logger.warning(
            f"Token expired for {provider_name} but no refresh_token available."
        )
        return False

    logger.info(f"Token expired for {provider_name}, attempting to refresh...")
    try:
        from ...core.utils.encryption import decrypt_value
        from ..models.oauth_provider import OAuthProvider

        provider_config = (
            db.query(OAuthProvider)
            .filter(OAuthProvider.provider_name == provider_name)
            .first()
        )
        if not provider_config:
            logger.warning(f"Unknown provider for refresh: {provider_name}")
            return False

        client_id = decrypt_value(provider_config.client_id)
        client_secret = decrypt_value(provider_config.client_secret)

        if not client_id or not client_secret:
            logger.warning(
                f"{provider_name} OAuth not configured (missing CLIENT_ID or SECRET)."
            )
            return False

        data = {
            "grant_type": "refresh_token",
            "refresh_token": oauth_account.refresh_token,
            "client_id": client_id,
            "client_secret": client_secret,
        }

        headers = {}
        if provider_name == "linkedin":
            headers["Content-Type"] = "application/x-www-form-urlencoded"

        async with httpx.AsyncClient() as client:
            response = await client.post(
                provider_config.token_url, data=data, headers=headers, timeout=10.0
            )

        if response.status_code == 200:
            data = response.json()
            if "access_token" in data:
                oauth_account.access_token = data["access_token"]
                if "refresh_token" in data:
                    oauth_account.refresh_token = data["refresh_token"]
                if "expires_in" in data:
                    oauth_account.expires_at = datetime.now(timezone.utc) + timedelta(
                        seconds=data["expires_in"]
                    )
                db.commit()
                logger.info(
                    f"Successfully refreshed {provider_name} token for user {oauth_account.user_id}"
                )
                return True
        else:
            logger.error(f"Failed to refresh {provider_name} token: {response.text}")

    except Exception as e:
        logger.error(
            f"Exception refreshing token for {provider_name}: {e}", exc_info=True
        )

    return False


class WebToolConfig(BaseToolConfig):
    """Web-specific tool configuration that loads from database."""

    @staticmethod
    def _coerce_user_id(value: Any) -> Optional[int]:
        return value if isinstance(value, int) else None

    def __init__(
        self,
        db: Any,
        request: Any,
        user_id: Optional[int] = None,
        is_admin: bool = False,
        workspace_config: Optional[Dict[str, Any]] = None,
        vision_model: Optional[Any] = None,
        llm: Optional[Any] = None,
        include_mcp_tools: bool = True,
        task_id: Optional[str] = None,
        workspace_base_dir: Optional[str] = None,
        browser_tools_enabled: bool = True,
        allowed_collections: Optional[List[str]] = None,
        allowed_skills: Optional[List[str]] = None,
        allowed_tools: Optional[List[str]] = None,
    ):
        self.db = db
        self.request = request
        self._user_id = (
            user_id if user_id is not None else self._get_user_id_from_request(request)
        )
        self._is_admin_value = is_admin or self._get_is_admin_from_request(request)
        # Initialize workspace_config with base_dir and task_id if provided
        if workspace_config is None:
            workspace_config = {}
        if task_id:
            workspace_config["task_id"] = task_id
        # Use uploads dir if workspace_base_dir not explicitly provided
        if workspace_base_dir is None:
            workspace_base_dir = str(get_uploads_dir())
        # Ensure base_dir is in workspace_config (required by ToolFactory._create_workspace)
        if "base_dir" not in workspace_config:
            workspace_config["base_dir"] = workspace_base_dir
        self._workspace_config = workspace_config
        self._explicit_vision_model = vision_model
        self._explicit_llm = llm
        self._include_mcp_tools = include_mcp_tools
        self._task_id = task_id
        self._browser_tools_enabled = browser_tools_enabled
        self._allowed_collections = allowed_collections
        self._allowed_skills = allowed_skills
        self._allowed_tools = allowed_tools
        self._excluded_agent_id: Optional[int] = None

        # Sandbox instance - only store reference, lifecycle managed by upper layer
        self._sandbox: Optional[Any] = None

        # Cache for loaded configurations
        self._cached_vision_config: Optional[Any] = None
        self._cached_image_configs: Optional[Dict[str, Any]] = None
        self._cached_image_generate_model: Optional[Any] = None
        self._cached_image_edit_model: Optional[Any] = None
        self._cached_asr_models: Optional[Dict[str, Any]] = None
        self._cached_asr_model: Optional[Any] = None
        self._cached_tts_models: Optional[Dict[str, Any]] = None
        self._cached_tts_model: Optional[Any] = None
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
                    if user is not None:
                        user_id = self._coerce_user_id(getattr(user, "id", None))
                        if user_id is not None:
                            return user_id

            # If request has a user attribute directly, use it
            if hasattr(request, "user") and request.user:
                user_id = self._coerce_user_id(getattr(request.user, "id", None))
                if user_id is not None:
                    return user_id

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

    async def get_mcp_server_configs(self) -> List[Dict[str, Any]]:
        """Load MCP server configurations from database."""
        if not self._include_mcp_tools:
            return []

        if self._cached_mcp_configs is None:
            self._cached_mcp_configs = await self._load_mcp_server_configs()
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

    def get_sandbox(self) -> Optional[Any]:
        """Get sandbox instance. Returns None if not available."""
        return self._sandbox

    def get_tool_credential(self, tool_name: str, field_name: str) -> Optional[str]:
        return resolve_tool_credential(self.db, tool_name, field_name)

    def get_sql_connections(self) -> Dict[str, str]:
        return get_sql_connection_map(self.db, self._user_id)

    def set_sandbox(self, sandbox: Any) -> None:
        """Set sandbox instance for this config."""
        self._sandbox = sandbox

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

    def get_asr_models(self) -> Dict[str, Any]:
        """Load ASR models from database."""
        if self._cached_asr_models is None:
            self._cached_asr_models = self._load_asr_models()
        return self._cached_asr_models

    def _load_asr_models(self) -> Dict[str, Any]:
        """Load ASR models from database via model service."""
        try:
            from ...web.services.model_service import get_asr_models

            return get_asr_models(self.db, self._user_id)

        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to load ASR models: {e}")
            return {}

    def get_asr_model(self) -> Optional[Any]:
        """Get default ASR model from database."""
        if self._cached_asr_model is None:
            self._cached_asr_model = self._load_asr_model()
        return self._cached_asr_model

    def _load_asr_model(self) -> Optional[Any]:
        """Load default ASR model from database via model service."""
        try:
            from ...web.services.model_service import get_default_asr_model

            return get_default_asr_model(self._user_id)

        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to load default ASR model: {e}")
            return None

    def get_tts_models(self) -> Dict[str, Any]:
        """Load TTS models from database."""
        if self._cached_tts_models is None:
            self._cached_tts_models = self._load_tts_models()
        return self._cached_tts_models

    def _load_tts_models(self) -> Dict[str, Any]:
        """Load TTS models from database via model service."""
        try:
            from ...web.services.model_service import get_tts_models

            return get_tts_models(self.db, self._user_id)

        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to load TTS models: {e}")
            return {}

    def get_tts_model(self) -> Optional[Any]:
        """Get default TTS model from database."""
        if self._cached_tts_model is None:
            self._cached_tts_model = self._load_tts_model()
        return self._cached_tts_model

    def get_llm(self) -> Optional[Any]:
        """Get LLM from constructor parameter."""
        return self._explicit_llm

    def _load_tts_model(self) -> Optional[Any]:
        """Load default TTS model from database via model service."""
        try:
            from ...web.services.model_service import get_default_tts_model

            return get_default_tts_model(self._user_id)

        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to load default TTS model: {e}")
            return None

    async def _load_mcp_server_configs(self) -> List[Dict[str, Any]]:
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

                # Handle OAuth credentials
                if server.transport == "oauth":
                    # Find corresponding OAuth account
                    # The provider might be linkedin, google, etc. based on the app config
                    from ...web.mcp_apps import get_app_by_name
                    from ...web.models.user_oauth import UserOAuth

                    app_info = get_app_by_name(self.db, str(server.name))
                    provider_name = (
                        app_info.get("provider") if app_info else server.name.lower()
                    )

                    # Some oauth records might be saved with the app_id as provider instead of the general provider_name
                    # For example, "google-drive" instead of "google"
                    app_id = app_info.get("id") if app_info else None

                    if app_id:
                        providers_to_check = [provider_name, app_id]
                        oauth_account = (
                            self.db.query(UserOAuth)
                            .filter(
                                UserOAuth.user_id == self._user_id,
                                UserOAuth.provider.in_(providers_to_check),
                            )
                            .first()
                        )
                        logger.info(
                            f"OAUTH CONFIG: Checked providers {providers_to_check} for user {self._user_id}. Found: {oauth_account is not None}"
                        )
                    else:
                        oauth_account = (
                            self.db.query(UserOAuth)
                            .filter(
                                UserOAuth.user_id == self._user_id,
                                UserOAuth.provider == provider_name,
                            )
                            .first()
                        )
                        logger.info(
                            f"OAUTH CONFIG: Checked provider '{provider_name}' for user {self._user_id}. Found: {oauth_account is not None}"
                        )

                    if oauth_account and oauth_account.access_token:
                        logger.info(
                            f"OAUTH CONFIG: Token found for '{provider_name}'. Refresh token present: {oauth_account.refresh_token is not None}, Expires: {oauth_account.expires_at}"
                        )
                        # Check and refresh token if needed before using it
                        is_valid = await refresh_oauth_token_if_needed(
                            self.db,
                            oauth_account,
                            str(provider_name) if provider_name else "",
                        )

                        if not is_valid:
                            logger.warning(
                                f"OAUTH CONFIG: Token for '{provider_name}' is invalid and could not be refreshed. "
                                "Deleting OAuth record to prompt user for reconnection."
                            )
                            # Delete the invalid oauth record so UI shows it as disconnected
                            self.db.delete(oauth_account)
                            self.db.commit()
                            continue

                        if is_valid and app_info:
                            app_id = app_info.get("id")
                            logger.info(
                                f"OAUTH CONFIG: Mapping '{app_id}' to executable proxy"
                            )

                            launch_config = app_info.get("launch_config")
                            if launch_config:
                                config["transport"] = "stdio"
                                transport_config["transport"] = "stdio"
                                transport_config["command"] = launch_config["command"]
                                transport_config["args"] = launch_config.get(
                                    "args", []
                                ).copy()

                                env = {}
                                for env_key, token_type in launch_config.get(
                                    "env_mapping", {}
                                ).items():
                                    if token_type == "access_token":
                                        env[env_key] = oauth_account.access_token

                                env.update(
                                    {
                                        "HTTPS_PROXY": os.environ.get(
                                            "HTTPS_PROXY", ""
                                        ),
                                        "HTTP_PROXY": os.environ.get("HTTP_PROXY", ""),
                                        "https_proxy": os.environ.get(
                                            "https_proxy", ""
                                        ),
                                        "http_proxy": os.environ.get("http_proxy", ""),
                                    }
                                )
                                transport_config["env"] = env  # type: ignore
                            else:
                                config["transport"] = "stdio"
                                transport_config["transport"] = "stdio"
                                transport_config["command"] = "npx"
                                transport_config["args"] = [  # type: ignore
                                    "-y",
                                    f"@mcp-servers/{str(server.name).lower().replace(' ', '-')}",
                                ]
                                transport_config["env"] = {  # type: ignore
                                    f"{str(server.name).upper().replace(' ', '_')}_ACCESS_TOKEN": oauth_account.access_token,
                                    "HTTPS_PROXY": os.environ.get("HTTPS_PROXY", ""),
                                    "HTTP_PROXY": os.environ.get("HTTP_PROXY", ""),
                                    "https_proxy": os.environ.get("https_proxy", ""),
                                    "http_proxy": os.environ.get("http_proxy", ""),
                                }

                    else:
                        logger.info(
                            f"OAUTH CONFIG: No valid token found for '{provider_name}'."
                        )

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

    def get_custom_api_configs(self) -> List[Dict[str, Any]]:
        """Get custom API configurations."""
        if not self._user_id:
            return []

        try:
            from ..models.custom_api import UserCustomApi

            user_apis = (
                self.db.query(UserCustomApi)
                .filter(
                    UserCustomApi.user_id == int(self._user_id),
                    UserCustomApi.is_active,
                )
                .all()
            )

            if not user_apis:
                return []

            custom_api_configs = []
            for user_api in user_apis:
                api = user_api.custom_api
                if api:
                    custom_api_configs.append(
                        {
                            "name": api.name,
                            "description": api.description or "",
                            "env": api.env or {},
                        }
                    )
            return custom_api_configs

        except Exception as e:
            logger.error(
                f"Failed to get Custom API configs from database: {e}", exc_info=True
            )
            return []
