"""
Web-specific tool configuration for xagent

Provides web-specific configuration classes that load from database
and other web-specific sources.
"""

import inspect
import logging
import os
import shlex
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from functools import partial
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Literal, Mapping, Optional

import httpx

from ...config import get_uploads_dir
from ...core.agent.result import ClassifiedToolFailure, normalize_tool_failure_code
from ...core.tools.adapters.vibe.config import (
    BaseToolConfig,
    normalize_tool_allowlist,
)
from ...core.tools.adapters.vibe.connector_runtime import (
    ERROR_CONNECTOR_RUNTIME_UNAVAILABLE,
    MISSING_RUNTIME_VALUE,
    RUNTIME_INPUT_AUTH_SELECTOR,
    RUNTIME_INPUT_SECRETS,
    TARGET_TRANSPORT_HEADERS,
    ConnectorRuntimeError,
    binding_source_value,
    binding_target,
    runtime_bindings_from_config,
)
from ..services.tool_credentials import (
    get_sql_connection_map,
    get_user_tool_allowlist,
    get_user_tool_overrides,
    resolve_tool_credential,
)

logger = logging.getLogger(__name__)


OAUTH_TOKEN_EXPIRY_SKEW = timedelta(minutes=5)
OAUTH_TOKEN_GENERATION_MAX_LENGTH = 1024
OAUTH_TOKEN_RESOLVER_FAILURE_CODE = "oauth_token_resolver_failed"
OAUTH_TOKEN_RESOLVER_FAILURE_MESSAGE = "OAuth token resolver failed"
UNAVAILABLE_MCP_CREDENTIAL_MESSAGE = "MCP server credentials are unavailable."


@dataclass(frozen=True)
class OAuthRefreshContext:
    reason: Literal["invalid_token"]
    resource_metadata_url: str | None
    challenge_scope: str | None
    failed_generation: str | None = field(repr=False)


@dataclass(frozen=True)
class TokenRequest:
    """Request passed to the OAuth token resolver hook.

    Registered MCP apps use provider name followed by app id, de-duplicated.
    Remote MCP servers without a matching app use the server name as a neutral
    compatibility candidate; embedders must not treat that name as an identity
    boundary. The first resolver hit wins. ``resource`` is the configured MCP
    OAuth resource URI for the current app/server when present, passed verbatim
    without canonicalization. ``scope`` is the current execution scope from
    ``WebToolConfig.get_execution_scope()`` when present; it is typed as
    Optional[Any] to avoid importing the core scope type into this config layer.
    """

    provider: str
    user_id: int
    scope: Optional[Any] = None
    resource: str | None = None
    refresh: OAuthRefreshContext | None = None


@dataclass(frozen=True)
class ResolvedToken:
    """OAuth access token supplied by the resolver hook.

    ``expires_at`` should be an aware UTC datetime when set. Naive datetimes
    are interpreted as UTC for compatibility with the existing OAuth refresh
    comparison. Resolvers SHOULD set ``expires_at`` to enable MCP config
    caching; ``expires_at=None`` means the token is usable for this build only
    and this ``WebToolConfig`` instance will reload MCP configs on later calls.
    """

    access_token: str = field(repr=False)
    expires_at: datetime | None = None
    generation: str | None = field(default=None, repr=False)


TokenResolverResult = ResolvedToken | Awaitable[ResolvedToken | None] | None
TokenResolver = Callable[[TokenRequest], TokenResolverResult]

_oauth_token_resolver_hook: TokenResolver | None = None
_oauth_token_resolver_generation = 0


def set_oauth_token_resolver_hook(resolver: TokenResolver | None) -> None:
    """Register or clear the process-wide OAuth token resolver hook.

    Resolvers may return ``ResolvedToken`` or ``None`` directly, or return an
    awaitable that resolves to either value.

    Every registration invalidates existing per-instance MCP config caches, even
    when the callable identity is unchanged. Embedders can re-register the hook
    after external token-store changes to force already-created ``WebToolConfig``
    instances to reload credentials.
    """
    global _oauth_token_resolver_generation, _oauth_token_resolver_hook

    _oauth_token_resolver_hook = resolver
    _oauth_token_resolver_generation += 1


def _get_oauth_token_resolver_hook() -> tuple[TokenResolver | None, int]:
    return _oauth_token_resolver_hook, _oauth_token_resolver_generation


def _oauth_token_resolver_registration_matches(
    resolver: TokenResolver, registration_generation: int
) -> bool:
    current_resolver, current_generation = _get_oauth_token_resolver_hook()
    return (
        current_resolver is resolver and current_generation == registration_generation
    )


async def _maybe_await_oauth_token_resolver_result(result: object) -> object:
    if inspect.isawaitable(result):
        return await result
    return result


@dataclass(frozen=True)
class _ResolvedHookToken:
    provider: str
    access_token: str = field(repr=False)
    expires_at: datetime | None
    generation: str | None = field(repr=False)


class _OAuthTokenResolverFailed(Exception):
    def __init__(
        self,
        *,
        providers: list[str],
        exception_type: str,
        resource: str | None = None,
        actor_id: str | None = None,
        failure_code: str | None = None,
    ) -> None:
        super().__init__(OAUTH_TOKEN_RESOLVER_FAILURE_CODE)
        self.providers = providers
        self.exception_type = exception_type
        self.resource = resource
        self.actor_id = actor_id
        self.failure_code = normalize_tool_failure_code(failure_code)


class _OAuthLaunchConfigInvalid(Exception):
    def __init__(self, *, field: str) -> None:
        super().__init__(field)
        self.field = field


def _bounded_oauth_metadata(value: Any, *, max_length: int = 128) -> str:
    text = str(value)
    if len(text) <= max_length:
        return text
    return f"{text[: max_length - 3]}..."


def _extract_oauth_token_resolver_diagnostic_actor_id(exc: Exception) -> str | None:
    try:
        raw_actor_id = getattr(exc, "oauth_token_resolver_diagnostic_actor_id", None)
        if type(raw_actor_id) is not str:
            return None
        return _bounded_oauth_metadata(raw_actor_id)
    except Exception:
        return None


def _extract_oauth_token_resolver_failure_code(exc: Exception) -> str | None:
    try:
        raw_failure_code = getattr(exc, "oauth_token_resolver_failure_code", None)
    except Exception:
        return None
    return normalize_tool_failure_code(raw_failure_code)


def _normalize_oauth_expires_at(expires_at: datetime | None) -> datetime | None:
    if expires_at is None:
        return None
    if expires_at.tzinfo is None:
        return expires_at.replace(tzinfo=timezone.utc)
    return expires_at.astimezone(timezone.utc)


def _oauth_token_is_expired(expires_at: datetime) -> bool:
    return expires_at <= datetime.now(timezone.utc)


def _oauth_token_expires_after_cache_window(expires_at: datetime) -> bool:
    return expires_at > datetime.now(timezone.utc) + OAUTH_TOKEN_EXPIRY_SKEW


def _oauth_token_provider_candidates(app_info: Mapping[str, Any]) -> list[str]:
    return list(
        dict.fromkeys(
            value
            for value in (app_info.get("provider"), app_info.get("id"))
            if isinstance(value, str) and value
        )
    )


def _oauth_token_configured_resource(app_info: Mapping[str, Any]) -> str | None:
    resource = app_info.get("resource")
    if isinstance(resource, str) and resource != "":
        return resource
    launch_config = app_info.get("launch_config")
    if isinstance(launch_config, Mapping):
        resource = launch_config.get("resource")
        if isinstance(resource, str) and resource != "":
            return resource
    return None


def _oauth_launch_config_args(launch_config: Mapping[str, Any]) -> list[Any]:
    args = launch_config.get("args")
    if args is None:
        return []
    if isinstance(args, list):
        return args.copy()
    if isinstance(args, str):
        try:
            return shlex.split(args)
        except ValueError as exc:
            logger.warning(
                "Falling back to whitespace split for OAuth MCP launch config args because args string could not be parsed: %s",
                type(exc).__name__,
            )
            return args.split()
    logger.warning(
        "Ignoring OAuth MCP launch config args because args must be a list or a string"
    )
    return []


def _oauth_launch_config_command(launch_config: Mapping[str, Any]) -> str:
    command = launch_config.get("command")
    if isinstance(command, str) and command:
        return command
    raise _OAuthLaunchConfigInvalid(field="command")


def _oauth_launch_config_env_mapping(
    launch_config: Mapping[str, Any],
) -> Mapping[str, Any]:
    env_mapping = launch_config.get("env_mapping")
    if env_mapping is None:
        return {}
    if isinstance(env_mapping, Mapping):
        return env_mapping
    logger.warning(
        "Ignoring OAuth MCP launch config env_mapping because env_mapping must be a mapping"
    )
    return {}


def _oauth_launch_config_mapping(
    launch_config: Any,
) -> Mapping[str, Any] | None:
    if launch_config is None:
        return None
    if isinstance(launch_config, Mapping):
        return launch_config
    raise _OAuthLaunchConfigInvalid(field="type")


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

    if expires_at > now + OAUTH_TOKEN_EXPIRY_SKEW:
        return True  # Token is still valid

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

        if provider_name == "meta":
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    provider_config.token_url,
                    params={
                        "grant_type": "fb_exchange_token",
                        "client_id": client_id,
                        "client_secret": client_secret,
                        "fb_exchange_token": oauth_account.access_token,
                    },
                    timeout=10.0,
                )

            if response.status_code == 200:
                data = response.json()
                if "access_token" in data:
                    oauth_account.access_token = data["access_token"]
                    if "expires_in" in data:
                        oauth_account.expires_at = datetime.now(
                            timezone.utc
                        ) + timedelta(seconds=int(data["expires_in"]))
                    db.commit()
                    logger.info(
                        f"Successfully refreshed {provider_name} token for user {oauth_account.user_id}"
                    )
                    return True
            else:
                logger.error(
                    f"Failed to refresh {provider_name} token: {response.text}"
                )
            return False

        if not oauth_account.refresh_token:
            logger.warning(
                f"Token expired for {provider_name} but no refresh_token available."
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
        db_factory: Optional[Any] = None,
        user_id: Optional[int] = None,
        is_admin: Optional[bool] = None,
        user: Optional[Any] = None,
        workspace_config: Optional[Dict[str, Any]] = None,
        vision_model: Optional[Any] = None,
        llm: Optional[Any] = None,
        include_mcp_tools: bool = True,
        task_id: Optional[str] = None,
        workspace_base_dir: Optional[str] = None,
        browser_tools_enabled: bool = True,
        allowed_collections: Optional[List[str]] = None,
        allowed_skills: Optional[List[str]] = None,
        allowed_agent_ids: Optional[List[int]] = None,
        agent_tool_overrides: Optional[Dict[int, Dict[str, Any]]] = None,
        a2a_agent_configs: Optional[List[Dict[str, Any]]] = None,
        enable_global_agent_tools: bool = True,
        allow_cross_user_agent_ids: bool = False,
        parent_task_id: Optional[str] = None,
        parent_tracer: Optional[Any] = None,
        agent_call_stack: Optional[List[int]] = None,
        sandbox: Optional[Any] = None,
        tool_selection_spec: Optional[Any] = None,
        mcp_auth_context: Optional[Dict[str, Any]] = None,
        execution_scope: Optional[Any] = None,
        connector_runtime_turn_id: Optional[str] = None,
    ):
        # ``tool_selection_spec`` accepts :class:`ToolSelectionSpec` from
        # the tools adapter package; typed as ``Any`` here to avoid an
        # import cycle (web.tools → core.tools.adapters). The factory
        # reads ``config.get_tool_selection_spec()``. ``None`` defaults
        # to the ``_SpecAll`` ALL-mode (build every default tool).
        self._tool_selection_spec = tool_selection_spec
        self._live_db = db
        self._db_factory = db_factory
        self._lazy_db = None
        self.request = request
        self._user_id = (
            user_id if user_id is not None else self._get_user_id_from_request(request)
        )
        # Tri-state: an explicit ``is_admin`` (including ``False``) is
        # authoritative and is NOT OR-ed with the request's admin flag. This
        # is the privilege-isolation boundary: when the runtime builds a tool
        # config for a task owner (passing ``is_admin=bool(owner.is_admin)``),
        # an admin *actor* on the request must not silently widen the config
        # to admin scope. Only when ``is_admin`` is unset do we fall back to
        # the request.
        self._is_admin_value = (
            bool(is_admin)
            if is_admin is not None
            else self._get_is_admin_from_request(request)
        )
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
        if self._user_id is not None and "user_id" not in workspace_config:
            workspace_config["user_id"] = self._user_id
        if mcp_auth_context is None:
            raw_auth_context = workspace_config.get("mcp_auth_context")
            mcp_auth_context = (
                raw_auth_context if isinstance(raw_auth_context, dict) else None
            )
        self._workspace_config = workspace_config
        # ExecutionScope (typed as Any to avoid importing core into every
        # config consumer) the tool set is built under. Nested agent tools
        # snapshot it at construction so delegated executions re-activate
        # the parent turn's scope instead of re-resolving.
        self._execution_scope = execution_scope
        self._mcp_auth_context = (
            mcp_auth_context if isinstance(mcp_auth_context, dict) else {}
        )
        if connector_runtime_turn_id is None:
            raw_turn_id = workspace_config.get("turn_id")
            connector_runtime_turn_id = (
                raw_turn_id if isinstance(raw_turn_id, str) else None
            )
        self._connector_runtime_turn_id = connector_runtime_turn_id
        self._connector_runtime_view: Optional[Dict[str, Any]] = None
        self._mcp_oauth_diagnostics: List[Dict[str, Any]] = []
        self._explicit_vision_model = vision_model
        self._explicit_llm = llm
        self._include_mcp_tools = include_mcp_tools
        self._task_id = task_id
        self._browser_tools_enabled = browser_tools_enabled
        self._allowed_collections = allowed_collections
        self._allowed_skills = allowed_skills
        self._allowed_agent_ids = allowed_agent_ids
        self._agent_tool_overrides = (
            agent_tool_overrides if isinstance(agent_tool_overrides, dict) else {}
        )
        self._a2a_agent_configs = (
            a2a_agent_configs if isinstance(a2a_agent_configs, list) else []
        )
        self._enable_global_agent_tools = bool(enable_global_agent_tools)
        self._allow_cross_user_agent_ids = bool(allow_cross_user_agent_ids)
        self._parent_task_id = parent_task_id
        self._parent_tracer = parent_tracer
        self._agent_call_stack = list(agent_call_stack or [])
        self._excluded_agent_id: Optional[int] = None

        # Cache user object for hook queries.
        # Use explicit user param first; fall back to request.user.
        self._user = user if user is not None else getattr(request, "user", None)
        self._cached_tool_overrides: Optional[dict] = None
        # ``None`` is a meaningful allowlist value ("no allowlist"), so a
        # separate flag tracks whether the hook has been consulted yet.
        self._cached_tool_allowlist: Optional[list] = None
        self._tool_allowlist_cached: bool = False

        # Sandbox instance - only store reference, lifecycle managed by upper layer
        self._sandbox: Optional[Any] = sandbox

        # Cache for loaded configurations
        self._cached_vision_config: Optional[Any] = None
        self._cached_image_configs: Optional[Dict[str, Any]] = None
        self._cached_video_configs: Optional[Dict[str, Any]] = None
        self._cached_image_generate_model: Optional[Any] = None
        self._cached_image_edit_model: Optional[Any] = None
        self._cached_video_model: Optional[Any] = None
        self._cached_asr_models: Optional[Dict[str, Any]] = None
        self._cached_asr_model: Optional[Any] = None
        self._cached_tts_models: Optional[Dict[str, Any]] = None
        self._cached_tts_model: Optional[Any] = None
        self._cached_sound_effect_models: Optional[Dict[str, Any]] = None
        self._cached_sound_effect_model: Optional[Any] = None
        self._cached_music_models: Optional[Dict[str, Any]] = None
        self._cached_music_model: Optional[Any] = None
        self._cached_mcp_configs: Optional[List[Dict[str, Any]]] = None
        self._mcp_hook_token_cache_expires_at: datetime | None = None
        self._mcp_hook_token_cache_uncacheable = False
        self._mcp_hook_generation_at_load: int | None = None
        self._mcp_hook_resolution_failed = False
        self._cached_embedding_model: Optional[str] = None
        self._cached_rerank_model: Optional[str] = None

    def _build_mcp_file_allowed_dirs(self) -> str:
        """Build comma-separated file roots that local MCP tools may read."""
        dirs: list[str] = []
        base_dir = Path(str(self._workspace_config.get("base_dir", get_uploads_dir())))
        task_id = self._workspace_config.get("task_id")
        if task_id:
            dirs.append(str((base_dir / str(task_id)).expanduser().resolve()))

        for raw_dir in self._workspace_config.get("allowed_external_dirs") or []:
            dirs.append(str(Path(str(raw_dir)).expanduser().resolve()))

        seen: set[str] = set()
        unique_dirs = []
        for dir_path in dirs:
            if dir_path not in seen:
                unique_dirs.append(dir_path)
                seen.add(dir_path)
        return ",".join(unique_dirs)

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
        """Extract is_admin flag from the request user, defaulting to False.

        Uses ``getattr`` so a minimal request object (e.g. one carrying only a
        user id) doesn't trip the broad ``except`` and log a spurious warning.
        """
        user = getattr(request, "user", None)
        return bool(getattr(user, "is_admin", False)) if user is not None else False

    def get_workspace_config(self) -> Optional[Dict[str, Any]]:
        """Get workspace configuration."""
        return self._workspace_config

    def get_execution_scope(self) -> Optional[Any]:
        """ExecutionScope the tool set was built under (None = unscoped)."""
        return self._execution_scope

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

    def get_video_models(self) -> Dict[str, Any]:
        """Load video models from database."""
        if self._cached_video_configs is None:
            self._cached_video_configs = self._load_video_models()
        return self._cached_video_configs

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

    def get_video_model(self) -> Optional[Any]:
        """Get default video generation model from database."""
        if self._cached_video_model is None:
            self._cached_video_model = self._load_video_model()
        return self._cached_video_model

    async def get_mcp_server_configs(self) -> List[Dict[str, Any]]:
        """Load MCP server configurations from database."""
        if not self._include_mcp_tools:
            return []

        if self._cached_mcp_configs is not None and self._mcp_config_cache_is_valid():
            return self._cached_mcp_configs

        configs = await self._load_mcp_server_configs()
        self._store_mcp_config_cache_if_cacheable(configs)
        return configs

    def _mcp_config_cache_is_valid(self) -> bool:
        # MCP config caching is aware of hook-supplied token expiry only. The
        # legacy UserOAuth path keeps the pre-existing per-instance cache shape.
        _, current_generation = _get_oauth_token_resolver_hook()
        if self._mcp_hook_generation_at_load != current_generation:
            return False
        if self._mcp_hook_resolution_failed:
            return False
        if self._mcp_hook_token_cache_uncacheable:
            return False
        if self._mcp_hook_token_cache_expires_at is not None:
            return _oauth_token_expires_after_cache_window(
                self._mcp_hook_token_cache_expires_at
            )
        return True

    def _reset_mcp_config_load_cache_state(self) -> None:
        _, current_generation = _get_oauth_token_resolver_hook()
        self._mcp_hook_token_cache_expires_at = None
        self._mcp_hook_token_cache_uncacheable = False
        self._mcp_hook_generation_at_load = current_generation
        self._mcp_hook_resolution_failed = False

    def _store_mcp_config_cache_if_cacheable(
        self, configs: List[Dict[str, Any]]
    ) -> None:
        if self._mcp_hook_resolution_failed or self._mcp_hook_token_cache_uncacheable:
            self._cached_mcp_configs = None
            return
        self._cached_mcp_configs = configs

    def get_mcp_oauth_diagnostics(self) -> List[Dict[str, Any]]:
        """Return structured MCP OAuth runtime diagnostics from the last load."""
        return list(self._mcp_oauth_diagnostics)

    def _get_connector_runtime_for(
        self, connector_type: str, connector_id: int
    ) -> Optional[Dict[str, Any]]:
        view = self._load_connector_runtime_view()
        value = view.get(f"{connector_type}:{connector_id}")
        return dict(value) if isinstance(value, dict) else None

    def _load_connector_runtime_view(self) -> Dict[str, Any]:
        if self._connector_runtime_view is not None:
            return self._connector_runtime_view
        self._connector_runtime_view = {}
        task_id = self._parse_numeric_task_id()
        if task_id is None or self._user_id is None:
            return self._connector_runtime_view
        try:
            from ..services.connector_runtime import load_connector_runtime_view

            self._connector_runtime_view = load_connector_runtime_view(
                db=self.db,
                task_id=task_id,
                turn_id=self._connector_runtime_turn_id,
                user_id=int(self._user_id),
            )
        except ConnectorRuntimeError:
            raise
        except Exception as exc:
            logging.getLogger(__name__).warning(
                "Failed to resolve connector runtime view for task %s",
                self._task_id,
                exc_info=True,
            )
            self._connector_runtime_view = None
            raise ConnectorRuntimeError(
                ERROR_CONNECTOR_RUNTIME_UNAVAILABLE,
                "Connector runtime context is unavailable.",
                details={"reason": "runtime_view_resolution_failed"},
                status_code=503,
            ) from exc
        return self._connector_runtime_view

    def set_connector_runtime_turn_id(self, turn_id: Optional[str]) -> bool:
        """Switch the per-turn connector runtime source for reused agents.

        ``WebToolConfig`` instances are cached with ``AgentService`` by task.
        Runtime secrets/auth selectors are intentionally per-turn, so an append
        turn must not keep using the first turn's resolved connector runtime
        view or MCP config cache.
        """

        normalized_turn_id = turn_id if isinstance(turn_id, str) else None
        if self._connector_runtime_turn_id == normalized_turn_id:
            return False
        self._connector_runtime_turn_id = normalized_turn_id
        self._connector_runtime_view = None
        self._cached_mcp_configs = None
        return True

    def _parse_numeric_task_id(self) -> Optional[int]:
        task_id = self._task_id
        if not isinstance(task_id, str) or not task_id:
            return None
        if task_id.startswith("web_task_"):
            task_id = task_id.removeprefix("web_task_")
        try:
            return int(task_id)
        except (TypeError, ValueError):
            return None

    def _runtime_transport_headers(
        self,
        *,
        runtime_values: Optional[Dict[str, Any]],
        runtime_bindings: Any,
        allow_delegated_authorization: bool,
        warn_on_rejected_authorization: bool = True,
    ) -> Dict[str, str]:
        if not isinstance(runtime_values, dict):
            return {}
        headers: Dict[str, str] = {}
        for binding in runtime_bindings_from_config(
            {"runtime_bindings": runtime_bindings}
        ):
            target = binding_target(binding)
            if target.get("target_type") != TARGET_TRANSPORT_HEADERS:
                continue
            header_name = target.get("key")
            if not isinstance(header_name, str) or not header_name:
                continue
            if (
                header_name.lower() == "authorization"
                and not allow_delegated_authorization
            ):
                if warn_on_rejected_authorization:
                    logger.warning(
                        "Ignoring runtime MCP Authorization header binding because "
                        "delegated authorization is disabled"
                    )
                continue
            value = binding_source_value(
                binding,
                runtime_values,
                allowed_input_types={RUNTIME_INPUT_SECRETS},
            )
            if value is MISSING_RUNTIME_VALUE or isinstance(value, (dict, list)):
                continue
            headers[header_name] = str(value)
        return headers

    def _delegated_mcp_connection(
        self,
        *,
        server: Any,
        runtime_values: Optional[Dict[str, Any]],
        runtime_bindings: Any,
        allow_delegated_authorization: bool,
    ) -> dict[str, Any] | None:
        delegated_headers = self._runtime_transport_headers(
            runtime_values=runtime_values,
            runtime_bindings=runtime_bindings,
            allow_delegated_authorization=allow_delegated_authorization,
        )
        if not delegated_headers:
            return None
        return self._mcp_connection_with_runtime_headers(
            server=server, runtime_headers=delegated_headers
        )

    @staticmethod
    def _mcp_connection_with_runtime_headers(
        *, server: Any, runtime_headers: Mapping[str, str]
    ) -> dict[str, Any]:
        from ...web.services.mcp_runtime import connection_without_authorization

        connection = connection_without_authorization(server.to_connection_dict())
        connection["headers"].update(runtime_headers)
        connection.pop("auth", None)
        return connection

    def _non_auth_mcp_connection(
        self,
        *,
        server: Any,
        runtime_values: Optional[Dict[str, Any]],
        runtime_bindings: Any,
    ) -> dict[str, Any]:
        runtime_headers = self._runtime_transport_headers(
            runtime_values=runtime_values,
            runtime_bindings=runtime_bindings,
            allow_delegated_authorization=False,
            warn_on_rejected_authorization=False,
        )
        return self._mcp_connection_with_runtime_headers(
            server=server, runtime_headers=runtime_headers
        )

    def _refresh_delegated_mcp_connection(
        self,
        *,
        server: Any,
        runtime_bindings: Any,
        allow_delegated_authorization: bool,
    ) -> dict[str, Any] | None:
        self._connector_runtime_view = None
        runtime_values = self._get_connector_runtime_for("mcp", int(server.id))
        return self._delegated_mcp_connection(
            server=server,
            runtime_values=runtime_values,
            runtime_bindings=runtime_bindings,
            allow_delegated_authorization=allow_delegated_authorization,
        )

    def _mcp_auth_context_for_server(
        self,
        *,
        server_id: int,
        runtime_values: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        context = dict(self._mcp_auth_context)
        auth_selector = (
            runtime_values.get(RUNTIME_INPUT_AUTH_SELECTOR)
            if isinstance(runtime_values, dict)
            else None
        )
        if isinstance(auth_selector, dict) and auth_selector:
            context[str(server_id)] = dict(auth_selector)
        return context

    def get_embedding_model(self) -> Optional[str]:
        """Load default embedding model ID from database."""
        if self._cached_embedding_model is None:
            self._cached_embedding_model = self._load_embedding_model()
        return self._cached_embedding_model

    def get_rerank_model(self) -> Optional[str]:
        """Load default rerank model ID from database."""
        if self._cached_rerank_model is None:
            self._cached_rerank_model = self._load_rerank_model()
        return self._cached_rerank_model

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

    def get_skill_scope_context(self) -> Any:
        """Build generic context for scoped skill providers."""
        from ...skills.library import SkillScopeContext

        return SkillScopeContext(
            user=self._user,
            user_id=self._user_id,
            db=self.db,
            request=self.request,
        )

    def get_tool_selection_spec(self) -> Optional[Any]:
        """Typed spec accessor (preferred over :meth:`get_allowed_tools`).

        Returns a :class:`ToolSelectionSpec` instance when the caller
        supplied one via ``tool_selection_spec=ToolSelectionSpec.from_raw(...)``.
        ``ToolFactory.create_all_tools`` reads this first; falls back to
        ``get_allowed_tools()`` only if this returns ``None`` (legacy
        backward-compat).
        """
        return self._tool_selection_spec

    def get_allowed_agent_ids(self) -> Optional[List[int]]:
        """Get explicitly allowed published agent IDs. None means use defaults."""
        return self._allowed_agent_ids

    def get_agent_tool_overrides(self) -> Dict[int, Dict[str, Any]]:
        """Get per-agent tool metadata/runtime overrides for delegation."""
        return self._agent_tool_overrides

    def get_a2a_agent_configs(self) -> List[Dict[str, Any]]:
        """Get remote A2A agent tool configurations."""
        return self._a2a_agent_configs

    def get_enable_global_agent_tools(self) -> bool:
        """Whether to include globally visible published agents as tools."""
        return self._enable_global_agent_tools

    def get_allow_cross_user_agent_ids(self) -> bool:
        """Whether explicit allowed agent IDs may cross the current user boundary."""
        return self._allow_cross_user_agent_ids

    def get_parent_task_id(self) -> Optional[str]:
        """Get parent task ID for delegated tool execution."""
        return self._parent_task_id

    def get_parent_tracer(self) -> Optional[Any]:
        """Get parent tracer for delegated tool execution."""
        return self._parent_tracer

    def get_agent_call_stack(self) -> List[int]:
        """Get active agent delegation call stack for recursion prevention."""
        return self._agent_call_stack

    def get_user_tool_overrides(self) -> dict:
        """Return per-user tool overrides from the registered hook.

        Both display layer and execution layer use this as the single
        source of truth for per-user tool policies.
        """
        if self._cached_tool_overrides is not None:
            return self._cached_tool_overrides
        if self._user is None:
            self._cached_tool_overrides = {}
            return {}
        try:
            self._cached_tool_overrides = get_user_tool_overrides(self.db, self._user)
        except Exception:
            logger.exception("Failed to get user tool overrides")
            self._cached_tool_overrides = {}
        return self._cached_tool_overrides

    def refresh_user_tool_overrides(self) -> dict:
        """Reload per-user tool overrides from the registered hook."""
        # The policy can change while an AgentService instance is reused.
        self._cached_tool_overrides = None
        return self.get_user_tool_overrides()

    def get_user_tool_allowlist(self) -> Optional[list]:
        """Return the positive tool allowlist from the registered hook.

        ``None`` means "no allowlist configured" — no filtering. A concrete
        list means keep only those tool names (execution layer only). The
        allowlist is resolved from the active execution scope by the hook, so
        it can differ per turn even for the same user.
        """
        if self._tool_allowlist_cached:
            return self._cached_tool_allowlist
        try:
            self._cached_tool_allowlist = normalize_tool_allowlist(
                get_user_tool_allowlist(self.db, self._user)
            )
        except Exception:
            logger.exception("Failed to get user tool allowlist")
            self._cached_tool_allowlist = None
        self._tool_allowlist_cached = True
        return self._cached_tool_allowlist

    def refresh_user_tool_allowlist(self) -> Optional[list]:
        """Reload the positive tool allowlist from the registered hook."""
        # The active execution scope (hence the CA allowlist) can change while
        # an AgentService instance is reused across turns.
        self._tool_allowlist_cached = False
        self._cached_tool_allowlist = None
        return self.get_user_tool_allowlist()

    def get_excluded_agent_id(self) -> Optional[int]:
        """Get agent ID to exclude from agent tools (to prevent self-calls)."""
        return getattr(self, "_excluded_agent_id", None)

    def get_user_id(self) -> Optional[int]:
        """Get current user ID for multi-tenancy."""
        return self._user_id

    def get_session_factory(self) -> Any:
        """Return the sessionmaker used to mint per-call tool sessions."""
        if self._db_factory is not None:
            return self._db_factory
        from ..models.database import get_session_local

        return get_session_local()

    @property
    def db(self) -> Any:
        """Construction-time DB session.

        Request path: the caller-owned live session, returned verbatim.
        Factory path (nested child config): a lazily-opened, cached session
        minted from the factory and closed by ``close()``.

        Exposing this as a property keeps every DB-backed config loader that
        reads ``self.db.query(...)`` working whether the config was built with
        a live session or with only a factory — without each loader having to
        route through ``get_db()`` explicitly.
        """
        if self._live_db is not None:
            return self._live_db
        if self._db_factory is not None:
            if self._lazy_db is None:
                self._lazy_db = self._db_factory()
            return self._lazy_db
        return None

    def get_db(self) -> Any:
        """Get database session (see the :attr:`db` property)."""
        return self.db

    def close(self) -> None:
        """Close the lazily-opened factory session, if any."""
        if self._lazy_db is not None:
            self._lazy_db.close()
            self._lazy_db = None

    def release_db_connection(self) -> None:
        """Return the pooled connection held by this config's session(s).

        See :meth:`BaseToolConfig.release_db_connection`. Rolls back only
        clean (read-only) transactions via
        ``release_db_connection_if_clean``; sessions with pending writes are
        left untouched. Both the caller-owned live session and the lazily
        minted factory session are released — either one may have run the
        MCP/agent config SELECTs whose transaction would otherwise stay open
        across the MCP network await (issue #889).
        """
        from ..models.database import release_db_connection_if_clean

        release_db_connection_if_clean(self._live_db)
        release_db_connection_if_clean(self._lazy_db)

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

    def _load_rerank_model(self) -> Optional[str]:
        """Load rerank model ID from database via model service."""
        from ...web.services.model_service import get_default_rerank_model

        return get_default_rerank_model(self._user_id)

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

    def _load_video_models(self) -> Dict[str, Any]:
        """Load video models from database via model service."""
        try:
            from ...web.services.model_service import get_video_models

            return get_video_models(self.db, self._user_id)

        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to load video models: {e}")

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

    def _load_video_model(self) -> Optional[Any]:
        """Load default video generation model from database via model service."""
        try:
            from ...web.services.model_service import get_default_video_model

            return get_default_video_model(self._user_id)

        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to load default video model: {e}")
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

    def get_sound_effect_models(self) -> Dict[str, Any]:
        """Load sound effect models from the independent model category."""
        if self._cached_sound_effect_models is None:
            self._cached_sound_effect_models = self._load_sound_effect_models()
        return self._cached_sound_effect_models

    def _load_sound_effect_models(self) -> Dict[str, Any]:
        try:
            from ...web.services.model_service import get_sound_effect_models

            return get_sound_effect_models(self.db, self._user_id)
        except Exception as exc:
            logger = logging.getLogger(__name__)
            logger.warning("Failed to load sound effect models: %s", exc)
            return {}

    def get_sound_effect_model(self) -> Optional[Any]:
        """Get the user's default sound effect model."""
        if self._cached_sound_effect_model is None:
            self._cached_sound_effect_model = self._load_sound_effect_model()
        return self._cached_sound_effect_model

    def _load_sound_effect_model(self) -> Optional[Any]:
        try:
            from ...web.services.model_service import get_default_sound_effect_model

            return get_default_sound_effect_model(self._user_id)
        except Exception as exc:
            logger = logging.getLogger(__name__)
            logger.warning("Failed to load default sound effect model: %s", exc)
            return None

    def get_music_models(self) -> Dict[str, Any]:
        """Load music models from the independent model category."""
        if self._cached_music_models is None:
            self._cached_music_models = self._load_music_models()
        return self._cached_music_models

    def _load_music_models(self) -> Dict[str, Any]:
        try:
            from ...web.services.model_service import get_music_models

            return get_music_models(self.db, self._user_id)
        except Exception as exc:
            logger = logging.getLogger(__name__)
            logger.warning("Failed to load music models: %s", exc)
            return {}

    def get_music_model(self) -> Optional[Any]:
        """Get the user's default music model."""
        if self._cached_music_model is None:
            self._cached_music_model = self._load_music_model()
        return self._cached_music_model

    def _load_music_model(self) -> Optional[Any]:
        try:
            from ...web.services.model_service import get_default_music_model

            return get_default_music_model(self._user_id)
        except Exception as exc:
            logger = logging.getLogger(__name__)
            logger.warning("Failed to load default music model: %s", exc)
            return None

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

    async def _resolve_oauth_token_from_hook(
        self,
        *,
        providers: list[str],
        resource: str | None,
        resolver: TokenResolver | None = None,
    ) -> _ResolvedHookToken | None:
        if resolver is None:
            resolver, _ = _get_oauth_token_resolver_hook()
        if resolver is None or not providers or self._user_id is None:
            return None

        for provider in providers:
            request = TokenRequest(
                provider=provider,
                user_id=int(self._user_id),
                scope=self.get_execution_scope(),
                resource=resource,
            )
            try:
                resolved = await _maybe_await_oauth_token_resolver_result(
                    resolver(request)
                )
            except ConnectorRuntimeError:
                raise
            except Exception as exc:
                raise _OAuthTokenResolverFailed(
                    providers=providers,
                    exception_type=_bounded_oauth_metadata(type(exc).__name__),
                    resource=resource,
                    actor_id=_extract_oauth_token_resolver_diagnostic_actor_id(exc),
                    failure_code=_extract_oauth_token_resolver_failure_code(exc),
                ) from exc

            if resolved is None:
                continue
            return self._normalize_resolved_oauth_token_from_hook(
                provider=provider,
                providers=providers,
                resource=resource,
                resolved=resolved,
            )

        return None

    async def _refresh_resolver_owned_mcp_connection(
        self,
        *,
        challenge: object,
        resolver: TokenResolver,
        registration_generation: int,
        provider: str,
        providers: list[str],
        user_id: int,
        scope: Any,
        resource: str | None,
        failed_generation: str | None,
        non_auth_connection: dict[str, Any],
    ) -> dict[str, Any] | ClassifiedToolFailure | None:
        from ...web.services.mcp_oauth import MCPAuthorizationChallenge

        if (
            not isinstance(challenge, MCPAuthorizationChallenge)
            or failed_generation is None
        ):
            return None
        if not _oauth_token_resolver_registration_matches(
            resolver, registration_generation
        ):
            return None

        request = TokenRequest(
            provider=provider,
            user_id=user_id,
            scope=scope,
            resource=resource,
            refresh=OAuthRefreshContext(
                reason="invalid_token",
                resource_metadata_url=challenge.resource_metadata_url,
                challenge_scope=challenge.scope,
                failed_generation=failed_generation,
            ),
        )
        try:
            resolved = await _maybe_await_oauth_token_resolver_result(resolver(request))
        except Exception as exc:
            if not _oauth_token_resolver_registration_matches(
                resolver, registration_generation
            ):
                return None
            failure_code = _extract_oauth_token_resolver_failure_code(exc)
            if failure_code is not None:
                return ClassifiedToolFailure(failure_code=failure_code)
            return None

        if (
            not _oauth_token_resolver_registration_matches(
                resolver, registration_generation
            )
            or resolved is None
        ):
            return None
        try:
            normalized = self._normalize_resolved_oauth_token_from_hook(
                provider=provider,
                providers=providers,
                resource=resource,
                resolved=resolved,
            )
        except _OAuthTokenResolverFailed:
            return None
        if normalized.generation is None or normalized.generation == failed_generation:
            return None

        return self._build_resolver_owned_mcp_connection(
            resolver=resolver,
            registration_generation=registration_generation,
            resolved=normalized,
            providers=providers,
            user_id=user_id,
            scope=scope,
            resource=resource,
            non_auth_connection=non_auth_connection,
        )

    def _build_resolver_owned_mcp_connection(
        self,
        *,
        resolver: TokenResolver,
        registration_generation: int,
        resolved: _ResolvedHookToken,
        providers: list[str],
        user_id: int,
        scope: Any,
        resource: str | None,
        non_auth_connection: dict[str, Any],
    ) -> dict[str, Any]:
        from ...web.services.mcp_runtime import connection_with_bearer_authorization

        async def refresh(
            challenge: object,
        ) -> dict[str, Any] | ClassifiedToolFailure | None:
            return await self._refresh_resolver_owned_mcp_connection(
                challenge=challenge,
                resolver=resolver,
                registration_generation=registration_generation,
                provider=resolved.provider,
                providers=providers,
                user_id=user_id,
                scope=scope,
                resource=resource,
                failed_generation=resolved.generation,
                non_auth_connection=non_auth_connection,
            )

        connection = connection_with_bearer_authorization(
            non_auth_connection, resolved.access_token
        )
        connection["_oauth_token_resolver_refresh"] = refresh
        connection.pop("_connector_runtime_refresh", None)
        return connection

    def _normalize_resolved_oauth_token_from_hook(
        self,
        *,
        provider: str,
        providers: list[str],
        resource: str | None,
        resolved: object,
    ) -> _ResolvedHookToken:
        if not isinstance(resolved, ResolvedToken):
            raise _OAuthTokenResolverFailed(
                providers=providers,
                exception_type=_bounded_oauth_metadata(type(resolved).__name__),
                resource=resource,
            )
        if not isinstance(resolved.access_token, str) or not resolved.access_token:
            raise _OAuthTokenResolverFailed(
                providers=providers,
                exception_type="InvalidAccessToken",
                resource=resource,
            )
        if resolved.expires_at is not None and not isinstance(
            resolved.expires_at, datetime
        ):
            raise _OAuthTokenResolverFailed(
                providers=providers,
                exception_type="InvalidExpiresAt",
                resource=resource,
            )
        if resolved.generation is not None and (
            type(resolved.generation) is not str
            or not resolved.generation
            or len(resolved.generation) > OAUTH_TOKEN_GENERATION_MAX_LENGTH
        ):
            raise _OAuthTokenResolverFailed(
                providers=providers,
                exception_type="InvalidGeneration",
                resource=resource,
            )

        expires_at = _normalize_oauth_expires_at(resolved.expires_at)
        if expires_at is not None and _oauth_token_is_expired(expires_at):
            raise _OAuthTokenResolverFailed(
                providers=providers,
                exception_type="ExpiredAccessToken",
                resource=resource,
            )

        return _ResolvedHookToken(
            provider=provider,
            access_token=resolved.access_token,
            expires_at=expires_at,
            generation=resolved.generation,
        )

    def _mark_hook_token_cache_metadata(self, resolved: _ResolvedHookToken) -> None:
        if resolved.expires_at is None:
            self._mcp_hook_token_cache_uncacheable = True
            return
        if not _oauth_token_expires_after_cache_window(resolved.expires_at):
            self._mcp_hook_token_cache_uncacheable = True
            return
        if self._mcp_hook_token_cache_expires_at is None:
            self._mcp_hook_token_cache_expires_at = resolved.expires_at
            return
        self._mcp_hook_token_cache_expires_at = min(
            self._mcp_hook_token_cache_expires_at,
            resolved.expires_at,
        )

    def _build_oauth_token_resolver_diagnostic(
        self,
        *,
        server: Any,
        error: _OAuthTokenResolverFailed,
    ) -> Dict[str, Any]:
        from ...web.services.mcp_runtime import mcp_oauth_runtime_diagnostic

        diagnostic = mcp_oauth_runtime_diagnostic(
            server,
            code=OAUTH_TOKEN_RESOLVER_FAILURE_CODE,
            message=OAUTH_TOKEN_RESOLVER_FAILURE_MESSAGE,
            resource=_bounded_oauth_metadata(error.resource)
            if error.resource is not None
            else None,
        )
        diagnostic["providers"] = [
            _bounded_oauth_metadata(provider) for provider in error.providers[:2]
        ]
        diagnostic["exception_type"] = _bounded_oauth_metadata(error.exception_type)
        if error.actor_id:
            diagnostic["actor_id"] = error.actor_id
        return diagnostic

    def _resolver_failure_config(
        self,
        *,
        server: Any,
        error: _OAuthTokenResolverFailed,
    ) -> Dict[str, Any]:
        self._mcp_hook_resolution_failed = True
        diagnostic = self._build_oauth_token_resolver_diagnostic(
            server=server,
            error=error,
        )
        self._mcp_oauth_diagnostics.append(diagnostic)
        logger.warning(
            "OAuth token resolver failed for MCP server '%s' with %s",
            getattr(server, "name", "<unknown>"),
            error.exception_type,
        )
        return self._build_unavailable_oauth_mcp_config(
            server=server,
            diagnostic=diagnostic,
            failure_code=error.failure_code,
        )

    def _build_unavailable_oauth_mcp_config(
        self,
        *,
        server: Any,
        diagnostic: Dict[str, Any],
        failure_code: str | None,
    ) -> Dict[str, Any]:
        inner_config: Dict[str, Any] = {
            "unavailable": True,
            "reason": OAUTH_TOKEN_RESOLVER_FAILURE_CODE,
            "message": UNAVAILABLE_MCP_CREDENTIAL_MESSAGE,
            "server_id": getattr(server, "id", None),
            "diagnostic": diagnostic,
        }
        if failure_code is not None:
            inner_config["failure_code"] = failure_code
        return {
            "name": server.name,
            "transport": "unavailable",
            "description": server.description,
            "config": inner_config,
            "user_id": str(self._user_id),
            "allow_users": [str(self._user_id)],
        }

    def _build_oauth_mcp_stdio_transport_config(
        self,
        *,
        server: Any,
        app_info: Mapping[str, Any],
        access_token: str,
    ) -> Dict[str, Any]:
        launch_config = _oauth_launch_config_mapping(app_info.get("launch_config"))
        if launch_config:
            transport_config: Dict[str, Any] = {
                "transport": "stdio",
                "command": _oauth_launch_config_command(launch_config),
                "args": _oauth_launch_config_args(launch_config),
            }

            env = {}
            for env_key, token_type in _oauth_launch_config_env_mapping(
                launch_config
            ).items():
                if token_type == "access_token":
                    env[env_key] = access_token

            env.update(
                {
                    "HTTPS_PROXY": os.environ.get("HTTPS_PROXY", ""),
                    "HTTP_PROXY": os.environ.get("HTTP_PROXY", ""),
                    "https_proxy": os.environ.get("https_proxy", ""),
                    "http_proxy": os.environ.get("http_proxy", ""),
                }
            )
            allowed_file_dirs = self._build_mcp_file_allowed_dirs()
            if allowed_file_dirs:
                env["XAGENT_LINKEDIN_IMAGE_ALLOWED_DIRS"] = allowed_file_dirs
            transport_config["env"] = env
            return transport_config

        return {
            "transport": "stdio",
            "command": "npx",
            "args": [
                "-y",
                f"@mcp-servers/{str(server.name).lower().replace(' ', '-')}",
            ],
            "env": {
                f"{str(server.name).upper().replace(' ', '_')}_ACCESS_TOKEN": access_token,
                "HTTPS_PROXY": os.environ.get("HTTPS_PROXY", ""),
                "HTTP_PROXY": os.environ.get("HTTP_PROXY", ""),
                "https_proxy": os.environ.get("https_proxy", ""),
                "http_proxy": os.environ.get("http_proxy", ""),
            },
        }

    async def _load_mcp_server_configs(self) -> List[Dict[str, Any]]:
        """Load MCP server configurations from database with user context."""
        logger = logging.getLogger(__name__)
        configs = []
        self._mcp_oauth_diagnostics = []
        self._reset_mcp_config_load_cache_state()

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

            # Per-user env overrides (decrypted), merged over each server's global
            # env at runtime. Prefetched once to avoid an N+1 per-server lookup.
            from ..services.mcp_runtime import (
                load_shared_env_overrides,
                load_user_env_overrides,
                load_user_env_sources,
            )

            user_env_by_id = load_user_env_overrides(self.db, self._user_id)
            shared_env_by_id = load_shared_env_overrides(self.db, self._user_id)
            env_source_by_id = load_user_env_sources(self.db, self._user_id)

            for server in servers:
                # Build config dict from server model
                runtime_bindings = getattr(server, "runtime_bindings", None)
                allow_delegated_authorization = bool(
                    getattr(server, "allow_delegated_authorization", False)
                )
                runtime_values = self._get_connector_runtime_for("mcp", int(server.id))
                config: Dict[str, Any] = {
                    "id": int(server.id),
                    "name": server.name,
                    "transport": server.transport,
                    "description": server.description,
                    "runtime_input_schema": getattr(
                        server, "runtime_input_schema", None
                    ),
                    "runtime_bindings": runtime_bindings,
                    "allow_delegated_authorization": allow_delegated_authorization,
                }
                if runtime_values:
                    context_values = runtime_values.get("context")
                    config["connector_runtime"] = {
                        "context": context_values
                        if isinstance(context_values, dict)
                        else {},
                        "secrets": {},
                        "auth_selector": {},
                    }

                # Add transport-specific configuration
                transport_config: Dict[str, Any] = {}

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

                    hook_token: _ResolvedHookToken | None = None
                    if app_info:
                        configured_resource = _oauth_token_configured_resource(app_info)
                        providers_to_resolve = _oauth_token_provider_candidates(
                            app_info
                        )
                        try:
                            hook_token = await self._resolve_oauth_token_from_hook(
                                providers=providers_to_resolve,
                                resource=configured_resource,
                            )
                        except _OAuthTokenResolverFailed as error:
                            configs.append(
                                self._resolver_failure_config(
                                    server=server,
                                    error=error,
                                )
                            )
                            continue

                    if app_info and hook_token is not None:
                        self._mark_hook_token_cache_metadata(hook_token)
                        try:
                            transport_config = (
                                self._build_oauth_mcp_stdio_transport_config(
                                    server=server,
                                    app_info=app_info,
                                    access_token=hook_token.access_token,
                                )
                            )
                        except _OAuthLaunchConfigInvalid as error:
                            logger.warning(
                                "Skipping OAuth MCP server '%s' because launch_config.%s is invalid",
                                getattr(server, "name", "<unknown>"),
                                error.field,
                            )
                            continue
                        config["transport"] = "stdio"
                        logger.info(
                            "OAuth token resolver supplied token for MCP server '%s' via provider '%s'",
                            getattr(server, "name", "<unknown>"),
                            hook_token.provider,
                        )
                    else:
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
                                try:
                                    transport_config = (
                                        self._build_oauth_mcp_stdio_transport_config(
                                            server=server,
                                            app_info=app_info,
                                            access_token=oauth_account.access_token,
                                        )
                                    )
                                except _OAuthLaunchConfigInvalid as error:
                                    logger.warning(
                                        "Skipping OAuth MCP server '%s' because launch_config.%s is invalid",
                                        getattr(server, "name", "<unknown>"),
                                        error.field,
                                    )
                                    continue
                                config["transport"] = "stdio"

                        else:
                            logger.info(
                                f"OAUTH CONFIG: No valid token found for '{provider_name}'."
                            )

                if server.transport == "stdio":
                    if server.command:
                        transport_config["command"] = server.command
                    if server.args:
                        transport_config["args"] = server.args
                    # Decrypt global env and merge per-user override (user wins).
                    from ...core.utils.encryption import decrypt_env_dict
                    from ..services.mcp_runtime import resolve_stdio_env

                    merged_env = resolve_stdio_env(
                        env_source_by_id.get(server.id),
                        decrypt_env_dict(getattr(server, "env", None)),
                        shared_env_by_id.get(server.id),
                        user_env_by_id.get(server.id),
                    )
                    if merged_env:
                        transport_config["env"] = merged_env
                    if server.cwd:
                        transport_config["cwd"] = server.cwd

                elif server.transport in ["sse", "websocket", "streamable_http"]:
                    from ...web.mcp_apps import get_app_by_name
                    from ...web.services.mcp_runtime import (
                        build_mcp_runtime_connection,
                        connection_to_transport_config,
                        effective_mcp_oauth_resource,
                    )

                    auth_context = self._mcp_auth_context_for_server(
                        server_id=int(server.id),
                        runtime_values=runtime_values,
                    )
                    resolver, registration_generation = _get_oauth_token_resolver_hook()
                    remote_providers_to_resolve: list[str] = []
                    remote_configured_resource: str | None = None
                    remote_hook_token: _ResolvedHookToken | None = None
                    if resolver is not None:
                        app_info = get_app_by_name(self.db, str(server.name))
                        remote_providers_to_resolve = (
                            _oauth_token_provider_candidates(app_info)
                            if app_info
                            else [str(server.name)]
                        )
                        remote_configured_resource = effective_mcp_oauth_resource(
                            server,
                            mcp_auth_context=auth_context,
                        )
                        if remote_providers_to_resolve:
                            try:
                                remote_hook_token = (
                                    await self._resolve_oauth_token_from_hook(
                                        providers=remote_providers_to_resolve,
                                        resource=remote_configured_resource,
                                        resolver=resolver,
                                    )
                                )
                            except _OAuthTokenResolverFailed as error:
                                configs.append(
                                    self._resolver_failure_config(
                                        server=server,
                                        error=error,
                                    )
                                )
                                continue

                    if remote_hook_token is not None and resolver is not None:
                        self._mark_hook_token_cache_metadata(remote_hook_token)
                        resolver_connection = self._build_resolver_owned_mcp_connection(
                            resolver=resolver,
                            registration_generation=registration_generation,
                            resolved=remote_hook_token,
                            providers=remote_providers_to_resolve,
                            user_id=int(self._user_id),
                            scope=self.get_execution_scope(),
                            resource=remote_configured_resource,
                            non_auth_connection=self._non_auth_mcp_connection(
                                server=server,
                                runtime_values=runtime_values,
                                runtime_bindings=runtime_bindings,
                            ),
                        )
                        transport_config.update(
                            connection_to_transport_config(resolver_connection)
                        )
                    else:
                        delegated_connection = self._delegated_mcp_connection(
                            server=server,
                            runtime_values=runtime_values,
                            runtime_bindings=runtime_bindings,
                            allow_delegated_authorization=allow_delegated_authorization,
                        )
                        if delegated_connection:
                            delegated_connection["_connector_runtime_refresh"] = (
                                partial(
                                    self._refresh_delegated_mcp_connection,
                                    server=server,
                                    runtime_bindings=runtime_bindings,
                                    allow_delegated_authorization=allow_delegated_authorization,
                                )
                            )
                            transport_config.update(
                                connection_to_transport_config(delegated_connection)
                            )
                        else:
                            runtime_build = await build_mcp_runtime_connection(
                                self.db,
                                server,
                                user_id=self._user_id,
                                mcp_auth_context=auth_context,
                            )
                            if runtime_build.connection is None:
                                if runtime_build.diagnostic is not None:
                                    self._mcp_oauth_diagnostics.append(
                                        runtime_build.diagnostic
                                    )
                                continue
                            transport_config.update(
                                connection_to_transport_config(runtime_build.connection)
                            )

                transport_config["concurrency_safe"] = bool(
                    getattr(server, "concurrency_safe", False)
                )
                transport_config["concurrent_tools"] = list(
                    getattr(server, "concurrent_tools", None) or []
                )

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

        except ConnectorRuntimeError:
            raise
        except Exception as e:
            # Preserve the legacy partial-return behavior for unrelated load
            # errors. Resolver and OAuth launch-config failures are handled
            # per server above so later servers can still load.
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
                            "id": int(api.id),
                            "name": api.name,
                            "description": api.description or "",
                            "url": api.url,
                            "method": api.method or "GET",
                            "headers": api.headers or {},
                            "body": api.body,
                            "env": api.env or {},
                            "runtime_input_schema": getattr(
                                api, "runtime_input_schema", None
                            ),
                            "runtime_bindings": getattr(api, "runtime_bindings", None),
                            "allow_delegated_authorization": bool(
                                getattr(api, "allow_delegated_authorization", False)
                            ),
                            "connector_runtime": self._get_connector_runtime_for(
                                "custom_api", int(api.id)
                            ),
                        }
                    )
            return custom_api_configs

        except ConnectorRuntimeError:
            raise
        except Exception as e:
            logger.error(
                f"Failed to get Custom API configs from database: {e}", exc_info=True
            )
            return []
