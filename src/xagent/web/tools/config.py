"""
Web-specific tool configuration for xagent

Provides web-specific configuration classes that load from database
and other web-specific sources.
"""

import copy
import inspect
import logging
import os
import re
import shlex
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from functools import partial
from pathlib import Path
from types import MappingProxyType
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    TypeVar,
    cast,
)

import httpx

from ...config import get_uploads_dir
from ...core.agent.result import (
    ClassifiedToolFailure,
    is_oauth_token_required_code,
    normalize_tool_failure_code,
)
from ...core.tools.adapters.vibe.config import (
    BaseToolConfig,
    MCPConfigLoadError,
    MCPFailurePolicy,
    MCPToolLoadSummary,
    ToolFactoryRuntimeSessionBoundaryError,
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
from ...core.tools.adapters.vibe.db_session import tool_session_scope
from ..services.tool_credentials import (
    TOOL_CREDENTIAL_SPECS,
    get_sql_connection_map,
    get_user_tool_allowlist,
    get_user_tool_overrides,
    has_user_tool_overrides_hook,
    has_user_tool_policy_hooks,
    resolve_tool_credential,
    unresolved_tool_policy_allowlist,
)
from ..services.user_oauth import (
    get_scoped_user_oauth_account,
    scoped_user_oauth_query,
)

logger = logging.getLogger(__name__)


OAUTH_TOKEN_EXPIRY_SKEW = timedelta(minutes=5)
OAUTH_TOKEN_GENERATION_MAX_LENGTH = 1024
OAUTH_TOKEN_RESOLVER_FAILURE_CODE = "oauth_token_resolver_failed"
OAUTH_TOKEN_RESOLVER_FAILURE_MESSAGE = "OAuth token resolver failed"
UNAVAILABLE_MCP_MESSAGE = "MCP server is unavailable."
UNAVAILABLE_MCP_CREDENTIAL_MESSAGE = "MCP server credentials are unavailable."
# This web-runtime allowlist is intentionally narrower than the adapter-layer
# public summary allowlist. It accepts only credential/config resolution reasons
# produced at this boundary; adapter/list-tools phases are sanitized separately
# after loading and must not be admitted here by sharing one broad constant.
MCP_UNAVAILABLE_REASONS = frozenset(
    {
        "authorization_required",
        "catalog_app_not_found",
        "config_load_failed",
        "insufficient_scope",
        "invalid_launch_config",
        "oauth_token_refresh_failed",
        "oauth_token_required",
        OAUTH_TOKEN_RESOLVER_FAILURE_CODE,
        "runtime_connection_failed",
        "token_refresh_failed",
    }
)


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
    ``instance_url`` carries the per-org API host a provider like Salesforce
    returns alongside its access token; providers without one leave it None.
    """

    access_token: str = field(repr=False)
    expires_at: datetime | None = None
    generation: str | None = field(default=None, repr=False)
    instance_url: str | None = None


@dataclass(frozen=True)
class _LegacyOAuthTokenResolution:
    access_token: str | None
    refresh_failed: bool = False
    # Set only for providers that return a per-org API host instead of
    # using a fixed domain (Salesforce) -- None for everyone else.
    instance_url: str | None = None


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


def oauth_token_resolver_installed() -> bool:
    """Whether an embedding application registered a token resolver hook.

    Deployment-level, not per-connector: the resolver is keyed on provider and
    end user and is embedder-implemented, so the only question answerable
    without calling it -- once per listed connector, on a list request, with
    unknown side effects -- is whether one exists at all. Read by
    ``list_mcp_apps`` to decide whether an mcp_oauth connector can plausibly
    obtain credentials without an ``MCPOAuthGrant``, and whether advertising
    interactive consent for it is meaningful (#1347).

    Selection is on hook presence, mirroring ``team_env_hook_installed``: an
    installed resolver that answers ``None`` for a given provider is a
    legitimate answer, not an absent hook.
    """
    resolver, _ = _get_oauth_token_resolver_hook()
    return resolver is not None


def _oauth_token_resolver_registration_matches(
    resolver: TokenResolver, registration_generation: int
) -> bool:
    current_resolver, current_generation = _get_oauth_token_resolver_hook()
    return (
        current_resolver is resolver and current_generation == registration_generation
    )


def _refresh_delegated_mcp_connection_from_snapshot(
    *,
    session_factory: Any,
    task_id: int | None,
    turn_id: str | None,
    user_id: int | None,
    server_id: int,
    connection_snapshot: Mapping[str, Any],
    runtime_bindings: Any,
    allow_delegated_authorization: bool,
    agent_team_id: int | None = None,
) -> dict[str, Any] | None:
    """Refresh delegated MCP headers without retaining construction objects."""
    if task_id is None or user_id is None:
        return None

    from ..services.connector_runtime import load_connector_runtime_view

    with session_factory() as db:
        runtime_view = load_connector_runtime_view(
            db=db,
            task_id=task_id,
            turn_id=turn_id,
            user_id=user_id,
            agent_team_id=agent_team_id,
        )
    runtime_values = runtime_view.get(f"mcp:{server_id}")
    runtime_headers = WebToolConfig._runtime_transport_headers(
        runtime_values=runtime_values if isinstance(runtime_values, dict) else None,
        runtime_bindings=runtime_bindings,
        allow_delegated_authorization=allow_delegated_authorization,
    )
    if not runtime_headers:
        return None

    connection = copy.deepcopy(dict(connection_snapshot))
    connection["headers"] = dict(connection.get("headers") or {})
    connection["headers"].update(runtime_headers)
    connection.pop("auth", None)
    return connection


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
    instance_url: str | None = None


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


class _OAuthInstanceUrlRequired(Exception):
    """The launch_config declares an instance_url env mapping, but the
    resolved token (hook or legacy DB path) didn't supply one.

    Raised instead of silently omitting the env var so the connector comes
    back as unavailable/reconnect-required, matching how a missing
    access_token is already surfaced, rather than launching a subprocess
    that fails opaquely on its first real tool call. Carries the env_mapping
    key that triggered it, mirroring _OAuthLaunchConfigInvalid.field, so a
    second provider adding its own instance_url-mapped key someday doesn't
    leave both call sites' log lines unable to say which one failed.
    """

    def __init__(self, *, env_key: str) -> None:
        super().__init__(env_key)
        self.env_key = env_key


@dataclass(frozen=True)
class _ToolFactoryRuntimeLoadPlan:
    """Detached inputs describing the synchronous factory reads to prefetch."""

    user_id: int | None
    task_id: str | None
    connector_runtime_turn_id: str | None
    load_policy: bool
    load_basic: bool
    load_sql: bool
    load_custom_api: bool
    load_vision: bool
    load_image: bool
    load_video: bool
    load_audio: bool
    published_agent_policy: Any | None
    # The governing agent's owning team, detached from the ORM row. ``None``
    # is the fail-closed default: an un-migrated construction site (or a run
    # with no governing agent) resolves personal-only connectors.
    connector_team_id: int | None = None


@dataclass(frozen=True)
class _ToolRuntimePolicySnapshot:
    """Detached per-turn policy loaded by a worker-owned Session."""

    tool_overrides: dict[str, Any] = field(default_factory=dict)
    tool_allowlist: list[str] | None = None


@dataclass(frozen=True)
class _ToolFactoryRuntimeSnapshot:
    """Worker-produced values consumed synchronously by tool creators."""

    plan: _ToolFactoryRuntimeLoadPlan
    tool_credentials: dict[tuple[str, str], str | None] = field(default_factory=dict)
    sql_connections: dict[str, str] = field(default_factory=dict)
    failed_inputs: frozenset[str] = frozenset()
    custom_api_configs: list[dict[str, Any]] = field(default_factory=list)
    tool_overrides: dict[str, Any] = field(default_factory=dict)
    tool_allowlist: list[str] | None = None
    vision_model: Any | None = None
    image_models: dict[str, Any] = field(default_factory=dict)
    image_generate_model: Any | None = None
    image_edit_model: Any | None = None
    video_models: dict[str, Any] = field(default_factory=dict)
    video_model: Any | None = None
    asr_models: dict[str, Any] = field(default_factory=dict)
    asr_model: Any | None = None
    tts_models: dict[str, Any] = field(default_factory=dict)
    tts_model: Any | None = None
    sound_effect_models: dict[str, Any] = field(default_factory=dict)
    sound_effect_model: Any | None = None
    music_models: dict[str, Any] = field(default_factory=dict)
    music_model: Any | None = None
    published_agent_records: list[Any] = field(default_factory=list)


@dataclass(frozen=True)
class _RetainedFactoryModelState:
    """Detached model values that remain readable after factory handoff."""

    load_vision: bool
    load_image: bool
    load_video: bool
    load_audio: bool
    vision_model: Any | None
    image_models: Mapping[str, Any]
    image_generate_model: Any | None
    image_edit_model: Any | None
    video_models: Mapping[str, Any]
    video_model: Any | None
    asr_models: Mapping[str, Any]
    asr_model: Any | None
    tts_models: Mapping[str, Any]
    tts_model: Any | None
    sound_effect_models: Mapping[str, Any]
    sound_effect_model: Any | None
    music_models: Mapping[str, Any]
    music_model: Any | None

    def __post_init__(self) -> None:
        for field_name in (
            "image_models",
            "video_models",
            "asr_models",
            "tts_models",
            "sound_effect_models",
            "music_models",
        ):
            object.__setattr__(
                self,
                field_name,
                MappingProxyType(dict(getattr(self, field_name))),
            )

    @classmethod
    def from_factory_snapshot(
        cls, snapshot: _ToolFactoryRuntimeSnapshot
    ) -> "_RetainedFactoryModelState":
        plan = snapshot.plan
        return cls(
            load_vision=plan.load_vision,
            load_image=plan.load_image,
            load_video=plan.load_video,
            load_audio=plan.load_audio,
            vision_model=snapshot.vision_model if plan.load_vision else None,
            image_models=snapshot.image_models if plan.load_image else {},
            image_generate_model=(
                snapshot.image_generate_model if plan.load_image else None
            ),
            image_edit_model=snapshot.image_edit_model if plan.load_image else None,
            video_models=snapshot.video_models if plan.load_video else {},
            video_model=snapshot.video_model if plan.load_video else None,
            asr_models=snapshot.asr_models if plan.load_audio else {},
            asr_model=snapshot.asr_model if plan.load_audio else None,
            tts_models=snapshot.tts_models if plan.load_audio else {},
            tts_model=snapshot.tts_model if plan.load_audio else None,
            sound_effect_models=(
                snapshot.sound_effect_models if plan.load_audio else {}
            ),
            sound_effect_model=(
                snapshot.sound_effect_model if plan.load_audio else None
            ),
            music_models=snapshot.music_models if plan.load_audio else {},
            music_model=snapshot.music_model if plan.load_audio else None,
        )


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
    if not is_oauth_token_required_code(raw_failure_code):
        return None
    return raw_failure_code


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
    from ...web.mcp_apps import restrict_to_app_scoped_oauth_grant

    return restrict_to_app_scoped_oauth_grant(
        app_info.get("id"), (app_info.get("provider"), app_info.get("id"))
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


def _oauth_launch_config_static_env(
    launch_config: Mapping[str, Any],
) -> Mapping[str, str]:
    """Server-only static secrets forwarded verbatim from the host process env.

    Unlike env_mapping (per-user OAuth token values), these are platform-wide
    values read from this process's own environment at transport-config build
    time — e.g. a shared API developer token that isn't tied to any one user's
    OAuth grant.

    A static_env entry can name *any* host env var to forward, so this is
    only safe as long as launch_config is written exclusively by migrations,
    the builtin registry, and admin-gated API routes — never by an
    end-user-writable path.
    """
    static_env = launch_config.get("static_env")
    if static_env is None:
        return {}
    if isinstance(static_env, Mapping):
        return static_env
    logger.warning(
        "Ignoring OAuth MCP launch config static_env because static_env must be a mapping"
    )
    return {}


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
        from ..api.auth import _resolve_oauth_redirect_uri, _resolve_oauth_secret
        from ..models.oauth_provider import OAuthProvider
        from ..oauth_provider_quirks import requires_json_accept_header

        provider_config = (
            db.query(OAuthProvider)
            .filter(OAuthProvider.provider_name == provider_name)
            .first()
        )
        if not provider_config:
            logger.warning(f"Unknown provider for refresh: {provider_name}")
            return False

        # Matches the connect path (_resolve_oauth_secret, api/auth.py) --
        # without the same env-var fallback here, a provider row seeded with
        # blank credentials (e.g. a migration that ran before the app's env
        # was fully populated) connects fine via the env fallback but then
        # fails every refresh, since this used to read only the DB row.
        client_id = _resolve_oauth_secret(
            provider_name, provider_config.client_id, "CLIENT_ID"
        )
        client_secret = _resolve_oauth_secret(
            provider_name, provider_config.client_secret, "CLIENT_SECRET"
        )

        if not client_id or not client_secret:
            logger.warning(
                f"{provider_name} OAuth not configured (missing CLIENT_ID or SECRET)."
            )
            return False

        # Normalize once for the special-case comparisons below; DB lookups
        # and log messages above/below keep using the original provider_name
        # so an admin-created provider's display casing is unaffected.
        normalized_provider = provider_name.lower()

        if normalized_provider == "meta":
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
                    db.flush([oauth_account])
                    logger.info(
                        f"Successfully refreshed {provider_name} token for user {oauth_account.user_id}"
                    )
                    return True
            else:
                logger.error(
                    "Failed to refresh %s token (status %s)",
                    provider_name,
                    response.status_code,
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
        }
        post_kwargs: dict[str, Any] = {}
        # Matches the code-exchange branch in api/auth.py: an admin-created
        # provider named "Zoom" would otherwise connect fine but silently
        # fail every refresh an hour later.
        if normalized_provider == "zoom":
            # Zoom's token endpoint requires HTTP Basic Auth for client
            # credentials (client_id:client_secret, base64) on every refresh,
            # same as the initial code exchange.
            post_kwargs["auth"] = httpx.BasicAuth(client_id, client_secret)
        else:
            data["client_id"] = client_id
            data["client_secret"] = client_secret

        refresh_token_url = provider_config.token_url
        if normalized_provider == "deputy":
            # Deputy's docs (both the code-exchange and refresh legs) list
            # `redirect_uri` and `scope` as required body params here too,
            # matching the code-exchange branch in api/auth.py. scope is
            # read from provider_config.default_scopes -- same source, and
            # same "no app-level oauth_scopes override" caveat, as that
            # code-exchange leg (see its comment) -- rather than a
            # hardcoded literal, so an admin who edits this provider row's
            # scopes doesn't leave refresh silently still sending the old
            # value.
            data["redirect_uri"] = _resolve_oauth_redirect_uri(
                provider_name, provider_config
            )
            data["scope"] = " ".join(provider_config.default_scopes or []) or (
                "longlife_refresh_token"
            )
            # Deputy's generic once.deputy.com host only serves the initial
            # code exchange -- token renewal must go to the same per-install
            # host returned as `endpoint` in that exchange (and persisted as
            # UserOAuth.instance_url), not the static token_url on the
            # provider row. See deputy.py's _instance_url() for the matching
            # use-time validation of that same stored value.
            stored_instance_url = getattr(oauth_account, "instance_url", None)
            if not stored_instance_url:
                logger.warning(
                    f"Cannot refresh Deputy token for user "
                    f"{oauth_account.user_id}: no instance_url stored on "
                    "this connection."
                )
                return False
            refresh_token_url = f"{stored_instance_url}/oauth/access_token"

        headers = {}
        if normalized_provider == "linkedin":
            headers["Content-Type"] = "application/x-www-form-urlencoded"
        if requires_json_accept_header(normalized_provider):
            headers["Accept"] = "application/json"

        # Matches the code-exchange branch in api/auth.py: Atlassian's token
        # endpoint requires a JSON body on refresh too, not form-urlencoded.
        body_kwarg: dict[str, Any] = {"data": data}
        if normalized_provider == "jira":
            headers["Content-Type"] = "application/json"
            body_kwarg = {"json": data}

        async with httpx.AsyncClient() as client:
            response = await client.post(
                refresh_token_url,
                headers=headers,
                timeout=10.0,
                **body_kwarg,
                **post_kwargs,
            )

        if response.status_code == 200:
            data = response.json()
            if "access_token" in data:
                oauth_account.access_token = data["access_token"]
                if "refresh_token" in data:
                    oauth_account.refresh_token = data["refresh_token"]
                if "instance_url" in data:
                    # Matches the code-exchange branch in api/auth.py:
                    # Salesforce can return a different instance_url on
                    # refresh (e.g. after an org migration), so this is
                    # re-persisted here too, not just at initial connect.
                    # Type/non-empty checked (not full host/scheme
                    # validation -- that stays salesforce.py's own
                    # use-time job) before overwriting: this row's
                    # existing instance_url is a previously-valid value,
                    # and a malformed refresh response replacing it would
                    # break the connector on its next use with no signal
                    # at refresh time that anything went wrong.
                    refreshed_instance_url = data["instance_url"]
                    if (
                        isinstance(refreshed_instance_url, str)
                        and refreshed_instance_url
                    ):
                        oauth_account.instance_url = refreshed_instance_url
                    else:
                        logger.warning(
                            f"Refresh response for {provider_name} (user "
                            f"{oauth_account.user_id}) had a malformed "
                            "instance_url; keeping the previously stored value"
                        )
                if normalized_provider == "deputy" and "endpoint" in data:
                    # Deputy's equivalent of the block above -- its refresh
                    # response carries the per-install host under `endpoint`,
                    # not `instance_url`, and without a scheme (matches the
                    # code-exchange branch in api/auth.py).
                    from ..api.auth import _normalize_deputy_endpoint

                    refreshed_endpoint = _normalize_deputy_endpoint(data["endpoint"])
                    if refreshed_endpoint:
                        oauth_account.instance_url = refreshed_endpoint
                    else:
                        logger.warning(
                            f"Refresh response for {provider_name} (user "
                            f"{oauth_account.user_id}) had a malformed "
                            "endpoint; keeping the previously stored value"
                        )
                if "expires_in" in data:
                    oauth_account.expires_at = datetime.now(timezone.utc) + timedelta(
                        seconds=data["expires_in"]
                    )
                db.flush([oauth_account])
                logger.info(
                    f"Successfully refreshed {provider_name} token for user {oauth_account.user_id}"
                )
                return True
        else:
            logger.error(
                "Failed to refresh %s token (status %s)",
                provider_name,
                response.status_code,
            )

    except Exception as e:
        logger.error(
            "Exception refreshing token for %s with %s",
            provider_name,
            type(e).__name__,
        )

    return False


def _parse_custom_api_task_id(task_id: str | None) -> int | None:
    if not isinstance(task_id, str) or not task_id:
        return None
    if task_id.startswith("web_task_"):
        task_id = task_id.removeprefix("web_task_")
    try:
        return int(task_id)
    except (TypeError, ValueError):
        return None


def _load_custom_api_runtime_view_sync(
    db: Any,
    *,
    task_id: str | None,
    connector_runtime_turn_id: str | None,
    user_id: int | None,
    agent_team_id: int | None = None,
) -> dict[str, Any]:
    numeric_task_id = _parse_custom_api_task_id(task_id)
    if numeric_task_id is None or user_id is None:
        return {}
    try:
        from ..services.connector_runtime import load_connector_runtime_view

        return load_connector_runtime_view(
            db=db,
            task_id=numeric_task_id,
            turn_id=connector_runtime_turn_id,
            user_id=user_id,
            agent_team_id=agent_team_id,
        )
    except ConnectorRuntimeError:
        raise
    except Exception as exc:
        logger.warning(
            "Failed to resolve connector runtime view for task %s",
            task_id,
            exc_info=True,
        )
        raise ConnectorRuntimeError(
            ERROR_CONNECTOR_RUNTIME_UNAVAILABLE,
            "Connector runtime context is unavailable.",
            details={"reason": "runtime_view_resolution_failed"},
            status_code=503,
        ) from exc


def _visible_custom_api_query(
    db: Any, *, owner_user_id: int | None, team_api_ids: frozenset[int]
) -> Any:
    """Production custom-API visibility query, shared by both read points.

    Module-level, not a ``WebToolConfig`` method: ``_load_custom_api_factory_inputs``
    below is a module-level function that never sees a ``WebToolConfig``
    instance (the detached-plan contract documented on
    ``_load_tool_factory_runtime_snapshot``'s docstring, this module: "never
    receives the caller's ``WebToolConfig``, request Session, or ORM user"),
    so a method could serve only one of the two read points.
    """
    from ..models.custom_api import CustomApi
    from ..services.connector_team_scope import visible_custom_api_clause

    return (
        db.query(CustomApi)
        .filter(visible_custom_api_clause(owner_user_id, team_api_ids))
        .order_by(CustomApi.id)
    )


def _load_custom_api_factory_inputs(
    db: Any,
    *,
    user_id: int | None,
    task_id: str | None,
    connector_runtime_turn_id: str | None,
    connector_team_id: int | None = None,
) -> list[dict[str, Any]]:
    if user_id is None:
        return []

    # Resolved before the caller's guarded region (this helper carries no
    # try/except of its own): that region reports "every selected API is
    # unavailable", which is the wrong answer for "the scope could not be
    # resolved". The typed error is what survives the tool-creator frame --
    # an untyped one is dropped there with an ERROR and no tool set at all.
    from ..services.connector_team_scope import resolve_team_connector_ids_or_raise

    team_api_ids = frozenset(
        resolve_team_connector_ids_or_raise(
            db, team_id=connector_team_id, log_subject=user_id
        )["custom_api"]
    )

    rows = _visible_custom_api_query(
        db, owner_user_id=user_id, team_api_ids=team_api_ids
    ).all()
    if not rows:
        return []

    runtime_view = _load_custom_api_runtime_view_sync(
        db,
        task_id=task_id,
        connector_runtime_turn_id=connector_runtime_turn_id,
        user_id=user_id,
        agent_team_id=connector_team_id,
    )
    configs: list[dict[str, Any]] = []
    for api in rows:
        runtime_values = runtime_view.get(f"custom_api:{int(api.id)}")
        configs.append(
            _custom_api_config_from_model(
                api,
                dict(runtime_values) if isinstance(runtime_values, dict) else None,
            )
        )
    return configs


def _custom_api_config_from_model(
    api: Any,
    connector_runtime: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "id": int(api.id),
        "name": api.name,
        "description": api.description or "",
        "url": api.url,
        "method": api.method or "GET",
        "headers": api.headers or {},
        "body": api.body,
        "env": api.env or {},
        "runtime_input_schema": getattr(api, "runtime_input_schema", None),
        "runtime_bindings": getattr(api, "runtime_bindings", None),
        "allow_delegated_authorization": bool(
            getattr(api, "allow_delegated_authorization", False)
        ),
        "connector_runtime": connector_runtime,
    }


_SessionResultT = TypeVar("_SessionResultT")
_CreatorFailedInputKey = Literal["basic", "database"]


def _run_with_checked_out_session(
    session_factory: Any,
    operation: Callable[[Any], _SessionResultT],
) -> _SessionResultT:
    """Run one operation through a newly checked-out, always-closed Session."""
    with tool_session_scope(session_factory) as db:
        # Eager checkout keeps pool exhaustion outside callers' fallback
        # handlers, so it cannot be mistaken for an absent optional input.
        db.connection()
        return operation(db)


def _load_tool_factory_runtime_snapshot(
    session_factory: Any,
    plan: _ToolFactoryRuntimeLoadPlan,
    policy_snapshot: _ToolRuntimePolicySnapshot | None = None,
) -> _ToolFactoryRuntimeSnapshot:
    """Load factory-only DB inputs using worker-owned sessions.

    This function accepts only a session factory plus detached scalar policy. It
    never receives the caller's ``WebToolConfig``, request Session, or ORM user.
    """
    from ..services.db_runtime import is_database_pool_timeout

    tool_credentials: dict[tuple[str, str], str | None] = {}
    sql_connections: dict[str, str] = {}
    failed_inputs: set[str] = set()
    custom_api_configs: list[dict[str, Any]] = []
    runtime_policy = policy_snapshot or _ToolRuntimePolicySnapshot()
    vision_model: Any | None = None
    image_models: dict[str, Any] = {}
    image_generate_model: Any | None = None
    image_edit_model: Any | None = None
    video_models: dict[str, Any] = {}
    video_model: Any | None = None
    asr_models: dict[str, Any] = {}
    asr_model: Any | None = None
    tts_models: dict[str, Any] = {}
    tts_model: Any | None = None
    sound_effect_models: dict[str, Any] = {}
    sound_effect_model: Any | None = None
    music_models: dict[str, Any] = {}
    music_model: Any | None = None
    published_agent_records: list[Any] = []

    def load_snapshot_input(
        input_name: str,
        loader: Callable[[Any], Any],
        default: Any,
        *,
        failed_input_key: _CreatorFailedInputKey | None = None,
        propagated_exceptions: tuple[type[Exception], ...] = (),
        log_level: int = logging.WARNING,
        log_message: str | None = None,
    ) -> Any:
        """Load one logical input through an isolated Session boundary.

        ``basic`` credentials and ``database`` connections are the only
        creator-scoped fail-closed inputs: a plain loader failure records their
        key so that the matching creator raises while unrelated creators can
        continue. Custom API, published-agent, and model discovery retain their
        legacy soft defaults. Pool timeouts always propagate, and callers can
        name additional typed exceptions that must propagate.
        """

        def load(db: Any) -> Any:
            try:
                return loader(db)
            except Exception as exc:
                if isinstance(exc, propagated_exceptions):
                    raise
                if is_database_pool_timeout(exc):
                    raise
                if failed_input_key is not None:
                    failed_inputs.add(failed_input_key)
                if log_message is None:
                    logger.log(
                        log_level,
                        "Failed to prefetch %s tool input",
                        input_name,
                        exc_info=True,
                    )
                else:
                    logger.log(log_level, log_message, exc_info=True)
                return default

        return _run_with_checked_out_session(session_factory, load)

    if plan.load_policy and policy_snapshot is None:
        runtime_policy = _load_tool_runtime_policy_snapshot(
            session_factory,
            plan.user_id,
        )

    if plan.load_basic:

        def load_tool_credentials(db: Any) -> dict[tuple[str, str], str | None]:
            loaded_credentials: dict[tuple[str, str], str | None] = {}
            for tool_name, field_specs in TOOL_CREDENTIAL_SPECS.items():
                for field_name in field_specs:
                    loaded_credentials[(tool_name, field_name)] = (
                        resolve_tool_credential(db, tool_name, field_name)
                    )
            return loaded_credentials

        tool_credentials = load_snapshot_input(
            "tool credentials",
            load_tool_credentials,
            {},
            failed_input_key="basic",
            log_message="Failed to prefetch tool credentials",
        )

    if plan.load_sql:
        sql_connections = load_snapshot_input(
            "SQL connections",
            lambda db: get_sql_connection_map(db, plan.user_id),
            {},
            failed_input_key="database",
            log_message="Failed to prefetch SQL connections",
        )

    if plan.load_custom_api:
        custom_api_configs = load_snapshot_input(
            "Custom API configs",
            lambda db: _load_custom_api_factory_inputs(
                db,
                user_id=plan.user_id,
                task_id=plan.task_id,
                connector_runtime_turn_id=plan.connector_runtime_turn_id,
                connector_team_id=plan.connector_team_id,
            ),
            [],
            propagated_exceptions=(ConnectorRuntimeError,),
            log_level=logging.ERROR,
            log_message="Failed to get Custom API configs from database",
        )

    if (
        plan.published_agent_policy is not None
        and plan.published_agent_policy.query_required
        and plan.user_id is not None
    ):
        from ...core.tools.adapters.vibe.agent_tool import (
            load_published_agent_tool_records,
        )

        published_agent_user_id = plan.user_id
        published_agent_policy = plan.published_agent_policy
        published_agent_records = load_snapshot_input(
            "published-agent tools",
            lambda db: load_published_agent_tool_records(
                db,
                user_id=published_agent_user_id,
                policy=published_agent_policy,
            ),
            [],
            log_message="Failed to prefetch published-agent tools",
        )

    from ..services import model_service

    if plan.load_image:
        image_models = load_snapshot_input(
            "image",
            lambda db: model_service.get_image_models(db, plan.user_id),
            {},
        )
    if plan.load_video:
        video_models = load_snapshot_input(
            "video",
            lambda db: model_service.get_video_models(db, plan.user_id),
            {},
        )
    if plan.load_audio:
        asr_models = load_snapshot_input(
            "audio:asr-models",
            lambda db: model_service.get_asr_models(db, plan.user_id),
            {},
        )
        tts_models = load_snapshot_input(
            "audio:tts-models",
            lambda db: model_service.get_tts_models(db, plan.user_id),
            {},
        )
        sound_effect_models = load_snapshot_input(
            "audio:sound-effect-models",
            lambda db: model_service.get_sound_effect_models(db, plan.user_id),
            {},
        )
        music_models = load_snapshot_input(
            "audio:music-models",
            lambda db: model_service.get_music_models(db, plan.user_id),
            {},
        )

    if plan.load_vision:
        vision_model = load_snapshot_input(
            "vision",
            lambda db: model_service.get_default_vision_model(plan.user_id, db=db),
            None,
        )
    if plan.load_image and image_models:
        image_generate_model = load_snapshot_input(
            "image",
            lambda db: model_service.get_default_image_generate_model(
                plan.user_id, db=db
            ),
            None,
        )
        image_edit_model = load_snapshot_input(
            "image",
            lambda db: model_service.get_default_image_edit_model(plan.user_id, db=db),
            None,
        )
    if plan.load_video and video_models:
        video_model = load_snapshot_input(
            "video",
            lambda db: model_service.get_default_video_model(plan.user_id, db=db),
            None,
        )
    if plan.load_audio:
        if asr_models or tts_models:
            asr_model = load_snapshot_input(
                "audio:default-asr",
                lambda db: model_service.get_default_asr_model(plan.user_id, db=db),
                None,
            )
            tts_model = load_snapshot_input(
                "audio:default-tts",
                lambda db: model_service.get_default_tts_model(plan.user_id, db=db),
                None,
            )
        if sound_effect_models:
            sound_effect_model = load_snapshot_input(
                "audio:default-sound-effect",
                lambda db: model_service.get_default_sound_effect_model(
                    plan.user_id, db=db
                ),
                None,
            )
        if music_models:
            music_model = load_snapshot_input(
                "audio:default-music",
                lambda db: model_service.get_default_music_model(plan.user_id, db=db),
                None,
            )

    return _ToolFactoryRuntimeSnapshot(
        plan=plan,
        tool_credentials=tool_credentials,
        sql_connections=sql_connections,
        failed_inputs=frozenset(failed_inputs),
        custom_api_configs=custom_api_configs,
        tool_overrides=runtime_policy.tool_overrides,
        tool_allowlist=runtime_policy.tool_allowlist,
        vision_model=vision_model,
        image_models=image_models,
        image_generate_model=image_generate_model,
        image_edit_model=image_edit_model,
        video_models=video_models,
        video_model=video_model,
        asr_models=asr_models,
        asr_model=asr_model,
        tts_models=tts_models,
        tts_model=tts_model,
        sound_effect_models=sound_effect_models,
        sound_effect_model=sound_effect_model,
        music_models=music_models,
        music_model=music_model,
        published_agent_records=published_agent_records,
    )


def _load_tool_runtime_policy_snapshot(
    session_factory: Any,
    user_id: int | None,
) -> _ToolRuntimePolicySnapshot:
    """Load each detached policy input through its own worker-owned Session."""
    from ..services.db_runtime import is_database_pool_timeout

    if user_id is None:
        return _ToolRuntimePolicySnapshot()

    from ..models.user import User

    # A registering application enforces authorization through the policy
    # hooks, so "could not resolve the policy" must not be reported as "no
    # policy configured". Both branches below reach the hooks not at all, so
    # the application has nothing to intercept; the loader itself has to fail
    # closed. Recorded per input and collapsed into a deny-all allowlist after
    # every input has had its own isolated Session, so an unresolvable
    # overrides read cannot skip the independent allowlist read.
    unresolved: set[str] = set()

    def load_policy_input(
        input_name: str,
        loader: Callable[[Any, Any], Any],
        default: Any,
    ) -> Any:
        def load(db: Any) -> Any:
            try:
                user = db.query(User).filter(User.id == user_id).first()
                if user is None:
                    unresolved.add(input_name)
                    logger.warning(
                        "Tool policy %s unresolved: user %s could not be reloaded",
                        input_name,
                        user_id,
                    )
                    return default
                return loader(db, user)
            except Exception as exc:
                # Pool timeouts keep propagating (the caller retries the turn),
                # and CancelledError is a BaseException that is deliberately
                # not caught here.
                if is_database_pool_timeout(exc):
                    raise
                unresolved.add(input_name)
                logger.exception("Failed to get user tool %s", input_name)
                return default

        return _run_with_checked_out_session(session_factory, load)

    overrides = load_policy_input(
        "overrides",
        lambda db, user: get_user_tool_overrides(db, user),
        {},
    )
    tool_overrides = dict(overrides) if isinstance(overrides, dict) else {}
    tool_allowlist = load_policy_input(
        "allowlist",
        lambda db, user: normalize_tool_allowlist(get_user_tool_allowlist(db, user)),
        None,
    )
    if unresolved:
        fail_closed = unresolved_tool_policy_allowlist()
        if fail_closed is not None:
            logger.error(
                "Tool policy unresolved for user %s (%s); denying every tool "
                "for this turn",
                user_id,
                ", ".join(sorted(unresolved)),
            )
            tool_allowlist = fail_closed
    return _ToolRuntimePolicySnapshot(
        tool_overrides=tool_overrides,
        tool_allowlist=tool_allowlist,
    )


# Maps the frontend's `app_locale` cookie (see i18n-context.tsx, which only
# ever sets "en" or "zh") to a Playwright-compatible locale tag.
_APP_LOCALE_TO_BROWSER_LOCALE = {"en": "en-US", "zh": "zh-CN"}


class WebToolConfig(BaseToolConfig):
    """Web-specific tool configuration that loads from database."""

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
        mcp_failure_policy: MCPFailurePolicy = MCPFailurePolicy.BEST_EFFORT,
        mcp_load_summary_tracer: Optional[Any] = None,
        mcp_load_summary_trace_task_id: Optional[str] = None,
        connector_team_id: Optional[int] = None,
        agent_creator_user_id: Optional[int] = None,
        declared_knowledge_bases: Optional[List[str]] = None,
        # Appended after every pre-existing parameter (not inserted
        # alongside its closest siblings above) so a caller still using
        # positional arguments for anything after agent_call_stack keeps
        # binding the same values it always did.
        voice: Optional[str] = None,
    ):
        # ``tool_selection_spec`` accepts :class:`ToolSelectionSpec` from
        # the tools adapter package; typed as ``Any`` here to avoid an
        # import cycle (web.tools → core.tools.adapters). The factory
        # reads ``config.get_tool_selection_spec()``. ``None`` defaults
        # to the ``_SpecAll`` ALL-mode (build every default tool).
        self._tool_selection_spec = tool_selection_spec
        self._mcp_failure_policy = mcp_failure_policy
        self._mcp_load_summary_tracer = mcp_load_summary_tracer
        self._mcp_load_summary_trace_task_id = mcp_load_summary_trace_task_id
        # The governing agent's owning team, never the acting/request user's
        # own team membership. ``None`` is the closed default: no request
        # path ever supplies this directly, it is read off a loaded ``Agent``
        # row (or a frozen snapshot of one) by the caller.
        self._connector_team_id = connector_team_id
        # The governing agent's creator -- always the same agent as
        # ``_connector_team_id`` above, populated by the same caller. Used by
        # the knowledge-base resolution path to tell the agent's creator
        # apart from any other runner of a team-governed agent.
        self._agent_creator_user_id = agent_creator_user_id
        # The governing agent's own STORED ``knowledge_bases`` declaration --
        # never the model-supplied value on a search request. Read from the
        # same place ``allowed_collections`` above already is; kept as a
        # separate field because ``allowed_collections`` on a tool_args
        # object can be overwritten by the model, and this value must not
        # be confusable with that one at the resolution point.
        self._declared_knowledge_bases = declared_knowledge_bases
        self._task_runtime_contribution: Any = None
        self._task_runtime_workspace: Any = None
        self._live_db = db
        self._db_factory = db_factory
        self._lazy_db = None
        self.request = request
        self._user_id = user_id
        # No identity can carry administrative privilege. For identified
        # configs, an explicit value remains authoritative; only an unset value
        # may fall back to the authenticated request user.
        if self._user_id is None:
            self._is_admin_value = False
        elif is_admin is not None:
            self._is_admin_value = bool(is_admin)
        else:
            self._is_admin_value = self._get_is_admin_from_request(request)
        # Initialize workspace_config with base_dir and task_id if provided
        if workspace_config is None:
            workspace_config = {}
        if task_id:
            workspace_config["task_id"] = task_id
        # Use uploads dir if workspace_base_dir not explicitly provided
        if workspace_base_dir is None:
            workspace_base_dir = str(get_uploads_dir())
        # Ensure base_dir is in workspace_config (required by ToolFactory.create_workspace)
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
        # Already-resolved onboarding output-voice preference (see
        # get_voice's docstring on BaseToolConfig for why this threads into
        # delegated AgentTool children).
        self._voice = voice
        self._excluded_agent_id: Optional[int] = None

        # Cache user object for hook queries.
        # Use explicit user param first; fall back to request.user.
        self._user = user if user is not None else getattr(request, "user", None)
        self._cached_tool_overrides: Optional[dict] = None
        # ``None`` is a meaningful allowlist value ("no allowlist"), so a
        # separate flag tracks whether the hook has been consulted yet.
        self._cached_tool_allowlist: Optional[list] = None
        self._tool_allowlist_cached: bool = False
        # Names the policy inputs whose read could not be resolved (the hook
        # never ran). ``get_user_tool_allowlist`` turns a non-empty set into a
        # deny-all allowlist so the execution layer fails closed instead of
        # building every tool. Each accessor clears its own entry before
        # re-reading, so a transient failure cannot latch deny-all onto a config
        # that is reused across turns.
        self._unresolved_tool_policy_inputs: set[str] = set()

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
        self._factory_runtime_snapshot: _ToolFactoryRuntimeSnapshot | None = None
        self._retained_factory_model_state: _RetainedFactoryModelState | None = None
        self._factory_runtime_handed_off = False
        self._pending_runtime_policy: _ToolRuntimePolicySnapshot | None = None
        # get_browser_locale() memoizes on first call: _detach_factory_runtime_resources()
        # nulls self.request once tools are built, but AgentService can rebuild tools on
        # this same config instance later (_ensure_tools_initialized), at which point
        # re-deriving from self.request would silently lose the already-resolved locale
        # to the deployment default rather than reusing what was actually resolved.
        self._browser_locale_resolved = False
        self._cached_browser_locale: Optional[str] = None

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

    def _get_is_admin_from_request(self, request: Any) -> bool:
        """Extract is_admin flag from the request user, defaulting to False.

        Uses ``getattr`` so a minimal request object (e.g. one carrying only a
        user id) doesn't trip the broad ``except`` and log a spurious warning.

        Only safe as long as ``request`` never reaches here as a real
        Starlette ``Request``/``HTTPConnection`` without ``AuthenticationMiddleware``
        installed (which this app never installs): accessing ``.user`` on one
        raises ``AssertionError``, not ``AttributeError``, so ``getattr``'s
        default would not save it. Currently unreachable because
        ``create_default_tools`` always passes an explicit ``user=``/``is_admin=``
        (this path only runs when ``is_admin`` is left unset), but a future
        caller that omits both and passes a real ``Request`` here would crash
        instead of defaulting to ``False``.
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

    def _resolve_factory_model_field(
        self,
        *,
        load_flag: str,
        field_name: str,
        cache_name: str,
        loader: Callable[[], Any],
        terminal_neutral: Any,
    ) -> Any:
        snapshot = self._factory_runtime_snapshot
        if snapshot is not None and getattr(snapshot.plan, load_flag):
            return getattr(snapshot, field_name)

        retained = self._retained_factory_model_state
        if retained is not None and getattr(retained, load_flag):
            return getattr(retained, field_name)

        if self._factory_runtime_handed_off:
            return terminal_neutral

        cached = getattr(self, cache_name)
        if cached is None:
            cached = loader()
            setattr(self, cache_name, cached)
        return cached

    def _get_factory_model_value(
        self,
        *,
        load_flag: str,
        field_name: str,
        cache_name: str,
        loader: Callable[[], Any | None],
    ) -> Any | None:
        return self._resolve_factory_model_field(
            load_flag=load_flag,
            field_name=field_name,
            cache_name=cache_name,
            loader=loader,
            terminal_neutral=None,
        )

    def _get_factory_model_mapping(
        self,
        *,
        load_flag: str,
        field_name: str,
        cache_name: str,
        loader: Callable[[], Mapping[str, Any]],
    ) -> Dict[str, Any]:
        resolved = self._resolve_factory_model_field(
            load_flag=load_flag,
            field_name=field_name,
            cache_name=cache_name,
            loader=loader,
            terminal_neutral={},
        )
        return dict(resolved)

    def get_vision_model(self) -> Optional[Any]:
        """Get vision model, prioritizing explicitly provided model over database."""
        if hasattr(self, "_explicit_vision_model") and self._explicit_vision_model:
            return self._explicit_vision_model

        return self._get_factory_model_value(
            load_flag="load_vision",
            field_name="vision_model",
            cache_name="_cached_vision_config",
            loader=self._load_vision_model,
        )

    def get_image_models(self) -> Dict[str, Any]:
        """Load image models from database."""
        return self._get_factory_model_mapping(
            load_flag="load_image",
            field_name="image_models",
            cache_name="_cached_image_configs",
            loader=self._load_image_models,
        )

    def get_video_models(self) -> Dict[str, Any]:
        """Load video models from database."""
        return self._get_factory_model_mapping(
            load_flag="load_video",
            field_name="video_models",
            cache_name="_cached_video_configs",
            loader=self._load_video_models,
        )

    def get_image_generate_model(self) -> Optional[Any]:
        """Get default image generation model from database."""
        return self._get_factory_model_value(
            load_flag="load_image",
            field_name="image_generate_model",
            cache_name="_cached_image_generate_model",
            loader=self._load_image_generate_model,
        )

    def get_image_edit_model(self) -> Optional[Any]:
        """Get default image editing model from database."""
        return self._get_factory_model_value(
            load_flag="load_image",
            field_name="image_edit_model",
            cache_name="_cached_image_edit_model",
            loader=self._load_image_edit_model,
        )

    def get_video_model(self) -> Optional[Any]:
        """Get default video generation model from database."""
        return self._get_factory_model_value(
            load_flag="load_video",
            field_name="video_model",
            cache_name="_cached_video_model",
            loader=self._load_video_model,
        )

    async def get_mcp_server_configs(self) -> List[Dict[str, Any]]:
        """Load MCP server configurations from database."""
        if self._user_id is None:
            return []

        if not self._include_mcp_tools:
            return []

        if self._cached_mcp_configs is not None:
            if self._mcp_config_cache_is_valid():
                return self._cached_mcp_configs
            # Once rejected, an executable config cache must not become valid
            # again merely because a refresh updated generation/expiry metadata
            # before failing.
            self._cached_mcp_configs = None

        configs = await self._load_mcp_server_configs()
        self._store_mcp_config_cache_if_cacheable(configs)
        return configs

    def _serialize_mcp_user_id(self) -> str:
        """Return the explicit identity used to isolate an MCP config."""
        if self._user_id is None:
            raise RuntimeError("MCP configs require a user identity")
        return str(self._user_id)

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
                agent_team_id=self._connector_team_id,
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
        self._factory_runtime_snapshot = None
        self._pending_runtime_policy = None
        return True

    def set_execution_scope(self, scope: Optional[Any]) -> bool:
        """Switch the per-turn execution scope for a reused tool config.

        ``WebToolConfig`` is cached with ``AgentService`` by task. The scope is
        per-turn state exactly like the connector runtime turn id: an embedder
        resolver may return a scope carrying turn-varying data that the base
        namespace fingerprint does not cover, and a cached config must not keep
        serving the first turn's object to the OAuth resolver hook.
        Namespace-affecting changes never reach here -- the caller evicts on a
        fingerprint change before this runs.

        No-op only for the same object, or for two plain base
        ``ExecutionScope`` instances comparing equal: the base class's
        compared fields fully describe it, and the persisted-snapshot path
        decodes a fresh equal instance every turn that must not force a tool
        rebuild. A subclass instance always swaps (unless identical) --
        subclasses may carry per-turn payload declared ``compare=False``, so
        value equality cannot prove freshness for them, and two same-type
        instances differing only in that payload compare equal.
        """
        from xagent.core.execution_scope import ExecutionScope

        previous = self._execution_scope
        if previous is scope:
            return False
        if (
            type(previous) is ExecutionScope
            and type(scope) is ExecutionScope
            and previous == scope
        ):
            return False
        self._execution_scope = scope
        self._cached_mcp_configs = None
        self._factory_runtime_snapshot = None
        self._pending_runtime_policy = None
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

    @staticmethod
    def _runtime_transport_headers(
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

    def _build_delegated_mcp_refresh_callback(
        self,
        *,
        server: Any,
        runtime_bindings: Any,
        allow_delegated_authorization: bool,
    ) -> Callable[[], dict[str, Any] | None]:
        """Return a refresh callback containing only scalar and copied values."""
        connection_snapshot = self._mcp_connection_with_runtime_headers(
            server=server,
            runtime_headers={},
        )
        return partial(
            _refresh_delegated_mcp_connection_from_snapshot,
            session_factory=self.get_session_factory(),
            task_id=self._parse_numeric_task_id(),
            turn_id=self._connector_runtime_turn_id,
            user_id=self._user_id,
            server_id=int(server.id),
            connection_snapshot=connection_snapshot,
            runtime_bindings=copy.deepcopy(runtime_bindings),
            allow_delegated_authorization=allow_delegated_authorization,
            # Memoised at build time: the whole tool set is built for one
            # agent, so this avoids re-deriving the agent's team id from the
            # ORM row per tool. The team hook itself still runs on every
            # per-tool-call refresh -- nothing in that chain caches it.
            agent_team_id=self._connector_team_id,
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

    def get_browser_locale(self) -> Optional[str]:
        """Derive a browser-automation locale from the ``app_locale``
        cookie the web UI's language switcher sets (see
        frontend/src/contexts/i18n-context.tsx), so a task's Playwright
        sessions request pages in the language the browser is currently
        set to, rather than a single locale hardcoded for every
        deployment or the browser's own Accept-Language header (which
        reflects OS/browser settings, not a deliberate in-app choice, and
        does not necessarily match the language the UI is displaying).

        This is a browser cookie, not a persisted account attribute: there
        is no ``locale`` column on ``User`` and no ``locale`` field on
        ``TaskCreateRequest``. It's device- and browser-scoped, lost when
        cookies are cleared, and not synced across a user's sessions.

        ``None`` (no request, no cookie, or an unrecognized value) lets the
        browser tool fall back to its own deployment default.

        Memoized on first call: ``_detach_factory_runtime_resources`` nulls
        ``self.request`` once tools are built, but ``AgentService`` can
        rebuild tools on this same config instance later
        (``_ensure_tools_initialized``) -- re-deriving at that point would
        silently lose the already-resolved locale to the deployment default
        instead of reusing what this task actually resolved.
        """
        if not self._browser_locale_resolved:
            cookies = getattr(self.request, "cookies", None)
            app_locale = getattr(cookies, "get", lambda _key: None)("app_locale")
            self._cached_browser_locale = self._normalize_app_locale_cookie(app_locale)
            self._browser_locale_resolved = True
        return self._cached_browser_locale

    @staticmethod
    def _normalize_app_locale_cookie(app_locale: Any) -> Optional[str]:
        """Map an ``app_locale`` cookie value to a Playwright locale tag.

        Matches on the primary language subtag so an already-valid BCP-47-ish
        variant (``zh-CN``, ``zh_CN``, ``EN``, ...) resolves the same as the
        exact ``"en"``/``"zh"`` the frontend writes today (see
        ``frontend/src/contexts/i18n-context.tsx``), instead of only matching
        those two literal strings.

        Primary-subtag-only means ``zh-TW``/``zh-HK`` would also map to the
        Simplified ``zh-CN`` in ``_APP_LOCALE_TO_BROWSER_LOCALE`` rather than
        a Traditional-Chinese locale -- harmless today since the frontend's
        ``Locale`` type is a hard ``"en" | "zh"`` union with no way to write
        those values, but revisit this if a Traditional-Chinese UI locale is
        ever added.
        """
        if not isinstance(app_locale, str) or not app_locale:
            return None
        primary = re.split(r"[-_]", app_locale, maxsplit=1)[0].lower()
        return _APP_LOCALE_TO_BROWSER_LOCALE.get(primary)

    def set_task_runtime_contribution(self, contribution: Any) -> None:
        """Attach the detached contribution built for this task."""

        self._task_runtime_contribution = contribution

    def get_task_runtime_contribution(self) -> Any:
        """Return the contribution consumed while building ``AgentService``."""

        return self._task_runtime_contribution

    def set_task_runtime_workspace(self, workspace: Any) -> None:
        """Retain the workspace already prepared for runtime providers."""

        self._task_runtime_workspace = workspace

    def get_task_runtime_workspace(self) -> Any:
        """Return the workspace shared by providers and sandbox setup."""

        return self._task_runtime_workspace

    def get_task_id(self) -> Optional[str]:
        """Get task ID for session tracking."""
        return self._task_id

    def get_allowed_collections(self) -> Optional[List[str]]:
        """Get allowed knowledge base collections. None means all collections are allowed."""
        return self._allowed_collections

    def get_agent_creator_user_id(self) -> Optional[int]:
        """Get the governing agent's creator, for the knowledge-base seam."""
        return self._agent_creator_user_id

    def get_declared_knowledge_bases(self) -> Optional[List[str]]:
        """Get the governing agent's stored knowledge-base declaration.

        Distinct from ``get_allowed_collections()``: both are populated from
        the same agent configuration today, but this value is threaded
        through to the knowledge-base resolution point as its own input and
        must never be conflated with a tool-args field a model can
        overwrite.
        """
        return self._declared_knowledge_bases

    def get_allowed_skills(self) -> Optional[List[str]]:
        """Get allowed skill names. None means all skills are allowed."""
        return self._allowed_skills

    def get_skill_scope_context(self) -> Any:
        """Build detached runtime identity for read-only skill providers."""
        from ..services.skill_runtime import build_detached_skill_scope

        return build_detached_skill_scope(user_id=self._user_id)

    def get_tool_selection_spec(self) -> Optional[Any]:
        """Typed spec accessor (preferred over :meth:`get_allowed_tools`).

        Returns a :class:`ToolSelectionSpec` instance when the caller
        supplied one via ``tool_selection_spec=ToolSelectionSpec.from_raw(...)``.
        ``ToolFactory.create_all_tools`` reads this first; falls back to
        ``get_allowed_tools()`` only if this returns ``None`` (legacy
        backward-compat).
        """
        return self._tool_selection_spec

    def get_mcp_failure_policy(self) -> MCPFailurePolicy:
        return self._mcp_failure_policy

    async def emit_mcp_load_summary(self, summary: MCPToolLoadSummary) -> None:
        """Persist one fixed-schema MCP setup audit event when configured."""
        if self._mcp_load_summary_tracer is None:
            return

        from ...core.agent.trace import SYSTEM_INFO

        await self._mcp_load_summary_tracer.trace_event(
            SYSTEM_INFO,
            task_id=self._mcp_load_summary_trace_task_id,
            data={
                "__audit_only__": True,
                "event_type": "mcp_load_summary",
                "requested_servers": list(summary.requested_servers),
                "loaded_servers": list(summary.loaded_servers),
                "failures": [
                    {
                        "server_name": failure.server_name,
                        "reason": failure.reason,
                    }
                    for failure in summary.failures
                ],
                "successful_tool_count": summary.successful_tool_count,
            },
            require_persisted=False,
        )

    def get_allowed_agent_ids(self) -> Optional[List[int]]:
        """Get explicitly allowed published agent IDs. None means use defaults."""
        return self._allowed_agent_ids

    def get_agent_tool_overrides(self) -> Dict[int, Dict[str, Any]]:
        """Get per-agent tool metadata/runtime overrides for delegation."""
        return self._agent_tool_overrides

    def get_published_agent_tool_records(self) -> Optional[List[Any]]:
        """Return worker-prefetched, ORM-free published-agent rows."""
        snapshot = self._factory_runtime_snapshot
        if snapshot is None or snapshot.plan.published_agent_policy is None:
            return None
        return list(snapshot.published_agent_records)

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

    def get_voice(self) -> Optional[str]:
        """See BaseToolConfig.get_voice's docstring."""
        return self._voice

    def _note_unresolved_tool_policy(self, input_name: str, reason: str) -> None:
        """Record that a policy input could not be resolved for this turn.

        The registered hook never ran on these paths, so the application that
        owns authorization has nothing to intercept and cannot repair the read.
        ``get_user_tool_allowlist`` converts a recorded input into a deny-all
        allowlist so the execution layer builds no tools instead of the full
        default set. Entries live only as long as the cached read that produced
        them: each accessor drops its own entry before consulting the hook
        again, so a transient failure denies one turn rather than latching.
        """
        if not has_user_tool_policy_hooks():
            # No application policy to lose: standalone xagent keeps its
            # unrestricted default rather than denying every tool.
            return
        self._unresolved_tool_policy_inputs.add(input_name)
        # Invalidate any allowlist already cached by an earlier read. The two
        # inputs are read in either order, so a clean allowlist cached before
        # this failure would otherwise keep reporting "no filtering" and hand
        # the turn the full tool set. Re-deriving it applies the denial.
        if input_name != "allowlist":
            self._tool_allowlist_cached = False
            self._cached_tool_allowlist = None
        logger.error(
            "Tool policy %s unresolved for user %s (%s); denying every tool "
            "for this turn",
            input_name,
            self._user_id,
            reason,
        )

    def get_user_tool_overrides(self) -> dict:
        """Return per-user tool overrides from the registered hook.

        Both display layer and execution layer read per-user tool policy from
        here, but this is no longer the whole picture: ``{}`` means either "no
        overrides configured" or "the policy could not be resolved", and the
        two are not distinguishable from the return value. The fail-closed
        signal for an unresolved read is carried by
        :meth:`get_user_tool_allowlist`, so the execution layer must consult
        both.
        """
        from ..services.db_runtime import is_database_pool_timeout

        if self._cached_tool_overrides is not None:
            return self._cached_tool_overrides
        # This read supersedes whatever the previous one concluded.
        self._unresolved_tool_policy_inputs.discard("overrides")
        if self._user is None:
            # No user to hand the hook: the policy is unresolved, not absent.
            # The overrides mapping stays a dict (the tool-listing API indexes
            # it); ``get_user_tool_allowlist`` carries the fail-closed signal.
            #
            # Gated on the overrides hook specifically, not on either hook:
            # with no overrides hook registered ``get_user_tool_overrides``
            # ignores ``user`` and returns ``{}`` regardless, so a missing user
            # resolves this input. Recording it would deny an allowlist-only
            # deployment whose allowlist hook answered successfully.
            if has_user_tool_overrides_hook():
                self._note_unresolved_tool_policy("overrides", "no runtime user")
            self._cached_tool_overrides = {}
            return {}
        try:
            self._cached_tool_overrides = get_user_tool_overrides(self.db, self._user)
        except Exception as exc:
            # A pool checkout timeout is not an unresolved policy: the very next
            # step needs the same pool, so propagate it for the caller to retry
            # rather than spending the turn with no tools. Matches
            # ``_load_tool_runtime_policy_snapshot``. Nothing is cached and no
            # input is recorded, so the retry re-reads from scratch.
            if is_database_pool_timeout(exc):
                raise
            logger.exception("Failed to get user tool overrides")
            self._note_unresolved_tool_policy("overrides", "hook read failed")
            self._cached_tool_overrides = {}
        return self._cached_tool_overrides

    def _build_factory_runtime_load_plan(self) -> _ToolFactoryRuntimeLoadPlan:
        spec = self._tool_selection_spec

        def wants_category(category: str) -> bool:
            if spec is None:
                return True
            includes_category = getattr(spec, "includes_category", None)
            if callable(includes_category):
                return bool(includes_category(category))
            return True

        includes_custom_api = (
            getattr(spec, "includes_custom_api", None) if spec is not None else None
        )
        includes_published_agent = (
            getattr(spec, "includes_published_agent", None)
            if spec is not None
            else None
        )
        published_agent_policy = None
        if (
            self._user_id
            and self.get_enable_agent_tools()
            and (
                spec is None
                or not callable(includes_published_agent)
                or bool(includes_published_agent())
            )
        ):
            from ...core.tools.adapters.vibe.agent_tool import (
                build_published_agent_tool_query_policy,
            )

            published_agent_policy = build_published_agent_tool_query_policy(
                excluded_agent_id=self.get_excluded_agent_id(),
                include_draft=False,
                allowed_agent_ids=self.get_allowed_agent_ids(),
                agent_tool_overrides=self.get_agent_tool_overrides(),
                enable_global_agent_tools=self.get_enable_global_agent_tools(),
                allow_cross_user_agent_ids=self.get_allow_cross_user_agent_ids(),
                agent_call_stack=self.get_agent_call_stack(),
            )
        return _ToolFactoryRuntimeLoadPlan(
            user_id=int(self._user_id) if self._user_id is not None else None,
            task_id=self._task_id,
            connector_runtime_turn_id=self._connector_runtime_turn_id,
            connector_team_id=self._connector_team_id,
            load_policy=(self._user_id is not None and has_user_tool_policy_hooks()),
            load_basic=(wants_category("basic") or wants_category("web_search")),
            load_sql=wants_category("database"),
            load_custom_api=(
                True
                if spec is None or not callable(includes_custom_api)
                else bool(includes_custom_api())
            ),
            load_vision=(
                wants_category("vision") and not bool(self._explicit_vision_model)
            ),
            load_image=wants_category("image"),
            load_video=wants_category("video"),
            load_audio=wants_category("audio"),
            published_agent_policy=published_agent_policy,
        )

    def _apply_factory_runtime_snapshot(
        self, snapshot: _ToolFactoryRuntimeSnapshot
    ) -> None:
        self._factory_runtime_snapshot = snapshot
        self._cached_tool_overrides = snapshot.tool_overrides
        self._cached_tool_allowlist = snapshot.tool_allowlist
        self._tool_allowlist_cached = True
        # The worker resolved the policy itself and already folded any
        # unresolvable input into ``snapshot.tool_allowlist``; stale entries from
        # an earlier in-request read must not latch deny-all onto this snapshot.
        self._unresolved_tool_policy_inputs.clear()

    async def prepare_factory_runtime(self) -> None:
        """Prefetch synchronous ToolFactory inputs without blocking its loop.

        The worker receives only a session factory and a detached load plan. It
        owns and closes every Session it creates; the request Session and ORM
        user retained by this config never cross the thread boundary.
        """
        if self._db_factory is None:
            from sqlalchemy.orm import Session

            # Some standalone/test configs supply duck-typed DB objects rather
            # than a real SQLAlchemy Session. They retain the legacy synchronous
            # getter contract; there is no safe engine from which to mint a
            # worker-owned Session.
            if not isinstance(self._live_db, Session):
                return

        from ..services.db_runtime import run_db_io_cancellation_safe

        plan = self._build_factory_runtime_load_plan()
        policy_snapshot = self._pending_runtime_policy
        self._pending_runtime_policy = None
        session_factory = self.get_session_factory()
        # The worker mints its own Session from the same engine. A live request
        # Session may already hold the pool's only connection after a SELECT,
        # so release clean read transactions before the worker checks out.
        self.release_db_connection()
        snapshot = await run_db_io_cancellation_safe(
            lambda: _load_tool_factory_runtime_snapshot(
                session_factory,
                plan,
                policy_snapshot,
            )
        )
        self._apply_factory_runtime_snapshot(snapshot)

    def discard_prepared_factory_runtime(self) -> None:
        """Discard construction-only snapshots without changing DB ownership."""
        self._factory_runtime_snapshot = None
        self._pending_runtime_policy = None

    def release_prepared_factory_runtime(self) -> None:
        """Compatibility alias for configs that only discard snapshots."""
        self.discard_prepared_factory_runtime()

    @staticmethod
    def _terminally_close_owned_lazy_db(lazy_db: Any) -> bool:
        """Close or invalidate an owned Session without losing retry ownership."""
        try:
            lazy_db.close()
            return True
        except Exception:
            try:
                lazy_db.invalidate()
                return True
            except Exception:
                logger.warning(
                    "Failed to invalidate lazy tool-factory database session",
                    exc_info=True,
                )
                return False

    def handoff_factory_runtime(self) -> None:
        """Verify and detach construction-only database resources.

        The request session remains caller-owned: a failed clean-release leaves
        it, the request, and its ORM user untouched. A lazily-created session is
        owned by this config and is therefore terminally closed on every path.
        """
        snapshot = self._factory_runtime_snapshot
        self._detach_factory_runtime_resources()
        if snapshot is not None:
            self._retained_factory_model_state = (
                _RetainedFactoryModelState.from_factory_snapshot(snapshot)
            )
        self._factory_runtime_handed_off = True
        self.discard_prepared_factory_runtime()

    def abort_factory_runtime(self) -> None:
        """Discard factory-only values while detaching request-owned resources."""
        self.discard_prepared_factory_runtime()
        try:
            self._detach_factory_runtime_resources()
        finally:
            self._factory_runtime_handed_off = True

    def _detach_factory_runtime_resources(self) -> None:
        """Detach every construction-owned session reference after verification."""
        from sqlalchemy.orm import Session

        from ..models.database import release_db_connection_if_clean

        live_db = self._live_db
        lazy_db = self._lazy_db
        if self._db_factory is None and isinstance(live_db, Session):
            self._db_factory = self.get_session_factory()

        # Only a real SQLAlchemy Session participates in the verified pool
        # handoff. Standalone embedders and unit tests may supply a duck-typed
        # object for synchronous getters; prepare_factory_runtime() deliberately
        # keeps that legacy path out of the worker/session-factory boundary.
        live_released = (
            release_db_connection_if_clean(live_db)
            if isinstance(live_db, Session)
            else True
        )
        lazy_released = (
            release_db_connection_if_clean(lazy_db) if lazy_db is not None else True
        )

        if lazy_db is not None:
            if not lazy_released:
                try:
                    lazy_db.rollback()
                except Exception:
                    logger.warning(
                        "Failed to roll back lazy tool-factory database session",
                        exc_info=True,
                    )
            if self._terminally_close_owned_lazy_db(lazy_db):
                self._lazy_db = None
            else:
                lazy_released = False

        if not live_released or not lazy_released:
            raise ToolFactoryRuntimeSessionBoundaryError()

        self._live_db = None
        self.request = None
        self._user = None

    async def refresh_runtime_policy(self) -> None:
        """Refresh only detached per-turn policy before signature comparison."""
        self.discard_prepared_factory_runtime()
        if self._user_id is None or not has_user_tool_policy_hooks():
            policy_snapshot = _ToolRuntimePolicySnapshot()
        else:
            from sqlalchemy.orm import Session

            if self._db_factory is None and not isinstance(self._live_db, Session):
                self._cached_tool_overrides = None
                self._tool_allowlist_cached = False
                self._cached_tool_allowlist = None
                # The accessors below each drop their own entry before
                # re-reading, so nothing from a previous turn survives; clearing
                # here keeps that explicit for the in-request branch.
                self._unresolved_tool_policy_inputs.clear()
                self.refresh_user_tool_overrides()
                self.refresh_user_tool_allowlist()
                return

            from ..services.db_runtime import run_db_io_cancellation_safe

            session_factory = self.get_session_factory()
            self.release_db_connection()
            policy_snapshot = await run_db_io_cancellation_safe(
                lambda: _load_tool_runtime_policy_snapshot(
                    session_factory,
                    int(cast(int, self._user_id)),
                )
            )

        self._cached_tool_overrides = policy_snapshot.tool_overrides
        self._cached_tool_allowlist = policy_snapshot.tool_allowlist
        self._tool_allowlist_cached = True
        # This snapshot is authoritative for the turn (the loader already
        # fail-closed any input it could not resolve), so drop entries left by
        # an earlier read rather than denying every tool for the rest of the run.
        self._unresolved_tool_policy_inputs.clear()
        self._pending_runtime_policy = policy_snapshot

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

        When a policy hook is registered but a policy input could not be
        resolved, this returns the empty list ("no tools allowed") rather than
        ``None``: the hook never ran, so the application enforcing
        authorization has nothing to intercept, and reporting "no allowlist"
        would build the globally available tool set. With no hook registered
        there is no policy to lose and the unrestricted default is kept.
        """
        from ..services.db_runtime import is_database_pool_timeout

        if self._tool_allowlist_cached:
            return self._cached_tool_allowlist
        # This read supersedes whatever the previous one concluded. The
        # overrides entry is left alone: an unresolved overrides read in this
        # same turn must still deny below.
        self._unresolved_tool_policy_inputs.discard("allowlist")
        try:
            # ``self._user`` may legitimately be ``None`` here: the allowlist is
            # resolved from the active execution scope, so the hook is consulted
            # with whatever user the config holds rather than short-circuited.
            self._cached_tool_allowlist = normalize_tool_allowlist(
                get_user_tool_allowlist(self.db, self._user)
            )
        except Exception as exc:
            # See get_user_tool_overrides: a pool timeout propagates for retry
            # instead of being recorded as an unresolved policy. Left uncached
            # so the retry re-reads, and no deny-all is applied on the way out.
            if is_database_pool_timeout(exc):
                raise
            logger.exception("Failed to get user tool allowlist")
            self._note_unresolved_tool_policy("allowlist", "hook read failed")
            self._cached_tool_allowlist = None
        if self._unresolved_tool_policy_inputs:
            # An unresolved read on either policy input denies every tool
            # rather than reporting "no allowlist configured", which would
            # skip the execution layer's positive filter entirely.
            fail_closed = unresolved_tool_policy_allowlist()
            if fail_closed is not None:
                self._cached_tool_allowlist = fail_closed
        self._tool_allowlist_cached = True
        return self._cached_tool_allowlist

    def refresh_user_tool_allowlist(self) -> Optional[list]:
        """Reload the positive tool allowlist from the registered hook."""
        # The active execution scope (hence the CA allowlist) can change while
        # an AgentService instance is reused across turns.
        #
        # ``get_user_tool_allowlist`` drops only its own unresolved entry before
        # re-reading, so a fresh allowlist answer lifts the denial it caused
        # while an unresolved overrides read from the same turn still denies.
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
        from sqlalchemy.orm import Session, sessionmaker

        if isinstance(self._live_db, Session):
            bind = self._live_db.get_bind()
            engine = getattr(bind, "engine", bind)
            return sessionmaker(
                bind=engine,
                autocommit=False,
                autoflush=False,
            )
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
        """Finalize the current factory runtime generation and owned resources.

        This discards the current prepared and retained factory state and
        closes any owned lazy database session. Old factory/database-backed
        media getters are neutral after close until a later
        ``prepare_factory_runtime()`` installs the next generation. An
        explicit constructor-supplied vision model is independent and remains
        authoritative after close.
        """
        try:
            self.discard_prepared_factory_runtime()
            lazy_db = self._lazy_db
            if lazy_db is not None:
                if self._terminally_close_owned_lazy_db(lazy_db):
                    self._lazy_db = None
                else:
                    raise ToolFactoryRuntimeSessionBoundaryError()
        finally:
            self._retained_factory_model_state = None
            self._factory_runtime_handed_off = True

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
        snapshot = self._factory_runtime_snapshot
        if snapshot is not None and snapshot.plan.load_basic:
            if "basic" in snapshot.failed_inputs:
                raise RuntimeError("Tool credential snapshot is unavailable")
            return snapshot.tool_credentials.get((tool_name, field_name))
        return resolve_tool_credential(self.db, tool_name, field_name)

    def get_sql_connections(self) -> Dict[str, str]:
        snapshot = self._factory_runtime_snapshot
        if snapshot is not None and snapshot.plan.load_sql:
            if "database" in snapshot.failed_inputs:
                raise RuntimeError("SQL connection snapshot is unavailable")
            return snapshot.sql_connections
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
        return self._get_factory_model_mapping(
            load_flag="load_audio",
            field_name="asr_models",
            cache_name="_cached_asr_models",
            loader=self._load_asr_models,
        )

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
        return self._get_factory_model_value(
            load_flag="load_audio",
            field_name="asr_model",
            cache_name="_cached_asr_model",
            loader=self._load_asr_model,
        )

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
        return self._get_factory_model_mapping(
            load_flag="load_audio",
            field_name="tts_models",
            cache_name="_cached_tts_models",
            loader=self._load_tts_models,
        )

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
        return self._get_factory_model_value(
            load_flag="load_audio",
            field_name="tts_model",
            cache_name="_cached_tts_model",
            loader=self._load_tts_model,
        )

    def get_sound_effect_models(self) -> Dict[str, Any]:
        """Load sound effect models from the independent model category."""
        return self._get_factory_model_mapping(
            load_flag="load_audio",
            field_name="sound_effect_models",
            cache_name="_cached_sound_effect_models",
            loader=self._load_sound_effect_models,
        )

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
        return self._get_factory_model_value(
            load_flag="load_audio",
            field_name="sound_effect_model",
            cache_name="_cached_sound_effect_model",
            loader=self._load_sound_effect_model,
        )

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
        return self._get_factory_model_mapping(
            load_flag="load_audio",
            field_name="music_models",
            cache_name="_cached_music_models",
            loader=self._load_music_models,
        )

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
        return self._get_factory_model_value(
            load_flag="load_audio",
            field_name="music_model",
            cache_name="_cached_music_model",
            loader=self._load_music_model,
        )

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
        if resolved.instance_url is not None and (
            type(resolved.instance_url) is not str or not resolved.instance_url
        ):
            raise _OAuthTokenResolverFailed(
                providers=providers,
                exception_type="InvalidInstanceUrl",
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
            instance_url=resolved.instance_url,
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
        return self._build_unavailable_mcp_config(
            server=server,
            reason=OAUTH_TOKEN_RESOLVER_FAILURE_CODE,
            message=UNAVAILABLE_MCP_CREDENTIAL_MESSAGE,
            diagnostic=diagnostic,
            failure_code=error.failure_code,
        )

    def _build_unavailable_mcp_config(
        self,
        *,
        server: Any,
        reason: str,
        message: str = UNAVAILABLE_MCP_MESSAGE,
        diagnostic: Mapping[str, Any] | None = None,
        failure_code: object = None,
    ) -> Dict[str, Any]:
        safe_reason = (
            reason
            if type(reason) is str and reason in MCP_UNAVAILABLE_REASONS
            else "runtime_connection_failed"
        )
        inner_config: Dict[str, Any] = {
            "unavailable": True,
            "reason": safe_reason,
            "message": message,
            "server_id": getattr(server, "id", None),
        }
        if diagnostic is not None:
            inner_config["diagnostic"] = dict(diagnostic)
        normalized_failure_code = normalize_tool_failure_code(failure_code)
        if normalized_failure_code is not None:
            inner_config["failure_code"] = normalized_failure_code
        serialized_user_id = self._serialize_mcp_user_id()
        return {
            "name": getattr(server, "name", ""),
            "transport": "unavailable",
            "description": getattr(server, "description", None),
            "config": inner_config,
            "user_id": serialized_user_id,
            "allow_users": [serialized_user_id],
        }

    def _build_oauth_mcp_stdio_transport_config(
        self,
        *,
        server: Any,
        app_info: Mapping[str, Any],
        access_token: str,
        instance_url: str | None = None,
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
                elif token_type == "instance_url":
                    if not instance_url:
                        raise _OAuthInstanceUrlRequired(env_key=env_key)
                    env[env_key] = instance_url
                else:
                    # A typo'd env_mapping value (e.g. "acess_token") would
                    # otherwise silently emit neither an env var nor an
                    # error -- the exact opaque failure mode
                    # _OAuthInstanceUrlRequired exists to prevent for the
                    # one token_type above it. Not developer-only: an admin
                    # can reach this through POST /admin/mcp/apps, whose
                    # launch_config is an unvalidated free-form dict (see
                    # PublicMCPAppCreate in admin_mcp.py -- its validator
                    # checks command/required_env/url/auth.type, not
                    # env_mapping's values), so a hand-typed custom OAuth
                    # app's env_mapping can carry this too.
                    logger.warning(
                        "Unrecognized launch_config.env_mapping token_type "
                        "'%s' for env var '%s'; no value forwarded",
                        token_type,
                        env_key,
                    )

            for env_key, host_env_var in _oauth_launch_config_static_env(
                launch_config
            ).items():
                value = os.environ.get(str(host_env_var))
                if value is not None:
                    env[env_key] = str(value)

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
                env["XAGENT_SLACK_FILE_ALLOWED_DIRS"] = allowed_file_dirs
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

    def _new_legacy_oauth_session(self) -> Any:
        """Open the transaction owner for legacy OAuth token maintenance."""
        if self._db_factory is not None:
            return self._db_factory()
        if self._live_db is not None:
            from sqlalchemy.orm import Session

            return Session(bind=self._live_db.get_bind().engine, autoflush=False)
        return self.get_session_factory()()

    async def _resolve_legacy_oauth_access_token(
        self,
        *,
        provider_name: object,
        app_id: object,
    ) -> _LegacyOAuthTokenResolution:
        """Resolve and persist one legacy OAuth account in an isolated transaction."""
        from ...web.mcp_apps import restrict_to_app_scoped_oauth_grant
        from ...web.models.user_oauth import UserOAuth

        if self._user_id is None:
            return _LegacyOAuthTokenResolution(access_token=None)
        user_id = int(self._user_id)
        oauth_db = self._new_legacy_oauth_session()
        try:
            if app_id:
                # A bare provider-level grant (e.g. UserOAuth.provider ==
                # "meta") never requested this app's own oauth_scopes, so it
                # can't be trusted to carry a permission added after that flow
                # already existed. See APPS_REQUIRING_APP_SCOPED_OAUTH_GRANT.
                providers_to_check = restrict_to_app_scoped_oauth_grant(
                    app_id, [provider_name, app_id]
                )
                oauth_account = (
                    scoped_user_oauth_query(
                        oauth_db,
                        user_id=user_id,
                        resource_owner_key=None,
                    )
                    .filter(UserOAuth.provider.in_(providers_to_check))
                    .first()
                )
                logger.info(
                    "OAUTH CONFIG: Checked providers %s for user %s. Found: %s",
                    providers_to_check,
                    self._user_id,
                    oauth_account is not None,
                )
            else:
                oauth_account = (
                    scoped_user_oauth_query(
                        oauth_db,
                        user_id=user_id,
                        resource_owner_key=None,
                    )
                    .filter(UserOAuth.provider == provider_name)
                    .first()
                )
                logger.info(
                    "OAUTH CONFIG: Checked provider '%s' for user %s. Found: %s",
                    provider_name,
                    self._user_id,
                    oauth_account is not None,
                )

            if not oauth_account or not oauth_account.access_token:
                return _LegacyOAuthTokenResolution(access_token=None)

            logger.info(
                "OAUTH CONFIG: Token found for '%s'. Refresh token present: %s, Expires: %s",
                provider_name,
                oauth_account.refresh_token is not None,
                oauth_account.expires_at,
            )
            account_id = int(oauth_account.id)
            is_valid = await refresh_oauth_token_if_needed(
                oauth_db,
                oauth_account,
                str(provider_name) if provider_name else "",
            )
            if not is_valid:
                logger.warning(
                    "OAUTH CONFIG: Token for '%s' is invalid and could not be refreshed. "
                    "Deleting OAuth record to prompt user for reconnection.",
                    provider_name,
                )
                # A failed flush leaves the transaction unusable. Roll it back,
                # reload the account, then persist the disconnection atomically.
                oauth_db.rollback()
                oauth_account = get_scoped_user_oauth_account(
                    oauth_db,
                    user_id=user_id,
                    account_id=account_id,
                    resource_owner_key=None,
                )
                if oauth_account is not None:
                    oauth_db.delete(oauth_account)
                    oauth_db.commit()
                return _LegacyOAuthTokenResolution(
                    access_token=None,
                    refresh_failed=True,
                )

            access_token = str(oauth_account.access_token)
            # Not direct attribute access: mypy infers oauth_account's
            # instance_url as Column[str] here (get_scoped_user_oauth_account
            # returns a type the SQLAlchemy plugin doesn't narrow the same
            # way as a plain query result), so getattr's own 3-arg overload
            # is what actually produces the correct `str | None` this
            # function's return type declares.
            instance_url = getattr(oauth_account, "instance_url", None)
            oauth_db.commit()
            return _LegacyOAuthTokenResolution(
                access_token=access_token, instance_url=instance_url
            )
        except Exception:
            oauth_db.rollback()
            raise
        finally:
            oauth_db.close()

    async def _build_mcp_server_config(
        self,
        *,
        server: Any,
        user_env_by_id: Mapping[int, Any],
        shared_env_by_id: Mapping[int, Any],
        env_source_by_id: Mapping[int, Any],
    ) -> Dict[str, Any]:
        """Build one MCP server config, preserving explicit unavailable outcomes."""
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
            "runtime_input_schema": getattr(server, "runtime_input_schema", None),
            "runtime_bindings": runtime_bindings,
            "allow_delegated_authorization": allow_delegated_authorization,
        }
        if runtime_values:
            context_values = runtime_values.get("context")
            config["connector_runtime"] = {
                "context": context_values if isinstance(context_values, dict) else {},
                "secrets": {},
                "auth_selector": {},
            }

        # Add transport-specific configuration
        transport_config: Dict[str, Any] = {}

        # Handle OAuth credentials
        if server.transport == "oauth":
            # Find corresponding OAuth account
            # The provider might be linkedin, google, etc. based on the app config
            from ...web.mcp_apps import get_app_for_mcp_server

            app_info = get_app_for_mcp_server(self.db, server)
            if app_info is None:
                logger.warning(
                    "OAuth MCP server '%s' has no matching catalog app",
                    getattr(server, "name", "<unknown>"),
                )
                return self._build_unavailable_mcp_config(
                    server=server,
                    reason="catalog_app_not_found",
                )
            provider_name = (
                app_info.get("provider") if app_info else server.name.lower()
            )

            # Some oauth records might be saved with the app_id as provider instead of the general provider_name
            # For example, "google-drive" instead of "google"
            app_id = app_info.get("id") if app_info else None

            hook_token: _ResolvedHookToken | None = None
            if app_info:
                configured_resource = _oauth_token_configured_resource(app_info)
                providers_to_resolve = _oauth_token_provider_candidates(app_info)
                try:
                    hook_token = await self._resolve_oauth_token_from_hook(
                        providers=providers_to_resolve,
                        resource=configured_resource,
                    )
                except _OAuthTokenResolverFailed as error:
                    return self._resolver_failure_config(
                        server=server,
                        error=error,
                    )

            if app_info and hook_token is not None:
                self._mark_hook_token_cache_metadata(hook_token)
                try:
                    transport_config = self._build_oauth_mcp_stdio_transport_config(
                        server=server,
                        app_info=app_info,
                        access_token=hook_token.access_token,
                        instance_url=hook_token.instance_url,
                    )
                except _OAuthLaunchConfigInvalid as error:
                    logger.warning(
                        "Skipping OAuth MCP server '%s' because launch_config.%s is invalid",
                        getattr(server, "name", "<unknown>"),
                        error.field,
                    )
                    return self._build_unavailable_mcp_config(
                        server=server,
                        reason="invalid_launch_config",
                    )
                except _OAuthInstanceUrlRequired as error:
                    logger.info(
                        "OAuth token resolver hook did not supply %s for MCP server '%s'",
                        error.env_key,
                        getattr(server, "name", "<unknown>"),
                    )
                    return self._build_unavailable_mcp_config(
                        server=server,
                        reason="oauth_token_required",
                        message=UNAVAILABLE_MCP_CREDENTIAL_MESSAGE,
                        failure_code="oauth_token_required",
                    )
                config["transport"] = "stdio"
                logger.info(
                    "OAuth token resolver supplied token for MCP server '%s' via provider '%s'",
                    getattr(server, "name", "<unknown>"),
                    hook_token.provider,
                )
            else:
                legacy_token = await self._resolve_legacy_oauth_access_token(
                    provider_name=provider_name,
                    app_id=app_id,
                )
                if legacy_token.refresh_failed:
                    return self._build_unavailable_mcp_config(
                        server=server,
                        reason="oauth_token_refresh_failed",
                        message=UNAVAILABLE_MCP_CREDENTIAL_MESSAGE,
                        failure_code="oauth_token_required",
                    )
                if legacy_token.access_token is None:
                    logger.info(
                        f"OAUTH CONFIG: No valid token found for '{provider_name}'."
                    )
                    return self._build_unavailable_mcp_config(
                        server=server,
                        reason="oauth_token_required",
                        message=UNAVAILABLE_MCP_CREDENTIAL_MESSAGE,
                        failure_code="oauth_token_required",
                    )
                logger.info("OAUTH CONFIG: Mapping '%s' to executable proxy", app_id)
                try:
                    transport_config = self._build_oauth_mcp_stdio_transport_config(
                        server=server,
                        app_info=app_info,
                        access_token=legacy_token.access_token,
                        instance_url=legacy_token.instance_url,
                    )
                except _OAuthLaunchConfigInvalid as error:
                    logger.warning(
                        "Skipping OAuth MCP server '%s' because launch_config.%s is invalid",
                        getattr(server, "name", "<unknown>"),
                        error.field,
                    )
                    return self._build_unavailable_mcp_config(
                        server=server,
                        reason="invalid_launch_config",
                    )
                except _OAuthInstanceUrlRequired as error:
                    logger.info(
                        "OAUTH CONFIG: No %s found for '%s'.",
                        error.env_key,
                        provider_name,
                    )
                    return self._build_unavailable_mcp_config(
                        server=server,
                        reason="oauth_token_required",
                        message=UNAVAILABLE_MCP_CREDENTIAL_MESSAGE,
                        failure_code="oauth_token_required",
                    )
                config["transport"] = "stdio"

        if server.transport == "stdio":
            if server.command:
                transport_config["command"] = server.command
            if server.args:
                transport_config["args"] = server.args
            # Decrypt global env and merge per-user override (user wins).
            from ...core.utils.encryption import decrypt_env_dict
            from ..services.mcp_runtime import caller_id_env, resolve_stdio_env

            merged_env = resolve_stdio_env(
                env_source_by_id.get(server.id),
                decrypt_env_dict(getattr(server, "env", None)),
                shared_env_by_id.get(server.id),
                user_env_by_id.get(server.id),
            )
            combined_env = {**(merged_env or {}), **caller_id_env(self._user_id)}
            if combined_env:
                transport_config["env"] = combined_env
            if server.cwd:
                transport_config["cwd"] = server.cwd

        elif server.transport in ["sse", "websocket", "streamable_http"]:
            from ...web.mcp_apps import get_app_for_mcp_server
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
                app_info = get_app_for_mcp_server(self.db, server)
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
                        remote_hook_token = await self._resolve_oauth_token_from_hook(
                            providers=remote_providers_to_resolve,
                            resource=remote_configured_resource,
                            resolver=resolver,
                        )
                    except _OAuthTokenResolverFailed as error:
                        return self._resolver_failure_config(
                            server=server,
                            error=error,
                        )

            if remote_hook_token is not None and resolver is not None:
                self._mark_hook_token_cache_metadata(remote_hook_token)
                resolver_connection = self._build_resolver_owned_mcp_connection(
                    resolver=resolver,
                    registration_generation=registration_generation,
                    resolved=remote_hook_token,
                    providers=remote_providers_to_resolve,
                    user_id=int(cast(int, self._user_id)),
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
                        self._build_delegated_mcp_refresh_callback(
                            server=server,
                            runtime_bindings=runtime_bindings,
                            allow_delegated_authorization=allow_delegated_authorization,
                        )
                    )
                    transport_config.update(
                        connection_to_transport_config(delegated_connection)
                    )
                else:
                    try:
                        runtime_build = await build_mcp_runtime_connection(
                            self.db,
                            server,
                            user_id=self._user_id,
                            mcp_auth_context=auth_context,
                        )
                    except ConnectorRuntimeError:
                        raise
                    except Exception as error:
                        logger.warning(
                            "MCP runtime connection failed for server '%s' with %s",
                            getattr(server, "name", "<unknown>"),
                            type(error).__name__,
                        )
                        return self._build_unavailable_mcp_config(
                            server=server,
                            reason="runtime_connection_failed",
                        )
                    if runtime_build.connection is None:
                        if runtime_build.diagnostic is not None:
                            self._mcp_oauth_diagnostics.append(runtime_build.diagnostic)
                        diagnostic = runtime_build.diagnostic
                        diagnostic_code = (
                            diagnostic.get("code")
                            if isinstance(diagnostic, Mapping)
                            else None
                        )
                        reason = (
                            diagnostic_code
                            if isinstance(diagnostic_code, str)
                            else "runtime_connection_failed"
                        )
                        return self._build_unavailable_mcp_config(
                            server=server,
                            reason=reason,
                            message=UNAVAILABLE_MCP_CREDENTIAL_MESSAGE,
                            diagnostic=diagnostic,
                            failure_code="oauth_token_required",
                        )
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
                transport_config["docker_environment"] = server.docker_environment
            if server.docker_working_dir:
                transport_config["docker_working_dir"] = server.docker_working_dir
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
        serialized_user_id = self._serialize_mcp_user_id()
        config["user_id"] = serialized_user_id
        config["allow_users"] = [serialized_user_id]  # Only allow current user

        logger.debug(f"Loaded MCP server config: {server.name} ({server.transport})")
        return config

    async def _load_mcp_server_config(
        self,
        *,
        server: Any,
        user_env_by_id: Mapping[int, Any],
        shared_env_by_id: Mapping[int, Any],
        env_source_by_id: Mapping[int, Any],
    ) -> Dict[str, Any]:
        """Isolate unexpected failures while loading one MCP server config."""
        try:
            return await self._build_mcp_server_config(
                server=server,
                user_env_by_id=user_env_by_id,
                shared_env_by_id=shared_env_by_id,
                env_source_by_id=env_source_by_id,
            )
        except ConnectorRuntimeError:
            raise
        except Exception as error:
            logger.warning(
                "Failed to load MCP server config for '%s' with %s",
                getattr(server, "name", "<unknown>"),
                type(error).__name__,
            )
            return self._build_unavailable_mcp_config(
                server=server,
                reason="config_load_failed",
            )

    def _visible_mcp_server_query(self, team_mcp_ids: frozenset[int]) -> Any:
        """Compile the production semi-join query for visible MCP servers.

        No join on ``user_mcpservers`` -- an inner join would drop every
        team-only server before the visibility predicate's ``OR`` could be
        evaluated. Extracted to its own method so a compiled-query test can
        exercise the exact object ``_load_mcp_server_configs`` uses, not a
        parallel reconstruction of it.
        """
        from ...web.models.mcp import MCPServer
        from ..services.connector_team_scope import visible_mcp_server_clause

        return (
            self.db.query(MCPServer)
            .filter(visible_mcp_server_clause(self._user_id, team_mcp_ids))
            .order_by(MCPServer.id)
        )

    async def _load_mcp_server_configs(self) -> List[Dict[str, Any]]:
        """Load MCP server configurations visible to this run: the user's
        personal servers, unioned with the governing agent's team-owned
        servers when a team-scope hook is installed. For each team-owned
        server id, this method also re-keys the shared env layer onto the
        governing team's own row -- see the team-env block below -- instead
        of leaving the shared layer keyed on the run owner's personal
        shared-env hook answer."""
        self._mcp_oauth_diagnostics = []
        self._reset_mcp_config_load_cache_state()

        # Resolved before the guarded region below: that region reports
        # "every selected server is unavailable", which is the wrong answer
        # for "the scope could not be resolved". The typed error is what
        # survives the tool-creator frame -- an untyped one is dropped there
        # with a WARNING and no tool set at all.
        from ..services.connector_team_scope import resolve_team_connector_ids_or_raise

        team_mcp_ids = frozenset(
            resolve_team_connector_ids_or_raise(
                self.db, team_id=self._connector_team_id, log_subject=self._user_id
            )["mcp"]
        )

        try:
            from ..services.mcp_runtime import (
                load_shared_env_overrides,
                load_team_env_overrides,
                load_user_env_overrides,
                load_user_env_sources,
                team_env_hook_installed,
                warn_team_env_hook_missing_once,
            )

            servers = self._visible_mcp_server_query(team_mcp_ids).all()
            logger.info(
                "Found %s visible MCP servers for user %s (connector_team_id=%s)",
                len(servers),
                self._user_id,
                self._connector_team_id,
            )

            # Prefetch shared runtime state once before entering the isolated
            # per-server formatter.
            user_env_by_id = load_user_env_overrides(self.db, self._user_id)
            shared_env_by_id = load_shared_env_overrides(self.db, self._user_id)
            env_source_by_id = load_user_env_sources(self.db, self._user_id)

            # Re-key the shared env layer, for team-owned ids only, onto the
            # governing team's own row -- never the run owner's team, and
            # never any other team's. The outer condition is the scope test:
            # a run with no governing team, or one whose governing team owns
            # nothing visible here, pays no team query at all. The inner
            # condition then decides between re-keying and reporting that
            # the credential-side hook was never installed -- the shared
            # layer stays user-keyed in that state, which is exactly the
            # cross-team influence this block exists to remove.
            if self._connector_team_id is not None and team_mcp_ids:
                if not team_env_hook_installed():
                    warn_team_env_hook_missing_once(
                        team_id=self._connector_team_id,
                        connector_count=len(team_mcp_ids),
                    )
                else:
                    team_env_by_id = load_team_env_overrides(
                        self.db, self._connector_team_id
                    )
                    # Both copies are load-bearing, for different reasons.
                    # shared_env_by_id: load_shared_env_overrides hands back the
                    # installing application's own object unwrapped, so writing
                    # into it below would mutate the hook's dict.
                    # env_source_by_id: load_user_env_sources always builds a
                    # fresh dict, so no hook holds it -- but the degrade below
                    # removes entries from it in place (`del ...[server_id]`),
                    # and the loader's result is not this block's to shrink.
                    # Shallow outer copies suffice for both: every downstream
                    # consumer only reads the inner per-server values --
                    # resolve_stdio_env's lookups and this file's own per-server
                    # env merge in _build_mcp_server_config's stdio branch (an
                    # unconditional-overwrite `{**a, **b}` merge) -- and neither
                    # assigns back into either map.
                    shared_env_by_id = dict(shared_env_by_id)
                    env_source_by_id = dict(env_source_by_id)
                    for server_id in team_mcp_ids:
                        if server_id in team_env_by_id:
                            shared_env_by_id[server_id] = team_env_by_id[server_id]
                        else:
                            shared_env_by_id.pop(server_id, None)
                            if env_source_by_id.get(server_id) == "shared":
                                # Unconditional, not only when the pop above
                                # removed something: firing it only on an actual
                                # removal would make a non-member runner's
                                # outcome depend on whether the RUNNER'S OWN team
                                # happens to hold a row for this server -- the
                                # exact cross-team influence this seam forbids,
                                # re-entering through a side door. Unconditional,
                                # the outcome depends on exactly three inputs:
                                # the governing team's row, the runner's own key,
                                # and this pick -- the runner's own team's rows
                                # never enter the computation.
                                del env_source_by_id[server_id]
        except ConnectorRuntimeError:
            raise
        except Exception as error:
            logger.warning(
                "Failed to scan MCP server configs with %s",
                type(error).__name__,
            )
            raise MCPConfigLoadError() from error

        configs = [
            await self._load_mcp_server_config(
                server=server,
                user_env_by_id=user_env_by_id,
                shared_env_by_id=shared_env_by_id,
                env_source_by_id=env_source_by_id,
            )
            for server in servers
        ]
        logger.info("Loaded %s MCP server configurations", len(configs))
        return configs

    def get_custom_api_configs(self) -> List[Dict[str, Any]]:
        """Get custom API configurations."""
        snapshot = self._factory_runtime_snapshot
        if snapshot is not None and snapshot.plan.load_custom_api:
            return snapshot.custom_api_configs
        if not self._user_id:
            return []

        # Resolved before the guarded region below: that region reports
        # "every selected API is unavailable", which is the wrong answer
        # for "the scope could not be resolved". The typed error is what
        # survives the tool-creator frame -- an untyped one is dropped there
        # with an ERROR and no tool set at all.
        from ..services.connector_team_scope import resolve_team_connector_ids_or_raise

        team_api_ids = frozenset(
            resolve_team_connector_ids_or_raise(
                self.db, team_id=self._connector_team_id, log_subject=self._user_id
            )["custom_api"]
        )

        # Fails closed on the query itself, matching the MCP twin
        # (_load_mcp_server_configs): a genuine DB/query failure here is not
        # "the caller selected zero custom APIs", so it must not degrade to
        # an empty list. The MCP twin raises its own MCPConfigLoadError,
        # whose summaries/failure-policy machinery (enforce_mcp_failure_policy)
        # has no custom-API equivalent and would be new surface to build for
        # this fix; ConnectorRuntimeError is the typed error this same
        # function already raises for the team-scope-resolution failure
        # above, is already propagated by every frame between here and the
        # tool creator (create_db_custom_api_tools's own
        # ``except ConnectorRuntimeError: raise``, then the registry loop's),
        # and needs no new plumbing. Per-row config-building below keeps its
        # existing swallow behavior unchanged -- neither side treats an
        # individual row's build failure as fatal -- but the granularity
        # still differs and is a remaining difference, out of scope here:
        # the MCP twin isolates a failed row into an "unavailable" config
        # and keeps loading the others, while this loop's guard spans the
        # whole loop, so one bad row still discards every row.
        try:
            apis = _visible_custom_api_query(
                self.db,
                owner_user_id=int(self._user_id),
                team_api_ids=team_api_ids,
            ).all()
        except ConnectorRuntimeError:
            raise
        except Exception as error:
            logger.warning(
                "Failed to scan Custom API configs with %s",
                type(error).__name__,
            )
            raise ConnectorRuntimeError(
                ERROR_CONNECTOR_RUNTIME_UNAVAILABLE,
                "Custom API configurations could not be loaded.",
                details={"reason": "custom_api_config_load_failed"},
                status_code=503,
            ) from error

        try:
            custom_api_configs = []
            for api in apis:
                custom_api_configs.append(
                    _custom_api_config_from_model(
                        api,
                        self._get_connector_runtime_for("custom_api", int(api.id)),
                    )
                )
            return custom_api_configs

        except ConnectorRuntimeError:
            raise
        except Exception as e:
            logger.error(
                f"Failed to get Custom API configs from database: {e}", exc_info=True
            )
            return []
