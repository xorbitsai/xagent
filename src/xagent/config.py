"""Core configuration for xagent.

Provides unified configuration for all paths and directories that can be used
by both core and web modules without creating circular dependencies.

All paths support environment variable overrides for portable deployments.

Environment Variable Naming Convention:
    Most config variables use the XAGENT_* prefix for consistency.
    Exceptions (without XAGENT_ prefix) are kept for backward compatibility:
    - SANDBOX_*: Sandbox container configuration (predates this module)
    - BOXLITE_HOME_DIR: Boxlite sandbox home directory
    - DATABASE_URL: Standard database connection URL
    - LANCEDB_PATH: LanceDB database path

Future enhancement: Consider migrating to pydantic-settings for more robust
configuration management with validation, type safety, and better structure.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit, urlunsplit

logger = logging.getLogger(__name__)

# Environment variable names
UPLOADS_DIR = "XAGENT_UPLOADS_DIR"
WEB_DIR = "XAGENT_WEB_DIR"
FRONTEND_DIST_DIR = "XAGENT_FRONTEND_DIST_DIR"
EXTERNAL_UPLOAD_DIRS = "XAGENT_EXTERNAL_UPLOAD_DIRS"
EXTERNAL_SKILLS_LIBRARY_DIRS = "XAGENT_EXTERNAL_SKILLS_LIBRARY_DIRS"
AGENT_RUNTIME = "XAGENT_AGENT_RUNTIME"
INTERACTION_PROTOCOL_MODE = "XAGENT_INTERACTION_PROTOCOL_MODE"
INTERACTION_NATIVE_SOURCES = "XAGENT_INTERACTION_NATIVE_SOURCES"
TASK_LEASE_TTL_SECONDS = "XAGENT_TASK_LEASE_TTL_SECONDS"
TASK_LEASE_HEARTBEAT_SECONDS = "XAGENT_TASK_LEASE_HEARTBEAT_SECONDS"
TASK_LEASE_RECOVERY_INTERVAL_SECONDS = "XAGENT_TASK_LEASE_RECOVERY_INTERVAL_SECONDS"
TASK_LEASE_RECOVERY_BATCH_SIZE = "XAGENT_TASK_LEASE_RECOVERY_BATCH_SIZE"
UPLOADED_FILE_RECOVERY_INTERVAL_SECONDS = (
    "XAGENT_UPLOADED_FILE_RECOVERY_INTERVAL_SECONDS"
)
UPLOADED_FILE_RECOVERY_STALE_SECONDS = "XAGENT_UPLOADED_FILE_RECOVERY_STALE_SECONDS"
UPLOADED_FILE_RECOVERY_BATCH_SIZE = "XAGENT_UPLOADED_FILE_RECOVERY_BATCH_SIZE"
TEMP_FILE_CLEANUP_SHUTDOWN_TIMEOUT_SECONDS = (
    "XAGENT_TEMP_FILE_CLEANUP_SHUTDOWN_TIMEOUT_SECONDS"
)
STORAGE_ROOT = "XAGENT_STORAGE_ROOT"
NATIVE_BROWSER_ENABLED = "XAGENT_NATIVE_BROWSER_ENABLED"
NATIVE_BROWSER_APP_NAME = "XAGENT_NATIVE_BROWSER_APP_NAME"
BROWSER_TOOL_DEFAULT_LOCALE = "XAGENT_BROWSER_TOOL_DEFAULT_LOCALE"
BROWSER_TOOL_DEFAULT_TIMEZONE = "XAGENT_BROWSER_TOOL_DEFAULT_TIMEZONE"
BROWSER_CUA_DRIVER_COMMAND = "XAGENT_BROWSER_CUA_DRIVER_COMMAND"
BROWSER_CUA_DRIVER_SOCKET = "XAGENT_BROWSER_CUA_DRIVER_SOCKET"
BROWSER_CUA_DRIVER_TIMEOUT_SECONDS = "XAGENT_BROWSER_CUA_DRIVER_TIMEOUT_SECONDS"
BROWSER_CUA_DRIVER_MAX_ELEMENTS = "XAGENT_BROWSER_CUA_DRIVER_MAX_ELEMENTS"
SUPPORTED_NATIVE_BROWSER_APP_NAMES = frozenset(
    {
        "Brave Browser",
        "Google Chrome",
        "Google Chrome Canary",
        "Chromium",
        "Microsoft Edge",
        "Vivaldi",
    }
)
_NATIVE_BROWSER_APP_NAMES_BY_CASEFOLD = {
    name.casefold(): name for name in SUPPORTED_NATIVE_BROWSER_APP_NAMES
}
MAX_UPLOAD_SIZE = "XAGENT_MAX_UPLOAD_SIZE"
FILE_STORAGE_URI = "XAGENT_FILE_STORAGE_URI"
FILE_STORAGE_OPTIONS = "XAGENT_FILE_STORAGE_OPTIONS"
FILE_MATERIALIZE_DIR = "XAGENT_FILE_MATERIALIZE_DIR"
PREVIEW_TMP_DIR = "XAGENT_PREVIEW_TMP_DIR"
FILE_STORAGE_STARTUP_SYNC_ENABLED = "XAGENT_FILE_STORAGE_STARTUP_SYNC_ENABLED"
FILE_DELIVERY_REDIRECT_ENABLED = "XAGENT_FILE_DELIVERY_REDIRECT_ENABLED"
FILE_DELIVERY_SIGNED_URL_TTL_SECONDS = "XAGENT_FILE_DELIVERY_SIGNED_URL_TTL_SECONDS"
FILE_DELIVERY_ACCEL_REDIRECT_ENABLED = "XAGENT_FILE_DELIVERY_ACCEL_REDIRECT_ENABLED"
FILE_DELIVERY_ACCEL_REDIRECT_PREFIX = "XAGENT_FILE_DELIVERY_ACCEL_REDIRECT_PREFIX"
FILE_STREAM_TICKET_TTL_SECONDS = "XAGENT_FILE_STREAM_TICKET_TTL_SECONDS"
SANDBOX_IMAGE = "SANDBOX_IMAGE"
LANCEDB_PATH = "LANCEDB_PATH"
KB_COLLECTIONS_TIMEOUT_SECONDS = "XAGENT_KB_COLLECTIONS_TIMEOUT_SECONDS"
KB_SEARCH_TIMEOUT_SECONDS = "XAGENT_KB_SEARCH_TIMEOUT_SECONDS"
GOOGLE_DRIVE_DOWNLOAD_TIMEOUT_SECONDS = "XAGENT_GOOGLE_DRIVE_DOWNLOAD_TIMEOUT_SECONDS"
DEEPDOC_XINFERENCE_URL = "XAGENT_DEEPDOC_XINFERENCE_URL"
DEEPDOC_XINFERENCE_API_KEY = "XAGENT_DEEPDOC_XINFERENCE_API_KEY"
DEEPDOC_XINFERENCE_TIMEOUT_SECONDS = "XAGENT_DEEPDOC_XINFERENCE_TIMEOUT_SECONDS"
DEEPDOC_XINFERENCE_MODEL_UID = "XAGENT_DEEPDOC_XINFERENCE_MODEL_UID"
DEEPDOC_XINFERENCE_USERNAME = "XAGENT_DEEPDOC_XINFERENCE_USERNAME"
DEEPDOC_XINFERENCE_PASSWORD = "XAGENT_DEEPDOC_XINFERENCE_PASSWORD"
DATABASE_URL = "DATABASE_URL"
DB_POOL_SIZE = "XAGENT_DB_POOL_SIZE"
DB_MAX_OVERFLOW = "XAGENT_DB_MAX_OVERFLOW"
DB_POOL_TIMEOUT_SECONDS = "XAGENT_DB_POOL_TIMEOUT_SECONDS"
MCP_TOOL_INIT_TIMEOUT_SECONDS = "XAGENT_MCP_TOOL_INIT_TIMEOUT_SECONDS"
SANDBOX_CPUS = "SANDBOX_CPUS"
SANDBOX_MEMORY = "SANDBOX_MEMORY"
SANDBOX_ENV = "SANDBOX_ENV"
SANDBOX_VOLUMES = "SANDBOX_VOLUMES"
# Set only inside the sandbox tool runner, which has no database credentials.
SANDBOX_TOOL_RUNNER = "XAGENT_SANDBOX_TOOL_RUNNER"
SANDBOX_HOST_PROJECT_ROOT = "XAGENT_SANDBOX_HOST_PROJECT_ROOT"
SANDBOX_HOST_STORAGE_ROOT = "XAGENT_SANDBOX_HOST_STORAGE_ROOT"
SANDBOX_MAX_CONCURRENCY = "XAGENT_SANDBOX_MAX_CONCURRENCY"
SANDBOX_IDLE_TTL = "XAGENT_SANDBOX_IDLE_TTL"
SANDBOX_SWEEP_INTERVAL = "XAGENT_SANDBOX_SWEEP_INTERVAL"
SANDBOX_MAX_CONTAINERS = "XAGENT_SANDBOX_MAX_CONTAINERS"
SANDBOX_ALLOW_LOCAL_FALLBACK_ON_CAPACITY = (
    "XAGENT_SANDBOX_ALLOW_LOCAL_FALLBACK_ON_CAPACITY"
)
SANDBOX_NAMESPACE = "XAGENT_SANDBOX_NAMESPACE"
BOXLITE_HOME_DIR = "BOXLITE_HOME_DIR"
WEB_SEARCH_PROVIDER = "XAGENT_WEB_SEARCH_PROVIDER"
WEB_CRAWL_TLS_IMPERSONATE = "XAGENT_WEB_CRAWL_TLS_IMPERSONATE"
TOOL_PARALLEL_ENABLED = "XAGENT_TOOL_PARALLEL_ENABLED"
TOOL_MAX_CONCURRENCY = "XAGENT_TOOL_MAX_CONCURRENCY"
TASK_RUNTIME_HOOK_MAX_WORKERS = "XAGENT_TASK_RUNTIME_HOOK_MAX_WORKERS"
TASK_RUNTIME_HOOK_QUEUE_TIMEOUT_SECONDS = (
    "XAGENT_TASK_RUNTIME_HOOK_QUEUE_TIMEOUT_SECONDS"
)
CHECKPOINT_ENCODING_V2 = "XAGENT_CHECKPOINT_ENCODING_V2"
CHECKPOINT_HISTORY_LIMIT = "XAGENT_CHECKPOINT_HISTORY_LIMIT"
COMPACT_THRESHOLD_RATIO = "XAGENT_COMPACT_THRESHOLD_RATIO"
COMPACT_THRESHOLD_DEFAULT = "XAGENT_COMPACT_THRESHOLD_DEFAULT"
REDIS_URL = "XAGENT_REDIS_URL"
HOT_PATH_CACHE_ENABLED = "XAGENT_HOT_PATH_CACHE_ENABLED"
HOT_PATH_CACHE_TTL_SECONDS = "XAGENT_HOT_PATH_CACHE_TTL_SECONDS"
HOT_PATH_TASK_CACHE_TTL_SECONDS = "XAGENT_HOT_PATH_TASK_CACHE_TTL_SECONDS"
CELERY_ENABLED = "XAGENT_CELERY_ENABLED"
CELERY_BROKER_URL = "XAGENT_CELERY_BROKER_URL"
CELERY_RESULT_BACKEND = "XAGENT_CELERY_RESULT_BACKEND"
BACKGROUND_JOB_VISIBILITY_TIMEOUT_SECONDS = (
    "XAGENT_BACKGROUND_JOB_VISIBILITY_TIMEOUT_SECONDS"
)
BACKGROUND_JOB_MAX_RETRIES = "XAGENT_BACKGROUND_JOB_MAX_RETRIES"
BACKGROUND_JOB_STALE_SECONDS = "XAGENT_BACKGROUND_JOB_STALE_SECONDS"
BACKGROUND_JOB_SWEEP_INTERVAL_SECONDS = "XAGENT_BACKGROUND_JOB_SWEEP_INTERVAL_SECONDS"
TASKLESS_UPLOAD_TTL_SECONDS = "XAGENT_TASKLESS_UPLOAD_TTL_SECONDS"
DETACHED_UPLOAD_RETENTION_SECONDS = "XAGENT_DETACHED_UPLOAD_RETENTION_SECONDS"
ORPHAN_UPLOAD_SWEEP_INTERVAL_SECONDS = "XAGENT_ORPHAN_UPLOAD_SWEEP_INTERVAL_SECONDS"
WORKFORCE_PREVIEW_RUN_STALE_SECONDS = "XAGENT_WORKFORCE_PREVIEW_RUN_STALE_SECONDS"
TRIGGER_DISPATCHER_ENABLED = "XAGENT_TRIGGER_DISPATCHER_ENABLED"
TRIGGER_DISPATCHER_INTERVAL_SECONDS = "XAGENT_TRIGGER_DISPATCHER_INTERVAL_SECONDS"
TRIGGER_DISPATCHER_BATCH_SIZE = "XAGENT_TRIGGER_DISPATCHER_BATCH_SIZE"
TRIGGER_DISPATCHER_STARTUP_JITTER_SECONDS = (
    "XAGENT_TRIGGER_DISPATCHER_STARTUP_JITTER_SECONDS"
)
TRIGGER_CALLBACK_RATE_LIMIT = "XAGENT_TRIGGER_CALLBACK_RATE_LIMIT"
TRIGGER_CALLBACK_IP_RATE_LIMIT = "XAGENT_TRIGGER_CALLBACK_IP_RATE_LIMIT"
TRIGGER_CRUD_RATE_LIMIT = "XAGENT_TRIGGER_CRUD_RATE_LIMIT"
TRUSTED_PROXY_HOPS = "XAGENT_TRUSTED_PROXY_HOPS"
# Public share-channel abuse controls (#973). Each is a rate string in the
# ``limits`` notation, e.g. "60/minute" or "500/day".
SHARE_AUTH_RATE_LIMIT = "XAGENT_SHARE_AUTH_RATE_LIMIT"
SHARE_AUTH_IP_RATE_LIMIT = "XAGENT_SHARE_AUTH_IP_RATE_LIMIT"
SHARE_TASK_CREATE_RATE_LIMIT = "XAGENT_SHARE_TASK_CREATE_RATE_LIMIT"
SHARE_TASK_CREATE_TOKEN_RATE_LIMIT = "XAGENT_SHARE_TASK_CREATE_TOKEN_RATE_LIMIT"
SHARE_WS_TURN_RATE_LIMIT = "XAGENT_SHARE_WS_TURN_RATE_LIMIT"
SHARE_WS_CONNECT_IP_RATE_LIMIT = "XAGENT_SHARE_WS_CONNECT_IP_RATE_LIMIT"
SHARE_UPLOAD_RATE_LIMIT = "XAGENT_SHARE_UPLOAD_RATE_LIMIT"
WIDGET_UPLOAD_RATE_LIMIT = "XAGENT_WIDGET_UPLOAD_RATE_LIMIT"
WIDGET_UPLOAD_IP_RATE_LIMIT = "XAGENT_WIDGET_UPLOAD_IP_RATE_LIMIT"
WIDGET_WS_CONNECT_IP_RATE_LIMIT = "XAGENT_WIDGET_WS_CONNECT_IP_RATE_LIMIT"
WIDGET_WS_TURN_IP_RATE_LIMIT = "XAGENT_WIDGET_WS_TURN_IP_RATE_LIMIT"
WIDGET_WS_TURN_RATE_LIMIT = "XAGENT_WIDGET_WS_TURN_RATE_LIMIT"
WIDGET_AUTH_RATE_LIMIT = "XAGENT_WIDGET_AUTH_RATE_LIMIT"
WIDGET_AUTH_IP_RATE_LIMIT = "XAGENT_WIDGET_AUTH_IP_RATE_LIMIT"
WIDGET_TASK_CREATE_RATE_LIMIT = "XAGENT_WIDGET_TASK_CREATE_RATE_LIMIT"
WIDGET_TASK_CREATE_IP_RATE_LIMIT = "XAGENT_WIDGET_TASK_CREATE_IP_RATE_LIMIT"
WIDGET_RUN_QUOTA = "XAGENT_WIDGET_RUN_QUOTA"
WIDGET_RUN_IP_QUOTA = "XAGENT_WIDGET_RUN_IP_QUOTA"
SHARE_RUN_QUOTA = "XAGENT_SHARE_RUN_QUOTA"
SHARE_RUN_GUEST_QUOTA = "XAGENT_SHARE_RUN_GUEST_QUOTA"
GMAIL_PUBSUB_PROJECT_ID = "XAGENT_GMAIL_PUBSUB_PROJECT_ID"
GMAIL_PUBSUB_TOPIC_PREFIX = "XAGENT_GMAIL_PUBSUB_TOPIC_PREFIX"
GMAIL_PUBSUB_SUBSCRIPTION_PREFIX = "XAGENT_GMAIL_PUBSUB_SUBSCRIPTION_PREFIX"
GMAIL_PUBSUB_PUSH_SERVICE_ACCOUNT = "XAGENT_GMAIL_PUBSUB_PUSH_SERVICE_ACCOUNT"
GMAIL_PUBSUB_TRANSPORT = "XAGENT_GMAIL_PUBSUB_TRANSPORT"
GMAIL_REGISTRATION_TIMEOUT_SECONDS = "XAGENT_GMAIL_REGISTRATION_TIMEOUT_SECONDS"
PUBLIC_API_BASE_URL = "XAGENT_PUBLIC_API_BASE_URL"
S2S_API_BASE_URL = "XAGENT_S2S_API_BASE_URL"
TRIGGER_CALLBACK_BASE_URL = "XAGENT_TRIGGER_CALLBACK_BASE_URL"
GMAIL_WATCH_ENABLED = "XAGENT_GMAIL_WATCH_ENABLED"
GMAIL_WATCH_RENEWAL_INTERVAL_SECONDS = "XAGENT_GMAIL_WATCH_RENEWAL_INTERVAL_SECONDS"
GMAIL_WATCH_RENEWAL_LEAD_SECONDS = "XAGENT_GMAIL_WATCH_RENEWAL_LEAD_SECONDS"
PASSWORD_RESET_EXPIRE_MINUTES = "XAGENT_PASSWORD_RESET_EXPIRE_MINUTES"
APP_BASE_URL = "XAGENT_APP_BASE_URL"
SLACK_CLIENT_ID = "XAGENT_SLACK_CLIENT_ID"
SLACK_CLIENT_SECRET = "XAGENT_SLACK_CLIENT_SECRET"
SLACK_APP_TOKEN = "XAGENT_SLACK_APP_TOKEN"
SLACK_REDIRECT_URI = "XAGENT_SLACK_REDIRECT_URI"
SMTP_HOST = "XAGENT_SMTP_HOST"
SMTP_PORT = "XAGENT_SMTP_PORT"
SMTP_USERNAME = "XAGENT_SMTP_USERNAME"
SMTP_PASSWORD = "XAGENT_SMTP_PASSWORD"
SMTP_USE_TLS = "XAGENT_SMTP_USE_TLS"
SMTP_USE_SSL = "XAGENT_SMTP_USE_SSL"
SMTP_FROM_EMAIL = "XAGENT_SMTP_FROM_EMAIL"
SMTP_FROM_NAME = "XAGENT_SMTP_FROM_NAME"
GOOGLE_OIDC_CLIENT_ID = "XAGENT_GOOGLE_OIDC_CLIENT_ID"
GOOGLE_OIDC_CLIENT_SECRET = "XAGENT_GOOGLE_OIDC_CLIENT_SECRET"
GOOGLE_OIDC_REDIRECT_URI = "XAGENT_GOOGLE_OIDC_REDIRECT_URI"
FRONTEND_URL = "XAGENT_FRONTEND_URL"
OIDC_LOGIN_TTL_SECONDS = "XAGENT_OIDC_LOGIN_TTL_SECONDS"
OIDC_EXCHANGE_TTL_SECONDS = "XAGENT_OIDC_EXCHANGE_TTL_SECONDS"
SESSION_SECRET = "XAGENT_SESSION_SECRET"
OPENROUTER_OFFICIAL_PROVIDERS_ONLY = "XAGENT_OPENROUTER_OFFICIAL_PROVIDERS_ONLY"
XROUTER_EXCLUDED_MODELS = "XAGENT_XROUTER_EXCLUDED_MODELS"
MCP_OAUTH_ALLOW_PRIVATE_HOSTS = "XAGENT_MCP_OAUTH_ALLOW_PRIVATE_HOSTS"
MCP_OAUTH_PROXY_URL = "XAGENT_MCP_OAUTH_PROXY_URL"
TRUSTED_EGRESS_PROXY = "XAGENT_TRUSTED_EGRESS_PROXY"

TOOL_MAX_OUTPUT_LENGTH = "XAGENT_TOOL_MAX_OUTPUT_LENGTH"
TOOL_MAX_RECURSION_DEPTH = "XAGENT_TOOL_MAX_RECURSION_DEPTH"
TOOL_MAX_FIELD_COUNT = "XAGENT_TOOL_MAX_FIELD_COUNT"
MAX_TRACE_PAYLOAD_BYTES = "XAGENT_MAX_TRACE_PAYLOAD_BYTES"

WEB_SEARCH_PROVIDERS = {"auto", "google", "tavily", "exa", "zhipu"}


def get_agent_runtime() -> Literal["v1", "v2"]:
    """Get the agent execution runtime version.

    Priority:
        1. XAGENT_AGENT_RUNTIME environment variable
        2. "v1" default for compatibility

    Returns:
        "v1" or "v2"
    """
    runtime = os.getenv(AGENT_RUNTIME, "v1").strip().lower()
    if runtime == "v1":
        return "v1"
    if runtime == "v2":
        return "v2"
    logger.warning("Invalid %s=%r; falling back to v1", AGENT_RUNTIME, runtime)
    return "v1"


def get_interaction_protocol_mode() -> str:
    """Raw XAGENT_INTERACTION_PROTOCOL_MODE reading: stripped and lowercased.

    Returns the env value normalized for whitespace and case, or "legacy" if
    the variable is unset or blank. Does not check that the result is one
    of the three valid modes -- validating the parsed value and building a
    policy out of it belongs to the interaction rollout policy owner
    (``web/services/interaction_rollout.py``), not to this module. Unlike
    ``get_agent_runtime`` above, an unrecognized value is not this
    function's problem to warn about or fall back from.
    """
    value = os.getenv(INTERACTION_PROTOCOL_MODE)
    if value is None or not value.strip():
        return "legacy"
    return value.strip().lower()


def get_interaction_native_sources() -> list[str]:
    """Raw XAGENT_INTERACTION_NATIVE_SOURCES reading: a normalized token list.

    Comma-splits the env value, strips and lowercases each token, and skips
    blank tokens -- the same shape ``get_external_upload_dirs`` below uses
    for its own comma-separated list. Duplicate tokens are preserved and
    tokens are not checked against any vocabulary here: deduplication and
    vocabulary validation need to raise two distinguishable errors, and
    producing those belongs to the interaction rollout policy owner, not to
    this module.
    """
    raw = os.getenv(INTERACTION_NATIVE_SOURCES, "")
    if not raw:
        return []

    result: list[str] = []
    for token in raw.split(","):
        token = token.strip().lower()
        if token:
            result.append(token)
    return result


def get_agent_pattern_for_execution_mode(execution_mode: str | None) -> str:
    """Map UI execution mode to agent pattern name.

    Supported modes:
        flash: strict single call
        balanced: ReAct
        think: DAG plan-execute
        auto: LLM-selected final answer / ReAct / DAG
    """
    mode = (execution_mode or "").strip().lower()
    mapping = {
        "flash": "single_call",
        "balanced": "react",
        "think": "dag_plan_execute",
        "auto": "auto",
    }
    return mapping.get(mode, "react")


def get_default_task_execution_mode(
    *,
    agent_id: object | None = None,
    agent_runtime: str | None = None,
) -> str:
    """Get the default UI execution mode for a newly-created task.

    Standalone tasks default to auto so simple prompts can answer directly while
    complex prompts can still route into ReAct or DAG. Explicit v1 deployments
    keep the legacy standalone DAG default for compatibility. Agent Builder
    tasks keep balanced because the agent's explicit tool/KB setup is usually
    better served by ReAct.
    """
    if agent_id is not None:
        return "balanced"

    if agent_runtime is not None:
        runtime = agent_runtime.strip().lower()
    else:
        runtime = (os.getenv(AGENT_RUNTIME) or "").strip().lower()

    if runtime == "v1":
        return "think"
    return "auto"


def get_task_lease_ttl_seconds() -> int:
    """Get task execution lease TTL in seconds.

    Priority:
        1. XAGENT_TASK_LEASE_TTL_SECONDS environment variable
        2. 60 seconds
    """
    value = os.getenv(TASK_LEASE_TTL_SECONDS, "60")
    try:
        seconds = int(value)
    except ValueError:
        logger.warning(
            "Invalid %s=%r; falling back to 60",
            TASK_LEASE_TTL_SECONDS,
            value,
        )
        return 60
    if seconds < 10:
        logger.warning(
            "%s=%r is too small; falling back to 60",
            TASK_LEASE_TTL_SECONDS,
            value,
        )
        return 60
    return seconds


def get_task_lease_heartbeat_seconds() -> int:
    """Get task execution lease heartbeat interval in seconds.

    Priority:
        1. XAGENT_TASK_LEASE_HEARTBEAT_SECONDS environment variable
        2. One third of the lease TTL, at least 5 seconds
    """
    default = max(5, get_task_lease_ttl_seconds() // 3)
    value = os.getenv(TASK_LEASE_HEARTBEAT_SECONDS)
    if value is None:
        return default
    try:
        seconds = int(value)
    except ValueError:
        logger.warning(
            "Invalid %s=%r; falling back to %s",
            TASK_LEASE_HEARTBEAT_SECONDS,
            value,
            default,
        )
        return default
    if seconds < 1:
        logger.warning(
            "%s=%r is too small; falling back to %s",
            TASK_LEASE_HEARTBEAT_SECONDS,
            value,
            default,
        )
        return default
    return min(seconds, max(1, get_task_lease_ttl_seconds() - 1))


def get_task_lease_recovery_interval_seconds() -> int:
    """Get the interval between automatic expired-lease recovery scans."""

    default = max(5, get_task_lease_ttl_seconds() // 3)
    return _get_positive_int_env(
        TASK_LEASE_RECOVERY_INTERVAL_SECONDS,
        default,
    )


def get_task_lease_recovery_batch_size() -> int:
    """Get the maximum number of expired leases scanned per recovery batch."""

    return _get_positive_int_env(TASK_LEASE_RECOVERY_BATCH_SIZE, 100)


def get_uploaded_file_recovery_interval_seconds() -> int:
    """Get the interval between stale file-compensation recovery scans."""

    return _get_positive_int_env(UPLOADED_FILE_RECOVERY_INTERVAL_SECONDS, 60)


def get_uploaded_file_recovery_stale_seconds() -> int:
    """Get the minimum age of a compensation claim eligible for recovery."""

    return _get_positive_int_env(UPLOADED_FILE_RECOVERY_STALE_SECONDS, 300)


def get_uploaded_file_recovery_batch_size() -> int:
    """Get the maximum file-compensation claims examined per polling tick."""

    return _get_positive_int_env(UPLOADED_FILE_RECOVERY_BATCH_SIZE, 100)


def get_temp_file_cleanup_shutdown_timeout_seconds() -> int:
    """Get how long shutdown waits for the orphaned temp-file sweep to unwind.

    At shutdown the sweep's cooperative stop flag is set and the handler waits
    up to this long for the walk to reach its next directory boundary and exit.
    This bounds only the wait, not the walk: the executor thread is not
    cancellable, so a long overrun is ultimately joined by asyncio.run()'s
    teardown. Operators on very large uploads trees may want a larger value.

    Unlike XAGENT_MCP_TOOL_INIT_TIMEOUT_SECONDS and similar getters in this
    module, "0" is not treated as "disable the timeout" here: it already has
    a distinct, meaningful value for a wait bound -- "don't wait at all" --
    which is the opposite of disabling it (waiting forever). So "0" falls
    back to the default like any other invalid value instead.

    Priority:
        1. XAGENT_TEMP_FILE_CLEANUP_SHUTDOWN_TIMEOUT_SECONDS environment variable
        2. Default of 10 seconds

    Returns:
        Shutdown grace period in seconds
    """

    return _get_positive_int_env(TEMP_FILE_CLEANUP_SHUTDOWN_TIMEOUT_SECONDS, 10)


def _get_positive_int_env(env_var: str, default: int, *, minimum: int = 1) -> int:
    value = os.getenv(env_var)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError:
        logger.warning("Invalid %s=%r; falling back to %s", env_var, value, default)
        return default
    if parsed < minimum:
        logger.warning("Invalid %s=%r; falling back to %s", env_var, value, default)
        return default
    return parsed


def _get_bool_env(env_var: str, default: bool) -> bool:
    value = os.getenv(env_var)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _normalized_env_url(env_var: str) -> str | None:
    """Return the env var's value normalized as a base URL.

    Strips surrounding whitespace and trailing slashes; unset or blank
    values normalize to None.
    """
    value = (os.getenv(env_var) or "").strip()
    return value.rstrip("/") or None


def _normalized_http_env_url(env_var: str) -> str | None:
    """Return a normalized HTTP(S) base URL or reject invalid configuration.

    Server-to-server URLs are copied into externally consumed callback,
    audience, and Agent Card fields. Validating them at the configuration
    boundary produces an actionable error before those integrations receive a
    malformed endpoint.
    """
    value = _normalized_env_url(env_var)
    if value is None:
        return None
    parts = urlsplit(value)
    if (
        parts.scheme not in {"http", "https"}
        or not parts.netloc
        or parts.query
        or parts.fragment
    ):
        raise ValueError(
            f"Invalid {env_var} value: {value!r}. "
            "Expected an absolute http:// or https:// URL without a query or fragment."
        )
    return value


def _reject_url_userinfo(env_var: str, value: str | None) -> str | None:
    """Reject a URL carrying ``user:password@``, returning it otherwise.

    httpx does not send URL userinfo as Basic auth, so credentials placed there
    authenticate nothing -- but ``httpx.HTTPStatusError`` renders the full URL
    unredacted, and callers put that string into log messages, so a password
    there ends up in plaintext in the application log.

    Kept separate from :func:`_normalized_http_env_url` on purpose. That helper
    has pre-existing callers whose own call sites do not catch ``ValueError``
    and rely on ``or``-chained fallbacks, so rejecting inside it would turn a
    working (if ill-advised) configuration into a runtime failure for them.
    """
    if value is not None and "@" in urlsplit(value).netloc:
        raise ValueError(
            f"Invalid {env_var} value: credentials embedded in the URL are not "
            "supported, because error messages built from it are logged. "
            "Remove the 'user:password@' part and configure the credential "
            "separately."
        )
    return value


def get_password_reset_expire_minutes() -> int:
    """Return the password reset token expiry window in minutes."""
    return _get_positive_int_env(PASSWORD_RESET_EXPIRE_MINUTES, 30)


def get_app_base_url() -> str | None:
    """Return the trusted frontend base URL used in email links."""
    return _normalized_env_url(APP_BASE_URL)


def get_slack_client_id() -> str | None:
    """Return the Slack app client ID used by the workspace OAuth flow."""
    value = (os.getenv(SLACK_CLIENT_ID) or "").strip()
    return value or None


def get_slack_client_secret() -> str | None:
    """Return the Slack app client secret used to exchange OAuth codes."""
    value = (os.getenv(SLACK_CLIENT_SECRET) or "").strip()
    return value or None


def get_slack_app_token() -> str | None:
    """Return the shared Slack Socket Mode app-level token."""
    value = (os.getenv(SLACK_APP_TOKEN) or "").strip()
    return value or None


def get_slack_oauth_redirect_uri() -> str | None:
    """Return the externally reachable Slack OAuth callback URL.

    An explicit redirect URI wins. Otherwise derive the callback from the
    public backend base URL so all advertised provider callbacks share the
    same deployment-level source of truth.
    """
    explicit = _normalized_env_url(SLACK_REDIRECT_URI)
    if explicit is not None:
        return explicit
    public_base_url = get_public_api_base_url()
    if public_base_url is None:
        return None
    return f"{public_base_url}/api/channels/slack/oauth/callback"


def get_smtp_host() -> str:
    return os.getenv(SMTP_HOST, "").strip()


def get_smtp_port() -> int:
    return _get_positive_int_env(SMTP_PORT, 587)


def get_smtp_username() -> str:
    return os.getenv(SMTP_USERNAME, "").strip()


def get_smtp_password() -> str:
    return os.getenv(SMTP_PASSWORD, "")


def get_smtp_use_tls() -> bool:
    return _get_bool_env(SMTP_USE_TLS, True)


def get_smtp_use_ssl() -> bool:
    return _get_bool_env(SMTP_USE_SSL, False)


def get_smtp_from_email() -> str:
    return os.getenv(SMTP_FROM_EMAIL, "").strip()


def get_smtp_from_name(default: str) -> str:
    return os.getenv(SMTP_FROM_NAME, default).strip() or default


def get_openrouter_official_providers_only() -> bool:
    """Return whether OpenRouter requests should pin official provider endpoints."""
    return _get_bool_env(OPENROUTER_OFFICIAL_PROVIDERS_ONLY, False)


def get_xrouter_excluded_models() -> tuple[str, ...]:
    """Return model slugs excluded from xrouter candidate sets.

    The environment value is a comma-separated list. Empty entries are ignored,
    and duplicates are removed while preserving the configured order.
    """
    value = os.getenv(XROUTER_EXCLUDED_MODELS, "")
    return tuple(
        dict.fromkeys(item.strip() for item in value.split(",") if item.strip())
    )


def get_mcp_oauth_allow_private_hosts() -> bool:
    """Return whether MCP OAuth URL policy may target local/private hosts.

    This is intended only for local development with local authorization
    servers. Production deployments should leave it disabled.
    """
    return _get_bool_env(MCP_OAUTH_ALLOW_PRIVATE_HOSTS, False)


def get_trusted_egress_proxy_enabled() -> bool:
    """Return whether the ambient HTTP(S)_PROXY may be used for public fetches.

    Outbound SSRF guarding pins the DNS resolution used at validation time
    to the one used at connect time. Routing through a proxy breaks that
    guarantee, since the proxy performs its own, independent DNS resolution
    of the target host. This flag is an explicit opt-in acknowledging that
    the configured proxy is trusted to enforce its own private-range egress
    policy; leave it disabled unless that is true.
    """
    return _get_bool_env(TRUSTED_EGRESS_PROXY, False)


def get_mcp_oauth_proxy_url() -> str | None:
    """Return an explicit trusted proxy URL for outbound MCP OAuth HTTP calls."""
    value = os.getenv(MCP_OAUTH_PROXY_URL)
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    parts = urlsplit(value)
    if parts.scheme not in {"http", "https"} or not parts.netloc:
        raise ValueError(
            f"Invalid {MCP_OAUTH_PROXY_URL} value: {value!r}. "
            "Expected an absolute HTTP(S) proxy URL."
        )
    return value


def _redis_url_with_database(url: str, database: int) -> str:
    """Return a Redis URL pointing at a different logical database."""
    parts = urlsplit(url)
    if parts.scheme not in {"redis", "rediss", "redis+socket"}:
        return url
    return urlunsplit((parts.scheme, parts.netloc, f"/{database}", "", ""))


def get_redis_url() -> str | None:
    """Return the optional Redis URL used by hot-path cache backends."""
    value = os.getenv(REDIS_URL)
    if value is None:
        return None
    value = value.strip()
    return value or None


def get_hot_path_cache_enabled() -> bool:
    """Return whether optional hot-path caching is enabled.

    Caching is inert unless ``XAGENT_REDIS_URL`` is configured or tests install
    an explicit cache backend. This flag gives operators a kill switch without
    changing the Redis URL.
    """
    value = os.getenv(HOT_PATH_CACHE_ENABLED, "true").strip().lower()
    return value in {"1", "true", "yes", "on"}


def get_hot_path_cache_ttl_seconds() -> int:
    """Default TTL for agent/model hot-path response caches."""
    return _get_positive_int_env(HOT_PATH_CACHE_TTL_SECONDS, 30)


def get_hot_path_task_cache_ttl_seconds() -> int:
    """Default TTL for task polling response caches."""
    return _get_positive_int_env(HOT_PATH_TASK_CACHE_TTL_SECONDS, 30)


def get_celery_enabled() -> bool:
    """Return whether durable background jobs should be enqueued to Celery."""
    return _get_bool_env(CELERY_ENABLED, False)


def get_tool_parallel_enabled() -> bool:
    """Whether independent tool calls in a ReAct turn run concurrently.

    Priority:
        1. XAGENT_TOOL_PARALLEL_ENABLED environment variable
        2. Default ``False`` (serial; byte-for-byte equivalent to before)

    Returns:
        True if concurrency-safe tool calls in a turn should run as a batch.
    """
    return _get_bool_env(TOOL_PARALLEL_ENABLED, False)


def get_tool_max_concurrency() -> int:
    """Maximum concurrent tool calls per ReAct turn batch (Semaphore bound).

    Priority:
        1. XAGENT_TOOL_MAX_CONCURRENCY environment variable
        2. Default ``3`` (kept low to limit API rate-limit pressure;
           per-API throttling is tracked separately)

    Invalid or non-positive values fall back to the default.

    Returns:
        The per-batch concurrency cap (>= 1).
    """
    return _get_positive_int_env(TOOL_MAX_CONCURRENCY, 3)


def get_task_runtime_hook_max_workers() -> int:
    """Maximum process-wide worker threads for task runtime provider hooks."""

    return _get_positive_int_env(TASK_RUNTIME_HOOK_MAX_WORKERS, 8)


def get_task_runtime_hook_queue_timeout_seconds() -> int:
    """Seconds a provider hook may wait for a runtime worker before starting."""

    return _get_positive_int_env(TASK_RUNTIME_HOOK_QUEUE_TIMEOUT_SECONDS, 30)


def get_native_browser_enabled() -> bool:
    """Whether tasks may control a browser on the interactive Xagent host."""

    return _get_bool_env(NATIVE_BROWSER_ENABLED, False)


def get_native_browser_app_name() -> str:
    """Browser application exposed by the Local browser runtime."""

    configured = (
        os.getenv(NATIVE_BROWSER_APP_NAME, "Google Chrome").strip() or "Google Chrome"
    )
    canonical = _NATIVE_BROWSER_APP_NAMES_BY_CASEFOLD.get(configured.casefold())
    if canonical is None:
        supported = ", ".join(sorted(SUPPORTED_NATIVE_BROWSER_APP_NAMES))
        raise ValueError(
            f"{NATIVE_BROWSER_APP_NAME} must name a supported browser: {supported}"
        )
    return canonical


_BCP47_LOCALE_RE = re.compile(r"^[a-zA-Z]{2,3}(-[a-zA-Z0-9]{2,8})*$")


def get_browser_tool_default_locale() -> str:
    """Get the fallback Playwright context locale for the browser_use tool.

    Used when a task/request carries no resolvable locale of its own (see
    ``WebToolConfig.get_browser_locale``). Was previously hardcoded to
    ``"zh-CN"``, which forced every automated browser session -- regardless
    of the requesting user's own language -- to request and render
    Chinese-localized pages.

    Priority:
        1. XAGENT_BROWSER_TOOL_DEFAULT_LOCALE environment variable
        2. "en-US"

    Raises:
        ValueError: if the env var is set but isn't a plausible BCP-47 tag
            (e.g. "en-US"). This getter is called lazily, from
            BrowserSession.__init__ on first browser tool use rather than at
            process startup, so a typo still fails as a clean tool-call
            error instead of an opaque Playwright error at session creation.
    """
    configured = os.getenv(BROWSER_TOOL_DEFAULT_LOCALE, "").strip()
    if not configured:
        return "en-US"
    if not _BCP47_LOCALE_RE.match(configured):
        raise ValueError(
            f"{BROWSER_TOOL_DEFAULT_LOCALE} must be a BCP-47 locale tag "
            f"(e.g. 'en-US', 'zh-CN'), got {configured!r}"
        )
    return configured


def get_browser_tool_default_timezone() -> str | None:
    """Get the fallback Playwright context timezone for the browser_use tool.

    Priority:
        1. XAGENT_BROWSER_TOOL_DEFAULT_TIMEZONE environment variable
        2. None (Playwright falls back to the host's own system timezone)

    Raises:
        ValueError: if the env var is set but isn't a recognized IANA
            timezone name (e.g. "Asia/Shanghai"). Like
            get_browser_tool_default_locale, this is read lazily on first
            browser tool use, not at process startup.
    """
    configured = os.getenv(BROWSER_TOOL_DEFAULT_TIMEZONE, "").strip()
    if not configured:
        return None
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

    try:
        ZoneInfo(configured)
    except (ZoneInfoNotFoundError, ValueError, KeyError) as exc:
        raise ValueError(
            f"{BROWSER_TOOL_DEFAULT_TIMEZONE} must be a valid IANA timezone "
            f"name (e.g. 'Asia/Shanghai'), got {configured!r}"
        ) from exc
    return configured


def get_browser_cua_driver_command() -> str:
    """Executable used to start the Local browser cua-driver MCP server."""

    return os.getenv(BROWSER_CUA_DRIVER_COMMAND, "cua-driver").strip() or "cua-driver"


def get_browser_cua_driver_socket() -> str | None:
    """Optional cua-driver daemon socket endpoint for Local browser."""

    value = os.getenv(BROWSER_CUA_DRIVER_SOCKET, "").strip()
    return value or None


def get_browser_cua_driver_timeout_seconds() -> float:
    """Per-call timeout for the Local browser driver."""

    raw_value = os.getenv(BROWSER_CUA_DRIVER_TIMEOUT_SECONDS, "30").strip()
    try:
        value = float(raw_value)
    except ValueError:
        value = 30.0
    if value > 0:
        return value
    logger.warning(
        "Invalid %s=%r; falling back to 30 seconds",
        BROWSER_CUA_DRIVER_TIMEOUT_SECONDS,
        raw_value,
    )
    return 30.0


def get_browser_cua_driver_max_elements() -> int:
    """Maximum AX elements requested for one Local browser observation."""

    return _get_positive_int_env(BROWSER_CUA_DRIVER_MAX_ELEMENTS, 2_000)


def get_checkpoint_encoding_v2_enabled() -> bool:
    """Whether checkpoint trace events use the v2 storage encoding.

    v2 extends the v1 refs/blob encoding to nested DAG/auto contexts,
    per-record tool-ledger dedup, and system-prompt blobs. Decode support
    for v2 is unconditional; this flag only gates NEW writes so a fleet can
    be rolled out in two phases (deploy with ``0`` first so every instance
    can read v2, then remove the override to start writing it).

    Priority:
        1. XAGENT_CHECKPOINT_ENCODING_V2 environment variable
        2. Default ``True``

    Returns:
        True if new checkpoints are written with the v2 encoding.
    """
    return _get_bool_env(CHECKPOINT_ENCODING_V2, True)


def get_checkpoint_history_limit() -> int:
    """How many checkpoint trace events to retain per task execution.

    Older checkpoint rows beyond this limit are deleted when a new
    checkpoint is persisted (resume only ever reads the most recent
    readable checkpoint; a few older rows are kept as a fallback for
    unreadable-latest recovery and debugging).

    Priority:
        1. XAGENT_CHECKPOINT_HISTORY_LIMIT environment variable
        2. Default ``8``

    ``0`` disables pruning entirely. Invalid values fall back to the
    default.

    Returns:
        The number of checkpoint rows to keep per execution (>= 0).
    """
    return _get_positive_int_env(CHECKPOINT_HISTORY_LIMIT, 8, minimum=0)


def get_compact_threshold_ratio() -> float:
    """Fraction of a model's context window at which to trigger compaction.

    Priority:
        1. XAGENT_COMPACT_THRESHOLD_RATIO environment variable
        2. Default ``0.75``

    Values outside ``(0, 1]`` fall back to the default.

    Returns:
        The compaction trigger ratio.
    """
    raw = os.getenv(COMPACT_THRESHOLD_RATIO)
    if raw:
        try:
            value = float(raw)
        except ValueError:
            logger.warning(
                "Invalid %s=%r (not a float); using default 0.75",
                COMPACT_THRESHOLD_RATIO,
                raw,
            )
            return 0.75
        if 0.0 < value <= 1.0:
            return value
        logger.warning(
            "%s=%r is outside (0, 1]; using default 0.75",
            COMPACT_THRESHOLD_RATIO,
            raw,
        )
    return 0.75


def get_compact_threshold_default() -> int:
    """Fallback compaction threshold (tokens) when a model has no context window set.

    Priority:
        1. XAGENT_COMPACT_THRESHOLD_DEFAULT environment variable
        2. Default ``32000``
    """
    return _get_positive_int_env(COMPACT_THRESHOLD_DEFAULT, 32000)


def get_celery_broker_url() -> str | None:
    """Return the Celery broker URL.

    If only ``XAGENT_REDIS_URL`` is configured, derive DB 1 so Celery broker
    traffic does not share the short-TTL hot-path cache database.
    """
    value = os.getenv(CELERY_BROKER_URL)
    if value is not None:
        value = value.strip()
        return value or None

    redis_url = get_redis_url()
    if redis_url is None:
        return None
    return _redis_url_with_database(redis_url, 1)


def get_celery_result_backend() -> str | None:
    """Return the optional Celery result backend URL.

    Background job state is persisted in the application database, so Celery
    results are disabled unless explicitly configured.
    """
    value = os.getenv(CELERY_RESULT_BACKEND)
    if value is None:
        return None
    value = value.strip()
    return value or None


def get_background_job_visibility_timeout_seconds() -> int:
    """Return Celery broker visibility timeout for long-running jobs."""
    return _get_positive_int_env(
        BACKGROUND_JOB_VISIBILITY_TIMEOUT_SECONDS,
        3600,
        minimum=60,
    )


def get_background_job_max_retries() -> int:
    """Return the default max attempts for durable background jobs."""
    return _get_positive_int_env(BACKGROUND_JOB_MAX_RETRIES, 3)


def get_background_job_stale_seconds() -> int:
    """Return the longest gap without a durable row update before requeueing.

    Measured against ``updated_at`` -- progress or status persistence -- not
    against how long the job has been running, so a job that keeps reporting is
    never requeued for being long. Liveness rides on whatever the work loop
    persists; there is no timer heartbeat.
    """
    return _get_positive_int_env(BACKGROUND_JOB_STALE_SECONDS, 7200, minimum=60)


def get_background_job_sweep_interval_seconds() -> int:
    """Return how often the scheduler scans for stale background jobs."""
    return _get_positive_int_env(
        BACKGROUND_JOB_SWEEP_INTERVAL_SECONDS,
        300,
        minimum=30,
    )


def get_taskless_upload_ttl_seconds() -> int:
    """Age after which an unbound task-less public upload is GC-eligible (#973).

    A task-less public-share upload is bound to its task at run start; if the
    guest never completes task creation it stays orphaned. Rows older than
    this (and still unbound) are reaped. Default 48h — long enough that a slow
    but real first turn is never reaped mid-flow.

    Priority:
        1. XAGENT_TASKLESS_UPLOAD_TTL_SECONDS environment variable
        2. Default 172800 (48 hours)
    """
    return _get_positive_int_env(
        TASKLESS_UPLOAD_TTL_SECONDS,
        48 * 60 * 60,
        minimum=60 * 60,
    )


def get_detached_upload_retention_seconds() -> int:
    """Age after which an attachment detached by Task deletion is reclaimed."""

    return _get_positive_int_env(
        DETACHED_UPLOAD_RETENTION_SECONDS,
        7 * 24 * 60 * 60,
        minimum=60 * 60,
    )


def get_orphan_upload_sweep_interval_seconds() -> int:
    """How often the orphan task-less-upload GC sweep runs (#973).

    Priority:
        1. XAGENT_ORPHAN_UPLOAD_SWEEP_INTERVAL_SECONDS environment variable
        2. Default 3600 (hourly)
    """
    return _get_positive_int_env(
        ORPHAN_UPLOAD_SWEEP_INTERVAL_SECONDS,
        60 * 60,
        minimum=60,
    )


def get_workforce_preview_run_stale_seconds() -> int:
    """Age after which an abandoned builder preview run is GC-eligible.

    Preview runs (workforce builder "test before save", ``is_preview`` true —
    either an ephemeral create-mode draft or an edit-mode test against an
    already-saved workforce) are only invalidated client-side when the draft
    changes; a closed tab, crashed browser, or network drop leaves the run
    (and its hidden Task) active server-side forever with no owner left to
    invalidate it. Rows still non-terminal past this age are reaped by the
    scheduled sweep.

    Priority:
        1. XAGENT_WORKFORCE_PREVIEW_RUN_STALE_SECONDS environment variable
        2. Default 7200 (2 hours)

    Clamped to a minimum of 300 seconds (5 minutes), so a misconfigured tiny
    value can't reap a preview run that is still genuinely in progress.
    """
    return _get_positive_int_env(WORKFORCE_PREVIEW_RUN_STALE_SECONDS, 7200, minimum=300)


def get_trigger_dispatcher_enabled() -> bool:
    """Return whether the backend should start prepared trigger runs."""
    return _get_bool_env(TRIGGER_DISPATCHER_ENABLED, True)


def get_trigger_dispatcher_interval_seconds() -> int:
    """Return how often backend processes poll prepared trigger runs."""
    return _get_positive_int_env(
        TRIGGER_DISPATCHER_INTERVAL_SECONDS,
        5,
        minimum=1,
    )


def get_trigger_dispatcher_batch_size() -> int:
    """Return max prepared trigger runs one backend poll should start."""
    return _get_positive_int_env(
        TRIGGER_DISPATCHER_BATCH_SIZE,
        20,
        minimum=1,
    )


def get_trigger_dispatcher_startup_jitter_seconds() -> int:
    """Return the max random delay before the dispatcher's first tick.

    Priority:
        1. XAGENT_TRIGGER_DISPATCHER_STARTUP_JITTER_SECONDS environment
           variable
        2. Default 30

    A container restart brings every trigger that fell due while it was
    down (Gmail watch renewals, scheduled triggers) up for processing all
    at once, and the dispatcher's first tick runs immediately on startup --
    before egress networking may be fully warmed up. This delay pushes that
    first tick past the likely warm-up window; it does not shrink how much
    that tick processes (still gated by the dispatcher's own batch-size and
    scan-limit settings), only when it starts. On a multi-replica rolling
    restart it also desyncs every replica's first tick from firing at the
    same instant, spreading their combined load across the window instead
    of concentrating it in a single moment. 0 disables the delay.
    """
    return _get_positive_int_env(
        TRIGGER_DISPATCHER_STARTUP_JITTER_SECONDS,
        30,
        minimum=0,
    )


def get_trigger_callback_rate_limit() -> str:
    """Rate limit for the public trigger callback endpoint.

    Priority:
        1. XAGENT_TRIGGER_CALLBACK_RATE_LIMIT environment variable
        2. Default "120/minute"

    Returns:
        A rate string in the ``limits`` notation, e.g. "120/minute".
    """
    value = (os.getenv(TRIGGER_CALLBACK_RATE_LIMIT) or "").strip()
    return value or "120/minute"


def get_trigger_callback_ip_rate_limit() -> str:
    """Per-IP ceiling across ALL callback ids on the public callback endpoint.

    Priority:
        1. XAGENT_TRIGGER_CALLBACK_IP_RATE_LIMIT environment variable
        2. Default "600/minute"

    The per-callback-id bucket alone can be bypassed by rotating random
    callback ids; this IP-wide bucket caps that traffic before audit writes.

    Returns:
        A rate string in the ``limits`` notation, e.g. "600/minute".
    """
    value = (os.getenv(TRIGGER_CALLBACK_IP_RATE_LIMIT) or "").strip()
    return value or "600/minute"


def get_trigger_crud_rate_limit() -> str:
    """Rate limit for trigger create/update/delete API calls per user.

    Priority:
        1. XAGENT_TRIGGER_CRUD_RATE_LIMIT environment variable
        2. Default "60/minute"

    Returns:
        A rate string in the ``limits`` notation, e.g. "60/minute".
    """
    value = (os.getenv(TRIGGER_CRUD_RATE_LIMIT) or "").strip()
    return value or "60/minute"


def _get_rate_limit(env_var: str, default: str) -> str:
    """Read a ``limits``-notation rate string with an env override.

    Priority: the given ``XAGENT_*`` env var, else ``default``. The value is
    validated (and defaulted) at parse time by the rate limiter, so this only
    trims and falls back on an empty/unset var.
    """
    return (os.getenv(env_var) or "").strip() or default


def get_share_auth_rate_limit() -> str:
    """Per-share-token limit on ``POST /api/share/auth`` (#973).

    No ``guest_id`` exists before auth, so this bounds one share link's token
    minting; the per-IP ceiling below bounds a single client across links.
    """
    return _get_rate_limit(SHARE_AUTH_RATE_LIMIT, "60/minute")


def get_share_auth_ip_rate_limit() -> str:
    """Per-IP ceiling on ``POST /api/share/auth`` across all share links."""
    return _get_rate_limit(SHARE_AUTH_IP_RATE_LIMIT, "300/minute")


def get_share_task_create_rate_limit() -> str:
    """Per-guest limit on public share task creation (#973).

    Task creation is the costly surface (each spawns an owner-billed run), so
    this is the tighter of the two task-create buckets.
    """
    return _get_rate_limit(SHARE_TASK_CREATE_RATE_LIMIT, "30/minute")


def get_share_task_create_token_rate_limit() -> str:
    """Per-share-token ceiling on public share task creation.

    Stops a client rotating fresh ``guest_id`` tokens (one auth each) from
    bypassing the per-guest bucket on one share link.
    """
    return _get_rate_limit(SHARE_TASK_CREATE_TOKEN_RATE_LIMIT, "120/minute")


def get_share_ws_turn_rate_limit() -> str:
    """Per-guest limit on share websocket turns (#973).

    Follow-up turns bypass task-create and each starts an owner-billed run;
    this caps the burst rate before a turn is enqueued.
    """
    return _get_rate_limit(SHARE_WS_TURN_RATE_LIMIT, "60/minute")


def get_share_ws_connect_ip_rate_limit() -> str:
    """Per-IP limit on share websocket connection attempts (#973).

    The share websocket accepts the handshake before auth so denial reasons
    reach the client, which means even a garbage token completes a full 101
    upgrade before rejection. This caps how many of those handshakes one IP
    can open; over-limit attempts are refused pre-accept (no upgrade cost).
    """
    return _get_rate_limit(SHARE_WS_CONNECT_IP_RATE_LIMIT, "120/minute")


def get_share_upload_rate_limit() -> str:
    """Per-guest limit on public share file uploads (#973)."""
    return _get_rate_limit(SHARE_UPLOAD_RATE_LIMIT, "60/minute")


def get_widget_upload_rate_limit() -> str:
    """Per-widget-entity limit on public widget file uploads (#973).

    Keyed on the embedded agent/workforce (not the widget ``guest_id``,
    which is client-supplied and rotatable at will). Loose: one widget
    serves many legitimate guests.
    """
    return _get_rate_limit(WIDGET_UPLOAD_RATE_LIMIT, "240/minute")


def get_widget_upload_ip_rate_limit() -> str:
    """Per-caller-IP limit on public widget file uploads (#973).

    The tighter bucket: bounds one abuser without a trustworthy per-guest
    key. Kept loose enough for enterprise NAT."""
    return _get_rate_limit(WIDGET_UPLOAD_IP_RATE_LIMIT, "60/minute")


def get_widget_ws_connect_ip_rate_limit() -> str:
    """Per-IP limit on widget websocket connection attempts (#1056).

    The widget websocket path mirrors the share handshake budget but in its
    own bucket, so probes against one public channel cannot consume the
    other's. Keyed per IP because the gate runs pre-auth; over-limit attempts
    are refused pre-accept (no upgrade cost).
    """
    return _get_rate_limit(WIDGET_WS_CONNECT_IP_RATE_LIMIT, "120/minute")


def get_widget_ws_turn_ip_rate_limit() -> str:
    """Per-caller-IP limit on widget websocket turns (#1056).

    The tighter turn bucket. Unlike the share turn gate this cannot key on
    the guest: the widget ``guest_id`` is client-supplied and rotatable at
    will, so the caller IP is the only per-abuser key available. The default
    numerically matches the share per-guest turn rate, but this bucket is
    per-IP: N guests behind one NAT egress share a single 60/minute budget
    here, where the share path gives each guest its own. Raise it for
    deployments with large shared-egress populations — and behind a reverse
    proxy, set XAGENT_TRUSTED_PROXY_HOPS so all traffic does not collapse
    onto the proxy's IP.
    """
    return _get_rate_limit(WIDGET_WS_TURN_IP_RATE_LIMIT, "60/minute")


def get_widget_ws_turn_rate_limit() -> str:
    """Per-widget-entity limit on widget websocket turns (#1056).

    The loose backstop bounding total owner-billed turn volume through one
    embedded agent/workforce across all callers (one widget serves many
    legitimate guests).
    """
    return _get_rate_limit(WIDGET_WS_TURN_RATE_LIMIT, "240/minute")


def get_widget_auth_rate_limit() -> str:
    """Per-widget-entity limit on widget auth + embed-ticket minting (#1108).

    The loose aggregate backstop bounding total auth/ticket volume through one
    embedded agent/workforce across all callers. Deliberately loose: both
    endpoints fire on every widget page load and the entity key is shared by
    all of a widget's visitors, so a tight per-entity bucket would 429
    ordinary visitors on a busy embed. The per-IP limit below is the tight
    per-visitor / per-abuser bound. Kept at the same 4:1 entity:IP ratio as
    the sibling widget upload / ws-turn gates — and auth is the one gate whose
    denial is fail-closed client-side (the widget never loads), so its
    aggregate backstop is deliberately the most tolerant, not the tightest.
    Raise this for very high-traffic embeds.
    """
    return _get_rate_limit(WIDGET_AUTH_RATE_LIMIT, "1200/minute")


def get_widget_auth_ip_rate_limit() -> str:
    """Per-caller-IP limit on widget auth + embed-ticket minting (#1108).

    The tight per-visitor / per-abuser bound (visitors have distinct IPs):
    bounds one caller minting guest tokens / embed tickets (each call does DB
    lookups and signs a JWT) regardless of how many widget keys or tickets
    they cycle through, since the IP is not cheaply rotatable. Raise this
    where many genuine visitors share one address (corporate NAT, carrier
    CGNAT), and set XAGENT_TRUSTED_PROXY_HOPS correctly behind a reverse proxy
    — otherwise every caller resolves to the proxy's IP and this becomes one
    global cap. (example.env carries the same caveat for all per-IP buckets.)
    """
    return _get_rate_limit(WIDGET_AUTH_IP_RATE_LIMIT, "300/minute")


def get_widget_task_create_rate_limit() -> str:
    """Per-widget-entity limit on public widget task creation (#1108).

    The loose backstop bounding total task-create volume through one embedded
    agent/workforce across all callers (one widget serves many legitimate
    guests). The per-IP bucket below is the tighter per-abuser gate; unlike the
    share path this cannot key on the guest, whose id is client-supplied and
    rotatable at will. Kept at the 4:1 entity:IP ratio shared by every widget
    gate: the entity bucket only accumulates on admitted requests, so ``ratio``
    cooperating under-cap IPs are needed to saturate it — 4 here, not 2, on the
    surface where each admitted request spawns an owner-billed run.
    """
    return _get_rate_limit(WIDGET_TASK_CREATE_RATE_LIMIT, "240/minute")


def get_widget_task_create_ip_rate_limit() -> str:
    """Per-caller-IP limit on public widget task creation (#1108).

    The tighter bucket: task creation is the costly surface (each spawns an
    owner-billed run), and the caller IP is the only trustworthy per-abuser
    key on the widget path. Numerically matches the widget upload/turn IP
    default. Raise this where many genuine visitors share one address
    (corporate NAT, carrier CGNAT), and set XAGENT_TRUSTED_PROXY_HOPS correctly
    behind a reverse proxy — otherwise every caller resolves to the proxy's IP
    and this becomes one global cap. (example.env carries the same caveat for
    all per-IP buckets.)
    """
    return _get_rate_limit(WIDGET_TASK_CREATE_IP_RATE_LIMIT, "60/minute")


def get_widget_run_quota() -> str:
    """Per-widget-entity rolling run quota (#1108).

    The widget mirror of :func:`get_share_run_quota`, in its own bucket so a
    popular/abused widget cannot drain the owner's whole team quota. Keyed on
    the embedded agent/workforce, with a per-creating-IP sub-quota
    (:func:`get_widget_run_ip_quota`) as the per-abuser dimension — the widget
    ``guest_id`` is client-supplied (rotatable at will), so unlike the share
    path there is no per-guest sub-quota. NOTE: this quota applies to
    already-live widget tasks as soon as it deploys (their ``agent_config``
    carries the widget markers); the only opt-out is raising the env var.
    """
    return _get_rate_limit(WIDGET_RUN_QUOTA, "500/day")


def get_widget_run_ip_quota() -> str:
    """Per-creating-IP, per-widget rolling run sub-quota (#1108).

    The per-abuser sub-quota under :func:`get_widget_run_quota`, mirroring the
    share path's per-guest window. Its bucket is keyed ``entity|ip``, i.e.
    scoped to one widget: a caller's budget on one embedded agent/workforce is
    independent of every other widget on the instance. That scoping matters
    because this quota is charged per *turn* — a bare-IP bucket would make one
    NAT/CGNAT egress share a single turn budget across unrelated widgets.

    Sizing: this is a share of a rolling owner-billed budget, not a burst
    throttle, so it is deliberately far below the per-minute burst gates
    (widget WS turn / task-create, both 60/minute per IP) and instead sized as
    a fraction of :func:`get_widget_run_quota` — at the defaults one caller
    needs several sustained hours to drain a widget's daily budget. It does
    NOT stop a multi-IP abuser: roughly ``entity_quota / ip_quota`` IPs, each
    staying under its own window, still exhaust the entity quota —
    structurally the same limit as the share path's per-guest quota.

    The IP is the one the server observed at task creation (stamped into
    ``agent_config``, never client-supplied); tasks created before this deploy
    carry no marker and are bounded by the entity quota alone. Raise it for
    deployments fronted by large shared egress (corporate NAT, carrier CGNAT),
    where many genuine visitors present one address, and set
    XAGENT_TRUSTED_PROXY_HOPS correctly behind a reverse proxy — otherwise
    every caller resolves to the proxy's IP and this becomes one global cap.
    (example.env carries the same caveat for all per-IP buckets.)
    """
    return _get_rate_limit(WIDGET_RUN_IP_QUOTA, "120/hour")


def get_share_run_quota() -> str:
    """Per-share rolling run quota (#973).

    Bounds the owner-billed runs one share link can start per window so a
    single popular/abused link cannot exhaust the owner's whole team quota.
    Rolling (not cumulative) so a legitimately busy link self-clears rather
    than being permanently bricked.
    """
    return _get_rate_limit(SHARE_RUN_QUOTA, "500/day")


def get_share_run_guest_quota() -> str:
    """Per-guest rolling run quota within a share link (#973).

    Shorter window than the per-share quota: bounds a single visitor's burst
    of runs so one guest cannot consume the whole link's budget.
    """
    return _get_rate_limit(SHARE_RUN_GUEST_QUOTA, "60/hour")


def get_trusted_proxy_hops() -> int:
    """Number of trusted reverse-proxy hops in front of the backend.

    Priority:
        1. XAGENT_TRUSTED_PROXY_HOPS environment variable
        2. Default 0 (forwarded headers are not trusted)

    Returns:
        How many trailing X-Forwarded-For entries were appended by trusted
        proxies. 0 means the raw peer address is used.
    """
    return _get_positive_int_env(TRUSTED_PROXY_HOPS, 0, minimum=0)


def get_gmail_pubsub_project_id() -> str | None:
    """GCP project id used for per-mailbox Gmail Pub/Sub provisioning.

    Priority:
        1. XAGENT_GMAIL_PUBSUB_PROJECT_ID environment variable
        2. None (Gmail provisioning is not configured)
    """
    value = (os.getenv(GMAIL_PUBSUB_PROJECT_ID) or "").strip()
    return value or None


def get_gmail_pubsub_topic_prefix() -> str:
    """Prefix for deterministic per-mailbox Gmail Pub/Sub topic names.

    Priority:
        1. XAGENT_GMAIL_PUBSUB_TOPIC_PREFIX environment variable
        2. Default "xagent-gmail"
    """
    value = (os.getenv(GMAIL_PUBSUB_TOPIC_PREFIX) or "").strip()
    return value or "xagent-gmail"


def get_gmail_pubsub_subscription_prefix() -> str:
    """Prefix for deterministic per-mailbox Gmail push subscription names.

    Priority:
        1. XAGENT_GMAIL_PUBSUB_SUBSCRIPTION_PREFIX environment variable
        2. Default "xagent-gmail-push"
    """
    value = (os.getenv(GMAIL_PUBSUB_SUBSCRIPTION_PREFIX) or "").strip()
    return value or "xagent-gmail-push"


def get_gmail_pubsub_transport() -> str:
    """Transport used by the Gmail Pub/Sub provisioning clients.

    Priority:
        1. XAGENT_GMAIL_PUBSUB_TRANSPORT environment variable ("grpc" or "rest")
        2. Default "grpc"

    "rest" exists for environments whose egress proxy cannot tunnel gRPC:
    the default gRPC channel hangs indefinitely there, leaving Gmail watch
    provisioning stuck at pending with no recorded error.
    """
    value = (os.getenv(GMAIL_PUBSUB_TRANSPORT) or "").strip().lower()
    return value if value in ("grpc", "rest") else "grpc"


def get_gmail_pubsub_push_service_account() -> str | None:
    """Service-account email used for Pub/Sub push OIDC tokens.

    Priority:
        1. XAGENT_GMAIL_PUBSUB_PUSH_SERVICE_ACCOUNT environment variable
        2. None

    Used both when provisioning the push subscription (oidc_token identity)
    and, optionally, to validate the pushing service account on delivery.
    """
    value = (os.getenv(GMAIL_PUBSUB_PUSH_SERVICE_ACCOUNT) or "").strip()
    return value or None


def get_gmail_registration_timeout_seconds() -> int:
    """How long a trigger create/update waits for Gmail provisioning.

    Priority:
        1. XAGENT_GMAIL_REGISTRATION_TIMEOUT_SECONDS environment variable
        2. Default 10 seconds

    Slow registration returns a pending state after this timeout while the
    reconcile flow converges to active or failed in the background.
    """
    return _get_positive_int_env(GMAIL_REGISTRATION_TIMEOUT_SECONDS, 10)


def get_public_api_base_url() -> str | None:
    """Public base URL of the backend API for advertised routes and callbacks.

    Priority:
        1. XAGENT_PUBLIC_API_BASE_URL environment variable
        2. None (consumers apply their own compatibility or validation policy)

    Deliberately separate from XAGENT_APP_BASE_URL (the frontend URL used in
    e.g. password-reset emails): externally advertised browser and MCP API URLs
    should normally use the public backend origin. Server-to-server consumers
    can override this via XAGENT_S2S_API_BASE_URL (see
    get_s2s_api_base_url).
    """
    return _normalized_http_env_url(PUBLIC_API_BASE_URL)


def get_s2s_api_base_url() -> str | None:
    """Backend base URL advertised to server-to-server integrations.

    Priority:
        1. XAGENT_S2S_API_BASE_URL environment variable
        2. XAGENT_PUBLIC_API_BASE_URL for backward-compatible deployments
        3. None

    The separate value lets deployments send provider callbacks and A2A
    traffic directly to a regional backend while browser and MCP traffic keep
    using the canonical public API. Trailing slashes are removed so consumers
    can append absolute API paths without producing duplicate separators.
    """
    return _normalized_http_env_url(S2S_API_BASE_URL) or get_public_api_base_url()


def get_gmail_callback_base_url() -> str | None:
    """Return the base URL used for Gmail Pub/Sub callbacks.

    ``XAGENT_TRIGGER_CALLBACK_BASE_URL`` was the Gmail-specific override
    before the broader S2S URL was introduced. Keep it as a deprecated
    fallback so upgrading does not silently move existing subscriptions back
    to a browser-facing public edge. A2A deliberately uses
    :func:`get_s2s_api_base_url` directly and never advertises this legacy
    Gmail-only endpoint.

    Priority:
        1. XAGENT_S2S_API_BASE_URL
        2. XAGENT_TRIGGER_CALLBACK_BASE_URL (deprecated)
        3. XAGENT_PUBLIC_API_BASE_URL
        4. None
    """
    return (
        _normalized_http_env_url(S2S_API_BASE_URL)
        or _normalized_http_env_url(TRIGGER_CALLBACK_BASE_URL)
        or get_public_api_base_url()
    )


def get_gmail_watch_enabled() -> bool:
    """Return whether the Gmail watch feature is enabled.

    Gates both watch registration (OAuth connect, Gmail trigger
    create/update/enable) and the background renewal/retry scans. With the
    flag off (the default), no new watch is created and Gmail triggers report
    a failed provisioning status with an explicit disabled error where
    applicable. An existing Gmail watch is not stopped by disabling this flag:
    callbacks can remain deliverable until the watch expires or its mailbox
    resources are explicitly torn down.

    Teardown is deliberately left ungated: rebinding, disabling, or deleting
    a Gmail trigger still releases the old mailbox's watch and Pub/Sub
    resources while this flag is off, so switching it off never strands
    those resources.

    The operator endpoint-reconciliation CLI
    (``reconcile_gmail_push_endpoints``) is also deliberately ungated, so
    push endpoints can be migrated ahead of enabling this flag.
    """
    return _get_bool_env(GMAIL_WATCH_ENABLED, False)


def get_gmail_watch_renewal_interval_seconds() -> int:
    """Return how often the backend scans Gmail watches for renewal."""
    return _get_positive_int_env(
        GMAIL_WATCH_RENEWAL_INTERVAL_SECONDS,
        3600,
        minimum=60,
    )


def get_gmail_watch_renewal_lead_seconds() -> int:
    """Return how early Gmail watches should be renewed before expiration."""
    return _get_positive_int_env(
        GMAIL_WATCH_RENEWAL_LEAD_SECONDS,
        24 * 60 * 60,
        minimum=60,
    )


def get_google_oidc_client_id() -> str | None:
    """Return the configured Google OIDC client ID, if any."""
    value = os.getenv(GOOGLE_OIDC_CLIENT_ID)
    return value.strip() if value and value.strip() else None


def get_google_oidc_client_secret() -> str | None:
    """Return the configured Google OIDC client secret, if any."""
    value = os.getenv(GOOGLE_OIDC_CLIENT_SECRET)
    return value.strip() if value and value.strip() else None


def get_google_oidc_redirect_uri() -> str | None:
    """Return the configured Google OIDC callback URI, if any."""
    value = os.getenv(GOOGLE_OIDC_REDIRECT_URI)
    return value.strip() if value and value.strip() else None


def get_frontend_url() -> str:
    """Return the public frontend origin used after browser auth callbacks."""
    value = os.getenv(FRONTEND_URL)
    if value and value.strip():
        return value.strip().rstrip("/")
    return "http://localhost:3000"


def get_oidc_login_ttl_seconds() -> int:
    """Return the short-lived OIDC login transaction TTL."""
    return _get_positive_int_env(OIDC_LOGIN_TTL_SECONDS, 600)


def get_oidc_exchange_ttl_seconds() -> int:
    """Return the short-lived OIDC frontend exchange-code TTL."""
    return _get_positive_int_env(OIDC_EXCHANGE_TTL_SECONDS, 120)


def get_session_secret() -> str:
    """Return the Starlette session secret used by browser OAuth flows."""
    value = os.getenv(SESSION_SECRET)
    if value and value.strip():
        return value.strip()
    return os.getenv("XAGENT_JWT_SECRET", "your-secret-key-change-in-production")


def get_web_dir() -> Path:
    """Get the web directory path.

    Priority:
    1. XAGENT_WEB_DIR environment variable
    2. Default to src/xagent/web relative to this file

    Returns:
        Path object for web directory
    """
    env_dir = os.getenv(WEB_DIR)
    if env_dir:
        return Path(env_dir)

    # Default: src/xagent/web relative to this file
    # This file is at: src/xagent/config.py
    # Web dir is at: src/xagent/web/
    return Path(__file__).parent / "web"


class UploadsDirConfigurationError(Exception):
    """The configured uploads directory has no single physical meaning.

    Raised where the value is read, which for the web app is
    ``app.py``'s module body: the failure lands during import, before the
    application object exists, so no request-handling frame is on the stack to
    swallow it and the process simply refuses to start. That -- not this
    exception's place in the hierarchy -- is what keeps a misconfigured
    deployment from being reported as a transient per-request failure.
    """


class ExternalUploadsDirConfigurationError(Exception):
    """A configured external upload directory has two physical meanings."""


# First absolutized reading of each cwd-dependent uploads root, kept so the
# root cannot move mid-process (see _require_unambiguous_uploads_dir). Keyed by
# the configured spelling, so changing the configuration still takes effect.
_pinned_relative_uploads_roots: dict[str, Path] = {}

# External upload dirs feed both the chat allowlist and sandbox mount building
# through separate calls. Pin cwd-dependent spellings on first use so a later
# process-wide chdir cannot make those consumers name different directories.
_pinned_relative_external_upload_dirs: dict[str, Path] = {}


def _reset_path_config_caches_for_tests() -> None:
    """Clear cwd-dependent path pins between tests.

    Production code deliberately keeps these values for the process lifetime;
    tests need an explicit reset boundary so their result cannot depend on
    which working directory an earlier test pinned for the same spelling.
    """
    _pinned_relative_uploads_roots.clear()
    _pinned_relative_external_upload_dirs.clear()


def _require_unambiguous_uploads_dir(uploads_dir: Path) -> Path:
    """Reject an uploads dir whose two normalizations name different places.

    Paths under the uploads root reach consumers that normalize differently,
    and both normalizations are load-bearing:

    - lexical (``canonical_sandbox_path``) is still used by generic sandbox
      configuration identities and must not preserve ``..`` segments that a
      backend will report differently;
    - physical (``realpath``) is what ``TaskWorkspace``, the upload writers
      and ``files.py``'s containment checks use, because files have to be
      found.

    They agree on every spelling but one: a symlink followed by ``..``, where
    the lexical form discards the symlink the physical form follows. That
    configuration gives one logical directory two readings. Workspace mount
    producers retain both readings -- the lexical one for Docker-host path
    translation and the physical one for file identity -- so rejecting this
    ambiguous spelling at the shared configuration boundary prevents those
    two load-bearing views from naming different directories.

    An ordinary symlink is untouched -- following one is not a disagreement,
    both spellings still name a single directory.

    Returns the absolutized value rather than the configured one, so callers
    receive the path this check actually examined. A relative or
    ``~``/``$VAR``-prefixed value is resolved against the environment as it
    stands here; returning it unresolved would let a later ``os.chdir`` (the
    Python execution tool does exactly that, process-wide, while a task runs)
    move the directory out from under the guarantee.
    """
    from .sandbox.base import canonical_sandbox_path

    raw = str(uploads_dir)
    expanded = Path(os.path.expandvars(raw)).expanduser()
    if expanded.is_absolute():
        absolute = expanded
    else:
        # A relative value means whatever the working directory says, and this
        # process changes it: the Python execution tool chdirs process-wide for
        # the duration of a task's code. Pinning the first reading keeps one
        # root for the process, so two callers cannot compose paths from two
        # different directories depending on when they asked.
        absolute = _pinned_relative_uploads_roots.setdefault(raw, Path.cwd() / expanded)
    canonical = canonical_sandbox_path(str(absolute))
    if os.path.realpath(canonical) != os.path.realpath(absolute):
        # Name whichever variable actually produced this root, so the
        # operator edits the one that is set.
        source = UPLOADS_DIR if os.getenv(UPLOADS_DIR) else WEB_DIR
        raise UploadsDirConfigurationError(
            f"The uploads root {str(uploads_dir)!r} (from {source}) names two "
            f"different directories depending on how it is normalized: "
            f"lexically it is {canonical!r}, resolving to "
            f"{os.path.realpath(canonical)!r}, while resolving the configured "
            f"spelling directly gives {os.path.realpath(absolute)!r}. A '..' "
            "segment after a symlink does that. Configure the directory you "
            "actually mean."
        )
    return absolute


def get_uploads_dir() -> Path:
    """Get the uploads directory path.

    Priority:
    1. XAGENT_UPLOADS_DIR environment variable
    2. Default to WEB_DIR/uploads for backward compatibility

    Validated here rather than at each consumer: this is the root every
    workspace, upload, knowledge-base and sandbox-mount path is composed
    from, so one check covers all of them -- see
    :func:`_require_unambiguous_uploads_dir`. The validation is on the value
    this function returns, not on one of the two branches that produce it:
    ``XAGENT_WEB_DIR`` reaches the uploads root just as directly as
    ``XAGENT_UPLOADS_DIR`` does, and an ambiguous spelling in either is the
    same ambiguity downstream.

    Returns:
        Path object for uploads directory

    Raises:
        UploadsDirConfigurationError: The resulting root's lexical and
            physical normalizations name different directories.
    """
    env_dir = os.getenv(UPLOADS_DIR)
    if env_dir:
        uploads_dir = Path(env_dir)
    else:
        # Default: web/uploads
        uploads_dir = get_web_dir() / "uploads"
    return _require_unambiguous_uploads_dir(uploads_dir)


def get_frontend_dist_dir() -> Path:
    """Get the directory holding the built frontend static export.

    Priority:
        1. XAGENT_FRONTEND_DIST_DIR environment variable
        2. Default to WEB_DIR/frontend_dist (where the wheel bundles the export)

    The directory may not exist (e.g. an editable install without a frontend
    build, or the multi-container Docker deployment where Next.js serves the
    frontend). Callers must handle a missing directory by falling back to
    API-only serving.

    Returns:
        Path object for the bundled frontend static export directory
    """
    env_dir = os.getenv(FRONTEND_DIST_DIR)
    if env_dir:
        return Path(env_dir)

    return get_web_dir() / "frontend_dist"


def get_max_upload_size_bytes() -> int:
    """Get the maximum allowed upload size in bytes.

    Priority:
    1. XAGENT_MAX_UPLOAD_SIZE environment variable
    2. Default to 100MB

    Supported formats:
    - Raw bytes: ``104857600``
    - Human-readable: ``100M``, ``100MB``, ``1G``, ``512K``

    Returns:
        Maximum upload size in bytes.

    Raises:
        ValueError: If the configured value is invalid.
    """

    env_value = os.getenv(MAX_UPLOAD_SIZE)
    if not env_value:
        return 100 * 1024 * 1024

    normalized = env_value.strip().upper()
    if not normalized:
        return 100 * 1024 * 1024

    suffix_multipliers = [
        ("GB", 1024 * 1024 * 1024),
        ("G", 1024 * 1024 * 1024),
        ("MB", 1024 * 1024),
        ("M", 1024 * 1024),
        ("KB", 1024),
        ("K", 1024),
        ("B", 1),
    ]

    result: int | None = None
    for suffix, multiplier in suffix_multipliers:
        if normalized.endswith(suffix):
            number_part = normalized[: -len(suffix)].strip()
            if not number_part:
                raise ValueError(
                    f"Invalid {MAX_UPLOAD_SIZE} value: {env_value!r}. Missing numeric value."
                )
            try:
                result = int(float(number_part) * multiplier)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid {MAX_UPLOAD_SIZE} value: {env_value!r}."
                ) from exc
            break

    if result is None:
        try:
            result = int(float(normalized))
        except ValueError as exc:
            raise ValueError(
                f"Invalid {MAX_UPLOAD_SIZE} value: {env_value!r}."
            ) from exc

    if result <= 0:
        raise ValueError(
            f"Invalid {MAX_UPLOAD_SIZE} value: {env_value!r}. Value must be positive."
        )

    return result


def get_file_storage_uri() -> str:
    """Get the durable file storage URI.

    Priority:
        1. XAGENT_FILE_STORAGE_URI environment variable
        2. file://<storage-root>/files

    Returns:
        fsspec-compatible URI for durable user-visible file storage.
    """
    env_value = os.getenv(FILE_STORAGE_URI)
    if env_value:
        return env_value

    return (get_storage_root().expanduser().resolve() / "files").as_uri()


def get_file_storage_options() -> dict[str, Any]:
    """Get fsspec provider options for durable file storage.

    The value must be a JSON object. Provider-specific details such as S3
    endpoint URL, region, or credentials profile live here to keep the config
    surface small.
    """
    env_value = os.getenv(FILE_STORAGE_OPTIONS)
    if not env_value:
        return {}

    try:
        parsed = json.loads(env_value)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid {FILE_STORAGE_OPTIONS} value: {env_value!r}"
        ) from exc

    if not isinstance(parsed, dict):
        raise ValueError(f"Invalid {FILE_STORAGE_OPTIONS} value: must be a JSON object")

    return parsed


def get_file_materialize_dir() -> Path:
    """Get the local directory used for temporary durable-file materialization."""
    env_value = os.getenv(FILE_MATERIALIZE_DIR)
    if env_value:
        return Path(env_value)

    return Path(tempfile.gettempdir()) / "xagent-materialized"


def get_preview_tmp_dir() -> Path:
    """Get the local directory used for temporary build-preview files."""
    env_value = os.getenv(PREVIEW_TMP_DIR)
    if env_value:
        return Path(env_value).expanduser()

    return Path(tempfile.gettempdir()) / "xagent-preview"


def get_file_storage_startup_sync_enabled() -> bool:
    """Return whether registered local files should sync to durable storage at startup."""
    env_value = os.getenv(FILE_STORAGE_STARTUP_SYNC_ENABLED)
    if env_value is None or not env_value.strip():
        return True

    normalized = env_value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False

    raise ValueError(
        f"Invalid {FILE_STORAGE_STARTUP_SYNC_ENABLED} value: {env_value!r}. "
        "Expected a boolean value."
    )


def get_file_delivery_redirect_enabled() -> bool:
    """Return whether private file endpoints may redirect to durable object URLs."""
    env_value = os.getenv(FILE_DELIVERY_REDIRECT_ENABLED)
    if env_value is None or not env_value.strip():
        return False

    normalized = env_value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False

    raise ValueError(
        f"Invalid {FILE_DELIVERY_REDIRECT_ENABLED} value: {env_value!r}. "
        "Expected a boolean value."
    )


def get_file_delivery_signed_url_ttl_seconds() -> int:
    """Get signed durable-object URL lifetime for private file delivery redirects."""
    env_value = os.getenv(FILE_DELIVERY_SIGNED_URL_TTL_SECONDS)
    if env_value is None or not env_value.strip():
        return 300

    try:
        ttl = int(env_value)
    except ValueError as exc:
        raise ValueError(
            f"Invalid {FILE_DELIVERY_SIGNED_URL_TTL_SECONDS} value: {env_value!r}."
        ) from exc

    if ttl <= 0:
        raise ValueError(
            f"Invalid {FILE_DELIVERY_SIGNED_URL_TTL_SECONDS} value: {env_value!r}. "
            "Value must be positive."
        )

    return ttl


def get_file_stream_ticket_ttl_seconds() -> int:
    """Get the lifetime of a media-streaming preview ticket.

    Priority:
        1. XAGENT_FILE_STREAM_TICKET_TTL_SECONDS environment variable
        2. Default of 600 (10 minutes)

    Kept independent of, and far shorter than, the user's own access token
    TTL: unlike a Bearer header, this credential rides in a URL a media
    element loads directly. The frontend never puts it anywhere a user could
    put it in the address bar, browser history, or a copied link, so the
    realistic exposure is proxy/CDN/server access logs and a devtools
    network panel -- a leaked or logged ticket should still stop being
    replayable long before a stolen access token would need to.

    Returns:
        Ticket lifetime in seconds.
    """
    env_value = os.getenv(FILE_STREAM_TICKET_TTL_SECONDS)
    if env_value is None or not env_value.strip():
        return 600

    try:
        ttl = int(env_value)
    except ValueError as exc:
        raise ValueError(
            f"Invalid {FILE_STREAM_TICKET_TTL_SECONDS} value: {env_value!r}."
        ) from exc

    if ttl <= 0:
        raise ValueError(
            f"Invalid {FILE_STREAM_TICKET_TTL_SECONDS} value: {env_value!r}. "
            "Value must be positive."
        )

    return ttl


def get_file_delivery_accel_redirect_enabled() -> bool:
    """Return whether private file endpoints may use nginx X-Accel-Redirect."""
    env_value = os.getenv(FILE_DELIVERY_ACCEL_REDIRECT_ENABLED)
    if env_value is None or not env_value.strip():
        return False

    normalized = env_value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False

    raise ValueError(
        f"Invalid {FILE_DELIVERY_ACCEL_REDIRECT_ENABLED} value: {env_value!r}. "
        "Expected a boolean value."
    )


def get_file_delivery_accel_redirect_prefix() -> str:
    """Get the internal nginx URI prefix used for X-Accel-Redirect."""
    env_value = os.getenv(FILE_DELIVERY_ACCEL_REDIRECT_PREFIX)
    prefix = (
        env_value.strip()
        if env_value is not None and env_value.strip()
        else "/_xagent_internal_files/"
    )
    if not prefix.startswith("/"):
        raise ValueError(
            f"Invalid {FILE_DELIVERY_ACCEL_REDIRECT_PREFIX} value: {env_value!r}. "
            "Value must start with '/'."
        )
    if not prefix.endswith("/"):
        prefix += "/"
    return prefix


def format_file_size(size_bytes: int) -> str:
    """Format a byte count for user-facing messages."""
    units = [("GB", 1024 * 1024 * 1024), ("MB", 1024 * 1024), ("KB", 1024)]

    for unit, divisor in units:
        value = size_bytes / divisor
        if value >= 0.9995:
            rounded = round(value, 1)
            if float(rounded).is_integer():
                return f"{int(rounded)}{unit}"
            return f"{rounded:.1f}{unit}"

    return f"{size_bytes}B"


def get_external_upload_dirs() -> list[Path]:
    """Get external upload directories from environment variable.

    The XAGENT_EXTERNAL_UPLOAD_DIRS environment variable should contain
    a comma-separated list of directory paths. Environment variables and
    ``~`` are expanded and relative paths are pinned to their first observed
    working directory. The returned path deliberately preserves its symlink
    spelling: Docker sibling-mode translation needs that backend-relative
    spelling, while file-access consumers resolve it in their own domain.

    Example: /path/to/uploads1,/path/to/uploads2

    Only directories that exist are included in the result.

    Returns:
        List of Path objects for existing external directories
    """
    env_dirs = os.getenv(EXTERNAL_UPLOAD_DIRS, "")
    if not env_dirs:
        return []

    result = []
    for dir_path in env_dirs.split(","):
        dir_path = dir_path.strip()
        if dir_path:
            expanded = Path(os.path.expandvars(dir_path)).expanduser()
            if expanded.is_absolute():
                absolute = expanded
            else:
                absolute = _pinned_relative_external_upload_dirs.setdefault(
                    dir_path, Path.cwd() / expanded
                )
            from .sandbox.base import canonical_sandbox_path

            canonical = Path(canonical_sandbox_path(str(absolute)))
            physical_configured = os.path.realpath(absolute)
            physical_canonical = os.path.realpath(canonical)
            if physical_configured != physical_canonical:
                raise ExternalUploadsDirConfigurationError(
                    f"External upload directory {dir_path!r} names two different "
                    "directories depending on normalization: lexically it is "
                    f"{str(canonical)!r}, resolving to {physical_canonical!r}, "
                    "while resolving the configured spelling directly gives "
                    f"{physical_configured!r}. A '..' segment after a symlink "
                    "does that; configure the directory you actually mean."
                )
            if canonical.is_dir():
                result.append(canonical)
            else:
                logger.warning(
                    "External upload directory does not exist or is not a directory: %r",
                    canonical,
                )

    return result


def get_external_skills_dirs() -> list[Path]:
    """Get external skills library directories from environment variable.

    The XAGENT_EXTERNAL_SKILLS_LIBRARY_DIRS environment variable should contain
    a comma-separated list of directory paths. Supports ~ expansion and environment
    variable expansion in paths.

    Example: ~/my-skills,/opt/skills,$PROJECT_DIR/skills

    Note: Unlike get_external_upload_dirs(), this includes all configured paths
    even if they don't exist yet. This allows users to configure skills directories
    before creating them.

    Returns:
        List of Path objects for external skills directories
    """
    env_dirs = os.getenv(EXTERNAL_SKILLS_LIBRARY_DIRS, "")
    if not env_dirs:
        return []

    result = []
    for dir_path in env_dirs.split(","):
        dir_path = dir_path.strip()
        if not dir_path:
            continue

        # Check for URL-like paths before path expansion
        if "://" in dir_path:
            logger.warning(f"Skipping non-local path (not supported yet): {dir_path}")
            continue

        # Expand environment variables and user home directory
        expanded_path = os.path.expanduser(os.path.expandvars(dir_path))
        path = Path(expanded_path)

        result.append(path)

    return result


def get_storage_root() -> Path:
    """Get the storage root directory path.

    Priority:
    1. XAGENT_STORAGE_ROOT environment variable
    2. Default to ~/.xagent

    Returns:
        Path object for storage root directory
    """
    env_dir = os.getenv(STORAGE_ROOT)
    if env_dir:
        # Expand ~ here so every consumer (including sqlite3, which opens a
        # literal ./~/... path) sees the same absolute location.
        return Path(env_dir).expanduser()

    # Default: ~/.xagent
    return Path.home() / ".xagent"


def get_sandbox_image() -> str:
    """Get the default sandbox image name.

    Priority:
    1. SANDBOX_IMAGE environment variable
    2. Default to xprobe/xagent-sandbox:latest

    Returns:
        Sandbox image name
    """
    return os.getenv(SANDBOX_IMAGE, "xprobe/xagent-sandbox:latest")


def get_sandbox_max_concurrency() -> int:
    """Maximum concurrent worker sandboxes per lifecycle.

    Priority:
        1. XAGENT_SANDBOX_MAX_CONCURRENCY environment variable
        2. Default ``3`` to match the default tool batch width

    Invalid or non-positive values fall back to the default.

    Returns:
        The per-lifecycle sandbox worker cap (>= 1).
    """
    return _get_positive_int_env(SANDBOX_MAX_CONCURRENCY, 3)


def _get_positive_float_env(env_var: str, default: float | None) -> float | None:
    value = os.getenv(env_var)
    if value is None or not value.strip():
        return default
    try:
        parsed = float(value)
    except ValueError:
        logger.warning("Invalid %s=%r; falling back to %s", env_var, value, default)
        return default
    # float() also parses "nan"/"inf"; nan comparisons are always False and
    # inf would make asyncio.sleep hang, so require a finite positive value.
    if not math.isfinite(parsed) or parsed <= 0:
        logger.warning("Invalid %s=%r; falling back to %s", env_var, value, default)
        return default
    return parsed


def get_sandbox_idle_ttl() -> float | None:
    """Idle TTL in seconds after which unreferenced sandboxes are reclaimed.

    Priority:
        1. XAGENT_SANDBOX_IDLE_TTL environment variable (seconds)
        2. Default ``None`` — idle reclamation disabled

    Invalid or non-positive values keep reclamation disabled.

    Reclamation deletes the container: anything written outside the
    bind-mounted workspace/uploads paths (e.g. lazily installed pip
    packages, files under ``/tmp`` or ``$HOME``) is lost. Workspace data on
    bind mounts survives, and the sandbox is transparently recreated on
    next use.

    Returns:
        TTL in seconds, or None when idle reclamation is disabled.
    """
    return _get_positive_float_env(SANDBOX_IDLE_TTL, None)


def get_sandbox_sweep_interval() -> float:
    """Interval in seconds between idle sandbox sweep runs.

    Priority:
        1. XAGENT_SANDBOX_SWEEP_INTERVAL environment variable (seconds)
        2. Default ``60``

    Only meaningful when XAGENT_SANDBOX_IDLE_TTL is set. Invalid or
    non-positive values fall back to the default.

    Returns:
        Sweep interval in seconds (> 0).
    """
    interval = _get_positive_float_env(SANDBOX_SWEEP_INTERVAL, None)
    return 60.0 if interval is None else interval


def get_sandbox_max_containers() -> int | None:
    """Maximum number of concurrently existing sandbox containers.

    Priority:
        1. XAGENT_SANDBOX_MAX_CONTAINERS environment variable
        2. Default ``None`` — no cap (previous behavior)

    The cap counts all managed containers including per-lifecycle workers;
    the transient warmup container is excluded. When the cap is reached,
    the least-recently-used idle sandbox is evicted to make room; if
    nothing is evictable the request fails with ``SandboxCapacityError``.
    Invalid or non-positive values keep the cap disabled.

    Returns:
        The container cap, or None when no cap is enforced.
    """
    # Sentinel default 0 (< minimum) maps every unset/invalid case to None.
    return _get_positive_int_env(SANDBOX_MAX_CONTAINERS, 0, minimum=1) or None


def get_sandbox_allow_local_fallback_on_capacity() -> bool:
    """Whether tasks may fall back to local (host) execution at capacity.

    Priority:
        1. XAGENT_SANDBOX_ALLOW_LOCAL_FALLBACK_ON_CAPACITY environment variable
        2. Default ``False`` — capacity exhaustion rejects the task

    By default a task that cannot get a sandbox because the container cap
    is reached is rejected with a clear error. Deployments that prefer
    availability over strict sandboxing can enable this to run such tasks
    on the host instead (with a warning log). Sandbox-service
    unavailability keeps its local fallback regardless of this setting.

    Returns:
        True when local fallback on capacity exhaustion is allowed.
    """
    return _get_bool_env(SANDBOX_ALLOW_LOCAL_FALLBACK_ON_CAPACITY, False)


def get_lancedb_path() -> Path:
    """Get the LanceDB database path.

    Priority:
    1. LANCEDB_PATH environment variable
    2. Default to STORAGE_ROOT/data/lancedb

    Returns:
        Path object for LanceDB directory
    """
    env_path = os.getenv(LANCEDB_PATH)
    if env_path:
        return Path(env_path)

    # Default: storage_root/data/lancedb
    return get_storage_root() / "data" / "lancedb"


def get_google_drive_download_timeout_seconds() -> int:
    """Get the maximum wait for a Google Drive long-running download.

    Native Google Workspace exports can return a pending Drive operation. This
    timeout bounds polling inside the cloud-ingest HTTP request. External proxy
    timeouts must also allow time for the final file transfer.

    Priority:
        1. XAGENT_GOOGLE_DRIVE_DOWNLOAD_TIMEOUT_SECONDS environment variable
        2. Default of 600 seconds

    Returns:
        Maximum operation wait in seconds.
    """
    return _get_positive_int_env(GOOGLE_DRIVE_DOWNLOAD_TIMEOUT_SECONDS, 600)


def get_kb_collections_timeout_seconds() -> int:
    """Get the deadline for a single knowledge base collection listing scan.

    Bounds one ``list_collections`` scan so the /api/kb/collections endpoint
    fails fast instead of hanging a request. The default is deliberately
    generous: the scan runs off the event loop in a worker thread, so a slow
    deployment now genuinely hits this deadline where previously the blocking
    scan starved the timer and always eventually succeeded. Measured cost is
    roughly 4s for a 400k-row table, so 30s keeps ~7x headroom while still
    bounding the request.

    Priority:
        1. XAGENT_KB_COLLECTIONS_TIMEOUT_SECONDS environment variable
        2. Default of 30 seconds

    Returns:
        Per-scan timeout in seconds
    """
    return _get_positive_int_env(KB_COLLECTIONS_TIMEOUT_SECONDS, 30)


def get_kb_search_timeout_seconds() -> int:
    """Get the deadline for searching a single knowledge base collection.

    Bounds one ``run_document_search`` call so an agent's knowledge_search does
    not wait forever on a stuck backend. The collections of one search run
    concurrently, so each of them holds a shared default-executor worker for as
    long as it runs; without a deadline an agent bound to N knowledge bases can
    pin N workers indefinitely. The default is generous because the budget has
    to cover embedding the query, the LanceDB scan and an optional rerank round
    trip.

    Known limitation, same as the collection listing endpoint: cancelling the
    ``asyncio.wait_for`` coroutine does not stop the underlying ``to_thread``
    worker, so a timed-out search keeps running to completion in the default
    executor. The deadline frees the caller, not the worker. A worker stuck in
    rerank outlives it by that model's own budget (``RerankModelConfig.timeout``,
    180s by default), so it can outlast this deadline threefold.

    Priority:
        1. XAGENT_KB_SEARCH_TIMEOUT_SECONDS environment variable
        2. Default of 60 seconds

    Returns:
        Per-collection search timeout in seconds
    """
    # ponytail: the rerank budget is not clamped to this one - clamping means
    # threading a per-call timeout through SearchConfig into the rerank adapter.
    # Do it if leaked workers actually starve the executor.
    return _get_positive_int_env(KB_SEARCH_TIMEOUT_SECONDS, 60)


def get_deepdoc_xinference_url() -> str | None:
    """Return the Xinference base URL that DeepDoc parsing is offloaded to.

    Leaving this unset keeps document parsing entirely local. It must
    otherwise be an absolute http:// or https:// base URL carrying no query
    or fragment, since request paths are appended to it. A malformed value
    raises rather than silently downgrading to local parsing, so the
    misconfiguration surfaces instead of showing up only as unexplained
    slowness.
    """
    return _reject_url_userinfo(
        DEEPDOC_XINFERENCE_URL, _normalized_http_env_url(DEEPDOC_XINFERENCE_URL)
    )


def get_deepdoc_xinference_api_key() -> str | None:
    """Return the API key for remote DeepDoc parsing, if one is configured.

    A dedicated key wins over the bare ``XINFERENCE_API_KEY`` shared with the
    other Xinference clients. Returning None is valid: a self-hosted
    Xinference deployment often runs without authentication.
    """
    value = (os.getenv(DEEPDOC_XINFERENCE_API_KEY) or "").strip()
    if value:
        return value
    return (os.getenv("XINFERENCE_API_KEY") or "").strip() or None


def get_deepdoc_xinference_timeout_seconds() -> int:
    """Return the read timeout for one remote DeepDoc document parse.

    Parsing a large PDF can take minutes, so the default matches the
    ``timeout=1800`` precedent in deepdoc-lib's own MinerU API client
    (``deepdoc/parser/mineru_parser.py``).
    """
    return _get_positive_int_env(DEEPDOC_XINFERENCE_TIMEOUT_SECONDS, 1800)


def get_deepdoc_xinference_model_uid() -> str:
    """Return the Xinference model UID that remote DeepDoc requests target.

    The OCR endpoint dispatches on this ``model`` form field, so it must name a
    launched DeepDoc model. ``DeepDoc`` is the model name Xinference registers
    the family under, which is also the UID a launch gets when none is chosen.
    """
    return (os.getenv(DEEPDOC_XINFERENCE_MODEL_UID) or "").strip() or "DeepDoc"


def get_deepdoc_xinference_username() -> str | None:
    """Return the username for the remote DeepDoc JWT exchange, if configured.

    Xinference clusters started with authentication mint a bearer token from
    ``POST /token``; deployments that instead issue a long-lived API key leave
    this unset and configure the key.
    """
    return (os.getenv(DEEPDOC_XINFERENCE_USERNAME) or "").strip() or None


def get_deepdoc_xinference_password() -> str | None:
    """Return the password for the remote DeepDoc JWT exchange, if configured."""
    return os.getenv(DEEPDOC_XINFERENCE_PASSWORD) or None


def get_default_sqlite_db_path() -> str:
    """Get the default SQLite database file path string.

    Returns:
        Path string for SQLite database file in storage root
    """
    # The original implementation in manager.py returned str
    # So we need to convert it to str here
    storage_root = get_storage_root()
    return str(storage_root / "xagent.db")


def get_database_url() -> str:
    """Get the database URL.

    Priority:
    1. DATABASE_URL environment variable (full connection string)
    2. Default to SQLite in storage root

    Returns:
        Database connection string
    """
    database_url = os.getenv(DATABASE_URL)
    if database_url is not None:
        return database_url

    # Default: SQLite in storage root
    db_path = get_default_sqlite_db_path()
    return f"sqlite:///{db_path}"


def get_db_pool_size() -> int:
    """Get the SQLAlchemy connection pool size for the shared web engine.

    Priority:
        1. XAGENT_DB_POOL_SIZE environment variable
        2. 10

    Returns:
        Number of persistent connections kept in the pool per process.
    """
    return _get_positive_int_env(DB_POOL_SIZE, 10)


def get_db_max_overflow() -> int:
    """Get the SQLAlchemy connection pool max overflow for the shared web engine.

    Priority:
        1. XAGENT_DB_MAX_OVERFLOW environment variable
        2. 20

    Returns:
        Number of extra connections allowed beyond the pool size (0 disables
        overflow).
    """
    return _get_positive_int_env(DB_MAX_OVERFLOW, 20, minimum=0)


def get_db_pool_timeout_seconds() -> int:
    """Get the timeout for acquiring a connection from the pool, in seconds.

    Priority:
        1. XAGENT_DB_POOL_TIMEOUT_SECONDS environment variable
        2. 30

    Returns:
        Seconds to wait for a free pooled connection before raising.
    """
    return _get_positive_int_env(DB_POOL_TIMEOUT_SECONDS, 30)


def get_db_pool_kwargs() -> dict[str, Any]:
    """Shared SQLAlchemy pool kwargs for every pooled (non-SQLite) engine.

    Single source for the pool sizing/health knobs so the shared web engine
    and the ad-hoc storage engine cannot drift apart. Note the same values
    apply to EACH engine that uses them: a process running both holds up to
    2 x (pool_size + max_overflow) connections (see example.env).

    Returns:
        Keyword arguments for :func:`sqlalchemy.create_engine`.
    """
    return {
        "pool_size": get_db_pool_size(),
        "max_overflow": get_db_max_overflow(),
        "pool_timeout": get_db_pool_timeout_seconds(),
        "pool_recycle": 3600,  # Recycle connections after 1 hour
        "pool_pre_ping": True,  # Verify connections before using
    }


def get_mcp_tool_init_timeout_seconds() -> int:
    """Get the per-server timeout for MCP tool initialization, in seconds.

    Bounds the connect + initialize + list-tools handshake for one MCP server
    during agent setup, so a hung server cannot stall task startup (and pin
    resources such as DB connections) indefinitely. A timed-out server is
    skipped; the remaining servers still load.

    Priority:
        1. XAGENT_MCP_TOOL_INIT_TIMEOUT_SECONDS environment variable
        2. 60

    Returns:
        Seconds allowed per MCP server; 0 disables the timeout.
    """
    return _get_positive_int_env(MCP_TOOL_INIT_TIMEOUT_SECONDS, 60, minimum=0)


def get_sandbox_cpus() -> int | None:
    """Get the CPU count for sandbox containers.

    Returns:
        CPU count from SANDBOX_CPUS env var, or None
    """
    env_str = os.getenv(SANDBOX_CPUS)
    if env_str:
        try:
            return int(env_str)
        except ValueError:
            logger.warning(f"Invalid {SANDBOX_CPUS} value: {env_str}")
    return None


def get_sandbox_memory() -> int | None:
    """Get the memory limit for sandbox containers (in MB).

    Returns:
        Memory value from SANDBOX_MEMORY env var, or None
    """
    env_str = os.getenv(SANDBOX_MEMORY)
    if env_str:
        try:
            return int(env_str)
        except ValueError:
            logger.warning(f"Invalid {SANDBOX_MEMORY} value: {env_str}")
    return None


def get_sandbox_env() -> dict[str, str]:
    """Get the environment variables for sandbox containers.

    Format: KEY1=value1;KEY2=value2

    Returns:
        Dictionary of environment variables
    """
    env_str = os.getenv(SANDBOX_ENV, "").strip()
    if not env_str:
        return {}

    env = {}
    for pair in env_str.split(";"):
        try:
            key, value = pair.strip().split("=", 1)
        except ValueError:
            logger.warning("Invalid sandbox env config: must be in KEY=value format")
            continue

        key = key.strip()
        value = value.strip()
        if key and value:
            env[key] = value
        elif not key:
            logger.warning("Environment variable has empty key")
        elif not value:
            logger.warning(f"Environment variable {key!r} has empty value")

    return env


def get_sandbox_volumes(
    *, host_side_sources: bool = False
) -> list[tuple[str, str, str]]:
    """Get the volume mappings for sandbox containers.

    Format: src:dst[:mode];src2:dst2[:mode2]
    - src: source path on host
    - dst: destination path in container
    - mode: ro or rw (default: ro)

    Args:
        host_side_sources: When True, source paths are already Docker-host paths.
            Only environment variables are expanded; relative paths and ``~`` are
            rejected instead of being normalized inside the backend container.

    Returns:
        List of (src, dst, mode) tuples
    """
    env_str = os.getenv(SANDBOX_VOLUMES, "").strip()
    if not env_str:
        return []

    volumes = []
    for item in env_str.split(";"):
        item = item.strip()
        if not item:
            continue

        parts = item.split(":", 2)
        if len(parts) < 2:
            logger.warning(f"Invalid sandbox volume config: {item}")
            continue

        src = os.path.expandvars(parts[0].strip())
        dst = parts[1].strip()
        if not src or not dst:
            logger.warning(f"Invalid sandbox volume: {item}")
            continue

        if host_side_sources:
            if src.startswith("~") or not Path(src).is_absolute():
                logger.warning(
                    "Invalid sandbox host volume source in Docker sibling mode: %s",
                    item,
                )
                continue
        else:
            # Normalize paths to resolve any relative components
            src = os.path.abspath(os.path.expanduser(src))

        mode = parts[2].strip().lower() if len(parts) > 2 else "ro"
        if mode not in ("ro", "rw"):
            logger.warning(f"Invalid sandbox volume mode: {item}, using 'ro'")
            mode = "ro"

        volumes.append((src, dst, mode))

    return volumes


def get_sandbox_host_project_root() -> Path | None:
    """Get the host project root used for Docker sibling sandbox code mounts.

    Priority:
    1. XAGENT_SANDBOX_HOST_PROJECT_ROOT environment variable
    2. None, which lets callers use their local runtime project root

    Returns:
        Path to the project root as resolved from the Docker host's perspective,
        or None when not configured.
    """
    env_str = os.getenv(SANDBOX_HOST_PROJECT_ROOT)
    if env_str:
        return Path(os.path.expandvars(env_str.strip()))
    return None


# Read once at import, never per call: the sandbox runner inherits this marker
# before it imports anything, while agent-authored code runs late and must not
# be able to flip host registration into sandbox mode process-wide.
_IN_SANDBOX_TOOL_RUNNER = _get_bool_env(SANDBOX_TOOL_RUNNER, False)


def in_sandbox_tool_runner() -> bool:
    """Return whether this process is the sandbox tool runner.

    Priority:
    1. XAGENT_SANDBOX_TOOL_RUNNER as it stood at process start
    2. False, the host process default

    Returns:
        True when running inside the sandbox, where no database or object
        storage credentials are available.
    """
    return _IN_SANDBOX_TOOL_RUNNER


def get_sandbox_host_storage_root() -> Path | None:
    """Get the Docker host storage root used for sibling sandbox bind mounts.

    Priority:
    1. XAGENT_SANDBOX_HOST_STORAGE_ROOT environment variable
    2. None, which lets callers use backend paths directly

    Returns:
        Path to the Xagent storage root as seen by the Docker host, or None when
        not configured.
    """
    env_str = os.getenv(SANDBOX_HOST_STORAGE_ROOT)
    if env_str:
        return Path(os.path.expandvars(env_str.strip()))
    return None


_SANDBOX_NAMESPACE_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


def validate_sandbox_namespace(namespace: str) -> None:
    """Validate a sandbox ownership namespace.

    Enforces the Docker Compose project-name grammar (lowercase letters,
    decimal digits, dashes and underscores, beginning with a lowercase letter
    or digit). Callers that accept a namespace from outside the environment
    (e.g. the Docker sandbox service constructor) must run this so a
    malformed value can never recreate a shared ownership domain.

    Raises:
        ValueError: The value does not match the grammar.
    """
    if not _SANDBOX_NAMESPACE_RE.fullmatch(namespace):
        raise ValueError(
            f"Invalid sandbox namespace {namespace!r}: must match the Docker "
            "Compose project-name grammar (lowercase letters, digits, dashes, "
            "underscores; start with a letter or digit)"
        )


def get_sandbox_namespace() -> str | None:
    """Get the stable per-deployment namespace for sandbox resources.

    The namespace is the ownership boundary between deployments that share one
    Docker daemon: every sandbox container (physical name and owner labels)
    a deployment creates is scoped to it, and every lookup/list/cleanup
    operation is restricted to it. It must be stable across restarts and
    unique per deployment; the Docker Compose project name
    (``COMPOSE_PROJECT_NAME``) is the canonical source.

    Accepts the Docker Compose project-name grammar: lowercase letters,
    decimal digits, dashes and underscores, beginning with a lowercase letter
    or digit.

    Returns:
        The configured namespace, or None when unset/empty.

    Raises:
        ValueError: The variable is set but does not match the grammar.
    """
    raw = os.getenv(SANDBOX_NAMESPACE, "").strip()
    if not raw:
        return None
    validate_sandbox_namespace(raw)
    return raw


def get_boxlite_home_dir() -> Path | None:
    """Get the BoxLite home directory path.

    Returns:
        Path from BOXLITE_HOME_DIR env var, or None
    """
    env_str = os.getenv(BOXLITE_HOME_DIR)
    if env_str:
        return Path(env_str)
    return None


def get_tool_max_output_length() -> int:
    """Get the maximum per-string output length for tools.

    This limit applies to individual string values within the output structure,
    not the total output size. The total output size is indirectly controlled
    by the combination of per-string limit, max field count, and max recursion depth.

    Returns:
        Maximum per-string length from TOOL_MAX_OUTPUT_LENGTH env var, or 50k by default
    """
    env_str = os.getenv(TOOL_MAX_OUTPUT_LENGTH)
    if env_str:
        try:
            return int(env_str)
        except ValueError:
            logger.warning("Invalid TOOL_MAX_OUTPUT_LENGTH value: {env_str}")
    return 50 * 1024


def get_web_search_provider() -> str:
    """Get the preferred web search provider.

    Priority:
        1. XAGENT_WEB_SEARCH_PROVIDER environment variable
        2. "auto"

    Valid values are: auto, google, tavily, exa, zhipu.
    """
    provider = (os.getenv(WEB_SEARCH_PROVIDER) or "auto").strip().lower()
    if provider in WEB_SEARCH_PROVIDERS:
        return provider

    logger.warning(
        "Invalid %s value: %r. Falling back to 'auto'.",
        WEB_SEARCH_PROVIDER,
        provider,
    )
    return "auto"


def get_web_crawl_tls_impersonate() -> str | None:
    """Get the optional TLS impersonation spec for website crawling.

    Priority:
        1. XAGENT_WEB_CRAWL_TLS_IMPERSONATE environment variable
        2. None (plain httpx)

    Values:
        - unset, empty, "none", or "null": None
        - "auto": built-in crawler fallback chain
        - any other non-empty value: curl_cffi impersonate spec
    """
    value = os.getenv(WEB_CRAWL_TLS_IMPERSONATE)
    if value is None:
        return None

    normalized = value.strip()
    if not normalized or normalized.lower() in {"none", "null"}:
        return None
    return normalized


def get_tool_max_recursion_depth() -> int:
    """Get the maximum recursion depth for tools.

    Returns:
        Maximum recursion depth from TOOL_MAX_RECURSION_DEPTH env var, or 20 by default.
        20 layers is sufficient for most real-world data structures while preventing
        excessively deep nesting that could cause performance issues.
    """
    env_str = os.getenv(TOOL_MAX_RECURSION_DEPTH)
    if env_str:
        try:
            return int(env_str)
        except ValueError:
            logger.warning("Invalid TOOL_MAX_RECURSION_DEPTH value: {env_str}")
    return 20


def get_tool_max_field_count() -> int:
    """Get the maximum number of fields/items in dict/list for tools.

    This helps control total output size by limiting the cardinality of
    collections. Combined with per-string length and recursion depth limits,
    it provides reasonable protection against excessive output without
    requiring expensive total size calculation.

    Returns:
        Maximum fields from TOOL_MAX_FIELD_COUNT env var, or 1000 by default
    """
    env_str = os.getenv(TOOL_MAX_FIELD_COUNT)
    if env_str:
        try:
            return int(env_str)
        except ValueError:
            logger.warning("Invalid TOOL_MAX_FIELDS value: {env_str}")
    return 1000


def get_max_trace_payload_bytes() -> int:
    """Max byte size for individual trace payload fields (e.g. data.messages,
    data.response) before truncation.

    Applies to the LLM I/O audit trace: a long DAG task hitting all 9 audit
    sites can otherwise write multi-MB rows into trace_events. Also bounds
    the rendered size of every trace category's console log line, not only
    LLM audit events.

    Priority:
        1. XAGENT_MAX_TRACE_PAYLOAD_BYTES env var
        2. Default 50_000 (~50KB, large enough for typical compacted
           messages while bounding worst case)

    Returns:
        Maximum bytes per truncated trace field.
    """
    env_str = os.getenv(MAX_TRACE_PAYLOAD_BYTES)
    if env_str:
        try:
            value = int(env_str)
            if value < 0:
                logger.warning(
                    f"Invalid {MAX_TRACE_PAYLOAD_BYTES} value (negative): {env_str!r}"
                )
            else:
                return value
        except ValueError:
            logger.warning(f"Invalid {MAX_TRACE_PAYLOAD_BYTES} value: {env_str!r}")
    return 50_000
