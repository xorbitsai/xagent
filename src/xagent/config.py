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
TASK_LEASE_TTL_SECONDS = "XAGENT_TASK_LEASE_TTL_SECONDS"
TASK_LEASE_HEARTBEAT_SECONDS = "XAGENT_TASK_LEASE_HEARTBEAT_SECONDS"
STORAGE_ROOT = "XAGENT_STORAGE_ROOT"
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
SANDBOX_IMAGE = "SANDBOX_IMAGE"
LANCEDB_PATH = "LANCEDB_PATH"
DATABASE_URL = "DATABASE_URL"
SANDBOX_CPUS = "SANDBOX_CPUS"
SANDBOX_MEMORY = "SANDBOX_MEMORY"
SANDBOX_ENV = "SANDBOX_ENV"
SANDBOX_VOLUMES = "SANDBOX_VOLUMES"
SANDBOX_HOST_PROJECT_ROOT = "XAGENT_SANDBOX_HOST_PROJECT_ROOT"
SANDBOX_HOST_STORAGE_ROOT = "XAGENT_SANDBOX_HOST_STORAGE_ROOT"
SANDBOX_MAX_CONCURRENCY = "XAGENT_SANDBOX_MAX_CONCURRENCY"
SANDBOX_IDLE_TTL = "XAGENT_SANDBOX_IDLE_TTL"
SANDBOX_SWEEP_INTERVAL = "XAGENT_SANDBOX_SWEEP_INTERVAL"
SANDBOX_MAX_CONTAINERS = "XAGENT_SANDBOX_MAX_CONTAINERS"
SANDBOX_ALLOW_LOCAL_FALLBACK_ON_CAPACITY = (
    "XAGENT_SANDBOX_ALLOW_LOCAL_FALLBACK_ON_CAPACITY"
)
BOXLITE_HOME_DIR = "BOXLITE_HOME_DIR"
WEB_SEARCH_PROVIDER = "XAGENT_WEB_SEARCH_PROVIDER"
WEB_CRAWL_TLS_IMPERSONATE = "XAGENT_WEB_CRAWL_TLS_IMPERSONATE"
TOOL_PARALLEL_ENABLED = "XAGENT_TOOL_PARALLEL_ENABLED"
TOOL_MAX_CONCURRENCY = "XAGENT_TOOL_MAX_CONCURRENCY"
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
TRIGGER_DISPATCHER_ENABLED = "XAGENT_TRIGGER_DISPATCHER_ENABLED"
TRIGGER_DISPATCHER_INTERVAL_SECONDS = "XAGENT_TRIGGER_DISPATCHER_INTERVAL_SECONDS"
TRIGGER_DISPATCHER_BATCH_SIZE = "XAGENT_TRIGGER_DISPATCHER_BATCH_SIZE"
TRIGGER_CALLBACK_RATE_LIMIT = "XAGENT_TRIGGER_CALLBACK_RATE_LIMIT"
TRIGGER_CALLBACK_IP_RATE_LIMIT = "XAGENT_TRIGGER_CALLBACK_IP_RATE_LIMIT"
TRIGGER_CRUD_RATE_LIMIT = "XAGENT_TRIGGER_CRUD_RATE_LIMIT"
TRUSTED_PROXY_HOPS = "XAGENT_TRUSTED_PROXY_HOPS"
GMAIL_PUBSUB_PROJECT_ID = "XAGENT_GMAIL_PUBSUB_PROJECT_ID"
GMAIL_PUBSUB_TOPIC_PREFIX = "XAGENT_GMAIL_PUBSUB_TOPIC_PREFIX"
GMAIL_PUBSUB_SUBSCRIPTION_PREFIX = "XAGENT_GMAIL_PUBSUB_SUBSCRIPTION_PREFIX"
GMAIL_PUBSUB_PUSH_SERVICE_ACCOUNT = "XAGENT_GMAIL_PUBSUB_PUSH_SERVICE_ACCOUNT"
GMAIL_REGISTRATION_TIMEOUT_SECONDS = "XAGENT_GMAIL_REGISTRATION_TIMEOUT_SECONDS"
PUBLIC_API_BASE_URL = "XAGENT_PUBLIC_API_BASE_URL"
GMAIL_WATCH_ENABLED = "XAGENT_GMAIL_WATCH_ENABLED"
GMAIL_WATCH_RENEWAL_INTERVAL_SECONDS = "XAGENT_GMAIL_WATCH_RENEWAL_INTERVAL_SECONDS"
GMAIL_WATCH_RENEWAL_LEAD_SECONDS = "XAGENT_GMAIL_WATCH_RENEWAL_LEAD_SECONDS"
PASSWORD_RESET_EXPIRE_MINUTES = "XAGENT_PASSWORD_RESET_EXPIRE_MINUTES"
APP_BASE_URL = "XAGENT_APP_BASE_URL"
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
MCP_OAUTH_ALLOW_PRIVATE_HOSTS = "XAGENT_MCP_OAUTH_ALLOW_PRIVATE_HOSTS"
MCP_OAUTH_PROXY_URL = "XAGENT_MCP_OAUTH_PROXY_URL"

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


def get_password_reset_expire_minutes() -> int:
    """Return the password reset token expiry window in minutes."""
    return _get_positive_int_env(PASSWORD_RESET_EXPIRE_MINUTES, 30)


def get_app_base_url() -> str | None:
    """Return the trusted frontend base URL used in email links."""
    value = os.getenv(APP_BASE_URL)
    if value is None:
        return None
    value = value.strip()
    return value.rstrip("/") or None


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


def get_mcp_oauth_allow_private_hosts() -> bool:
    """Return whether MCP OAuth URL policy may target local/private hosts.

    This is intended only for local development with local authorization
    servers. Production deployments should leave it disabled.
    """
    return _get_bool_env(MCP_OAUTH_ALLOW_PRIVATE_HOSTS, False)


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
    """Return the age after which non-terminal jobs should be requeued."""
    return _get_positive_int_env(BACKGROUND_JOB_STALE_SECONDS, 7200, minimum=60)


def get_background_job_sweep_interval_seconds() -> int:
    """Return how often the scheduler scans for stale background jobs."""
    return _get_positive_int_env(
        BACKGROUND_JOB_SWEEP_INTERVAL_SECONDS,
        300,
        minimum=30,
    )


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
    """Public base URL of the backend API for provider callbacks.

    Priority:
        1. XAGENT_PUBLIC_API_BASE_URL environment variable
        2. None (Gmail registration fails explicitly)

    Deliberately separate from XAGENT_APP_BASE_URL (the frontend URL used in
    e.g. password-reset emails): Pub/Sub OIDC audiences must match backend
    callback URLs, so the frontend base URL is never a valid substitute.
    """
    value = (os.getenv(PUBLIC_API_BASE_URL) or "").strip()
    return value.rstrip("/") or None


def get_gmail_watch_enabled() -> bool:
    """Return whether Gmail automatic watch registration is enabled."""
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


def get_uploads_dir() -> Path:
    """Get the uploads directory path.

    Priority:
    1. XAGENT_UPLOADS_DIR environment variable
    2. Default to WEB_DIR/uploads for backward compatibility

    Returns:
        Path object for uploads directory
    """
    env_dir = os.getenv(UPLOADS_DIR)
    if env_dir:
        return Path(env_dir)

    # Default: web/uploads
    web_dir = get_web_dir()
    return web_dir / "uploads"


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
    a comma-separated list of directory paths.

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
            path = Path(dir_path)
            if path.is_dir():
                result.append(path)
            else:
                logger.warning(
                    "External upload directory does not exist or is not a directory: %r",
                    path,
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
        return Path(env_dir)

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

    Applies to the LLM I/O audit trace added in fix/llm-trace-coverage. A
    long DAG task hitting all 9 audit sites can otherwise write multi-MB
    rows into trace_events.

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
