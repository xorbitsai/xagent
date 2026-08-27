"""Unit tests for core/config.py configuration functions."""

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from tempfile import gettempdir

import pytest

import xagent.config as config
from xagent.config import (
    AGENT_RUNTIME,
    APP_BASE_URL,
    BACKGROUND_JOB_MAX_RETRIES,
    BACKGROUND_JOB_STALE_SECONDS,
    BACKGROUND_JOB_SWEEP_INTERVAL_SECONDS,
    BACKGROUND_JOB_VISIBILITY_TIMEOUT_SECONDS,
    BOXLITE_HOME_DIR,
    BROWSER_CUA_DRIVER_COMMAND,
    BROWSER_CUA_DRIVER_MAX_ELEMENTS,
    BROWSER_CUA_DRIVER_SOCKET,
    BROWSER_CUA_DRIVER_TIMEOUT_SECONDS,
    CELERY_BROKER_URL,
    CELERY_ENABLED,
    CELERY_RESULT_BACKEND,
    DATABASE_URL,
    DB_MAX_OVERFLOW,
    DB_POOL_SIZE,
    DB_POOL_TIMEOUT_SECONDS,
    DEEPDOC_XINFERENCE_API_KEY,
    DEEPDOC_XINFERENCE_MODEL_UID,
    DEEPDOC_XINFERENCE_PASSWORD,
    DEEPDOC_XINFERENCE_TIMEOUT_SECONDS,
    DEEPDOC_XINFERENCE_URL,
    DEEPDOC_XINFERENCE_USERNAME,
    EXTERNAL_SKILLS_LIBRARY_DIRS,
    EXTERNAL_UPLOAD_DIRS,
    FILE_DELIVERY_ACCEL_REDIRECT_ENABLED,
    FILE_DELIVERY_ACCEL_REDIRECT_PREFIX,
    FILE_DELIVERY_REDIRECT_ENABLED,
    FILE_DELIVERY_SIGNED_URL_TTL_SECONDS,
    FILE_MATERIALIZE_DIR,
    FILE_STORAGE_OPTIONS,
    FILE_STORAGE_STARTUP_SYNC_ENABLED,
    FILE_STORAGE_URI,
    FILE_STREAM_TICKET_TTL_SECONDS,
    FRONTEND_DIST_DIR,
    GMAIL_PUBSUB_PROJECT_ID,
    GMAIL_PUBSUB_PUSH_SERVICE_ACCOUNT,
    GMAIL_PUBSUB_SUBSCRIPTION_PREFIX,
    GMAIL_PUBSUB_TOPIC_PREFIX,
    GMAIL_REGISTRATION_TIMEOUT_SECONDS,
    HOT_PATH_CACHE_ENABLED,
    HOT_PATH_CACHE_TTL_SECONDS,
    HOT_PATH_TASK_CACHE_TTL_SECONDS,
    KB_COLLECTIONS_TIMEOUT_SECONDS,
    KB_SEARCH_TIMEOUT_SECONDS,
    LANCEDB_PATH,
    MAX_TRACE_PAYLOAD_BYTES,
    MAX_UPLOAD_SIZE,
    MCP_OAUTH_ALLOW_PRIVATE_HOSTS,
    MCP_OAUTH_PROXY_URL,
    MCP_TOOL_INIT_TIMEOUT_SECONDS,
    NATIVE_BROWSER_APP_NAME,
    NATIVE_BROWSER_ENABLED,
    OPENROUTER_OFFICIAL_PROVIDERS_ONLY,
    PASSWORD_RESET_EXPIRE_MINUTES,
    PREVIEW_TMP_DIR,
    PUBLIC_API_BASE_URL,
    REDIS_URL,
    S2S_API_BASE_URL,
    SANDBOX_ALLOW_LOCAL_FALLBACK_ON_CAPACITY,
    SANDBOX_CPUS,
    SANDBOX_ENV,
    SANDBOX_HOST_PROJECT_ROOT,
    SANDBOX_HOST_STORAGE_ROOT,
    SANDBOX_IDLE_TTL,
    SANDBOX_IMAGE,
    SANDBOX_MAX_CONTAINERS,
    SANDBOX_MEMORY,
    SANDBOX_NAMESPACE,
    SANDBOX_SWEEP_INTERVAL,
    SANDBOX_TOOL_RUNNER,
    SANDBOX_VOLUMES,
    SLACK_APP_TOKEN,
    SLACK_CLIENT_ID,
    SLACK_CLIENT_SECRET,
    SLACK_REDIRECT_URI,
    SMTP_FROM_EMAIL,
    SMTP_FROM_NAME,
    SMTP_HOST,
    SMTP_PASSWORD,
    SMTP_PORT,
    SMTP_USE_SSL,
    SMTP_USE_TLS,
    SMTP_USERNAME,
    STORAGE_ROOT,
    TASK_LEASE_RECOVERY_BATCH_SIZE,
    TASK_LEASE_RECOVERY_INTERVAL_SECONDS,
    TASK_LEASE_TTL_SECONDS,
    TASK_RUNTIME_HOOK_MAX_WORKERS,
    TASK_RUNTIME_HOOK_QUEUE_TIMEOUT_SECONDS,
    TEMP_FILE_CLEANUP_SHUTDOWN_TIMEOUT_SECONDS,
    TRIGGER_DISPATCHER_BATCH_SIZE,
    TRIGGER_DISPATCHER_ENABLED,
    TRIGGER_DISPATCHER_INTERVAL_SECONDS,
    TRUSTED_EGRESS_PROXY,
    UPLOADED_FILE_RECOVERY_BATCH_SIZE,
    UPLOADED_FILE_RECOVERY_INTERVAL_SECONDS,
    UPLOADED_FILE_RECOVERY_STALE_SECONDS,
    UPLOADS_DIR,
    WEB_CRAWL_TLS_IMPERSONATE,
    WEB_DIR,
    WEB_SEARCH_PROVIDER,
    ExternalUploadsDirConfigurationError,
    format_file_size,
    get_agent_pattern_for_execution_mode,
    get_agent_runtime,
    get_app_base_url,
    get_background_job_max_retries,
    get_background_job_stale_seconds,
    get_background_job_sweep_interval_seconds,
    get_background_job_visibility_timeout_seconds,
    get_boxlite_home_dir,
    get_browser_cua_driver_command,
    get_browser_cua_driver_max_elements,
    get_browser_cua_driver_socket,
    get_browser_cua_driver_timeout_seconds,
    get_celery_broker_url,
    get_celery_enabled,
    get_celery_result_backend,
    get_database_url,
    get_db_max_overflow,
    get_db_pool_size,
    get_db_pool_timeout_seconds,
    get_deepdoc_xinference_api_key,
    get_deepdoc_xinference_model_uid,
    get_deepdoc_xinference_password,
    get_deepdoc_xinference_timeout_seconds,
    get_deepdoc_xinference_url,
    get_deepdoc_xinference_username,
    get_default_sqlite_db_path,
    get_default_task_execution_mode,
    get_external_skills_dirs,
    get_external_upload_dirs,
    get_file_delivery_accel_redirect_enabled,
    get_file_delivery_accel_redirect_prefix,
    get_file_delivery_redirect_enabled,
    get_file_delivery_signed_url_ttl_seconds,
    get_file_materialize_dir,
    get_file_storage_options,
    get_file_storage_startup_sync_enabled,
    get_file_storage_uri,
    get_file_stream_ticket_ttl_seconds,
    get_frontend_dist_dir,
    get_gmail_pubsub_project_id,
    get_gmail_pubsub_push_service_account,
    get_gmail_pubsub_subscription_prefix,
    get_gmail_pubsub_topic_prefix,
    get_gmail_registration_timeout_seconds,
    get_hot_path_cache_enabled,
    get_hot_path_cache_ttl_seconds,
    get_hot_path_task_cache_ttl_seconds,
    get_kb_collections_timeout_seconds,
    get_kb_search_timeout_seconds,
    get_lancedb_path,
    get_max_trace_payload_bytes,
    get_max_upload_size_bytes,
    get_mcp_oauth_allow_private_hosts,
    get_mcp_oauth_proxy_url,
    get_mcp_tool_init_timeout_seconds,
    get_native_browser_app_name,
    get_native_browser_enabled,
    get_openrouter_official_providers_only,
    get_password_reset_expire_minutes,
    get_preview_tmp_dir,
    get_public_api_base_url,
    get_redis_url,
    get_s2s_api_base_url,
    get_sandbox_allow_local_fallback_on_capacity,
    get_sandbox_cpus,
    get_sandbox_env,
    get_sandbox_host_project_root,
    get_sandbox_host_storage_root,
    get_sandbox_idle_ttl,
    get_sandbox_image,
    get_sandbox_max_containers,
    get_sandbox_memory,
    get_sandbox_namespace,
    get_sandbox_sweep_interval,
    get_sandbox_volumes,
    get_slack_app_token,
    get_slack_client_id,
    get_slack_client_secret,
    get_slack_oauth_redirect_uri,
    get_smtp_from_email,
    get_smtp_from_name,
    get_smtp_host,
    get_smtp_password,
    get_smtp_port,
    get_smtp_use_ssl,
    get_smtp_use_tls,
    get_smtp_username,
    get_storage_root,
    get_task_lease_recovery_batch_size,
    get_task_lease_recovery_interval_seconds,
    get_task_runtime_hook_max_workers,
    get_task_runtime_hook_queue_timeout_seconds,
    get_temp_file_cleanup_shutdown_timeout_seconds,
    get_trigger_dispatcher_batch_size,
    get_trigger_dispatcher_enabled,
    get_trigger_dispatcher_interval_seconds,
    get_trusted_egress_proxy_enabled,
    get_uploaded_file_recovery_batch_size,
    get_uploaded_file_recovery_interval_seconds,
    get_uploaded_file_recovery_stale_seconds,
    get_uploads_dir,
    get_web_crawl_tls_impersonate,
    get_web_dir,
    get_web_search_provider,
    in_sandbox_tool_runner,
    validate_sandbox_namespace,
)


class TestEnvironmentVariableConstants:
    """Test environment variable constant names."""

    def test_upload_dir_constant(self):
        assert UPLOADS_DIR == "XAGENT_UPLOADS_DIR"

    def test_web_dir_constant(self):
        assert WEB_DIR == "XAGENT_WEB_DIR"

    def test_frontend_dist_dir_constant(self):
        assert FRONTEND_DIST_DIR == "XAGENT_FRONTEND_DIST_DIR"

    def test_external_upload_dirs_constant(self):
        assert EXTERNAL_UPLOAD_DIRS == "XAGENT_EXTERNAL_UPLOAD_DIRS"

    def test_external_skills_dirs_constant(self):
        assert EXTERNAL_SKILLS_LIBRARY_DIRS == "XAGENT_EXTERNAL_SKILLS_LIBRARY_DIRS"

    def test_agent_runtime_constant(self):
        assert AGENT_RUNTIME == "XAGENT_AGENT_RUNTIME"

    def test_storage_root_constant(self):
        assert STORAGE_ROOT == "XAGENT_STORAGE_ROOT"

    def test_local_browser_constants(self):
        assert NATIVE_BROWSER_ENABLED == "XAGENT_NATIVE_BROWSER_ENABLED"
        assert NATIVE_BROWSER_APP_NAME == "XAGENT_NATIVE_BROWSER_APP_NAME"
        assert BROWSER_CUA_DRIVER_COMMAND == "XAGENT_BROWSER_CUA_DRIVER_COMMAND"
        assert BROWSER_CUA_DRIVER_SOCKET == "XAGENT_BROWSER_CUA_DRIVER_SOCKET"
        assert (
            BROWSER_CUA_DRIVER_TIMEOUT_SECONDS
            == "XAGENT_BROWSER_CUA_DRIVER_TIMEOUT_SECONDS"
        )
        assert (
            BROWSER_CUA_DRIVER_MAX_ELEMENTS == "XAGENT_BROWSER_CUA_DRIVER_MAX_ELEMENTS"
        )

    def test_sandbox_image_constant(self):
        assert SANDBOX_IMAGE == "SANDBOX_IMAGE"

    def test_sandbox_host_project_root_constant(self):
        assert SANDBOX_HOST_PROJECT_ROOT == "XAGENT_SANDBOX_HOST_PROJECT_ROOT"

    def test_sandbox_namespace_constant(self):
        assert SANDBOX_NAMESPACE == "XAGENT_SANDBOX_NAMESPACE"

    def test_sandbox_host_storage_root_constant(self):
        assert SANDBOX_HOST_STORAGE_ROOT == "XAGENT_SANDBOX_HOST_STORAGE_ROOT"

    def test_sandbox_tool_runner_constant(self):
        assert SANDBOX_TOOL_RUNNER == "XAGENT_SANDBOX_TOOL_RUNNER"

    def test_in_sandbox_tool_runner_defaults_to_false(self):
        assert in_sandbox_tool_runner() is False

    def test_in_sandbox_tool_runner_reads_the_env_at_process_start(self, monkeypatch):
        for value, expected in (
            ("1", True),
            ("true", True),
            ("yes", True),
            ("on", True),
            ("0", False),
            ("false", False),
            ("no", False),
            ("off", False),
            ("", False),
        ):
            monkeypatch.setenv(SANDBOX_TOOL_RUNNER, value)
            assert config._get_bool_env(SANDBOX_TOOL_RUNNER, False) is expected

    def test_in_sandbox_tool_runner_ignores_later_env_mutation(self, monkeypatch):
        """Agent-authored code runs late; it must not flip the whole process."""
        monkeypatch.setenv(SANDBOX_TOOL_RUNNER, "1")
        assert in_sandbox_tool_runner() is False

    def test_in_sandbox_tool_runner_does_read_the_env_at_import(self):
        """The other tests patch the snapshot, so pin that it is populated."""
        probe = "import xagent.config as c; print(c.in_sandbox_tool_runner())"
        env = {**os.environ, SANDBOX_TOOL_RUNNER: "1"}
        proc = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
            env=env,
            check=True,
        )
        assert proc.stdout.strip() == "True"

    def test_lancedb_path_constant(self):
        assert LANCEDB_PATH == "LANCEDB_PATH"

    def test_database_url_constant(self):
        assert DATABASE_URL == "DATABASE_URL"

    def test_task_lease_recovery_constants(self):
        assert (
            TASK_LEASE_RECOVERY_INTERVAL_SECONDS
            == "XAGENT_TASK_LEASE_RECOVERY_INTERVAL_SECONDS"
        )
        assert TASK_LEASE_RECOVERY_BATCH_SIZE == "XAGENT_TASK_LEASE_RECOVERY_BATCH_SIZE"

    def test_uploaded_file_recovery_constants(self):
        assert (
            UPLOADED_FILE_RECOVERY_INTERVAL_SECONDS
            == "XAGENT_UPLOADED_FILE_RECOVERY_INTERVAL_SECONDS"
        )
        assert (
            UPLOADED_FILE_RECOVERY_STALE_SECONDS
            == "XAGENT_UPLOADED_FILE_RECOVERY_STALE_SECONDS"
        )
        assert (
            UPLOADED_FILE_RECOVERY_BATCH_SIZE
            == "XAGENT_UPLOADED_FILE_RECOVERY_BATCH_SIZE"
        )

    def test_max_upload_size_constant(self):
        assert MAX_UPLOAD_SIZE == "XAGENT_MAX_UPLOAD_SIZE"

    def test_web_search_provider_constant(self):
        assert WEB_SEARCH_PROVIDER == "XAGENT_WEB_SEARCH_PROVIDER"

    def test_web_crawl_tls_impersonate_constant(self):
        assert WEB_CRAWL_TLS_IMPERSONATE == "XAGENT_WEB_CRAWL_TLS_IMPERSONATE"

    def test_openrouter_official_providers_only_constant(self):
        assert (
            OPENROUTER_OFFICIAL_PROVIDERS_ONLY
            == "XAGENT_OPENROUTER_OFFICIAL_PROVIDERS_ONLY"
        )

    def test_mcp_oauth_allow_private_hosts_constant(self):
        assert MCP_OAUTH_ALLOW_PRIVATE_HOSTS == "XAGENT_MCP_OAUTH_ALLOW_PRIVATE_HOSTS"

    def test_mcp_oauth_proxy_url_constant(self):
        assert MCP_OAUTH_PROXY_URL == "XAGENT_MCP_OAUTH_PROXY_URL"

    def test_file_storage_uri_constant(self):
        assert FILE_STORAGE_URI == "XAGENT_FILE_STORAGE_URI"

    def test_file_storage_options_constant(self):
        assert FILE_STORAGE_OPTIONS == "XAGENT_FILE_STORAGE_OPTIONS"

    def test_file_materialize_dir_constant(self):
        assert FILE_MATERIALIZE_DIR == "XAGENT_FILE_MATERIALIZE_DIR"

    def test_preview_tmp_dir_constant(self):
        assert PREVIEW_TMP_DIR == "XAGENT_PREVIEW_TMP_DIR"

    def test_file_storage_startup_sync_enabled_constant(self):
        assert (
            FILE_STORAGE_STARTUP_SYNC_ENABLED
            == "XAGENT_FILE_STORAGE_STARTUP_SYNC_ENABLED"
        )

    def test_file_delivery_accel_redirect_constants(self):
        assert (
            FILE_DELIVERY_ACCEL_REDIRECT_ENABLED
            == "XAGENT_FILE_DELIVERY_ACCEL_REDIRECT_ENABLED"
        )
        assert (
            FILE_DELIVERY_ACCEL_REDIRECT_PREFIX
            == "XAGENT_FILE_DELIVERY_ACCEL_REDIRECT_PREFIX"
        )

    def test_redis_url_constant(self):
        assert REDIS_URL == "XAGENT_REDIS_URL"

    def test_hot_path_cache_constants(self):
        assert HOT_PATH_CACHE_ENABLED == "XAGENT_HOT_PATH_CACHE_ENABLED"
        assert HOT_PATH_CACHE_TTL_SECONDS == "XAGENT_HOT_PATH_CACHE_TTL_SECONDS"
        assert (
            HOT_PATH_TASK_CACHE_TTL_SECONDS == "XAGENT_HOT_PATH_TASK_CACHE_TTL_SECONDS"
        )

    def test_celery_background_job_constants(self):
        assert CELERY_ENABLED == "XAGENT_CELERY_ENABLED"
        assert CELERY_BROKER_URL == "XAGENT_CELERY_BROKER_URL"
        assert CELERY_RESULT_BACKEND == "XAGENT_CELERY_RESULT_BACKEND"
        assert (
            BACKGROUND_JOB_VISIBILITY_TIMEOUT_SECONDS
            == "XAGENT_BACKGROUND_JOB_VISIBILITY_TIMEOUT_SECONDS"
        )
        assert BACKGROUND_JOB_MAX_RETRIES == "XAGENT_BACKGROUND_JOB_MAX_RETRIES"
        assert BACKGROUND_JOB_STALE_SECONDS == "XAGENT_BACKGROUND_JOB_STALE_SECONDS"
        assert (
            BACKGROUND_JOB_SWEEP_INTERVAL_SECONDS
            == "XAGENT_BACKGROUND_JOB_SWEEP_INTERVAL_SECONDS"
        )
        assert TRIGGER_DISPATCHER_ENABLED == "XAGENT_TRIGGER_DISPATCHER_ENABLED"
        assert (
            TRIGGER_DISPATCHER_INTERVAL_SECONDS
            == "XAGENT_TRIGGER_DISPATCHER_INTERVAL_SECONDS"
        )
        assert TRIGGER_DISPATCHER_BATCH_SIZE == "XAGENT_TRIGGER_DISPATCHER_BATCH_SIZE"

    def test_auth_email_config_constants(self):
        assert PASSWORD_RESET_EXPIRE_MINUTES == "XAGENT_PASSWORD_RESET_EXPIRE_MINUTES"
        assert APP_BASE_URL == "XAGENT_APP_BASE_URL"
        assert SMTP_HOST == "XAGENT_SMTP_HOST"
        assert SMTP_PORT == "XAGENT_SMTP_PORT"
        assert SMTP_USERNAME == "XAGENT_SMTP_USERNAME"
        assert SMTP_PASSWORD == "XAGENT_SMTP_PASSWORD"
        assert SMTP_USE_TLS == "XAGENT_SMTP_USE_TLS"
        assert SMTP_USE_SSL == "XAGENT_SMTP_USE_SSL"
        assert SMTP_FROM_EMAIL == "XAGENT_SMTP_FROM_EMAIL"
        assert SMTP_FROM_NAME == "XAGENT_SMTP_FROM_NAME"

    def test_slack_oauth_config_constants(self):
        assert SLACK_CLIENT_ID == "XAGENT_SLACK_CLIENT_ID"
        assert SLACK_CLIENT_SECRET == "XAGENT_SLACK_CLIENT_SECRET"
        assert SLACK_APP_TOKEN == "XAGENT_SLACK_APP_TOKEN"
        assert SLACK_REDIRECT_URI == "XAGENT_SLACK_REDIRECT_URI"


class TestAuthEmailConfig:
    def test_password_reset_expire_minutes_defaults_to_30(self, monkeypatch):
        monkeypatch.delenv(PASSWORD_RESET_EXPIRE_MINUTES, raising=False)
        assert get_password_reset_expire_minutes() == 30

    @pytest.mark.parametrize("value", ["abc", "0", "-5"])
    def test_password_reset_expire_minutes_invalid_values_fall_back(
        self, monkeypatch, value
    ):
        monkeypatch.setenv(PASSWORD_RESET_EXPIRE_MINUTES, value)
        assert get_password_reset_expire_minutes() == 30

    def test_app_base_url_returns_none_when_unset_or_blank(self, monkeypatch):
        monkeypatch.delenv(APP_BASE_URL, raising=False)
        assert get_app_base_url() is None

        monkeypatch.setenv(APP_BASE_URL, "   ")
        assert get_app_base_url() is None

    def test_app_base_url_strips_and_removes_trailing_slash(self, monkeypatch):
        monkeypatch.setenv(APP_BASE_URL, " https://app.example.com/base/ ")
        assert get_app_base_url() == "https://app.example.com/base"

    def test_slack_oauth_config_defaults_to_unconfigured(self, monkeypatch):
        for env_name in (
            SLACK_CLIENT_ID,
            SLACK_CLIENT_SECRET,
            SLACK_APP_TOKEN,
            SLACK_REDIRECT_URI,
            PUBLIC_API_BASE_URL,
        ):
            monkeypatch.delenv(env_name, raising=False)

        assert get_slack_client_id() is None
        assert get_slack_client_secret() is None
        assert get_slack_app_token() is None
        assert get_slack_oauth_redirect_uri() is None

    def test_slack_oauth_config_uses_explicit_values(self, monkeypatch):
        monkeypatch.setenv(SLACK_CLIENT_ID, " client-id ")
        monkeypatch.setenv(SLACK_CLIENT_SECRET, " client-secret ")
        monkeypatch.setenv(SLACK_APP_TOKEN, " xapp-test ")
        monkeypatch.setenv(
            SLACK_REDIRECT_URI,
            " https://api.example.com/api/channels/slack/oauth/callback/ ",
        )

        assert get_slack_client_id() == "client-id"
        assert get_slack_client_secret() == "client-secret"
        assert get_slack_app_token() == "xapp-test"
        assert (
            get_slack_oauth_redirect_uri()
            == "https://api.example.com/api/channels/slack/oauth/callback"
        )

    def test_slack_redirect_uri_falls_back_to_public_api_base(self, monkeypatch):
        monkeypatch.delenv(SLACK_REDIRECT_URI, raising=False)
        monkeypatch.setenv(PUBLIC_API_BASE_URL, " https://api.example.com/ ")

        assert (
            get_slack_oauth_redirect_uri()
            == "https://api.example.com/api/channels/slack/oauth/callback"
        )

    def test_smtp_host_and_credentials_strip_expected_values(self, monkeypatch):
        monkeypatch.setenv(SMTP_HOST, " smtp.example.com ")
        monkeypatch.setenv(SMTP_USERNAME, " user ")
        monkeypatch.setenv(SMTP_PASSWORD, "secret ")
        monkeypatch.setenv(SMTP_FROM_EMAIL, " noreply@example.com ")

        assert get_smtp_host() == "smtp.example.com"
        assert get_smtp_username() == "user"
        assert get_smtp_password() == "secret "
        assert get_smtp_from_email() == "noreply@example.com"

    def test_smtp_port_defaults_and_invalid_values_fall_back(self, monkeypatch):
        monkeypatch.delenv(SMTP_PORT, raising=False)
        assert get_smtp_port() == 587

        monkeypatch.setenv(SMTP_PORT, "abc")
        assert get_smtp_port() == 587

        monkeypatch.setenv(SMTP_PORT, "0")
        assert get_smtp_port() == 587

    @pytest.mark.parametrize(
        "env_var,getter,default,true_value,false_value",
        [
            (SMTP_USE_TLS, get_smtp_use_tls, True, "true", "false"),
            (SMTP_USE_SSL, get_smtp_use_ssl, False, "yes", "off"),
        ],
    )
    def test_smtp_bool_settings(
        self, monkeypatch, env_var, getter, default, true_value, false_value
    ):
        monkeypatch.delenv(env_var, raising=False)
        assert getter() is default

        monkeypatch.setenv(env_var, true_value)
        assert getter() is True

        monkeypatch.setenv(env_var, false_value)
        assert getter() is False

    def test_smtp_from_name_uses_default_and_trimmed_override(self, monkeypatch):
        monkeypatch.delenv(SMTP_FROM_NAME, raising=False)
        assert get_smtp_from_name("Xagent") == "Xagent"

        monkeypatch.setenv(SMTP_FROM_NAME, " Support Team ")
        assert get_smtp_from_name("Xagent") == "Support Team"


class TestOpenRouterConfig:
    def test_official_providers_only_defaults_false(self, monkeypatch):
        monkeypatch.delenv(OPENROUTER_OFFICIAL_PROVIDERS_ONLY, raising=False)
        assert get_openrouter_official_providers_only() is False

    @pytest.mark.parametrize("value", ["true", "1", "yes", "on", " TRUE "])
    def test_official_providers_only_true_values(self, monkeypatch, value):
        monkeypatch.setenv(OPENROUTER_OFFICIAL_PROVIDERS_ONLY, value)
        assert get_openrouter_official_providers_only() is True

    @pytest.mark.parametrize("value", ["false", "0", "no", "off", "", "unknown"])
    def test_official_providers_only_false_values(self, monkeypatch, value):
        monkeypatch.setenv(OPENROUTER_OFFICIAL_PROVIDERS_ONLY, value)
        assert get_openrouter_official_providers_only() is False


class TestMCPOAuthConfig:
    def test_allow_private_hosts_defaults_false(self, monkeypatch):
        monkeypatch.delenv(MCP_OAUTH_ALLOW_PRIVATE_HOSTS, raising=False)
        assert get_mcp_oauth_allow_private_hosts() is False

    @pytest.mark.parametrize("value", ["true", "1", "yes", "on", " TRUE "])
    def test_allow_private_hosts_true_values(self, monkeypatch, value):
        monkeypatch.setenv(MCP_OAUTH_ALLOW_PRIVATE_HOSTS, value)
        assert get_mcp_oauth_allow_private_hosts() is True

    @pytest.mark.parametrize("value", ["false", "0", "no", "off", "", "unknown"])
    def test_allow_private_hosts_false_values(self, monkeypatch, value):
        monkeypatch.setenv(MCP_OAUTH_ALLOW_PRIVATE_HOSTS, value)
        assert get_mcp_oauth_allow_private_hosts() is False

    def test_proxy_url_returns_none_when_unset_or_blank(self, monkeypatch):
        monkeypatch.delenv(MCP_OAUTH_PROXY_URL, raising=False)
        assert get_mcp_oauth_proxy_url() is None

        monkeypatch.setenv(MCP_OAUTH_PROXY_URL, "   ")
        assert get_mcp_oauth_proxy_url() is None

    def test_proxy_url_accepts_absolute_http_proxy(self, monkeypatch):
        monkeypatch.setenv(MCP_OAUTH_PROXY_URL, " http://proxy.example.com:8080 ")
        assert get_mcp_oauth_proxy_url() == "http://proxy.example.com:8080"

    @pytest.mark.parametrize("value", ["proxy.example.com:8080", "socks5://proxy:1080"])
    def test_proxy_url_rejects_unsupported_values(self, monkeypatch, value):
        monkeypatch.setenv(MCP_OAUTH_PROXY_URL, value)
        with pytest.raises(ValueError, match="XAGENT_MCP_OAUTH_PROXY_URL"):
            get_mcp_oauth_proxy_url()


class TestTrustedEgressProxyConfig:
    def test_trusted_egress_proxy_constant(self):
        assert TRUSTED_EGRESS_PROXY == "XAGENT_TRUSTED_EGRESS_PROXY"

    def test_trusted_egress_proxy_defaults_false(self, monkeypatch):
        monkeypatch.delenv(TRUSTED_EGRESS_PROXY, raising=False)
        assert get_trusted_egress_proxy_enabled() is False

    @pytest.mark.parametrize("value", ["true", "1", "yes", "on", " TRUE "])
    def test_trusted_egress_proxy_true_values(self, monkeypatch, value):
        monkeypatch.setenv(TRUSTED_EGRESS_PROXY, value)
        assert get_trusted_egress_proxy_enabled() is True

    @pytest.mark.parametrize("value", ["false", "0", "no", "off", "", "unknown"])
    def test_trusted_egress_proxy_false_values(self, monkeypatch, value):
        monkeypatch.setenv(TRUSTED_EGRESS_PROXY, value)
        assert get_trusted_egress_proxy_enabled() is False


class TestHotPathCacheConfig:
    def test_redis_url_empty_is_none(self, monkeypatch):
        monkeypatch.delenv(REDIS_URL, raising=False)
        assert get_redis_url() is None
        monkeypatch.setenv(REDIS_URL, "  ")
        assert get_redis_url() is None

    def test_redis_url_strips_value(self, monkeypatch):
        monkeypatch.setenv(REDIS_URL, " redis://localhost:6379/0 ")
        assert get_redis_url() == "redis://localhost:6379/0"

    def test_hot_path_cache_enabled_defaults_true(self, monkeypatch):
        monkeypatch.delenv(HOT_PATH_CACHE_ENABLED, raising=False)
        assert get_hot_path_cache_enabled() is True

    def test_hot_path_cache_enabled_false(self, monkeypatch):
        monkeypatch.setenv(HOT_PATH_CACHE_ENABLED, "false")
        assert get_hot_path_cache_enabled() is False

    def test_hot_path_ttls(self, monkeypatch):
        monkeypatch.delenv(HOT_PATH_CACHE_TTL_SECONDS, raising=False)
        monkeypatch.delenv(HOT_PATH_TASK_CACHE_TTL_SECONDS, raising=False)
        assert get_hot_path_cache_ttl_seconds() == 30
        assert get_hot_path_task_cache_ttl_seconds() == 30

        monkeypatch.setenv(HOT_PATH_CACHE_TTL_SECONDS, "45")
        monkeypatch.setenv(HOT_PATH_TASK_CACHE_TTL_SECONDS, "3")
        assert get_hot_path_cache_ttl_seconds() == 45
        assert get_hot_path_task_cache_ttl_seconds() == 3


class TestCeleryBackgroundJobConfig:
    def test_celery_disabled_by_default(self, monkeypatch):
        monkeypatch.delenv(CELERY_ENABLED, raising=False)
        assert get_celery_enabled() is False

    def test_celery_enabled_true_values(self, monkeypatch):
        monkeypatch.setenv(CELERY_ENABLED, "yes")
        assert get_celery_enabled() is True

    def test_celery_broker_explicit(self, monkeypatch):
        monkeypatch.setenv(CELERY_BROKER_URL, " redis://localhost:6379/7 ")
        monkeypatch.setenv(REDIS_URL, "redis://localhost:6379/0")
        assert get_celery_broker_url() == "redis://localhost:6379/7"

    def test_celery_broker_derives_from_redis_url_db1(self, monkeypatch):
        monkeypatch.delenv(CELERY_BROKER_URL, raising=False)
        monkeypatch.setenv(REDIS_URL, "redis://localhost:6379/0")
        assert get_celery_broker_url() == "redis://localhost:6379/1"

    def test_celery_broker_none_without_redis(self, monkeypatch):
        monkeypatch.delenv(CELERY_BROKER_URL, raising=False)
        monkeypatch.delenv(REDIS_URL, raising=False)
        assert get_celery_broker_url() is None

    def test_celery_result_backend_optional(self, monkeypatch):
        monkeypatch.delenv(CELERY_RESULT_BACKEND, raising=False)
        assert get_celery_result_backend() is None
        monkeypatch.setenv(CELERY_RESULT_BACKEND, " redis://localhost:6379/2 ")
        assert get_celery_result_backend() == "redis://localhost:6379/2"

    def test_background_job_tuning_defaults(self, monkeypatch):
        monkeypatch.delenv(BACKGROUND_JOB_VISIBILITY_TIMEOUT_SECONDS, raising=False)
        monkeypatch.delenv(BACKGROUND_JOB_MAX_RETRIES, raising=False)
        monkeypatch.delenv(BACKGROUND_JOB_STALE_SECONDS, raising=False)
        monkeypatch.delenv(BACKGROUND_JOB_SWEEP_INTERVAL_SECONDS, raising=False)
        assert get_background_job_visibility_timeout_seconds() == 3600
        assert get_background_job_max_retries() == 3
        assert get_background_job_stale_seconds() == 7200
        assert get_background_job_sweep_interval_seconds() == 300

    def test_trigger_dispatcher_tuning(self, monkeypatch):
        monkeypatch.delenv(TRIGGER_DISPATCHER_ENABLED, raising=False)
        monkeypatch.delenv(TRIGGER_DISPATCHER_INTERVAL_SECONDS, raising=False)
        monkeypatch.delenv(TRIGGER_DISPATCHER_BATCH_SIZE, raising=False)
        assert get_trigger_dispatcher_enabled() is True
        assert get_trigger_dispatcher_interval_seconds() == 5
        assert get_trigger_dispatcher_batch_size() == 20

        monkeypatch.setenv(TRIGGER_DISPATCHER_ENABLED, "false")
        monkeypatch.setenv(TRIGGER_DISPATCHER_INTERVAL_SECONDS, "9")
        monkeypatch.setenv(TRIGGER_DISPATCHER_BATCH_SIZE, "3")
        assert get_trigger_dispatcher_enabled() is False
        assert get_trigger_dispatcher_interval_seconds() == 9
        assert get_trigger_dispatcher_batch_size() == 3

    def test_task_lease_recovery_tuning(self, monkeypatch):
        monkeypatch.setenv(TASK_LEASE_TTL_SECONDS, "60")
        monkeypatch.delenv(TASK_LEASE_RECOVERY_INTERVAL_SECONDS, raising=False)
        monkeypatch.delenv(TASK_LEASE_RECOVERY_BATCH_SIZE, raising=False)
        assert get_task_lease_recovery_interval_seconds() == 20
        assert get_task_lease_recovery_batch_size() == 100

        monkeypatch.setenv(TASK_LEASE_RECOVERY_INTERVAL_SECONDS, "7")
        monkeypatch.setenv(TASK_LEASE_RECOVERY_BATCH_SIZE, "3")
        assert get_task_lease_recovery_interval_seconds() == 7
        assert get_task_lease_recovery_batch_size() == 3

        monkeypatch.setenv(TASK_LEASE_RECOVERY_INTERVAL_SECONDS, "")
        monkeypatch.setenv(TASK_LEASE_RECOVERY_BATCH_SIZE, "0")
        assert get_task_lease_recovery_interval_seconds() == 20
        assert get_task_lease_recovery_batch_size() == 100

    def test_uploaded_file_recovery_tuning(self, monkeypatch):
        monkeypatch.delenv(UPLOADED_FILE_RECOVERY_INTERVAL_SECONDS, raising=False)
        monkeypatch.delenv(UPLOADED_FILE_RECOVERY_STALE_SECONDS, raising=False)
        monkeypatch.delenv(UPLOADED_FILE_RECOVERY_BATCH_SIZE, raising=False)
        assert get_uploaded_file_recovery_interval_seconds() == 60
        assert get_uploaded_file_recovery_stale_seconds() == 300
        assert get_uploaded_file_recovery_batch_size() == 100

        monkeypatch.setenv(UPLOADED_FILE_RECOVERY_INTERVAL_SECONDS, "7")
        monkeypatch.setenv(UPLOADED_FILE_RECOVERY_STALE_SECONDS, "45")
        monkeypatch.setenv(UPLOADED_FILE_RECOVERY_BATCH_SIZE, "3")
        assert get_uploaded_file_recovery_interval_seconds() == 7
        assert get_uploaded_file_recovery_stale_seconds() == 45
        assert get_uploaded_file_recovery_batch_size() == 3

        monkeypatch.setenv(UPLOADED_FILE_RECOVERY_INTERVAL_SECONDS, "")
        monkeypatch.setenv(UPLOADED_FILE_RECOVERY_STALE_SECONDS, "0")
        monkeypatch.setenv(UPLOADED_FILE_RECOVERY_BATCH_SIZE, "-1")
        assert get_uploaded_file_recovery_interval_seconds() == 60
        assert get_uploaded_file_recovery_stale_seconds() == 300
        assert get_uploaded_file_recovery_batch_size() == 100

    def test_temp_file_cleanup_shutdown_timeout_tuning(self, monkeypatch):
        assert (
            TEMP_FILE_CLEANUP_SHUTDOWN_TIMEOUT_SECONDS
            == "XAGENT_TEMP_FILE_CLEANUP_SHUTDOWN_TIMEOUT_SECONDS"
        )

        monkeypatch.delenv(TEMP_FILE_CLEANUP_SHUTDOWN_TIMEOUT_SECONDS, raising=False)
        assert get_temp_file_cleanup_shutdown_timeout_seconds() == 10

        monkeypatch.setenv(TEMP_FILE_CLEANUP_SHUTDOWN_TIMEOUT_SECONDS, "45")
        assert get_temp_file_cleanup_shutdown_timeout_seconds() == 45

        # Invalid / non-positive values fall back to the default.
        monkeypatch.setenv(TEMP_FILE_CLEANUP_SHUTDOWN_TIMEOUT_SECONDS, "0")
        assert get_temp_file_cleanup_shutdown_timeout_seconds() == 10

        monkeypatch.setenv(TEMP_FILE_CLEANUP_SHUTDOWN_TIMEOUT_SECONDS, "abc")
        assert get_temp_file_cleanup_shutdown_timeout_seconds() == 10

        monkeypatch.setenv(TEMP_FILE_CLEANUP_SHUTDOWN_TIMEOUT_SECONDS, "-5")
        assert get_temp_file_cleanup_shutdown_timeout_seconds() == 10


class TestGetWebSearchProvider:
    """Test get_web_search_provider() function."""

    def test_default_web_search_provider(self, monkeypatch):
        monkeypatch.delenv(WEB_SEARCH_PROVIDER, raising=False)
        assert get_web_search_provider() == "auto"

    def test_normalizes_web_search_provider(self, monkeypatch):
        monkeypatch.setenv(WEB_SEARCH_PROVIDER, " Google ")
        assert get_web_search_provider() == "google"

    def test_invalid_web_search_provider_falls_back_to_auto(self, monkeypatch):
        monkeypatch.setenv(WEB_SEARCH_PROVIDER, "bing")
        assert get_web_search_provider() == "auto"


class TestGetWebCrawlTlsImpersonate:
    """Test get_web_crawl_tls_impersonate() function."""

    def test_default_web_crawl_tls_impersonate(self, monkeypatch):
        monkeypatch.delenv(WEB_CRAWL_TLS_IMPERSONATE, raising=False)
        assert get_web_crawl_tls_impersonate() is None

    @pytest.mark.parametrize("value", ["", "   ", "none", "None", "NULL"])
    def test_empty_like_web_crawl_tls_impersonate(self, monkeypatch, value):
        monkeypatch.setenv(WEB_CRAWL_TLS_IMPERSONATE, value)
        assert get_web_crawl_tls_impersonate() is None

    def test_auto_web_crawl_tls_impersonate(self, monkeypatch):
        monkeypatch.setenv(WEB_CRAWL_TLS_IMPERSONATE, " auto ")
        assert get_web_crawl_tls_impersonate() == "auto"

    def test_specific_web_crawl_tls_impersonate(self, monkeypatch):
        monkeypatch.setenv(WEB_CRAWL_TLS_IMPERSONATE, "safari17_0")
        assert get_web_crawl_tls_impersonate() == "safari17_0"


class TestGetMaxUploadSizeBytes:
    """Test get_max_upload_size_bytes() function."""

    def test_default_max_upload_size(self, monkeypatch):
        monkeypatch.delenv(MAX_UPLOAD_SIZE, raising=False)
        assert get_max_upload_size_bytes() == 100 * 1024 * 1024

    def test_numeric_max_upload_size(self, monkeypatch):
        monkeypatch.setenv(MAX_UPLOAD_SIZE, "2048")
        assert get_max_upload_size_bytes() == 2048

    def test_numeric_float_max_upload_size(self, monkeypatch):
        monkeypatch.setenv(MAX_UPLOAD_SIZE, "1.5")
        assert get_max_upload_size_bytes() == 1

    def test_rejects_non_positive_max_upload_size(self, monkeypatch):
        monkeypatch.setenv(MAX_UPLOAD_SIZE, "0")
        with pytest.raises(ValueError, match="positive"):
            get_max_upload_size_bytes()

        monkeypatch.setenv(MAX_UPLOAD_SIZE, "-1")
        with pytest.raises(ValueError, match="positive"):
            get_max_upload_size_bytes()

    def test_human_readable_max_upload_size(self, monkeypatch):
        monkeypatch.setenv(MAX_UPLOAD_SIZE, "150M")
        assert get_max_upload_size_bytes() == 150 * 1024 * 1024

    def test_invalid_max_upload_size_raises(self, monkeypatch):
        monkeypatch.setenv(MAX_UPLOAD_SIZE, "banana")
        with pytest.raises(ValueError, match="XAGENT_MAX_UPLOAD_SIZE"):
            get_max_upload_size_bytes()


class TestFormatFileSize:
    def test_formats_kilobytes(self):
        assert format_file_size(512 * 1024) == "512KB"

    def test_formats_fractional_megabytes(self):
        assert format_file_size(1572864) == "1.5MB"

    def test_promotes_boundary_values_to_next_unit(self):
        assert format_file_size(1048575) == "1MB"


class TestFileStorageConfig:
    def test_default_file_storage_uri_uses_storage_root(self, monkeypatch):
        monkeypatch.delenv(FILE_STORAGE_URI, raising=False)
        monkeypatch.setenv(STORAGE_ROOT, "/custom/storage")

        assert get_file_storage_uri() == "file:///custom/storage/files"

    def test_default_file_storage_uri_resolves_relative_storage_root(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.delenv(FILE_STORAGE_URI, raising=False)
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv(STORAGE_ROOT, ".xagent")

        assert get_file_storage_uri() == (tmp_path / ".xagent" / "files").as_uri()

    def test_file_storage_uri_with_env_var(self, monkeypatch):
        monkeypatch.setenv(FILE_STORAGE_URI, "s3://bucket/prefix")

        assert get_file_storage_uri() == "s3://bucket/prefix"

    def test_file_storage_options_default_to_empty_dict(self, monkeypatch):
        monkeypatch.delenv(FILE_STORAGE_OPTIONS, raising=False)

        assert get_file_storage_options() == {}

    def test_file_storage_options_parse_json_object(self, monkeypatch):
        monkeypatch.setenv(
            FILE_STORAGE_OPTIONS,
            '{"endpoint_url":"https://s3.example.com","region_name":"us-east-1"}',
        )

        assert get_file_storage_options() == {
            "endpoint_url": "https://s3.example.com",
            "region_name": "us-east-1",
        }

    def test_file_storage_options_reject_non_object_json(self, monkeypatch):
        monkeypatch.setenv(FILE_STORAGE_OPTIONS, '["not", "an", "object"]')

        with pytest.raises(ValueError, match="XAGENT_FILE_STORAGE_OPTIONS"):
            get_file_storage_options()

    def test_file_materialize_dir_default(self, monkeypatch):
        monkeypatch.delenv(FILE_MATERIALIZE_DIR, raising=False)

        assert get_file_materialize_dir() == Path(gettempdir()) / "xagent-materialized"

    def test_file_materialize_dir_with_env_var(self, monkeypatch):
        monkeypatch.setenv(FILE_MATERIALIZE_DIR, "/custom/materialized")

        assert get_file_materialize_dir() == Path("/custom/materialized")

    def test_preview_tmp_dir_default(self, monkeypatch):
        monkeypatch.delenv(PREVIEW_TMP_DIR, raising=False)

        assert get_preview_tmp_dir() == Path(gettempdir()) / "xagent-preview"

    def test_preview_tmp_dir_with_env_var(self, monkeypatch):
        monkeypatch.setenv(PREVIEW_TMP_DIR, "/custom/preview-tmp")

        assert get_preview_tmp_dir() == Path("/custom/preview-tmp")

    def test_file_storage_startup_sync_enabled_defaults_true(self, monkeypatch):
        monkeypatch.delenv(FILE_STORAGE_STARTUP_SYNC_ENABLED, raising=False)

        assert get_file_storage_startup_sync_enabled() is True

    @pytest.mark.parametrize(
        "value,expected",
        [
            ("true", True),
            ("1", True),
            ("yes", True),
            ("on", True),
            ("false", False),
            ("0", False),
            ("no", False),
            ("off", False),
        ],
    )
    def test_file_storage_startup_sync_enabled_parses_bool(
        self, monkeypatch, value, expected
    ):
        monkeypatch.setenv(FILE_STORAGE_STARTUP_SYNC_ENABLED, value)

        assert get_file_storage_startup_sync_enabled() is expected

    def test_file_storage_startup_sync_enabled_rejects_invalid(self, monkeypatch):
        monkeypatch.setenv(FILE_STORAGE_STARTUP_SYNC_ENABLED, "maybe")

        with pytest.raises(
            ValueError, match="XAGENT_FILE_STORAGE_STARTUP_SYNC_ENABLED"
        ):
            get_file_storage_startup_sync_enabled()

    def test_file_delivery_redirect_enabled_defaults_false(self, monkeypatch):
        monkeypatch.delenv(FILE_DELIVERY_REDIRECT_ENABLED, raising=False)

        assert get_file_delivery_redirect_enabled() is False

    @pytest.mark.parametrize(
        "value,expected",
        [
            ("true", True),
            ("1", True),
            ("yes", True),
            ("on", True),
            ("false", False),
            ("0", False),
            ("no", False),
            ("off", False),
        ],
    )
    def test_file_delivery_redirect_enabled_parses_bool(
        self, monkeypatch, value, expected
    ):
        monkeypatch.setenv(FILE_DELIVERY_REDIRECT_ENABLED, value)

        assert get_file_delivery_redirect_enabled() is expected

    def test_file_delivery_redirect_enabled_rejects_invalid(self, monkeypatch):
        monkeypatch.setenv(FILE_DELIVERY_REDIRECT_ENABLED, "maybe")

        with pytest.raises(ValueError, match="XAGENT_FILE_DELIVERY_REDIRECT_ENABLED"):
            get_file_delivery_redirect_enabled()

    def test_file_delivery_signed_url_ttl_defaults_to_300(self, monkeypatch):
        monkeypatch.delenv(FILE_DELIVERY_SIGNED_URL_TTL_SECONDS, raising=False)

        assert get_file_delivery_signed_url_ttl_seconds() == 300

    def test_file_delivery_signed_url_ttl_with_env_var(self, monkeypatch):
        monkeypatch.setenv(FILE_DELIVERY_SIGNED_URL_TTL_SECONDS, "60")

        assert get_file_delivery_signed_url_ttl_seconds() == 60

    @pytest.mark.parametrize("value", ["0", "-1", "abc"])
    def test_file_delivery_signed_url_ttl_rejects_invalid(self, monkeypatch, value):
        monkeypatch.setenv(FILE_DELIVERY_SIGNED_URL_TTL_SECONDS, value)

        with pytest.raises(
            ValueError, match="XAGENT_FILE_DELIVERY_SIGNED_URL_TTL_SECONDS"
        ):
            get_file_delivery_signed_url_ttl_seconds()

    def test_file_stream_ticket_ttl_defaults_to_600(self, monkeypatch):
        monkeypatch.delenv(FILE_STREAM_TICKET_TTL_SECONDS, raising=False)

        assert get_file_stream_ticket_ttl_seconds() == 600

    def test_file_stream_ticket_ttl_with_env_var(self, monkeypatch):
        monkeypatch.setenv(FILE_STREAM_TICKET_TTL_SECONDS, "120")

        assert get_file_stream_ticket_ttl_seconds() == 120

    @pytest.mark.parametrize("value", ["0", "-1", "abc"])
    def test_file_stream_ticket_ttl_rejects_invalid(self, monkeypatch, value):
        monkeypatch.setenv(FILE_STREAM_TICKET_TTL_SECONDS, value)

        with pytest.raises(ValueError, match="XAGENT_FILE_STREAM_TICKET_TTL_SECONDS"):
            get_file_stream_ticket_ttl_seconds()

    def test_file_delivery_accel_redirect_enabled_defaults_false(self, monkeypatch):
        monkeypatch.delenv(FILE_DELIVERY_ACCEL_REDIRECT_ENABLED, raising=False)

        assert get_file_delivery_accel_redirect_enabled() is False

    @pytest.mark.parametrize(
        "value,expected",
        [
            ("true", True),
            ("1", True),
            ("yes", True),
            ("on", True),
            ("false", False),
            ("0", False),
            ("no", False),
            ("off", False),
        ],
    )
    def test_file_delivery_accel_redirect_enabled_parses_bool(
        self, monkeypatch, value, expected
    ):
        monkeypatch.setenv(FILE_DELIVERY_ACCEL_REDIRECT_ENABLED, value)

        assert get_file_delivery_accel_redirect_enabled() is expected

    def test_file_delivery_accel_redirect_enabled_rejects_invalid(self, monkeypatch):
        monkeypatch.setenv(FILE_DELIVERY_ACCEL_REDIRECT_ENABLED, "maybe")

        with pytest.raises(
            ValueError, match="XAGENT_FILE_DELIVERY_ACCEL_REDIRECT_ENABLED"
        ):
            get_file_delivery_accel_redirect_enabled()

    def test_file_delivery_accel_redirect_prefix_defaults_to_internal_uri(
        self, monkeypatch
    ):
        monkeypatch.delenv(FILE_DELIVERY_ACCEL_REDIRECT_PREFIX, raising=False)

        assert get_file_delivery_accel_redirect_prefix() == "/_xagent_internal_files/"

    def test_file_delivery_accel_redirect_prefix_normalizes_trailing_slash(
        self, monkeypatch
    ):
        monkeypatch.setenv(FILE_DELIVERY_ACCEL_REDIRECT_PREFIX, "/private-files")

        assert get_file_delivery_accel_redirect_prefix() == "/private-files/"

    def test_file_delivery_accel_redirect_prefix_requires_absolute_uri(
        self, monkeypatch
    ):
        monkeypatch.setenv(FILE_DELIVERY_ACCEL_REDIRECT_PREFIX, "private-files")

        with pytest.raises(
            ValueError, match="XAGENT_FILE_DELIVERY_ACCEL_REDIRECT_PREFIX"
        ):
            get_file_delivery_accel_redirect_prefix()


class TestGetUploadsDir:
    """Test get_uploads_dir() function."""

    def test_default_uploads_dir(self, monkeypatch):
        """Test default uploads directory path."""
        monkeypatch.delenv(UPLOADS_DIR, raising=False)
        monkeypatch.delenv(WEB_DIR, raising=False)
        result = get_uploads_dir()
        # Default is src/xagent/web/uploads
        assert result.name == "uploads"
        assert result.parent.name == "web"

    def test_uploads_dir_with_env_var(self, monkeypatch):
        """Test uploads directory with environment variable."""
        monkeypatch.setenv(UPLOADS_DIR, "/tmp/test_uploads")
        result = get_uploads_dir()
        assert result == Path("/tmp/test_uploads")

    def test_uploads_dir_env_overrides_web_dir(self, monkeypatch):
        """Test that UPLOADS_DIR env var overrides computed default."""
        monkeypatch.setenv(WEB_DIR, "/custom/web")
        monkeypatch.setenv(UPLOADS_DIR, "/custom/uploads")
        result = get_uploads_dir()
        assert result == Path("/custom/uploads")


class TestGetWebDir:
    """Test get_web_dir() function."""

    def test_default_web_dir(self, monkeypatch):
        """Test default web directory path."""
        monkeypatch.delenv(WEB_DIR, raising=False)
        result = get_web_dir()
        assert result.name == "web"

    def test_web_dir_with_env_var(self, monkeypatch):
        """Test web directory with environment variable."""
        monkeypatch.setenv(WEB_DIR, "/custom/web")
        result = get_web_dir()
        assert result == Path("/custom/web")


class TestGetFrontendDistDir:
    """Test get_frontend_dist_dir() function."""

    def test_default_frontend_dist_dir(self, monkeypatch):
        """Defaults to WEB_DIR/frontend_dist."""
        monkeypatch.delenv(FRONTEND_DIST_DIR, raising=False)
        monkeypatch.delenv(WEB_DIR, raising=False)
        result = get_frontend_dist_dir()
        assert result.name == "frontend_dist"
        assert result.parent.name == "web"

    def test_frontend_dist_dir_with_env_var(self, monkeypatch):
        """Environment variable overrides the default."""
        monkeypatch.setenv(FRONTEND_DIST_DIR, "/custom/dist")
        result = get_frontend_dist_dir()
        assert result == Path("/custom/dist")

    def test_frontend_dist_dir_follows_web_dir_default(self, monkeypatch):
        """Default is computed under the configured WEB_DIR."""
        monkeypatch.delenv(FRONTEND_DIST_DIR, raising=False)
        monkeypatch.setenv(WEB_DIR, "/custom/web")
        result = get_frontend_dist_dir()
        assert result == Path("/custom/web/frontend_dist")


class TestGetAgentRuntime:
    """Test get_agent_runtime() function."""

    def test_default_agent_runtime(self, monkeypatch):
        monkeypatch.delenv(AGENT_RUNTIME, raising=False)
        assert get_agent_runtime() == "v1"

    def test_agent_runtime_v2(self, monkeypatch):
        monkeypatch.setenv(AGENT_RUNTIME, "v2")
        assert get_agent_runtime() == "v2"

    def test_agent_runtime_normalizes_case_and_spaces(self, monkeypatch):
        monkeypatch.setenv(AGENT_RUNTIME, " V2 ")
        assert get_agent_runtime() == "v2"

    def test_invalid_agent_runtime_falls_back_to_v1(self, monkeypatch):
        monkeypatch.setenv(AGENT_RUNTIME, "unknown")
        assert get_agent_runtime() == "v1"


class TestGetAgentPatternForExecutionMode:
    """Test get_agent_pattern_for_execution_mode() function."""

    def test_known_execution_modes(self):
        assert get_agent_pattern_for_execution_mode("flash") == "single_call"
        assert get_agent_pattern_for_execution_mode("balanced") == "react"
        assert get_agent_pattern_for_execution_mode("think") == "dag_plan_execute"
        assert get_agent_pattern_for_execution_mode("auto") == "auto"

    def test_normalizes_mode(self):
        assert get_agent_pattern_for_execution_mode(" AUTO ") == "auto"

    def test_unknown_mode_falls_back_to_react(self):
        assert get_agent_pattern_for_execution_mode("unknown") == "react"
        assert get_agent_pattern_for_execution_mode(None) == "react"


class TestGetDefaultTaskExecutionMode:
    """Test default task execution mode selection."""

    def test_default_standalone_defaults_to_auto(self, monkeypatch):
        monkeypatch.delenv(AGENT_RUNTIME, raising=False)
        assert get_default_task_execution_mode() == "auto"

    def test_v1_standalone_defaults_to_think(self, monkeypatch):
        monkeypatch.setenv(AGENT_RUNTIME, "v1")
        assert get_default_task_execution_mode() == "think"

    def test_v2_standalone_defaults_to_auto(self, monkeypatch):
        monkeypatch.setenv(AGENT_RUNTIME, "v2")
        assert get_default_task_execution_mode() == "auto"

    def test_agent_tasks_default_to_balanced_in_v2(self, monkeypatch):
        monkeypatch.setenv(AGENT_RUNTIME, "v2")
        assert get_default_task_execution_mode(agent_id=123) == "balanced"

    def test_explicit_runtime_can_be_passed(self):
        assert get_default_task_execution_mode(agent_runtime="v2") == "auto"


class TestGetExternalUploadDirs:
    """Test get_external_upload_dirs() function."""

    def test_no_env_var_returns_empty_list(self, monkeypatch):
        """Test that missing env var returns empty list."""
        monkeypatch.delenv(EXTERNAL_UPLOAD_DIRS, raising=False)
        result = get_external_upload_dirs()
        assert result == []

    def test_empty_env_var_returns_empty_list(self, monkeypatch):
        """Test that empty env var returns empty list."""
        monkeypatch.setenv(EXTERNAL_UPLOAD_DIRS, "")
        result = get_external_upload_dirs()
        assert result == []

    def test_nonexistent_dirs_are_filtered(self, monkeypatch):
        """Test that nonexistent directories are not included."""
        monkeypatch.setenv(
            EXTERNAL_UPLOAD_DIRS, "/nonexistent/path1,/nonexistent/path2"
        )
        result = get_external_upload_dirs()
        assert result == []

    def test_existing_dirs_are_included(self, monkeypatch):
        """Test that existing directories are included."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dir1 = Path(tmpdir) / "uploads1"
            dir2 = Path(tmpdir) / "uploads2"
            dir1.mkdir()
            dir2.mkdir()

            monkeypatch.setenv(EXTERNAL_UPLOAD_DIRS, f"{dir1},{dir2}")
            result = get_external_upload_dirs()
            assert len(result) == 2
            assert dir1 in result
            assert dir2 in result

    def test_expands_environment_tilde_and_preserves_symlink_spelling(
        self, tmp_path, monkeypatch
    ):
        physical = tmp_path / "physical" / "uploads"
        physical.mkdir(parents=True)
        alias = tmp_path / "alias"
        alias.symlink_to(physical, target_is_directory=True)
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("EXTERNAL_NAME", "alias")
        monkeypatch.setenv(EXTERNAL_UPLOAD_DIRS, "~/$EXTERNAL_NAME")

        assert get_external_upload_dirs() == [alias]

    def test_rejects_symlink_followed_by_dotdot(self, tmp_path, monkeypatch):
        base = tmp_path / "base"
        outside = tmp_path / "outside" / "nested"
        base.mkdir()
        outside.mkdir(parents=True)
        (base / "link").symlink_to(outside, target_is_directory=True)
        monkeypatch.setenv(EXTERNAL_UPLOAD_DIRS, str(base / "link" / ".."))

        with pytest.raises(ExternalUploadsDirConfigurationError, match="two different"):
            get_external_upload_dirs()

    def test_relative_dir_is_pinned_to_first_working_directory(
        self, tmp_path, monkeypatch
    ):
        first_cwd = tmp_path / "first"
        second_cwd = tmp_path / "second"
        external = first_cwd / "relative-external"
        external.mkdir(parents=True)
        second_cwd.mkdir()
        relative_spelling = "relative-external"
        monkeypatch.setenv(EXTERNAL_UPLOAD_DIRS, relative_spelling)

        monkeypatch.chdir(first_cwd)
        assert get_external_upload_dirs() == [external]

        monkeypatch.chdir(second_cwd)
        assert get_external_upload_dirs() == [external]


class TestGetExternalSkillsDirs:
    """Test get_external_skills_dirs() function."""

    def test_no_env_var_returns_empty_list(self, monkeypatch):
        """Test that missing env var returns empty list."""
        monkeypatch.delenv(EXTERNAL_SKILLS_LIBRARY_DIRS, raising=False)
        result = get_external_skills_dirs()
        assert result == []

    def test_tilde_expansion(self, monkeypatch):
        """Test that tilde (~) is expanded to home directory."""
        monkeypatch.setenv(EXTERNAL_SKILLS_LIBRARY_DIRS, "~/skills")
        result = get_external_skills_dirs()
        assert len(result) == 1
        assert result[0] == Path.home() / "skills"

    def test_env_var_expansion(self, monkeypatch):
        """Test that environment variables in paths are expanded."""
        monkeypatch.setenv("CUSTOM_SKILLS_DIR", "/opt/skills")
        monkeypatch.setenv(EXTERNAL_SKILLS_LIBRARY_DIRS, "$CUSTOM_SKILLS_DIR")
        result = get_external_skills_dirs()
        assert len(result) == 1
        assert result[0] == Path("/opt/skills")

    def test_url_like_paths_are_skipped(self, monkeypatch):
        """Test that URL-like paths are skipped with warning."""
        monkeypatch.setenv(EXTERNAL_SKILLS_LIBRARY_DIRS, "https://example.com/skills")
        result = get_external_skills_dirs()
        assert result == []


class TestGetStorageRoot:
    """Test get_storage_root() function."""

    def test_default_storage_root(self, monkeypatch):
        """Test default storage root path."""
        monkeypatch.delenv(STORAGE_ROOT, raising=False)
        result = get_storage_root()
        assert result == Path.home() / ".xagent"

    def test_storage_root_with_env_var(self, monkeypatch):
        """Test storage root with environment variable."""
        monkeypatch.setenv(STORAGE_ROOT, "/custom/storage")
        result = get_storage_root()
        assert result == Path("/custom/storage")

    def test_storage_root_expands_tilde(self, monkeypatch, tmp_path):
        """A ~-prefixed env value resolves to the home directory."""
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv(STORAGE_ROOT, "~/custom-root")
        result = get_storage_root()
        assert result == tmp_path / "custom-root"


class TestGetSandboxImage:
    """Test get_sandbox_image() function."""

    def test_default_sandbox_image(self, monkeypatch):
        """Test default sandbox image name."""
        monkeypatch.delenv(SANDBOX_IMAGE, raising=False)
        result = get_sandbox_image()
        assert result == "xprobe/xagent-sandbox:latest"

    def test_sandbox_image_with_env_var(self, monkeypatch):
        """Test sandbox image with environment variable."""
        monkeypatch.setenv(SANDBOX_IMAGE, "custom/sandbox:v1.0")
        result = get_sandbox_image()
        assert result == "custom/sandbox:v1.0"


class TestGetLancedbPath:
    """Test get_lancedb_path() function."""

    def test_default_lancedb_path(self, monkeypatch):
        """Test default LanceDB path (relative to storage root)."""
        monkeypatch.delenv(LANCEDB_PATH, raising=False)
        monkeypatch.delenv(STORAGE_ROOT, raising=False)
        result = get_lancedb_path()
        assert result == Path.home() / ".xagent" / "data" / "lancedb"

    def test_lancedb_path_with_env_var(self, monkeypatch):
        """Test LanceDB path with environment variable."""
        monkeypatch.setenv(LANCEDB_PATH, "/custom/lancedb")
        result = get_lancedb_path()
        assert result == Path("/custom/lancedb")


class TestGoogleDriveDownloadTimeout:
    """Test the Google Drive LRO timeout configuration."""

    def test_constant_name(self):
        assert (
            config.GOOGLE_DRIVE_DOWNLOAD_TIMEOUT_SECONDS
            == "XAGENT_GOOGLE_DRIVE_DOWNLOAD_TIMEOUT_SECONDS"
        )

    def test_default(self, monkeypatch):
        monkeypatch.delenv(
            "XAGENT_GOOGLE_DRIVE_DOWNLOAD_TIMEOUT_SECONDS", raising=False
        )
        assert config.get_google_drive_download_timeout_seconds() == 600

    def test_environment_override(self, monkeypatch):
        monkeypatch.setenv("XAGENT_GOOGLE_DRIVE_DOWNLOAD_TIMEOUT_SECONDS", "120")
        assert config.get_google_drive_download_timeout_seconds() == 120

    @pytest.mark.parametrize("value", ["0", "-1", "not-a-number"])
    def test_invalid_value_uses_default(self, monkeypatch, value):
        monkeypatch.setenv("XAGENT_GOOGLE_DRIVE_DOWNLOAD_TIMEOUT_SECONDS", value)
        assert config.get_google_drive_download_timeout_seconds() == 600


class TestGetKbCollectionsTimeoutSeconds:
    """Test get_kb_collections_timeout_seconds() function."""

    def test_env_var_constant(self):
        assert KB_COLLECTIONS_TIMEOUT_SECONDS == "XAGENT_KB_COLLECTIONS_TIMEOUT_SECONDS"

    def test_default(self, monkeypatch):
        monkeypatch.delenv(KB_COLLECTIONS_TIMEOUT_SECONDS, raising=False)
        assert get_kb_collections_timeout_seconds() == 30

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv(KB_COLLECTIONS_TIMEOUT_SECONDS, "90")
        assert get_kb_collections_timeout_seconds() == 90

    def test_invalid_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv(KB_COLLECTIONS_TIMEOUT_SECONDS, "not-a-number")
        assert get_kb_collections_timeout_seconds() == 30

    def test_non_positive_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv(KB_COLLECTIONS_TIMEOUT_SECONDS, "0")
        assert get_kb_collections_timeout_seconds() == 30


class TestGetKbSearchTimeoutSeconds:
    """Test get_kb_search_timeout_seconds() function."""

    def test_env_var_constant(self):
        assert KB_SEARCH_TIMEOUT_SECONDS == "XAGENT_KB_SEARCH_TIMEOUT_SECONDS"

    def test_default(self, monkeypatch):
        monkeypatch.delenv(KB_SEARCH_TIMEOUT_SECONDS, raising=False)
        assert get_kb_search_timeout_seconds() == 60

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv(KB_SEARCH_TIMEOUT_SECONDS, "5")
        assert get_kb_search_timeout_seconds() == 5

    def test_invalid_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv(KB_SEARCH_TIMEOUT_SECONDS, "not-a-number")
        assert get_kb_search_timeout_seconds() == 60

    def test_non_positive_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv(KB_SEARCH_TIMEOUT_SECONDS, "0")
        assert get_kb_search_timeout_seconds() == 60


class TestDeepDocRemoteParsingConfig:
    """Test the remote DeepDoc parsing (Xinference) configuration getters."""

    def test_env_var_constants(self):
        assert DEEPDOC_XINFERENCE_URL == "XAGENT_DEEPDOC_XINFERENCE_URL"
        assert DEEPDOC_XINFERENCE_API_KEY == "XAGENT_DEEPDOC_XINFERENCE_API_KEY"
        assert (
            DEEPDOC_XINFERENCE_TIMEOUT_SECONDS
            == "XAGENT_DEEPDOC_XINFERENCE_TIMEOUT_SECONDS"
        )
        assert DEEPDOC_XINFERENCE_MODEL_UID == "XAGENT_DEEPDOC_XINFERENCE_MODEL_UID"
        assert DEEPDOC_XINFERENCE_USERNAME == "XAGENT_DEEPDOC_XINFERENCE_USERNAME"
        assert DEEPDOC_XINFERENCE_PASSWORD == "XAGENT_DEEPDOC_XINFERENCE_PASSWORD"

    def test_url_unset_defaults_to_local_mode(self, monkeypatch):
        monkeypatch.delenv(DEEPDOC_XINFERENCE_URL, raising=False)
        assert get_deepdoc_xinference_url() is None

    def test_url_blank_defaults_to_local_mode(self, monkeypatch):
        monkeypatch.setenv(DEEPDOC_XINFERENCE_URL, "   ")
        assert get_deepdoc_xinference_url() is None

    def test_url_strips_whitespace_and_trailing_slash(self, monkeypatch):
        monkeypatch.setenv(DEEPDOC_XINFERENCE_URL, " http://gpu-host:9997/ ")
        assert get_deepdoc_xinference_url() == "http://gpu-host:9997"

    @pytest.mark.parametrize(
        "value",
        [
            "gpu-host:9997",
            "ftp://gpu-host:9997",
            "https:///missing-host",
            "http://gpu-host:9997?model=deepdoc",
            "http://gpu-host:9997#deepdoc",
            "http://gpu-host:9997/v1/deepdoc?model=deepdoc",
        ],
    )
    def test_url_rejects_invalid_values(self, monkeypatch, value):
        monkeypatch.setenv(DEEPDOC_XINFERENCE_URL, value)

        with pytest.raises(ValueError, match="XAGENT_DEEPDOC_XINFERENCE_URL"):
            get_deepdoc_xinference_url()

    def test_api_key_dedicated_var_wins(self, monkeypatch):
        monkeypatch.setenv(DEEPDOC_XINFERENCE_API_KEY, " deepdoc-key ")
        monkeypatch.setenv("XINFERENCE_API_KEY", "shared-key")
        assert get_deepdoc_xinference_api_key() == "deepdoc-key"

    def test_api_key_falls_back_to_shared_xinference_key(self, monkeypatch):
        monkeypatch.delenv(DEEPDOC_XINFERENCE_API_KEY, raising=False)
        monkeypatch.setenv("XINFERENCE_API_KEY", " shared-key ")
        assert get_deepdoc_xinference_api_key() == "shared-key"

    def test_api_key_blank_dedicated_var_falls_back(self, monkeypatch):
        monkeypatch.setenv(DEEPDOC_XINFERENCE_API_KEY, "  ")
        monkeypatch.setenv("XINFERENCE_API_KEY", "shared-key")
        assert get_deepdoc_xinference_api_key() == "shared-key"

    def test_api_key_unset_means_no_auth(self, monkeypatch):
        monkeypatch.delenv(DEEPDOC_XINFERENCE_API_KEY, raising=False)
        monkeypatch.delenv("XINFERENCE_API_KEY", raising=False)
        assert get_deepdoc_xinference_api_key() is None

    def test_timeout_default(self, monkeypatch):
        monkeypatch.delenv(DEEPDOC_XINFERENCE_TIMEOUT_SECONDS, raising=False)
        assert get_deepdoc_xinference_timeout_seconds() == 1800

    def test_timeout_env_override(self, monkeypatch):
        monkeypatch.setenv(DEEPDOC_XINFERENCE_TIMEOUT_SECONDS, "600")
        assert get_deepdoc_xinference_timeout_seconds() == 600

    @pytest.mark.parametrize("value", ["not-a-number", "", "0", "-1"])
    def test_timeout_invalid_or_non_positive_falls_back(self, monkeypatch, value):
        monkeypatch.setenv(DEEPDOC_XINFERENCE_TIMEOUT_SECONDS, value)
        assert get_deepdoc_xinference_timeout_seconds() == 1800

    def test_model_uid_defaults_to_the_family_name(self, monkeypatch):
        monkeypatch.delenv(DEEPDOC_XINFERENCE_MODEL_UID, raising=False)
        assert get_deepdoc_xinference_model_uid() == "DeepDoc"

    def test_model_uid_env_override_is_trimmed(self, monkeypatch):
        monkeypatch.setenv(DEEPDOC_XINFERENCE_MODEL_UID, "  deepdoc-gpu-1  ")
        assert get_deepdoc_xinference_model_uid() == "deepdoc-gpu-1"

    @pytest.mark.parametrize("value", ["", "   "])
    def test_model_uid_blank_falls_back_to_the_default(self, monkeypatch, value):
        monkeypatch.setenv(DEEPDOC_XINFERENCE_MODEL_UID, value)
        assert get_deepdoc_xinference_model_uid() == "DeepDoc"

    def test_username_unset_means_no_token_exchange(self, monkeypatch):
        monkeypatch.delenv(DEEPDOC_XINFERENCE_USERNAME, raising=False)
        assert get_deepdoc_xinference_username() is None

    def test_username_is_trimmed(self, monkeypatch):
        monkeypatch.setenv(DEEPDOC_XINFERENCE_USERNAME, "  admin  ")
        assert get_deepdoc_xinference_username() == "admin"

    @pytest.mark.parametrize("value", ["", "   "])
    def test_username_blank_means_no_token_exchange(self, monkeypatch, value):
        monkeypatch.setenv(DEEPDOC_XINFERENCE_USERNAME, value)
        assert get_deepdoc_xinference_username() is None

    def test_password_unset_means_no_token_exchange(self, monkeypatch):
        monkeypatch.delenv(DEEPDOC_XINFERENCE_PASSWORD, raising=False)
        assert get_deepdoc_xinference_password() is None

    def test_password_blank_means_no_token_exchange(self, monkeypatch):
        monkeypatch.setenv(DEEPDOC_XINFERENCE_PASSWORD, "")
        assert get_deepdoc_xinference_password() is None

    def test_password_preserves_surrounding_whitespace(self, monkeypatch):
        """Whitespace can be significant in a secret, so it is not stripped."""
        monkeypatch.setenv(DEEPDOC_XINFERENCE_PASSWORD, "  pa ss  ")
        assert get_deepdoc_xinference_password() == "  pa ss  "


class TestGetDefaultSqliteDbPath:
    """Test get_default_sqlite_db_path() function."""

    def test_default_sqlite_db_path(self, monkeypatch):
        """Test default SQLite database path."""
        monkeypatch.delenv(STORAGE_ROOT, raising=False)
        result = get_default_sqlite_db_path()
        assert result == str(Path.home() / ".xagent" / "xagent.db")

    def test_sqlite_db_path_respects_storage_root(self, monkeypatch):
        """Test that SQLite path respects STORAGE_ROOT env var."""
        monkeypatch.setenv(STORAGE_ROOT, "/custom/storage")
        result = get_default_sqlite_db_path()
        assert result == "/custom/storage/xagent.db"


class TestGetDatabaseUrl:
    """Test get_database_url() function."""

    def test_default_database_url(self, monkeypatch):
        """Test default database URL (SQLite)."""
        monkeypatch.delenv(DATABASE_URL, raising=False)
        monkeypatch.delenv(STORAGE_ROOT, raising=False)
        result = get_database_url()
        assert result.startswith("sqlite:///")
        assert result.endswith("xagent.db")

    def test_database_url_with_env_var(self, monkeypatch):
        """Test database URL with environment variable."""
        monkeypatch.setenv(DATABASE_URL, "postgresql://user:pass@localhost/db")
        result = get_database_url()
        assert result == "postgresql://user:pass@localhost/db"


class TestDbPoolConfig:
    """Test DB pool sizing getters."""

    def test_defaults(self, monkeypatch):
        monkeypatch.delenv(DB_POOL_SIZE, raising=False)
        monkeypatch.delenv(DB_MAX_OVERFLOW, raising=False)
        monkeypatch.delenv(DB_POOL_TIMEOUT_SECONDS, raising=False)
        assert get_db_pool_size() == 10
        assert get_db_max_overflow() == 20
        assert get_db_pool_timeout_seconds() == 30

    def test_env_overrides(self, monkeypatch):
        monkeypatch.setenv(DB_POOL_SIZE, "25")
        monkeypatch.setenv(DB_MAX_OVERFLOW, "0")
        monkeypatch.setenv(DB_POOL_TIMEOUT_SECONDS, "5")
        assert get_db_pool_size() == 25
        assert get_db_max_overflow() == 0
        assert get_db_pool_timeout_seconds() == 5

    def test_invalid_values_fall_back(self, monkeypatch):
        monkeypatch.setenv(DB_POOL_SIZE, "abc")
        monkeypatch.setenv(DB_MAX_OVERFLOW, "-1")
        monkeypatch.setenv(DB_POOL_TIMEOUT_SECONDS, "0")
        assert get_db_pool_size() == 10
        assert get_db_max_overflow() == 20
        assert get_db_pool_timeout_seconds() == 30


class TestMcpToolInitTimeout:
    """Test get_mcp_tool_init_timeout_seconds() function."""

    def test_default(self, monkeypatch):
        monkeypatch.delenv(MCP_TOOL_INIT_TIMEOUT_SECONDS, raising=False)
        assert get_mcp_tool_init_timeout_seconds() == 60

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv(MCP_TOOL_INIT_TIMEOUT_SECONDS, "120")
        assert get_mcp_tool_init_timeout_seconds() == 120

    def test_zero_disables(self, monkeypatch):
        monkeypatch.setenv(MCP_TOOL_INIT_TIMEOUT_SECONDS, "0")
        assert get_mcp_tool_init_timeout_seconds() == 0

    def test_invalid_falls_back(self, monkeypatch):
        monkeypatch.setenv(MCP_TOOL_INIT_TIMEOUT_SECONDS, "not-a-number")
        assert get_mcp_tool_init_timeout_seconds() == 60


class TestGetSandboxCpus:
    """Test get_sandbox_cpus() function."""

    def test_no_env_var_returns_none(self, monkeypatch):
        """Test that missing env var returns None."""
        monkeypatch.delenv(SANDBOX_CPUS, raising=False)
        result = get_sandbox_cpus()
        assert result is None

    def test_valid_cpu_count(self, monkeypatch):
        """Test valid CPU count from env var."""
        monkeypatch.setenv(SANDBOX_CPUS, "4")
        result = get_sandbox_cpus()
        assert result == 4

    def test_invalid_cpu_count_returns_none(self, monkeypatch):
        """Test that invalid CPU count returns None."""
        monkeypatch.setenv(SANDBOX_CPUS, "invalid")
        result = get_sandbox_cpus()
        assert result is None


class TestGetSandboxMemory:
    """Test get_sandbox_memory() function."""

    def test_no_env_var_returns_none(self, monkeypatch):
        """Test that missing env var returns None."""
        monkeypatch.delenv(SANDBOX_MEMORY, raising=False)
        result = get_sandbox_memory()
        assert result is None

    def test_valid_memory_value(self, monkeypatch):
        """Test valid memory value from env var."""
        monkeypatch.setenv(SANDBOX_MEMORY, "2048")
        result = get_sandbox_memory()
        assert result == 2048

    def test_invalid_memory_value_returns_none(self, monkeypatch):
        """Test that invalid memory value returns None."""
        monkeypatch.setenv(SANDBOX_MEMORY, "invalid")
        result = get_sandbox_memory()
        assert result is None


class TestGetSandboxEnv:
    """Test get_sandbox_env() function."""

    def test_no_env_var_returns_empty_dict(self, monkeypatch):
        """Test that missing env var returns empty dict."""
        monkeypatch.delenv(SANDBOX_ENV, raising=False)
        result = get_sandbox_env()
        assert result == {}

    def test_empty_env_var_returns_empty_dict(self, monkeypatch):
        """Test that empty env var returns empty dict."""
        monkeypatch.setenv(SANDBOX_ENV, "")
        result = get_sandbox_env()
        assert result == {}

    def test_valid_env_config(self, monkeypatch):
        """Test valid environment variable configuration."""
        monkeypatch.setenv(SANDBOX_ENV, "KEY1=value1;KEY2=value2")
        result = get_sandbox_env()
        assert result == {"KEY1": "value1", "KEY2": "value2"}

    def test_env_config_with_spaces(self, monkeypatch):
        """Test that spaces around keys/values are trimmed."""
        monkeypatch.setenv(SANDBOX_ENV, " KEY1 = value1 ; KEY2 = value2 ")
        result = get_sandbox_env()
        assert result == {"KEY1": "value1", "KEY2": "value2"}


class TestGetSandboxVolumes:
    """Test get_sandbox_volumes() function."""

    def test_no_env_var_returns_empty_list(self, monkeypatch):
        """Test that missing env var returns empty list."""
        monkeypatch.delenv(SANDBOX_VOLUMES, raising=False)
        result = get_sandbox_volumes()
        assert result == []

    def test_empty_env_var_returns_empty_list(self, monkeypatch):
        """Test that empty env var returns empty list."""
        monkeypatch.setenv(SANDBOX_VOLUMES, "")
        result = get_sandbox_volumes()
        assert result == []

    def test_valid_volume_config(self, monkeypatch):
        """Test valid volume configuration."""
        monkeypatch.setenv(SANDBOX_VOLUMES, "/host:/container:ro")
        result = get_sandbox_volumes()
        assert len(result) == 1
        assert result[0] == ("/host", "/container", "ro")

    def test_volume_with_explicit_mode(self, monkeypatch):
        """Test volume configuration with explicit mode."""
        monkeypatch.setenv(SANDBOX_VOLUMES, "/host:/container:rw")
        result = get_sandbox_volumes()
        assert result[0][2] == "rw"

    def test_volume_defaults_to_readonly(self, monkeypatch):
        """Test that volume defaults to readonly mode."""
        monkeypatch.setenv(SANDBOX_VOLUMES, "/host:/container")
        result = get_sandbox_volumes()
        assert result[0][2] == "ro"

    def test_invalid_mode_defaults_to_readonly(self, monkeypatch):
        """Test that invalid mode defaults to readonly."""
        monkeypatch.setenv(SANDBOX_VOLUMES, "/host:/container:invalid")
        result = get_sandbox_volumes()
        assert result[0][2] == "ro"

    def test_tilde_expansion_in_volume_src(self, monkeypatch):
        """Test that tilde is expanded in volume source path."""
        monkeypatch.setenv(SANDBOX_VOLUMES, "~/data:/container:ro")
        result = get_sandbox_volumes()
        assert result[0][0] == str(Path.home() / "data")

    def test_multiple_volumes(self, monkeypatch):
        """Test multiple volume configurations."""
        monkeypatch.setenv(
            SANDBOX_VOLUMES, "/host1:/container1:ro;/host2:/container2:rw"
        )
        result = get_sandbox_volumes()
        assert len(result) == 2
        assert result[0] == ("/host1", "/container1", "ro")
        assert result[1] == ("/host2", "/container2", "rw")

    def test_host_side_sources_preserve_absolute_paths(self, monkeypatch):
        """Docker sibling volume sources are already host paths."""
        monkeypatch.setenv(SANDBOX_VOLUMES, "/host/data:/container:rw")
        result = get_sandbox_volumes(host_side_sources=True)
        assert result == [("/host/data", "/container", "rw")]

    def test_host_side_sources_reject_relative_paths(self, monkeypatch):
        """Docker sibling mode should not absolutize relative paths in backend."""
        monkeypatch.setenv(SANDBOX_VOLUMES, "relative/path:/container:ro")
        result = get_sandbox_volumes(host_side_sources=True)
        assert result == []

    def test_host_side_sources_reject_tilde_paths(self, monkeypatch):
        """Docker sibling mode should not expand backend-container home paths."""
        monkeypatch.setenv(SANDBOX_VOLUMES, "~/data:/container:ro")
        result = get_sandbox_volumes(host_side_sources=True)
        assert result == []


class TestGetSandboxHostProjectRoot:
    """Test get_sandbox_host_project_root() function."""

    def test_no_env_var_returns_none(self, monkeypatch):
        """Test that missing env var returns None."""
        monkeypatch.delenv(SANDBOX_HOST_PROJECT_ROOT, raising=False)
        result = get_sandbox_host_project_root()
        assert result is None

    def test_project_root_with_env_var(self, monkeypatch):
        """Test project root with environment variable."""
        monkeypatch.setenv(SANDBOX_HOST_PROJECT_ROOT, "/host/xagent")
        result = get_sandbox_host_project_root()
        assert result == Path("/host/xagent")

    def test_project_root_expands_env_vars_without_user_or_abspath(self, monkeypatch):
        """Host paths should not be resolved against the backend container."""
        monkeypatch.setenv("HOST_PROJECT_ROOT", "/host/xagent")
        monkeypatch.setenv(SANDBOX_HOST_PROJECT_ROOT, "$HOST_PROJECT_ROOT/../xagent")
        result = get_sandbox_host_project_root()
        assert result == Path("/host/xagent/../xagent")


class TestGetSandboxHostStorageRoot:
    """Test get_sandbox_host_storage_root() function."""

    def test_no_env_var_returns_none(self, monkeypatch):
        """Test that missing env var returns None."""
        monkeypatch.delenv(SANDBOX_HOST_STORAGE_ROOT, raising=False)
        result = get_sandbox_host_storage_root()
        assert result is None

    def test_storage_root_with_env_var(self, monkeypatch):
        """Test storage root with environment variable."""
        monkeypatch.setenv(SANDBOX_HOST_STORAGE_ROOT, "/host/.xagent")
        result = get_sandbox_host_storage_root()
        assert result == Path("/host/.xagent")

    def test_storage_root_expands_env_vars_without_user_or_abspath(self, monkeypatch):
        """Host paths should not be resolved against the backend container."""
        monkeypatch.setenv("HOST_STORAGE_ROOT", "/host/.xagent")
        monkeypatch.setenv(SANDBOX_HOST_STORAGE_ROOT, "$HOST_STORAGE_ROOT/../.xagent")
        result = get_sandbox_host_storage_root()
        assert result == Path("/host/.xagent/../.xagent")


class TestGetSandboxNamespace:
    """Test get_sandbox_namespace() function."""

    def test_no_env_var_returns_none(self, monkeypatch):
        """Test that a missing namespace returns None."""
        monkeypatch.delenv(SANDBOX_NAMESPACE, raising=False)
        result = get_sandbox_namespace()
        assert result is None

    def test_blank_env_var_returns_none(self, monkeypatch):
        """Test that a blank namespace returns None."""
        monkeypatch.setenv(SANDBOX_NAMESPACE, "   ")
        result = get_sandbox_namespace()
        assert result is None

    def test_valid_namespace(self, monkeypatch):
        """Test that a Compose-project-name-shaped namespace is accepted."""
        monkeypatch.setenv(SANDBOX_NAMESPACE, "my-project-1")
        result = get_sandbox_namespace()
        assert result == "my-project-1"

    @pytest.mark.parametrize(
        "value",
        [
            "Upper-Case",
            "-leading-dash",
            "has space",
            ":",
            "a::b",
            "a/b",
            "a.b",
            "café",
        ],
    )
    def test_invalid_namespace_raises(self, monkeypatch, value):
        """Test that malformed namespaces raise instead of silently degrading."""
        monkeypatch.setenv(SANDBOX_NAMESPACE, value)
        with pytest.raises(ValueError, match="Invalid sandbox namespace"):
            get_sandbox_namespace()

    def test_validation_helper_rejects_empty_namespace(self):
        """Direct callers must not bypass the getter's blank-to-None policy."""
        with pytest.raises(ValueError, match="Invalid sandbox namespace"):
            validate_sandbox_namespace("")

    def test_long_namespace_accepted_like_compose(self, monkeypatch):
        """Compose accepts arbitrary-length project names; so must we.

        Container identities hash the full namespace, label values preserve
        it, and snapshot repository names use a bounded sanitized token plus
        a digest, so backend identifiers do not impose a namespace length cap.
        """
        monkeypatch.setenv(SANDBOX_NAMESPACE, "a" * 100)
        assert get_sandbox_namespace() == "a" * 100


class TestGetBoxliteHomeDir:
    """Test get_boxlite_home_dir() function."""

    def test_no_env_var_returns_none(self, monkeypatch):
        """Test that missing env var returns None."""
        monkeypatch.delenv(BOXLITE_HOME_DIR, raising=False)
        result = get_boxlite_home_dir()
        assert result is None

    def test_boxlite_home_dir_with_env_var(self, monkeypatch):
        """Test BoxLite home directory with environment variable."""
        monkeypatch.setenv(BOXLITE_HOME_DIR, "/custom/boxlite")
        result = get_boxlite_home_dir()
        assert result == Path("/custom/boxlite")


class TestGetMaxTracePayloadBytes:
    """Test get_max_trace_payload_bytes() function."""

    def test_default(self, monkeypatch):
        monkeypatch.delenv(MAX_TRACE_PAYLOAD_BYTES, raising=False)
        assert get_max_trace_payload_bytes() == 50_000

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv(MAX_TRACE_PAYLOAD_BYTES, "1234")
        assert get_max_trace_payload_bytes() == 1234

    def test_zero_passes_through(self, monkeypatch):
        """Zero disables truncation (handled by truncate_for_trace)."""
        monkeypatch.setenv(MAX_TRACE_PAYLOAD_BYTES, "0")
        assert get_max_trace_payload_bytes() == 0

    def test_invalid_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv(MAX_TRACE_PAYLOAD_BYTES, "not-a-number")
        assert get_max_trace_payload_bytes() == 50_000

    def test_negative_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv(MAX_TRACE_PAYLOAD_BYTES, "-100")
        assert get_max_trace_payload_bytes() == 50_000


class TestToolConcurrencyConfig:
    """Config for in-turn tool concurrency (design §7)."""

    def test_parallel_enabled_default_is_false(self, monkeypatch):
        from xagent.config import get_tool_parallel_enabled

        monkeypatch.delenv("XAGENT_TOOL_PARALLEL_ENABLED", raising=False)
        assert get_tool_parallel_enabled() is False

    def test_parallel_enabled_env_override(self, monkeypatch):
        from xagent.config import get_tool_parallel_enabled

        monkeypatch.setenv("XAGENT_TOOL_PARALLEL_ENABLED", "true")
        assert get_tool_parallel_enabled() is True
        monkeypatch.setenv("XAGENT_TOOL_PARALLEL_ENABLED", "0")
        assert get_tool_parallel_enabled() is False

    def test_max_concurrency_default_is_three(self, monkeypatch):
        from xagent.config import get_tool_max_concurrency

        monkeypatch.delenv("XAGENT_TOOL_MAX_CONCURRENCY", raising=False)
        assert get_tool_max_concurrency() == 3

    def test_max_concurrency_env_override(self, monkeypatch):
        from xagent.config import get_tool_max_concurrency

        monkeypatch.setenv("XAGENT_TOOL_MAX_CONCURRENCY", "8")
        assert get_tool_max_concurrency() == 8

    def test_max_concurrency_invalid_falls_back_to_default(self, monkeypatch):
        from xagent.config import get_tool_max_concurrency

        monkeypatch.setenv("XAGENT_TOOL_MAX_CONCURRENCY", "not-a-number")
        assert get_tool_max_concurrency() == 3
        monkeypatch.setenv("XAGENT_TOOL_MAX_CONCURRENCY", "0")
        assert get_tool_max_concurrency() == 3


class TestTaskRuntimeHookConfig:
    def test_defaults(self, monkeypatch):
        monkeypatch.delenv(TASK_RUNTIME_HOOK_MAX_WORKERS, raising=False)
        monkeypatch.delenv(TASK_RUNTIME_HOOK_QUEUE_TIMEOUT_SECONDS, raising=False)

        assert get_task_runtime_hook_max_workers() == 8
        assert get_task_runtime_hook_queue_timeout_seconds() == 30

    def test_env_overrides(self, monkeypatch):
        monkeypatch.setenv(TASK_RUNTIME_HOOK_MAX_WORKERS, "12")
        monkeypatch.setenv(TASK_RUNTIME_HOOK_QUEUE_TIMEOUT_SECONDS, "45")

        assert get_task_runtime_hook_max_workers() == 12
        assert get_task_runtime_hook_queue_timeout_seconds() == 45

    def test_invalid_values_fall_back(self, monkeypatch):
        monkeypatch.setenv(TASK_RUNTIME_HOOK_MAX_WORKERS, "0")
        monkeypatch.setenv(TASK_RUNTIME_HOOK_QUEUE_TIMEOUT_SECONDS, "invalid")

        assert get_task_runtime_hook_max_workers() == 8
        assert get_task_runtime_hook_queue_timeout_seconds() == 30


class TestLocalBrowserConfig:
    def test_defaults(self, monkeypatch):
        for name in (
            NATIVE_BROWSER_ENABLED,
            NATIVE_BROWSER_APP_NAME,
            BROWSER_CUA_DRIVER_COMMAND,
            BROWSER_CUA_DRIVER_SOCKET,
            BROWSER_CUA_DRIVER_TIMEOUT_SECONDS,
            BROWSER_CUA_DRIVER_MAX_ELEMENTS,
        ):
            monkeypatch.delenv(name, raising=False)

        assert get_native_browser_enabled() is False
        assert get_native_browser_app_name() == "Google Chrome"
        assert get_browser_cua_driver_command() == "cua-driver"
        assert get_browser_cua_driver_socket() is None
        assert get_browser_cua_driver_timeout_seconds() == 30
        assert get_browser_cua_driver_max_elements() == 2_000

    def test_env_overrides(self, monkeypatch):
        monkeypatch.setenv(NATIVE_BROWSER_ENABLED, "true")
        monkeypatch.setenv(NATIVE_BROWSER_APP_NAME, "Chromium")
        monkeypatch.setenv(BROWSER_CUA_DRIVER_COMMAND, "/opt/bin/cua-driver")
        monkeypatch.setenv(BROWSER_CUA_DRIVER_SOCKET, "/tmp/cua.sock")
        monkeypatch.setenv(BROWSER_CUA_DRIVER_TIMEOUT_SECONDS, "12.5")
        monkeypatch.setenv(BROWSER_CUA_DRIVER_MAX_ELEMENTS, "250")

        assert get_native_browser_enabled() is True
        assert get_native_browser_app_name() == "Chromium"
        assert get_browser_cua_driver_command() == "/opt/bin/cua-driver"
        assert get_browser_cua_driver_socket() == "/tmp/cua.sock"
        assert get_browser_cua_driver_timeout_seconds() == 12.5
        assert get_browser_cua_driver_max_elements() == 250

    def test_invalid_values_fall_back(self, monkeypatch):
        monkeypatch.setenv(BROWSER_CUA_DRIVER_TIMEOUT_SECONDS, "0")
        monkeypatch.setenv(BROWSER_CUA_DRIVER_MAX_ELEMENTS, "invalid")

        assert get_browser_cua_driver_timeout_seconds() == 30
        assert get_browser_cua_driver_max_elements() == 2_000

    def test_rejects_non_browser_native_application(self, monkeypatch):
        monkeypatch.setenv(NATIVE_BROWSER_APP_NAME, "Terminal")

        with pytest.raises(ValueError, match="must name a supported browser"):
            get_native_browser_app_name()


class TestBrowserToolDefaultLocaleConfig:
    """Fallback locale/timezone for browser_use (Playwright) sessions."""

    def test_defaults(self, monkeypatch):
        monkeypatch.delenv(config.BROWSER_TOOL_DEFAULT_LOCALE, raising=False)
        monkeypatch.delenv(config.BROWSER_TOOL_DEFAULT_TIMEZONE, raising=False)

        assert config.get_browser_tool_default_locale() == "en-US"
        assert config.get_browser_tool_default_timezone() is None

    def test_env_overrides(self, monkeypatch):
        monkeypatch.setenv(config.BROWSER_TOOL_DEFAULT_LOCALE, "ja-JP")
        monkeypatch.setenv(config.BROWSER_TOOL_DEFAULT_TIMEZONE, "Asia/Tokyo")

        assert config.get_browser_tool_default_locale() == "ja-JP"
        assert config.get_browser_tool_default_timezone() == "Asia/Tokyo"

    def test_blank_values_fall_back(self, monkeypatch):
        monkeypatch.setenv(config.BROWSER_TOOL_DEFAULT_LOCALE, "   ")
        monkeypatch.setenv(config.BROWSER_TOOL_DEFAULT_TIMEZONE, "")

        assert config.get_browser_tool_default_locale() == "en-US"
        assert config.get_browser_tool_default_timezone() is None


class TestCheckpointStorageConfig:
    """Config for checkpoint trace-event storage encoding and retention."""

    def test_encoding_v2_default_is_true(self, monkeypatch):
        from xagent.config import get_checkpoint_encoding_v2_enabled

        monkeypatch.delenv("XAGENT_CHECKPOINT_ENCODING_V2", raising=False)
        assert get_checkpoint_encoding_v2_enabled() is True

    def test_encoding_v2_env_override(self, monkeypatch):
        from xagent.config import get_checkpoint_encoding_v2_enabled

        monkeypatch.setenv("XAGENT_CHECKPOINT_ENCODING_V2", "false")
        assert get_checkpoint_encoding_v2_enabled() is False
        monkeypatch.setenv("XAGENT_CHECKPOINT_ENCODING_V2", "1")
        assert get_checkpoint_encoding_v2_enabled() is True

    def test_history_limit_default_is_eight(self, monkeypatch):
        from xagent.config import get_checkpoint_history_limit

        monkeypatch.delenv("XAGENT_CHECKPOINT_HISTORY_LIMIT", raising=False)
        assert get_checkpoint_history_limit() == 8

    def test_history_limit_env_override_and_zero_disables(self, monkeypatch):
        from xagent.config import get_checkpoint_history_limit

        monkeypatch.setenv("XAGENT_CHECKPOINT_HISTORY_LIMIT", "20")
        assert get_checkpoint_history_limit() == 20
        monkeypatch.setenv("XAGENT_CHECKPOINT_HISTORY_LIMIT", "0")
        assert get_checkpoint_history_limit() == 0

    def test_history_limit_invalid_falls_back_to_default(self, monkeypatch):
        from xagent.config import get_checkpoint_history_limit

        monkeypatch.setenv("XAGENT_CHECKPOINT_HISTORY_LIMIT", "not-a-number")
        assert get_checkpoint_history_limit() == 8
        monkeypatch.setenv("XAGENT_CHECKPOINT_HISTORY_LIMIT", "-1")
        assert get_checkpoint_history_limit() == 8


class TestSandboxConcurrencyConfig:
    """Config for sandbox worker concurrency."""

    def test_sandbox_max_concurrency_default_is_three(self, monkeypatch):
        from xagent.config import get_sandbox_max_concurrency

        monkeypatch.delenv("XAGENT_SANDBOX_MAX_CONCURRENCY", raising=False)
        assert get_sandbox_max_concurrency() == 3

    def test_sandbox_max_concurrency_env_override(self, monkeypatch):
        from xagent.config import get_sandbox_max_concurrency

        monkeypatch.setenv("XAGENT_SANDBOX_MAX_CONCURRENCY", "5")
        assert get_sandbox_max_concurrency() == 5

    def test_sandbox_max_concurrency_invalid_falls_back_to_default(self, monkeypatch):
        from xagent.config import get_sandbox_max_concurrency

        monkeypatch.setenv("XAGENT_SANDBOX_MAX_CONCURRENCY", "not-a-number")
        assert get_sandbox_max_concurrency() == 3
        monkeypatch.setenv("XAGENT_SANDBOX_MAX_CONCURRENCY", "0")
        assert get_sandbox_max_concurrency() == 3


class TestTriggerRateLimitConfig:
    """Config for trigger callback and CRUD rate limits."""

    def test_callback_rate_limit_default(self, monkeypatch):
        from xagent.config import get_trigger_callback_rate_limit

        monkeypatch.delenv("XAGENT_TRIGGER_CALLBACK_RATE_LIMIT", raising=False)
        assert get_trigger_callback_rate_limit() == "120/minute"

    def test_callback_rate_limit_env_override(self, monkeypatch):
        from xagent.config import get_trigger_callback_rate_limit

        monkeypatch.setenv("XAGENT_TRIGGER_CALLBACK_RATE_LIMIT", "10/second")
        assert get_trigger_callback_rate_limit() == "10/second"

    def test_crud_rate_limit_default(self, monkeypatch):
        from xagent.config import get_trigger_crud_rate_limit

        monkeypatch.delenv("XAGENT_TRIGGER_CRUD_RATE_LIMIT", raising=False)
        assert get_trigger_crud_rate_limit() == "60/minute"

    def test_crud_rate_limit_env_override(self, monkeypatch):
        from xagent.config import get_trigger_crud_rate_limit

        monkeypatch.setenv("XAGENT_TRIGGER_CRUD_RATE_LIMIT", "5/minute")
        assert get_trigger_crud_rate_limit() == "5/minute"


class TestShareRateLimitConfig:
    """Config for public share-channel rate limits and run quotas (#973)."""

    _CASES = [
        ("get_share_auth_rate_limit", "XAGENT_SHARE_AUTH_RATE_LIMIT", "60/minute"),
        (
            "get_share_auth_ip_rate_limit",
            "XAGENT_SHARE_AUTH_IP_RATE_LIMIT",
            "300/minute",
        ),
        (
            "get_share_task_create_rate_limit",
            "XAGENT_SHARE_TASK_CREATE_RATE_LIMIT",
            "30/minute",
        ),
        (
            "get_share_task_create_token_rate_limit",
            "XAGENT_SHARE_TASK_CREATE_TOKEN_RATE_LIMIT",
            "120/minute",
        ),
        (
            "get_share_ws_turn_rate_limit",
            "XAGENT_SHARE_WS_TURN_RATE_LIMIT",
            "60/minute",
        ),
        (
            "get_share_ws_connect_ip_rate_limit",
            "XAGENT_SHARE_WS_CONNECT_IP_RATE_LIMIT",
            "120/minute",
        ),
        ("get_share_upload_rate_limit", "XAGENT_SHARE_UPLOAD_RATE_LIMIT", "60/minute"),
        (
            "get_widget_upload_rate_limit",
            "XAGENT_WIDGET_UPLOAD_RATE_LIMIT",
            "240/minute",
        ),
        (
            "get_widget_upload_ip_rate_limit",
            "XAGENT_WIDGET_UPLOAD_IP_RATE_LIMIT",
            "60/minute",
        ),
        ("get_share_run_quota", "XAGENT_SHARE_RUN_QUOTA", "500/day"),
        ("get_share_run_guest_quota", "XAGENT_SHARE_RUN_GUEST_QUOTA", "60/hour"),
        (
            "get_widget_ws_connect_ip_rate_limit",
            "XAGENT_WIDGET_WS_CONNECT_IP_RATE_LIMIT",
            "120/minute",
        ),
        (
            "get_widget_ws_turn_ip_rate_limit",
            "XAGENT_WIDGET_WS_TURN_IP_RATE_LIMIT",
            "60/minute",
        ),
        (
            "get_widget_ws_turn_rate_limit",
            "XAGENT_WIDGET_WS_TURN_RATE_LIMIT",
            "240/minute",
        ),
        (
            "get_widget_auth_rate_limit",
            "XAGENT_WIDGET_AUTH_RATE_LIMIT",
            "1200/minute",
        ),
        (
            "get_widget_auth_ip_rate_limit",
            "XAGENT_WIDGET_AUTH_IP_RATE_LIMIT",
            "300/minute",
        ),
        (
            "get_widget_task_create_rate_limit",
            "XAGENT_WIDGET_TASK_CREATE_RATE_LIMIT",
            "240/minute",
        ),
        (
            "get_widget_task_create_ip_rate_limit",
            "XAGENT_WIDGET_TASK_CREATE_IP_RATE_LIMIT",
            "60/minute",
        ),
        ("get_widget_run_quota", "XAGENT_WIDGET_RUN_QUOTA", "500/day"),
        ("get_widget_run_ip_quota", "XAGENT_WIDGET_RUN_IP_QUOTA", "120/hour"),
    ]

    @pytest.mark.parametrize("func_name,env_var,default", _CASES)
    def test_default(self, monkeypatch, func_name, env_var, default):
        import xagent.config as config

        monkeypatch.delenv(env_var, raising=False)
        assert getattr(config, func_name)() == default

    @pytest.mark.parametrize("func_name,env_var,default", _CASES)
    def test_env_override(self, monkeypatch, func_name, env_var, default):
        import xagent.config as config

        monkeypatch.setenv(env_var, "7/second")
        assert getattr(config, func_name)() == "7/second"

    @pytest.mark.parametrize("func_name,env_var,default", _CASES)
    def test_blank_env_falls_back_to_default(
        self, monkeypatch, func_name, env_var, default
    ):
        import xagent.config as config

        monkeypatch.setenv(env_var, "   ")
        assert getattr(config, func_name)() == default


class TestOrphanUploadGcConfig:
    """Config for task-less public-upload orphan GC (#973)."""

    def test_ttl_default(self, monkeypatch):
        from xagent.config import get_taskless_upload_ttl_seconds

        monkeypatch.delenv("XAGENT_TASKLESS_UPLOAD_TTL_SECONDS", raising=False)
        assert get_taskless_upload_ttl_seconds() == 48 * 60 * 60

    def test_ttl_env_override(self, monkeypatch):
        from xagent.config import get_taskless_upload_ttl_seconds

        monkeypatch.setenv("XAGENT_TASKLESS_UPLOAD_TTL_SECONDS", "86400")
        assert get_taskless_upload_ttl_seconds() == 86400

    def test_ttl_below_minimum_falls_back_to_default(self, monkeypatch):
        from xagent.config import get_taskless_upload_ttl_seconds

        # Below the 1h floor -> default (guards against reaping mid-first-turn).
        monkeypatch.setenv("XAGENT_TASKLESS_UPLOAD_TTL_SECONDS", "5")
        assert get_taskless_upload_ttl_seconds() == 48 * 60 * 60

    def test_sweep_interval_default(self, monkeypatch):
        from xagent.config import get_orphan_upload_sweep_interval_seconds

        monkeypatch.delenv("XAGENT_ORPHAN_UPLOAD_SWEEP_INTERVAL_SECONDS", raising=False)
        assert get_orphan_upload_sweep_interval_seconds() == 3600

    def test_sweep_interval_env_override(self, monkeypatch):
        from xagent.config import get_orphan_upload_sweep_interval_seconds

        monkeypatch.setenv("XAGENT_ORPHAN_UPLOAD_SWEEP_INTERVAL_SECONDS", "900")
        assert get_orphan_upload_sweep_interval_seconds() == 900


class TestWorkforcePreviewRunReapConfig:
    """PR #1060 review: get_workforce_preview_run_stale_seconds() had no
    test, unlike its sibling TTL config functions above."""

    def test_default(self, monkeypatch):
        from xagent.config import get_workforce_preview_run_stale_seconds

        monkeypatch.delenv("XAGENT_WORKFORCE_PREVIEW_RUN_STALE_SECONDS", raising=False)
        assert get_workforce_preview_run_stale_seconds() == 7200

    def test_env_override(self, monkeypatch):
        from xagent.config import get_workforce_preview_run_stale_seconds

        monkeypatch.setenv("XAGENT_WORKFORCE_PREVIEW_RUN_STALE_SECONDS", "3600")
        assert get_workforce_preview_run_stale_seconds() == 3600

    def test_below_minimum_falls_back_to_default(self, monkeypatch):
        from xagent.config import get_workforce_preview_run_stale_seconds

        # Below the 300s floor -> default (guards against reaping a preview
        # run that is still genuinely in progress).
        monkeypatch.setenv("XAGENT_WORKFORCE_PREVIEW_RUN_STALE_SECONDS", "5")
        assert get_workforce_preview_run_stale_seconds() == 7200

    def test_invalid_value_falls_back_to_default(self, monkeypatch):
        from xagent.config import get_workforce_preview_run_stale_seconds

        monkeypatch.setenv("XAGENT_WORKFORCE_PREVIEW_RUN_STALE_SECONDS", "not-a-number")
        assert get_workforce_preview_run_stale_seconds() == 7200


class TestGmailPubSubProvisioningConfig:
    """Config for per-mailbox Gmail Pub/Sub provisioning."""

    def test_constants(self):
        assert GMAIL_PUBSUB_PROJECT_ID == "XAGENT_GMAIL_PUBSUB_PROJECT_ID"
        assert GMAIL_PUBSUB_TOPIC_PREFIX == "XAGENT_GMAIL_PUBSUB_TOPIC_PREFIX"
        assert (
            GMAIL_PUBSUB_SUBSCRIPTION_PREFIX
            == "XAGENT_GMAIL_PUBSUB_SUBSCRIPTION_PREFIX"
        )
        assert (
            GMAIL_PUBSUB_PUSH_SERVICE_ACCOUNT
            == "XAGENT_GMAIL_PUBSUB_PUSH_SERVICE_ACCOUNT"
        )
        assert (
            GMAIL_REGISTRATION_TIMEOUT_SECONDS
            == "XAGENT_GMAIL_REGISTRATION_TIMEOUT_SECONDS"
        )
        assert PUBLIC_API_BASE_URL == "XAGENT_PUBLIC_API_BASE_URL"
        assert S2S_API_BASE_URL == "XAGENT_S2S_API_BASE_URL"

    def test_defaults(self, monkeypatch):
        monkeypatch.delenv("XAGENT_GMAIL_PUBSUB_PROJECT_ID", raising=False)
        monkeypatch.delenv("XAGENT_GMAIL_PUBSUB_TOPIC_PREFIX", raising=False)
        monkeypatch.delenv("XAGENT_GMAIL_PUBSUB_SUBSCRIPTION_PREFIX", raising=False)
        monkeypatch.delenv("XAGENT_GMAIL_PUBSUB_PUSH_SERVICE_ACCOUNT", raising=False)
        monkeypatch.delenv("XAGENT_GMAIL_REGISTRATION_TIMEOUT_SECONDS", raising=False)
        monkeypatch.delenv("XAGENT_PUBLIC_API_BASE_URL", raising=False)
        monkeypatch.delenv("XAGENT_S2S_API_BASE_URL", raising=False)

        assert get_gmail_pubsub_project_id() is None
        assert get_gmail_pubsub_topic_prefix() == "xagent-gmail"
        assert get_gmail_pubsub_subscription_prefix() == "xagent-gmail-push"
        assert get_gmail_pubsub_push_service_account() is None
        assert get_gmail_registration_timeout_seconds() == 10
        assert get_public_api_base_url() is None
        assert get_s2s_api_base_url() is None

    def test_env_overrides(self, monkeypatch):
        monkeypatch.setenv("XAGENT_GMAIL_PUBSUB_PROJECT_ID", " demo ")
        monkeypatch.setenv("XAGENT_GMAIL_PUBSUB_TOPIC_PREFIX", " mail-topic ")
        monkeypatch.setenv("XAGENT_GMAIL_PUBSUB_SUBSCRIPTION_PREFIX", " mail-sub ")
        monkeypatch.setenv(
            "XAGENT_GMAIL_PUBSUB_PUSH_SERVICE_ACCOUNT",
            " push@demo.iam.gserviceaccount.com ",
        )
        monkeypatch.setenv("XAGENT_GMAIL_REGISTRATION_TIMEOUT_SECONDS", "3")
        monkeypatch.setenv("XAGENT_PUBLIC_API_BASE_URL", " https://api.example.com/ ")
        monkeypatch.setenv(
            "XAGENT_S2S_API_BASE_URL", " https://sg-origin.example.com/ "
        )

        assert get_gmail_pubsub_project_id() == "demo"
        assert get_gmail_pubsub_topic_prefix() == "mail-topic"
        assert get_gmail_pubsub_subscription_prefix() == "mail-sub"
        assert (
            get_gmail_pubsub_push_service_account()
            == "push@demo.iam.gserviceaccount.com"
        )
        assert get_gmail_registration_timeout_seconds() == 3
        assert get_public_api_base_url() == "https://api.example.com"
        assert get_s2s_api_base_url() == "https://sg-origin.example.com"

    def test_s2s_base_falls_back_to_public_api_base(self, monkeypatch):
        monkeypatch.delenv("XAGENT_S2S_API_BASE_URL", raising=False)
        monkeypatch.setenv(
            "XAGENT_PUBLIC_API_BASE_URL", " https://api.example.com/base/ "
        )

        assert get_s2s_api_base_url() == "https://api.example.com/base"

    @pytest.mark.parametrize(
        "value",
        [
            "api.example.com",
            "ftp://api.example.com",
            "https:///missing-host",
        ],
    )
    def test_s2s_base_rejects_invalid_http_urls(self, monkeypatch, value) -> None:
        monkeypatch.setenv("XAGENT_S2S_API_BASE_URL", value)

        with pytest.raises(ValueError, match="XAGENT_S2S_API_BASE_URL"):
            get_s2s_api_base_url()

    @pytest.mark.parametrize(
        "value",
        [
            "https://api.example.com/base?region=sg",
            "https://api.example.com/base#callbacks",
        ],
    )
    def test_s2s_base_rejects_query_and_fragment(self, monkeypatch, value) -> None:
        monkeypatch.setenv("XAGENT_S2S_API_BASE_URL", value)

        with pytest.raises(ValueError, match="XAGENT_S2S_API_BASE_URL"):
            get_s2s_api_base_url()

    def test_s2s_base_validates_public_api_fallback(self, monkeypatch) -> None:
        monkeypatch.delenv("XAGENT_S2S_API_BASE_URL", raising=False)
        monkeypatch.setenv(
            "XAGENT_PUBLIC_API_BASE_URL",
            "https://api.example.com?region=sg",
        )

        with pytest.raises(ValueError, match="XAGENT_PUBLIC_API_BASE_URL"):
            get_s2s_api_base_url()

    def test_gmail_callback_validates_legacy_fallback(self, monkeypatch) -> None:
        monkeypatch.delenv("XAGENT_S2S_API_BASE_URL", raising=False)
        monkeypatch.setenv(
            "XAGENT_TRIGGER_CALLBACK_BASE_URL",
            "https://legacy-callback.example.com#gmail",
        )

        resolver = getattr(config, "get_gmail_callback_base_url", lambda: None)
        with pytest.raises(ValueError, match="XAGENT_TRIGGER_CALLBACK_BASE_URL"):
            resolver()

    def test_gmail_callback_preserves_legacy_base_url_fallback(self, monkeypatch):
        monkeypatch.delenv("XAGENT_S2S_API_BASE_URL", raising=False)
        monkeypatch.setenv(
            "XAGENT_TRIGGER_CALLBACK_BASE_URL",
            " https://legacy-callback.example.com/ ",
        )
        monkeypatch.setenv("XAGENT_PUBLIC_API_BASE_URL", "https://api.example.com")

        assert (
            getattr(config, "TRIGGER_CALLBACK_BASE_URL", None)
            == "XAGENT_TRIGGER_CALLBACK_BASE_URL"
        )
        resolver = getattr(config, "get_gmail_callback_base_url", lambda: None)
        assert resolver() == "https://legacy-callback.example.com"
        assert get_s2s_api_base_url() == "https://api.example.com"

    def test_invalid_timeout_uses_default(self, monkeypatch):
        monkeypatch.setenv("XAGENT_GMAIL_REGISTRATION_TIMEOUT_SECONDS", "0")
        assert get_gmail_registration_timeout_seconds() == 10
        monkeypatch.setenv("XAGENT_GMAIL_REGISTRATION_TIMEOUT_SECONDS", "not-a-number")
        assert get_gmail_registration_timeout_seconds() == 10

    def test_pubsub_transport_defaults_to_grpc(self, monkeypatch):
        from xagent.config import get_gmail_pubsub_transport

        monkeypatch.delenv("XAGENT_GMAIL_PUBSUB_TRANSPORT", raising=False)
        assert get_gmail_pubsub_transport() == "grpc"
        monkeypatch.setenv("XAGENT_GMAIL_PUBSUB_TRANSPORT", " REST ")
        assert get_gmail_pubsub_transport() == "rest"
        monkeypatch.setenv("XAGENT_GMAIL_PUBSUB_TRANSPORT", "carrier-pigeon")
        assert get_gmail_pubsub_transport() == "grpc"


class TestTrustedProxyHopsConfig:
    """Config for proxy-aware remote IP derivation."""

    def test_default_is_zero(self, monkeypatch):
        from xagent.config import get_trusted_proxy_hops

        monkeypatch.delenv("XAGENT_TRUSTED_PROXY_HOPS", raising=False)
        assert get_trusted_proxy_hops() == 0

    def test_env_override(self, monkeypatch):
        from xagent.config import get_trusted_proxy_hops

        monkeypatch.setenv("XAGENT_TRUSTED_PROXY_HOPS", "2")
        assert get_trusted_proxy_hops() == 2

    def test_invalid_value_falls_back_to_zero(self, monkeypatch):
        from xagent.config import get_trusted_proxy_hops

        monkeypatch.setenv("XAGENT_TRUSTED_PROXY_HOPS", "not-a-number")
        assert get_trusted_proxy_hops() == 0
        monkeypatch.setenv("XAGENT_TRUSTED_PROXY_HOPS", "-1")
        assert get_trusted_proxy_hops() == 0


class TestTriggerCallbackIpRateLimitConfig:
    """Config for the IP-wide callback rate ceiling."""

    def test_default(self, monkeypatch):
        from xagent.config import get_trigger_callback_ip_rate_limit

        monkeypatch.delenv("XAGENT_TRIGGER_CALLBACK_IP_RATE_LIMIT", raising=False)
        assert get_trigger_callback_ip_rate_limit() == "600/minute"

    def test_env_override(self, monkeypatch):
        from xagent.config import get_trigger_callback_ip_rate_limit

        monkeypatch.setenv("XAGENT_TRIGGER_CALLBACK_IP_RATE_LIMIT", "50/second")
        assert get_trigger_callback_ip_rate_limit() == "50/second"


class TestGetSandboxIdleTtl:
    """Test get_sandbox_idle_ttl() function."""

    def test_no_env_var_disables_reclamation(self, monkeypatch):
        monkeypatch.delenv(SANDBOX_IDLE_TTL, raising=False)
        assert get_sandbox_idle_ttl() is None

    def test_blank_env_var_disables_reclamation(self, monkeypatch):
        monkeypatch.setenv(SANDBOX_IDLE_TTL, "   ")
        assert get_sandbox_idle_ttl() is None

    def test_valid_ttl_seconds(self, monkeypatch):
        monkeypatch.setenv(SANDBOX_IDLE_TTL, "86400")
        assert get_sandbox_idle_ttl() == 86400.0

    def test_fractional_ttl_seconds(self, monkeypatch):
        monkeypatch.setenv(SANDBOX_IDLE_TTL, "0.5")
        assert get_sandbox_idle_ttl() == 0.5

    def test_invalid_ttl_disables_reclamation(self, monkeypatch):
        monkeypatch.setenv(SANDBOX_IDLE_TTL, "invalid")
        assert get_sandbox_idle_ttl() is None

    def test_non_positive_ttl_disables_reclamation(self, monkeypatch):
        monkeypatch.setenv(SANDBOX_IDLE_TTL, "0")
        assert get_sandbox_idle_ttl() is None
        monkeypatch.setenv(SANDBOX_IDLE_TTL, "-60")
        assert get_sandbox_idle_ttl() is None

    def test_non_finite_ttl_disables_reclamation(self, monkeypatch):
        for value in ("nan", "NaN", "inf", "-inf"):
            monkeypatch.setenv(SANDBOX_IDLE_TTL, value)
            assert get_sandbox_idle_ttl() is None


class TestGetSandboxSweepInterval:
    """Test get_sandbox_sweep_interval() function."""

    def test_no_env_var_returns_default(self, monkeypatch):
        monkeypatch.delenv(SANDBOX_SWEEP_INTERVAL, raising=False)
        assert get_sandbox_sweep_interval() == 60.0

    def test_valid_interval(self, monkeypatch):
        monkeypatch.setenv(SANDBOX_SWEEP_INTERVAL, "5")
        assert get_sandbox_sweep_interval() == 5.0

    def test_invalid_interval_returns_default(self, monkeypatch):
        monkeypatch.setenv(SANDBOX_SWEEP_INTERVAL, "invalid")
        assert get_sandbox_sweep_interval() == 60.0

    def test_non_positive_interval_returns_default(self, monkeypatch):
        monkeypatch.setenv(SANDBOX_SWEEP_INTERVAL, "0")
        assert get_sandbox_sweep_interval() == 60.0

    def test_non_finite_interval_returns_default(self, monkeypatch):
        for value in ("nan", "inf", "-inf"):
            monkeypatch.setenv(SANDBOX_SWEEP_INTERVAL, value)
            assert get_sandbox_sweep_interval() == 60.0


class TestGetSandboxMaxContainers:
    """Test get_sandbox_max_containers() function."""

    def test_no_env_var_disables_cap(self, monkeypatch):
        monkeypatch.delenv(SANDBOX_MAX_CONTAINERS, raising=False)
        assert get_sandbox_max_containers() is None

    def test_valid_cap(self, monkeypatch):
        monkeypatch.setenv(SANDBOX_MAX_CONTAINERS, "20")
        assert get_sandbox_max_containers() == 20

    def test_invalid_cap_disables_cap(self, monkeypatch):
        monkeypatch.setenv(SANDBOX_MAX_CONTAINERS, "invalid")
        assert get_sandbox_max_containers() is None

    def test_non_positive_cap_disables_cap(self, monkeypatch):
        monkeypatch.setenv(SANDBOX_MAX_CONTAINERS, "0")
        assert get_sandbox_max_containers() is None
        monkeypatch.setenv(SANDBOX_MAX_CONTAINERS, "-5")
        assert get_sandbox_max_containers() is None


class TestGetSandboxAllowLocalFallbackOnCapacity:
    """Test get_sandbox_allow_local_fallback_on_capacity() function."""

    def test_defaults_to_false(self, monkeypatch):
        monkeypatch.delenv(SANDBOX_ALLOW_LOCAL_FALLBACK_ON_CAPACITY, raising=False)
        assert get_sandbox_allow_local_fallback_on_capacity() is False

    def test_enabled_values(self, monkeypatch):
        for value in ("1", "true", "True", "yes", "on"):
            monkeypatch.setenv(SANDBOX_ALLOW_LOCAL_FALLBACK_ON_CAPACITY, value)
            assert get_sandbox_allow_local_fallback_on_capacity() is True

    def test_disabled_values(self, monkeypatch):
        for value in ("0", "false", "off", "junk"):
            monkeypatch.setenv(SANDBOX_ALLOW_LOCAL_FALLBACK_ON_CAPACITY, value)
            assert get_sandbox_allow_local_fallback_on_capacity() is False


class TestCompactThresholdConfig:
    """Config for context-compaction threshold derivation."""

    def test_ratio_defaults_to_075(self, monkeypatch):
        from xagent.config import COMPACT_THRESHOLD_RATIO, get_compact_threshold_ratio

        monkeypatch.delenv(COMPACT_THRESHOLD_RATIO, raising=False)
        assert get_compact_threshold_ratio() == 0.75

    def test_ratio_env_override(self, monkeypatch):
        from xagent.config import COMPACT_THRESHOLD_RATIO, get_compact_threshold_ratio

        monkeypatch.setenv(COMPACT_THRESHOLD_RATIO, "0.8")
        assert get_compact_threshold_ratio() == 0.8

    @pytest.mark.parametrize("value", ["abc", "0", "-0.5", "1.5", "0.0"])
    def test_ratio_invalid_or_out_of_range_falls_back(self, monkeypatch, value):
        from xagent.config import COMPACT_THRESHOLD_RATIO, get_compact_threshold_ratio

        monkeypatch.setenv(COMPACT_THRESHOLD_RATIO, value)
        assert get_compact_threshold_ratio() == 0.75

    def test_default_defaults_to_32000(self, monkeypatch):
        from xagent.config import (
            COMPACT_THRESHOLD_DEFAULT,
            get_compact_threshold_default,
        )

        monkeypatch.delenv(COMPACT_THRESHOLD_DEFAULT, raising=False)
        assert get_compact_threshold_default() == 32000

    def test_default_env_override(self, monkeypatch):
        from xagent.config import (
            COMPACT_THRESHOLD_DEFAULT,
            get_compact_threshold_default,
        )

        monkeypatch.setenv(COMPACT_THRESHOLD_DEFAULT, "64000")
        assert get_compact_threshold_default() == 64000

    @pytest.mark.parametrize("value", ["abc", "0", "-1"])
    def test_default_invalid_or_non_positive_falls_back(self, monkeypatch, value):
        from xagent.config import (
            COMPACT_THRESHOLD_DEFAULT,
            get_compact_threshold_default,
        )

        monkeypatch.setenv(COMPACT_THRESHOLD_DEFAULT, value)
        assert get_compact_threshold_default() == 32000


class TestUrlUserinfoRejectionIsScopedToDeepDoc:
    """Only the DeepDoc URL rejects embedded credentials.

    The rejection guards a specific hazard: the remote client interpolates
    ``httpx.HTTPStatusError`` — which renders the URL unredacted — into an
    exception the caller logs. Putting the check inside the shared
    ``_normalized_http_env_url`` would also apply it to
    ``PUBLIC_API_BASE_URL``/``S2S_API_BASE_URL``, whose call sites do not catch
    ``ValueError`` and depend on ``or``-chained fallbacks, so a pre-existing
    (if ill-advised) config would start failing at runtime.
    """

    def test_deepdoc_url_rejects_credentials(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(DEEPDOC_XINFERENCE_URL, "http://user:pw@gpu-host:9997")
        with pytest.raises(ValueError, match="credentials embedded"):
            get_deepdoc_xinference_url()

    def test_deepdoc_url_without_credentials_is_accepted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(DEEPDOC_XINFERENCE_URL, "http://gpu-host:9997/base")
        assert get_deepdoc_xinference_url() == "http://gpu-host:9997/base"

    def test_shared_helper_consumers_keep_working(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The other consumers of the shared helper must be unaffected."""
        monkeypatch.setenv(PUBLIC_API_BASE_URL, "http://user:pw@api.example.com")
        monkeypatch.delenv(S2S_API_BASE_URL, raising=False)

        assert get_public_api_base_url() == "http://user:pw@api.example.com"
        assert get_s2s_api_base_url() == "http://user:pw@api.example.com"
