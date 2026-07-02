"""Unit tests for core/config.py configuration functions."""

import tempfile
from pathlib import Path
from tempfile import gettempdir

import pytest

from xagent.config import (
    AGENT_RUNTIME,
    APP_BASE_URL,
    BACKGROUND_JOB_MAX_RETRIES,
    BACKGROUND_JOB_STALE_SECONDS,
    BACKGROUND_JOB_SWEEP_INTERVAL_SECONDS,
    BACKGROUND_JOB_VISIBILITY_TIMEOUT_SECONDS,
    BOXLITE_HOME_DIR,
    CELERY_BROKER_URL,
    CELERY_ENABLED,
    CELERY_RESULT_BACKEND,
    DATABASE_URL,
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
    FRONTEND_DIST_DIR,
    HOT_PATH_CACHE_ENABLED,
    HOT_PATH_CACHE_TTL_SECONDS,
    HOT_PATH_TASK_CACHE_TTL_SECONDS,
    LANCEDB_PATH,
    MAX_TRACE_PAYLOAD_BYTES,
    MAX_UPLOAD_SIZE,
    OPENROUTER_OFFICIAL_PROVIDERS_ONLY,
    PASSWORD_RESET_EXPIRE_MINUTES,
    PREVIEW_TMP_DIR,
    REDIS_URL,
    SANDBOX_CPUS,
    SANDBOX_ENV,
    SANDBOX_HOST_PROJECT_ROOT,
    SANDBOX_HOST_STORAGE_ROOT,
    SANDBOX_IMAGE,
    SANDBOX_MEMORY,
    SANDBOX_VOLUMES,
    SMTP_FROM_EMAIL,
    SMTP_FROM_NAME,
    SMTP_HOST,
    SMTP_PASSWORD,
    SMTP_PORT,
    SMTP_USE_SSL,
    SMTP_USE_TLS,
    SMTP_USERNAME,
    STORAGE_ROOT,
    TRIGGER_DISPATCHER_BATCH_SIZE,
    TRIGGER_DISPATCHER_ENABLED,
    TRIGGER_DISPATCHER_INTERVAL_SECONDS,
    UPLOADS_DIR,
    WEB_CRAWL_TLS_IMPERSONATE,
    WEB_DIR,
    WEB_SEARCH_PROVIDER,
    format_file_size,
    get_agent_pattern_for_execution_mode,
    get_agent_runtime,
    get_app_base_url,
    get_background_job_max_retries,
    get_background_job_stale_seconds,
    get_background_job_sweep_interval_seconds,
    get_background_job_visibility_timeout_seconds,
    get_boxlite_home_dir,
    get_celery_broker_url,
    get_celery_enabled,
    get_celery_result_backend,
    get_database_url,
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
    get_frontend_dist_dir,
    get_hot_path_cache_enabled,
    get_hot_path_cache_ttl_seconds,
    get_hot_path_task_cache_ttl_seconds,
    get_lancedb_path,
    get_max_trace_payload_bytes,
    get_max_upload_size_bytes,
    get_openrouter_official_providers_only,
    get_password_reset_expire_minutes,
    get_preview_tmp_dir,
    get_redis_url,
    get_sandbox_cpus,
    get_sandbox_env,
    get_sandbox_host_project_root,
    get_sandbox_host_storage_root,
    get_sandbox_image,
    get_sandbox_memory,
    get_sandbox_volumes,
    get_smtp_from_email,
    get_smtp_from_name,
    get_smtp_host,
    get_smtp_password,
    get_smtp_port,
    get_smtp_use_ssl,
    get_smtp_use_tls,
    get_smtp_username,
    get_storage_root,
    get_trigger_dispatcher_batch_size,
    get_trigger_dispatcher_enabled,
    get_trigger_dispatcher_interval_seconds,
    get_uploads_dir,
    get_web_crawl_tls_impersonate,
    get_web_dir,
    get_web_search_provider,
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

    def test_sandbox_image_constant(self):
        assert SANDBOX_IMAGE == "SANDBOX_IMAGE"

    def test_sandbox_host_project_root_constant(self):
        assert SANDBOX_HOST_PROJECT_ROOT == "XAGENT_SANDBOX_HOST_PROJECT_ROOT"

    def test_sandbox_host_storage_root_constant(self):
        assert SANDBOX_HOST_STORAGE_ROOT == "XAGENT_SANDBOX_HOST_STORAGE_ROOT"

    def test_lancedb_path_constant(self):
        assert LANCEDB_PATH == "LANCEDB_PATH"

    def test_database_url_constant(self):
        assert DATABASE_URL == "DATABASE_URL"

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
