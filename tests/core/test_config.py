"""Unit tests for core/config.py configuration functions."""

import tempfile
from pathlib import Path

import pytest

from xagent.config import (
    AGENT_RUNTIME,
    BOXLITE_HOME_DIR,
    DATABASE_URL,
    EXTERNAL_SKILLS_LIBRARY_DIRS,
    EXTERNAL_UPLOAD_DIRS,
    LANCEDB_PATH,
    MAX_UPLOAD_SIZE,
    SANDBOX_CPUS,
    SANDBOX_ENV,
    SANDBOX_IMAGE,
    SANDBOX_MEMORY,
    SANDBOX_VOLUMES,
    STORAGE_ROOT,
    UPLOADS_DIR,
    WEB_DIR,
    WEB_SEARCH_PROVIDER,
    format_file_size,
    get_agent_pattern_for_execution_mode,
    get_agent_runtime,
    get_boxlite_home_dir,
    get_database_url,
    get_default_sqlite_db_path,
    get_default_task_execution_mode,
    get_external_skills_dirs,
    get_external_upload_dirs,
    get_lancedb_path,
    get_max_upload_size_bytes,
    get_sandbox_cpus,
    get_sandbox_env,
    get_sandbox_image,
    get_sandbox_memory,
    get_sandbox_volumes,
    get_storage_root,
    get_uploads_dir,
    get_web_dir,
    get_web_search_provider,
)


class TestEnvironmentVariableConstants:
    """Test environment variable constant names."""

    def test_upload_dir_constant(self):
        assert UPLOADS_DIR == "XAGENT_UPLOADS_DIR"

    def test_web_dir_constant(self):
        assert WEB_DIR == "XAGENT_WEB_DIR"

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

    def test_lancedb_path_constant(self):
        assert LANCEDB_PATH == "LANCEDB_PATH"

    def test_database_url_constant(self):
        assert DATABASE_URL == "DATABASE_URL"

    def test_max_upload_size_constant(self):
        assert MAX_UPLOAD_SIZE == "XAGENT_MAX_UPLOAD_SIZE"

    def test_web_search_provider_constant(self):
        assert WEB_SEARCH_PROVIDER == "XAGENT_WEB_SEARCH_PROVIDER"


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
