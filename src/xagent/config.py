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

import logging
import os
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

# Environment variable names
UPLOADS_DIR = "XAGENT_UPLOADS_DIR"
WEB_DIR = "XAGENT_WEB_DIR"
EXTERNAL_UPLOAD_DIRS = "XAGENT_EXTERNAL_UPLOAD_DIRS"
EXTERNAL_SKILLS_LIBRARY_DIRS = "XAGENT_EXTERNAL_SKILLS_LIBRARY_DIRS"
AGENT_RUNTIME = "XAGENT_AGENT_RUNTIME"
TASK_LEASE_TTL_SECONDS = "XAGENT_TASK_LEASE_TTL_SECONDS"
TASK_LEASE_HEARTBEAT_SECONDS = "XAGENT_TASK_LEASE_HEARTBEAT_SECONDS"
STORAGE_ROOT = "XAGENT_STORAGE_ROOT"
MAX_UPLOAD_SIZE = "XAGENT_MAX_UPLOAD_SIZE"
SANDBOX_IMAGE = "SANDBOX_IMAGE"
LANCEDB_PATH = "LANCEDB_PATH"
DATABASE_URL = "DATABASE_URL"
SANDBOX_CPUS = "SANDBOX_CPUS"
SANDBOX_MEMORY = "SANDBOX_MEMORY"
SANDBOX_ENV = "SANDBOX_ENV"
SANDBOX_VOLUMES = "SANDBOX_VOLUMES"
BOXLITE_HOME_DIR = "BOXLITE_HOME_DIR"
WEB_SEARCH_PROVIDER = "XAGENT_WEB_SEARCH_PROVIDER"

TOOL_MAX_OUTPUT_LENGTH = "XAGENT_TOOL_MAX_OUTPUT_LENGTH"
TOOL_MAX_RECURSION_DEPTH = "XAGENT_TOOL_MAX_RECURSION_DEPTH"
TOOL_MAX_FIELD_COUNT = "XAGENT_TOOL_MAX_FIELD_COUNT"

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

    Standalone v2 tasks default to auto so simple prompts can answer directly
    while complex prompts can still route into ReAct or DAG. v1 keeps the legacy
    standalone DAG default for compatibility. Agent Builder tasks keep balanced
    in both runtimes because the agent's explicit tool/KB setup is usually
    better served by ReAct.
    """
    if agent_id is not None:
        return "balanced"

    runtime = (agent_runtime or get_agent_runtime()).strip().lower()
    if runtime == "v2":
        return "auto"
    return "think"


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


def get_sandbox_volumes() -> list[tuple[str, str, str]]:
    """Get the volume mappings for sandbox containers.

    Format: src:dst[:mode];src2:dst2[:mode2]
    - src: source path on host (expanded ~ and env vars)
    - dst: destination path in container
    - mode: ro or rw (default: ro)

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

        src = os.path.expanduser(os.path.expandvars(parts[0].strip()))
        dst = parts[1].strip()
        if not src or not dst:
            logger.warning(f"Invalid sandbox volume: {item}")
            continue

        # Normalize paths to resolve any relative components
        src = os.path.abspath(src)
        mode = parts[2].strip().lower() if len(parts) > 2 else "ro"
        if mode not in ("ro", "rw"):
            logger.warning(f"Invalid sandbox volume mode: {item}, using 'ro'")
            mode = "ro"

        volumes.append((src, dst, mode))

    return volumes


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
