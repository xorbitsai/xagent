from importlib import resources
from typing import Any

from alembic.config import Config as AlembicConfig
from sqlalchemy import Engine


def create_alembic_config(engine: Engine) -> Any:
    """Create Alembic configuration programmatically."""
    cfg: Any = AlembicConfig()

    # Basic configuration
    migrations_path = resources.files("xagent") / "migrations"
    cfg.set_main_option("script_location", str(migrations_path))
    cfg.set_main_option("prepend_sys_path", ".")
    cfg.set_main_option("version_path_separator", "os")
    cfg.set_main_option("output_encoding", "utf-8")

    # Database URL from engine
    cfg.set_main_option("sqlalchemy.url", str(engine.url))

    # Logging configuration
    cfg.set_main_option("keys", "root,sqlalchemy,alembic")
    cfg.set_section_option("loggers", "keys", "root,sqlalchemy,alembic")
    cfg.set_section_option("handlers", "keys", "console")
    cfg.set_section_option("formatters", "keys", "generic")

    # Logger levels
    cfg.set_section_option("logger_root", "level", "WARN")
    cfg.set_section_option("logger_root", "handlers", "console")
    cfg.set_section_option("logger_root", "qualname", "")

    cfg.set_section_option("logger_sqlalchemy", "level", "WARN")
    cfg.set_section_option("logger_sqlalchemy", "handlers", "")
    cfg.set_section_option("logger_sqlalchemy", "qualname", "sqlalchemy.engine")

    cfg.set_section_option("logger_alembic", "level", "INFO")
    cfg.set_section_option("logger_alembic", "handlers", "")
    cfg.set_section_option("logger_alembic", "qualname", "alembic")

    # Console handler
    cfg.set_section_option("handler_console", "class", "StreamHandler")
    cfg.set_section_option("handler_console", "args", "(sys.stderr,)")
    cfg.set_section_option("handler_console", "level", "NOTSET")
    cfg.set_section_option("handler_console", "formatter", "generic")

    return cfg
