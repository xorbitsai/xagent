"""Unified logging configuration for xagent web application."""

import logging
import os
from logging.config import dictConfig
from typing import Literal, cast

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


_is_applied = False


def setup_logging(level: LogLevel | None = None) -> None:
    """Configure logging for the entire application.

    Args:
        level: Log level. If None, reads from XAGENT_LOG_LEVEL env var,
               defaults to "INFO" if env var is not set or invalid.
    """
    global _is_applied
    if _is_applied:
        return
    # Read log level from env var if not provided
    if level is None:
        level = cast(LogLevel, os.getenv("XAGENT_LOG_LEVEL", "INFO").upper())
    else:
        level = cast(LogLevel, level.upper())
    # Validate and fallback to INFO if invalid
    original_level = level
    if invalid_level := level not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
        level = "INFO"
    # apply logging config
    dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(asctime)s %(levelname)-8s %(name)s - %(message)s",
                    "datefmt": "%Y-%m-%d %H:%M:%S",
                }
            },
            "handlers": {
                "default": {
                    "class": "logging.StreamHandler",
                    "formatter": "default",
                }
            },
            "loggers": {
                "aiohttp": {"level": "WARNING"},
                "sqlalchemy": {"level": "WARNING"},
                "urllib3": {"level": "WARNING"},
                "uvicorn.access": {"level": "WARNING"},
                "uvicorn.error": {"level": "INFO"},
                "httpx": {"level": "WARNING"},
                "httpcore": {"level": "WARNING"},
            },
            "root": {
                "level": level,
                "handlers": ["default"],
            },
        }
    )

    if invalid_level:
        logging.warning(
            "Invalid log level '%r', falling back to 'INFO'", original_level
        )

    _is_applied = True
