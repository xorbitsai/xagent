"""Unified logging configuration for xagent web application."""

import logging
import os
from collections.abc import Callable
from logging.config import dictConfig
from typing import Literal, cast

from ..core.utils.security import redact_log_text
from .request_log_redaction import install_request_log_redaction

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


_is_applied = False

# Transport loggers whose request lines carry full URLs (query strings
# included). Their level is pinned to WARNING below, but a third-party library
# can call ``setLevel`` on them at any time; a logger-level filter keeps the
# INFO lines out regardless of the level.
_TRANSPORT_LOGGER_NAMES = ("httpx", "httpcore")
# Marker set on the record factory this module wraps around a previous one, so
# repeated ``setup_logging(force=True)`` calls (and a third party wrapping the
# chain later) cannot stack another redaction layer.
_RECORD_FACTORY_MARKER = "_xagent_redacts_log_records"
# Marker set on a record once this module has redacted it; two layers of the
# same factory would otherwise format and scan the message twice.
_RECORD_MARKER = "_xagent_log_redacted"

_EXCEPTION_FORMATTER = logging.Formatter()


class _BelowWarningFilter(logging.Filter):
    """Drop records below WARNING regardless of the logger's level."""

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno >= logging.WARNING


_TRANSPORT_FILTER = _BelowWarningFilter()


def _redacting_record_factory(
    previous: Callable[..., logging.LogRecord],
) -> Callable[..., logging.LogRecord]:
    """Wrap ``previous`` so every record is redacted before any handler sees it.

    A record factory, rather than a root-handler filter, because a handler
    filter misses loggers with ``propagate = False`` and their own handlers,
    handlers a library attaches to ``httpx`` directly, and handlers replaced
    later by ``logging.basicConfig(force=True)``.
    """

    def factory(*args: object, **kwargs: object) -> logging.LogRecord:
        record = previous(*args, **kwargs)
        if getattr(record, _RECORD_MARKER, False):
            return record
        message = record.getMessage()
        redacted = redact_log_text(message)
        if redacted != message:
            # The formatted text is final; leaving the arguments in place
            # would let a formatter interpolate the raw values again.
            record.msg = redacted
            record.args = ()
        if record.exc_text:
            record.exc_text = redact_log_text(record.exc_text)
        elif record.exc_info:
            record.exc_text = redact_log_text(
                _EXCEPTION_FORMATTER.formatException(record.exc_info)
            )
        if record.stack_info:
            record.stack_info = redact_log_text(record.stack_info)
        setattr(record, _RECORD_MARKER, True)
        return record

    setattr(factory, _RECORD_FACTORY_MARKER, True)
    setattr(factory, "__wrapped__", previous)
    return factory


def _install_record_redaction() -> None:
    """Chain the redacting record factory in front of the current one (idempotent)."""
    current = logging.getLogRecordFactory()
    factory: object | None = current
    seen: set[int] = set()
    while factory is not None and id(factory) not in seen:
        seen.add(id(factory))
        if getattr(factory, _RECORD_FACTORY_MARKER, False):
            return
        factory = getattr(factory, "__wrapped__", None)
    logging.setLogRecordFactory(_redacting_record_factory(current))


def _install_transport_log_filters() -> None:
    """Keep httpx/httpcore request lines out of INFO wherever they are handled.

    httpcore logs through child loggers (``httpcore.connection`` and
    friends); a filter on the parent is not consulted when a child record
    propagates, so the child loggers that already exist get it too.
    """
    names = set(_TRANSPORT_LOGGER_NAMES)
    names.update(
        name
        for name, logger in list(logging.Logger.manager.loggerDict.items())
        if isinstance(logger, logging.Logger)
        and name.startswith(("httpx.", "httpcore."))
    )
    for name in sorted(names):
        logger = logging.getLogger(name)
        if _TRANSPORT_FILTER not in logger.filters:
            logger.addFilter(_TRANSPORT_FILTER)


def setup_logging(level: LogLevel | None = None, force: bool = False) -> None:
    """Configure logging for the entire application.

    Args:
        level: Log level. If None, reads from XAGENT_LOG_LEVEL env var,
               defaults to "INFO" if env var is not set or invalid.
        force: If True, reconfigure logging even if already applied.
    """
    global _is_applied
    if _is_applied and not force:
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
                "xagent": {"level": level},
            },
            "root": {
                "level": level,
                "handlers": ["default"],
            },
        }
    )

    # Redact every record at creation: this also covers loggers that do not
    # propagate and handlers a library attaches after this call.
    _install_record_redaction()
    # Keep httpx/httpcore request lines (full URL, query included) out of INFO
    # even when a library lowers those loggers' levels.
    _install_transport_log_filters()
    # Server access and WebSocket handshake lines carry the raw request
    # target; mask credential-like query values before they reach a handler.
    install_request_log_redaction()

    if invalid_level:
        logging.warning(
            "Invalid log level '%r', falling back to 'INFO'", original_level
        )

    _is_applied = True
