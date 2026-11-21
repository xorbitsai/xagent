"""Unified logging configuration for xagent web application."""

import logging
import os
from collections.abc import Callable
from logging.config import dictConfig
from typing import Any, Literal, cast

from ..core.utils.security import redact_log_text, redact_log_text_pattern_only
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


# Loggers whose record arguments are also read by a formatter of their own.
# uvicorn's ``AccessFormatter`` unpacks the five-tuple in ``record.args`` and
# ignores ``record.msg``, so flattening such a record into a string drops the
# access line entirely. The request target inside ``args[2]`` is redacted
# here (and again by ``RequestTargetRedactionFilter``).
_ARG_PRESERVING_LOGGERS = frozenset({"uvicorn.access"})


def _format_with_args(msg: object, args: object) -> str:
    text = str(msg)
    if args:
        return text % args
    return text


def _redact_argument(value: object) -> object:
    if isinstance(value, str):
        return redact_log_text(value)
    return value


def _redact_args(args: object) -> object:
    """Redact string elements of a tuple/dict of log arguments in place."""
    if isinstance(args, tuple):
        return tuple(_redact_argument(item) for item in args)
    if isinstance(args, dict):
        return {key: _redact_argument(value) for key, value in args.items()}
    return args


def _redact_record(record: logging.LogRecord) -> None:
    """Redact one record, falling back to a pattern-only pass on failure.

    The full pass commits every change only at the end. If it raises anyway,
    a header/assignment/JWT-only pass still masks the common shapes instead
    of leaving the whole record unredacted. If that pass fails too, the
    exception reaches the factory, which returns the record exactly as the
    previous factory produced it.
    """
    if getattr(record, _RECORD_MARKER, False):
        return
    try:
        _redact_record_fully(record)
    except Exception:
        _redact_record_pattern_only(record)


def _redact_record_fully(record: logging.LogRecord) -> None:
    """Redact one record with the full structured redactor."""
    message = record.getMessage()
    redacted_message = redact_log_text(message)

    new_msg = record.msg
    new_args: Any = record.args
    if redacted_message != message:
        redacted_args = _redact_args(record.args)
        try:
            rebuilt = _format_with_args(record.msg, redacted_args)
        except Exception:
            rebuilt = None
        if rebuilt == redacted_message:
            # Every masked value was visible inside one argument. Keep the
            # argument structure: filters and formatters (uvicorn's access
            # formatter in particular) still read it.
            new_args = redacted_args
        elif (
            record.name in _ARG_PRESERVING_LOGGERS
            and isinstance(record.args, tuple)
            and len(record.args) == 5
        ):
            # The access formatter unpacks exactly this five-tuple and
            # ignores record.msg, so flattening would drop the line. Any
            # other shape on that logger is an ordinary record and is
            # formatted like one; keeping the raw msg there would leak a
            # credential to a plain ``%(message)s`` handler.
            new_args = redacted_args
        else:
            # The credential only appears after interpolation. The formatted
            # text is final; drop the raw arguments so a formatter cannot
            # interpolate them again.
            new_msg = redacted_message
            new_args = ()

    exc_text = record.exc_text
    if exc_text:
        redacted_exc = redact_log_text(exc_text)
        if redacted_exc != exc_text:
            exc_text = redacted_exc
    elif record.exc_info:
        formatted_exc = _EXCEPTION_FORMATTER.formatException(record.exc_info)
        redacted_exc = redact_log_text(formatted_exc)
        if redacted_exc != formatted_exc:
            # Only pre-format when redaction changed something. Otherwise a
            # handler's own formatException (custom or structured) must still
            # run, exactly as it did before this factory existed.
            exc_text = redacted_exc

    stack_info = record.stack_info
    if stack_info:
        redacted_stack = redact_log_text(stack_info)
        if redacted_stack != stack_info:
            stack_info = redacted_stack

    record.msg = new_msg
    record.args = new_args
    record.exc_text = exc_text
    record.stack_info = stack_info
    setattr(record, _RECORD_MARKER, True)


def _redact_record_pattern_only(record: logging.LogRecord) -> None:
    """Mask a record with the pattern-only fallback when the full pass failed.

    The message is written flat: the fallback does not know which argument a
    masked value came from, so keeping ``%`` arguments could let a formatter
    interpolate the raw secret again. Exception and stack text go through the
    same pass; as in the full pass, a formatted exception is only stored when
    redaction changed it, so a custom handler formatter still runs otherwise.
    """
    message = record.getMessage()
    redacted_message = redact_log_text_pattern_only(message)
    changed = redacted_message != message

    exc_text = record.exc_text
    if exc_text:
        exc_text = redact_log_text_pattern_only(exc_text)
    elif record.exc_info:
        formatted_exc = _EXCEPTION_FORMATTER.formatException(record.exc_info)
        redacted_exc = redact_log_text_pattern_only(formatted_exc)
        if redacted_exc != formatted_exc:
            exc_text = redacted_exc

    stack_info = record.stack_info
    if stack_info:
        stack_info = redact_log_text_pattern_only(stack_info)

    if changed:
        record.msg = redacted_message
        record.args = ()
    record.exc_text = exc_text
    record.stack_info = stack_info
    setattr(record, _RECORD_MARKER, True)


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
        try:
            _redact_record(record)
        except Exception:
            # Redaction must never change the semantics of a logging call.
            # Return the record exactly as the previous factory produced it;
            # logging's own error path (handler ``handleError``) then handles
            # a bad format string, a broken ``__str__`` or a failing
            # redactor just as it would without this factory.
            return record
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
        # A transport child logger (``httpcore.http11`` and friends) can be
        # created lazily after the first call. Re-scan on a repeated setup so
        # a caller that imports the app after configuring logging still gets
        # the below-WARNING filter onto the children that now exist.
        _install_transport_log_filters()
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
