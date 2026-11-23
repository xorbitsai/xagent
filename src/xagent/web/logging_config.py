"""Unified logging configuration for xagent web application."""

import logging
import os
from collections.abc import Callable
from logging.config import dictConfig
from typing import Any, Literal, cast

from ..config import get_log_transport_request_lines
from ..core.utils.security import (
    REDACTED,
    redact_log_text,
    redact_log_text_pattern_only,
)
from .request_log_redaction import install_request_log_redaction

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


_is_applied = False

# Transport loggers whose request lines carry full URLs (query strings
# included). Their level is pinned to WARNING below, but a third-party library
# can call ``setLevel`` on them at any time; a logger-level filter keeps the
# INFO lines out regardless of the level. ``XAGENT_LOG_TRANSPORT_REQUEST_LINES``
# turns that filter off and leaves the record factory as the only guard.
_TRANSPORT_LOGGER_NAMES = ("httpx", "httpcore")
# Marker set on the record factory this module wraps around a previous one, so
# repeated ``setup_logging(force=True)`` calls (and a third party wrapping the
# chain later) cannot stack another redaction layer.
_RECORD_FACTORY_MARKER = "_xagent_redacts_log_records"
# Marker set on a record once this module has redacted it; two layers of the
# same factory would otherwise format and scan the message twice.
_RECORD_MARKER = "_xagent_log_redacted"

# A record whose message, arguments, exception text and stack info contain
# none of these markers cannot hold a shape any redactor recognizes. The
# pre-check is several plain substring scans instead of the ~9 regex and URL
# passes the full redactor runs, and it is what makes an ordinary record
# (``ConsoleTraceHandler`` INFO payloads reach 50 kB) cheap.
# ``://`` is covered by ``:`` and ``xagent-auth-token`` by ``auth-token``, so
# the shorter needles are enough; ``auth-token`` is also much cheaper to scan
# for than the seventeen-character literal.
_REDACTION_HINTS = ("=", ":", "eyJ", "?", "auth-token")

_EXCEPTION_FORMATTER = logging.Formatter()


class _BelowWarningFilter(logging.Filter):
    """Drop records below WARNING regardless of the logger's level."""

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno >= logging.WARNING


_TRANSPORT_FILTER = _BelowWarningFilter()


def _format_with_args(msg: object, args: object) -> str:
    text = str(msg)
    if args:
        return text % args
    return text


def _redact_argument(value: object) -> object:
    if isinstance(value, str):
        return redact_log_text(value)
    return value


def _redact_argument_pattern_only(value: object) -> object:
    if isinstance(value, str):
        return redact_log_text_pattern_only(value)
    return value


def _redact_args(args: object) -> Any:
    """Redact string elements of a tuple/dict of log arguments in place."""
    if isinstance(args, tuple):
        return tuple(_redact_argument(item) for item in args)
    if isinstance(args, dict):
        return {key: _redact_argument(value) for key, value in args.items()}
    return args


def _text_may_hold_a_credential(text: str) -> bool:
    """Cheap check for a character sequence a redaction rule needs.

    Every shape the redactors recognize contains at least one of these
    markers, so a text with none of them cannot be redacted into a different
    string. The scan is a handful of plain substring searches instead of the
    full pass's regex and URL work.
    """
    return any(hint in text for hint in _REDACTION_HINTS)


def _record_may_hold_a_credential(record: logging.LogRecord) -> bool:
    """Whether any text the full pass reads could hold a recognized shape."""
    if _text_may_hold_a_credential(str(record.msg)):
        return True
    args = record.args
    if isinstance(args, dict):
        arg_values: Any = args.values()
    elif isinstance(args, tuple):
        arg_values = args
    else:
        arg_values = (args,)
    if any(_text_may_hold_a_credential(str(value)) for value in arg_values):
        return True
    if record.exc_info:
        # The traceback text only exists after formatting, and an upstream
        # error body can echo a header line into it, so this record always
        # takes the full pass.
        return True
    if record.exc_text and _text_may_hold_a_credential(record.exc_text):
        return True
    if record.stack_info:
        return _text_may_hold_a_credential(record.stack_info)
    return False


def _transport_record_is_dropped(record: logging.LogRecord) -> bool:
    """Whether a logger filter drops this record before any handler sees it.

    Only the record's own logger filter runs before its handlers, so this is
    exact: when the below-WARNING transport filter is attached to that logger,
    the record never reaches a handler and needs no redaction. A transport
    child logger created after ``setup_logging`` has no filter, so the record
    factory still redacts its request line.
    """
    if record.levelno >= logging.WARNING:
        return False
    logger_object = logging.Logger.manager.loggerDict.get(record.name)
    return (
        isinstance(logger_object, logging.Logger)
        and _TRANSPORT_FILTER in logger_object.filters
    )


def _redact_record(record: logging.LogRecord) -> None:
    """Redact one record, falling back to a pattern-only pass on failure.

    Two cheap pre-checks run first: a transport request line that the
    below-WARNING filter will drop anyway is not redacted, and a record whose
    text holds none of the redaction hints skips the full pass. The full pass
    commits every change only at the end. If it raises anyway, a
    header/assignment/JWT-only pass still masks the common shapes instead of
    leaving the whole record unredacted. If that pass fails too, the exception
    reaches the factory, which returns the record exactly as the previous
    factory produced it.
    """
    if getattr(record, _RECORD_MARKER, False):
        return
    if _transport_record_is_dropped(record):
        return
    if not _record_may_hold_a_credential(record):
        # Nothing a redactor recognizes. Mark it so a chained factory does not
        # re-run the pre-check either.
        setattr(record, _RECORD_MARKER, True)
        return
    try:
        _redact_record_fully(record)
    except Exception:
        _redact_record_pattern_only(record)


def _redact_text_fields(
    record: logging.LogRecord, redact: Callable[[str], str]
) -> None:
    """Redact the exception and stack text of ``record`` in place.

    A formatted exception is stored only when redaction changed it. Otherwise
    a handler's own ``formatException`` (custom or structured) must still run,
    exactly as it did before this factory existed.
    """
    exc_text = record.exc_text
    if exc_text:
        redacted_exc = redact(exc_text)
        if redacted_exc != exc_text:
            record.exc_text = redacted_exc
    elif record.exc_info:
        formatted_exc = _EXCEPTION_FORMATTER.formatException(record.exc_info)
        redacted_exc = redact(formatted_exc)
        if redacted_exc != formatted_exc:
            record.exc_text = redacted_exc
    stack_info = record.stack_info
    if stack_info:
        redacted_stack = redact(stack_info)
        if redacted_stack != stack_info:
            record.stack_info = redacted_stack


def _redact_record_fully(record: logging.LogRecord) -> None:
    """Redact one record with the full structured redactor."""
    message = record.getMessage()
    redacted_message = redact_log_text(message)
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
            record.args = redacted_args
        elif (
            record.name == "uvicorn.access"
            and isinstance(record.args, tuple)
            and len(record.args) == 5
        ):
            # uvicorn's access formatter unpacks exactly this five-tuple and
            # ignores record.msg, so flattening would drop the line. Here the
            # whole-message pass masked at least one value the per-argument
            # pass cannot see (a header value split across two arguments), so
            # keep the shape but write the mask into every string field
            # instead of keeping a partly masked value. Any other shape on
            # that logger is an ordinary record and is formatted like one;
            # keeping the raw msg there would leak a credential to a plain
            # ``%(message)s`` handler.
            record.msg = redact_log_text(str(record.msg))
            record.args = tuple(
                REDACTED if isinstance(item, str) else item for item in record.args
            )
        else:
            # The credential only appears after interpolation. The formatted
            # text is final; drop the raw arguments so a formatter cannot
            # interpolate them again.
            record.msg = redacted_message
            record.args = ()
    _redact_text_fields(record, redact_log_text)
    setattr(record, _RECORD_MARKER, True)


def _redact_record_pattern_only(record: logging.LogRecord) -> None:
    """Mask a record with the pattern-only fallback when the full pass failed.

    The message is written flat: the fallback does not know which argument a
    masked value came from, so keeping ``%`` arguments could let a formatter
    interpolate the raw secret again. The access logger's five-tuple is the
    exception -- its formatter unpacks the arguments and ignores the message,
    so those elements are masked in place and the shape is kept.
    """
    message = record.getMessage()
    redacted_message = redact_log_text_pattern_only(message)
    if redacted_message != message:
        if (
            record.name == "uvicorn.access"
            and isinstance(record.args, tuple)
            and len(record.args) == 5
        ):
            # uvicorn's access formatter unpacks the five-tuple and ignores
            # record.msg. Flattening it here would drop an access line that
            # the full pass could not redact, so mask the tuple elements and
            # keep the shape the formatter needs. The format string is masked
            # too, for a plain formatter that reads it.
            record.msg = _redact_argument_pattern_only(record.msg)
            record.args = tuple(
                _redact_argument_pattern_only(item) for item in record.args
            )
        else:
            record.msg = redacted_message
            record.args = ()
    _redact_text_fields(record, redact_log_text_pattern_only)
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

    ``XAGENT_LOG_TRANSPORT_REQUEST_LINES`` turns the filter off, including on
    a repeated setup after the environment changed; the record factory still
    redacts the request line. A logger that already carries the filter loses
    it in that case so the toggle works without a process restart.
    """
    keep_below_warning = get_log_transport_request_lines()
    names = set(_TRANSPORT_LOGGER_NAMES)
    names.update(
        name
        for name, logger in list(logging.Logger.manager.loggerDict.items())
        if isinstance(logger, logging.Logger)
        and name.startswith(("httpx.", "httpcore."))
    )
    for name in sorted(names):
        logger = logging.getLogger(name)
        if keep_below_warning:
            if _TRANSPORT_FILTER in logger.filters:
                logger.removeFilter(_TRANSPORT_FILTER)
        elif _TRANSPORT_FILTER not in logger.filters:
            logger.addFilter(_TRANSPORT_FILTER)


def install_log_redaction() -> None:
    """Install record redaction and the transport filter, and nothing else.

    A record factory is process-wide and a logger filter travels with its
    logger, so neither is undone by another ``dictConfig`` or by a library
    that replaces the root handlers. That is what a process that configures
    its own logging needs: Celery builds the worker and beat log configuration
    itself (its ``worker_hijack_root_logger`` setting replaces the root
    handlers) and never calls ``setup_logging``, so those processes call this
    from Celery's ``worker_init``, ``after_setup_logger``,
    ``worker_process_init`` and ``beat_init`` signals. The order does not
    matter: reconfiguring logging cannot drop a record factory or a logger
    filter.

    Handlers, levels and the logging configuration are left untouched, so this
    is safe to call at any point after a process starts. Idempotent.
    """
    _install_record_redaction()
    _install_transport_log_filters()


def setup_logging(level: LogLevel | None = None, force: bool = False) -> None:
    """Configure logging for the entire application.

    Every record created through the standard ``logging`` machinery is
    redacted at creation by a process-wide record factory, so loggers that do
    not propagate, handlers a library attaches later and ``basicConfig``
    replacements are covered too. The factory covers the message, the ``%``
    arguments, exception text and stack info. It does not cover:

    * ``extra=`` fields: they are attached to the record after the factory
      returns and are never inspected;
    * a handler that renders ``exc_info`` itself (a structured traceback
      renderer, for instance): it sees the original exception objects, not
      the pre-formatted ``exc_text``;
    * an MCP stdio server child process: it configures its own logging with
      ``basicConfig`` and its lines never pass through this configuration.

    ``httpx``/``httpcore`` request lines below WARNING are dropped at the
    logger so a third-party SDK that lowers either level cannot put a full
    request URL into INFO. Set ``XAGENT_LOG_TRANSPORT_REQUEST_LINES`` to true
    to keep those lines; the record factory still masks credential values in
    them.

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

    # Redact every record at creation (this also covers loggers that do not
    # propagate and handlers a library attaches after this call) and keep
    # httpx/httpcore request lines (full URL, query included) out of INFO even
    # when a library lowers those loggers' levels.
    install_log_redaction()
    # Server access and WebSocket handshake lines carry the raw request
    # target; mask credential-like query values before they reach a handler.
    install_request_log_redaction()

    if invalid_level:
        logging.warning(
            "Invalid log level '%r', falling back to 'INFO'", original_level
        )

    _is_applied = True
