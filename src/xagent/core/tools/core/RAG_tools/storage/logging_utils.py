"""Structured logging utilities for storage operations.

This module provides utilities for structured logging with performance tracking
and audit capabilities for RAG storage operations.
"""

import logging
import time
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Dict, Iterator, Optional

logger = logging.getLogger(__name__)


@contextmanager
def log_operation(operation: str, **extra_context: Any) -> Iterator[None]:
    """Context manager for logging operation with timing and structured output.

    Usage:
        with log_operation("upsert_documents", table="chunks", count=100):
            # ... perform operation ...
            # Will log: operation_started, operation_completed (with duration_ms)
            # On exception: operation_failed (with error details)

    Args:
        operation: Name of the operation being performed
        **extra_context: Additional context to include in all log entries

    Yields:
        None
    """
    start_time = time.time()
    try:
        logger.info(
            "operation_started", extra={"operation": operation, **extra_context}
        )
        yield
    except Exception as e:
        logger.error(
            "operation_failed",
            extra={
                "operation": operation,
                "error": str(e),
                "error_type": type(e).__name__,
                **extra_context,
            },
            exc_info=True,
        )
        raise
    finally:
        duration_ms = (time.time() - start_time) * 1000
        logger.info(
            "operation_completed",
            extra={
                "operation": operation,
                "duration_ms": round(duration_ms, 2),
                **extra_context,
            },
        )


def log_async_operation(operation: str, **extra_context: Any) -> Callable:
    """Decorator for async operations with automatic timing and structured logging.

    Usage:
        @log_async_operation("search_vectors", table="embeddings_test")
        async def search_vectors_async(self, ...):
            # ... async operation ...

    Args:
        operation: Name of the operation being performed
        **extra_context: Additional context to include in all log entries

    Returns:
        Decorator function
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.time()
            # Extract context from args/kwargs if possible
            context = dict(extra_context)

            # Try to extract self and method name for better logging
            if args and hasattr(args[0], "__class__"):
                context["class"] = args[0].__class__.__name__

            try:
                logger.info(
                    "operation_started", extra={"operation": operation, **context}
                )
                result = await func(*args, **kwargs)

                duration_ms = (time.time() - start_time) * 1000
                logger.info(
                    "operation_completed",
                    extra={
                        "operation": operation,
                        "duration_ms": round(duration_ms, 2),
                        **context,
                    },
                )
                return result
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                logger.error(
                    "operation_failed",
                    extra={
                        "operation": operation,
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "duration_ms": round(duration_ms, 2),
                        **context,
                    },
                    exc_info=True,
                )
                raise

        return wrapper

    return decorator


def log_audit(operation: str, **context: Any) -> None:
    """Log an audit event for security and compliance tracking.

    Args:
        operation: The operation being performed (e.g., "data_access", "permission_check")
        **context: Audit context (user_id, collection, doc_id, etc.)
    """
    logger.info("audit", extra={"operation": operation, **context})


def log_performance(
    metric_name: str, value: Optional[float] = None, unit: str = "ms", **context: Any
) -> None:
    """Log a performance metric.

    Args:
        metric_name: Name of the metric (e.g., "query_duration", "batch_size")
        value: Numeric value of the metric (optional for metrics that only need context)
        unit: Unit of measurement (default: "ms")
        **context: Additional context
    """
    extra: Dict[str, Any] = {"metric": metric_name, **context}
    if value is not None:
        extra["value"] = value
        extra["unit"] = unit
    logger.debug("performance", extra=extra)
