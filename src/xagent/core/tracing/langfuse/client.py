"""Langfuse client lifecycle and configuration."""

from __future__ import annotations

import logging
import os
import threading
from typing import Optional

from langfuse import Langfuse

logger = logging.getLogger(__name__)

_CLIENT_LOCK = threading.Lock()
_LANGFUSE_CLIENT: Optional[Langfuse] = None
_LANGFUSE_INIT_ATTEMPTED = False
_PLACEHOLDER_VALUES = {
    "",
    "public key",
    "secret key",
    "your-public-key",
    "your-secret-key",
}


def _is_langfuse_env_configured() -> bool:
    if os.getenv("LANGFUSE_TRACING_ENABLED", "true").lower() == "false":
        return False

    public_key = (os.getenv("LANGFUSE_PUBLIC_KEY") or "").strip()
    secret_key = (os.getenv("LANGFUSE_SECRET_KEY") or "").strip()

    if public_key.lower() in _PLACEHOLDER_VALUES:
        return False
    if secret_key.lower() in _PLACEHOLDER_VALUES:
        return False

    return bool(public_key and secret_key)


def get_langfuse_client() -> Optional[Langfuse]:
    """Return a shared Langfuse client when tracing is configured."""
    global _LANGFUSE_CLIENT, _LANGFUSE_INIT_ATTEMPTED

    with _CLIENT_LOCK:
        if _LANGFUSE_INIT_ATTEMPTED:
            return _LANGFUSE_CLIENT

        _LANGFUSE_INIT_ATTEMPTED = True

        if not _is_langfuse_env_configured():
            logger.info("Langfuse tracing is disabled or not configured")
            return None

        try:
            base_url = (
                (os.getenv("LANGFUSE_BASE_URL") or "").strip()
                or (os.getenv("LANGFUSE_HOST") or "").strip()
                or None
            )
            if base_url:
                _LANGFUSE_CLIENT = Langfuse(base_url=base_url)
            else:
                _LANGFUSE_CLIENT = Langfuse()
            logger.info("Langfuse tracing initialized")
        except Exception as exc:
            logger.warning(f"Failed to initialize Langfuse client: {exc}")
            _LANGFUSE_CLIENT = None

        return _LANGFUSE_CLIENT


def initialize_langfuse() -> bool:
    """Initialize Langfuse eagerly during app startup."""
    return get_langfuse_client() is not None


def flush_langfuse() -> None:
    """Flush pending Langfuse spans before shutdown."""
    client = _LANGFUSE_CLIENT
    if client is None:
        return

    try:
        client.flush()
    except Exception as exc:
        logger.warning(f"Failed to flush Langfuse traces: {exc}")


def reset_langfuse_client() -> None:
    """Reset the shared client. Intended for tests."""
    global _LANGFUSE_CLIENT, _LANGFUSE_INIT_ATTEMPTED
    with _CLIENT_LOCK:
        _LANGFUSE_CLIENT = None
        _LANGFUSE_INIT_ATTEMPTED = False
