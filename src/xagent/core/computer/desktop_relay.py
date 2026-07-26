"""Dedicated relay registry for user-authorized desktop targets.

The transport primitives are shared with the browser relay, but credentials,
connections, and task claims live in a separate namespace so a user may keep a
browser extension and a desktop companion connected at the same time.
"""

from __future__ import annotations

from ...config import get_browser_relay_backend, get_redis_url
from .relay import BrowserRelayRegistry, BrowserRelayRegistryProtocol

DESKTOP_RELAY_PROTOCOL_VERSION = 1

DesktopRelayRegistryProtocol = BrowserRelayRegistryProtocol

_desktop_relay_registry: DesktopRelayRegistryProtocol | None = None


def get_desktop_relay_registry() -> DesktopRelayRegistryProtocol:
    global _desktop_relay_registry
    if _desktop_relay_registry is None:
        backend = get_browser_relay_backend()
        redis_url = get_redis_url()
        if backend == "redis" or (backend == "auto" and redis_url):
            if not redis_url:
                raise RuntimeError("Redis desktop relay requires XAGENT_REDIS_URL.")
            from .redis_relay import RedisBrowserRelayRegistry

            _desktop_relay_registry = RedisBrowserRelayRegistry(
                redis_url,
                namespace="xagent:desktop-relay",
                target_kind="desktop",
            )
        else:
            _desktop_relay_registry = BrowserRelayRegistry(target_kind="desktop")
    return _desktop_relay_registry


def reset_desktop_relay_registry() -> None:
    """Reset the process singleton for isolated tests."""
    global _desktop_relay_registry
    _desktop_relay_registry = None
