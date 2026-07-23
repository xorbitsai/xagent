"""Process-wide hook for injecting the SSH target provider (closed-source SaaS
installs a DB-backed provider; open-source/self-hosted can install a local one)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from xagent.core.ssh import SshTargetProvider

# factory(session_factory) -> SshTargetProvider. The provider is long-lived and
# opens its own one-shot session per call from the factory the tool passes in;
# it must NOT be handed a single live session (unsafe under tool concurrency).
SshTargetProviderFactory = Callable[[Any], SshTargetProvider]

_ssh_target_provider_factory: SshTargetProviderFactory | None = None


def set_ssh_target_provider_hook(factory: SshTargetProviderFactory | None) -> None:
    """Register (or clear) the SSH target provider factory."""
    global _ssh_target_provider_factory
    _ssh_target_provider_factory = factory


def get_ssh_target_provider(session_factory: Any) -> SshTargetProvider | None:
    """Build the provider for this call, or None if no hook is installed."""
    if _ssh_target_provider_factory is None:
        return None
    return _ssh_target_provider_factory(session_factory)
