"""Process-wide hook for injecting the SSH target provider (closed-source
xagent-cloud installs a DB-backed provider; open-source/self-hosted can install
a local one)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from xagent.core.ssh import SshAuditSink, SshTargetProvider

# factory(session_factory) -> SshTargetProvider. The provider is long-lived and
# opens its own one-shot session per call from the factory the tool passes in;
# it must NOT be handed a single live session (unsafe under tool concurrency).
SshTargetProviderFactory = Callable[[Any], SshTargetProvider]
# factory(session_factory) -> SshAuditSink. Same session discipline as above:
# the sink opens its own one-shot session per audit event.
SshAuditSinkFactory = Callable[[Any], SshAuditSink]

_ssh_target_provider_factory: SshTargetProviderFactory | None = None
_ssh_audit_sink_factory: SshAuditSinkFactory | None = None


def set_ssh_target_provider_hook(factory: SshTargetProviderFactory | None) -> None:
    """Register (or clear) the SSH target provider factory."""
    global _ssh_target_provider_factory
    _ssh_target_provider_factory = factory


def get_ssh_target_provider(session_factory: Any) -> SshTargetProvider | None:
    """Build the provider for this call, or None if no hook is installed."""
    if _ssh_target_provider_factory is None:
        return None
    provider = _ssh_target_provider_factory(session_factory)
    # Structural guard at the injection seam: the factory is first-party but
    # installed out-of-tree (xagent-cloud), so a mis-wired factory fails loudly
    # here rather than as an AttributeError deep inside a tool call. The
    # Protocol is @runtime_checkable, so this is a cheap method-presence check.
    if not isinstance(provider, SshTargetProvider):
        raise TypeError(
            "SSH target provider factory returned an object that does not "
            "implement SshTargetProvider"
        )
    return provider


def set_ssh_audit_sink_hook(factory: SshAuditSinkFactory | None) -> None:
    """Register (or clear) the SSH audit sink factory."""
    global _ssh_audit_sink_factory
    _ssh_audit_sink_factory = factory


def get_ssh_audit_sink(session_factory: Any) -> SshAuditSink | None:
    """Build the audit sink for this call, or None if no hook is installed."""
    if _ssh_audit_sink_factory is None:
        return None
    sink = _ssh_audit_sink_factory(session_factory)
    # Same structural guard as the provider seam (see above).
    if not isinstance(sink, SshAuditSink):
        raise TypeError(
            "SSH audit sink factory returned an object that does not "
            "implement SshAuditSink"
        )
    return sink
