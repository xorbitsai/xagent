"""Process-wide hook for injecting the SSH target provider (closed-source
xagent-cloud installs a DB-backed provider; open-source/self-hosted can install
a local one)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from xagent.core.ssh import SshAuditSink, SshTargetProvider

_Hook = TypeVar("_Hook")

# factory(session_factory) -> SshTargetProvider. The provider is long-lived and
# opens its own one-shot session per call from the factory the tool passes in;
# it must NOT be handed a single live session (unsafe under tool concurrency).
SshTargetProviderFactory = Callable[[Any], SshTargetProvider]
# factory(session_factory) -> SshAuditSink. Same session discipline as above:
# the sink opens its own one-shot session per audit event.
SshAuditSinkFactory = Callable[[Any], SshAuditSink]

_ssh_target_provider_factory: SshTargetProviderFactory | None = None
_ssh_audit_sink_factory: SshAuditSinkFactory | None = None


def _resolve_hook(
    factory: Callable[[Any], _Hook] | None,
    session_factory: Any,
    protocol: Any,  # a @runtime_checkable Protocol; Any dodges type-abstract
    label: str,
) -> _Hook | None:
    """Build an injected hook for this call, or None if no factory installed.

    Structural guard at the injection seam: the factory is first-party but
    installed out-of-tree (xagent-cloud), so a mis-wired one fails loudly here
    rather than as an AttributeError deep inside a tool call. The Protocols are
    @runtime_checkable, so this is a cheap method-presence check."""
    if factory is None:
        return None
    obj = factory(session_factory)
    if not isinstance(obj, protocol):
        raise TypeError(
            f"SSH {label} factory returned an object that does not "
            f"implement {protocol.__name__}"
        )
    return obj


def set_ssh_target_provider_hook(factory: SshTargetProviderFactory | None) -> None:
    """Register (or clear) the SSH target provider factory."""
    global _ssh_target_provider_factory
    _ssh_target_provider_factory = factory


def get_ssh_target_provider(session_factory: Any) -> SshTargetProvider | None:
    """Build the provider for this call, or None if no hook is installed."""
    return _resolve_hook(
        _ssh_target_provider_factory,
        session_factory,
        SshTargetProvider,
        "target provider",
    )


def set_ssh_audit_sink_hook(factory: SshAuditSinkFactory | None) -> None:
    """Register (or clear) the SSH audit sink factory."""
    global _ssh_audit_sink_factory
    _ssh_audit_sink_factory = factory


def get_ssh_audit_sink(session_factory: Any) -> SshAuditSink | None:
    """Build the audit sink for this call, or None if no hook is installed."""
    return _resolve_hook(
        _ssh_audit_sink_factory, session_factory, SshAuditSink, "audit sink"
    )
