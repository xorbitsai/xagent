"""In-process registry of degraded-mode operational signals.

Xagent has no metrics stack, so security-relevant degradations (e.g. Gmail
OIDC verification running without service-account email checks) would
otherwise only exist as log lines. This registry gives them a
machine-readable surface — the ``/health`` endpoint reports active
degradations — that uptime monitors and dashboards can alert on.

``/health`` is unauthenticated, so it exposes only the signal *names*;
the detail strings describe security-relevant misconfiguration and are
reserved for logs and authenticated diagnostics.

Signals are per-process and idempotent: registering the same name twice
updates its detail, and clearing a name that is not active is a no-op.
"""

from __future__ import annotations

import threading

GMAIL_OIDC_SERVICE_ACCOUNT_UNVERIFIED = "gmail_oidc_service_account_unverified"
CHECKPOINT_DECODE_FALLBACK = "checkpoint_decode_fallback"

_signals: dict[str, str] = {}
_lock = threading.Lock()


def register_degradation(name: str, detail: str) -> None:
    """Mark a named degradation as active with a human-readable detail."""
    with _lock:
        _signals[name] = detail


def clear_degradation(name: str) -> None:
    """Mark a named degradation as resolved."""
    with _lock:
        _signals.pop(name, None)


def active_degradations() -> dict[str, str]:
    """Snapshot of currently active degradations, keyed by signal name."""
    with _lock:
        return dict(_signals)
