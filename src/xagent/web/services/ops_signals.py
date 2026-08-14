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
# Set at startup when a Pub/Sub project is configured but
# XAGENT_GMAIL_WATCH_ENABLED is not: Gmail watch registration and renewal
# are disabled, so Gmail triggers report failed provisioning and existing
# watches expire unrenewed.
GMAIL_WATCH_REGISTRATION_DISABLED = "gmail_watch_registration_disabled"
CHECKPOINT_DECODE_FALLBACK = "checkpoint_decode_fallback"
CHECKPOINT_LEGACY_POINTER_AMBIGUOUS = "checkpoint_legacy_pointer_ambiguous"
CHECKPOINT_LOAD_UNAVAILABLE = "checkpoint_load_unavailable"
CHECKPOINT_PK_ANCHOR_DANGLING = "checkpoint_pk_anchor_dangling"
CHECKPOINT_PRUNE_FAILED = "checkpoint_prune_failed"
# Set when the interaction rollout gate (or /ready, in native mode) finds
# task_interaction_requests missing; cleared at either site once the table
# is observed to exist.
INTERACTION_ROLLOUT_SCHEMA_ABSENT = "interaction_rollout_schema_absent"
# Set when the interaction rollout gate sees a Task.source value that does
# not normalize to a known origin. Deliberately not paired with a clear
# site: an unrecognized source is a property of persisted data, and no
# in-process signal can observe that data being fixed. Auto-clearing would
# assert "the data got fixed" without evidence for it. See
# INTERACTION_ANCHOR_CORRUPT below for the same reasoning applied to a
# different persisted-data corruption.
INTERACTION_ROLLOUT_UNKNOWN_TASK_SOURCE = "interaction_rollout_unknown_task_source"
# interaction_handoff (task_interaction_staging.py) degrades instead of
# losing a caller's turn on five expected failures, sharing this signal.
INTERACTION_HANDOFF_DEGRADED = "interaction_handoff_degraded"
# A sixth, separately addressable signal for InteractionRunPartitionMismatch
# specifically: degrading rather than propagating that one exception is an
# override with no production reachability data behind it (see
# interaction_handoff's docstring), so it stays distinguishable on /health
# from the other five.
INTERACTION_RUN_PARTITION_MISMATCH_DEGRADED = (
    "interaction_run_partition_mismatch_degraded"
)
# Set by resolve_interaction_anchor (task_interaction_anchor.py) when the
# trace_events row its pointer names fails the ownership/type/partition/
# identity checks that make it a valid anchor. Deliberately not paired with
# a clear site, the same reasoning as INTERACTION_ROLLOUT_UNKNOWN_TASK_SOURCE
# above: a corrupt anchor is a property of a persisted trace_events row, and
# no in-process registry can observe that row being fixed. Auto-clearing
# would assert "the data got fixed" without evidence for it.
INTERACTION_ANCHOR_CORRUPT = "interaction_anchor_corrupt"
# Set when the supersede helper's own UPDATE fails; cleared by the next
# supersede that completes normally, for any task. Task-scoped detail,
# process-scoped signal: it reports whether the sweep mechanism is
# working, not whether a particular task has been repaired.
CLARIFICATION_LEGACY_SUPERSEDE_FAILED = "clarification_legacy_supersede_failed"

# Set when a legacy-resume interaction close (task_interaction_close.py)
# matches more than one active row for a single task -- structurally
# impossible under uq_task_interaction_active_slot unless that constraint
# has already been violated. Not paired with a clear_degradation() call,
# like INTERACTION_ROLLOUT_UNKNOWN_TASK_SOURCE above: it reports evidence
# that a uniqueness constraint was already broken, a schema fact this
# module has no way to observe being fixed.
INTERACTION_LEGACY_RESUME_CLOSE_ROWCOUNT_ANOMALY = (
    "interaction_legacy_resume_close_rowcount_anomaly"
)

# task_clarification_draft.py registers this when a waiting run's result
# carries no clarification draft to publish, and clears it the next time a
# draft is successfully resolved to a publishable payload.
CLARIFICATION_DRAFT_MISSING = "clarification_draft_missing"
# task_clarification_draft.py registers this when the same batch produced
# more than one waiting step and a losing step's draft was discarded.
# Deliberately not cleared by the very publish that reports it -- a
# successful publish this round does not undo the fact that a sibling's
# question was dropped this round, so this stays visible until a later,
# conflict-free resolution clears it.
CLARIFICATION_MULTIPLE_DRAFTS = "clarification_multiple_drafts"

# Set by the compatibility seam in _handle_resume_task_unserialized when a
# legacy (non-receipt-carrying) resume command is refused because the task
# still has an active native interaction row anchored to its current run --
# i.e. a resume that should have gone through respond() instead. This
# is a boolean degradation flag, not a counter: this registry has no notion
# of "count," only "name -> detail string" (see the module docstring). The
# measurable half of "how often" lives in a structured log line the seam
# also emits, not here. Removal condition: the log line is silent for one
# full release cycle and every registered consumer of this signal has
# migrated off the legacy resume path.
INTERACTION_LEGACY_RESUME_SHIM = "interaction_legacy_resume_shim"

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
