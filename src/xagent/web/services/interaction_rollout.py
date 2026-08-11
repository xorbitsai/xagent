"""Owner of the default-off interaction rollout policy and its publication gate.

This module answers exactly one question for the rest of the codebase: "is
a task allowed to publish a native interaction row right now." It owns the
env-derived policy (parsed once, frozen, fail-fast at startup), the
publication gate that answers the question, an in-process counter registry
for the gate's outcomes, and a one-way readiness latch consumed by
``/ready``. It does not read interaction rows, answer them, close them, or
decide what a reader sees -- those stay untouched by every mode this
module can be configured into (see the module-level invariant below).

Gating vocabulary vs. origin vocabulary
    ``INTERACTION_GATING_SOURCES`` (seven entries) is a proper superset of
    ``INTERACTION_ORIGIN_VOCABULARY`` (six entries, defined on
    ``task_interaction.py`` next to the ``origin`` column it constrains).
    The extra entry, ``"channel"``, is a synthetic split of the
    ``"internal"`` bucket by ``task.channel_id`` -- it exists only to pick a
    rollout batch and must never be written into the ``origin`` column or
    added to that column's CHECK constraint. Any change that merges the two
    vocabularies, or adds ``"channel"`` to the origin CHECK, needs to justify
    why a temporary rollout-axis split is worth a database migration.

Retired sources stay in the vocabulary
    A gating source that has stopped producing new tasks is "legitimate but
    dead," not invalid. Retiring a ``Task.source`` value means "no new tasks
    of this kind," not "no task of this kind will ever reach a finalizer
    again" -- historical rows keep flowing through the same finalizers.
    Seeing a gating entry with zero current traffic is not a reason to
    remove it from an allow list.

Suppression markers never ride the read-side key
    (Applies once a write-side suppression mechanism exists; there is none
    in this PR, but the constraint is pinned here because it binds whatever
    key this gate produces.) A suppression marker must never be written into
    any column the read-side decision key already consumes. The read-side
    key answers "what has been published;" a suppression marker answers
    "what the write side intends to do next." Letting the marker share the
    read side's column would let a write-side intention masquerade as a
    published fact.

Hard invariant enforced by this module and by the static guards that check
it: the interaction rollout mode can only decide whether a *new* native row
gets produced. It can never decide how an already-published row is read,
answered, or closed.
"""

from __future__ import annotations

import enum
import logging
import os
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

from ...config import (
    INTERACTION_NATIVE_SOURCES,
    INTERACTION_PROTOCOL_MODE,
    get_interaction_native_sources,
    get_interaction_protocol_mode,
)
from ..models.task_interaction import (
    INTERACTION_ORIGIN_VOCABULARY,
    INTERACTION_PROTOCOL_VERSION,
    normalize_interaction_origin,
)
from .ops_signals import (
    INTERACTION_ROLLOUT_SCHEMA_ABSENT,
    INTERACTION_ROLLOUT_UNKNOWN_TASK_SOURCE,
    clear_degradation,
    register_degradation,
)
from .task_interaction_schema import interaction_requests_table_exists

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from ..models.task import Task

logger = logging.getLogger(__name__)

_VALID_MODES = ("legacy", "read", "native")


class InteractionRolloutConfigError(ValueError):
    """Raised at startup when the interaction rollout configuration is invalid."""


# ---------------------------------------------------------------------------
# Gating vocabulary and the synthetic "channel" split.
# ---------------------------------------------------------------------------

# The right-hand operand here must stay INTERACTION_ORIGIN_VOCABULARY, not
# TASK_SOURCE_LITERALS: the latter is missing "internal" today (nothing in
# the codebase constructs a Task with source="internal" as a literal -- it
# only ever arrives via the column default or normalize_interaction_origin's
# falsy fallback), and deriving the gating vocabulary from it would silently
# drop the single most important gating source.
INTERACTION_GATING_SOURCES = INTERACTION_ORIGIN_VOCABULARY | {"channel"}


def gating_key(task: "Task") -> str | None:
    """Map a task to the gating vocabulary entry that governs its publication.

    Returns ``None`` for a task whose ``Task.source`` does not normalize to
    a known origin -- callers must treat that as "unknown," not fall back to
    any bucket.

    The ``origin == "internal"`` guard below is required, not incidental:
    Web UI, WebSocket, and IM-channel tasks all land in the ``"internal"``
    origin bucket, but a ``widget`` task can also carry a non-null
    ``channel_id`` (public chat access sets one). Without the guard, that
    widget task's synthetic key would collapse to ``"channel"`` even though
    its actual origin is ``"widget"``, silently pulling widget traffic into
    whatever rollout batch was meant only for IM channels.

    The ``"channel"`` key is persisted but not stable over a task's
    lifetime: ``Task.channel_id`` has ``ondelete=SET NULL``, so deleting
    the channel a task belongs to nulls that column on the task row that
    survives the deletion. A task gated as ``"channel"`` today can be
    re-read later, after its channel is deleted, and gate as
    ``"internal"`` instead -- the same task, two different answers,
    neither of them wrong for the state of the data at the time each was
    computed.
    """
    # Task.source is a legacy Column-typed attribute (not Mapped[str | None]),
    # so mypy sees it as opaque; cast rather than loosen
    # normalize_interaction_origin's own signature, which is a documented
    # public contract (see task_interaction.py) independent of this cast.
    origin = normalize_interaction_origin(cast("str | None", task.source))
    if origin is None:
        return None
    if origin == "internal" and task.channel_id is not None:
        return "channel"
    return origin


# ---------------------------------------------------------------------------
# In-process counter registry.
#
# The full counter namespace for the mixed-version rollout is pinned here as
# constants -- even the families this PR does not increment -- so that later
# producers (materialization, response handling, repair, command staging)
# reuse this one registry instead of standing up a second counter parser.
# This mirrors ops_signals's own "one owner, many producers" shape. Only the
# rollout.decision.* family is incremented in this PR; every other constant
# below is a placeholder naming its future owner. The nine placeholder
# constants below are intentionally dead code until the producer named in
# each one's comment lands and starts incrementing it -- not an oversight
# to flag.
# ---------------------------------------------------------------------------
COUNTER_ROLLOUT_DECISION_ALLOWED = "rollout.decision.allowed"
COUNTER_ROLLOUT_DECISION_BLOCKED_MODE = "rollout.decision.blocked_mode"
COUNTER_ROLLOUT_DECISION_BLOCKED_SOURCE = "rollout.decision.blocked_source"
COUNTER_ROLLOUT_DECISION_BLOCKED_UNKNOWN_SOURCE = (
    "rollout.decision.blocked_unknown_source"
)
COUNTER_ROLLOUT_DECISION_BLOCKED_SCHEMA_ABSENT = (
    "rollout.decision.blocked_schema_absent"
)

# Not incremented by this PR -- placeholders for later owners sharing this
# registry.
COUNTER_COMPAT_READ_FALLBACK = "compat.read_fallback"  # owner: read-path wiring
COUNTER_LIFECYCLE_RESPONSE_CONFLICT = "lifecycle.response_conflict"  # owner: #1075
COUNTER_MATERIALIZE_SUCCESS = "materialize.success"  # owner: #1078
COUNTER_MATERIALIZE_TRANSIENT = "materialize.transient"  # owner: #1078
COUNTER_MATERIALIZE_PERMANENT = "materialize.permanent"  # owner: #1078
COUNTER_MATERIALIZE_AMBIGUOUS = "materialize.ambiguous"  # owner: #1078
COUNTER_REPAIR_COUNT = "repair.count"  # owner: #1082
COUNTER_REPAIR_TERMINAL = "repair.terminal"  # owner: #1082
COUNTER_COMMAND_LAG_SECONDS_MAX = "command.lag_seconds_max"  # owner: #1073 / #1075

_counters: dict[str, int] = {}
_counters_lock = threading.Lock()


def increment_counter(name: str) -> None:
    """Increment one named counter by one. Process-local, like ops_signals."""
    with _counters_lock:
        _counters[name] = _counters.get(name, 0) + 1


def counters_snapshot() -> dict[str, int]:
    """Point-in-time copy of every counter's current value.

    Returns a copy, not the live registry -- mutating the returned dict
    never affects what a later call to this function returns. Counters are
    per-process, the same limitation ``ops_signals`` documents for its own
    registry: a diagnostic reader behind a load balancer only ever sees its
    own process's counts.
    """
    with _counters_lock:
        return dict(_counters)


# ---------------------------------------------------------------------------
# Policy: parsed once at startup, frozen, and fail-fast.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InteractionRolloutPolicy:
    """The frozen, fully-validated interaction rollout configuration."""

    mode: Literal["legacy", "read", "native"]
    native_sources: frozenset[str]
    native_protocol_version: int


_policy: InteractionRolloutPolicy | None = None
_policy_lock = threading.Lock()

# One-way readiness latch for /ready's native-mode schema check. Lives here,
# alongside the policy singleton, rather than on app.state: cache True
# forever (the table is only ever added, never dropped, so a stale True
# cannot occur) and let False re-query every time (the table may not exist
# yet during a legitimate "service rolled before migration ran" window).
# This latch is consumed only by /ready -- the gate's own schema check
# (evaluate_native_publication step 5) always queries fresh.
_native_schema_ready = False


def is_native_schema_ready() -> bool:
    """Current state of the one-way /ready schema latch."""
    return _native_schema_ready


def mark_native_schema_ready() -> None:
    """Flip the one-way /ready schema latch. Idempotent; never unflips."""
    global _native_schema_ready
    _native_schema_ready = True


def validate_interaction_rollout_at_startup() -> InteractionRolloutPolicy:
    """Parse, validate, and freeze the interaction rollout policy.

    Idempotent: the first call parses env and freezes the result; every
    later call returns that same object (``is``-identical), without
    re-reading env. Raises :class:`InteractionRolloutConfigError` (a
    ``ValueError`` subclass) for anything invalid enough to make the
    process refuse to start -- deliberately, a departure from this
    codebase's more common three-state warn-and-fall-back pattern (see
    ``get_agent_runtime`` in ``config.py``). A rollout switch that silently
    falls back to legacy on a typo would defeat the entire purpose of this
    module: it exists so a misconfigured deployment fails loudly at
    startup instead of quietly staying in legacy while an operator
    believes they enabled native mode.
    """
    global _policy
    with _policy_lock:
        if _policy is not None:
            return _policy

        raw_mode = os.getenv(INTERACTION_PROTOCOL_MODE)
        mode = get_interaction_protocol_mode()
        if mode not in _VALID_MODES:
            raise InteractionRolloutConfigError(
                f"Invalid {INTERACTION_PROTOCOL_MODE} value: {raw_mode!r}. "
                f"Expected one of: {', '.join(_VALID_MODES)}."
            )

        raw_sources_env = os.getenv(INTERACTION_NATIVE_SOURCES)
        tokens = get_interaction_native_sources()
        # Same split and blank-filter as get_interaction_native_sources(),
        # but without its case/whitespace normalization, so the duplicate
        # check below can quote what the operator actually typed. Without
        # this, "SDK,sdk" would report the collision as "'sdk' is listed
        # more than once" with no indication that the two entries were
        # spelled differently in the first place.
        raw_tokens = [t for t in (raw_sources_env or "").split(",") if t.strip()]

        seen: dict[str, str] = {}
        for token, raw_token in zip(tokens, raw_tokens):
            if token in seen:
                raise InteractionRolloutConfigError(
                    f"Invalid {INTERACTION_NATIVE_SOURCES}: entry {raw_token!r} "
                    f"is listed more than once (already present as "
                    f"{seen[token]!r}; both normalize to {token!r})."
                )
            seen[token] = raw_token
            if token not in INTERACTION_GATING_SOURCES:
                raise InteractionRolloutConfigError(
                    f"Invalid {INTERACTION_NATIVE_SOURCES} entry {token!r}: "
                    "not a member of the gating source vocabulary "
                    f"{sorted(INTERACTION_GATING_SOURCES)}. A retired "
                    "Task.source value is a legitimate but dead entry -- it "
                    "stays in the gating vocabulary even after it stops "
                    "producing new rows, so if this entry used to be valid "
                    "check whether it was removed from the vocabulary in "
                    "error before assuming this configuration is wrong."
                )

        if mode == "native" and not tokens:
            raise InteractionRolloutConfigError(
                f"{INTERACTION_PROTOCOL_MODE}=native requires at least one "
                f"entry in {INTERACTION_NATIVE_SOURCES}."
            )

        native_sources = frozenset(tokens)

        # legacy/read with a non-empty source list is normal --
        # operators pre-stage the source list before flipping the mode
        # switch. Not a failure, just worth an INFO line.
        if mode != "native" and native_sources:
            logger.info(
                "%s is set while %s=%s; it has no effect until mode is "
                "switched to native.",
                INTERACTION_NATIVE_SOURCES,
                INTERACTION_PROTOCOL_MODE,
                mode,
            )

        # "trigger" tasks are hidden tasks with no interactive responder --
        # allowing them is never wrong on its own, but it can never be
        # answered, so operators get a WARNING instead of silence.
        if "trigger" in native_sources:
            logger.warning(
                "%s includes 'trigger': trigger-originated tasks are "
                "hidden tasks with no interactive responder, so native "
                "publication for them will never be answered.",
                INTERACTION_NATIVE_SOURCES,
            )

        # Normalization (case/whitespace) changing the raw value is by
        # design, not a fault -- INFO only, so operators can see it happened.
        normalized_sources = ",".join(tokens)
        if (raw_mode is not None and raw_mode != mode) or (
            raw_sources_env is not None and raw_sources_env != normalized_sources
        ):
            logger.info(
                "Interaction rollout config normalized: %s=%r->%r %s=%r->%r",
                INTERACTION_PROTOCOL_MODE,
                raw_mode,
                mode,
                INTERACTION_NATIVE_SOURCES,
                raw_sources_env,
                normalized_sources,
            )

        # mode was already checked against _VALID_MODES above; mypy cannot
        # narrow a plain str to the Literal from that membership check, so
        # the cast documents an invariant already enforced by the raise.
        policy = InteractionRolloutPolicy(
            mode=cast('Literal["legacy", "read", "native"]', mode),
            native_sources=native_sources,
            native_protocol_version=INTERACTION_PROTOCOL_VERSION,
        )

        # Unconditional, same shape as app.py's "Agent runtime configured:"
        # startup line -- this is the one line an operator can grep for to
        # know what a given process actually resolved, every time, not just
        # on failure.
        logger.info(
            "Interaction rollout policy configured: mode=%s native_sources=%s "
            "native_protocol_version=%s",
            policy.mode,
            sorted(policy.native_sources),
            policy.native_protocol_version,
        )

        _policy = policy
        return policy


def _require_policy() -> InteractionRolloutPolicy:
    """Shared uninitialized-singleton guard.

    Not itself a tracked consumer-facing symbol -- both the public
    accessor below and the gate's own internal read go through this, so
    the gate reading its own owner module's state does not inflate the
    call-site count the ``get_interaction_rollout_policy`` guard tracks
    for *external* consumers.
    """
    if _policy is None:
        raise RuntimeError(
            "get_interaction_rollout_policy() called before "
            "validate_interaction_rollout_at_startup() initialized the "
            "interaction rollout policy singleton."
        )
    return _policy


def get_interaction_rollout_policy() -> InteractionRolloutPolicy:
    """Return the frozen policy singleton.

    Raises ``RuntimeError`` if called before
    ``validate_interaction_rollout_at_startup()`` -- this accessor never
    lazily parses env itself. In production this path is unreachable: the
    startup event calls the validator before the process accepts traffic.
    A caller that hits this is a programming error (a new process entry
    point that forgot to call the validator), not a runtime condition to
    recover from, so it is treated as one.
    """
    return _require_policy()


# ---------------------------------------------------------------------------
# The publication gate.
# ---------------------------------------------------------------------------


def _truncated_repr(value: object, *, limit: int = 64) -> str:
    """``repr()`` a value and cap the result to ``limit`` characters.

    ``Task.source`` is a ``String(20)`` column with no CHECK constraint, and
    SQLite does not enforce column length at all, so it can hold arbitrary-
    length text. ``repr()`` (not plain ``str()``) makes whitespace and other
    invisible characters visible -- the most common shape of an unknown
    source once normalization stops stripping/lowercasing -- and the length
    cap keeps a pathological value out of logs and degradation detail.
    """
    text = repr(value)
    if len(text) > limit:
        return text[:limit] + "..."
    return text


class NativePublicationDecision(enum.Enum):
    """The five possible outcomes of :func:`evaluate_native_publication`.

    ``ALLOWED`` is the only member carrying a payload:
    :attr:`protocol_version`, which mirrors ``INTERACTION_PROTOCOL_VERSION``
    (``task_interaction.py``) rather than defining its own value. No member
    carries an ``origin`` field -- the normalized origin value computed
    during evaluation is used only inside the gate, for vocabulary lookups
    and logging. It never becomes part of what the gate returns, because the
    gate answers "may a native row be published," and letting a write-side
    input ride along on that answer invites a future caller to treat the
    gate's return value as if it were an authoritative reading of already-
    published data.
    """

    ALLOWED = "allowed"
    BLOCKED_MODE = "blocked_mode"
    BLOCKED_UNKNOWN_SOURCE = "blocked_unknown_source"
    BLOCKED_SOURCE = "blocked_source"
    BLOCKED_SCHEMA_ABSENT = "blocked_schema_absent"

    @property
    def protocol_version(self) -> int | None:
        if self is NativePublicationDecision.ALLOWED:
            return INTERACTION_PROTOCOL_VERSION
        return None


def evaluate_native_publication(
    db: "Session", task: "Task"
) -> NativePublicationDecision:
    """Decide whether ``task`` may publish a native interaction row now.

    Call this only when a finalizer is about to transition a task into
    ``WAITING_FOR_USER`` -- this function does not check task status
    itself, it assumes that precondition already gated the call.

    Guard order is the contract, not an implementation detail:

    1. The mode check runs first. The default deployment carries 100% of
       traffic in legacy mode, so putting the cheapest possible check first
       (one attribute read off the frozen policy singleton, one string
       compare -- no environment read happens here, unlike in earlier,
       unfrozen designs) drives the gate's total cost to zero queries for
       the common case.
    2. The schema-existence check is the only step that issues a query, so
       it runs last among the blocking checks -- after every check capable
       of rejecting for free has had its turn.
    3. Unknown source is checked before allow-list membership, not after:
       an unrecognized ``Task.source`` value is a data anomaly worth a
       human's attention, while a recognized-but-not-yet-allowed source is
       ordinary rollout progress. Merging the two into one outcome would
       bury the anomaly inside the routine case.

    Never opens a transaction, writes a row, resolves a resume anchor, or
    takes a lock -- it only answers the yes/no question, never acts on the
    answer. That boundary is what lets this gate exist with zero production
    callers and full unit coverage.
    """
    policy = _require_policy()

    if policy.mode != "native":
        logger.debug("Native publication blocked: mode=%s", policy.mode)
        increment_counter(COUNTER_ROLLOUT_DECISION_BLOCKED_MODE)
        return NativePublicationDecision.BLOCKED_MODE

    key = gating_key(task)
    if key is None:
        detail = (
            f"task {task.id}: unrecognized Task.source {_truncated_repr(task.source)}"
        )
        logger.info("Native publication blocked: %s", detail)
        register_degradation(INTERACTION_ROLLOUT_UNKNOWN_TASK_SOURCE, detail)
        increment_counter(COUNTER_ROLLOUT_DECISION_BLOCKED_UNKNOWN_SOURCE)
        return NativePublicationDecision.BLOCKED_UNKNOWN_SOURCE

    if key not in policy.native_sources:
        logger.info("Native publication blocked: source %r not in allowed sources", key)
        increment_counter(COUNTER_ROLLOUT_DECISION_BLOCKED_SOURCE)
        return NativePublicationDecision.BLOCKED_SOURCE

    if not interaction_requests_table_exists(db):
        detail = "task_interaction_requests table not present"
        logger.info("Native publication blocked: %s", detail)
        register_degradation(INTERACTION_ROLLOUT_SCHEMA_ABSENT, detail)
        increment_counter(COUNTER_ROLLOUT_DECISION_BLOCKED_SCHEMA_ABSENT)
        return NativePublicationDecision.BLOCKED_SCHEMA_ABSENT

    clear_degradation(INTERACTION_ROLLOUT_SCHEMA_ABSENT)
    logger.info("Native publication allowed: source %r", key)
    increment_counter(COUNTER_ROLLOUT_DECISION_ALLOWED)
    return NativePublicationDecision.ALLOWED
