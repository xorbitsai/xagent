"""Decide whether a clarification draft attached to a finalizing run can be
published as a structured interaction request, and build/parse the JSON
payload a published row carries.

This module answers one question: given the result dict a pattern run just
produced, the task row and lease the caller already holds, and the resume
anchor the caller already resolved, does the clarification draft attached to
that result (if any) qualify to become a durable, structured interaction
row? The answer is one of three outcomes, never folded into a boolean:

* :class:`Publishable` -- the draft passed every precondition and is shaped
  into the three keyword arguments ``interaction_handoff.stage()`` expects
  (``request_payload``, ``request_idempotency_key``, ``expires_at``).
* :class:`NotApplicable` -- nothing durable should be written this round,
  but nothing failed either: either this run never produced a clarification
  in the first place, or a waiting run's draft could not be turned into a
  publishable payload (no resume anchor, or the payload could not be shaped
  within the size and character limits below). The caller's own settlement
  proceeds exactly as it would have before this module existed.
* :class:`FailClosed` -- carries one of five reason strings that split into
  two different consequences, not one. Two of them (from the guard that
  pairs waiting status against draft presence) do not discard the round at
  all: no structured row is written, but the finalizer's existing path for
  that status proceeds unchanged. The other three (an unfenced lease, a
  task row no longer owned by the lease that produced this result, or an
  attempt that is no longer current) do mean the caller must discard this
  round's settlement wholesale, the same way it already discards a late,
  no-longer-owned result. See :class:`FailClosed` itself for which reason
  is which.

Guard order is a contract: the checks inside :func:`resolve_publishable_clarification`
run in a fixed sequence, most specific and least data-exposing failure
first, so that the classification a caller observes always names the first
thing that actually went wrong -- never a later, coincidentally-true guard
that happens to fail on the same input for an unrelated reason.

Callers. This function's caller set is exactly the finalizers that settle a
task run against its lease and load the task row under a lock: nothing else
holds the combination of task, lease and anchor this function's guards
depend on. A caller reached from anywhere else -- a builder chat session, a
nested sub-agent's own turn -- was never going to be a caller in the first
place, and this module makes no claim about what happens there. Extending
the caller set to a fourth site is a reason to come back and re-read this
module, not to assume its guards still describe the new caller's situation.

Timing is as strict as the caller set: the finalizer must call this
function before it releases the lease, against the same task row it
already holds locked under ``SELECT ... FOR UPDATE`` -- never a row
re-read afterward, which could already reflect a different attempt's
ownership by the time this function's ownership and attempt guards run.

This module is also the only place allowed to construct or parse the JSON
shape a published row's ``request_payload`` carries -- see
:func:`build_clarification_payload` and :func:`parse_clarification_payload`
for the pairing rule and the version-evolution promise that comes with it.

Two of that payload's fields, ``source`` and ``requests``, have no reader
anywhere in this delivery. They are there for whichever change adds the
response-routing path that answers a multi-tool waiting turn: that reader
needs to know which source produced the draft and which pending tool call
each answer belongs to, and once this payload is published, it is the only
place that information still lives. Do not remove either field for being
unread today -- removing them now would trade the free "add an optional
field" path above for a version bump later, to add back exactly what was
just deleted.

An application-layer invariant worth stating plainly: a task can have at
most one *active* interaction row at a time (the database enforces this
with a unique constraint on the active slot). Nothing in this module
inserts a row -- that is the caller's job, via ``stage()`` -- but every
guard below exists to make sure that when the caller does insert one, it is
the single row this run is entitled to produce.

One more precondition lives outside this module entirely and cannot be
checked at run time: whichever deployment decides to start publishing
structured rows for a given ``Task.source`` value needs that value to
already appear in the model's ``INTERACTION_ORIGIN_VOCABULARY`` (see
``xagent.web.models.task_interaction``). A source value outside that
vocabulary is not rejected here -- it degrades one layer down, inside
``interaction_handoff``, as an unrecognized origin swallowed into a generic
degradation signal. Confirming vocabulary coverage against a live
deployment's actual ``tasks.source`` values before turning publication on
is the caller's obligation, not this module's.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Literal, Mapping

from ...core.agent.clarification import ClarificationDraft
from ..models.task import Task
from .ops_signals import (
    CLARIFICATION_DRAFT_MISSING,
    CLARIFICATION_MULTIPLE_DRAFTS,
    clear_degradation,
    register_degradation,
)
from .task_interaction_staging import InteractionAnchor
from .task_lease_service import (
    TaskLease,
    lease_is_fenced,
    task_row_matches_lease_attempt,
    task_row_matches_lease_owner,
)

logger = logging.getLogger(__name__)

# Deployment policy, not an invariant: this only bounds how long an
# unanswered row sits before a reclaim job retires it. The read surface
# does not consult ``expires_at`` at all, so a longer-lived row is still
# shown to whoever is waiting on it; widen or narrow this once real
# publication volume gives a basis for the number, not before.
CLARIFICATION_REQUEST_TTL = timedelta(hours=24)

# The character domain a payload's free-text leaves may contain, kept
# byte-for-byte identical to ``xagent.core.agent.clarification``'s
# ``_marker_clean`` whitelist. That module cannot import this one (core
# layer never imports web layer), so the two copies cannot share code --
# ``_marker_clean``'s own docstring enumerates this copy among its five.
# What actually pins the two domains against drifting apart is
# ``test_character_domain_matches_the_marker_cleaning_domain`` in
# ``tests/web/services/test_task_clarification_draft.py``, which calls the
# real ``_marker_clean`` and compares it against this filter directly. If
# one domain ever changes, that test is what will turn red.
_CONTROL_CHAR_KEEP = "\n\r\t"

# Three independent size ceilings, checked in this order because each one
# can only be evaluated once the field(s) before it are already final.
_QUESTION_MAX_BYTES = 16 * 1024
_INTERACTIONS_MAX_BYTES = 64 * 1024
_TOTAL_PAYLOAD_MAX_BYTES = 128 * 1024

_QUESTION_TRUNCATION_SUFFIX = "\n\n[question truncated]"


def _strip_control_characters(value: str) -> str:
    """Remove NUL and C0 control characters, keeping the same three
    whitespace exceptions ``_marker_clean`` keeps. See the module-level
    comment above for the cross-layer contract this filter must not drift
    from."""

    # Both replaces below target the same code point, NUL (U+0000); the
    # second is a no-op after the first. It is kept so this filter's
    # character-handling semantics match clarification.py's
    # ``_marker_clean`` exactly, for the five-copy comparison the
    # cross-layer contract in the comment above ``_CONTROL_CHAR_KEEP``
    # depends on -- not because the two copies are written alike
    # line-for-line: that one uses two separate statements, this one
    # chains both replaces into a single line.
    cleaned = value.replace("\x00", "").replace("\u0000", "")
    return "".join(ch for ch in cleaned if ord(ch) >= 32 or ch in _CONTROL_CHAR_KEEP)


def _clean_leaves(value: Any) -> Any:
    """Recursively apply :func:`_strip_control_characters` to every string
    leaf inside a JSON-like structure, leaving every other value untouched."""

    if isinstance(value, str):
        return _strip_control_characters(value)
    if isinstance(value, dict):
        return {key: _clean_leaves(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_leaves(item) for item in value]
    return value


def _serialized_byte_length(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False).encode("utf-8"))


def _truncate_to_byte_limit(text: str, max_bytes: int) -> str:
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    # Decoding a byte-sliced UTF-8 string can land mid-character; dropping
    # whatever partial tail remains is preferable to raising here, since
    # this path must never fail the round it is degrading.
    return encoded[:max_bytes].decode("utf-8", errors="ignore")


# The full value sets below, enumerated from every construction site in
# :func:`resolve_publishable_clarification` and from :class:`NotApplicable`
# and :class:`FailClosed`'s own docstrings. Both are ``str`` at runtime --
# ``Literal`` only narrows what a type checker accepts, so a caller reading
# ``resolution.fail_closed_reason == "attempt_mismatch"`` needs no change.
NotApplicableReason = Literal["no_anchor", "payload_too_large", "empty_question"]
FailClosedReason = Literal[
    "missing_draft",
    "draft_status_mismatch",
    "unfenced_lease",
    "ownership_changed",
    "attempt_mismatch",
]


@dataclass(frozen=True)
class Publishable:
    """The draft passed every precondition; these three values are exactly
    the keyword arguments ``interaction_handoff.stage()`` expects beyond
    ``kind`` and ``protocol_version``, which the caller supplies directly."""

    request_payload: dict[str, Any]
    request_idempotency_key: str
    expires_at: datetime
    fail_closed_reason: None = field(default=None, init=False, repr=False)


@dataclass(frozen=True)
class NotApplicable:
    """No durable row should be written this round, and nothing failed.

    ``reason`` is ``None`` for the ordinary case -- this finalization was
    never a waiting-for-user turn with a draft attached, so publication
    simply does not apply, and the caller's settlement proceeds exactly as
    it would without this module. ``reason`` is one of ``"no_anchor"``,
    ``"payload_too_large"``, or ``"empty_question"`` when a waiting turn's
    draft existed but could not be shaped into a publishable payload; each
    of those is a degrade, not a failure -- the run is not discarded.
    """

    reason: NotApplicableReason | None
    fail_closed_reason: None = field(default=None, init=False, repr=False)


@dataclass(frozen=True)
class FailClosed:
    """No structured row is published this round. What the caller must do
    about the *round itself* depends on which guard produced ``reason`` --
    the five values split into two different consequences, not one.

    ``"missing_draft"`` and ``"draft_status_mismatch"`` come from guard 1
    (the waiting-status/draft-presence pairing) and do not discard the
    round: no structured row is written and a stray draft is dropped, but
    the finalizer's existing path for that status proceeds exactly as it
    already does today.

    ``"unfenced_lease"``, ``"ownership_changed"``, and ``"attempt_mismatch"``
    come from guards 2 through 4 and mean the opposite: this round's
    settlement must be discarded wholesale, the same way the caller already
    discards a late, no-longer-owned result, via that caller's own existing
    late-result exit -- this module adds no new one.
    """

    reason: FailClosedReason

    @property
    def fail_closed_reason(self) -> FailClosedReason:
        return self.reason


ClarificationResolution = Publishable | NotApplicable | FailClosed


def clarification_idempotency_key(draft: ClarificationDraft) -> str:
    """Derive a ``stage()`` idempotency key from the draft's content.

    Reads only ``turn_marker`` -- never ``message`` or ``interactions`` --
    so that truncating or otherwise reshaping the payload in
    :func:`build_clarification_payload` can never change the key a re-entrant
    resume of the same turn produces. A key that moved with the payload
    would turn a replay that should cost nothing (the active row already
    matches) into a fresh request every time the payload's shape changed.
    """

    digest = hashlib.sha256(draft.turn_marker.encode("utf-8")).hexdigest()[:32]
    return f"clarification.{digest}"


def build_clarification_payload(draft: ClarificationDraft) -> dict[str, Any]:
    """Build the v1 ``request_payload`` shape for a clarification draft.

    Processing order is fixed: character-domain filtering happens first,
    length truncation second. Reversing the order would make the same input
    produce a different truncated length depending on how many control
    characters a truncated tail happened to contain -- the same input must
    always truncate to the same length.

    Every step here degrades rather than raises: a control character is
    dropped silently (it was never visible to begin with), an over-length
    ``question`` is cut with a fixed suffix and flagged, and an over-length
    ``interactions`` list is dropped to an empty list and flagged --
    truncating half of a multi-field form is worse than presenting none of
    it. The one condition worth a log line is a ``question`` that comes out
    empty after filtering: that is not a truncation, it means the visible
    question is gone.

    ``protocol_version`` is not a key in this dict on purpose -- the caller
    passes it to ``stage()`` directly, matching that primitive's own
    signature. This payload's own version-evolution promise, independent of
    that column, is: as long as this shape stays v1, only optional fields
    may be added here. Renaming a key, changing a key's type, removing a
    key, or changing what an existing key means requires bumping the
    protocol version and giving :func:`parse_clarification_payload` an
    explicit branch for the old shape -- never a silent redefinition of what
    v1 means. The one exception already taken under this promise is the
    ``question`` -> ``message`` rename below: it shipped without a version
    bump only because the table this payload is written to held no rows at
    the time, so there was no persisted v1 shape to keep reading. Any
    later rename, once the table has writers, must bump the version like
    every other case above.

    The free-text field is called ``message``, not ``question``, because
    the persisted row's reader is ``parse_v1_request_payload``
    (``task_interaction_service.py``), which validates against
    ``AskUserQuestionArgs`` (``core/tools/adapters/vibe/ask_user_tool.py``),
    whose required fields are ``message`` and ``interactions`` (neither has
    a default). That model is also the
    ``ask_user_question`` tool's own argument schema, so it is the fixed
    side of this pairing: this builder aligns to it, never the reverse. A
    payload keyed ``question`` fails that validation, and the reader
    swallows the failure and falls back to the legacy transcript with no
    log, no signal and no counter -- a shape that is silent by
    construction, which is why the round-trip test below crosses both
    modules instead of checking this function against itself.
    """

    original_question = draft.message
    cleaned_question = _strip_control_characters(original_question)
    stripped_characters = len(original_question) - len(cleaned_question)

    if not cleaned_question:
        logger.warning(
            "clarification question was empty after removing control characters",
            extra={
                "original_length": len(original_question),
                "stripped_characters": stripped_characters,
            },
        )

    message_truncated = False
    question = cleaned_question
    if len(cleaned_question.encode("utf-8")) > _QUESTION_MAX_BYTES:
        suffix_bytes = len(_QUESTION_TRUNCATION_SUFFIX.encode("utf-8"))
        question = (
            _truncate_to_byte_limit(
                cleaned_question, _QUESTION_MAX_BYTES - suffix_bytes
            )
            + _QUESTION_TRUNCATION_SUFFIX
        )
        message_truncated = True

    interactions_cleaned = [_clean_leaves(dict(item)) for item in draft.interactions]
    interactions_dropped = False
    interactions_payload: list[Any] = interactions_cleaned
    if _serialized_byte_length(interactions_cleaned) > _INTERACTIONS_MAX_BYTES:
        interactions_payload = []
        interactions_dropped = True

    payload: dict[str, Any] = {
        "message": question,
        "interactions": interactions_payload,
        "message_type": _strip_control_characters(draft.message_type),
        "source": draft.source,
        "requests": [_clean_leaves(item.to_dict()) for item in draft.requests],
    }
    if message_truncated:
        payload["message_truncated"] = True
    if interactions_dropped:
        payload["interactions_dropped"] = True

    total_bytes = _serialized_byte_length(payload)
    if total_bytes > _TOTAL_PAYLOAD_MAX_BYTES:
        logger.warning(
            "clarification payload exceeded the maximum size even after truncation",
            extra={"serialized_length": total_bytes},
        )

    return payload


def parse_clarification_payload(
    payload: Mapping[str, Any],
) -> tuple[str | None, list[dict[str, Any]] | None]:
    """Read a v1 clarification payload back into the ``(question,
    interactions)`` shape ``get_latest_waiting_question``
    (``chat_history_service.py``) already returns for the legacy transcript
    path -- a cross-module contract with no shared base class to enforce it,
    pinned by ``test_payload_round_trip_matches_the_legacy_reader_shape*``
    in ``tests/web/services/test_task_clarification_draft.py``, which calls
    the real ``get_latest_waiting_question`` and compares its return shape
    against this function's, rather than re-deriving the expectation from
    this function's own return type.

    ``interactions`` comes back ``None``, not ``[]``, whenever the payload
    carries no interaction items. ``get_latest_waiting_question`` itself is
    not this consistent: it returns ``None`` for a persisted row whose
    ``interactions`` column is ``NULL`` (the shape a ``send_message``-
    sourced draft always produces, since that source's payload never has
    any interactions to carry), but it returns ``[]`` for a row whose
    column holds an actual empty list -- the shape an empty-form
    ``ask_user_question`` call produces (a legal call: ``ask_user_question``
    is classified by whether its request carries an ``interactions`` key at
    all, not by whether that key's value is non-empty; see
    ``draft_from_waiting_request``). That ``NULL``-vs-``[]`` split on the
    legacy side is a known, un-reconciled divergence, not something this
    function reproduces: every empty case collapses to ``None`` here,
    deliberately, so a caller checking "did this turn have interactions"
    never has to handle two different falsy shapes.

    This function and ``parse_v1_request_payload``
    (``task_interaction_service.py``) are two readers of the same v1 shape,
    not two competing authorities. That one validates the whole envelope
    against ``AskUserQuestionArgs`` and is what the read surface
    (``materialize_compatibility_view``) uses on a persisted row. This one
    is the lossy projection down to the legacy ``(question, interactions)``
    tuple ``get_latest_waiting_question`` (``chat_history_service.py``)
    returns, for callers that need that tuple and nothing else. Both read
    the same key, ``message``; neither may reach into ``payload["message"]``
    at a call site, so that a future protocol version bump needs a new
    branch in these two functions and nowhere else.
    """

    question = payload.get("message")
    interactions = payload.get("interactions")
    return (
        question if isinstance(question, str) else None,
        interactions if isinstance(interactions, list) and interactions else None,
    )


def resolve_publishable_clarification(
    result: Mapping[str, Any],
    *,
    task: Task,
    lease: TaskLease,
    anchor: InteractionAnchor | None,
    now: datetime,
) -> ClarificationResolution:
    """Classify one finalizing run's clarification draft.

    Guard order is the contract (most specific, least data-exposing failure
    first):

    1. ``result["status"] == "waiting_for_user"`` paired against draft
       presence. A waiting result with no draft is a fail-closed
       ``"missing_draft"``: nothing about this run says a question should
       have existed, but the pattern says the user should be asked one, and
       there is nothing to ask. A non-waiting result carrying a draft
       (never expected from the core pattern layer, only reachable if a
       caller mixes up two results) is fail-closed ``"draft_status_mismatch"``,
       and the stray draft is discarded rather than acted on. Either of the
       two matched pairs -- waiting-with-draft, or non-waiting-with-nothing
       -- is not a failure: the first continues to the remaining guards, the
       second is simply not applicable, and the caller's own non-waiting
       settlement proceeds untouched.
    2. ``lease.run_id is not None`` -- an unfenced lease cannot prove which
       run produced this result.
    3. ``task.runner_id == lease.runner_id and task.run_id == lease.run_id``
       -- the task row must still be owned by the lease that produced the
       result.
    4. ``lease.attempt_id is None or task.lease_attempt_id == lease.attempt_id``
       -- ``None`` means the lease cannot prove attempt identity at all and
       must be treated as "skip this check", never as "matches" (the same
       reading ``interaction_handoff`` itself uses for the identical
       sentinel). A concrete mismatch means a later attempt has already
       claimed the row, and this settlement must be discarded wholesale
       rather than degrade -- an attempt that is no longer current has no
       business writing anything at all.

       Guards 2 through 4 evaluate ``lease_is_fenced``,
       ``task_row_matches_lease_owner`` and ``task_row_matches_lease_attempt``
       (``task_lease_service.py``). The booleans are shared with the
       interaction handoff's own re-check so the two cannot drift; the
       fail-closed classification each one produces here is this function's
       alone.
    5. ``anchor is not None`` -- no resume anchor means this run's most
       recent checkpoint was never one a structured interaction can resume
       against; this degrades, it does not fail the round.
    5b. The payload built from the draft fits the character domain and size
        limits in :func:`build_clarification_payload`; if it does not
        (empty after filtering, or reduced to only whitespace, or still
        oversized after truncation), that also degrades rather than fails.
        This check runs last because it is the only one that requires
        building the payload first.

    A cross-run anchor mismatch is deliberately absent from this sequence:
    that check belongs to ``interaction_handoff`` (it already raises a
    dedicated exception for exactly this case, reported under its own
    degradation signal), not to this resolver.

    ``now`` must be an aware UTC datetime -- the same obligation
    ``stage_interaction_request`` documents for its own ``now`` parameter
    (``task_interaction_staging.py``). It is folded into ``expires_at``
    (``now + CLARIFICATION_REQUEST_TTL``) on the ``Publishable`` path, and a
    naive or non-UTC value would corrupt that column the same way an
    unvalidated ``expires_at`` would. This function does not itself
    validate ``now``: ``stage_interaction_request``'s own validation of the
    ``expires_at`` this function hands it already enforces the obligation
    downstream, so a second check here would only duplicate that guard.

    ``CLARIFICATION_MULTIPLE_DRAFTS`` is registered before any guard below
    gets a chance to return, not only on the ``Publishable`` path: whether
    ``result["clarification_superseded_step_ids"]`` is non-empty is a fact
    about this round's ``result`` that holds regardless of which guard
    ultimately classifies it -- a losing waiting step can be superseded by
    an interrupt that wins the same round (see ``AgentExecutionAdapter``'s
    ``"interrupted"`` branch), so ``NotApplicable(None)`` and every other
    non-``Publishable`` outcome must still be able to report the conflict.
    Clearing it is the opposite: confined to the ``Publishable`` path only,
    alongside ``CLARIFICATION_DRAFT_MISSING`` -- a successful publish
    clears the signal only when nothing was superseded *this same round*
    either, since publishing this round's draft does not retroactively
    undo a sibling step's draft having been discarded.
    """

    superseded_step_ids = result.get("clarification_superseded_step_ids") or []
    if superseded_step_ids:
        register_degradation(
            CLARIFICATION_MULTIPLE_DRAFTS,
            f"task {task.id}: run {lease.run_id}: superseded step ids "
            f"{list(superseded_step_ids)}",
        )

    status = result.get("status")
    draft = result.get("clarification_draft")
    is_waiting = status == "waiting_for_user"

    if is_waiting and draft is None:
        logger.warning(
            "a waiting run produced no clarification draft to publish",
            extra={"task_id": task.id, "run_id": lease.run_id},
        )
        register_degradation(
            CLARIFICATION_DRAFT_MISSING,
            f"task {task.id}: waiting run produced no clarification draft",
        )
        return FailClosed("missing_draft")

    if not is_waiting and draft is not None:
        # Still error level, unlike the three warnings above: those three sit
        # on paths that degrade and let the round proceed, while this one is
        # only reachable when a caller hands this function two different
        # runs' results mixed together -- a programming error the core
        # pattern layer cannot produce (see guard 1 in this function's
        # docstring), not a degradation.
        logger.error(
            "a non-waiting run carried a clarification draft; discarding it",
            extra={"task_id": task.id, "run_id": lease.run_id, "status": status},
        )
        return FailClosed("draft_status_mismatch")

    if not is_waiting:
        return NotApplicable(None)

    assert draft is not None  # narrowed by the two branches above

    if not lease_is_fenced(lease):
        return FailClosed("unfenced_lease")

    if not task_row_matches_lease_owner(task, lease):
        return FailClosed("ownership_changed")

    if not task_row_matches_lease_attempt(task, lease):
        return FailClosed("attempt_mismatch")

    if anchor is None:
        return NotApplicable("no_anchor")

    payload = build_clarification_payload(draft)

    if not payload["message"].strip():
        return NotApplicable("empty_question")

    if _serialized_byte_length(payload) > _TOTAL_PAYLOAD_MAX_BYTES:
        return NotApplicable("payload_too_large")

    clear_degradation(CLARIFICATION_DRAFT_MISSING)
    if not superseded_step_ids:
        clear_degradation(CLARIFICATION_MULTIPLE_DRAFTS)

    return Publishable(
        request_payload=payload,
        request_idempotency_key=clarification_idempotency_key(draft),
        expires_at=now + CLARIFICATION_REQUEST_TTL,
    )
