"""Typed lifecycle service for blocking task interaction requests: the
answer-side counterpart to ``task_interaction_staging.py``'s ask-side
primitive.

Not merged into ``task_interaction_staging.py``. That module's own
docstring pins its merge reason to a fact this module does not share: its
two entry points exist in one file because they share every exception type
and one nesting invariant that must never be split across a file boundary,
both live on the ask side, and both require the caller to already hold the
transaction they run inside. This module's future ``respond()`` -- not
implemented in this delivery, see below for what is -- is designed to be
the opposite shape on every one of those points: it will own its own
session and its own commit, and will not nest inside anyone else's
savepoint. Nothing in this delivery opens a session or commits anything;
``create()`` takes a caller-owned session and only reads through it. Putting
the future seam in the same file as the staging primitive would make that
staging module's merge-reason docstring false the day it lands. What this
delivery reuses from the staging module
instead of duplicating is narrow and one-directional: the private kind
vocabulary (``_KIND_VOCABULARY``), imported by name. ``create()`` never
calls ``stage_interaction_request`` in this delivery and therefore never
raises or catches any of that module's nine exception classes, and this
module's own read-direction anchor resolver validates a real
``trace_events`` row against a stored interaction row's fields -- a
different check from the staging module's ``_validate_anchor_fields``,
which validates an ``InteractionAnchor`` value object before an INSERT --
so it is not reused here either. See ``create()``'s own docstring for the
exception-family accounting, and ``_resolve_read_direction_anchor``'s for
the anchor-check distinction.

Concurrency precondition, stated once here because every rowcount-based
branch this service will grow depends on it: every rowcount-based branch
below assumes READ COMMITTED (PostgreSQL's default, and what this
deployment uses): a blocked UPDATE re-evaluates its WHERE clause after the
lock is released. Under REPEATABLE READ or stricter the same conflict
surfaces as a serialization failure instead, and the classification this
service documents does not hold.

Audit identity: a row's ``responder_user_id`` foreign key is cleared by the
database itself (``ON DELETE SET NULL``) if the responding user is later
deleted; ``responder_identity`` is not touched by that FK and is the only
field this table's audit trail can rely on staying populated. Anything that
records who answered a request must treat ``responder_identity`` as
authoritative and ``responder_user_id`` as a convenience join that can go
missing under normal account deletion, not a corruption signal.

Delivered here: the ``InteractionPrincipal`` value object and the shared
public-chat ownership predicate extracted from ``public_chat_access.py``;
the ``RespondOutcome`` and ``CreateOutcome`` discriminated unions and their
reason vocabularies; the ``create()`` typed seam (validates and returns,
does not stage a row); ``get()``/``list_active()``; and the three-tier compatibility
materialization view. Not delivered here: ``respond()``'s call body, the
answer fence, the compatibility seam into the existing resume coordinator,
and any new counter. Those land with later changes; this module's own
zero-production-caller gate
(``tests/web/services/test_task_interaction_service_production_gate.py``)
gives that boundary a regression guard against import bindings, not an
absolute one -- the gate's own docstring lists what it cannot see
(dynamic access, alias/re-export chains, filename-stem exclusion, and
code outside the scanned package tree).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError as _PydanticValidationError

from ...core.agent.checkpoint import (
    CHECKPOINT_EVENT_TYPE,
    READABLE_CHECKPOINT_TYPES,
    checkpoint_execution_id,
)
from ...core.tools.adapters.vibe.ask_user_tool import AskUserQuestionArgs
from ..models.task import Task, TraceEvent
from ..models.task_interaction import (
    INTERACTION_PROTOCOL_VERSION,
    TaskInteractionRequest,
)
from .chat_history_service import get_latest_waiting_question
from .ops_signals import (
    CHECKPOINT_LOAD_UNAVAILABLE,
    CHECKPOINT_PK_ANCHOR_DANGLING,
    register_degradation,
)
from .task_command_transport import _normalize_command_id
from .task_interaction_schema import interaction_requests_table_exists
from .task_interaction_staging import _KIND_VOCABULARY
from .task_lease_service import TASK_RUN_ID_TRACE_FIELD

if TYPE_CHECKING:
    from datetime import datetime

    from sqlalchemy.orm import Session

# ---------------------------------------------------------------------------
# Principal and the shared public-chat ownership predicate
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InteractionPrincipal:
    """A resolved caller identity: either an authenticated web user or an
    anonymous widget/share guest, carrying every field the public-chat
    ownership predicate below and the future answer-side authorization
    predicate need.

    This is not a transport-authentication object -- it does not verify a
    JWT, a widget key, or an embed ticket. Callers resolve those themselves
    (``public_chat_access.py`` already does, for the widget and share
    channels) and construct this value object from the result. Splitting it
    this way keeps authentication where the transport already lives and
    puts only the authorization predicate here, which is the reading this
    module's callers need: the four public-chat entry points already decode
    their own credentials before ever calling the ownership predicate.

    ``kind`` is ``"user"`` for an authenticated web session or ``"guest"``
    for a widget/share visitor. ``user_id`` is populated for both kinds --
    an authenticated user is also the task owner or an admin acting on it,
    and a guest's owning user is the entity (agent/workforce) owner the
    guest is chatting through, not the guest itself. ``is_admin`` records
    whether this is an admin acting on someone else's task (the turn still
    runs as the task owner; ``is_admin`` is audit-only). ``auth_mode``, the
    four ``*_id`` entity-binding fields, and ``guest_id`` mirror the fields
    ``public_chat_access.py``'s access-context dataclasses already carry
    (see that module's ``PublicChatAccessContext`` and
    ``ShareChatAccessContext``); exactly one of ``widget_agent_id`` /
    ``widget_workforce_id`` / ``share_agent_id`` / ``share_workforce_id`` is
    populated for a guest principal, none for a ``"user"`` principal.

    ``identity_string`` produces the same ``"user:{id}"`` / ``"guest:{gid}"``
    namespacing the ``responder_identity`` column comment documents
    (``models/task_interaction.py``): the prefix is a namespace, not
    decoration, because a bare user id and a bare guest id share no type and
    must never compare equal by accident.
    """

    kind: str  # "user" | "guest"
    user_id: int | None
    is_admin: bool
    auth_mode: str | None  # "widget" | "share" | None
    widget_agent_id: int | None = None
    widget_workforce_id: int | None = None
    share_agent_id: int | None = None
    share_workforce_id: int | None = None
    guest_id: str | None = None

    def identity_string(self) -> str:
        if self.kind == "guest":
            return f"guest:{self.guest_id}"
        return f"user:{self.user_id}"


def _json_entity_binding_matches(config_value: Any, expected: int) -> bool:
    """True when a JSON-sourced entity-binding value equals ``expected``.

    ``config_value`` comes from ``Task.agent_config``, a JSON column
    another writer controls -- it is untrusted data, not a value this
    predicate can assume is even convertible to ``int``. A non-empty
    string that is not a number, a list, or a dict, or a JSON float of
    infinity or NaN, would make a bare ``int(config_value)`` raise
    ``TypeError``/``ValueError``/``OverflowError``; this treats every such
    value as "does not match" instead of letting the exception escape, the
    same fail-closed default every other conjunct in this predicate uses
    for a missing or wrong value.

    This is not a behavior change from the pre-existing code's
    ``int(x or 0) != expected`` shape for the inputs that shape actually
    handled: ``or 0`` only ever substitutes ``0`` for a *falsy*
    ``config_value`` (``None``, ``""``, ``0``, an empty list or dict), and
    ``int(0)`` never raises, so the pre-existing shape never crashed on
    falsy input either -- it just couldn't be reached with a genuinely
    unconvertible value, because the only inputs it was ever fed were
    already int-convertible ids or an absent key. A *truthy*
    non-int-convertible value (a non-numeric string, a non-empty list, a
    non-empty dict) is new ground neither shape ever needed to cover
    safely until this predicate started being called with untrusted JSON
    it does not otherwise validate the shape of.

    ``bool`` values are excluded before the ``int(...)`` conversion even
    runs, for either argument. ``bool`` is a subclass of ``int`` in Python,
    so ``int(True)`` folds to ``1`` and ``int(False)`` folds to ``0`` --
    without this exclusion, a JSON ``true``/``false`` entity-binding value
    would silently compare equal to entity id ``1``/``0``. This one *is* a
    behavior change from the pre-existing ``int(x or 0) != expected``
    shape: that shape had no notion of "not an int" at all, so a
    ``config_value`` of ``True`` folded to ``1`` and matched an
    ``expected`` of ``1`` the same as a genuine id would. This predicate
    treats that fold as a bug, not a feature, and rejects it.
    """

    if config_value is None:
        return False
    # ``bool`` is a subclass of ``int``, so ``int(True)`` folds to ``1`` and
    # would compare equal to an entity id of ``1``. Two existing guards on
    # this authorization chain already reject booleans for that reason:
    # ``_is_strict_int`` in ``web/api/public_chat_access.py`` screens the
    # JWT claim before it reaches ``expected`` here, and
    # ``_coerce_optional_entity_id`` in ``web/api/chat.py`` screens
    # ``agent_config`` reads on a separate path. The ``expected`` side is
    # therefore already pre-screened by the time it gets here -- the
    # ``isinstance(expected, bool)`` check is a second line of defense, not
    # this predicate's only one. ``config_value`` has no such upstream
    # guard, since it is read straight off ``task.agent_config``, so this
    # predicate applies the same judgement to both sides itself.
    if isinstance(config_value, bool) or isinstance(expected, bool):
        return False
    try:
        return int(config_value) == expected
    except (TypeError, ValueError, OverflowError):
        return False


def task_is_owned_by_public_principal(
    task: "Task", principal: InteractionPrincipal
) -> bool:
    """The shared conjunction extracted from ``public_chat_access.py``'s
    four widget/share ownership checks (``_get_task_for_workforce_widget_context``,
    ``get_task_for_public_context``, ``_get_task_for_workforce_share_context``,
    ``get_task_for_share_context``), plus three deliberate tightenings over
    that pre-existing code.

    Direction is inferred from which single entity-binding field is
    populated on ``principal``: exactly one of ``widget_agent_id`` /
    ``widget_workforce_id`` / ``share_agent_id`` / ``share_workforce_id``
    must be set, matching how each of the four existing entry points already
    commits to one direction structurally. Zero or more than one populated
    field is a caller error (an ambiguous principal must never be treated as
    "try every direction until one matches") and raises ``ValueError`` --
    fail closed on a malformed caller, not fail open on a guess.

    Ownership by ``Task.user_id`` is deliberately **not** one of this
    predicate's conjuncts, even though ``principal.user_id`` is a field on
    it: all four pre-existing entry points enforce it purely as a filter on
    the SQL query that loads ``task`` in the first place (``Task.user_id ==
    access_context.user.id``), never as a post-load Python attribute check,
    and this predicate preserves that division of labor rather than adding
    a second, redundant one -- adding it here would touch ``task.user_id``
    unconditionally, which the pre-existing stub-based tests for these entry
    points do not always populate on their fixture objects (they build a
    minimal double carrying only the fields the pre-existing post-load
    checks actually read). ``principal.user_id`` exists on this value object
    for the answer-side authorization predicate's own, separate use, not for
    this one.

    The conjunction, in order:

    1. ``task.channel_id is None``. All three production callers wired to
       this predicate require it (a widget/share task is never bound to a
       bot channel); the fourth direction (plain widget-agent, matched by
       ``get_task_for_public_context``) keeps its own channel-id-or-None
       filter and is deliberately *not* routed through this predicate (see
       ``get_task_for_public_context``'s own comment) -- this predicate's
       simpler "must be None" is exercised only by tests that address the
       widget-agent direction directly.
    2. ``task.agent_config`` is a ``dict``.
    3. ``task.agent_config["auth_mode"] == principal.auth_mode``. This is
       the first of the three tightenings: the pre-existing widget-agent branch of
       ``get_task_for_public_context`` never checks ``auth_mode`` at all,
       while the other three entry points do. A widget guest whose
       self-chosen ``guest_id`` happens to equal a share guest's
       server-minted one is rejected here even though the same inputs fed
       to the untouched ``get_task_for_public_context`` return the task --
       that divergence is intentional (see that function's own comment for
       the candidate-issue this predicate does not fix).
    4. Entity binding for whichever single direction ``principal``
       addresses, requiring the task-side value to be genuinely present --
       not folded from ``None``/missing into a comparable default. This is
       the second tightening: the pre-existing code's ``int(x or 0) !=
       expected`` shape treats a missing task-side value the same as a
       task-side value of exactly zero, which would accept a
       zero-or-``None`` principal field against a task whose
       ``agent_config`` simply never set the corresponding key. Requiring
       presence first closes that gap in both directions the pre-existing
       code took it: a missing config key never satisfies any principal
       value, and a missing/``None`` principal field never satisfies any
       config value. The three JSON-level comparisons below go through
       ``_json_entity_binding_matches``, which also treats a *present but
       non-int-convertible* config value (a non-numeric string, a list, a
       dict) as not matching rather than letting ``int(...)`` raise -- see
       that function's own docstring for why this is not a behavior change
       from the pre-existing shape for the inputs that shape actually saw.
       The third tightening lives in that same helper: a ``config_value``
       or ``expected`` of ``True``/``False`` is rejected outright before
       the ``int(...)`` conversion runs, because ``bool`` is a subclass of
       ``int`` and would otherwise fold to ``1``/``0`` and match a genuine
       entity id of ``1``/``0`` -- unlike the previous two, this one *is* a
       behavior change from the pre-existing ``int(x or 0)`` shape, which
       had no way to distinguish a boolean from a real id and would have
       accepted the fold.
       - widget-agent: ``task.agent_id == principal.widget_agent_id``
         (row-level only; the pre-existing code has no JSON-level widget
         entity binding to mirror -- see the candidate issue below).
       - widget-workforce: ``task.agent_config["widget_workforce_id"] ==
         principal.widget_workforce_id`` (JSON-level only).
       - share-agent: **both** ``task.agent_id ==
         principal.share_agent_id`` (row-level) **and**
         ``task.agent_config["share_agent_id"] == principal.share_agent_id``
         (JSON-level). The row-level half is, in the pre-existing
         ``get_task_for_share_context``, only a SQL ``WHERE`` filter on the
         query that loads ``task`` -- the same category of check as
         ``Task.user_id`` above. This predicate is what turns it into a
         post-load Python check for the first time; that is a choice to
         duplicate one pre-existing filter and not the other, not a
         contradiction of the ``Task.user_id`` exclusion above.
       - share-workforce: ``task.agent_config["share_workforce_id"] ==
         principal.share_workforce_id`` (JSON-level only).
    5. ``task.agent_config["guest_id"] == principal.guest_id``, with
       ``principal.guest_id`` required non-empty first -- an empty or
       ``None`` guest id must never match a task whose own ``guest_id`` is
       also empty or missing.

    Candidate issue, logged and not fixed here (#1304): the widget-agent
    direction (``get_task_for_public_context``, not routed through this
    predicate)
    has no JSON-level entity binding to mirror asymmetrically with the
    other three directions, and (separately) has no ``auth_mode`` check in
    the pre-existing code this predicate does not touch. Fixing either is a
    behavior change to a production authorization path and is out of scope
    for this change, which only extracts and adds a new, additively-called
    predicate.
    """

    populated = [
        name
        for name, value in (
            ("widget_agent", principal.widget_agent_id),
            ("widget_workforce", principal.widget_workforce_id),
            ("share_agent", principal.share_agent_id),
            ("share_workforce", principal.share_workforce_id),
        )
        if value is not None
    ]
    if len(populated) != 1:
        raise ValueError(
            "InteractionPrincipal must populate exactly one of widget_agent_id / "
            "widget_workforce_id / share_agent_id / share_workforce_id to address "
            f"task_is_owned_by_public_principal; got {populated!r}"
        )
    direction = populated[0]

    if task.channel_id is not None:
        return False
    if not isinstance(task.agent_config, dict):
        return False
    if task.agent_config.get("auth_mode") != principal.auth_mode:
        return False

    if direction == "widget_agent":
        if task.agent_id is None or task.agent_id != principal.widget_agent_id:
            return False
    elif direction == "widget_workforce":
        if not _json_entity_binding_matches(
            task.agent_config.get("widget_workforce_id"),
            principal.widget_workforce_id,
        ):
            return False
    elif direction == "share_agent":
        if task.agent_id is None or task.agent_id != principal.share_agent_id:
            return False
        if not _json_entity_binding_matches(
            task.agent_config.get("share_agent_id"), principal.share_agent_id
        ):
            return False
    else:  # share_workforce
        if not _json_entity_binding_matches(
            task.agent_config.get("share_workforce_id"),
            principal.share_workforce_id,
        ):
            return False

    if not principal.guest_id:
        return False
    if task.agent_config.get("guest_id") != principal.guest_id:
        return False

    return True


def public_chat_identity_matches(task: "Task", principal: InteractionPrincipal) -> bool:
    """Just the guest-identity slice of
    ``task_is_owned_by_public_principal``'s conjunction, exposed separately
    so a caller can choose which of its own two 403 messages to raise when
    the full conjunction fails.

    All three wired entry points collapse a ``guest_id`` mismatch into the
    same detail text as "task does not exist" -- an anti-enumeration
    property the pre-existing code documented on its own now-inlined guest
    gate ("a distinguishable guest-mismatch message would tell a probing
    visitor which task ids exist") -- while every other ownership conjunct
    (channel binding, config shape, auth_mode, entity binding) gets a
    distinguishable "widget/share is unavailable" message. A caller uses
    this function to pick between the two without re-deriving the identity
    check inline: when this returns ``False``, raise the not-found-shaped
    message; when it returns ``True`` but
    ``task_is_owned_by_public_principal`` still returned ``False``, raise
    the unavailable-shaped message.

    Deliberately does not also check ``Task.user_id`` -- see
    ``task_is_owned_by_public_principal``'s docstring for why ownership by
    user stays a caller-side SQL filter, never a post-load attribute read,
    across every function this predicate is wired into.

    **Identity mismatch takes precedence over every other criterion, not
    just entity binding.** This function is checked first at each of the
    three wired call sites; when it returns ``False`` the caller raises the
    not-found-shaped message immediately, without ever asking whether
    channel binding, config shape, auth_mode, or entity binding also
    failed. Concretely: for *any* input where ``guest_id`` does not match,
    the caller now gets the not-found message, regardless of what else
    about that input is also wrong.

    This is a deliberate narrowing of the pre-existing behavior, not a
    byte-identical extraction of it. Each of the three pre-existing
    functions checked identity at a different point in its own chain of
    early returns -- widget-workforce checked it before entity binding,
    both share-side functions checked entity binding (and, for
    share-agent, two separate binding checks) before it -- so for a
    combined-failure input (identity wrong *and* one or more of the other
    criteria also wrong), the pre-existing code could return either
    message depending on which check happened to run first for that
    specific function. Routing every combined failure through identity
    first collapses that per-function variance into one rule. Verified
    against the full input space per entry point: 55 = 1 (a non-``dict``
    ``agent_config`` collapses every other input into a single cell:
    both this predicate and the identity check return ``False`` on a
    non-``dict`` config, so no other axis can change the decision or
    the message) + 2 x 3 x 3 x 3
    (``channel_id``'s two values, crossed with the three-valued
    ``auth_mode``, entity-binding, and ``guest_id`` judgments). This
    enumeration does not vary the share-agent direction's row-level
    ``agent_id`` half -- that value is pinned by the SQL ``WHERE`` filter
    at its entry point before this predicate ever runs, per the
    share-agent bullet above. A majority of the combined-failure cells --
    on the order of 31-35 of the 55 enumerated input cells (54 of which
    deny), per entry point -- now return the not-found text where the
    pre-existing, per-function order would have returned the unavailable
    text for that same input. The allow/deny decision itself is unchanged
    for every cell; only which of the two denial messages a combined failure gets
    can differ, and only in the direction of revealing less: the
    not-found message is the one that tells a probing visitor nothing
    about which criterion failed, so this shift narrows the information a
    denied response leaks, it never widens it. A single-criterion failure
    (identity wrong and nothing else, or something else wrong and identity
    fine) is unaffected -- both messages are already pinned by the
    single-cause tests, and this change only touches cells where more than
    one criterion fails at once.
    """

    if not principal.guest_id:
        return False
    if not isinstance(task.agent_config, dict):
        return False
    return task.agent_config.get("guest_id") == principal.guest_id


# ---------------------------------------------------------------------------
# RespondOutcome: the answer-side discriminated union. Types only in this
# delivery -- respond() itself, and every branch that could actually
# produce most of these variants, lands with a later change (see the module
# docstring). Defining the full union now, rather than growing it
# incrementally alongside respond()'s call body, is what lets the reason
# vocabulary be pinned once and the counting below be a real guard instead
# of a moving target.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InteractionResponseReceipt:
    """Everything a caller needs to know about a successful answer, taken
    before commit from an explicit ``RETURNING``/``SELECT`` -- never from a
    lazy ORM relationship read after commit, which could re-issue SQL
    against a session whose transaction has already ended. Appears only on
    ``RespondAccepted`` and ``RespondReplayed``; every other outcome carries
    no receipt because no answer became durable.

    Field-for-field, this mirrors ``StagedInteractionRequest``'s and
    ``StagedTaskCommand``'s own after-flush value objects: a name that
    marks "already durable", not merely "staged in this call's session".
    """

    interaction_id: int
    task_id: int
    run_id: str
    status: str  # "answered"
    responded_at: "datetime"
    responder_identity: str
    idempotency_key: str
    command_db_id: int
    task_state_version: int
    task_control_state: str


@dataclass(frozen=True)
class RespondAccepted:
    receipt: InteractionResponseReceipt


@dataclass(frozen=True)
class RespondValidationRejected:
    reason: str


@dataclass(frozen=True)
class RespondUnauthorized:
    reason: str


@dataclass(frozen=True)
class RespondUnavailable:
    reason: str


@dataclass(frozen=True)
class RespondReplayed:
    receipt: InteractionResponseReceipt


@dataclass(frozen=True)
class RespondConflict:
    reason: str


@dataclass(frozen=True)
class RespondStale:
    reason: str


@dataclass(frozen=True)
class RespondOutcomeUnknown:
    """A commit whose acknowledgment was ambiguous, reconciled against the
    durable graph, and still unresolved after every reconciliation attempt.
    Not an exception this service lets escape -- a stable, typed result a
    caller can act on (e.g. surface "we could not confirm this went
    through" rather than crash)."""


RespondOutcome = (
    RespondAccepted
    | RespondValidationRejected
    | RespondUnauthorized
    | RespondUnavailable
    | RespondReplayed
    | RespondConflict
    | RespondStale
    | RespondOutcomeUnknown
)

# The (outcome type, reason) pairs RespondOutcome can produce once
# respond() is implemented, keyed by outcome class name. ``None`` in the
# reason set stands for the reason-less variants (Accepted / Replayed /
# OutcomeUnknown carry no reason code). This is the vocabulary guard: it
# proves the reason word list stays closed at exactly the pairs enumerated
# here, nothing more -- it does NOT prove every pair has a test written
# against it (several reasons are reachable from more than one triggering
# condition, e.g. every "A" row of the design's failure matrix collapses to
# the single (Unauthorized, not_task_principal) pair here; a guard over
# this dict cannot and does not distinguish which condition produced a
# given pair). Do not read "count matches" as "coverage complete".
RESPOND_OUTCOME_REASON_VOCABULARY: dict[str, frozenset[str | None]] = {
    "RespondAccepted": frozenset({None}),
    "RespondValidationRejected": frozenset(
        {
            "unknown_kind",
            "unknown_protocol_version",
            "malformed_idempotency_key",
            "invalid_values",
            "kind_version_mismatch",
        }
    ),
    "RespondUnauthorized": frozenset({"not_task_principal"}),
    "RespondUnavailable": frozenset(
        {"task_missing", "interaction_missing", "checkpoint_unavailable"}
    ),
    "RespondReplayed": frozenset({None}),
    "RespondConflict": frozenset({"already_answered", "idempotency_key_reused"}),
    "RespondStale": frozenset(
        {
            "expired",
            "run_superseded",
            "answered_via_chat",
            "run_ended",
            "foreign_run",
            "state_version_advanced",
            "anchor_dangling",
        }
    ),
    "RespondOutcomeUnknown": frozenset({None}),
}


# ---------------------------------------------------------------------------
# CreateOutcome: the create() seam's own discriminated union. Same family,
# same style as RespondOutcome, but a separate set of classes -- the two
# unions are not reused between each other even where a reason string
# happens to be spelled the same way (e.g. "not_task_principal" appears in
# both vocabularies below because both seams reuse the shared ownership
# predicate's verdict, not because the two outcome types share a base
# class).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CreatedInteractionReceipt:
    """Same family as ``InteractionResponseReceipt``: taken before commit,
    never from a lazy relationship read after it. Produced only once
    ``CreateCreated`` becomes producible -- not in this delivery (see
    ``create()``'s own docstring)."""

    interaction_id: int
    task_id: int
    run_id: str
    active_slot: int | None
    protocol_version: int
    request_idempotency_key: str
    expires_at: "datetime"


@dataclass(frozen=True)
class CreateCreated:
    receipt: CreatedInteractionReceipt


@dataclass(frozen=True)
class CreateValidationRejected:
    reason: str


@dataclass(frozen=True)
class CreateUnauthorized:
    reason: str


@dataclass(frozen=True)
class CreateUnavailable:
    reason: str


@dataclass(frozen=True)
class CreateConflict:
    reason: str


@dataclass(frozen=True)
class CreateStale:
    reason: str


@dataclass(frozen=True)
class CreateNotWired:
    """A well-formed, authorized create() call has nothing to report but
    that fact: this seam validates and returns, it does not stage a row.

    Retired by the change that fills this seam's call
    body. Until then a well-formed, authorized request has nothing to
    report but that fact: this seam validates and returns, it does not
    stage. That later change deletes this variant and its reason constant
    together; the vocabulary guard then asserts one fewer pair, which is
    what makes the deletion impossible to forget.
    """

    reason: str  # always "seam_not_wired" in this delivery


CreateOutcome = (
    CreateCreated
    | CreateValidationRejected
    | CreateUnauthorized
    | CreateUnavailable
    | CreateConflict
    | CreateStale
    | CreateNotWired
)

# The reason word list is defined once, for both delivery periods, because
# it is the same closed vocabulary either way -- only which pairs are
# *producible* changes (see CREATE_OUTCOME_PRODUCIBLE_REASONS below). 13
# reasons total (the "Created" / None pair is not a reason string and is
# counted separately). Do not update this number by recounting the set
# literal below -- it is pinned as part of the vocabulary's contract.
CREATE_OUTCOME_REASON_WORDS: frozenset[str] = frozenset(
    {
        "unknown_kind",
        "unknown_protocol_version",
        "malformed_idempotency_key",
        "invalid_values",
        "not_task_principal",
        "task_missing",
        "checkpoint_unavailable",
        "anchor_run_mismatch",
        "slot_taken",
        "idempotency_key_reused",
        "anchor_dangling",
        "run_ended",
        "seam_not_wired",
    }
)

# The (outcome type, reason) pairs this delivery's create() can actually
# produce today: validation and authorization run, but the seam never
# stages a row, so nothing past those two categories -- plus NotWired
# itself -- is reachable. 7 pairs. Once a later change fills create()'s
# call body, this dict is replaced by one covering all 13 reasons (minus
# seam_not_wired, which is deleted along with CreateNotWired) -- not
# extended in place, so the guard's expected count changes atomically with
# the variant it is pinned to.
CREATE_OUTCOME_REASON_VOCABULARY: dict[str, frozenset[str]] = {
    "CreateValidationRejected": frozenset(
        {
            "unknown_kind",
            "unknown_protocol_version",
            "malformed_idempotency_key",
            "invalid_values",
        }
    ),
    "CreateUnauthorized": frozenset({"not_task_principal"}),
    "CreateUnavailable": frozenset({"task_missing"}),
    "CreateNotWired": frozenset({"seam_not_wired"}),
}


# ---------------------------------------------------------------------------
# v1 request_payload construct/parse function pair. The shape is
# ``AskUserQuestionArgs`` (``core/tools/adapters/vibe/ask_user_tool.py``),
# imported by name rather than redeclared, so the two cannot drift the way
# a hand-written mirror would. ``create()`` below uses the parse half to
# validate an incoming envelope's ``values``; the future write path is the
# intended importer of the construct half, so that the
# payload the write path stages and the payload the read path
# (``materialize_compatibility_view``) decodes come from the same function
# pair, not two independently maintained copies.
# ---------------------------------------------------------------------------


def parse_v1_request_payload(values: Any) -> AskUserQuestionArgs:
    """Validate ``values`` against the v1 request_payload contract.

    Raises ``pydantic.ValidationError`` on any shape mismatch -- the type,
    not a boolean or an outcome, because both of this function's callers
    need the distinction between "validation failed" and "validation
    infrastructure failed" that only an exception type preserves, and each
    translates it into its own outcome shape at its own call site.
    """

    return AskUserQuestionArgs.model_validate(values)


def build_v1_request_payload(parsed: AskUserQuestionArgs) -> dict[str, Any]:
    """Render a validated ``AskUserQuestionArgs`` instance into the exact
    JSON-shaped dict ``stage_interaction_request``'s ``request_payload``
    parameter expects.

    Raises ``ValueError`` if the rendered dict is not JSON-serializable
    with ``allow_nan=False`` -- the identical probe
    ``stage_interaction_request`` (``task_interaction_staging.py``) runs on
    ``request_payload`` before its own INSERT, for the identical reason:
    pydantic's ``float`` field accepts ``float('nan')`` /
    ``float('inf')`` by default (``InteractionArg.default_value`` is one
    such field), and the default ``json.dumps`` would render either as the
    bare, non-standard JSON tokens ``NaN`` / ``Infinity`` instead of
    raising. Closing this probe here means a payload this function
    produces can never be the thing that trips staging's own probe later
    -- the two are meant to agree because they are validating the same
    contract from opposite ends of the same pipe, not because one copies
    the other.
    """

    payload = parsed.model_dump(mode="json")
    try:
        json.dumps(payload, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"request_payload is not JSON-serializable: {exc}") from exc
    return payload


# TTL policy interval for a create() envelope's optional ttl_seconds
# override. Enforcing the bound here in the facade is deliberate: clamping
# silently would be fail-open, and is explicitly rejected in favor of an
# outright validation failure. The concrete numbers are not pinned by
# anything: no existing config or constant anywhere in this codebase
# defines an interaction TTL policy today. The two bounds below are this
# delivery's own placeholder, not a fact recovered from source, logs, or
# the database -- flagged here as a value that needs an explicit policy
# decision, not a discovered one.
_MIN_INTERACTION_TTL_SECONDS = 60
_MAX_INTERACTION_TTL_SECONDS = 7 * 24 * 3600


@dataclass(frozen=True)
class CreateInteractionEnvelope:
    """The caller-supplied intent for ``create()``: what interaction to
    publish, not yet validated. Every field is checked by ``create()``'s
    validation step before anything else runs; none of them are trusted
    as-is."""

    kind: str
    protocol_version: int
    request_idempotency_key: str
    values: Any
    ttl_seconds: float | int | None = None


def create(
    db: "Session",
    *,
    task_id: int,
    principal: InteractionPrincipal,
    envelope: CreateInteractionEnvelope,
) -> CreateOutcome:
    """Typed seam for publishing a native clarification interaction.

    The body validates the envelope and returns a typed outcome. It does
    NOT call ``stage_interaction_request``: that call is what retires the
    zero-production-caller gate in
    ``tests/web/services/test_interaction_staging_production_gate.py``, and
    that gate retires only for the change that wires all three finalizers,
    adds the Task-side protocol marker, and switches the read surface
    together -- i.e. the wiring change that supplies this seam's call body,
    not this one.

    Validation order, each step short-circuiting on the first failure:
    ``kind`` against ``str`` first (a non-string value -- a ``list``, a
    ``dict`` -- is rejected before the ``in _KIND_VOCABULARY`` membership
    test ever runs, not caught as a side effect of the ``TypeError:
    unhashable type`` that test would otherwise raise for an unhashable
    value) and then against the v1 vocabulary, ``protocol_version``
    against ``int`` first excluding ``bool`` (``True`` and ``1.0`` both
    equal ``1`` in Python, so the type check has to run before the
    version comparison, not instead of it) and then against the v1
    version, ``request_idempotency_key`` against ``str`` first (a
    non-string value -- ``None``, an ``int``, ``bytes`` -- is rejected
    before the normalizer is ever called, not caught as a side effect of
    whatever ``TypeError``/``AttributeError`` a non-string would raise
    inside it) and then ``COMMAND_ID_PATTERN`` (via
    ``task_command_transport``'s own
    normalizer, not a copy of its regex), ``values`` against the v1
    ``request_payload`` contract (shape, via ``parse_v1_request_payload``,
    then JSON-serializability with ``allow_nan=False``, via
    ``build_v1_request_payload`` -- a shape-valid payload carrying a
    NaN/Infinity ``default_value`` fails the second check and is rejected
    the same way a shape failure is), and an optional ``ttl_seconds``
    against this facade's policy interval -- out of range is a rejection,
    never a silent clamp. Authorization runs only after every one of those
    passes,
    and only against a task this call itself loads by id: a ``"user"``
    principal must own the task or be an admin; a ``"guest"`` principal is
    checked with the same shared ownership predicate ``respond()`` will
    reuse (``task_is_owned_by_public_principal``), not a re-derived
    conjunction; a principal whose ``kind`` is neither ``"user"`` nor
    ``"guest"`` is always unauthorized -- there is no third branch that
    defaults to allow. A malformed principal that populates zero or more
    than one of the guest entity-binding fields makes the ownership
    predicate raise ``ValueError``; this function catches only that one
    exception type from that one call and treats it as unauthorized, the
    same fail-closed-on-a-malformed-caller behavior the predicate itself
    documents. A task that does not exist at all is ``Unavailable``, not
    ``Unauthorized`` -- that branch is reached before the ownership check
    even runs, because there is no row to check ownership against.

    ``origin`` is deliberately not part of this envelope or this
    validation step in this delivery: the reason vocabulary deliberately
    has no origin-related entry, and ``stage_interaction_request``
    (which this seam does not call) already validates it against the
    model's public vocabulary once a production write path calls it. Adding an
    origin check here now would validate a field this seam never uses for
    anything.

    This seam does not consult the native-publication gate
    (``interaction_rollout.evaluate_native_publication``): that gate's own
    docstring scopes it to "a finalizer is about to transition a task into
    WAITING_FOR_USER", and this seam is not a finalizer -- gating belongs to
    the finalizer layer that will supply this seam's call body, not to the
    validation-and-authorization facade in front of it.

    Of the staging module's nine exception classes, three
    (``InteractionAttemptMismatch``, ``InteractionHandoffMisuse``,
    ``InteractionOriginUnknown``) are raised only from inside
    ``_InteractionHandoff``'s own call path and are therefore unreachable
    from this function, which never enters that context manager -- this
    function has no call to any of the staging module's exception-raising
    code at all in this delivery, since it never calls
    ``stage_interaction_request``.
    """

    if not isinstance(envelope.kind, str) or envelope.kind not in _KIND_VOCABULARY:
        return CreateValidationRejected(reason="unknown_kind")
    if (
        not isinstance(envelope.protocol_version, int)
        or isinstance(envelope.protocol_version, bool)
        or envelope.protocol_version != INTERACTION_PROTOCOL_VERSION
    ):
        return CreateValidationRejected(reason="unknown_protocol_version")
    if not isinstance(envelope.request_idempotency_key, str):
        return CreateValidationRejected(reason="malformed_idempotency_key")
    try:
        _normalize_command_id(envelope.request_idempotency_key)
    except ValueError:
        return CreateValidationRejected(reason="malformed_idempotency_key")
    try:
        parsed_values = parse_v1_request_payload(envelope.values)
    except _PydanticValidationError:
        return CreateValidationRejected(reason="invalid_values")
    try:
        build_v1_request_payload(parsed_values)
    except ValueError:
        # e.g. a NaN/Infinity default_value: shape-valid per
        # AskUserQuestionArgs, but not JSON-serializable with
        # allow_nan=False -- see build_v1_request_payload's own docstring.
        return CreateValidationRejected(reason="invalid_values")
    if envelope.ttl_seconds is not None:
        if isinstance(envelope.ttl_seconds, bool) or not isinstance(
            envelope.ttl_seconds, (int, float)
        ):
            return CreateValidationRejected(reason="invalid_values")
        if not (
            _MIN_INTERACTION_TTL_SECONDS
            <= envelope.ttl_seconds
            <= _MAX_INTERACTION_TTL_SECONDS
        ):
            return CreateValidationRejected(reason="invalid_values")

    task = db.query(Task).filter(Task.id == task_id).first()
    if task is None:
        return CreateUnavailable(reason="task_missing")

    if principal.kind == "user":
        authorized = principal.is_admin or (
            principal.user_id is not None and task.user_id == principal.user_id
        )
    elif principal.kind == "guest":
        try:
            authorized = task_is_owned_by_public_principal(task, principal)
        except ValueError:
            authorized = False
    else:
        authorized = False
    if not authorized:
        return CreateUnauthorized(reason="not_task_principal")

    return CreateNotWired(reason="seam_not_wired")


# ---------------------------------------------------------------------------
# The shared active-row predicate (list_active()'s run scoping and the
# compatibility view's T2/T3 read share the same four fields the design
# pins together; the future answer fence and write-side reclaim predicate
# are meant to import this too, once they land -- see the docstring below
# for why all of them must move as one).
# ---------------------------------------------------------------------------


def _active_native_row_criteria() -> tuple[Any, ...]:
    """The four-field predicate for "this task's one live interaction row",
    excluding the ``task_id`` equality itself (every caller already scopes
    by task_id its own way -- a join column here, a plain filter there).

    ``status == "active"`` and ``active_slot IS NOT NULL`` are the row's own
    lifecycle state; ``TaskInteractionRequest.run_id == Task.run_id`` is
    what makes a *stale* active row (one staged in a run the task has since
    moved on from) invisible to every reader -- a query using this
    predicate must join against ``Task`` on ``task_id`` for that comparison
    to resolve. Deliberately does **not** include ``Task.status`` -- "is the
    task currently WAITING_FOR_USER" is a separate concern the future
    answer fence adds on top of this predicate, not a part of "which row is
    this task's live one".

    This predicate is meant to be imported by every future caller that
    needs the same notion of "the live row" -- the answer fence and the
    write-side reclaim statement both land later, and the design commits
    all of this predicate's callers (today: ``list_active()`` and this
    module's own ``_active_native_row()``; later: the fence and reclaim) to
    changing together if the predicate ever does, specifically so it cannot
    drift into two different definitions of "active" across the read and
    write sides. ``get()`` is deliberately **not** one of those callers --
    it fetches one row by id, scoped only by ``task_id``, with no
    active-row filtering at all; see its own docstring.
    """

    return (
        TaskInteractionRequest.status == "active",
        TaskInteractionRequest.active_slot.isnot(None),
        TaskInteractionRequest.run_id == Task.run_id,
    )


def get(
    db: "Session", *, task_id: int, interaction_id: int
) -> TaskInteractionRequest | None:
    """Fetch one interaction request row by id, scoped to the given task --
    fetching a resource by id always carries an ownership predicate in the
    same query, never a bare id lookup a caller trusts on its own."""

    return (
        db.query(TaskInteractionRequest)
        .filter(
            TaskInteractionRequest.id == interaction_id,
            TaskInteractionRequest.task_id == task_id,
        )
        .first()
    )


def list_active(db: "Session", *, task_id: int) -> list[TaskInteractionRequest]:
    """The ``list()`` deliverable's get-list seam. Named ``list_active``,
    not the bare ``list``, because this module annotates other functions'
    return types with the builtin ``list[...]`` generic throughout, and a
    module-level ``list`` name shadows that builtin for every annotation
    resolved after its definition, not just call sites.

    Every row for this task whose (status, active_slot, run_id) conjunction
    matches the shared active-row predicate. At most one row today
    (``uq_task_interaction_active_slot`` caps a task at one active slot),
    but this accessor's contract is "the live rows for this task", not "the
    one active row" -- answered/terminated history is out of scope here."""

    return (
        db.query(TaskInteractionRequest)
        .join(Task, Task.id == TaskInteractionRequest.task_id)
        .filter(
            TaskInteractionRequest.task_id == task_id,
            *_active_native_row_criteria(),
        )
        .all()
    )


def _active_native_row(db: "Session", task_id: int) -> TaskInteractionRequest | None:
    return (
        db.query(TaskInteractionRequest)
        .join(Task, Task.id == TaskInteractionRequest.task_id)
        .filter(
            TaskInteractionRequest.task_id == task_id,
            *_active_native_row_criteria(),
        )
        .first()
    )


# ---------------------------------------------------------------------------
# Read-direction anchor resolution + the three-tier compatibility view.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _AnchorUnresolved:
    reason: str  # "anchor_dangling" | "checkpoint_unavailable"


def _resolve_read_direction_anchor(
    db: "Session", row: TaskInteractionRequest
) -> _AnchorUnresolved | None:
    """Resolve an active interaction row's resume anchor for the read
    direction: does ``row.resume_trace_event_id`` still point at a
    structurally valid checkpoint row -- the right task, event type,
    checkpoint type, run partition, and (when present) execution identity?
    Returns ``None`` on success, or an ``_AnchorUnresolved`` naming which of
    the two outcomes applies otherwise.

    This validates the row's *identity*, not its *payload*: unlike
    ``trace_handlers``, which also calls ``decode_trace_event_data`` and
    can fail there too (registering ``CHECKPOINT_DECODE_FALLBACK``), this
    resolver never attempts to decode ``trace_row.data`` beyond reading the
    handful of fields the validity judgment itself needs. A row that
    passes this resolver is confirmed to be the legitimate checkpoint the
    interaction row's anchor names; whether its full payload would also
    decode cleanly is a question this function does not ask, because
    nothing downstream of it in this delivery needs that answer -- the
    native projection this resolver feeds reads ``request_payload`` on the
    interaction row itself, never the checkpoint row's own ``data``.

    Deliberate divergences from trace_handlers' anchor path
    (``api/trace_handlers.py``'s own by-primary-key resolver), listed
    because the two are close enough in shape that "align them" is the
    obvious next edit for a reader who has not seen this list:

    - The row-validity judgment (``task_id``, ``event_type``, ``build_id``,
      ``checkpoint_type in READABLE_CHECKPOINT_TYPES``, run partition, and
      -- when the checkpoint row carries one -- execution identity) is
      copied deliberately: this set must agree with trace_handlers' own,
      because both are answering the same question, "is this a legitimate
      checkpoint row", from different directions. (A shared, stateless
      predicate the two resolvers could both import is a real
      simplification, left as a follow-up, not done here -- see the
      module docstring's delivered-here accounting.)
    - ``CHECKPOINT_LOAD_UNAVAILABLE`` is registered exactly the way
      trace_handlers registers it: a read failure is a read failure on
      either side.
    - ``CHECKPOINT_PK_ANCHOR_DANGLING``'s registration surface is
      deliberately *wider* here than in trace_handlers: trace_handlers only
      registers it when the pointer names a row that does not exist, and
      raises (unregistered) when a row exists but fails validation.
      trace_handlers has a second candidate set to fall back to (the legacy
      scan) and treats "this specific row is bad" as different from "there
      is truly nothing to find"; this resolver has no second candidate set
      -- a pointer that resolves to an invalid row and a pointer that
      resolves to no row at all are the same fact from here (this answer
      cannot be recovered), so both register the one signal.
    - **This resolver never calls ``clear_degradation``.** Clearing
      ``CHECKPOINT_PK_ANCHOR_DANGLING`` is process-wide and coarse by
      design (trace_handlers' own comment: "one healthy task's read clears
      another task's dangling signal"); trace_handlers clears it before
      every read because it runs at high frequency across many tasks and
      that coarseness pays for itself. This resolver runs far less often,
      scoped to one task's one active row -- clearing here would erase a
      signal trace_handlers (or an earlier call here, for a different
      task) just registered, with no comparable frequency to earn back the
      false-clear rate. The clear right belongs to the read surface that
      runs at that frequency, not to every caller that happens to touch the
      same registry. This is a fact about ``CHECKPOINT_PK_ANCHOR_DANGLING``
      and ``CHECKPOINT_LOAD_UNAVAILABLE`` specifically, not a claim that
      every degradation signal in the registry has one exclusive clearer --
      ``INTERACTION_HANDOFF_DEGRADED`` and
      ``INTERACTION_RUN_PARTITION_MISMATCH_DEGRADED`` are a separate pair
      with no clearer at all yet, and this module does not touch them.
    - This resolver never falls back to a legacy scan on its own -- a
      resolution failure is reported as "unanswerable", not silently
      retried against a different candidate set. See
      ``materialize_compatibility_view``'s own docstring for why folding
      this into "no active row" and retrying legacy is prohibited outright.

    A further divergence, this one against the write-direction ("finalizer
    side") anchor resolver a later change supplies rather than against
    trace_handlers: that resolver treats a legacy checkpoint type as "no
    anchor available" and never stages an active row anchored to one at
    all. This resolver, by contrast, accepts every member of
    ``READABLE_CHECKPOINT_TYPES`` -- the current type and the legacy ones
    -- exactly as trace_handlers does. The two disagreeing is not a bug to
    reconcile: because the write side never produces an active row
    anchored to a legacy-type checkpoint, this resolver's broader
    acceptance of legacy types is unreachable in production regardless --
    there is no row for it to ever apply to. It stays broader than
    strictly necessary only because narrowing it here would make this
    resolver's row-validity judgment diverge from trace_handlers' for no
    reachable gain, which is the exact kind of drift the shared-judgment
    bullet above exists to prevent. Separately, if the write-direction
    resolver ever classifies its own dangling-pointer case, it is expected
    to register its own signal at its own call site, not
    ``CHECKPOINT_PK_ANCHOR_DANGLING`` -- that constant's registration in
    this module describes only the read direction; the two sides do not
    share a signal budget any more than they share a resolver.
    """

    if row.resume_trace_event_id is None:
        register_degradation(
            CHECKPOINT_PK_ANCHOR_DANGLING,
            f"task {row.task_id}: active interaction {row.id} has no resume anchor",
        )
        return _AnchorUnresolved(reason="anchor_dangling")

    try:
        trace_row = db.get(TraceEvent, row.resume_trace_event_id)
    except Exception:
        register_degradation(
            CHECKPOINT_LOAD_UNAVAILABLE,
            f"task {row.task_id}: interaction {row.id} anchor row fetch failed",
        )
        return _AnchorUnresolved(reason="checkpoint_unavailable")

    if trace_row is None:
        register_degradation(
            CHECKPOINT_PK_ANCHOR_DANGLING,
            f"task {row.task_id}: interaction {row.id} anchor "
            f"{row.resume_trace_event_id} has no matching trace_events row",
        )
        return _AnchorUnresolved(reason="anchor_dangling")

    row_data: dict[str, Any] = (
        trace_row.data if isinstance(trace_row.data, dict) else {}
    )
    run_field = row_data.get(TASK_RUN_ID_TRACE_FIELD)
    partition_matches = run_field == row.resume_run_partition
    row_execution_id = checkpoint_execution_id(row_data)
    execution_matches = (
        not row_execution_id or row_execution_id == row.resume_execution_id
    )
    if (
        trace_row.task_id != row.task_id
        or trace_row.event_type != str(CHECKPOINT_EVENT_TYPE)
        or trace_row.build_id is not None
        or row_data.get("checkpoint_type") not in READABLE_CHECKPOINT_TYPES
        or not partition_matches
        or not execution_matches
    ):
        register_degradation(
            CHECKPOINT_PK_ANCHOR_DANGLING,
            f"task {row.task_id}: interaction {row.id} anchor "
            f"{row.resume_trace_event_id} does not match the row it points at",
        )
        return _AnchorUnresolved(reason="anchor_dangling")

    return None


@dataclass(frozen=True)
class CompatibilityQuestionView:
    """The rich three-tier result ``materialize_compatibility_view``
    produces. Not the legacy ``(question, interactions)`` tuple
    ``chat_history_service.get_latest_waiting_question`` returns -- that
    lossy projection, and the four call sites that consume it, belong to
    a later adapter change (see the module docstring), which
    imports this type and projects it down. ``reason`` carries the reason
    code #1079's endpoint needs and the legacy tuple has no slot for; it is
    only set on the ``"unanswerable"`` tier.
    """

    tier: str  # "legacy" | "native" | "unanswerable"
    question: str | None
    interactions: list[dict[str, Any]] | None
    reason: str | None = None


def _legacy_view(db: "Session", task_id: int) -> CompatibilityQuestionView:
    question, interactions = get_latest_waiting_question(db, task_id)
    return CompatibilityQuestionView(
        tier="legacy", question=question, interactions=interactions
    )


def materialize_compatibility_view(
    db: "Session", task_id: int
) -> CompatibilityQuestionView:
    """The single rich implementation of "what is this waiting task's
    question", three-tiered:

    T1 (tier ``"legacy"``) -- the ``task_interaction_requests`` table does
    not exist yet, there is no active native row for this task, the active
    row's ``protocol_version`` is not one this reader recognizes, or its
    ``request_payload`` does not parse against the v1 shape: falls back,
    internally, to ``get_latest_waiting_question`` and returns exactly what
    that function would have, unconditionally. This tier's table-existence
    gate is not defensive decoration for a table that might never exist --
    ``interaction_rollout.py``'s own ``/ready`` gate treats "the service
    deploys before its own migration has run" as a real, accepted window,
    and this reader has to survive being called inside it.

    T2 (tier ``"native"``) -- an active native row exists and its resume
    anchor resolves (see ``_resolve_read_direction_anchor`` for exactly
    what "resolves" checks -- the checkpoint row's identity, not its
    decoded payload): the native projection, ``question`` and
    ``interactions`` decoded from ``request_payload`` itself, not the
    legacy transcript.

    T3 (tier ``"unanswerable"``) -- an active native row exists but its
    anchor does not resolve (see ``_resolve_read_direction_anchor``):
    ``question`` is still the row's own question text (there genuinely is
    one, waiting), ``interactions`` is ``None`` (it cannot currently be
    answered), and ``reason`` names why. **This tier must never fold into
    "no active row" and retry the T1 fallback** -- doing so would show the
    caller a legacy transcript question whose answer would land in a slot
    a native row has already claimed, producing a self-contradictory
    result. The one thing that makes this projection honest rather than a
    lie by omission: the compatibility seam that accepts continuation
    commands (landing with a later change) refuses a legacy-shaped answer
    on exactly this same condition -- an active native row with an
    unresolved anchor -- so "no controls shown" and "a free-text answer
    would be refused anyway" agree. If that seam's refusal is ever
    loosened, this tier's projection needs re-deciding alongside it, not
    independently.

    Consumers, and how much of this result they get: a later
    adapter (not written here) projects this down to the legacy
    ``(question, interactions)`` tuple for the four existing call sites,
    dropping ``reason`` -- lossy by design, not an oversight. #1079's own
    endpoint (not written here) is meant to consume this rich result
    directly, keeping ``reason`` for its own outcome classification.
    """

    if not interaction_requests_table_exists(db):
        return _legacy_view(db, task_id)

    row = _active_native_row(db, task_id)
    if row is None:
        return _legacy_view(db, task_id)
    if row.protocol_version != INTERACTION_PROTOCOL_VERSION:
        return _legacy_view(db, task_id)

    try:
        parsed = parse_v1_request_payload(row.request_payload)
    except _PydanticValidationError:
        return _legacy_view(db, task_id)

    unresolved = _resolve_read_direction_anchor(db, row)
    if unresolved is not None:
        return CompatibilityQuestionView(
            tier="unanswerable",
            question=parsed.message,
            interactions=None,
            reason=unresolved.reason,
        )

    return CompatibilityQuestionView(
        tier="native",
        question=parsed.message,
        interactions=[
            interaction.model_dump(mode="json") for interaction in parsed.interactions
        ],
    )
