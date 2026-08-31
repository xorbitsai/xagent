"""Typed lifecycle service for blocking task interaction requests: the
answer-side counterpart to ``task_interaction_staging.py``'s ask-side
primitive.

Not merged into ``task_interaction_staging.py``. That module's own
docstring pins its merge reason to a fact this module does not share: its
two entry points exist in one file because they share every exception type
and one nesting invariant that must never be split across a file boundary,
both live on the ask side, and both require the caller to already hold the
transaction they run inside. This module's ``respond()`` is the opposite
shape on every one of those points: it owns its own session end to end
(opens it, commits or rolls it back, retires it) and never nests inside a
caller's transaction -- the one savepoint it opens
(``db.begin_nested()``, around staging its own resume command) is its
own, not a caller's. ``create()``, by contrast, takes a caller-owned
session and only reads through it, never opening or committing one of its
own. Putting ``respond()`` in the same file as the staging primitive would
make that staging module's merge-reason docstring false. What this module
reuses from the staging module instead of duplicating is narrow and
one-directional: the private kind vocabulary (``_KIND_VOCABULARY``),
imported by name. Neither ``create()`` nor ``respond()`` calls
``stage_interaction_request`` and neither raises or catches any of that
module's nine exception classes, and this module's own read-direction
anchor resolver validates a real ``trace_events`` row against a stored
interaction row's fields -- a different check from the staging module's
``_validate_anchor_fields``, which validates an ``InteractionAnchor``
value object before an INSERT -- so it is not reused here either. See
``create()``'s own docstring for the exception-family accounting, and
``_resolve_read_direction_anchor``'s for the anchor-check distinction.

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

This module now delivers the full answer side: the ``InteractionPrincipal``
value object and the shared public-chat ownership predicate extracted from
``public_chat_access.py``; the ``RespondOutcome`` and ``CreateOutcome``
discriminated unions; the ``create()`` typed seam (validates and returns,
does not stage a row); ``get()``/``list_active()``; the three-tier
compatibility materialization view; the answer fence and its active-row
predicate; ``respond()``'s own call body (validation, authorization, the
idempotency pre-read, anchor resolution, the answer fence and its
zero-rowcount classification, the task control-state transition, staging
the resume command, and committing or reconciling an ambiguous
acknowledgment -- this delivery classifies why the fence missed,
classifies a staging conflict, and reconciles an ambiguous commit against
the durable graph, so ``RespondOutcomeUnknown`` is left reporting only a
commit whose acknowledgment was lost and whose reconciliation could not
confirm what landed); and the response-conflict counter
(``COUNTER_LIFECYCLE_RESPONSE_CONFLICT``).

Not delivered here, and named so a reader does not go looking for them in
this module: the compatibility seam that routes the existing resume
coordinator (``websocket.py``'s ``_handle_resume_task_unserialized``)
through ``_active_native_row_criteria()`` shipped as its own change in this
same series and has already merged -- the seam now reads through
``task_interaction_close.active_interaction_id_sync``, which imports
``_active_native_row_criteria`` from this module; ``websocket.py`` itself
no longer imports it directly -- but is not part of this
change. ``create()``'s own call body -- the write
that actually stages a row -- is not delivered here either; its own
zero-production-caller gate
(``tests/web/services/test_task_interaction_service_create_gate.py``)
watches both ``create()`` and ``respond()``: zero production code calls
either name today, and the change that wires a caller is the change that
takes the corresponding name out of the gated set. The gate gives
``create()``'s boundary a regression guard against import bindings, not
an absolute one -- its docstring lists what it cannot see (dynamic
access, alias/re-export chains, filename-stem exclusion, and code outside
the scanned package tree).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Literal, cast

import sqlalchemy as sa
from pydantic import ValidationError as _PydanticValidationError
from sqlalchemy.exc import IntegrityError

from ...core.agent.checkpoint import (
    CHECKPOINT_EVENT_TYPE,
    READABLE_CHECKPOINT_TYPES,
    checkpoint_execution_id,
)
from ...core.tools.adapters.vibe.ask_user_tool import AskUserQuestionArgs
from ...core.tools.adapters.vibe.interaction_types import INTERACTION_TYPES
from ..models.task import Task, TaskStatus, TaskStatusPredicate, TraceEvent
from ..models.task_command import TaskExecutionCommand
from ..models.task_interaction import (
    INTERACTION_PROTOCOL_VERSION,
    TaskInteractionRequest,
)
from .chat_history_service import get_latest_waiting_question
from .interaction_rollout import (
    COUNTER_COMPAT_READ_FALLBACK,
    COUNTER_LIFECYCLE_RESPONSE_CONFLICT,
    increment_counter,
)
from .ops_signals import (
    CHECKPOINT_LOAD_UNAVAILABLE,
    CHECKPOINT_PK_ANCHOR_DANGLING,
    INTERACTION_READ_PAYLOAD_UNREADABLE,
    INTERACTION_READ_PROTOCOL_UNRECOGNIZED,
    register_degradation,
)
from .task_command_transport import (
    TaskCommandConflictKind,
    TaskCommandKind,
    _canonical_payload,
    _matches_existing,
    _normalize_command_id,
    classify_task_command_conflict,
    notify_task_command_dispatcher,
    stage_task_command,
)
from .task_execution_controller import (
    TaskControlState,
    apply_task_control_transition,
)
from .task_interaction_schema import interaction_requests_table_exists
from .task_interaction_staging import _KIND_VOCABULARY
from .task_lease_service import TASK_RUN_ID_TRACE_FIELD

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

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
        """The ``responder_identity`` value for this principal.

        Raises ``ValueError`` for a principal this namespacing cannot
        describe, rather than rendering the gap into the string. There are
        two such gaps, and both used to produce something that looks like
        an identity and is not one: a missing id interpolated into the
        literal ``"user:None"`` / ``"guest:None"``, and an unrecognized
        ``kind`` falling through to the user branch and being recorded as a
        user.

        Nothing downstream stops either --
        ``ck_task_interaction_requests_responder_identity_nonempty``
        (``models/task_interaction.py``) only requires a non-empty string,
        and both of those are non-empty -- and this column is the one field
        this table's audit trail can rely on staying populated, so a value
        it cannot trust is worse here than a failure.

        The empty string is the same gap wearing a different shape: a
        ``guest_id`` of ``""`` used to render as the literal ``"guest:"``,
        another non-empty value the CHECK above lets through, and it names
        nobody either. The falsy test below is the same one the ownership
        predicates in this module already use for ``guest_id``. It stops
        at falsy and does not strip: the two token decoders that build
        these principals do not agree on whitespace, and the widget path
        admits a guest id that is only spaces, so stripping here would
        start rejecting principals that path produces today.

        ``ValueError``, not a typed rejection, because this is a pure
        function of the principal: reaching it with one this incomplete
        means the caller built the principal wrong, which is a programming
        error rather than a request that can be answered. It is what
        ``task_is_owned_by_public_principal`` already raises for a
        malformed guest principal. Both write-side facades reject such a
        principal at their authorization step, so no caller should be in a
        position to see this raise.
        """

        if self.kind == "guest":
            if not self.guest_id:
                raise ValueError("guest principal carries no guest_id")
            return f"guest:{self.guest_id}"
        if self.kind != "user":
            raise ValueError(f"principal kind {self.kind!r} has no identity namespace")
        if self.user_id is None:
            raise ValueError("user principal carries no user_id")
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
    predicate) has no JSON-level entity binding to mirror asymmetrically
    with the other three directions, and (separately) has no
    ``auth_mode`` check in the pre-existing code this predicate does not
    touch. Fixing either is a behavior change to a production
    authorization path and is out of scope for this change, which only
    extracts and adds a new, additively-called predicate.
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
# RespondOutcome: the answer-side discriminated union. respond() itself,
# and the call body that produces every one of these variants, are both
# delivered in this module (see the module docstring). Defining the full
# union up front, rather than growing it incrementally alongside that call
# body, is what lets the reason vocabulary be pinned once and the counting
# below be a real guard instead of a moving target.
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


# Each outcome that carries a reason declares its own closed word list as a
# ``Literal``, so the vocabulary is the type itself rather than a separate
# dict a reader has to trust stays in sync with the dataclasses below. A
# reason string outside its outcome's Literal is a static type error at
# every construction site; it is not a runtime check (dataclasses do not
# validate their field types at construction), the same limitation a
# runtime membership check against a dict would not have removed either,
# since nothing in this module ever constructed one of these outcomes with
# an unvalidated, caller-supplied reason string to begin with -- every
# reason literal below is written by this module's own code, at a call
# site a type checker already sees.
RespondValidationRejectedReason = Literal[
    "unknown_kind",
    "unknown_protocol_version",
    "malformed_idempotency_key",
    "invalid_values",
    "kind_version_mismatch",
]
RespondUnauthorizedReason = Literal["not_task_principal"]
RespondUnavailableReason = Literal[
    "task_missing", "interaction_missing", "checkpoint_unavailable"
]
RespondConflictReason = Literal["already_answered", "idempotency_key_reused"]
RespondStaleReason = Literal[
    "expired",
    "run_superseded",
    "answered_via_chat",
    "run_ended",
    "foreign_run",
    "anchor_dangling",
]


@dataclass(frozen=True)
class RespondAccepted:
    receipt: InteractionResponseReceipt


@dataclass(frozen=True)
class RespondValidationRejected:
    reason: RespondValidationRejectedReason


@dataclass(frozen=True)
class RespondUnauthorized:
    reason: RespondUnauthorizedReason


@dataclass(frozen=True)
class RespondUnavailable:
    reason: RespondUnavailableReason


@dataclass(frozen=True)
class RespondReplayed:
    receipt: InteractionResponseReceipt


@dataclass(frozen=True)
class RespondConflict:
    reason: RespondConflictReason


@dataclass(frozen=True)
class RespondStale:
    reason: RespondStaleReason


@dataclass(frozen=True)
class RespondOutcomeUnknown:
    """A call this function could not resolve to one of the specific
    outcomes above, for the one reason that still reaches it: committing
    raised, and the durable-graph reconciliation this build runs
    afterward could not confirm the write landed on every one of its
    three attempts (step 9) -- the acknowledgment could have been lost
    after the server applied the write, and reading the graph back did
    not settle it either way, so this build reports the ambiguity rather
    than guessing. Neither of the other two races arrives here any more.
    The answer fence's UPDATE matching zero rows (step 6) is *not* one of
    them: this build rereads the row and classifies every such miss into
    a specific outcome. Step 8's staging race is not one either: both of
    its doors are classified into ``Replayed`` or ``Conflict`` now. Not an
    exception this service lets escape -- a stable, typed result a caller
    can act on (e.g. surface "we could not confirm this went through"
    rather than crash).

    Retrying under the same idempotency key resolves every reason that
    still reaches this outcome, and that is now the whole set. A retry
    after an ambiguous commit finds the command this call staged and
    returns ``Replayed``. What a retry could never have clarified on its
    own is a fence miss -- a terminated, superseded, foreign-run, or
    foreign-owned row stays that way and misses again -- but that case no
    longer lands on this outcome: step 6 hands the caller the specific
    reason directly, and its own warning log records the reread row state
    (``active_slot``, ``terminal_reason``, ``run_id``,
    ``responder_identity``) that no outcome variant carries.
    """


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


# ---------------------------------------------------------------------------
# CreateOutcome: the create() seam's own discriminated union. Same family,
# same style as RespondOutcome, but two different mechanisms, and the
# asymmetry is intentional rather than unfinished cleanup. Respond's
# outcomes declare each reason as a ``Literal`` on the dataclass field, so
# the type is the vocabulary. Create's still carry ``reason: str`` beside
# the runtime dictionaries below, which record something a ``Literal``
# cannot: which reasons are *producible* in this delivery as opposed to
# merely declared, a distinction that exists only because create()'s call
# body is not delivered yet. The two vocabularies overlap in nine strings
# and are deliberately not shared -- ``"not_task_principal"`` appears in
# both because both seams reuse the shared ownership predicate's verdict,
# not because the two outcome types share a base class, and a change to
# one side's word list must never silently move the other's. When
# create()'s call body lands and the producible/declared distinction
# disappears with it, that side converts to the Literal mechanism and this
# note goes away.
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
#
# There is a second producer of this same shape that does NOT go through
# ``build_v1_request_payload``: ``build_clarification_payload``
# (``task_clarification_draft.py``) builds a v1 payload directly from a
# clarification draft, because it also has to filter and truncate that
# draft's free text before anything is staged. Its output must satisfy
# ``parse_v1_request_payload`` below -- that is a cross-module contract with
# no shared type to enforce it, pinned by the round-trip test named in that
# function's own docstring. Anything changed here about what v1 accepts has
# to be changed there in the same commit.
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


# The seven interaction types the render surface implements. The list
# itself lives on ``interaction_types.INTERACTION_TYPES``
# (``core/tools/adapters/vibe/``), an import-free module beside the tool
# whose argument schema carries the field; this is the write side's
# membership view of it. Importing rather than restating is what keeps the tool's
# JSON-Schema enum, the tool's own argument description and this
# admissibility set from drifting apart -- all three used to carry the
# same seven names independently.
#
# Deriving the set here does not narrow ``InteractionArg.type`` to a
# ``Literal``: that model is also the ``ask_user_question`` tool's
# argument schema, so narrowing it would narrow what the model itself is
# allowed to emit, which is a different decision from what this service is
# willing to persist.
#
# The seven are the same ones the frontend's own normalizer accepts
# (``normalizeInteractions``, ``frontend/src/contexts/app-context-chat.tsx``),
# which drops an item typed anything else before it reaches a renderer.
# That normalizer is not the only way in, though, so an unknown type is
# not simply invisible: the agent builder's chat
# (``frontend/src/components/build/agent-builder-chat.tsx``) takes the
# interactions off its own stream and hands them to the same renderer
# unnormalized, where an unrecognized type falls to
# ``clarification-form.tsx``'s ``default`` branch and shows the user a
# red "unsupported type" line. Rejecting the type here on the write side
# is what keeps either surface from having to.
_V1_INTERACTION_TYPES = frozenset(INTERACTION_TYPES)

# The three types whose whole purpose is picking from a supplied list, and
# the four that render their own control and have nothing to pick from.
# The split is the render surface's, not this module's: ``select_one``,
# ``select_multiple``, and ``action_cards`` iterate ``interaction.options``
# (``frontend/src/components/chat/clarification-form.tsx``), while
# ``confirm`` renders a switch, ``text_input`` a field, ``number_input`` a
# spinner, and ``file_upload`` a picker -- none of which read ``options``.
_V1_TYPES_REQUIRING_OPTIONS = frozenset(
    {"select_one", "select_multiple", "action_cards"}
)
_V1_TYPES_REJECTING_OPTIONS = _V1_INTERACTION_TYPES - _V1_TYPES_REQUIRING_OPTIONS


def validate_v1_write_payload(parsed: AskUserQuestionArgs) -> None:
    """Reject a shape-valid v1 payload that must not be persisted.

    Raises ``ValueError`` describing the first violation found; returns
    ``None`` when the payload may be written.

    Deliberately separate from ``parse_v1_request_payload`` rather than
    folded into it, because the two directions have different failure
    policies for the same payload and folding them would collapse both
    into one. The read direction meets these payloads as rows that are
    already persisted: a payload it cannot make sense of has to degrade to
    something the waiting user can still act on, so widening what it
    rejects turns readable-but-odd rows into unanswerable ones. The write
    direction meets them before anything is stored, where the only useful
    answer is to refuse -- a question naming an interaction type no
    renderer implements, or a select with nothing to select, reaches the
    user as a form they cannot complete and a run that can never be
    resumed. ``parse_v1_request_payload`` therefore keeps accepting
    exactly what it accepts today, and this runs on top of it on the write
    side only.

    ``build_clarification_payload`` (``task_clarification_draft.py``) is a
    second producer of this same shape and its output has to keep passing
    here, pinned by a test that feeds this function that builder's real
    output. Two of that builder's shapes are the reason for the boundaries
    drawn below. An empty ``interactions`` list is accepted: a
    ``send_message``-sourced draft never carries interactions, and an
    over-size form is deliberately dropped to ``[]`` rather than truncated
    to half a form -- both are questions with prose and no form, which the
    read surface renders. A blank ``message`` is rejected, and that costs
    the builder nothing: ``resolve_publishable_clarification`` already
    classifies a payload whose message is blank after filtering as
    ``NotApplicable("empty_question")`` and never offers it for writing.

    Every rule here refuses something that makes the *answer* wrong or
    unrecoverable, and nothing here refuses something that only makes the
    *rendering* worse. Three shapes are deliberately accepted for that
    reason, each of them a thing the render surface already handles and
    none of them changing what answer comes back:

    * A blank interaction-level ``label``. ``clarification-form.tsx``
      renders ``interaction.label || interaction.field``, so the field name
      stands in. Refusing it would also start refusing payloads the second
      producer really emits: ``_normalize_ask_user_interactions``
      (``react.py``), which every ``ask_user_question`` payload passes
      through, repairs a blank ``field`` and never touches ``label``, so a
      model that emits ``label=""`` reaches ``build_clarification_payload``
      with it -- and losing the whole question to the legacy transcript is
      a worse outcome than a label that reads as the field name.
    * ``accept=[]`` on a ``file_upload``. It renders as ``accept=""``,
      which is what the browser does with no restriction at all.
    * ``min``/``max`` on a type that does not render them. Only
      ``number_input`` reads the pair -- ``clarification-form.tsx`` passes
      ``min``/``max`` to the input in its ``number_input`` branch and
      nowhere else -- so on the other six they are an ignored hint, and the
      question still asks exactly what it asks. The ``min > max`` rule
      below is scoped to ``number_input`` for that same reason: on the type
      that reads the pair, an inverted range makes every answer invalid and
      the write is refused; on the six that ignore it, the pair never
      reaches the user at all, so an inverted one is a hint nobody reads
      rather than a question nobody can answer.

    Answers are out of scope here and in every other function in this
    module. Until the answer-side field schema lands (issue #1368),
    everything downstream of an interaction row must treat
    ``response_payload`` as unvalidated input: nothing checks a submitted
    answer against the ``InteractionArg.type`` / ``InteractionArg.field``
    definitions this function checks on the question side, so a
    malformed-but-dict-shaped answer is stored as submitted.
    """

    if not parsed.message.strip():
        raise ValueError("request_payload.message is blank")

    seen_fields: set[str] = set()
    for index, interaction in enumerate(parsed.interactions):
        where = f"request_payload.interactions[{index}]"
        if interaction.type not in _V1_INTERACTION_TYPES:
            raise ValueError(f"{where}.type {interaction.type!r} is not a v1 type")
        # A blank field is refused for the same reason a duplicated one is,
        # and not for the reason the rules above exist. The renderer copes
        # with both: clarification-form.tsx substitutes ``response_{index}``
        # for a blank or all-whitespace field and appends ``_{index}`` to a
        # repeat, so neither one produces a form the user cannot complete.
        # What neither substitution can fix is that the answer then arrives
        # under a key the persisted question never named. Today nothing
        # compares the two; the answer-side field schema (#1368) is what
        # will, and it will compare against what is stored here. Refusing
        # the write is the only point at which the stored key can still be
        # made to be the key the answer will carry.
        #
        # Two checks, not one: blank (including all-whitespace) and
        # surrounding whitespace on an otherwise non-blank field are both
        # refused, for the same key-integrity reason -- ``" a "`` never
        # equal-matches an answer keyed ``"a"``, so a stored field with
        # leading or trailing whitespace is exactly as unmatchable against
        # #1368 as a blank one. Stripped here only to test, not to
        # normalize what gets stored: the model-facing producer
        # (``_normalize_ask_user_interactions``, ``react.py``) already
        # trims a field -- against a table covering every code point
        # either Python's ``str.strip()`` or JavaScript's ``trim()``
        # treats as whitespace, including a leading BOM (U+FEFF), which
        # ``str.strip()`` alone does not -- and substitutes for a blank
        # one before it ever reaches this validator, so a well-formed
        # field arrives here pre-trimmed and this pair of checks costs
        # that producer nothing. These two checks themselves use plain
        # ``str.strip()``, though, not that table: they exist for
        # whatever reaches this write side some other way, and for a
        # field that does, a leading or trailing BOM would pass both of
        # them unnoticed even though it would not survive the producer's
        # own trim. Narrowing these two checks to match would mean
        # importing that same table here.
        #
        # Both react.py paths run that normalizer, so both arrive
        # pre-trimmed, and there the resemblance stops -- whoever wires the
        # first production writer is wiring one of two different producers.
        # The single-tool ``ask_user_question`` path calls the normalizer
        # and stops: it does not deduplicate, so two interactions the model
        # named the same field reach this validator unchanged and the
        # duplicate rule below refuses the write, costing that path the
        # whole question rather than one field. The multi-tool waiting path
        # (``_pause_for_tool_results``) runs its own dedup loop afterwards
        # across every waiting tool, appending ``_2``, ``_3`` to a repeated
        # base: it cannot trip that rule, and the field it hands over is
        # the renamed one, so the key stored for #1368 to match answers
        # against is not the key the tool asked under.
        if not interaction.field.strip():
            raise ValueError(f"{where}.field is blank")
        if interaction.field != interaction.field.strip():
            raise ValueError(f"{where}.field carries surrounding whitespace")
        if interaction.field in seen_fields:
            raise ValueError(f"{where}.field {interaction.field!r} is duplicated")
        seen_fields.add(interaction.field)
        if interaction.type in _V1_TYPES_REQUIRING_OPTIONS and not interaction.options:
            raise ValueError(f"{where} is a {interaction.type} carrying no options")
        if interaction.type in _V1_TYPES_REJECTING_OPTIONS and interaction.options:
            raise ValueError(f"{where} is a {interaction.type} carrying options")
        # Either half blank, not both: an option the user can see but not
        # submit, and one they can submit but not see, are both unusable.
        # The renderer's own filter (clarification-form.tsx) keeps an
        # option only when value and label are non-blank under
        # JavaScript's own ``trim()``. The model-facing producer
        # (``_normalize_ask_user_interactions``, react.py) keeps an option
        # only when both are non-blank under a wider table -- every code
        # point either Python's ``str.strip()`` or JavaScript's ``trim()``
        # treats as whitespace, a strict superset of what the renderer
        # checks. The two are not equivalent: a value like a single
        # ``"\x1c"`` (a control code point Python treats as whitespace but
        # JavaScript's ``trim()`` does not) survives the renderer's filter
        # but is dropped by the producer, so the producer is the stricter
        # of the two, not the looser one. This check here does not rely on
        # either of them: it is falsy, not trim-aware, and
        # ``InteractionOption.label``/``value`` (ask_user_tool.py) are
        # required ``str`` with no ``min_length`` or whitespace
        # constraint, so a whitespace-only label or value -- not just the
        # empty string -- reaches this line, and this line lets it
        # through. A payload that reaches this validator some way other
        # than the react.py producer could still carry one past it;
        # catching that here too would mean checking against the same
        # trim table the producer uses, which this validator does not
        # import today. An option dropped by the renderer's own filter
        # leaves a select the user cannot complete, or, if it was the
        # only one, a form with an empty control; refusing the write is
        # the only place that outcome can still be prevented for whatever
        # this check does catch.
        for option_index, option in enumerate(interaction.options or ()):
            if not option.label or not option.value:
                raise ValueError(
                    f"{where}.options[{option_index}] has a blank label or value"
                )
        # Values, not labels: the renderer resolves a submitted answer back
        # to an option by matching on ``value`` and taking the first hit
        # (clarification-form.tsx's ``options.find(o => o.value === value)``,
        # used by select_one, select_multiple and action_cards alike), so two
        # options sharing a value make the answer unable to say which of them
        # was chosen. Two options sharing a label are merely confusing to
        # look at; the answer still names exactly one of them.
        seen_option_values: set[str] = set()
        for option_index, option in enumerate(interaction.options or ()):
            if option.value in seen_option_values:
                raise ValueError(
                    f"{where}.options[{option_index}].value "
                    f"{option.value!r} is duplicated"
                )
            seen_option_values.add(option.value)
        # Scoped to ``number_input`` deliberately -- this function's
        # docstring carries the reason, under the third of the three
        # deliberately accepted shapes.
        if (
            interaction.type == "number_input"
            and interaction.min is not None
            and interaction.max is not None
            and interaction.min > interaction.max
        ):
            raise ValueError(f"{where} has min greater than max")


# The interval a create() envelope's optional ttl_seconds override must
# fall inside. Enforcing the bound here in the facade is deliberate:
# clamping silently would be fail-open, so an out-of-range override is an
# outright validation failure instead.
#
# These two bound what a caller may ask for. Neither is the TTL anything
# is published with: that is CLARIFICATION_REQUEST_TTL (24 hours,
# task_clarification_draft.py), the value the publication path adds to
# "now" to get a row's expires_at. The two are different quantities -- an
# interval and a value -- and are deliberately not unified; the value sits
# inside the interval, which is the only relationship between them that
# has to hold.
#
# No config or constant anywhere in this codebase defines an interaction
# TTL policy, so the numbers are decided here rather than derived: the
# widest interval whose ends both still mean something. The floor is a
# minute because a question the user cannot plausibly answer within the
# window is worse than no question, and because
# ck_task_interaction_requests_expiry_after_creation
# (models/task_interaction.py) requires expires_at > created_at -- any
# positive floor satisfies that constraint and a minute is the smallest
# one a person answering a question can use. The ceiling is a week
# because a row that effectively never expires is a row a reclaim job can
# never retire.
_MIN_INTERACTION_TTL_SECONDS = 60
_MAX_INTERACTION_TTL_SECONDS = 7 * 24 * 3600


@dataclass(frozen=True)
class CreateInteractionEnvelope:
    """The caller-supplied intent for ``create()``: what interaction to
    publish, not yet validated. Every field is checked by ``create()``'s
    validation step, which runs after that call has authorized the
    principal against the task; none of them are trusted as-is."""

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

    Authorization runs first, against a task this call itself loads. Only a
    caller that has been authorized against a real task row reaches the
    envelope checks below, so a rejection reason describing the payload
    shape is never returned to a caller that is not entitled to the task in
    the first place. Validation order within that step, each step
    short-circuiting on the first failure: ``kind`` against ``str`` first (a
    non-string value -- a ``list``, a
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
    the same way a shape failure is -- and then the write side's own
    admissibility rules, via ``validate_v1_write_payload``, which is where
    an unrenderable interaction type or a select with no options is
    refused), and an optional ``ttl_seconds``
    against this facade's policy interval -- out of range is a rejection,
    never a silent clamp.

    The load is owner-scoped for the two branches whose ownership includes
    a column-level term and id-only for the one that has none, and the
    difference is deliberate:

    - A non-admin ``"user"`` principal's ownership is one equality on a
      column, so it is a predicate on the lookup itself
      (``Task.user_id == principal.user_id``) rather than a Python
      comparison run after the row is already in hand. Such a principal
      never loads a row it does not own.
    - An admin ``"user"`` principal is authorized without owning the task,
      so there is no owner predicate to add; the lookup stays id-only.
      Authorized here is not allowed to write: ``respond()`` lets an admin
      clear the same step and then refuses it at the write point, on
      ``_answer_fence_task_predicate``'s ``Task.user_id`` term. This seam
      writes nothing and so has no such fence -- a writer wired behind it
      inherits an admin whose ownership nothing has checked.
    - A ``"guest"`` principal's ownership is split across the two. The
      column-level half is the same ``Task.user_id == principal.user_id``:
      a guest's owning user is the entity owner the guest is chatting
      through, and all three entry points that build a guest principal
      already load their task under ``Task.user_id ==
      access_context.user.id`` (``web/api/public_chat_access.py``).
      ``task_is_owned_by_public_principal`` deliberately excludes that term
      from its own conjunction and leaves it to whatever loads the task
      (see its docstring), so this lookup carries it. The rest of the
      guest conjunction reads the task's ``agent_config`` JSON, which
      cannot be compiled into this lookup's WHERE clause, and stays in the
      shared Python predicate ``respond()`` reuses rather than being
      re-derived here. Without the column-level half, a guest whose
      ``agent_config`` values happened to match would be authorized
      against a task belonging to another user; the answer side already
      refuses exactly that at its write point
      (``_answer_fence_task_predicate``'s ``Task.user_id`` term).
    - A ``"user"`` or ``"guest"`` principal carrying no ``user_id`` is
      unauthorized before the lookup is even built; the guard itself
      carries why it is placed there.

    A principal whose ``kind`` is neither ``"user"`` nor ``"guest"`` is
    always unauthorized -- there is no third branch that defaults to allow
    -- and is rejected before the lookup is built rather than at the end of
    the branch chain, because an unrecognized kind is not owner-scoped, so
    the lookup it would have reached is the id-only one that reports
    ``task_missing``.
    ``InteractionPrincipal.identity_string`` refuses the same principal on
    the audit side by raising ``ValueError``; the two are not redundant.
    This one decides whether a caller may act at all and answers with a
    typed outcome, while that one runs at the write point on a caller
    already authorized, where being unnameable is a programming error and
    not a rejection to report.

    A malformed principal that populates zero or more than one of the guest
    entity-binding fields makes the ownership predicate raise
    ``ValueError``; this function catches only that one exception type from
    that one call and treats it as unauthorized, the same
    fail-closed-on-a-malformed-caller behavior the predicate itself
    documents.

    Consequence of the owner-scoped lookup, and the reason it is stated
    here rather than left for a reader to derive from the SQL: for a
    non-admin ``"user"`` principal and for a guest, "this task does not
    exist" and "this task is not yours" are the same empty result set, and
    both return ``CreateUnauthorized(reason="not_task_principal")``.
    Neither principal can use this function to learn whether a ``task_id``
    exists. ``CreateUnavailable(reason="task_missing")`` stays reachable
    from the id-only admin branch, and that puts an obligation on whoever
    consumes these outcomes: exposing the distinction externally hands the
    requester a task-existence oracle, so an endpoint mapping this
    function's outcome onto an HTTP response has to collapse the two into
    one client-facing shape. The three existing public-chat entry points
    already do the equivalent one layer up: a task that does not exist and
    a task whose ``guest_id`` belongs to another visitor both produce the
    identical not-found-shaped 403. The same obligation covers the
    respond-side twin pair (``RespondUnavailable(reason="task_missing")``
    versus ``RespondUnauthorized(reason="not_task_principal")``), which
    stays distinguishable for every principal kind because ``respond()``
    loads its task by id under a lock -- and it covers every other outcome
    consumer built on this module, not only ``create()``'s own two
    variants.

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

    if principal.kind in ("user", "guest") and principal.user_id is None:
        # Rejected before the lookup, on every branch that carries a
        # user_id at all. An admin passing on the flag alone would reach
        # the write point with no identity to record as who acted. An
        # owner-scoped branch's predicate would be built from that same
        # absent id and render as ``Task.user_id IS NULL``, which rejects
        # only because ``Task.user_id`` is NOT NULL today -- an explicit
        # rejection here says what is meant instead of borrowing a schema
        # detail to mean it.
        return CreateUnauthorized(reason="not_task_principal")

    if principal.kind not in ("user", "guest"):
        # Rejected before the lookup, beside the missing-id guard above and
        # for the same kind of reason. Such a principal is unauthorized
        # either way -- the branch chain below has no third arm that
        # authorizes -- but reaching that chain means the lookup ran first,
        # and an unrecognized kind is not owner-scoped, so it would have run
        # by id alone and answered CreateUnavailable(reason="task_missing")
        # for a task that does not exist. That is the existence oracle the
        # owner-scoped branches are written to withhold, handed to a
        # principal this module cannot even name.
        return CreateUnauthorized(reason="not_task_principal")

    owner_scoped = principal.kind == "guest" or (
        principal.kind == "user" and not principal.is_admin
    )

    task_lookup = db.query(Task).filter(Task.id == task_id)
    if owner_scoped:
        task_lookup = task_lookup.filter(Task.user_id == principal.user_id)
    task = task_lookup.first()
    if task is None:
        # An owner-scoped lookup cannot tell "no such task" from "not your
        # task", and must not appear to -- see the consequence paragraph in
        # this function's docstring.
        if owner_scoped:
            return CreateUnauthorized(reason="not_task_principal")
        return CreateUnavailable(reason="task_missing")

    if principal.kind == "user":
        # A non-admin reached this line only by matching the owner
        # predicate in SQL; an admin reached it without one and is
        # authorized on the flag alone.
        authorized = True
    elif principal.kind == "guest":
        # The owner term is already proved by the lookup above, the same
        # way the three public-chat entry points prove it; what is left for
        # the Python predicate is the agent_config conjunction, which no
        # WHERE clause can express.
        try:
            authorized = task_is_owned_by_public_principal(task, principal)
        except ValueError:
            authorized = False
    else:
        authorized = False
    if not authorized:
        return CreateUnauthorized(reason="not_task_principal")

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
    try:
        validate_v1_write_payload(parsed_values)
    except ValueError as exc:
        # Shape-valid and serializable, but not a question this service is
        # willing to persist -- an unrenderable interaction type, a select
        # with nothing to select, a blank or duplicated field name, two
        # options sharing a value, a blank option half, an inverted numeric
        # range. Same reason as every other payload rejection: the caller
        # learns its values were not accepted, not which of the checks in
        # front of them said so.
        #
        # The precise diagnostic is not lost, only kept server-side: this
        # validator names the exact position it refused
        # ("request_payload.interactions[3].options[1] has a blank label or
        # value"), which is the only thing an operator debugging a rejected
        # write has to go on. Logged at warning, unconditionally -- an
        # observability line that can be switched off is one that is off
        # when it is needed.
        #
        # What the message can contain: positions, and three identifiers
        # the question's author chose -- an interaction's ``type`` (quoted
        # by the rule that refuses a type outside the v1 vocabulary), an
        # interaction's ``field``, and an option's ``value`` (the latter
        # two quoted by the rules that refuse a duplicate of either). All
        # three are unconstrained ``str`` on the tool model, so nothing at
        # the type level keeps caller text out of them; what keeps it out
        # today is that no production path puts user input in any of the
        # three, and the response side is not reachable from here at all.
        # Constraining them belongs at the schema, not here. The payload
        # is never logged whole.
        logger.warning(
            "v1 interaction write payload refused for task_id=%s: %s",
            task_id,
            exc,
        )
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
    # The unanswerable tier's full reason vocabulary lives on this one
    # line, because ``CompatibilityQuestionView.reason`` is a bare string
    # with no type to carry it. Two values come from this resolver:
    # "anchor_dangling" | "checkpoint_unavailable". Two more are produced
    # directly by ``materialize_compatibility_view`` without going through
    # this class, because they are decided before the anchor is ever
    # looked at: "protocol_version_unrecognized" | "payload_unreadable".
    # Converging all four into a Literal belongs with the endpoint that
    # first classifies on them.
    reason: str


def _resolve_read_direction_anchor(
    db: "Session", row: TaskInteractionRequest
) -> _AnchorUnresolved | None:
    """Resolve an active interaction row's resume anchor for the read
    direction: does ``row.resume_trace_event_id`` still point at a
    structurally valid checkpoint row -- the right task, event type,
    checkpoint type, run partition, (when present) execution identity, and
    the event id the anchor itself names? Returns ``None`` on success, or
    an ``_AnchorUnresolved`` naming which of the two outcomes applies
    otherwise.

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
      checkpoint row", from different directions. (That shared predicate now
      exists -- ``failed_checkpoint_row_conditions``,
      ``trace_event_staging.py`` -- and the two by-primary-key resolvers
      read it. This resolver does not: its judgment carries a seventh
      condition that one has no input for (``trace_row.event_id`` against
      ``row.resume_event_id``) and compares the partition against a
      non-null ``resume_run_partition`` rather than a task's possibly-null
      ``run_id``. Adopting it would mean adding an optional condition and a
      second partition rule for one caller.)
    - One condition in that judgment is this resolver's alone and is not
      expected to appear in trace_handlers': ``trace_row.event_id`` must
      equal ``row.resume_event_id``. It is the identity the write-direction
      resolver stored on the interaction row for exactly this comparison,
      and it is checkable only from here -- trace_handlers reaches a
      checkpoint by primary key with no interaction row in hand, so it has
      no second identity to compare against. The "must agree with
      trace_handlers' own" rule above does not reach it, and matching the
      two sides is not a reason to remove it.
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
    side") resolver ``resolve_interaction_anchor``
    (``task_interaction_anchor.py``) rather than against trace_handlers:
    that resolver treats a legacy checkpoint type as "no anchor available"
    (its own step 5) and never stages an active row anchored to one at all.
    This resolver, by contrast, accepts every member of
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
    except (
        sa.exc.OperationalError,  # connection loss / lock wait / timeout
        sa.exc.InterfaceError,  # DBAPI-level connection failure
        sa.exc.DisconnectionError,  # pool detected a dropped connection
        sa.exc.TimeoutError,  # pool checkout timed out
    ):
        # Narrow on purpose, and narrower than ``sa.exc.SQLAlchemyError``:
        # fallback is open only to transient, recoverable infrastructure
        # failures. Anything that instead indicates a programming defect
        # or a session that is itself unrecoverably broken must propagate
        # -- swallowing it here would disguise a bug as an operational
        # condition. That is why ``ProgrammingError``, ``DataError``,
        # ``InternalError``, ``ArgumentError``, ``CompileError``,
        # ``InvalidRequestError``, ``NoResultFound``,
        # ``ResourceClosedError`` and ``PendingRollbackError`` are all
        # deliberately absent from this list. ``PendingRollbackError`` in
        # particular is not reachable here on either entry path, for a
        # different reason on each. materialize_compatibility_view calls
        # ``interaction_requests_table_exists`` first, which issues
        # ``db.connection()`` on the same session, so a session broken
        # badly enough to raise ``PendingRollbackError`` raises it there.
        # respond() does not go through that check -- it reaches this
        # resolver directly -- but it owns its session and has already
        # run several statements on it by then, so the same failure
        # surfaces before this fetch as well. Its source is also mixed
        # -- sometimes a
        # connection that failed mid-transaction, sometimes a prior flush
        # failure that left the session itself unrecoverable -- so even if
        # it were reachable, it would not belong on a transient-only list.
        #
        # No db.rollback() here: respond() holds the tasks row's FOR
        # UPDATE lock from its earlier step and owns its own session end
        # to end, so a rollback here would release that lock mid-flow.
        # Note the backend asymmetry after a genuine DBAPI failure at this
        # fetch: SQLite keeps the caller's staged, uncommitted writes;
        # PostgreSQL has already invalidated the server-side transaction,
        # so a later commit() returns successfully while discarding them.
        # A rollback here would not recover those writes -- deciding what
        # to do about the failed transaction belongs to the session
        # owner, not this read helper.
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
    event_id_matches = trace_row.event_id == row.resume_event_id
    if (
        trace_row.task_id != row.task_id
        or trace_row.event_type != str(CHECKPOINT_EVENT_TYPE)
        or trace_row.build_id is not None
        or row_data.get("checkpoint_type") not in READABLE_CHECKPOINT_TYPES
        or not partition_matches
        or not execution_matches
        or not event_id_matches
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
    lossy projection, and the four call sites that consume it, are handled
    by ``task_interaction_read.get_pending_interaction_question``, which
    imports this type and projects it down. ``reason`` carries the reason
    code #1079's endpoint needs and the legacy tuple has no slot for; it is
    only set on the ``"unanswerable"`` tier.
    """

    tier: str  # "legacy" | "native" | "unanswerable"
    question: str | None
    interactions: list[dict[str, Any]] | None
    reason: str | None = None


def _legacy_view(
    db: "Session", task_id: int, *, allow_superseded: bool = False
) -> CompatibilityQuestionView:
    question, interactions = get_latest_waiting_question(
        db, task_id, allow_superseded=allow_superseded
    )
    return CompatibilityQuestionView(
        tier="legacy", question=question, interactions=interactions
    )


def _validation_error_summary(exc: _PydanticValidationError) -> list[str]:
    """Which fields failed validation and how -- never what was in them.

    ``str(ValidationError)`` embeds ``input_value=``, and for this payload
    the input is the question text written for an end user. This summary
    is built from ``errors()`` with the input and the docs URL excluded,
    keeping the field path and the error type and nothing else. Capped so
    a payload with many fields cannot turn one log record into a dump.
    """

    summary: list[str] = []
    for error in exc.errors(include_url=False, include_input=False)[:10]:
        location = ".".join(str(part) for part in error.get("loc", ()))
        summary.append(f"{location or '<root>'}:{error.get('type', 'unknown')}")
    return summary


def materialize_compatibility_view(
    db: "Session", task_id: int, *, allow_superseded: bool = False
) -> CompatibilityQuestionView:
    """The single rich implementation of "what is this waiting task's
    question", three-tiered:

    T1 (tier ``"legacy"``) -- the ``task_interaction_requests`` table does
    not exist yet, or there is no active native row for this task: falls
    back, internally, to ``get_latest_waiting_question`` and returns
    exactly what that function hands back, with no filtering of its own
    and no reason code (the caller's ``allow_superseded`` is passed
    through to it -- see below). This tier's
    table-existence gate is not defensive decoration for a table that
    might never exist -- ``interaction_rollout.py``'s own ``/ready`` gate
    treats "the service deploys before its own migration has run" as a
    real, accepted window, and this reader has to survive being called
    inside it.

    T2 (tier ``"native"``) -- an active native row exists and its resume
    anchor resolves (see ``_resolve_read_direction_anchor`` for exactly
    what "resolves" checks -- the checkpoint row's identity, not its
    decoded payload): the native projection, ``question`` and
    ``interactions`` decoded from ``request_payload`` itself, not the
    legacy transcript.

    T3 (tier ``"unanswerable"``) -- an active native row exists but this
    reader cannot answer with it, for one of four reasons, and ``reason``
    names which: the row's ``protocol_version`` is not one this reader
    recognizes (``"protocol_version_unrecognized"``); its
    ``request_payload`` does not parse against the v1 shape
    (``"payload_unreadable"``); or its resume anchor does not resolve
    (``"anchor_dangling"`` / ``"checkpoint_unavailable"``, see
    ``_resolve_read_direction_anchor``). Only the anchor-resolution pair
    can still read the row's own question text -- the other two cannot
    recover it at all, so both slots come out empty. **This tier must
    never fold into "no active row" and retry the T1 fallback**, on any of
    the four reasons -- doing so would show the caller a legacy transcript
    question whose answer would land in a slot a native row has already
    claimed, producing a self-contradictory result. For the anchor-
    resolution pair, the one thing that makes this projection honest
    rather than a lie by omission: the compatibility seam that accepts
    continuation commands (landing with a later change) refuses a
    legacy-shaped answer on exactly this same condition -- an active
    native row with an unresolved anchor -- so "no controls shown" and "a
    free-text answer would be refused anyway" agree. If that seam's
    refusal is ever loosened, this tier's projection needs re-deciding
    alongside it, not independently.

    Both slots empty is a dead end for the user -- the task is plainly
    waiting and the interface has nothing to answer -- and it is accepted
    only because operations is told at the same moment and can go look.
    Remove or silence that signal and this tier's projection has to be
    decided again.

    Consumers, and how much of this result they get:
    ``task_interaction_read.get_pending_interaction_question`` projects
    this down to the legacy ``(question, interactions)`` tuple for the
    four existing call sites, dropping ``reason`` -- lossy by design, not
    an oversight. Three of the four already hold a loaded ``Task`` row
    when they need the answer; the fourth resolves and authorizes one
    first, inside a worker-owned short session, before calling this view.
    #1079's own endpoint (not written here) is meant to consume this rich
    result directly, keeping ``reason`` for its own outcome classification.

    ``allow_superseded`` is passed straight to
    ``get_latest_waiting_question`` on both T1 branches and nowhere else.
    It lets the transcript reader reach a question row a structured
    publication has already relabelled. Both T1 branches mean "no active
    native row", which is not on its own enough to make that honest: the
    window where ``respond()`` has retired the row to ``answered`` while
    the task stays ``WAITING_FOR_USER`` until its staged resume command is
    consumed means it too, and a read landing there would re-offer the
    relabelled row for a question already answered. That window is
    unreachable here -- ``respond()`` is its only writer and has no
    production caller, kept true by its own zero-caller gate -- and what
    the read surface owes it belongs to the change that publishes an
    interaction atomically with the task's status transition. The three T3
    branches never call ``_legacy_view`` at all, so the parameter has no
    place to appear in them -- not an omission.

    Deciding "no active row" and reading the transcript are two separate
    statements, and on PostgreSQL's default READ COMMITTED each statement
    takes a fresh snapshot, so another session can commit an active row
    between them. The no-active-row branch therefore looks once more, after
    the transcript read and before returning, and answers from the row if
    one has appeared. **This narrows the window, it does not close it**:
    another session can still commit an active row between that recheck and
    this function's return, and the caller gets the legacy tier anyway. The
    consequence of landing in that remaining window is bounded and worth
    naming: the user answers through the legacy channel, the native row
    retires as ``answered_via_legacy_resume``, and the answer is neither
    lost nor the task left stuck -- what that one turn does not get is a
    structured record of the answer. The table-absent branch does not
    recheck: with no table there is nothing an active row could have been
    written into.

    The recheck's inverse is not narrowed by any of this and does not need
    to be: a row it finds can retire between the find and whatever the
    caller does next. Nothing is corrupted when that happens, because the
    answer path never trusts this read -- ``respond()`` re-selects the row
    inside ``_answer_fence_stmt``'s compare-and-swap and classifies the
    zero rowcount, rather than writing into a slot that has since moved on.

    ``compat.read_fallback`` counts one per legacy tier returned from here,
    and only from here. It is a raw count and not a rate: nothing in this
    registry records how often this function ran or how often it answered
    from a native row, so there is no denominator to read it against, and
    a reader who wants one has to bring their own request-volume figure.
    Two different states increment it and they are worth keeping apart --
    the table-absent branch counts a deployment whose interaction-table
    migration has not landed yet, and the no-active-row branch counts a
    read that genuinely had to fall back to the transcript. Only the second
    says anything about the rollout; past the migration the first cannot
    fire at all, so a non-zero count on a migrated deployment is entirely
    the second. A run the recheck rescues is not a fallback and is not
    counted. Like every counter in ``interaction_rollout``, this one lives
    in process memory: it starts at zero on each start, a redeploy resets
    it, and a reader behind a load balancer sees one process's share.
    """

    if not interaction_requests_table_exists(db):
        # No recheck on this branch: with no table there is nowhere for an
        # active row to have been written. The count goes up after
        # ``_legacy_view`` returns, not before it is called, so a raise on
        # the way through cannot leave a fallback counted that no caller
        # ever received -- same ordering as the no-active-row branch below.
        fallback = _legacy_view(db, task_id, allow_superseded=allow_superseded)
        increment_counter(COUNTER_COMPAT_READ_FALLBACK)
        return fallback

    row = _active_native_row(db, task_id)
    if row is None:
        fallback = _legacy_view(db, task_id, allow_superseded=allow_superseded)
        row = _active_native_row(db, task_id)
        if row is None:
            increment_counter(COUNTER_COMPAT_READ_FALLBACK)
            return fallback
        # An active row appeared between the two looks, so this task's tier
        # is decided from it below instead of from the transcript. Once:
        # there is no second recheck, and a row found here is answered
        # from rather than looked at again.

    if row.protocol_version != INTERACTION_PROTOCOL_VERSION:
        # An active row holds this task's answer slot, so this cannot fold
        # back into "there is no active row" and re-offer the legacy
        # transcript: that would surface a question whose answer would land
        # in a slot this row has already claimed. The row is unreadable
        # rather than absent, so both slots come out empty and the caller
        # shows nothing to answer -- which is only acceptable because of
        # the signal raised right here.
        #
        # ``ck_task_interaction_requests_active_protocol`` pins every
        # active row to protocol version 1, so on a schema carrying that
        # constraint this branch cannot be reached at all: the database is
        # the first line and this check is the second. It is kept, and
        # kept observable, because a model-level constraint is not a
        # licence for the reader to assume -- and because that constraint
        # only converged on SQLite recently, so an older SQLite file may
        # not carry it.
        register_degradation(
            INTERACTION_READ_PROTOCOL_UNRECOGNIZED,
            f"task {task_id}: active interaction {row.id} carries protocol "
            f"version {row.protocol_version}",
        )
        logger.warning(
            "active native interaction row carries an unrecognized protocol "
            "version; projecting an empty pending question",
            extra={
                "task_id": task_id,
                "interaction_id": row.id,
                "row_protocol_version": row.protocol_version,
                "reason": "protocol_version_unrecognized",
            },
        )
        return CompatibilityQuestionView(
            tier="unanswerable",
            question=None,
            interactions=None,
            reason="protocol_version_unrecognized",
        )

    try:
        parsed = parse_v1_request_payload(row.request_payload)
    except _PydanticValidationError as exc:
        # An active row holds the answer slot but its stored
        # request_payload does not parse against the v1 shape, so its
        # question text cannot be recovered at all -- ``parsed`` does not
        # exist in this branch. Same rule as the unrecognized-version
        # branch above: unreadable is not absent, so this does not fold
        # back into the legacy transcript; both slots come out empty and
        # the signal below is what makes that acceptable.
        register_degradation(
            INTERACTION_READ_PAYLOAD_UNREADABLE,
            f"task {task_id}: active interaction {row.id} request_payload "
            "does not parse against the v1 shape",
        )
        logger.warning(
            "active native interaction row failed v1 payload validation; "
            "projecting an empty pending question",
            extra={
                "task_id": task_id,
                "interaction_id": row.id,
                "reason": "payload_unreadable",
                "validation_errors": _validation_error_summary(exc),
            },
        )
        return CompatibilityQuestionView(
            tier="unanswerable",
            question=None,
            interactions=None,
            reason="payload_unreadable",
        )

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


# ---------------------------------------------------------------------------
# The answer fence: the one write this module makes to an active interaction
# row, and the shared active-row predicate's write-side counterpart the
# module docstring on ``_active_native_row_criteria`` already commits to.
# ---------------------------------------------------------------------------


def _answer_fence_task_predicate(principal: InteractionPrincipal) -> list[Any]:
    """The task-side terms of the answer fence: is this task still waiting,
    and does ``principal`` own it -- evaluated against the same ``tasks``
    row the fence statement joins in via ``Task.id == task_id`` (see
    ``_answer_fence_stmt``), not a second, independently-scoped read.

    The ``Task.user_id`` term requires the owner in person, on both
    backends: step 3's authorization admits an admin acting on another
    user's task, and the guest ownership predicate never reads
    ``principal.user_id`` (see ``task_is_owned_by_public_principal``), so
    for both of those callers this term is the only ownership constraint
    in the path and the fence misses. Answering on another user's behalf
    is not delivered in this build; whether an admin may do so is a policy
    decision left to the change that wires the first production caller. A
    ``principal.user_id`` of ``None`` compiles to ``tasks.user_id IS
    NULL`` and matches nothing (the column is ``nullable=False``), so such
    a caller fails closed here rather than matching another user's row.

    Against a *concurrent* ownership change the term is redundant on
    PostgreSQL -- step 2 of ``respond()`` already holds this row's
    ``FOR NO KEY UPDATE`` lock -- and load-bearing on SQLite, where the
    dialect drops every locking clause silently and nothing serializes a
    caller until this statement, the transaction's first write (see
    ``respond()``'s own docstring for the two backends' different
    serialization points). Enforcing ownership at the write point rather
    than only in ``respond()``'s step 3 is what makes the two backends
    agree on the terms this statement actually carries -- which is not
    every term step 3 evaluated. Three are re-asserted here: the task is
    still ``WAITING_FOR_USER``, ``Task.user_id`` is still the principal's,
    and, for a guest, the ``guest_id`` in ``agent_config`` still matches.
    The rest of the guest conjunction -- ``auth_mode``, the entity binding,
    and the channel binding -- is evaluated once in Python at step 3 and
    not re-checked in SQL. A successful answer therefore proves the three
    terms above held at write time and that the other three held at read
    time, not that all six held at write time.

    ``Task.status`` is compared through ``TaskStatusPredicate.eq``, never a
    raw string literal beside the column -- this module's own convention
    (see ``TaskStatusPredicate``'s docstring for why a raw literal is a
    query-time failure waiting to happen, not merely a style preference).

    The two entity-binding backends disagree on how a JSON key read
    compiles (``->>`` on PostgreSQL, ``json_extract`` on SQLite) -- both are
    exercised by ``test_task_interaction_service_postgresql.py`` and this
    module's SQLite unit tests, deliberately, rather than assumed
    equivalent from one compiled form.
    """

    terms: list[Any] = [
        TaskStatusPredicate.eq(TaskStatus.WAITING_FOR_USER),
        Task.user_id == principal.user_id,
    ]
    if principal.kind == "guest":
        terms.append(Task.agent_config["guest_id"].as_string() == principal.guest_id)
    return terms


def _answer_fence_stmt(
    *,
    interaction_id: int,
    task_id: int,
    principal: InteractionPrincipal,
    response_payload: dict[str, Any],
    now: "datetime",
    responder_user_id: int | None,
    responder_identity: str,
) -> Any:
    """The Core UPDATE that is this module's one and only write to an active
    interaction row: a compare-and-swap on row status, task state, and
    ownership, all evaluated in the same statement's WHERE clause so that a
    concurrent change to any one of them is what makes ``rowcount`` land on
    zero rather than one.

    ``Task.id == task_id`` pins the implicit ``FROM tasks`` join this
    statement needs (both for the reused active-row predicate's own
    ``Task.run_id`` comparison and for ``_answer_fence_task_predicate``'s
    terms) to exactly one row. The ownership terms are deliberately a flat
    conjunction here, not their own correlated ``EXISTS(...)``: a subquery
    correlated against this UPDATE's target table auto-correlates every
    table the two queries have in common, including
    ``TaskInteractionRequest`` itself, which either raises
    ``InvalidRequestError`` or -- if correlation is pinned away from
    ``TaskInteractionRequest`` -- leaves the outer join unpinned to a
    specific ``tasks`` row (verified empirically against both backends'
    compiled SQL). A flat conjunction against a ``tasks`` row already pinned
    by primary key has no such ambiguity and compiles to one join, not a
    join plus a subquery.

    The three active-row criteria (``status``, ``active_slot``, and the
    ``run_id`` comparison against this same joined ``tasks`` row) are
    imported from ``_active_native_row_criteria`` rather than rewritten --
    that function's own docstring already commits every future caller
    needing "the live row" to changing together with it.

    Writes exactly the columns the model's paired CHECK constraints require
    for an answered row: ``status``, ``active_slot`` (cleared), and,
    together, ``response_payload`` / ``responded_at`` / responder identity.
    Never writes ``terminated_at`` or ``terminal_reason`` -- doing so on a
    row this statement is simultaneously marking ``answered`` trips
    ``ck_task_interaction_requests_terminal_pairs_status`` /
    ``ck_task_interaction_requests_terminated_at_pairs_status``, the
    database refusing a row that claims to be both answered and
    terminated. ``responder_user_id`` is the caller's to set correctly:
    populated only for a ``"user"`` principal, left ``None`` for a guest
    (see ``respond()``'s own docstring for why that column and
    ``responder_identity`` are allowed to disagree).
    """

    return (
        sa.update(TaskInteractionRequest)
        .where(
            TaskInteractionRequest.id == interaction_id,
            TaskInteractionRequest.task_id == task_id,
            Task.id == task_id,
            *_active_native_row_criteria(),
            *_answer_fence_task_predicate(principal),
        )
        .values(
            status="answered",
            active_slot=None,
            response_payload=response_payload,
            responded_at=now,
            responder_user_id=responder_user_id,
            responder_identity=responder_identity,
            updated_at=now,
        )
    )


# ---------------------------------------------------------------------------
# respond(): the answer-side entry point. Everything above this point in the
# module either already shipped (the principal, the outcome unions, create(),
# the shared active-row predicate, the anchor resolver, the compatibility
# view) or is this function's own supporting statement (the fence, just
# above). What follows is the function this module was always building
# toward, per its own docstring's "not delivered here" list -- which this
# change retires.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RespondEnvelope:
    """The caller-supplied answer for ``respond()``: what interaction this
    answers, with what values, and under which idempotency key -- not yet
    validated.

    Deliberately carries no resume-locator field mirroring
    ``CreateInteractionEnvelope``'s absence of one: the only documented
    caller today (#1079's own AC) never hands one back, and the only thing
    an echoed locator could prove -- "this is the same value the server
    handed out" -- is already proven by ``interaction_id`` plus the row and
    anchor checks ``respond()`` runs on its own account (steps 4 and 5.5).
    """

    kind: str
    protocol_version: int
    values: Any
    idempotency_key: str


def _respond_command_payload(
    *, interaction_id: int, principal: InteractionPrincipal, values: Any
) -> dict[str, Any]:
    """The RESUME command payload ``respond()`` stages in step 8, and the
    payload it re-derives at steps 5 and 8 to look an existing command up by
    the same shape.

    Carries ``responder_identity`` alongside the answer ``values`` so that
    two different principals submitting the same idempotency key with the
    same ``values`` are *not* treated as the same idempotent request: step
    7's ``actor_user_id`` is the task owner for both a ``"user"`` and a
    ``"guest"`` principal (see ``respond()``'s own docstring), so
    ``_matches_existing``'s ``actor_user_id`` comparison alone cannot tell
    them apart -- putting ``responder_identity`` in the payload that
    ``_canonical_payload`` hashes is what does: two principals reusing one
    key now canonicalize to two different payloads, which
    ``_matches_existing`` reports as ``idempotency_key_reused``, not a
    replay.
    """

    return {
        "interaction_id": interaction_id,
        "responder_identity": principal.identity_string(),
        "values": values,
    }


def _respond_receipt(
    *,
    interaction: "TaskInteractionRequest",
    task: "Task",
    command_db_id: int,
    idempotency_key: str,
) -> InteractionResponseReceipt:
    """Build the receipt from already-loaded, already-committed-or-about-to-
    commit rows, never from a lazy relationship read after commit (see
    ``InteractionResponseReceipt``'s own docstring). ``responder_identity``
    is read from the interaction row's own column, never reconstructed from
    a caller's ``principal`` -- that column, not ``principal``, is this
    table's audit-authoritative record of who answered (see ``respond()``'s
    own docstring for why ``responder_user_id`` cannot be used for the same
    purpose). This is not the only receipt this module builds: the accept
    path builds ``RespondAccepted``'s receipt from plain locals captured in
    the same transaction before the commit below, including
    ``principal.identity_string()`` for ``responder_identity`` rather than
    a read of this row's column. The two are not competing sources of
    truth -- the answer fence this function's own precondition depends on
    is the statement that wrote the column from that same
    ``principal.identity_string()`` value, so the column and the local are
    equal by the fence write's own semantics, not by coincidence.

    Three callers, each with its own precondition for why the row it hands
    in is already answered. ``respond()``'s idempotent-replay pre-read
    branch (step 5) calls this on a row it found by this call's own
    idempotency key -- a staged RESUME command this service itself
    committed in some earlier transaction alongside the answer fence
    UPDATE, which implies an answered row. ``respond()``'s fence-miss
    classification (step 6) calls it on the row its own reread just read
    back as ``status == "answered"``, paired with a command row found
    under this call's own idempotency key whose payload matches -- the
    same two facts the pre-read branch stands on, established one
    statement earlier instead of before the fence attempt.
    ``_verify_respond_durable_graph`` calls this only after its own three
    checks already confirmed the row it is holding matches: ``status ==
    "answered"``, the answering identity, and the canonical submitted
    payload. In all three cases, the paired CHECK constraints
    (``ck_task_interaction_requests_responded_at_pairs_status`` and
    ``ck_task_interaction_requests_responder_pairs_responded_at``) make an
    answered row with a NULL ``responded_at`` or ``responder_identity``
    impossible. That reasoning spans two modules, so the guard below turns
    it into a loud failure instead of trusting it silently: a receipt must
    never carry a coerced ``'None'`` identity or a ``None`` timestamp.
    """

    if interaction.responded_at is None or interaction.responder_identity is None:
        raise RuntimeError(
            f"interaction {interaction.id} on task {interaction.task_id} "
            "matched a staged RESUME command but carries no answer; "
            "ck_task_interaction_requests_responder_pairs_responded_at "
            "makes this impossible"
        )
    return InteractionResponseReceipt(
        interaction_id=int(interaction.id),
        task_id=int(interaction.task_id),
        run_id=str(interaction.run_id),
        status=str(interaction.status),
        responded_at=cast("datetime", interaction.responded_at),
        responder_identity=str(interaction.responder_identity),
        idempotency_key=idempotency_key,
        command_db_id=command_db_id,
        task_state_version=int(task.state_version),
        task_control_state=str(task.control_state),
    )


_RESPOND_DURABLE_GRAPH_ATTEMPTS = 3
_RESPOND_DURABLE_GRAPH_RETRY_SLEEP_SECONDS = 0.01


def _retire_respond_session_best_effort(db: "Session") -> None:
    """Release an owned Session without letting a close/invalidate failure
    replace the transaction error that is already in flight. Same shape as
    ``task_orchestrator._retire_turn_session_best_effort`` -- a different
    copy because that helper is private to its own module and this
    service's session lifecycle (see the module docstring: ``respond()``
    owns and retires its own session) is not the turn-commit lifecycle that
    helper is named for.
    """

    try:
        db.close()
        return
    except Exception:
        logger.warning("failed to close respond() session", exc_info=True)
    try:
        db.invalidate()
    except Exception:
        logger.warning("failed to invalidate respond() session", exc_info=True)


def _notify_dispatcher_best_effort(*, interaction_id: int, task_id: int) -> None:
    """Wake the task command dispatcher after an answer is already durable,
    and never let that wakeup turn a committed answer into a raised
    exception.

    ``notify_task_command_dispatcher`` reads two module globals and calls
    ``loop.call_soon_threadsafe`` after checking ``loop.is_closed()``; the
    loop can close between that check and that call, raising
    ``RuntimeError``. Letting it out would make the caller read a
    committed answer as a failure, retry, land on the replay branch, and
    get a receipt without a second notify anyway. Skipping the wakeup
    costs at most ``DISPATCHER_IDLE_SECONDS`` of latency because the
    dispatcher's idle poll still finds the staged command; that is the
    documented fallback (see ``stage_task_command``'s caller obligation
    (a)).

    Narrow on purpose: this wraps one post-commit best-effort
    notification and nothing else -- no statement above a commit is
    inside it. ``respond()`` calls it from both of its two post-commit
    exits: the ordinary one after step 9's own commit, and the one after
    step 9's durable-graph reconciliation recovers a receipt from a
    commit whose acknowledgment was lost. The second exit is the reason
    this is a function rather than an inline ``try`` -- the answer is
    just as durable there, so it must not raise there either.
    """

    try:
        notify_task_command_dispatcher()
    except Exception:
        logger.warning(
            "failed to notify the task command dispatcher after "
            "committing the answer for interaction %s on task %s; "
            "the dispatcher's idle poll will still pick the command up",
            interaction_id,
            task_id,
            exc_info=True,
        )


def _verify_respond_durable_graph(
    *,
    task_id: int,
    interaction_id: int,
    expected_run_id: str,
    expected_state_version_after: int,
    principal: InteractionPrincipal,
    canonical_submitted_values: str,
    command_id: str,
    command_kind: TaskCommandKind,
    command_payload: dict[str, Any],
    actor_user_id: int | None,
) -> InteractionResponseReceipt | None:
    """Check the complete accepted answer graph in a fresh, owned Session,
    up to three times with a short sleep between attempts. Same retry shape
    as ``task_orchestrator._reconcile_claimed_turn_after_commit_ack_failure``:
    a new ``SessionLocal()`` per attempt, ``time.sleep(0.01)`` between
    failures, the session retired in every case before the next attempt or
    return.

    Three independent facts must all hold -- the row's key, the answer's
    hash, and the answering principal's identity: (1) the
    ``tasks`` row has advanced past ``expected_state_version_after - 1`` on
    the same ``run_id`` -- ``>=``, not ``==``, and ``control_state`` is not
    compared at all, because the resume coordinator re-issues the same
    ``RESUME_REQUESTED`` transition when it applies the command this call
    staged (see ``respond()``'s own docstring for the verbatim reasoning);
    (2) the interaction row is answered, by this identity, with a
    ``response_payload`` that canonicalizes to the same string as the
    answer ``values`` the caller submitted (``canonical_submitted_values``
    -- the interaction row stores the answer values alone, not the wrapping
    command payload the third check below compares); (3) exactly one
    command row exists for this idempotency key and it matches what this
    call staged. Returns the receipt on success, ``None`` if every attempt
    fails to confirm all three.
    """

    from ..models.database import get_session_local

    SessionLocal = get_session_local()
    for attempt in range(_RESPOND_DURABLE_GRAPH_ATTEMPTS):
        reconcile_db: "Session | None" = None
        try:
            reconcile_db = SessionLocal()
            task = (
                reconcile_db.query(Task)
                .filter(
                    Task.id == task_id,
                    Task.run_id == expected_run_id,
                    Task.state_version >= expected_state_version_after,
                )
                .first()
            )
            if task is not None:
                ir = (
                    reconcile_db.query(TaskInteractionRequest)
                    .filter(
                        TaskInteractionRequest.id == interaction_id,
                        TaskInteractionRequest.task_id == task_id,
                        TaskInteractionRequest.status == "answered",
                        TaskInteractionRequest.responder_identity
                        == principal.identity_string(),
                    )
                    .first()
                )
                if (
                    ir is not None
                    and _canonical_payload(
                        ir.response_payload
                        if isinstance(ir.response_payload, dict)
                        else {}
                    )
                    == canonical_submitted_values
                ):
                    commands = (
                        reconcile_db.query(TaskExecutionCommand)
                        .filter(
                            TaskExecutionCommand.task_id == task_id,
                            TaskExecutionCommand.command_id == command_id,
                        )
                        .all()
                    )
                    if len(commands) == 1 and _matches_existing(
                        commands[0],
                        actor_user_id=actor_user_id,
                        kind=command_kind,
                        payload=command_payload,
                    ):
                        return _respond_receipt(
                            interaction=ir,
                            task=task,
                            command_db_id=int(commands[0].id),
                            idempotency_key=command_id,
                        )
        except Exception:
            # Deliberately broad, and it does not widen ``respond()``'s
            # escape surface: this whole function runs only after step 9
            # already caught a commit exception, so anything raised while
            # reading the graph back means the attempt could not answer
            # "did it land?", not that a new failure needs reporting. Each
            # attempt logs on its own, and exhausting all three returns
            # ``None`` -- the same ``OutcomeUnknown`` the caller would have
            # been given had this reconciliation never run.
            logger.warning(
                "respond() durable-graph reconciliation attempt %s failed for "
                "task %s interaction %s",
                attempt + 1,
                task_id,
                interaction_id,
                exc_info=True,
            )
        finally:
            if reconcile_db is not None:
                _retire_respond_session_best_effort(reconcile_db)
        if attempt < _RESPOND_DURABLE_GRAPH_ATTEMPTS - 1:
            time.sleep(_RESPOND_DURABLE_GRAPH_RETRY_SLEEP_SECONDS)
    return None


def respond(
    *,
    interaction_id: int,
    task_id: int,
    principal: InteractionPrincipal,
    envelope: RespondEnvelope,
) -> RespondOutcome:
    """Answer one active interaction row and stage the RESUME command that
    lets the task's execution resume with that answer, in one all-or-
    nothing transaction on a session this function owns end to end (opens
    it, commits or rolls it back, and retires it -- the caller never passes
    one in, because step 9 below needs to be able to retire it on an
    ambiguous commit and start a fresh one to verify what actually landed).

    Every rowcount-based branch below assumes READ COMMITTED (PostgreSQL's
    default, and what this deployment uses; see the module docstring for
    the general statement). The branch that actually depends on it is step
    6's zero-rowcount reread: a blocked fence UPDATE re-evaluates its WHERE
    clause once the lock holder's transaction ends, so a second caller's
    UPDATE naturally resolves to a `rowcount` of zero that this function can
    then classify against the row's now-current state. Under REPEATABLE
    READ or stricter the same conflict surfaces as a serialization failure
    instead of a rowcount of zero, and step 6's classification does not
    hold.

    `_handle_resume_task_unserialized` re-issues the same
    `RESUME_REQUESTED` transition when it applies the command this call
    staged (`websocket.py`, reached from `_execute_durable_task_command`).
    `apply_task_control_transition` has no legality table, so that second
    transition succeeds and bumps `state_version` again. Post-commit
    verification must therefore be monotone in `state_version` and must not
    compare `control_state`. One concrete consequence, recorded here
    because a future reader who treats `state_version` as an operation
    counter will be surprised by it: answering one interaction bumps
    `state_version` twice -- once in step 7 below, once when the
    coordinator re-applies the transition -- so a consumer that expects
    "one answer, one version bump" will observe +2.

    That second bump is not reachable today: nothing in production calls
    `respond()`, so no RESUME command carrying this function's payload is
    ever staged or executed. It becomes reachable with the change that
    gives `respond()` its first production caller, and deciding which of
    the two writers owns the transition belongs to that change, not this
    one. Making the transition single-writer here is not an option this
    change can take on its own: `apply_task_control_transition` is shared
    by eight direct call sites across five modules plus five more through
    `transition_task_control_state_sync`, and adding a legality table to it
    is a change to the task state machine, not to this service.

    `responder_user_id` and `actor_user_id` answer two different questions
    and are allowed to disagree. `responder_user_id` (written on the
    interaction row) records who actually answered; for a guest principal
    it stays `NULL`, and it can also be cleared for a `"user"` principal
    later by that user account's own deletion (`ON DELETE SET NULL`) -- see
    the module docstring's audit-identity paragraph for why
    `responder_identity` and not this column is this table's audit source
    of truth. `actor_user_id` (written on the staged command, step 8)
    records whose identity the resulting RESUME command executes as. For a
    guest principal, and for a non-admin `"user"` principal, that is always
    the task's owning user: a guest's turn already runs as the entity owner
    it is chatting through, matching the existing `public_chat_access.py`
    precedent of dispatching guest-originated work under the owner's
    identity, and step 3's non-admin branch requires `task.user_id ==
    principal.user_id` to even reach here. The admin branch carries no
    ownership requirement -- `principal.is_admin` authorizes without one --
    so an admin's `actor_user_id` is the admin's own `principal.user_id`,
    belonging to someone other than the task's owner. It is never absent:
    step 3 requires a `"user"` principal to carry a `user_id` on both
    branches, so an identity-less caller cannot pass on the admin flag
    alone and arrive here with nothing to record. Do not "fix" the
    disagreement between the two columns into agreement; they are answers
    to different questions.

    Step 3 letting an admin through without owning the task is the
    intended behavior, not a missed check, and the reason is that step 3
    is not the step that authorizes the write. Two guards run, in this
    order and for different purposes:

    - Step 3, before the idempotency preread, keeps an unauthorized caller
      from reaching a read at all: without it, anyone who could guess an
      idempotency key could pull back someone else's receipt. It rejects
      early and broadly, and being an admin is enough to clear it.
    - Step 6's answer fence carries `_answer_fence_task_predicate`, whose
      ownership term requires `principal.user_id` to be the task's owner.
      An admin does not satisfy it. The fence matches zero rows, the
      reread classifies the miss, and the call returns
      `RespondUnauthorized(reason="not_task_principal")`.

    So an admin passes the authorization step and is still refused at the
    write point -- refused precisely, after the reread has established why,
    rather than refused early on a guess. Any reader tempted to "align" the
    two by rejecting admins at step 3 should note that this would change
    what the service does for admins, not just where it says no, and that
    the two guards are answering different questions on purpose: one is
    "may this caller read anything here", the other is "may this caller
    write this row".

    Of the two audit-relevant columns this function fills on the interaction
    row, `responder_identity` is the one a reader can trust to stay
    populated across account deletion; `responder_user_id` is a convenience
    join that account deletion can silently null out from under it. Every
    branch below that has to report `responder_identity` (the receipt
    builder, the durable-graph check) reads it back off the interaction
    row's own column rather than recomputing it from `principal`.

    Step 2 loads `task` as a mapped entity, not a column tuple, specifically
    so `apply_task_control_transition` (step 7) can call `object_session(task)`
    on it. The cost of that choice: `task` is now a live object in this
    session's identity map, and `apply_task_control_transition` flushes it
    (`session.flush([task])`) before issuing its own Core UPDATE. This
    function must not assign any attribute on `task` between step 2 and
    step 7 -- doing so would be picked up by that flush and written out
    alongside the CAS whether or not that was intended. Nothing between
    those two steps needs to set an attribute on `task` at all: the fence
    (step 6) and the CAS (step 7) are both Core statements operating on
    ``Task``/``TaskInteractionRequest`` directly, not on this loaded
    instance's attributes.

    Statement order, each step's position load-bearing:

    1. Pure Python validation against ``envelope`` alone: ``kind`` checked
       for ``str`` before it is compared against the shared vocabulary (a
       non-``str`` -- a ``list``, a ``dict`` -- is rejected outright rather
       than reaching a membership test that would raise ``TypeError`` on an
       unhashable value), ``protocol_version`` checked for ``int`` and not
       ``bool`` before it is compared against the current version (``bool``
       is excluded explicitly because it is a subclass of ``int`` and
       ``True == 1`` would otherwise pass; a ``float`` like ``1.0`` is
       rejected by the ``isinstance`` check for the same reason it would
       otherwise compare equal), ``idempotency_key`` through
       ``_normalize_command_id``, and ``values`` required to be a ``dict``.
       This mirrors ``stage_interaction_request``'s own type-before-value
       order (``task_interaction_staging.py``) for the identical reason.
       ``values``'s dict check is a structural check
       only, not the question-side ``parse_v1_request_payload`` contract --
       an answer's ``values`` shape is keyed by the *question's own*
       interaction fields, which are only known once the row is read in
       step 4, not from ``envelope`` alone.
       A dict whose contents cannot be rendered as JSON -- a ``datetime``,
       a ``set``, ``bytes``, or a ``nan``/infinite float -- is rejected
       here too, by the same ``json.dumps(..., allow_nan=False)`` probe
       the question side runs (see ``build_v1_request_payload``). Without
       it the first two would surface as a ``StatementError`` raised
       inside step 6's fence UPDATE, outside the ``RespondOutcome``
       contract entirely, and the third would be stored silently as a
       non-JSON token.
       Validating an answer against
       those per-field types (its ``InteractionArg.type`` /
       ``InteractionArg.field`` definitions) is not implemented in this
       change; a malformed-but-dict-shaped answer reaches the fence and is
       stored as submitted.
    2. The first SQL statement: load ``tasks`` by id with
       ``with_for_update(key_share=True)`` (``FOR NO KEY UPDATE`` on
       PostgreSQL) -- absent on the given id, ``Unavailable(task_missing)``.
    3. Pure Python authorization against the row step 2 loaded: a
       ``"user"`` principal must own the task or be an admin; a
       ``"guest"`` principal must satisfy the shared
       ``task_is_owned_by_public_principal`` predicate. A principal whose
       ``kind`` is neither ``"user"`` nor ``"guest"`` is always
       unauthorized -- there is no third branch that defaults to allow. A
       malformed guest principal that populates zero or more than one
       entity-binding field makes the ownership predicate raise
       ``ValueError``; this function catches only that one exception type
       from that one call and treats it as unauthorized -- the same
       fail-closed-on-a-malformed-caller behavior the predicate itself
       documents and ``create()`` applies to the same two cases. Runs
       before the idempotency pre-read (step 5) so an unauthorized caller
       can never use a guessed idempotency key to read back someone else's
       receipt.
    4. Load the interaction row by ``(id, task_id)`` -- absent,
       ``Unavailable(interaction_missing)``. Present but its own ``kind`` /
       ``protocol_version`` columns disagree with ``envelope``'s,
       ``ValidationRejected(kind_version_mismatch)``.
    5. Idempotency pre-read against ``task_execution_commands``: a hit
       matching this call's payload is ``Replayed`` (built from the current
       row state); a hit that does not match is
       ``Conflict(idempotency_key_reused)``. Runs before anchor resolution:
       an answered row's anchor is prunable, and replay recognition must
       not depend on one.
    5.5. Resolve the row's resume anchor via
       ``_resolve_read_direction_anchor`` -- ``checkpoint_unavailable`` maps
       to ``Unavailable``, ``anchor_dangling`` to ``Stale``. Never falls back
       to a legacy scan on failure (see that resolver's own docstring for
       why).
    6. The answer fence UPDATE. ``rowcount == 1`` continues; ``rowcount == 0``
       rereads the interaction row -- which also confirms it did not
       disappear out from under this transaction's own row lock -- and
       classifies the miss: already-answered replay/conflict, three
       terminated reasons, wrong task state, foreign run, or an ownership
       miss. That last one is reachable on both backends, not only on
       SQLite: ``_answer_fence_task_predicate``'s ``Task.user_id`` term
       requires the owner in person, so an admin acting on another user's
       task and a guest whose ``principal.user_id`` is not the owner's
       both reach it on PostgreSQL too, alongside SQLite's own concurrent
       ownership change and, for a guest, SQLite's own concurrent write to
       ``agent_config["guest_id"]`` between step 2's read and the fence
       (see that predicate's own docstring). The reread row's state is
       logged before the classification runs, because no outcome variant
       carries those columns. ``rowcount > 1`` is a schema invariant
       violation (``uq_task_interaction_active_slot``) and raises.
    7. The Task CAS via ``apply_task_control_transition``, called with no
       ``expected_run_id`` / ``expected_state_version`` -- this function
       takes no caller-supplied optimistic-concurrency token, so neither of
       that helper's own staleness checks can fire. ``status`` is
       deliberately never passed either; flipping ``Task.status`` is the
       resume coordinator's job, not this function's. The one remaining way
       ``apply_task_control_transition`` can raise ``StaleTaskRunError`` --
       its own atomic UPDATE matching zero rows -- is unreachable here: step
       2's ``FOR NO KEY UPDATE`` already holds this exact ``tasks`` row for
       the rest of the transaction, so the CAS's ``Task.id == task_id``
       WHERE clause cannot fail to match. This call is therefore left
       uncaught; ``StaleTaskRunError`` surfacing here would mean that
       invariant broke, not a normal stale-answer outcome.
    8. Stage the RESUME command inside this function's own
       ``db.begin_nested()``, and require the result to be a row this call
       itself created carrying this call's own payload. Two different
       signals report the same race -- a second writer took this
       idempotency key between step 5's pre-read and this statement -- and
       both are classified the same way. An ``IntegrityError`` is raised
       when that writer's row lands on the unique constraint;
       ``created=False`` is returned instead when the row was already
       committed and visible, because ``stage_task_command`` checks for an
       existing row before inserting and returns it rather than raising.
       Either way this function rolls back to its savepoint (not the whole
       transaction -- see ``classify_task_command_conflict``'s own
       docstring for why a savepoint rollback satisfies the
       post-rollback-state precondition it documents) and asks one
       question about the row that won: does it carry this call's own
       ``actor_user_id``, kind and canonical payload? The
       ``IntegrityError`` door answers it through
       ``classify_task_command_conflict``, the ``created=False`` door
       through the ``payload_matches`` ``stage_task_command`` already
       computed; both compute it with the same ``_matches_existing``, so
       the two doors cannot disagree. A match means the RESUME that will
       execute carries exactly this answer, so this call commits its own
       fence UPDATE and CAS and reports ``Replayed`` naming the winner's
       row -- not ``Accepted``, because the command row is the other
       writer's. A mismatch means two different answers under one key: the
       whole transaction rolls back, undoing this call's own fence UPDATE
       and CAS along with it, and this call reports
       ``Conflict(idempotency_key_reused)``. A ``TASK_MISSING``
       classification maps to ``Unavailable(task_missing)``; an
       ``UNRELATED`` one is re-raised, because it means a foreign key
       other than this call's own duplicate failed.
    9. Commit. A raised exception here does not mean the write failed --
       the acknowledgment could have been lost after the server applied it
       -- so this function retires its session and checks the durable graph
       in a fresh one (``_verify_respond_durable_graph``) before deciding
       between ``Accepted`` and ``OutcomeUnknown``.
    10. After a successful commit, outside any transaction:
        ``_notify_dispatcher_best_effort()``. A raise there would turn a
        committed answer into a reported failure, so it is caught, logged
        as a warning, and the dispatcher's idle poll delivers the command
        instead. Both post-commit exits go through it -- step 9's own
        commit and step 9's reconciliation recovering a lost
        acknowledgment -- because the answer is equally durable on both.

    What this function lets escape, deliberately, and what it does not.
    The eight ``RespondOutcome`` variants cover every outcome this build
    classifies; they do not cover operational failure. Three families are
    left to propagate rather than folded into ``OutcomeUnknown``, because
    a caller that cannot tell "the database is down" from "your answer was
    ambiguous" will retry the first one forever:

    - Database-level failures outside the units caught above -- a
      deadlock, a lost connection, a pool checkout timeout -- raised by
      any statement from step 2 onward. Three of the four catches are
      narrow on purpose: step 8's is ``IntegrityError`` only (and it
      re-raises an ``UNRELATED`` classification rather than swallowing
      it), step 9's covers the commit only, and step 10's wraps one
      post-commit best-effort notification whose failure is logged and
      swallowed. The fourth is the broad ``except Exception`` inside
      ``_verify_respond_durable_graph``, which is deliberately broad and
      does not widen this surface: it runs only after step 9 already
      caught a commit exception, its own failure is logged per attempt,
      and exhausting all three attempts returns ``None``, which this
      function reports as ``OutcomeUnknown`` -- exactly the answer it
      would have given had the reconciliation not existed.
    - ``TaskCommandOwnerStateError`` and ``TaskCommandTaskMissing`` from
      ``stage_task_command``. Neither is an ``IntegrityError`` subclass, so
      step 8's catch does not see them, and both mean a precondition this
      function already checked has changed underneath it.
    - ``RuntimeError`` from the five structural invariants asserted above:
      a fence rowcount above one, an interaction row that disappeared
      while this transaction held the tasks row lock, a
      ``RACED_DUPLICATE`` classification arriving from step 8 with no
      raced projection attached, and -- both added by step 6's
      classification -- a reread row that is ``terminated`` under a
      ``terminal_reason`` outside the three this service maps, or one
      whose ``status`` is a value this module does not know. The last two
      are deliberately loud rather than folded into ``OutcomeUnknown``:
      each means the interaction table grew a state this classification
      has not been taught, and answering "we could not confirm this went
      through" would hide a schema change behind a normal-looking result.

    On every one of those paths the ``finally`` below retires the session,
    which rolls back an uncommitted transaction, so an escaping exception
    leaves no partial write behind. ``StaleTaskRunError`` from step 7 is
    documented above as unreachable and is in this same category.
    """

    if not isinstance(envelope.kind, str) or envelope.kind not in _KIND_VOCABULARY:
        return RespondValidationRejected(reason="unknown_kind")
    if (
        not isinstance(envelope.protocol_version, int)
        or isinstance(envelope.protocol_version, bool)
        or envelope.protocol_version != INTERACTION_PROTOCOL_VERSION
    ):
        return RespondValidationRejected(reason="unknown_protocol_version")
    if not isinstance(envelope.idempotency_key, str):
        return RespondValidationRejected(reason="malformed_idempotency_key")
    try:
        normalized_key = _normalize_command_id(envelope.idempotency_key)
    except ValueError:
        return RespondValidationRejected(reason="malformed_idempotency_key")
    if not isinstance(envelope.values, dict):
        return RespondValidationRejected(reason="invalid_values")
    try:
        json.dumps(envelope.values, allow_nan=False, sort_keys=True)
    except (TypeError, ValueError):
        # The same probe ``build_v1_request_payload`` runs on the question
        # side (see its own docstring), applied here for the two failure
        # modes this side has. A ``values`` dict holding a ``datetime``,
        # ``set``, ``bytes``, or any other object the JSON encoder does not
        # know raises ``TypeError`` at bind time, inside the fence UPDATE
        # -- far past every typed return above, so it would leave this
        # function through ``sqlalchemy.exc.StatementError`` instead of one
        # of the eight ``RespondOutcome`` variants. A float that is ``nan``
        # or an infinity is worse because it is silent: the default encoder
        # renders it as the bare ``NaN`` / ``Infinity`` tokens, which are
        # not JSON, and stores them. ``allow_nan=False`` turns that second
        # case into the ``ValueError`` caught here. Structural, like every
        # other check in step 1 -- it asks whether these values can be
        # stored at all, not whether they answer this particular question.
        # ``sort_keys=True`` matches ``_canonical_payload``
        # (``task_command_transport.py``), which sorts keys recursively: a
        # dict mixing int and str keys passes an unsorted dump (int keys
        # are silently coerced to strings -- the same silent-corruption
        # class as the nan case) and then raises ``TypeError`` inside the
        # replay comparison. The probe has to be a strict superset of every
        # serializer downstream of step 1.
        return RespondValidationRejected(reason="invalid_values")

    from ..models.database import get_session_local

    SessionLocal = get_session_local()
    db: "Session" = SessionLocal()
    session_retired = False
    try:
        task = (
            db.execute(
                sa.select(Task)
                .where(Task.id == task_id)
                .with_for_update(key_share=True)
            )
            .scalars()
            .first()
        )
        if task is None:
            return RespondUnavailable(reason="task_missing")

        if principal.kind == "user":
            # user_id is required of both branches, not only of the owner
            # comparison: an admin authorized on the flag alone would
            # otherwise reach step 8 with no identity to write into
            # responder_identity or actor_user_id.
            authorized = principal.user_id is not None and (
                principal.is_admin or task.user_id == principal.user_id
            )
        elif principal.kind == "guest":
            try:
                authorized = task_is_owned_by_public_principal(task, principal)
            except ValueError:
                authorized = False
        else:
            authorized = False
        if not authorized:
            return RespondUnauthorized(reason="not_task_principal")

        ir = get(db, task_id=task_id, interaction_id=interaction_id)
        if ir is None:
            return RespondUnavailable(reason="interaction_missing")
        if ir.kind != envelope.kind or ir.protocol_version != envelope.protocol_version:
            return RespondValidationRejected(reason="kind_version_mismatch")

        command_payload = _respond_command_payload(
            interaction_id=interaction_id, principal=principal, values=envelope.values
        )
        canonical_submitted_values = _canonical_payload(envelope.values)
        actor_user_id = principal.user_id

        # Runs before anchor resolution below, not after it. Answering
        # clears ``active_slot``, and the checkpoint retention pruner only
        # protects rows whose ``active_slot`` is still set
        # (``trace_handlers.py``), so an answered row's anchor becomes
        # prunable the moment this service answers it and the foreign key's
        # ``ON DELETE SET NULL`` then empties the pointer. With anchor
        # resolution first, a retry arriving after that pruning would be
        # told ``Stale(anchor_dangling)`` about an answer that was in fact
        # accepted, and would never reach the replay branch below. Replay
        # recognition is a question about this call's idempotency key and
        # the row it already wrote; it does not need a live anchor, and
        # must not be gated on one. Still after step 3's authorization, so
        # an unauthorized caller can never use a guessed idempotency key to
        # read back someone else's receipt.
        existing_command = (
            db.query(TaskExecutionCommand)
            .filter(
                TaskExecutionCommand.task_id == task_id,
                TaskExecutionCommand.command_id == normalized_key,
            )
            .first()
        )
        if existing_command is not None:
            if _matches_existing(
                existing_command,
                actor_user_id=actor_user_id,
                kind=TaskCommandKind.RESUME,
                payload=command_payload,
            ):
                return RespondReplayed(
                    receipt=_respond_receipt(
                        interaction=ir,
                        task=task,
                        command_db_id=int(existing_command.id),
                        idempotency_key=normalized_key,
                    )
                )
            increment_counter(COUNTER_LIFECYCLE_RESPONSE_CONFLICT)
            return RespondConflict(reason="idempotency_key_reused")

        unresolved = _resolve_read_direction_anchor(db, ir)
        if unresolved is not None:
            if unresolved.reason == "checkpoint_unavailable":
                return RespondUnavailable(reason="checkpoint_unavailable")
            return RespondStale(reason="anchor_dangling")

        now = datetime.now(timezone.utc)
        responder_user_id = principal.user_id if principal.kind == "user" else None
        fence_result = db.execute(
            _answer_fence_stmt(
                interaction_id=interaction_id,
                task_id=task_id,
                principal=principal,
                response_payload=envelope.values,
                now=now,
                responder_user_id=responder_user_id,
                responder_identity=principal.identity_string(),
            )
        )
        rowcount = int(getattr(fence_result, "rowcount", 0) or 0)
        if rowcount > 1:
            raise RuntimeError(
                f"answer fence updated {rowcount} rows for interaction "
                f"{interaction_id} on task {task_id}; "
                "uq_task_interaction_active_slot makes this impossible"
            )
        if rowcount == 0:
            # The fence UPDATE's own re-evaluation just proved this session's
            # identity map is stale for this row (rowcount 0 means the row
            # changed since step 4's read); without expiring first, the
            # ORM would hand this query's result back through the same
            # already-loaded, now-stale Python object instead of the fresh
            # row this reread exists to see. This has to expire the whole
            # session, not just ``ir`` -- the classification below also
            # reads ``task.status`` and ``task.run_id`` (the "active"
            # branch further down), and those need refreshing too: without
            # it, ``task.status`` would still read back step 2's own
            # WAITING_FOR_USER value even after some other writer moved the
            # task on, and a genuine ``run_ended`` miss would misclassify
            # as an ownership miss (``not_task_principal``) instead.
            db.expire_all()
            reread = get(db, task_id=task_id, interaction_id=interaction_id)
            if reread is None:
                raise RuntimeError(
                    f"interaction {interaction_id} on task {task_id} disappeared "
                    "while this transaction held the tasks row lock"
                )
            # The reread above already has the row in hand, so recording
            # what it found costs no extra statement. The classification
            # below tells the *caller* which miss this was; this line tells
            # an *operator* what the row actually looked like, and the two
            # are not the same information. The reason it hands back
            # describes the row and task as this reread found them, not
            # necessarily the exact condition that made the fence UPDATE's
            # own WHERE clause fail to match: neither backend holds a lock
            # across the gap between that UPDATE and this SELECT, so in
            # principle the row could change again in between and the
            # label would name whatever this reread actually saw, not the
            # original miss. No ``RespondOutcome`` variant
            # carries ``active_slot``, ``terminal_reason``, ``run_id`` or
            # ``responder_identity``: ``Stale(run_superseded)`` names the
            # reason without naming the run, and ``Conflict
            # (already_answered)`` names neither who answered nor when. A
            # fence miss is also still the one exit a retry cannot clarify
            # on its own -- a terminated, superseded, or foreign-owned row
            # produces the same miss every time, unlike the commit-exception
            # door below, where a retry under the same idempotency key
            # resolves the ambiguity by itself -- so the row state at the
            # moment of the miss is worth a line whether or not the caller
            # got a specific reason back. Logged before the classification
            # runs so that the two ``RuntimeError`` exits below (an
            # unrecognized ``terminal_reason``, an unrecognized ``status``)
            # are covered by it too.
            logger.warning(
                "answer fence matched zero rows for interaction %s on task "
                "%s; reread status=%s active_slot=%s terminal_reason=%s "
                "run_id=%s responder_identity=%s",
                interaction_id,
                task_id,
                reread.status,
                reread.active_slot,
                reread.terminal_reason,
                reread.run_id,
                reread.responder_identity,
            )
            if reread.status == "answered":
                fresh_command = (
                    db.query(TaskExecutionCommand)
                    .filter(
                        TaskExecutionCommand.task_id == task_id,
                        TaskExecutionCommand.command_id == normalized_key,
                    )
                    .first()
                )
                if fresh_command is not None:
                    # A command row under this exact idempotency key exists
                    # now, even though step 5's pre-read (run before the
                    # fence attempt) found none: the row that answered this
                    # interaction was staged concurrently, in the same
                    # window this call's own fence attempt lost. Whether
                    # that command is this call's own request replaying, or
                    # a different payload racing under the same reused key,
                    # is exactly the question step 5's own two-way split
                    # answers, so this reread applies the same test: a
                    # payload match is a replay, a mismatch is
                    # ``idempotency_key_reused`` (see step 5 above).
                    if _matches_existing(
                        fresh_command,
                        actor_user_id=actor_user_id,
                        kind=TaskCommandKind.RESUME,
                        payload=command_payload,
                    ):
                        return RespondReplayed(
                            receipt=_respond_receipt(
                                interaction=reread,
                                task=task,
                                command_db_id=int(fresh_command.id),
                                idempotency_key=normalized_key,
                            )
                        )
                    increment_counter(COUNTER_LIFECYCLE_RESPONSE_CONFLICT)
                    return RespondConflict(reason="idempotency_key_reused")
                # No command row exists under this key at all: some other,
                # unrelated write answered this row (not one racing this
                # call's own idempotency key), so there is nothing to
                # replay or compare payloads against.
                increment_counter(COUNTER_LIFECYCLE_RESPONSE_CONFLICT)
                return RespondConflict(reason="already_answered")
            if reread.status == "terminated":
                # No ``or ""`` fallback needed on the lookup key --
                # ``ck_task_interaction_requests_terminal_pairs_status`` is
                # a biconditional (``status = 'terminated'`` iff
                # ``terminal_reason IS NOT NULL``), so a row that reaches
                # this branch is guaranteed by the database itself to carry
                # a non-``None`` ``terminal_reason``. The ``cast`` below is
                # only for mypy's benefit (the mapped column's static type
                # is ``str | None``); it asserts that guarantee, it does
                # not substitute a runtime value the way ``or ""`` did.
                terminal_stale_reasons: dict[str, RespondStaleReason] = {
                    "deadline_elapsed": "expired",
                    "run_superseded": "run_superseded",
                    "answered_via_legacy_resume": "answered_via_chat",
                }
                mapped_reason = terminal_stale_reasons.get(
                    cast(str, reread.terminal_reason)
                )
                if mapped_reason is None:
                    raise RuntimeError(
                        f"interaction {interaction_id} on task {task_id} is "
                        f"terminated with an unrecognized terminal_reason "
                        f"{reread.terminal_reason!r}"
                    )
                return RespondStale(reason=mapped_reason)
            if reread.status == "active":
                # Same order as the fence's own WHERE clause narrows: the
                # task-level terms first, the ownership term last. Reaching
                # the final return means the row is still live and the task
                # is still waiting on this very run, so the only fence term
                # left that can have failed is
                # ``_answer_fence_task_predicate``'s ownership conjunction
                # -- which requires ``principal.user_id`` to be the task's
                # owner on both backends. Four callers land here: an admin
                # answering someone else's task, a guest whose bindings
                # match but whose ``principal.user_id`` is not the owner's
                # (both of which pass step 3's Python authorization and
                # fail only at the write point), and, on SQLite alone,
                # ``Task.user_id`` changing between step 2's read and this
                # statement, or, also SQLite-only, a guest's
                # ``agent_config["guest_id"]`` changing in that same window.
                # See ``_answer_fence_task_predicate``'s own docstring for
                # which of its six terms are re-asserted in SQL and which
                # are checked once in Python.
                if task.status != TaskStatus.WAITING_FOR_USER:
                    # Covers every other ``TaskStatus`` member: PENDING,
                    # RUNNING, PAUSED, COMPLETED, FAILED -- five non-waiting
                    # states this one comparison treats alike, without
                    # distinguishing which of them the task actually moved
                    # to.
                    return RespondStale(reason="run_ended")
                if task.run_id != reread.run_id:
                    return RespondStale(reason="foreign_run")
                return RespondUnauthorized(reason="not_task_principal")
            raise RuntimeError(
                f"interaction {interaction_id} on task {task_id} has an "
                f"unrecognized status {reread.status!r} after a zero-rowcount "
                "answer fence"
            )

        # No expected_run_id/expected_state_version to pass: this function
        # takes no caller-supplied optimistic-concurrency token, so
        # apply_task_control_transition's own staleness checks never fire.
        # StaleTaskRunError from its atomic UPDATE matching zero rows is
        # provably unreachable here -- step 2's row lock is still held --
        # and is deliberately left uncaught (see this function's own
        # docstring, step 7).
        apply_task_control_transition(task, TaskControlState.RESUME_REQUESTED)

        expected_state_version_after = int(task.state_version)
        run_id_for_verification = str(task.run_id)
        savepoint = db.begin_nested()
        # Capture every receipt value this transaction can still commit as
        # plain Python locals now, before the commit below. SQLAlchemy's
        # default ``expire_on_commit=True`` invalidates every ORM attribute
        # on ``ir``/``task`` the instant ``db.commit()`` returns -- reading
        # ``ir.run_id`` or ``task.state_version`` afterward would silently
        # re-issue a SELECT against a session this function is about to
        # decide whether to keep or retire. The fence UPDATE above is a
        # Core statement, so ``ir``'s in-memory attributes are not synced
        # by it; ``task``'s, by contrast, already are --
        # ``apply_task_control_transition`` re-reads it with its own
        # ``session.refresh(task)`` right after its atomic UPDATE. Either
        # way these locals still have to be captured now: ``expire_on_commit``
        # is what invalidates them the instant ``db.commit()`` returns, not
        # whether this transaction has kept them current up to this point.
        answered_run_id = str(ir.run_id)
        answered_responder_identity = principal.identity_string()
        answered_responded_at = now
        committed_state_version = int(task.state_version)
        committed_control_state = str(task.control_state)

        # Both doors of step 8's race funnel into these two locals, so the
        # decision below is written once. ``None`` means no race: this call
        # created the row itself.
        raced_command_db_id: int | None = None
        raced_payload_matches = False
        try:
            staged = stage_task_command(
                db,
                task_id=task_id,
                actor_user_id=actor_user_id,
                command_id=normalized_key,
                kind=TaskCommandKind.RESUME,
                payload=command_payload,
            )
        except IntegrityError:
            # Door one: the other writer's row was not visible at
            # ``stage_task_command``'s own existing-row check and landed on
            # the unique constraint at the flush. Roll back to this
            # function's own savepoint rather than the whole transaction,
            # so the fence UPDATE and the CAS survive long enough for the
            # classification below to decide whether they should commit.
            savepoint.rollback()
            classification = classify_task_command_conflict(
                db,
                task_id=task_id,
                command_id=normalized_key,
                actor_user_id=actor_user_id,
                kind=TaskCommandKind.RESUME,
                payload=command_payload,
            )
            if classification.kind == TaskCommandConflictKind.TASK_MISSING:
                db.rollback()
                return RespondUnavailable(reason="task_missing")
            if classification.kind != TaskCommandConflictKind.RACED_DUPLICATE:
                # ``UNRELATED``: no duplicate row exists and the task is
                # still there, so some other constraint on the row this
                # call tried to insert failed -- most plausibly the
                # ``actor_user_id`` foreign key. That is not a race this
                # function has an answer for, and folding it into a typed
                # outcome would tell the caller "your answer was
                # ambiguous" about a database-level failure. Re-raised.
                db.rollback()
                raise
            if classification.raced is None:
                raise RuntimeError(
                    f"classify_task_command_conflict reported RACED_DUPLICATE "
                    f"for interaction {interaction_id} on task {task_id} "
                    "with no raced projection attached; "
                    "TaskCommandConflictClassification's own constructor "
                    "always pairs RACED_DUPLICATE with a raced projection, "
                    "making this impossible"
                )
            raced_command_db_id = classification.raced.command_db_id
            raced_payload_matches = classification.raced.payload_matches
        else:
            if not (staged.created and staged.payload_matches):
                # Door two: the same race, seen one statement earlier.
                # ``stage_task_command`` does not raise when a row for this
                # ``(task_id, command_id)`` already exists -- it returns
                # that row with ``created=False`` -- so without this branch
                # the transaction would commit the fence and the CAS while
                # the RESUME carrying this answer was never staged at all,
                # and return ``RespondAccepted`` naming the other writer's
                # row. ``created=True`` is only ever returned together with
                # ``payload_matches=True`` (see ``stage_task_command``), so
                # reaching here means ``created=False`` and
                # ``staged.payload_matches`` is the verdict on the winner's
                # row -- computed by the same ``_matches_existing``
                # ``classify_task_command_conflict`` uses on the other
                # door, which is why the two doors cannot disagree and are
                # decided together below. The savepoint holds nothing at
                # this point (this call inserted nothing), and is rolled
                # back anyway to keep both doors on one shape.
                savepoint.rollback()
                raced_command_db_id = staged.staged_db_id
                raced_payload_matches = staged.payload_matches

        if raced_command_db_id is not None:
            if not raced_payload_matches:
                # Two different answers under one idempotency key. The
                # whole transaction rolls back, undoing this call's own
                # fence UPDATE and CAS along with it.
                db.rollback()
                increment_counter(COUNTER_LIFECYCLE_RESPONSE_CONFLICT)
                return RespondConflict(reason="idempotency_key_reused")
            # The winner's row carries this call's own actor, kind and
            # canonical payload, so the RESUME that executes is this
            # answer. Commit the fence UPDATE and the CAS -- they are what
            # makes the interaction row agree with the command that will
            # run -- and report ``Replayed`` naming the winner's row rather
            # than ``Accepted``, because the command row is not this
            # call's. No dispatcher notification is sent here: the writer
            # that staged that row owes it (see ``stage_task_command``'s
            # caller obligation (a)), and the dispatcher's idle poll is the
            # documented fallback either way.
            try:
                db.commit()
            except Exception:
                # Same ambiguity as step 9's commit below, on the other
                # door of this function's single race funnel: the fence
                # UPDATE and CAS this call issued may have landed durably
                # even though this call never saw the acknowledgment. The
                # winner's command row is not this transaction's own write
                # and is not rolled back by this exception, so a bare
                # re-raise would turn an already-committed answer into a
                # crash loop on retry -- a retry under the same
                # idempotency key would land on the replay branch above
                # and hit this same commit again. Reconcile against a
                # fresh session instead, exactly as step 9 does.
                logger.warning(
                    "commit failed while answering interaction %s on task %s; "
                    "the write may or may not be durable -- reconciling against "
                    "the durable graph",
                    interaction_id,
                    task_id,
                    exc_info=True,
                )
                session_retired = True
                _retire_respond_session_best_effort(db)
                receipt = _verify_respond_durable_graph(
                    task_id=task_id,
                    interaction_id=interaction_id,
                    expected_run_id=run_id_for_verification,
                    expected_state_version_after=expected_state_version_after,
                    principal=principal,
                    canonical_submitted_values=canonical_submitted_values,
                    command_id=normalized_key,
                    command_kind=TaskCommandKind.RESUME,
                    command_payload=command_payload,
                    actor_user_id=actor_user_id,
                )
                if receipt is not None:
                    return RespondReplayed(receipt=receipt)
                return RespondOutcomeUnknown()
            return RespondReplayed(
                receipt=InteractionResponseReceipt(
                    interaction_id=interaction_id,
                    task_id=task_id,
                    run_id=answered_run_id,
                    status="answered",
                    responded_at=answered_responded_at,
                    responder_identity=answered_responder_identity,
                    idempotency_key=normalized_key,
                    command_db_id=raced_command_db_id,
                    task_state_version=committed_state_version,
                    task_control_state=committed_control_state,
                )
            )

        command_db_id = staged.staged_db_id
        try:
            db.commit()
        except Exception:
            # A raised exception here does not mean the write failed -- the
            # acknowledgment could have been lost after the server applied
            # it -- so this call retires its session and asks the durable
            # graph, in a fresh one, what actually landed. Logged first,
            # unconditionally: the reconciliation below can turn this into
            # an ``Accepted``, but an operator still needs the record that
            # a commit acknowledgment went missing at all, and the
            # reconciliation's own per-attempt logs only fire when *it*
            # fails.
            logger.warning(
                "commit failed while answering interaction %s on task %s; "
                "the write may or may not be durable -- reconciling against "
                "the durable graph",
                interaction_id,
                task_id,
                exc_info=True,
            )
            session_retired = True
            _retire_respond_session_best_effort(db)
            receipt = _verify_respond_durable_graph(
                task_id=task_id,
                interaction_id=interaction_id,
                expected_run_id=run_id_for_verification,
                expected_state_version_after=expected_state_version_after,
                principal=principal,
                canonical_submitted_values=canonical_submitted_values,
                command_id=normalized_key,
                command_kind=TaskCommandKind.RESUME,
                command_payload=command_payload,
                actor_user_id=actor_user_id,
            )
            if receipt is not None:
                _notify_dispatcher_best_effort(
                    interaction_id=interaction_id, task_id=task_id
                )
                return RespondAccepted(receipt=receipt)
            return RespondOutcomeUnknown()

        _notify_dispatcher_best_effort(interaction_id=interaction_id, task_id=task_id)
        return RespondAccepted(
            receipt=InteractionResponseReceipt(
                interaction_id=interaction_id,
                task_id=task_id,
                run_id=answered_run_id,
                status="answered",
                responded_at=answered_responded_at,
                responder_identity=answered_responder_identity,
                idempotency_key=normalized_key,
                command_db_id=command_db_id,
                task_state_version=committed_state_version,
                task_control_state=committed_control_state,
            )
        )
    finally:
        if not session_retired:
            _retire_respond_session_best_effort(db)
