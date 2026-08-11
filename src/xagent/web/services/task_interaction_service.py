"""Typed lifecycle service for blocking task interaction requests: the
answer-side counterpart to ``task_interaction_staging.py``'s ask-side
primitive.

Not merged into ``task_interaction_staging.py``. That module's own
docstring pins its merge reason to a fact this module does not share: its
two entry points exist in one file because they share every exception type
and one nesting invariant that must never be split across a file boundary,
both live on the ask side, and both require the caller to already hold the
transaction they run inside. This module's answering seam is the opposite
shape on every one of those points -- it owns its own session and its own
commit, and it does not nest inside anyone else's savepoint. Putting it in
the same file would make that staging module's merge-reason docstring false
the day this file lands. What this module reuses instead of duplicating is
narrow: the staging module's exception base class and its anchor
self-consistency check, imported by name, not copied.

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
does not stage a row); ``get()``/``list()``; and the three-tier compatibility
materialization view. Not delivered here: ``respond()``'s call body, the
answer fence, the compatibility seam into the existing resume coordinator,
and any new counter. Those land with later changes; this module's own
zero-production-caller gate
(``tests/web/services/test_task_interaction_service_production_gate.py``)
is what keeps that boundary enforced rather than aspirational.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError as _PydanticValidationError

from ...core.tools.adapters.vibe.ask_user_tool import AskUserQuestionArgs
from ..models.task import Task
from ..models.task_interaction import INTERACTION_PROTOCOL_VERSION
from .task_command_transport import _normalize_command_id
from .task_interaction_staging import _KIND_VOCABULARY

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
    runs as the task owner; ``is_admin`` is audit-only). ``channel_id``,
    ``auth_mode``, the four ``*_id`` entity-binding fields, and ``guest_id``
    mirror the fields ``public_chat_access.py``'s access-context dataclasses
    already carry (see that module's ``PublicChatAccessContext`` and
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
    channel_id: int | None
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


def task_is_owned_by_public_principal(
    task: "Task", principal: InteractionPrincipal
) -> bool:
    """The shared conjunction extracted from ``public_chat_access.py``'s
    four widget/share ownership checks (``_get_task_for_workforce_widget_context``,
    ``get_task_for_public_context``, ``_get_task_for_workforce_share_context``,
    ``get_task_for_share_context``), plus two deliberate tightenings over
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
       one of the two tightenings: the pre-existing widget-agent branch of
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
       config value.
       - widget-agent: ``task.agent_id == principal.widget_agent_id``
         (row-level only; the pre-existing code has no JSON-level widget
         entity binding to mirror -- see the candidate issue below).
       - widget-workforce: ``task.agent_config["widget_workforce_id"] ==
         principal.widget_workforce_id`` (JSON-level only).
       - share-agent: **both** ``task.agent_id ==
         principal.share_agent_id`` (row-level) **and**
         ``task.agent_config["share_agent_id"] == principal.share_agent_id``
         (JSON-level) -- the pre-existing ``get_task_for_share_context`` is
         the one entry point that checks both.
       - share-workforce: ``task.agent_config["share_workforce_id"] ==
         principal.share_workforce_id`` (JSON-level only).
    5. ``task.agent_config["guest_id"] == principal.guest_id``, with
       ``principal.guest_id`` required non-empty first -- an empty or
       ``None`` guest id must never match a task whose own ``guest_id`` is
       also empty or missing.

    Candidate issue, logged and not fixed here: the widget-agent direction
    (``get_task_for_public_context``, not routed through this predicate)
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
        config_value = task.agent_config.get("widget_workforce_id")
        if config_value is None or int(config_value) != principal.widget_workforce_id:
            return False
    elif direction == "share_agent":
        if task.agent_id is None or task.agent_id != principal.share_agent_id:
            return False
        config_value = task.agent_config.get("share_agent_id")
        if config_value is None or int(config_value) != principal.share_agent_id:
            return False
    else:  # share_workforce
        config_value = task.agent_config.get("share_workforce_id")
        if config_value is None or int(config_value) != principal.share_workforce_id:
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

    Message-selection priority under a hypothetical *simultaneous* identity
    **and** entity-binding failure is not preserved identically across all
    three pre-existing functions by this split: the pre-existing
    widget-workforce code checked guest_id before entity binding (identity
    wins), while both pre-existing share-side functions checked entity
    binding before guest_id (entity wins). This helper always gives
    identity priority, matching the widget-workforce order and diverging
    from the share-side order only when both dimensions are wrong at once
    -- a combination no caller here can construct today (auth_mode alone
    already separates the widget and share channels) and no existing test
    exercises. The security property that matters -- a guest_id mismatch
    alone is indistinguishable from not-found -- holds under either
    priority.
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

    Retired by the wiring batch's ignition PR, which fills this seam's call
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
# counted separately). Do not recount this at implementation time -- it is
# taken directly from the frozen design's own line-by-line count.
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
# itself -- is reachable. 7 pairs. Once the wiring batch fills create()'s
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
# validate an incoming envelope's ``values``; the wiring batch's write side
# is the intended future importer of the construct half, so that the
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

    if isinstance(values, AskUserQuestionArgs):
        return values
    return AskUserQuestionArgs.model_validate(values)


def build_v1_request_payload(parsed: AskUserQuestionArgs) -> dict[str, Any]:
    """Render a validated ``AskUserQuestionArgs`` instance into the exact
    JSON-shaped dict ``stage_interaction_request``'s ``request_payload``
    parameter expects."""

    return parsed.model_dump(mode="json")


# TTL policy interval for a create() envelope's optional ttl_seconds
# override. The frozen design requires this bound to be enforced in the
# facade (clamping silently would be fail-open and is explicitly rejected
# in favor of an outright validation failure), but does not pin concrete
# numbers -- it describes the requirement as "a [min, max] policy interval"
# without giving one. No existing config or constant anywhere in this
# codebase defines an interaction TTL policy today. The two bounds below
# are this delivery's own placeholder, not a fact recovered from source,
# logs, or the database -- flagged here, and in this delivery's own report,
# as a value that needs an explicit policy decision, not a discovered one.
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
    together -- i.e. the wiring batch's ignition PR (W5), not this one.

    Validation order, each step short-circuiting on the first failure:
    ``kind`` and ``protocol_version`` against the v1 vocabulary and
    version, ``request_idempotency_key`` against
    ``COMMAND_ID_PATTERN`` (via ``task_command_transport``'s own
    normalizer, not a copy of its regex), ``values`` against the v1
    ``request_payload`` contract, and an optional ``ttl_seconds`` against
    this facade's policy interval -- out of range is a rejection, never a
    silent clamp. Authorization runs only after every one of those passes,
    and only against a task this call itself loads by id: a ``"user"``
    principal must own the task or be an admin; a ``"guest"`` principal is
    checked with the same shared ownership predicate ``respond()`` will
    reuse (``task_is_owned_by_public_principal``), not a re-derived
    conjunction. A task that does not exist at all is ``Unavailable``, not
    ``Unauthorized`` -- that branch is reached before the ownership check
    even runs, because there is no row to check ownership against.

    ``origin`` is deliberately not part of this envelope or this
    validation step in this delivery: the frozen design's own reason
    vocabulary has no origin-related entry, and ``stage_interaction_request``
    (which this seam does not call) already validates it against the
    model's public vocabulary when the wiring batch does call it. Adding an
    origin check here now would validate a field this seam never uses for
    anything.

    Of the staging module's nine exception classes, three
    (``InteractionAttemptMismatch``, ``InteractionHandoffMisuse``,
    ``InteractionOriginUnknown``) are raised only from inside
    ``_InteractionHandoff``'s own call path and are therefore unreachable
    from this function, which never enters that context manager -- this
    function has no call to any of the staging module's exception-raising
    code at all in this delivery, since it never calls
    ``stage_interaction_request``.
    """

    if envelope.kind not in _KIND_VOCABULARY:
        return CreateValidationRejected(reason="unknown_kind")
    if envelope.protocol_version != INTERACTION_PROTOCOL_VERSION:
        return CreateValidationRejected(reason="unknown_protocol_version")
    try:
        _normalize_command_id(envelope.request_idempotency_key)
    except ValueError:
        return CreateValidationRejected(reason="malformed_idempotency_key")
    try:
        parse_v1_request_payload(envelope.values)
    except _PydanticValidationError:
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
