"""Optional application hooks for team-owned MCP and Custom API connectors.

Standalone xagent keeps connectors user-owned. A multi-tenant application can
install these hooks to overlay team visibility without teaching xagent about
the application's team tables.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Collection, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypeVar

from sqlalchemy.sql.elements import ColumnElement

if TYPE_CHECKING:
    from ..models.custom_api import UserCustomApi
    from ..models.mcp import UserMCPServer

from ...core.tools.adapters.vibe.connector_runtime import (
    ERROR_CONNECTOR_RUNTIME_UNAVAILABLE,
    ConnectorRuntimeError,
)

logger = logging.getLogger(__name__)

ConnectorType = Literal["mcp", "custom_api"]
ConnectorRenamedHook = Callable[[Any, int, ConnectorType, int, str, str], None]


@dataclass(frozen=True)
class ConnectorDeleteDecision:
    team_owned: bool = False
    authorized: bool = False
    delete_definition: bool = False
    # Set when the delete is refused because the connector is still selected by a
    # team agent. The endpoint surfaces this as a 403 before any mutation, mirroring
    # the unshare path's "still used by a team agent" guard.
    blocked_reason: str | None = None


ConnectorDeletedHook = Callable[[Any, int, ConnectorType, int], ConnectorDeleteDecision]


@dataclass(frozen=True)
class ConnectorAccess:
    """Whether the caller's team links a connector, and may edit it.

    A verdict that reaches a caller always carries ``team_owned=True``:
    the only way to say "the caller's team does not link this connector"
    is to leave its ref out of the hook's answer map entirely, not to
    return a verdict with ``team_owned=False``. ``can_edit`` is otherwise
    independent -- a team can link a connector without granting edit
    rights to it, which is a legal answer on its own, not an intermediate
    or partial state. Both fields are validated as exact bools on the way
    in (see ``_validate_connector_access_answer``); the dataclass defaults
    below stay ``False``/``False`` on purpose so that constructing a bare
    ``ConnectorAccess()`` remains the shape the validator rejects, rather
    than quietly becoming a legitimate "not linked" answer.
    """

    team_owned: bool = False
    can_edit: bool = False


ConnectorRef = tuple[ConnectorType, int]

ConnectorAccessHook = Callable[
    [Any, int, "Collection[ConnectorRef]"], "dict[ConnectorRef, ConnectorAccess]"
]

ConnectorVisibilityHook = Callable[[Any, int], dict[str, set[int]]]


class TeamConnectorVisibilityHook(Protocol):
    """Resolver for the connectors a team owns.

    Typed as a ``Protocol`` with a keyword-only ``team_id`` so it cannot be
    bound where ``ConnectorVisibilityHook`` (user-keyed) is expected, or vice
    versa: the two shapes are otherwise identical and a swapped install would
    type-check while resolving an unrelated team's connectors.

    Keyed strictly on the team that owns the *governing agent* -- never on
    the running user's own team membership, and the hook has no way to
    express "is the runner a member of this team": there is no ``user_id``
    parameter, and the hook is a process-global singleton with no
    per-request state, so the hook body cannot learn which runner a
    resolution is for. An application that wants runner-scoped narrowing
    enforces it at its agent-access policy layer -- deciding who may run a
    team's agents at all -- not at this seam, because a published team
    agent is meant to serve runners -- including anonymous end users --
    who are not members of the team that owns it.

    A platform admin's tasks reach this seam through the agent-access
    policy layer's admin short-circuit, which returns every agent
    regardless of team -- so the runner of a team-governed agent is not
    necessarily a member of that team even in deployments whose policy
    layer is otherwise membership-scoped. This is accepted platform
    behavior, not a defect this hook is meant to close.

    Every connector id a hook returns becomes executable by *any* runner of
    that team's agents, not only members: for stdio and other static-auth
    MCP transports, a team-owned server with no personal link for the
    running user executes using the shared definition row's own stored
    credentials, not the runner's. (OAuth transports fail closed instead,
    for lack of a token to resolve.) For stdio specifically, **once an
    application has also installed the separate, credential-side hook**
    (``mcp_runtime.set_mcp_team_env_hook`` -- a different socket from this
    one, installed independently), the shared env layer a team-owned
    server resolves is keyed on the team this hook answers for -- the team
    that owns the *governing agent* -- never on any other team, including
    one the running user happens to belong to: when the governing team has
    no stored row for that server, there is no shared layer at all, and a
    runner whose own env-source pick is "shared" falls back to their own
    stored key instead of silently borrowing another team's row. An
    application that installs only this visibility hook, without also
    installing ``set_mcp_team_env_hook``, gets none of that: the shared env
    layer then still resolves however the pre-existing, user-keyed
    ``set_mcp_shared_env_hook`` was wired, which can and typically does key
    on the *running user's own* team rather than the governing one --
    exactly the cross-team leak the paragraph above describes as closed.
    Deciding to share a team-owned MCP server through this hook is a
    visibility decision only; closing the credential leak is a second,
    separate installation step. Custom APIs are starker: there is no
    per-member credential override layer at all, and no transport-level
    exception either -- every custom API always executes with whatever is
    stored on its shared definition row (headers, a static API key, and so
    on), for every runner, member or not. Sharing a custom API with a team
    means sharing whatever credential it holds with everyone who can run
    the team's agents.
    """

    def __call__(self, db: Any, *, team_id: int) -> dict[str, set[int]]: ...


_connector_deleted_hook: ConnectorDeletedHook | None = None
_connector_renamed_hook: ConnectorRenamedHook | None = None
_connector_visibility_hook: ConnectorVisibilityHook | None = None
_team_connector_visibility_hook: TeamConnectorVisibilityHook | None = None
_connector_access_hook: ConnectorAccessHook | None = None


def set_connector_team_hooks(
    *,
    deleted: ConnectorDeletedHook | None = None,
    renamed: ConnectorRenamedHook | None = None,
    visibility: ConnectorVisibilityHook | None = None,
    team_visibility: TeamConnectorVisibilityHook | None = None,
    access: ConnectorAccessHook | None = None,
) -> None:
    """Install application-owned connector lifecycle hooks.

    Installs the complete hook set: every hook not supplied to a given call
    is cleared, even one a previous call installed. This is a reset-all
    setter, not a merge -- an application that installs hooks across more
    than one call must pass its complete set each time.
    """

    global _connector_deleted_hook, _connector_renamed_hook
    global _connector_visibility_hook, _team_connector_visibility_hook
    global _connector_access_hook
    _connector_deleted_hook = deleted
    _connector_renamed_hook = renamed
    _connector_visibility_hook = visibility
    _team_connector_visibility_hook = team_visibility
    _connector_access_hook = access


def visible_team_connector_ids(db: Any, user_id: int) -> dict[str, set[int]]:
    """Team-shared connector ids visible to user; empty when no hook/standalone.

    Answers list membership only. Direct-id reachability and edit
    authority come from a different hook -- ``resolve_connector_access``
    -- and xagent enforces no relationship between the two answers:
    they are separate module-level slots, installed separately, and
    nothing cross-checks them. An installing application must derive
    both from one and the same link query, because it is the only side
    that can see its own link table; xagent cannot verify that and does
    not try.

    When the two answers disagree, xagent does not reconcile them: each
    question is answered from the hook that owns it, and whatever that
    hook said is what the caller gets. A connector one hook reports and
    the other omits is not a defect in xagent -- it is what the installed
    answers said.
    """
    if _connector_visibility_hook is None:
        return {"mcp": set(), "custom_api": set()}
    return _call_connector_hook_gate(db, _connector_visibility_hook, db, int(user_id))


def _validate_team_connector_answer(answer: Any) -> dict[str, set[int]]:
    """Validate the team-visibility hook's answer shape.

    This is an authorization input, not user-facing data: a malformed
    answer must fail loudly, never be normalized, coerced, or defaulted to
    empty. The hook must return a ``dict`` carrying both ``"mcp"`` and
    ``"custom_api"`` keys, each mapped to a ``set`` whose members are all
    ``int`` (``bool`` is a subclass of ``int`` in Python but is rejected
    here -- a truthy/falsy value is never a legitimate connector id). The
    ``set`` requirement is exact and deliberate: ``frozenset`` and other
    iterables are rejected too, matching the hook Protocol's declared
    ``set[int]`` rather than duck-typing around it.
    Extra keys beyond the two required ones are accepted and silently
    ignored: only ``"mcp"`` and ``"custom_api"`` are ever read by callers,
    so this function does not iterate the answer's keys at all, only probe
    the two it needs.
    """
    if not isinstance(answer, dict):
        raise ValueError(
            "team visibility hook returned a malformed answer: expected a "
            f"dict, got {type(answer).__name__}"
        )
    for key in ("mcp", "custom_api"):
        if key not in answer:
            raise ValueError(
                f"team visibility hook returned a malformed answer: missing key {key!r}"
            )
        value = answer[key]
        if not isinstance(value, set):
            raise ValueError(
                "team visibility hook returned a malformed answer: key "
                f"{key!r} must be a set, got {type(value).__name__}"
            )
        for member in value:
            if isinstance(member, bool) or not isinstance(member, int):
                raise ValueError(
                    "team visibility hook returned a malformed answer: key "
                    f"{key!r} contains a member {member!r} that is not a "
                    "connector id (must be int, not bool)"
                )
    # Defensive copy at the authorization boundary: callers must not be able
    # to mutate the installing application's own answer object (or vice
    # versa). Extra keys are dropped by the copy -- only the two probed keys
    # are part of the contract.
    return {"mcp": set(answer["mcp"]), "custom_api": set(answer["custom_api"])}


def team_connector_ids(db: Any, *, team_id: int | None) -> dict[str, set[int]]:
    """Connector ids owned by ``team_id``; empty for no team and for standalone.

    ``team_id`` is the team that owns the *governing agent*, never the
    running user's current team. ``None`` resolves empty without calling the
    hook. A non-``None`` answer is shape-validated (see
    ``_validate_team_connector_answer``) before it reaches any caller.
    """
    if team_id is None or _team_connector_visibility_hook is None:
        return {"mcp": set(), "custom_api": set()}
    return _call_connector_hook_gate(
        db,
        _team_connector_visibility_hook,
        db,
        team_id=int(team_id),
        validate=_validate_team_connector_answer,
    )


def team_connector_hook_installed() -> bool:
    """Whether an application installed a team-keyed visibility hook.

    Callers select on this, never on an empty return value: an installed
    hook legitimately answers with empty sets for a team that owns nothing.
    """
    return _team_connector_visibility_hook is not None


def _normalize_connector_refs(
    refs: "Collection[ConnectorRef]",
) -> "frozenset[ConnectorRef]":
    """Canonicalize a batch of connector refs, rejecting any id that is not
    already an ``int``.

    Written once because the result is not merely an argument passed on to
    the hook -- it is also the baseline the hook's answer is checked
    against: ``_validate_connector_access_answer`` rejects every key that
    is not a member of this set. Two separately written normalizations
    could drift, and the question asked would then be measured against a
    different notion of canonical shape than the one it was built from,
    with nothing failing to say so.

    An id that is not already an ``int`` is rejected here -- never coerced,
    skipped, or defaulted. Callers are expected to build their refs from
    FastAPI-validated ``int`` path parameters or from non-nullable integer
    primary keys, so anything else means xagent's own code built the list
    wrong; it surfaces as the ``TypeError``/``ValueError`` it is. Dropping
    such a ref instead would silently turn a caller's bug into a connector
    the hook was never asked about, and the caller would read the resulting
    gap as "the team does not link it".

    Coercing was worse than either, which is why it is gone. ``int("11")``
    looked harmless, because the batch answer did come back correct -- but it
    asked about a different tuple than the caller was holding. The answer came
    back keyed ``("mcp", 11)`` while the caller still had ``("mcp", "11")``,
    so ``resolve_one_connector_access_or_raise`` looked its own ref up, missed,
    and returned ``None`` -- this seam's word for "the team does not link it"
    -- about a connector the hook had just granted. Rejecting instead means
    the ref the hook is asked about is always the ref the caller passed, so an
    answer key can never fail to match the ref it answers.

    ``bool`` is excluded explicitly, the same way
    ``_validate_connector_access_answer`` and
    ``_validate_team_connector_answer`` exclude it: ``isinstance(True, int)``
    is ``True`` in Python, and ``("mcp", True)`` compares and hashes equal to
    ``("mcp", 1)``, so tolerating it would resolve a different connector's
    access with nothing failing to say so.

    Only the id half is checked at run time. The connector type is carried
    by the ``ConnectorRef`` annotation, so a non-``str`` type is not
    rejected here and reaches the hook as it was passed in: if the hook
    answers with a verdict for it, the answer validator rejects the key,
    and if the hook leaves it out, the caller reads that gap as "the team
    does not link it". Whichever route first feeds this seam something a
    type checker has not already constrained is the one that owes a
    run-time check.
    """
    validated: "set[ConnectorRef]" = set()
    for connector_type, connector_id in refs:
        if isinstance(connector_id, bool) or not isinstance(connector_id, int):
            raise TypeError(
                "connector ref id must already be an int (bool is a subclass "
                "of int in Python and is never a legitimate connector id), got "
                f"{connector_id!r} for connector type {connector_type!r}"
            )
        validated.add((connector_type, connector_id))
    return frozenset(validated)


def _validate_connector_access_answer(
    answer: Any, requested: "frozenset[ConnectorRef]"
) -> "dict[ConnectorRef, ConnectorAccess]":
    """Validate the access hook's batch answer shape.

    An authorization input, not user-facing data: a malformed answer must
    fail loudly, never be normalized, coerced, or defaulted to empty. The
    hook answers a ``dict`` keyed on the connectors it was asked about; a
    connector the caller's team does not link is expressed by leaving its
    ref out of the answer entirely, never by a verdict with
    ``team_owned=False`` -- unlike the team-visibility hook's
    ``_validate_team_connector_answer`` above, where extra keys beyond the
    two required ones are silently accepted because nothing ever reads
    them, here the keys of the answer *are* the question: a verdict for a
    ref that was never asked about means the hook answered a different
    question than the one it was asked, and silently dropping it would
    hide that the hook and the caller have gone out of sync.

    Each value must be a ``ConnectorAccess`` whose ``team_owned`` is
    exactly ``True`` and whose ``can_edit`` is a ``bool``. The two are
    deliberately spelled differently because they are different
    requirements: ``team_owned`` admits one specific value, so it is an
    identity check matching ``knowledge_base_team_scope.py``'s
    ``element.team_owned is not True``, while ``can_edit`` admits either
    ``True`` or ``False``, which is a type check and reads as one. Neither
    is a truthiness check: ``bool`` is a subclass of ``int`` in Python, so
    a merely truthy stand-in such as ``1`` is never accepted as a
    legitimate grant.

    Each key must be an exact ``(str, int)`` pair before it is even checked
    for membership: ``bool``, ``float`` and ``Decimal`` all compare equal
    to the ``int`` they alias (``True == 1``, ``1.0 == 1``,
    ``Decimal("1") == 1``), and Python's ordinary tuple equality carries
    that through to a key like ``("mcp", True)`` -- which would compare
    equal to, and pass the membership check for, ``("mcp", 1)``. The keys
    of this answer *are* the question (see above), so a key that only
    resembles one of the refs asked about is not a legitimate answer to
    the question, and must fail loudly here rather than being accepted as
    the connector it merely aliases.
    """
    if not isinstance(answer, dict):
        raise ValueError(
            "connector access hook returned a malformed answer: expected a "
            f"dict, got {type(answer).__name__}"
        )
    validated: "dict[ConnectorRef, ConnectorAccess]" = {}
    for key, verdict in answer.items():
        if not isinstance(key, tuple) or len(key) != 2:
            raise ValueError(
                "connector access hook returned a malformed answer: key "
                f"{key!r} is not a (connector_type, connector_id) pair"
            )
        connector_type, connector_id = key
        if not isinstance(connector_type, str):
            raise ValueError(
                "connector access hook returned a malformed answer: key "
                f"{key!r} has a connector type that is not a str, got "
                f"{type(connector_type).__name__}"
            )
        if isinstance(connector_id, bool) or not isinstance(connector_id, int):
            raise ValueError(
                "connector access hook returned a malformed answer: key "
                f"{key!r} has a connector id that is not an int (bool is a "
                "subclass of int in Python and is never a legitimate "
                f"connector id), got {connector_id!r}"
            )
        if key not in requested:
            raise ValueError(
                "connector access hook returned a malformed answer: a "
                f"verdict for {key!r}, which was not among the connectors "
                "asked about"
            )
        if not isinstance(verdict, ConnectorAccess):
            raise ValueError(
                "connector access hook returned a malformed answer: "
                f"expected ConnectorAccess values, got "
                f"{type(verdict).__name__} for {key!r}"
            )
        if verdict.team_owned is not True:
            raise ValueError(
                "connector access hook returned a malformed answer for "
                f"{key!r}: team_owned must be True -- a connector the "
                "caller's team does not link is expressed by leaving it "
                f"out of the answer, not by a verdict, got {verdict.team_owned!r}"
            )
        if not isinstance(verdict.can_edit, bool):
            raise ValueError(
                "connector access hook returned a malformed answer for "
                f"{key!r}: can_edit must be exactly True or False (bool is "
                "a subclass of int in Python, and a truthy value is never "
                f"a legitimate grant), got {verdict.can_edit!r}"
            )
        validated[key] = verdict
    return validated


def _resolve_normalized_connector_access(
    db: Any,
    user_id: int,
    hook: "ConnectorAccessHook",
    requested: "frozenset[ConnectorRef]",
) -> "dict[ConnectorRef, ConnectorAccess]":
    """Ask an already-resolved hook about an already-normalized ref set.

    Exists so each public entry point normalizes exactly once.
    ``resolve_connector_access_or_raise`` has to normalize above its ``try``,
    so it cannot hand ``refs`` down; before this split it handed the
    normalized set to ``resolve_connector_access``, which normalized it again.

    ``hook`` is a parameter rather than a read of the module global, so "is a
    hook installed" is answered once per call, by the entry point, and that
    same answer is what gets used. Reading the global here instead would make
    the ``None`` check every future caller's job to remember.
    """
    if not requested:
        return {}
    return _call_connector_hook_gate(
        db,
        hook,
        db,
        int(user_id),
        requested,
        validate=lambda answer: _validate_connector_access_answer(answer, requested),
    )


def resolve_connector_access(
    db: Any, user_id: int, refs: "Collection[ConnectorRef]"
) -> "dict[ConnectorRef, ConnectorAccess]":
    """Whether the caller's team links each of ``refs``, and may edit it.

    Asks the installed access hook, if any, at most once per call
    regardless of how many refs are passed -- batching is the point of
    this signature, not an incidental property, because the seam's whole
    reason to exist is to answer "what is this caller's team's
    relationship to these connectors" without paying one hook call per
    connector. Returns ``{}`` immediately, without calling the hook at
    all, when no hook is installed or when ``refs`` is empty: an empty
    request is never worth a call, and a standalone deployment with no
    hook installed sees zero queries and zero behavior change.

    The returned map is keyed on the refs as the caller passed them. That is
    a consequence of ``_normalize_connector_refs`` rejecting an id that is not
    already an ``int`` rather than coercing it: there is no input that reaches
    the hook under one key and comes back under another, so a caller may look
    its own ref straight back up in this map.

    A ref missing from the returned map means "the caller's team does not
    link this connector" -- the only way that fact is ever expressed (see
    ``_validate_connector_access_answer``). The answer is shape-validated
    before it reaches any caller.

    Answers direct-id reachability and edit authority only. List
    membership comes from a different hook -- ``visible_team_connector_ids``
    -- and xagent enforces no relationship between the two answers: they
    are separate module-level slots, installed separately, and nothing
    cross-checks them. An installing application must derive both from one
    and the same link query, because it is the only side that can see its
    own link table; xagent cannot verify that and does not try.

    When the two answers disagree, xagent does not reconcile them: each
    question is answered from the hook that owns it, and whatever that
    hook said is what the caller gets. A connector one hook reports and
    the other omits is not a defect in xagent -- it is what the installed
    answers said.
    """
    hook = _connector_access_hook
    if hook is None:
        return {}
    return _resolve_normalized_connector_access(
        db, user_id, hook, _normalize_connector_refs(refs)
    )


def _restore_session_after_hook_failure(db: Any) -> None:
    """Roll back whatever a hook left on the shared session before its call
    failed -- whether the hook raised, or answered with a shape this seam
    rejected.

    Hooks are handed the endpoint's own live session (see
    ``delete_team_connector``'s contract note). A hook whose own statement
    failed leaves that transaction unusable on PostgreSQL, and an ORM
    ``flush`` failure leaves it unusable on every backend -- so every
    later statement in the request, including the ones a degradation path
    needs to build its response, would be refused. Rolling back here, at
    the one door every hook call and every answer check passes through, is
    what keeps the degradation contract true.

    No durable work is discarded, and not because of where the rollback
    sits relative to a commit: every call site today invokes its hook
    *before* the route's own ``db.commit()`` -- both rename paths and both
    delete paths do. What the rollback can reach is therefore only the
    aborting request's own pending mutations, which the re-raised
    exception was going to strand uncommitted anyway. A call site that
    ever decorated *after* committing would keep the same property for
    the opposite reason, its work already being durable by then.

    A rollback that itself fails is logged and swallowed: this runs on an
    already-failing path, the original failure is re-raised by the caller
    either way, and there is no further recovery available.
    """
    rollback = getattr(db, "rollback", None)
    if rollback is None:
        return
    # Unconditional, unlike ``release_db_connection_if_clean``
    # (``models/database.py``), which refuses to roll back a session
    # carrying pending ORM changes: that helper runs on the success path,
    # where discarding work the request still intends to commit would be
    # data loss. Here the request is aborting -- the caller re-raises --
    # so there is no such work to protect, and leaving the session
    # unusable is the worse outcome.
    try:
        rollback()
    except Exception:
        logger.warning(
            "Rolling back after a failed connector hook failed", exc_info=True
        )


_HookResult = TypeVar("_HookResult")


def _call_connector_hook_gate(
    db: Any,
    hook: "Callable[..., _HookResult]",
    *args: Any,
    validate: "Callable[[Any], _HookResult] | None" = None,
    **kwargs: Any,
) -> _HookResult:
    """The one door every installed connector hook is called through, and
    the one place its answer is checked.

    Hooks run on the endpoint's own live session (see
    ``delete_team_connector``'s contract note). A hook whose own statement
    failed leaves that transaction unusable on PostgreSQL, and a failed
    ORM ``flush`` leaves it unusable on every backend -- so restoring the
    session belongs to the invocation itself, not to whichever caller
    happens to wrap it. Placing it here is what makes the property hold
    for a hook slot added to this module later, without that slot's author
    having to know about it: every one of the five slots this module
    defines is invoked through this one function.

    ``validate``, when given, runs inside the same ``try`` because a hook
    can poison the session *without* raising: run a statement that fails,
    catch that itself, and answer with a shape this seam then rejects. The
    rejection is this module's own exception rather than the hook's, so a
    restore placed around the call alone would not fire for it -- the
    session would stay unusable for everything the request does next. Two
    of the five slots have an answer this seam validates; the other three
    pass nothing, which says at the call site that this seam checks
    nothing about those answers, rather than leaving that silent.

    The exception is re-raised unchanged, whichever of the two raised it;
    this function decides nothing about how the failure is classified or
    translated. That stays with the ``*_or_raise`` wrappers below, which
    own the seam's typed-error contract.

    One shape stays uncovered, deliberately: a hook that poisons the
    session, swallows its own failure, and still returns a well-formed
    answer produces no exception at all -- neither here nor in a
    validator -- so nothing triggers a restore. Closing it would mean
    probing the session's health after every hook call, which is a
    different design than a failure path.
    """
    try:
        answer = hook(*args, **kwargs)
        if validate is None:
            return answer
        return validate(answer)
    except Exception:
        _restore_session_after_hook_failure(db)
        raise


def resolve_team_connector_ids_or_raise(
    db: Any, *, team_id: int | None, log_subject: int | None
) -> dict[str, set[int]]:
    """``team_connector_ids(db, team_id=team_id)``, with every non-typed
    failure converted into the seam's one typed 503.

    Every team-scope resolution call site -- both of ``WebToolConfig``'s
    custom-API read points, its MCP read point, and the runtime-context
    view loader -- wraps ``team_connector_ids`` the same way: a
    ``ConnectorRuntimeError`` passes through unchanged, and any other
    exception is logged at ``WARNING`` and converted into
    ``ConnectorRuntimeError(ERROR_CONNECTOR_RUNTIME_UNAVAILABLE, "Connector
    team scope is unavailable.", details={"reason":
    "team_scope_resolution_failed"}, status_code=503)``. This function is
    that wrap, written once. Returns the full ``{"mcp": ..., "custom_api":
    ...}`` dict unmodified -- callers that need one connector kind extract
    it themselves (e.g. ``result["mcp"]``); the runtime-context view loader
    needs both and takes the dict as-is.

    ``log_subject`` is the user id that identifies the caller in the
    warning log line -- ``None`` at any call site whose own identity guard
    the caller has not yet reached (the MCP read point's private loader has
    no identity guard of its own; production only reaches it through its
    guarded public wrapper). It is only ever formatted into the log
    message, never interpreted.

    The session restore lives on ``_call_connector_hook_gate``, the single
    door every installed hook is invoked through, rather than on either
    failure arm below, and it covers both ways that call can fail: a
    hook can leave a statement failed on the session and *then* raise its
    own ``ConnectorRuntimeError``, and a hook can leave one failed, swallow
    that itself, and answer with a shape this seam's own validator then
    rejects. Neither is something the generic-exception arm below could
    own, and both are restored before either arm sees the exception.
    """
    try:
        return team_connector_ids(db, team_id=team_id)
    except ConnectorRuntimeError:
        raise
    except Exception as exc:
        logger.warning(
            "Failed to resolve team connector scope for user %s",
            log_subject,
            exc_info=True,
        )
        raise ConnectorRuntimeError(
            ERROR_CONNECTOR_RUNTIME_UNAVAILABLE,
            "Connector team scope is unavailable.",
            details={"reason": "team_scope_resolution_failed"},
            status_code=503,
        ) from exc


def resolve_connector_access_or_raise(
    db: Any, user_id: int, refs: "Collection[ConnectorRef]"
) -> "dict[ConnectorRef, ConnectorAccess]":
    """``resolve_connector_access(db, user_id, refs)``, with every failure of the
    hook call and of its answer validation converted into the seam's one typed 503.

    Returns ``{}`` without reading ``refs`` at all when no hook is installed,
    before any normalization -- the same first move ``resolve_connector_access``
    makes, and for the same reason: a deployment with no application installed
    must not be able to raise on a ref shape only an installing application
    would ever produce. The rest of this docstring describes what happens once
    one is installed.

    Validating ``refs`` then happens before the 503 conversion and is
    deliberately outside it. An id that is not already an ``int`` is a defect
    in the calling route, not an outage of the installing application: every
    caller builds its refs from FastAPI-validated ``int`` path parameters or
    from non-nullable integer primary keys, so an id that is not already an
    ``int`` means xagent's own code is wrong. It surfaces as the
    ``TypeError``/``ValueError`` it is, raised before the hook is reached,
    rather than as a retryable "connector access is unavailable" that would
    send an operator to look at the application instead of at the route.

    Only the id half is checked that way. The connector type is carried by the
    ``ConnectorRef`` annotation and is not re-checked at run time, so a
    non-``str`` type reaches the hook as it was passed in: if the hook answers
    with a verdict for it, the answer validator rejects the key and the generic
    arm below folds that into the same 503; if the hook leaves it out, the
    caller reads the gap as "the team does not link it" and nothing raises at
    all.

    A ``ConnectorRuntimeError`` -- whether raised by the hook itself or by
    ``_validate_connector_access_answer`` rejecting what the hook answered --
    passes through unchanged (same object, not re-wrapped). Any other
    exception is logged at ``WARNING`` and converted into
    ``ConnectorRuntimeError(ERROR_CONNECTOR_RUNTIME_UNAVAILABLE, "Connector
    access is unavailable.", details={"reason":
    "connector_access_resolution_failed"}, status_code=503)``. Unlike
    ``resolve_team_connector_ids_or_raise``, there is no separate
    ``log_subject`` parameter: ``user_id`` here already identifies the
    caller directly, so it doubles as the value logged. The logged refs are
    the validated ``(connector_type, int)`` tuples this function resolved
    with, which are the caller's own refs -- ids are rejected rather than
    rewritten, so nothing appears in the log under a shape the caller never
    passed. They are sorted for a stable log line, and are never an ORM
    attribute read off a row, which could itself fail if the session is left
    unusable by whatever just failed.

    The session restore lives on ``_call_connector_hook_gate``, the single
    door every installed hook is invoked through, rather than on either
    failure arm below, and it covers both ways that call can fail: a
    hook can leave a statement failed on the session and *then* raise its
    own ``ConnectorRuntimeError``, and a hook can leave one failed, swallow
    that itself, and answer with a shape this seam's own validator then
    rejects. Neither is something the generic-exception arm below could
    own, and both are restored before either arm sees the exception.
    """
    # Answered before the normalization below, which is itself deliberately
    # above the ``try``. Both public entry points owe the same answer here:
    # without this gate a standalone deployment raises on a malformed ref
    # through this entry point while ``resolve_connector_access`` returns
    # ``{}`` for the very same input. Each entry point reads the slot once and
    # passes what it read down, so the call below cannot act on a hook other
    # than the one this check just cleared.
    hook = _connector_access_hook
    if hook is None:
        return {}
    requested = _normalize_connector_refs(refs)
    try:
        return _resolve_normalized_connector_access(db, user_id, hook, requested)
    except ConnectorRuntimeError:
        raise
    except Exception as exc:
        logger.warning(
            "Failed to resolve connector access for user %s across %s connectors: %s",
            user_id,
            len(requested),
            # ``key=str`` because the failure arm must not fail: the refs a caller
            # hands in are only type-checked statically, so a mixed-type collection
            # reaches here intact and a bare ``sorted`` would raise TypeError from
            # inside this handler, replacing the typed 503 with an untyped error.
            sorted(requested, key=str),
            exc_info=True,
        )
        raise ConnectorRuntimeError(
            ERROR_CONNECTOR_RUNTIME_UNAVAILABLE,
            "Connector access is unavailable.",
            details={"reason": "connector_access_resolution_failed"},
            status_code=503,
        ) from exc


def resolve_one_connector_access_or_raise(
    db: Any, user_id: int, ref: "ConnectorRef"
) -> "ConnectorAccess | None":
    """Single-``ref`` convenience wrapper around
    ``resolve_connector_access_or_raise``: wraps ``ref`` in a one-element
    collection, calls the batch resolver, and unwraps the answer for that
    ref. ``None`` means the same thing it means for any ref missing from a
    batch answer -- not linked, or a legitimate answer the hook chose to
    omit -- never a failure, which still raises ``ConnectorRuntimeError``
    same as the batch form. Exists so a caller resolving a single
    connector does not repeat the wrap-then-``.get(ref)`` shape by hand.
    """
    return resolve_connector_access_or_raise(db, user_id, [ref]).get(ref)


@contextmanager
def snapshot_connector_team_hooks() -> Iterator[None]:
    """Save every module-level hook slot, restore it on exit.

    Intended for tests: entering the block, replacing any slot (through
    ``set_connector_team_hooks`` or a direct module-attribute monkeypatch),
    and leaving restores every slot to the exact object it held before the
    block, including a slot the block never touched. A slot added to this
    module later must be added here too, or a snapshot taken before that
    slot exists will silently fail to restore it -- covered by the
    discovery-based coverage test in
    tests/web/services/test_connector_team_scope.py, which enumerates every
    module global ending in ``_hook`` and asserts this snapshot restores
    each one by identity. Saving and restoring lives on the module because
    the state being saved lives on the module: a test-side helper would
    have to name and reach these globals from outside, and would go stale
    the moment a slot is added here.
    """
    global _connector_deleted_hook, _connector_renamed_hook
    global _connector_visibility_hook, _team_connector_visibility_hook
    global _connector_access_hook
    saved = (
        _connector_deleted_hook,
        _connector_renamed_hook,
        _connector_visibility_hook,
        _team_connector_visibility_hook,
        _connector_access_hook,
    )
    try:
        yield
    finally:
        (
            _connector_deleted_hook,
            _connector_renamed_hook,
            _connector_visibility_hook,
            _team_connector_visibility_hook,
            _connector_access_hook,
        ) = saved


def connector_visible_to_user(
    *,
    association: "UserMCPServer | UserCustomApi | None",
    connector_id: int,
    team_ids: Collection[int],
) -> bool:
    """In-Python twin of the ``visible_*_clause`` predicates below.

    Same rule, expressed for callers that hold already-loaded ORM rows rather
    than a query to filter: an active personal association, unioned with the
    team-owned ids. ``is_active`` gates only the personal arm, so a
    team-visible connector stays visible through a deactivated personal link
    -- the corner that separates this from a naive ``association.is_active``
    check, and the one ``/api/mcp/apps`` got wrong before #1384.

    ``association`` is None when the caller resolved the connector through the
    team overlay alone. Kept beside the clauses it mirrors so the declarative
    and imperative forms of the rule move together;
    ``_load_visible_runtime_connectors`` expresses the same union a third time
    in connector_runtime.py, against its own queries.
    """

    if association is not None and bool(association.is_active):
        return True
    return connector_id in team_ids


def visible_mcp_server_clause(
    owner_user_id: int | None, team_mcp_ids: Collection[int]
) -> ColumnElement[bool]:
    """Predicate selecting the MCP servers visible to ``owner_user_id``.

    Personal servers (an active ``UserMCPServer`` link) unioned with the
    team-owned servers named in ``team_mcp_ids``. ``is_active`` gates only
    the personal arm -- a team-owned server is visible regardless of the
    member's own personal link state, which is intentional (see the
    module's callers for the credential consequence). Pure function over
    ORM constructs -- no I/O. An empty ``team_mcp_ids`` reduces to exactly
    the personal semi-join, so a deployment with no team overlay compiles
    the same query shape it always has. A ``None`` owner also reduces to
    the personal arm regardless of ``team_mcp_ids`` -- defense in depth so
    this function is fail-closed on its own terms, not only because of how
    its one caller happens to be gated today.
    """

    from sqlalchemy import or_, select

    from ..models.mcp import MCPServer, UserMCPServer

    personal = MCPServer.id.in_(
        select(UserMCPServer.mcpserver_id).where(
            UserMCPServer.user_id == owner_user_id,
            UserMCPServer.is_active,
        )
    )
    if not team_mcp_ids or owner_user_id is None:
        return personal
    return or_(personal, MCPServer.id.in_(set(team_mcp_ids)))


def visible_custom_api_clause(
    owner_user_id: int | None, team_api_ids: Collection[int]
) -> ColumnElement[bool]:
    """Predicate selecting the custom APIs visible to ``owner_user_id``.

    Personal custom APIs (an active ``UserCustomApi`` link) unioned with the
    team-owned APIs named in ``team_api_ids``. ``is_active`` gates only the
    personal arm -- a team-owned API is visible regardless of the member's
    own personal link state, which is intentional (see the module's callers
    for the credential consequence). Pure function over ORM constructs -- no
    I/O. An empty ``team_api_ids`` reduces to exactly the personal
    semi-join, so a deployment with no team overlay compiles the same query
    shape it always has. A ``None`` owner also reduces to the personal arm
    regardless of ``team_api_ids`` -- defense in depth so this function is
    fail-closed on its own terms, not only because of how its callers
    happen to be gated today.
    """

    from sqlalchemy import or_, select

    from ..models.custom_api import CustomApi, UserCustomApi

    personal = CustomApi.id.in_(
        select(UserCustomApi.custom_api_id).where(
            UserCustomApi.user_id == owner_user_id,
            UserCustomApi.is_active,
        )
    )
    if not team_api_ids or owner_user_id is None:
        return personal
    return or_(personal, CustomApi.id.in_(set(team_api_ids)))


def delete_team_connector(
    db: Any, user_id: int, connector_type: ConnectorType, connector_id: int
) -> ConnectorDeleteDecision:
    """Remove team ownership before a global delete.

    Returns whether the application recognized the connector as team-owned.
    Hooks must use the passed session and must not commit independently, so a
    refused endpoint request can discard all hook-side mutations atomically.
    """

    if _connector_deleted_hook is None:
        return ConnectorDeleteDecision()
    return _call_connector_hook_gate(
        db, _connector_deleted_hook, db, user_id, connector_type, connector_id
    )


def rename_team_connector(
    db: Any,
    user_id: int,
    connector_type: ConnectorType,
    connector_id: int,
    old_name: str,
    new_name: str,
) -> None:
    """Keep application-owned connector selectors aligned after a rename."""

    if _connector_renamed_hook is not None and old_name != new_name:
        _call_connector_hook_gate(
            db,
            _connector_renamed_hook,
            db,
            user_id,
            connector_type,
            connector_id,
            old_name,
            new_name,
        )
