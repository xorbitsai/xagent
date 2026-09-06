"""Optional application hooks for team-owned MCP and Custom API connectors.

Standalone xagent keeps connectors user-owned. A multi-tenant application can
install these hooks to overlay team visibility without teaching xagent about
the application's team tables.

Session contract
----------------

Every hook here is called with the endpoint's own live database session --
the same object the route is using, mid-request, sometimes with row locks
already taken. A hook that ends that transaction releases those locks, and
the route finds out about none of it: it keeps running, writes, and commits
believing it still holds them. Two concurrent edits of one connector can
then interleave into a final row neither of them submitted.

A hook must not:

- call ``commit()`` on the session it was handed;
- call ``rollback()`` on it;
- call ``close()`` on it;
- call a helper that conditionally does one of those three. This repository
  ships one such helper: ``release_db_connection_if_clean``
  (``models/database.py``), which rolls back and returns the connection when
  the session has no pending writes -- which is exactly the state a hook is
  usually called in, so a hook reusing it lands on the rollback branch.

A hook may:

- open a savepoint with ``begin_nested()`` and commit or roll back that
  savepoint. This is the supported way for a hook to recover from a failure
  of its own, and it does not end the caller's transaction;
- raise. The seam restores the session and propagates the failure, and
  raising is not a contract violation;
- write on the session and ``flush()`` without ending the transaction. The
  ``deleted`` and ``renamed`` slots exist so a hook can do exactly this;
- open a ``Session`` of its own and do whatever it likes on that one.

Call sites and what the caller holds
------------------------------------

Each row states what the caller's transaction is holding while the hook
runs, whether the caller had already committed work of its own before
asking, and whether the call declares ``caller_holds_lock``. A call site
that declares it is checked; one that does not is not.

| Call site | Caller holds while the hook runs | Committed before asking | ``caller_holds_lock`` |
| --- | --- | --- | --- |
| ``custom_api.update_custom_api`` | the ``custom_apis`` definition row, ``FOR UPDATE``, on the payloads that write that row | no | ``True`` |
| ``custom_api.delete_custom_api`` | the ``custom_apis`` definition row, ``FOR UPDATE`` | no | ``True`` |
| ``mcp.update_mcp_server`` | the ``mcp_servers`` definition row, ``FOR UPDATE ... KEY SHARE``, on the payloads that write that row | no | ``True`` |
| ``mcp._teardown_mcp_app_server_locally`` | three row locks: ``public_mcp_apps``, ``mcp_servers``, ``user_mcpservers`` | no, within this function -- see the note below | ``True`` |
| ``mcp.delete_mcp_server`` | two row locks: ``mcp_servers`` and ``user_mcpservers``, taken by ``_lock_active_mcp_oauth_lifecycle`` before this call | no | ``True`` |

``mcp._teardown_mcp_app_server_locally`` is a helper, not a route: it has no
route decorator and no caller in this repository outside tests (the async
``teardown_mcp_app_server`` route only dispatches to it with
``asyncio.to_thread`` and, after it returns, runs the external revocation that
must not itself hold any of the three row locks). "Nothing committed before
asking" therefore holds inside its own body only. A future caller that commits
and then calls it would turn its declaration into a report of failure for work
that already succeeded, and nothing here would notice -- the check that keeps
this table honest compares declarations against call sites, not against a
caller's commit history.

While a hook runs at any of the row-locking call sites above, every
concurrent request touching that connector is queued behind it. Keep the
hook's work local to the database: blocking on an external network call there
holds that queue open for as long as the call takes. xagent has no way to
check this at run time, so it is stated here rather than enforced.

The remaining slots declare nothing. Every ``visibility`` and
``team_visibility`` call site is lock-free, and one of the ``team_visibility``
paths runs on a lazily created session that may not be in a transaction at
all. ``access`` has no call site in this repository; a caller that adds one
while holding a lock owes this table a row and owes the call
``caller_holds_lock=True``. One shape must never declare it: a call site that
has already committed its own work before asking, because refusing there
reports a failure for an operation that fully succeeded.

What the check is not
---------------------

- The count lives in ``session.info``, which a hook can read and overwrite.
  This detects a hook author who does not know the rule. It is not a barrier
  against a hook deliberately working around it, and must not be cited as
  one.
- A failure of the check itself refuses the request. It is never caught and
  read as a pass, and the session is restored before the refusal.
- Restoring the session recovers a hook's uncommitted work only. Work a hook
  already committed stays committed -- on the rename slot that commit carries
  the route's own staged field writes with it, so the caller sees a failure
  over a durable write. That is why ``commit()`` is forbidden outright rather
  than treated as something the check cleans up after.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Collection, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypeVar, cast

from fastapi import Request
from fastapi.responses import JSONResponse
from sqlalchemy.sql.elements import ColumnElement

if TYPE_CHECKING:
    from ..models.custom_api import UserCustomApi
    from ..models.mcp import UserMCPServer

from ...core.tools.adapters.vibe.connector_runtime import (
    ERROR_CONNECTOR_RUNTIME_UNAVAILABLE,
    ConnectorRuntimeError,
)
from ..models.database import root_transaction_end_count

logger = logging.getLogger(__name__)


class ConnectorHookSessionBoundaryError(RuntimeError):
    """An installed connector hook ended the caller's own transaction."""


async def connector_hook_session_boundary_error_handler(
    request: Request, exc: ConnectorHookSessionBoundaryError
) -> JSONResponse:
    """Return one stable public response for a broken hook session contract.

    500 rather than the 503 this module uses elsewhere: 503 announces a
    transient outage of the installing application and invites a retry,
    while a hook that ends the caller's transaction does the same thing on
    every request until its code changes. The body names nothing about the
    hook, the slot, or the session -- the operator reads the log line, the
    caller does not.
    """
    logger.error(
        "Connector hook ended the caller transaction for %s",
        request.url.path,
        exc_info=exc,
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "Connector team integration is unavailable."},
    )


ConnectorType = Literal["mcp", "custom_api"]

ConnectorHookSlot = Literal[
    "visibility", "team_visibility", "access", "deleted", "renamed"
]

# Called with the endpoint's live session; see the session contract in this
# module's docstring for what a hook may and may not do to it.
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


# Called with the endpoint's live session; see the session contract in this
# module's docstring for what a hook may and may not do to it.
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

# Called with the endpoint's live session; see the session contract in this
# module's docstring for what a hook may and may not do to it.
ConnectorAccessHook = Callable[
    [Any, int, "Collection[ConnectorRef]"], "dict[ConnectorRef, ConnectorAccess]"
]

# Called with the endpoint's live session; see the session contract in this
# module's docstring for what a hook may and may not do to it.
ConnectorVisibilityHook = Callable[[Any, int], dict[str, set[int]]]
TEAM_OWNED_MCP_DEFINITIONS_KEY = "owned_mcp_definitions"


@dataclass(frozen=True)
class TeamConnectorSelection:
    """Team-visible connector ids plus positive definition ownership."""

    mcp_ids: frozenset[int]
    custom_api_ids: frozenset[int]
    owned_mcp_definition_ids: frozenset[int]

    def connector_ids(self) -> dict[str, set[int]]:
        return {
            "mcp": set(self.mcp_ids),
            "custom_api": set(self.custom_api_ids),
        }


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

    The optional ``owned_mcp_definitions`` set marks MCP definitions the team
    owns. It must be a subset of ``mcp``. Absence means no positive ownership
    evidence; xagent never infers ownership from visibility alone.

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

    Called with the endpoint's live session; see the session contract in
    this module's docstring for what a hook may and may not do to it.
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
    return _call_connector_hook_gate(
        db, _connector_visibility_hook, db, int(user_id), slot="visibility"
    )


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
    Extra keys beyond the two required ones are accepted here. The selection
    validator separately checks the reserved ``owned_mcp_definitions`` key.
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


def _validate_team_connector_selection(answer: Any) -> TeamConnectorSelection:
    connector_ids = _validate_team_connector_answer(answer)
    owned = answer.get(TEAM_OWNED_MCP_DEFINITIONS_KEY, set())
    if not isinstance(owned, set):
        raise ValueError(
            "team visibility hook returned a malformed answer: key "
            f"{TEAM_OWNED_MCP_DEFINITIONS_KEY!r} must be a set, got "
            f"{type(owned).__name__}"
        )
    for member in owned:
        if isinstance(member, bool) or not isinstance(member, int):
            raise ValueError(
                "team visibility hook returned a malformed answer: key "
                f"{TEAM_OWNED_MCP_DEFINITIONS_KEY!r} contains a member "
                f"{member!r} that is not a connector id (must be int, not bool)"
            )
    if not owned.issubset(connector_ids["mcp"]):
        raise ValueError(
            "team visibility hook returned definition ownership for a hidden MCP connector"
        )
    return TeamConnectorSelection(
        mcp_ids=frozenset(connector_ids["mcp"]),
        custom_api_ids=frozenset(connector_ids["custom_api"]),
        owned_mcp_definition_ids=frozenset(owned),
    )


def team_connector_selection(db: Any, *, team_id: int | None) -> TeamConnectorSelection:
    """Resolve team visibility and positive definition ownership once."""
    if team_id is None or _team_connector_visibility_hook is None:
        return TeamConnectorSelection(frozenset(), frozenset(), frozenset())
    return cast(
        TeamConnectorSelection,
        _call_connector_hook_gate(
            db,
            _team_connector_visibility_hook,
            db,
            team_id=int(team_id),
            slot="team_visibility",
            validate=_validate_team_connector_selection,
        ),
    )


def team_connector_ids(db: Any, *, team_id: int | None) -> dict[str, set[int]]:
    """Connector ids owned by ``team_id``; empty for no team and for standalone.

    ``team_id`` is the team that owns the *governing agent*, never the
    running user's current team. ``None`` resolves empty without calling the
    hook. A non-``None`` answer is shape-validated (see
    ``_validate_team_connector_answer``) before it reaches any caller.
    """
    return team_connector_selection(db, team_id=team_id).connector_ids()


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
    *,
    caller_holds_lock: bool = False,
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

    ``caller_holds_lock``: see this module's "Call sites and what the
    caller holds" section and ``delete_team_connector`` for what it means
    and who must declare it.
    """
    if not requested:
        return {}
    return _call_connector_hook_gate(
        db,
        hook,
        db,
        int(user_id),
        requested,
        slot="access",
        validate=lambda answer: _validate_connector_access_answer(answer, requested),
        session_boundary_checked=caller_holds_lock,
    )


def resolve_connector_access(
    db: Any,
    user_id: int,
    refs: "Collection[ConnectorRef]",
    *,
    caller_holds_lock: bool = False,
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

    ``caller_holds_lock``: see this module's "Call sites and what the
    caller holds" section and ``delete_team_connector`` for what it means
    and who must declare it.
    """
    hook = _connector_access_hook
    if hook is None:
        return {}
    return _resolve_normalized_connector_access(
        db,
        user_id,
        hook,
        _normalize_connector_refs(refs),
        caller_holds_lock=caller_holds_lock,
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
    slot: ConnectorHookSlot,
    validate: "Callable[[Any], _HookResult] | None" = None,
    session_boundary_checked: bool = False,
    **kwargs: Any,
) -> _HookResult:
    """The one door every installed connector hook is called through, and
    the one place its answer is checked.

    Hooks run on the endpoint's own live session (see
    ``delete_team_connector``'s contract note). A hook whose own statement
    failed leaves that transaction unusable on PostgreSQL, and a failed
    ORM ``flush`` leaves it unusable on every backend -- so restoring the
    session belongs to the invocation itself, not to whichever caller
    happens to wrap it. Placing it here is what makes restoration hold
    for a hook slot added to this module later, without that slot's author
    having to know about it: every one of the five slots this module
    defines is invoked through this one function. Session boundary
    checking is the one thing here that is not inherited that way -- see
    ``session_boundary_checked`` below.

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

    One shape produces no exception at all: a hook that ends this
    session's own transaction, swallows whatever it was doing, and still
    returns a well-formed answer. Neither this function nor a validator
    sees a failure, while the caller's row locks are already gone. That
    one is caught by comparing a root-transaction-end count across the
    call (``root_transaction_end_count``, ``models/database.py``) on the
    success path, and refusing with
    ``ConnectorHookSessionBoundaryError`` when it moved. A hook that
    raised is not compared: the restore above is itself an unconditional
    rollback, so it moves the count on its own, and comparing there would
    read every ordinary hook failure as a contract violation.

    ``session_boundary_checked`` says whether to compare at all, and it
    is off by default. Session boundary checking is declared per call
    site, not inherited by every slot: a call site holding a row lock
    across a hook call declares it, and a call site that has already
    committed its own work before asking must not, because refusing there
    would report a failure for an operation that already succeeded.
    Which call sites declare it, and why, is the table in this module's
    docstring. The cost of that default is stated there too: a new
    lock-holding call site that forgets to declare is not checked.

    ``slot`` names which of this module's five slots the call belongs to
    (``visibility``, ``team_visibility``, ``access``, ``deleted``,
    ``renamed``). It appears only in the log lines below, never in an
    exception message or a response body.
    """
    # Read before the ``try`` on purpose. The hook has not run yet, so a
    # counter this session cannot report -- a value that is present but is
    # not a count, left there by an earlier hook -- refuses the request
    # with the hook never called and nothing to restore. Moving this read
    # inside the ``try`` would call ``_restore_session_after_hook_failure``
    # on a session no hook has touched.
    end_count_before = (
        root_transaction_end_count(db) if session_boundary_checked else None
    )
    if session_boundary_checked and end_count_before is None:
        # A duck-typed session -- ``web/tools/config.py`` documents that
        # those reach this seam -- cannot report the count, so this call
        # site's declaration buys it nothing. Say so rather than skipping
        # silently: a skipped check and a passed check look identical.
        logger.debug(
            "Connector hook session boundary check skipped: %s cannot report "
            "a root transaction end count",
            type(db).__name__,
        )
    already_restored = False
    try:
        answer = hook(*args, **kwargs)
        if validate is not None:
            answer = validate(answer)
        if end_count_before is not None:
            end_count_after = root_transaction_end_count(db)
            if end_count_after is None:
                # Symmetric with the pre-hook read above: a skipped check
                # and a passed check look identical, so say which one this
                # was rather than falling through silently.
                logger.debug(
                    "Connector hook session boundary check skipped after "
                    "the hook ran: %s cannot report a root transaction end "
                    "count",
                    type(db).__name__,
                )
            elif end_count_after > end_count_before:
                _restore_session_after_hook_failure(db)
                already_restored = True
                # The hook's identity and its slot go in the log line, not
                # in the exception message: one route converts a stray
                # exception into a response body carrying ``str(exc)``, so
                # anything put here can reach a caller.
                logger.error(
                    "Connector hook %r for the %s slot ended the caller's "
                    "database transaction",
                    getattr(hook, "__name__", type(hook).__name__),
                    slot,
                )
                raise ConnectorHookSessionBoundaryError(
                    "An installed connector hook ended the caller's "
                    "database transaction"
                )
        return answer
    except Exception:
        # Read, not merely assigned: the restore has to happen exactly
        # once whichever way this call failed, and matching on the
        # exception's type instead would restore zero times for a hook
        # that raises this module's own boundary error.
        if not already_restored:
            _restore_session_after_hook_failure(db)
        raise


def resolve_team_connector_selection_or_raise(
    db: Any, *, team_id: int | None, log_subject: int | None
) -> TeamConnectorSelection:
    """Resolve team connector visibility and provenance with a typed failure."""
    try:
        return team_connector_selection(db, team_id=team_id)
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
    return resolve_team_connector_selection_or_raise(
        db,
        team_id=team_id,
        log_subject=log_subject,
    ).connector_ids()


def resolve_connector_access_or_raise(
    db: Any,
    user_id: int,
    refs: "Collection[ConnectorRef]",
    *,
    caller_holds_lock: bool = False,
) -> "dict[ConnectorRef, ConnectorAccess]":
    """``resolve_connector_access(db, user_id, refs)``, with every failure of the
    hook call and of its answer validation converted into the seam's one typed 503.

    ``caller_holds_lock``: see this module's "Call sites and what the
    caller holds" section and ``delete_team_connector`` for what it means
    and who must declare it.

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
        return _resolve_normalized_connector_access(
            db, user_id, hook, requested, caller_holds_lock=caller_holds_lock
        )
    except ConnectorHookSessionBoundaryError:
        # A hook that ended this session's transaction is a permanent
        # defect in the installing application's code, not the transient
        # outage ``ConnectorRuntimeError`` describes. Converting it here
        # would send an operator looking for an outage that is not
        # happening, and would give this slot a different answer from
        # every other slot for the same failure.
        raise
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
    db: Any,
    user_id: int,
    ref: "ConnectorRef",
    *,
    caller_holds_lock: bool = False,
) -> "ConnectorAccess | None":
    """Single-``ref`` convenience wrapper around
    ``resolve_connector_access_or_raise``: wraps ``ref`` in a one-element
    collection, calls the batch resolver, and unwraps the answer for that
    ref. ``None`` means the same thing it means for any ref missing from a
    batch answer -- not linked, or a legitimate answer the hook chose to
    omit -- never a failure, which still raises ``ConnectorRuntimeError``
    same as the batch form. Exists so a caller resolving a single
    connector does not repeat the wrap-then-``.get(ref)`` shape by hand.

    ``caller_holds_lock``: see this module's "Call sites and what the
    caller holds" section and ``delete_team_connector`` for what it means
    and who must declare it.
    """
    return resolve_connector_access_or_raise(
        db, user_id, [ref], caller_holds_lock=caller_holds_lock
    ).get(ref)


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
    db: Any,
    user_id: int,
    connector_type: ConnectorType,
    connector_id: int,
    *,
    caller_holds_lock: bool = False,
) -> ConnectorDeleteDecision:
    """Remove team ownership before a global delete.

    Returns whether the application recognized the connector as team-owned.
    Hooks must use the passed session and must not commit independently, so a
    refused endpoint request can discard all hook-side mutations atomically.

    ``caller_holds_lock`` says this call happens while the caller's
    transaction is holding something it cannot afford to lose -- a row
    lock, typically -- and turns on the session boundary check for this
    one call. It is off by default, so a call site that holds a lock and
    does not pass it is not checked; the call site table in this module's
    docstring is where every call site and what it declares are listed.
    """

    if _connector_deleted_hook is None:
        return ConnectorDeleteDecision()
    return _call_connector_hook_gate(
        db,
        _connector_deleted_hook,
        db,
        user_id,
        connector_type,
        connector_id,
        slot="deleted",
        session_boundary_checked=caller_holds_lock,
    )


def rename_team_connector(
    db: Any,
    user_id: int,
    connector_type: ConnectorType,
    connector_id: int,
    old_name: str,
    new_name: str,
    *,
    caller_holds_lock: bool = False,
) -> None:
    """Keep application-owned connector selectors aligned after a rename.

    ``caller_holds_lock`` says this call happens while the caller's
    transaction is holding something it cannot afford to lose -- a row
    lock, typically -- and turns on the session boundary check for this
    one call. It is off by default, so a call site that holds a lock and
    does not pass it is not checked; the call site table in this module's
    docstring is where every call site and what it declares are listed.
    """

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
            slot="renamed",
            session_boundary_checked=caller_holds_lock,
        )
