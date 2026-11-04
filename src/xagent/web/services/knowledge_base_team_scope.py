"""Optional application hooks for team-owned knowledge bases.

Standalone xagent keeps collections scoped to the requesting user's id. A
multi-tenant application can install these hooks to expose a logical team
collection while its files and vector rows remain in the original storage
user's namespace.

Every knowledge base a team-keyed hook returns becomes searchable and
quotable by *any* runner of that team's agents -- not only members, and not
only interactive ones. A published team agent is meant to serve runners,
including anonymous end users of a widget or other embedded surface, who are
not members of the team that owns it. Once such a runner's turn resolves a
declared collection onto the team's copy, the collection's document text
becomes part of what that run can retrieve and quote back, regardless of who
is actually asking. An application that wants to narrow who may run a
team's agents at all enforces that at its agent-access policy layer, not at
this seam -- this seam only answers "which knowledge base does this team
own", never "is the current runner allowed to run this agent".

A visibility hook's answer must set ``can_edit`` and ``can_delete`` to
exactly ``False`` on every entry: ``KnowledgeBaseAccess`` defaults both to
``True``, and an answer that leaves either at its default -- or omits it --
is rejected outright rather than narrowed, which surfaces to every caller
as the seam's one typed 503.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Literal, Protocol

from sqlalchemy.orm import Session

from ...core.tools.core.knowledge_base_scope import KnowledgeBaseScopeError

logger = logging.getLogger(__name__)

KnowledgeBaseAction = Literal["read", "edit", "delete"]

ERROR_KNOWLEDGE_BASE_SCOPE_UNAVAILABLE = "knowledge_base_scope_unavailable"


@dataclass(frozen=True)
class KnowledgeBaseAccess:
    name: str
    storage_user_id: int
    team_owned: bool = False
    can_edit: bool = True
    can_delete: bool = True


KnowledgeBaseVisibilityHook = Callable[[Session | None, int], list[KnowledgeBaseAccess]]
KnowledgeBaseAccessHook = Callable[
    [Session | None, int, str, KnowledgeBaseAction], KnowledgeBaseAccess | None
]
KnowledgeBaseLifecycleHook = Callable[[Session | None, int, str, str | None], None]


class TeamKnowledgeBaseVisibilityHook(Protocol):
    """Resolver for the knowledge bases a team owns.

    Typed as a ``Protocol`` with a keyword-only ``team_id`` so it cannot be
    bound where ``KnowledgeBaseVisibilityHook`` (user-keyed) is expected, or
    vice versa. The two shapes would otherwise be easy to confuse -- both
    take an optional session and return a list of the same element type --
    and the danger is not symmetric.

    Binding the by-team function into the user-keyed slot is the dangerous
    direction: ``visible_team_knowledge_bases`` calls that slot
    *positionally* (``_visibility_hook(db, int(user_id))``), so a by-team
    function bound there receives a plain ``user_id`` where it expects a
    team id, and its answer is returned to the caller unvalidated -- it
    silently resolves against whichever team happens to share that
    numeric id, or misbehaves in some other way the type checker cannot
    catch, because both callables type-check as functions from
    ``(Session | None, int) -> list[KnowledgeBaseAccess]``.

    The reverse swap -- binding the user-keyed function into this
    protocol's slot -- is caught immediately: this slot is always called
    with ``team_id=`` as a keyword, and a function whose second parameter
    is positional-only or differently named raises ``TypeError`` at call
    time instead of resolving against the wrong tenant.
    """

    def __call__(
        self, db: Session | None, *, team_id: int
    ) -> list[KnowledgeBaseAccess]: ...


_visibility_hook: KnowledgeBaseVisibilityHook | None = None
_access_hook: KnowledgeBaseAccessHook | None = None
_renamed_hook: KnowledgeBaseLifecycleHook | None = None
_deleted_hook: KnowledgeBaseLifecycleHook | None = None
_team_visibility_hook: TeamKnowledgeBaseVisibilityHook | None = None


def set_knowledge_base_team_hooks(
    *,
    visibility: KnowledgeBaseVisibilityHook | None = None,
    access: KnowledgeBaseAccessHook | None = None,
    renamed: KnowledgeBaseLifecycleHook | None = None,
    deleted: KnowledgeBaseLifecycleHook | None = None,
    team_visibility: TeamKnowledgeBaseVisibilityHook | None = None,
) -> None:
    """Install application-owned knowledge-base hooks.

    Installs the complete hook set: every hook not supplied to a given call
    is cleared, even one a previous call installed. This is a reset-all
    setter, not a merge -- an application that installs hooks across more
    than one call must pass its complete set each time. A caller that adds
    a new slot to this function later must audit every existing installer
    for the same reason: an installer that does not know about the new
    keyword clears it.
    """

    global _visibility_hook, _access_hook, _renamed_hook, _deleted_hook
    global _team_visibility_hook
    _visibility_hook = visibility
    _access_hook = access
    _renamed_hook = renamed
    _deleted_hook = deleted
    _team_visibility_hook = team_visibility


def has_knowledge_base_visibility_hook() -> bool:
    """Return whether the application installed a team visibility hook."""

    return _visibility_hook is not None


def visible_team_knowledge_bases(
    db: Session | None, user_id: int
) -> list[KnowledgeBaseAccess]:
    """Return visible team KBs through the application-owned hook.

    Callers may invoke the hook from a worker thread. When ``db`` is ``None``,
    the hook must create, use, and close its own thread-confined session and
    must not depend on event-loop thread-local state.
    """
    if _visibility_hook is None:
        return []
    return list(_visibility_hook(db, int(user_id)))


def _validate_team_knowledge_base_answer(
    answer: object,
) -> list[KnowledgeBaseAccess]:
    """Validate the team-visibility hook's answer shape.

    This is an authorization input, not user-facing data: a malformed
    answer must fail loudly, never be normalized, coerced, or defaulted to
    empty.

    Two different classes of rejection are mixed into one list on purpose,
    and each element below says which class it is:

    - **Fail-open guards.** ``can_edit`` and ``can_delete`` default to
      ``True`` on ``KnowledgeBaseAccess``, and the consumer that reads a
      by-team answer copies both flags straight onto the collection object
      it hands back to the caller. An answer that simply omits them would
      therefore grant edit *and* delete on the team's knowledge base to
      every runner who can search it. Rejecting anything other than
      ``False`` is what closes that gap; nothing about this check is
      informational.
    - **Coherence check.** ``team_owned`` is fixed ``True`` for every
      element a by-team hook may legitimately return, but the consumer
      that reads a by-team answer never looks at this field -- it sets
      ``ownership: "team"`` itself once it decides to trust the answer at
      all. A wrong ``team_owned`` therefore cannot fail open on that path
      today. The check exists so the answer cannot assert something false
      that a later consumer might start believing, not because today's
      consumer is fooled by it.
    """
    if not isinstance(answer, Iterable) or isinstance(answer, (str, bytes)):
        raise ValueError(
            "team knowledge-base visibility hook returned a malformed answer: "
            f"expected an iterable of KnowledgeBaseAccess, got {type(answer).__name__}"
        )
    validated: list[KnowledgeBaseAccess] = []
    for element in answer:
        if not isinstance(element, KnowledgeBaseAccess):
            # Wrong element type.
            raise ValueError(
                "team knowledge-base visibility hook returned a malformed "
                f"answer: expected KnowledgeBaseAccess elements, got "
                f"{type(element).__name__}"
            )
        if not isinstance(element.name, str) or not element.name:
            # Empty (or non-string) name.
            raise ValueError(
                "team knowledge-base visibility hook returned a malformed "
                f"answer: name must be a non-empty str, got {element.name!r}"
            )
        if isinstance(element.storage_user_id, bool) or not isinstance(
            element.storage_user_id, int
        ):
            # bool storage_user_id, or a non-int storage_user_id.
            raise ValueError(
                "team knowledge-base visibility hook returned a malformed "
                f"answer: storage_user_id must be an int (not bool), got "
                f"{element.storage_user_id!r} for {element.name!r}"
            )
        if element.can_edit is not False or element.can_delete is not False:
            # Fail-open guard: see the docstring above -- these are copied
            # onto the collection object unchanged, and the dataclass
            # defaults are True, so an omitted field grants edit and
            # delete to every runner who can search the collection.
            raise ValueError(
                "team knowledge-base visibility hook returned a malformed "
                f"answer for {element.name!r}: can_edit and can_delete must "
                "both be False on a team-owned entry"
            )
        if element.team_owned is not True:
            # Coherence check: see the docstring above -- the consumer
            # does not read this field, so a wrong value cannot fail open
            # on today's path, but it would assert something false to a
            # future reader.
            raise ValueError(
                "team knowledge-base visibility hook returned a malformed "
                f"answer for {element.name!r}: team_owned must be True"
            )
        validated.append(element)
    return validated


def team_knowledge_bases(
    db: Session | None, *, team_id: int
) -> list[KnowledgeBaseAccess]:
    """Knowledge bases owned by ``team_id``; empty for no hook installed.

    ``team_id`` is the team that owns the *governing agent*, never the
    running user's current team. A non-empty answer is shape-validated
    (see ``_validate_team_knowledge_base_answer``) before it reaches any
    caller.
    """
    if _team_visibility_hook is None:
        return []
    answer = _team_visibility_hook(db, team_id=int(team_id))
    return _validate_team_knowledge_base_answer(answer)


def team_knowledge_base_hook_installed() -> bool:
    """Whether an application installed a team-keyed visibility hook.

    Callers select on this, never on an empty return value: an installed
    hook legitimately answers with an empty list for a team that owns
    nothing.
    """
    return _team_visibility_hook is not None


def resolve_team_knowledge_bases_or_raise(
    db: Session | None, *, team_id: int, log_subject: int | None
) -> list[KnowledgeBaseAccess]:
    """``team_knowledge_bases(db, team_id=team_id)``, with every non-typed
    failure converted into the seam's one typed 503.

    Mirrors ``resolve_team_connector_ids_or_raise``: a
    ``KnowledgeBaseScopeError`` passes through unchanged, and any other
    exception (including a validator ``ValueError``) is logged and
    converted into ``KnowledgeBaseScopeError`` so a caller can distinguish
    "the team owns nothing" (an empty list) from "resolution failed" (a
    raised, typed error) instead of the two collapsing into the same
    silent empty answer.

    ``log_subject`` is the id that identifies the caller in the warning
    log line, formatted only, never interpreted.
    """
    try:
        return team_knowledge_bases(db, team_id=team_id)
    except KnowledgeBaseScopeError:
        raise
    except Exception as exc:
        logger.warning(
            "Failed to resolve team knowledge-base scope for team %s (caller %s)",
            team_id,
            log_subject,
            exc_info=True,
        )
        raise KnowledgeBaseScopeError(
            ERROR_KNOWLEDGE_BASE_SCOPE_UNAVAILABLE,
            "Knowledge base team scope is unavailable.",
            details={"reason": "team_scope_resolution_failed"},
            status_code=503,
        ) from exc


@contextmanager
def snapshot_knowledge_base_team_hooks() -> Iterator[None]:
    """Save every module-level hook slot, restore it on exit.

    Intended for tests: entering the block, replacing any slot (through
    ``set_knowledge_base_team_hooks`` or a direct module-attribute
    monkeypatch), and leaving restores every slot to the exact object it
    held before the block, including a slot the block never touched. A
    slot added to this module later must be added here too, or a snapshot
    taken before that slot exists will silently fail to restore it.

    The connector seam this module otherwise mirrors has its own equivalent,
    ``connector_team_scope.snapshot_connector_team_hooks``, with the same
    save-and-restore shape. The two primitives are independent: each saves
    and restores only its own module's hook slots, and installing or
    resetting one has no effect on the other. Tests that do not use either
    primitive reset by calling the relevant setter with no arguments,
    which restores the *empty* state, not the state the test found. Every
    slot here is process-global, so a test that installs one and resets by
    clearing leaves any hook the process had installed before it gone, and
    the next test to depend on that hook fails somewhere else. Saving and
    restoring lives on the module because the state being saved lives on the
    module: a test-side helper would have to name and reach these globals
    from outside, and would go stale the moment a slot is added here.
    """
    global _visibility_hook, _access_hook, _renamed_hook, _deleted_hook
    global _team_visibility_hook
    saved = (
        _visibility_hook,
        _access_hook,
        _renamed_hook,
        _deleted_hook,
        _team_visibility_hook,
    )
    try:
        yield
    finally:
        (
            _visibility_hook,
            _access_hook,
            _renamed_hook,
            _deleted_hook,
            _team_visibility_hook,
        ) = saved


def resolve_knowledge_base_access(
    db: Session | None,
    user_id: int,
    name: str,
    action: KnowledgeBaseAction = "read",
) -> KnowledgeBaseAccess:
    """Resolve ownership; hooks must open a session when ``db`` is None."""

    if _access_hook is not None:
        resolved = _access_hook(db, int(user_id), name, action)
        if resolved is not None:
            return resolved
    return KnowledgeBaseAccess(name=name, storage_user_id=int(user_id))


def notify_knowledge_base_renamed(
    db: Session | None, user_id: int, old_name: str, new_name: str
) -> None:
    """Notify the app; hooks must open a session when ``db`` is None."""
    if _renamed_hook is not None and old_name != new_name:
        _renamed_hook(db, int(user_id), old_name, new_name)


def notify_knowledge_base_deleted(db: Session | None, user_id: int, name: str) -> None:
    """Notify the app; hooks must open a session when ``db`` is None."""
    if _deleted_hook is not None:
        _deleted_hook(db, int(user_id), name, None)
