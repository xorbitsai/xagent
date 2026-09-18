"""Deployment hooks for classifying ``source="external"`` conversation logs.

The OSS runtime never stamps ``Task.source = "external"`` itself. The SaaS
session transport (session widget, header-WS, v1 external API) does, and the
same stored value covers both REST/SDK tasks created through a client
application and widget-session tasks. Telling them apart needs deployment-side
state (widget session linkage) that this repository does not model, so the
Conversation Logs page asks the deployment layer through the two hooks below.

Both hooks are optional. With none registered every ``external`` row is shown
under "REST API" and carries no public context, which keeps the rows visible
instead of dropping them from every channel filter.

This module is a transitional bridge for xorbitsai/xagent-saas#1366. The
stored value cannot change on the write side today because the SaaS external
task resolvers use ``Task.source == "external"`` as an ownership predicate.
Once the transport stamps a durable, per-transport origin that this page can
map directly in ``DIRECT_SOURCE_TO_UI_SOURCE`` (and the existing rows are
backfilled), both hooks and the validator in ``api/conversation_logs.py``
can be deleted.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Sequence

from sqlalchemy.orm import Session

if TYPE_CHECKING:
    from ..models.task import Task

EXTERNAL_TASK_SOURCE = "external"

# Hook signature:
#     (db: Session) -> Sequence[tuple[<SQL predicate>, ui_source]] | None
# Each pair is a boolean SQLAlchemy expression over ``Task`` columns and the UI
# source ("widget", "rest_api" or "shared_link") it selects. The Conversation
# Logs query evaluates the pairs in the order the returned list holds them
# inside its classification CASE, restricted to rows whose stored source is
# "external"; rows no predicate matches fall back to "rest_api". ``None`` means
# "no branches". The hook is SQL-shaped so per-source counts, channel filters
# and pagination stay in the database instead of loading every task to
# classify it in Python. A predicate may reference ``Task`` columns directly;
# anything in another table must be reached through a self-contained subquery
# -- ``exists().where(Other.task_id == Task.id)`` or ``Task.id.in_(subquery)``
# -- because the consuming queries join different tables and a bare
# cross-table comparison would multiply rows. The consumer validates the
# shape of each entry, not its semantics: a predicate that is not
# boolean-typed (wrap SQL functions with ``type_=Boolean`` or
# ``cast(..., Boolean)``), one that adds a FROM entry other than the ``tasks``
# table (another table or a ``Task`` alias), or an entry that is not a
# ``(predicate, ui_source)`` pair (tuple or list), is skipped one at a time
# with a warning; a hook that raises is treated as unregistered.
#
# Raw SQL (``text()``, ``literal_column()``) is rejected as well: it declares
# no FROM entries, so the consumer cannot check which tables it names. Build
# predicates from ``Task`` columns and bound parameters.
#
# READ-ONLY CONTRACT: both hooks run inside a GET request on the caller-owned
# ``db`` session. They may query through it but must not add, flush, commit,
# roll back or close it. If a hook statement fails, the consumer rolls the
# session back before rendering the default.
#
# DETERMINISM: the list and the detail endpoint call the source hook
# independently, with no shared snapshot. The pairs must depend only on
# ``Task`` columns and on state that does not change between two requests,
# or list and detail can disagree about the same task.
#
# Application layers inject it via set_external_task_source_hook(). Only one
# hook of each kind is held: a later set_* call replaces the earlier one.
ExternalTaskSourceHook = Callable[[Session], Sequence[tuple[Any, str]] | None]
_external_task_source_hook: ExternalTaskSourceHook | None = None

# Hook signature: (db: Session, task: Task, ui_source: str) -> dict | None
# Returns the ``public_context`` block for one external task's detail view,
# for example the widget session and end-user the task belongs to. ``None``
# means "no deployment context"; external rows never fall back to the
# agent_config-derived context of the legacy widget/share transports because
# the session transport does not populate those keys. Same read-only contract
# as above. Application layers inject it via set_external_task_context_hook().
ExternalTaskContextHook = Callable[[Session, "Task", str], dict[str, Any] | None]
_external_task_context_hook: ExternalTaskContextHook | None = None


def set_external_task_source_hook(hook: ExternalTaskSourceHook | None) -> None:
    global _external_task_source_hook
    _external_task_source_hook = hook


def get_external_task_source_branches(db: Session) -> list[tuple[Any, str]]:
    """Return the deployment's ``(predicate, ui_source)`` pairs, in order."""
    if _external_task_source_hook is None:
        return []
    branches = _external_task_source_hook(db)
    return [] if branches is None else list(branches)


def set_external_task_context_hook(hook: ExternalTaskContextHook | None) -> None:
    global _external_task_context_hook
    _external_task_context_hook = hook


def get_external_task_public_context(
    db: Session, task: Task, ui_source: str
) -> dict[str, Any] | None:
    """Return the deployment-provided public context for one external task."""
    if _external_task_context_hook is None:
        return None
    return _external_task_context_hook(db, task, ui_source)


__all__ = [
    "EXTERNAL_TASK_SOURCE",
    "ExternalTaskContextHook",
    "ExternalTaskSourceHook",
    "get_external_task_public_context",
    "get_external_task_source_branches",
    "set_external_task_context_hook",
    "set_external_task_source_hook",
]
