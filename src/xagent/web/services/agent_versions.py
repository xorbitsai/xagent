"""Changes to an agent's versioned fields, recorded as version rows.

Eight columns of an ``agents`` row are versioned (``VERSIONED_FIELDS``).
Changes made through :func:`apply_agent_change` are recorded: it updates the
row and records the new state as an ``agent_versions`` row in the caller's
transaction. Creating an agent does not record a version yet; its first row is
written the first time this function is called for it.
:func:`rewrite_agent_references` rewrites a renamed connector or knowledge
base in the agent row and in every version row the same way, without
recording a new version.

Neither function commits or rolls back the caller's transaction.
``AgentStore.update_agent_fields`` is the committing wrapper used by the
product's save paths.

Only READ COMMITTED isolation is supported; concurrent saves of one agent
queue on the row lock. Callers must not await between taking the row lock and
ending the transaction.
"""

from __future__ import annotations

import copy
import json
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from sqlalchemy import (
    JSON,
    Integer,
    bindparam,
    column,
    insert,
)
from sqlalchemy import inspect as sa_inspect
from sqlalchemy import (
    select,
    text,
    update,
    values,
)
from sqlalchemy.orm import InstanceState, Session
from sqlalchemy.orm.util import identity_key

from ..models.agent import Agent, AgentOrigin
from ..models.agent_version import AgentVersion
from ..models.database import root_transaction_end_count
from .agent_team_scope import owned_agent_clause

logger = logging.getLogger(__name__)

# The versioned fields, in the order version content lists them.
VERSIONED_FIELDS: tuple[str, ...] = (
    "instructions",
    "execution_mode",
    "prompt_context_fields",
    "suggested_prompts",
    "skills",
    "models",
    "knowledge_bases",
    "tool_categories",
)

# Canonical form, used only to drop requested values that would not change
# the agent and to decide whether a state is a new version:
#
# - instructions: ``None`` equals ``""``; compared without leading and
#   trailing whitespace.
# - execution_mode: compared as stored.
# - models: ``None`` equals ``{}``; empty slots are dropped, because the
#   runtime treats an empty slot like a missing one.
# - skills: ``None`` equals ``[]``; order does not matter (membership test).
# - knowledge_bases: ``None`` equals ``[]``; order matters (it is the order
#   the prompt lists them in).
# - tool_categories: ``None`` (the full default tool set) differs from ``[]``
#   (no tools); order of a non-empty list does not matter.
# - suggested_prompts: ``None`` equals ``[]``; order and empty entries matter.
# - prompt_context_fields: compared after ``clean_prompt_context_fields``,
#   the parser every reader uses.
#
# Canonical values are compared as JSON (``same_value``), so ``True`` differs
# from ``1`` and ``1.0`` from ``1``: a model slot holding ``True`` does not
# name model 1, and replacing it with ``1`` is a change.
_EMPTY_VALUES: dict[str, Any] = {
    "instructions": "",
    "models": {},
    "skills": [],
    "knowledge_bases": [],
    "suggested_prompts": [],
    "prompt_context_fields": [],
}
EMPTY_EQUIVALENT: frozenset[str] = frozenset(_EMPTY_VALUES)
ORDER_INSENSITIVE: frozenset[str] = frozenset({"skills", "tool_categories"})

ACTOR_KINDS: frozenset[str] = frozenset(
    {"user", "api_key", "workspace_key", "system", "unspecified"}
)
# A change records a ``save`` version; ``initial`` and ``untracked`` are
# written only by the baseline check.
_CHANGE_VERSION_KIND = "save"
# Version rows rewritten per PostgreSQL statement by ``rewrite_agent_references``.
_VERSION_WRITE_CHUNK = 1000
_BASELINE_SOURCE = "baseline"

REFERENCE_FIELDS: frozenset[str] = frozenset({"tool_categories", "knowledge_bases"})

_AGENTS = Agent.__table__
_VERSIONS = AgentVersion.__table__


@dataclass(frozen=True)
class ConfigActor:
    """Who made a configuration change, as recorded on the version row."""

    kind: str
    user_id: int | None = None
    # Personal key prefix, or the id of a workspace key.
    key_prefix: str | None = None
    # Entry point, e.g. ``web``, ``builder_tool``, ``v1``, ``import``.
    source: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in ACTOR_KINDS:
            raise ValueError(f"Unsupported actor kind: {self.kind}")
        if self.key_prefix is not None and len(self.key_prefix) > 32:
            raise ValueError("actor key_prefix is longer than 32 characters")
        if self.source is not None and len(self.source) > 64:
            raise ValueError("actor source is longer than 64 characters")

    @classmethod
    def user(cls, user_id: int, *, source: str) -> ConfigActor:
        return cls(kind="user", user_id=int(user_id), source=source)

    @classmethod
    def api_key(
        cls, user_id: int, key_prefix: str, *, source: str = "v1"
    ) -> ConfigActor:
        return cls(
            kind="api_key", user_id=int(user_id), key_prefix=key_prefix, source=source
        )

    @classmethod
    def system(cls, source: str) -> ConfigActor:
        return cls(kind="system", source=source)

    @classmethod
    def unspecified(cls) -> ConfigActor:
        return cls(kind="unspecified")


_BASELINE_ACTOR = ConfigActor.system(_BASELINE_SOURCE)


@dataclass(frozen=True)
class OwnedBy:
    """Change the agent only if ``user_id`` may manage it under ``team_scope``.

    The ownership predicate is part of the locking read, so an agent the user
    cannot manage is ``not_found`` and nothing is written.
    """

    user_id: int
    team_scope: Any


@dataclass(frozen=True)
class SystemWrite:
    """Change the agent without an ownership check (system rewrites)."""


@dataclass(frozen=True)
class AppliedChange:
    """Outcome of :func:`apply_agent_change`.

    - ``not_found``: no such agent, or ``OwnedBy`` does not match; nothing was
      written and ``version_no`` is ``None``.
    - ``no_change``: no versioned field changed. Requested unversioned fields
      were written, and a baseline version may have been recorded.
      ``version_no`` is the agent's current version.
    - ``applied``: a versioned field changed and version ``version_no`` was
      recorded.
    """

    status: Literal["not_found", "no_change", "applied"]
    agent_id: int
    version_no: int | None = None


@dataclass(frozen=True)
class RewriteResult:
    """Outcome of :func:`rewrite_agent_references`."""

    status: Literal["not_found", "unchanged", "rewritten"]
    agent_id: int
    rewritten_versions: int = 0


class PendingVersionedChange(RuntimeError):
    """The session holds unflushed changes to the agent's versioned fields.

    They would either be lost or later overwrite the change being applied, so
    the call is refused before anything is read or written.
    """


class SessionBoundaryError(RuntimeError):
    """A ``validate`` callback ended the caller's transaction.

    That released the row lock the change was decided under.
    """


Change = Mapping[str, Any] | Callable[[dict[str, Any]], Mapping[str, Any]]
Validate = Callable[[dict[str, Any], dict[str, Any]], None]


def _sort_key(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def canonical_value(field: str, value: Any) -> Any:
    """``value`` of versioned ``field`` in canonical form (see the table above)."""
    if field not in VERSIONED_FIELDS:
        raise ValueError(f"Not a versioned agent field: {field}")
    if field == "prompt_context_fields":
        from .agent_store import clean_prompt_context_fields

        return clean_prompt_context_fields(value)
    if value is None and field in EMPTY_EQUIVALENT:
        value = copy.deepcopy(_EMPTY_VALUES[field])
    if field == "instructions" and isinstance(value, str):
        return value.strip()
    if field == "models" and isinstance(value, dict):
        return {slot: model for slot, model in value.items() if model}
    if field in ORDER_INSENSITIVE and isinstance(value, list):
        return sorted(value, key=_sort_key)
    return value


def same_value(field: str, first: Any, second: Any) -> bool:
    """Whether two values of versioned ``field`` are equal in canonical form,
    compared as JSON."""
    return _sort_key(canonical_value(field, first)) == _sort_key(
        canonical_value(field, second)
    )


def same_content(first: Mapping[str, Any], second: Mapping[str, Any]) -> bool:
    """Whether two states of the versioned fields are the same version."""
    return all(
        same_value(field, first.get(field), second.get(field))
        for field in VERSIONED_FIELDS
    )


def _refuse_pending(db: Session, agent_id: int) -> None:
    obj = db.identity_map.get(identity_key(Agent, agent_id))
    if obj is None:
        return
    state: InstanceState[Any] = sa_inspect(obj)
    if not state.persistent:
        return
    pending = [
        field for field in VERSIONED_FIELDS if state.attrs[field].history.has_changes()
    ]
    if pending:
        raise PendingVersionedChange(
            f"agent {agent_id} has unflushed changes to {', '.join(pending)}"
        )


def _lock_and_read(
    db: Session, agent_id: int, authority: OwnedBy | SystemWrite
) -> dict[str, Any] | None:
    """Lock the agent row and read every column, or ``None`` if not found.

    SQLite ignores ``FOR UPDATE``; a no-op write takes its database write lock
    instead. It runs before any savepoint, so the transaction has already
    begun when one is opened. Raw SQL so ``updated_at``'s ``onupdate`` does
    not fire. PostgreSQL takes ``FOR NO KEY UPDATE``, which still serializes
    writers of this row but does not block inserts of rows that reference it
    (their foreign-key check takes ``FOR KEY SHARE``).
    """
    query = select(_AGENTS).where(_AGENTS.c.id == agent_id)
    if isinstance(authority, OwnedBy):
        query = query.where(
            owned_agent_clause(authority.user_id, authority.team_scope),
            _AGENTS.c.origin != AgentOrigin.WORKFORCE_GENERATED_MANAGER.value,
        )
    with db.no_autoflush:
        if db.get_bind().dialect.name == "sqlite":
            db.execute(
                text("UPDATE agents SET id = id WHERE id = :agent_id"),
                {"agent_id": agent_id},
            )
        else:
            query = query.with_for_update(key_share=True)
        row = db.execute(query).mappings().first()
    return dict(row) if row is not None else None


def _insert_version(
    db: Session,
    *,
    agent_id: int,
    version_no: int,
    kind: str,
    actor: ConfigActor,
    content: dict[str, Any],
) -> None:
    db.execute(
        insert(_VERSIONS).values(
            agent_id=agent_id,
            version_no=version_no,
            kind=kind,
            actor_kind=actor.kind,
            actor_user_id=actor.user_id,
            actor_key_prefix=actor.key_prefix,
            actor_source=actor.source,
            content=content,
        )
    )


def apply_agent_change(
    db: Session,
    agent_id: int,
    change: Change,
    *,
    actor: ConfigActor,
    authority: OwnedBy | SystemWrite,
    validate: Validate | None = None,
) -> AppliedChange:
    """Apply ``change`` to one agent and record a version when it changes one.

    ``change`` maps column names to new values, or is a callable that receives
    a deep copy of the locked row (every column) and returns that mapping.
    Callers that change several agents call this once per agent in ascending
    id order. A change records a version of kind ``save``.

    In order:

    1. Refuse with :class:`PendingVersionedChange` when the session holds
       unflushed changes to this agent's versioned fields.
    2. Lock the row and read its columns (``_lock_and_read``); with
       ``OwnedBy`` the ownership predicate is part of that query. Not found:
       ``not_found``.
    3. Compute the change from a deep copy of what was read. A requested
       column the ``agents`` table does not have raises ``ValueError``, even
       when other values would be dropped in step 4.
    4. Drop versioned values whose canonical form equals the stored one,
       compared as JSON (:func:`same_value`), and other values equal
       (``==``) to the stored one; they are not written.
    5. Decide: ``validate(row, change)`` (both deep copies) may raise; it
       must not commit, roll back, open another session or await, and a
       transaction it ends raises :class:`SessionBoundaryError`. It sees only
       the values left after step 4. Up to here the only statement that
       writes is SQLite's no-op lock write.
    6. In a savepoint (opening it flushes the session's other pending
       changes, as every ``begin_nested`` does): the baseline check records
       the stored state when the agent has no version yet (``initial``) or
       its newest version differs from it (``untracked``); one ``UPDATE``
       writes the remaining change, and none is issued when nothing is left
       (so ``updated_at`` keeps its value); a new version is recorded when a
       versioned field changed. The savepoint holds only these statements.
       A database error rolls back the savepoint only, then propagates.
    7. Expire ``updated_at`` on the session's agent object when it wrote.

    Never commits or rolls back the caller's transaction: on any exception
    the caller rolls back.
    """
    agent_id = int(agent_id)

    _refuse_pending(db, agent_id)

    row = _lock_and_read(db, agent_id, authority)
    if row is None:
        return AppliedChange("not_found", agent_id)

    requested = dict(
        change(copy.deepcopy(row)) if callable(change) else copy.deepcopy(change)
    )
    # Checked on every requested field, before equal values are dropped.
    unknown = set(requested) - (set(_AGENTS.c.keys()) - {"id"})
    if unknown:
        raise ValueError(
            f"Unsupported agent update fields: {', '.join(sorted(unknown))}"
        )
    pending = {
        field: value
        for field, value in requested.items()
        if (
            not same_value(field, value, row[field])
            if field in VERSIONED_FIELDS
            else value != row[field]
        )
    }

    if validate is not None:
        ends_before = root_transaction_end_count(db)
        validate(copy.deepcopy(row), copy.deepcopy(pending))
        ends_after = root_transaction_end_count(db)
        if ends_before is not None and ends_after != ends_before:
            raise SessionBoundaryError(
                f"validate ended the transaction of agent {agent_id}'s change"
            )

    stored = {field: row[field] for field in VERSIONED_FIELDS}
    changed = [field for field in VERSIONED_FIELDS if field in pending]
    with db.no_autoflush, db.begin_nested():
        latest = db.execute(
            select(_VERSIONS.c.version_no, _VERSIONS.c.content)
            .where(_VERSIONS.c.agent_id == agent_id)
            .order_by(_VERSIONS.c.version_no.desc())
            .limit(1)
        ).first()
        version_no = int(latest.version_no) if latest is not None else 0
        baseline_kind: str | None = None
        if latest is None:
            baseline_kind = "initial"
        elif not isinstance(latest.content, Mapping) or not same_content(
            latest.content, stored
        ):
            baseline_kind = "untracked"
        if baseline_kind is not None:
            version_no += 1
            _insert_version(
                db,
                agent_id=agent_id,
                version_no=version_no,
                kind=baseline_kind,
                actor=_BASELINE_ACTOR,
                content=stored,
            )
        if pending:
            db.execute(
                update(Agent)
                .where(Agent.id == agent_id)
                .values(**pending)
                .execution_options(synchronize_session="fetch")
            )
        if changed:
            version_no += 1
            _insert_version(
                db,
                agent_id=agent_id,
                version_no=version_no,
                kind=_CHANGE_VERSION_KIND,
                actor=actor,
                content={**stored, **{field: pending[field] for field in changed}},
            )
    obj = db.identity_map.get(identity_key(Agent, agent_id))
    if obj is not None and pending:
        db.expire(obj, ["updated_at"])
    return AppliedChange("applied" if changed else "no_change", agent_id, version_no)


def _write_version_contents(db: Session, rewrites: list[dict[str, Any]]) -> None:
    """Replace the ``content`` of several version rows in few statements.

    ``rewrites`` holds ``{"b_id": ..., "b_content": ...}`` items. The number of
    rows is unbounded, so the write must not cost one database round trip per
    row. psycopg2's ``executemany`` (SQLAlchemy's default ``executemany_mode``)
    still sends each parameter set separately, so PostgreSQL gets one
    ``UPDATE ... FROM (VALUES ...)`` per ``_VERSION_WRITE_CHUNK`` rows, which
    keeps each statement's size and bound parameter count (two per row)
    bounded. SQLite runs in-process, where an ``executemany`` costs no round
    trips.
    """
    if not rewrites:
        return
    if db.get_bind().dialect.name == "sqlite":
        db.execute(
            update(_VERSIONS)
            .where(_VERSIONS.c.id == bindparam("b_id"))
            .values(content=bindparam("b_content", type_=_VERSIONS.c.content.type)),
            rewrites,
        )
        return
    for start in range(0, len(rewrites), _VERSION_WRITE_CHUNK):
        chunk = rewrites[start : start + _VERSION_WRITE_CHUNK]
        rows = values(
            column("b_id", Integer), column("b_content", JSON), name="rewrites"
        ).data([(item["b_id"], item["b_content"]) for item in chunk])
        db.execute(
            update(_VERSIONS)
            .where(_VERSIONS.c.id == rows.c.b_id)
            .values(content=rows.c.b_content)
        )


def rewrite_agent_references(
    db: Session,
    agent_id: int,
    *,
    field: Literal["tool_categories", "knowledge_bases"],
    transform: Callable[[Any], Any],
    source: str,
) -> RewriteResult:
    """Rewrite a renamed reference in one agent row and all of its versions.

    For connector and knowledge base renames: ``transform`` maps the stored
    value of ``field`` to the value with the new name. It must be a pure
    function that only replaces names and returns any value it has nothing to
    replace in unchanged, including ``None``. The agent row and each version
    row go through the same ``transform``, so a version equal to the row
    before the rename is equal to it afterwards.

    Locks the row like :func:`apply_agent_change` but without an ownership
    check. Records no version, runs no baseline check. Never commits or rolls
    back: the rename's transaction does, and any exception must roll back
    the whole rename.
    """
    if field not in REFERENCE_FIELDS:
        raise ValueError(f"Not a reference field: {field}")
    agent_id = int(agent_id)

    def _rewritten(value: Any) -> Any:
        new_value = transform(copy.deepcopy(value))
        if value is None and new_value is not None:
            raise ValueError(
                f"transform for {field} turned an unset value into {new_value!r}"
            )
        return new_value

    row = _lock_and_read(db, agent_id, SystemWrite())
    if row is None:
        return RewriteResult("not_found", agent_id)
    with db.no_autoflush:
        new_value = _rewritten(row[field])
        agent_rewritten = new_value != row[field]
        if agent_rewritten:
            db.execute(
                update(Agent)
                .where(Agent.id == agent_id)
                .values({field: new_value})
                .execution_options(synchronize_session="fetch")
            )
        rewrites: list[dict[str, Any]] = []
        for version_id, content in db.execute(
            select(_VERSIONS.c.id, _VERSIONS.c.content)
            .where(_VERSIONS.c.agent_id == agent_id)
            .order_by(_VERSIONS.c.version_no)
        ).all():
            if not isinstance(content, Mapping) or field not in content:
                continue
            version_value = _rewritten(content[field])
            if version_value == content[field]:
                continue
            rewrites.append(
                {"b_id": version_id, "b_content": {**content, field: version_value}}
            )
        _write_version_contents(db, rewrites)
        rewritten_versions = len(rewrites)
    if not agent_rewritten and not rewritten_versions:
        return RewriteResult("unchanged", agent_id)
    logger.info(
        "agent references rewritten: source=%s agent_id=%d field=%s "
        "agent_row=%s versions=%d",
        source,
        agent_id,
        field,
        agent_rewritten,
        rewritten_versions,
    )
    return RewriteResult("rewritten", agent_id, rewritten_versions)
