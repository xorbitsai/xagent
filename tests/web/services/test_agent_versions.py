"""Agent version rows written by ``apply_agent_change`` and its committing wrapper.

The fixtures run every dialect-sensitive test on SQLite (with the repository's
connection pragmas, so foreign keys are enforced) and on PostgreSQL. The
PostgreSQL half comes from ``disposable_database_factory``, which reads
``XAGENT_TEST_POSTGRES_URL`` and skips when it is unset.
"""

from __future__ import annotations

import ast
import inspect
import shutil
import subprocess
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import create_engine, event, func, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from tests.shared.postgres_disposable import disposable_database_factory
from xagent.db.sqlite import apply_sqlite_concurrency_pragmas
from xagent.web.models.agent import Agent
from xagent.web.models.agent_version import AgentVersion
from xagent.web.models.database import Base
from xagent.web.models.task import Task
from xagent.web.models.user import User
from xagent.web.services import agent_versions
from xagent.web.services.agent_store import AgentStore
from xagent.web.services.agent_versions import (
    EMPTY_EQUIVALENT,
    ORDER_INSENSITIVE,
    VERSIONED_FIELDS,
    AppliedChange,
    ConfigActor,
    OwnedBy,
    PendingVersionedChange,
    SessionBoundaryError,
    SystemWrite,
    apply_agent_change,
    model_slot_is_unset,
    rewrite_agent_references,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
WEB = ConfigActor.user(1, entry_point="web")


def _sqlite_sessions(path: Path) -> Iterator[sessionmaker[Session]]:
    engine = create_engine(f"sqlite:///{path}")
    apply_sqlite_concurrency_pragmas(engine)
    Base.metadata.create_all(engine)
    try:
        yield sessionmaker(autocommit=False, autoflush=False, bind=engine)
    finally:
        engine.dispose()


@pytest.fixture()
def sqlite_sessions(tmp_path: Path) -> Iterator[sessionmaker[Session]]:
    yield from _sqlite_sessions(tmp_path / "versions.db")


@pytest.fixture()
def pg_sessions() -> Iterator[sessionmaker[Session]]:
    with disposable_database_factory("xagent_agent_versions") as make_database:
        engine = make_database("v")
        Base.metadata.create_all(engine)
        yield sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture(
    params=["sqlite", pytest.param("postgresql", marks=pytest.mark.postgresql)]
)
def sessions(request: pytest.FixtureRequest) -> sessionmaker[Session]:
    name = "sqlite_sessions" if request.param == "sqlite" else "pg_sessions"
    return request.getfixturevalue(name)


def _user(db: Session, name: str = "owner") -> int:
    user = User(username=name, password_hash="x", is_admin=False)
    db.add(user)
    db.commit()
    return int(user.id)


def _agent(db: Session, user_id: int, name: str = "a", **fields: Any) -> int:
    values: dict[str, Any] = {
        "description": None,
        "instructions": "v1",
        "tool_categories": ["basic_tools"],
        **fields,
    }
    agent = AgentStore(db).add_agent(user_id=user_id, name=name, **values)
    db.commit()
    return int(agent.id)


def _versions(db: Session, agent_id: int) -> list[AgentVersion]:
    return list(
        db.execute(
            select(AgentVersion)
            .where(AgentVersion.agent_id == agent_id)
            .order_by(AgentVersion.version_no)
        ).scalars()
    )


def _row_content(db: Session, agent_id: int) -> dict[str, Any]:
    row = (
        db.execute(select(Agent.__table__).where(Agent.__table__.c.id == agent_id))
        .mappings()
        .one()
    )
    return {field: row[field] for field in VERSIONED_FIELDS}


def _save(db: Session, user_id: int, agent_id: int, updates: dict[str, Any]) -> Any:
    return AgentStore(db).update_agent_fields(
        user_id, agent_id, updates, actor=ConfigActor.user(user_id, entry_point="web")
    )


# -- constants and canonical form ------------------------------------------


def test_versioned_field_constants() -> None:
    assert len(VERSIONED_FIELDS) == 8
    assert set(VERSIONED_FIELDS) <= set(Agent.__table__.c.keys())
    assert "tool_categories" not in EMPTY_EQUIVALENT
    assert "knowledge_bases" not in ORDER_INSENSITIVE
    assert "suggested_prompts" not in ORDER_INSENSITIVE
    assert EMPTY_EQUIVALENT <= set(VERSIONED_FIELDS)
    assert ORDER_INSENSITIVE <= set(VERSIONED_FIELDS)


PCF_A = [{"name": "site", "max_length": 100, "description": None}]
PCF_B = [{"name": "site", "max_length": 200, "description": None}]

# Stored model slot values and whether each is unset (model_slot_is_unset).
SLOT_VALUES_UNSET: list[tuple[Any, bool]] = [
    (None, True),
    (False, True),
    ("", True),
    (0, True),
    (0.0, False),
    ([], False),
    ({}, False),
    (True, False),
    (1.0, False),
    ("0", False),
    (1, False),
]

# field -> (stored value, requested value, is it a new version?)
CANONICAL_MATRIX: dict[str, list[tuple[Any, Any, bool]]] = {
    "instructions": [
        (None, "", False),
        ("v1", "  v1\n", False),
        ("v1", "v2", True),
    ],
    "execution_mode": [("graph", "graph", False), ("graph", "flash", True)],
    "prompt_context_fields": [
        (None, [], False),
        (PCF_A, [dict(PCF_A[0])], False),
        (PCF_A, PCF_B, True),
    ],
    "suggested_prompts": [
        (None, [], False),
        (["a", "b"], ["a", "b"], False),
        (["a", "b"], ["b", "a"], True),
        (["a", ""], ["a"], True),
    ],
    "skills": [
        (None, [], False),
        (["a", "b"], ["b", "a"], False),
        (["a"], ["a", "c"], True),
    ],
    "models": [
        (None, {}, False),
        ({"general": 1, "compact": None}, {"general": 1}, False),
        ({"general": 1}, {"general": 2}, True),
        # Compared as JSON: ``True`` and ``1.0`` do not name model 1.
        ({"general": True}, {"general": 1}, True),
        ({"general": 1.0}, {"general": 1}, True),
        # Clearing a slot, with ``None`` or by leaving it out, is a change
        # only when the stored value is not unset (model_slot_is_unset).
        ({"general": 0}, {"general": None}, False),
        ({"general": 0}, {}, False),
        ({"general": False}, {"general": None}, False),
        ({"general": False}, {}, False),
        ({"general": ""}, {"general": None}, False),
        ({"general": ""}, {}, False),
        ({"general": 0.0}, {"general": None}, True),
        ({"general": 0.0}, {}, True),
        ({"general": []}, {"general": None}, True),
        ({"general": []}, {}, True),
        ({"general": {}}, {"general": None}, True),
        ({"general": {}}, {}, True),
    ],
    "knowledge_bases": [
        (None, [], False),
        (["a", "b"], ["a", "b"], False),
        (["a", "b"], ["b", "a"], True),
        (["a"], ["b"], True),
    ],
    "tool_categories": [
        (None, [], True),
        ([], None, True),
        (["a", "b"], ["b", "a"], False),
        (["a"], ["b"], True),
    ],
}


def test_canonical_matrix_covers_every_versioned_field() -> None:
    assert set(CANONICAL_MATRIX) == set(VERSIONED_FIELDS)


@pytest.mark.parametrize(
    ("value", "unset"),
    SLOT_VALUES_UNSET,
    ids=[repr(value) for value, _ in SLOT_VALUES_UNSET],
)
def test_model_slot_unset_rule(value: Any, unset: bool) -> None:
    assert model_slot_is_unset(value) is unset


def test_models_matrix_follows_the_unset_rule() -> None:
    """Every clear-a-slot row of the matrix expects a change exactly when the
    stored value is not unset."""
    clears = [
        (stored["general"], new)
        for stored, requested, new in CANONICAL_MATRIX["models"]
        if isinstance(stored, dict)
        and list(stored) == ["general"]
        and requested in ({"general": None}, {})
    ]
    assert len(clears) == 12
    for value, new in clears:
        assert new is (not model_slot_is_unset(value)), value


@pytest.mark.parametrize(
    ("field", "stored", "requested", "is_new_version"),
    [
        (field, stored, requested, new)
        for field, cases in CANONICAL_MATRIX.items()
        for stored, requested, new in cases
    ],
)
def test_canonical_form_decides_new_versions(
    sqlite_sessions: sessionmaker[Session],
    field: str,
    stored: Any,
    requested: Any,
    is_new_version: bool,
) -> None:
    with sqlite_sessions() as db:
        user_id = _user(db)
        agent_id = _agent(db, user_id)
        db.execute(
            Agent.__table__.update()
            .where(Agent.__table__.c.id == agent_id)
            .values({field: stored})
        )
        db.commit()
        result = apply_agent_change(
            db,
            agent_id,
            {field: requested},
            actor=WEB,
            authority=SystemWrite(),
        )
        db.commit()
        stored_after = _row_content(db, agent_id)[field]
        kinds = [v.kind for v in _versions(db, agent_id)]

    assert result.status == ("applied" if is_new_version else "no_change")
    assert kinds == (["initial", "save"] if is_new_version else ["initial"])
    # A value equal in canonical form is not written at all.
    assert stored_after == (requested if is_new_version else stored)


# -- version sequence, baseline check --------------------------------------


def test_saves_number_versions_from_one(sessions: sessionmaker[Session]) -> None:
    with sessions() as db:
        user_id = _user(db)
        agent_id = _agent(db, user_id)
        before = _row_content(db, agent_id)
        _save(db, user_id, agent_id, {"instructions": "v2"})
        _save(db, user_id, agent_id, {"instructions": "v3"})
        agent = _save(db, user_id, agent_id, {"instructions": "v3"})
        versions = _versions(db, agent_id)

    assert agent is not None and agent.instructions == "v3"
    assert [(v.version_no, v.kind) for v in versions] == [
        (1, "initial"),
        (2, "save"),
        (3, "save"),
    ]
    assert versions[0].content == before
    assert (versions[0].actor_kind, versions[0].actor_source) == ("system", "baseline")
    assert versions[0].actor_user_id is None
    assert (versions[2].actor_kind, versions[2].actor_user_id) == ("user", user_id)
    assert versions[2].actor_source == "web"
    assert versions[2].content["instructions"] == "v3"


def test_raw_sql_bypass_is_recorded_as_untracked(
    sessions: sessionmaker[Session],
) -> None:
    with sessions() as db:
        user_id = _user(db)
        agent_id = _agent(db, user_id)
        _save(db, user_id, agent_id, {"instructions": "v2"})
        db.execute(
            text("UPDATE agents SET instructions = 'bypass' WHERE id = :id"),
            {"id": agent_id},
        )
        db.commit()
        _save(db, user_id, agent_id, {"instructions": "v3"})
        versions = _versions(db, agent_id)

    assert [(v.version_no, v.kind) for v in versions] == [
        (1, "initial"),
        (2, "save"),
        (3, "untracked"),
        (4, "save"),
    ]
    assert versions[2].content["instructions"] == "bypass"
    assert (versions[2].actor_kind, versions[2].actor_source) == ("system", "baseline")
    assert versions[3].content["instructions"] == "v3"


def test_a_bypass_that_changes_only_a_value_type_is_recorded_as_untracked(
    sessions: sessionmaker[Session],
) -> None:
    with sessions() as db:
        user_id = _user(db)
        agent_id = _agent(db, user_id)
        _save(db, user_id, agent_id, {"models": {"general": 1}})
        # ``True == 1`` in Python; the baseline check compares as JSON.
        db.execute(
            Agent.__table__.update()
            .where(Agent.__table__.c.id == agent_id)
            .values(models={"general": True})
        )
        db.commit()
        _save(db, user_id, agent_id, {"instructions": "v3"})
        versions = _versions(db, agent_id)

    assert [(v.version_no, v.kind) for v in versions] == [
        (1, "initial"),
        (2, "save"),
        (3, "untracked"),
        (4, "save"),
    ]
    assert versions[2].content["models"]["general"] is True


@pytest.mark.parametrize("updates", [{"name": "renamed"}, {"instructions": "v2"}])
def test_agent_without_versions_gets_initial_first(
    sessions: sessionmaker[Session], updates: dict[str, Any]
) -> None:
    with sessions() as db:
        user_id = _user(db)
        db.execute(
            text(
                "INSERT INTO agents (user_id, visibility, name, instructions, "
                "execution_mode, tool_categories, widget_enabled, share_enabled, "
                "widget_voice_enabled, origin, status, created_at, updated_at) "
                "VALUES (:user_id, 'team', 'raw', 'v1', 'graph', '[\"basic_tools\"]', "
                "false, false, false, 'user', 'draft', CURRENT_TIMESTAMP, "
                "CURRENT_TIMESTAMP)"
            ),
            {"user_id": user_id},
        )
        db.commit()
        agent_id = int(db.execute(select(func.max(Agent.id))).scalar_one())
        before = _row_content(db, agent_id)
        _save(db, user_id, agent_id, updates)
        versions = _versions(db, agent_id)

    assert versions[0].kind == "initial"
    assert versions[0].content == before
    assert [v.kind for v in versions] == (
        ["initial"] if "name" in updates else ["initial", "save"]
    )


# -- outcomes, sync contract -----------------------------------------------


def test_three_outcomes(sessions: sessionmaker[Session]) -> None:
    with sessions() as db:
        owner = _user(db, "owner")
        other = _user(db, "other")
        agent_id = _agent(db, owner)

        def call(user_id: int, target: int, change: dict[str, Any]) -> AppliedChange:
            result = apply_agent_change(
                db,
                target,
                change,
                actor=WEB,
                authority=OwnedBy(user_id, None),
            )
            db.commit()
            return result

        missing = call(owner, agent_id + 100, {"instructions": "x"})
        not_owned = call(other, agent_id, {"instructions": "x"})
        versions_after_refusals = len(_versions(db, agent_id))
        applied = call(owner, agent_id, {"instructions": "x"})
        unchanged = call(owner, agent_id, {"instructions": "x"})

    assert missing == AppliedChange("not_found", agent_id + 100)
    assert not_owned == AppliedChange("not_found", agent_id)
    assert versions_after_refusals == 0
    assert applied == AppliedChange("applied", agent_id, 2)
    assert unchanged == AppliedChange("no_change", agent_id, 2)


def test_core_functions_are_synchronous() -> None:
    assert not inspect.iscoroutinefunction(apply_agent_change)
    assert not inspect.iscoroutinefunction(rewrite_agent_references)


def test_change_callable_gets_a_deep_copy(
    sqlite_sessions: sessionmaker[Session],
) -> None:
    def mutate_in_place(row: dict[str, Any]) -> dict[str, Any]:
        row["tool_categories"].append("web_tools")
        return {"tool_categories": row["tool_categories"]}

    with sqlite_sessions() as db:
        user_id = _user(db)
        agent_id = _agent(db, user_id)
        result = apply_agent_change(
            db,
            agent_id,
            mutate_in_place,
            actor=WEB,
            authority=SystemWrite(),
        )
        db.commit()
        versions = _versions(db, agent_id)
        stored = _row_content(db, agent_id)["tool_categories"]

    assert result.status == "applied"
    assert stored == ["basic_tools", "web_tools"]
    assert [(v.kind, v.content["tool_categories"]) for v in versions] == [
        ("initial", ["basic_tools"]),
        ("save", ["basic_tools", "web_tools"]),
    ]


def test_pending_versioned_change_is_refused(
    sqlite_sessions: sessionmaker[Session],
) -> None:
    with sqlite_sessions() as db:
        user_id = _user(db)
        agent_id = _agent(db, user_id)
        agent = db.get(Agent, agent_id)
        assert agent is not None
        agent.instructions = "unflushed"  # type: ignore[assignment]
        with pytest.raises(PendingVersionedChange):
            _save(db, user_id, agent_id, {"name": "renamed"})
        assert agent.instructions == "v1"
        assert _versions(db, agent_id) == []


def test_validate_that_ends_the_transaction_is_refused(
    sqlite_sessions: sessionmaker[Session],
) -> None:
    with sqlite_sessions() as db:
        user_id = _user(db)
        agent_id = _agent(db, user_id)
        with pytest.raises(SessionBoundaryError):
            apply_agent_change(
                db,
                agent_id,
                {"instructions": "x"},
                actor=WEB,
                authority=SystemWrite(),
                validate=lambda _row, _change: db.commit(),
            )
        db.rollback()
        assert _versions(db, agent_id) == []
        assert _row_content(db, agent_id)["instructions"] == "v1"


def test_database_error_undoes_only_the_failing_change(
    sessions: sessionmaker[Session],
) -> None:
    """A unique-name violation inside the core rolls back its savepoint only:
    the caller can still commit the rest of the transaction, and the failing
    agent keeps no half-written baseline row."""
    with sessions() as db:
        user_id = _user(db)
        first = _agent(db, user_id, "first")
        second = _agent(db, user_id, "second")

        def call(agent_id: int, change: dict[str, Any]) -> AppliedChange:
            return apply_agent_change(
                db,
                agent_id,
                change,
                actor=WEB,
                authority=SystemWrite(),
            )

        assert call(first, {"instructions": "kept"}).status == "applied"
        with pytest.raises(IntegrityError):
            call(second, {"name": "first", "instructions": "lost"})
        db.commit()
        first_versions = [v.kind for v in _versions(db, first)]
        second_versions = _versions(db, second)
        stored = _row_content(db, first)["instructions"]
        second_stored = _row_content(db, second)["instructions"]

    assert stored == "kept"
    assert first_versions == ["initial", "save"]
    assert second_versions == []
    assert second_stored == "v1"


def test_outer_savepoint_rollback_restores_the_session_object(
    sessions: sessionmaker[Session],
) -> None:
    """Rolling back a savepoint that contains a change also rolls back the
    in-memory agent object, not only the database row."""
    with sessions() as db:
        user_id = _user(db)
        agent_id = _agent(db, user_id)
        agent = db.get(Agent, agent_id)
        assert agent is not None and agent.instructions == "v1"
        outer = db.begin_nested()
        apply_agent_change(
            db,
            agent_id,
            {"instructions": "inside"},
            actor=WEB,
            authority=SystemWrite(),
        )
        assert agent.instructions == "inside"
        outer.rollback()
        in_memory = agent.instructions
        in_database = _row_content(db, agent_id)["instructions"]
        db.commit()
        versions = _versions(db, agent_id)

    assert (in_memory, in_database) == ("v1", "v1")
    assert versions == []


def test_deleted_agent_takes_its_versions_along(
    sessions: sessionmaker[Session],
) -> None:
    """On SQLite a new agent can get a deleted agent's id; the deleted agent's
    versions are gone by then, so the new agent starts at version 1."""
    with sessions() as db:
        user_id = _user(db)
        _agent(db, user_id, "keep")
        newest = _agent(db, user_id, "newest")
        _save(db, user_id, newest, {"instructions": "v2"})
        store = AgentStore(db)
        agent = store.get_owned_agent(user_id, newest)
        assert agent is not None
        store.stage_delete_agent(agent)
        db.commit()
        left = len(_versions(db, newest))
        replacement = _agent(db, user_id, "replacement")
        _save(db, user_id, replacement, {"instructions": "fresh"})
        versions = [(v.version_no, v.kind) for v in _versions(db, replacement)]
        dialect = db.get_bind().dialect.name

    assert left == 0
    if dialect == "sqlite":
        assert replacement == newest
    assert versions == [(1, "initial"), (2, "save")]


@pytest.mark.postgresql
def test_held_change_does_not_block_new_tasks(
    pg_sessions: sessionmaker[Session],
) -> None:
    """While a change holds the agent row lock, another connection can still
    insert a task that references the agent (its foreign-key check takes FOR
    KEY SHARE, which FOR NO KEY UPDATE does not block)."""
    with pg_sessions() as db:
        user_id = _user(db)
        agent_id = _agent(db, user_id)

    holder = pg_sessions()
    try:
        apply_agent_change(
            holder,
            agent_id,
            {"instructions": "held"},
            actor=WEB,
            authority=OwnedBy(user_id, None),
        )
        outcome: dict[str, Any] = {}

        def insert_task() -> None:
            started = time.monotonic()
            with pg_sessions() as other:
                other.add(Task(user_id=user_id, title="t", agent_id=agent_id))
                other.commit()
            outcome["seconds"] = time.monotonic() - started

        worker = threading.Thread(target=insert_task)
        worker.start()
        worker.join(timeout=5)
        assert not worker.is_alive()
        assert outcome["seconds"] < 1.0
        holder.commit()
    finally:
        holder.close()


# -- the committing wrapper ------------------------------------------------


def test_wrapper_signature_keeps_existing_parameters() -> None:
    params = inspect.signature(AgentStore.update_agent_fields).parameters
    original = [
        ("self", inspect.Parameter.POSITIONAL_OR_KEYWORD),
        ("user_id", inspect.Parameter.POSITIONAL_OR_KEYWORD),
        ("agent_id", inspect.Parameter.POSITIONAL_OR_KEYWORD),
        ("updates", inspect.Parameter.POSITIONAL_OR_KEYWORD),
        ("team_scope", inspect.Parameter.KEYWORD_ONLY),
        ("agent", inspect.Parameter.KEYWORD_ONLY),
    ]
    assert [(name, params[name].kind) for name, _ in original] == original
    assert list(params)[: len(original)] == [name for name, _ in original]
    assert params["agent"].default is None
    assert params["actor"].kind is inspect.Parameter.KEYWORD_ONLY
    assert params["actor"].default is None


def test_wrapper_returns_none_or_the_refreshed_agent(
    sqlite_sessions: sessionmaker[Session],
) -> None:
    with sqlite_sessions() as db:
        owner = _user(db, "owner")
        other = _user(db, "other")
        agent_id = _agent(db, owner)
        store = AgentStore(db)
        assert store.update_agent_fields(other, agent_id, {"name": "x"}) is None
        # The wrapper ended the transaction its row lock opened.
        assert not db.in_transaction()
        updated = store.update_agent_fields(owner, agent_id, {"instructions": "v2"})
        assert isinstance(updated, Agent)
        assert updated.instructions == "v2"
        versions = _versions(db, agent_id)

    # Omitting ``actor`` records ``unspecified``.
    assert (versions[-1].actor_kind, versions[-1].actor_user_id) == (
        "unspecified",
        None,
    )


def test_wrapper_rolls_back_before_raising(
    sqlite_sessions: sessionmaker[Session],
) -> None:
    with sqlite_sessions() as db:
        user_id = _user(db)
        agent_id = _agent(db, user_id)
        db.add(User(username="staged", password_hash="x", is_admin=False))
        db.flush()
        with pytest.raises(ValueError):
            _save(db, user_id, agent_id, {"instructions": "v2", "not_a_field": 1})
        staged = db.query(User).filter(User.username == "staged").count()
        assert staged == 0
        assert _versions(db, agent_id) == []
        assert _row_content(db, agent_id)["instructions"] == "v1"


def test_wrapper_validates_against_the_locked_row(
    sqlite_sessions: sessionmaker[Session],
) -> None:
    """The visibility guard sees the stored value, not the caller's stale
    object: the object still says ``team`` while the row says ``admins``."""
    with sqlite_sessions() as db:
        user_id = _user(db)
        agent_id = _agent(db, user_id)
        stale = db.get(Agent, agent_id)
        assert stale is not None and stale.visibility == "team"
        with sqlite_sessions() as other:
            other.execute(
                text("UPDATE agents SET visibility = 'admins' WHERE id = :id"),
                {"id": agent_id},
            )
            other.commit()
        assert stale.visibility == "team"
        with pytest.raises(PermissionError):
            AgentStore(db).update_agent_fields(
                user_id,
                agent_id,
                {"visibility": "team"},
                agent=stale,
                actor=WEB,
            )
        stored = db.execute(
            text("SELECT visibility FROM agents WHERE id = :id"), {"id": agent_id}
        ).scalar_one()
    assert stored == "admins"


def test_unchanged_save_writes_nothing(sessions: sessionmaker[Session]) -> None:
    """Saving every field with its stored value issues no UPDATE: no new
    version, and ``updated_at`` keeps its value."""
    with sessions() as db:
        user_id = _user(db)
        agent_id = _agent(db, user_id, description="d")
        _save(db, user_id, agent_id, {"instructions": "v2"})
        before = db.execute(
            select(Agent.__table__.c.updated_at).where(Agent.__table__.c.id == agent_id)
        ).scalar_one()
        time.sleep(0.01)
        _save(
            db,
            user_id,
            agent_id,
            {"name": "a", "description": "d", "instructions": "v2", "skills": []},
        )
        after = db.execute(
            select(Agent.__table__.c.updated_at).where(Agent.__table__.c.id == agent_id)
        ).scalar_one()
        count = len(_versions(db, agent_id))
    assert after == before
    assert count == 2


def _shell_calls() -> list[ast.Call]:
    if shutil.which("git") is None:
        pytest.skip("git is not available")
    listed = subprocess.run(
        ["git", "grep", "-l", "update_agent_fields(", "--", "src"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    calls = []
    for relative in listed:
        tree = ast.parse((REPO_ROOT / relative).read_text())
        calls.extend(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "update_agent_fields"
        )
    return calls


def test_every_wrapper_call_names_its_actor() -> None:
    calls = _shell_calls()
    # web agents API 8, builder tool 1, personal-key skills, connectors and
    # update 3, import tool 1. A new call site updates this count and passes
    # ``actor``.
    assert len(calls) == 13
    assert all(any(k.arg == "actor" for k in call.keywords) for call in calls)


# -- entry points ----------------------------------------------------------


def _holder_names() -> Any:
    from xagent.web.services.mcp_server_management import HolderConnectorNames

    return HolderConnectorNames(
        visible=frozenset({"github"}),
        own_active=frozenset({"github"}),
        own_inactive=frozenset(),
        team_keys=frozenset(),
    )


@pytest.mark.parametrize("entry", ["skills", "connectors", "update"])
def test_personal_key_entries_record_api_key_versions(
    sqlite_sessions: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
    entry: str,
) -> None:
    from xagent.web.services import agent_management
    from xagent.web.services.agent_management import AgentManagementRuntime

    monkeypatch.setattr(agent_management, "get_session_local", lambda: sqlite_sessions)
    with sqlite_sessions() as db:
        user_id = _user(db)
        agent_id = _agent(db, user_id, skills=["b", "a"])

    if entry == "skills":
        written = AgentManagementRuntime._set_agent_skills_sync(
            user_id=user_id,
            key_prefix="xk_abc",
            agent_id=agent_id,
            skills=["a", "b"],
            visible=frozenset({"a", "b"}),
        )
        # Reordering only is no change; the response reads back what is stored.
        assert written == (("b", "a"), ())
        written = AgentManagementRuntime._set_agent_skills_sync(
            user_id=user_id,
            key_prefix="xk_abc",
            agent_id=agent_id,
            skills=["a", "c"],
            visible=frozenset({"a", "c"}),
        )
        assert written == (("a", "c"), ("c",))
    elif entry == "update":
        detail = AgentManagementRuntime._update_agent_sync(
            user_id=user_id,
            key_prefix="xk_abc",
            agent_id=agent_id,
            changes={"instructions": "changed through the personal key"},
        )
        assert detail is not None
    else:
        snapshot = AgentManagementRuntime._set_agent_connectors_sync(
            user_id=user_id,
            key_prefix="xk_abc",
            agent_id=agent_id,
            connectors=["github"],
            all_connectors=False,
            names=_holder_names(),
        )
        assert snapshot is not None

    with sqlite_sessions() as db:
        versions = _versions(db, agent_id)
        row = _row_content(db, agent_id)
    if entry == "connectors":
        assert row["tool_categories"] == ["basic_tools", "mcp:github"]
    assert versions[-1].content == row
    assert (
        versions[-1].actor_kind,
        versions[-1].actor_user_id,
        versions[-1].actor_key_prefix,
        versions[-1].actor_source,
    ) == ("api_key", user_id, "xk_abc", "v1")


def _bundle(persona: str) -> Any:
    from xagent.migration.bundle import MigrationBundle, PersonaItem

    return MigrationBundle(
        source="hermes",
        source_root="/nowhere",
        agent_name="Hermes",
        persona=PersonaItem(instructions=persona),
    )


def test_import_rerun_records_an_import_version(
    sqlite_sessions: sessionmaker[Session],
) -> None:
    from xagent.migration.loaders import MigrationLoader

    with sqlite_sessions() as db:
        user = User(username="importer", password_hash="x", is_admin=False)
        db.add(user)
        db.commit()
        MigrationLoader(db, user=user).load(_bundle("first persona"))
        report = MigrationLoader(db, user=user).load(_bundle("second persona"))
        agent_id = int(
            db.execute(select(Agent.id).where(Agent.user_id == user.id)).scalar_one()
        )
        versions = _versions(db, agent_id)
        row = _row_content(db, agent_id)

    assert report.agent_reused is True
    assert row["instructions"] == "second persona"
    assert [v.kind for v in versions] == ["initial", "save"]
    assert versions[-1].content == row
    assert (versions[-1].actor_kind, versions[-1].actor_source) == ("user", "import")
    assert versions[-1].actor_user_id == int(user.id)


def _move_agent_away(
    sessions: sessionmaker[Session], agent_id: int, new_owner: int
) -> None:
    """Hand the agent to another user from another session, so the caller's
    ownership no longer holds when its write locks the row."""
    with sessions() as other:
        other.execute(
            text("UPDATE agents SET user_id = :owner WHERE id = :id"),
            {"owner": new_owner, "id": agent_id},
        )
        other.commit()


def _owner_other_agent(sessions: sessionmaker[Session]) -> tuple[int, int, int]:
    with sessions() as db:
        user_id = _user(db, "owner")
        other_id = _user(db, "other")
        agent_id = _agent(db, user_id)
    return user_id, other_id, agent_id


@pytest.mark.asyncio
async def test_web_save_records_a_web_version(
    sqlite_sessions: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    from xagent.web.api import agents as agents_api
    from xagent.web.api.agents import AgentUpdateRequest

    user_id, _other, agent_id = _owner_other_agent(sqlite_sessions)

    async def visible(*_args: Any, **_kwargs: Any) -> None:
        return None

    monkeypatch.setattr(agents_api, "_validate_knowledge_bases_exist", visible)
    with sqlite_sessions() as db:
        user = db.get(User, user_id)
        response = await agents_api.update_agent(
            agent_id,
            AgentUpdateRequest(instructions="from web"),
            current_user=user,
            db=db,
        )
        versions = _versions(db, agent_id)
        row = _row_content(db, agent_id)

    assert response.instructions == "from web"
    assert versions[-1].content == row
    assert (versions[-1].actor_kind, versions[-1].actor_source) == ("user", "web")
    assert versions[-1].actor_user_id == user_id


@pytest.mark.asyncio
async def test_web_save_of_an_agent_gone_meanwhile_is_404(
    sqlite_sessions: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    from fastapi import HTTPException

    from xagent.web.api import agents as agents_api
    from xagent.web.api.agents import AgentUpdateRequest

    user_id, other_id, agent_id = _owner_other_agent(sqlite_sessions)

    async def taken_away(*_args: Any, **_kwargs: Any) -> None:
        _move_agent_away(sqlite_sessions, agent_id, other_id)

    monkeypatch.setattr(agents_api, "_validate_knowledge_bases_exist", taken_away)
    with sqlite_sessions() as db:
        user = db.get(User, user_id)
        with pytest.raises(HTTPException) as raised:
            await agents_api.update_agent(
                agent_id,
                AgentUpdateRequest(instructions="lost"),
                current_user=user,
                db=db,
            )
        versions = _versions(db, agent_id)
        row = _row_content(db, agent_id)

    assert raised.value.status_code == 404
    assert versions == []
    assert row["instructions"] == "v1"


@pytest.mark.asyncio
async def test_builder_tool_records_a_builder_version(
    sqlite_sessions: sessionmaker[Session],
) -> None:
    from xagent.core.tools.adapters.vibe.agent_tool import UpdateAgentTool

    with sqlite_sessions() as db:
        user_id = _user(db)
        agent_id = _agent(db, user_id)

    result = await UpdateAgentTool(
        session_factory=sqlite_sessions, user_id=user_id
    ).run_json_async({"agent_id": agent_id, "instructions": "from builder"})

    with sqlite_sessions() as db:
        versions = _versions(db, agent_id)
        row = _row_content(db, agent_id)
    assert result["status"] == "success"
    assert row["instructions"] == "from builder"
    assert versions[-1].content == row
    assert (versions[-1].actor_kind, versions[-1].actor_source) == (
        "user",
        "builder_tool",
    )


class _StoreThatLosesTheAgent(AgentStore):
    """Hands the agent to another user right before the write locks it."""

    sessions: sessionmaker[Session]
    agent_id: int
    new_owner: int

    def update_agent_fields(self, *args: Any, **kwargs: Any) -> Any:
        _move_agent_away(self.sessions, self.agent_id, self.new_owner)
        return super().update_agent_fields(*args, **kwargs)


@pytest.mark.parametrize("entry", ["skills", "connectors"])
def test_personal_key_write_of_an_agent_gone_meanwhile_is_not_found(
    sqlite_sessions: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
    entry: str,
) -> None:
    from xagent.web.services import agent_management
    from xagent.web.services.agent_management import AgentManagementRuntime

    monkeypatch.setattr(agent_management, "get_session_local", lambda: sqlite_sessions)
    with sqlite_sessions() as db:
        user_id = _user(db, "owner")
        other_id = _user(db, "other")
        agent_id = _agent(db, user_id, skills=["a"], tool_categories=["basic_tools"])
    monkeypatch.setattr(
        agent_management,
        "AgentStore",
        _losing_store(sqlite_sessions, agent_id, other_id),
    )

    if entry == "skills":
        result = AgentManagementRuntime._set_agent_skills_sync(
            user_id=user_id,
            key_prefix="xk_abc",
            agent_id=agent_id,
            skills=["a", "b"],
            visible=frozenset({"a", "b"}),
        )
    else:
        names = _holder_names()
        result = AgentManagementRuntime._set_agent_connectors_sync(
            user_id=user_id,
            key_prefix="xk_abc",
            agent_id=agent_id,
            connectors=["github"],
            all_connectors=False,
            names=names,
        )

    with sqlite_sessions() as db:
        versions = _versions(db, agent_id)
        row = _row_content(db, agent_id)
    assert result is None
    assert versions == []
    assert (row["skills"], row["tool_categories"]) == (["a"], ["basic_tools"])


def _losing_store(
    sessions: sessionmaker[Session], agent_id: int, new_owner: int
) -> type[AgentStore]:
    return type(
        "Store",
        (_StoreThatLosesTheAgent,),
        {"sessions": sessions, "agent_id": agent_id, "new_owner": new_owner},
    )


def test_fields_callers_may_not_write_are_refused_even_unchanged(
    sqlite_sessions: sessionmaker[Session],
) -> None:
    with sqlite_sessions() as db:
        user_id = _user(db)
        agent_id = _agent(db, user_id)
        with pytest.raises(ValueError):
            _save(db, user_id, agent_id, {"user_id": user_id})
        with pytest.raises(ValueError):
            apply_agent_change(
                db,
                agent_id,
                {"id": agent_id},
                actor=WEB,
                authority=SystemWrite(),
            )
        db.rollback()
        assert _versions(db, agent_id) == []


def test_unchanged_admins_visibility_is_not_rechecked(
    sqlite_sessions: sessionmaker[Session],
) -> None:
    """A non-admin owner whose personal agent was created admins-only (agent
    creation does not run the visibility guard) can save it with the same
    visibility: only a value that changes is checked."""
    with sqlite_sessions() as db:
        user_id = _user(db)
        agent_id = _agent(db, user_id, visibility="admins")
        updated = _save(
            db, user_id, agent_id, {"visibility": "admins", "instructions": "v2"}
        )
        assert updated is not None and updated.visibility == "admins"
        with pytest.raises(PermissionError):
            _save(db, user_id, agent_id, {"visibility": "team"})


def test_workforce_manager_agents_are_not_found(
    sqlite_sessions: sessionmaker[Session],
) -> None:
    from xagent.web.models.agent import AgentOrigin

    with sqlite_sessions() as db:
        user_id = _user(db)
        agent_id = _agent(
            db, user_id, origin=AgentOrigin.WORKFORCE_GENERATED_MANAGER.value
        )
        result = apply_agent_change(
            db,
            agent_id,
            {"instructions": "x"},
            actor=WEB,
            authority=OwnedBy(user_id, None),
        )
        db.rollback()
        assert result == AppliedChange("not_found", agent_id)
        assert _versions(db, agent_id) == []


@pytest.mark.asyncio
async def test_logo_upload_of_an_agent_gone_meanwhile_is_404(
    sqlite_sessions: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    from fastapi import HTTPException

    from xagent.web.api import agents as agents_api

    user_id, other_id, agent_id = _owner_other_agent(sqlite_sessions)
    monkeypatch.setattr(agents_api, "_save_logo", lambda *_args: "/uploads/new.png")
    monkeypatch.setattr(
        agents_api, "AgentStore", _losing_store(sqlite_sessions, agent_id, other_id)
    )
    with sqlite_sessions() as db:
        user = db.get(User, user_id)
        with pytest.raises(HTTPException) as raised:
            await agents_api.upload_agent_logo(
                agent_id, "data:image/png;base64,", current_user=user, db=db
            )
        versions = _versions(db, agent_id)

    assert raised.value.status_code == 404
    assert versions == []


@pytest.mark.asyncio
async def test_builder_tool_update_of_an_agent_gone_meanwhile_is_not_found(
    sqlite_sessions: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    from xagent.core.tools.adapters.vibe import agent_tool
    from xagent.core.tools.adapters.vibe.agent_tool import UpdateAgentTool

    with sqlite_sessions() as db:
        user_id = _user(db, "owner")
        other_id = _user(db, "other")
        agent_id = _agent(db, user_id)
    monkeypatch.setattr(
        agent_tool, "AgentStore", _losing_store(sqlite_sessions, agent_id, other_id)
    )

    result = await UpdateAgentTool(
        session_factory=sqlite_sessions, user_id=user_id
    ).run_json_async({"agent_id": agent_id, "instructions": "lost"})

    with sqlite_sessions() as db:
        versions = _versions(db, agent_id)
        row = _row_content(db, agent_id)
    assert result["status"] == "error"
    assert "not found" in result["message"]
    assert versions == []
    assert row["instructions"] == "v1"


def test_import_rerun_on_an_unmanageable_agent_reports_a_skip(
    sqlite_sessions: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    from xagent.migration import loaders
    from xagent.migration.loaders import MigrationLoader

    with sqlite_sessions() as db:
        user = User(username="importer", password_hash="x", is_admin=False)
        db.add(user)
        db.commit()
        other_id = _user(db, "other")
        MigrationLoader(db, user=user).load(_bundle("first persona"))
        agent_id = int(
            db.execute(select(Agent.id).where(Agent.user_id == user.id)).scalar_one()
        )
        monkeypatch.setattr(
            loaders, "AgentStore", _losing_store(sqlite_sessions, agent_id, other_id)
        )
        report = MigrationLoader(db, user=user).load(_bundle("second persona"))
        row = _row_content(db, agent_id)
        versions = _versions(db, agent_id)

    assert report.agent_reused is False
    assert report.errors == [
        "agent Hermes: not updated, the importing user can no longer manage it"
    ]
    assert row["instructions"] == "first persona"
    assert versions == []


# -- several agents, reference rewrites ------------------------------------


def test_system_rewrite_of_several_agents_rolls_back_together(
    sessions: sessionmaker[Session],
) -> None:
    """System callers change agents one by one in ascending id order inside
    one transaction; one failure rolls the whole rewrite back."""
    with sessions() as db:
        user_id = _user(db)
        ids = [_agent(db, user_id, f"a{i}") for i in range(3)]

        def refuse_last(row: dict[str, Any], _change: dict[str, Any]) -> None:
            if row["id"] == ids[-1]:
                raise RuntimeError("refused")

        with pytest.raises(RuntimeError):
            for agent_id in sorted(ids):
                apply_agent_change(
                    db,
                    agent_id,
                    {"instructions": "rewritten"},
                    actor=ConfigActor.system("system_reconcile"),
                    authority=SystemWrite(),
                    validate=refuse_last,
                )
        db.rollback()
        stored = [_row_content(db, agent_id)["instructions"] for agent_id in ids]
        version_rows = db.execute(select(func.count(AgentVersion.id))).scalar_one()

    assert stored == ["v1", "v1", "v1"]
    assert version_rows == 0


def _kb_rename(old: str, new: str):  # type: ignore[no-untyped-def]
    def transform(value: Any) -> Any:
        if not isinstance(value, list) or old not in value:
            return value
        return [new if name == old else name for name in value]

    return transform


def test_reference_rewrite_covers_the_row_and_every_version(
    sessions: sessionmaker[Session],
) -> None:
    """The knowledge-base rename shape: every candidate agent in ascending id
    order, whether or not its current row still uses the old name."""
    with sessions() as db:
        user_id = _user(db)
        current = _agent(db, user_id, "current", knowledge_bases=["old", "x"])
        _save(db, user_id, current, {"instructions": "v2"})
        former = _agent(db, user_id, "former", knowledge_bases=["old"])
        _save(db, user_id, former, {"knowledge_bases": ["y"]})
        unrelated = _agent(db, user_id, "unrelated")
        # An unset list must stay unset (``add_agent`` would store ``[]``).
        db.execute(
            Agent.__table__.update()
            .where(Agent.__table__.c.id == unrelated)
            .values(knowledge_bases=None)
        )
        db.commit()
        _save(db, user_id, unrelated, {"instructions": "v2"})
        empty = _agent(db, user_id, "empty", knowledge_bases=[])
        _save(db, user_id, empty, {"instructions": "v2"})

        results = {
            agent_id: rewrite_agent_references(
                db,
                agent_id,
                field="knowledge_bases",
                transform=_kb_rename("old", "new"),
                source="kb_rename",
            )
            for agent_id in sorted([current, former, unrelated, empty])
        }
        db.commit()
        rows = {i: _row_content(db, i) for i in (current, former, unrelated, empty)}
        versions = {i: _versions(db, i) for i in (current, former, unrelated, empty)}
        # The row and its newest version stay equal: the next save records
        # no untracked version.
        for agent_id in (current, former, unrelated, empty):
            _save(db, user_id, agent_id, {"name": f"renamed-{agent_id}"})
        kinds_after = {i: [v.kind for v in _versions(db, i)] for i in rows}

    assert results[current].status == "rewritten"
    assert results[current].rewritten_versions == 2
    assert rows[current]["knowledge_bases"] == ["new", "x"]
    assert [v.content["knowledge_bases"] for v in versions[current]] == [
        ["new", "x"],
        ["new", "x"],
    ]
    assert results[former].status == "rewritten"
    assert results[former].rewritten_versions == 1
    assert rows[former]["knowledge_bases"] == ["y"]
    assert [v.content["knowledge_bases"] for v in versions[former]] == [
        ["new"],
        ["y"],
    ]
    assert results[unrelated].status == "unchanged"
    assert rows[unrelated]["knowledge_bases"] is None
    assert [v.content["knowledge_bases"] for v in versions[unrelated]] == [
        None,
        None,
    ]
    # An empty list stays empty in the row and in every version.
    assert results[empty].status == "unchanged"
    assert rows[empty]["knowledge_bases"] == []
    assert [v.content["knowledge_bases"] for v in versions[empty]] == [[], []]
    assert kinds_after == {
        current: ["initial", "save"],
        former: ["initial", "save"],
        unrelated: ["initial", "save"],
        empty: ["initial", "save"],
    }


def _rewrite_many_versions(
    sessions: sessionmaker[Session],
) -> tuple[Any, int, list[AgentVersion], dict[str, Any]]:
    """Rewrite an agent with 200 versions, 150 of which use the old name.

    Returns the result, the number of ``UPDATE agent_versions`` executions,
    those 200 versions afterwards and the agent row's content."""
    numbers = range(1001, 1201)
    with sessions() as db:
        user_id = _user(db)
        agent_id = _agent(db, user_id, knowledge_bases=["x"])
        content = _row_content(db, agent_id)
        db.execute(
            AgentVersion.__table__.insert(),
            [
                {
                    "agent_id": agent_id,
                    "version_no": no,
                    "kind": "save",
                    "actor_kind": "user",
                    "actor_user_id": user_id,
                    "content": {
                        **content,
                        "instructions": f"v{no}",
                        "knowledge_bases": ["old", f"kb{no}"] if no <= 1150 else ["y"],
                    },
                }
                for no in numbers
            ],
        )
        db.commit()
        statements: list[str] = []

        def record(conn, cursor, statement, parameters, context, executemany):  # type: ignore[no-untyped-def]
            if statement.lstrip().upper().startswith("UPDATE AGENT_VERSIONS"):
                statements.append(statement)

        engine = db.get_bind()
        event.listen(engine, "before_cursor_execute", record)
        try:
            result = rewrite_agent_references(
                db,
                agent_id,
                field="knowledge_bases",
                transform=_kb_rename("old", "new"),
                source="kb_rename",
            )
        finally:
            event.remove(engine, "before_cursor_execute", record)
        db.commit()
        versions = [v for v in _versions(db, agent_id) if v.version_no in numbers]
        row = _row_content(db, agent_id)
    return result, len(statements), versions, row


def _assert_many_versions_rewritten(
    result: Any, versions: list[AgentVersion], row: dict[str, Any]
) -> None:
    numbers = range(1001, 1201)
    assert result.status == "rewritten"
    assert result.rewritten_versions == 150
    assert row["knowledge_bases"] == ["x"]
    assert [v.content["knowledge_bases"] for v in versions] == [
        ["new", f"kb{no}"] if no <= 1150 else ["y"] for no in numbers
    ]
    # Only the rewritten field changed.
    assert [v.content["instructions"] for v in versions] == [f"v{no}" for no in numbers]


def test_reference_rewrite_of_many_versions_is_one_statement(
    sessions: sessionmaker[Session],
) -> None:
    """Version rows are unbounded, so rewriting them must not cost one
    round trip per row: 150 of 200 versions change in a single UPDATE."""
    result, updates, versions, row = _rewrite_many_versions(sessions)

    assert updates == 1
    _assert_many_versions_rewritten(result, versions, row)


@pytest.mark.postgresql
def test_reference_rewrite_splits_many_versions_into_chunks(
    pg_sessions: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    """PostgreSQL writes at most ``_VERSION_WRITE_CHUNK`` rows per statement."""
    monkeypatch.setattr(agent_versions, "_VERSION_WRITE_CHUNK", 50)
    result, updates, versions, row = _rewrite_many_versions(pg_sessions)

    assert updates == 3
    _assert_many_versions_rewritten(result, versions, row)


def test_reference_rewrite_rejects_other_fields_and_none_to_empty(
    sqlite_sessions: sessionmaker[Session],
) -> None:
    with sqlite_sessions() as db:
        user_id = _user(db)
        agent_id = _agent(db, user_id, tool_categories=None)
        with pytest.raises(ValueError):
            rewrite_agent_references(
                db,
                agent_id,
                field="models",  # type: ignore[arg-type]
                transform=lambda value: value,
                source="test",
            )
        with pytest.raises(ValueError):
            rewrite_agent_references(
                db,
                agent_id,
                field="tool_categories",
                transform=lambda value: value or [],
                source="connector_rename",
            )
        db.rollback()
        missing = rewrite_agent_references(
            db,
            agent_id + 100,
            field="tool_categories",
            transform=lambda value: value,
            source="connector_rename",
        )
        assert missing.status == "not_found"


def test_failed_reference_rewrite_rolls_back_the_whole_rename(
    sessions: sessionmaker[Session],
) -> None:
    with sessions() as db:
        user_id = _user(db)
        first = _agent(db, user_id, "first", tool_categories=["mcp:old"])
        second = _agent(db, user_id, "second", tool_categories=["mcp:old"])

        def rename(value: Any) -> Any:
            if value == ["mcp:old"] and rename.calls:
                raise RuntimeError("second agent fails")
            rename.calls += 1
            return ["mcp:new"] if value == ["mcp:old"] else value

        rename.calls = 0  # type: ignore[attr-defined]
        with pytest.raises(RuntimeError):
            for agent_id in sorted([first, second]):
                rewrite_agent_references(
                    db,
                    agent_id,
                    field="tool_categories",
                    transform=rename,
                    source="connector_rename",
                )
        db.rollback()
        stored = [_row_content(db, i)["tool_categories"] for i in (first, second)]

    assert stored == [["mcp:old"], ["mcp:old"]]


@pytest.mark.postgresql
def test_concurrent_saves_of_one_agent_queue_on_the_row_lock(
    pg_sessions: sessionmaker[Session],
) -> None:
    """Two connections change the same agent at once. The second waits for
    the first's row lock, then numbers its version after the first's."""
    with pg_sessions() as db:
        user_id = _user(db)
        agent_id = _agent(db, user_id)

    first_locked = threading.Event()
    errors: list[BaseException] = []

    def save(text_: str, hold: bool) -> None:
        try:
            with pg_sessions() as db:
                if not hold:
                    assert first_locked.wait(timeout=10)
                apply_agent_change(
                    db,
                    agent_id,
                    {"instructions": text_},
                    actor=WEB,
                    authority=OwnedBy(user_id, None),
                )
                if hold:
                    first_locked.set()
                    time.sleep(0.5)
                db.commit()
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)
            first_locked.set()

    threads = [
        threading.Thread(target=save, args=("first", True)),
        threading.Thread(target=save, args=("second", False)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=15)

    with pg_sessions() as db:
        versions = [(v.version_no, v.kind) for v in _versions(db, agent_id)]
        contents = [v.content["instructions"] for v in _versions(db, agent_id)]
        stored = _row_content(db, agent_id)["instructions"]

    assert errors == []
    assert versions == [(1, "initial"), (2, "save"), (3, "save")]
    assert contents == ["v1", "first", "second"]
    assert stored == "second"


# -- schema ----------------------------------------------------------------


def test_every_foreign_key_column_leads_an_index() -> None:
    table = AgentVersion.__table__
    leading = {index.columns[0].name for index in table.indexes} | {
        constraint.columns[0].name
        for constraint in table.constraints
        if constraint.__class__.__name__ == "UniqueConstraint"
    }
    foreign_key_columns = {fk.parent.name for fk in table.foreign_keys}
    assert foreign_key_columns == {"agent_id", "actor_user_id"}
    assert foreign_key_columns <= leading
