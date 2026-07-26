"""Slice 4 of #757: scope propagation to nested and delegated executions.

Covers the AgentTool construction-time scope snapshot, the workforce
task-config scope persistence (``agent_config`` JSON, no schema migration),
the Task-backed snapshot loader, and the per-task resolution preferring a
persisted snapshot over the resolver — including across a simulated
process restart.
"""

import contextvars
from unittest.mock import MagicMock

import pytest

from xagent.core.execution_scope import (
    EXECUTION_SCOPE_AGENT_CONFIG_KEY,
    ExecutionScope,
    ExecutionScopeContext,
    get_execution_scope,
    resolve_execution_scope,
    set_execution_scope_resolver,
    set_execution_scope_snapshot_loader,
    turn_execution_scope,
)
from xagent.core.tools.adapters.vibe.agent_tool import AgentTool
from xagent.web.services.workforce_snapshot import build_workforce_task_config

SCOPE = ExecutionScope(
    sandbox_key_suffix="tenant-a",
    workspace_segments=("tenant-a",),
    memory_dimensions={"tenant": "a"},
)


@pytest.fixture(autouse=True)
def _clear_hooks():
    set_execution_scope_resolver(None)
    set_execution_scope_snapshot_loader(None)
    yield
    set_execution_scope_resolver(None)
    set_execution_scope_snapshot_loader(None)


class TestScopeSerialization:
    def test_round_trip(self):
        assert ExecutionScope.from_dict(SCOPE.to_dict()) == SCOPE

    def test_from_dict_revalidates(self):
        data = SCOPE.to_dict()
        data["workspace_segments"] = ["../escape"]
        with pytest.raises(Exception):
            ExecutionScope.from_dict(data)

    def test_to_dict_is_json_serializable(self):
        import json

        assert json.loads(json.dumps(SCOPE.to_dict())) == SCOPE.to_dict()


class TestAgentToolSnapshot:
    def _tool(self, **kwargs) -> AgentTool:
        return AgentTool(
            agent_id=7,
            agent_name="a",
            agent_description="d",
            session_factory=MagicMock(),
            user_id=1,
            **kwargs,
        )

    def test_captures_ambient_scope_at_construction(self):
        with ExecutionScopeContext(SCOPE):
            tool = self._tool()
        assert tool._execution_scope is SCOPE

    def test_explicit_scope_wins_over_ambient(self):
        other = ExecutionScope(sandbox_key_suffix="tenant-b")
        with ExecutionScopeContext(other):
            tool = self._tool(execution_scope=SCOPE)
        assert tool._execution_scope is SCOPE

    def test_unscoped_construction_snapshots_none(self):
        assert self._tool()._execution_scope is None


class TestWorkforceTaskConfigSnapshot:
    def test_scope_persisted_when_active(self):
        with ExecutionScopeContext(SCOPE):
            config = build_workforce_task_config(
                {"workforce": {"id": 3}}, workforce_run_id=9
            )
        assert config[EXECUTION_SCOPE_AGENT_CONFIG_KEY] == SCOPE.to_dict()
        assert config["workforce_run_id"] == 9

    def test_unscoped_config_is_byte_identical(self):
        config = build_workforce_task_config({"workforce": {"id": 3}})
        assert EXECUTION_SCOPE_AGENT_CONFIG_KEY not in config


class TestSnapshotLoaderPreference:
    def test_snapshot_preferred_over_resolver(self):
        resolver_calls = []

        def resolver(task_id):
            resolver_calls.append(task_id)
            return ExecutionScope(sandbox_key_suffix="from-resolver")

        set_execution_scope_resolver(resolver)
        set_execution_scope_snapshot_loader(
            lambda task_id: SCOPE if task_id == "42" else None
        )

        assert resolve_execution_scope(42) == SCOPE
        assert resolver_calls == []
        # Tasks without a snapshot still resolve through the resolver.
        assert resolve_execution_scope(43).sandbox_key_suffix == "from-resolver"
        assert resolver_calls == ["43"]

    def test_caller_supplied_snapshot_skips_registered_loader(self):
        loader_calls = []
        set_execution_scope_snapshot_loader(
            lambda task_id: loader_calls.append(task_id)
        )

        assert resolve_execution_scope(42, persisted_snapshot=SCOPE) == SCOPE
        assert loader_calls == []

    def test_caller_supplied_missing_snapshot_falls_back_to_resolver(self):
        loader_calls = []
        resolver_calls = []
        resolved = ExecutionScope(sandbox_key_suffix="from-resolver")
        set_execution_scope_snapshot_loader(
            lambda task_id: loader_calls.append(task_id)
        )
        set_execution_scope_resolver(
            lambda task_id: resolver_calls.append(task_id) or resolved
        )

        assert resolve_execution_scope(42, persisted_snapshot=None) == resolved
        assert loader_calls == []
        assert resolver_calls == ["42"]

    def test_loader_exception_fails_the_turn(self):
        def loader(task_id):
            raise RuntimeError("db down")

        set_execution_scope_snapshot_loader(loader)
        with pytest.raises(RuntimeError, match="db down"):
            resolve_execution_scope(42)

    def test_delegated_subtask_executes_scoped_after_restart(self):
        """A sub-task of a scoped parent executes fully scoped after a
        process restart: the snapshot persisted at creation is re-loaded
        per turn, with no resolver knowing the internal task id."""
        persisted = {"42": SCOPE.to_dict()}  # the agent_config JSON store

        def run_turn():
            # Fresh process: the loader is registered at startup; no
            # resolver mapping exists for the internally created task id.
            set_execution_scope_snapshot_loader(
                lambda task_id: (
                    ExecutionScope.from_dict(persisted[task_id])
                    if task_id in persisted
                    else None
                )
            )
            with turn_execution_scope("42"):
                return get_execution_scope()

        first = contextvars.copy_context().run(run_turn)
        set_execution_scope_snapshot_loader(None)
        second = contextvars.copy_context().run(run_turn)

        assert first == SCOPE
        assert first == second


class TestTaskBackedSnapshotLoader:
    @pytest.fixture
    def db_session(self, tmp_path):
        from xagent.web.models.database import Base, get_db, get_engine, init_db

        init_db(db_url=f"sqlite:///{tmp_path / 'scope_snapshot.db'}")
        db = next(get_db())
        try:
            yield db
        finally:
            db.close()
            Base.metadata.drop_all(bind=get_engine())

    def _make_task(self, db, agent_config):
        from xagent.web.models.task import Task, TaskStatus
        from xagent.web.models.user import User

        user = User(username="scope-user", password_hash="x")
        db.add(user)
        db.commit()
        task = Task(
            user_id=int(user.id),
            title="t",
            description="d",
            status=TaskStatus.PENDING,
            agent_config=agent_config,
        )
        db.add(task)
        db.commit()
        db.refresh(task)
        return task

    def test_loads_persisted_snapshot(self, db_session):
        from xagent.web.services.execution_scope_snapshot import (
            load_task_execution_scope_snapshot,
        )

        task = self._make_task(
            db_session, {EXECUTION_SCOPE_AGENT_CONFIG_KEY: SCOPE.to_dict()}
        )
        assert load_task_execution_scope_snapshot(str(task.id)) == SCOPE

    def test_task_without_snapshot_returns_none(self, db_session):
        from xagent.web.services.execution_scope_snapshot import (
            load_task_execution_scope_snapshot,
        )

        task = self._make_task(db_session, {"workforce_run_id": 9})
        assert load_task_execution_scope_snapshot(str(task.id)) is None

    def test_non_integer_task_id_returns_none(self, db_session):
        from xagent.web.services.execution_scope_snapshot import (
            load_task_execution_scope_snapshot,
        )

        assert load_task_execution_scope_snapshot("agent_7_ab12cd34") is None
