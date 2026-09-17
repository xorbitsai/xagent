"""Slice 4 of #757: scope propagation to nested and delegated executions.

Covers the AgentTool construction-time scope snapshot, the workforce
task-config scope persistence (``agent_config`` JSON, no schema migration),
the Task-backed snapshot loader, and the per-task resolution's
resolver/snapshot precedence — including across a simulated process
restart.

With a resolver registered, the
persisted snapshot is a corroborating *candidate* for the resolver's
authoritative answer, not an override of it (a namespace-affecting
disagreement fails the turn instead of the snapshot silently winning). With
no resolver registered — the case that matters for standalone/workforce
restart-and-resume — the snapshot is still the sole answer, byte-identical
to before. See ``TestSnapshotCandidatePrecedence`` below and
``tests/core/test_execution_scope.py::TestSnapshotCandidateAuthority`` for
the full precedence matrix.

The resolver/snapshot-loader globals are reset by the root-level
``isolate_execution_scope_hooks`` autouse fixture in ``tests/conftest.py``,
not by a fixture in this module.
"""

import contextvars
from unittest.mock import MagicMock

import pytest

from tests.shared.execution_scope import register_scope_resolver
from xagent.core.execution_scope import (
    EXECUTION_SCOPE_AGENT_CONFIG_KEY,
    ExecutionScope,
    ExecutionScopeAuthorityError,
    ExecutionScopeContext,
    get_execution_scope,
    resolve_execution_scope,
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

    def test_persists_the_current_shape_version_not_the_ambient_scopes_version(
        self,
    ):
        """The ambient scope may itself have been decoded from an older
        persisted snapshot (``version`` stamped at 0 or some prior shape).
        Re-persisting it here for a *new* workforce sub-task must stamp the
        current shape version -- the dict is being built now, in the
        current shape -- not propagate the decoded-from scope's stale
        version, which would make this brand-new snapshot look pre-existing
        forever and get permanently ignored as a candidate by
        resolve_execution_scope's stale-version check."""
        from xagent.core.execution_scope import EXECUTION_SCOPE_SHAPE_VERSION

        legacy_data = SCOPE.to_dict()
        legacy_data["version"] = 0
        decoded_legacy_scope = ExecutionScope.from_dict(legacy_data)
        assert decoded_legacy_scope.version == 0

        with ExecutionScopeContext(decoded_legacy_scope):
            config = build_workforce_task_config({"workforce": {"id": 3}})

        assert (
            config[EXECUTION_SCOPE_AGENT_CONFIG_KEY]["version"]
            == EXECUTION_SCOPE_SHAPE_VERSION
        )


class TestSnapshotCandidatePrecedence:
    """Resolver/snapshot precedence for the resolver-registered case.

    A persisted snapshot is a candidate, not an authority. The resolver is
    authoritative and always called; the snapshot is a corroborating
    candidate -- consistent with it wins silently, but a namespace-affecting
    disagreement fails the turn instead of the snapshot winning. See
    ``tests/core/test_execution_scope.py::TestSnapshotCandidateAuthority``
    for the full precedence matrix (defer/None/ExecutionScope resolver
    values x snapshot present/absent/stale-version/corrupt).
    """

    def test_resolver_is_authoritative_and_always_called(self):
        resolver_calls = []

        def resolver(task_id):
            resolver_calls.append(task_id)
            return SCOPE

        register_scope_resolver(resolver)
        set_execution_scope_snapshot_loader(
            lambda task_id: SCOPE if task_id == "42" else None
        )

        # A snapshot equal to the resolver's answer corroborates silently.
        assert resolve_execution_scope(42) == SCOPE
        assert resolver_calls == ["42"]
        # Tasks without a snapshot still resolve through the resolver.
        assert (
            resolve_execution_scope(43).sandbox_key_suffix == SCOPE.sandbox_key_suffix
        )
        assert resolver_calls == ["42", "43"]

    def test_namespace_mismatch_between_resolver_and_snapshot_fails_the_turn(self):
        register_scope_resolver(
            lambda task_id: ExecutionScope(sandbox_key_suffix="from-resolver"),
        )
        set_execution_scope_snapshot_loader(lambda task_id: SCOPE)

        with pytest.raises(ExecutionScopeAuthorityError) as exc_info:
            resolve_execution_scope(42)
        assert exc_info.value.resolver_scope.sandbox_key_suffix == "from-resolver"
        assert exc_info.value.snapshot_scope == SCOPE
        assert "sandbox_key_suffix" in exc_info.value.mismatched_fields

    def test_caller_supplied_agent_config_skips_registered_loader(self):
        """A caller that already read the task row hands over the raw
        ``agent_config``; the snapshot inside it is decoded here rather than by
        the caller, so a malformed one is tolerated or fatal per branch."""
        loader_calls = []
        set_execution_scope_snapshot_loader(
            lambda task_id: loader_calls.append(task_id)
        )

        resolved = resolve_execution_scope(
            42,
            persisted_agent_config={EXECUTION_SCOPE_AGENT_CONFIG_KEY: SCOPE.to_dict()},
        )
        assert resolved == SCOPE
        assert loader_calls == []

    def test_caller_supplied_missing_agent_config_leaves_resolver_authoritative(self):
        loader_calls = []
        resolver_calls = []
        resolved = ExecutionScope(sandbox_key_suffix="from-resolver")
        set_execution_scope_snapshot_loader(
            lambda task_id: loader_calls.append(task_id)
        )
        register_scope_resolver(
            lambda task_id: resolver_calls.append(task_id) or resolved,
        )

        # Explicit ``None`` means "this task carries no agent_config", which
        # leaves the resolver's answer standing rather than forcing unscoped.
        assert resolve_execution_scope(42, persisted_agent_config=None) == resolved
        assert loader_calls == []
        assert resolver_calls == ["42"]

    def test_loader_exception_fails_the_turn_with_no_resolver_registered(self):
        def loader(task_id):
            raise RuntimeError("db down")

        set_execution_scope_snapshot_loader(loader)
        with pytest.raises(RuntimeError, match="db down"):
            resolve_execution_scope(42)

    def test_loader_exception_does_not_veto_resolver_authority(self):
        """A broken snapshot candidate must not veto an already-given
        authoritative resolver answer."""
        resolved = ExecutionScope(sandbox_key_suffix="from-resolver")

        def loader(task_id):
            raise RuntimeError("db down")

        set_execution_scope_snapshot_loader(loader)
        register_scope_resolver(
            lambda task_id: resolved,
        )

        assert resolve_execution_scope(42) == resolved

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
