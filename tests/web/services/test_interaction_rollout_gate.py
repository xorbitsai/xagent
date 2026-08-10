"""Publication gate decision logic: guard order is the contract."""

from __future__ import annotations

import ast
import inspect as pyinspect

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import xagent.web.services.interaction_rollout as ir
from tests.web.services.task_interaction_schema_shared import (
    tables_excluding_interaction_requests,
)
from xagent.web.models.database import Base
from xagent.web.services.ops_signals import (
    INTERACTION_ROLLOUT_SCHEMA_ABSENT,
    INTERACTION_ROLLOUT_UNKNOWN_TASK_SOURCE,
    active_degradations,
    clear_degradation,
)


class _FakeTask:
    """Duck-typed stand-in: the gate only ever reads .source/.channel_id/.id."""

    def __init__(self, source, channel_id=None, id=1):  # noqa: A002 - matches Task.id
        self.source = source
        self.channel_id = channel_id
        self.id = id


class _PoisonSession:
    """Raises on any DB access -- proves the gate never reached the query step."""

    def connection(self):
        raise AssertionError("gate must not touch the database at this step")

    def execute(self, *args, **kwargs):
        raise AssertionError("gate must not touch the database at this step")


@pytest.fixture(autouse=True)
def _reset_state(monkeypatch):
    monkeypatch.setattr(ir, "_policy", None)
    ir._counters.clear()
    clear_degradation(INTERACTION_ROLLOUT_SCHEMA_ABSENT)
    clear_degradation(INTERACTION_ROLLOUT_UNKNOWN_TASK_SOURCE)
    yield
    monkeypatch.setattr(ir, "_policy", None)
    ir._counters.clear()
    clear_degradation(INTERACTION_ROLLOUT_SCHEMA_ABSENT)
    clear_degradation(INTERACTION_ROLLOUT_UNKNOWN_TASK_SOURCE)


def _set_policy(monkeypatch, *, mode="native", native_sources=frozenset()):
    monkeypatch.setattr(
        ir,
        "_policy",
        ir.InteractionRolloutPolicy(
            mode=mode,
            native_sources=frozenset(native_sources),
            native_protocol_version=1,
        ),
    )


@pytest.fixture()
def sqlite_session_with_table(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'with_table.db'}")
    Base.metadata.create_all(bind=engine)
    session = sessionmaker(bind=engine)()
    yield session
    session.close()
    engine.dispose()


@pytest.fixture()
def sqlite_session_without_table(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'without_table.db'}")
    Base.metadata.create_all(
        bind=engine, tables=tables_excluding_interaction_requests()
    )
    session = sessionmaker(bind=engine)()
    yield session
    session.close()
    engine.dispose()


@pytest.mark.parametrize("non_native_mode", ["legacy", "read"])
def test_tg1_tg2_non_native_modes_block_with_zero_queries(monkeypatch, non_native_mode):
    _set_policy(monkeypatch, mode=non_native_mode)
    decision = ir.evaluate_native_publication(_PoisonSession(), _FakeTask("sdk"))
    assert decision is ir.NativePublicationDecision.BLOCKED_MODE


def test_tg3_native_mode_unknown_source_blocks_zero_queries_registers_degradation(
    monkeypatch,
):
    _set_policy(monkeypatch, mode="native", native_sources={"sdk"})
    decision = ir.evaluate_native_publication(_PoisonSession(), _FakeTask("bogus"))
    assert decision is ir.NativePublicationDecision.BLOCKED_UNKNOWN_SOURCE
    assert INTERACTION_ROLLOUT_UNKNOWN_TASK_SOURCE in active_degradations()


def test_tg4_native_mode_disallowed_known_source_blocks_zero_queries_no_unknown_degradation(
    monkeypatch,
):
    _set_policy(monkeypatch, mode="native", native_sources={"a2a"})
    decision = ir.evaluate_native_publication(_PoisonSession(), _FakeTask("sdk"))
    assert decision is ir.NativePublicationDecision.BLOCKED_SOURCE
    assert INTERACTION_ROLLOUT_UNKNOWN_TASK_SOURCE not in active_degradations()


def test_tg5_native_mode_allowed_source_table_absent_blocks_with_degradation(
    monkeypatch, sqlite_session_without_table
):
    _set_policy(monkeypatch, mode="native", native_sources={"sdk"})
    decision = ir.evaluate_native_publication(
        sqlite_session_without_table, _FakeTask("sdk")
    )
    assert decision is ir.NativePublicationDecision.BLOCKED_SCHEMA_ABSENT
    assert INTERACTION_ROLLOUT_SCHEMA_ABSENT in active_degradations()


def test_tg6_native_mode_allowed_source_table_present_is_allowed_no_origin_field(
    monkeypatch, sqlite_session_with_table
):
    _set_policy(monkeypatch, mode="native", native_sources={"sdk"})
    decision = ir.evaluate_native_publication(
        sqlite_session_with_table, _FakeTask("sdk")
    )
    assert decision is ir.NativePublicationDecision.ALLOWED
    assert decision.protocol_version == 1
    assert not hasattr(decision, "origin")


def test_tg7_none_source_with_no_channel_allowed_as_internal(
    monkeypatch, sqlite_session_with_table
):
    _set_policy(monkeypatch, mode="native", native_sources={"internal"})
    decision = ir.evaluate_native_publication(
        sqlite_session_with_table, _FakeTask(None, channel_id=None)
    )
    assert decision is ir.NativePublicationDecision.ALLOWED


def test_tg8_policy_singleton_identity_stable_after_env_mutated(monkeypatch):
    import xagent.config as config

    monkeypatch.delenv(config.INTERACTION_NATIVE_SOURCES, raising=False)
    monkeypatch.setenv(config.INTERACTION_PROTOCOL_MODE, "legacy")
    ir.validate_interaction_rollout_at_startup()
    policy_before = ir.get_interaction_rollout_policy()

    monkeypatch.setenv(config.INTERACTION_PROTOCOL_MODE, "native")
    monkeypatch.setenv(config.INTERACTION_NATIVE_SOURCES, "sdk")

    policy_after = ir.get_interaction_rollout_policy()
    assert policy_after is policy_before
    assert policy_after.mode == "legacy"

    decision = ir.evaluate_native_publication(_PoisonSession(), _FakeTask("sdk"))
    assert decision is ir.NativePublicationDecision.BLOCKED_MODE


def test_tg10_counters_increment_only_the_matching_outcome(
    monkeypatch, sqlite_session_with_table, sqlite_session_without_table
):
    _set_policy(monkeypatch, mode="legacy")
    ir.evaluate_native_publication(_PoisonSession(), _FakeTask("sdk"))
    assert ir.counters_snapshot() == {ir.COUNTER_ROLLOUT_DECISION_BLOCKED_MODE: 1}
    ir._counters.clear()

    _set_policy(monkeypatch, mode="native", native_sources={"sdk"})
    ir.evaluate_native_publication(_PoisonSession(), _FakeTask("bogus"))
    assert ir.counters_snapshot() == {
        ir.COUNTER_ROLLOUT_DECISION_BLOCKED_UNKNOWN_SOURCE: 1
    }
    ir._counters.clear()

    ir.evaluate_native_publication(_PoisonSession(), _FakeTask("a2a"))
    assert ir.counters_snapshot() == {ir.COUNTER_ROLLOUT_DECISION_BLOCKED_SOURCE: 1}
    ir._counters.clear()

    ir.evaluate_native_publication(sqlite_session_without_table, _FakeTask("sdk"))
    assert ir.counters_snapshot() == {
        ir.COUNTER_ROLLOUT_DECISION_BLOCKED_SCHEMA_ABSENT: 1
    }
    ir._counters.clear()

    ir.evaluate_native_publication(sqlite_session_with_table, _FakeTask("sdk"))
    assert ir.counters_snapshot() == {ir.COUNTER_ROLLOUT_DECISION_ALLOWED: 1}


def test_tg11_gate_function_body_has_no_try_except():
    source = pyinspect.getsource(ir.evaluate_native_publication)
    tree = ast.parse(source)
    func = tree.body[0]
    assert isinstance(func, ast.FunctionDef)
    for node in ast.walk(func):
        assert not isinstance(node, ast.Try), (
            "gate must not silently degrade via try/except"
        )
