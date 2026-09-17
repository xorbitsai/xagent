"""Vocabulary pairings and the synthetic "channel" gating key.

Two related checks are intentionally not implemented here: verifying how
a sibling staging module binds against this vocabulary, and its
behavior when gated on the synthetic "channel" key. That module does not
exist yet on this branch, so there is nothing to test against --
whichever change introduces it is responsible for adding that coverage
alongside its own guards.
"""

from __future__ import annotations

import ast
import inspect

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import xagent.web.services.interaction_rollout as ir
from xagent.web.models.database import Base
from xagent.web.models.task_interaction import (
    INTERACTION_ORIGIN_VOCABULARY,
    TaskInteractionRequest,
)


class _FakeTask:
    def __init__(self, source, channel_id=None):
        self.source = source
        self.channel_id = channel_id


def test_tv1_gating_sources_superset_with_channel_as_the_only_addition():
    assert ir.INTERACTION_GATING_SOURCES >= INTERACTION_ORIGIN_VOCABULARY
    assert ir.INTERACTION_GATING_SOURCES - INTERACTION_ORIGIN_VOCABULARY == {"channel"}


def test_tv2a_channel_is_not_in_origin_vocabulary_or_the_origin_check():
    assert "channel" not in INTERACTION_ORIGIN_VOCABULARY

    check = next(
        c
        for c in TaskInteractionRequest.__table__.constraints
        if getattr(c, "name", None) == "ck_task_interaction_requests_origin"
    )
    import re

    values = set(re.findall(r"'([^']+)'", check.sqltext.text))
    assert "channel" not in values


@pytest.mark.parametrize(
    "source,channel_id,expected",
    [
        (None, 7, "channel"),
        (None, None, "internal"),
        ("shared_link", None, "shared_link"),
    ],
)
def test_tv3_tv5_tv6_gating_key_direct_lookups(source, channel_id, expected):
    assert ir.gating_key(_FakeTask(source, channel_id=channel_id)) == expected


def test_tv4_widget_source_with_channel_id_keeps_widget_key_not_channel():
    """An explicit non-internal origin must never be swallowed into the
    synthetic 'channel' bucket just because channel_id happens to be set --
    a widget task can carry a non-null channel_id too. Kept as its own
    test (not folded into the parametrized lookup table above): this is
    the core regression this rollout's design exists to guard, not a
    routine lookup."""
    assert ir.gating_key(_FakeTask("widget", channel_id=7)) == "widget"


@pytest.fixture()
def sqlite_session_with_table(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'with_table.db'}")
    Base.metadata.create_all(bind=engine)
    session = sessionmaker(bind=engine)()
    yield session
    session.close()
    engine.dispose()


@pytest.fixture(autouse=True)
def _reset_state(monkeypatch):
    monkeypatch.setattr(ir, "_policy", None)
    ir._counters.clear()
    yield
    monkeypatch.setattr(ir, "_policy", None)
    ir._counters.clear()


def test_tv7_internal_only_allowlist_does_not_let_channel_traffic_through(
    monkeypatch, sqlite_session_with_table
):
    """The core regression this rollout exists to prevent: allowing
    "internal" must not silently open the door for IM-channel traffic --
    that traffic must be gated by its own "channel" entry."""
    monkeypatch.setattr(
        ir,
        "_policy",
        ir.InteractionRolloutPolicy(
            mode="native",
            native_sources=frozenset({"internal"}),
            native_protocol_version=1,
        ),
    )
    decision = ir.evaluate_native_publication(
        sqlite_session_with_table, _FakeTask(None, channel_id=42)
    )
    assert decision is ir.NativePublicationDecision.BLOCKED_SOURCE


def test_tv8_channel_gated_decision_has_internal_origin_not_leaked_into_audit_column():
    """The gating key ("channel") must never leak out as if it were the
    origin value that would be written to the origin column -- the
    normalized origin underneath a channel-gated task is still
    "internal"."""
    from xagent.web.models.task_interaction import normalize_interaction_origin

    task = _FakeTask(None, channel_id=42)
    assert ir.gating_key(task) == "channel"
    assert normalize_interaction_origin(task.source) == "internal"


def test_tv9_gating_sources_definition_derives_from_origin_vocabulary_not_literals():
    """AST assertion: the right-hand operand of the INTERACTION_GATING_SOURCES
    assignment must name INTERACTION_ORIGIN_VOCABULARY, not
    TASK_SOURCE_LITERALS -- deriving from the literals constant would drop
    "internal" (absent from that constant today) from the gating vocabulary
    entirely.
    """
    source = inspect.getsource(ir)
    tree = ast.parse(source)

    assign = None
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "INTERACTION_GATING_SOURCES"
        ):
            assign = node
            break
    assert assign is not None, "INTERACTION_GATING_SOURCES definition not found"

    names_used = {n.id for n in ast.walk(assign.value) if isinstance(n, ast.Name)}
    assert "INTERACTION_ORIGIN_VOCABULARY" in names_used, (
        "retired sources are legitimate but dead entries -- they stay in the "
        "gating vocabulary even after they stop producing new rows"
    )
    assert "TASK_SOURCE_LITERALS" not in names_used
