"""Observability surfaces for the interaction rollout gate: /health,
/ready, and the admin diagnostics endpoint.

/health and /ready are exercised by calling the route functions directly
(``asyncio.run`` / ``await``), the same pattern
``tests/web/test_health_degradations.py`` uses -- it avoids booting the
full ASGI lifespan (and the real startup event) just to reach two plain
async functions. The admin diagnostics endpoint is exercised the same way
``tests/web/api/test_admin_users_hidden.py`` calls its own admin routes:
directly, with a hand-built ``User``/``Session`` pair, rather than through
TestClient + JWT headers.
"""

from __future__ import annotations

from importlib import import_module

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import xagent.web.models.database as database_module
import xagent.web.services.interaction_rollout as ir
from tests.web.services.task_interaction_schema_shared import (
    tables_excluding_interaction_requests,
)
from xagent.web.api.admin_interaction_rollout import (
    get_interaction_rollout_diagnostics,
)
from xagent.web.models.database import Base
from xagent.web.models.user import User
from xagent.web.services.ops_signals import (
    INTERACTION_ROLLOUT_SCHEMA_ABSENT,
    INTERACTION_ROLLOUT_UNKNOWN_TASK_SOURCE,
    active_degradations,
    clear_degradation,
)

app_module = import_module("xagent.web.app")


@pytest.fixture(autouse=True)
def _reset_state(monkeypatch):
    monkeypatch.setattr(ir, "_policy", None)
    monkeypatch.setattr(ir, "_native_schema_ready", False)
    ir._counters.clear()
    clear_degradation(INTERACTION_ROLLOUT_SCHEMA_ABSENT)
    clear_degradation(INTERACTION_ROLLOUT_UNKNOWN_TASK_SOURCE)
    yield
    monkeypatch.setattr(ir, "_policy", None)
    monkeypatch.setattr(ir, "_native_schema_ready", False)
    ir._counters.clear()
    clear_degradation(INTERACTION_ROLLOUT_SCHEMA_ABSENT)
    clear_degradation(INTERACTION_ROLLOUT_UNKNOWN_TASK_SOURCE)


def _set_policy(monkeypatch, *, mode="legacy", native_sources=frozenset()):
    monkeypatch.setattr(
        ir,
        "_policy",
        ir.InteractionRolloutPolicy(
            mode=mode,
            native_sources=frozenset(native_sources),
            native_protocol_version=1,
        ),
    )


class _PoisonSessionLocal:
    def __call__(self):
        raise AssertionError("must not open a DB session at this step")


# ---------------------------------------------------------------------------
# /health surfaces the two new signal names, never their detail
# ---------------------------------------------------------------------------


async def test_to1_health_surfaces_new_signal_names_but_not_detail():
    import xagent.web.services.ops_signals as ops_signals

    ops_signals.register_degradation(
        INTERACTION_ROLLOUT_SCHEMA_ABSENT, "task_interaction_requests table absent"
    )
    ops_signals.register_degradation(
        INTERACTION_ROLLOUT_UNKNOWN_TASK_SOURCE, "task 42: unrecognized source"
    )

    payload = await app_module.health_check()

    assert payload["status"] == "ok"
    assert INTERACTION_ROLLOUT_SCHEMA_ABSENT in payload["degradations"]
    assert INTERACTION_ROLLOUT_UNKNOWN_TASK_SOURCE in payload["degradations"]
    assert "task 42" not in str(payload)
    assert "table absent" not in str(payload)


# ---------------------------------------------------------------------------
# /ready: default-mode passthrough, the native-mode schema check, and its
# one-way latch in both directions
# ---------------------------------------------------------------------------


async def test_to2_default_mode_ready_behavior_is_unchanged_zero_queries(
    monkeypatch,
):
    _set_policy(monkeypatch, mode="legacy")
    monkeypatch.setattr(database_module, "get_session_local", _PoisonSessionLocal())
    app_module.app.state.file_storage_startup_sync_task = None
    app_module.app.state.file_storage_startup_sync_error = None

    response = await app_module.readiness_check()

    assert response.status_code == 200


async def test_to3_native_mode_table_absent_returns_503_without_table_name(
    monkeypatch, tmp_path
):
    _set_policy(monkeypatch, mode="native", native_sources={"sdk"})
    app_module.app.state.file_storage_startup_sync_task = None
    app_module.app.state.file_storage_startup_sync_error = None

    engine = create_engine(f"sqlite:///{tmp_path / 'no_table.db'}")
    Base.metadata.create_all(
        bind=engine, tables=tables_excluding_interaction_requests()
    )
    session_local = sessionmaker(bind=engine)
    monkeypatch.setattr(database_module, "get_session_local", lambda: session_local)

    response = await app_module.readiness_check()

    assert response.status_code == 503
    body = response.body.decode()
    assert "Interaction rollout schema not ready" in body
    assert "task_interaction_requests" not in body


async def test_to4_native_mode_table_present_returns_200_and_clears_degradation(
    monkeypatch, tmp_path
):
    _set_policy(monkeypatch, mode="native", native_sources={"sdk"})
    app_module.app.state.file_storage_startup_sync_task = None
    app_module.app.state.file_storage_startup_sync_error = None
    import xagent.web.services.ops_signals as ops_signals

    ops_signals.register_degradation(
        INTERACTION_ROLLOUT_SCHEMA_ABSENT, "task_interaction_requests table absent"
    )

    engine = create_engine(f"sqlite:///{tmp_path / 'with_table.db'}")
    Base.metadata.create_all(bind=engine)
    session_local = sessionmaker(bind=engine)
    monkeypatch.setattr(database_module, "get_session_local", lambda: session_local)

    response = await app_module.readiness_check()

    assert response.status_code == 200
    assert INTERACTION_ROLLOUT_SCHEMA_ABSENT not in active_degradations()


async def test_to4b_latch_positive_means_zero_queries_on_next_probe(
    monkeypatch, tmp_path
):
    _set_policy(monkeypatch, mode="native", native_sources={"sdk"})
    app_module.app.state.file_storage_startup_sync_task = None
    app_module.app.state.file_storage_startup_sync_error = None

    engine = create_engine(f"sqlite:///{tmp_path / 'with_table.db'}")
    Base.metadata.create_all(bind=engine)
    session_local = sessionmaker(bind=engine)
    monkeypatch.setattr(database_module, "get_session_local", lambda: session_local)

    first = await app_module.readiness_check()
    assert first.status_code == 200
    assert ir.is_native_schema_ready() is True

    # Second probe: swap in a poison session-factory. If the latch were not
    # honored, this would raise instead of returning 200.
    monkeypatch.setattr(database_module, "get_session_local", _PoisonSessionLocal())
    second = await app_module.readiness_check()
    assert second.status_code == 200


async def test_to4c_latch_negative_rechecks_every_probe(monkeypatch, tmp_path):
    _set_policy(monkeypatch, mode="native", native_sources={"sdk"})
    app_module.app.state.file_storage_startup_sync_task = None
    app_module.app.state.file_storage_startup_sync_error = None

    engine = create_engine(f"sqlite:///{tmp_path / 'absent.db'}")
    Base.metadata.create_all(
        bind=engine, tables=tables_excluding_interaction_requests()
    )
    session_local = sessionmaker(bind=engine)
    monkeypatch.setattr(database_module, "get_session_local", lambda: session_local)

    first = await app_module.readiness_check()
    assert first.status_code == 503
    assert ir.is_native_schema_ready() is False

    # Table shows up between probes (a migration applied while the process
    # is running) -- the negative latch must re-query and see it.
    Base.metadata.create_all(bind=engine)
    second = await app_module.readiness_check()
    assert second.status_code == 200
    assert ir.is_native_schema_ready() is True


# ---------------------------------------------------------------------------
# Admin diagnostics endpoint
#
# SQLite only, deliberately: the active-row stats query
# (SELECT count(*), min(created_at) FROM task_interaction_requests WHERE
# status = 'active' AND active_slot IS NOT NULL) uses no PostgreSQL-specific
# syntax -- no JSON operators, no ::casts, no RETURNING, nothing beyond
# ANSI count()/min()/WHERE -- so its behavior does not vary by backend and
# a PostgreSQL run would exercise nothing this SQLite run does not already
# cover.
# ---------------------------------------------------------------------------


@pytest.fixture()
def sqlite_session_with_table(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'diag_with_table.db'}")
    Base.metadata.create_all(bind=engine)
    session = sessionmaker(bind=engine)()
    yield session
    session.close()
    engine.dispose()


@pytest.fixture()
def sqlite_session_without_table(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'diag_without_table.db'}")
    Base.metadata.create_all(
        bind=engine, tables=tables_excluding_interaction_requests()
    )
    session = sessionmaker(bind=engine)()
    yield session
    session.close()
    engine.dispose()


def _admin_user() -> User:
    return User(id=1, username="admin", password_hash="x", is_admin=True)


def _regular_user() -> User:
    return User(id=2, username="regular", password_hash="x", is_admin=False)


async def test_to5_non_admin_gets_403_with_exact_detail(
    monkeypatch, sqlite_session_with_table
):
    _set_policy(monkeypatch, mode="legacy")
    with pytest.raises(HTTPException) as exc:
        await get_interaction_rollout_diagnostics(
            current_user=_regular_user(), db=sqlite_session_with_table
        )
    assert exc.value.status_code == 403
    assert exc.value.detail == "Admin access required"


async def test_to5_admin_gets_200(monkeypatch, sqlite_session_with_table):
    _set_policy(monkeypatch, mode="legacy")
    response = await get_interaction_rollout_diagnostics(
        current_user=_admin_user(), db=sqlite_session_with_table
    )
    assert response["policy"]["mode"] == "legacy"


async def test_to6_empty_table_returns_zero_active_count_and_no_oldest_age(
    monkeypatch, sqlite_session_with_table
):
    _set_policy(monkeypatch, mode="legacy")
    response = await get_interaction_rollout_diagnostics(
        current_user=_admin_user(), db=sqlite_session_with_table
    )
    assert response["active_count"] == 0
    assert response["oldest_age_seconds"] is None
    assert "schema_absent" not in response


async def test_to7_table_absent_returns_schema_absent_without_count_fields(
    monkeypatch, sqlite_session_without_table
):
    _set_policy(monkeypatch, mode="legacy")
    response = await get_interaction_rollout_diagnostics(
        current_user=_admin_user(), db=sqlite_session_without_table
    )
    assert response["schema_absent"] is True
    assert "active_count" not in response
    assert "oldest_age_seconds" not in response


async def test_to8_stats_query_exception_returns_503_not_a_fake_zero(
    monkeypatch, sqlite_session_with_table
):
    _set_policy(monkeypatch, mode="legacy")

    def _raise(*args, **kwargs):
        raise RuntimeError("simulated DB hiccup")

    monkeypatch.setattr(sqlite_session_with_table, "execute", _raise)

    with pytest.raises(HTTPException) as exc:
        await get_interaction_rollout_diagnostics(
            current_user=_admin_user(), db=sqlite_session_with_table
        )
    assert exc.value.status_code == 503


# ---------------------------------------------------------------------------
# Counter registry snapshot isolation
# ---------------------------------------------------------------------------


def test_to9_counters_snapshot_is_a_copy_not_a_live_reference():
    ir.increment_counter(ir.COUNTER_ROLLOUT_DECISION_ALLOWED)
    snapshot = ir.counters_snapshot()
    snapshot[ir.COUNTER_ROLLOUT_DECISION_ALLOWED] = 999
    snapshot["bogus"] = 1

    fresh = ir.counters_snapshot()
    assert fresh[ir.COUNTER_ROLLOUT_DECISION_ALLOWED] == 1
    assert "bogus" not in fresh


def test_to9_counters_snapshot_concurrent_increments_total_correctly():
    import threading

    def _bump():
        for _ in range(100):
            ir.increment_counter(ir.COUNTER_ROLLOUT_DECISION_ALLOWED)

    threads = [threading.Thread(target=_bump) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert ir.counters_snapshot()[ir.COUNTER_ROLLOUT_DECISION_ALLOWED] == 800


# ---------------------------------------------------------------------------
# Unconditional startup INFO log
# ---------------------------------------------------------------------------


def test_to10_startup_logs_resolved_policy_unconditionally(monkeypatch, caplog):
    import logging

    import xagent.config as config

    monkeypatch.delenv(config.INTERACTION_PROTOCOL_MODE, raising=False)
    monkeypatch.delenv(config.INTERACTION_NATIVE_SOURCES, raising=False)
    monkeypatch.setattr(ir, "_policy", None)

    with caplog.at_level(
        logging.INFO, logger="xagent.web.services.interaction_rollout"
    ):
        ir.validate_interaction_rollout_at_startup()

    assert any(
        "Interaction rollout policy configured" in r.message for r in caplog.records
    )
    assert all(
        r.levelno == logging.INFO
        for r in caplog.records
        if "Interaction rollout policy configured" in r.message
    )


# ---------------------------------------------------------------------------
# Truncated repr of the offending Task.source in degradation detail
# ---------------------------------------------------------------------------


def test_to11_unknown_source_detail_is_repr_and_truncated_to_64_chars(
    monkeypatch, sqlite_session_with_table
):
    class _FakeTask:
        def __init__(self, source):
            self.source = source
            self.channel_id = None
            self.id = 7

    _set_policy(monkeypatch, mode="native", native_sources={"sdk"})
    long_source = "x" * 500
    ir.evaluate_native_publication(sqlite_session_with_table, _FakeTask(long_source))

    detail = active_degradations()[INTERACTION_ROLLOUT_UNKNOWN_TASK_SOURCE]
    assert "..." in detail
    # The embedded repr must be capped -- nowhere near the full 500 chars.
    assert len(detail) < 200
