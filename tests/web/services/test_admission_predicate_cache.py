"""The admission predicate is built once per process and keyed on the DB clock."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlalchemy import bindparam, literal, update
from sqlalchemy.dialects import postgresql, sqlite
from sqlalchemy.orm import Session

from tests.web.services.test_task_execution_admission import engine as engine_fixture
from tests.web.services.test_task_execution_admission import enqueue
from tests.web.services.test_task_execution_admission import host as host_fixture
from xagent.web.models.task_admission_pacing import TaskAdmissionPacing
from xagent.web.services import task_admission_pacing as pacing
from xagent.web.services import task_command_transport as transport
from xagent.web.services import task_execution_admission as admission
from xagent.web.services.task_dispatch_cap import DispatchCapSnapshot, DispatchPolicy

host = host_fixture
engine = engine_fixture

DIALECTS = {"sqlite": sqlite.dialect, "postgresql": postgresql.dialect}
REAL_CLOCK_TEXT = {"sqlite": "julianday", "postgresql": "clock_timestamp()"}


@pytest.fixture(autouse=True)
def fresh_cache():
    admission.reset_admission_eligible_cache()
    yield
    admission.reset_admission_eligible_cache()


def clock_text(dialect_name: str) -> str:
    compiled = admission.admission_eligible().compile(
        dialect=DIALECTS[dialect_name](), compile_kwargs={"literal_binds": True}
    )
    return str(compiled)


def test_admission_predicate_is_built_once_and_shared():
    first = admission.admission_eligible()
    assert admission.admission_eligible() is first


def test_reset_makes_the_next_call_build_a_new_predicate():
    first = admission.admission_eligible()
    admission.reset_admission_eligible_cache()
    assert admission.admission_eligible() is not first


@pytest.mark.parametrize("dialect_name", sorted(DIALECTS))
def test_admission_predicate_rebuilds_when_clock_swapped(dialect_name, monkeypatch):
    real_clock = REAL_CLOCK_TEXT[dialect_name]
    assert real_clock in clock_text(dialect_name)

    monkeypatch.setattr(pacing, "database_time", lambda: literal(100.5))
    fixed_text = clock_text(dialect_name)
    assert "100.5" in fixed_text
    assert real_clock not in fixed_text

    now = [200.5]
    monkeypatch.setattr(
        pacing,
        "database_time",
        lambda: bindparam("fake_db_now", callable_=lambda: now[0]),
    )
    advancing_text = clock_text(dialect_name)
    assert "200.5" in advancing_text
    assert "100.5" not in advancing_text
    now[0] = 300.5
    assert "300.5" in clock_text(dialect_name)

    monkeypatch.undo()
    assert real_clock in clock_text(dialect_name)
    assert "200.5" not in clock_text(dialect_name)


POLICY = DispatchPolicy(limit=4, interactive_reserve=1)
CAPS = {
    "none": None,
    "open": DispatchCapSnapshot(policy=POLICY, active_ids=frozenset()),
    "batch_line": DispatchCapSnapshot(policy=POLICY, active_ids=frozenset({1, 2, 3})),
    "closed": DispatchCapSnapshot(policy=POLICY, active_ids=frozenset({1, 2, 3, 4})),
}
SCAN_TIME = datetime(2026, 1, 1, tzinfo=timezone.utc)


def compiled_scan(dialect_name: str, *, shared: bool, cap):
    query = transport._dispatcher_candidate_query(
        Session(),
        runner_id="runner",
        command_db_id=None,
        shared=shared,
        cap=cap,
        now=SCAN_TIME,
    )
    compiled = query.statement.compile(dialect=DIALECTS[dialect_name]())
    return str(compiled), dict(compiled.params)


@pytest.mark.parametrize("cap_name", sorted(CAPS))
@pytest.mark.parametrize("shared", [True, False], ids=["shared", "non_shared"])
@pytest.mark.parametrize("dialect_name", sorted(DIALECTS))
def test_cached_admission_predicate_compiles_identically(
    dialect_name, shared, cap_name, monkeypatch
):
    monkeypatch.setenv("XAGENT_SHARED_TASK_EXECUTION_ENABLED", str(shared).lower())
    cap = CAPS[cap_name]

    compiled_scan(dialect_name, shared=shared, cap=cap)
    cached = compiled_scan(dialect_name, shared=shared, cap=cap)

    monkeypatch.setattr(
        admission, "admission_eligible", admission._build_admission_eligible
    )
    rebuilt_first = compiled_scan(dialect_name, shared=shared, cap=cap)
    rebuilt_second = compiled_scan(dialect_name, shared=shared, cap=cap)

    assert cached == rebuilt_first == rebuilt_second


async def test_fake_database_clock_advances_through_cached_predicate(host, monkeypatch):
    now = [100.0]
    monkeypatch.setattr(
        pacing,
        "database_time",
        lambda: bindparam("fake_db_now", callable_=lambda: now[0]),
    )
    admission.set_task_admission_hook(
        lambda db, command: admission.AdmissionPolicy(
            "paced",
            10,
            20,
            pacing=pacing.StartupPacing(interval_seconds=1, burst=1),
        )
    )
    command = enqueue(host)
    with host.sessions() as db:
        db.execute(
            update(TaskAdmissionPacing)
            .where(TaskAdmissionPacing.bucket_key == "paced")
            .values(next_start_at=101.0)
        )
        db.commit()

    predicate = admission.admission_eligible()
    for instant, waits in [
        (100.0, True),
        (100.999, True),
        (101.0, False),
        (1000.0, False),
    ]:
        now[0] = instant
        with host.sessions() as db:
            assert admission.waiting_for_capacity(db, command.command_id) is waits
        # The clock advanced through the same shared expression, not a rebuild.
        assert admission.admission_eligible() is predicate
