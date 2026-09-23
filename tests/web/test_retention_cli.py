"""``xagent retention preview`` -- the read-only diagnostic (#2562).

The property that matters most here is negative: the command must not write
anything. It is the one piece of this PR an operator will point at a
production database, and it exists to inform a policy decision, not to act on
one.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session

from xagent.web import retention_cli
from xagent.web.models import database
from xagent.web.models.database import Base
from xagent.web.models.task import Task, TaskStatus, TraceEvent
from xagent.web.models.user import User
from xagent.web.services.task_retention import retention_cutoff

NOW = datetime.now(timezone.utc)


@pytest.fixture(autouse=True)
def restore_process_database_binding():
    """Undo the process-global bind these tests deliberately let happen.

    ``configure_db`` assigns module globals (``_SessionLocal``, ``_engine``),
    and this file exercises the real binding rather than stubbing it, because
    "the preview opens the database read-only" is one of the properties under
    test. Left unrestored, the last test's ``tmp_path`` SQLite file stays
    installed as the process-wide session factory, and any later test that
    reaches for it silently gets this database instead of its own -- a hazard
    already recorded in
    tests/core/tools/adapters/sandboxed_tool/test_sandbox_output_registration.py.
    """
    previous_sessions = database._SessionLocal
    previous_engine = database._engine
    try:
        yield
    finally:
        engine = database._engine
        if engine is not None and engine is not previous_engine:
            engine.dispose()
        database._SessionLocal = previous_sessions
        database._engine = previous_engine


@pytest.fixture
def database_url(tmp_path):
    """A SQLite database holding four terminal tasks and one live one."""
    url = f"sqlite:///{tmp_path / 'retention.db'}"
    engine = sa.create_engine(url)
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        user = User(username="owner", password_hash="unused")
        db.add(user)
        db.flush()
        # Distinct trace cardinalities per age, none of them equal to a task
        # count, so the trace column cannot be satisfied by counting tasks.
        traces_by_age = {400: 5, 200: 3, 100: 2, 10: 7}
        for index, age in enumerate((400, 200, 100, 10)):
            task = Task(
                user_id=user.id,
                title=f"terminal-{index}",
                status=TaskStatus.COMPLETED,
                last_activity_at=NOW - timedelta(days=age),
            )
            db.add(task)
            db.flush()
            for n in range(traces_by_age[age]):
                db.add(
                    TraceEvent(
                        task_id=task.id,
                        event_id=f"{task.id}-{n}",
                        event_type="step",
                        timestamp=NOW - timedelta(days=age),
                        data={},
                    )
                )
        # Ancient, but still running: must never be counted, and its traces
        # must not leak into the trace column either.
        live = Task(
            user_id=user.id,
            title="running",
            status=TaskStatus.RUNNING,
            last_activity_at=NOW - timedelta(days=4000),
        )
        db.add(live)
        db.flush()
        for n in range(11):
            db.add(
                TraceEvent(
                    task_id=live.id,
                    event_id=f"live-{n}",
                    event_type="step",
                    timestamp=NOW - timedelta(days=4000),
                    data={},
                )
            )
        db.commit()
    engine.dispose()
    return url


def _run(database_url: str, *args: str) -> int:
    return retention_cli.main(["preview", "--database-url", database_url, *args])


def _row(out: str, days: int) -> list[str]:
    """The complete output row for one period, as [days, tasks, traces]."""
    for line in out.splitlines():
        fields = line.split()
        if len(fields) == 3 and fields[0] == str(days):
            return fields
    raise AssertionError(f"no output row for --days {days} in:\n{out}")


def test_preview_reports_one_row_per_requested_period(capsys, database_url):
    assert _run(database_url, "--days", "90", "--days", "365") == 0
    out = capsys.readouterr().out
    # Ages 400/200/100 are past 90 days; 10 is not. Only 400 is past 365.
    # Traces follow their tasks: 5+3+2 at 90 days, 5 alone at 365.
    assert _row(out, 90) == ["90", "3", "10"]
    assert _row(out, 365) == ["365", "1", "5"]


def test_preview_counts_traces_of_eligible_tasks_only(capsys, database_url):
    """The trace column must follow eligibility, not the whole table.

    The fixture holds 28 trace rows. At one day every terminal task is
    eligible (5+3+2+7 = 17) and the RUNNING task's 11 are excluded -- so a
    count that ignored the quiescence filter, returned a constant, or counted
    tasks instead of traces all produce a different number here.
    """
    assert _run(database_url, "--days", "1") == 0
    out = capsys.readouterr().out
    assert _row(out, 1) == ["1", "4", "17"]


def test_preview_reports_no_traces_when_no_task_is_eligible(capsys, database_url):
    """A period nothing has aged past must report zero, not the table total."""
    assert _run(database_url, "--days", "5000") == 0
    assert _row(capsys.readouterr().out, 5000) == ["5000", "0", "0"]


def test_preview_excludes_live_tasks_from_every_count(capsys, database_url):
    """A 4000-day-old RUNNING task is older than any period and still excluded."""
    assert _run(database_url, "--days", "1") == 0
    out = capsys.readouterr().out
    assert "Quiescent tasks" in out
    assert out.splitlines()[1].endswith(": 4")
    assert "1     4" in out


def test_preview_defaults_to_the_policy_candidates(capsys, database_url):
    assert _run(database_url) == 0
    out = capsys.readouterr().out
    for days in retention_cli.DEFAULT_PREVIEW_DAYS:
        assert f"\n{days}" in out


def test_preview_deduplicates_and_orders_requested_periods(capsys, database_url):
    assert _run(database_url, "--days", "365", "--days", "90", "--days", "365") == 0
    body = capsys.readouterr().out.split("trace events")[1]
    assert body.index("\n90") < body.index("\n365")
    assert body.count("\n365") == 1


def test_preview_rejects_a_negative_period(capsys, database_url):
    assert _run(database_url, "--days", "-5") == 2
    assert "must not be negative" in capsys.readouterr().err


@pytest.mark.parametrize(
    "days",
    [
        pytest.param(1_000_000_000, id="timedelta-refuses-to-construct"),
        pytest.param(999_999_999, id="subtraction-overflows"),
    ],
)
def test_preview_rejects_a_period_no_date_can_express(capsys, database_url, days):
    """An unrepresentable period is an input error, not a traceback.

    The two cases fail on different legs -- one cannot build the timedelta,
    the other builds it and then overflows the subtraction -- which is why the
    CLI asks the arithmetic instead of comparing against a constant.
    """
    assert _run(database_url, "--days", str(days)) == 2
    assert "too large to express as a date" in capsys.readouterr().err


def test_preview_rejects_an_oversized_period_before_opening_the_database(
    capsys, monkeypatch
):
    """Validation runs before any connection is made."""
    monkeypatch.setattr(
        retention_cli,
        "configure_db",
        lambda *a, **k: pytest.fail("validation must precede configure_db"),
    )
    assert retention_cli.main(["preview", "--days", "1000000000"]) == 2
    assert "too large to express as a date" in capsys.readouterr().err


def test_retention_cutoff_still_raises_for_library_callers():
    """The helper keeps raising; only the CLI translates.

    #2563 consumes ``retention_cutoff`` directly and should see the failure
    rather than a silently clamped period.
    """
    with pytest.raises(OverflowError):
        retention_cutoff(now=NOW, days=1_000_000_000)


def test_preview_writes_nothing(capsys, database_url):
    """The whole table, byte for byte, before and after."""
    engine = sa.create_engine(database_url)
    with Session(engine) as db:
        before = db.execute(sa.text("SELECT * FROM tasks ORDER BY id")).all()
    engine.dispose()

    assert _run(database_url, "--days", "1") == 0
    capsys.readouterr()

    engine = sa.create_engine(database_url)
    with Session(engine) as db:
        after = db.execute(sa.text("SELECT * FROM tasks ORDER BY id")).all()
    engine.dispose()
    assert after == before


def test_preview_says_it_deleted_nothing(capsys, database_url):
    """An operator reading the output must not mistake it for a purge report."""
    assert _run(database_url, "--days", "1") == 0
    assert "deletes nothing" in capsys.readouterr().out


def test_retention_is_dispatched_from_the_module_entry_point(monkeypatch):
    """``xagent retention ...`` must reach the CLI, like ``xagent migrate``."""
    from xagent.web import __main__ as entry

    seen: dict[str, list[str]] = {}

    def fake_main(argv):
        seen["argv"] = argv
        return 0

    monkeypatch.setattr(retention_cli, "main", fake_main)
    monkeypatch.setattr(
        entry.sys, "argv", ["xagent", "retention", "preview", "--days", "7"]
    )
    with pytest.raises(SystemExit) as exit_info:
        entry.main()
    assert exit_info.value.code == 0
    assert seen["argv"] == ["preview", "--days", "7"]
