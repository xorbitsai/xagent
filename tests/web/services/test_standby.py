"""Standby start mode: closed hold, no writes, tolerant schema, ordered activation."""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import pytest
import sqlalchemy as sa
from alembic.script import ScriptDirectory

from xagent.db.migration import check_standby_schema, create_alembic_config
from xagent.web.models import database as database_module
from xagent.web.services.standby import (
    STANDBY_ENV,
    StandbyActivationError,
    StandbyController,
    StandbyPhase,
    standby_requested,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, False),
        ("", False),
        ("0", False),
        ("off", False),
        ("1", True),
        ("True", True),
    ],
)
def test_standby_requested_parses_boolean(
    monkeypatch: pytest.MonkeyPatch, raw: str | None, expected: bool
) -> None:
    if raw is None:
        monkeypatch.delenv(STANDBY_ENV, raising=False)
    else:
        monkeypatch.setenv(STANDBY_ENV, raw)
    assert standby_requested() is expected


def test_standby_requested_rejects_garbage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(STANDBY_ENV, "maybe")
    with pytest.raises(ValueError):
        standby_requested()


@pytest.mark.parametrize(("raw", "closed"), [("1", True), ("", False)])
def test_hold_is_closed_and_sealed_from_import_in_standby(
    raw: str, closed: bool
) -> None:
    code = (
        "from xagent.web.services.release_hold import release_hold as h;"
        "print(h.closed, h.sealed)"
    )
    env = {"PATH": "", STANDBY_ENV: raw, "PYTHONPATH": ":".join(sys.path)}
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    ).stdout.split()
    assert out == [str(closed), str(closed)]


@pytest.mark.asyncio
async def test_controller_runs_deferred_steps_in_order_once() -> None:
    controller = StandbyController()
    ran: list[str] = []

    def step(name: str):
        async def run() -> None:
            ran.append(name)

        return run

    controller.begin()
    controller.defer("a", step("a"))
    controller.defer("b", step("b"))
    assert controller.standby and controller.phase is StandbyPhase.STANDBY
    await controller.activate()
    await controller.activate()  # idempotent
    assert ran == ["a", "b"]
    assert controller.phase is StandbyPhase.ACTIVE and not controller.standby
    assert controller.activated_at is not None


@pytest.mark.asyncio
async def test_failed_activation_stays_failed_and_is_not_retried() -> None:
    controller = StandbyController()
    ran: list[str] = []

    async def ok() -> None:
        ran.append("ok")

    async def boom() -> None:
        raise RuntimeError("migration check failed")

    controller.begin()
    controller.defer("ok", ok)
    controller.defer("boom", boom)
    with pytest.raises(RuntimeError):
        await controller.activate()
    assert controller.phase is StandbyPhase.FAILED and controller.standby
    with pytest.raises(StandbyActivationError):
        await controller.activate()
    assert ran == ["ok"]


def test_defer_outside_standby_is_refused() -> None:
    controller = StandbyController()

    async def noop() -> None:
        return None

    with pytest.raises(StandbyActivationError):
        controller.defer("x", noop)


def _sqlite_with_revision(tmp_path: Path, revision: str | None) -> sa.Engine:
    engine = sa.create_engine(f"sqlite:///{tmp_path / 'db.sqlite'}")
    with engine.begin() as conn:
        conn.execute(
            sa.text("CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL)")
        )
        if revision is not None:
            conn.execute(
                sa.text("INSERT INTO alembic_version VALUES (:r)"), {"r": revision}
            )
    return engine


def _older_revision(engine: sa.Engine) -> str:
    script = ScriptDirectory.from_config(create_alembic_config(engine))
    heads = set(script.get_heads())
    older = [r.revision for r in script.walk_revisions() if r.revision not in heads]
    assert older
    return older[len(older) // 2]


def test_schema_admission_accepts_older_and_head_revisions(tmp_path: Path) -> None:
    engine = _sqlite_with_revision(tmp_path, None)
    script = ScriptDirectory.from_config(create_alembic_config(engine))
    for revision in (_older_revision(engine), script.get_current_head()):
        with engine.begin() as conn:
            conn.execute(sa.text("DELETE FROM alembic_version"))
            conn.execute(
                sa.text("INSERT INTO alembic_version VALUES (:r)"), {"r": revision}
            )
        check_standby_schema(engine)


@pytest.mark.parametrize("revision", ["from_a_newer_release", None])
def test_schema_admission_refuses_unknown_or_missing_revision(
    tmp_path: Path, revision: str | None
) -> None:
    with pytest.raises(RuntimeError):
        check_standby_schema(_sqlite_with_revision(tmp_path, revision))


def test_standby_bind_against_older_schema_issues_no_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = _sqlite_with_revision(tmp_path, None)
    older = _older_revision(engine)
    with engine.begin() as conn:
        conn.execute(sa.text("INSERT INTO alembic_version VALUES (:r)"), {"r": older})
    engine.dispose()
    url = f"sqlite:///{tmp_path / 'db.sqlite'}"
    monkeypatch.setattr(database_module, "_engine", None)
    monkeypatch.setattr(database_module, "_SessionLocal", None)

    statements: list[str] = []

    @sa.event.listens_for(sa.engine.Engine, "before_cursor_execute")
    def record(conn, cursor, statement, parameters, context, executemany):  # type: ignore[no-untyped-def]
        statements.append(statement)

    try:
        database_module.init_db_standby(url)
    finally:
        sa.event.remove(sa.engine.Engine, "before_cursor_execute", record)
        database_module.get_engine().dispose()
    writes = [
        s
        for s in statements
        if not s.lstrip().upper().startswith(("SELECT", "PRAGMA", "BEGIN", "ROLLBACK"))
    ]
    assert writes == []


@pytest.mark.asyncio
async def test_startup_in_standby_defers_writes_and_activation_runs_them_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app_module = importlib.import_module("xagent.web.app")
    hold_module = importlib.import_module("xagent.web.services.release_hold")
    hold = hold_module.release_hold
    events: list[str] = []

    monkeypatch.setenv(STANDBY_ENV, "1")
    monkeypatch.setattr(app_module, "init_db", lambda: events.append("init_db"))
    monkeypatch.setattr(
        app_module, "init_db_standby", lambda: events.append("bind_readonly")
    )
    monkeypatch.setattr(
        app_module,
        "complete_standby_db_init",
        lambda: events.append("init_db_deferred"),
    )

    async def admissions(_app) -> None:
        events.append("admissions")

    monkeypatch.setattr(app_module, "run_host_startup_admissions", admissions)
    monkeypatch.setattr(
        "xagent.web.services.trace_database.get_trace_database_runtime",
        lambda: events.append("trace_db"),
    )
    monkeypatch.setattr(
        app_module, "register_local_browser_runtime", lambda: events.append("browser")
    )
    for name in (
        "start_file_storage_startup_sync_task",
        "start_trigger_dispatcher_task",
        "start_task_lease_recovery_task",
        "start_task_auto_resume_task",
        "start_uploaded_file_recovery_task",
        "start_orphan_upload_gc_task",
        "start_retention_purge_task",
        "start_task_cleanup_retry_task",
    ):
        monkeypatch.setattr(
            app_module, name, lambda _app, name=name: events.append(name)
        )

    app_module.standby.begin()
    hold.seal("standby")
    try:
        await app_module._initialize_database_and_admit_runtime(app_module.app)
        assert events == ["bind_readonly", "browser"]
        assert hold.closed and hold.sealed
        assert app_module.standby.deferred == [
            "database and host admission",
            "open admission",
            "local runtime loops",
        ]
        await app_module.standby.activate()
        assert events[2:5] == ["init_db_deferred", "admissions", "trace_db"]
        assert not hold.closed and not hold.sealed
        assert events.index("start_task_lease_recovery_task") > events.index("trace_db")
        assert "init_db" not in events
    finally:
        app_module.standby.reset()
        hold.reopen()


class _FakePgBind:
    """Just enough of a PostgreSQL connection for ``check_task_status_enum_drift``."""

    class dialect:  # noqa: N801
        name = "postgresql"

    def __init__(self, labels: list[str]) -> None:
        self._labels = labels

    def scalars(self, _statement):  # type: ignore[no-untyped-def]
        return iter(self._labels)


class _FakeInspector:
    def has_table(self, _name: str) -> bool:
        return True


def test_enum_drift_tolerates_missing_labels_only_for_standby(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xagent.web.models import task as task_module

    monkeypatch.setattr(task_module, "sa_inspect", lambda _bind: _FakeInspector())
    expected = list(task_module.Task.__table__.c.status.type.enums)

    older = _FakePgBind(expected[:-1])
    with pytest.raises(task_module.TaskStatusEnumDriftError):
        task_module.check_task_status_enum_drift(older)
    task_module.check_task_status_enum_drift(older, allow_missing_labels=True)

    newer = _FakePgBind([*expected, "LABEL_FROM_A_NEWER_RELEASE"])
    with pytest.raises(task_module.TaskStatusEnumDriftError):
        task_module.check_task_status_enum_drift(newer, allow_missing_labels=True)
