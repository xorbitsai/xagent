"""The backend, the agent worker and the Celery worker refuse a mismatched KB engine."""

from __future__ import annotations

import asyncio
import logging
import re
import subprocess
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import lancedb
import pytest
from cryptography.fernet import Fernet

from xagent.core.tools.core.RAG_tools.core.exceptions import ConfigurationError
from xagent.core.tools.core.RAG_tools.storage.vector_backend import KB_ENGINE_RECORD


class _Started(Exception):
    pass


@pytest.fixture(autouse=True)
def milvus_record(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("LANCEDB_DIR", str(tmp_path))
    monkeypatch.setenv("XAGENT_VECTOR_BACKEND", "lancedb")
    (tmp_path / KB_ENGINE_RECORD).write_text("milvus\n")


@pytest.mark.asyncio
async def test_backend_startup_refuses_before_touching_the_database(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xagent.web import app

    initialize = AsyncMock()
    monkeypatch.setattr(app, "_initialize_database_and_admit_runtime", initialize)

    with pytest.raises(ConfigurationError, match="KB engine is milvus"):
        await app.startup_event()
    initialize.assert_not_called()


@pytest.mark.asyncio
async def test_agent_worker_refuses_before_accepting_tasks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xagent.web import worker

    monkeypatch.setenv("XAGENT_SHARED_TASK_EXECUTION_ENABLED", "true")
    monkeypatch.setenv("XAGENT_TASK_EXECUTION_ROLE", "worker")
    monkeypatch.setenv("XAGENT_REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setenv("ENCRYPTION_KEY", Fernet.generate_key().decode())
    for name in (
        "configure_db",
        "validate_worker_schema",
        "validate_interaction_rollout_at_startup",
        "register_local_browser_runtime",
    ):
        monkeypatch.setattr(worker, name, Mock())

    with pytest.raises(ConfigurationError, match="KB engine is milvus"):
        await worker.run_worker(stop=asyncio.Event())
    worker.register_local_browser_runtime.assert_not_called()


def test_celery_worker_exits_at_worker_init() -> None:
    from celery.signals import worker_init

    from xagent.web.jobs import celery_app  # noqa: F401 - connects the handler

    with pytest.raises(
        SystemExit,
        match="Refusing to start the Celery worker: This deployment's KB engine is milvus",
    ):
        worker_init.send(sender=None)


@pytest.mark.parametrize(
    ("returncode", "stdout", "stderr", "message"),
    [
        (3, "disk unavailable\n", "", "disk unavailable"),
        (
            3,
            "KB engine detected\nThis deployment's KB engine is milvus\n",
            "Cannot record the KB engine: [Errno 30] Read-only file system\n",
            "This deployment's KB engine is milvus",
        ),
        (
            -11,
            "",
            "Fatal Python error\nSegmentation fault\n",
            "the engine check was killed by signal 11: Segmentation fault",
        ),
        (
            -11,
            "",
            "WARNING\tCannot record the KB engine in /d/.kb-engine: [Errno 30] "
            "Read-only file system\n",
            "the engine check was killed by signal 11: Cannot record the KB engine "
            "in /d/.kb-engine: [Errno 30] Read-only file system",
        ),
        (-11, "", "", "the engine check was killed by signal 11"),
        (1, "", "", "the engine check exited with code 1"),
        (1, "stray output\n", "", "the engine check exited with code 1"),
    ],
)
def test_celery_worker_exits_on_any_engine_check_failure(
    monkeypatch: pytest.MonkeyPatch,
    returncode: int,
    stdout: str,
    stderr: str,
    message: str,
) -> None:
    from celery.signals import worker_init

    from xagent.web.jobs import celery_app  # noqa: F401 - connects the handler

    monkeypatch.setattr(
        celery_app.subprocess,
        "run",
        lambda *_, **__: subprocess.CompletedProcess([], returncode, stdout, stderr),
    )

    with pytest.raises(
        SystemExit,
        match=f"Refusing to start the Celery worker: {re.escape(message)}$",
    ):
        worker_init.send(sender=None)


@pytest.mark.parametrize(
    ("empty_dir", "level", "message"),
    [
        (True, logging.WARNING, "LANCEDB_DIR is empty"),
        (False, logging.INFO, "Recorded KB engine lancedb in"),
    ],
)
def test_celery_worker_relays_what_a_passing_engine_check_logs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    empty_dir: bool,
    level: int,
    message: str,
) -> None:
    from celery.signals import worker_init

    from xagent.web.jobs import celery_app  # noqa: F401 - connects the handler

    if empty_dir:
        monkeypatch.setenv("LANCEDB_DIR", "")
    else:
        (tmp_path / KB_ENGINE_RECORD).unlink()
    caplog.set_level(logging.INFO)

    worker_init.send(sender=None)

    assert any(
        record.levelno == level and message in record.getMessage()
        for record in caplog.records
    )


def test_celery_worker_relays_the_engine_check_output_at_its_levels(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from celery.signals import worker_init

    from xagent.web.jobs import celery_app  # noqa: F401 - connects the handler

    script = (
        "import sys\n"
        "print('plain stdout')\n"
        "sys.stderr.write('ERROR\\tparsed error\\n')\n"
        "sys.stderr.write('raw stderr line\\n')\n"
    )
    monkeypatch.setattr(celery_app, "_LOCK_KB_ENGINE", script)
    caplog.set_level(logging.INFO)

    worker_init.send(sender=None)

    assert [(r.levelno, r.getMessage()) for r in caplog.records] == [
        (logging.INFO, "KB engine check: plain stdout"),
        (logging.ERROR, "KB engine check: parsed error"),
        (logging.WARNING, "KB engine check: raw stderr line"),
    ]


def test_celery_worker_relays_the_output_of_a_refusing_engine_check(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from celery.signals import worker_init

    from xagent.web.jobs import celery_app  # noqa: F401 - connects the handler

    script = (
        "import sys\n"
        "sys.stderr.write('WARNING\\tCannot record the KB engine\\n')\n"
        "print('refused')\n"
        "raise SystemExit(3)\n"
    )
    monkeypatch.setattr(celery_app, "_LOCK_KB_ENGINE", script)
    caplog.set_level(logging.INFO)

    with pytest.raises(SystemExit, match="Celery worker: refused$"):
        worker_init.send(sender=None)

    assert (logging.WARNING, "KB engine check: Cannot record the KB engine") in [
        (r.levelno, r.getMessage()) for r in caplog.records
    ]


def test_celery_worker_refuses_even_when_relaying_the_output_raises(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from celery.signals import worker_init

    from xagent.web.jobs import celery_app  # noqa: F401 - connects the handler

    def failing_filter(record: logging.LogRecord) -> bool:
        raise RuntimeError("the log filter failed")

    caplog.set_level(logging.INFO)
    celery_app.logger.addFilter(failing_filter)
    try:
        with pytest.raises(
            SystemExit,
            match="Refusing to start the Celery worker: This deployment's KB engine is milvus",
        ):
            worker_init.send(sender=None)
    finally:
        celery_app.logger.removeFilter(failing_filter)


def test_celery_worker_exits_when_the_engine_check_cannot_start(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from celery.signals import worker_init

    from xagent.web.jobs import celery_app  # noqa: F401 - connects the handler

    monkeypatch.setattr(celery_app.sys, "executable", str(tmp_path / "no-python"))

    with pytest.raises(SystemExit, match="cannot run the engine check: .*no-python"):
        worker_init.send(sender=None)


def test_celery_worker_reads_an_engine_check_that_prints_undecodable_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from celery.signals import worker_init

    from xagent.web.jobs import celery_app  # noqa: F401 - connects the handler

    script = (
        "import sys\n"
        "sys.stdout.buffer.write(b'\\xff the path is broken\\n')\n"
        "raise SystemExit(3)\n"
    )
    monkeypatch.setattr(celery_app, "_LOCK_KB_ENGINE", script)

    with pytest.raises(
        SystemExit, match="Refusing to start the Celery worker: . the path is broken$"
    ):
        worker_init.send(sender=None)


def test_celery_worker_exits_when_the_engine_check_output_cannot_be_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from celery.signals import worker_init

    from xagent.web.jobs import celery_app  # noqa: F401 - connects the handler

    def undecodable(*_: object, **__: object) -> None:
        raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")

    monkeypatch.setattr(celery_app.subprocess, "run", undecodable)

    with pytest.raises(SystemExit, match="cannot run the engine check: .*0xff"):
        worker_init.send(sender=None)


def test_celery_parent_leaves_the_engine_check_to_a_child_process(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from celery.signals import worker_init

    from xagent.core.tools.core.RAG_tools.storage import vector_backend
    from xagent.web.jobs import celery_app  # noqa: F401 - connects the handler

    def in_the_parent() -> None:
        # Celery swallows an Exception raised by a signal handler, so exit instead.
        raise SystemExit("the engine check opened LanceDB in the Celery parent")

    monkeypatch.setattr(vector_backend, "lock_deployment_kb_engine", in_the_parent)
    (tmp_path / KB_ENGINE_RECORD).write_text("lancedb\n")

    worker_init.send(sender=None)


@pytest.fixture
def add_on(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """The Milvus add-on applied to a deployment that has no engine record yet."""
    (tmp_path / KB_ENGINE_RECORD).unlink()
    monkeypatch.setenv("XAGENT_VECTOR_BACKEND", "milvus")
    return tmp_path


def _seed_lancedb(directory: Path) -> None:
    lancedb.connect(str(directory)).create_table("documents", [{"id": "x"}])


@pytest.mark.asyncio
async def test_backend_starts_with_the_add_on_on_a_new_deployment(
    monkeypatch: pytest.MonkeyPatch, add_on: Path
) -> None:
    from xagent.web import app

    initialize = AsyncMock(side_effect=_Started)
    monkeypatch.setattr(app, "_initialize_database_and_admit_runtime", initialize)

    with pytest.raises(_Started):
        await app.startup_event()
    initialize.assert_awaited_once()
    assert (add_on / KB_ENGINE_RECORD).read_text() == "milvus\n"


@pytest.mark.asyncio
async def test_backend_refuses_the_add_on_over_existing_lancedb_data(
    monkeypatch: pytest.MonkeyPatch, add_on: Path
) -> None:
    from xagent.web import app

    _seed_lancedb(add_on)
    initialize = AsyncMock()
    monkeypatch.setattr(app, "_initialize_database_and_admit_runtime", initialize)

    with pytest.raises(ConfigurationError, match=r"is lancedb \(documents hold data\)"):
        await app.startup_event()
    initialize.assert_not_called()


def test_celery_worker_starts_with_the_add_on_on_a_new_deployment(
    add_on: Path,
) -> None:
    from celery.signals import worker_init

    from xagent.web.jobs import celery_app  # noqa: F401 - connects the handler

    worker_init.send(sender=None)
    assert (add_on / KB_ENGINE_RECORD).read_text() == "milvus\n"


def test_celery_worker_exits_with_the_add_on_over_existing_lancedb_data(
    add_on: Path,
) -> None:
    from celery.signals import worker_init

    from xagent.web.jobs import celery_app  # noqa: F401 - connects the handler

    _seed_lancedb(add_on)

    with pytest.raises(SystemExit, match=r"is lancedb \(documents hold data\)"):
        worker_init.send(sender=None)
