"""The Celery worker and beat processes redact records and drop transport INFO.

Celery configures logging itself and never calls ``setup_logging``, so the app
installs the record factory and the httpx/httpcore filter from Celery's own
signals. Every test bootstraps a real Celery entry point in a child interpreter
and reads what reaches stderr: a process-global factory and a logger filter can
only be trusted from a process that did not inherit them from a previous test.
"""

from __future__ import annotations

import io
import logging
import os
import subprocess
import sys
import textwrap
from collections.abc import Iterator
from pathlib import Path

import pytest

from xagent.web.logging_config import install_log_redaction

REPO_ROOT = Path(__file__).resolve().parents[2]

_QUERY_SECRET = "SECRET-QUERY-VALUE"
_REFRESH_SECRET = "SECRET-REFRESH-VALUE"
_BEARER_SECRET = "SECRET-BEARER-VALUE"

# The same records every process must mask: a request target with two
# credential values and a header line. The logger is an application logger,
# not ``celery.*``, because task modules log through their own module logger.
_LOG_CALLS = """\
logging.getLogger("xagent.web.jobs.tasks").info(
    "callback %s",
    "https://api.example.com/cb?token=SECRET-QUERY-VALUE&refresh_token=SECRET-REFRESH-VALUE",
)
logging.getLogger("xagent.web.jobs.tasks").info(
    "Authorization: Bearer SECRET-BEARER-VALUE"
)
# A transport request line at INFO must be dropped, at WARNING kept.
logging.getLogger("httpx").info(
    "HTTP Request: GET https://api.example.com/x?token=SECRET-QUERY-VALUE"
)
logging.getLogger("httpx").warning("transport failed with status 503")
"""

# ``Worker.setup_logging`` calls ``app.log.setup``, which emits
# ``setup_logging``/``after_setup_logger``/``after_setup_task_logger`` after
# Celery replaced the root handlers. A worker that sets up its logging without
# going through ``worker_init`` first (``celery.contrib.testing.worker`` does)
# is covered by ``after_setup_logger`` alone.
_WORKER_LOG_SETUP = """\
import logging

from xagent.web.jobs.celery_app import celery_app

celery_app.log.setup(
    logging.INFO,
    None,
    redirect_stdouts=False,
    colorize=False,
    hostname="probe@worker",
)
"""

# The same worker after a ``setup_logging`` receiver took the log
# configuration over: any receiver, whatever it returns, makes Celery skip its
# own configuration and never emit ``after_setup_logger``, so only
# ``worker_init`` can cover the main process.
_HIJACKED_WORKER_LOG_SETUP = """\
import logging
import sys

from celery.signals import setup_logging, worker_init

from xagent.web.jobs.celery_app import celery_app


@setup_logging.connect
def configure_logging(**_: object) -> None:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(logging.INFO)


worker_init.send(sender=None)
celery_app.log.setup(
    logging.INFO,
    None,
    redirect_stdouts=False,
    colorize=False,
    hostname="probe@worker",
)
"""


@pytest.fixture
def restore_logging_state() -> Iterator[None]:
    """Restore levels, handlers, filters and the record factory after a test."""
    manager = logging.Logger.manager
    names_before = set(manager.loggerDict)
    loggers = [logging.getLogger()] + [
        logger
        for logger in list(manager.loggerDict.values())
        if isinstance(logger, logging.Logger)
    ]
    snapshot = [
        (
            logger,
            logger.level,
            list(logger.handlers),
            list(logger.filters),
            logger.propagate,
            logger.disabled,
        )
        for logger in loggers
    ]
    record_factory = logging.getLogRecordFactory()
    try:
        yield
    finally:
        for logger, level, handlers, filters, propagate, disabled in snapshot:
            logger.setLevel(level)
            logger.handlers[:] = handlers
            logger.filters[:] = filters
            logger.propagate = propagate
            logger.disabled = disabled
        logging.setLogRecordFactory(record_factory)
        # ``install_log_redaction`` creates the transport loggers if the rest
        # of the suite did not import httpx yet; they did not exist to restore.
        for name in set(manager.loggerDict) - names_before:
            created = manager.loggerDict.get(name)
            if isinstance(created, logging.Logger):
                created.filters.clear()


def _child_env(tmp_path: Path) -> dict[str, str]:
    """A development environment, so admission checks never need real keys."""
    env = {
        **os.environ,
        "XAGENT_ENVIRONMENT": "development",
        "DATABASE_URL": f"sqlite:///{tmp_path / 'celery-log.db'}",
        "LANCEDB_DIR": str(tmp_path / "lancedb"),
        "XAGENT_VECTOR_BACKEND": "lancedb",
        "XAGENT_LOG_TRANSPORT_REQUEST_LINES": "false",
    }
    for name in (
        "XAGENT_REQUIRE_SECURE_SECRETS",
        "XAGENT_CELERY_BROKER_URL",
        "XAGENT_CELERY_RESULT_BACKEND",
    ):
        env.pop(name, None)
    return env


def _run_script(script: str, tmp_path: Path) -> str:
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env=_child_env(tmp_path),
        timeout=300,
    )
    output = completed.stdout + completed.stderr
    assert completed.returncode == 0, output
    return output


def _assert_redacted(output: str) -> None:
    for secret in (_QUERY_SECRET, _REFRESH_SECRET, _BEARER_SECRET):
        assert secret not in output
    assert "token=[REDACTED]&refresh_token=[REDACTED]" in output
    assert "Authorization: Bearer [REDACTED]" in output
    # Dropped at the logger because it is below WARNING...
    assert "HTTP Request: GET" not in output
    # ...while a WARNING transport line still comes through.
    assert "transport failed with status 503" in output


def test_install_log_redaction_is_idempotent_and_keeps_handlers(
    restore_logging_state: None,
) -> None:
    root = logging.getLogger()
    handlers_before = list(root.handlers)
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(handler)
    root.setLevel(logging.INFO)

    install_log_redaction()
    factory = logging.getLogRecordFactory()
    install_log_redaction()

    logging.getLogger("xagent.web.jobs.probe").info(
        "callback https://api.example.com/cb?token=%s", _QUERY_SECRET
    )

    # A second install adds no second layer and rewrites no handler or level.
    assert logging.getLogRecordFactory() is factory
    assert root.handlers == [*handlers_before, handler]
    assert root.level == logging.INFO
    text = stream.getvalue()
    assert _QUERY_SECRET not in text
    assert "token=[REDACTED]" in text


def test_worker_logging_bootstrap_redacts_records(tmp_path: Path) -> None:
    # The ``after_setup_logger`` path: ``app.log.setup`` and nothing else.
    output = _run_script(_WORKER_LOG_SETUP + _LOG_CALLS, tmp_path)
    _assert_redacted(output)


def test_worker_init_redacts_records_when_setup_logging_is_hijacked(
    tmp_path: Path,
) -> None:
    # A ``setup_logging`` receiver makes Celery skip its own log configuration,
    # so ``after_setup_logger`` never fires; ``worker_init`` has to cover the
    # worker main process on its own. The receiver below is the shape a real
    # one has: it configures logging itself and returns None.
    output = _run_script(_HIJACKED_WORKER_LOG_SETUP + _LOG_CALLS, tmp_path)
    _assert_redacted(output)


def test_prefork_child_init_redacts_records(tmp_path: Path) -> None:
    # ``process_initializer`` is the real pool child entry: it re-runs
    # ``app.log.setup`` (a no-op unless the logfile has ``%i``) and then sends
    # ``worker_process_init``. The child starts from stock logging state on
    # purpose: a child started with execv does not inherit the parent's
    # factory, and the signal must install it there too.
    script = (
        _WORKER_LOG_SETUP
        + "\n"
        + "import multiprocessing\n"
        + "\n"
        + "\n"
        + "def child() -> None:\n"
        + "    from celery.concurrency.prefork import process_initializer\n"
        + "\n"
        + "    logging.setLogRecordFactory(logging.LogRecord)\n"
        + '    for name in ("httpx", "httpcore"):\n'
        + "        logging.getLogger(name).filters.clear()\n"
        + '    process_initializer(celery_app, "probe@worker")\n'
        + textwrap.indent(_LOG_CALLS, "    ")
        + "\n"
        + 'process = multiprocessing.get_context("fork").Process(target=child)\n'
        + "process.start()\n"
        + "process.join()\n"
        + "raise SystemExit(process.exitcode)\n"
    )
    _assert_redacted(_run_script(script, tmp_path))


def test_beat_logging_bootstrap_redacts_records(tmp_path: Path) -> None:
    # ``Beat.start_scheduler`` calls ``setup_logging()`` and then
    # ``Service.start()``, whose first statement is
    # ``signals.beat_init.send(sender=self)``; the ticking loop is not entered.
    script = (
        '''"""Beat logging bootstrap, stopped right after ``beat_init``."""\n'''
        "import logging\n"
        "\n"
        "from celery.apps.beat import Beat\n"
        "from celery.signals import beat_init\n"
        "\n"
        "from xagent.web.jobs.celery_app import celery_app\n"
        "\n"
        "beat = Beat(\n"
        "    app=celery_app,\n"
        '    loglevel="INFO",\n'
        "    logfile=None,\n"
        "    no_color=True,\n"
        "    redirect_stdouts=False,\n"
        ")\n"
        "beat.setup_logging()\n"
        "beat_init.send(sender=None)\n" + _LOG_CALLS
    )
    _assert_redacted(_run_script(script, tmp_path))


def test_a_real_solo_worker_redacts_task_output(tmp_path: Path) -> None:
    # The celery CLI with a process-local in-memory broker: the probe task is
    # sent from a thread the worker imports, then terminates the worker so the
    # test does not have to wait out a timeout.
    probe_module = (
        "import logging\n"
        "import os\n"
        "import signal\n"
        "import threading\n"
        "import time\n"
        "\n"
        "from xagent.web.jobs.celery_app import celery_app\n"
        "\n"
        "\n"
        '@celery_app.task(name="probe_log_redaction")\n'
        "def probe_log_redaction() -> None:\n"
        + textwrap.indent(_LOG_CALLS, "    ")
        + "    os.kill(os.getpid(), signal.SIGTERM)\n"
        "\n"
        "\n"
        "def _enqueue() -> None:\n"
        "    time.sleep(3.0)\n"
        "    probe_log_redaction.delay()\n"
        "\n"
        "\n"
        "threading.Thread(target=_enqueue, daemon=True).start()\n"
    )
    (tmp_path / "celery_log_redaction_probe.py").write_text(probe_module)

    env = _child_env(tmp_path)
    env["PYTHONPATH"] = str(tmp_path)
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "celery",
            "-b",
            "memory://",
            "-A",
            "xagent.web.jobs.celery_app:celery_app",
            "worker",
            "--pool=solo",
            "--loglevel=INFO",
            "--include",
            "celery_log_redaction_probe",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env=env,
        # Below the 300s pytest timeout, so a hang keeps the captured output.
        timeout=120,
    )
    output = completed.stdout + completed.stderr
    assert completed.returncode == 0, output
    _assert_redacted(output)
