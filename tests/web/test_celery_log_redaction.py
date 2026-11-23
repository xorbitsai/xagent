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

from xagent.web.logging_config import _TRANSPORT_FILTER, install_log_redaction

REPO_ROOT = Path(__file__).resolve().parents[2]

_QUERY_SECRET = "SECRET-QUERY-VALUE"
_REFRESH_SECRET = "SECRET-REFRESH-VALUE"
_BEARER_SECRET = "SECRET-BEARER-VALUE"

# The loggers ``install_log_redaction`` attaches the below-WARNING filter to.
_TRANSPORT_LOGGERS = ("httpx", "httpcore")

# Below the 300s pytest timeout, so a hang keeps the captured output.
_SUBPROCESS_TIMEOUT_SECONDS = 120

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

# A ``setup_logging`` receiver, the shape a real one has: it installs its own
# root handler and returns None. Any receiver makes Celery skip its own log
# configuration and never emit ``after_setup_logger``, so a test that connects
# this first can only pass through the signal it is meant to prove.
_SETUP_LOGGING_RECEIVER = """\
import logging
import sys

from celery.signals import setup_logging


@setup_logging.connect
def configure_logging(**_: object) -> None:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(logging.INFO)
"""

_LOG_SETUP_CALL = """\
celery_app.log.setup(
    logging.INFO,
    None,
    redirect_stdouts=False,
    colorize=False,
    hostname="probe@worker",
)
"""

# ``Worker.setup_logging`` calls ``app.log.setup``, which emits
# ``setup_logging``/``after_setup_logger``/``after_setup_task_logger`` after
# Celery replaced the root handlers. With no ``setup_logging`` receiver, the
# worker main process is covered by ``after_setup_logger``.
_WORKER_LOG_SETUP = (
    """\
import logging

from xagent.web.jobs.celery_app import celery_app

"""
    + _LOG_SETUP_CALL
)

# The same worker once ``_SETUP_LOGGING_RECEIVER`` took the log configuration
# over: Celery never emits ``after_setup_logger``, so ``worker_init``, which
# ``Worker.init_worker`` sends before ``app.log.setup``, has to cover the main
# process on its own.
_HIJACKED_WORKER_LOG_SETUP = (
    _SETUP_LOGGING_RECEIVER
    + textwrap.dedent(
        """
        from celery.signals import worker_init

        from xagent.web.jobs.celery_app import celery_app

        worker_init.send(sender=None)
        """
    )
    + _LOG_SETUP_CALL
)

# ``process_initializer`` is the real pool child entry: it re-runs
# ``app.log.setup`` (a no-op unless the logfile has ``%i``) and then sends
# ``worker_process_init``. The child starts from stock logging state on
# purpose: a child started with execv does not inherit the parent's factory,
# and the signal must install it there too.
_PREFORK_CHILD_INIT = (
    _WORKER_LOG_SETUP
    + textwrap.dedent(
        """
        import multiprocessing


        def child() -> None:
            from celery.concurrency.prefork import process_initializer

            logging.setLogRecordFactory(logging.LogRecord)
            for name in ("httpx", "httpcore"):
                logging.getLogger(name).filters.clear()
            process_initializer(celery_app, "probe@worker")
        """
    )
    + textwrap.indent(_LOG_CALLS, "    ")
    + textwrap.dedent(
        """
        process = multiprocessing.get_context("fork").Process(target=child)
        process.start()
        process.join()
        raise SystemExit(process.exitcode)
        """
    )
)

# ``Beat.start_scheduler`` calls ``setup_logging()`` and then
# ``Service.start()``, whose first statement is
# ``signals.beat_init.send(sender=self)``; the ticking loop is not entered.
# ``_SETUP_LOGGING_RECEIVER`` takes the log configuration over first, so
# ``Beat.setup_logging`` never emits ``after_setup_logger`` and only
# ``beat_init`` can install redaction here.
_BEAT_LOG_SETUP = (
    _SETUP_LOGGING_RECEIVER
    + textwrap.dedent(
        """
        from celery.apps.beat import Beat
        from celery.signals import beat_init

        from xagent.web.jobs.celery_app import celery_app

        beat = Beat(
            app=celery_app,
            loglevel="INFO",
            logfile=None,
            no_color=True,
            redirect_stdouts=False,
        )
        beat.setup_logging()
        beat_init.send(sender=None)
        """
    )
    + _LOG_CALLS
)

# The task the real worker runs. ``worker_ready`` is sent from the consumer
# loop right before it reads the broker, so the task is enqueued there instead
# of after a guessed sleep; it terminates the worker once it has logged.
_PROBE_MODULE = (
    textwrap.dedent(
        """\
        import logging
        import os
        import signal

        from celery.signals import worker_ready

        from xagent.web.jobs.celery_app import celery_app


        @celery_app.task(name="probe_log_redaction")
        def probe_log_redaction() -> None:
        """
    )
    + textwrap.indent(_LOG_CALLS, "    ")
    + textwrap.dedent(
        """\
            os.kill(os.getpid(), signal.SIGTERM)


        @worker_ready.connect
        def _enqueue(**_: object) -> None:
            probe_log_redaction.delay()
        """
    )
)


def _transport_loggers() -> dict[str, logging.Logger]:
    """The loggers ``install_log_redaction`` attaches the filter to."""
    return {
        name: logger
        for name, logger in logging.Logger.manager.loggerDict.items()
        if isinstance(logger, logging.Logger)
        and (name in _TRANSPORT_LOGGERS or name.startswith(("httpx.", "httpcore.")))
    }


@pytest.fixture
def restore_logging_state() -> Iterator[None]:
    """Restore what this test rewrites and what ``install_log_redaction`` sets.

    Only the root handler and level this test adds, the process-wide record
    factory and the transport logger filters can be left behind; nothing else
    on the logger tree is touched.
    """
    root = logging.getLogger()
    handlers_before = list(root.handlers)
    level_before = root.level
    filters_before = {
        name: list(logger.filters) for name, logger in _transport_loggers().items()
    }
    record_factory = logging.getLogRecordFactory()
    try:
        yield
    finally:
        root.handlers[:] = handlers_before
        root.setLevel(level_before)
        logging.setLogRecordFactory(record_factory)
        for name, logger in _transport_loggers().items():
            # A logger this test created has no snapshot and drops the filter.
            logger.filters[:] = filters_before.get(name, [])


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
        timeout=_SUBPROCESS_TIMEOUT_SECONDS,
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
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("XAGENT_LOG_TRANSPORT_REQUEST_LINES", "false")
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

    # A second install adds no second layer or filter and rewrites no handler
    # or level.
    assert logging.getLogRecordFactory() is factory
    assert root.handlers == [*handlers_before, handler]
    assert root.level == logging.INFO
    for name in _TRANSPORT_LOGGERS:
        assert logging.getLogger(name).filters.count(_TRANSPORT_FILTER) == 1
    text = stream.getvalue()
    assert _QUERY_SECRET not in text
    assert "token=[REDACTED]" in text


@pytest.mark.slow
@pytest.mark.parametrize(
    "script",
    [
        # No receiver on ``setup_logging``: ``after_setup_logger`` alone.
        pytest.param(_WORKER_LOG_SETUP, id="after_setup_logger"),
        # A receiver took the log configuration over: ``worker_init`` alone.
        pytest.param(_HIJACKED_WORKER_LOG_SETUP, id="worker_init"),
    ],
)
def test_worker_logging_bootstrap_redacts_records(script: str, tmp_path: Path) -> None:
    """Either signal covers the worker main process without the other."""
    _assert_redacted(_run_script(script + _LOG_CALLS, tmp_path))


@pytest.mark.slow
def test_prefork_child_init_redacts_records(tmp_path: Path) -> None:
    _assert_redacted(_run_script(_PREFORK_CHILD_INIT, tmp_path))


@pytest.mark.slow
def test_beat_logging_bootstrap_redacts_records(tmp_path: Path) -> None:
    _assert_redacted(_run_script(_BEAT_LOG_SETUP, tmp_path))


@pytest.mark.slow
def test_a_real_solo_worker_redacts_task_output(tmp_path: Path) -> None:
    # The celery CLI with a process-local in-memory broker; the probe module
    # enqueues on ``worker_ready`` and terminates the worker from the task, so
    # the test neither guesses a startup delay nor waits out a timeout.
    (tmp_path / "celery_log_redaction_probe.py").write_text(_PROBE_MODULE)

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
        timeout=_SUBPROCESS_TIMEOUT_SECONDS,
    )
    output = completed.stdout + completed.stderr
    assert completed.returncode == 0, output
    _assert_redacted(output)
