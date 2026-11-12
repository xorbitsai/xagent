"""``python -m xagent.web.connector_checks``: one JSON document on stdout.

The document is the run result; logs go to stderr. Most tests replace the
database and the adapter; the refusal test runs the real ``user`` adapter
against a seeded database.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import subprocess
import sys
from collections.abc import Iterator
from datetime import datetime, timezone
from typing import Any

import jsonschema
import pytest

from tests.web.services.test_connector_probe import (
    _connect_catalog_stripe,
    _github_ok,
    _Loader,
    _stripe_ok,
)
from tests.web.services.test_mcp_connector_session import (
    LOADER,
    _connect_github,
)
from xagent.web.connector_checks import cli
from xagent.web.connector_checks.identity import ConnectorTarget
from xagent.web.connector_checks.results import RunResult, load_result_schema
from xagent.web.connector_checks.sweeper import load_sweep_schema
from xagent.web.services.mcp_connector_session import ConnectorToolsOutcome

from .fakes import FakeAdapter, FakeTool, failed, listed

SECRET = "postgresql://ops:hunter2@db.internal/xagent"


def _with_database(monkeypatch: pytest.MonkeyPatch, db: Any) -> None:
    @contextlib.contextmanager
    def database() -> Iterator[Any]:
        yield db

    monkeypatch.setattr(cli, "_database", database)


def _install_adapter(
    monkeypatch: pytest.MonkeyPatch, adapter: FakeAdapter
) -> list[str]:
    """Run the CLI under ``adapter``, without a database; returns the user
    names the adapter was built for."""
    names: list[str] = []

    def build(db: Any, user_name: str) -> FakeAdapter:
        names.append(user_name)
        return adapter

    _with_database(monkeypatch, None)
    monkeypatch.setattr(cli, "UserIdentityAdapter", build)
    return names


def _record_engine(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Replace the engine with one that records its keywords."""
    seen: dict[str, Any] = {}

    async def run_checks(adapter: Any, **kwargs: Any) -> RunResult:
        seen.update(kwargs)
        now = datetime.now(timezone.utc)
        return RunResult(adapter.label, now, now, True, ())

    monkeypatch.setattr(cli, "run_checks", run_checks)
    return seen


def _document(out: str) -> dict[str, Any]:
    """The whole of stdout, read as one JSON document of the result schema."""
    document = json.loads(out)
    jsonschema.validate(document, load_result_schema())
    return document


def _slack(tool: FakeTool | None = None) -> FakeAdapter:
    tool = tool or FakeTool("slack_list_channels")
    return FakeAdapter([ConnectorTarget("slack")], {"slack": listed(tool)})


def _stripe() -> FakeAdapter:
    """A legacy connector, whose one check is its L3 probe."""
    tool = FakeTool("stripe_list_customers")
    return FakeAdapter([ConnectorTarget("stripe")], {"stripe": listed(tool)})


def test_a_passing_run_prints_its_document_and_exits_0(monkeypatch, capfd):
    names = _install_adapter(monkeypatch, _stripe())

    code = cli.main(["--user", "monitor"])

    out, _ = capfd.readouterr()
    document = _document(out)
    assert code == 0
    assert names == ["monitor"]
    assert document["identity"] == "user"
    assert document["complete"] is True
    assert [row["outcome"] for row in document["rows"]] == ["pass", "pass"]


def test_stdout_carries_only_the_document(monkeypatch, capfd):
    """Whatever a connector, a tool or a library writes during the run, at
    the Python level, to file descriptor 1 or from a child process, goes to
    stderr."""

    async def noisy_session():
        print("print from a tool")
        sys.stdout.flush()
        os.write(1, b"raw write to fd 1\n")
        subprocess.run(
            [sys.executable, "-c", "print('child process output')"], check=True
        )
        logging.getLogger("xagent.test").warning("a log line")
        return listed(FakeTool("slack_list_channels"))

    _install_adapter(
        monkeypatch, FakeAdapter([ConnectorTarget("slack")], {"slack": noisy_session})
    )
    # pytest's root logger has handlers, which would make basicConfig a no-op.
    root = logging.getLogger()
    monkeypatch.setattr(root, "handlers", [])
    monkeypatch.setattr(root, "level", root.level)

    code = cli.main(["--user", "monitor"])

    out, err = capfd.readouterr()
    assert code == 0
    assert out.endswith("}\n") and out.count("\n") == 1
    _document(out)
    for line in ("print from a tool", "raw write to fd 1", "child process output"):
        assert line in err
    assert "a log line" in err
    assert "a log line" not in out
    # Stdout is back on its own stream after the run.
    print("after the run")
    assert capfd.readouterr().out == "after the run\n"


def test_a_failed_row_exits_1(monkeypatch, capfd):
    adapter = FakeAdapter(
        [ConnectorTarget("slack")],
        {"slack": failed(ConnectorToolsOutcome.NOT_CONNECTED, "token_refresh_failed")},
    )
    _install_adapter(monkeypatch, adapter)

    code = cli.main(["--user", "monitor"])

    out, _ = capfd.readouterr()
    assert code == 1
    assert _document(out)["rows"][0]["failure_class"] == "not_connected"


def test_a_run_the_total_timeout_cut_exits_1(monkeypatch, capfd):
    _install_adapter(monkeypatch, _slack(FakeTool("slack_list_channels", delay=1.0)))

    code = cli.main(["--user", "monitor", "--total-timeout", "0.05"])

    out, _ = capfd.readouterr()
    assert code == 1
    assert _document(out)["complete"] is False


def test_skipped_rows_do_not_fail_the_run(monkeypatch, capfd):
    adapter = FakeAdapter(
        [ConnectorTarget("stripe", skip_reason="not_catalog_identity")]
    )
    _install_adapter(monkeypatch, adapter)

    code = cli.main(["--user", "monitor"])

    out, _ = capfd.readouterr()
    assert code == 0
    assert [row["outcome"] for row in _document(out)["rows"]] == ["skip", "skip"]


def test_selectors_and_bounds_reach_the_engine(monkeypatch, capfd):
    _install_adapter(monkeypatch, _slack())
    seen = _record_engine(monkeypatch)

    cli.main(
        [
            "--user",
            "monitor",
            "--connector",
            "slack, notion",
            "--connector",
            "jira",
            "--check",
            "slack.l3_probe",
            "--environment",
            "stage",
            "--check-timeout",
            "5",
            "--total-timeout",
            "60",
        ]
    )

    assert seen == {
        "environment": "stage",
        "connectors": frozenset({"slack", "notion", "jira"}),
        "checks": frozenset({"slack.l3_probe"}),
        "check_timeout": 5.0,
        "total_timeout": 60.0,
    }


def test_without_selectors_every_connector_and_check_runs(monkeypatch, capfd):
    _install_adapter(monkeypatch, _slack())
    seen = _record_engine(monkeypatch)

    cli.main(["--user", "monitor"])

    assert seen["connectors"] is None and seen["checks"] is None
    assert seen["environment"] is None and seen["check_timeout"] is None
    assert seen["total_timeout"] == cli.DEFAULT_TOTAL_TIMEOUT_SECONDS


def test_a_database_that_cannot_open_is_a_run_failure(monkeypatch, capfd, caplog):
    @contextlib.contextmanager
    def broken() -> Iterator[None]:
        raise RuntimeError(SECRET)
        yield

    monkeypatch.setattr(cli, "_database", broken)

    code = cli.main(["--user", "monitor", "--environment", "stage"])

    out, err = capfd.readouterr()
    document = _document(out)
    assert code == 1
    assert document["failure"] == "setup:database"
    assert document["environment"] == "stage"
    assert document["rows"] == []
    assert "RuntimeError" in caplog.text
    assert SECRET not in out + err + caplog.text


def test_an_engine_that_raises_is_a_run_failure(monkeypatch, capfd, caplog):
    async def run_checks(adapter: Any, **kwargs: Any) -> RunResult:
        raise ValueError(SECRET)

    _install_adapter(monkeypatch, _slack())
    monkeypatch.setattr(cli, "run_checks", run_checks)

    code = cli.main(["--user", "monitor"])

    out, err = capfd.readouterr()
    assert code == 1
    assert _document(out)["failure"] == "run:exception"
    assert "ValueError" in caplog.text
    assert SECRET not in out + err + caplog.text


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["--user", "monitor", "--environment", "Not A Label"],
        ["--user", "monitor", "--total-timeout", "0"],
        ["--user", "monitor", "--check-timeout", "nan"],
        ["--user", "monitor", "--check-timeout", "inf"],
        ["--user", "monitor", "--total-timeout", "soon"],
        # Selectors only: a tool name or an argument is not an option.
        ["--user", "monitor", "--tool", "slack_list_channels"],
        ["--user", "monitor", "--arguments", "{}"],
        # A selector must name something.
        ["--user", "monitor", "--connector", ""],
        ["--user", "monitor", "--connector", ","],
        ["--user", "monitor", "--check", ""],
        # A sweep has no checks to select.
        ["--user", "monitor", "--sweep", "--check", "slack.l3_probe"],
    ],
)
def test_invalid_arguments_exit_2_with_nothing_on_stdout(monkeypatch, capfd, argv):
    names = _install_adapter(monkeypatch, _slack())

    with pytest.raises(SystemExit) as exited:
        cli.main(argv)

    out, _ = capfd.readouterr()
    assert exited.value.code == 2
    assert out == ""
    assert names == []


def test_sweep_prints_the_sweep_document(monkeypatch, capfd):
    adapter = _slack()
    names = _install_adapter(monkeypatch, adapter)

    code = cli.main(["--user", "monitor", "--sweep", "--environment", "stage"])

    out, _ = capfd.readouterr()
    document = json.loads(out)
    jsonschema.validate(document, load_sweep_schema())
    assert document["schema"] == "xagent.connector_smoke.sweep"
    assert document["environment"] == "stage"
    # Writes are off, so whatever the registry declares, nothing starts.
    assert document["complete"] is True
    assert code == 0
    assert names == ["monitor"]
    assert adapter.opened == []


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        ([], {"total_timeout": 600.0, "call_timeout": None, "connectors": None}),
        (
            ["--total-timeout", "60", "--check-timeout", "5", "--connector", "jira"],
            {"total_timeout": 60.0, "call_timeout": 5.0, "connectors": {"jira"}},
        ),
    ],
)
def test_sweep_options_reach_the_sweeper(monkeypatch, capfd, argv, expected):
    _install_adapter(monkeypatch, _slack())
    seen: dict[str, Any] = {}

    async def record(adapter: Any, **kwargs: Any) -> Any:
        seen.update(kwargs)
        raise RuntimeError("stop here")

    monkeypatch.setattr(cli, "run_sweep", record)

    cli.main(["--user", "monitor", "--sweep", *argv])

    assert seen["total_timeout"] == expected["total_timeout"]
    assert seen["call_timeout"] == expected["call_timeout"]
    assert seen["connectors"] == expected["connectors"]


def test_a_sweep_that_cannot_open_the_database_prints_a_sweep_failure(
    monkeypatch, capfd
):
    @contextlib.contextmanager
    def database() -> Iterator[Any]:
        raise RuntimeError(SECRET)
        yield

    monkeypatch.setattr(cli, "_database", database)

    code = cli.main(["--user", "monitor", "--sweep"])

    out, _ = capfd.readouterr()
    document = json.loads(out)
    jsonschema.validate(document, load_sweep_schema())
    assert document["failure"] == "setup:database"
    assert SECRET not in out
    assert code == 1


# --- the real adapter ------------------------------------------------------


def test_a_user_outside_the_probe_users_is_refused_before_a_connector_starts(
    db_users, probe_users, monkeypatch, capfd
):
    db, _, other_user = db_users
    _connect_github(db, other_user)
    loader = _Loader(_github_ok())
    monkeypatch.setattr(LOADER, loader)
    _with_database(monkeypatch, db)

    code = cli.main(["--user", "other"])

    out, _ = capfd.readouterr()
    document = _document(out)
    assert code == 1
    assert document["failure"] == "identity:user_not_allowed"
    assert document["rows"] == []
    assert loader.calls == []


def test_a_probe_user_runs_their_connectors_checks(
    db_users, probe_users, monkeypatch, capfd
):
    db, user, _ = db_users
    _connect_catalog_stripe(db, user)
    monkeypatch.setattr(LOADER, _Loader(_stripe_ok()))
    _with_database(monkeypatch, db)

    code = cli.main(["--user", "monitor@example.com"])

    out, _ = capfd.readouterr()
    assert code == 0
    assert [
        (row["connector"], row["check"], row["outcome"])
        for row in _document(out)["rows"]
    ] == [("stripe", None, "pass"), ("stripe", "stripe.l3_probe", "pass")]


@pytest.mark.parametrize(
    ("user", "failure"),
    [("other", "identity:user_not_allowed"), ("monitor", "enumeration:no_connectors")],
)
def test_the_module_runs_as_a_one_shot_command(
    db_users, tmp_path, user: str, failure: str
):
    """``python -m`` in a fresh process, on the service's database: the
    document is the whole of stdout."""
    db, _, _ = db_users
    env = {
        **os.environ,
        "DATABASE_URL": db.get_bind().url.render_as_string(hide_password=False),
        "XAGENT_CONNECTOR_PROBE_USERS": "monitor",
    }

    completed = subprocess.run(
        [sys.executable, "-m", "xagent.web.connector_checks", "--user", user],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert completed.returncode == 1, completed.stderr
    assert _document(completed.stdout)["failure"] == failure


def test_a_database_that_cannot_connect_is_setup_database(tmp_path):
    """The engine connects lazily: an unreachable database must fail at
    setup, not inside the enumeration. SQLite cannot open a directory."""
    directory = tmp_path / "not-a-file"
    directory.mkdir()
    env = {
        **os.environ,
        "DATABASE_URL": f"sqlite:///{directory}",
        "XAGENT_CONNECTOR_PROBE_USERS": "monitor",
    }

    completed = subprocess.run(
        [sys.executable, "-m", "xagent.web.connector_checks", "--user", "monitor"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert completed.returncode == 1, completed.stderr
    assert _document(completed.stdout)["failure"] == "setup:database"
