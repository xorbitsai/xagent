import json

import pytest
from sqlalchemy import inspect
from sqlalchemy.exc import OperationalError

from xagent.web import reconcile_gmail_push_endpoints as cli
from xagent.web.models import database
from xagent.web.services.gmail_provisioning import (
    GmailProvisioningError,
    GmailPushEndpointReconciliation,
)


class FakeSession:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def test_cli_defaults_to_audit_mode(monkeypatch, capsys) -> None:
    session = FakeSession()
    calls: list[bool] = []
    configured: list[bool] = []
    monkeypatch.setattr(
        cli,
        "configure_db",
        lambda *, read_only: configured.append(read_only),
        raising=False,
    )
    monkeypatch.setattr(
        cli,
        "init_db",
        lambda: (_ for _ in ()).throw(AssertionError("audit initialized the schema")),
    )
    monkeypatch.setattr(cli, "get_session_local", lambda: lambda: session)
    monkeypatch.setattr(
        cli,
        "reconcile_gmail_push_endpoints",
        lambda _db, *, execute: (
            calls.append(execute)
            or GmailPushEndpointReconciliation(
                scanned=2,
                changed=1,
                unchanged=1,
                failed=0,
            )
        ),
    )

    assert cli.run([]) == 0
    assert configured == [True]
    assert calls == [False]
    assert session.closed is True
    assert json.loads(capsys.readouterr().out) == {
        "mode": "audit",
        "scanned": 2,
        "changed": 1,
        "unchanged": 1,
        "skipped": 0,
        "failed": 0,
        "errors": [],
    }


def test_cli_execute_returns_failure_when_any_subscription_fails(
    monkeypatch, capsys
) -> None:
    session = FakeSession()
    initialized: list[bool] = []
    monkeypatch.setattr(cli, "init_db", lambda: initialized.append(True))
    monkeypatch.setattr(
        cli,
        "configure_db",
        lambda: (_ for _ in ()).throw(
            AssertionError("execute bypassed schema initialization")
        ),
        raising=False,
    )
    monkeypatch.setattr(cli, "get_session_local", lambda: lambda: session)
    monkeypatch.setattr(
        cli,
        "reconcile_gmail_push_endpoints",
        lambda _db, *, execute: GmailPushEndpointReconciliation(
            scanned=1,
            changed=0,
            unchanged=0,
            failed=1,
            errors=("watch 9: permission denied",),
        ),
    )

    assert cli.run(["--execute"]) == 1
    assert initialized == [True]
    assert json.loads(capsys.readouterr().out)["mode"] == "execute"
    assert session.closed is True


@pytest.mark.parametrize(
    "error",
    [
        GmailProvisioningError("missing Gmail Pub/Sub project"),
        ValueError("invalid XAGENT_S2S_API_BASE_URL"),
    ],
)
def test_cli_reports_invalid_provisioning_config_as_json(
    monkeypatch,
    capsys,
    error: Exception,
) -> None:
    session = FakeSession()
    monkeypatch.setattr(cli, "configure_db", lambda *, read_only: None)
    monkeypatch.setattr(cli, "get_session_local", lambda: lambda: session)
    monkeypatch.setattr(
        cli,
        "reconcile_gmail_push_endpoints",
        lambda _db, *, execute: (_ for _ in ()).throw(error),
    )

    assert cli.run([]) == 1
    assert session.closed is True
    assert json.loads(capsys.readouterr().out) == {
        "mode": "audit",
        "scanned": 0,
        "changed": 0,
        "unchanged": 0,
        "skipped": 0,
        "failed": 1,
        "errors": [str(error)],
    }


def test_configure_db_read_only_does_not_create_a_missing_sqlite_file(
    tmp_path,
) -> None:
    database_path = tmp_path / "missing-audit.db"
    database.configure_db(
        f"sqlite:///{database_path}",
        read_only=True,
    )

    with pytest.raises(OperationalError):
        inspect(database.get_engine()).get_table_names()
    assert not database_path.exists()


def test_configure_db_read_only_preserves_named_memory_sqlite_mode() -> None:
    database.configure_db(
        "sqlite:///file:gmail-audit-memory?mode=memory&cache=shared&uri=true",
        read_only=True,
    )

    engine = database.get_engine()
    assert engine.url.query["mode"] == "memory"
    with engine.connect() as connection:
        assert connection.exec_driver_sql("SELECT 1").scalar_one() == 1


def test_configure_db_enforces_read_only_postgresql_transactions(
    monkeypatch,
) -> None:
    created: dict[str, object] = {}
    engine = object()

    def fake_create_engine(url: str, **kwargs):
        created["url"] = url
        created["kwargs"] = kwargs
        return engine

    monkeypatch.setattr(database, "create_engine", fake_create_engine)
    monkeypatch.setattr(database, "sessionmaker", lambda **_kwargs: object())

    database.configure_db(
        "postgresql://xagent:secret@db.example.com/xagent",
        read_only=True,
    )

    assert created["kwargs"] == {
        "poolclass": database.QueuePool,
        **database.get_db_pool_kwargs(),
        "hide_parameters": True,
        "execution_options": {"postgresql_readonly": True},
    }
