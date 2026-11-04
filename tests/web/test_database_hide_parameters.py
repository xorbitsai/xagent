"""configure_db must always pass hide_parameters=True to create_engine.

Without it, a SQLAlchemy StatementError's default __str__ includes bound
parameter values -- any `except Exception: logger.x(e)` or `str(e)`
anywhere in the app that touches a commit/execute failure on a
credential-bearing table (UserOAuth.access_token/refresh_token,
User.password_hash) can otherwise put a live secret into a log or a
client-facing response.
"""

from __future__ import annotations

from unittest.mock import Mock

from xagent.web.models import database as database_module


def test_configure_db_hides_parameters_for_sqlite(monkeypatch, tmp_path):
    create_engine_mock = Mock(wraps=database_module.create_engine)
    monkeypatch.setattr(database_module, "create_engine", create_engine_mock)
    monkeypatch.setattr(database_module, "apply_sqlite_concurrency_pragmas", Mock())

    database_module.configure_db(f"sqlite:///{tmp_path / 'test.db'}")

    assert create_engine_mock.call_args.kwargs["hide_parameters"] is True


def test_configure_db_hides_parameters_for_non_sqlite(monkeypatch):
    create_engine_mock = Mock(return_value=Mock())
    monkeypatch.setattr(database_module, "create_engine", create_engine_mock)

    database_module.configure_db("postgresql://user:pass@localhost/db")

    assert create_engine_mock.call_args.kwargs["hide_parameters"] is True
