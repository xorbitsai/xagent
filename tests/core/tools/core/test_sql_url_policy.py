"""Deployment SQL policies apply before persistence and runtime dispatch."""

from unittest.mock import Mock, patch

import pytest
from sqlalchemy.engine import make_url

from xagent.core.tools.core import sql_tool
from xagent.web.services import tool_credentials

ERROR = "File-backed SQLite is retired; migrate this connection to PostgreSQL or MySQL."


@pytest.fixture
def deployment_policy(monkeypatch):
    def reject_files(url):
        if url.get_backend_name() == "sqlite" and url.database not in (
            None,
            "",
            ":memory:",
        ):
            raise ValueError(ERROR)

    monkeypatch.setattr(sql_tool, "_external_database_url_policy", reject_files)


@pytest.mark.parametrize(
    "url", ["sqlite:///old.db", "sqlite+pysqlite:////stored/old.db"]
)
def test_policy_blocks_new_saves_before_any_persistence(deployment_policy, url):
    db = Mock()
    with pytest.raises(ValueError, match="File-backed SQLite"):
        tool_credentials.set_sql_connection(db, 1, "legacy", url)
    db.query.assert_not_called()
    db.commit.assert_not_called()


@pytest.mark.parametrize("source", ["explicit", "environment"])
def test_existing_urls_are_rejected_before_worker_or_local_execution(
    deployment_policy, monkeypatch, source
):
    monkeypatch.setenv("XAGENT_SQL_WORKER_URL", "http://sql-worker:8090")
    monkeypatch.setenv("XAGENT_EXTERNAL_DB_LEGACY", "sqlite:///old.db")
    with (
        patch.object(sql_tool, "execute_sql_query_remote") as remote,
        patch.object(sql_tool, "create_engine") as engine,
        pytest.raises(ValueError, match="File-backed SQLite"),
    ):
        sql_tool.execute_sql_query(
            "legacy",
            "SELECT 1",
            connection_url="sqlite:///old.db" if source == "explicit" else None,
        )
    remote.assert_not_called()
    engine.assert_not_called()


def test_listing_retired_credentials_preserves_them_and_shows_migration(
    deployment_policy,
):
    stored = {"LEGACY": {"masked": "sqlite:///old.db", "ciphertext": "keep-unchanged"}}
    db = Mock()
    db.query.return_value.filter.return_value.first.return_value = Mock(
        config={"sql_connections": stored}
    )
    result = tool_credentials.list_sql_connections(db, 1)
    item = next(item for item in result if item["name"] == "LEGACY")
    assert item["configured"] is False
    assert item["error"] == ERROR
    assert stored["LEGACY"]["ciphertext"] == "keep-unchanged"
    db.commit.assert_not_called()


@pytest.mark.parametrize(
    "url",
    [
        "sqlite://",
        "sqlite:///:memory:",
        "postgresql://customer/db",
        "mysql://customer/db",
    ],
)
def test_policy_keeps_supported_memory_and_remote_databases(deployment_policy, url):
    sql_tool.validate_external_database_url_policy(url)


def test_unregistered_policy_preserves_standalone_file_connections(monkeypatch):
    monkeypatch.setattr(sql_tool, "_external_database_url_policy", None)
    sql_tool.validate_external_database_url_policy("sqlite:///old.db")
    sql_tool.set_external_database_url_policy(lambda url: None)
    sql_tool.validate_external_database_url_policy(make_url("sqlite:///old.db"))
    sql_tool.set_external_database_url_policy(None)
