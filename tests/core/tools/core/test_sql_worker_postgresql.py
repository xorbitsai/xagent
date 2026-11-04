"""Real psycopg2 regressions for the disposable SQL worker, not driver mocks."""

import json
import os
from contextlib import nullcontext

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import event, text
from sqlalchemy.engine import make_url

from tests.shared.postgres_disposable import disposable_database_factory
from xagent.core.tools.core.sql_tool import _execute_sql_query_local
from xagent.core.tools.core.sql_worker_client import execute_sql_query_remote
from xagent.core.workspace import TaskWorkspace
from xagent.sql_worker import (
    SENSITIVE_ENVIRONMENT_SUFFIXES,
    SENSITIVE_PLATFORM_ENVIRONMENT_VARIABLES,
    QueryRequest,
    app,
    execute_in_disposable_process,
)

pytestmark = pytest.mark.postgresql


@pytest.fixture
def worker_database(monkeypatch, tmp_path):
    for variable in tuple(os.environ):
        if variable in SENSITIVE_PLATFORM_ENVIRONMENT_VARIABLES or variable.endswith(
            SENSITIVE_ENVIRONMENT_SUFFIXES
        ):
            monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("XAGENT_SQL_WORKER_TOKEN", "x" * 32)
    monkeypatch.setenv("XAGENT_SQL_WORKER_QUERY_TIMEOUT_SECONDS", "30")
    monkeypatch.setenv("XAGENT_SQL_WORKER_MAX_ROWS", "10")
    monkeypatch.setenv("XAGENT_SQL_WORKER_MAX_RESULT_BYTES", "4096")
    monkeypatch.setattr(
        "xagent.sql_worker.DOCKER_SOCKET_PATH", tmp_path / "absent.sock"
    )
    with disposable_database_factory("sql_worker_review") as make_database:
        engine = make_database("queries")
        with engine.connect() as connection:
            name = connection.execute(text("SELECT current_database()")).scalar_one()
        url = (
            make_url(os.environ["XAGENT_TEST_POSTGRES_URL"])
            .set(drivername="postgresql+psycopg2", database=name)
            .render_as_string(hide_password=False)
        )
        yield engine, url


def test_postgresql_worker_commits_dml_ddl_and_returning(worker_database):
    engine, url = worker_database

    def run(query):
        result = execute_in_disposable_process(
            QueryRequest(connection_name="external", connection_url=url, query=query)
        )
        assert result["success"] is True
        return result

    run("CREATE TABLE metrics (value integer)")
    run("INSERT INTO metrics VALUES (42)")
    run("UPDATE metrics SET value = 43")
    with engine.connect() as connection:
        assert connection.execute(text("SELECT value FROM metrics")).scalar_one() == 43
    assert run("INSERT INTO metrics VALUES (44) RETURNING value")["rows"] == [
        {"value": 44}
    ]
    result = run(
        "WITH changed AS (UPDATE metrics SET value = value + 1 RETURNING value) SELECT value FROM changed ORDER BY value"
    )
    assert result["rows"] == [{"value": 44}, {"value": 45}]
    with engine.connect() as connection:
        assert connection.execute(
            text("SELECT value FROM metrics ORDER BY value")
        ).scalars().all() == [44, 45]
    assert (
        run(
            "WITH rows AS (SELECT value FROM metrics) SELECT value FROM rows ORDER BY value"
        )["row_count"]
        == 2
    )
    run("SELECT 7 AS value INTO copied_metrics")
    with engine.connect() as connection:
        assert (
            connection.execute(text("SELECT value FROM copied_metrics")).scalar_one()
            == 7
        )
    run("DELETE FROM metrics")
    with engine.connect() as connection:
        assert (
            connection.execute(text("SELECT count(*) FROM metrics")).scalar_one() == 0
        )
    run("DROP TABLE metrics")
    with engine.connect() as connection:
        assert (
            connection.execute(text("SELECT to_regclass('metrics')")).scalar_one()
            is None
        )


def test_postgresql_reads_stream_but_writes_use_ordinary_cursors(
    worker_database, monkeypatch
):
    engine, url = worker_database
    observed = []

    @event.listens_for(engine, "before_cursor_execute")
    def record_cursor(_connection, cursor, statement, _parameters, _context, _many):
        observed.append((statement, bool(cursor.name)))

    monkeypatch.setattr(
        "xagent.core.tools.core.sql_tool.create_engine", lambda *_a, **_k: engine
    )
    queries = [
        ("CREATE TABLE cursor_metrics (value int)", False),
        ("INSERT INTO cursor_metrics VALUES (1) RETURNING value", False),
        ("SELECT value FROM cursor_metrics", True),
        ("WITH rows AS (SELECT value FROM cursor_metrics) SELECT * FROM rows", True),
        ("SELECT value INTO cursor_copy FROM cursor_metrics", False),
    ]
    for query, streamed in queries:
        _execute_sql_query_local(
            "external", query, connection_url=url, max_rows=10, max_result_bytes=4096
        )
        assert observed[-1] == (query, streamed)


def test_postgresql_bytea_survives_worker_pipe_http_and_remote_export(
    worker_database, monkeypatch, tmp_path
):
    engine, url = worker_database
    query = "SELECT decode('ff00', 'hex') AS payload"
    with engine.connect() as connection:
        raw = connection.execute(text(query)).scalar_one()
        assert isinstance(raw, memoryview)  # Actual psycopg2 bytea representation.
    monkeypatch.setenv("XAGENT_SQL_WORKER_URL", "http://sql-worker:8090")
    workspace = TaskWorkspace("bytea-export", base_dir=str(tmp_path))
    with TestClient(app) as client:
        monkeypatch.setattr(
            "xagent.core.tools.core.sql_worker_client.httpx.Client",
            lambda **_: nullcontext(client),
        )
        result = execute_sql_query_remote(
            connection_name="external",
            connection_url=url,
            query=query,
            output_file="binary.jsonl",
            workspace=workspace,
        )
    assert result["success"] is True
    exported = workspace.resolve_write_path("binary.jsonl", default_dir="output")
    assert json.loads(exported.read_text()) == {"payload": "base64:/wA="}


def test_postgresql_scalar_precision_and_nonfinite_values_survive_http_and_export(
    worker_database, monkeypatch, tmp_path
):
    import math
    from datetime import timedelta
    from decimal import Decimal

    engine, url = worker_database
    query = """SELECT 9007199254740993.01::numeric AS amount,
                      INTERVAL '1 day 2 hours' AS duration,
                      'NaN'::double precision AS nan,
                      'Infinity'::double precision AS positive,
                      '-Infinity'::double precision AS negative,
                      42 AS "_salary"
    """
    with engine.connect() as connection:
        raw = connection.execute(text(query)).mappings().one()
        assert raw["amount"] == Decimal("9007199254740993.01")
        assert raw["duration"] == timedelta(days=1, hours=2)
        assert math.isnan(raw["nan"])
        assert math.isinf(raw["positive"]) and math.isinf(raw["negative"])
    expected = {
        "amount": "9007199254740993.01",
        "duration": "P1DT2H",
        "nan": None,
        "positive": None,
        "negative": None,
        "_salary": 42,
    }
    monkeypatch.setenv("XAGENT_SQL_WORKER_URL", "http://sql-worker:8090")
    workspace = TaskWorkspace("scalar-export", base_dir=str(tmp_path))
    with TestClient(app) as client:
        response = client.post(
            "/v1/query",
            headers={"Authorization": f"Bearer {'x' * 32}"},
            json={"connection_name": "external", "connection_url": url, "query": query},
        )
        assert response.status_code == 200
        assert response.json()["rows"] == [expected]
        monkeypatch.setattr(
            "xagent.core.tools.core.sql_worker_client.httpx.Client",
            lambda **_: nullcontext(client),
        )
        result = execute_sql_query_remote(
            connection_name="external",
            connection_url=url,
            query=query,
            output_file="scalars.jsonl",
            workspace=workspace,
        )
    assert result["success"] is True
    assert (
        json.loads(
            workspace.resolve_write_path(
                "scalars.jsonl", default_dir="output"
            ).read_text()
        )
        == expected
    )
