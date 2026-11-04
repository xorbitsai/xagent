"""Security boundary tests for the isolated SQL execution service."""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import nullcontext
from datetime import UTC

import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from fastapi.testclient import TestClient

from xagent.core.tools.core.sql_worker_client import execute_sql_query_remote
from xagent.core.workspace import TaskWorkspace
from xagent.sql_worker import (
    SENSITIVE_ENVIRONMENT_SUFFIXES,
    SENSITIVE_PLATFORM_ENVIRONMENT_VARIABLES,
    QueryRequest,
    _authorize,
    _json_safe,
    _scrub_child_environment,
    app,
    assert_isolated_environment,
    execute_in_disposable_process,
    validate_worker_configuration,
)


@pytest.fixture(autouse=True)
def _remove_ambient_platform_secrets(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "xagent.sql_worker.DOCKER_SOCKET_PATH", tmp_path / "docker.sock"
    )
    for variable in tuple(os.environ):
        if variable != "XAGENT_SQL_WORKER_TOKEN" and (
            variable in SENSITIVE_PLATFORM_ENVIRONMENT_VARIABLES
            or variable.endswith(SENSITIVE_ENVIRONMENT_SUFFIXES)
        ):
            monkeypatch.delenv(variable, raising=False)


@pytest.mark.parametrize(
    "variable",
    [
        "DATABASE_URL",
        "POSTGRES_PASSWORD",
        "XAGENT_REDIS_URL",
        "XAGENT_SECRET_ENCRYPTION_KEY",
        "AWS_SECRET_ACCESS_KEY",
        "DOCKER_HOST",
        "OPENAI_API_KEY",
        "SECRET_KEY",
        "XAGENT_ENCRYPTION_MIGRATION_KEY",
        "XAGENT_TOOL_CREDENTIALS_MIGRATION_KEY",
        "LANGFUSE_SECRET_KEY",
    ],
)
def test_worker_refuses_platform_credentials_and_host_control(variable):
    with pytest.raises(RuntimeError, match=variable):
        assert_isolated_environment({variable: "present"})


def test_worker_accepts_only_its_own_runtime_configuration():
    assert_isolated_environment(
        {
            "XAGENT_SQL_WORKER_TOKEN": "x" * 32,
            "XAGENT_SQL_WORKER_MAX_ROWS": "100",
            "HOME": "/tmp",
        }
    )


def test_worker_refuses_a_short_authentication_token(monkeypatch):
    monkeypatch.setenv("XAGENT_SQL_WORKER_TOKEN", "short")
    with pytest.raises(RuntimeError, match="at least 32"):
        validate_worker_configuration()


def test_worker_refuses_a_mounted_docker_socket(monkeypatch, tmp_path):
    socket_path = tmp_path / "docker.sock"
    socket_path.touch()
    monkeypatch.setattr("xagent.sql_worker.DOCKER_SOCKET_PATH", socket_path)
    monkeypatch.setenv("XAGENT_SQL_WORKER_TOKEN", "x" * 32)
    with pytest.raises(RuntimeError, match="docker.sock"):
        validate_worker_configuration()


def test_child_environment_drops_worker_and_secret_values(monkeypatch):
    monkeypatch.setenv("XAGENT_SQL_WORKER_TOKEN", "worker-secret")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "platform-secret")
    monkeypatch.setenv("PATH", "/safe/bin")
    _scrub_child_environment()
    assert "XAGENT_SQL_WORKER_TOKEN" not in os.environ
    assert "LANGFUSE_SECRET_KEY" not in os.environ
    assert os.environ["PATH"] == "/safe/bin"


def test_binary_values_are_encoded_for_json():
    assert _json_safe({"value": b"\xff\x00"}) == {"value": "base64:/wA="}


@pytest.mark.parametrize("buffer", [bytes, bytearray, memoryview])
def test_binary_buffers_are_normalized_before_the_process_pipe(buffer):
    import pickle

    payload = _json_safe({"rows": [{"payload": buffer(b"\xff\x00")}]})
    assert pickle.loads(pickle.dumps(payload)) == {"rows": [{"payload": "base64:/wA="}]}


def test_result_limit_matches_the_actual_json_response():
    from datetime import datetime
    from decimal import Decimal

    from xagent.core.tools.core.sql_result import bound_sql_result, serialize_sql_result

    payload = {
        "rows": [
            {
                "binary": memoryview(b"\xff\x00"),
                "text": "你好",
                "time": datetime(2026, 1, 1, tzinfo=UTC),
                "amount": Decimal("1.25"),
            }
        ]
    }
    normalized = _json_safe(payload)
    size = len(serialize_sql_result(normalized))
    assert bound_sql_result(payload, size) == normalized
    with pytest.raises(ValueError, match="byte limit"):
        bound_sql_result(payload, size - 1)


def test_buffer_normalization_preserves_sql_scalars_and_column_names():
    from datetime import timedelta
    from decimal import Decimal

    payload = {
        "rows": [
            {
                "_salary": Decimal("9007199254740993.01"),
                "duration": timedelta(days=1, hours=2),
            }
        ]
    }
    assert _json_safe(payload) == payload


@pytest.mark.parametrize("query", ["SELECT 1e999 AS value", "SELECT -1e999 AS value"])
def test_nonfinite_sqlite_values_use_the_endpoint_null_policy(monkeypatch, query):
    monkeypatch.setenv("XAGENT_SQL_WORKER_TOKEN", "x" * 32)
    with TestClient(app) as client:
        response = client.post(
            "/v1/query",
            headers={"Authorization": f"Bearer {'x' * 32}"},
            json={
                "connection_name": "external",
                "connection_url": "sqlite:///:memory:",
                "query": query,
            },
        )
    assert response.status_code == 200
    assert response.json()["rows"] == [{"value": None}]


def test_worker_limit_matches_actual_endpoint_bytes_at_the_boundary(monkeypatch):
    monkeypatch.setenv("XAGENT_SQL_WORKER_TOKEN", "x" * 32)
    monkeypatch.setenv("XAGENT_SQL_WORKER_MAX_RESULT_BYTES", "4096")
    with TestClient(app) as client:

        def run():
            return client.post(
                "/v1/query",
                headers={"Authorization": f"Bearer {'x' * 32}"},
                json={
                    "connection_name": "external",
                    "connection_url": "sqlite:///:memory:",
                    "query": "SELECT 0.00001 AS value",
                },
            )

        initial = run()
        assert initial.status_code == 200
        size = len(initial.content)
        monkeypatch.setenv("XAGENT_SQL_WORKER_MAX_RESULT_BYTES", str(size))
        exact = run()
        assert exact.status_code == 200
        assert exact.content == initial.content
        monkeypatch.setenv("XAGENT_SQL_WORKER_MAX_RESULT_BYTES", str(size - 1))
        assert run().status_code == 400


def test_sa_prefixed_columns_survive_worker_http_and_remote_export(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("XAGENT_SQL_WORKER_TOKEN", "x" * 32)
    monkeypatch.setenv("XAGENT_SQL_WORKER_URL", "http://sql-worker:8090")
    workspace = TaskWorkspace("salary-export", base_dir=str(tmp_path))
    with TestClient(app) as client:
        response = client.post(
            "/v1/query",
            headers={"Authorization": f"Bearer {'x' * 32}"},
            json={
                "connection_name": "external",
                "connection_url": "sqlite:///:memory:",
                "query": 'SELECT 42 AS "_salary"',
            },
        )
        assert response.status_code == 200
        assert response.json()["rows"] == [{"_salary": 42}]
        assert response.json()["columns"] == ["_salary"]
        monkeypatch.setattr(
            "xagent.core.tools.core.sql_worker_client.httpx.Client",
            lambda **_: nullcontext(client),
        )
        result = execute_sql_query_remote(
            connection_name="external",
            connection_url="sqlite:///:memory:",
            query='SELECT 42 AS "_salary"',
            output_file="salary.jsonl",
            workspace=workspace,
        )
    assert result["success"] is True
    assert json.loads(
        workspace.resolve_write_path("salary.jsonl", default_dir="output").read_text()
    ) == {"_salary": 42}


@pytest.mark.parametrize("suffix", [".parquet", ".csv"])
def test_empty_select_preserves_columns_through_worker_and_remote_export(
    monkeypatch, tmp_path, suffix
):
    monkeypatch.setenv("XAGENT_SQL_WORKER_TOKEN", "x" * 32)
    monkeypatch.setenv("XAGENT_SQL_WORKER_URL", "http://sql-worker:8090")
    workspace = TaskWorkspace("empty-export", base_dir=str(tmp_path))
    query = "SELECT 1 AS id, 'test' AS name WHERE 1 = 0"
    output_file = f"empty{suffix}"
    with TestClient(app) as client:
        response = client.post(
            "/v1/query",
            headers={"Authorization": f"Bearer {'x' * 32}"},
            json={
                "connection_name": "external",
                "connection_url": "sqlite:///:memory:",
                "query": query,
            },
        )
        assert response.status_code == 200
        assert response.json()["rows"] == []
        assert response.json()["columns"] == ["id", "name"]
        assert response.json()["row_count"] == 0
        monkeypatch.setattr(
            "xagent.core.tools.core.sql_worker_client.httpx.Client",
            lambda **_: nullcontext(client),
        )
        result = execute_sql_query_remote(
            connection_name="external",
            connection_url="sqlite:///:memory:",
            query=query,
            output_file=output_file,
            workspace=workspace,
        )

    assert result["success"] is True
    assert result["columns"] == ["id", "name"]
    assert result["row_count"] == 0
    assert result["rows"] == []
    assert result["file_id"]
    assert result["relative_path"] == f"output/{output_file}"
    exported = workspace.output_dir / output_file
    if suffix == ".parquet":
        import pyarrow.parquet as pq

        table = pq.read_table(exported)
        assert table.column_names == ["id", "name"]
        assert table.num_rows == 0
    else:
        assert exported.read_text().splitlines() == ["id,name"]


@pytest.mark.parametrize("size,limit", [(3000, 4096), (7_340_032, 8_388_608)])
def test_binary_result_limit_applies_after_base64_encoding(monkeypatch, size, limit):
    monkeypatch.setenv("XAGENT_SQL_WORKER_TOKEN", "x" * 32)
    monkeypatch.setenv("XAGENT_SQL_WORKER_MAX_RESULT_BYTES", str(limit))
    monkeypatch.setenv("XAGENT_SQL_WORKER_QUERY_TIMEOUT_SECONDS", "30")
    with TestClient(app) as client:
        response = client.post(
            "/v1/query",
            headers={"Authorization": f"Bearer {'x' * 32}"},
            json={
                "connection_name": "external",
                "connection_url": "sqlite:///:memory:",
                "query": f"SELECT CAST(replace(hex(zeroblob({size // 2})), '0', 'a') AS BLOB) AS payload",
            },
        )
    assert response.status_code == 400
    assert response.json() == {
        "detail": "SQL query result exceeded the configured byte limit"
    }


def test_query_endpoint_requires_bearer_token(monkeypatch):
    monkeypatch.setenv("XAGENT_SQL_WORKER_TOKEN", "x" * 32)
    with TestClient(app) as client:
        response = client.post(
            "/v1/query",
            json={
                "connection_name": "external",
                "connection_url": "sqlite:///:memory:",
                "query": "SELECT 1",
            },
        )
    assert response.status_code == 401
    assert response.json() == {"detail": "Unauthorized"}


def test_wrong_and_non_ascii_tokens_are_unauthorized(monkeypatch):
    monkeypatch.setenv("XAGENT_SQL_WORKER_TOKEN", "x" * 32)
    for token in ("wrong", "é"):
        with pytest.raises(HTTPException) as exc_info:
            _authorize(HTTPAuthorizationCredentials(scheme="Bearer", credentials=token))
        assert exc_info.value.status_code == 401


def test_authentication_runs_before_body_validation(monkeypatch):
    monkeypatch.setenv("XAGENT_SQL_WORKER_TOKEN", "x" * 32)
    with TestClient(app) as client:
        response = client.post("/v1/query", content=b"not-json")
    assert response.status_code == 401


def test_capacity_wait_returns_429_before_query_execution(monkeypatch):
    class FullCapacity:
        def acquire(self, *, timeout):
            assert timeout == 1.0
            return False

        def release(self):
            raise AssertionError("unacquired capacity must not be released")

    monkeypatch.setattr("xagent.sql_worker._query_slots", FullCapacity())
    monkeypatch.setenv("XAGENT_SQL_WORKER_TOKEN", "x" * 32)
    with TestClient(app) as client:
        response = client.post(
            "/v1/query",
            headers={"Authorization": f"Bearer {'x' * 32}"},
            json={
                "connection_name": "external",
                "connection_url": "sqlite:///:memory:",
                "query": "SELECT 1",
            },
        )
    assert response.status_code == 429


def test_query_endpoint_does_not_echo_credentials_on_failure(monkeypatch):
    monkeypatch.setenv("XAGENT_SQL_WORKER_TOKEN", "x" * 32)
    password = "must-not-appear"
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/v1/query",
            headers={"Authorization": f"Bearer {'x' * 32}"},
            json={
                "connection_name": "external",
                "connection_url": f"not-a-driver://user:{password}@db.example/data",
                "query": "SELECT 1",
            },
        )
    assert response.status_code == 400
    assert password not in response.text
    assert response.json() == {"detail": "Invalid SQLAlchemy connection URL"}


def test_query_size_limit_is_enforced_before_process_start(monkeypatch):
    monkeypatch.setenv("XAGENT_SQL_WORKER_MAX_QUERY_BYTES", "4")
    request = QueryRequest(
        connection_name="external",
        connection_url="sqlite:///:memory:",
        query="SELECT 1",
    )
    with pytest.raises(HTTPException) as exc_info:
        execute_in_disposable_process(request)
    assert exc_info.value.status_code == 413


def test_query_runs_in_disposable_process(monkeypatch, tmp_path):
    db_path = tmp_path / "external.db"
    with sqlite3.connect(db_path) as connection:
        connection.execute("CREATE TABLE metrics (value INTEGER)")
        connection.executemany("INSERT INTO metrics VALUES (?)", [(10,), (20,)])
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("XAGENT_SQL_WORKER_QUERY_TIMEOUT_SECONDS", "10")
    monkeypatch.setenv("XAGENT_SQL_WORKER_MAX_ROWS", "10")
    monkeypatch.setenv("XAGENT_SQL_WORKER_MAX_RESULT_BYTES", "4096")

    result = execute_in_disposable_process(
        QueryRequest(
            connection_name="external",
            connection_url=f"sqlite:///{db_path}",
            query="SELECT value FROM metrics ORDER BY value",
        )
    )

    assert result["rows"] == [{"value": 10}, {"value": 20}]
    assert result["row_count"] == 2


def test_write_query_commits_in_disposable_process(monkeypatch, tmp_path):
    db_path = tmp_path / "external.db"
    with sqlite3.connect(db_path) as connection:
        connection.execute("CREATE TABLE metrics (value INTEGER)")
    monkeypatch.setenv("XAGENT_SQL_WORKER_QUERY_TIMEOUT_SECONDS", "10")
    monkeypatch.setenv("XAGENT_SQL_WORKER_MAX_ROWS", "10")
    monkeypatch.setenv("XAGENT_SQL_WORKER_MAX_RESULT_BYTES", "4096")

    result = execute_in_disposable_process(
        QueryRequest(
            connection_name="external",
            connection_url=f"sqlite:///{db_path}",
            query="INSERT INTO metrics VALUES (42)",
        )
    )

    assert result["success"] is True
    assert result["row_count"] == 1
    with sqlite3.connect(db_path) as connection:
        assert connection.execute("SELECT value FROM metrics").fetchall() == [(42,)]
