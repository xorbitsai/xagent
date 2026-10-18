"""Security boundary tests for the isolated SQL execution service."""

from __future__ import annotations

import os
import sqlite3

import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from fastapi.testclient import TestClient

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
        if variable in SENSITIVE_PLATFORM_ENVIRONMENT_VARIABLES or variable.endswith(
            SENSITIVE_ENVIRONMENT_SUFFIXES
        ):
            if variable != "XAGENT_SQL_WORKER_TOKEN":
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
