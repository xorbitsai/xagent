"""Tests for the authenticated client of the isolated SQL worker."""

from __future__ import annotations

import httpx
import pytest

from xagent.core.tools.core.sql_worker_client import (
    SqlWorkerError,
    execute_sql_query_remote,
)


def _client_with_transport(
    transport: httpx.MockTransport,
    seen_kwargs: dict[str, object] | None = None,
):
    real_client = httpx.Client

    def client(*args, **kwargs):
        if seen_kwargs is not None:
            seen_kwargs.update(kwargs)
        return real_client(*args, transport=transport, **kwargs)

    return client


def test_remote_request_is_authenticated_and_credentials_stay_in_body(
    monkeypatch,
):
    seen: dict[str, object] = {}
    client_kwargs: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["authorization"] = request.headers.get("Authorization")
        seen["body"] = request.read().decode()
        return httpx.Response(
            200,
            json={
                "success": True,
                "rows": [{"one": 1}],
                "row_count": 1,
                "columns": ["one"],
                "message": "ok",
            },
        )

    monkeypatch.setenv("XAGENT_SQL_WORKER_URL", "http://sql-worker:8090/")
    monkeypatch.setenv("XAGENT_SQL_WORKER_TOKEN", "shared-secret")
    monkeypatch.setattr(
        "xagent.core.tools.core.sql_worker_client.httpx.Client",
        _client_with_transport(httpx.MockTransport(handler), client_kwargs),
    )
    connection_url = "postgresql://user:password@customer.example/analytics"

    result = execute_sql_query_remote(
        connection_name="analytics",
        connection_url=connection_url,
        query="SELECT 1 AS one",
        output_file=None,
        workspace=None,
    )

    assert result["rows"] == [{"one": 1}]
    assert seen["url"] == "http://sql-worker:8090/v1/query"
    assert "password" not in str(seen["url"])
    assert seen["authorization"] == "Bearer shared-secret"
    assert connection_url in str(seen["body"])
    assert client_kwargs["trust_env"] is False


def test_remote_error_does_not_expose_http_exception(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection failed", request=request)

    monkeypatch.setenv("XAGENT_SQL_WORKER_URL", "http://sql-worker:8090")
    monkeypatch.setenv("XAGENT_SQL_WORKER_TOKEN", "shared-secret")
    monkeypatch.setattr(
        "xagent.core.tools.core.sql_worker_client.httpx.Client",
        _client_with_transport(httpx.MockTransport(handler)),
    )

    with pytest.raises(SqlWorkerError, match="worker is unavailable"):
        execute_sql_query_remote(
            connection_name="analytics",
            connection_url="postgresql://user:password@customer.example/data",
            query="SELECT 1",
            output_file=None,
            workspace=None,
        )


def test_remote_timeout_reports_unknown_completion_state(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    monkeypatch.setenv("XAGENT_SQL_WORKER_URL", "http://sql-worker:8090")
    monkeypatch.setenv("XAGENT_SQL_WORKER_TOKEN", "shared-secret")
    monkeypatch.setattr(
        "xagent.core.tools.core.sql_worker_client.httpx.Client",
        _client_with_transport(httpx.MockTransport(handler)),
    )
    with pytest.raises(SqlWorkerError, match="must not be retried automatically"):
        execute_sql_query_remote(
            connection_name="analytics",
            connection_url="sqlite:///:memory:",
            query="SELECT 1",
            output_file=None,
            workspace=None,
        )


def test_remote_non_200_uses_only_worker_safe_detail(monkeypatch):
    transport = httpx.MockTransport(
        lambda _request: httpx.Response(400, json={"detail": "safe query error"})
    )
    monkeypatch.setenv("XAGENT_SQL_WORKER_URL", "http://sql-worker:8090")
    monkeypatch.setenv("XAGENT_SQL_WORKER_TOKEN", "shared-secret")
    monkeypatch.setattr(
        "xagent.core.tools.core.sql_worker_client.httpx.Client",
        _client_with_transport(transport),
    )
    with pytest.raises(SqlWorkerError, match="safe query error"):
        execute_sql_query_remote(
            connection_name="analytics",
            connection_url="sqlite:///:memory:",
            query="SELECT 1",
            output_file=None,
            workspace=None,
        )
